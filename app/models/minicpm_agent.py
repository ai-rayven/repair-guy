"""MiniCPM5-1B: the text "brain" that drives the find-and-point loop.

Where models/minicpm.py is the MiniCPM-V VLM — the "eyes" that ground the circle
and write the ingest figure/table descriptions — this is the small TEXT model
that decides what to do. Each step it sees the conversation so far, the manual's
table of contents, and the WHOLE text of the page being viewed, and picks ONE
tool:

    search(query)            semantic-search the manual; the best page is shown
                             and its text added to the conversation
    go_to_section(section)   jump to a numbered table-of-contents section
    circle(target)           circle something on the CURRENT page (its text is
                             in context); the VLM grounds the box
    done(message)            nothing more to do, or it can't be found

It never writes answers — the manual does the talking. History is used only to
resolve references ("circle the other one", "go back to that bolt").

Standard LlamaForCausalLM (no trust_remote_code). Loaded as a module-level CUDA
global for the same ZeroGPU reason as the other models: module-level CUDA tensors
are shared with the GPU worker, whereas function arguments are pickled.

Entry points (plain functions; the caller runs them inside its @spaces.GPU
context). decide() takes the running message list the pipeline maintains and
returns a parsed tool call; rerank() picks the best of N candidate pages by their
text — the search tool's reranker over ColEmbed's shortlist. The system prompt
and the message builders live here so the wording stays with the model.
"""

from __future__ import annotations

import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.constants import (
    AGENT_MAX_NEW_TOKENS,
    MINICPM_AGENT_MODEL_ID,
    MINICPM_AGENT_REVISION,
)

# The tools the agent may emit, and the JSON shape of each. Kept here so the
# prompt and the parser can't drift apart.
TOOLS = ("search", "go_to_section", "circle", "done")

SYSTEM_PROMPT = (
    "You are the assistant for a hands-busy mechanic reading a repair manual on "
    "a page viewer. You do NOT answer questions or explain — the manual does the "
    "talking. Your only job is to FIND the right page and POINT at things on it.\n\n"
    "Each step, choose exactly ONE tool and reply with ONLY its JSON object — no "
    "prose, no markdown, nothing else. Replace every <...> placeholder with the "
    "real value — NEVER output the angle brackets.\n\n"
    "Tools:\n"
    '- Search the manual for a part/topic/procedure:\n'
    '  {"tool": "search", "query": "<focused search phrase>"}\n'
    "- Jump to a section — use its number from the TABLE OF CONTENTS:\n"
    '  {"tool": "go_to_section", "section": <number>}\n'
    "- Circle something on the CURRENT page (its full text is given to you):\n"
    '  {"tool": "circle", "target": "<short name of the thing to circle>"}\n'
    "- Finish — nothing more to do, or it isn't in the manual:\n"
    '  {"tool": "done", "message": "<one short line for the mechanic>"}\n\n'
    "How to choose:\n"
    '- They say "go to" / "take me to" / name a section → go_to_section.\n'
    "- The thing they want is on the CURRENT page → circle it. The target MUST "
    "be what the mechanic asked for — match it to the page's exact wording if it "
    "appears there; NEVER circle a different component.\n"
    "- Otherwise → search. After a search shows a page, that page becomes the "
    "CURRENT page, so circle on it or search again.\n"
    "- Use the conversation history only to resolve what they mean (e.g. "
    '"circle the other one"); never restate earlier answers.\n\n'
    "Examples (copy the FORMAT, not the values):\n"
    'Mechanic: "go to the cooling system" → {"tool": "go_to_section", "section": 5}\n'
    'Mechanic: "where do I replace the fuel filter" (not on this page) → '
    '{"tool": "search", "query": "fuel filter replacement"}\n'
    'Mechanic: "circle the bleeder screw" (it is on this page) → '
    '{"tool": "circle", "target": "bleeder screw"}'
)


def system_message() -> dict:
    return {"role": "system", "content": SYSTEM_PROMPT}


def state_message(
    request: str,
    toc: list[dict],
    page: int | None,
    section: str,
    page_text: str,
) -> dict:
    """The user message for the current step: what the mechanic just said, the
    table of contents (numbered, the go_to_section index), and the whole text of
    the page being viewed. page_text is the parsed page rendered to text (figures
    and tables as their descriptions) — empty when no page is open or the manual
    has no parse."""
    toc_lines = "\n".join(
        f"{i + 1}. {s['title']} (p.{s['page']})" for i, s in enumerate(toc)
    ) or "(none)"
    where = f"p.{page}" + (f', section "{section}"' if section else "") if page else "(no page open)"
    page_block = (
        f"CURRENT PAGE ({where}) — full text:\n{page_text}"
        if page_text
        else f"CURRENT PAGE: {where} (no text available)"
    )
    # The request goes LAST (after the long page text) so it stays freshest —
    # otherwise the page block dominates and the agent acts on the page instead
    # of what was asked.
    return {
        "role": "user",
        "content": (
            f"{page_block}\n\n"
            f"TABLE OF CONTENTS:\n{toc_lines}\n\n"
            f"The mechanic said: {request!r}\n"
            "Choose ONE tool and reply with ONLY its JSON object."
        ),
    }


def tool_result_message(text: str) -> dict:
    """A tool's outcome fed back into the loop as the next observation (e.g. the
    page a search landed on, with its text, so the agent can circle on it)."""
    return {"role": "user", "content": text}


def assistant_action_message(tool: dict) -> dict:
    """Record an action the agent took, so it stays in the running transcript
    (this turn) and the compact history (later turns)."""
    return {"role": "assistant", "content": json.dumps(tool, separators=(",", ":"))}


_TOKENIZER = AutoTokenizer.from_pretrained(
    MINICPM_AGENT_MODEL_ID, revision=MINICPM_AGENT_REVISION
)
_MODEL = (
    AutoModelForCausalLM.from_pretrained(
        MINICPM_AGENT_MODEL_ID,
        revision=MINICPM_AGENT_REVISION,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    .to("cuda")
    .eval()
)


def _generate(messages: list[dict], max_new_tokens: int) -> str:
    """Greedy decode the assistant's next message. enable_thinking=False — tool
    routing wants a terse decision, not a reasoning trace."""
    inputs = _TOKENIZER.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to(_MODEL.device)
    # apply_chat_template emits token_type_ids, which this LlamaForCausalLM's
    # generate() rejects as an unused kwarg.
    inputs.pop("token_type_ids", None)
    with torch.no_grad():
        out = _MODEL.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = _TOKENIZER.decode(
        out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return text.strip()


def _parse_tool(raw: str) -> dict | None:
    """Pull the JSON tool call out of the reply. Tolerant of a ```json fence or
    a stray lead-in. Returns a validated
    {tool, ...} or None when the reply isn't usable (the caller decides the
    fallback). Range-checking go_to_section is the caller's job — it has the
    TOC; this only validates shape."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0].strip()
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    tool = obj.get("tool")
    if tool == "search":
        query = str(obj.get("query") or "").strip()
        return {"tool": "search", "query": query} if _real(query) else None
    if tool == "go_to_section":
        try:
            return {"tool": "go_to_section", "section": int(obj.get("section"))}
        except (TypeError, ValueError):
            return None
    if tool == "circle":
        target = str(obj.get("target") or "").strip()
        return {"tool": "circle", "target": target} if _real(target) else None
    if tool == "done":
        return {"tool": "done", "message": str(obj.get("message") or "").strip()}
    return None


def _real(value: str) -> bool:
    """A usable arg, not an echoed placeholder. Small models sometimes copy the
    schema example verbatim ("<short name of the thing to circle>") — angle
    brackets are the tell; reject so the loop re-asks for a real value."""
    return bool(value) and "<" not in value and ">" not in value


def decide(messages: list[dict]) -> tuple[dict | None, str]:
    """One agentic step. messages is the running conversation the pipeline
    maintains (system + past turns + this turn's state and any tool results).
    Returns (parsed tool call, raw reply); the tool is None when the reply isn't
    a usable JSON tool call. Must run on GPU."""
    raw = _generate(messages, AGENT_MAX_NEW_TOKENS)
    return _parse_tool(raw), raw


RERANK_PROMPT = (
    "A mechanic is searching a repair manual for: {query!r}\n\n"
    "Below are {n} candidate pages, each its page number and the text on it. "
    "Pick the ONE page that best covers what the mechanic wants — the component, "
    "system, or procedure involved (judge by topic, not exact wording). Reply "
    "with ONLY that page's number, nothing else.\n\n{candidates}"
)


def rerank(query: str, candidates: list[tuple[int, str]]) -> tuple[int, str]:
    """Pick the best of ColEmbed's shortlist by page text. candidates is
    [(page, text)] in retrieval order. Returns (index into candidates, raw
    reply); falls back to index 0 (the top ColEmbed hit) when the reply can't be
    read. Must run on GPU."""
    if not candidates:
        return 0, ""
    listing = "\n\n".join(
        f"PAGE {page}:\n{text or '(no text)'}" for page, text in candidates
    )
    prompt = RERANK_PROMPT.format(query=query, n=len(candidates), candidates=listing)
    raw = _generate([{"role": "user", "content": prompt}], max_new_tokens=8)
    m = re.search(r"\d+", raw)
    if m:
        picked = int(m.group())
        for i, (page, _) in enumerate(candidates):
            if page == picked:
                return i, raw
    return 0, raw
