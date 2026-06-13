"""MiniCPM5-1B: the text "brain" that drives the find-and-point loop.

Where models/minicpm.py is the MiniCPM-V VLM — the "eyes" that ground the circle
and write the ingest figure/table descriptions — this is the small TEXT model
that decides what to do. Each step it sees the conversation so far, the manual's
table of contents, and the WHOLE text of the page being viewed, and picks ONE
tool:

    search(query)            semantic-search the manual; the best page is shown
                             and its text added to the conversation
    find_answer(query)       dense TEXT search of the parsed chunks for a
                             spec/value/fact; the page that states it is shown,
                             its text added so the agent can circle the answer
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

import gc
import json
import logging
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.constants import (
    AGENT_MAX_NEW_TOKENS,
    AGENT_MODELS,
    DEFAULT_AGENT_MODEL,
)

log = logging.getLogger("repairguy.agent")

# The tools the agent may emit, and the JSON shape of each. Kept here so the
# prompt and the parser can't drift apart.
TOOLS = ("search", "find_answer", "go_to_section", "go_to_page", "circle", "done")

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
    "- Look up the ANSWER to a question — a spec, value, or fact (fuel type, a "
    "torque, an oil/coolant capacity). This searches the manual TEXT, so it "
    "finds the page that STATES the answer when a topic search would miss the "
    "plain specs page:\n"
    '  {"tool": "find_answer", "query": "<the spec or fact being asked for>"}\n'
    "- Jump to a section — use its number from the TABLE OF CONTENTS:\n"
    '  {"tool": "go_to_section", "section": <number>}\n'
    "- Jump straight to a known PHYSICAL page number:\n"
    '  {"tool": "go_to_page", "page": <number>}\n'
    "- Circle something on a page CURRENTLY ON SCREEN (their full text is given "
    "to you) — a part, OR the line/value that answers their question. When two "
    "pages are shown, set page to the one whose text has it:\n"
    '  {"tool": "circle", "target": "<short name or printed words to circle>", '
    '"page": <the on-screen page number it is on>}\n'
    "- Finish — nothing more to do, or it isn't in the manual:\n"
    '  {"tool": "done", "message": "<one short line for the mechanic>"}\n\n'
    "How to choose:\n"
    '- They say "go to" / "take me to" / name a section → go_to_section.\n'
    "- You already KNOW the page number — e.g. the CURRENT page is an index or "
    'contents that lists the part with a page number ("Actuators .... 855"), or '
    "history gave one → go_to_page that number. Do NOT circle the index line; "
    "go to the page it points to.\n"
    "- The thing they want is on a page ON SCREEN → circle it, and set page to "
    "the one it is on. The target MUST be what the mechanic asked for — match it "
    "to that page's wording if it appears there (a part named inside a figure or "
    "diagram description still counts as on the page); NEVER circle a different "
    "component.\n"
    "- They ask a QUESTION whose answer is printed on a page on screen — a spec, "
    'a value, or a line (e.g. "what fuel does it take" answered by "Engine fuel '
    '- Gasoline") → circle that spot, using the printed words as the target. '
    "Point at the answer; do NOT keep searching for a better page.\n"
    "- They ask a QUESTION for a spec/value/fact that is NOT on screen yet "
    '("what fuel does it take", "engine oil capacity", "drain-plug torque") → '
    "find_answer with the fact as the query; it searches the manual text and "
    "shows the page that states it, which you then circle. Use find_answer for "
    "fact questions; use search to find a part, diagram, or procedure.\n"
    "- Otherwise → search. After a search shows a page, that page is on screen, "
    "so circle on it or search again — but NEVER repeat a search that did not "
    "move you; act on the page instead.\n"
    "- Use the conversation history only to resolve what they mean (e.g. "
    '"circle the other one"); never restate earlier answers.\n\n'
    "Examples (copy the FORMAT, not the values):\n"
    'Mechanic: "go to the cooling system" → {"tool": "go_to_section", "section": 5}\n'
    'Mechanic: "where do I replace the fuel filter" (not on this page) → '
    '{"tool": "search", "query": "fuel filter replacement"}\n'
    'Mechanic: "the wastegate actuator" (current page is an index reading '
    '"Actuators .... 855") → {"tool": "go_to_page", "page": 855}\n'
    'Mechanic: "circle the bleeder screw" (it is on p.412, which is on screen) → '
    '{"tool": "circle", "target": "bleeder screw", "page": 412}\n'
    'Mechanic: "what fuel does it take" (p.5 on screen shows "Engine fuel - '
    'Gasoline") → {"tool": "circle", "target": "Engine fuel - Gasoline", "page": 5}\n'
    'Mechanic: "what\'s the engine oil capacity" (not on this page) → '
    '{"tool": "find_answer", "query": "engine oil capacity"}'
)


def system_message() -> dict:
    return {"role": "system", "content": SYSTEM_PROMPT}


def state_message(
    request: str,
    toc: list[dict],
    shown: list[dict],
    section: str,
) -> dict:
    """The user message for the current step: what the mechanic just said, the
    table of contents (numbered, the go_to_section index), and the whole text of
    the page(s) currently on the viewer. The viewer shows a two-page spread, so
    `shown` is [{page, text}] for each page on screen (one or two). Each page's
    text is the parsed page rendered to text (figures/tables as descriptions) —
    empty when the manual has no parse. When the agent circles, it names which of
    these pages the target is on."""
    toc_lines = "\n".join(
        f"{i + 1}. {s['title']} (p.{s['page']})" for i, s in enumerate(toc)
    ) or "(none)"
    if shown:
        where = " and ".join(f"p.{s['page']}" for s in shown) + (
            f' (section "{section}")' if section else ""
        )
        blocks = "\n\n".join(
            f"PAGE {s['page']} — full text:\n{s['text'] or '(no text available)'}"
            for s in shown
        )
        page_block = (
            f"CURRENTLY ON SCREEN — {where}. You can circle on "
            + (
                "either of these pages (say which in the circle call):\n"
                if len(shown) > 1
                else "this page:\n"
            )
            + blocks
        )
    else:
        page_block = "CURRENTLY ON SCREEN: (no page open)"
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


def search_result_message(request: str, page: int, text: str, stuck: bool) -> str:
    """The observation fed back after a search shows a page. `stuck` is set by the
    pipeline when the search was a no-op — the same query again, or it landed on a
    page already shown this turn. Then the message FORCES a decision and DROPS the
    "search again" option, because a 1B on greedy decoding will otherwise re-issue
    the same search and loop on the same page forever (seen live: 5x the identical
    search, all landing on p.876, never circling)."""
    body = (
        f"Search showed p.{page}. It is now the CURRENT page.\n"
        f"CURRENT PAGE (p.{page}) — full text:\n"
        f"{text or '(no text available)'}\n\n"
    )
    if stuck:
        return body + (
            f"You are still on p.{page}; that search did not move you, and "
            "repeating it will NOT help. Decide now — do not search again:\n"
            f"- If this page shows {request!r} OR the line/value that answers it "
            "(it counts even when named inside a figure or diagram description), "
            'circle that spot: {"tool": "circle", "target": "<the printed words>", '
            f'"page": {page}}}.\n'
            "- If it belongs on a different page, use go_to_section or go_to_page.\n"
            "- Only if it is truly not in this manual, use done."
        )
    return body + (
        f"The mechanic asked for: {request!r}. If this page shows THAT — or the "
        "line/value that answers it, including when named inside a figure or "
        "diagram description — circle that spot (use the words as printed on the "
        "page). Otherwise search again or go to a section. Do not circle a "
        "different component."
    )


def ground_failed_message(request: str, target: str, page: int) -> str:
    """The observation fed back when the VLM could not locate `target` on p.`page`.
    A failed grounding almost always means the target is NOT on this page (the
    agent circled on the wrong one), so this FORCES a relocate — search or
    navigate — and forbids re-circling the same spot, the same greedy-loop guard
    used after a no-op search. Without it, circle is a dead-end: the page shows
    with no pin and the turn ends."""
    return (
        f"I could not find {target!r} on p.{page} — it does not appear to be on "
        "this page, so circling here will not work. Do NOT circle that on this "
        "page again. It is almost certainly on a DIFFERENT page: search for "
        f"{request!r}, or go to the right section or page, and only circle once "
        "you are on the page whose text actually shows it. Use done only if it is "
        "truly not in this manual."
    )


def assistant_action_message(tool: dict) -> dict:
    """Record an action the agent took, so it stays in the running transcript
    (this turn) and the compact history (later turns)."""
    return {"role": "assistant", "content": json.dumps(tool, separators=(",", ":"))}


# --- the resident brain: ONE model in VRAM, swapped on demand ---------------
# The agent model is selectable from the UI (core.constants.AGENT_MODELS). Only
# one is held on the GPU at a time: use_model() evicts the current one before
# loading the next ("load-on-switch"), so VRAM stays flat as the user A/Bs
# models. The model/tokenizer live in module globals for the same ZeroGPU reason
# as the other models — module-level CUDA tensors are shared with the GPU worker.
_REGISTRY = {m["key"]: m for m in AGENT_MODELS}
_active_key: str | None = None
_MODEL = None
_TOKENIZER = None
# Whether the active model's chat template accepts enable_thinking (Qwen3 /
# MiniCPM yes, Cohere no) — drives whether we pass the kwarg below.
_THINKING = False


def _spec(key: str | None) -> dict:
    return _REGISTRY.get(key or "", _REGISTRY[DEFAULT_AGENT_MODEL])


def use_model(key: str | None = None) -> str:
    """Make `key` the resident agent brain, loading it (and evicting the
    previous one) when it isn't already active — one model in VRAM at a time.
    Unknown keys fall back to the default. Called once per turn by the pipeline,
    inside the GPU context. Returns the active key. Must run on GPU."""
    global _active_key, _MODEL, _TOKENIZER, _THINKING
    spec = _spec(key)
    if spec["key"] == _active_key and _MODEL is not None:
        return _active_key
    # Drop the current model first so VRAM holds only one brain at a time.
    if _MODEL is not None:
        log.info("agent model: evicting %s", _active_key)
        _MODEL = _TOKENIZER = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log.info("agent model: loading %s (%s)", spec["key"], spec["model_id"])
    _TOKENIZER = AutoTokenizer.from_pretrained(
        spec["model_id"], revision=spec["revision"]
    )
    _MODEL = (
        AutoModelForCausalLM.from_pretrained(
            spec["model_id"],
            revision=spec["revision"],
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    _active_key, _THINKING = spec["key"], spec["thinking"]
    return _active_key


# Load the default brain eagerly at import — same as the other models, so the
# ZeroGPU startup packing covers it and the common (no-switch) path pays no
# load cost on the first turn.
use_model(DEFAULT_AGENT_MODEL)


def _template_kwargs() -> dict:
    """Extra apply_chat_template kwargs for the active model. Tool routing wants
    a terse decision, not a reasoning trace, so disable thinking — but only for
    templates that accept the kwarg (others would ignore or choke on it)."""
    return {"enable_thinking": False} if _THINKING else {}


def _generate(messages: list[dict], max_new_tokens: int) -> str:
    """Greedy decode the assistant's next message."""
    inputs = _TOKENIZER.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **_template_kwargs(),
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


def render_prompt(messages: list[dict]) -> str:
    """The exact text fed to the model this step — the chat template applied to
    the running message list, special tokens and all. Mirrors _generate's
    template call but returns the string instead of tokenizing, so the
    Diagnostics view can show precisely what the brain was asked. CPU-only."""
    return _TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **_template_kwargs(),
    )


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
    if tool == "find_answer":
        query = str(obj.get("query") or "").strip()
        return {"tool": "find_answer", "query": query} if _real(query) else None
    if tool == "go_to_section":
        try:
            return {"tool": "go_to_section", "section": int(obj.get("section"))}
        except (TypeError, ValueError):
            return None
    if tool == "go_to_page":
        try:
            return {"tool": "go_to_page", "page": int(obj.get("page"))}
        except (TypeError, ValueError):
            return None
    if tool == "circle":
        target = str(obj.get("target") or "").strip()
        if not _real(target):
            return None
        out = {"tool": "circle", "target": target}
        # Optional: which on-screen page the target is on (the pipeline validates
        # it against the pages actually shown and defaults to the active page).
        try:
            if obj.get("page") is not None:
                out["page"] = int(obj.get("page"))
        except (TypeError, ValueError):
            pass
        return out
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
