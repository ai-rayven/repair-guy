"""MiniCPM5-1B: the text "brain" that drives the find-and-point loop.

Where models/minicpm.py is the MiniCPM-V VLM — the "eyes" that ground the circle
and write the ingest figure/table descriptions — this is the small TEXT model
that decides what to do. Each step it sees the conversation so far and the WHOLE
text of the page being viewed (no table of contents — it navigates by retrieval,
not a chapter index), and picks ONE tool:

    search(query)            visual-search the manual (ColEmbed); the best page
                             is shown and its text added to the conversation
    circle(target)           circle something on the CURRENT page (its text is
                             in context); the VLM grounds the box
    done(message)            nothing more to do, or it can't be found

It never writes answers — the manual does the talking. History is used only to
resolve references ("circle the other one", "go back to that bolt").

The default brain is a standard LlamaForCausalLM (no trust_remote_code); other
selectable brains may ship custom modeling code (loaded with trust_remote_code
per their AGENT_MODELS spec — see use_model). Loaded as a module-level CUDA
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

from core import tracing
from core.constants import (
    AGENT_MAX_NEW_TOKENS,
    AGENT_MODELS,
    DEFAULT_AGENT_MODEL,
)
from core.vram import log_vram

log = logging.getLogger("repairguy.agent")

# The tools the agent may emit, and the JSON shape of each. Kept here so the
# prompt and the parser can't drift apart.
TOOLS = ("search", "go_to_page", "circle", "done")

SYSTEM_PROMPT = (
    "You are the assistant for a hands-busy mechanic reading a repair manual on "
    "a page viewer. You do NOT answer questions or explain — the manual does the "
    "talking. Your only job is to FIND the right page and POINT at things on it.\n\n"
    "Each step, choose exactly ONE tool and reply with ONLY its JSON object — no "
    "prose, no markdown, nothing else. Replace every <...> placeholder with the "
    "real value — NEVER output the angle brackets.\n\n"
    "Tools:\n"
    '- Search the manual for a part/topic/procedure, OR for the page that states '
    "a spec/value/fact (a torque, a fuel type, an oil/coolant capacity) — search "
    "with the thing being asked for as the query:\n"
    '  {"tool": "search", "query": "<focused search phrase>"}\n'
    "- Jump straight to a known PHYSICAL page number:\n"
    '  {"tool": "go_to_page", "page": <number>}\n'
    "- Circle the ONE thing on a page CURRENTLY ON SCREEN (their full text is "
    "given to you) that answers the mechanic — the specific component, the "
    "torque/spec VALUE, or the table row. `target` is the EXACT words printed on "
    "that page for that thing (copy them from the page text you were given): a "
    "part name, a spec label, or the value itself. NEVER the mechanic's sentence, "
    "their question, or a paraphrase. If those words are NOT in the page text you "
    "were given, it is not on screen — do NOT circle it; search instead. When two "
    "pages are shown, set page to the one whose text has it:\n"
    '  {"tool": "circle", "target": "<exact printed words for the part/value>", '
    '"page": <the on-screen page number it is on>}\n'
    "- Finish — nothing more to do, or it isn't in the manual:\n"
    '  {"tool": "done", "message": "<one short line for the mechanic>"}\n\n'
    "How to choose — do this IN ORDER every step:\n"
    "1. FIRST read the CURRENT PAGE text you were given. If the thing the mechanic "
    "wants is there — the part itself, or the line/value that answers their "
    'question (e.g. "what fuel does it take" answered by a printed "Engine fuel - '
    'Gasoline") → CIRCLE it: copy the exact printed words as the target and set '
    "page to the page it is on (a part named inside a figure or diagram "
    "description still counts as being on the page). This holds NO MATTER which "
    'verb they use ("find", "search", "show", "where is", "circle") — if it is on '
    "the page in front of you, you point at it; you do NOT search for a better "
    "page, and you do NOT circle a different component than the one asked for.\n"
    "If instead they asked to SEE or SHOW a whole diagram, overview, or components "
    "view (not one specific part) and it is already on the screen, the PAGE ITSELF "
    "is the answer → reply done with a brief confirmation; do NOT circle one "
    "component out of a diagram they asked to see in full.\n"
    "2. If it is NOT in the current page text, it is not on screen. Then:\n"
    "   - A chapter, system, or section named by topic, or any part / procedure / "
    'spec you cannot see ("go to the cooling system", "engine oil capacity", '
    '"remove the EWGA actuator") → search with that thing as the query.\n'
    "   - go_to_page ONLY with an EXPLICIT page number — one the mechanic gives "
    'outright ("go to page 612"), a relative step from the page(s) on screen '
    '("go back a page" / "previous page" → the page just BEFORE the first one '
    'shown; "next page" → the page just AFTER the last one shown — a spread shows '
    'two), an index line on this page that lists it ("Actuators .... 855"), or one '
    "history gave you. A navigation request is NEVER answered by circling or "
    "searching — it is a page move. NEVER invent or guess a page number.\n"
    "   - Truly absent from the manual → done.\n"
    "Never circle a target whose words are not in the page text in front of you. "
    "After a search shows a page, that page is on screen — circle on it; NEVER "
    "repeat a search that did not move you.\n"
    "Every step answers THIS step's request — re-decide from scratch. If the "
    "mechanic changes topic, names a different part/system, says the page is "
    'wrong or "this doesn\'t help" or "forget it", or asks to navigate, then the '
    "thing on screen is NOT the answer: run the steps for the NEW request — circle "
    "the new thing only if it is in the page text on screen, otherwise search for "
    "it or navigate. NEVER repeat the exact action you just took, and NEVER keep "
    "circling the same page once the mechanic has moved on or said it did not "
    "help — repeating your last answer is the one thing you must never do; if they "
    "are still asking, that answer did not satisfy them, so MOVE.\n"
    "Use the conversation history only to resolve what they mean (e.g. "
    '"circle the other one"); never restate earlier answers.\n\n'
    "Examples (copy the FORMAT, not the values):\n"
    'Mechanic: "go to the cooling system" → {"tool": "search", "query": "cooling system"}\n'
    'Mechanic: "where do I replace the fuel filter" (not on this page) → '
    '{"tool": "search", "query": "fuel filter replacement"}\n'
    'Mechanic: "the wastegate actuator" (current page is an index reading '
    '"Actuators .... 855") → {"tool": "go_to_page", "page": 855}\n'
    'Mechanic: "go back a page" (p.850 on screen) → {"tool": "go_to_page", "page": 849}\n'
    'Mechanic: "next page" (p.46 and p.47 on screen) → {"tool": "go_to_page", "page": 48}\n'
    'Mechanic: "show me how to bleed the brakes" (parked on the engine-oil page, '
    'brakes not on screen) → {"tool": "search", "query": "brake bleeding"}\n'
    'Mechanic: "circle the bleeder screw" (it is on p.412, which is on screen) → '
    '{"tool": "circle", "target": "bleeder screw", "page": 412}\n'
    'Mechanic: "search the thermostat" (p.826 on screen shows "Thermostat" in the '
    'cooling diagram) → {"tool": "circle", "target": "Thermostat", "page": 826}\n'
    'Mechanic: "what fuel does it take" (p.5 on screen shows "Engine fuel - '
    'Gasoline") → {"tool": "circle", "target": "Engine fuel - Gasoline", "page": 5}\n'
    'Mechanic: "what\'s the clutch bolt torque" (p.630 shows a row "Clutch cover '
    'bolt .... 27.5") → {"tool": "circle", "target": "Clutch cover bolt torque spec", "page": 630}\n'
    'Mechanic: "what\'s the engine oil capacity" (not on this page) → '
    '{"tool": "search", "query": "engine oil capacity"}'
)


def system_message() -> dict:
    return {"role": "system", "content": SYSTEM_PROMPT}


def state_message(
    request: str,
    shown: list[dict],
    section: str,
) -> dict:
    """The user message for the current step: what the mechanic just said and the
    whole text of the page(s) currently on the viewer. No table of contents — the
    agent navigates by search (visual retrieval) and the current page's
    text, not a chapter index. The viewer shows a two-page spread, so `shown` is
    [{page, text}] for each page on screen (one or two). Each page's text is the
    parsed page rendered to text (figures/tables as descriptions) — empty when the
    manual has no parse. When the agent circles, it names which of these pages the
    target is on."""
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
            'circle it — set "target" to the words exactly as printed on '
            f"p.{page} (copy them from the page text above), NEVER the mechanic's "
            'question: {"tool": "circle", "target": "<exact printed words for the '
            f'part/value>", "page": {page}}}.\n'
            "- If it belongs on a different page, use go_to_page.\n"
            "- Only if it is truly not in this manual, use done."
        )
    return body + (
        f"The mechanic asked for: {request!r}. First, is THIS the right page? Check "
        "the title/section at the top: if the page is about a DIFFERENT system than "
        'they asked about — a part that merely shares a word (a "gear" inside a '
        "fuel-system actuator is NOT a transmission gear) — it is the WRONG page, "
        "so do NOT circle; search again with a more specific query. If it IS the "
        "right page and shows the part, or the line/value that answers it (it "
        "counts even when named inside a figure or diagram description), circle it "
        'now — set "target" to '
        f"a part name or the exact words printed on p.{page} (copy them from the "
        "page text above), NOT the mechanic's question: "
        '{"tool": "circle", "target": "<exact printed words for the part/value>", '
        f'"page": {page}}}. Do not circle a different component. Only if it is NOT '
        "on this page, move: go_to_page if you know where, or search with a "
        "DIFFERENT query — never repeat the search you just ran, it will land here "
        "again."
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
        f"{request!r}, or go to the right page, and only circle once "
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
# all current brains do) — drives whether we pass the kwarg below.
_THINKING = False


def _spec(key: str | None) -> dict:
    return _REGISTRY.get(key or "", _REGISTRY[DEFAULT_AGENT_MODEL])


def _active_model_id() -> str:
    """The HF id of the resident brain — the `model` for its generation spans."""
    return _spec(_active_key).get("model_id", _active_key or "unknown")


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
        log_vram(f"before-evict-{_active_key}")
        _MODEL = _TOKENIZER = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_vram(f"after-evict-{_active_key}")
    log.info("agent model: loading %s (%s)", spec["key"], spec["model_id"])
    trust = spec.get("trust_remote_code", False)
    _TOKENIZER = AutoTokenizer.from_pretrained(
        spec["model_id"], revision=spec["revision"], trust_remote_code=trust
    )
    load_kwargs = dict(
        revision=spec["revision"],
        dtype=torch.bfloat16,
        trust_remote_code=trust,
    )
    # attn_implementation defaults to "sdpa" but a spec can override it — None
    # omits the kwarg entirely so a model's own modeling code picks its default
    # (e.g. a custom sparse-attention arch we don't want to force to sdpa).
    attn = spec.get("attn_implementation", "sdpa")
    if attn is not None:
        load_kwargs["attn_implementation"] = attn
    # A spec can ask accelerate to place weights straight on the GPU (device_map)
    # instead of the default load-on-CPU-then-.to("cuda"). Generic knob, unused by
    # the current brains (all small enough for the plain path); kept for a future
    # large brain where the bulk host→device copy would be worth skipping.
    device_map = spec.get("device_map")
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(spec["model_id"], **load_kwargs)
    if device_map is None:
        model = model.to("cuda")
    _MODEL = model.eval()
    _active_key, _THINKING = spec["key"], spec["thinking"]
    log_vram(f"after-load-{_active_key}")
    return _active_key


# Load the default brain eagerly at import so ZeroGPU's startup tensor-packing
# covers it and the common (no-switch) turn pays no per-turn load cost. A plain
# .to("cuda") model is safe at import on ZeroGPU: the `spaces` library patches
# torch and "packs" it into the forked GPU worker (the bf16 8B default included),
# so it stays resident without ever being rebuilt per turn.
use_model(DEFAULT_AGENT_MODEL)


def _template_kwargs() -> dict:
    """Extra apply_chat_template kwargs for the active model. Tool routing wants
    a terse decision, not a reasoning trace, so disable thinking — but only for
    templates that accept the kwarg (others would ignore or choke on it)."""
    return {"enable_thinking": False} if _THINKING else {}


def _generate(
    messages: list[dict], max_new_tokens: int, trace_name: str = "agent-generate"
) -> str:
    """Greedy decode the assistant's next message. Traced as one `generation`
    (the resident brain as the model, the messages as input, the reply and the
    in/out token counts attached) when Langfuse is configured."""
    # Defensive: ensure a brain is resident. The default loads at import, so this
    # is a no-op in normal operation; it only fires if that eager load was skipped
    # (e.g. a future deferred default). Always reached inside a @spaces.GPU window.
    if _MODEL is None:
        use_model(DEFAULT_AGENT_MODEL)
    with tracing.generation(
        trace_name, model=_active_model_id(), input=messages
    ) as gen:
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
        n_in = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            out = _MODEL.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        new_ids = out[0, n_in:]
        text = _TOKENIZER.decode(new_ids, skip_special_tokens=True)
        if gen is not None:
            n_out = int(new_ids.shape[0])
            gen.update(
                output=text.strip(),
                usage_details={"input": n_in, "output": n_out, "total": n_in + n_out},
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
    fallback). Range-checking go_to_page is the caller's job — it knows the
    page count; this only validates shape."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0].strip()
    start = text.find("{")
    if start < 0:
        return None
    body = text[start:]
    try:
        obj, _ = json.JSONDecoder().raw_decode(body)
    except ValueError:
        # A 1B greedy-decodes the closing brace away: it emits EOS right after
        # the query/target string's closing quote, so the object is complete and
        # correct except for the final "}". Re-close that one case rather than
        # discard a good reply (this is what turns a right "engine diagram"
        # search into a thrown-away reply, then a drifted retry). Anything more
        # broken than a missing brace falls through to the caller's re-ask.
        body = body.rstrip()
        if not body.endswith('"'):
            return None
        try:
            obj, _ = json.JSONDecoder().raw_decode(body + "}")
        except ValueError:
            return None
    if not isinstance(obj, dict):
        return None
    tool = obj.get("tool")
    if tool == "search":
        query = str(obj.get("query") or "").strip()
        return {"tool": "search", "query": query} if _real(query) else None
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
    raw = _generate(messages, AGENT_MAX_NEW_TOKENS, trace_name="agent-decide")
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
    raw = _generate(
        [{"role": "user", "content": prompt}], max_new_tokens=8, trace_name="agent-rerank"
    )
    m = re.search(r"\d+", raw)
    if m:
        picked = int(m.group())
        for i, (page, _) in enumerate(candidates):
            if page == picked:
                return i, raw
    return 0, raw
