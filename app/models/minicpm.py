"""MiniCPM-V: the local VLM shared by both approaches.

- generate_answer: answers a question grounded in retrieved page images
  (one-shot; used by scripts/eval_answers_modal.py).
- generate_step / parse_tool_call / summarize_history / history_tokens: the
  agent-chat surface (pipelines/agent_ask.py) — multi-turn generation under a
  tool-calling system prompt, plus history budgeting. MiniCPM-V has no native
  function calling, so tools are a prompted protocol: a reply that *starts
  with* a JSON object is a tool call, anything else is the final answer.
- describe_batch (+ describe_figure / describe_table wrappers): used only by
  the parsed ingest pipeline (on Modal) to turn Picture crops and Table
  markdown into searchable text, batched through chat()'s batched mode.

The model and tokenizer are module-level globals: ZeroGPU packs module-level
CUDA tensors at startup and shares them with the GPU worker, whereas function
arguments are pickled — and trust_remote_code model classes are not picklable.

All functions are plain (not @spaces.GPU): callers run them inside their own
GPU context — the ask pipelines' single @spaces.GPU call on the Space, or a
real CUDA process on Modal.
"""

from __future__ import annotations

import json

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from core.constants import (
    ANSWER_MAX_NEW_TOKENS,
    DESCRIBE_MAX_NEW_TOKENS,
    HISTORY_KEEP_MESSAGES,
    MINICPM_MODEL_ID,
    MINICPM_REVISION,
    SUMMARY_MAX_NEW_TOKENS,
)

PROMPT = (
    "You are a repair-manual assistant. The images are the manual pages most "
    "relevant to the user's question, each preceded by its label (manual name "
    "and page number).\n\n"
    "Answer the question using ONLY what is printed on these pages, following "
    "these rules:\n"
    "1. If the answer is a procedure, reproduce EVERY step in order, numbered "
    "exactly as in the manual. Never skip, merge, or summarize steps. Keep each "
    "step's notes, model exceptions, and specifications (e.g. torque values) "
    "with that step, exactly as printed.\n"
    "2. Quote exact values (torques, clearances, part numbers, capacities) as "
    "printed, including units.\n"
    "3. End with the page label(s) you used, e.g. (Manual — p.238). If a "
    "procedure clearly continues on a page you were not given, say so.\n"
    "4. If the pages do not contain the answer, say so instead of guessing.\n"
    "5. Start directly with the answer — no preamble like 'Based on the "
    "provided pages'.\n\n"
    "Question: {question}"
)

# Agent chat: same grounding rules as PROMPT, restated for a multi-turn
# conversation where the model fetches its own pages with tools. The JSON
# examples use doubled braces because the template is .format()ed.
AGENT_SYSTEM_PROMPT = (
    'You are Repair Guy, an assistant helping a user with the repair manual '
    '"{manual}". You answer ONLY from the manual\'s pages, which you retrieve '
    "with tools.\n\n"
    "TOOLS\n"
    "- search_docs: semantic search over the manual; returns the most "
    "relevant pages as labeled images.\n"
    "- show_page: displays a page in the user's PDF viewer (the right side of "
    "their screen).\n\n"
    "To call a tool, reply with ONLY one JSON object and nothing else:\n"
    '{{"tool": "search_docs", "query": "<focused search phrase>"}}\n'
    '{{"tool": "show_page", "page": <page number>}}\n\n'
    "RULES\n"
    "1. For every new question about the manual, FIRST call search_docs with "
    "a focused query built from the question — never answer from memory. "
    "Follow-ups about pages already shown in this conversation may be "
    "answered directly.\n"
    "2. If the answer is a procedure, reproduce EVERY step in order, numbered "
    "exactly as in the manual. Never skip, merge, or summarize steps. Keep "
    "each step's notes, model exceptions, and specifications (e.g. torque "
    "values) with that step, exactly as printed.\n"
    "3. Quote exact values (torques, clearances, part numbers, capacities) as "
    "printed, including units.\n"
    "4. End answers with the page label(s) you used, e.g. (Manual — p.238). "
    "If a procedure clearly continues on a page you were not given, say so.\n"
    "5. If the retrieved pages do not contain the answer, you may try ONE "
    "more search_docs call with a different query; if it still is not there, "
    "say so instead of guessing.\n"
    "6. Use show_page when the user asks to see a page, or to bring up the "
    "page your answer is grounded in.\n"
    "7. When not calling a tool, reply to the user in plain markdown — never "
    "output JSON, and start directly with the answer, no preamble."
)

SUMMARY_PROMPT = (
    "Below is the transcript of an ongoing conversation between a user and a "
    "repair-manual assistant. Summarize it in at most 200 words so the "
    "assistant can seamlessly continue the conversation: keep the components "
    "and procedures discussed, every exact value mentioned (torques, "
    "clearances, capacities, part numbers), the page numbers cited, and "
    "anything the user said about their situation or vehicle. Plain text, no "
    "preamble.\n\nTranscript:\n{transcript}"
)

_TOOLS = ("search_docs", "show_page")

FIGURE_PROMPT = (
    "The image is a figure cropped from a page of a repair manual. Context "
    "from the page it appears on:\n{context}\n\n"
    "Using the context's terminology, write a search-index description of the "
    "figure in 1-3 sentences: name the specific component or assembly shown "
    "(and which models it applies to, if stated), the kind of view or diagram, "
    "and what the labeled parts or callouts are. Be definite — never write "
    "'likely', 'possibly' or 'appears to'. Example of the expected style:\n"
    "'Dimensional reference drawing of the cross-type universal joint used on "
    "the Pn35-50 and Cu35-55 propeller shafts: two yokes joined by a spider "
    "and four bearing cups, shown in side and end views with the overall "
    "length, width, and height dimensions.'\n"
    "Start directly with the description — no preamble."
)

TABLE_PROMPT = (
    "Below is a table extracted from a repair manual as markdown, followed by "
    "context from the page it appears on.\n\nTable:\n{markdown}\n\n"
    "Context:\n{context}\n\n"
    "Using the context's terminology, write a search-index description of the "
    "table in 1-3 sentences: what the table is for, which components, models "
    "or operations it covers, and what values it lists (with units). Example "
    "of the expected style:\n"
    "'Propeller shaft specifications by forklift model (Pn35-50, Pn60-80, "
    "Cu35-55, Cu60/70): the universal joint type, the three shaft length "
    "dimensions in millimeters and inches, and whether the upper and lower "
    "propeller shaft covers are fitted.'\n"
    "Start directly with the description — no preamble."
)

_MODEL = (
    AutoModel.from_pretrained(
        MINICPM_MODEL_ID,
        revision=MINICPM_REVISION,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    .to("cuda")
    .eval()
)
_TOKENIZER = AutoTokenizer.from_pretrained(
    MINICPM_MODEL_ID, revision=MINICPM_REVISION, trust_remote_code=True
)
# Pre-build the processor chat() would otherwise lazily create per GPU worker
# (it caches on this exact attribute, see modeling_minicpmv.chat).
_MODEL.processor = AutoProcessor.from_pretrained(
    MINICPM_MODEL_ID, revision=MINICPM_REVISION, trust_remote_code=True
)


def generate_answer(question: str, pages: list[tuple[str, Image.Image]]) -> str:
    """pages: [(label, page image)] in retrieval order. Must run on GPU
    (called from within a @spaces.GPU context)."""
    content = []
    for label, img in pages:  # chat() accepts interleaved strings and PIL images
        content.append(f"[{label}]")
        content.append(img.convert("RGB"))
    content.append(PROMPT.format(question=question))
    with torch.no_grad():
        answer = _MODEL.chat(
            msgs=[{"role": "user", "content": content}],
            tokenizer=_TOKENIZER,
            enable_thinking=False,
            max_new_tokens=ANSWER_MAX_NEW_TOKENS,
        )
    return str(answer).strip()


def generate_step(msgs: list[dict], manual_name: str) -> str:
    """One agent-loop generation: the full conversation so far (text history,
    current question, tool-call/tool-result exchanges with page images) under
    the tool-calling system prompt. Must run on GPU."""
    with torch.no_grad():
        out = _MODEL.chat(
            msgs=msgs,
            tokenizer=_TOKENIZER,
            system_prompt=AGENT_SYSTEM_PROMPT.format(manual=manual_name),
            enable_thinking=False,
            max_new_tokens=ANSWER_MAX_NEW_TOKENS,
        )
    return str(out).strip()


def parse_tool_call(reply: str) -> dict | None:
    """{"tool": ..., "args": {...}} if the reply is a tool call, else None.

    Only a *leading* JSON object counts (optionally inside a ``` fence) — JSON
    quoted later inside prose is part of a normal answer. Accepts both the
    prompted flat shape {"tool", "query"/"page"} and the nested {"tool",
    "args": {...}} an instruction-tuned model sometimes produces."""
    text = reply.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0].strip()
    if not text.startswith("{"):
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
    except ValueError:
        return None
    if not isinstance(obj, dict) or obj.get("tool") not in _TOOLS:
        return None
    args = obj.get("args")
    if not isinstance(args, dict):
        args = {k: v for k, v in obj.items() if k != "tool"}
    return {"tool": obj["tool"], "args": args}


def history_tokens(history: list[dict]) -> int:
    """Token count of a text-only chat history ([{role, content}])."""
    return sum(
        len(_TOKENIZER.encode(m.get("content") or "", add_special_tokens=False))
        for m in history
    )


def summarize_history(history: list[dict]) -> list[dict]:
    """Collapse all but the last HISTORY_KEEP_MESSAGES messages into one
    summary exchange, keeping role alternation intact (turns are appended in
    user/assistant pairs, so the kept tail starts with a user message).
    Must run on GPU."""
    if len(history) <= HISTORY_KEEP_MESSAGES:
        return history
    old, kept = history[:-HISTORY_KEEP_MESSAGES], history[-HISTORY_KEEP_MESSAGES:]
    transcript = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in old)
    with torch.no_grad():
        summary = _MODEL.chat(
            msgs=[{"role": "user", "content": SUMMARY_PROMPT.format(transcript=transcript)}],
            tokenizer=_TOKENIZER,
            enable_thinking=False,
            max_new_tokens=SUMMARY_MAX_NEW_TOKENS,
        )
    return [
        {
            "role": "user",
            "content": "Summary of our conversation so far (for your reference):\n"
            + str(summary).strip(),
        },
        {"role": "assistant", "content": "Got it — I'll keep that context in mind."},
    ] + kept


def describe_batch(jobs: list[tuple[str, object, str]]) -> list[str]:
    """Searchable descriptions for a batch of parsed elements in ONE chat()
    call (its batched mode: msgs = list of conversations). Each job is
    ("figure", crop image, context) or ("table", markdown, context); the
    context (manual name, section heading, caption, page text) is what lets
    MiniCPM use the manual's own terminology. Returns descriptions in job
    order. Parsed ingest only; must run on GPU."""
    msgs = []
    for kind, payload, context in jobs:
        if kind == "figure":
            content = [payload.convert("RGB"), FIGURE_PROMPT.format(context=context)]
        else:
            content = [TABLE_PROMPT.format(markdown=payload, context=context)]
        msgs.append([{"role": "user", "content": content}])
    with torch.no_grad():
        out = _MODEL.chat(
            msgs=msgs,
            tokenizer=_TOKENIZER,
            enable_thinking=False,
            max_new_tokens=DESCRIBE_MAX_NEW_TOKENS,
        )
    return [str(answer).strip() for answer in out]


def describe_figure(image: Image.Image, context: str) -> str:
    """Single-element convenience wrapper around describe_batch."""
    return describe_batch([("figure", image, context)])[0]


def describe_table(markdown: str, context: str) -> str:
    """Single-element convenience wrapper around describe_batch."""
    return describe_batch([("table", markdown, context)])[0]
