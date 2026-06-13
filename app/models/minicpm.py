"""MiniCPM-V: the local VLM shared by both approaches.

- route_request / classify_pages / ground_box: the find-and-point surface
  (pipelines/find_ask.py) — route a request against the page being viewed,
  YES/NO-classify candidate pages in one batched call, and return the
  bounding box of a described object on a page image.
- generate_answer: answers a question grounded in retrieved page images
  (one-shot; used by scripts/eval_answers_modal.py and the pipelines' .run()).
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
import re

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from core.constants import (
    ANSWER_MAX_NEW_TOKENS,
    DESCRIBE_MAX_NEW_TOKENS,
    MINICPM_MODEL_ID,
    MINICPM_REVISION,
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

# Find-and-point router: the model sees the page the mechanic is viewing plus
# a numbered list of manual sections, and chooses ONE action — circle on this
# page, jump to a section, or search the manual. The JSON examples use doubled
# braces because the template is .format()ed.
ROUTER_PROMPT = (
    'The image is the page of the repair manual "{manual}" that a mechanic is '
    "viewing right now (page {page}{section_part}). The mechanic said: "
    "{request!r}\n\n"
    "Choose ONE action and reply with ONLY its JSON object:\n"
    "- Circle something that is visible on THIS page:\n"
    '  {{"tool": "point_here", "target": "<short name of the thing to circle>"}}\n'
    "- Jump to a section (pick its number from the SECTIONS list below):\n"
    '  {{"tool": "go_to_section", "section": <number>}}\n'
    "- Search the whole manual for a part/topic, to circle once found:\n"
    '  {{"tool": "search", "query": "<focused search phrase>", '
    '"target": "<short name of the thing to circle>"}}\n\n'
    "Prefer point_here when the thing is plainly on this page; prefer "
    "go_to_section when they name or describe a section/procedure; otherwise "
    "search.\n\n"
    "SECTIONS:\n{sections}\n\n"
    "Reply with ONLY the JSON object, nothing else."
)

# Parallel page classification: one batched call, one YES/NO per candidate.
# Relevance-by-topic, NOT literal containment — the page that helps with a
# "radiator leak" is the radiator page, which never literally shows a leak.
# Too strict a test gives up on good retrievals; this still rejects pages about
# an unrelated system (and truly absent things, so the give-up path survives).
CLASSIFY_PROMPT = (
    "The image is one page of a repair manual. A mechanic is looking for "
    "{target!r}. Is THIS page relevant to that — does it show or cover the "
    "component, system, or procedure involved (as a diagram, table, "
    "specification, or steps)? Judge by topic, not exact wording: a page "
    "about the radiator is relevant to a 'radiator leak'; a page about an "
    "unrelated system is not. Reply with ONLY YES or NO."
)

# Visual grounding for "circle the <thing>": the mechanic asks to circle
# something on the page they are viewing. MiniCPM-V grounding replies in
# <box>x1 y1 x2 y2</box> form with coordinates normalized to 0-1000.
GROUND_PROMPT = (
    "The image is one page of a repair manual. A mechanic asked to circle "
    "{query!r} on this page. Locate it precisely:\n"
    "- In an exploded or assembly diagram, parts carry callout numbers/letters "
    "on leader lines, and a legend lists what each number is. Find {query!r} in "
    "the legend to get its number, then follow that number's leader line to the "
    "part in the drawing and box THAT part (not the legend text).\n"
    "- Otherwise it may be a row in a table, a specification value, or a "
    "heading — box that.\n"
    "Reply with ONLY the box, as <box>x1 y1 x2 y2</box>: four integers "
    "normalized to 0-1000 (x left→right, y top→bottom) over the whole page, "
    "and nothing else. If it is not on this page, reply exactly: NOT FOUND"
)

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


def route_request(
    image: Image.Image,
    request: str,
    manual: str,
    page: int,
    section: str,
    options: list[dict],
) -> tuple[dict | None, str]:
    """(route, raw reply). options is the numbered section list shown to the
    model ([{title, page}]). route is one of:
        {"tool": "point_here", "target": str}
        {"tool": "go_to_section", "page": int, "title": str}  (index resolved)
        {"tool": "search", "query": str, "target": str}
    or None when the reply isn't usable (caller falls back to a manual-wide
    search with the raw request). Must run on GPU."""
    sections = "\n".join(
        f"{i + 1}. {o['title']} (p.{o['page']})" for i, o in enumerate(options)
    )
    prompt = ROUTER_PROMPT.format(
        manual=manual,
        page=page,
        section_part=f', section "{section}"' if section else "",
        request=request,
        sections=sections or "(none)",
    )
    with torch.no_grad():
        out = _MODEL.chat(
            msgs=[{"role": "user", "content": [image.convert("RGB"), prompt]}],
            tokenizer=_TOKENIZER,
            enable_thinking=False,
            max_new_tokens=128,
        )
    raw = str(out).strip()
    text = raw
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0].strip()
    start = text.find("{")
    if start < 0:
        return None, raw
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return None, raw
    if not isinstance(obj, dict):
        return None, raw
    tool = obj.get("tool")
    if tool == "point_here":
        target = str(obj.get("target") or request).strip()
        return {"tool": "point_here", "target": target}, raw
    if tool == "go_to_section":
        try:
            idx = int(obj.get("section")) - 1
        except (TypeError, ValueError):
            return None, raw
        if not 0 <= idx < len(options):
            return None, raw
        return {
            "tool": "go_to_section",
            "page": int(options[idx]["page"]),
            "title": options[idx]["title"],
        }, raw
    if tool == "search":
        target = str(obj.get("target") or request).strip()
        return {
            "tool": "search",
            "query": str(obj.get("query") or target).strip(),
            "target": target,
        }, raw
    return None, raw


def classify_pages(images: list[Image.Image], target: str) -> list[bool]:
    """Whether each page shows the target — all pages classified in ONE
    chat() call (its batched mode: msgs = list of conversations), so the
    candidates are checked in parallel. Must run on GPU."""
    prompt = CLASSIFY_PROMPT.format(target=target)
    msgs = [
        [{"role": "user", "content": [img.convert("RGB"), prompt]}]
        for img in images
    ]
    with torch.no_grad():
        out = _MODEL.chat(
            msgs=msgs,
            tokenizer=_TOKENIZER,
            enable_thinking=False,
            max_new_tokens=8,
        )
    return [str(a).strip().upper().startswith("YES") for a in out]


def ground_box(
    image: Image.Image, query: str
) -> tuple[tuple[float, float, float, float] | None, str]:
    """(bbox, raw reply): the bounding box of the described object on a page
    image, in that image's pixel coordinates — or None when the model can't
    place it (no box in the reply, or a degenerate one). Must run on GPU."""
    with torch.no_grad():
        out = _MODEL.chat(
            msgs=[
                {
                    "role": "user",
                    "content": [image.convert("RGB"), GROUND_PROMPT.format(query=query)],
                }
            ],
            tokenizer=_TOKENIZER,
            enable_thinking=False,
            max_new_tokens=64,
        )
    raw = str(out).strip()
    if "NOT FOUND" in raw.upper():
        return None, raw
    # Read the coordinates from the <box>…</box> tag specifically. MiniCPM-V
    # may prefix a <ref>…</ref> (e.g. echoing "5. Rod"); that digit would
    # otherwise be grabbed as the first coordinate and shift the whole box.
    m = re.search(r"<box>(.*?)</box>", raw, re.IGNORECASE | re.DOTALL)
    nums = re.findall(r"\d+(?:\.\d+)?", m.group(1) if m else raw)
    if len(nums) < 4:
        return None, raw
    x1, y1, x2, y2 = (min(1000.0, max(0.0, float(n))) for n in nums[:4])
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None, raw
    w, h = image.size
    return (x1 * w / 1000, y1 * h / 1000, x2 * w / 1000, y2 * h / 1000), raw


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
