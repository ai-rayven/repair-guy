"""MiniCPM-V: the local VLM — the find-and-point "eyes".

- ground_box: return the bounding box of a described object on a page image —
  the circle the agent (models/minicpm_agent, the text "brain") asks for.
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
    "Box ONLY that one part, as TIGHTLY as possible — just the part itself. Do "
    "NOT box the whole figure, the whole diagram, a group of parts, or the page; "
    "if the part is small, the box must be small. A box wider than about half "
    "the page is almost always wrong.\n"
    "Reply with ONLY the box, as <box>x1 y1 x2 y2</box>: four integers "
    "normalized to 0-1000 (x left→right, y top→bottom) over the whole page, "
    "and nothing else. If it is not on this page, reply exactly: NOT FOUND"
)

# Visual page rerank: score the search shortlist by LOOKING at the page images,
# instead of re-judging from page text (a 1B text rerank measured worse than raw
# ColEmbed top-1, because repair pages are figure-heavy and text retrieval
# surfaces index/spec pages that merely name-drop the part).
PAGE_RERANK_PROMPT = (
    "The image is one page of a repair manual. A mechanic wants: {query!r}\n"
    "How directly does THIS page give them what they need — the procedure, "
    "component, diagram, table, or value involved (judge by topic, not exact "
    "wording)? Reply with ONLY one digit 0-5: 5 = this is exactly the page, "
    "0 = unrelated."
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


def rerank_pages(images: list[Image.Image], query: str) -> tuple[int, list[int]]:
    """Pick the best of the search shortlist by LOOKING at the page images.
    Each candidate is scored 0-5 for how directly it answers the query, in ONE
    batched chat() call; the highest score wins, ties broken by retrieval order
    (so on a tie it never does worse than ColEmbed). Returns (index into images,
    the per-page scores; -1 where the reply had no digit). Must run on GPU."""
    prompt = PAGE_RERANK_PROMPT.format(query=query)
    msgs = [
        [{"role": "user", "content": [img.convert("RGB"), prompt]}] for img in images
    ]
    with torch.no_grad():
        out = _MODEL.chat(
            msgs=msgs,
            tokenizer=_TOKENIZER,
            enable_thinking=False,
            max_new_tokens=4,
        )
    scores = []
    for a in out:
        m = re.search(r"\d", str(a))
        scores.append(int(m.group()) if m else -1)
    # max() returns the FIRST argmax, so equal scores keep ColEmbed's order.
    best = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0
    return best, scores


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
