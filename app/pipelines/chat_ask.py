"""Chat ask pipeline: one retrieval + one grounded generation per Q&A turn.

This replaced the agent loop (the model no longer drives retrieval with
prompted tool calls): a question turn is search → render → answer, every step
in code. Navigation ("go to brakes", "next page") and parse-bbox circling
never reach this module — they are CPU routes in app.py; this pipeline is only
for real content questions, plus point_box for VLM circling.

The whole turn streams as events and runs in ONE @spaces.GPU call (generators
are supported): retrieval, page rendering, the answer generation, and — when
the text history has outgrown HISTORY_TOKEN_BUDGET — a summarization pass
first.

Events yielded (PIL images included; app.py converts them for the wire):
  {"type": "status", "text", "kind"?}                    progress for the UI
  {"type": "tool_call", "tool": "search_docs", "args"}   search starting
  {"type": "tool_result", "tool": "search_docs",
   "gallery": [(img, caption)], "page_refs"}             retrieved pages
  {"type": "answer", "answer", "history", "summarized",
   "grounded_page"}                                      final; grounded_page
                                                         is where the viewer
                                                         should land

History is the client's: a [{role, content}] text transcript sent with each
turn and returned updated (past page images cannot live in history under
chat()'s 16384-token input cap, so turns are stored as plain text).

viewer ({"page", "section"}) is what the right pane currently shows: it is
described to the model, and when the question points at it ("this", "here")
the viewed page's image is attached alongside the retrieved ones.
"""

from __future__ import annotations

import logging
import re

import spaces

from core.constants import (
    CHAT_GPU_DURATION,
    HISTORY_KEEP_MESSAGES,
    HISTORY_TOKEN_BUDGET,
    POINT_GPU_DURATION,
)
from core.pdf import render_page
from models import minicpm

log = logging.getLogger("repairguy.chat")


def _fmt_msgs(msgs: list[dict]) -> str:
    """The full conversation as sent to the model, for the logs — text parts
    verbatim, page images as placeholders."""
    lines = []
    for m in msgs:
        parts = [
            c if isinstance(c, str) else f"<page image {c.size[0]}x{c.size[1]}>"
            for c in m["content"]
        ]
        lines.append(f"--- {m['role']} ---\n" + "\n".join(parts))
    return "\n".join(lines)


def _searcher(approach: str):
    """Lazy import: parsed_ask/visual_ask import this module's pipeline class,
    and both retriever modules load models at import."""
    if approach == "visual":
        from models.colembed import maxsim_search

        return maxsim_search
    from pipelines.parsed_ask import retrieve_pages

    return retrieve_pages


def _grounded_page(answer: str, candidates: list[int]) -> int | None:
    """The page the viewer should land on: the last page number the answer
    cites that is actually among the pages the model was shown (so a
    hallucinated citation can't drive the viewer), else the top retrieved."""
    cited = [int(n) for n in re.findall(r"p\.?\s*(\d+)", answer)]
    for page in reversed(cited):
        if page in candidates:
            return page
    return candidates[0] if candidates else None


@spaces.GPU(duration=CHAT_GPU_DURATION)
def chat_events(
    question: str,
    store,
    doc_ids: list[str],
    top_k: int,
    names: dict[str, str],
    history: list[dict],
    approach: str,
    viewer: dict | None = None,
):
    """Yield the events of one Q&A turn (see module docstring)."""
    search = _searcher(approach)
    manual = names[doc_ids[0]]
    viewer = viewer or {}

    summarized = False
    if (
        len(history) > HISTORY_KEEP_MESSAGES
        and (tokens := minicpm.history_tokens(history)) > HISTORY_TOKEN_BUDGET
    ):
        log.info(
            "history %d msgs / %d tokens over budget %d — summarizing",
            len(history), tokens, HISTORY_TOKEN_BUDGET,
        )
        yield {
            "type": "status",
            "kind": "summarizing",
            "text": "Summarizing earlier conversation…",
        }
        history = minicpm.summarize_history(history)
        summarized = True
        log.info(
            "summarized to %d msgs / %d tokens",
            len(history), minicpm.history_tokens(history),
        )

    yield {"type": "tool_call", "tool": "search_docs", "args": {"query": question}}
    refs = search(question, store, doc_ids, top_k)
    log.info(
        "search(%r) → %s", question, [(d, p, round(s, 3)) for d, p, s in refs]
    )
    pages = [
        (f"{names[doc_id]} — p.{page}", render_page(store.pdf_path(doc_id), page))
        for doc_id, page, _ in refs
    ]
    yield {
        "type": "tool_result",
        "tool": "search_docs",
        "gallery": [
            (img, f"{label} (score {score:.3g})")
            for (label, img), (_, _, score) in zip(pages, refs)
        ],
        "page_refs": [(doc_id, page) for doc_id, page, _ in refs],
    }
    yield {"type": "status", "text": "Reading the pages…"}

    # What the right pane currently shows: always described in text; the page
    # image itself is attached when the question points at it ("this", "here")
    # and it isn't already among the retrieved pages.
    gen_pages = list(pages)
    candidates = [page for _, page, _ in refs]
    note = ""
    cur = int(viewer.get("page") or 0)
    if cur >= 1:
        section = str(viewer.get("section") or "").strip()
        note = f"The user is currently viewing page {cur}" + (
            f' (section "{section}")' if section else ""
        ) + " of the manual in their viewer."
        if cur not in candidates and re.search(
            r"\b(this|these|here|that|current(ly)?)\b", question, re.I
        ):
            try:
                img = render_page(store.pdf_path(doc_ids[0]), cur)
                gen_pages.insert(0, (f"{manual} — p.{cur} (the page being viewed)", img))
                candidates.insert(0, cur)
            except ValueError:
                pass

    content: list = []
    for label, img in gen_pages:
        content.append(f"[{label}]")
        content.append(img.convert("RGB"))
    if note:
        content.append(note)
    content.append(f"Question: {question}")
    msgs = [{"role": m["role"], "content": [m["content"]]} for m in history]
    msgs.append({"role": "user", "content": content})

    log.info("generate_chat input:\n%s", _fmt_msgs(msgs))
    answer = minicpm.generate_chat(msgs, manual)
    log.info("answer (%d chars): %r", len(answer), answer[:300])

    yield {
        "type": "answer",
        "answer": answer,
        "history": history
        + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "summarized": summarized,
        "grounded_page": _grounded_page(answer, candidates),
    }


@spaces.GPU(duration=POINT_GPU_DURATION)
def point_box(pdf_path: str, page: int, query: str) -> dict | None:
    """Visual grounding for the /point endpoint: the bounding box of the
    described object on a page, as {"page", "bbox"} in rendered-page pixels
    (the same space the /page route serves) — or None when the model can't
    place it."""
    img = render_page(pdf_path, page)
    box, raw = minicpm.ground_box(img, query)
    log.info("ground_box(%r) on p.%d → %s | raw=%r", query, page, box, raw[:200])
    if box is None:
        return None
    return {"page": page, "bbox": [round(v) for v in box]}
