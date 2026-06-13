"""Find-and-point pipeline: every non-obvious request is one stateless turn
that ends with a page shown (and usually a circle drawn) — never a generated
answer. The manual does the talking; the assistant finds and points.

Flow (one @spaces.GPU call, streamed as events):
  1. ROUTE   MiniCPM sees the page the mechanic is viewing plus a numbered
             section list and picks one tool:
               point_here(target)     circle it on this page
               go_to_section(n)       land at that section's first page
               search(query, target)  look manual-wide
  2a. point_here → ground the target on the viewed page and circle it.
  2b. go_to_section → jump to the section start (no circle). Landing only;
      we don't then hunt within the section (kept simple on purpose).
  2c. search → semantic retrieval with the routed query, then ONE batched
      generation classifies every candidate page in parallel ("does this page
      show <target>?"). The first YES page is shown and the target circled on
      it; all NO → give up for now.

There is no chat history: each request stands alone, grounded only in the
viewer state (the current page is sent with every request).

Events yielded (PIL images included; app.py converts them for the wire):
  {"type": "status", "text"}                              progress for the UI
  {"type": "route", "route": "local"|"section"|"search",
   "target"?, "query"?, "title"?, "page"?}                router decision
  {"type": "tool_result", "tool": "search_docs",
   "gallery": [(img, caption)], "page_refs"}              candidate pages
  {"type": "classified", "results": [[page, bool], ...]}  per-page verdicts
  {"type": "found", "page"}                               show this page now
  {"type": "done", "kind": "navigate"|"point",
   "found"?, "target"?, "page"?, "bbox"?, "title"?}       terminal; for a
                                                          point, bbox may be
                                                          null even when found
                                                          (page located but
                                                          not pinpointed)
"""

from __future__ import annotations

import logging

import spaces

from core.constants import FIND_GPU_DURATION
from core.pdf import render_page
from models import minicpm

log = logging.getLogger("repairguy.find")


def _searcher(approach: str):
    """Lazy import: parsed_ask/visual_ask import this module's pipeline class,
    and both retriever modules load models at import."""
    if approach == "visual":
        from models.colembed import maxsim_search

        return maxsim_search
    from pipelines.parsed_ask import retrieve_pages

    return retrieve_pages


def _search_and_circle(search, store, doc_ids, top_k, manual, target, query):
    """The search path, shared by the search route and the point_here
    fallback: retrieve → classify all candidates in parallel → circle the
    first match. A generator of events; ends with a terminal 'done'."""
    refs = search(query, store, doc_ids, top_k)
    log.info("search(%r) → %s", query, [(d, p, round(s, 3)) for d, p, s in refs])
    if not refs:
        yield {"type": "done", "kind": "point", "found": False, "target": target}
        return
    pages = [
        (page, render_page(store.pdf_path(doc_id), page))
        for doc_id, page, _ in refs
    ]
    yield {
        "type": "tool_result",
        "tool": "search_docs",
        "gallery": [
            (img, f"{manual} — p.{page} (score {score:.3g})")
            for (page, img), (_, _, score) in zip(pages, refs)
        ],
        "page_refs": [(doc_id, page) for doc_id, page, _ in refs],
    }

    yield {"type": "status", "text": f"Checking {len(pages)} pages…"}
    flags = minicpm.classify_pages([img for _, img in pages], target)
    log.info("classify(%r) → %s", target, [(p, f) for (p, _), f in zip(pages, flags)])
    yield {
        "type": "classified",
        "results": [[page, flag] for (page, _), flag in zip(pages, flags)],
    }

    for (page, img), flag in zip(pages, flags):
        if not flag:
            continue
        yield {"type": "found", "page": page}
        yield {"type": "status", "text": "Pinning it down…"}
        box, raw = minicpm.ground_box(img, target)
        log.info("ground_box(%r) on p.%d → %s | raw=%r", target, page, box, raw[:200])
        yield {
            "type": "done",
            "kind": "point",
            "found": True,
            "target": target,
            "page": page,
            "bbox": [round(v) for v in box] if box is not None else None,
        }
        return

    yield {"type": "done", "kind": "point", "found": False, "target": target}


@spaces.GPU(duration=FIND_GPU_DURATION)
def find_events(
    request: str,
    store,
    doc_ids: list[str],
    top_k: int,
    names: dict[str, str],
    approach: str,
    sections: list[dict],
    viewer: dict | None = None,
):
    """Yield the events of one find-and-point turn (see module docstring).
    sections is the numbered router option list ([{title, page}])."""
    search = _searcher(approach)
    manual = names[doc_ids[0]]
    viewer = viewer or {}
    cur = max(1, int(viewer.get("page") or 1))
    section = str(viewer.get("section") or "").strip()

    yield {"type": "status", "text": "Looking at your page…"}
    try:
        cur_img = render_page(store.pdf_path(doc_ids[0]), cur)
    except ValueError:  # stale viewer state; route without the local option
        cur_img = None

    route = None
    if cur_img is not None:
        route, raw = minicpm.route_request(cur_img, request, manual, cur, section, sections)
        log.info("route(%r) on p.%d → %s | raw=%r", request, cur, route, raw[:200])
    if route is None:  # unusable reply → treat as a plain manual-wide search
        route = {"tool": "search", "query": request, "target": request}

    if route["tool"] == "go_to_section":
        yield {
            "type": "route",
            "route": "section",
            "title": route["title"],
            "page": route["page"],
        }
        yield {
            "type": "done",
            "kind": "navigate",
            "page": route["page"],
            "title": route["title"],
        }
        return

    if route["tool"] == "point_here":
        target = route["target"]
        yield {"type": "route", "route": "local", "target": target}
        yield {"type": "status", "text": "Pinning it down…"}
        box, raw = minicpm.ground_box(cur_img, target)
        log.info("ground_box(%r) on p.%d → %s | raw=%r", target, cur, box, raw[:200])
        if box is not None:
            yield {
                "type": "done",
                "kind": "point",
                "found": True,
                "target": target,
                "page": cur,
                "bbox": [round(v) for v in box],
            }
            return
        # Router said "here" but grounding couldn't pin it — fall through to a
        # search rather than give up on a routing hunch.
        log.info("local grounding failed — falling through to search")
        query = target
    else:  # search
        target, query = route["target"], route["query"] or route["target"]

    yield {"type": "route", "route": "search", "target": target, "query": query}
    yield from _search_and_circle(
        search, store, doc_ids, top_k, manual, target, query
    )
