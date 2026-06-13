"""Agent find-and-point pipeline: one user message is one streamed turn driven
by MiniCPM5-1B (the text "brain"), which calls tools in a loop until the turn
ends with a page shown — usually a circle drawn — never a generated answer.

Flow (one @spaces.GPU call, streamed as events):
  Build the running conversation — system rules, the compact history of past
  turns (memory, for "circle the other one" / "go back"), and the live state:
  the request, the manual's table of contents, and the WHOLE text of the page
  being viewed (parsed page → text, figures/tables as their descriptions). Then
  loop, up to AGENT_MAX_STEPS:
    decide → ONE tool:
      go_to_section(n)   land at that section's first page                (terminal)
      search(query)      ColEmbed top-N → 1B rerank by page text → show
                         the best page; its text is fed back so the agent
                         can then circle on it                            (continues)
      circle(target)     ground the target on the CURRENT page (VLM) and
                         circle it                                        (terminal)
      done(message)      nothing to do / not in the manual                (terminal)

Retrieval is FUSED: ColEmbed (visual store) supplies the shortlist, the parsed
store supplies the page text the 1B reranks with and the agent reasons over — so
a manual must be indexed both ways.

History is used only to resolve references, never to restate answers. Each turn
is otherwise grounded in the viewer state the client sends (current page +
section) and the history it accumulates.

Events yielded (PIL images included; app.py converts them for the wire):
  {"type": "status", "text"}                                progress for the UI
  {"type": "step", "tool": "search"|"go_to_section"|"circle", ...}
                                                            the tool just chosen
  {"type": "tool_result", "tool": "search_docs",
   "gallery": [(img, caption)], "page_refs"}                search candidates
  {"type": "found", "page"}                                 show this page now
  {"type": "done", "kind": "navigate"|"point"|"reply", ...} terminal; point may
                                                            carry bbox=null (page
                                                            shown, not pinpointed)
"""

from __future__ import annotations

import logging

import spaces

from core.constants import (
    AGENT_HISTORY_TURNS,
    AGENT_MAX_STEPS,
    FIND_GPU_DURATION,
)
from core.page_context import index_pages, page_to_text
from core.pdf import page_count, render_page
from models import minicpm, minicpm_agent
from models.colembed import maxsim_search

log = logging.getLogger("repairguy.agent")


def _history_messages(history: list | None) -> list[dict]:
    """The compact memory of past turns as plain user/assistant turns: what the
    mechanic asked and what we did. The client sends [{request, action}]; only
    the last AGENT_HISTORY_TURNS are kept."""
    msgs = []
    for turn in (history or [])[-AGENT_HISTORY_TURNS:]:
        request = str((turn or {}).get("request") or "").strip()
        action = str((turn or {}).get("action") or "").strip()
        if request:
            msgs.append({"role": "user", "content": request})
        if action:
            msgs.append({"role": "assistant", "content": action})
    return msgs


@spaces.GPU(duration=FIND_GPU_DURATION)
def agent_events(
    request: str,
    visual_store,
    parsed_store,
    doc_ids: list[str],
    top_k: int,
    names: dict[str, str],
    sections: list[dict],
    viewer: dict | None = None,
    history: list | None = None,
):
    """Yield the events of one agent turn (see module docstring). sections is the
    numbered table of contents shown to the agent ([{title, page}]); the agent's
    go_to_section index is 1-based into it."""
    doc_id = doc_ids[0]
    manual = names[doc_id]
    viewer = viewer or {}
    cur = max(1, int(viewer.get("page") or 1))
    section = str(viewer.get("section") or "").strip()

    # Parsed pages read once; page_text(p) is the whole-page text for the agent
    # (and the reranker). Empty for a page with no parse.
    page_elements = index_pages(parsed_store.parsed_pages(doc_id))

    def page_text(p: int) -> str:
        return page_to_text(page_elements.get(p, []))

    # The page(s) on the viewer — a two-page spread shows the active page plus
    # the next. The agent sees the text of all of them and may circle on any;
    # the active page stays first. Falls back to the single current page.
    shown_pages = [int(p) for p in (viewer.get("pages") or []) if int(p) >= 1] or [cur]
    if cur in shown_pages:
        shown_pages = [cur] + [p for p in shown_pages if p != cur]
    shown_pages = shown_pages[:2]
    shown = [{"page": p, "text": page_text(p)} for p in shown_pages]

    messages = [minicpm_agent.system_message()]
    messages += _history_messages(history)
    messages.append(minicpm_agent.state_message(request, sections, shown, section))

    current_page = shown_pages[0]  # the active page circle defaults to
    circleable = set(shown_pages)  # pages the agent may circle on right now
    seen_pages = set(shown_pages)  # pages already put on screen this turn
    tried_queries = set()  # normalized search queries already issued this turn
    yield {"type": "status", "text": "Thinking…"}

    for step in range(AGENT_MAX_STEPS):
        # Render the exact prompt BEFORE deciding so the trace can show what the
        # brain was asked, not just what it answered.
        prompt = minicpm_agent.render_prompt(messages)
        tool, raw = minicpm_agent.decide(messages)
        log.info("step %d: tool=%s | raw=%r", step, tool, raw[:200])
        # Diagnostic event: the prompt fed in, the raw 1B reply, and the parsed
        # tool for this step, so the UI's trace view shows exactly what the brain
        # was asked and decided (and why a reply was rejected). Not used by the
        # normal chip flow.
        yield {"type": "trace", "step": step, "tool": tool, "raw": raw,
               "prompt": prompt}
        if tool is None:
            # Unusable reply (bad JSON, or an echoed placeholder target). Correct
            # it and let the agent try again rather than abandon the turn.
            messages.append(
                minicpm_agent.tool_result_message(
                    "Your last reply was not one complete JSON object. Reply with "
                    "ONE complete JSON object and nothing else, e.g. "
                    '{"tool": "search", "query": "fuel filter"}. If you circle, the '
                    "target MUST be copied from the page text above — never invent "
                    "a part that is not printed there."
                )
            )
            continue
        messages.append(minicpm_agent.assistant_action_message(tool))

        if tool["tool"] == "go_to_section":
            idx = tool["section"] - 1
            if not 0 <= idx < len(sections):
                messages.append(
                    minicpm_agent.tool_result_message(
                        f"There is no section {tool['section']}. Pick a number from "
                        "the table of contents, or use search."
                    )
                )
                continue
            opt = sections[idx]
            yield {"type": "step", "tool": "go_to_section",
                   "title": opt["title"], "page": int(opt["page"])}
            yield {"type": "done", "kind": "navigate", "nav": "section",
                   "page": int(opt["page"]), "title": opt["title"]}
            return

        if tool["tool"] == "go_to_page":
            page = tool["page"]
            n = page_count(visual_store.pdf_path(doc_id))
            if not 1 <= page <= n:
                messages.append(
                    minicpm_agent.tool_result_message(
                        f"There is no page {page}; this manual has pages 1–{n}. "
                        "Pick a page in range, search, or go to a section."
                    )
                )
                continue
            yield {"type": "step", "tool": "go_to_page", "page": page}
            yield {"type": "done", "kind": "navigate", "nav": "page",
                   "page": page, "title": f"Page {page}"}
            return

        if tool["tool"] == "search":
            query = tool["query"]
            yield {"type": "step", "tool": "search", "query": query}
            yield {"type": "status", "text": f"Searching for “{query}”…"}
            # k (the viewer's slider) is the shortlist size; ColEmbed's top page
            # is the one shown. A 1B text rerank measured WORSE than raw ColEmbed
            # top-1 (0.68 vs 0.84 hit@1) — visual late interaction already ranks
            # these (figure-heavy) pages better than re-judging from page text.
            hits = maxsim_search(query, visual_store, doc_ids, top_k)
            log.info("search(%r) → %s", query, [(p, round(s, 3)) for _, p, s in hits])
            if not hits:
                messages.append(
                    minicpm_agent.tool_result_message(f"Search for {query!r} found nothing.")
                )
                continue
            pages = [(p, render_page(visual_store.pdf_path(doc_id), p)) for _, p, _ in hits]
            yield {
                "type": "tool_result",
                "tool": "search_docs",
                "gallery": [
                    (img, f"{manual} — p.{p} (score {s:.3g})")
                    for (p, img), (_, _, s) in zip(pages, hits)
                ],
                "page_refs": [(doc_id, p) for _, p, _ in hits],
            }
            best_page = hits[0][1]
            yield {"type": "found", "page": best_page}
            current_page = best_page
            circleable = {best_page}  # search landed here — circle on this page
            # A no-op search — the same query again, or it lands on a page already
            # shown this turn — means the agent is stuck. Feed back a FORCING
            # message (no "search again") so a greedy 1B can't loop the identical
            # search forever; otherwise the normal "circle or search again" prompt.
            qkey = " ".join(query.lower().split())
            stuck = qkey in tried_queries or best_page in seen_pages
            tried_queries.add(qkey)
            seen_pages.add(best_page)
            messages.append(
                minicpm_agent.tool_result_message(
                    minicpm_agent.search_result_message(
                        request, best_page, page_text(best_page), stuck
                    )
                )
            )
            continue

        if tool["tool"] == "circle":
            target = tool["target"]
            # The agent says which shown page the target is on (it has both pages'
            # text). Default to the active page when it's unspecified or not one of
            # the pages on screen — so the box is grounded on, and drawn over, the
            # RIGHT page.
            page = tool.get("page")
            if page not in circleable:
                page = current_page
            yield {"type": "step", "tool": "circle", "target": target, "page": page}
            yield {"type": "status", "text": "Pinning it down…"}
            img = render_page(visual_store.pdf_path(doc_id), page)
            box, braw = minicpm.ground_box(img, target)
            log.info("ground_box(%r) on p.%d → %s | raw=%r",
                     target, page, box, braw[:200])
            yield {
                "type": "done",
                "kind": "point",
                "found": True,
                "target": target,
                "page": page,
                "bbox": [round(v) for v in box] if box is not None else None,
                # the VLM's raw grounding reply — diagnostic only (helps explain
                # where/why a box landed); shown in the trace view.
                "ground_raw": braw[:300],
            }
            return

        if tool["tool"] == "done":
            yield {"type": "done", "kind": "reply",
                   "message": tool.get("message") or "Done."}
            return

    yield {
        "type": "done",
        "kind": "reply",
        "message": "I went in circles on that one — try rephrasing?",
    }


class AgentPipeline:
    """Stateless: the stores are passed per call (fused — visual for retrieval,
    parsed for page text)."""

    def run_find(
        self,
        visual_store,
        parsed_store,
        request: str,
        doc_ids: list[str] | None,
        top_k: int,
        sections: list[dict],
        viewer: dict | None = None,
        history: list | None = None,
    ):
        """One streamed agent turn (the event generator of agent_events)."""
        request = (request or "").strip()
        if not request:
            raise ValueError("Tell me what to find.")
        docs = visual_store.list_docs()
        if not docs:
            raise ValueError("No manuals in this library yet.")
        names = {d["doc_id"]: d["name"] for d in docs}
        return agent_events(
            request, visual_store, parsed_store, doc_ids or list(names),
            int(top_k), names, sections, viewer, history,
        )
