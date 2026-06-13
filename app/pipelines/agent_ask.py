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
from core.pdf import render_page
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

    messages = [minicpm_agent.system_message()]
    messages += _history_messages(history)
    messages.append(
        minicpm_agent.state_message(request, sections, cur, section, page_text(cur))
    )

    current_page = cur  # the page circle acts on; moves when search shows one
    yield {"type": "status", "text": "Thinking…"}

    for step in range(AGENT_MAX_STEPS):
        tool, raw = minicpm_agent.decide(messages)
        log.info("step %d: tool=%s | raw=%r", step, tool, raw[:200])
        if tool is None:
            yield {
                "type": "done",
                "kind": "reply",
                "message": "Sorry, I didn't catch that — try rephrasing.",
            }
            return
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
            yield {"type": "done", "kind": "navigate",
                   "page": int(opt["page"]), "title": opt["title"]}
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
            messages.append(
                minicpm_agent.tool_result_message(
                    f"Search showed p.{best_page}. It is now the CURRENT page.\n"
                    f"CURRENT PAGE (p.{best_page}) — full text:\n"
                    f"{page_text(best_page) or '(no text available)'}"
                )
            )
            continue

        if tool["tool"] == "circle":
            target = tool["target"]
            yield {"type": "step", "tool": "circle", "target": target}
            yield {"type": "status", "text": "Pinning it down…"}
            img = render_page(visual_store.pdf_path(doc_id), current_page)
            box, braw = minicpm.ground_box(img, target)
            log.info("ground_box(%r) on p.%d → %s | raw=%r",
                     target, current_page, box, braw[:200])
            yield {
                "type": "done",
                "kind": "point",
                "found": True,
                "target": target,
                "page": current_page,
                "bbox": [round(v) for v in box] if box is not None else None,
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
