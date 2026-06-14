"""Agent find-and-point pipeline: one user message is one streamed turn driven
by MiniCPM5-1B (the text "brain"), which calls tools in a loop until the turn
ends with a page shown — usually a circle drawn — never a generated answer.

Flow (one @spaces.GPU call, streamed as events):
  Build the running conversation — system rules, the compact history of past
  turns (memory, for "circle the other one" / "go back"), and the live state:
  the request and the WHOLE text of the page being viewed (parsed page → text,
  figures/tables as their descriptions). No table of contents is injected. Then
  loop, up to AGENT_MAX_STEPS:
    decide → ONE tool:
      search(query)      ColEmbed top-N → 1B rerank by page text → show
                         the best page; its text is fed back so the agent
                         can then circle on it                            (continues)
      find_answer(query) dense TEXT retrieval over the parsed chunks → show
                         the page that STATES the answer (where a visual
                         search would miss the plain specs page); its text
                         is fed back so the agent can circle it           (continues)
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
  {"type": "step", "tool": "search"|"go_to_page"|"circle", ...}
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

from core import tracing
from core.constants import (
    AGENT_HISTORY_TURNS,
    AGENT_MAX_STEPS,
    FIND_GPU_DURATION,
)
from core.page_context import index_pages, page_to_text
from core.pdf import page_count, render_page
from core.vram import log_vram, reset_peak, set_enabled
from models import minicpm, minicpm_agent
from models.colembed import maxsim_search
from pipelines.parsed_ask import retrieve_pages

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
    viewer: dict | None = None,
    history: list | None = None,
    ground_thinking: bool | None = None,
    agent_model: str | None = None,
    vram_log: bool = False,
    session_id: str | None = None,
):
    """Yield the events of one agent turn (see module docstring). The agent gets
    no table of contents — it works from retrieval (search / find_answer) and the
    current page's text. ground_thinking toggles MiniCPM-V's
    reasoning for the circle grounding (None → server default); agent_model picks
    which brain drives the loop (None → default), loaded on switch inside this
    GPU window. vram_log enables the per-turn VRAM probe (UI setting)."""
    # Apply the UI's VRAM-logging toggle for this turn. Done here (inside the GPU
    # worker) rather than in the parent so a reused ZeroGPU worker always honors
    # the current request's setting. When off, every log_vram/reset_peak below is
    # a no-op.
    set_enabled(vram_log)
    # Snapshot the GPU budget for this turn: reset the peak counter, then log the
    # resident set (VLM + ColEmbed + embedder + current brain) as it stands
    # before any swap. use_model() logs its own evict/load deltas; the
    # after-ground snapshot below captures the activation high-water (peak) of the
    # heaviest op — the VLM grounding on a full-res page.
    reset_peak()
    log_vram("turn-start")
    # Swap in the selected brain (evicts the previous one) before any decide/
    # rerank. Inside this @spaces.GPU window, so the load happens on the GPU.
    active = minicpm_agent.use_model(agent_model)
    log.info("agent brain: %s", active)
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
    messages.append(minicpm_agent.state_message(request, shown, section))

    current_page = shown_pages[0]  # the active page circle defaults to
    circleable = set(shown_pages)  # pages the agent may circle on right now
    seen_pages = set(shown_pages)  # pages already put on screen this turn
    tried_queries = set()  # normalized search queries already issued this turn
    ground_failed = set()  # (page, normalized target) the VLM already missed
    yield {"type": "status", "text": "Thinking…"}

    def present_hits(hits, qkey):
        """Shared tail for the two retrieval tools (search / find_answer): show
        the shortlist, land on the top page, and feed its text back — FORCING a
        decision when the landing is a no-op (the same query again, or a page
        already shown this turn), so a greedy 1B can't loop the identical lookup
        forever. The only thing that differs between the tools is the retriever
        that produced `hits`; everything downstream is identical."""
        nonlocal current_page, circleable
        rendered = [
            (p, render_page(visual_store.pdf_path(doc_id), p)) for _, p, _ in hits
        ]
        yield {
            "type": "tool_result",
            "tool": "search_docs",
            "gallery": [
                (img, f"{manual} — p.{p} (score {s:.3g})")
                for (p, img), (_, _, s) in zip(rendered, hits)
            ],
            "page_refs": [(doc_id, p) for _, p, _ in hits],
        }
        best_page = hits[0][1]
        yield {"type": "found", "page": best_page}
        current_page = best_page
        circleable = {best_page}  # the lookup landed here — circle on this page
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

    # Open the trace for this turn (no-op when Langfuse is unconfigured). The
    # whole loop runs under try/finally so the root span is always ended and
    # flushed — on a terminal return, a mid-turn error, OR an early client
    # disconnect (GeneratorExit raised at a yield). `turn_output` is the terminal
    # `done` event, recorded as the trace's output. Children (the brain's
    # decisions, searches, the grounding) attach to this span explicitly.
    span = tracing.start_turn(
        name="agent-find",
        input=request,
        session_id=session_id,
        tags=[t for t in (manual, active) if t],
        metadata={
            "manual": manual,
            "agent_model": active,
            "k": int(top_k),
            "thinking": bool(ground_thinking),
            "viewer_pages": shown_pages,
        },
    )
    turn_output = None
    try:
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
                # Re-ask, but RESTATE the request at the end so the most recent
                # (highest-attention) text is the mechanic's actual ask — not a
                # generic format scold that lets a 1B drift onto whatever is
                # printed on the page currently on screen. Deliberately no "copy
                # from the page text" line here: that guidance is for circling,
                # and it was steering search queries onto leftover on-screen text.
                messages.append(
                    minicpm_agent.tool_result_message(
                        "That was not one complete JSON object — it must start with "
                        "{ and end with }. Reply with ONE complete JSON object and "
                        'nothing else, e.g. {"tool": "search", "query": "fuel '
                        f'filter"}}. The mechanic asked: {request!r} — answer THAT.'
                    )
                )
                continue
            messages.append(minicpm_agent.assistant_action_message(tool))

            if tool["tool"] == "go_to_page":
                page = tool["page"]
                n = page_count(visual_store.pdf_path(doc_id))
                if not 1 <= page <= n:
                    messages.append(
                        minicpm_agent.tool_result_message(
                            f"There is no page {page}; this manual has pages 1–{n}. "
                            "Pick a page in range, or search."
                        )
                    )
                    continue
                yield {"type": "step", "tool": "go_to_page", "page": page}
                turn_output = {"type": "done", "kind": "navigate", "nav": "page",
                               "page": page, "title": f"Page {page}"}
                yield turn_output
                return

            if tool["tool"] == "search":
                query = tool["query"]
                yield {"type": "step", "tool": "search", "query": query}
                yield {"type": "status", "text": f"Searching for “{query}”…"}
                # k (the viewer's slider) is the shortlist size; ColEmbed's top page
                # is the one shown. A 1B text rerank measured WORSE than raw ColEmbed
                # top-1 (0.68 vs 0.84 hit@1) — visual late interaction already ranks
                # these (figure-heavy) pages better than re-judging from page text.
                with tracing.retriever("search", input=query,
                                       metadata={"retriever": "colembed", "k": int(top_k)}) as rsp:
                    hits = maxsim_search(query, visual_store, doc_ids, top_k)
                    if rsp is not None:
                        rsp.update(output=[{"page": p, "score": round(float(s), 4)}
                                           for _, p, s in hits])
                log.info("search(%r) → %s", query, [(p, round(s, 3)) for _, p, s in hits])
                if not hits:
                    messages.append(
                        minicpm_agent.tool_result_message(f"Search for {query!r} found nothing.")
                    )
                    continue
                yield from present_hits(hits, "search:" + " ".join(query.lower().split()))
                continue

            if tool["tool"] == "find_answer":
                query = tool["query"]
                yield {"type": "step", "tool": "find_answer", "query": query}
                yield {"type": "status", "text": f"Looking up “{query}”…"}
                # Dense retrieval over the PARSED chunks (text/semantic) — the index
                # the parsed store was built for. A fact lookup ("what fuel does it
                # take") is a TEXT match: ColEmbed ranks pages by VISUAL similarity
                # and misses the plain specs page, so fact questions route here. Same
                # (doc_id, page, score) shape as maxsim_search; the agent then circles
                # the answering line on the page shown.
                with tracing.retriever("find_answer", input=query,
                                       metadata={"retriever": "parsed-dense", "k": int(top_k)}) as rsp:
                    hits = retrieve_pages(query, parsed_store, doc_ids, top_k)
                    if rsp is not None:
                        rsp.update(output=[{"page": p, "score": round(float(s), 4)}
                                           for _, p, s in hits])
                log.info("find_answer(%r) → %s", query, [(p, round(s, 3)) for _, p, s in hits])
                if not hits:
                    messages.append(
                        minicpm_agent.tool_result_message(f"Looking up {query!r} found nothing.")
                    )
                    continue
                yield from present_hits(hits, "answer:" + " ".join(query.lower().split()))
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
                box, braw = minicpm.ground_box(img, target, enable_thinking=ground_thinking)
                # Heaviest op of the turn — the VLM vision encoder runs on a full-res
                # page. peak here is the turn's activation high-water (since
                # turn-start), the number that decides whether a big brain still fits.
                log_vram("after-ground")
                log.info("ground_box(%r) on p.%d → %s | raw=%r",
                         target, page, box, braw[:200])
                # The VLM couldn't find the target on this page — almost always
                # because it's on a DIFFERENT page (the agent circled too early).
                # Don't end the turn with an empty pin: push it to relocate and try
                # again. Only fall through to showing the page un-pinned once we've
                # already missed this exact (page, target) — a repeat means retrying
                # here won't help, same guard as the no-op search.
                tkey = (page, " ".join(target.lower().split()))
                if box is None and tkey not in ground_failed:
                    ground_failed.add(tkey)
                    messages.append(
                        minicpm_agent.tool_result_message(
                            minicpm_agent.ground_failed_message(request, target, page)
                        )
                    )
                    continue
                turn_output = {
                    "type": "done",
                    "kind": "point",
                    "found": True,
                    "target": target,
                    "page": page,
                    "bbox": [round(v) for v in box] if box is not None else None,
                    # The pixel size of the image the box was GROUNDED on — the bbox
                    # is in this coordinate space. The frontend sizes its SVG viewBox
                    # from this (not the browser-loaded <img>), so the circle lands
                    # correctly even if the displayed page PNG is served at a
                    # different/stale resolution than this grounding render.
                    "dims": [img.width, img.height],
                    # the VLM's raw grounding reply — diagnostic only (helps explain
                    # where/why a box landed); shown in the trace view.
                    "ground_raw": braw[:300],
                }
                yield turn_output
                return

            if tool["tool"] == "done":
                turn_output = {"type": "done", "kind": "reply",
                               "message": tool.get("message") or "Done."}
                yield turn_output
                return

        turn_output = {
            "type": "done",
            "kind": "reply",
            "message": "I went in circles on that one — try rephrasing?",
        }
        yield turn_output
    finally:
        tracing.finish_turn(span, output=turn_output)


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
        viewer: dict | None = None,
        history: list | None = None,
        ground_thinking: bool | None = None,
        agent_model: str | None = None,
        vram_log: bool = False,
        session_id: str | None = None,
    ):
        """One streamed agent turn (the event generator of agent_events).
        vram_log forwards the UI's VRAM-logging toggle to the probe; session_id
        groups a page session's turns together in Langfuse."""
        request = (request or "").strip()
        if not request:
            raise ValueError("Tell me what to find.")
        docs = visual_store.list_docs()
        if not docs:
            raise ValueError("No manuals in this library yet.")
        names = {d["doc_id"]: d["name"] for d in docs}
        return agent_events(
            request, visual_store, parsed_store, doc_ids or list(names),
            int(top_k), names, viewer, history, ground_thinking,
            agent_model, vram_log, session_id,
        )
