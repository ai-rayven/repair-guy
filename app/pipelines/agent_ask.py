"""Agent ask pipeline: multi-turn chat where MiniCPM drives retrieval itself.

Instead of the one-shot retrieve-then-answer flow, the model converses: it
sees the (text-only) chat history plus the new question and replies either
with a tool call — search_docs (semantic search over the manual, the same
retrievers the one-shot pipelines use) or show_page (a pure UI effect on the
viewer pane) — or with the final answer. Tool results are fed back into the
conversation (search results as labeled page images) and the loop continues,
bounded by AGENT_MAX_STEPS.

The whole turn streams as events and runs in ONE @spaces.GPU call (generators
are supported): every generation, retrieval and page render between them, and
— when the text history has outgrown HISTORY_TOKEN_BUDGET — a summarization
pass first.

Events yielded (PIL images included; app.py converts them for the wire):
  {"type": "status", "text", "kind"?}                    progress for the UI
  {"type": "tool_call", "tool", "args"}                  model called a tool
  {"type": "tool_result", "tool", ...}                   search: gallery
                                                         [(img, caption)] +
                                                         page_refs; show_page:
                                                         page + doc_id
  {"type": "answer", "answer", "history", "summarized"}  final, with the
                                                         updated text history

History is the client's: a [{role, content}] text transcript sent with each
turn and returned updated (past page images are collapsed to a bracketed tool
trace inside the assistant turns — at ~600-1000 tokens per page image they
cannot live in history under chat()'s 16384-token input cap).
"""

from __future__ import annotations

import spaces

from core.constants import (
    AGENT_GPU_DURATION,
    AGENT_MAX_STEPS,
    HISTORY_KEEP_MESSAGES,
    HISTORY_TOKEN_BUDGET,
)
from core.pdf import render_page
from models import minicpm

GIVE_UP_ANSWER = (
    "I couldn't put together a grounded answer within my tool budget — try "
    "rephrasing the question or pointing me at a section of the manual."
)


def _searcher(approach: str):
    """Lazy import: parsed_ask/visual_ask import this module's pipeline class,
    and both retriever modules load models at import."""
    if approach == "visual":
        from models.colembed import maxsim_search

        return maxsim_search
    from pipelines.parsed_ask import retrieve_pages

    return retrieve_pages


@spaces.GPU(duration=AGENT_GPU_DURATION)
def agent_events(
    question: str,
    store,
    doc_ids: list[str],
    top_k: int,
    names: dict[str, str],
    history: list[dict],
    approach: str,
):
    """Yield the events of one chat turn (see module docstring)."""
    search = _searcher(approach)
    manual = names[doc_ids[0]]

    summarized = False
    if (
        len(history) > HISTORY_KEEP_MESSAGES
        and minicpm.history_tokens(history) > HISTORY_TOKEN_BUDGET
    ):
        yield {
            "type": "status",
            "kind": "summarizing",
            "text": "Summarizing earlier conversation…",
        }
        history = minicpm.summarize_history(history)
        summarized = True

    msgs = [{"role": m["role"], "content": [m["content"]]} for m in history]
    msgs.append({"role": "user", "content": [question]})

    trace: list[str] = []  # text record of tool calls for the durable history
    answer = None
    for _ in range(AGENT_MAX_STEPS):
        yield {"type": "status", "text": "Thinking…"}
        reply = minicpm.generate_step(msgs, manual)
        call = minicpm.parse_tool_call(reply)
        if call is None:
            answer = reply
            break
        msgs.append({"role": "assistant", "content": [reply]})

        if call["tool"] == "search_docs":
            query = str(call["args"].get("query") or "").strip() or question
            yield {"type": "tool_call", "tool": "search_docs", "args": {"query": query}}
            refs = search(query, store, doc_ids, top_k)
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
            content: list = [
                f'Result of search_docs("{query}") — the {len(pages)} most '
                "relevant pages, each preceded by its label:"
            ]
            for label, img in pages:
                content.append(f"[{label}]")
                content.append(img.convert("RGB"))
            content.append(
                "Answer the user's question using ONLY what is printed on "
                "these pages, or call another tool."
            )
            msgs.append({"role": "user", "content": content})
            labels = ", ".join(label for label, _ in pages)
            trace.append(f'[searched the manual for "{query}" → {labels or "no pages"}]')

        else:  # show_page
            try:
                page = int(call["args"].get("page"))
            except (TypeError, ValueError):
                page = 0
            if page < 1:
                msgs.append(
                    {"role": "user", "content": ["Invalid show_page call: give a 1-based page number."]}
                )
                continue
            yield {"type": "tool_call", "tool": "show_page", "args": {"page": page}}
            yield {"type": "tool_result", "tool": "show_page", "page": page, "doc_id": doc_ids[0]}
            msgs.append(
                {
                    "role": "user",
                    "content": [
                        f"Page {page} is now displayed in the user's viewer. "
                        "Continue — answer the user or call another tool."
                    ],
                }
            )
            trace.append(f"[displayed page {page} in the viewer]")

    if answer is None:  # tool budget exhausted: force a plain answer
        yield {"type": "status", "text": "Wrapping up…"}
        msgs.append(
            {
                "role": "user",
                "content": [
                    "You have used all your tool calls. Give your final answer "
                    "to the user now, in plain markdown — do not output JSON."
                ],
            }
        )
        answer = minicpm.generate_step(msgs, manual)
        if minicpm.parse_tool_call(answer) is not None:
            answer = GIVE_UP_ANSWER

    durable = ("\n".join(trace) + "\n\n" if trace else "") + answer
    yield {
        "type": "answer",
        "answer": answer,
        "history": history
        + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": durable},
        ],
        "summarized": summarized,
    }
