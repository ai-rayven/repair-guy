"""Visual ask pipeline: question -> MaxSim retrieval over page embeddings ->
top-K page images -> MiniCPM answer grounded in those pages.

The whole question runs in ONE @spaces.GPU call (query embedding + MaxSim +
page rendering + answer generation), so each question pays the ZeroGPU
allocation wait once.
"""

from __future__ import annotations

import spaces

from core.constants import ASK_GPU_DURATION
from core.pdf import render_page
from core.visual_store import VisualStore
from models.colembed import maxsim_search
from models.minicpm import generate_answer
from pipelines.chat_ask import chat_events


@spaces.GPU(duration=ASK_GPU_DURATION)
def _ask_on_gpu(
    question: str,
    store: VisualStore,
    doc_ids: list[str] | None,
    top_k: int,
    names: dict[str, str],
):
    hits = maxsim_search(question, store, doc_ids, top_k)
    pages = [
        (f"{names[doc_id]} — p.{page}", render_page(store.pdf_path(doc_id), page), score)
        for doc_id, page, score in hits
    ]
    answer = generate_answer(question, [(label, img) for label, img, _ in pages])
    gallery = [(img, f"{label} (score {score:.1f})") for label, img, score in pages]
    page_refs = [(doc_id, page) for doc_id, page, _ in hits]
    return answer, gallery, page_refs


class VisualAskPipeline:
    """Stateless: the store is passed per call."""

    def run(self, store: VisualStore, question: str, doc_ids: list[str] | None, top_k: int):
        """Return (answer markdown, gallery items [(image, caption)], page_refs
        [(doc_id, page_num)] for the retrieved pages, in answer order)."""
        question = (question or "").strip()
        if not question:
            raise ValueError("Please enter a question.")
        docs = store.list_docs()
        if not docs:
            raise ValueError("No manuals in this library yet.")
        names = {d["doc_id"]: d["name"] for d in docs}
        return _ask_on_gpu(question, store, doc_ids or None, int(top_k), names)

    def run_chat(
        self,
        store: VisualStore,
        question: str,
        history: list[dict],
        doc_ids: list[str] | None,
        top_k: int,
        viewer: dict | None = None,
    ):
        """One streamed Q&A turn: the event generator of chat_ask.py, with
        MaxSim as the search step."""
        question = (question or "").strip()
        if not question:
            raise ValueError("Please enter a question.")
        docs = store.list_docs()
        if not docs:
            raise ValueError("No manuals in this library yet.")
        names = {d["doc_id"]: d["name"] for d in docs}
        return chat_events(
            question, store, doc_ids or list(names), int(top_k), names, history,
            "visual", viewer,
        )
