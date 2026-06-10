"""Ask pipeline: question -> MaxSim retrieval over the store -> top-K page
images -> MiniCPM answer grounded in those pages."""

from __future__ import annotations

from core.pdf import render_page
from core.store import Store
from models.colembed import ColEmbed
from models.minicpm import MiniCPM


class AskPipeline:
    def __init__(self, embedder: ColEmbed, store: Store, llm: MiniCPM):
        self.embedder = embedder
        self.store = store
        self.llm = llm

    def run(self, question: str, doc_ids: list[str] | None, top_k: int):
        """Return (answer markdown, gallery items [(image, caption)])."""
        question = (question or "").strip()
        if not question:
            raise ValueError("Please enter a question.")
        docs = self.store.list_docs()
        if not docs:
            raise ValueError("No manuals indexed yet — add one in the Library tab.")

        hits = self.embedder.search(question, self.store, doc_ids or None, int(top_k))
        names = {d["doc_id"]: d["name"] for d in docs}
        pages = [
            (f"{names[doc_id]} — p.{page}", render_page(self.store.pdf_path(doc_id), page), score)
            for doc_id, page, score in hits
        ]
        answer = self.llm.answer(question, [(label, img) for label, img, _ in pages])
        gallery = [(img, f"{label} (score {score:.1f})") for label, img, score in pages]
        return answer, gallery
