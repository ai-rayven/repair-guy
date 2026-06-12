"""Parsed ask pipeline: question -> dense cosine retrieval over chunks ->
parent pages -> MiniCPM answer grounded in those pages.

Retrieval is parent-document style: chunks (sections / figure descriptions /
table descriptions) are what's scored, but MiniCPM reads the FULL pages the
top chunks came from, so it sees figures and layout the chunk text only
summarizes.

Like the visual pipeline, the whole question runs in ONE @spaces.GPU call
(query embedding + scoring + page rendering + answer generation).
"""

from __future__ import annotations

import numpy as np
import spaces

from core.constants import ASK_GPU_DURATION, PARSED_TOP_CHUNKS
from core.parsed_store import ParsedStore
from core.pdf import render_page
from models.minicpm import generate_answer
from models.nemotron_embed import embed_query


def _chunk_pages(chunk: dict) -> list[int]:
    return chunk["pages"] if chunk["type"] == "section" else [chunk["page"]]


def retrieve_pages(
    question: str, store: ParsedStore, doc_ids: list[str], top_k: int
) -> list[tuple[str, int, float]]:
    """Top-K (doc_id, page_num, score): chunks scored by cosine, then the
    parent-document step where best chunks vote for pages, budgeted to top_k.
    Same shape as the visual side's maxsim_search, so the two retrievers are
    directly comparable (scripts/eval_modal.py relies on this). Must run on
    GPU (called from within a @spaces.GPU context)."""
    q = embed_query(question)  # [dim] float32, normalized

    hits = []  # (score, doc_id, chunk)
    for doc_id in doc_ids:
        if not store.exists(doc_id):  # e.g. deleted while still selected in the UI
            continue
        chunks, embeddings = store.load(doc_id)
        scores = embeddings.astype(np.float32) @ q  # cosine: both sides normalized
        for i in np.argsort(scores)[::-1][:PARSED_TOP_CHUNKS]:
            hits.append((float(scores[i]), doc_id, chunks[i]))
    hits.sort(key=lambda h: h[0], reverse=True)
    hits = hits[:PARSED_TOP_CHUNKS]

    page_refs: list[tuple[str, int]] = []
    page_score: dict[tuple[str, int], float] = {}
    for score, doc_id, chunk in hits:
        for page in _chunk_pages(chunk):
            ref = (doc_id, page)
            if ref not in page_score:
                page_refs.append(ref)
                page_score[ref] = score
    return [(doc_id, page, page_score[(doc_id, page)]) for doc_id, page in page_refs[:top_k]]


@spaces.GPU(duration=ASK_GPU_DURATION)
def _ask_on_gpu(
    question: str,
    store: ParsedStore,
    doc_ids: list[str],
    top_k: int,
    names: dict[str, str],
):
    refs = retrieve_pages(question, store, doc_ids, top_k)
    pages = [
        (f"{names[doc_id]} — p.{page}", render_page(store.pdf_path(doc_id), page))
        for doc_id, page, _ in refs
    ]
    answer = generate_answer(question, pages)
    gallery = [
        (img, f"{label} (cosine {score:.3f})")
        for (label, img), (_, _, score) in zip(pages, refs)
    ]
    return answer, gallery


class ParsedAskPipeline:
    """Stateless: the store is passed per call."""

    def run(self, store: ParsedStore, question: str, doc_ids: list[str] | None, top_k: int):
        """Return (answer markdown, gallery items [(image, caption)])."""
        question = (question or "").strip()
        if not question:
            raise ValueError("Please enter a question.")
        docs = store.list_docs()
        if not docs:
            raise ValueError("No manuals in this library yet.")
        names = {d["doc_id"]: d["name"] for d in docs}
        doc_ids = doc_ids or list(names)
        return _ask_on_gpu(question, store, doc_ids, int(top_k), names)
