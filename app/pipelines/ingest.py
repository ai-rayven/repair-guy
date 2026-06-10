"""Ingest pipeline: PDF -> page images -> ColEmbed embeddings -> on-disk store.

Pages are embedded in chunks of EMBED_PAGES_PER_CALL so each ZeroGPU call stays
short; progress is reported between chunks.
"""

from __future__ import annotations

import os

from core.constants import EMBED_PAGES_PER_CALL, RENDER_DPI
from core.pdf import page_count, render_pages
from core.store import Store, slugify
from models.colembed import ColEmbed


class IngestPipeline:
    def __init__(self, embedder: ColEmbed, store: Store):
        self.embedder = embedder
        self.store = store

    def run(self, pdf_path: str | None, doc_name: str = "", progress=None) -> dict:
        """Index one PDF; returns the stored doc's summary. Re-indexing a manual
        with the same name overwrites it."""
        if not pdf_path:
            raise ValueError("Please upload a PDF first.")
        if not pdf_path.lower().endswith(".pdf"):
            raise ValueError("Only PDFs can be indexed.")

        name = doc_name.strip() or (
            os.path.splitext(os.path.basename(pdf_path))[0].replace("_", " ")
        )
        doc_id = slugify(name)
        total = page_count(pdf_path)

        writer = self.store.create(doc_id, name, pdf_path, RENDER_DPI, self.embedder.MODEL_ID)
        try:
            for start in range(1, total + 1, EMBED_PAGES_PER_CALL):
                nums = list(range(start, min(start + EMBED_PAGES_PER_CALL, total + 1)))
                images = render_pages(pdf_path, nums)
                for num, emb in zip(nums, self.embedder.embed_pages(images)):
                    writer.add_page(num, emb)
                if progress:
                    progress(nums[-1] / total, f"Embedded {nums[-1]}/{total} pages")
            writer.finalize()
        except BaseException:
            writer.abort()
            raise
        return {"doc_id": doc_id, "name": name, "pages": total}
