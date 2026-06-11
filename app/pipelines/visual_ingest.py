"""Visual ingest pipeline: PDF -> page images -> ColEmbed embeddings -> store.

Pages are embedded in chunks of EMBED_PAGES_PER_CALL so each GPU call stays
short; progress events are yielded between chunks so callers can stream them.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from core.constants import EMBED_PAGES_PER_CALL, RENDER_DPI
from core.pdf import page_count, render_pages
from core.store import slugify
from core.visual_store import VisualStore
from models.colembed import ColEmbed


class VisualIngestPipeline:
    def __init__(self, embedder: ColEmbed, store: VisualStore):
        self.embedder = embedder
        self.store = store

    def run(
        self, pdf_path: str | None, doc_name: str = "", max_pages: int | None = None
    ) -> Iterator[tuple]:
        """Index one PDF. Generator yielding ("progress", pages_done, total)
        after each embedded chunk, then ("done", doc summary dict) last.
        Re-indexing a manual with the same name overwrites it."""
        if not pdf_path:
            raise ValueError("Please provide a PDF first.")
        if not pdf_path.lower().endswith(".pdf"):
            raise ValueError("Only PDFs can be indexed.")

        name = doc_name.strip() or (
            os.path.splitext(os.path.basename(pdf_path))[0].replace("_", " ")
        )
        doc_id = slugify(name)
        total = page_count(pdf_path)
        if max_pages and total > max_pages:
            raise ValueError(
                f"This PDF has {total} pages — over the {max_pages}-page cap."
            )

        writer = self.store.create(doc_id, name, pdf_path, RENDER_DPI, self.embedder.MODEL_ID)
        try:
            for start in range(1, total + 1, EMBED_PAGES_PER_CALL):
                nums = list(range(start, min(start + EMBED_PAGES_PER_CALL, total + 1)))
                images = render_pages(pdf_path, nums)
                for num, emb in zip(nums, self.embedder.embed_pages(images)):
                    writer.add_page(num, emb)
                yield ("progress", nums[-1], total)
            writer.finalize()
        except BaseException:
            writer.abort()
            raise
        yield ("done", {"doc_id": doc_id, "name": name, "pages": total})
