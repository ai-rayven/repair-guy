"""Parsed-approach store: section/figure/table chunks + dense embeddings.

Per-doc files (on top of the DocStore conventions):
    chunks.json      [{chunk_id, type, heading, pages|page, bbox?, text}]
    embeddings.npy   float16 [n_chunks, dim], L2-normalized; row i = chunk i
    parsed.json      raw Nemotron Parse output per page — kept so chunking or
                     embedding can be redone later without re-parsing
    index.json       name, method, dpi, model id, dim, page_count

Unlike the visual store this is small (a 1000-page manual is a few MB of
vectors), so load() returns everything in memory and retrieval is one matmul.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np

from core.store import EMB_DTYPE, DocStore


class ParsedStore(DocStore):
    method = "parsed"

    def _page_count(self, meta: dict) -> int:
        return meta["page_count"]

    def save(
        self,
        doc_id: str,
        name: str,
        pdf_path: str,
        chunks: list[dict],
        embeddings: np.ndarray,
        page_count: int,
        dpi: int,
        model_id: str,
        parsed_pages: list[dict] | None = None,
    ) -> None:
        """Write one fully-ingested document atomically (index.json last).
        embeddings: [n_chunks, dim], row-aligned with chunks."""
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"{len(chunks)} chunks but {embeddings.shape[0]} embedding rows"
            )
        self.delete(doc_id)
        doc_dir = self._dir(doc_id)
        os.makedirs(doc_dir)
        try:
            shutil.copyfile(pdf_path, os.path.join(doc_dir, "doc.pdf"))
            chunks = [{"chunk_id": i, **c} for i, c in enumerate(chunks)]
            with open(os.path.join(doc_dir, "chunks.json"), "w") as f:
                json.dump(chunks, f)
            np.save(
                os.path.join(doc_dir, "embeddings.npy"),
                np.ascontiguousarray(embeddings, dtype=EMB_DTYPE),
            )
            if parsed_pages is not None:
                with open(os.path.join(doc_dir, "parsed.json"), "w") as f:
                    json.dump(parsed_pages, f)
            meta = {
                "name": name,
                "method": self.method,
                "dpi": dpi,
                "model_id": model_id,
                "dim": int(embeddings.shape[1]),
                "page_count": page_count,
            }
            with open(os.path.join(doc_dir, "index.json"), "w") as f:
                json.dump(meta, f)
        except BaseException:
            shutil.rmtree(doc_dir, ignore_errors=True)
            raise

    def chunks(self, doc_id: str) -> list[dict]:
        """Just the chunk dicts (no embeddings) — section/element structure
        for the CPU navigation routes."""
        with open(os.path.join(self._dir(doc_id), "chunks.json")) as f:
            return json.load(f)

    def load(self, doc_id: str) -> tuple[list[dict], np.ndarray]:
        """Return (chunks, embeddings [n_chunks, dim] float16)."""
        doc_dir = self._dir(doc_id)
        with open(os.path.join(doc_dir, "chunks.json")) as f:
            chunks = json.load(f)
        embeddings = np.load(os.path.join(doc_dir, "embeddings.npy"))
        return chunks, embeddings
