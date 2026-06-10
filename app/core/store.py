"""On-disk store of per-page ColEmbed token embeddings.

Layout: STORE_DIR/<doc_id>/
    doc.pdf          original PDF, kept to re-render retrieved pages at answer time
    embeddings.bin   raw float16 [total_tokens, dim], pages stored back to back
    index.json       manual name, dim, dpi, model id, per-page (offset, count)

embeddings.bin is read back as a numpy memmap, so scoring streams pages from
disk in batches without ever loading a whole document into memory. index.json
is written last, so a directory without it is an interrupted ingest and is
ignored (and overwritten on the next attempt).
"""

from __future__ import annotations

import json
import os
import re
import shutil

import numpy as np

from core.constants import STORE_DIR

EMB_DTYPE = np.float16


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "doc"


class DocWriter:
    """Streams one document's page embeddings to disk during ingest."""

    def __init__(self, doc_dir: str, name: str, pdf_path: str, dpi: int, model_id: str):
        self.doc_dir = doc_dir
        os.makedirs(doc_dir, exist_ok=True)
        shutil.copyfile(pdf_path, os.path.join(doc_dir, "doc.pdf"))
        self._bin = open(os.path.join(doc_dir, "embeddings.bin"), "wb")
        self._meta = {
            "name": name,
            "dpi": dpi,
            "model_id": model_id,
            "dim": None,
            "pages": [],
        }
        self._offset = 0

    def add_page(self, page_num: int, emb: np.ndarray) -> None:
        """emb: [n_tokens, dim] for one page, padding rows already removed."""
        emb = np.ascontiguousarray(emb, dtype=EMB_DTYPE)
        if self._meta["dim"] is None:
            self._meta["dim"] = int(emb.shape[1])
        self._bin.write(emb.tobytes())
        self._meta["pages"].append(
            {"page": page_num, "offset": self._offset, "count": int(emb.shape[0])}
        )
        self._offset += int(emb.shape[0])

    def finalize(self) -> None:
        self._bin.close()
        with open(os.path.join(self.doc_dir, "index.json"), "w") as f:
            json.dump(self._meta, f)

    def abort(self) -> None:
        self._bin.close()
        shutil.rmtree(self.doc_dir, ignore_errors=True)


class Store:
    def __init__(self, root: str = STORE_DIR):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def _dir(self, doc_id: str) -> str:
        return os.path.join(self.root, doc_id)

    def exists(self, doc_id: str) -> bool:
        return os.path.isfile(os.path.join(self._dir(doc_id), "index.json"))

    def meta(self, doc_id: str) -> dict:
        with open(os.path.join(self._dir(doc_id), "index.json")) as f:
            return json.load(f)

    def pdf_path(self, doc_id: str) -> str:
        return os.path.join(self._dir(doc_id), "doc.pdf")

    def list_docs(self) -> list[dict]:
        docs = []
        for doc_id in sorted(os.listdir(self.root)):
            if not self.exists(doc_id):
                continue
            meta = self.meta(doc_id)
            size = os.path.getsize(os.path.join(self._dir(doc_id), "embeddings.bin"))
            docs.append(
                {
                    "doc_id": doc_id,
                    "name": meta["name"],
                    "pages": len(meta["pages"]),
                    "size_mb": size / 1e6,
                }
            )
        return docs

    def create(self, doc_id: str, name: str, pdf_path: str, dpi: int, model_id: str) -> DocWriter:
        self.delete(doc_id)
        return DocWriter(self._dir(doc_id), name, pdf_path, dpi, model_id)

    def delete(self, doc_id: str) -> None:
        shutil.rmtree(self._dir(doc_id), ignore_errors=True)

    def iter_page_batches(self, doc_ids: list[str] | None = None, pages_per_batch: int = 32):
        """Yield (refs, embs): refs is [(doc_id, page_num)] and embs is a
        zero-padded float16 array [batch, max_tokens, dim].

        Zero rows are inert under MaxSim (the model zeroes padding before
        L2-normalizing real tokens), matching the reference scorer.
        """
        if doc_ids is None:
            doc_ids = [d["doc_id"] for d in self.list_docs()]
        for doc_id in doc_ids:
            if not self.exists(doc_id):  # e.g. deleted while still selected in the UI
                continue
            meta = self.meta(doc_id)
            pages, dim = meta["pages"], meta["dim"]
            if not pages:
                continue
            total = pages[-1]["offset"] + pages[-1]["count"]
            mm = np.memmap(
                os.path.join(self._dir(doc_id), "embeddings.bin"),
                dtype=EMB_DTYPE,
                mode="r",
                shape=(total, dim),
            )
            for i in range(0, len(pages), pages_per_batch):
                chunk = pages[i : i + pages_per_batch]
                t_max = max(p["count"] for p in chunk)
                out = np.zeros((len(chunk), t_max, dim), dtype=EMB_DTYPE)
                for j, p in enumerate(chunk):
                    out[j, : p["count"]] = mm[p["offset"] : p["offset"] + p["count"]]
                yield [(doc_id, p["page"]) for p in chunk], out
