"""Shared on-disk document store conventions (base class for both approaches).

Layout: <store root>/<doc_id>/
    doc.pdf       original PDF, kept to re-render retrieved pages at answer time
    index.json    metadata; always contains "name" and "method", written LAST —
                  a directory without it is an interrupted ingest and is
                  ignored (and overwritten on the next attempt)
    ...           per-method data files (see VisualStore / ParsedStore)

The "method" field guards against pointing a store at the other method's
directory: docs whose method doesn't match are skipped loudly in list_docs
and rejected by meta()/exists().
"""

from __future__ import annotations

import json
import os
import re
import shutil

import numpy as np

EMB_DTYPE = np.float16


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "doc"


class DocStore:
    method: str = ""  # subclasses set "visual" / "parsed"

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def _dir(self, doc_id: str) -> str:
        return os.path.join(self.root, doc_id)

    def _read_meta(self, doc_id: str) -> dict | None:
        path = os.path.join(self._dir(doc_id), "index.json")
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return json.load(f)

    def exists(self, doc_id: str) -> bool:
        meta = self._read_meta(doc_id)
        return meta is not None and meta.get("method") == self.method

    def meta(self, doc_id: str) -> dict:
        meta = self._read_meta(doc_id)
        if meta is None:
            raise FileNotFoundError(f"No indexed doc at {self._dir(doc_id)}")
        if meta.get("method") != self.method:
            raise ValueError(
                f"{doc_id} was indexed with method={meta.get('method')!r}, "
                f"but this is a {self.method!r} store"
            )
        return meta

    def pdf_path(self, doc_id: str) -> str:
        return os.path.join(self._dir(doc_id), "doc.pdf")

    def _page_count(self, meta: dict) -> int:
        raise NotImplementedError

    def list_docs(self) -> list[dict]:
        docs = []
        for doc_id in sorted(os.listdir(self.root)):
            if not os.path.isdir(self._dir(doc_id)):
                continue
            meta = self._read_meta(doc_id)
            if meta is None:
                continue
            if meta.get("method") != self.method:
                print(
                    f"Skipping {doc_id}: method={meta.get('method')!r} in a "
                    f"{self.method!r} store"
                )
                continue
            size = sum(
                os.path.getsize(os.path.join(self._dir(doc_id), f))
                for f in os.listdir(self._dir(doc_id))
            )
            docs.append(
                {
                    "doc_id": doc_id,
                    "name": meta["name"],
                    "pages": self._page_count(meta),
                    "size_mb": size / 1e6,
                }
            )
        return docs

    def delete(self, doc_id: str) -> None:
        shutil.rmtree(self._dir(doc_id), ignore_errors=True)
