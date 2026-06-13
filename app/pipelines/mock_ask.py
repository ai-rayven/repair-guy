"""Mock store + ask pipeline for local UI iteration (MOCK_MODELS=1).

Drop any PDF into MOCK_PDF_DIR (default app/data/mock_pdfs/) and app.py serves a
canned answer grounded in real, rendered pages of that PDF — no GPU, no model
downloads, no HF library sync. The same MockStore instance backs both
approaches, so the manual dropdown, the page viewer, navigation, circling, the
cited-pages gallery and the jump-to-page behaviour all exercise the real
wiring; only the answer text, the section structure and the page/bbox picks
are faked.

Nothing here imports torch / spaces / the model modules (those load CUDA at
import), so this module is safe to load on a laptop.
"""

from __future__ import annotations

import glob
import hashlib
import os
import time

from core.constants import MOCK_PDF_DIR
from core.pdf import page_count, page_size, render_pages
from core.store import slugify

# Canned section titles spread evenly over each PDF's pages, realistic enough
# to exercise "go to <section>" fuzzy matching in local tests.
MOCK_SECTION_TITLES = [
    "General Information",
    "Engine Mechanical System",
    "Engine Electrical System",
    "Fuel System",
    "Cooling System",
    "Transmission System",
    "Brake System — Bleeding and Adjustment",
    "Steering System",
    "Suspension System",
    "Body Electrical System",
]

MOCK_ANSWER = (
    "🧪 **Mock mode** — canned answer for UI iteration, not a real model "
    "response.\n\n"
    "1. Loosen the three retaining bolts in a star pattern (reassembly torque: "
    "24 N·m / 18 ft-lb).\n"
    "2. Withdraw the assembly and inspect the seal lip for wear or scoring.\n"
    "3. Refit in reverse order, checking clearances against the spec table.\n\n"
    "The pages this answer *would* be grounded in are shown on the right.\n\n"
    "({label})"
)


class MockStore:
    """A folder of PDFs exposed through the slice of the DocStore API that
    app.py and MockAskPipeline use. doc_id is the slug of the file name."""

    def __init__(self, pdf_dir: str = MOCK_PDF_DIR):
        self.pdf_dir = os.path.abspath(pdf_dir)
        os.makedirs(self.pdf_dir, exist_ok=True)

    def _docs(self) -> dict[str, dict]:
        """Re-scanned on every call so PDFs dropped in while the app is running
        appear after a 🔄 Sync library click (which just rebuilds the dropdown)."""
        docs: dict[str, dict] = {}
        for path in sorted(glob.glob(os.path.join(self.pdf_dir, "*.pdf"))):
            name = os.path.splitext(os.path.basename(path))[0]
            docs[slugify(name)] = {"name": name, "path": path}
        return docs

    def list_docs(self) -> list[dict]:
        return [
            {
                "doc_id": doc_id,
                "name": info["name"],
                "pages": page_count(info["path"]),
                "size_mb": os.path.getsize(info["path"]) / 1e6,
            }
            for doc_id, info in self._docs().items()
        ]

    def exists(self, doc_id: str) -> bool:
        return doc_id in self._docs()

    def pdf_path(self, doc_id: str) -> str | None:
        info = self._docs().get(doc_id)
        return info["path"] if info else None

    def sections(self, doc_id: str) -> list[dict]:
        """Canned sections spread evenly over the PDF's real pages — same
        shape as core.sections.sections_from_chunks."""
        info = self._docs().get(doc_id)
        if not info:
            return []
        n = page_count(info["path"])
        k = min(len(MOCK_SECTION_TITLES), n)
        starts = [round(i * n / k) + 1 for i in range(k)]
        return [
            {
                "title": MOCK_SECTION_TITLES[i],
                "page_start": starts[i],
                "page_end": (starts[i + 1] - 1) if i + 1 < k else n,
            }
            for i in range(k)
        ]

    def elements(self, doc_id: str) -> list[dict]:
        """Canned figure/table chunks (one per section, bbox in rendered-page
        pixels) so /locate works in mock mode."""
        info = self._docs().get(doc_id)
        if not info:
            return []
        w, h = page_size(info["path"], 1)
        out = []
        for i, s in enumerate(self.sections(doc_id)):
            kind = "figure" if i % 2 == 0 else "table"
            out.append(
                {
                    "type": kind,
                    "heading": s["title"],
                    "page": s["page_start"],
                    "bbox": [w * 0.15, h * 0.25, w * 0.85, h * 0.6],
                    "text": f"Mock {kind}: exploded view and specifications "
                    f"for the {s['title'].lower()}.",
                }
            )
        return out


class MockAskPipeline:
    """Stateless mock matching the real ask pipelines' contracts: picks a
    deterministic spread of real pages and returns a canned answer."""

    def run(self, store: MockStore, question: str, doc_ids: list[str] | None, top_k: int):
        """Return (answer markdown, gallery [(image, caption)], page_refs
        [(doc_id, page)]) — same shape as the real ask pipelines."""
        question = (question or "").strip()
        if not question:
            raise ValueError("Please enter a question.")
        # Real questions take seconds on the GPU; the mock is instant, so the
        # loading indicator never shows. MOCK_DELAY (seconds) fakes that latency
        # for local UI work — e.g. MOCK_DELAY=2.
        time.sleep(float(os.environ.get("MOCK_DELAY", "0")))
        doc_id, info = self._pick_doc(store, doc_ids)
        pages = self._pick_pages(question, info["pages"], int(top_k))
        images = render_pages(store.pdf_path(doc_id), pages)
        labels = [f"{info['name']} — p.{p}" for p in pages]
        answer = MOCK_ANSWER.format(label=labels[0])
        gallery = [(img, f"{label} (mock)") for label, img in zip(labels, images)]
        page_refs = [(doc_id, p) for p in pages]
        return answer, gallery, page_refs

    def run_chat(
        self,
        store: MockStore,
        question: str,
        history: list[dict],
        doc_ids: list[str] | None,
        top_k: int,
        viewer: dict | None = None,
    ):
        """Yield the same event sequence as pipelines/chat_ask.py — a search
        grounded in real rendered pages, then the answer with the updated text
        history. Every third turn the history is "summarized" so the UI's
        compaction status can be exercised. MOCK_DELAY (seconds) paces the
        events."""
        question = (question or "").strip()
        if not question:
            raise ValueError("Please enter a question.")
        doc_id, info = self._pick_doc(store, doc_ids)
        return self._chat_events(
            store, question, list(history or []), doc_id, info, int(top_k)
        )

    def _chat_events(self, store, question, history, doc_id, info, top_k):
        delay = float(os.environ.get("MOCK_DELAY", "0"))

        summarized = False
        if len(history) >= 6:  # 3 stored turns: mimic the token-budget trigger
            yield {
                "type": "status",
                "kind": "summarizing",
                "text": "Summarizing earlier conversation…",
            }
            time.sleep(delay)
            history = [
                {
                    "role": "user",
                    "content": "Summary of our conversation so far (for your "
                    f"reference):\n(mock summary of {len(history) - 4} earlier messages)",
                },
                {"role": "assistant", "content": "Got it — I'll keep that context in mind."},
            ] + history[-4:]
            summarized = True

        yield {"type": "tool_call", "tool": "search_docs", "args": {"query": question}}
        time.sleep(delay)
        pages = self._pick_pages(question, info["pages"], top_k)
        images = render_pages(store.pdf_path(doc_id), pages)
        labels = [f"{info['name']} — p.{p}" for p in pages]
        yield {
            "type": "tool_result",
            "tool": "search_docs",
            "gallery": [(img, f"{label} (mock)") for label, img in zip(labels, images)],
            "page_refs": [(doc_id, p) for p in pages],
        }
        yield {"type": "status", "text": "Reading the pages…"}
        time.sleep(delay)

        answer = MOCK_ANSWER.format(label=labels[0])
        yield {
            "type": "answer",
            "answer": answer,
            "history": history
            + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "summarized": summarized,
            "grounded_page": pages[0],
        }

    @staticmethod
    def point(store: MockStore, doc_id: str, page: int, query: str) -> dict | None:
        """Mock visual grounding: a deterministic, query-dependent box on the
        requested page, in rendered-page pixels (None for one query in ~8, so
        the not-found path can be exercised with e.g. 'circle the xyzzy')."""
        path = store.pdf_path(doc_id)
        if not path:
            return None
        seed = int(hashlib.sha1(query.encode()).hexdigest(), 16)
        if seed % 8 == 7:
            return None
        w, h = page_size(path, page)
        x1 = w * (0.1 + (seed % 5) * 0.1)
        y1 = h * (0.15 + (seed // 5 % 5) * 0.12)
        return {"page": page, "bbox": [round(x1), round(y1), round(x1 + w * 0.3), round(y1 + h * 0.18)]}

    @staticmethod
    def _pick_doc(store: MockStore, doc_ids: list[str] | None):
        docs = {d["doc_id"]: d for d in store.list_docs()}
        if not docs:
            raise ValueError(
                f"No PDFs in the mock library yet — drop one into {store.pdf_dir}."
            )
        doc_id = (doc_ids or list(docs))[0]
        if doc_id not in docs:
            raise ValueError("That manual isn't in the mock library.")
        return doc_id, docs[doc_id]

    @staticmethod
    def _pick_pages(question: str, n_pages: int, top_k: int) -> list[int]:
        """A deterministic, question-dependent spread of 1-based pages, so
        different questions cite different pages (nice for clicking through the
        viewer) while the same question is stable across reloads."""
        k = max(1, min(top_k, n_pages))
        seed = int(hashlib.sha1(question.encode()).hexdigest(), 16)
        start = seed % n_pages  # 0-based anchor
        step = max(1, n_pages // k)
        out: list[int] = []
        for i in range(k):
            page = (start + i * step) % n_pages + 1  # back to 1-based
            if page not in out:
                out.append(page)
        return out
