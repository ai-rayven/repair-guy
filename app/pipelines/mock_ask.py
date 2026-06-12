"""Mock store + ask pipeline for local UI iteration (MOCK_MODELS=1).

Drop any PDF into MOCK_PDF_DIR (default app/data/mock_pdfs/) and app.py serves a
canned answer grounded in real, rendered pages of that PDF — no GPU, no model
downloads, no HF library sync. The same MockStore instance backs both
approaches, so the manual dropdown, the PDF viewer, the cited-pages gallery and
the jump-to-page behaviour all exercise the real wiring; only the answer text
and page selection are faked.

Nothing here imports torch / spaces / the model modules (those load CUDA at
import), so this module is safe to load on a laptop.
"""

from __future__ import annotations

import glob
import hashlib
import os
import time

from core.constants import MOCK_PDF_DIR
from core.pdf import page_count, render_pages
from core.store import slugify

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

    def run_agent(
        self,
        store: MockStore,
        question: str,
        history: list[dict],
        doc_ids: list[str] | None,
        top_k: int,
    ):
        """Yield the same event sequence as pipelines/agent_ask.py — status,
        a fake search_docs call grounded in real rendered pages, a show_page
        call, then the answer with the updated text history. Every third turn
        the history is "summarized" so the UI's compaction status can be
        exercised. MOCK_DELAY (seconds) paces the events."""
        question = (question or "").strip()
        if not question:
            raise ValueError("Please enter a question.")
        doc_id, info = self._pick_doc(store, doc_ids)
        return self._agent_events(
            store, question, list(history or []), doc_id, info, int(top_k)
        )

    def _agent_events(self, store, question, history, doc_id, info, top_k):
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

        yield {"type": "status", "text": "Thinking…"}
        time.sleep(delay)

        query = " ".join(question.split()[:6])
        yield {"type": "tool_call", "tool": "search_docs", "args": {"query": query}}
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

        yield {"type": "status", "text": "Thinking…"}
        time.sleep(delay)
        yield {"type": "tool_call", "tool": "show_page", "args": {"page": pages[0]}}
        yield {"type": "tool_result", "tool": "show_page", "page": pages[0], "doc_id": doc_id}
        time.sleep(delay)

        answer = MOCK_ANSWER.format(label=labels[0])
        trace = (
            f'[searched the manual for "{query}" → {", ".join(labels)}]\n'
            f"[displayed page {pages[0]} in the viewer]"
        )
        yield {
            "type": "answer",
            "answer": answer,
            "history": history
            + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": trace + "\n\n" + answer},
            ],
            "summarized": summarized,
        }

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
