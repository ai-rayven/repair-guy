"""Mock store + find-and-point pipeline for local UI iteration (MOCK_MODELS=1).

Drop any PDF into MOCK_PDF_DIR (default app/data/mock_pdfs/) and the whole
find-and-point UX runs over real, rendered pages of that PDF — no GPU, no
model downloads, no HF library sync. The same MockStore instance backs both
approaches, so the manual dropdown, the page viewer, navigation, the three
router branches (go_to_section / point_here / search→classify→circle), the
circle overlay and the candidate-pages strip all exercise the real wiring;
only the router's tool choice, the page/bbox picks and the page classification
are faked (by simple keyword heuristics, so the branches are predictable in
tests).

Nothing here imports torch / spaces / the model modules (those load CUDA at
import), so this module is safe to load on a laptop.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import time

from core.constants import MOCK_PDF_DIR
from core.pdf import page_count, page_size, render_pages
from core.sections import match_section
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

    def run_find(
        self,
        visual_store: MockStore,
        parsed_store: MockStore,
        request: str,
        doc_ids: list[str] | None,
        top_k: int,
        sections: list[dict],
        viewer: dict | None = None,
        history: list | None = None,
    ):
        """Yield the same event sequence as pipelines/agent_ask.py, with a
        keyword-driven stand-in for the agent's tool choice. Both stores are the
        one MockStore; history is ignored. MOCK_DELAY (seconds) paces events."""
        request = (request or "").strip()
        if not request:
            raise ValueError("Tell me what to find.")
        doc_id, info = self._pick_doc(visual_store, doc_ids)
        return self._find_events(
            visual_store, request, doc_id, info, int(top_k), sections or [], viewer or {}
        )

    def _find_events(self, store, request, doc_id, info, top_k, sections, viewer):
        delay = float(os.environ.get("MOCK_DELAY", "0"))
        cur = max(1, int(viewer.get("page") or 1))
        shown = [int(p) for p in (viewer.get("pages") or []) if str(p).isdigit()][:2] or [cur]
        prompt = self._mock_prompt(request, sections, shown)
        yield {"type": "status", "text": "Thinking…"}
        time.sleep(delay)

        kind, a, b = self._mock_route(request, sections)

        if kind == "go_to_section":
            yield self._trace(0, {"tool": "go_to_section", "section": b}, prompt)
            yield {"type": "step", "tool": "go_to_section", "title": b, "page": a}
            yield {"type": "done", "kind": "navigate", "nav": "section",
                   "page": a, "title": b}
            return

        if kind == "go_to_page":
            page = max(1, min(int(a), info["pages"]))
            yield self._trace(0, {"tool": "go_to_page", "page": page}, prompt)
            yield {"type": "step", "tool": "go_to_page", "page": page}
            yield {"type": "done", "kind": "navigate", "nav": "page",
                   "page": page, "title": f"Page {page}"}
            return

        if kind == "point_here":
            target = a
            yield self._trace(0, {"tool": "circle", "target": target}, prompt)
            yield from self._circle(store, doc_id, cur, target, delay)
            return

        # search → show the best page → circle the target on it (mirrors the
        # agent's search-then-circle), unless a magic target hits the give-up path
        target, query = a, b
        yield self._trace(0, {"tool": "search", "query": query}, prompt)
        yield {"type": "step", "tool": "search", "query": query}
        pages = self._pick_pages(request, info["pages"], top_k)
        images = render_pages(store.pdf_path(doc_id), pages)
        yield {
            "type": "tool_result",
            "tool": "search_docs",
            "gallery": [
                (img, f"{info['name']} — p.{p} (mock)")
                for p, img in zip(pages, images)
            ],
            "page_refs": [(doc_id, p) for p in pages],
        }
        yield {"type": "status", "text": f"Reading {len(pages)} candidate pages…"}
        time.sleep(delay)

        if any(w in target.lower() for w in ("xyzzy", "flux capacitor", "nonexistent")):
            msg = f"Couldn't find “{target}” in this manual."
            yield self._trace(1, {"tool": "done", "message": msg}, prompt)
            yield {"type": "done", "kind": "reply", "message": msg}
            return
        best = pages[0]
        yield {"type": "found", "page": best}
        yield self._trace(1, {"tool": "circle", "target": target}, prompt)
        yield from self._circle(store, doc_id, best, target, delay)

    def _circle(self, store, doc_id, page, target, delay):
        """Emit the circle step + terminal point event for a target on a page."""
        yield {"type": "step", "tool": "circle", "target": target, "page": page}
        yield {"type": "status", "text": "Pinning it down…"}
        time.sleep(delay)
        box = self._mock_box(store, doc_id, page, target)
        yield {"type": "done", "kind": "point", "found": True,
               "target": target, "page": page, "bbox": box,
               "ground_raw": f"(mock) grounding for {target!r} → {box}"}

    @staticmethod
    def _trace(step: int, tool: dict, prompt: str = "") -> dict:
        """A mock diagnostic 'trace' event mirroring agent_ask's: the prompt fed
        in, the raw model reply (here just the tool JSON) + the parsed tool, for
        the UI trace view."""
        return {"type": "trace", "step": step, "tool": tool,
                "raw": json.dumps(tool, separators=(",", ":")), "prompt": prompt}

    @staticmethod
    def _mock_prompt(request: str, sections: list[dict], shown: list[int]) -> str:
        """A representative stand-in for the rendered chat prompt, so the
        Diagnostics 'prompt' view is exercisable in MOCK_MODELS=1. Not the real
        template — just the same shape (system rules + the on-screen pages +
        TOC + request)."""
        toc = "\n".join(
            f"{i + 1}. {s['title']} (p.{s.get('page') or s.get('page_start')})"
            for i, s in enumerate(sections)
        ) or "(none)"
        where = " and ".join(f"p.{p}" for p in shown) or "(no page open)"
        return (
            "<|im_start|>system\n(mock) You FIND the right page and POINT at "
            "things — reply with ONE tool JSON, no prose.<|im_end|>\n"
            f"<|im_start|>user\nCURRENTLY ON SCREEN — {where} (mock text omitted)\n\n"
            f"TABLE OF CONTENTS:\n{toc}\n\n"
            f"The mechanic said: {request!r}\n"
            "Choose ONE tool and reply with ONLY its JSON object."
            "<|im_end|>\n<|im_start|>assistant\n"
        )

    @staticmethod
    def _mock_route(request: str, sections: list[dict]):
        """Fake the LLM router with keyword heuristics. Returns one of:
        ("go_to_section", page, title) / ("go_to_page", page, None) /
        ("point_here", target, None) / ("search", target, query)."""
        r = request.lower()
        nav_verb = any(
            v in r for v in ("go to", "take me", "open", "bring up", "pull up",
                             "navigate", "jump to")
        )
        is_circle = any(
            v in r for v in ("circle", "point", "highlight", "mark", "show me where")
        )
        here = any(v in r for v in ("here", "this page", "this", "current"))
        # A bare page number that the client's strict nav regex didn't catch
        # (extra words around it) → exercise the go_to_page tool, like the agent
        # reading a page number off an index.
        pm = re.search(r"\bp(?:age|g|\.)?\s*(\d+)\b", r)
        if pm and (nav_verb or "index" in r or "contents" in r):
            return "go_to_page", int(pm.group(1)), None
        secs = [{"title": o["title"], "page_start": o["page"]} for o in sections]
        best = match_section(request, secs) if secs else None

        if (nav_verb or "section" in r or "chapter" in r) and best and best["score"] >= 0.4:
            return "go_to_section", best["page"], best["title"]
        target = MockAskPipeline._clean_target(request)
        if is_circle and here:
            return "point_here", target, None
        if is_circle:
            return "search", target, request
        if best and best["score"] >= 0.6:
            return "go_to_section", best["page"], best["title"]
        return "search", target, request

    @staticmethod
    def _clean_target(request: str) -> str:
        r = re.sub(
            r"^(?:can you |please )?(?:circle|point (?:at|to)|highlight|mark|"
            r"show me where)\s+",
            "", request.strip(), flags=re.I,
        )
        r = re.sub(r"^(the |a |an )", "", r, flags=re.I)
        r = re.sub(r"\b(?:on |in )?(?:this|the current) page\b", "", r, flags=re.I)
        r = re.sub(r"\bhere\b", "", r, flags=re.I)
        r = re.sub(r"\s+(?:is|are)(?:\s+located)?$", "", r, flags=re.I)
        return " ".join(r.split()) or request

    @staticmethod
    def _mock_box(store: MockStore, doc_id: str, page: int, target: str):
        """A deterministic, target-dependent bbox in rendered-page pixels;
        None for ~1 in 8 targets so the "located but not pinpointed" path is
        exercised too."""
        path = store.pdf_path(doc_id)
        if not path:
            return None
        seed = int(hashlib.sha1(target.encode()).hexdigest(), 16)
        if seed % 8 == 7:
            return None
        w, h = page_size(path, page)
        x1 = w * (0.1 + (seed % 5) * 0.1)
        y1 = h * (0.15 + (seed // 5 % 5) * 0.12)
        return [round(x1), round(y1), round(x1 + w * 0.3), round(y1 + h * 0.18)]

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
