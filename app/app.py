"""Gradio Server Mode backend for the Repair Guy Space.

Showcases two local-only indexing approaches over the same manuals (all
ingestion happens offline via scripts/index_modal.py; the Space only syncs
the pre-indexed library and answers questions):

  Visual — every page embedded as an image with Nemotron ColEmbed v2
           (multi-vector late interaction, no parsing); retrieval is MaxSim
           over page embeddings streamed from disk.
  Parsed — pages parsed with Nemotron Parse, figures/tables described by
           MiniCPM-V, section chunks embedded with Llama Nemotron Embed;
           retrieval is dense cosine over chunks with parent-page lookup.

The product is a hands-busy mechanic's assistant: a page viewer driven by
short requests. Intents are tiered by cost —
  Tier 0 (client, instant)  next/previous page, "page 412", back, next
                            section: pure frontend state, no server call.
  Tier 1 (CPU routes)       "go to <section>" → /navigate fuzzy-matches the
                            parse-derived section index (core/sections.py);
                            "circle the <thing>" → /locate matches figure/
                            table descriptions and returns the parse bbox.
  Tier 2 (ZeroGPU)          real questions → /ask, one retrieval + one
                            MiniCPM-V generation grounded in the retrieved
                            page images (pipelines/chat_ask.py), streamed as
                            events; "circle ..." with no parse match →
                            /point, MiniCPM-V visual grounding on the viewed
                            page. Chat history is kept client-side and
                            summarized by the model when it outgrows its
                            token budget.

UI architecture — this is NOT a gr.Blocks app. It runs in Gradio *Server Mode*
(`gradio.Server`, a FastAPI server with Gradio's engine: queueing, streaming
and — crucially — ZeroGPU auth). The Python pipelines below are exposed as
`@app.api` endpoints; the frontend is a fully custom single page (Tailwind +
Alpine, no build step) served from frontend/index.html, which calls those
endpoints with @gradio/client so the HF iframe auth headers ZeroGPU needs are
forwarded. The source PDFs are served straight from FastAPI at /pdf/<doc_id>.

Module layout:
  models/colembed.py        ColEmbed       — visual: page embeddings + MaxSim
  models/nemotron_embed.py  NemotronEmbed  — parsed: dense chunk/query embeddings
  models/minicpm.py         MiniCPM        — shared: answers over page images
  core/visual_store.py      VisualStore    — on-disk per-page token embeddings
  core/parsed_store.py      ParsedStore    — chunks + dense embedding matrix
  core/sections.py          section index + fuzzy matching (CPU navigation)
  pipelines/chat_ask.py     chat_events / point_box — the shared GPU turns
  pipelines/visual_ask.py   VisualAskPipeline
  pipelines/parsed_ask.py   ParsedAskPipeline
  pipelines/mock_ask.py     MockAskPipeline — local UI iteration (MOCK_MODELS=1)
  frontend/index.html       the custom UI
"""

import base64
import io
import json
import logging
import os
import shutil
import time
import warnings

import gradio as gr
from fastapi.responses import FileResponse, HTMLResponse, Response
from huggingface_hub import snapshot_download

from core.constants import (
    DEFAULT_TOP_K,
    LIBRARY_DATASET_ID,
    MAX_TOP_K,
    MOCK_MODELS,
    MOCK_PDF_DIR,
    PARSED_SUBDIR,
    PREINDEXED_DIR,
    VISUAL_SUBDIR,
)
from core.pdf import render_page_png
from core.sections import match_element, match_section, sections_from_chunks

# gradio 6.17.3 (pinned — see README frontmatter) still uses starlette's old
# 422 constant, so every queue join emits a StarletteDeprecationWarning. Not
# ours to fix; silence it so the Space logs stay readable.
warnings.filterwarnings(
    "ignore", message=r"'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated"
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)  # logs every hub request
log = logging.getLogger("repairguy")


def _build_libraries() -> dict:
    """The {approach: (store, pipeline)} map, constructed once at startup.

    In MOCK_MODELS mode a single PDF-folder mock backs both approaches and no
    GPU/model code is imported (those modules load CUDA at import). Otherwise
    the real stores and pipelines load — the models onto cuda here, in the main
    process. Either way both pipelines expose run(store, question, doc_ids, k)."""
    if MOCK_MODELS:
        from pipelines.mock_ask import MockAskPipeline, MockStore

        store, pipeline = MockStore(), MockAskPipeline()
        print(f"⚠️  MOCK_MODELS — canned answers over PDFs in {MOCK_PDF_DIR}")
        return {"visual": (store, pipeline), "parsed": (store, pipeline)}

    from core.parsed_store import ParsedStore
    from core.visual_store import VisualStore
    from pipelines.parsed_ask import ParsedAskPipeline
    from pipelines.visual_ask import VisualAskPipeline

    return {
        "visual": (
            VisualStore(os.path.join(PREINDEXED_DIR, VISUAL_SUBDIR)),
            VisualAskPipeline(),
        ),
        "parsed": (
            ParsedStore(os.path.join(PREINDEXED_DIR, PARSED_SUBDIR)),
            ParsedAskPipeline(),
        ),
    }


LIBRARIES = _build_libraries()


def sync_library() -> None:
    """Pull pre-indexed manuals from the library dataset into /data.
    A missing or empty dataset just means an empty library."""
    if MOCK_MODELS:  # mock library is the local PDF folder; nothing to sync
        return
    try:
        snapshot_download(
            LIBRARY_DATASET_ID, repo_type="dataset", local_dir=PREINDEXED_DIR
        )
        log.info("library synced from %s", LIBRARY_DATASET_ID)
    except Exception as e:
        log.warning("library dataset not synced (%s): %s", LIBRARY_DATASET_ID, e)
        return
    # PREINDEXED_DIR mirrors the dataset (one dir per method); prune top-level
    # leftovers from the pre-method-prefix layout, which snapshot_download
    # never deletes.
    for entry in os.listdir(PREINDEXED_DIR):
        path = os.path.join(PREINDEXED_DIR, entry)
        if entry.startswith(".") or entry in (VISUAL_SUBDIR, PARSED_SUBDIR):
            continue
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "index.json")):
            print(f"Pruning stale pre-migration doc dir: {entry}")
            shutil.rmtree(path, ignore_errors=True)


sync_library()


# Labels for the approach picker shown in the settings panel.
APPROACHES = {
    "visual": {
        "label": "Visual",
        "blurb": "ColEmbed late interaction — pages stay images, nothing parsed.",
    },
    "parsed": {
        "label": "Parsed",
        "blurb": "Dense chunks over parsed text + figure / table descriptions.",
    },
}


def _manual_choices() -> list[dict]:
    """The manuals shown in the picker, one shared list across both libraries
    (doc ids are name slugs, so the same manual lands on the same id in both);
    manuals indexed under only one approach are tagged with it. pages drives
    the viewer's page count."""
    docs: dict[str, dict] = {}
    for method, (store, _) in LIBRARIES.items():
        for d in store.list_docs():
            entry = docs.setdefault(
                d["doc_id"], {"name": d["name"], "methods": [], "pages": 0}
            )
            entry["methods"].append(method)
            entry["pages"] = max(entry["pages"], d["pages"])
    choices = []
    for doc_id, info in sorted(docs.items(), key=lambda kv: kv[1]["name"].lower()):
        label = info["name"]
        if len(info["methods"]) < len(LIBRARIES):
            label += f" — {info['methods'][0]} only"
        choices.append({"value": doc_id, "label": label, "pages": info["pages"]})
    return choices


def _pdf_path(doc_id: str) -> str | None:
    """The source PDF for a manual (kept identically in whichever store indexed
    it — both copy doc.pdf at ingest)."""
    for store, _ in LIBRARIES.values():
        if store.exists(doc_id):
            return store.pdf_path(doc_id)
    return None


# Section/element structure for the CPU navigation routes, derived from the
# parsed store's chunks (the mock store fakes both). Cached per doc — cleared
# on library re-sync. Manuals indexed only visually have neither: /navigate
# and /locate then return no match and the frontend falls back to Q&A.
_SECTIONS_CACHE: dict[str, list[dict]] = {}
_ELEMENTS_CACHE: dict[str, list[dict]] = {}


def _doc_chunks(doc_id: str) -> list[dict]:
    store, _ = LIBRARIES["parsed"]
    if MOCK_MODELS or not store.exists(doc_id):
        return []
    return store.chunks(doc_id)


def _doc_sections(doc_id: str) -> list[dict]:
    if doc_id not in _SECTIONS_CACHE:
        store, _ = LIBRARIES["parsed"]
        if MOCK_MODELS:
            _SECTIONS_CACHE[doc_id] = store.sections(doc_id)
        else:
            _SECTIONS_CACHE[doc_id] = sections_from_chunks(_doc_chunks(doc_id))
    return _SECTIONS_CACHE[doc_id]


def _doc_elements(doc_id: str) -> list[dict]:
    if doc_id not in _ELEMENTS_CACHE:
        store, _ = LIBRARIES["parsed"]
        if MOCK_MODELS:
            _ELEMENTS_CACHE[doc_id] = store.elements(doc_id)
        else:
            _ELEMENTS_CACHE[doc_id] = [
                c for c in _doc_chunks(doc_id) if c.get("type") in ("figure", "table")
            ]
    return _ELEMENTS_CACHE[doc_id]


def _thumb_data_uri(img, width: int = 280) -> str:
    """A downscaled JPEG data URI for a rendered page, small enough to ship in
    the JSON answer (cited-page thumbnails are decorative, not full-res)."""
    img = img.convert("RGB")
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Server Mode: API engine (ZeroGPU auth, queueing) with a custom frontend.
# ---------------------------------------------------------------------------
app = gr.Server()


@app.api(name="manuals")
def api_manuals() -> list[dict]:
    """The manual picker's options: [{value, label}]."""
    return _manual_choices()


@app.api(name="refresh")
def api_refresh() -> list[dict]:
    """Re-sync the library dataset (incremental) and return the refreshed
    picker options, so manuals indexed after boot show up without a restart."""
    sync_library()
    _SECTIONS_CACHE.clear()
    _ELEMENTS_CACHE.clear()
    return _manual_choices()


def _clean_history(history) -> list[dict]:
    """The model-facing transcript the client sends back with each turn,
    reduced to what the pipelines expect: alternating {role, content} text
    messages. Anything malformed is dropped rather than rejected."""
    cleaned = []
    for m in history or []:
        if (
            isinstance(m, dict)
            and m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str)
        ):
            cleaned.append({"role": m["role"], "content": m["content"]})
    return cleaned


@app.api(name="ask")
def api_ask(
    question: str,
    manual: str = "",
    approach: str = "visual",
    k: int = DEFAULT_TOP_K,
    history: list | None = None,
    page: int = 0,
    section: str = "",
) -> dict:  # the per-yield type: Server.api infers outputs from this annotation
    """One Q&A turn with the chosen approach (one ZeroGPU call), streamed as
    events (see pipelines/chat_ask.py for the protocol). page/section are what
    the viewer currently shows, so the model can answer "what torque is listed
    here?".

    Yields {type: status|tool_call|tool_result|answer|error, ...}; tool_result
    galleries are converted to JSON-able thumbnails here, and the final answer
    event carries the updated history (possibly summarized) plus elapsed/
    approach/k. Soft errors so the frontend renders them as messages."""
    question = (question or "").strip()
    if not question:
        yield {"type": "error", "error": "Please enter a question."}
        return
    if not manual:
        yield {"type": "error", "error": "Pick a manual first ☝️"}
        return
    if approach not in LIBRARIES:
        approach = "visual"
    store, pipeline = LIBRARIES[approach]
    if not store.exists(manual):
        other = "parsed" if approach == "visual" else "visual"
        yield {
            "type": "error",
            "error": f"This manual isn't indexed with the **{approach}** approach "
            f"yet — switch to **{other}** in settings, or pick another manual.",
        }
        return
    start = time.monotonic()
    hist = _clean_history(history)
    viewer = {"page": int(page or 0), "section": str(section or "")}
    log.info(
        "ask: manual=%s approach=%s k=%s history=%d viewer=%s q=%r",
        manual, approach, k, len(hist), viewer, question[:200],
    )
    try:
        events = pipeline.run_chat(store, question, hist, [manual], int(k), viewer)
        for ev in events:
            if ev.get("type") == "tool_result" and "gallery" in ev:
                pages = [p for _, p in ev["page_refs"]]
                yield {
                    "type": "tool_result",
                    "tool": ev["tool"],
                    "pages": pages,
                    "thumbnails": [
                        {"page": p, "src": _thumb_data_uri(img), "caption": cap}
                        for (img, cap), p in zip(ev["gallery"], pages)
                    ],
                }
            elif ev.get("type") == "answer":
                elapsed = round(time.monotonic() - start, 1)
                log.info(
                    "ask: answered in %.1fs (%d chars, history=%d, summarized=%s)",
                    elapsed, len(ev.get("answer") or ""),
                    len(ev.get("history") or []), ev.get("summarized"),
                )
                yield {**ev, "elapsed": elapsed, "approach": approach, "k": int(k)}
            else:
                yield ev
    except ValueError as e:
        log.warning("ask: rejected — %s", e)
        yield {"type": "error", "error": f"⚠️ {e}"}
    except Exception as e:
        log.exception("ask: failed after %.1fs", time.monotonic() - start)
        yield {"type": "error", "error": f"⚠️ Something went wrong: {e}"}


@app.api(name="point")
def api_point(manual: str = "", query: str = "", page: int = 0) -> dict:
    """Visual grounding for "circle the <thing>" when the parse-bbox match
    (/locate) came up empty: one short ZeroGPU call that asks MiniCPM-V for
    the bounding box of the described object on the viewed page. Returns
    {found, page?, bbox?} with bbox in rendered-page pixels (the /page image's
    coordinate space)."""
    page = int(page or 0)
    if not manual or not query or page < 1:
        return {"found": False}
    path = _pdf_path(manual)
    if not path or not os.path.isfile(path):
        return {"found": False}
    log.info("point: manual=%s page=%d query=%r", manual, page, query)
    if MOCK_MODELS:
        store, pipeline = LIBRARIES["parsed"]
        res = pipeline.point(store, manual, page, query)
    else:
        from pipelines.chat_ask import point_box  # GPU import chain — lazy

        res = point_box(path, page, query)
    return {"found": bool(res), **(res or {})}


# --- custom FastAPI routes: serve the SPA and the source PDFs ---------------
_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")


_APPROACHES_JS = [{"key": k, **v} for k, v in APPROACHES.items()]


@app.get("/")
def index():
    """Serve the single-page UI, injecting the small bit of server config the
    frontend needs (approach options, default/max k) so it needs no extra
    round-trip on load."""
    with open(os.path.join(_FRONTEND_DIR, "index.html")) as f:
        html = f.read()
    html = (
        html.replace("__APPROACHES__", json.dumps(_APPROACHES_JS))
        .replace("__DEFAULT_K__", str(DEFAULT_TOP_K))
        .replace("__MAX_K__", str(MAX_TOP_K))
    )
    return HTMLResponse(html)


@app.get("/media/{filename}")
def serve_media(filename: str):
    """Static frontend assets (logo etc.) from frontend/assets/."""
    path = os.path.join(_FRONTEND_DIR, "assets", os.path.basename(filename))
    if not os.path.isfile(path):
        return Response(status_code=404)
    return FileResponse(path)


@app.get("/pdf/{doc_id}")
def serve_pdf(doc_id: str):
    """A manual's source PDF (kept for direct download/open-in-tab; the viewer
    pane shows /page images, not the PDF)."""
    path = _pdf_path(doc_id)
    if not path or not os.path.isfile(path):
        return Response(status_code=404)
    return FileResponse(
        path, media_type="application/pdf", content_disposition_type="inline"
    )


@app.get("/page/{doc_id}/{page}")
def serve_page(doc_id: str, page: int):
    """One manual page rendered to PNG at RENDER_DPI — the viewer pane's <img>
    source. Parse bboxes (/locate) and grounding boxes (/point) are in this
    image's pixel coordinates, so the SVG circle overlay maps 1:1. Immutable
    content → long browser cache, which is what makes page flips instant."""
    path = _pdf_path(doc_id)
    if not path or not os.path.isfile(path):
        return Response(status_code=404)
    try:
        data = render_page_png(path, page)
    except ValueError:
        return Response(status_code=404)
    return Response(
        data,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/sections/{doc_id}")
def serve_sections(doc_id: str):
    """The manual's section index [{title, page_start, page_end}] — drives the
    breadcrumb and next/previous-section navigation in the frontend."""
    return {"sections": _doc_sections(doc_id)}


@app.get("/navigate/{doc_id}")
def serve_navigate(doc_id: str, q: str = ""):
    """Fuzzy-match a "go to <X>" request against the section index. CPU-only;
    returns {match: {title, page, score} | null} — the frontend applies its
    acceptance threshold and falls back to Q&A below it."""
    match = match_section(q, _doc_sections(doc_id)) if q.strip() else None
    log.info("navigate %s q=%r → %s", doc_id, q, match)
    return {"match": match}


@app.get("/locate/{doc_id}")
def serve_locate(doc_id: str, q: str = ""):
    """Fuzzy-match a "circle the <X>" request against the parsed figure/table
    descriptions. CPU-only; returns {match: {page, bbox, kind, label, score} |
    null} with bbox in /page pixel coordinates."""
    match = match_element(q, _doc_elements(doc_id)) if q.strip() else None
    log.info("locate %s q=%r → %s", doc_id, q, match)
    return {"match": match}


if __name__ == "__main__":
    app.launch()
