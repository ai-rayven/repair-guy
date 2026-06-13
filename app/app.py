"""Gradio Server Mode backend for the Repair Guy Space.

A hands-busy mechanic's assistant: a page viewer driven by short requests. It
finds and points — it never writes answers. All ingestion happens offline via
scripts/index_modal.py; the Space only syncs the pre-indexed library and serves
turns.

  Obvious nav (client)  next/previous page, "page 412", back, next/previous
                        section: pure frontend state, no server call.
  Everything else       → /find, one ZeroGPU call (pipelines/agent_ask.py):
  (ZeroGPU agent turn)  MiniCPM5-1B (the text "brain") sees the conversation so
                        far, the table of contents, and the WHOLE text of the
                        page being viewed, and calls tools in a loop until a page
                        is shown — circle on the current page (MiniCPM-V grounds
                        the box), jump to a section, or search. Search is FUSED:
                        ColEmbed (visual store) shortlists pages, the 1B reranks
                        by their parsed text. Streamed as events; the UI shows
                        tool chips and the resulting page/circle, never model
                        prose. History is kept only to resolve references.

The table of contents is the manual's clean PDF-bookmark chapters plus a
per-request fuzzy shortlist of fine parse headings (core/sections.py).

UI architecture — this is NOT a gr.Blocks app. It runs in Gradio *Server Mode*
(`gradio.Server`, a FastAPI server with Gradio's engine: queueing, streaming
and — crucially — ZeroGPU auth). The Python pipelines below are exposed as
`@app.api` endpoints; the frontend is a fully custom single page (Tailwind +
Alpine, no build step) served from frontend/index.html, which calls those
endpoints with @gradio/client so the HF iframe auth headers ZeroGPU needs are
forwarded. The source PDFs are served straight from FastAPI at /pdf/<doc_id>.

Module layout:
  models/colembed.py        ColEmbed       — search shortlist: page embeddings + MaxSim
  models/minicpm_agent.py   MiniCPM5-1B    — the agent "brain": tool loop + rerank
  models/minicpm.py         MiniCPM-V      — the "eyes": grounds the circle
  models/nemotron_embed.py  NemotronEmbed  — parsed: dense chunk/query embeddings (ingest)
  core/visual_store.py      VisualStore    — on-disk per-page token embeddings
  core/parsed_store.py      ParsedStore    — chunks, embeddings, raw parsed pages
  core/page_context.py      whole-page → text for the agent's context
  core/sections.py          section index + fuzzy matching (TOC shortlist)
  pipelines/agent_ask.py    AgentPipeline — the agent find-and-point GPU turn
  pipelines/mock_ask.py     MockAskPipeline — local UI iteration (MOCK_MODELS=1)
  frontend/index.html       the custom UI
"""

import base64
import io
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
from core.pdf import pdf_outline, render_page_png
from core.sections import sections_from_chunks, top_sections

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


def _build_libraries():
    """(visual_store, parsed_store, agent_pipeline), constructed once at startup.

    The agent path is FUSED: ColEmbed (visual store) supplies the search
    shortlist, the parsed store supplies the page text the 1B reranks with and
    the agent reasons over — so both stores are live at once.

    In MOCK_MODELS mode a single PDF-folder mock backs both stores and no
    GPU/model code is imported (those modules load CUDA at import). Otherwise the
    real stores and the agent pipeline load — the models onto cuda here, in the
    main process."""
    if MOCK_MODELS:
        from pipelines.mock_ask import MockAskPipeline, MockStore

        store = MockStore()
        print(f"⚠️  MOCK_MODELS — fake agent over PDFs in {MOCK_PDF_DIR}")
        return store, store, MockAskPipeline()

    from core.parsed_store import ParsedStore
    from core.visual_store import VisualStore
    from pipelines.agent_ask import AgentPipeline

    return (
        VisualStore(os.path.join(PREINDEXED_DIR, VISUAL_SUBDIR)),
        ParsedStore(os.path.join(PREINDEXED_DIR, PARSED_SUBDIR)),
        AgentPipeline(),
    )


VISUAL_STORE, PARSED_STORE, PIPELINE = _build_libraries()
# method -> store, for the picker / pdf lookups. In mock both keys map to the
# one MockStore, so every mock manual reads as indexed under both methods.
_METHOD_STORES = {"visual": VISUAL_STORE, "parsed": PARSED_STORE}


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


def _manual_choices() -> list[dict]:
    """The manuals shown in the picker (doc ids are name slugs, so the same
    manual lands on the same id in both stores). The fused agent needs BOTH
    indexes; a manual missing one is tagged and the agent turn rejects it.
    pages drives the viewer's page count."""
    docs: dict[str, dict] = {}
    for method, store in _METHOD_STORES.items():
        for d in store.list_docs():
            entry = docs.setdefault(
                d["doc_id"], {"name": d["name"], "methods": set(), "pages": 0}
            )
            entry["methods"].add(method)
            entry["pages"] = max(entry["pages"], d["pages"])
    choices = []
    for doc_id, info in sorted(docs.items(), key=lambda kv: kv[1]["name"].lower()):
        label = info["name"]
        if info["methods"] != {"visual", "parsed"}:
            label += " — needs reindex"
        choices.append({"value": doc_id, "label": label, "pages": info["pages"]})
    return choices


def _pdf_path(doc_id: str) -> str | None:
    """The source PDF for a manual (kept identically in whichever store indexed
    it — both copy doc.pdf at ingest)."""
    for store in _METHOD_STORES.values():
        if store.exists(doc_id):
            return store.pdf_path(doc_id)
    return None


# Two section views, both cached per doc and cleared on library re-sync:
#   outline  — the manual's own clean bookmark chapters (pdf_outline), the
#              frontend's breadcrumb + next/previous-section navigation, and
#              the always-shown part of the router's section list.
#   headings — the fine-grained, noisy parse headings (1000+ for a big
#              manual): far too many for a prompt, but a per-request fuzzy
#              shortlist of them gives the router precise targets like
#              "Brake System Bleeding — p.532".
# A visual-only manual with no PDF bookmarks has neither; section navigation
# then no-ops and the router works from the page image alone.
_OUTLINE_CACHE: dict[str, list[dict]] = {}
_HEADINGS_CACHE: dict[str, list[dict]] = {}


def _doc_outline(doc_id: str) -> list[dict]:
    if doc_id not in _OUTLINE_CACHE:
        if MOCK_MODELS:
            _OUTLINE_CACHE[doc_id] = PARSED_STORE.sections(doc_id)
        else:
            path = _pdf_path(doc_id)
            _OUTLINE_CACHE[doc_id] = pdf_outline(path) if path else []
    return _OUTLINE_CACHE[doc_id]


def _doc_headings(doc_id: str) -> list[dict]:
    if doc_id not in _HEADINGS_CACHE:
        if MOCK_MODELS:
            _HEADINGS_CACHE[doc_id] = PARSED_STORE.sections(doc_id)
        elif PARSED_STORE.exists(doc_id):
            _HEADINGS_CACHE[doc_id] = sections_from_chunks(PARSED_STORE.chunks(doc_id))
        else:
            _HEADINGS_CACHE[doc_id] = []
    return _HEADINGS_CACHE[doc_id]


def _router_options(doc_id: str, request: str, max_total: int = 24) -> list[dict]:
    """The numbered section list shown to the router as [{title, page}]: the
    clean chapters always, plus the fine headings best matching this request,
    deduped by page. The router replies with a 1-based index into this list."""
    options = [
        {"title": s["title"], "page": s["page_start"]} for s in _doc_outline(doc_id)
    ]
    seen = {o["page"] for o in options}
    for s in top_sections(request, _doc_headings(doc_id), n=8):
        if s["page"] not in seen:
            options.append(s)
            seen.add(s["page"])
    return options[:max_total]


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
    _OUTLINE_CACHE.clear()
    _HEADINGS_CACHE.clear()
    return _manual_choices()


@app.api(name="find")
def api_find(
    request: str,
    manual: str = "",
    k: int = DEFAULT_TOP_K,
    page: int = 0,
    section: str = "",
    history: list = None,
) -> dict:  # the per-yield type: Server.api infers outputs from this annotation
    """One agent turn (one ZeroGPU call), streamed as events (see
    pipelines/agent_ask.py for the protocol). page/section are what the viewer
    currently shows (the agent's current page); history is the compact memory of
    past turns ([{request, action}]) for resolving references.

    Yields {type: status|step|tool_result|found|done|error, ...}; tool_result
    galleries are converted to JSON-able thumbnails here, and the terminal `done`
    carries elapsed/k. Soft errors so the frontend renders them as chips."""
    request = (request or "").strip()
    if not request:
        yield {"type": "error", "error": "Tell me what to find."}
        return
    if not manual:
        yield {"type": "error", "error": "Pick a manual first ☝️"}
        return
    if not (VISUAL_STORE.exists(manual) and PARSED_STORE.exists(manual)):
        yield {
            "type": "error",
            "error": "This manual needs reindexing — the assistant needs both the "
            "visual and parsed indexes. Pick another manual, or re-run indexing.",
        }
        return
    start = time.monotonic()
    viewer = {"page": int(page or 0), "section": str(section or "")}
    options = _router_options(manual, request)
    log.info(
        "find: manual=%s k=%s viewer=%s hist=%d opts=%d q=%r",
        manual, k, viewer, len(history or []), len(options), request[:200],
    )
    try:
        events = PIPELINE.run_find(
            VISUAL_STORE, PARSED_STORE, request, [manual], int(k), options,
            viewer, history,
        )
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
            elif ev.get("type") == "done":
                elapsed = round(time.monotonic() - start, 1)
                log.info("find: done in %.1fs (%s)", elapsed, ev.get("kind"))
                yield {**ev, "elapsed": elapsed, "k": int(k)}
            else:
                yield ev
    except ValueError as e:
        log.warning("find: rejected — %s", e)
        yield {"type": "error", "error": f"⚠️ {e}"}
    except Exception as e:
        log.exception("find: failed after %.1fs", time.monotonic() - start)
        yield {"type": "error", "error": f"⚠️ Something went wrong: {e}"}


# --- custom FastAPI routes: serve the SPA and the source PDFs ---------------
_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")


@app.get("/")
def index():
    """Serve the single-page UI, injecting the small bit of server config the
    frontend needs (default/max k) so it needs no extra round-trip on load."""
    with open(os.path.join(_FRONTEND_DIR, "index.html")) as f:
        html = f.read()
    html = html.replace("__DEFAULT_K__", str(DEFAULT_TOP_K)).replace(
        "__MAX_K__", str(MAX_TOP_K)
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
    source. Grounding boxes from the find turn are in this image's pixel
    coordinates, so the SVG circle overlay maps 1:1. Immutable content → long
    browser cache, which is what makes page flips instant."""
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
    """The manual's clean chapter outline [{title, page_start, page_end}] —
    drives the breadcrumb and next/previous-section navigation in the
    frontend. Empty for a visual-only manual with no PDF bookmarks."""
    return {"sections": _doc_outline(doc_id)}


if __name__ == "__main__":
    app.launch()
