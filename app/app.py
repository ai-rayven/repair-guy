"""Gradio + ZeroGPU Space: ask questions over repair manuals.

Visual RAG with no parsing/chunking: every PDF page is embedded as an image
with Nemotron ColEmbed v2 (multi-vector, late interaction). A question is
answered by embedding the query, running MaxSim over page embeddings streamed
from disk, and handing the top pages to MiniCPM-V — all in one ZeroGPU call.

Two tabs:
  Library     — large manuals pre-indexed offline (scripts/index_local.py),
                synced at startup from the library dataset on the Hub.
  Upload      — index your own small PDF on ZeroGPU (page-capped to protect
                the Space's GPU quota), then ask it questions.

Module layout:
  models/colembed.py   ColEmbed     — embedding model + GPU embed / MaxSim
  models/minicpm.py    MiniCPM      — local VLM that answers over page images
  core/store.py        Store        — on-disk per-page token embeddings
  core/pdf.py          render_pages — PDF -> RGB page images (CPU)
  pipelines/ingest.py  IngestPipeline — PDF -> embeddings -> store
  pipelines/ask.py     AskPipeline    — question -> retrieve -> answer
"""

import os

import gradio as gr
from huggingface_hub import snapshot_download

from core.constants import (
    DEFAULT_TOP_K,
    LIBRARY_DATASET_ID,
    MAX_UPLOAD_PAGES,
    PREINDEXED_DIR,
    UPLOADS_DIR,
)
from core.store import Store, slugify
from models.colembed import ColEmbed
from pipelines.ask import AskPipeline
from pipelines.ingest import IngestPipeline

# Construct once at startup (the models load onto cuda here, in the main process).
library = Store(PREINDEXED_DIR)
uploads = Store(UPLOADS_DIR)
embedder = ColEmbed()
ingest_pipeline = IngestPipeline(embedder, uploads)
ask_pipeline = AskPipeline()


def sync_library() -> None:
    """Pull pre-indexed manuals from the library dataset into /data.
    A missing or empty dataset just means an empty library tab."""
    try:
        snapshot_download(
            LIBRARY_DATASET_ID, repo_type="dataset", local_dir=library.root
        )
    except Exception as e:
        print(f"Library dataset not synced ({LIBRARY_DATASET_ID}): {e}")


sync_library()


def _choices(store: Store) -> list[tuple[str, str]]:
    return [(d["name"], d["doc_id"]) for d in store.list_docs()]


def ask_library(question, doc_id):
    if not doc_id:
        raise gr.Error("Pick a manual first.")
    try:
        return ask_pipeline.run(library, question, [doc_id], DEFAULT_TOP_K)
    except ValueError as e:
        raise gr.Error(str(e)) from e


def ask_upload(question, doc_id):
    if not doc_id:
        raise gr.Error("Upload a repair manual first.")
    try:
        return ask_pipeline.run(uploads, question, [doc_id], DEFAULT_TOP_K)
    except ValueError as e:
        raise gr.Error(str(e)) from e


def index_manual(pdf_file, progress=gr.Progress()):
    """Runs on upload: embed the manual's pages (or reuse a previous index).
    Generator — each yield streams (doc_state, status markdown) to the UI."""
    if not pdf_file:
        yield None, ""
        return
    name = os.path.splitext(os.path.basename(pdf_file))[0].replace("_", " ")
    doc_id = slugify(name)
    if uploads.exists(doc_id):
        pages = len(uploads.meta(doc_id)["pages"])
        yield doc_id, f"**{name}** is already indexed ({pages} pages) — ask away."
        return
    yield None, f"⏳ Indexing **{name}** — preparing…"
    try:
        for event in ingest_pipeline.run(pdf_file, name, max_pages=MAX_UPLOAD_PAGES):
            if event[0] == "progress":
                _, done, total = event
                progress(done / total, desc=f"Embedding pages {done}/{total}")
                yield None, f"⏳ Indexing **{name}** — {done}/{total} pages embedded…"
            else:
                doc = event[1]
    except ValueError as e:
        raise gr.Error(str(e)) from e
    yield doc_id, f"✅ Indexed **{doc['name']}** ({doc['pages']} pages) — ask away."


with gr.Blocks(title="Repair Guy") as demo:
    gr.Markdown(
        "# 🔧 Repair Guy\n"
        "Ask questions over repair manuals. Pages are retrieved visually with "
        "[Nemotron ColEmbed v2](https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2) "
        "(late interaction, no parsing) and answered by MiniCPM-V reading the "
        "most relevant pages."
    )
    with gr.Tab("📚 Library"):
        with gr.Row():
            with gr.Column(scale=1):
                manual_in = gr.Dropdown(label="Manual", choices=[])
                lib_question_in = gr.Textbox(
                    label="Question",
                    lines=2,
                    placeholder="e.g. What is the tightening torque for the universal joint flange bolts?",
                )
                lib_ask_btn = gr.Button("Ask", variant="primary")
            with gr.Column(scale=2):
                lib_answer_out = gr.Markdown(label="Answer")
                lib_pages_out = gr.Gallery(label="Pages used", columns=3, height=420)
    with gr.Tab("📄 Upload your own"):
        gr.Markdown(
            f"Upload a PDF of up to **{MAX_UPLOAD_PAGES} pages** (indexing runs "
            "on this Space's GPU quota — big manuals live in the Library tab)."
        )
        doc_state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                pdf_in = gr.File(
                    label="Repair manual (PDF)", file_types=[".pdf"], type="filepath"
                )
                status_out = gr.Markdown()
                up_question_in = gr.Textbox(label="Question", lines=2)
                up_ask_btn = gr.Button("Ask", variant="primary")
            with gr.Column(scale=2):
                up_answer_out = gr.Markdown(label="Answer")
                up_pages_out = gr.Gallery(label="Pages used", columns=3, height=420)

    lib_ask_btn.click(
        ask_library, inputs=[lib_question_in, manual_in],
        outputs=[lib_answer_out, lib_pages_out],
    )
    lib_question_in.submit(
        ask_library, inputs=[lib_question_in, manual_in],
        outputs=[lib_answer_out, lib_pages_out],
    )
    pdf_in.upload(index_manual, inputs=[pdf_in], outputs=[doc_state, status_out])
    up_ask_btn.click(
        ask_upload, inputs=[up_question_in, doc_state],
        outputs=[up_answer_out, up_pages_out],
    )
    up_question_in.submit(
        ask_upload, inputs=[up_question_in, doc_state],
        outputs=[up_answer_out, up_pages_out],
    )
    demo.load(lambda: gr.update(choices=_choices(library)), outputs=[manual_in])


if __name__ == "__main__":
    demo.launch()
