"""Gradio + ZeroGPU Space: ask questions over repair manuals.

Showcases two local-only indexing approaches over the same manuals (all
ingestion happens offline via scripts/index_modal.py; the Space only syncs
the pre-indexed library and answers questions):

  Visual — every page embedded as an image with Nemotron ColEmbed v2
           (multi-vector late interaction, no parsing); retrieval is MaxSim
           over page embeddings streamed from disk.
  Parsed — pages parsed with Nemotron Parse, figures/tables described by
           MiniCPM-V, section chunks embedded with Llama Nemotron Embed;
           retrieval is dense cosine over chunks with parent-page lookup.

Both hand the retrieved page images to MiniCPM-V for the grounded answer,
in one ZeroGPU call per question.

Module layout:
  models/colembed.py        ColEmbed       — visual: page embeddings + MaxSim
  models/nemotron_embed.py  NemotronEmbed  — parsed: dense chunk/query embeddings
  models/minicpm.py         MiniCPM        — shared: answers over page images
  core/visual_store.py      VisualStore    — on-disk per-page token embeddings
  core/parsed_store.py      ParsedStore    — chunks + dense embedding matrix
  pipelines/visual_ask.py   VisualAskPipeline
  pipelines/parsed_ask.py   ParsedAskPipeline  (not wired into the UI yet)
"""

import os
import shutil

import gradio as gr
from huggingface_hub import snapshot_download

from core.constants import (
    DEFAULT_TOP_K,
    LIBRARY_DATASET_ID,
    PARSED_SUBDIR,
    PREINDEXED_DIR,
    VISUAL_SUBDIR,
)
from core.visual_store import VisualStore
from pipelines.visual_ask import VisualAskPipeline

# Construct once at startup (the models load onto cuda here, in the main process).
library = VisualStore(os.path.join(PREINDEXED_DIR, VISUAL_SUBDIR))
ask_pipeline = VisualAskPipeline()


def sync_library() -> None:
    """Pull pre-indexed manuals from the library dataset into /data.
    A missing or empty dataset just means an empty library."""
    try:
        snapshot_download(
            LIBRARY_DATASET_ID, repo_type="dataset", local_dir=PREINDEXED_DIR
        )
    except Exception as e:
        print(f"Library dataset not synced ({LIBRARY_DATASET_ID}): {e}")
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


def _choices(store) -> list[tuple[str, str]]:
    return [(d["name"], d["doc_id"]) for d in store.list_docs()]


def refresh_library():
    """Re-pull the library dataset (incremental) and refresh the dropdown,
    so manuals indexed after the Space booted show up without a restart."""
    sync_library()
    return gr.update(choices=_choices(library))


def ask_library(question, doc_id):
    if not doc_id:
        raise gr.Error("Pick a manual first.")
    try:
        return ask_pipeline.run(library, question, [doc_id], DEFAULT_TOP_K)
    except ValueError as e:
        raise gr.Error(str(e)) from e


with gr.Blocks(title="Repair Guy") as demo:
    gr.Markdown(
        "# 🔧 Repair Guy\n"
        "Ask questions over repair manuals. Pages are retrieved visually with "
        "[Nemotron ColEmbed v2](https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2) "
        "(late interaction, no parsing) and answered by MiniCPM-V reading the "
        "most relevant pages."
    )
    with gr.Row():
        with gr.Column(scale=1):
            manual_in = gr.Dropdown(label="Manual", choices=[])
            lib_refresh_btn = gr.Button("🔄 Sync library", size="sm")
            lib_question_in = gr.Textbox(
                label="Question",
                lines=2,
                placeholder="e.g. What is the tightening torque for the universal joint flange bolts?",
            )
            lib_ask_btn = gr.Button("Ask", variant="primary")
        with gr.Column(scale=2):
            lib_answer_out = gr.Markdown(label="Answer")
            lib_pages_out = gr.Gallery(label="Pages used", columns=3, height=420)

    lib_refresh_btn.click(refresh_library, outputs=[manual_in])
    lib_ask_btn.click(
        ask_library, inputs=[lib_question_in, manual_in],
        outputs=[lib_answer_out, lib_pages_out],
    )
    lib_question_in.submit(
        ask_library, inputs=[lib_question_in, manual_in],
        outputs=[lib_answer_out, lib_pages_out],
    )
    demo.load(lambda: gr.update(choices=_choices(library)), outputs=[manual_in])


if __name__ == "__main__":
    demo.launch()
