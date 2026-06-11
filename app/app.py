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
in one ZeroGPU call per question. The approach is picked per question in
the UI; each approach has its own store directory and manual list.

Module layout:
  models/colembed.py        ColEmbed       — visual: page embeddings + MaxSim
  models/nemotron_embed.py  NemotronEmbed  — parsed: dense chunk/query embeddings
  models/minicpm.py         MiniCPM        — shared: answers over page images
  core/visual_store.py      VisualStore    — on-disk per-page token embeddings
  core/parsed_store.py      ParsedStore    — chunks + dense embedding matrix
  pipelines/visual_ask.py   VisualAskPipeline
  pipelines/parsed_ask.py   ParsedAskPipeline
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
from core.parsed_store import ParsedStore
from core.visual_store import VisualStore
from pipelines.parsed_ask import ParsedAskPipeline
from pipelines.visual_ask import VisualAskPipeline

# Construct once at startup (the models load onto cuda here, in the main
# process). Both pipelines expose the same run(store, question, doc_ids, top_k).
LIBRARIES = {
    "visual": (
        VisualStore(os.path.join(PREINDEXED_DIR, VISUAL_SUBDIR)),
        VisualAskPipeline(),
    ),
    "parsed": (
        ParsedStore(os.path.join(PREINDEXED_DIR, PARSED_SUBDIR)),
        ParsedAskPipeline(),
    ),
}

METHOD_CHOICES = [
    ("Visual — ColEmbed page embeddings + MaxSim", "visual"),
    ("Parsed — Nemotron Parse chunks + dense retrieval", "parsed"),
]


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


def _manuals_update(method: str, current: str | None = None):
    """Dropdown update for the chosen approach's library; keeps the current
    selection when the same manual is indexed under both approaches."""
    store, _ = LIBRARIES[method]
    choices = [(d["name"], d["doc_id"]) for d in store.list_docs()]
    ids = [doc_id for _, doc_id in choices]
    return gr.update(choices=choices, value=current if current in ids else None)


def switch_method(method, doc_id):
    return _manuals_update(method, doc_id)


def refresh_library(method, doc_id):
    """Re-pull the library dataset (incremental) and refresh the dropdown,
    so manuals indexed after the Space booted show up without a restart."""
    sync_library()
    return _manuals_update(method, doc_id)


def ask_library(question, doc_id, method):
    if not doc_id:
        raise gr.Error("Pick a manual first.")
    store, pipeline = LIBRARIES[method]
    try:
        return pipeline.run(store, question, [doc_id], DEFAULT_TOP_K)
    except ValueError as e:
        raise gr.Error(str(e)) from e


with gr.Blocks(title="Repair Guy") as demo:
    gr.Markdown(
        "# 🔧 Repair Guy\n"
        "Ask questions over repair manuals, comparing two local-only retrieval "
        "approaches over the same library:\n"
        "- **Visual** — pages retrieved as images with "
        "[Nemotron ColEmbed v2](https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2) "
        "(late interaction, no parsing).\n"
        "- **Parsed** — pages parsed with "
        "[Nemotron Parse](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2), "
        "figures/tables described by MiniCPM-V, section chunks retrieved with "
        "[Llama Nemotron Embed](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2) "
        "and mapped back to their pages.\n\n"
        "Either way, MiniCPM-V reads the retrieved pages and answers."
    )
    with gr.Row():
        with gr.Column(scale=1):
            method_in = gr.Radio(
                METHOD_CHOICES, value="visual", label="Retrieval approach"
            )
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

    method_in.change(switch_method, inputs=[method_in, manual_in], outputs=[manual_in])
    lib_refresh_btn.click(
        refresh_library, inputs=[method_in, manual_in], outputs=[manual_in]
    )
    lib_ask_btn.click(
        ask_library, inputs=[lib_question_in, manual_in, method_in],
        outputs=[lib_answer_out, lib_pages_out],
    )
    lib_question_in.submit(
        ask_library, inputs=[lib_question_in, manual_in, method_in],
        outputs=[lib_answer_out, lib_pages_out],
    )
    demo.load(switch_method, inputs=[method_in, manual_in], outputs=[manual_in])


if __name__ == "__main__":
    demo.launch()
