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
in one ZeroGPU call per question. The UI is a side-by-side comparison: one
manual, one question, and each approach answers in its own column (its own
GPU call) so retrieval quality and latency can be compared directly.

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
import time

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


def _manual_choices() -> list[tuple[str, str]]:
    """One shared dropdown across both libraries (doc ids are name slugs, so
    the same manual lands on the same id in both); manuals indexed under only
    one approach are labeled with it."""
    docs: dict[str, dict] = {}
    for method, (store, _) in LIBRARIES.items():
        for d in store.list_docs():
            entry = docs.setdefault(d["doc_id"], {"name": d["name"], "methods": []})
            entry["methods"].append(method)
    choices = []
    for doc_id, info in sorted(docs.items(), key=lambda kv: kv[1]["name"].lower()):
        label = info["name"]
        if len(info["methods"]) < len(LIBRARIES):
            label += f" — {info['methods'][0]} only"
        choices.append((label, doc_id))
    return choices


def refresh_library(doc_id):
    """Re-pull the library dataset (incremental) and refresh the dropdown,
    so manuals indexed after the Space booted show up without a restart."""
    sync_library()
    choices = _manual_choices()
    ids = [v for _, v in choices]
    return gr.update(choices=choices, value=doc_id if doc_id in ids else None)


def _ask(method: str, question, doc_id):
    """One approach's column: (timing line, answer markdown, gallery).
    Soft in-column messages instead of gr.Error so that when both columns run
    off one click, one failing doesn't kill the other."""
    if not doc_id:
        return "", "*Pick a manual first.*", []
    store, pipeline = LIBRARIES[method]
    if not store.exists(doc_id):
        return "", f"*This manual isn't indexed with the {method} approach yet.*", []
    start = time.monotonic()
    try:
        answer, gallery = pipeline.run(store, question, [doc_id], DEFAULT_TOP_K)
    except ValueError as e:
        return "", f"*{e}*", []
    return f"⏱️ answered in {time.monotonic() - start:.1f}s", answer, gallery


def ask_visual(question, doc_id):
    return _ask("visual", question, doc_id)


def ask_parsed(question, doc_id):
    return _ask("parsed", question, doc_id)


CSS = """
.app-header { text-align: center; margin: 0.5em 0 0.2em; }
.app-header p { color: var(--body-text-color-subdued); margin-top: 0.3em; }
.approach-card {
    border-radius: 14px !important;
    border-top: 4px solid var(--card-accent) !important;
}
.visual-card { --card-accent: #e8590c; }
.parsed-card { --card-accent: #0c8599; }
.approach-card h3 { margin: 0.1em 0 0; }
.pipeline-steps {
    color: var(--body-text-color-subdued);
    font-size: 0.85em;
    line-height: 1.7;
}
.pipeline-steps code { font-size: 0.95em; }
.timing { color: var(--body-text-color-subdued); font-size: 0.9em; min-height: 1.2em; }
"""

VISUAL_CARD = """### 🖼️ Visual
**ColEmbed late interaction** — pages stay images; nothing is parsed or chunked.

<div class="pipeline-steps">

`page image` → `multi-vector embedding` → `MaxSim vs. query` → `top pages` → `MiniCPM‑V answers`

**Index:** heavy (5–12 MB/page) · **Strengths:** immune to parsing errors, sees layout and diagrams natively
</div>"""

PARSED_CARD = """### 📄 Parsed
**Parse + dense chunks** — pages become structured text; figures and tables become descriptions.

<div class="pipeline-steps">

`Nemotron Parse` → `MiniCPM‑V describes figures/tables` → `section chunks` → `dense cosine` → `parent pages` → `MiniCPM‑V answers`

**Index:** tiny (a few MB/manual) · **Strengths:** inspectable chunks, precise hits on one table or diagram
</div>"""


with gr.Blocks(title="Repair Guy") as demo:
    gr.Markdown(
        "# 🔧 Repair Guy\n"
        "Two local-only ways to search a repair manual, side by side — pick a "
        "manual, ask once, compare what each approach retrieves and answers.",
        elem_classes="app-header",
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            manual_in = gr.Dropdown(label="Manual", choices=[])
            refresh_btn = gr.Button("🔄 Sync library", size="sm")
        with gr.Column(scale=2):
            question_in = gr.Textbox(
                label="Question",
                lines=2,
                placeholder="e.g. What is the tightening torque for the universal joint flange bolts?",
            )
            with gr.Row():
                both_btn = gr.Button("⚡ Ask both", variant="primary")

    with gr.Row(equal_height=False):
        with gr.Column(variant="panel", elem_classes="approach-card visual-card"):
            gr.Markdown(VISUAL_CARD)
            vis_btn = gr.Button("Ask with Visual", size="sm")
            vis_time = gr.Markdown(elem_classes="timing")
            vis_answer = gr.Markdown(label="Answer")
            vis_pages = gr.Gallery(label="Pages used", columns=3, height=300)
        with gr.Column(variant="panel", elem_classes="approach-card parsed-card"):
            gr.Markdown(PARSED_CARD)
            par_btn = gr.Button("Ask with Parsed", size="sm")
            par_time = gr.Markdown(elem_classes="timing")
            par_answer = gr.Markdown(label="Answer")
            par_pages = gr.Gallery(label="Pages used", columns=3, height=300)

    inputs = [question_in, manual_in]
    vis_outputs = [vis_time, vis_answer, vis_pages]
    par_outputs = [par_time, par_answer, par_pages]

    vis_btn.click(ask_visual, inputs=inputs, outputs=vis_outputs)
    par_btn.click(ask_parsed, inputs=inputs, outputs=par_outputs)
    # Two listeners on one event: both columns run off a single click/submit.
    both_btn.click(ask_visual, inputs=inputs, outputs=vis_outputs)
    both_btn.click(ask_parsed, inputs=inputs, outputs=par_outputs)
    question_in.submit(ask_visual, inputs=inputs, outputs=vis_outputs)
    question_in.submit(ask_parsed, inputs=inputs, outputs=par_outputs)

    refresh_btn.click(refresh_library, inputs=[manual_in], outputs=[manual_in])
    demo.load(lambda: gr.update(choices=_manual_choices()), outputs=[manual_in])


# Gradio 6 takes theme/css at launch(), not in the Blocks constructor.
LAUNCH_KWARGS = dict(theme=gr.themes.Soft(primary_hue="orange"), css=CSS)

if __name__ == "__main__":
    demo.launch(**LAUNCH_KWARGS)
