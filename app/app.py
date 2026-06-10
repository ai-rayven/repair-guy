"""Gradio + ZeroGPU Space: upload a repair manual, ask questions.

Visual RAG with no parsing/chunking: every PDF page is embedded as an image
with Nemotron ColEmbed v2 (multi-vector, late interaction) when the manual is
uploaded. A question is answered by embedding the query, running MaxSim over
the page embeddings streamed from disk, and handing the top pages to MiniCPM-V.

Module layout:
  models/colembed.py   ColEmbed     — embedding model + GPU embed/search
  models/minicpm.py    MiniCPM      — remote VLM that answers over page images
  core/store.py        Store        — on-disk per-page token embeddings
  core/pdf.py          render_pages — PDF -> RGB page images (CPU)
  pipelines/ingest.py  IngestPipeline — PDF -> embeddings -> store
  pipelines/ask.py     AskPipeline    — question -> retrieve -> answer
  app.py               this file: builds the objects + Gradio UI
"""

import os

import gradio as gr

from core.constants import DEFAULT_TOP_K
from core.store import Store, slugify
from models.colembed import ColEmbed
from models.minicpm import MiniCPM
from pipelines.ask import AskPipeline
from pipelines.ingest import IngestPipeline

# Construct once at startup (the model loads onto cuda here, in the main process).
store = Store()
embedder = ColEmbed()
ingest_pipeline = IngestPipeline(embedder, store)
ask_pipeline = AskPipeline(embedder, store, MiniCPM())


def index_manual(pdf_file, progress=gr.Progress()):
    """Runs on upload: embed the manual's pages (or reuse a previous index)."""
    if not pdf_file:
        return None, ""
    name = os.path.splitext(os.path.basename(pdf_file))[0].replace("_", " ")
    doc_id = slugify(name)
    if store.exists(doc_id):
        pages = len(store.meta(doc_id)["pages"])
        return doc_id, f"**{name}** is already indexed ({pages} pages) — ask away."
    try:
        doc = ingest_pipeline.run(
            pdf_file, name, lambda frac, desc: progress(frac, desc=desc)
        )
    except ValueError as e:
        raise gr.Error(str(e)) from e
    return doc_id, f"Indexed **{doc['name']}** ({doc['pages']} pages) — ask away."


def answer_question(question, doc_id):
    if not doc_id:
        raise gr.Error("Upload a repair manual first.")
    try:
        return ask_pipeline.run(question, [doc_id], DEFAULT_TOP_K)
    except ValueError as e:
        raise gr.Error(str(e)) from e


with gr.Blocks(title="Repair Guy") as demo:
    gr.Markdown(
        "# 🔧 Repair Guy\n"
        "Upload a repair manual (PDF) and ask it questions. Pages are embedded "
        "with [Nemotron ColEmbed v2](https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2) "
        "on upload; answers come from MiniCPM-V reading the most relevant pages."
    )
    doc_state = gr.State(None)
    with gr.Row():
        with gr.Column(scale=1):
            pdf_in = gr.File(
                label="Repair manual (PDF)", file_types=[".pdf"], type="filepath"
            )
            status_out = gr.Markdown()
            question_in = gr.Textbox(
                label="Question",
                lines=2,
                placeholder="e.g. What is the tightening torque for the universal joint flange bolts?",
            )
            ask_btn = gr.Button("Ask", variant="primary")
        with gr.Column(scale=2):
            answer_out = gr.Markdown(label="Answer")
            pages_out = gr.Gallery(label="Pages used", columns=3, height=420)

    pdf_in.upload(index_manual, inputs=[pdf_in], outputs=[doc_state, status_out])
    ask_btn.click(
        answer_question, inputs=[question_in, doc_state], outputs=[answer_out, pages_out]
    )
    question_in.submit(
        answer_question, inputs=[question_in, doc_state], outputs=[answer_out, pages_out]
    )


if __name__ == "__main__":
    demo.launch()
