---
title: Repair Guy
emoji: 🔧
colorFrom: purple
colorTo: red
sdk: gradio
sdk_version: 6.18.0
python_version: '3.12'
app_file: app.py
pinned: false
preload_from_hub:
  - nvidia/nemotron-colembed-vl-4b-v2
  - openbmb/MiniCPM-V-4_5
  - nvidia/llama-nemotron-embed-vl-1b-v2
license: mit
---

# Repair Guy — two local-only RAG approaches over repair manuals

Ask questions over repair manuals, answered entirely by models running inside
the Space (no external endpoints). The project showcases two indexing
approaches over the same manuals:

- **Visual** — no parsing, no chunking: every page is embedded as an image
  with [Nemotron ColEmbed v2](https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2)
  (multi-vector, late interaction). At question time the query is embedded and
  scored against every page with MaxSim (batched torch matmuls, pages streamed
  from disk via numpy memmap).
- **Parsed** — pages are parsed with
  [Nemotron Parse v1.2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2),
  figures and tables are described by MiniCPM-V, and heading-based section
  chunks (descriptions spliced inline) are embedded with
  [Llama Nemotron Embed VL 1B v2](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2).
  Retrieval is dense cosine over chunks with parent-document lookup back to
  the pages they came from.

Either way, the top page images are read by MiniCPM-V 4.5 to produce a
grounded answer — one ZeroGPU call per question.

## Indexing (offline only)

All ingestion runs offline on Modal GPUs — `scripts/index_modal.py` in the
GitHub repo (`--method visual|parsed`) — and is pushed to the
[library dataset](https://huggingface.co/datasets/build-small-hackathon/repair-guy-library),
laid out as `<method>/<doc_id>/`. The Space syncs it to `/data/preindexed`
at startup (and via the 🔄 button). The parsed method runs as two Modal
functions because Nemotron Parse requires transformers 5.x while the rest of
the stack is on 4.x.

## Local UI development (mock mode)

To iterate on the Gradio UI without GPUs, model downloads or the HF library
sync, run with `MOCK_MODELS=1`. Drop any PDF into `app/data/mock_pdfs/` (or set
`MOCK_PDF_DIR`) and the app serves a canned answer grounded in real, rendered
pages of that PDF; both approaches are backed by the same mock, and the PDF
viewer, cited-pages gallery and jump-to-page all work. Only the answer text and
which pages it "cites" are faked.

```bash
cd app
uv pip install gradio pymupdf pillow numpy huggingface_hub   # light deps only
MOCK_MODELS=1 python app.py
```

The 🔄 Sync library button just re-scans the folder, so PDFs added while the app
is running show up without a restart. See `pipelines/mock_ask.py`.

## Space setup

- **Persistent storage** must be enabled (the library sync lives under
  `/data`). The visual index is the big one: roughly 5–12 MB per page of
  float16 token embeddings (a 1000-page manual is ~6–12 GB); the parsed index
  is a few MB per manual.
- Optional env vars: `LIBRARY_DATASET_ID`, `COLEMBED_MODEL_ID`,
  `MINICPM_MODEL_ID`, `NEMOTRON_EMBED_MODEL_ID`, `COLEMBED_ATTN` (defaults to
  `sdpa`; set `flash_attention_2` if flash-attn is installed).

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
