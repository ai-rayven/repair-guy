---
title: Repair Guy
emoji: 🔧
colorFrom: purple
colorTo: red
sdk: gradio
sdk_version: 6.16.0
python_version: '3.12'
app_file: app.py
pinned: false
preload_from_hub:
  - nvidia/nemotron-colembed-vl-4b-v2
  - openbmb/MiniCPM-V-4_5
license: mit
---

# Repair Guy — visual RAG over repair manuals

No parsing, no chunking, no figure descriptions: every PDF page is embedded
as an image with [Nemotron ColEmbed v2](https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2)
(multi-vector, late interaction). At question time the query is embedded and
scored against every page with MaxSim (batched torch matmuls on ZeroGPU,
pages streamed from disk via numpy memmap), and the top-K page images are
read by MiniCPM-V 4.5 — also on ZeroGPU — to produce a grounded answer.
Everything runs inside the Space; no external endpoints.

## Two tabs

- **Library** — large manuals indexed offline (`scripts/index_modal.py` in the
  GitHub repo runs it on a Modal GPU; `index_local.py` if you have a CUDA box)
  and pushed to the
  [library dataset](https://huggingface.co/datasets/build-small-hackathon/repair-guy-library);
  the Space syncs it to `/data/preindexed` at startup.
- **Upload your own** — indexes on the Space's ZeroGPU, capped at
  `MAX_UPLOAD_PAGES` (default 50) to protect quota.

## Space setup

- **Persistent storage** must be enabled (the library sync and uploads live
  under `/data`). Budget roughly 5–12 MB per page of float16 token
  embeddings; a 1000-page manual is ~6–12 GB — size the tier to the library.
- Optional env vars: `LIBRARY_DATASET_ID`, `MAX_UPLOAD_PAGES`,
  `COLEMBED_MODEL_ID` (defaults to the 4B model), `MINICPM_MODEL_ID`
  (defaults to `openbmb/MiniCPM-V-4_5`), `COLEMBED_ATTN` (defaults to `sdpa`;
  set `flash_attention_2` if flash-attn is installed).

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
