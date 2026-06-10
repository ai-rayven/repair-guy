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
license: mit
---

# Repair Guy — visual RAG over repair manuals

No parsing, no chunking, no figure descriptions: every PDF page is embedded
as an image with [Nemotron ColEmbed v2](https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2)
(multi-vector, late interaction). At question time the query is embedded and
scored against every page with MaxSim (batched torch matmuls on ZeroGPU,
pages streamed from disk via numpy memmap), and the top-K page images are
handed to a MiniCPM-V endpoint to produce a grounded answer.

## Space setup

- **Persistent storage** must be enabled (embeddings + PDFs live under
  `/data/library`). Budget roughly 5–12 MB per page of float16 token
  embeddings; a 300-page manual is ~2–3.5 GB.
- **Secret `MINICPM_API_KEY`** — bearer token for the MiniCPM endpoint
  (`MINICPM_BASE_URL` / `MINICPM_MODEL` are env-overridable).
- Optional: `COLEMBED_MODEL_ID` (defaults to the 4B model),
  `COLEMBED_ATTN` (defaults to `sdpa`; set `flash_attention_2` if flash-attn
  is installed).

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
