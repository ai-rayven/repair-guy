import os

# Embedding model (late-interaction / ColBERT-style page embeddings).
# Revisions are pinned because both models load trust_remote_code; bump
# deliberately after reviewing upstream changes.
COLEMBED_MODEL_ID = os.environ.get(
    "COLEMBED_MODEL_ID", "nvidia/nemotron-colembed-vl-4b-v2"
)
COLEMBED_REVISION = os.environ.get(
    "COLEMBED_REVISION", "0ed152d91f8ad4c5d48296b51c220f686641a398"
)
# sdpa works for this model and needs no extra wheels on ZeroGPU; set
# COLEMBED_ATTN=flash_attention_2 if flash-attn is installed.
COLEMBED_ATTN = os.environ.get("COLEMBED_ATTN", "sdpa")

# Page rendering (both for embedding at index time and for the answering model).
RENDER_DPI = 150

# Indexing: pages embedded per ZeroGPU call, and model batch size within a call.
# Chunking keeps each GPU call well under its duration limit; progress is
# reported between chunks. Bigger chunks mean fewer ZeroGPU queue waits per
# manual (~0.5s/page observed, so 64 pages ≈ 35-40s of a 240s budget — a
# 1000-page manual is ~16 GPU calls).
EMBED_PAGES_PER_CALL = 64
EMBED_BATCH_SIZE = 8
EMBED_GPU_DURATION = 240

# Retrieval: MaxSim is computed on GPU over fixed-size batches of pages
# streamed from the on-disk store.
SCORE_PAGES_PER_BATCH = 32
DEFAULT_TOP_K = 3
MAX_TOP_K = 5

# Answering model (runs locally on ZeroGPU).
MINICPM_MODEL_ID = os.environ.get("MINICPM_MODEL_ID", "openbmb/MiniCPM-V-4_5")
MINICPM_REVISION = os.environ.get(
    "MINICPM_REVISION", "fd3209b2e0580e346fc33d2c6f85b6e9332eecda"
)
ANSWER_MAX_NEW_TOKENS = 2048

# One ZeroGPU call covers the whole question: query embedding + MaxSim +
# page rendering + answer generation.
ASK_GPU_DURATION = 120

# Manual stores. HF Spaces persistent storage mounts at /data; fall back to a
# local directory for development. Pre-indexed manuals (large PDFs embedded
# offline, see scripts/index_local.py) are synced from the library dataset at
# startup; in-app uploads are indexed on ZeroGPU and page-capped to protect
# the Space's quota.
_DATA_ROOT = os.environ.get("DATA_ROOT") or (
    "/data"
    if os.path.isdir("/data")
    else os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
)
PREINDEXED_DIR = os.path.join(_DATA_ROOT, "preindexed")
UPLOADS_DIR = os.path.join(_DATA_ROOT, "uploads")
LIBRARY_DATASET_ID = os.environ.get(
    "LIBRARY_DATASET_ID", "build-small-hackathon/repair-guy-library"
)
MAX_UPLOAD_PAGES = int(os.environ.get("MAX_UPLOAD_PAGES", "50"))
