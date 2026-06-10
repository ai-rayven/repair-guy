import os

# Embedding model (late-interaction / ColBERT-style page embeddings).
COLEMBED_MODEL_ID = os.environ.get(
    "COLEMBED_MODEL_ID", "nvidia/nemotron-colembed-vl-4b-v2"
)
# sdpa works for this model and needs no extra wheels on ZeroGPU; set
# COLEMBED_ATTN=flash_attention_2 if flash-attn is installed.
COLEMBED_ATTN = os.environ.get("COLEMBED_ATTN", "sdpa")

# Page rendering (both for embedding at index time and for the answering model).
RENDER_DPI = 150

# Indexing: pages embedded per ZeroGPU call, and model batch size within a call.
# Chunking keeps each GPU call well under its duration limit; progress is
# reported between chunks.
EMBED_PAGES_PER_CALL = 16
EMBED_BATCH_SIZE = 4
EMBED_GPU_DURATION = 120

# Retrieval: MaxSim is computed on GPU over fixed-size batches of pages
# streamed from the on-disk store.
SCORE_PAGES_PER_BATCH = 32
SEARCH_GPU_DURATION = 60
DEFAULT_TOP_K = 3
MAX_TOP_K = 5

# Embedding store. HF Spaces persistent storage mounts at /data; fall back to a
# local directory for development.
STORE_DIR = os.environ.get("STORE_DIR") or (
    "/data/library"
    if os.path.isdir("/data")
    else os.path.join(os.path.dirname(os.path.dirname(__file__)), "library")
)
