import os

# ---------------------------------------------------------------------------
# Shared (both approaches)
# ---------------------------------------------------------------------------

# Page rendering (embedding/parsing at index time, and the answering model).
RENDER_DPI = 150

# Answering model (runs locally on ZeroGPU). Revisions are pinned because all
# models here load trust_remote_code; bump deliberately after reviewing
# upstream changes.
MINICPM_MODEL_ID = os.environ.get("MINICPM_MODEL_ID", "openbmb/MiniCPM-V-4_5")
MINICPM_REVISION = os.environ.get(
    "MINICPM_REVISION", "fd3209b2e0580e346fc33d2c6f85b6e9332eecda"
)
ANSWER_MAX_NEW_TOKENS = 2048

# One ZeroGPU call covers the whole question: query embedding + retrieval +
# page rendering + answer generation.
ASK_GPU_DURATION = 120

# Pages handed to the answering model.
DEFAULT_TOP_K = 3
MAX_TOP_K = 5

# ---------------------------------------------------------------------------
# Agent chat (multi-turn history + tool calls, pipelines/agent_ask.py)
# ---------------------------------------------------------------------------

# Tool-call iterations MiniCPM gets per question before being forced to
# answer. The expected pattern is search → show_page → answer (3 generations),
# so 5 leaves room for one retry search.
AGENT_MAX_STEPS = 5
# One ZeroGPU call covers a whole chat turn: up to AGENT_MAX_STEPS generations
# plus retrieval and page rendering between them, and possibly a history
# summarization pass first.
AGENT_GPU_DURATION = 240

# chat()'s max_inp_length is 16384 tokens and each retrieved page image costs
# roughly 600-1000 of them, so past turns are kept as text only and summarized
# once they grow past this budget (counted with the MiniCPM tokenizer).
HISTORY_TOKEN_BUDGET = 6000
# Most recent messages kept verbatim when the older history is summarized.
HISTORY_KEEP_MESSAGES = 4
SUMMARY_MAX_NEW_TOKENS = 512

# ---------------------------------------------------------------------------
# Visual approach: ColEmbed late-interaction page embeddings + MaxSim
# ---------------------------------------------------------------------------

COLEMBED_MODEL_ID = os.environ.get(
    "COLEMBED_MODEL_ID", "nvidia/nemotron-colembed-vl-4b-v2"
)
COLEMBED_REVISION = os.environ.get(
    "COLEMBED_REVISION", "0ed152d91f8ad4c5d48296b51c220f686641a398"
)
# sdpa works for this model and needs no extra wheels on ZeroGPU; set
# COLEMBED_ATTN=flash_attention_2 if flash-attn is installed.
COLEMBED_ATTN = os.environ.get("COLEMBED_ATTN", "sdpa")

# Indexing: pages embedded per GPU call, and model batch size within a call.
EMBED_PAGES_PER_CALL = 64
EMBED_BATCH_SIZE = 8
EMBED_GPU_DURATION = 240

# Retrieval: MaxSim is computed on GPU over fixed-size batches of pages
# streamed from the on-disk store.
SCORE_PAGES_PER_BATCH = 32

# ---------------------------------------------------------------------------
# Parsed approach: Nemotron Parse -> MiniCPM figure/table descriptions ->
# section chunks -> dense embeddings + cosine retrieval
# ---------------------------------------------------------------------------

# Nemotron Parse requires transformers==5.6.1, which is incompatible with
# ColEmbed/MiniCPM (<5). It therefore runs only in the dedicated parse stage
# on Modal (scripts/index_modal.py) and is never imported on the Space.
NEMOTRON_PARSE_MODEL_ID = os.environ.get(
    "NEMOTRON_PARSE_MODEL_ID", "nvidia/NVIDIA-Nemotron-Parse-v1.2"
)
NEMOTRON_PARSE_REVISION = os.environ.get(
    "NEMOTRON_PARSE_REVISION", "2bd0189bffd6cdded6280d9f22a4077b25a504e3"
)
# Pages parsed per generate call at index time (identical task prompt per
# page, so the batch needs no padding).
PARSE_BATCH_SIZE = 16

# Dense bi-encoder for chunk/query embeddings (2048-dim, mean-pooled).
# Its remote code supports transformers 4.56+ — same env as the Space.
NEMOTRON_EMBED_MODEL_ID = os.environ.get(
    "NEMOTRON_EMBED_MODEL_ID", "nvidia/llama-nemotron-embed-vl-1b-v2"
)
NEMOTRON_EMBED_REVISION = os.environ.get(
    "NEMOTRON_EMBED_REVISION", "0c6f636ed4c022e427277c4c336054d6cdffaa87"
)
EMBED_TEXT_MAX_LENGTH = 8192  # processor token budget for text-only inputs
EMBED_TEXT_BATCH_SIZE = 8

# MiniCPM descriptions generated at ingest for Picture/Table elements.
# Each description is conditioned on document context (manual name, section
# heading, adjacent caption, page text) so it uses the manual's terminology;
# the page-text part of that context is capped at this many characters.
DESCRIBE_MAX_NEW_TOKENS = 256
DESCRIBE_CONTEXT_MAX_CHARS = 1200
# Figure/table descriptions generated per batched MiniCPM chat() call.
DESCRIBE_BATCH_SIZE = 16
# Picture bboxes with either side smaller than this (pixels at RENDER_DPI)
# are skipped — icons, bullets, print artifacts.
FIGURE_MIN_SIDE_PX = 40

# Section chunking (core/chunking.py): a Title/Section-header closes the
# current section unless it is still under SECTION_MIN_CHARS (sparse pages
# merge forward into the next section); sections over SECTION_MAX_CHARS split
# at element boundaries with the heading repeated.
SECTION_MIN_CHARS = 200
SECTION_MAX_CHARS = 6000

# Retrieval: chunk candidates scored by cosine before the page budget
# (top_k pages) is applied via parent-document lookup.
PARSED_TOP_CHUNKS = 8

# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------

# HF Spaces persistent storage mounts at /data; fall back to a local directory
# for development. The library dataset mirrors PREINDEXED_DIR: one top-level
# directory per indexing method, one doc directory per manual under it.
_DATA_ROOT = os.environ.get("DATA_ROOT") or (
    "/data"
    if os.path.isdir("/data")
    else os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
)
PREINDEXED_DIR = os.path.join(_DATA_ROOT, "preindexed")
VISUAL_SUBDIR = "visual"
PARSED_SUBDIR = "parsed"
LIBRARY_DATASET_ID = os.environ.get(
    "LIBRARY_DATASET_ID", "build-small-hackathon/repair-guy-library"
)

# ---------------------------------------------------------------------------
# Local mock mode (UI iteration with no GPU / model downloads / HF sync)
# ---------------------------------------------------------------------------

# With MOCK_MODELS set, app.py serves canned answers grounded in real, rendered
# pages of any PDF dropped into MOCK_PDF_DIR. Nothing on the mock path imports
# torch / spaces / the model modules (which load CUDA at import), so the Gradio
# UI boots instantly on a laptop. See pipelines/mock_ask.py.
MOCK_MODELS = os.environ.get("MOCK_MODELS", "").lower() in ("1", "true", "yes")
MOCK_PDF_DIR = os.environ.get("MOCK_PDF_DIR") or os.path.join(_DATA_ROOT, "mock_pdfs")
