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

# Let MiniCPM-V reason (legend → callout-number → leader-line → part) before
# committing to a grounding box for "circle the <thing>". This is only the
# DEFAULT — the UI settings panel sends a per-request override (see api_find's
# `think`). Off by default: it roughly multiplies grounding latency (64 → ~512
# generated tokens) and mainly helps exploded-diagram callouts, not the
# dense-table wrong-row misses.
GROUND_ENABLE_THINKING = os.environ.get("GROUND_ENABLE_THINKING", "").lower() in (
    "1",
    "true",
    "yes",
)
# Token budgets for one grounding generation. A bare <box> fits in 64; a think
# trace does not (it gets cut off before the box), so the budget tracks whether
# thinking is on for that call.
GROUND_BOX_MAX_NEW_TOKENS = 64
GROUND_THINK_MAX_NEW_TOKENS = 512

# Agent brain: MiniCPM5-1B — a standard LlamaForCausalLM (no trust_remote_code),
# 131k context. The small TEXT model that drives the find-and-point loop: each
# step it picks ONE tool from the conversation so far, the manual's table of
# contents, and the whole text of the page being viewed. The MiniCPM-V VLM above
# stays the "eyes" (grounding the circle + the ingest figure/table descriptions);
# this is the "brain". Revision is left unpinned (standard architecture, no
# remote code to drift) but overridable; pin a commit before a real deploy.
MINICPM_AGENT_MODEL_ID = os.environ.get("MINICPM_AGENT_MODEL_ID", "openbmb/MiniCPM5-1B")
MINICPM_AGENT_REVISION = os.environ.get("MINICPM_AGENT_REVISION", "") or None

# Selectable agent brains, offered in the UI settings panel. ONE model is
# resident in VRAM at a time — switching evicts the previous and loads the next
# (models/minicpm_agent.use_model). Each loads as an AutoModelForCausalLM;
# `trust_remote_code` (default False) flags the ones that ship custom modeling
# code (MiniCPM4.1-8B). `thinking` flags whether the chat template accepts
# enable_thinking (Qwen3 and MiniCPM do — tool routing passes it False). The
# FIRST entry is the default at boot and tracks the MINICPM_AGENT_MODEL_ID/
# REVISION env overrides, so existing config still applies. Only one is resident
# at a time, so each stays well under the hackathon's 32B total-params budget.
AGENT_MODELS = [
    {
        "key": "minicpm5-1b",
        "label": "MiniCPM5 1B",
        "model_id": MINICPM_AGENT_MODEL_ID,
        "revision": MINICPM_AGENT_REVISION,
        "thinking": True,
    },
    {
        "key": "qwen3-1.7b",
        "label": "Qwen3 1.7B",
        "model_id": "Qwen/Qwen3-1.7B",
        "revision": None,
        "thinking": True,
    },
    {
        "key": "qwen3-0.6b",
        "label": "Qwen3 0.6B",
        "model_id": "Qwen/Qwen3-0.6B",
        "revision": None,
        "thinking": True,
    },
    {
        "key": "minicpm4.1-8b",
        "label": "MiniCPM4.1 8B",
        "model_id": "openbmb/MiniCPM4.1-8B",
        # trust_remote_code model — pin a reviewed commit before a real deploy
        # (see use_model). Left unpinned here so the entry tracks latest.
        "revision": None,
        "thinking": True,
        "trust_remote_code": True,
        # Ships custom sparse attention (InfLLM v2) — let its modeling code pick
        # the attn impl instead of forcing sdpa like the standard-arch brains.
        "attn_implementation": None,
    },
]
DEFAULT_AGENT_MODEL = AGENT_MODELS[0]["key"]
# A tool-call decision is short JSON; a rerank reply is a single number. 96 was
# too tight — the 1B writes verbose search queries and was getting CUT OFF
# mid-string (unterminated JSON → parse fail → wasted retry), seen live.
AGENT_MAX_NEW_TOKENS = 128
# Backstop on tool steps within one turn, so a confused loop can't run forever.
AGENT_MAX_STEPS = 6
# ColEmbed shortlist size the search tool retrieves (the eval default).
AGENT_SEARCH_CANDIDATES = 5
# Past turns of conversation fed back as memory (resolve "the other one", "go
# back"); the live turn carries the table of contents and full current page.
AGENT_HISTORY_TURNS = 6

# One ZeroGPU call covers the whole question: query embedding + retrieval +
# page rendering + answer generation.
ASK_GPU_DURATION = 120

# Pages handed to the answering model.
DEFAULT_TOP_K = 3
MAX_TOP_K = 5

# ---------------------------------------------------------------------------
# Find-and-point (pipelines/agent_ask.py) — every non-obvious request is one
# GPU turn: the 1B agent loops over tools (search / go_to_section / circle)
# against the table of contents and the current page's text. Deterministic
# navigation is CPU-only (core/sections.py) and never reaches the GPU.
# ---------------------------------------------------------------------------

# One ZeroGPU call covers a whole agent turn: up to AGENT_MAX_STEPS tool-choice
# generations, ColEmbed retrieval + page rendering and the 1B rerank inside a
# search step, and one MiniCPM-V grounding generation for a circle.
FIND_GPU_DURATION = 180

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
