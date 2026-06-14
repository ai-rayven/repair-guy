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

# Selectable agent brains, offered in the UI settings panel. ONE model is meant
# to be resident in VRAM at a time — switching evicts the previous and loads the
# next (models/minicpm_agent.use_model). NOTE: the FIRST/default brain is loaded at
# import (ZeroGPU emulation phase) and materialized into the forked GPU worker,
# where empty_cache() can't reclaim it — it stays stuck for the process. A brain
# SWITCHED IN at runtime must fit the headroom ABOVE the resident set
# (VLM+ColEmbed+embedder) + that stuck default (~15 GiB free on the 48 GiB `large`
# slice — see core/vram.py).
#
# The default is MiniCPM4.1-8B at int8 (~8.5 GiB via bitsandbytes). As the LONE
# stuck brain it REPLACES the old 1B default (it does not stack on it), so the
# resident set + one ~8.5 GiB brain still leaves room for the grounding spike — the
# same footprint class as MiniCPM3-4B, which was VRAM-vetted to fit. (bf16 8B at
# ~16 GiB still does NOT fit; int8 is what makes the 8B deployable as the default.)
# Caveat: with an ~8.5 GiB default already stuck, switching ANOTHER 4-8 GiB brain
# in at runtime for a UI A/B is tight and may OOM at grounding — the default path
# is fine.
# Each loads as an AutoModelForCausalLM; `trust_remote_code` (default False) flags
# the ones that ship custom modeling code (MiniCPM3 / MiniCPM4.1). `thinking` flags
# whether the chat template accepts enable_thinking (Qwen3, MiniCPM5, MiniCPM4.1 do
# — tool routing passes it False; MiniCPM3 does not). int8/4bit brains need
# bitsandbytes (requirements.txt). The FIRST entry is the default at boot; the
# minicpm5-1b entry stays selectable and still tracks the MINICPM_AGENT_MODEL_ID/
# REVISION env overrides. Only one brain is resident at a time.
AGENT_MODELS = [
    {
        "key": "minicpm4.1-8b-8bit",
        "label": "MiniCPM4.1 8B (8-bit)",
        "model_id": "openbmb/MiniCPM4.1-8B",
        "revision": None,
        # DEFAULT brain. Hybrid-reasoning 8B at int8 (~8.5 GiB) — the best
        # deployable eval config (0.85 tool / 0.91 args; with the v3 prompt it also
        # recovers half the coincidence fix). trust_remote_code custom modeling
        # (sparse "InfLLM v2" attention); runs clean on current transformers,
        # unlike MiniCPM3-4B. Quantized weights load straight onto the GPU, so
        # device_map is set (skips the host→device .to copy).
        "thinking": True,
        "trust_remote_code": True,
        "quantization": "8bit",
        "device_map": {"": 0},
    },
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
        "key": "qwen3-4b",
        "label": "Qwen3 4B",
        "model_id": "Qwen/Qwen3-4B",
        "revision": None,
        # Native Qwen3 arch (no trust_remote_code → immune to the custom-modeling
        # rot that breaks MiniCPM3-4B on current transformers). ~4B / ~8.1 GiB bf16,
        # same footprint class as MiniCPM3-4B, which was VRAM-vetted to fit; routing
        # passes enable_thinking=False.
        "thinking": True,
    },
    {
        "key": "minicpm3-4b",
        "label": "MiniCPM3 4B",
        "model_id": "openbmb/MiniCPM3-4B",
        # trust_remote_code model — pin a reviewed commit before a real deploy
        # (see use_model). Left unpinned here so the entry tracks latest.
        "revision": None,
        # MiniCPM3 has no hybrid-reasoning mode; its chat template doesn't take
        # enable_thinking, so leave it off (the kwarg is then omitted).
        "thinking": False,
        "trust_remote_code": True,
        # ~4B / ~8.6 GiB in bf16: with the resident VLM+ColEmbed+embedder (~28 GiB)
        # and the un-evictable default brain, it loads into the ~15.6 GiB free at
        # switch time with ~5 GiB to spare after the grounding spike. The 8B (16
        # GiB) didn't fit — see core/vram.py / the find-turn VRAM logs.
    },
    {
        "key": "minicpm4.1-8b",
        "label": "MiniCPM4.1 8B",
        "model_id": "openbmb/MiniCPM4.1-8B",
        "revision": None,
        # Hybrid-reasoning 8B (enable_thinking supported; routing passes it False).
        # trust_remote_code custom modeling (sparse "InfLLM v2" attention) — same
        # rot risk that broke MiniCPM3-4B on current transformers; pin a reviewed
        # commit for any real use. NOTE: 8B / ~16 GiB bf16 does NOT fit the
        # production find turn (the 8B didn't fit — VRAM logs); benchmarkable in
        # THIS eval (brain-only load) to gauge the quality ceiling, but shipping it
        # needs quantization (4-bit ~5-6 GiB) + a coexistence VRAM check.
        "thinking": True,
        "trust_remote_code": True,
    },
    {
        "key": "minicpm4.1-8b-4bit",
        "label": "MiniCPM4.1 8B (4-bit)",
        "model_id": "openbmb/MiniCPM4.1-8B",
        "revision": None,
        "thinking": True,
        "trust_remote_code": True,
        # bitsandbytes nf4 (~5-6 GiB vs ~16 GiB bf16) to fit the 8B into the
        # find-turn VRAM budget; quantized weights load straight onto the GPU, so
        # device_map is set (skips the host→device .to copy). The question this
        # answers: does 4-bit hold the bf16 8B's quality (0.87/0.86)?
        "quantization": "4bit",
        "device_map": {"": 0},
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
# back"); the live turn carries the full current page text (no table of contents).
AGENT_HISTORY_TURNS = 6

# One ZeroGPU call covers the whole question: query embedding + retrieval +
# page rendering + answer generation.
ASK_GPU_DURATION = 120

# Pages handed to the answering model.
DEFAULT_TOP_K = 3
MAX_TOP_K = 5

# ---------------------------------------------------------------------------
# Find-and-point (pipelines/agent_ask.py) — every non-obvious request is one
# GPU turn: the 1B agent loops over tools (search / go_to_page / circle) against
# the current page's text — no table of contents is injected.
# The frontend's breadcrumb section nav is client-side and never reaches the GPU.
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
