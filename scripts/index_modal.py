#!/usr/bin/env python3
"""Index a repair manual on Modal GPUs and push it to the library dataset
the Space syncs from.

One-time setup:
    uv tool install modal
    modal setup
    modal secret create huggingface HF_TOKEN=<token with write access to the dataset>

Usage (from the repo root):
    modal run scripts/index_modal.py --pdf manual.pdf --name "Toyota 8FGU"
    modal run scripts/index_modal.py --pdf manual.pdf --name "Toyota 8FGU" --method parsed
    modal run scripts/index_modal.py --pdf manual.pdf --no-push   # dry run

Methods (each lands under <method>/<doc_id>/ in the dataset):
    visual  — ColEmbed page embeddings (one GPU function).
    parsed  — two GPU functions in DIFFERENT images, because Nemotron Parse
              requires transformers==5.6.1 while MiniCPM/Nemotron Embed need
              <5: parse_manual (stage 1) -> build_parsed (stage 2).

The GPU type can be overridden at launch: MODAL_GPU=A100 modal run ...
"""

import os
from pathlib import Path

import modal

APP_DIR = Path(__file__).parent.parent / "app"

# Visual ingest + parsed build stage: the Space's environment (transformers 4.x).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "torchvision",
        "transformers>=4.57.2,<5",
        "accelerate",
        "datasets",  # imported by the ColEmbed remote modeling code
        "spaces",  # imported by the app code; its GPU decorator is a no-op off-Spaces
        "pymupdf",
        "pillow",
        "numpy",
        "huggingface_hub",
    )
    .add_local_dir(str(APP_DIR), remote_path="/root/app")
)

# Parsed parse stage: Nemotron Parse's own environment (transformers 5.6.1,
# plus the deps its remote code needs).
parse_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "torchvision",
        "transformers==5.6.1",
        "accelerate",
        "timm",
        "albumentations",
        "einops",
        "open_clip_torch",
        "beautifulsoup4",
        "lxml",
        "pymupdf",
        "pillow",
        "numpy",
        "huggingface_hub",
    )
    .add_local_dir(str(APP_DIR), remote_path="/root/app")
)

app = modal.App("repair-guy-indexer", image=image)

# Model weights download once into this volume, not per run.
hf_cache = modal.Volume.from_name("repair-guy-hf-cache", create_if_missing=True)

GPU = os.environ.get("MODAL_GPU", "L40S")
COMMON = dict(
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/root/.cache/huggingface": hf_cache},
)


def _setup(pdf_bytes: bytes) -> str:
    import sys

    sys.path.insert(0, "/root/app")
    pdf_path = "/tmp/manual.pdf"
    Path(pdf_path).write_bytes(pdf_bytes)
    return pdf_path


def _push_doc(store_root: str, method: str, doc: dict) -> None:
    from core.constants import LIBRARY_DATASET_ID  # _setup already put app on sys.path
    from huggingface_hub import create_repo, upload_folder

    create_repo(LIBRARY_DATASET_ID, repo_type="dataset", exist_ok=True)
    upload_folder(
        repo_id=LIBRARY_DATASET_ID,
        repo_type="dataset",
        folder_path=f"{store_root}/{doc['doc_id']}",
        path_in_repo=f"{method}/{doc['doc_id']}",
        commit_message=f"Add {doc['name']} ({doc['pages']} pages, {method})",
    )
    print(f"Pushed to https://huggingface.co/datasets/{LIBRARY_DATASET_ID}")


@app.function(gpu=GPU, timeout=3 * 60 * 60, **COMMON)
def index_visual(pdf_bytes: bytes, name: str, push: bool) -> dict:
    pdf_path = _setup(pdf_bytes)
    from core.visual_store import VisualStore
    from models.colembed import ColEmbed
    from pipelines.visual_ingest import VisualIngestPipeline

    pipeline = VisualIngestPipeline(ColEmbed(), VisualStore("/tmp/store"))
    doc = None
    for event in pipeline.run(pdf_path, name):
        if event[0] == "progress":
            print(f"Embedded {event[1]}/{event[2]} pages")
        else:
            doc = event[1]

    if push:
        _push_doc("/tmp/store", "visual", doc)
    return doc


@app.function(image=parse_image, gpu=GPU, timeout=2 * 60 * 60, **COMMON)
def parse_manual(pdf_bytes: bytes) -> list:
    """Parsed stage 1: page elements via Nemotron Parse (transformers 5.x)."""
    pdf_path = _setup(pdf_bytes)
    from pipelines.parsed_ingest import ParseStage

    stage = ParseStage()
    pages = None
    for event in stage.run(pdf_path):
        if event[0] == "progress":
            print(f"Parsed {event[1]}/{event[2]} pages")
        else:
            pages = event[1]
    return pages


@app.function(gpu=GPU, timeout=6 * 60 * 60, **COMMON)
def build_parsed(pdf_bytes: bytes, parsed_pages: list, name: str, push: bool) -> dict:
    """Parsed stage 2: MiniCPM descriptions + chunking + embeddings
    (transformers 4.x, same env as the Space)."""
    pdf_path = _setup(pdf_bytes)
    from core.parsed_store import ParsedStore
    from pipelines.parsed_ingest import BuildStage

    stage = BuildStage(ParsedStore("/tmp/store"))
    doc = None
    for event in stage.run(pdf_path, parsed_pages, name):
        if event[0] == "progress":
            print(f"Described {event[1]}/{event[2]} pages")
        else:
            doc = event[1]
    print(f"Built {doc['chunks']} chunks from {doc['pages']} pages")

    if push:
        _push_doc("/tmp/store", "parsed", doc)
    return doc


@app.local_entrypoint()
def main(pdf: str, name: str = "", method: str = "visual", push: bool = True):
    if method not in ("visual", "parsed"):
        raise SystemExit(f"Unknown --method {method!r} (visual or parsed)")
    pdf_bytes = Path(pdf).read_bytes()
    if method == "visual":
        doc = index_visual.remote(pdf_bytes, name, push)
    else:
        pages = parse_manual.remote(pdf_bytes)
        doc = build_parsed.remote(pdf_bytes, pages, name, push)
    print(f"Done: {doc['name']} ({doc['pages']} pages, doc_id={doc['doc_id']}, method={method})")
