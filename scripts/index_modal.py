#!/usr/bin/env python3
"""Index a repair manual on a Modal GPU and push it to the library dataset
the Space syncs from.

One-time setup:
    pip install modal
    modal setup
    modal secret create huggingface HF_TOKEN=<token with write access to the dataset>

Usage (from the repo root):
    modal run scripts/index_modal.py --pdf manual.pdf --name "Toyota 8FGU"
    modal run scripts/index_modal.py --pdf manual.pdf --no-push   # dry run

The GPU type can be overridden at launch: MODAL_GPU=A100 modal run ...
"""

import os
from pathlib import Path

import modal

APP_DIR = Path(__file__).parent.parent / "app"

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

app = modal.App("repair-guy-indexer", image=image)

# Model weights (~9.6 GB) download once into this volume, not per run.
hf_cache = modal.Volume.from_name("repair-guy-hf-cache", create_if_missing=True)


@app.function(
    gpu=os.environ.get("MODAL_GPU", "L40S"),
    timeout=3 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/root/.cache/huggingface": hf_cache},
)
def index_pdf(pdf_bytes: bytes, name: str, push: bool) -> dict:
    import sys

    sys.path.insert(0, "/root/app")

    from core.constants import LIBRARY_DATASET_ID
    from core.store import Store
    from models.colembed import ColEmbed
    from pipelines.ingest import IngestPipeline

    pdf_path = "/tmp/manual.pdf"
    Path(pdf_path).write_bytes(pdf_bytes)

    pipeline = IngestPipeline(ColEmbed(), Store("/tmp/store"))
    doc = None
    for event in pipeline.run(pdf_path, name):
        if event[0] == "progress":
            print(f"Embedded {event[1]}/{event[2]} pages")
        else:
            doc = event[1]

    if push:
        from huggingface_hub import create_repo, upload_folder

        create_repo(LIBRARY_DATASET_ID, repo_type="dataset", exist_ok=True)
        upload_folder(
            repo_id=LIBRARY_DATASET_ID,
            repo_type="dataset",
            folder_path=f"/tmp/store/{doc['doc_id']}",
            path_in_repo=doc["doc_id"],
            commit_message=f"Add {doc['name']} ({doc['pages']} pages)",
        )
        print(f"Pushed to https://huggingface.co/datasets/{LIBRARY_DATASET_ID}")
    return doc


@app.local_entrypoint()
def main(pdf: str, name: str = "", push: bool = True):
    doc = index_pdf.remote(Path(pdf).read_bytes(), name, push)
    print(f"Done: {doc['name']} ({doc['pages']} pages, doc_id={doc['doc_id']})")
