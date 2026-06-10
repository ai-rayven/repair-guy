"""Nemotron ColEmbed v2: late-interaction page embeddings + MaxSim retrieval.

Two ZeroGPU entry points:
  _embed_pages_on_gpu   page images -> per-page token embeddings (index time)
  _search_on_gpu        question -> top-K (doc, page, score) via MaxSim over
                        batches of page embeddings streamed from the store

The model is a module-level global: ZeroGPU packs module-level CUDA tensors at
startup and shares them with the GPU worker, whereas function arguments are
pickled — and the trust_remote_code model class is not picklable.

forward_images/forward_queries return zero-padded [batch, tokens, dim] tensors
with real tokens L2-normalized, so padding rows are exactly zero. We strip them
before storing and rely on the same property when scoring zero-padded batches.
"""

from __future__ import annotations

import sys

import numpy as np
import spaces
import torch
from PIL import Image
from transformers import AutoModel

from core.constants import (
    COLEMBED_ATTN,
    COLEMBED_MODEL_ID,
    EMBED_BATCH_SIZE,
    EMBED_GPU_DURATION,
    SCORE_PAGES_PER_BATCH,
    SEARCH_GPU_DURATION,
)

_MODEL = (
    AutoModel.from_pretrained(
        COLEMBED_MODEL_ID,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation=COLEMBED_ATTN,
    )
    .to("cuda")
    .eval()
)

# The remote code's forward_documents hardcodes DataLoader(num_workers=8), but
# the ZeroGPU worker is a daemonic process and may not spawn children
# ("daemonic processes are not allowed to have children"). Patch the DataLoader
# name in the model's own module to force in-process loading. A subclass (not a
# wrapper function) because the remote code also uses the name in isinstance().
_remote_module = sys.modules[type(_MODEL).__module__]


class _SingleProcessDataLoader(_remote_module.DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs["num_workers"] = 0
        super().__init__(*args, **kwargs)


_remote_module.DataLoader = _SingleProcessDataLoader


@spaces.GPU(duration=EMBED_GPU_DURATION)
def _embed_pages_on_gpu(images: list[Image.Image]) -> list[np.ndarray]:
    with torch.no_grad():
        embs = _MODEL.forward_images(images, batch_size=EMBED_BATCH_SIZE)
    out = []
    for emb in embs:  # [tokens, dim]; zero rows are padding
        mask = emb.abs().sum(dim=-1) > 0
        out.append(emb[mask].to(torch.float16).cpu().numpy())
    return out


@spaces.GPU(duration=SEARCH_GPU_DURATION)
def _search_on_gpu(question: str, store, doc_ids, top_k: int):
    results = []
    with torch.no_grad():
        q = _MODEL.forward_queries([question], batch_size=1)[0].to(torch.float16)
        for refs, batch in store.iter_page_batches(doc_ids, SCORE_PAGES_PER_BATCH):
            emb = torch.from_numpy(batch).to(q.device)  # [B, T, D] float16
            sim = torch.einsum("qd,btd->bqt", q, emb).float()
            scores = sim.amax(dim=2).sum(dim=1)  # MaxSim: max over doc tokens, sum over query tokens
            results.extend(
                (doc_id, page, s)
                for (doc_id, page), s in zip(refs, scores.tolist())
            )
    results.sort(key=lambda r: r[2], reverse=True)
    return results[:top_k]


class ColEmbed:
    MODEL_ID = COLEMBED_MODEL_ID

    def embed_pages(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Embed page images -> list of [n_tokens, dim] float16 arrays."""
        return _embed_pages_on_gpu(images)

    def search(
        self, question: str, store, doc_ids: list[str] | None, top_k: int
    ) -> list[tuple[str, int, float]]:
        """Return the top_k (doc_id, page_num, score) across the given docs."""
        return _search_on_gpu(question, store, doc_ids, top_k)
