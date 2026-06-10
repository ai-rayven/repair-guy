"""Nemotron ColEmbed v2: late-interaction page embeddings + MaxSim retrieval.

Two ZeroGPU entry points:
  _embed_pages_on_gpu   page images -> per-page token embeddings (index time)
  _search_on_gpu        question -> top-K (doc, page, score) via MaxSim over
                        batches of page embeddings streamed from the store

forward_images/forward_queries return zero-padded [batch, tokens, dim] tensors
with real tokens L2-normalized, so padding rows are exactly zero. We strip them
before storing and rely on the same property when scoring zero-padded batches.
"""

from __future__ import annotations

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


@spaces.GPU(duration=EMBED_GPU_DURATION)
def _embed_pages_on_gpu(model, images: list[Image.Image]) -> list[np.ndarray]:
    with torch.no_grad():
        embs = model.forward_images(images, batch_size=EMBED_BATCH_SIZE)
    out = []
    for emb in embs:  # [tokens, dim]; zero rows are padding
        mask = emb.abs().sum(dim=-1) > 0
        out.append(emb[mask].to(torch.float16).cpu().numpy())
    return out


@spaces.GPU(duration=SEARCH_GPU_DURATION)
def _search_on_gpu(model, question: str, store, doc_ids, top_k: int):
    results = []
    with torch.no_grad():
        q = model.forward_queries([question], batch_size=1)[0].to(torch.float16)
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

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.model = (
            AutoModel.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True,
                torch_dtype=dtype,
                attn_implementation=COLEMBED_ATTN,
            )
            .to(device)
            .eval()
        )

    def embed_pages(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Embed page images -> list of [n_tokens, dim] float16 arrays."""
        return _embed_pages_on_gpu(self.model, images)

    def search(
        self, question: str, store, doc_ids: list[str] | None, top_k: int
    ) -> list[tuple[str, int, float]]:
        """Return the top_k (doc_id, page_num, score) across the given docs."""
        return _search_on_gpu(self.model, question, store, doc_ids, top_k)
