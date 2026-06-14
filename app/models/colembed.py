"""Nemotron ColEmbed v2: late-interaction page embeddings + MaxSim retrieval.

The model is a module-level global, but — unlike the other models — it is NOT
built at import. It lazy-loads on first use (_load) so ZeroGPU does NOT pack it
at startup, freeing ~8 GiB for the bf16 8B agent brain to stay resident. The
trade: the FIRST visual-search call after a cold worker builds it inside the GPU
window (not packed). The default search index is "parsed" (no ColEmbed) and the
8B brain forbids visual search, so the production default path never loads it.

_embed_pages_on_gpu is a ZeroGPU entry point (used at index time);
maxsim_search is a plain function so the ask pipeline can run it inside its
own single GPU call together with answer generation. Both call _load() first.

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
from transformers import AutoModel, AutoProcessor

from core.constants import (
    COLEMBED_ATTN,
    COLEMBED_MODEL_ID,
    COLEMBED_REVISION,
    EMBED_BATCH_SIZE,
    EMBED_GPU_DURATION,
    SCORE_PAGES_PER_BATCH,
)
from core.vram import log_vram

_MODEL = None


def _load():
    """Build ColEmbed and cache it as the module global, on first use. Deliberately
    NOT called at import: keeping it off the GPU at startup means ZeroGPU does not
    pack it, freeing ~8 GiB so the bf16 8B agent brain fits the resident set. Must
    run on GPU (called from within a @spaces.GPU context). A no-op once loaded."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    model = (
        AutoModel.from_pretrained(
            COLEMBED_MODEL_ID,
            revision=COLEMBED_REVISION,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation=COLEMBED_ATTN,
        )
        .to("cuda")
        .eval()
    )
    # Pre-build the processor the remote code would otherwise lazily create per
    # GPU worker (it caches on this exact attribute, see _get_processor).
    model._processor = AutoProcessor.from_pretrained(
        COLEMBED_MODEL_ID, revision=COLEMBED_REVISION, trust_remote_code=True
    )
    # The remote code's forward_documents hardcodes DataLoader(num_workers=8), but
    # the ZeroGPU worker is a daemonic process and may not spawn children
    # ("daemonic processes are not allowed to have children"). Patch the DataLoader
    # name in the model's own module to force in-process loading. A subclass (not a
    # wrapper function) because the remote code also uses the name in isinstance().
    remote_module = sys.modules[type(model).__module__]

    class _SingleProcessDataLoader(remote_module.DataLoader):
        def __init__(self, *args, **kwargs):
            kwargs["num_workers"] = 0
            super().__init__(*args, **kwargs)

    remote_module.DataLoader = _SingleProcessDataLoader
    _MODEL = model
    log_vram("load-colembed")
    return _MODEL


@spaces.GPU(duration=EMBED_GPU_DURATION)
def _embed_pages_on_gpu(images: list[Image.Image]) -> list[np.ndarray]:
    model = _load()
    with torch.no_grad():
        embs = model.forward_images(images, batch_size=EMBED_BATCH_SIZE)
    out = []
    for emb in embs:  # [tokens, dim]; zero rows are padding
        mask = emb.abs().sum(dim=-1) > 0
        out.append(emb[mask].to(torch.float16).cpu().numpy())
    return out


def maxsim_search(
    question: str, store, doc_ids: list[str] | None, top_k: int
) -> list[tuple[str, int, float]]:
    """Top-K (doc_id, page_num, score) across docs. Must run on GPU (called
    from within a @spaces.GPU context)."""
    results = []
    model = _load()
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

    def embed_pages(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Embed page images -> list of [n_tokens, dim] float16 arrays."""
        return _embed_pages_on_gpu(images)
