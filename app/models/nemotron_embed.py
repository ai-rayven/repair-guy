"""Llama Nemotron Embed VL 1B v2: dense chunk/query embeddings (parsed approach).

Bi-encoder, 2048-dim, mean-pooled. We embed text only in v1 (section chunks
and MiniCPM figure/table descriptions); the model can also embed page/figure
images into the same space — a future upgrade path.

Its remote code supports transformers 4.56+, so it shares an environment with
MiniCPM (and the Space). The model is a module-level global for the same
ZeroGPU reason as the others: module-level CUDA tensors are shared with the
GPU worker, function arguments are pickled and trust_remote_code classes are
not picklable.

embed_texts runs at ingest (Modal); embed_query runs inside the ask pipeline's
single @spaces.GPU call. Outputs are L2-normalized, so cosine == dot product.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel

from core.constants import (
    EMBED_TEXT_BATCH_SIZE,
    EMBED_TEXT_MAX_LENGTH,
    NEMOTRON_EMBED_MODEL_ID,
    NEMOTRON_EMBED_REVISION,
)

_MODEL = (
    AutoModel.from_pretrained(
        NEMOTRON_EMBED_MODEL_ID,
        revision=NEMOTRON_EMBED_REVISION,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    .to("cuda")
    .eval()
)
# Token budget for text-only inputs (the model card's recommended setting).
_MODEL.processor.p_max_length = EMBED_TEXT_MAX_LENGTH


def _normalize(emb: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(emb.float(), p=2, dim=-1)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed chunk texts -> [n, dim] float16, L2-normalized. Must run on GPU."""
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), EMBED_TEXT_BATCH_SIZE):
            emb = _MODEL.encode_documents(texts=texts[i : i + EMBED_TEXT_BATCH_SIZE])
            out.append(_normalize(emb).to(torch.float16).cpu().numpy())
    return np.concatenate(out, axis=0)


def embed_query(question: str) -> np.ndarray:
    """Embed one query -> [dim] float32, L2-normalized. Must run on GPU
    (called from within a @spaces.GPU context on the Space)."""
    with torch.inference_mode():
        emb = _MODEL.encode_queries([question])
    return _normalize(emb)[0].cpu().numpy()


class NemotronEmbed:
    MODEL_ID = NEMOTRON_EMBED_MODEL_ID

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return embed_texts(texts)

    def embed_query(self, question: str) -> np.ndarray:
        return embed_query(question)
