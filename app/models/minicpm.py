"""MiniCPM-V: answers a question grounded in the retrieved repair-manual pages.

The model and tokenizer are module-level globals: ZeroGPU packs module-level
CUDA tensors at startup and shares them with the GPU worker, whereas function
arguments are pickled — and trust_remote_code model classes are not picklable.

generate_answer is a plain function: the ask pipeline calls it inside its own
single @spaces.GPU call, right after retrieval.
"""

from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from core.constants import ANSWER_MAX_NEW_TOKENS, MINICPM_MODEL_ID, MINICPM_REVISION

PROMPT = (
    "You are a repair-manual assistant. The images are the manual pages most "
    "relevant to the user's question, each preceded by its label (manual name "
    "and page number).\n\n"
    "Answer the question using ONLY what is printed on these pages, following "
    "these rules:\n"
    "1. If the answer is a procedure, reproduce EVERY step in order, numbered "
    "exactly as in the manual. Never skip, merge, or summarize steps. Keep each "
    "step's notes, model exceptions, and specifications (e.g. torque values) "
    "with that step, exactly as printed.\n"
    "2. Quote exact values (torques, clearances, part numbers, capacities) as "
    "printed, including units.\n"
    "3. End with the page label(s) you used, e.g. (Manual — p.238). If a "
    "procedure clearly continues on a page you were not given, say so.\n"
    "4. If the pages do not contain the answer, say so instead of guessing.\n"
    "5. Start directly with the answer — no preamble like 'Based on the "
    "provided pages'.\n\n"
    "Question: {question}"
)

_MODEL = (
    AutoModel.from_pretrained(
        MINICPM_MODEL_ID,
        revision=MINICPM_REVISION,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    .to("cuda")
    .eval()
)
_TOKENIZER = AutoTokenizer.from_pretrained(
    MINICPM_MODEL_ID, revision=MINICPM_REVISION, trust_remote_code=True
)
# Pre-build the processor chat() would otherwise lazily create per GPU worker
# (it caches on this exact attribute, see modeling_minicpmv.chat).
_MODEL.processor = AutoProcessor.from_pretrained(
    MINICPM_MODEL_ID, revision=MINICPM_REVISION, trust_remote_code=True
)


def generate_answer(question: str, pages: list[tuple[str, Image.Image]]) -> str:
    """pages: [(label, page image)] in retrieval order. Must run on GPU
    (called from within a @spaces.GPU context)."""
    content = []
    for label, img in pages:  # chat() accepts interleaved strings and PIL images
        content.append(f"[{label}]")
        content.append(img.convert("RGB"))
    content.append(PROMPT.format(question=question))
    with torch.no_grad():
        answer = _MODEL.chat(
            msgs=[{"role": "user", "content": content}],
            tokenizer=_TOKENIZER,
            enable_thinking=False,
            max_new_tokens=ANSWER_MAX_NEW_TOKENS,
        )
    return str(answer).strip()
