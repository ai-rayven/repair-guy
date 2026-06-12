"""Nemotron Parse v1.2: page image -> classified elements with bboxes (parsed approach).

IMPORTANT: this model requires transformers==5.6.1, which is incompatible with
ColEmbed and MiniCPM (transformers<5). It therefore runs only in the dedicated
parse stage of scripts/index_modal.py (its own Modal image) and must never be
imported on the Space or alongside the other models.

The model emits a tagged sequence; the repo's own postprocessing helpers turn
it into (classes, bboxes, texts) with bboxes mapped back to the input image's
pixel coordinates and tables rendered as markdown.
"""

from __future__ import annotations

import sys

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModel, AutoProcessor, GenerationConfig

from core.constants import NEMOTRON_PARSE_MODEL_ID, NEMOTRON_PARSE_REVISION

# bbox + class + markdown per element; figure contents are left to MiniCPM
# descriptions at the next stage, so no text-in-picture prediction.
_TASK_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"


def _load_postprocessing():
    """Import the model repo's postprocessing module (it has sibling .py
    imports, so the whole top level goes on sys.path)."""
    repo_dir = snapshot_download(
        repo_id=NEMOTRON_PARSE_MODEL_ID,
        revision=NEMOTRON_PARSE_REVISION,
        allow_patterns=["*.py"],
    )
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    import postprocessing

    return postprocessing


class NemotronParse:
    MODEL_ID = NEMOTRON_PARSE_MODEL_ID

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._pp = _load_postprocessing()
        self._model = (
            AutoModel.from_pretrained(
                NEMOTRON_PARSE_MODEL_ID,
                revision=NEMOTRON_PARSE_REVISION,
                trust_remote_code=True,
                dtype=torch.bfloat16,
            )
            .to(device)
            .eval()
        )
        self._processor = AutoProcessor.from_pretrained(
            NEMOTRON_PARSE_MODEL_ID,
            revision=NEMOTRON_PARSE_REVISION,
            trust_remote_code=True,
        )
        self._gen_config = GenerationConfig.from_pretrained(
            NEMOTRON_PARSE_MODEL_ID,
            revision=NEMOTRON_PARSE_REVISION,
            trust_remote_code=True,
        )

    def parse_pages(self, images: list[Image.Image]) -> list[list[dict]]:
        """Parse a batch of rendered pages in ONE generate call (every page
        uses the same task prompt, so the batch needs no padding). Returns one
        element list per page, in input order."""
        inputs = self._processor(
            images=list(images),
            text=[_TASK_PROMPT] * len(images),
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = {
            k: (
                v.to(self.device, torch.bfloat16)
                if torch.is_floating_point(v)
                else v.to(self.device)
            )
            for k, v in inputs.items()
        }
        with torch.no_grad():
            outputs = self._model.generate(**inputs, generation_config=self._gen_config)
        raws = self._processor.batch_decode(outputs, skip_special_tokens=True)
        return [self._elements(raw, image) for raw, image in zip(raws, images)]

    def parse_page(self, image: Image.Image) -> list[dict]:
        """Parse one rendered page -> [{"class", "bbox", "text"}] in reading
        order. bbox is [x1, y1, x2, y2] in the image's pixel coordinates."""
        return self.parse_pages([image])[0]

    def _elements(self, raw: str, image: Image.Image) -> list[dict]:
        classes, bboxes, texts = self._pp.extract_classes_bboxes(raw)
        bboxes = [
            self._pp.transform_bbox_to_original(b, image.width, image.height)
            for b in bboxes
        ]
        texts = [
            self._pp.postprocess_text(
                t, cls=c, table_format="markdown", text_format="markdown"
            )
            for t, c in zip(texts, classes)
        ]
        return [
            {"class": c, "bbox": [int(v) for v in b], "text": (t or "").strip()}
            for c, b, t in zip(classes, bboxes, texts)
        ]
