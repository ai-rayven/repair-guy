"""Client for a MiniCPM-V OpenAI-compatible vision endpoint: answers a question
grounded in the retrieved repair-manual pages."""

from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request

from PIL import Image

BASE_URL = os.environ.get("MINICPM_BASE_URL", "http://35.203.155.71:8003").rstrip("/")
MODEL = os.environ.get("MINICPM_MODEL", "MiniCPM-V-4.6")
API_KEY = os.environ.get("MINICPM_API_KEY", "")
MAX_EDGE = 1024  # downscale images; endpoint max_model_len is only 8192 and load-sensitive
RETRIES = 3

PROMPT = (
    "You are a repair-manual assistant. The images are the manual pages most "
    "relevant to the user's question, each preceded by its label (manual name "
    "and page number).\n\n"
    "Answer the question using ONLY these pages. Quote exact values (torques, "
    "clearances, part numbers, capacities) as printed, and cite the page label "
    "for each fact. If the pages do not contain the answer, say so instead of "
    "guessing.\n\nQuestion: {question}"
)


class MiniCPM:
    def __init__(
        self,
        base_url: str = BASE_URL,
        model: str = MODEL,
        api_key: str = API_KEY,
        max_edge: int = MAX_EDGE,
        retries: int = RETRIES,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_edge = max_edge
        self.retries = retries

    def answer(self, question: str, pages: list[tuple[str, Image.Image]]) -> str:
        """pages: [(label, page image)] in retrieval order."""
        if not self.api_key:
            raise ValueError(
                "MINICPM_API_KEY is not set — add it as a secret on the Space."
            )
        content = [{"type": "text", "text": PROMPT.format(question=question)}]
        for label, img in pages:
            content.append({"type": "text", "text": f"\n[{label}]"})
            content.append(
                {"type": "image_url", "image_url": {"url": self._data_uri(img)}}
            )
        return self._chat(content, max_tokens=900).strip()

    def _data_uri(self, img: Image.Image) -> str:
        im = img.convert("RGB")
        w, h = im.size
        if max(w, h) > self.max_edge:
            s = self.max_edge / max(w, h)
            im = im.resize((int(w * s), int(h * s)))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def _chat(self, content: list[dict], max_tokens: int = 512) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": content}],
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        last = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.loads(r.read())["choices"][0]["message"]["content"]
            except (urllib.error.URLError, TimeoutError, OSError) as ex:
                last = ex
                time.sleep(2 * (attempt + 1))  # 2s, 4s backoff between retries
        raise last
