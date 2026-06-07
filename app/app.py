"""Gradio + ZeroGPU Space for NVIDIA Nemotron Parse v1.2.

Upload a PDF, pick a page, and get back the parsed markdown, a structured JSON of
elements, and the page image annotated with bounding boxes.

Runs on ZeroGPU: the model is loaded onto cuda at module level (ZeroGPU emulates
CUDA at startup) and inference runs inside an @spaces.GPU-decorated function.

This file targets the Space (cuda/bfloat16). For local CPU testing use
parse_page.py in the repo root instead.
"""

import json
import sys

import fitz  # pymupdf
import gradio as gr
import spaces
import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor, GenerationConfig

MODEL_ID = "nvidia/NVIDIA-Nemotron-Parse-v1.2"
DEVICE = "cuda"
DTYPE = torch.bfloat16
MAX_PROMPT_DURATION = 120  # seconds of GPU time per page

# ---------------------------------------------------------------------------
# Load helpers + model once at module level (ZeroGPU loads cuda weights here).
# ---------------------------------------------------------------------------


def load_postprocessing():
    """Download the repo's .py helpers and import postprocessing.

    postprocessing.py imports sibling modules (latex2html, ...), so we pull all
    top-level .py files into one dir and put it on sys.path before importing.
    """
    repo_dir = snapshot_download(repo_id=MODEL_ID, allow_patterns=["*.py"])
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    import postprocessing  # noqa: E402  (resolved via sys.path above)

    return postprocessing


pp = load_postprocessing()

# Every load passes trust_remote_code=True so the nested C-RADIO encoder code is
# accepted non-interactively (no [y/N] prompt to hang the Space build).
model = (
    AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=DTYPE)
    .to(DEVICE)
    .eval()
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
generation_config = GenerationConfig.from_pretrained(MODEL_ID, trust_remote_code=True)


@spaces.GPU(duration=MAX_PROMPT_DURATION)
def run_model(image: Image.Image, task_prompt: str) -> str:
    """GPU-only step: preprocess + generate + decode. Returns raw model text."""
    inputs = processor(
        images=[image], text=task_prompt, return_tensors="pt", add_special_tokens=False
    )
    # Move to GPU; cast float tensors (pixel_values) to the model dtype.
    inputs = {
        k: (v.to(DEVICE, DTYPE) if torch.is_floating_point(v) else v.to(DEVICE))
        for k, v in inputs.items()
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]


# ---------------------------------------------------------------------------
# CPU-side orchestration: render page, call GPU, postprocess, annotate.
# ---------------------------------------------------------------------------


def render_page(pdf_path: str, page_num: int, dpi: int) -> Image.Image:
    doc = fitz.open(pdf_path)
    try:
        if page_num < 1 or page_num > doc.page_count:
            raise gr.Error(
                f"Page {page_num} out of range — this PDF has {doc.page_count} pages."
            )
        pix = doc.load_page(page_num - 1).get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()


def load_input(file_path: str, page_num: int, dpi: int) -> Image.Image:
    """Return an RGB image from either a PDF page or an image file."""
    if file_path.lower().endswith(".pdf"):
        return render_page(file_path, page_num, dpi)
    return Image.open(file_path).convert("RGB")


def parse(input_file, page_num, dpi, text_in_pic, table_format):
    if input_file is None:
        raise gr.Error("Please upload a PDF or image first.")

    image = load_input(input_file, int(page_num), int(dpi))

    fourth = "<predict_text_in_pic>" if text_in_pic else "<predict_no_text_in_pic>"
    task_prompt = f"</s><s><predict_bbox><predict_classes><output_markdown>{fourth}"

    generated_text = run_model(image, task_prompt)

    classes, bboxes, texts = pp.extract_classes_bboxes(generated_text)
    bboxes = [pp.transform_bbox_to_original(b, image.width, image.height) for b in bboxes]
    texts = [
        pp.postprocess_text(t, cls=c, table_format=table_format, text_format="markdown")
        for t, c in zip(texts, classes)
    ]

    markdown = "\n\n".join(texts)
    elements = [
        {"class": c, "bbox": b, "text": t} for c, b, t in zip(classes, bboxes, texts)
    ]

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for b in bboxes:
        draw.rectangle((b[0], b[1], b[2], b[3]), outline="red", width=2)

    return annotated, markdown, json.dumps(elements, indent=2)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Nemotron Parse — Repair Manuals") as demo:
    gr.Markdown(
        "# 🔧 Nemotron Parse v1.2 — Repair Manual Explorer\n"
        "Upload a PDF (choose a page) or an image, and parse it with "
        "[NVIDIA Nemotron Parse v1.2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2) "
        "on ZeroGPU. Returns structured markdown, a JSON of elements, and an "
        "annotated page image."
    )
    with gr.Row():
        with gr.Column(scale=1):
            pdf_in = gr.File(
                label="PDF or image",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp"],
                type="filepath",
            )
            page_in = gr.Number(
                label="Page (PDF only)", value=1, precision=0, minimum=1
            )
            dpi_in = gr.Slider(
                label="Render DPI (PDF only)", minimum=72, maximum=300, value=150, step=10
            )
            text_in_pic_in = gr.Checkbox(
                label="Extract text inside pictures/diagrams", value=False
            )
            table_format_in = gr.Dropdown(
                label="Table format",
                choices=["markdown", "latex", "HTML", "json", "csv"],
                value="markdown",
            )
            run_btn = gr.Button("Parse page", variant="primary")
        with gr.Column(scale=2):
            img_out = gr.Image(label="Annotated page", type="pil")
            with gr.Tab("Rendered markdown"):
                md_out = gr.Markdown()
            with gr.Tab("Structured JSON"):
                json_out = gr.Code(language="json")

    run_btn.click(
        parse,
        inputs=[pdf_in, page_in, dpi_in, text_in_pic_in, table_format_in],
        outputs=[img_out, md_out, json_out],
    )


if __name__ == "__main__":
    demo.launch()
