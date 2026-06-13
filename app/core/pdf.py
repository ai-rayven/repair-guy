import fitz
from PIL import Image

from core.constants import RENDER_DPI


def page_count(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()


def render_pages(
    pdf_path: str, page_nums: list[int], dpi: int = RENDER_DPI
) -> list[Image.Image]:
    """Render 1-based pages of a PDF to RGB images (document opened once)."""
    doc = fitz.open(pdf_path)
    try:
        images = []
        for num in page_nums:
            if num < 1 or num > doc.page_count:
                raise ValueError(
                    f"Page {num} out of range — this PDF has {doc.page_count} pages."
                )
            pix = doc.load_page(num - 1).get_pixmap(dpi=dpi)
            images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        return images
    finally:
        doc.close()


def render_page(pdf_path: str, page_num: int, dpi: int = RENDER_DPI) -> Image.Image:
    return render_pages(pdf_path, [page_num], dpi)[0]


def render_page_png(pdf_path: str, page_num: int, dpi: int = RENDER_DPI) -> bytes:
    """Render a 1-based page straight to PNG bytes (for the /page route).
    Same DPI as the model/parse renders, so parse bboxes map 1:1 onto it."""
    doc = fitz.open(pdf_path)
    try:
        if page_num < 1 or page_num > doc.page_count:
            raise ValueError(
                f"Page {page_num} out of range — this PDF has {doc.page_count} pages."
            )
        return doc.load_page(page_num - 1).get_pixmap(dpi=dpi).tobytes("png")
    finally:
        doc.close()


def pdf_outline(pdf_path: str) -> list[dict]:
    """The PDF's bookmark outline as [{title, page_start, page_end}] — the
    manual's own clean chapter structure (e.g. 16 entries for a 1151-page
    Hyundai manual), unlike the noisy parse-derived headings. Empty when the
    PDF has no bookmarks."""
    doc = fitz.open(pdf_path)
    try:
        entries = [
            {"title": " ".join(title.split()), "page_start": max(1, page)}
            for _, title, page in doc.get_toc()
            if title.strip()
        ]
        for i, e in enumerate(entries):
            e["page_end"] = (
                max(e["page_start"], entries[i + 1]["page_start"] - 1)
                if i + 1 < len(entries)
                else doc.page_count
            )
        return entries
    finally:
        doc.close()


def page_size(pdf_path: str, page_num: int, dpi: int = RENDER_DPI) -> tuple[int, int]:
    """(width, height) in pixels a 1-based page renders to at this DPI,
    without rendering it."""
    doc = fitz.open(pdf_path)
    try:
        rect = doc.load_page(page_num - 1).rect
        scale = dpi / 72
        return round(rect.width * scale), round(rect.height * scale)
    finally:
        doc.close()
