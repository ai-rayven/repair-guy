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
