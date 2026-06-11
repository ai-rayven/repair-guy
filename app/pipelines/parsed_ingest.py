"""Parsed ingest pipeline: PDF -> Nemotron Parse -> MiniCPM descriptions ->
section chunks -> Nemotron Embed -> store.

Two stages that CANNOT share a process: Nemotron Parse requires
transformers==5.6.1 while MiniCPM/Nemotron Embed need <5. scripts/index_modal.py
runs each stage in its own Modal container (separate images); the parse
output (plain dicts) is what crosses the boundary, and it is also persisted
as parsed.json in the store so later re-chunking needs no re-parse.

Model imports happen lazily inside each stage's __init__ — importing this
module must not pull in either transformers world.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from core.chunking import (
    CAPTION_CLASS,
    FIGURE_CLASS,
    HEADING_CLASSES,
    SKIP_CLASSES,
    TABLE_CLASS,
    build_chunks,
)
from core.constants import (
    DESCRIBE_CONTEXT_MAX_CHARS,
    FIGURE_MIN_SIDE_PX,
    RENDER_DPI,
)
from core.parsed_store import ParsedStore
from core.pdf import page_count, render_page
from core.store import slugify


class ParseStage:
    """Stage 1 (transformers 5.x env): render pages and parse them into
    classified elements."""

    def __init__(self):
        from models.nemotron_parse import NemotronParse

        self.parser = NemotronParse()

    def run(self, pdf_path: str) -> Iterator[tuple]:
        """Generator yielding ("progress", page, total) per parsed page, then
        ("done", pages) where pages = [{"page", "elements"}] in order."""
        total = page_count(pdf_path)
        pages = []
        for num in range(1, total + 1):
            image = render_page(pdf_path, num, RENDER_DPI)
            pages.append({"page": num, "elements": self.parser.parse_page(image)})
            yield ("progress", num, total)
        yield ("done", pages)


class BuildStage:
    """Stage 2 (transformers 4.x env, same as the Space): describe figures and
    tables with MiniCPM, build section chunks, embed them, save to the store."""

    def __init__(self, store: ParsedStore):
        from models import minicpm
        from models.nemotron_embed import NemotronEmbed

        self.describer = minicpm
        self.embedder = NemotronEmbed()
        self.store = store

    @staticmethod
    def _element_context(doc_name: str, heading: str, elements: list[dict], i: int) -> str:
        """Document context for describing element i: the manual's name, the
        section heading in force, the adjacent Caption (Parse keeps captions
        next to their figure/table in reading order), and the page's other
        text. This is what lets MiniCPM use the manual's own terminology
        instead of guessing from pixels."""
        lines = [f"Manual: {doc_name}"]
        if heading:
            lines.append(f"Section: {heading}")
        for j in (i + 1, i - 1):
            if 0 <= j < len(elements) and elements[j]["class"] == CAPTION_CLASS:
                caption = (elements[j].get("text") or "").strip()
                if caption:
                    lines.append(f"Caption: {caption}")
                break
        page_text = " ".join(
            t
            for el in elements
            if el["class"] not in SKIP_CLASSES | {FIGURE_CLASS, TABLE_CLASS}
            for t in [(el.get("text") or "").strip()]
            if t
        )
        if page_text:
            lines.append(f"Text on the page: {page_text[:DESCRIBE_CONTEXT_MAX_CHARS]}")
        return "\n".join(lines)

    def _describe_page(self, pdf_path: str, pg: dict, doc_name: str, heading: str) -> str:
        """Attach MiniCPM descriptions to this page's Picture/Table elements
        (in place). Tracks and returns the section heading in force so the
        next page's context carries it (sections span pages)."""
        image = None
        elements = pg["elements"]
        prev_was_heading = False
        for i, el in enumerate(elements):
            if el["class"] in HEADING_CLASSES:
                text = (el.get("text") or "").strip()
                if text:
                    # consecutive headings combine into a breadcrumb, like the chunker
                    heading = f"{heading} — {text}" if prev_was_heading and heading else text
                    prev_was_heading = True
                continue
            prev_was_heading = False
            if el["class"] == FIGURE_CLASS:
                x1, y1, x2, y2 = el.get("bbox") or (0, 0, 0, 0)
                if x2 - x1 < FIGURE_MIN_SIDE_PX or y2 - y1 < FIGURE_MIN_SIDE_PX:
                    continue  # icons, bullets, print artifacts
                if image is None:
                    image = render_page(pdf_path, pg["page"], RENDER_DPI)
                crop = image.crop(
                    (max(x1, 0), max(y1, 0), min(x2, image.width), min(y2, image.height))
                )
                el["description"] = self.describer.describe_figure(
                    crop, self._element_context(doc_name, heading, elements, i)
                )
            elif el["class"] == TABLE_CLASS and el.get("text", "").strip():
                el["description"] = self.describer.describe_table(
                    el["text"], self._element_context(doc_name, heading, elements, i)
                )
        return heading

    def run(self, pdf_path: str, parsed_pages: list[dict], doc_name: str = "") -> Iterator[tuple]:
        """Generator yielding ("progress", pages_done, total) through the
        describe phase, then ("done", doc summary dict) after embed + save."""
        name = doc_name.strip() or (
            os.path.splitext(os.path.basename(pdf_path))[0].replace("_", " ")
        )
        doc_id = slugify(name)
        total = len(parsed_pages)

        heading = ""
        for done, pg in enumerate(parsed_pages, start=1):
            heading = self._describe_page(pdf_path, pg, name, heading)
            yield ("progress", done, total)

        chunks = build_chunks(parsed_pages)
        if not chunks:
            raise ValueError("Parsing produced no text — nothing to index.")
        embeddings = self.embedder.embed_texts([c["text"] for c in chunks])
        self.store.save(
            doc_id,
            name,
            pdf_path,
            chunks,
            embeddings,
            page_count=total,
            dpi=RENDER_DPI,
            model_id=self.embedder.MODEL_ID,
            parsed_pages=parsed_pages,
        )
        yield (
            "done",
            {"doc_id": doc_id, "name": name, "pages": total, "chunks": len(chunks)},
        )
