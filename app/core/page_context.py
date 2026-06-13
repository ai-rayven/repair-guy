"""Whole-page text for the agent's context (parsed approach).

The agent (models/minicpm_agent) is a text model, so "the page the mechanic is
viewing" is given to it as text: a page's parsed elements rendered in reading
order, with Picture/Table elements replaced by their MiniCPM descriptions (the
same descriptions the chunker splices inline, but here we keep the WHOLE page,
not just the chunks that matched a query). Source is the parsed store's
parsed.json — [{"page": n, "elements": [{"class", "bbox", "text",
"description"?}]}].

Pure functions, no model/GPU dependency.
"""

from __future__ import annotations

from core.chunking import FIGURE_CLASS, SKIP_CLASSES, TABLE_CLASS


def index_pages(parsed_pages: list[dict]) -> dict[int, list[dict]]:
    """{page_number: elements} so a turn reads parsed.json once and looks up any
    page it shows without re-scanning."""
    return {pg["page"]: pg.get("elements", []) for pg in parsed_pages}


def page_to_text(elements: list[dict]) -> str:
    """A page's elements as one text block in reading order. Figures become
    "[Figure: <description>]" and tables "[Table: <description>] <cells>", so the
    agent reads what's on the page (and can name a figure/table to circle)
    without seeing the image. Headers/footers are dropped."""
    lines: list[str] = []
    for el in elements:
        cls = el.get("class", "")
        if cls in SKIP_CLASSES:
            continue
        text = (el.get("text") or "").strip()
        if cls == FIGURE_CLASS:
            desc = (el.get("description") or "").strip()
            if desc:
                lines.append(f"[Figure: {desc}]")
        elif cls == TABLE_CLASS:
            desc = (el.get("description") or "").strip()
            parts = []
            if desc:
                parts.append(f"[Table: {desc}]")
            if text:
                parts.append(text)
            if parts:
                lines.append("\n".join(parts))
        elif text:
            lines.append(text)
    return "\n".join(lines).strip()
