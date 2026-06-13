"""Section index + fuzzy matching for hands-busy navigation.

The mechanic-facing flow is "go to <section>" / "circle the <thing>": no model,
no GPU — just the section structure Nemotron Parse already extracted at ingest
(headings on the section chunks, bboxes on the figure/table chunks) matched
against the request with plain string similarity. app.py exposes these through
the CPU-only /sections, /navigate and /locate routes.

All scoring is 0..1; the frontend applies the acceptance thresholds (an
explicit "go to X" tolerates a looser match than a bare "brake bleeding").
"""

from __future__ import annotations

import difflib
import re

# Filler that carries no signal about which section is meant.
_STOPWORDS = {
    "a", "an", "and", "for", "in", "of", "on", "or", "the", "to",
    "section", "chapter", "page", "pages", "part",
}


def _tokens(text: str) -> list[str]:
    words = re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()
    return [w for w in words if w not in _STOPWORDS]


def _close(word: str, other: str) -> bool:
    """Tolerate inflection/typos: bleed ~ bleeding, breaks ~ brakes."""
    return difflib.SequenceMatcher(None, word, other).ratio() >= 0.8


def score(query: str, text: str) -> float:
    """How well a request matches a section title / element description:
    mostly "is every query word in there" (so a short request matches a long
    breadcrumb title), blended with whole-string similarity as a tiebreak."""
    q, t = _tokens(query), _tokens(text)
    if not q or not t:
        return 0.0
    contained = sum(
        1 for w in q if any(w == tw or _close(w, tw) for tw in t)
    ) / len(q)
    fuzzy = difflib.SequenceMatcher(None, " ".join(q), " ".join(t)).ratio()
    return 0.75 * contained + 0.25 * fuzzy


def sections_from_chunks(chunks: list[dict]) -> list[dict]:
    """[{title, page_start, page_end}] in document order, derived from the
    parsed store's section chunks (oversized sections are split into several
    chunks with the same heading — merged back here)."""
    sections: list[dict] = []
    for c in chunks:
        if c.get("type") != "section" or not c.get("heading") or not c.get("pages"):
            continue
        first, last = min(c["pages"]), max(c["pages"])
        if sections and sections[-1]["title"] == c["heading"]:
            sections[-1]["page_end"] = max(sections[-1]["page_end"], last)
        else:
            sections.append(
                {"title": c["heading"], "page_start": first, "page_end": last}
            )
    return sections


def match_section(query: str, sections: list[dict]) -> dict | None:
    """The best-matching section as {title, page, score}, or None."""
    best, best_score = None, 0.0
    for s in sections:
        sc = score(query, s["title"])
        if sc > best_score:
            best, best_score = s, sc
    if best is None:
        return None
    return {"title": best["title"], "page": best["page_start"], "score": round(best_score, 3)}


def match_element(query: str, elements: list[dict]) -> dict | None:
    """The best-matching figure/table as {page, bbox, kind, label, score}, or
    None. elements are parsed-store chunks of type figure/table: their text is
    the MiniCPM search description, their bbox is in rendered-page pixels (the
    same space the /page route serves)."""
    best, best_score = None, 0.0
    for el in elements:
        if el.get("type") not in ("figure", "table") or not el.get("bbox"):
            continue
        sc = score(query, f"{el.get('heading') or ''} {el.get('text') or ''}")
        if sc > best_score:
            best, best_score = el, sc
    if best is None:
        return None
    label = " ".join((best.get("text") or best.get("heading") or "").split())
    return {
        "page": best["page"],
        "bbox": [round(v) for v in best["bbox"]],
        "kind": best["type"],
        "label": label[:140],
        "score": round(best_score, 3),
    }
