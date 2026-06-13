"""Section index + fuzzy matching for the find-and-point router.

Two uses, both CPU-only and cheap:
- sections_from_chunks: turn the parsed store's section chunks into a clean
  [{title, page_start, page_end}] index.
- top_sections: the per-request shortlist of fine-grained headings shown to
  the LLM router (the full index is far too large for a prompt), so a precise
  heading like "Brake System Bleeding — p.532" is on offer alongside the
  manual's clean PDF-bookmark chapters.

All scoring is 0..1. The agent (models/minicpm_agent) makes the final call from
this table of contents plus the current page's text; nothing here navigates on
its own.
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


def top_sections(
    query: str, sections: list[dict], n: int = 8, floor: float = 0.3
) -> list[dict]:
    """The n best-matching sections as [{title, page}], best first, dropping
    anything below floor — the per-request shortlist of fine-grained headings
    shown to the router so a precise heading ("Brake System Bleeding") is on
    offer even when the full index is far too large to put in the prompt."""
    scored = [(score(query, s["title"]), s) for s in sections]
    scored = [(sc, s) for sc, s in scored if sc >= floor]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"title": s["title"], "page": s["page_start"]} for _, s in scored[:n]]
