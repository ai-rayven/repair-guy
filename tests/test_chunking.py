from core.chunking import build_chunks


def el(cls, text, bbox=None, description=None):
    e = {"class": cls, "text": text, "bbox": bbox or [0, 0, 100, 100]}
    if description is not None:
        e["description"] = description
    return e


def sections(chunks):
    return [c for c in chunks if c["type"] == "section"]


def test_headings_split_sections():
    pages = [
        {
            "page": 1,
            "elements": [
                el("Section-header", "REMOVAL"),
                el("Text", "a" * 300),
                el("Section-header", "INSTALLATION"),
                el("Text", "b" * 300),
            ],
        }
    ]
    secs = sections(build_chunks(pages, min_chars=200, max_chars=6000))
    assert [s["heading"] for s in secs] == ["REMOVAL", "INSTALLATION"]
    assert secs[0]["text"].startswith("REMOVAL\n\n")
    assert "b" not in secs[0]["text"]


def test_sparse_section_merges_forward():
    pages = [
        {
            "page": 1,
            "elements": [
                el("Section-header", "NOTES"),
                el("Text", "tiny"),  # under min_chars -> no standalone chunk
                el("Section-header", "REMOVAL"),
                el("Text", "c" * 300),
            ],
        }
    ]
    secs = sections(build_chunks(pages, min_chars=200, max_chars=6000))
    assert len(secs) == 1
    # the tiny section's heading and text fold into the surviving chunk
    assert "REMOVAL" in secs[0]["text"]
    assert "tiny" in secs[0]["text"]
    assert secs[0]["heading"] == "NOTES"


def test_consecutive_headings_combine_into_breadcrumb():
    pages = [
        {
            "page": 1,
            "elements": [
                el("Title", "REAR AXLE"),
                el("Section-header", "PROPELLER SHAFT"),
                el("Text", "d" * 300),
            ],
        }
    ]
    secs = sections(build_chunks(pages, min_chars=200, max_chars=6000))
    assert secs[0]["heading"] == "REAR AXLE — PROPELLER SHAFT"


def test_sections_span_pages_and_record_them():
    pages = [
        {"page": 3, "elements": [el("Section-header", "REMOVAL"), el("Text", "x" * 300)]},
        {"page": 4, "elements": [el("Text", "y" * 300)]},
    ]
    secs = sections(build_chunks(pages, min_chars=200, max_chars=6000))
    assert len(secs) == 1
    assert secs[0]["pages"] == [3, 4]


def test_oversized_section_splits_with_heading_repeated():
    pages = [
        {
            "page": 1,
            "elements": [
                el("Section-header", "SPECS"),
                el("Text", "e" * 500),
                el("Text", "f" * 500),
            ],
        }
    ]
    secs = sections(build_chunks(pages, min_chars=100, max_chars=600))
    assert len(secs) == 2
    assert all(s["text"].startswith("SPECS\n\n") for s in secs)


def test_figure_and_table_chunks():
    pages = [
        {
            "page": 7,
            "elements": [
                el("Section-header", "UNIVERSAL JOINT"),
                el("Text", "g" * 300),
                el("Picture", "", bbox=[10, 10, 200, 200], description="Exploded view of the universal joint."),
                el("Table", "| bolt | torque |\n| --- | --- |\n| M8 | 25 Nm |", description="Tightening torques."),
            ],
        }
    ]
    chunks = build_chunks(pages, min_chars=200, max_chars=6000)
    fig = next(c for c in chunks if c["type"] == "figure")
    tbl = next(c for c in chunks if c["type"] == "table")
    sec = sections(chunks)[0]
    assert fig["page"] == 7 and "Exploded view" in fig["text"]
    assert fig["text"].startswith("UNIVERSAL JOINT")  # heading context prepended
    assert "25 Nm" in tbl["text"] and "Tightening torques" in tbl["text"]
    # descriptions spliced inline into the section text
    assert "[Figure: Exploded view of the universal joint.]" in sec["text"]
    assert "[Table: Tightening torques.]" in sec["text"]


def test_undescribed_figure_is_dropped_and_headers_footers_skipped():
    pages = [
        {
            "page": 1,
            "elements": [
                el("Page-header", "TOYOTA 8FGU MANUAL"),
                el("Text", "h" * 300),
                el("Picture", ""),  # too small to describe -> no description
                el("Page-footer", "1-2"),
            ],
        }
    ]
    chunks = build_chunks(pages, min_chars=200, max_chars=6000)
    assert all(c["type"] == "section" for c in chunks)
    assert "TOYOTA 8FGU MANUAL" not in chunks[0]["text"]


def test_text_before_any_heading_still_chunks():
    pages = [{"page": 1, "elements": [el("Text", "intro " * 100)]}]
    secs = sections(build_chunks(pages, min_chars=200, max_chars=6000))
    assert len(secs) == 1
    assert secs[0]["heading"] == ""
