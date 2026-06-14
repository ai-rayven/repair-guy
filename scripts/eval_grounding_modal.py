#!/usr/bin/env python3
"""VLM grounding eval: does ground_box() circle the right thing — or abstain?

This grades the layer nothing else does. tools.json grades the 1B's tool CHOICE;
find.json grades RETRIEVAL (which page). This grades the VLM box: given a page
image and a target, does MiniCPM-V land the box on the right component/value, and
does it correctly say NOT FOUND when the target is not a groundable thing on the
page? Only the VLM (MiniCPM-V) loads here.

Two metrics, because a circle can fail two opposite ways:
  - region-hit (positives): the returned box CENTER falls inside the gold region.
    Coarse region, so it catches the "random spot" failure without needing
    pixel-perfect labels. A box that lands outside = wrong_spot; a NOT FOUND on a
    real target = false_not_found.
  - NOT-FOUND behavior (negatives): the page has only the query WORD (a breadcrumb)
    or not the thing at all → the right answer is to abstain. Boxing anyway =
    false_box (the failure the mechanic sees as "it pointed at the heading").

cross_page cases carry a `callout_target`: the part is on this diagram page but its
name->number legend is on ANOTHER page, so the page itself shows only the number.
We ground the NAME (expected to fail — single-page can't resolve it) AND the
resolved number ("callout number 10"), to measure whether handing the VLM the
number — which the 1B could read off the legend page's text — recovers the case.

    modal run scripts/eval_grounding_modal.py
    modal run scripts/eval_grounding_modal.py --cases eval/<doc>-ground.json --thinking
    modal run scripts/eval_grounding_modal.py --prompt v2   # no-think prompt variant
    modal run scripts/eval_grounding_modal.py --limit 3

Writes eval/results/<doc_id>-ground[-think][-<prompt>]-<timestamp>.json.

PROMPT VARIANTS (--prompt): the eval owns these so we can A/B without editing
the app. `baseline` defers to minicpm.GROUND_PROMPT (what ships). `v2` is the
abstention-first rewrite: it makes "is this actually a circleable thing on this
page?" the FIRST decision and names the breadcrumb/heading/nav-path trap as
NOT FOUND, instead of (baseline) listing "a heading" as a valid box target —
the line that makes the model box the 'Fuel System' breadcrumb. Goal: recover
thinking-mode's abstention wins without thinking-mode's latency.
"""
import json
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent))
from index_modal import GPU, hf_cache, image  # noqa: E402

app = modal.App(
    "repair-guy-eval-grounding", image=image.add_local_python_source("index_modal")
)

eval_data = modal.Volume.from_name("repair-guy-eval-data", create_if_missing=True)

# --- prompt variants (the eval owns these; baseline=None defers to the SHIPPED
# minicpm.GROUND_PROMPT). `old` is the pre-2026-06-13 prompt kept for regression
# A/B — it listed "a heading — box that" as a valid target, which made the VLM box
# the 'Fuel System' breadcrumb; the shipped prompt is abstention-first instead. ---
PROMPT_OLD = (
    "The image is one page of a repair manual. A mechanic asked to circle "
    "{query!r} on this page. If {query!r} reads like a question or request rather "
    "than a label, box the specific component or value it refers to. Locate it "
    "precisely:\n"
    "- In an exploded or assembly diagram, parts carry callout numbers/letters "
    "on leader lines, and a legend lists what each number is. Find {query!r} in "
    "the legend to get its number, then follow that number's leader line to the "
    "part in the drawing and box THAT part (not the legend text).\n"
    "- For a torque or specification, box the VALUE with its label/units (e.g. "
    "the number and N·m), not the whole table.\n"
    "- Otherwise it may be a row in a table, a specification value, or a "
    "heading — box that.\n"
    "Box ONLY that one part, as TIGHTLY as possible — just the part itself. Do "
    "NOT box the whole figure, the whole diagram, a group of parts, or the page; "
    "if the part is small, the box must be small. A box wider than about half "
    "the page is almost always wrong.\n"
    "Reply with ONLY the box, as <box>x1 y1 x2 y2</box>: four integers "
    "normalized to 0-1000 (x left→right, y top→bottom) over the whole page, "
    "and nothing else. If it is not on this page, reply exactly: NOT FOUND"
)

# Two-page composite probe: p872 (diagram) stitched LEFT of p873 (legend), so the
# VLM has both the callout numbers AND the name->number legend in one frame. The
# part is on the left page, so a correct box has its center in the left half.
PROMPT_TWOPAGE = (
    "This image shows TWO facing pages of a repair manual, side by side. The LEFT "
    "page is a parts/location DIAGRAM whose components are marked with callout "
    "numbers. The RIGHT page is the LEGEND listing what each number is. A mechanic "
    "wants {query!r} circled on the LEFT diagram.\n"
    "Look up {query!r} in the RIGHT legend to get its callout number, find that "
    "number on the LEFT diagram, follow its leader line to the part, and box THAT "
    "part. Your box MUST be on the LEFT page (its x coordinates well under 500). Do "
    "NOT box the legend text on the right page.\n"
    "Reply with ONLY the box, as <box>x1 y1 x2 y2</box>: four integers normalized "
    "0-1000 over this whole two-page image. If {query!r} cannot be found, reply "
    "exactly: NOT FOUND"
)

PROMPTS = {"baseline": None, "old": PROMPT_OLD}


def _hstack(left, right):
    """Stitch two page images side by side at a common height. Returns
    (composite, left_frac) where left_frac is the left page's share of the total
    width — used to remap a gold region defined over the left page alone into the
    composite's 0-1000 x coordinates."""
    from PIL import Image as PILImage

    h = max(left.height, right.height)
    lw = round(left.width * h / left.height)
    rw = round(right.width * h / right.height)
    canvas = PILImage.new("RGB", (lw + rw, h), "white")
    canvas.paste(left.resize((lw, h)), (0, 0))
    canvas.paste(right.resize((rw, h)), (lw, 0))
    return canvas, lw / (lw + rw)


def _center_in(box_norm, gold) -> bool:
    """box_norm: [x1,y1,x2,y2] in 0-1000. gold: same. True if box center ∈ gold."""
    cx = (box_norm[0] + box_norm[2]) / 2
    cy = (box_norm[1] + box_norm[3]) / 2
    return gold[0] <= cx <= gold[2] and gold[1] <= cy <= gold[3]


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@app.function(
    gpu=GPU,
    timeout=2 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/root/.cache/huggingface": hf_cache, "/eval_data": eval_data},
)
def run_eval(
    doc_id: str, cases: list, thinking: bool, prompt_variant: str, repeat: int = 1
) -> list:
    sys.path.insert(0, "/root/app")
    from core.constants import LIBRARY_DATASET_ID, VISUAL_SUBDIR
    from huggingface_hub import snapshot_download

    root = "/eval_data/preindexed"
    print(f"Syncing {doc_id} (visual store + PDF) from {LIBRARY_DATASET_ID} ...")
    snapshot_download(
        LIBRARY_DATASET_ID,
        repo_type="dataset",
        local_dir=root,
        allow_patterns=[f"{VISUAL_SUBDIR}/{doc_id}/*"],
    )
    eval_data.commit()

    from core.visual_store import VisualStore

    visual = VisualStore(f"{root}/{VISUAL_SUBDIR}")
    if not visual.exists(doc_id):
        raise SystemExit(
            f"{doc_id} is not indexed under {VISUAL_SUBDIR}/ in {LIBRARY_DATASET_ID} "
            f"(the grounding eval renders pages from that store's doc.pdf)."
        )
    pdf_path = visual.pdf_path(doc_id)

    from core.pdf import render_page
    from models import minicpm

    prompt = PROMPTS[prompt_variant]
    print(f"prompt variant: {prompt_variant}  thinking: {thinking}")

    def ground(page_img, target, prompt_override=...):
        """Returns ((box_norm | None, raw), seconds). box_norm in 0-1000 over
        page_img, or None on NOT FOUND. prompt_override defaults to the selected
        variant; pass an explicit string (e.g. the two-page prompt) to override."""
        p = prompt if prompt_override is ... else prompt_override
        t0 = time.time()
        box, raw = minicpm.ground_box(
            page_img, target, enable_thinking=thinking, prompt=p
        )
        secs = round(time.time() - t0, 2)
        if box is None:
            return (None, raw), secs
        w, h = page_img.size
        norm = [
            1000 * box[0] / w, 1000 * box[1] / h,
            1000 * box[2] / w, 1000 * box[3] / h,
        ]
        return (norm, raw), secs

    def score(box_norm, gold):
        """gold is 'NOT_FOUND' or a region. Returns (verdict, correct, iou)."""
        if gold == "NOT_FOUND":
            if box_norm is None:
                return "correct_not_found", True, None
            return "false_box", False, None
        # positive: a region is expected
        if box_norm is None:
            return "false_not_found", False, 0.0
        iou = round(_iou(box_norm, gold), 3)
        if _center_in(box_norm, gold):
            return "hit", True, iou
        return "wrong_spot", False, iou

    results = []
    page_cache = {}
    for rep in range(repeat):
      for n, c in enumerate(cases, start=1):
        img = page_cache.setdefault(c["page"], render_page(pdf_path, c["page"]))
        gold = c["gold"]

        (box_norm, raw), secs = ground(img, c["target"])
        verdict, correct, iou = score(box_norm, gold)
        row = {
            "id": c["id"],
            "rep": rep,
            "category": c["category"],
            "page": c["page"],
            "target": c["target"],
            "gold": gold,
            "box": [round(v) for v in box_norm] if box_norm else None,
            "verdict": verdict,
            "correct": correct,
            "iou": iou,
            "secs": secs,
            "raw": raw[:200],
        }
        # cross_page probes: where exactly does single-page grounding break, and
        # does more context recover it? Each is scored vs the same gold region.
        #   callout — hand the VLM the resolved number ("callout number 10")
        #   leader  — explicit "follow the leader line to the part" phrasing
        #   twopage — stitch the legend page (context_page) beside the diagram so
        #             the VLM has BOTH; box must land on the LEFT (diagram) half
        for probe in ("callout", "leader"):
            tgt = c.get(f"{probe}_target")
            if not tgt:
                continue
            (pbox, praw), psecs = ground(img, tgt)
            pverd, pcorr, piou = score(pbox, gold)
            row[f"{probe}_target"] = tgt
            row[f"{probe}_box"] = [round(v) for v in pbox] if pbox else None
            row[f"{probe}_verdict"] = pverd
            row[f"{probe}_correct"] = pcorr
            row[f"{probe}_iou"] = piou
            row[f"{probe}_secs"] = psecs
            row[f"{probe}_raw"] = praw[:200]
        if c.get("context_page") and isinstance(gold, list):
            ctx = page_cache.setdefault(
                c["context_page"], render_page(pdf_path, c["context_page"])
            )
            composite, left_frac = _hstack(img, ctx)
            # remap the left-page gold region into the composite's x scale
            gold_c = [gold[0] * left_frac, gold[1], gold[2] * left_frac, gold[3]]
            (tbox, traw), tsecs = ground(composite, c["target"], PROMPT_TWOPAGE)
            tverd, tcorr, tiou = score(tbox, gold_c)
            # a box whose center is in the right half = boxed the legend, not part
            on_left = tbox is not None and (tbox[0] + tbox[2]) / 2 < left_frac * 1000
            row["twopage_context_page"] = c["context_page"]
            row["twopage_box"] = [round(v) for v in tbox] if tbox else None
            row["twopage_verdict"] = tverd
            row["twopage_correct"] = tcorr
            row["twopage_on_left"] = on_left
            row["twopage_iou"] = tiou
            row["twopage_secs"] = tsecs
            row["twopage_raw"] = traw[:200]

        results.append(row)
        probes = " | ".join(
            f"{p}={row[f'{p}_verdict']}({'Y' if row[f'{p}_correct'] else '-'})"
            for p in ("callout", "leader", "twopage")
            if f"{p}_verdict" in row
        )
        probes = f"  | {probes}" if probes else ""
        print(
            f"[r{rep} {n}/{len(cases)}] {c['id']:<9} {c['category']:<10} "
            f"name={verdict:<17}({'Y' if correct else '-'}) iou={iou} "
            f"{secs}s{probes}"
        )
    return results


def _summarize(results: list) -> dict:
    categories = list(dict.fromkeys(r["category"] for r in results))
    summary = {}
    for cat in categories + ["overall"]:
        rows = [r for r in results if cat == "overall" or r["category"] == cat]
        n = len(rows)
        positives = [r for r in rows if r["gold"] != "NOT_FOUND"]
        negatives = [r for r in rows if r["gold"] == "NOT_FOUND"]
        e = {
            "n": n,
            "accuracy": sum(r["correct"] for r in rows) / n,
            "n_pos": len(positives),
            "region_hit": (
                sum(r["verdict"] == "hit" for r in positives) / len(positives)
                if positives else None
            ),
            "wrong_spot": (
                sum(r["verdict"] == "wrong_spot" for r in positives) / len(positives)
                if positives else None
            ),
            "false_not_found": (
                sum(r["verdict"] == "false_not_found" for r in positives) / len(positives)
                if positives else None
            ),
            "n_neg": len(negatives),
            "not_found_recall": (
                sum(r["verdict"] == "correct_not_found" for r in negatives) / len(negatives)
                if negatives else None
            ),
        }
        # mean grounding latency (name probe), the cost we trade abstention for
        secs = [r["secs"] for r in rows if r.get("secs") is not None]
        e["mean_secs"] = round(sum(secs) / len(secs), 2) if secs else None
        # cross_page probes: name-only vs each context-recovery probe
        for probe in ("callout", "leader", "twopage"):
            pr = [r for r in rows if f"{probe}_correct" in r]
            if pr:
                e.setdefault("ab_name_hit", sum(r["correct"] for r in pr) / len(pr))
                e[f"ab_{probe}_hit"] = sum(r[f"{probe}_correct"] for r in pr) / len(pr)
        summary[cat] = e
    return summary


def _fmt(v):
    return "  -  " if v is None else f"{v:.2f}"


def _print_table(summary: dict) -> None:
    cols = ["accuracy", "region_hit", "wrong_spot", "false_not_found",
            "not_found_recall", "mean_secs"]
    header = f"{'category':<12}{'n':>3}  " + "".join(f"{c:>17}" for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for cat, e in summary.items():
        cells = "".join(f"{_fmt(e[c]):>17}" for c in cols)
        print(f"{cat:<12}{e['n']:>3}  {cells}")
    # cross_page probe line, if present
    for cat, e in summary.items():
        if "ab_name_hit" not in e:
            continue
        parts = [f"name-only {e['ab_name_hit']:.2f}"]
        for probe in ("callout", "leader", "twopage"):
            if f"ab_{probe}_hit" in e:
                parts.append(f"{probe} {e[f'ab_{probe}_hit']:.2f}")
        print(f"\n{cat} cross-page probes — " + "  |  ".join(parts))


@app.local_entrypoint()
def main(
    cases: str = "eval/hyundai-genesis-2-0t-bk2-ground.json",
    thinking: bool = False,
    prompt: str = "baseline",
    repeat: int = 1,
    limit: int = 0,
):
    if prompt not in PROMPTS:
        raise SystemExit(f"--prompt must be one of {sorted(PROMPTS)} (got {prompt!r})")
    spec = json.loads(Path(cases).read_text())
    rows = spec["cases"][:limit] if limit else spec["cases"]
    results = run_eval.remote(spec["doc_id"], rows, thinking, prompt, repeat)

    summary = _summarize(results)
    _print_table(summary)

    tag = "ground-think" if thinking else "ground"
    if prompt != "baseline":
        tag += f"-{prompt}"
    out = Path("eval/results") / f"{spec['doc_id']}-{tag}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "doc_id": spec["doc_id"],
                "thinking": thinking,
                "prompt": prompt,
                "summary": summary,
                "cases": results,
            },
            indent=2,
        )
    )
    print(f"\nFull results written to {out}")
