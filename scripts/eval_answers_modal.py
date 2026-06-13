#!/usr/bin/env python3
# Answer eval: takes a retrieval-eval results file (eval_retrieval_modal.py),
# generates MiniCPM answers from each method's retrieved pages, scores them
# against expected_values, and grades them with a judge VLM that sees the
# GOLD pages. Two GPU stages: MiniCPM answers (app image), Qwen3-VL judge
# (its own image — different model family than MiniCPM, no self-judging bias).
#   modal run scripts/eval_answers_modal.py                          # latest results file
#   modal run scripts/eval_answers_modal.py --results eval/results/<file>.json --limit 4
#   modal run scripts/eval_answers_modal.py --results eval/results/<doc>-answers-<ts>.json
#     ^ rows that already have answers skip generation and go straight to the judge
import json
import os
import re
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent))
from index_modal import APP_DIR, GPU, hf_cache, image  # noqa: E402


app = modal.App(
    "repair-guy-eval-answers", image=image.add_local_python_source("index_modal")
)

eval_data = modal.Volume.from_name("repair-guy-eval-data", create_if_missing=True)

METHODS = ("visual", "parsed")
# Covers the longest procedure span in the eval set (15 pages); beyond that
# gold lists duplicate locations of one fact, so truncation loses nothing.
GOLD_PAGES_FOR_JUDGE = 15
# Total pixel budget across a row's gold pages, split evenly between them.
# The judge weighs ~66GB in bf16 on an 80GB card; 5 full-res pages (~2.1MP
# each at 150 DPI) fit the remaining activation/KV headroom, 15 OOM it.
# Short rows stay at full resolution (the per-page share exceeds the render),
# long rows downscale to ~0.8MP/page — still legible to the judge.
JUDGE_PIXEL_BUDGET = 12_000_000

JUDGE_MODEL_ID = os.environ.get("JUDGE_MODEL_ID", "Qwen/Qwen3-VL-32B-Instruct")
JUDGE_GPU = os.environ.get("MODAL_JUDGE_GPU", "A100-80GB")  # ~66GB in bf16
GRADE_SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}

JUDGE_PROMPT = (
    "You are grading an assistant's answer about a repair manual. The images "
    "are the manual pages that contain the correct information (the ground "
    "truth).\n\nQuestion: {question}\n\nAssistant's answer:\n{answer}\n\n"
    "The answer is expected to cite source pages like (Manual — p.123); the "
    "numbers refer to the source PDF, not the page numbers printed on the "
    "pages, so do not treat citations as fabrications or grade them.\n\n"
    "Grade strictly against the pages:\n"
    "- correct: the key values and steps match the pages, nothing important "
    "is missing or made up\n"
    "- partial: on the right track but missing steps/values or including "
    "minor unsupported claims\n"
    "- wrong: contradicts the pages, answers something else, or is mostly "
    "unsupported\n\n"
    "Reply in EXACTLY this format and nothing else:\n"
    "GRADE: correct|partial|wrong\nREASON: <one sentence>"
)

judge_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "torchvision",
        "transformers>=4.57.2,<5",
        "accelerate",
        "qwen-vl-utils",
        "pymupdf",
        "pillow",
        "numpy",
        "huggingface_hub",
    )
    # Variable-size image batches fragment the allocator (the judge sees a
    # different page count per row); expandable segments reclaim that slack.
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(str(APP_DIR), remote_path="/root/app")
    .add_local_python_source("index_modal")
)


def _ensure_pdf(doc_id: str) -> str:
    """The eval volume normally has the doc already (synced by the retrieval
    eval); pull the small parsed folder if not."""
    sys.path.insert(0, "/root/app")
    from core.constants import LIBRARY_DATASET_ID

    pdf_path = f"/eval_data/preindexed/parsed/{doc_id}/doc.pdf"
    if not os.path.exists(pdf_path):
        from huggingface_hub import snapshot_download

        snapshot_download(
            LIBRARY_DATASET_ID,
            repo_type="dataset",
            local_dir="/eval_data/preindexed",
            allow_patterns=[f"parsed/{doc_id}/*"],
        )
        eval_data.commit()
    return pdf_path


@app.function(
    gpu=GPU,
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/root/.cache/huggingface": hf_cache, "/eval_data": eval_data},
)
def generate_answers(doc_id: str, rows: list) -> list:
    """Stage 1: MiniCPM answers each question from each method's retrieved
    pages — the same call the app makes. Only MiniCPM loads here."""
    _ensure_pdf(doc_id)
    from core.parsed_store import ParsedStore
    from core.pdf import render_page
    from models import minicpm

    store = ParsedStore("/eval_data/preindexed/parsed")
    doc_name = store.meta(doc_id)["name"]
    pdf_path = store.pdf_path(doc_id)

    for n, row in enumerate(rows, start=1):
        marks = []
        for method in METHODS:
            pages = [
                (f"{doc_name} — p.{page}", render_page(pdf_path, page))
                for page in row[method]["pages"]
            ]
            start = time.monotonic()
            answer = minicpm.generate_answer(row["question"], pages)
            row[method]["answer_seconds"] = round(time.monotonic() - start, 1)
            row[method]["answer"] = answer
            vals = row.get("expected_values") or []
            row[method]["value_score"] = (
                sum(v.lower() in answer.lower() for v in vals) / len(vals)
                if vals
                else None
            )
            marks.append(f"{method}: val={row[method]['value_score']:.1f}")
        print(f"[{n}/{len(rows)}] {row['id']:<10} {' | '.join(marks)}")
    return rows


@app.function(
    image=judge_image,
    gpu=JUDGE_GPU,
    timeout=2 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/root/.cache/huggingface": hf_cache, "/eval_data": eval_data},
)
def judge_answers(doc_id: str, rows: list) -> list:
    """Stage 2: grade each answer against its gold pages with the judge VLM."""
    pdf_path = _ensure_pdf(doc_id)
    import torch
    from core.pdf import render_page
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading judge {JUDGE_MODEL_ID} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        JUDGE_MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(JUDGE_MODEL_ID)

    def grade(question: str, answer: str, gold_pages: list) -> dict:
        pages = gold_pages[:GOLD_PAGES_FOR_JUDGE]
        max_pixels = JUDGE_PIXEL_BUDGET // max(len(pages), 1)
        content = [
            {
                "type": "image",
                "image": render_page(pdf_path, p),
                "max_pixels": max_pixels,  # qwen_vl_utils resizes per image
            }
            for p in pages
        ]
        content.append(
            {
                "type": "text",
                "text": JUDGE_PROMPT.format(question=question, answer=answer),
            }
        )
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=images, return_tensors="pt").to(
            model.device
        )
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        raw = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0].strip()
        m = re.search(r"\b(correct|partial|wrong)\b", raw.lower())
        return {"grade": m.group(1) if m else None, "raw": raw}

    for n, row in enumerate(rows, start=1):
        marks = []
        for method in METHODS:
            if "answer" not in row[method]:
                continue
            row[method]["judge"] = grade(
                row["question"], row[method]["answer"], row["gold_pages"]
            )
            marks.append(f"{method}: {row[method]['judge']['grade']}")
        print(f"[{n}/{len(rows)}] {row['id']:<10} {' | '.join(marks)}")
    return rows


def _median(values: list) -> float:
    return sorted(values)[len(values) // 2] if values else 0.0


def _summarize(rows: list) -> dict:
    """Per-category and overall value / judge / latency per method. Unparsed
    judge verdicts and questions without expected_values are excluded from
    their averages rather than counted as zero."""
    categories = list(dict.fromkeys(r["category"] for r in rows))
    summary = {}
    for cat in categories + ["overall"]:
        cat_rows = [r for r in rows if cat == "overall" or r["category"] == cat]
        entry = {"n": len(cat_rows)}
        for method in METHODS:
            values = [
                r[method]["value_score"]
                for r in cat_rows
                if r[method].get("value_score") is not None
            ]
            grades = [
                GRADE_SCORE[g]
                for r in cat_rows
                for g in [(r[method].get("judge") or {}).get("grade")]
                if g in GRADE_SCORE
            ]
            entry[method] = {
                "value": sum(values) / len(values) if values else 0.0,
                "judge": sum(grades) / len(grades) if grades else 0.0,
                "ans_median_s": _median(
                    [r[method].get("answer_seconds", 0.0) for r in cat_rows]
                ),
            }
        summary[cat] = entry
    return summary


def _print_table(summary: dict) -> None:
    cols = ["value", "judge", "ans_median_s"]
    labels = {"ans_median_s": "ans_med_s"}
    header = f"{'category':<16}{'n':>3}  {'method':<8}" + "".join(
        f"{labels.get(c, c):>10}" for c in cols
    )
    print("\n" + header)
    print("-" * len(header))
    for cat, entry in summary.items():
        for method in METHODS:
            cells = "".join(f"{entry[method][c]:>10.2f}" for c in cols)
            name = f"{cat:<16}{entry['n']:>3}" if method == METHODS[0] else " " * 19
            print(f"{name}  {method:<8}{cells}")


@app.local_entrypoint()
def main(
    results: str = "",
    questions: str = "eval/hyundai-genesis-2-0t-bk2-repairs.json",
    limit: int = 0,
    judge: bool = True,
):
    if results:
        path = Path(results)
    else:
        candidates = [
            p for p in Path("eval/results").glob("*.json") if "-answers-" not in p.name
        ]
        if not candidates:
            raise SystemExit("No retrieval results found — run eval_retrieval_modal.py first.")
        path = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"Answering from retrieval results: {path}")
    data = json.loads(path.read_text())
    rows = data["questions"][:limit] if limit else data["questions"]

    # Older retrieval results predate expected_values riding along — backfill
    # from the question file by id.
    if any("expected_values" not in r for r in rows) and Path(questions).exists():
        qmap = {
            q["id"]: q.get("expected_values")
            for q in json.loads(Path(questions).read_text())["questions"]
        }
        for r in rows:
            r.setdefault("expected_values", qmap.get(r["id"]))

    out = Path("eval/results") / (
        f"{data['doc_id']}-answers-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )

    def save(judged: bool) -> None:
        out.write_text(
            json.dumps(
                {
                    "doc_id": data["doc_id"],
                    # carried through when resuming from an -answers- file
                    "retrieval_results": data.get("retrieval_results", str(path)),
                    "judge_model": JUDGE_MODEL_ID if judged else None,
                    "summary": _summarize(rows),
                    "questions": rows,
                },
                indent=2,
            )
        )

    if any("answer" not in r[m] for r in rows for m in METHODS):
        rows = generate_answers.remote(data["doc_id"], rows)
        save(judged=False)  # a judge crash must not cost the generation pass
        print(f"Answers checkpointed to {out}")
    else:
        print("All rows already have answers — skipping generation.")
    if judge:
        rows = judge_answers.remote(data["doc_id"], rows)

    _print_table(_summarize(rows))
    save(judged=judge)
    print(f"\nFull results written to {out}")
