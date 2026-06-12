#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent))
from index_modal import GPU, hf_cache, image  # noqa: E402


app = modal.App("repair-guy-eval", image=image.add_local_python_source("index_modal"))

eval_data = modal.Volume.from_name("repair-guy-eval-data", create_if_missing=True)

TOP_K = 5
METHODS = ("visual", "parsed")


@app.function(
    gpu=GPU,
    timeout=2 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/root/.cache/huggingface": hf_cache, "/eval_data": eval_data},
)
def run_eval(doc_id: str, questions: list, top_k: int) -> list:
    sys.path.insert(0, "/root/app")
    from core.constants import LIBRARY_DATASET_ID, PARSED_SUBDIR, VISUAL_SUBDIR
    from huggingface_hub import snapshot_download

    root = "/eval_data/preindexed"
    print(f"Syncing {doc_id} indexes from {LIBRARY_DATASET_ID} ...")
    snapshot_download(
        LIBRARY_DATASET_ID,
        repo_type="dataset",
        local_dir=root,
        allow_patterns=[f"{VISUAL_SUBDIR}/{doc_id}/*", f"{PARSED_SUBDIR}/{doc_id}/*"],
    )
    eval_data.commit()

    from core.parsed_store import ParsedStore
    from core.visual_store import VisualStore

    stores = {
        "visual": VisualStore(os.path.join(root, VISUAL_SUBDIR)),
        "parsed": ParsedStore(os.path.join(root, PARSED_SUBDIR)),
    }
    for method, store in stores.items():
        if not store.exists(doc_id):
            raise SystemExit(
                f"{doc_id} is not indexed under {method}/ in {LIBRARY_DATASET_ID} — "
                f"index it first: modal run scripts/index_modal.py --method {method} ..."
            )

    # Model loads happen here (module-level cuda); MiniCPM comes along with the
    # parsed_ask import but is unused — retrieval only.
    from models.colembed import maxsim_search
    from pipelines.parsed_ask import retrieve_pages

    retrievers = {"visual": maxsim_search, "parsed": retrieve_pages}

    results = []
    for n, q in enumerate(questions, start=1):
        row = {k: q[k] for k in ("id", "category", "question", "gold_pages")}
        gold = set(q["gold_pages"])
        marks = []
        for method in METHODS:
            start = time.monotonic()
            hits = retrievers[method](q["question"], stores[method], [doc_id], top_k)
            row[method] = {
                "pages": [page for _, page, _ in hits],
                "scores": [round(score, 4) for _, _, score in hits],
                "seconds": round(time.monotonic() - start, 2),
            }
            rank = next(
                (i for i, (_, page, _) in enumerate(hits, start=1) if page in gold),
                None,
            )
            marks.append(f"{method}: {'hit@' + str(rank) if rank else 'MISS'}")
        results.append(row)
        print(f"[{n}/{len(questions)}] {q['id']:<10} {' | '.join(marks)}")
    return results


def _summarize(results: list, top_k: int) -> dict:
    """Per-category and overall hit@k / MRR / median seconds per method."""
    categories = list(dict.fromkeys(r["category"] for r in results))
    summary = {}
    for cat in categories + ["overall"]:
        rows = [r for r in results if cat == "overall" or r["category"] == cat]
        entry = {"n": len(rows)}
        for method in METHODS:
            ranks, precisions, recalls = [], [], []
            for r in rows:
                gold = set(r["gold_pages"])
                pages = r[method]["pages"]
                ranks.append(
                    next((i for i, p in enumerate(pages, start=1) if p in gold), None)
                )
                found = len(gold & set(pages))
                # Precision over pages actually returned (parsed may return
                # fewer than top_k): how clean is the context MiniCPM gets.
                precisions.append(found / len(pages) if pages else 0.0)
                # Recall capped at top_k slots: gold often lists every duplicate
                # location of one fact (one suffices), so |gold| can exceed
                # top_k; for procedure page spans this measures coverage.
                recalls.append(found / min(len(gold), top_k))
            secs = sorted(r[method]["seconds"] for r in rows)
            entry[method] = {
                **{
                    f"hit@{k}": sum(1 for r in ranks if r and r <= k) / len(rows)
                    for k in (1, 3, top_k)
                },
                "precision": sum(precisions) / len(rows),
                "recall": sum(recalls) / len(rows),
                "mrr": sum(1 / r for r in ranks if r) / len(rows),
                "median_s": secs[len(secs) // 2],
            }
        summary[cat] = entry
    return summary


def _print_table(summary: dict, top_k: int) -> None:
    cols = ["hit@1", "hit@3", f"hit@{top_k}", "precision", "recall", "mrr", "median_s"]
    labels = {"precision": f"prec@{top_k}", "recall": f"rec@{top_k}", "median_s": "med_s"}
    header = f"{'category':<16}{'n':>3}  {'method':<8}" + "".join(
        f"{labels.get(c, c):>9}" for c in cols
    )
    print("\n" + header)
    print("-" * len(header))
    for cat, entry in summary.items():
        for method in METHODS:
            cells = "".join(f"{entry[method][c]:>9.2f}" for c in cols)
            name = f"{cat:<16}{entry['n']:>3}" if method == METHODS[0] else " " * 19
            print(f"{name}  {method:<8}{cells}")


@app.local_entrypoint()
def main(questions: str = "eval/hyundai-genesis-2-0t-bk2-repairs.json", top_k: int = TOP_K):
    spec = json.loads(Path(questions).read_text())
    results = run_eval.remote(spec["doc_id"], spec["questions"], top_k)

    summary = _summarize(results, top_k)
    _print_table(summary, top_k)

    out = Path("eval/results") / f"{spec['doc_id']}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"doc_id": spec["doc_id"], "top_k": top_k, "summary": summary, "questions": results},
            indent=2,
        )
    )
    print(f"\nFull results written to {out}")
