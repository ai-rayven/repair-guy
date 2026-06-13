#!/usr/bin/env python3
"""Agent eval (stage 1): the search tool's reranker.

The new search tool is ColEmbed's top-N shortlist reranked by MiniCPM5-1B over
each candidate's page text — it shows ONE page. This measures that pick against
gold pages and against the no-rerank baseline (ColEmbed top-1), so we know
whether the 1B reranker helps before wiring the full agent loop into the app.

    modal run scripts/eval_agent_modal.py
    modal run scripts/eval_agent_modal.py --questions eval/<doc>.json --limit 5

Writes eval/results/<doc_id>-agent-<timestamp>.json. ColEmbed (visual) and the
1B agent both load here — both module-level CUDA globals. The parsed store
supplies the candidate page text the reranker reads (the fused search path).
"""
import json
import os
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent))
from index_modal import GPU, hf_cache, image  # noqa: E402

app = modal.App(
    "repair-guy-eval-agent", image=image.add_local_python_source("index_modal")
)

eval_data = modal.Volume.from_name("repair-guy-eval-data", create_if_missing=True)


@app.function(
    gpu=GPU,
    timeout=2 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/root/.cache/huggingface": hf_cache, "/eval_data": eval_data},
)
def run_eval(doc_id: str, questions: list, candidates: int) -> list:
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

    visual = VisualStore(os.path.join(root, VISUAL_SUBDIR))
    parsed = ParsedStore(os.path.join(root, PARSED_SUBDIR))
    for method, store in (("visual", visual), ("parsed", parsed)):
        if not store.exists(doc_id):
            raise SystemExit(
                f"{doc_id} is not indexed under {method}/ in {LIBRARY_DATASET_ID} — "
                f"the fused search needs BOTH indexes. Index it: "
                f"modal run scripts/index_modal.py --method {method} ..."
            )

    # The page text the 1B reranker reads: parsed.json once, indexed by page.
    from core.page_context import index_pages, page_to_text
    from core.pdf import render_page

    page_elements = index_pages(parsed.parsed_pages(doc_id))
    pdf_path = parsed.pdf_path(doc_id)

    # Model loads (module-level cuda): ColEmbed (shortlist), the 1B (text rerank),
    # MiniCPM-V (visual rerank). Three rerankers compared on one shortlist.
    from models import minicpm, minicpm_agent
    from models.colembed import maxsim_search

    results = []
    for n, q in enumerate(questions, start=1):
        gold = set(q["gold_pages"])
        hits = maxsim_search(q["question"], visual, [doc_id], candidates)
        shortlist = [page for _, page, _ in hits]
        images = [render_page(pdf_path, page) for page in shortlist]

        # text rerank (1B over page text)
        cands = [(page, page_to_text(page_elements.get(page, []))) for page in shortlist]
        t_idx, t_raw = minicpm_agent.rerank(q["question"], cands)
        picked_text = shortlist[t_idx] if shortlist else None
        # visual rerank (MiniCPM-V over page images)
        start = time.monotonic()
        v_idx, v_scores = minicpm.rerank_pages(images, q["question"]) if images else (0, [])
        vlm_seconds = round(time.monotonic() - start, 2)
        picked_vlm = shortlist[v_idx] if shortlist else None

        row = {
            k: q.get(k)
            for k in ("id", "category", "question", "gold_pages", "expected_values")
        }
        row["shortlist"] = shortlist
        row["picked_text"] = picked_text
        row["picked_vlm"] = picked_vlm
        row["vlm_scores"] = v_scores
        row["vlm_seconds"] = vlm_seconds
        row["colembed_top1_hit"] = bool(shortlist and shortlist[0] in gold)
        row["shortlist_hit"] = bool(gold & set(shortlist))  # ceiling: gold reachable?
        row["text_rerank_hit"] = bool(picked_text in gold)
        row["vlm_rerank_hit"] = bool(picked_vlm in gold)
        results.append(row)

        def mk(hit):
            return "Y" if hit else "-"

        print(
            f"[{n}/{len(questions)}] {q['id']:<10} top1={mk(row['colembed_top1_hit'])} "
            f"text={mk(row['text_rerank_hit'])}(p.{picked_text}) "
            f"vlm={mk(row['vlm_rerank_hit'])}(p.{picked_vlm})"
        )
    return results


def _summarize(results: list) -> dict:
    """Per-category and overall: ColEmbed top-1 hit, the two reranked hits, and
    the shortlist ceiling (was a gold page even in the shortlist to pick)."""
    categories = list(dict.fromkeys(r["category"] for r in results))
    summary = {}
    for cat in categories + ["overall"]:
        rows = [r for r in results if cat == "overall" or r["category"] == cat]
        n = len(rows)
        summary[cat] = {
            "n": n,
            "colembed_top1": sum(r["colembed_top1_hit"] for r in rows) / n,
            "text_rerank": sum(r["text_rerank_hit"] for r in rows) / n,
            "vlm_rerank": sum(r["vlm_rerank_hit"] for r in rows) / n,
            "ceiling": sum(r["shortlist_hit"] for r in rows) / n,
            "vlm_med_s": sorted(r["vlm_seconds"] for r in rows)[n // 2],
        }
    return summary


def _print_table(summary: dict) -> None:
    cols = ["colembed_top1", "text_rerank", "vlm_rerank", "ceiling", "vlm_med_s"]
    header = f"{'category':<16}{'n':>3}  " + "".join(f"{c:>14}" for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for cat, e in summary.items():
        cells = "".join(f"{e[c]:>14.2f}" for c in cols)
        print(f"{cat:<16}{e['n']:>3}  {cells}")


@app.local_entrypoint()
def main(
    questions: str = "eval/hyundai-genesis-2-0t-bk2-repairs.json",
    candidates: int = 0,
    limit: int = 0,
):
    sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
    from core.constants import AGENT_SEARCH_CANDIDATES

    candidates = candidates or AGENT_SEARCH_CANDIDATES
    spec = json.loads(Path(questions).read_text())
    rows = spec["questions"][:limit] if limit else spec["questions"]
    results = run_eval.remote(spec["doc_id"], rows, candidates)

    summary = _summarize(results)
    _print_table(summary)

    out = Path("eval/results") / f"{spec['doc_id']}-agent-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "doc_id": spec["doc_id"],
                "candidates": candidates,
                "summary": summary,
                "questions": results,
            },
            indent=2,
        )
    )
    print(f"\nFull results written to {out}")
