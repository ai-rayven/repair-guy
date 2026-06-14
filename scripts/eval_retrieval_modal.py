#!/usr/bin/env python3
# Retrieval eval: modal run scripts/eval_retrieval_modal.py
# Scores both indexing approaches against a question set with gold pages and
# writes eval/results/<doc_id>-<timestamp>.json (input for eval_answers_modal.py).
import json
import os
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent))
from index_modal import GPU, hf_cache, image  # noqa: E402


app = modal.App(
    "repair-guy-eval-retrieval", image=image.add_local_python_source("index_modal")
)

eval_data = modal.Volume.from_name("repair-guy-eval-data", create_if_missing=True)

TOP_K = 5
# visual = ColEmbed late-interaction; parsed = dense text; rrf = reciprocal-rank
# fusion of the two (the candidate merge a single fused `search` tool would use).
# parsed+ / rrf+ = the same parsed / rrf but with an ELEMENT-TYPE BOOST applied to
# the parsed side: chunks whose type matches the question's kind get their cosine
# score multiplied by BOOST_FACTOR before the page vote. Tests whether knowing the
# query is a "diagram" (boost figure chunks) or a "table" (boost table chunks)
# lifts retrieval — using the eval's category as the ORACLE kind, i.e. the ceiling
# before any 1B has to emit that kind itself.
METHODS = ("visual", "parsed", "rrf", "parsed+", "rrf+")
# RRF constant: rank r (1-based) contributes 1/(RRF_K + r) to a page's score. 60
# is the canonical default (Cormack et al.); it damps the top rank's dominance so
# a page both retrievers rank mid-list can beat one retriever's lone top hit.
RRF_K = 60
# Element-type boost: only diagram and table queries get a boost (locate /
# troubleshoot stay unboosted as controls). diagram -> prefer pages with a Figure
# element; table -> prefer pages with a Table element. SOFT multiplier, not a hard
# filter, so gold content that landed in a section chunk (parser missed the
# element tag) is not excluded — it can still surface.
BOOST_FACTOR = 1.3
BOOST_TYPE = {"diagram": "figure", "table": "table"}


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

    # Model loads happen here (module-level cuda). We use the PUBLIC retrievers
    # (maxsim_search / retrieve_pages) only for a single-query latency probe +
    # drift check; the whole question set is scored in ONE pass over each store
    # via the models' lower-level encode/score paths (the speedup — see below).
    import numpy as np
    import torch

    from core.constants import PARSED_TOP_CHUNKS, SCORE_PAGES_PER_BATCH
    from models import colembed, nemotron_embed
    from models.colembed import maxsim_search
    from pipelines.parsed_ask import _chunk_pages, retrieve_pages

    # RRF fuses a DEEPER candidate list than the top_k we report, so a page ranked
    # mid-list by both retrievers can surface into the fused top_k. Pull at least
    # 20 from each (the late pages cost little and only help the fusion).
    pool = max(top_k * 4, 20)

    def rrf_fuse(ranked_lists):
        """Reciprocal-rank fusion. Each input is hits [(doc, page, score)] in rank
        order; pages are scored by sum over lists of 1/(RRF_K + rank) and returned
        as fused hits [(doc, page, rrf_score)], highest first. Score-scale-free —
        it uses only each retriever's RANKS, so ColEmbed MaxSim and dense cosine
        (different magnitudes) combine without normalization."""
        agg = {}
        for hits in ranked_lists:
            for rank, (doc, page, _) in enumerate(hits, start=1):
                cur = agg.get(page)
                contrib = 1.0 / (RRF_K + rank)
                if cur is None:
                    agg[page] = [doc, contrib]
                else:
                    cur[1] += contrib
        fused = sorted(agg.items(), key=lambda kv: kv[1][1], reverse=True)
        return [(doc, page, score) for page, (doc, score) in fused]

    qtexts = [q["question"] for q in questions]
    n_q = len(qtexts)

    # --- visual: stream the page store ONCE, score every query per batch --------
    # maxsim_search re-reads the WHOLE page-embedding store for each query; the
    # store I/O + host->device copy dominates, so N queries pay it N times. Here
    # we encode each query and then, for each page batch loaded from disk, score
    # every query against it while it is resident on the GPU. Identical MaxSim
    # math (max over doc tokens, sum over query tokens) and identical store
    # iteration order, so rankings match maxsim_search exactly (asserted below) —
    # only the per-query store re-streaming is collapsed into one pass.
    t0 = time.monotonic()
    with torch.no_grad():
        # Encode each query EXACTLY as maxsim_search does — ONE at a time, no
        # batch padding. Batching the encoder (forward_queries over the whole set)
        # pads each query to the batch's longest and shifts its embedding by a few
        # thousandths — enough to reorder pages tied to two decimals, so the
        # rankings would no longer match production. The encode is cheap; the real
        # win is the single page-store pass below, not batching the encode.
        queries = [
            colembed._MODEL.forward_queries([t], batch_size=1)[0].to(torch.float16)
            for t in qtexts
        ]
        vagg = [[] for _ in range(n_q)]
        for refs, batch in stores["visual"].iter_page_batches(
            [doc_id], SCORE_PAGES_PER_BATCH
        ):
            pe = torch.from_numpy(batch).to(queries[0].device)  # [B, T, D] f16
            for i, qv in enumerate(queries):
                sim = torch.einsum("qd,btd->bqt", qv, pe).float()
                scores = sim.amax(dim=2).sum(dim=1)
                vagg[i].extend(
                    (d, p, s) for (d, p), s in zip(refs, scores.tolist())
                )
    visual_pooled = [
        sorted(a, key=lambda r: r[2], reverse=True)[:pool] for a in vagg
    ]
    visual_secs = time.monotonic() - t0

    # --- parsed: load the chunk matrix ONCE, score each query against it --------
    # retrieve_pages reloads the chunk embeddings per query (the I/O cost); here
    # we load once and reuse for every query, scoring each with the SAME matrix-
    # vector product retrieve_pages uses (chunk_mat @ q) so the UNBOOSTED scores
    # are bit-identical to production — then the same top-chunks -> page-voting.
    # parsed+ reuses the same per-query scores with the element-type boost applied
    # before the vote, so it costs essentially nothing extra.
    t0 = time.monotonic()
    chunks, embeddings = stores["parsed"].load(doc_id)
    chunk_mat = embeddings.astype(np.float32)  # [C, dim], L2-normalized at index
    chunk_types = np.array([c.get("type", "") for c in chunks])  # for the boost

    def pages_from_scores(scores):
        """retrieve_pages' tail: top chunks -> parent pages -> pool. Shared by the
        boosted and unboosted variants so only the score vector differs."""
        top = np.argsort(scores)[::-1][:PARSED_TOP_CHUNKS]
        page_refs, page_score = [], {}
        for c in top:
            for page in _chunk_pages(chunks[c]):
                ref = (doc_id, page)
                if ref not in page_score:
                    page_refs.append(ref)
                    page_score[ref] = float(scores[c])
        return [(d, p, page_score[(d, p)]) for d, p in page_refs[:pool]]

    parsed_pooled, parsed_boost_pooled = [], []
    for q_item in questions:
        q = nemotron_embed.embed_query(q_item["question"]).astype(np.float32)
        scores = chunk_mat @ q  # [C] cosine (both sides normalized)
        parsed_pooled.append(pages_from_scores(scores))
        # parsed+: multiply matching-type chunk scores, then re-vote. Categories
        # not in BOOST_TYPE (locate / troubleshoot) get the unboosted result, so
        # they act as controls.
        btype = BOOST_TYPE.get(q_item["category"])
        if btype:
            mult = np.where(chunk_types == btype, BOOST_FACTOR, 1.0).astype(np.float32)
            parsed_boost_pooled.append(pages_from_scores(scores * mult))
        else:
            parsed_boost_pooled.append(parsed_pooled[-1])
    parsed_secs = time.monotonic() - t0

    # --- single-query latency probe + drift check ------------------------------
    # The batched paths above MUST rank identically to the production retrievers;
    # verify on one query, and time the production single-query path — the real
    # product latency, which the amortized batched numbers do NOT reflect.
    probe = qtexts[0]
    t0 = time.monotonic()
    pv = maxsim_search(probe, stores["visual"], [doc_id], top_k)
    lat_v = time.monotonic() - t0
    t0 = time.monotonic()
    pp = retrieve_pages(probe, stores["parsed"], [doc_id], top_k)
    lat_p = time.monotonic() - t0
    vb = [p for _, p, _ in visual_pooled[0][:top_k]]
    pb = [p for _, p, _ in parsed_pooled[0][:top_k]]
    assert vb == [p for _, p, _ in pv], f"visual batch drifted: {vb} vs {pv}"
    assert pb == [p for _, p, _ in pp], f"parsed batch drifted: {pb} vs {pp}"
    # parsed+ runs the same retriever as parsed (just a re-vote on boosted scores),
    # so it carries parsed's latency; rrf+ carries rrf's.
    latency = {
        "visual": round(lat_v, 3),
        "parsed": round(lat_p, 3),
        "rrf": round(lat_v + lat_p, 3),
        "parsed+": round(lat_p, 3),
        "rrf+": round(lat_v + lat_p, 3),
    }
    # Amortized per-query wall time of the batched run (what makes the EVAL fast);
    # rrf adds no retrieval of its own — just the in-memory fuse. parsed+/rrf+
    # share parsed/rrf timing (the boost is a cheap elementwise multiply).
    amort = {"visual": visual_secs / n_q, "parsed": parsed_secs / n_q}
    amort["rrf"] = amort["visual"] + amort["parsed"]
    amort["parsed+"] = amort["parsed"]
    amort["rrf+"] = amort["rrf"]
    print(
        f"single-query latency (production path): "
        f"visual={lat_v:.2f}s parsed={lat_p:.2f}s | "
        f"batched full pass: visual={visual_secs:.1f}s parsed={parsed_secs:.1f}s "
        f"for {n_q} q (amortized {amort['visual']:.3f}s/{amort['parsed']:.3f}s)"
    )

    results = []
    for n, q in enumerate(questions):
        # expected_values ride along so the answer eval can score without
        # needing the question file again.
        row = {
            k: q.get(k)
            for k in ("id", "category", "question", "gold_pages", "expected_values")
        }
        gold = set(q["gold_pages"])
        pooled = {"visual": visual_pooled[n], "parsed": parsed_pooled[n]}
        pooled["rrf"] = rrf_fuse([pooled["visual"], pooled["parsed"]])
        pooled["parsed+"] = parsed_boost_pooled[n]
        pooled["rrf+"] = rrf_fuse([visual_pooled[n], parsed_boost_pooled[n]])

        marks = []
        for method in METHODS:
            hits = pooled[method][:top_k]
            row[method] = {
                "pages": [page for _, page, _ in hits],
                "scores": [round(score, 4) for _, _, score in hits],
                # amortized batched per-query time; see latency_single_query in
                # the written results for the real production single-query cost.
                "seconds": round(amort[method], 3),
            }
            rank = next(
                (i for i, (_, page, _) in enumerate(hits, start=1) if page in gold),
                None,
            )
            marks.append(f"{method}: {'hit@' + str(rank) if rank else 'MISS'}")
        results.append(row)
        print(f"[{n + 1}/{n_q}] {q['id']:<10} {' | '.join(marks)}")
    return {"results": results, "latency": latency}


def _summarize(results: list, top_k: int) -> dict:
    """Per-category and overall hit@k / precision / recall / MRR per method."""
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
def main(questions: str = "eval/hyundai-genesis-2-0t-bk2-repairs.json", top_k: int = TOP_K):
    spec = json.loads(Path(questions).read_text())
    out_data = run_eval.remote(spec["doc_id"], spec["questions"], top_k)
    results, latency = out_data["results"], out_data["latency"]

    summary = _summarize(results, top_k)
    _print_table(summary, top_k)
    # median_s in the table is the AMORTIZED batched per-query time (all questions
    # scored in one pass over each store). The real product single-query latency
    # is the probe below — not amortizable, since a live request scores one query.
    print(
        "\nsingle-query latency (production path): "
        + "  ".join(f"{m}={latency[m]}s" for m in METHODS)
    )

    out = Path("eval/results") / f"{spec['doc_id']}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "doc_id": spec["doc_id"],
                "top_k": top_k,
                "latency_single_query": latency,
                "summary": summary,
                "questions": results,
            },
            indent=2,
        )
    )
    print(f"\nFull results written to {out}")
