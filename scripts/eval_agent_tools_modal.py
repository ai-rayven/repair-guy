#!/usr/bin/env python3
"""Tool-calling eval (Layer A): the agent's decide() step.

Given a viewer state + optional history + request, does MiniCPM5-1B pick an
ACCEPTABLE tool and emit a VALID JSON tool call? This is the gate that catches a
broken 1B before deploy — the redesign has no deterministic fallback. It grades
tool CHOICE against a per-scenario acceptable-set (not one golden tool) plus the
JSON parse rate; it does NOT grade args or run the loop (that's the end-to-end
eval). Only the 1B loads here — no retrieval/VLM.

    modal run scripts/eval_agent_tools_modal.py
    modal run scripts/eval_agent_tools_modal.py --scenarios eval/<doc>-tools.json

Writes eval/results/<doc_id>-tools-<timestamp>.json.
"""
import json
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent))
from index_modal import GPU, hf_cache, image  # noqa: E402

app = modal.App(
    "repair-guy-eval-agent-tools", image=image.add_local_python_source("index_modal")
)

eval_data = modal.Volume.from_name("repair-guy-eval-data", create_if_missing=True)


@app.function(
    gpu=GPU,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/root/.cache/huggingface": hf_cache, "/eval_data": eval_data},
)
def run_eval(doc_id: str, scenarios: list) -> list:
    sys.path.insert(0, "/root/app")
    from core.constants import LIBRARY_DATASET_ID, PARSED_SUBDIR
    from huggingface_hub import snapshot_download

    root = "/eval_data/preindexed"
    print(f"Syncing {doc_id} (parsed) from {LIBRARY_DATASET_ID} ...")
    snapshot_download(
        LIBRARY_DATASET_ID,
        repo_type="dataset",
        local_dir=root,
        allow_patterns=[f"{PARSED_SUBDIR}/{doc_id}/*"],
    )
    eval_data.commit()

    from core.page_context import index_pages, page_to_text
    from core.parsed_store import ParsedStore
    from core.pdf import pdf_outline
    from core.sections import sections_from_chunks, top_sections

    parsed = ParsedStore(f"{root}/{PARSED_SUBDIR}")
    if not parsed.exists(doc_id):
        raise SystemExit(f"{doc_id} is not indexed under parsed/ in {LIBRARY_DATASET_ID}.")
    pdf_path = parsed.pdf_path(doc_id)
    page_elements = index_pages(parsed.parsed_pages(doc_id))
    headings = sections_from_chunks(parsed.chunks(doc_id))
    outline = pdf_outline(pdf_path)

    def options_for(request: str, max_total: int = 24) -> list[dict]:
        """The table of contents shown to the agent — same construction as the
        app's _router_options: clean bookmark chapters plus a per-request fuzzy
        shortlist of fine headings, deduped by page."""
        opts = [{"title": s["title"], "page": s["page_start"]} for s in outline]
        seen = {o["page"] for o in opts}
        for s in top_sections(request, headings, n=8):
            if s["page"] not in seen:
                opts.append(s)
                seen.add(s["page"])
        return opts[:max_total]

    def history_messages(history: list | None) -> list[dict]:
        msgs = []
        for turn in history or []:
            req = str((turn or {}).get("request") or "").strip()
            act = str((turn or {}).get("action") or "").strip()
            if req:
                msgs.append({"role": "user", "content": req})
            if act:
                msgs.append({"role": "assistant", "content": act})
        return msgs

    from models import minicpm_agent

    results = []
    for n, s in enumerate(scenarios, start=1):
        viewer = s.get("viewer") or {}
        page = int(viewer.get("page") or 0)
        request = s["request"]
        # The page(s) on screen — a scenario may set viewer.pages (the two-page
        # spread); falls back to the single page. Each becomes a {page, text}.
        shown_pages = [int(p) for p in (viewer.get("pages") or []) if int(p) >= 1]
        if not shown_pages and page:
            shown_pages = [page]
        shown = [
            {"page": p, "text": page_to_text(page_elements.get(p, []))}
            for p in shown_pages
        ]
        messages = [minicpm_agent.system_message()]
        messages += history_messages(s.get("history"))
        messages.append(
            minicpm_agent.state_message(
                request,
                options_for(request),
                shown,
                str(viewer.get("section") or ""),
            )
        )
        tool, raw = minicpm_agent.decide(messages)
        chosen = tool["tool"] if tool else None
        parse_ok = tool is not None
        tool_ok = parse_ok and chosen in s["accept"]
        results.append({
            "id": s["id"],
            "category": s["category"],
            "request": request,
            "accept": s["accept"],
            "chosen": chosen,
            "parse_ok": parse_ok,
            "tool_ok": tool_ok,
            "raw": raw,
        })
        print(
            f"[{n}/{len(scenarios)}] {s['id']:<9} {s['category']:<12} "
            f"chose={str(chosen):<14} accept={s['accept']} "
            f"{'OK' if tool_ok else ('PARSE-FAIL' if not parse_ok else 'WRONG')}"
        )
    return results


def _summarize(results: list) -> dict:
    categories = list(dict.fromkeys(r["category"] for r in results))
    summary = {}
    for cat in categories + ["overall"]:
        rows = [r for r in results if cat == "overall" or r["category"] == cat]
        n = len(rows)
        summary[cat] = {
            "n": n,
            "tool_acc": sum(r["tool_ok"] for r in rows) / n,
            "parse_rate": sum(r["parse_ok"] for r in rows) / n,
        }
    return summary


def _print_table(summary: dict) -> None:
    header = f"{'category':<14}{'n':>3}  {'tool_acc':>10}{'parse_rate':>12}"
    print("\n" + header)
    print("-" * len(header))
    for cat, e in summary.items():
        print(f"{cat:<14}{e['n']:>3}  {e['tool_acc']:>10.2f}{e['parse_rate']:>12.2f}")


@app.local_entrypoint()
def main(
    scenarios: str = "eval/hyundai-genesis-2-0t-bk2-tools.json",
    limit: int = 0,
):
    spec = json.loads(Path(scenarios).read_text())
    rows = spec["scenarios"][:limit] if limit else spec["scenarios"]
    results = run_eval.remote(spec["doc_id"], rows)

    summary = _summarize(results)
    _print_table(summary)

    out = Path("eval/results") / f"{spec['doc_id']}-tools-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"doc_id": spec["doc_id"], "summary": summary, "scenarios": results}, indent=2)
    )
    print(f"\nFull results written to {out}")
