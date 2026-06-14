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
    modal run scripts/eval_agent_tools_modal.py --prompt topic   # A/B a prompt variant

Writes eval/results/<doc_id>-tools[-<prompt>]-<timestamp>.json.

PROMPT VARIANTS (--prompt): the eval owns these so we can A/B a candidate prompt
WITHOUT editing the app. A variant can change the system prompt (PROMPTS) and/or
the post-search message (POST_SEARCH); `baseline` defers to both shipped strings.
Draft a wording as a variant, run it against baseline, and only copy the winner
into the app — so "did it improve?" is always measured, never guessed.

A scenario is graded on either the FRESH-VIEWER decision (state_message — the
default) or, when it sets `landed: {page}`, the POST-SEARCH decision
(search_result_message — the model just searched and a page is now on screen).
The post-search turn is where a coincidental same-named part on the wrong system
can trip a bad circle, so the `topic` variant gates it on a title/section match.
"""
import json
import re
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


# --- system-prompt variants (the eval OWNS these so we can A/B a candidate prompt
# WITHOUT editing the app — pick with --prompt). A variant is one of:
#   None      → defer to the SHIPPED minicpm_agent.SYSTEM_PROMPT (the "baseline").
#   str       → use this full system prompt verbatim.
#   callable  → transform the shipped prompt, e.g. prepend a rule, so the variant
#               stays DRY instead of duplicating ~2.5KB that would rot.
# To test a new wording: add an entry, run `--prompt <name>` and `--prompt baseline`,
# compare the tables, and only copy the winner into the app's SYSTEM_PROMPT. ---
ANTI_LOOP_PREAMBLE = (
    "CRITICAL — read before anything else: each step answers ONLY the mechanic's "
    "LATEST request, decided fresh. NEVER repeat the action you just took, and "
    "never keep circling the same page once they change topic, reject it, or ask "
    "to navigate — if they are still asking, your last answer did not satisfy "
    "them, so MOVE (search or navigate). Pointing at the same thing again is the "
    "one thing you must never do.\n\n"
)


def _lead(base: str) -> str:
    """Variant: hoist the anti-loop rule to the very TOP of the prompt (a 1B
    over-weights its first tokens; in the shipped prompt that rule sits near the
    bottom). Tests whether position alone recovers the stuck-loop cases."""
    return ANTI_LOOP_PREAMBLE + base


def _v2(base: str) -> str:
    """Variant targeting the two-page weaknesses the spread eval exposed:
    (1) circle-here over-search — tell it to read BOTH shown pages before
        concluding the target is off-screen (the trailing-page answers fail
        because it only reads page 1);
    (2) navigate spread-edge math — add a worked 'go back a page' spread example
        beside the 'next page' one (the model returns an on-screen page instead
        of stepping past the spread edge).
    Implemented as surgical .replace()s so the variant tracks the shipped prompt."""
    s = base.replace(
        "FIRST read the CURRENT PAGE text you were given.",
        "FIRST read ALL the page text on screen — a spread shows TWO pages, so "
        "check BOTH before deciding the target is not on screen.",
    )
    s = s.replace(
        'Mechanic: "next page" (p.46 and p.47 on screen) → '
        '{"tool": "go_to_page", "page": 48}\n',
        'Mechanic: "next page" (p.46 and p.47 on screen) → '
        '{"tool": "go_to_page", "page": 48}\n'
        'Mechanic: "go back a page" (p.46 and p.47 on screen) → '
        '{"tool": "go_to_page", "page": 45}\n',
    )
    return s


def _diagram(base: str) -> str:
    """Variant for the diagram-shown gap (0.00 on EVERY brain — shown-01 circles
    one component out of a whole transmission diagram the mechanic asked to SEE).
    Adds a 'whole diagram already on screen → done' rule to step 1, scoped to
    'show me the WHOLE thing' (not 'circle this part'), so it shouldn't touch the
    circle-here cases. A prompt fix, not a capacity fix — worth testing on a brain
    capable enough to honor the distinction."""
    return base.replace(
        "the one asked for.\n"
        "2. If it is NOT in the current page text, it is not on screen.",
        "the one asked for.\n"
        "If instead they asked to SEE or SHOW a whole diagram, overview, or "
        "components view (not one specific part) and it is already on the screen, "
        "the PAGE ITSELF is the answer → reply done with a brief confirmation; do "
        "NOT circle one component out of a diagram they asked to see in full.\n"
        "2. If it is NOT in the current page text, it is not on screen.",
    )


# v3 bundles the two remaining prompt-shaped gaps: the diagram-shown rule (system
# prompt) + the coincidence gate (post-search, via POST_SEARCH below). Orthogonal
# scenarios (shown-01 vs cos-*), so they don't interact.
PROMPTS = {
    "baseline": None,
    "lead": _lead,
    "v2": _v2,
    "topic": None,
    "v3": _diagram,
}


def _resolve_prompt(variant, shipped: str) -> str:
    """The system-prompt string for a variant: shipped when None, the transform's
    output when callable, else the literal string."""
    if variant is None:
        return shipped
    if callable(variant):
        return variant(shipped)
    return variant


# --- post-search-message variants ------------------------------------------
# The OTHER prompt the agent sees: search_result_message, fed back after a search
# lands a page ("does THIS page show it? → circle"). Scenarios with a `landed`
# field exercise this turn (the existing 35 use state_message and never touch it,
# so a change here can't regress them). A POST_SEARCH entry wraps the shipped
# builder; names absent here use it verbatim. A variant name may live in PROMPTS
# (system prompt) and/or POST_SEARCH (this message) — "topic" only touches this one.
def _topic_post_search(builder):
    """Add a coincidence/topic gate before the 'circle it now' push. The shipped
    message asks only 'is the thing literally on this page?', so a search that
    lands on a topically-WRONG page with a coincidental word match — e.g. 'second
    gear grinding' landing on the fuel-system WGT-actuator page, whose parts table
    lists a part named 'Second gear' — gets a circle on the wrong part. This makes
    the model check the page's title/section matches the request FIRST. Rewrites
    only the non-stuck branch (the stuck branch already forces a decision), as a
    .replace() so it tracks the shipped wording."""
    target = (
        "First, does THIS page show it — the part, or the line/value that answers "
        "it (it counts even when named inside a figure or diagram description)? If "
        "so, circle it now"
    )
    repl = (
        "First, is THIS the right page? Check the title/section at the top: if the "
        "page is about a DIFFERENT system than they asked about — a part that "
        'merely shares a word (a "gear" inside a fuel-system actuator is NOT a '
        "transmission gear) — it is the WRONG page, so do NOT circle; search again "
        "with a more specific query. If it IS the right page and shows the part, or "
        "the line/value that answers it (it counts even when named inside a figure "
        "or diagram description), circle it now"
    )

    def build(request, page, text, stuck):
        msg = builder(request, page, text, stuck)
        return msg if stuck else msg.replace(target, repl)

    return build


POST_SEARCH = {"topic": _topic_post_search, "v3": _topic_post_search}


# --- two-stage decider (prototype) -----------------------------------------
# An alternate architecture (--decider two_stage): instead of one call that picks
# the tool AND its args, make TWO calls — a pure ROUTER that only names the tool
# (no JSON schema in front of it, so it isn't pulled toward "circle" by the
# copy-the-exact-words priming), then an ARGS call that fills the chosen tool's
# fields. Owned by the eval so we can A/B the architecture against the shipped
# single-call decide() WITHOUT rewriting the app pipeline. If it wins here, port
# it to minicpm_agent; if not, we learned it cheaply.
CLASSIFIER_SYSTEM = (
    "You route a hands-busy mechanic's request to ONE action on a repair-manual "
    "page viewer. Reply with ONLY one word: circle, search, go_to_page, or done. "
    "Nothing else — no JSON, no punctuation.\n\n"
    "Decide by THIS step's request, fresh each time:\n"
    "- circle — the REAL thing they asked about is on a page currently on screen "
    "(its full text is given to you). It must be the actual component/value they "
    "mean. A word that only COINCIDENTALLY matches a different-purpose part on an "
    "UNRELATED system is NOT it: check the page's title/section at the top matches "
    'what they asked (a "gear" inside a fuel-system actuator is not a transmission '
    "gear).\n"
    "- search — the part/procedure/spec/section they want is NOT on screen, or the "
    "page on screen is the wrong system. Use this for a symptom, or any topic you "
    "must go find.\n"
    "- go_to_page — they named or implied a specific page: an explicit number, or "
    "'next page' / 'previous page' / 'go back a page' relative to the spread shown.\n"
    "- done — nothing left to do, they are satisfied, or it is not in this manual.\n"
    "If they changed topic, rejected the page, or asked to navigate, do NOT keep "
    "circling the same page."
)

ARGS_SYSTEM = (
    "You fill in the arguments for an action a mechanic's assistant will take on a "
    "repair-manual viewer. Output ONLY the JSON object for the given action — no "
    "prose, no markdown. Replace every <...>; never output angle brackets.\n"
    "- circle: \"target\" is the EXACT words printed on the page (copy them from "
    'the page text you were given), and "page" is the on-screen page number they '
    "are on.\n"
    '- go_to_page: "page" is the physical page number — for "next page" the page '
    'just AFTER the last one shown, for "previous page" / "go back a page" the page '
    "just BEFORE the first one shown.\n"
    '- search: "query" is a short focused phrase for what they want.\n'
    '- done: "message" is one short line.'
)

_ARG_SCHEMA = {
    "search": '{"tool": "search", "query": "<focused search phrase>"}',
    "go_to_page": '{"tool": "go_to_page", "page": <number>}',
    "circle": '{"tool": "circle", "target": "<exact words printed on the page>", '
    '"page": <page number it is on>}',
    "done": '{"tool": "done", "message": "<one short line>"}',
}


def _two_stage_decide(m, history_msgs: list, ctx: str, request: str, router_system: str):
    """The split decider: one call to CHOOSE the tool, a second to FILL its args.
    Returns (tool_dict|None, raw) like minicpm_agent.decide. `m` is the
    minicpm_agent module (reuses its _generate / _parse_tool on the GPU).
    `router_system` is the router's system prompt — a lean dedicated CLASSIFIER_SYSTEM
    (decider=two_stage) or the full shipped SYSTEM_PROMPT (decider=two_stage_full),
    to separate 'is the SPLIT bad?' from 'is my router prompt under-tuned?'."""
    user = f"{ctx}\n\nThe mechanic said: {request!r}\n"
    cls_msgs = (
        [{"role": "system", "content": router_system}]
        + history_msgs
        + [{"role": "user", "content": user
            + "Reply with ONLY one word: circle, search, go_to_page, or done."}]
    )
    raw_cls = m._generate(cls_msgs, 8, trace_name="agent-classify")
    low = raw_cls.lower().replace(" ", "_")
    pos = {t: low.find(t) for t in ("go_to_page", "search", "circle", "done") if low.find(t) >= 0}
    chosen = min(pos, key=pos.get) if pos else None
    if chosen is None:
        return None, f"[classify] {raw_cls}"
    args_msgs = (
        [{"role": "system", "content": ARGS_SYSTEM}]
        + history_msgs
        + [{"role": "user", "content": user
            + f"You will use the {chosen} action. Output ONLY its JSON:\n{_ARG_SCHEMA[chosen]}"}]
    )
    raw_args = m._generate(args_msgs, m.AGENT_MAX_NEW_TOKENS, trace_name="agent-args")
    obj = m._parse_tool(raw_args)
    # The classifier owns the tool; the args step only fills fields. If args drifted
    # to a different tool (or didn't parse), keep the chosen tool and take what we can.
    if obj and obj.get("tool") == chosen:
        tool = obj
    elif obj:
        tool = {**obj, "tool": chosen}
    else:
        tool = {"tool": chosen}
    return tool, f"[classify] {raw_cls}\n[args] {raw_args}"


# --- navigation front gate (prototype) -------------------------------------
# A NARROWER, easier split than the 4-way router: --decider nav_gate runs a 1B
# BINARY classifier (navigation vs content) on the request alone, and on
# 'navigation' computes the page number IN CODE (_resolve_nav), never asking the
# model — which routes go_to_page well but botches the spread-edge arithmetic
# (navigate args 0.40 single-call). Additive and self-guarding: on 'content', or
# when the page can't be resolved, it falls through to the UNCHANGED single-call
# decider, so it can only peel off page-moves — a content request is only at risk
# if it ALSO parses to a page move. Binary intent from the request (no page text
# in front of it) sidesteps the circle-reflex that sank the full router.
NAV_CLASSIFIER_SYSTEM = (
    "Classify what KIND of request a mechanic made on a repair-manual viewer. "
    "Reply with ONLY one word: navigation or content.\n"
    "- navigation = move the viewer to a specific page: an explicit page number "
    "('go to page 612'), or a relative step from the current page ('next page', "
    "'go back a page', 'previous page', 'show me the page before this').\n"
    "- content = anything about a part, system, section, spec, procedure, or "
    "symptom — EVEN IF phrased 'go to' or 'take me to'. 'Go to the cooling system' "
    "and 'take me to the brakes' are CONTENT (they name a topic, not a page). "
    "Pointing at something on screen, and finishing, are also content.\n"
    "Reply with ONLY 'navigation' or 'content'."
)


def _resolve_nav(request: str, shown_pages: list) -> int | None:
    """Deterministic page math for a navigation command — the whole point of the
    gate: once the 1B says 'navigation', the destination is computed in code. An
    explicit 'page N' wins; otherwise a relative step from the spread on screen
    (next → after the last page shown, back/previous → before the first). Returns
    None when nothing resolves (→ let the normal decider handle it), which also
    makes a mis-fired 'navigation' on a topic request a safe no-op."""
    r = request.lower()
    m = re.search(r"page\s+(\d+)", r)
    if m:
        return int(m.group(1))
    if not shown_pages:
        return None
    if any(w in r for w in ("next", "forward", "ahead")):
        return shown_pages[-1] + 1
    if any(w in r for w in ("back", "previous", "prior", "preceding", "before")):
        return shown_pages[0] - 1
    return None


def _nav_gate(m, history_msgs: list, request: str, shown_pages: list):
    """Front gate. Returns (tool, raw) where tool is a go_to_page call when the
    classifier says 'navigation' AND the page resolves, else (None, raw) so the
    caller falls through to the unchanged single-call decider."""
    viewing = (
        f"(Currently viewing p.{', p.'.join(str(p) for p in shown_pages)}.)\n"
        if shown_pages
        else ""
    )
    msgs = (
        [{"role": "system", "content": NAV_CLASSIFIER_SYSTEM}]
        + history_msgs
        + [{"role": "user", "content": f"The mechanic said: {request!r}.\n{viewing}"
            "Reply with ONLY 'navigation' or 'content'."}]
    )
    raw = m._generate(msgs, 8, trace_name="agent-nav-gate")
    if "navigation" not in raw.lower():
        return None, raw
    page = _resolve_nav(request, shown_pages)
    if page is None:
        return None, raw
    return {"tool": "go_to_page", "page": page}, f"[nav-gate] {raw!r} -> page {page}"


# --- history → query rewrite (prototype) -----------------------------------
# --decider rewrite: instead of feeding the multi-turn history into the decider
# (where a 1B parrots its own prior 'circle' and loops), a separate 1B call
# COLLAPSES history + the latest message into ONE self-contained request, and the
# decider then runs with NO history — just that rewritten request. Tests whether
# history is noise the decider would be better off without, with references
# pre-resolved. The shipped decider only uses history to resolve references, so
# this swaps 'decider reads history' for 'rewriter resolves references first'.
REWRITE_SYSTEM = (
    "A mechanic is using a repair-manual assistant across several turns. Rewrite "
    "their LATEST message into ONE self-contained request an assistant could act "
    "on with no other context. Resolve references ('the other one', 'that bolt', "
    "'it') from the earlier turns into the explicit thing. Preserve their intent "
    "exactly:\n"
    "- asking to see/find/circle a part or spec → name it explicitly;\n"
    "- navigating ('next page', 'go back a page') → keep it as is;\n"
    "- rejecting the current page or changing topic → make the NEW target explicit "
    "and drop the old one;\n"
    "- satisfied or finished ('that's it', 'it's working now', 'cool enough') → "
    "say plainly that they are done.\n"
    "Output ONLY the rewritten request — one line, no quotes, no explanation."
)


def _format_history(history: list | None) -> str:
    lines = []
    for t in history or []:
        req = str((t or {}).get("request") or "").strip()
        act = str((t or {}).get("action") or "").strip()
        if req:
            lines.append(f"Mechanic: {req}")
        if act:
            lines.append(f"Assistant: {act}")
    return "\n".join(lines)


def _rewrite_query(m, history: list, request: str) -> str:
    """Collapse history + the latest message into a standalone request. Falls back
    to the raw request if the rewrite comes back empty."""
    transcript = _format_history(history)
    user = (
        (f"Earlier turns:\n{transcript}\n\n" if transcript else "")
        + f"Latest message: {request!r}\nRewrite it as ONE self-contained request:"
    )
    raw = m._generate(
        [{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": user}],
        64,
        trace_name="agent-rewrite",
    )
    rw = raw.strip().strip('"').strip()
    return (rw.splitlines()[0].strip() if rw else "") or request


def _check_args(tool: dict | None, shown_pages: list, expect: dict | None) -> bool | None:
    """Deterministic argument check for the chosen tool — the part of the call we
    can verify without judgement. Returns True/False, or None when no deterministic
    check applies (search query / done message are free text). Mirrors production's
    constraints (pipelines/agent_ask): a circle must land on a page actually on
    screen, and a scenario's `expect.page` pins the exact page the target (circle)
    or destination (go_to_page) must be on."""
    if not tool:
        return None
    name = tool.get("tool")
    expect = expect or {}
    if name == "circle":
        # The model may omit page; production defaults it to the active (first)
        # on-screen page, so validate against that.
        pg = tool.get("page", shown_pages[0] if shown_pages else None)
        if shown_pages and pg not in shown_pages:
            return False  # circled a page that isn't on screen
        if "page" in expect:
            return pg == expect["page"]
        return True
    if name == "go_to_page":
        if "page" in expect:
            return tool.get("page") == expect["page"]
        return None
    return None


@app.function(
    gpu=GPU,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/root/.cache/huggingface": hf_cache, "/eval_data": eval_data},
)
def run_eval(
    doc_id: str,
    scenarios: list,
    prompt_variant: str = "baseline",
    decider: str = "single",
    model_key: str | None = None,
) -> list:
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

    parsed = ParsedStore(f"{root}/{PARSED_SUBDIR}")
    if not parsed.exists(doc_id):
        raise SystemExit(f"{doc_id} is not indexed under parsed/ in {LIBRARY_DATASET_ID}.")
    page_elements = index_pages(parsed.parsed_pages(doc_id))

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

    # Swap the resident brain when a --model key is given (else the default 1B).
    # use_model falls back to the default on an unknown key, so print what actually
    # loaded — that's the brain every decision below ran on.
    active_model = minicpm_agent.use_model(model_key or None)
    print(f"agent model: {active_model}  ({minicpm_agent._active_model_id()})")

    system_content = _resolve_prompt(PROMPTS[prompt_variant], minicpm_agent.SYSTEM_PROMPT)
    ps_variant = POST_SEARCH.get(prompt_variant)
    post_search = (
        minicpm_agent.search_result_message
        if ps_variant is None
        else ps_variant(minicpm_agent.search_result_message)
    )
    print(f"prompt variant: {prompt_variant}  ({len(system_content)} chars)  decider: {decider}")

    results = []
    for n, s in enumerate(scenarios, start=1):
        viewer = s.get("viewer") or {}
        page = int(viewer.get("page") or 0)
        request = s["request"]
        # The page(s) on screen — a scenario sets viewer.pages (the two-page
        # spread); falls back to the single viewer.page. Built to MATCH production
        # (pipelines/agent_ask): the active page (viewer.page) leads the spread,
        # and it is capped at the two pages the viewer shows. Whether the answer
        # sits on the leading or trailing page is set per scenario (vary it to
        # avoid a first-page-position bias). Each page becomes a {page, text}.
        section = str(viewer.get("section") or "")
        history_msgs = history_messages(s.get("history"))
        # A `landed` scenario tests the POST-SEARCH decision: the model has just
        # searched (in history) and a page is now on screen — the exact context
        # that fires the circle-vs-search choice in production (state_message
        # scenarios never reach it). The landed page is the only page on screen.
        landed = s.get("landed")
        # A `landed_candidates` scenario tests the POST-SEARCH RERANK: a search just
        # returned an ordered shortlist (rank-1 first) and ALL of them are on screen
        # with their text; the brain must circle the target on whichever candidate
        # actually has it (expect.page), not blindly on retrieval's #1. Mirrors
        # present_hits in pipelines/agent_ask.
        lc = s.get("landed_candidates")
        if lc:
            cand_pages = [int(p) for p in lc]
            shown_pages = cand_pages
            shown = [
                {"page": p, "text": page_to_text(page_elements.get(p, []))}
                for p in cand_pages
            ]
        elif landed:
            lp = int(landed["page"])
            shown_pages = [lp]
            shown = [{"page": lp, "text": page_to_text(page_elements.get(lp, []))}]
        else:
            shown_pages = [int(p) for p in (viewer.get("pages") or []) if int(p) >= 1]
            if not shown_pages and page:
                shown_pages = [page]
            if page and page in shown_pages:
                shown_pages = [page] + [p for p in shown_pages if p != page]
            shown_pages = shown_pages[:2]
            shown = [
                {"page": p, "text": page_to_text(page_elements.get(p, []))}
                for p in shown_pages
            ]

        if decider in ("two_stage", "two_stage_full"):
            # Neutral context — no JSON schema / "circle it now" priming, so the
            # router isn't pulled toward circle. Mirrors the single-call wording
            # minus the tool instructions.
            if shown:
                where = " and ".join(f"p.{x['page']}" for x in shown)
                sec = f' (section "{section}")' if section else ""
                blocks = "\n\n".join(
                    f"PAGE {x['page']} — full text:\n{x['text'] or '(no text available)'}"
                    for x in shown
                )
                lead = (
                    f"A search just landed on p.{shown[0]['page']}; its text is below."
                    if landed
                    else f"CURRENTLY ON SCREEN — {where}{sec}."
                )
                ctx = f"{lead}\n{blocks}"
            else:
                ctx = "CURRENTLY ON SCREEN: (no page open)"
            router_system = system_content if decider == "two_stage_full" else CLASSIFIER_SYSTEM
            tool, raw = _two_stage_decide(
                minicpm_agent, history_msgs, ctx, request, router_system
            )
        else:
            # rewrite: collapse history into a standalone request and drop history
            # from the decider's input (references pre-resolved by the rewriter).
            eff_request = request
            use_history = history_msgs
            rewrite_note = None
            if decider == "rewrite" and s.get("history"):
                eff_request = _rewrite_query(minicpm_agent, s.get("history"), request)
                use_history = []
                rewrite_note = eff_request

            tool = raw = None
            gate_raw = None
            # nav_gate peels off page-moves first; None means "not a resolvable
            # navigation" → fall through to the unchanged single-call decider.
            if decider == "nav_gate":
                tool, gate_raw = _nav_gate(minicpm_agent, history_msgs, request, shown_pages)
            if tool is None:
                messages = [{"role": "system", "content": system_content}] + use_history
                if lc:
                    candidates = [
                        (p, page_to_text(page_elements.get(p, []))) for p in cand_pages
                    ]
                    messages.append(
                        minicpm_agent.tool_result_message(
                            minicpm_agent.search_results_message(
                                eff_request, candidates, bool(s.get("stuck"))
                            )
                        )
                    )
                elif landed:
                    messages.append(
                        minicpm_agent.tool_result_message(
                            post_search(
                                eff_request, shown_pages[0], shown[0]["text"],
                                bool(landed.get("stuck")),
                            )
                        )
                    )
                else:
                    messages.append(minicpm_agent.state_message(eff_request, shown, section))
                tool, raw = minicpm_agent.decide(messages)
                if gate_raw is not None:
                    # Keep the gate's classifier output visible even when we fell
                    # through, so a 'gate never fired' run is diagnosable.
                    raw = f"[nav-gate→content: {gate_raw!r}]\n{raw}"
                if rewrite_note is not None:
                    raw = f"[rewrite: {rewrite_note!r}]\n{raw}"
            else:
                raw = gate_raw
        chosen = tool["tool"] if tool else None
        parse_ok = tool is not None
        tool_ok = parse_ok and chosen in s["accept"]
        args_ok = _check_args(tool, shown_pages, s.get("expect"))
        results.append({
            "id": s["id"],
            "category": s["category"],
            "request": request,
            "accept": s["accept"],
            "expect": s.get("expect"),
            "chosen": chosen,
            "parse_ok": parse_ok,
            "tool_ok": tool_ok,
            "args_ok": args_ok,
            "raw": raw,
            # The exact page text the model saw — so a circle-vs-search miss can
            # be diagnosed (is the target/value actually in the page text, or did
            # parsing drop it?) without re-deriving the render.
            "shown": shown,
        })
        args_mark = "" if args_ok is None else (" args=OK" if args_ok else " args=BAD")
        print(
            f"[{n}/{len(scenarios)}] {s['id']:<9} {s['category']:<12} "
            f"chose={str(chosen):<14} accept={s['accept']} "
            f"{'OK' if tool_ok else ('PARSE-FAIL' if not parse_ok else 'WRONG')}"
            f"{args_mark}"
        )
    return results


def _summarize(results: list) -> dict:
    categories = list(dict.fromkeys(r["category"] for r in results))
    summary = {}
    for cat in categories + ["overall"]:
        rows = [r for r in results if cat == "overall" or r["category"] == cat]
        n = len(rows)
        # args_acc is over rows with a deterministic check only (args_ok not None).
        arg_rows = [r for r in rows if r.get("args_ok") is not None]
        summary[cat] = {
            "n": n,
            "tool_acc": sum(r["tool_ok"] for r in rows) / n,
            "args_acc": (sum(r["args_ok"] for r in arg_rows) / len(arg_rows))
            if arg_rows
            else None,
            "n_args": len(arg_rows),
            "parse_rate": sum(r["parse_ok"] for r in rows) / n,
        }
    return summary


def _print_table(summary: dict) -> None:
    header = (
        f"{'category':<14}{'n':>3}  {'tool_acc':>9}  {'args_acc':>9}{'(n)':>5}"
        f"  {'parse':>6}"
    )
    print("\n" + header)
    print("-" * len(header))
    for cat, e in summary.items():
        aa = " n/a" if e["args_acc"] is None else f"{e['args_acc']:.2f}"
        print(
            f"{cat:<14}{e['n']:>3}  {e['tool_acc']:>9.2f}  {aa:>9}{e['n_args']:>5}"
            f"  {e['parse_rate']:>6.2f}"
        )


@app.local_entrypoint()
def main(
    scenarios: str = "eval/hyundai-genesis-2-0t-bk2-tools.json",
    prompt: str = "baseline",
    decider: str = "single",
    model: str = "",
    limit: int = 0,
):
    if prompt not in PROMPTS:
        raise SystemExit(f"--prompt must be one of {sorted(PROMPTS)} (got {prompt!r})")
    if decider not in ("single", "two_stage", "two_stage_full", "nav_gate", "rewrite"):
        raise SystemExit(
            "--decider must be single | two_stage | two_stage_full | nav_gate | "
            f"rewrite (got {decider!r})"
        )
    spec = json.loads(Path(scenarios).read_text())
    rows = spec["scenarios"][:limit] if limit else spec["scenarios"]
    results = run_eval.remote(spec["doc_id"], rows, prompt, decider, model or None)

    summary = _summarize(results)
    _print_table(summary)

    # Non-baseline variants / the two-stage decider tag the filename so A/B runs
    # don't clobber each other.
    parts = ["tools"]
    if prompt != "baseline":
        parts.append(prompt)
    if decider != "single":
        parts.append(
            {
                "two_stage": "2stage",
                "two_stage_full": "2stagefull",
                "nav_gate": "navgate",
                "rewrite": "rewrite",
            }[decider]
        )
    if model:
        parts.append(model.replace("/", "_").replace(".", ""))
    tag = "-".join(parts)
    out = Path("eval/results") / f"{spec['doc_id']}-{tag}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "doc_id": spec["doc_id"],
                "prompt": prompt,
                "decider": decider,
                "model": model or "default",
                "summary": summary,
                "scenarios": results,
            },
            indent=2,
        )
    )
    print(f"\nFull results written to {out}")


@app.function(
    timeout=60 * 20,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/root/.cache/huggingface": hf_cache, "/eval_data": eval_data},
)
def dump_pages(doc_id: str, lo: int, hi: int) -> dict:
    """Return the parsed page text for pages lo..hi — a helper for authoring
    page-based scenarios (step-nav, rerank) against the manual's REAL content."""
    sys.path.insert(0, "/root/app")
    from core.constants import LIBRARY_DATASET_ID, PARSED_SUBDIR
    from huggingface_hub import snapshot_download

    root = "/eval_data/preindexed"
    snapshot_download(
        LIBRARY_DATASET_ID, repo_type="dataset", local_dir=root,
        allow_patterns=[f"{PARSED_SUBDIR}/{doc_id}/*"],
    )
    from core.page_context import index_pages, page_to_text
    from core.parsed_store import ParsedStore

    parsed = ParsedStore(f"{root}/{PARSED_SUBDIR}")
    page_elements = index_pages(parsed.parsed_pages(doc_id))
    return {p: page_to_text(page_elements.get(p, [])) for p in range(lo, hi + 1)}


@app.local_entrypoint()
def dump(doc_id: str = "hyundai-genesis-2-0t-bk2", lo: int = 634, hi: int = 646):
    pages = dump_pages.remote(doc_id, lo, hi)
    for p in sorted(pages):
        print(f"\n===== PAGE {p} =====\n{(pages[p] or '(no text)')[:1800]}")
