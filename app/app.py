"""Gradio + ZeroGPU Space: ask questions over repair manuals.

Showcases two local-only indexing approaches over the same manuals (all
ingestion happens offline via scripts/index_modal.py; the Space only syncs
the pre-indexed library and answers questions):

  Visual — every page embedded as an image with Nemotron ColEmbed v2
           (multi-vector late interaction, no parsing); retrieval is MaxSim
           over page embeddings streamed from disk.
  Parsed — pages parsed with Nemotron Parse, figures/tables described by
           MiniCPM-V, section chunks embedded with Llama Nemotron Embed;
           retrieval is dense cosine over chunks with parent-page lookup.

Both hand the retrieved page images to MiniCPM-V for the grounded answer,
in one ZeroGPU call per question. The UI is a two-panel assistant: a chat on
the left, and on the right a viewer of the source PDF with a strip of the
pages the answer was grounded in. Which approach answers (and how many pages
it retrieves, k) is chosen in the settings modal.

Module layout:
  models/colembed.py        ColEmbed       — visual: page embeddings + MaxSim
  models/nemotron_embed.py  NemotronEmbed  — parsed: dense chunk/query embeddings
  models/minicpm.py         MiniCPM        — shared: answers over page images
  core/visual_store.py      VisualStore    — on-disk per-page token embeddings
  core/parsed_store.py      ParsedStore    — chunks + dense embedding matrix
  pipelines/visual_ask.py   VisualAskPipeline
  pipelines/parsed_ask.py   ParsedAskPipeline
"""

import os
import shutil
import time

import gradio as gr
from huggingface_hub import snapshot_download

from core.constants import (
    DEFAULT_TOP_K,
    LIBRARY_DATASET_ID,
    MAX_TOP_K,
    PARSED_SUBDIR,
    PREINDEXED_DIR,
    VISUAL_SUBDIR,
)
from core.parsed_store import ParsedStore
from core.visual_store import VisualStore
from pipelines.parsed_ask import ParsedAskPipeline
from pipelines.visual_ask import VisualAskPipeline

# Construct once at startup (the models load onto cuda here, in the main
# process). Both pipelines expose the same run(store, question, doc_ids, top_k).
LIBRARIES = {
    "visual": (
        VisualStore(os.path.join(PREINDEXED_DIR, VISUAL_SUBDIR)),
        VisualAskPipeline(),
    ),
    "parsed": (
        ParsedStore(os.path.join(PREINDEXED_DIR, PARSED_SUBDIR)),
        ParsedAskPipeline(),
    ),
}


def sync_library() -> None:
    """Pull pre-indexed manuals from the library dataset into /data.
    A missing or empty dataset just means an empty library."""
    try:
        snapshot_download(
            LIBRARY_DATASET_ID, repo_type="dataset", local_dir=PREINDEXED_DIR
        )
    except Exception as e:
        print(f"Library dataset not synced ({LIBRARY_DATASET_ID}): {e}")
        return
    # PREINDEXED_DIR mirrors the dataset (one dir per method); prune top-level
    # leftovers from the pre-method-prefix layout, which snapshot_download
    # never deletes.
    for entry in os.listdir(PREINDEXED_DIR):
        path = os.path.join(PREINDEXED_DIR, entry)
        if entry.startswith(".") or entry in (VISUAL_SUBDIR, PARSED_SUBDIR):
            continue
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "index.json")):
            print(f"Pruning stale pre-migration doc dir: {entry}")
            shutil.rmtree(path, ignore_errors=True)


sync_library()


def _manual_choices() -> list[tuple[str, str]]:
    """One shared dropdown across both libraries (doc ids are name slugs, so
    the same manual lands on the same id in both); manuals indexed under only
    one approach are labeled with it."""
    docs: dict[str, dict] = {}
    for method, (store, _) in LIBRARIES.items():
        for d in store.list_docs():
            entry = docs.setdefault(d["doc_id"], {"name": d["name"], "methods": []})
            entry["methods"].append(method)
    choices = []
    for doc_id, info in sorted(docs.items(), key=lambda kv: kv[1]["name"].lower()):
        label = info["name"]
        if len(info["methods"]) < len(LIBRARIES):
            label += f" — {info['methods'][0]} only"
        choices.append((label, doc_id))
    return choices


def refresh_library(doc_id):
    """Re-pull the library dataset (incremental) and refresh the dropdown,
    so manuals indexed after the Space booted show up without a restart."""
    sync_library()
    choices = _manual_choices()
    ids = [v for _, v in choices]
    return gr.update(choices=choices, value=doc_id if doc_id in ids else None)


# Labels for the approach picker: value -> (label, one-line description).
APPROACHES = {
    "visual": "🖼️ Visual — ColEmbed late interaction (pages stay images)",
    "parsed": "📄 Parsed — dense chunks over parsed text + figure/table descriptions",
}

EMPTY_PDF = (
    "<div class='pdf-empty'>📄<br>The cited manual page will appear here "
    "once you ask a question.</div>"
)


def _pdf_path(doc_id: str) -> str | None:
    """The source PDF for a manual (kept identically in whichever store indexed
    it — both copy doc.pdf at ingest)."""
    for store, _ in LIBRARIES.values():
        if store.exists(doc_id):
            return store.pdf_path(doc_id)
    return None


def _pdf_viewer(doc_id: str, page: int = 1) -> str:
    """An <iframe> of the manual's PDF, opened at `page`. Gradio serves files
    under allowed_paths at /gradio_api/file=<abs path>; the #page/view fragment
    is honoured by the browser's built-in PDF viewer."""
    path = _pdf_path(doc_id)
    if not path:
        return EMPTY_PDF
    src = f"/gradio_api/file={path}#page={page}&view=FitH"
    return f"<iframe class='pdf-frame' src='{src}'></iframe>"


def _message_text(content) -> str:
    """A chat message's text. Gradio passes Chatbot history back into a function
    in normalized form, where `content` is a list of parts
    [{"type": "text", "text": ...}], not the plain string we appended."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
        ).strip()
    return ""


def add_user(question, history):
    """Show the question immediately and clear the input. Returns ('', history)
    so the textbox empties while the answer is generated by `bot`."""
    question = (question or "").strip()
    history = history or []
    if not question:
        return "", history
    return "", history + [{"role": "user", "content": question}]


def bot(history, manual, approach, k, pages):
    """Answer the last user message with the chosen approach, streaming a
    'searching' state while the single ZeroGPU call runs. `pages` is the
    current citation state, echoed on intermediate yields so it isn't disturbed.

    Yields (chatbot, pdf_html, pages_gallery, page_state)."""
    history = history or []
    hold = (gr.update(), gr.update(), pages)  # pdf, gallery, state: unchanged
    if not history or history[-1]["role"] != "user":
        yield history, *hold
        return
    question = _message_text(history[-1]["content"])

    def assistant_says(text):
        return history + [{"role": "assistant", "content": text}]

    if not manual:
        yield assistant_says("Pick a manual first ☝️"), *hold
        return

    store, pipeline = LIBRARIES[approach]
    if not store.exists(manual):
        other = "parsed" if approach == "visual" else "visual"
        yield assistant_says(
            f"This manual isn't indexed with the **{approach}** approach yet — "
            f"switch to **{other}** in ⚙️ settings, or pick another manual."
        ), *hold
        return

    history = history + [{"role": "assistant", "content": "_Searching the manual…_"}]
    yield history, *hold

    start = time.monotonic()
    try:
        answer, gallery, page_refs = pipeline.run(store, question, [manual], int(k))
    except ValueError as e:
        history[-1]["content"] = f"⚠️ {e}"
        yield history, *hold
        return

    new_pages = [p for _, p in page_refs]
    footer = f"\n\n<sub>⏱️ {time.monotonic() - start:.1f}s · {approach} · k={int(k)}</sub>"
    history[-1]["content"] = answer + footer
    pdf_html = _pdf_viewer(manual, new_pages[0] if new_pages else 1)
    yield history, pdf_html, gallery, new_pages


def jump_to_page(evt: gr.SelectData, manual, pages):
    """Clicking a cited-page thumbnail re-opens the PDF at that page."""
    if not manual or not pages or evt.index >= len(pages):
        return gr.update()
    return _pdf_viewer(manual, pages[evt.index])


def reset_pdf_on_manual_change(manual):
    """Switching manuals opens that manual at page 1 and clears stale citations."""
    return _pdf_viewer(manual, 1) if manual else EMPTY_PDF, [], []


CSS = """
:root { --rg-radius: 16px; }
/* Center the app: the real container class is version-suffixed
   (.gradio-container-6-18-0), so match it by substring and add auto margins. */
[class*="gradio-container"] { max-width: 1500px !important; margin: 0 auto !important; }
.fillable:not(.fill_width) { max-width: 100% !important; }

/* Header ----------------------------------------------------------------- */
#rg-header { align-items: center; margin: 0.4em 0 0.8em; }
#rg-title h1 { margin: 0; font-weight: 700; letter-spacing: -0.01em; }
#rg-title p { margin: 0.15em 0 0; color: var(--body-text-color-subdued); font-size: 0.95em; }
#rg-cog { display: flex; justify-content: flex-end; }
.icon-btn {
    max-width: 46px; min-width: 46px !important; height: 46px;
    border-radius: 50% !important; font-size: 1.25em !important;
    padding: 0 !important; box-shadow: none !important;
}

/* Panels ----------------------------------------------------------------- */
.rg-panel {
    border-radius: var(--rg-radius) !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06), 0 8px 24px rgba(15, 23, 42, 0.05) !important;
    border: 1px solid var(--border-color-primary) !important;
    padding: 14px !important;
}
.rg-panel .chatbot, #rg-chat .bubble-wrap { border: none !important; }

/* PDF viewer ------------------------------------------------------------- */
.pdf-frame {
    width: 100%; height: 620px; border: none;
    border-radius: 12px; background: var(--background-fill-secondary);
}
.pdf-empty {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    height: 620px; gap: 12px; border-radius: 12px; text-align: center;
    font-size: 1.05em; line-height: 1.5; color: var(--body-text-color-subdued);
    background: var(--background-fill-secondary);
}
.pdf-empty::first-line { font-size: 2.4em; }
.cited-label { font-size: 0.85em; color: var(--body-text-color-subdued); margin: 10px 2px 4px; }

/* Settings modal (native overlay, no extra deps) ------------------------- */
#rg-settings {
    position: fixed; inset: 0; z-index: 1000;
    background: rgba(15, 23, 42, 0.55);
    display: flex; align-items: center; justify-content: center;
}
#rg-settings .rg-settings-card {
    width: min(440px, 92vw); background: var(--background-fill-primary);
    border-radius: var(--rg-radius); padding: 22px 24px;
    box-shadow: 0 24px 60px rgba(15, 23, 42, 0.35);
}
.rg-settings-card h3 { margin-top: 0; }
"""


with gr.Blocks(title="Repair Guy") as demo:
    pages_state = gr.State([])  # page numbers of the current citations, gallery-aligned

    with gr.Row(elem_id="rg-header"):
        with gr.Column(scale=8, elem_id="rg-title"):
            gr.Markdown(
                "# 🔧 Repair Guy\n"
                "Your AI repair assistant — ask about a manual, get a grounded "
                "answer with the exact pages it came from."
            )
        with gr.Column(scale=1, elem_id="rg-cog", min_width=60):
            cog_btn = gr.Button("⚙️", elem_classes="icon-btn", variant="secondary")

    with gr.Row(equal_height=True):
        # Left: chat
        with gr.Column(scale=5, elem_classes="rg-panel"):
            manual_in = gr.Dropdown(
                label="Manual", choices=[], filterable=True, container=True
            )
            chatbot = gr.Chatbot(
                height=540,
                show_label=False,
                elem_id="rg-chat",
                avatar_images=(None, None),
                placeholder="### 🔧 Repair Guy\nPick a manual, then describe the issue "
                "or ask a question about it.",
            )
            with gr.Row():
                question_in = gr.Textbox(
                    show_label=False, scale=8, container=False,
                    placeholder="Describe the issue you're facing…",
                )
                submit_btn = gr.Button("Send", scale=1, variant="primary", min_width=90)

        # Right: source PDF + cited pages
        with gr.Column(scale=5, elem_classes="rg-panel"):
            pdf_view = gr.HTML(EMPTY_PDF)
            gr.Markdown("Cited pages", elem_classes="cited-label")
            pages_gallery = gr.Gallery(
                show_label=False, columns=4, height=140, object_fit="contain",
                preview=False, allow_preview=False,
            )

    # Settings modal -----------------------------------------------------------
    with gr.Column(elem_id="rg-settings", visible=False) as settings_modal:
        with gr.Column(elem_classes="rg-settings-card"):
            gr.Markdown("### ⚙️ Settings")
            approach_in = gr.Radio(
                choices=[(label, key) for key, label in APPROACHES.items()],
                value="visual",
                label="Retrieval approach",
            )
            k_in = gr.Slider(
                1, MAX_TOP_K, value=DEFAULT_TOP_K, step=1,
                label="Pages retrieved per question (k)",
            )
            with gr.Row():
                refresh_btn = gr.Button("🔄 Sync library", size="sm")
                close_btn = gr.Button("Done", variant="primary", size="sm")

    # Wiring -------------------------------------------------------------------
    bot_inputs = [chatbot, manual_in, approach_in, k_in, pages_state]
    chat_outputs = [chatbot, pdf_view, pages_gallery, pages_state]

    for trigger in (question_in.submit, submit_btn.click):
        trigger(
            add_user, [question_in, chatbot], [question_in, chatbot], queue=False
        ).then(bot, bot_inputs, chat_outputs)

    pages_gallery.select(
        jump_to_page, inputs=[manual_in, pages_state], outputs=pdf_view
    )
    manual_in.change(
        reset_pdf_on_manual_change, inputs=manual_in,
        outputs=[pdf_view, pages_gallery, pages_state],
    )

    cog_btn.click(lambda: gr.update(visible=True), outputs=settings_modal)
    close_btn.click(lambda: gr.update(visible=False), outputs=settings_modal)
    refresh_btn.click(refresh_library, inputs=[manual_in], outputs=[manual_in])

    demo.load(lambda: gr.update(choices=_manual_choices()), outputs=[manual_in])


# Gradio 6 takes theme/css at launch(), not in the Blocks constructor.
LAUNCH_KWARGS = dict(
    theme=gr.themes.Soft(
        primary_hue="orange",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ),
    css=CSS,
    # PDFs live under the pre-indexed library; allow the file route to serve them.
    allowed_paths=[PREINDEXED_DIR],
)

if __name__ == "__main__":
    demo.launch(**LAUNCH_KWARGS)
