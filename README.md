# Repair Guy: Hands-Free Manual Navigator

<p align="center">
  <img src="app/frontend/assets/app_screenshot.png" alt="Repair Guy screenshot" width="700">
</p>

**▶️ [Watch the Demo Video](LINK_HERE)** &nbsp;•&nbsp; **🐦 [Social Media Post](LINK_HERE)** &nbsp;•&nbsp; **📝 [Read the Field Notes Blog Post](LINK_HERE)**

> Also live as a Hugging Face Space — see [`app/README.md`](app/README.md) for the Space card.

## 💡 The Problem & Solution
Mechanics with greasy hands can't scroll through 500-page PDFs. **Repair Guy** is a fully local, voice-activated manual navigator.
It visually highlights exact diagrams and troubleshooting steps, and allows for precise page navigation, all hands-free.

## ⚙️ The Tech Stack (All <32B Parameters)
*   **Agent Model:** `openbmb/MiniCPM4.1-8B` (8B) - Handles core logic. *(Note: Fine-tuning a 1B model is planned so the entire stack can run locally on an iPad or iPhone)*.
*   **Vision Model:** `openbmb/MiniCPM-V-4_5` (1B) - Handles visual reasoning, component pinpointing, and generating table/image descriptions.
*   **Embedding Model:** `nvidia/llama-nemotron-embed-vl-1b-v2` (1B) - Extracts structure from dense Toyota Forklift and Hyundai Genesis manuals.
*   **Speech Model:** `moonshine/tiny` (27M) - Runs directly in-browser for ultra-fast, real-time Speech-to-Text.
*   **Infrastructure:** `Modal` - Powers the batch indexing pipeline and automated model evaluations. *(Note: Indexing was offloaded to the cloud as a time-saving measure and to prevent heavy battery drain on edge devices).*
*   **Observability:** `Langfuse/In App` - Stores agent execution traces (for future finetuning) and app displays a diagnostic tab.

## 🎛 Other Features (mostly for engineers that want to experiment)
*   **Speak Responses:** Toggle voice readouts for true hands-free feedback
*   **Careful Pointing:** Forces the VLM to reason before circling components, increasing accuracy on complex diagrams. (Increased latency but, if used with speak responses, you can get a ping when it's done)
*   **Dynamic Indices:** Swap between text-parsed indexing (best for specs/tables) and visual ColEmbed indexing (best for diagrams) for fun to see the difference ;)
*   **Model Swapping:** Swap between different models for the agent brain
*   **VRAM Logging:** Built-in logging to monitor GPU memory during model load/evict cycles.

## 🏆 Bonus Quests Achieved
*   **Off the Grid:** 100% local execution. Zero external cloud APIs used.
*   **Off-Brand:** Custom frontend architecture using `gr.Server`.
*   **Sharing is Caring:** Built-in UI Diagnostic Tab and Langfuse integration for agent traces.
*   **Field Notes:** Detailed write-up covering the architecture.

## 🚀 How to Test It
1. Select the **Toyota Forklift** or **Hyundai Genesis** manual.
2. Click the microphone.
3. Use commands like:
   * *"Show me the oil change procedure"*
   * *"Troubleshoot slipping clutch"*
   * *"Go to the next page"* / *"Go back a page"*
   * *"Go to page 512"*