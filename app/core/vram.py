"""Tiny GPU-memory probe for understanding the VRAM budget.

The Space runs on a ZeroGPU `large` slice (48 GB of an RTX Pro 6000). Several
models are resident at once — the MiniCPM-V "eyes", the ColEmbed search model,
the text embedder, plus one swappable agent "brain" — so it's easy to lose track
of how much headroom is actually left. log_vram() prints a one-line snapshot you
can drop at any interesting point (a model load, a brain evict/load) to watch the
numbers move.

Real device memory is only meaningful INSIDE a @spaces.GPU context: at module
import ZeroGPU runs a CUDA *emulation* mode (models go to "cuda" but no physical
GPU is held), so mem_get_info() either errors or reports the host. Calls are
therefore wrapped defensively — outside a real GPU window they degrade to just
the allocator counters (or nothing) instead of raising.

Four numbers to read:
  alloc    bytes backing live tensors (what the model weights + activations use)
  peak     high-water mark of alloc since the last reset_peak() — catches the
           transient activation/KV spike during generation that a point-in-time
           alloc misses
  reserved bytes the caching allocator holds from the driver (alloc + cached
           free blocks); this is the real pressure on the 48 GB ceiling
  free     driver-reported free VRAM on the device (only inside @spaces.GPU)

Typical usage across one find turn (all inside the @spaces.GPU worker, so the
numbers are real — see the toggle note below):
  set_enabled(vram_log)         # apply the UI toggle for this turn
  reset_peak()                  # zero the high-water mark
  log_vram("turn-start")        # resident models + current brain, idle
  ... use_model() logs evict/load deltas if the brain switches ...
  log_vram("after-ground")      # peak now reflects the VLM grounding spike
"""

import logging

import torch

log = logging.getLogger("repairguy.vram")

_GiB = 1024**3

# Off by default — the UI "VRAM logging" toggle flips this per turn (the find
# pipeline calls set_enabled() at the start of each turn, inside the GPU worker).
# When off, log_vram()/reset_peak() are no-ops, so the probe costs nothing and
# the logs stay quiet. Import-time snapshots (model loads) are therefore silent
# unless the toggle is on the next turn re-triggers a switch.
_ENABLED = False


def set_enabled(flag: bool) -> None:
    """Turn the VRAM probe on or off. Driven by the per-turn UI setting."""
    global _ENABLED
    _ENABLED = bool(flag)


def log_vram(label: str) -> None:
    """Log a GPU-memory snapshot tagged with `label`. No-op unless the probe is
    enabled (set_enabled). When on: safe to call anywhere — no-ops cleanly when
    CUDA is unavailable and tolerates ZeroGPU's import-time emulation mode (where
    device free/total can't be queried)."""
    if not _ENABLED:
        return
    if not torch.cuda.is_available():
        log.info("vram[%s]: cuda unavailable", label)
        return
    alloc = torch.cuda.memory_allocated() / _GiB
    peak = torch.cuda.max_memory_allocated() / _GiB
    reserved = torch.cuda.memory_reserved() / _GiB
    try:
        free, total = torch.cuda.mem_get_info()
        log.info(
            "vram[%s]: alloc=%.2f peak=%.2f reserved=%.2f free=%.2f/%.2f GiB",
            label,
            alloc,
            peak,
            reserved,
            free / _GiB,
            total / _GiB,
        )
    except Exception:
        # Import-time emulation mode: no real device to query free/total.
        log.info(
            "vram[%s]: alloc=%.2f peak=%.2f reserved=%.2f GiB (no device info)",
            label,
            alloc,
            peak,
            reserved,
        )


def reset_peak() -> None:
    """Reset the alloc high-water mark so the next log_vram() peak reflects only
    what happened since this call (e.g. one find turn). No-op when the probe is
    disabled or CUDA is unavailable."""
    if _ENABLED and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
