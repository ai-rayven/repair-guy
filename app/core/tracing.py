"""Langfuse tracing for the find-and-point agent turn — optional, no-op safe.

One agent turn (pipelines/agent_ask.agent_events) is the unit of observability:
the root is an `agent` observation whose children are the brain's decisions
(`generation`), the searches (`retriever`) and the circle grounding
(`generation`). The mapping to Langfuse:

    agent-find (agent)            input=the request, output=the terminal action
    ├─ agent-decide (generation)  the 1B "brain" picks a tool      [models/minicpm_agent]
    ├─ search (retriever)         ColEmbed visual lookup           [pipelines/agent_ask]
    └─ ground-circle (generation) MiniCPM-V places the box          [models/minicpm]

Two constraints shape the design, and are why this does NOT use the SDK's
context-manager "current span" nesting:

  1. The turn runs inside a @spaces.GPU worker (ZeroGPU forks a worker that
     shares the resident CUDA models). The whole trace must therefore be built
     and flushed *inside* that worker — so the root span is opened in
     agent_events, not in the FastAPI endpoint that streams it.
  2. agent_events is a generator. OTel's contextvar-based "current span" is
     fragile across generator yields, so instead we open a root span object,
     stash it on a threadlocal (one worker thread per turn), and attach every
     child to it EXPLICITLY via parent.start_observation(...). No reliance on
     the active context survives a yield.

Tracing is fully disabled — every call a cheap no-op — unless
LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set, so the Space (and local
MOCK_MODELS runs) work unchanged without any Langfuse config.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading

log = logging.getLogger("repairguy.tracing")

# Keys present → tracing on. Read once at import; the Space sets these as Space
# secrets, local dev via the shell / .env.
_ENABLED = bool(
    os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
)
# The skill/CLI use LANGFUSE_BASE_URL; the SDK reads LANGFUSE_HOST. Bridge them
# so a single var configures both.
if os.environ.get("LANGFUSE_BASE_URL") and not os.environ.get("LANGFUSE_HOST"):
    os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]

_client = None
_client_failed = False
# The root span of the turn currently running on this (worker) thread. Children
# attach to it explicitly; cleared by finish_turn so a later untraced turn on
# the same worker thread can't graft onto a stale parent.
_state = threading.local()


def _get_client():
    """The shared Langfuse client, or None when disabled / unconfigurable.
    Lazy so importing this module never touches the network or hard-requires
    the SDK; cached, including the failure case."""
    global _client, _client_failed
    if not _ENABLED or _client_failed:
        return None
    if _client is None:
        try:
            from langfuse import get_client

            _client = get_client()
        except Exception as e:  # SDK missing or misconfigured — degrade to no-op
            log.warning("Langfuse tracing disabled: %s", e)
            _client_failed = True
            return None
    return _client


def enabled() -> bool:
    return _get_client() is not None


def _parent():
    return getattr(_state, "parent", None)


def start_turn(
    *,
    name: str,
    input,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
):
    """Open the root `agent` span for one turn and register it as the active
    parent for this thread. Returns the span (or None when tracing is off).
    Always pair with finish_turn in a finally."""
    client = _get_client()
    if client is None:
        _state.parent = None
        return None
    try:
        from langfuse import propagate_attributes

        # session_id/user_id/tags are trace-level; propagate_attributes stamps
        # them onto the span we create inside the context (and thus its trace).
        # Coerce empty strings to None so we never set blank attributes.
        with propagate_attributes(
            session_id=session_id or None,
            user_id=user_id or None,
            tags=tags or None,
            trace_name=name,
        ):
            span = client.start_observation(
                name=name, as_type="agent", input=input, metadata=metadata
            )
        _state.parent = span
        return span
    except Exception as e:
        log.warning("tracing start_turn failed: %s", e)
        _state.parent = None
        return None


def finish_turn(span, *, output=None, level: str | None = None, status_message: str | None = None):
    """Close the turn: record its output, end the root span, clear the parent,
    and flush (the worker is short-lived, so unflushed events would be lost)."""
    _state.parent = None
    if span is None:
        return
    try:
        if output is not None or level is not None or status_message is not None:
            span.update(output=output, level=level, status_message=status_message)
        span.end()
    except Exception as e:
        log.warning("tracing finish_turn failed: %s", e)
    finally:
        client = _get_client()
        if client is not None:
            with contextlib.suppress(Exception):
                client.flush()


@contextlib.contextmanager
def generation(name: str, *, model: str | None = None, input=None, metadata: dict | None = None):
    """A child `generation` under the active turn — one model decode. Yields the
    span (or None when off / no active turn) so the caller can .update(output=,
    usage_details=). Synchronous body only (no yields inside), so the parent
    link is explicit and contextvar-safe."""
    parent = _parent()
    if parent is None:
        yield None
        return
    gen = None
    try:
        gen = parent.start_observation(
            name=name, as_type="generation", model=model, input=input, metadata=metadata
        )
    except Exception as e:
        log.warning("tracing generation(%s) failed: %s", name, e)
    try:
        yield gen
    finally:
        if gen is not None:
            with contextlib.suppress(Exception):
                gen.end()


@contextlib.contextmanager
def retriever(name: str, *, input=None, metadata: dict | None = None):
    """A child `retriever` under the active turn — one search/lookup. Yields the
    span (or None) for the caller to .update(output=the hits)."""
    parent = _parent()
    if parent is None:
        yield None
        return
    span = None
    try:
        span = parent.start_observation(
            name=name, as_type="retriever", input=input, metadata=metadata
        )
    except Exception as e:
        log.warning("tracing retriever(%s) failed: %s", name, e)
    try:
        yield span
    finally:
        if span is not None:
            with contextlib.suppress(Exception):
                span.end()
