"""Pre-warm the NEXT LLM in the rotation so the upcoming loop cycle
doesn't pay the cold-swap cost.

2026-05-11 Stage 3 Phase 1: minimal in-process prewarmer. Called from
``flush_llm_batch()`` right after the rotation counter increments. Issues
a 1-token dummy prompt to the next-slot model, which forces
``llama_server`` to swap to it synchronously inside the prewarm call —
the next real cycle (60 s away) then finds the model already loaded and
skips the swap entirely.

Why this is safe:

- All exceptions are swallowed and logged at WARNING. A broken
  prewarmer cannot regress the working ministral/qwen3/fingpt path
  because the call site in ``flush_llm_batch()`` wraps this in its own
  try/except as a belt-and-braces backstop.
- We rely solely on the public ``query_llama_server`` contract — we
  never touch ``llama_server`` internals or the file lock directly.
  The 1-token dummy prompt holds the same locks the real swap path
  holds; concurrent metals_loop swaps will simply queue behind us
  (~10-30 s typical).
- Chronos (``gpu_gate("chronos", timeout=30)``) is unaffected: this
  module never acquires ``gpu_gate`` itself. The win is that by the
  time Chronos runs in the *next* cycle, the LLM swap is already
  done — so Chronos's 30 s timeout no longer races a mid-flight swap.

Rotation order (must match ``llm_batch._LLM_ROTATION``):
``ministral → qwen3 → fingpt``. The llama_server slot names are
``ministral3`` / ``qwen3`` / ``finance-llama-8b`` respectively — see
``ROTATION_SLOTS`` below for the mapping.

State persistence: writes a single line to
``data/llm_rotation_state.jsonl`` after each prewarm attempt. On
restart, the most recent line is consulted so a process that just
prewarmed slot S at counter C doesn't redundantly prewarm S again
when restarted at the same counter (which would happen if a crash
left the rotation counter at the same value).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("portfolio.llm_prewarmer")

# Rotation order pinned to llm_batch._LLM_ROTATION. Each entry is the
# abstract rotation name; ROTATION_SLOT_TO_SERVER maps it to the actual
# llama_server slot name (which is what _read_pid_model() returns and
# what query_llama_server expects as `name`).
ROTATION_SLOTS: tuple[str, ...] = ("ministral", "qwen3", "fingpt")

ROTATION_SLOT_TO_SERVER: dict[str, str] = {
    "ministral": "ministral3",
    "qwen3": "qwen3",
    "fingpt": "finance-llama-8b",
}

# State file: one JSONL line per prewarm attempt. Used to short-circuit
# duplicate prewarms across process restarts at the same counter.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATE_FILE = DATA_DIR / "llm_rotation_state.jsonl"


def _next_slot(current_counter: int) -> str:
    """Compute the next rotation slot name. Mirrors the rotation logic in
    ``llm_batch.is_llm_on_cycle``: after a flush at counter C completes,
    the counter is incremented so the *upcoming* cycle's slot index is
    ``(new_counter - 1) % 3``. The next cycle after that is
    ``new_counter % 3`` — and that's what we want to pre-warm.

    But we're called with ``current_counter`` being the *just-incremented*
    counter (i.e. the counter that flush_llm_batch leaves in shared_state
    after the bump). The next cycle's slot is ``(current_counter - 1) % 3``;
    we want the slot AFTER that — i.e. ``current_counter % 3``.

    Worked example with ROTATION = (ministral, qwen3, fingpt):

        flush 1 finishes, counter=1, ran ministral.
          next cycle will run qwen3 (slot = (1-1) % 3 = 0 → ministral).
          Wait, that's wrong direction.

    Re-read is_llm_on_cycle: at enqueue time counter==1 means slot 0
    (ministral) is the active LLM for that cycle. The counter is
    incremented AFTER the flush. So when we're called by flush_llm_batch
    POST-increment with counter=2, the next real cycle has counter=2
    and slot = (2-1) % 3 = 1 → qwen3. We pre-warm qwen3.

    Therefore: prewarm slot index = ``(current_counter - 1) % 3``.
    """
    if not ROTATION_SLOTS:
        raise ValueError("ROTATION_SLOTS is empty")
    idx = (int(current_counter) - 1) % len(ROTATION_SLOTS)
    return ROTATION_SLOTS[idx]


def _read_last_state() -> dict | None:
    """Return the most recent entry from STATE_FILE, or None.

    Fix C 2026-05-11 (codex review): true tail read instead of
    ``f.readlines()``. The state file is bounded by log_rotation now
    (Fix B), but even pre-rotation we don't want to grow O(n) with
    file size — every prewarm pays this. We seek to the end and read
    a small trailing block, then split on newlines and take the last
    complete JSON line. If the block doesn't contain a complete line
    (extremely long single record, unlikely), fall back to a full read
    once.
    """
    import json

    TAIL_BLOCK = 8192

    try:
        if not STATE_FILE.exists():
            return None
        size = STATE_FILE.stat().st_size
        if size == 0:
            return None

        with open(STATE_FILE, "rb") as f:
            if size <= TAIL_BLOCK:
                f.seek(0)
                block = f.read()
            else:
                f.seek(size - TAIL_BLOCK)
                block = f.read()

        text = block.decode("utf-8", errors="replace")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return None

        # If we tail-read and the first kept line came from a partial
        # block (i.e. we started mid-line because there was no newline
        # at the seek point), it may parse garbage. The LAST line is
        # always safe because the writer terminates with \n via
        # atomic_append_jsonl. We try it first.
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            # Fallback: pathological case where the tail block does not
            # contain a complete JSON record. Read the whole file once.
            if size > TAIL_BLOCK:
                with open(STATE_FILE, encoding="utf-8") as f:
                    all_lines = [ln.strip() for ln in f if ln.strip()]
                if all_lines:
                    return json.loads(all_lines[-1])
            return None
    except Exception as e:
        logger.warning("llm_prewarmer state read failed: %s", e)
        return None


def _write_state(counter: int, prewarmed_slot: str, server_slot: str,
                 outcome: str, duration_s: float | None = None) -> None:
    """Append a single state record. Best-effort; swallows errors."""
    try:
        from portfolio.file_utils import atomic_append_jsonl
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "counter": int(counter),
            "prewarmed_slot": prewarmed_slot,
            "server_slot": server_slot,
            "outcome": outcome,
        }
        if duration_s is not None:
            entry["duration_s"] = round(float(duration_s), 3)
        atomic_append_jsonl(STATE_FILE, entry)
    except Exception as e:
        logger.warning("llm_prewarmer state write failed: %s", e)


def _current_loaded_server_slot() -> str | None:
    """Return the llama_server slot name currently loaded according to the
    PID file, or None on any error/missing file.

    Used both to short-circuit prewarm when the target is already loaded
    AND to validate the JSONL idempotency record against ground truth —
    see Fix A 2026-05-11 in ``prewarm_next_model``.
    """
    try:
        from portfolio.llama_server import _read_pid_model
        _, current_model = _read_pid_model()
        return current_model
    except Exception as e:
        logger.debug("llm_prewarmer load-check failed: %s", e)
        return None


def _is_slot_already_loaded(server_slot: str) -> bool:
    """Check llama_server's PID file to see if the target slot is already
    the active model. Returns False on any error (safe default — we'd
    rather prewarm an already-loaded model than skip a needed prewarm).
    """
    return _current_loaded_server_slot() == server_slot


def prewarm_next_model(current_counter: int) -> bool:
    """Issue a dummy 1-token prompt to the next-slot model.

    Args:
        current_counter: the value of ``shared_state._full_llm_cycle_count``
            *after* ``flush_llm_batch`` has incremented it. Must be a
            positive int — counter==0 means the warmup hasn't happened
            yet, no useful prewarm possible.

    Returns:
        True if a prewarm query was actually dispatched.
        False if the prewarm was a no-op (already loaded / duplicate /
        invalid counter / error). Never raises.

    Contract: this function MUST NOT propagate exceptions to the caller.
    A broken prewarmer cannot be allowed to regress the working LLM
    rotation path.
    """
    try:
        # Test-suite safety: pytest sets PYTEST_CURRENT_TEST for every test
        # function. We auto-skip when that's present so the real prewarmer
        # never fires during test collection of the broader suite (which
        # would issue real swap requests against a running llama-server
        # if one happens to be up on the dev box). Tests that DO want to
        # exercise the prewarmer directly call prewarm_next_model from
        # their own fixtures with mocked dependencies — those tests bypass
        # this guard by setting PF_PREWARM_FORCE_RUN=1.
        if (
            os.environ.get("PYTEST_CURRENT_TEST")
            and os.environ.get("PF_PREWARM_FORCE_RUN") != "1"
        ):
            logger.debug("llm_prewarmer skip: pytest detected, no-op")
            return False

        counter = int(current_counter)
        if counter <= 0:
            # Counter==0 means no flush has run yet, so the rotation hasn't
            # started; counter<0 is nonsense. Either way: no-op.
            logger.debug("llm_prewarmer skip: counter=%d not positive", counter)
            return False

        next_slot = _next_slot(counter)
        server_slot = ROTATION_SLOT_TO_SERVER.get(next_slot)
        if server_slot is None:
            logger.warning(
                "llm_prewarmer skip: no server mapping for slot=%s", next_slot,
            )
            return False

        # Fix A 2026-05-11 (codex review): reconcile JSONL idempotency
        # against the *currently loaded* slot. The rotation counter is
        # in-memory only and resets to 0 on process restart, which means a
        # fresh process will re-hit counter=1, counter=2, ... — and the
        # state JSONL from the previous process lifetime still has a
        # matching "warmed" line for those counters. Trusting it blindly
        # would let a restart skip a swap that is actually still needed.
        #
        # Rule: skip-by-state only if BOTH the JSONL record matches the
        # current (counter, slot) AND the llama_server PID file confirms
        # the expected slot is still loaded. Any mismatch → force prewarm.
        currently_loaded = _current_loaded_server_slot()
        last = _read_last_state()
        if (
            last is not None
            and int(last.get("counter", -1)) == counter
            and last.get("prewarmed_slot") == next_slot
            and last.get("outcome") == "warmed"
            and currently_loaded == server_slot
        ):
            logger.debug(
                "llm_prewarmer skip: counter=%d slot=%s already prewarmed "
                "and still loaded",
                counter, next_slot,
            )
            return False

        # If the target model is already the active llama-server model
        # (e.g. metals_loop happened to swap to it for an unrelated
        # reason), there is nothing to do.
        if currently_loaded == server_slot:
            logger.info(
                "llm_prewarmer noop: slot=%s server=%s already loaded",
                next_slot, server_slot,
            )
            _write_state(counter, next_slot, server_slot, outcome="already_loaded")
            return False

        # Fire the dummy query. n_predict=1 keeps the prompt-completion
        # work minimal — the load + KV-cache-prime cost dominates and
        # that's what we actually want to pay before the next loop cycle.
        from portfolio.llama_server import query_llama_server
        t0 = time.monotonic()
        logger.info(
            "llm_prewarmer start: counter=%d slot=%s server=%s",
            counter, next_slot, server_slot,
        )
        text = query_llama_server(
            server_slot, "test", n_predict=1, temperature=0.0,
        )
        duration = time.monotonic() - t0

        if text is None:
            # query_llama_server returns None on failure but does not
            # raise. Treat that as a soft failure: state is recorded but
            # outcome is logged so we can see prewarm failures in the
            # JSONL.
            logger.warning(
                "llm_prewarmer query returned None: counter=%d slot=%s in %.1fs",
                counter, next_slot, duration,
            )
            _write_state(counter, next_slot, server_slot,
                         outcome="query_none", duration_s=duration)
            return False

        logger.info(
            "llm_prewarmer warmed: counter=%d slot=%s server=%s in %.1fs",
            counter, next_slot, server_slot, duration,
        )
        _write_state(counter, next_slot, server_slot,
                     outcome="warmed", duration_s=duration)
        return True
    except Exception as e:
        # Defensive backstop. Anything else in this function should have
        # already caught its own errors; this is the contract guarantee
        # that the prewarmer NEVER raises.
        logger.warning(
            "llm_prewarmer unexpected failure: counter=%s err=%s",
            current_counter, e,
        )
        return False
