# Plan: Quiet Hours LLM Throttling

## Problem
After EU+US market close, `metals_loop.py` continues spawning Ministral-8B (every 5min)
and Chronos (every 60s) as subprocesses. Each subprocess cold-starts Python, imports
heavy libraries, and causes CPU spikes that spin up the CPU fan. With zero active positions,
the signals are all HOLD and nobody acts on them.

## Root Cause
`_llm_worker()` in `data/metals_llm.py:638-679` uses fixed intervals (`LLM_INTERVAL=300`,
`CHRONOS_INTERVAL=60`) with no awareness of market hours.

## Solution: Phase A — Time-aware interval throttling

### What
Add a `_is_quiet_hours()` function to `metals_llm.py` that returns True when outside
EU/US market hours (22:00-08:15 CET, weekday; all day weekends). When quiet, use
extended intervals:
- Ministral: 300s → 1800s (30 min)
- Chronos: 60s → 300s (5 min)

### Where
Single file change: `data/metals_llm.py`
- Add `_is_quiet_hours()` helper (uses `cet_hour()` logic, same as `is_market_hours()`)
- Add `LLM_INTERVAL_QUIET` and `CHRONOS_INTERVAL_QUIET` constants
- Modify `_llm_worker()` to pick interval based on `_is_quiet_hours()`
- Log when entering/leaving quiet mode

### What won't break
- Prediction accuracy tracking: unaffected — each sample is independent
- Accuracy stats: fewer samples overnight, but still valid
- Consensus computation: same logic, just runs less often
- Signal data format: unchanged
- metals_loop.py: no changes needed, reads LLM signals via `get_llm_signals()`

### Tests
- Existing tests in `tests/test_unified_loop.py` — verify they pass
- No new tests needed (interval change is config-level, not logic-level)

## Solution: Phase B — Persistent LLM server (attempt, revertible)

### What
Convert Ministral and Chronos from `subprocess.run()` per-call to a persistent
HTTP server or stdin/stdout daemon that stays loaded. Eliminates cold-start CPU spikes.

### Risk
Has failed before in previous attempts. Will implement on a worktree branch and
only merge if it works. Ready to revert.

### Approach
TBD after Phase A is verified.

---
*Written: 2026-04-01*
