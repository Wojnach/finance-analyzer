# PLAN — heartbeat() before Layer 2 invocation

**Date:** 2026-05-04
**Branch:** `feat/heartbeat-before-l2-2026-05-04`
**Worktree:** `.worktrees/heartbeat-before-l2-2026-05-04`

## Problem

Even after the regime_accuracy_cache fix landed (cycles dropped 595s → 67s),
the dashboard `/api/health` flag still flips fresh→stale on cycles that
trigger Layer 2. Reason: `update_health()` (which writes `last_heartbeat`)
is called at the END of each cycle in `portfolio/main.py:882`, AFTER the
Layer 2 invocation at line 830. T2 has a 600s timeout and T3 a 900s timeout,
so a single triggering cycle can leave the heartbeat stale for up to 15 min.

This is the residual cause of "stale" flags after the cache fix. Confirmed
in tonight's data: cycle 3 with `uptime=1566s` and `hb age=446s` happened
because BTC-USD triggered T2 at 15:21:06 and the cycle is still in flight
waiting on Claude CLI.

## Fix

Two pieces in `portfolio/health.py`:

1. `heartbeat()` — updates ONLY `last_heartbeat`. No cycle counters, no
   signals_ok churn, no errors. Cheap and thread-safe via `_health_lock`.
2. `heartbeat_keepalive(interval=60)` — context manager that beats once
   synchronously on `__enter__`, then ticks `heartbeat()` every interval
   seconds via a daemon thread until `__exit__`.

Initial design used a single pre-Layer-2 beat. Codex review (P1) caught
that this only delays staleness by 5 minutes — at 300s into a 600s T2
the gate trips again. The keepalive version keeps health fresh for the
full duration. Default interval=60s gives 5x headroom against the 300s
stale threshold.

Call site in `main.py:824-855` (only the actually-blocking branches —
`invoke_agent` and `autonomous_decision`, NOT the `NO_TELEGRAM` /
`outside agent window` skip-paths which are bounded short).

```python
with heartbeat_keepalive():
    result = invoke_agent(reasons_list, tier=tier)
```

The `with` statement guarantees `__exit__` runs even if the wrapped
block raises, so the daemon thread is always cleaned up. No outer
try/except needed; existing main-loop exception handling at line 1149+
catches anything that escapes.

## Why a separate function and not just `update_health()`?

`update_health()` requires `cycle_count`, `signals_ok`, `signals_failed`. At
the pre-Layer-2 point we have those values, so re-using it is technically
fine — but it would mean two writes of those counters per cycle (once
before Layer 2, once after with the same values). The dedicated `heartbeat()`
makes intent obvious in the call site (we ONLY want to bump the watchdog,
not record cycle progress) and avoids re-counting risk if a future refactor
moves the cycle-end accounting elsewhere.

## Tests

`tests/test_heartbeat_function.py`:

- `test_heartbeat_updates_only_last_heartbeat_field` — pre-populate state
  with cycle_count, signals_ok, errors. Call heartbeat(). Assert only
  `last_heartbeat` advanced; cycle_count/signals_ok/errors unchanged.
- `test_heartbeat_creates_state_when_missing` — empty data dir; call
  heartbeat(); state file exists with `last_heartbeat` set.
- `test_heartbeat_is_thread_safe` — N threads call heartbeat() in parallel;
  no torn reads, final timestamp parses, no exceptions.
- `test_heartbeat_uses_atomic_write` — patch `atomic_write_json`; assert
  it's called (not raw `open(...).write(...)`).

## Risk + rollback

- The pre-Layer-2 heartbeat advances `last_heartbeat` even if Layer 2
  fails. That's correct — we want "loop is alive" semantics, not "Layer 2
  succeeded" semantics. `agent_silence` checks already use `last_invocation_ts`
  (set inside `update_health` only when a trigger fires), which is unrelated.
- No existing caller of `update_health` is removed. The new function is
  additive.
- Rollback: revert the commit; behavior reverts to "stale during T2".

## Verification

After merge + restart:
- Trigger a Layer 2 invocation (any consensus flip).
- Watch `data/health_state.json.last_heartbeat` advance once before the
  Layer 2 subprocess starts.
- During the L2 run (visible in `portfolio.agent.Agent T2 invoked` log),
  heartbeat age should stay <120s, not creep toward 600s.
- After the cycle ends, `update_health()` writes both heartbeat AND
  cycle_count/signals_ok normally.
