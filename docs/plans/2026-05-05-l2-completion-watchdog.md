# PLAN — Layer 2 completion-detection watchdog

**Date:** 2026-05-05
**Branch:** `fix/t1-timeout-drift-2026-05-05`
**Goal:** Stop T1 / T2 / T3 invocation completions from being detected up to 8 minutes late.

## Context

`docs/plans/2026-05-05-dashboard-noise-followups.md` item (3) flagged that
T1 invocations across 2026-05-04 routinely report `duration_s` 400-540s,
even though `TIER_CONFIG[1] = {"timeout": 120}`. The hotfix
`0834bb71` widened `LAYER2_JOURNAL_GRACE_S_BY_TIER` to 12 min to mute
the false-positive `layer2_journal_activity` alerts, but the underlying
detection delay remained.

I confirmed hypothesis (3a) is the actual cause:

- `portfolio/main.py:431` calls `check_agent_completion()` exactly once
  per `run()` cycle, at the start.
- `portfolio/agent_invocation.py:989-1035` is where `subprocess.poll()`
  is read and the timeout-kill is enforced.
- When `run()` itself bloats (`cycle_duration` violations of 333-918s
  observed 2026-05-01 / 02 / 03 / 04), `check_agent_completion` only
  fires at that bloated cadence. The subprocess finishes inside its real
  budget but `duration_s` records `next_run_call - invoke`, i.e. up to
  the cycle duration.
- Sample of `data/invocations.jsonl` 2026-05-04 (UTC):

  ```
  12:07 invoked T1 → 12:15 success duration=492s   (real T1 budget: 120s)
  13:09 success T1 duration=646s
  14:29 success T1 duration=520s
  15:19 success T1 duration=254s
  16:29 success T1 duration=518s
  ```

  Completion timestamps cluster at HH:X9:16 — the cadence of `run()`
  finishing under bloat, not the cadence of the subprocess finishing.

This isn't theoretical: every `success` row in this window over-reports
duration by 4-6 minutes, which (a) inflates `get_completion_stats`'s
notion of agent latency, (b) delays the kill of a hung agent past its
real budget, and (c) was the root cause of the noise the 0834bb71
hotfix had to mask with a wider grace window.

## Fix

Add a daemon watchdog thread inside `portfolio/agent_invocation.py`
that calls `check_agent_completion()` every **30 s**, independent of
`run()`'s cadence. The watchdog:

- Starts lazily — first invocation of `try_invoke_agent()` ensures it's
  alive (idempotent: re-spawns only if previous thread died, which only
  happens on uncaught exceptions inside the watchdog).
- Holds an `_completion_lock` while calling `check_agent_completion()`
  so the main loop's call at `main.py:431` and the watchdog's tick
  cannot race on `_agent_proc` / `_agent_start` state.
- Is a daemon thread (process-exits cleanly without joining).
- Checks every 30 s — small enough that T1's 120 s timeout fires
  within at most 30 s of the real budget; large enough that the lock
  is essentially uncontended.

Why 30 s? `_kill_overrun_agent` writes a critical-error journal entry;
we don't want to write 24 of those in the 120 s before the timeout
actually fires. 30 s gives at most 4 ticks before the budget, only the
last of which exceeds it.

## Files to change

| File | Change |
|---|---|
| `portfolio/agent_invocation.py` | Add `_completion_lock`, `_watchdog_thread`, `_completion_watchdog`, `_ensure_completion_watchdog`. Take the lock inside `check_agent_completion()`. Call `_ensure_completion_watchdog()` from `try_invoke_agent()` after a successful spawn. |
| `portfolio/main.py` | Documentation comment at the line 431 call site noting the lock-protected path; no behaviour change. |
| `tests/test_agent_invocation_watchdog.py` (new) | Unit tests: (a) watchdog thread is started by first invoke, (b) watchdog tick is a no-op when no agent is running, (c) calling `check_agent_completion` from main + watchdog concurrently does not double-log to `invocations.jsonl`, (d) timeout kill fires when subprocess hangs past `_agent_timeout`. |

## Risks

- **Daemon thread leak.** Mitigated by `daemon=True` (joins to interpreter exit) and the lazy-start guard.
- **Lock contention with main thread.** Negligible — main calls `check_agent_completion` once per cycle, watchdog every 30 s. Both are sub-millisecond when no completion happened.
- **Test isolation.** Tests stop the watchdog after each test (`_watchdog_stop.set()`) so xdist parallel runs don't interfere.

## Verification

1. Unit tests: `python -m pytest tests/test_agent_invocation_watchdog.py -v`
2. Existing tests: `python -m pytest tests/test_agent_invocation.py tests/test_layer2_journal_contract.py -v`
3. Adversarial review: Codex `xhigh` effort, scope=branch
4. After merge: restart `PF-DataLoop` and observe next 24 h of `data/invocations.jsonl` — `duration_s` for T1 should drop from ~480s to ~120s when the subprocess actually completes within budget.

## Out of scope

- Fixing the `cycle_duration` root cause (separate bug).
- Reverting the `LAYER2_JOURNAL_GRACE_S_BY_TIER[1]` widening from 0834bb71 — keep as belt-and-suspenders until a week of post-watchdog data shows T1 reliably finishes < 180s.
