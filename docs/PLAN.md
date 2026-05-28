# PLAN — Fix inter-cycle heartbeat staleness (false "loop down")

Branch: `fix/heartbeat-intercycle-staleness`
Date: 2026-05-29

## Problem

The Layer 1 main loop refreshes its liveness heartbeat **only once per cycle**,
but the cycle interval (600s) is **2× the staleness threshold (300s)**, so the
heartbeat ages past "stale" for roughly half of every cycle even when the loop
is perfectly healthy.

### Evidence
- `portfolio/market_timing.py:20-22`: `INTERVAL_MARKET_OPEN/CLOSED/WEEKEND = 600`
  (bumped from 60s "pre-daemon era").
- `portfolio/main.py` loop body (1362-1404): each cycle does
  `_sleep_for_next_cycle(last_cycle_started, sleep_interval)` → a single
  `time.sleep(remaining)` with **no heartbeat**, then `run()` (whose end calls
  `update_health()` → `last_heartbeat`), then `atomic_write_text(heartbeat.txt)`.
  So both heartbeats refresh ~once / 600s and age 0→600s during the sleep.
- `portfolio/health.py:152` `check_staleness(max_age_seconds=300)` → consumed by
  `get_health_summary()` → dashboard `/api/health` status; and `digest.py:241-247`
  (`heartbeat_age < 300` = ok, else "Xm stale").
- `portfolio/main.py:1301` startup crash-detector: heartbeat.txt age > 300s →
  logs "previous loop likely crashed" + Telegram. With 600s cycles, every clean
  restart fires this false alarm (heartbeat.txt up to 600s old at restart).

### Scope / non-issues (verified during explore)
- `loop_health` watchdog monitors only the 60s swing loops
  (crypto/oil/mstr/metals/golddigger via `*_loop.heartbeat`), **NOT** the main
  loop — so there is no false restart of the main loop, only false
  dashboard/digest staleness + a false restart-alert Telegram.
- This is a latent regression from the 60s→600s interval bump, independent of
  the 2026-05-28 bug-hunt fixes (`health.py` / sleep path untouched there).
- The cold-start first-cycle delay observed during the restart (acc_load
  56-66s/ticker on a cold accuracy cache → ~125s signal phase) is a *separate*,
  transient symptom that also delayed the first post-restart `update_health`.
  Not fixed here (transient; self-resolves once the accuracy cache warms).

## Fix

Keep both heartbeats fresh **during the inter-cycle sleep** by beating every
~60s (5× headroom under the 300s gate), reusing the existing `health.heartbeat()`
(touches only `last_heartbeat`, leaves cycle stats intact).

### Change 1 — `portfolio/main.py::_sleep_for_next_cycle`
Add an optional `beat` callback + `beat_interval_s=60.0`. When `beat` is given,
chunk the remaining sleep into <=`beat_interval_s` slices and call `beat()` after
each slice. Deadline-anchored (preserves cadence). `beat()` exceptions are
caught + logged (never abort the loop). When `beat is None`, behaviour is
unchanged (single `time.sleep`) — keeps the helper generic and existing callers
unaffected.

### Change 2 — `portfolio/main.py::loop`
Define a loop-local `_cycle_beat()` that refreshes BOTH heartbeats:
`health.heartbeat()` (health_state.json `last_heartbeat`) and
`atomic_write_text(DATA_DIR/"heartbeat.txt", now)`. Pass it to
`_sleep_for_next_cycle(...)` at line ~1372. Failure-tolerant.

No change to `health.py`, `market_timing.py`, the interval values, or the
staleness thresholds. The 300s gate is correct; the cadence was the bug.

## Why not alternatives
- *Lower the interval back to 60s*: rejected — the 600s interval is intentional
  (token-budget / load reduction per the daemon-era change); the heartbeat
  cadence, not the work cadence, is what must stay < 300s.
- *Raise the 300s thresholds to >600s*: rejected — would blind the dashboard to
  a genuinely hung loop for 10+ min; the gate should stay tight.
- *Wrap the sleep in `heartbeat_keepalive()`*: viable but only beats
  health_state.json, not heartbeat.txt, and spins a daemon thread per cycle.
  The callback approach beats both on the main thread (no extra thread, no
  `_health_lock` cross-thread contention) and is trivially testable.

## Execution order
1. Write this plan, commit. (this step)
2. Premortem (fresh agent) -> append narratives -> commit.
3. Implement Change 1 + 2.
4. Tests: new `tests/test_intercycle_heartbeat.py` — assert
   `_sleep_for_next_cycle` beats ~N times under a fake clock and respects the
   deadline; assert `beat=None` path unchanged. Commit.
5. Adversarial review subagent on the branch; fix P1/P2.
6. `pytest -n auto` targeted (`test_loop_*`, `test_health*`, `test_heartbeat*`,
   `test_dashboard_system_status`) + the new test.
7. Merge to main, push (user runs `! git push`), restart loops.

## Premortem
_(populated by the premortem agent in the next step)_
