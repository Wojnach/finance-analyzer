# Avanza Playwright Resilience Plan

Owner: Ralph-loop driven work, started 2026-04-13.

## Context

`/mnt/q/finance-analyzer/` has 4 long-running Python processes talking to Avanza via
Playwright:

- `data/metals_loop.py`
- `portfolio/main.py --loop`
- `portfolio/golddigger/` (module, via `PF-GoldDigger` scheduled task)
- `portfolio/fin_snipe_manager.py`

Auth is shared via `data/avanza_storage_state.json`, seeded once by
`scripts/avanza_login.py` (BankID flow). Every loop is supposed to launch its own
Chromium browser and load those cookies — no live browser handoff.

## Failure Modes Observed 2026-04-13

1. `metals_loop` + `golddigger` each own their own browser but have NO auto-recovery.
   When the browser context dies (OS sleep, memory pressure, cookie expiry), every
   `page.evaluate()` throws `playwright._impl._errors.TargetClosedError`. The loop
   keeps ticking, emitting errors for days — silently not trading.
2. `portfolio/avanza_session.py` uses a singleton Playwright context guarded by
   `threading.RLock _pw_lock`, shared by `main.py` + `fin_snipe_manager`. When that
   singleton dies, both consumers fail in lockstep.
3. Re-authing Avanza (fresh `avanza_storage_state.json` on disk) does NOT rescue a
   running Python process — the in-memory browser handle is still dead. Only
   process restart picks up the new state.

Net: lost 3 days of metals trading (2026-04-10 → 2026-04-13). Discovered when user
asked "why isn't metals loop trading".

## Tasks (already in TaskList)

### Task 1 — Auto-recovery for metals_loop
In `data/metals_loop.py` + `data/metals_avanza_helpers.py`, wrap these call sites
with retry-on-TargetClosedError:
- `metals_avanza_helpers.py:34` `fetch_price`
- `metals_avanza_helpers.py:129` `fetch_account_cash`
- `metals_loop.py:1640` `detect_holdings`

On hit: teardown dead browser/context, relaunch Chromium, reload
`data/avanza_storage_state.json` into fresh context, retry call once. Log recovery.

### Task 2 — Add TargetClosedError auto-recovery to avanza_session.py (keep singleton)
**Revised 2026-04-13 11:59 after reading avanza_session.py line-by-line.**
Original plan was to remove the `_pw_lock` RLock + singleton. That is WRONG — the
lock prevents BUG-129: Playwright's sync_api is not thread-safe and the
`main.py` 8-worker thread pool + metals 10s fast-tick concurrently corrupt
`ctx.request.*` calls (e.g. a POST's CONFIRM response being stolen by another
thread's GET). The lock is intra-process, not cross-process — it does NOT
cause the metals_loop → main.py lockstep failure.

The real fix is surgical: `api_get`, `api_post`, and `api_delete` already call
`close_playwright()` on 401/403. Add the same teardown on `TargetClosedError`
(and `Error` with "browser has been closed" in message), then retry once.
Optional: add a shared `_api_call_with_recovery()` helper to avoid copy-paste
across the 3 public API functions.

### Task 3 — File-based avanza_order.lock
Add advisory file lock at `data/avanza_order.lock`, acquired only around actual
order placement (`place_order`, `place_stop_loss`, `cancel_order`). NOT around
reads. Prevents two loops from placing overlapping orders across
metals + golddigger + fin_snipe. 2s fail-fast. Use `fcntl.flock` equiv via
`msvcrt.locking` on Windows, or `portalocker` if already in requirements.

### Task 4 — Test + merge + restart
Tests:
1. Metals auto-recovery — simulate TargetClosedError → browser relaunched → call succeeds
2. avanza_session per-consumer independence — kill one consumer's browser, other still works
3. avanza_order.lock contention — two processes contend, only one places order, other waits or fails-fast

Run `.venv/Scripts/python.exe -m pytest tests/ -n auto`. Commit on a worktree,
merge into `main`, push, restart `PF-MetalsLoop` + `PF-DataLoop` + `PF-GoldDigger`
via schtasks.

## Constraints

- Python: always `.venv/Scripts/python.exe` with forward slashes.
- Git: use worktrees, not main directly. SSH keys on Windows, so
  `cmd.exe /c git push` from WSL (or `! git push` from user).
- File I/O: `file_utils.atomic_write_json`, `load_json`, `atomic_append_jsonl`.
- Reuse: check `portfolio/avanza/` unified package first (TOTP auth + 163 tests
  per user memory) before writing new code.
- NEVER commit `config.json` — symlink to external, API keys.
- Trade guards: do not break stop-loss path — must use `/_api/trading/stoploss/new`
  (per user memory note about Mar 3 incident).

## Completion Criteria

All 4 tasks complete AND:
- `pytest tests/ -n auto` green
- All 3 loops (`PF-MetalsLoop`, `PF-DataLoop`, `PF-GoldDigger`) restart cleanly
  via schtasks
- `tail data/metals_loop_out.txt` shows 2 consecutive cycles with zero
  `TargetClosedError`
- Cash sync succeeds on first cycle post-restart (verified via
  `[SWING] Cash synced: N SEK` log line)
- Changes merged to `main`, pushed to origin

## Progress Log

- 2026-04-13 11:57 CET — plan created, tasks in TaskList, metals_loop already
  restarted manually (working with cash=6036 SEK), PF-DataLoop also restarted,
  architectural fix still pending.
