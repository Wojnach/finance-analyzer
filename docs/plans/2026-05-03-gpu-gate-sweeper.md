# PLAN — gpu_gate stale-lock background sweeper (2026-05-03)

**Date:** 2026-05-03
**Branch:** `feat/gpu-gate-sweeper-2026-05-03`
**Worktree:** `/mnt/q/finance-analyzer/.worktrees/gpu-gate-sweeper-2026-05-03`

## Why

On 2026-05-02 02:14, chronos pid 13152 died holding the cross-process file lock
at `Q:/models/.gpu_lock`. Layer 1 wedged for ~25 hours because nothing tried
to acquire while the lock was leaked, so `_is_stale()` and `_pid_alive()` were
never checked. The wedge persisted until an nvlddmkm TDR took the system
down completely; recovery required manual reboot. Same pattern reproduced on
2026-05-03 18:53 (kronos pid 43428) and again 2026-05-03 20:35 (kronos pid
2912). See `docs/SESSION_PROGRESS.md` for the full forensic timeline.

The bug is structural: `gpu_gate._is_stale()` is only called from
`gpu_gate()` line 141 inside the acquire retry loop. If no one tries to
acquire while the lock is leaked, the lock stays leaked indefinitely.

## What this PR does

Add a background daemon thread (`_lock_sweeper`) inside
`portfolio/gpu_gate.py` that wakes every 30 seconds and runs the same
"stale + dead pid" check the reactive path runs at line 141. If both are
true, it deletes the lock file and emits a warning identical in format to
the reactive path so existing log-grep tools keep working.

## Files changed

| File | Change |
|---|---|
| `portfolio/gpu_gate.py` | Add `_start_sweeper()` (idempotent), `_sweeper_loop()` (the daemon body), and a module-level singleton guard. Exported lock-file inspection helpers stay unchanged. |
| `tests/test_gpu_gate.py` | Add `TestSweeper` class with 3 tests: (1) stale-lock-with-dead-pid is reaped within 1 sec of sweeper tick; (2) live-pid lock is NOT reaped; (3) fresh lock (less than `_STALE_SECONDS`) is NOT reaped. All tests monkeypatch `_GPU_LOCK_FILE` to `tmp_path / ".gpu_lock"` for xdist safety. |

## Implementation specifics

### `_lock_sweeper` daemon

- Infinite `while True: time.sleep(30); _try_break_stale_lock()` loop.
- `_try_break_stale_lock()` is a new helper that mirrors the existing logic at
  line 141: read lock, check stale, check pid alive, log + unlink if both
  conditions met. The reactive path at line 141 will be refactored to call
  the same helper to avoid drift.
- Daemon thread (`daemon=True`) so it dies with the process; no shutdown
  hook needed.
- Idempotent registration: `_SWEEPER_STARTED = False` module-level flag,
  protected by `_THREAD_LOCK` to handle the rare case where two threads
  hit `_start_sweeper` simultaneously during cold start.

### Where `_start_sweeper` is invoked

Lazily, on first call to `gpu_gate()`. Avoids:
- Side effects at import time (which would also fire in subprocess workers
  that import `portfolio.gpu_gate` indirectly).
- The need for an explicit init in `portfolio/main.py`.

The cost: the very first cycle pays a tiny one-time cost to spawn the
thread. Negligible compared to nvidia-smi calls.

### Test isolation strategy

xdist safety: each test must monkeypatch `_GPU_LOCK_FILE` to point under
`tmp_path` BEFORE writing any lock content. The sweeper itself is too
slow to run in tests (30 s tick) — so the test calls
`_try_break_stale_lock()` directly instead of waiting for the daemon.

The daemon-spawn behaviour is verified by a fourth test:
`test_sweeper_starts_only_once`.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Sweeper false-positives (breaks a real long-running lock) | Sweeper uses the same `_is_stale() AND not _pid_alive()` predicate as the existing reactive path. If that predicate is correct (and it has been in production since BUG-182 fix), the new sweeper inherits its correctness. No new false-positive surface. |
| Sweeper spawned in subprocess workers | Idempotent guard + lazy spawn means subprocesses only spawn a sweeper if they themselves call `gpu_gate()`. The kronos and chronos infer scripts use a different module (`Q:/models/gpu_lock.py`), so they will not trigger this path. |
| Daemon thread reaper mid-acquire | `_try_break_stale_lock` only acts when the predicate fires. The acquire path uses `O_CREAT \| O_EXCL` (atomic), so a concurrent unlink + retry-create is safe. |
| Test flakiness from real-time sleeps | Tests bypass the 30 s loop and call `_try_break_stale_lock()` directly. Only `test_sweeper_starts_only_once` actually starts the daemon, and it just checks the flag, not loop behaviour. |

## Out of scope (intentionally)

- VRAM-aware scheduling between kronos / chronos / llama-server. Real
  problem, but would require a design discussion. No code change here.
- Investigating *why* kronos / chronos processes die in the first place.
  Needs `faulthandler.enable()` + per-process crash log to capture the
  next occurrence. Separate work.
- Thermal-aware contracts tied to `data/hw_monitor.json`. Discussed in
  the same session but is its own change.
- Refactoring the duplicated `Q:/models/gpu_lock.py` (used by inference
  subprocesses) to share code with `portfolio/gpu_gate.py`. They use the
  same lock file but different code paths; not a deduplication target
  for this PR.

## Verification

1. Focused: `pytest tests/test_gpu_gate.py -v`
2. Full: `pytest tests/ -n auto`
3. Codex adversarial review on the branch.
4. Manual: read the sweeper code with the lens of "what could go wrong if
   the sweeper spawns inside the metals loop, the crypto loop, the oil
   loop, and the main loop simultaneously."

## Rollback

`git revert <commit>` followed by Windows-side `cmd.exe /c "cd /d Q:\finance-analyzer && git push"`.
