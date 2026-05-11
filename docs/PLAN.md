# Loop audit — fix plan (2026-05-11)

## Context

User asked "how many loops are running, any duplicates, why are some silent."
Audit found 8 healthy loops, no real duplicates (Windows venv-launcher
shim pattern explains 2 PIDs per loop). Investigation surfaced three
follow-up issues to fix.

## Issues

### 1. Log file growth — `data/loop_out.txt` = 105MB, no rotation
- `pf-loop.bat` uses `>> data\loop_out.txt 2>&1` (append, unbounded).
- `golddigger.bat` uses `>>` (golddigger_out.txt = 18MB, golddigger_log.jsonl = 84MB).
- `mstr-loop.bat` uses `>>` (small but unbounded).
- `metals/crypto/oil-loop.bat` use `>` (truncate per restart — fine).
- `portfolio/log_rotation.py` exists with policy `loop_out.txt 5MB/3 keep`
  but **nobody invokes it**. No scheduled task, no in-loop trigger.

### 2. `contract_violation` dedup race — 7-8 fires in 410ms
- File: `portfolio/loop_contract.py` lines 410-495.
- Dedup keyed on `current_trigger_iso == last_fired_trigger_ts` from
  `contract_state.json`. Should suppress repeat fires for same trigger.
- Empirically: 7 entries in 410ms for same trigger on 2026-05-10
  16:47:48. Marker write timing or state-file race lets duplicates land.
- Not fixing root cause (low value — investigation took longer than
  fix). Adding **wall-clock cooldown floor** as belt-and-braces: if
  `now - last_violation_wall_ts < 30s` and same invariant, suppress.

### 3. `accuracy_degradation` ESCALATED — false alarm replay
- Calendar signal already disabled via commit b56f653c (sell_in_may to
  HOLD May-Oct). Detector keeps re-firing on rolling-7d window until
  bad samples flush.
- Resolve by appending bulk `resolution` entries to
  `critical_errors.jsonl` so check_critical_errors stops surfacing them.

### 4. GoldDigger `last_poll_time` never persisted (minor)
- `portfolio/golddigger/bot.py:184` mutates in-memory only.
- `state.save()` only called on entry (332) and exit (392).
- Not breaking anything — state file is for position/equity. Live
  poll telemetry lives in `golddigger_log.jsonl`. Leaving as-is,
  documenting only.

## Execution plan

### Batch 1 — log rotation wiring
- Add `PF-LogRotate` scheduled task: hourly, runs
  `.venv/Scripts/python.exe -m portfolio.log_rotation`. Install script
  in `scripts/win/install-log-rotate-task.ps1`.
- Tighten `loop_out.txt` policy: 5MB stays (already aggressive). Keep 5 backups.
- Add `golddigger_out.txt` rotation policy (currently missing —
  golddigger_log.jsonl is covered, plain stdout isn't).
- Add `mstr_loop_out.txt` rotation policy.
- Files: `portfolio/log_rotation.py`, `scripts/win/install-log-rotate-task.ps1` (new).

### Batch 2 — contract dedup wall-clock cooldown
- In `check_layer2_journal_activity`: read
  `contract_state["layer2_last_violation_wall_ts"]` (UNIX ts). If now
  - last_wall < 30s AND same trigger, suppress.
- Write `layer2_last_violation_wall_ts = time.time()` alongside the
  existing trigger_ts marker.
- Backward compatible — missing key behaves like fresh state.
- File: `portfolio/loop_contract.py`.

### Batch 3 — bulk-resolve old critical errors
- Append `resolution` entries to `data/critical_errors.jsonl` for the
  43 unresolved entries from accuracy_degradation (calendar) and
  contract_violation (layer2_journal_activity) prior to 2026-05-11.
- Script: `scripts/resolve_loop_audit_errors.py` (one-shot).
- File: `data/critical_errors.jsonl` (append).

### Batch 4 — tests + restart
- Update `tests/test_loop_contract.py` for new cooldown branch.
- Run `pytest tests/test_loop_contract.py tests/test_log_rotation.py -v`.
- Full suite: `pytest tests/ -n auto`.
- After merge: restart PF-DataLoop + PF-MetalsLoop.

## What could break

- Log rotation truncates `loop_out.txt` while loop holds the file
  handle. `open(filepath, "w")` zeros the file content. cmd.exe `>>`
  re-opens with FILE_APPEND_DATA which always seeks to EOF (now 0) on
  each write — works on Windows.
- Contract cooldown could suppress a genuine *second* invocation
  failure within 30s. Acceptable: the underlying trigger is still
  alerted; the second fire would be redundant noise.
- Bulk resolution doesn't fix the root cause (auth_error chain). If
  Layer 2 has new auth failures they'll surface fresh.

## Out of scope

- GoldDigger last_poll_time persistence (low value, no breakage).
- 105MB existing log truncation while loop runs (will rotate next
  cycle of PF-LogRotate; if user wants immediate cleanup, restart loop).
- Layer 2 auth chain root cause (separate investigation).
