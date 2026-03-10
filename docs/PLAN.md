# Metals State Hardening Plan

Updated: 2026-03-10
Worktree: `Q:\wt\metals-state-store`
Branch: `metals-state-store`

## Goal

Harden the metals subsystem state files without disturbing the live environment.
The immediate target is the shared state used by `data/metals_loop.py`,
`data/metals_risk.py`, supporting scripts, and the Layer 2 trade-queue prompt.

## Findings

- The metals loop still persists critical shared state via raw `open(..., "w")`
  JSON overwrites:
  - `data/metals_positions_state.json`
  - `data/metals_trade_queue.json`
  - `data/metals_stop_orders.json`
  - `data/metals_spike_state.json`
- `data/metals_risk.py` persists `data/metals_guard_state.json` the same way and
  silently resets on read failure.
- The Layer 2 prompt in `data/metals_agent_prompt.txt` explicitly instructs the
  agent to overwrite `data/metals_trade_queue.json` directly.
- Several consumers outside the main loop still read the JSON files directly,
  including `data/silver_monitor.py` and docs such as `docs/STOP_LOSS_SETUP.md`.
- The repo already has two durability primitives that should be reused first:
  - `portfolio.file_utils.atomic_write_json`
  - `portfolio.signal_db.SignalDB` (SQLite WAL example)

## Decision

Use the smallest safe change first:

1. Keep the JSON file contract for now so existing readers and the dashboard are
   not broken.
2. Replace raw overwrites with atomic writes using shared file utilities.
3. Replace silent read-reset behavior with explicit logging and safe defaults.
4. Update the Layer 2 queue-writing guidance to use an atomic write pattern.

SQLite/WAL remains a follow-up option, but it is not the first batch because it
would require a wider contract migration for scripts, docs, and operator habits.

## Risks

- If malformed JSON is currently being tolerated silently, making failures more
  visible could surface operator issues that were previously hidden.
- Updating the prompt contract may require small test adjustments where file I/O
  is mocked broadly.
- The metals loop is a live-trading path, so changes must stay narrowly scoped to
  persistence helpers and must not alter trading decisions or thresholds.

## Batch Plan

### Batch 1: Tests First

- Add regression tests for:
  - atomic round-trips for positions, trade queue, stop orders, and spike state
  - malformed/corrupt JSON falling back with explicit logging
  - guard-state reads no longer failing silently

Files expected:
- `tests/test_metals_loop_functions.py`
- `tests/test_metals_risk.py`

### Batch 2: Shared State Hardening

- Introduce small shared helpers for metals JSON state reads/writes, or reuse
  `portfolio.file_utils` directly where that keeps the diff smaller.
- Update:
  - `data/metals_loop.py`
  - `data/metals_risk.py`

Goals:
- all shared JSON writes are atomic
- all read failures are logged explicitly
- no behavior change to trading logic

### Batch 3: Prompt and Docs

- Update `data/metals_agent_prompt.txt` so the trade queue is written atomically.
- Update affected docs to match the new state-handling contract and note the
  rationale.

Files expected:
- `data/metals_agent_prompt.txt`
- `docs/SYSTEM_OVERVIEW.md`
- `docs/STOP_LOSS_SETUP.md`

## Verification

Targeted first:

- `tests/test_metals_loop_functions.py`
- `tests/test_metals_risk.py`
- `tests/test_metals_loop_autonomous.py`
- `tests/test_unified_loop.py`

Broader follow-up if the targeted slice stays green:

- `pytest -n auto`
- `ruff check data/metals_loop.py data/metals_risk.py portfolio/file_utils.py`

## Rollback

1. Revert the batch commit in the worktree branch.
2. Leave live processes untouched until the branch is reviewed and explicitly
   merged.
3. If prompt/docs changes prove noisy, revert those separately from the runtime
   persistence helper changes.
