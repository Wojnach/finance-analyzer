# Worktree Notes

Branch: `metals-state-store`
Path: `Q:\wt\metals-state-store`

## What Changed

- Hardened metals shared state persistence in `data/metals_loop.py` and
  `data/metals_risk.py`.
- Replaced raw JSON overwrites for shared metals state with atomic writes.
- Added explicit corrupt-file logging for guard, spike, positions, stop-order,
  and trade-queue state.
- Fixed `data/metals_loop.py` path handling so imports and relative paths bind
  to the current checkout instead of the hardcoded live repo path.
- Updated the Layer 2 prompt and related docs to match the safer queue/state
  contract.

## Key Commands

Run targeted verification:

```powershell
Q:\finance-analyzer\.venv\Scripts\python.exe -m pytest -q `
  tests\test_metals_loop_autonomous.py `
  tests\test_unified_loop.py `
  tests\test_metals_loop_functions.py `
  tests\test_metals_risk.py
```

Run lint if available:

```powershell
Q:\finance-analyzer\.venv\Scripts\python.exe -m ruff check `
  data\metals_loop.py data\metals_risk.py portfolio\file_utils.py
```

## Commits In This Worktree

- `442581f` `docs: add metals state hardening plan`
- `0d38be1` `test: cover metals state persistence hardening`
- `a1ef4d9` `fix: harden metals state persistence`
- `03b511d` `fix: make metals loop worktree-safe`

## Verification Snapshot

- Targeted metals slice:
  - `134 passed`
- Full suite:
  - `3901 passed`
  - `18 failed` (pre-existing repo baseline outside this worktree scope)
