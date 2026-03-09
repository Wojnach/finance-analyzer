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

## GoldDigger Signal Upgrade

- Branch: `golddigger-signal-upgrade`
- Worktree: `Q:\finance-analyzer\.worktrees\golddigger-signal-upgrade`
- Scope: safer GoldDigger signal upgrades for intraday macro overlays and diagnostics

### What Changed

- Fixed dead GoldDigger volume confirmation by wiring volume fetches into live snapshots.
- Fixed ignored config knobs for `binance_gold_symbol` and `fred_series`.
- Added intraday proxy support for:
  - DXY via `yfinance`
  - 10Y yield proxy via `yfinance`
  - safe fallback to FRED or macro context
- Added event-risk blocking around high-impact macro events for metals.
- Added richer structured poll logging for proxy/event context.
- Added `--once` single-cycle dry-run path.
- Made GoldDigger config loading work from a clean git worktree by honoring:
  - `PORTFOLIO_CONFIG_PATH`
  - `GOLDDIGGER_CONFIG_PATH`
  - local worktree `config.json`
  - fallback shared checkout `config.json`

### Verification

- `Q:\finance-analyzer\.venv\Scripts\python.exe -m pytest -q tests\test_golddigger.py`
- `Q:\finance-analyzer\.venv\Scripts\python.exe -m portfolio.golddigger --once --dry-run --log-level INFO`

### Safe Dry-Run Example

```powershell
$env:GOLDDIGGER_CONFIG_PATH='Q:\path\to\temp-config.json'
Q:\finance-analyzer\.venv\Scripts\python.exe -m portfolio.golddigger --once --dry-run --log-level INFO
```

### Merge Notes

- New behavior is feature-flagged in `golddigger` config.
- `--once` suppresses Telegram notifications so it can be used from a worktree safely.
