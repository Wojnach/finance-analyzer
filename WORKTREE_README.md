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

- Main checkout was left untouched.
- New behavior is feature-flagged in `golddigger` config.
- `--once` suppresses Telegram notifications so it can be used from a worktree safely.
