# Session Progress — Midfinance: Multi-Asset Subsystem Completion (2026-05-01)

**Session start:** 2026-05-01 ~01:00 UTC
**Status:** IN PROGRESS — 5/5 batches done, codex review + merge pending
**Branch:** `feat/midfinance-2026-05-01` (worktree at `/mnt/q/finance-analyzer.midfinance`)

## Goal

Bring oil to parity with the crypto+MSTR paper-mode swing pattern shipped
2026-04-30 (merge `ae4f8705`). Add operational plumbing (install scripts,
heartbeats, exit-code-11) for crypto+MSTR loops so they can run unattended
without flipping any live-trading switches.

## What was done

### Phase 0: Parallel exploration (3 subagents)
- Mapped crypto+MSTR live-wiring gap (loop is autonomous but inert — no
  scheduled task, no exit-code-11, no heartbeat).
- Mapped oil parity build (mirror crypto+MSTR pattern, NOT fishtrader
  greenfield which is blocked on 6 user questions).
- Inventoried Avanza oil instruments — 5 OLJA warrants found in
  `data/avanza_instruments_live.json` but missing barriers + stale prices.

### Phase 1: Plan
- `docs/PLAN.md` written + committed on main (commit `024476f6`).

### Phase 2: Worktree
- Created `feat/midfinance-2026-05-01` worktree.

### Batch A: Oil swing config + warrant scaffold (commit `94ad2401`)
- `data/oil_swing_config.py`, `data/oil_warrant_catalog.json`,
  `data/oil_warrant_refresh.py`, 2 test files. 37 tests pass.

### Batch B: Oil swing trader + loop (commit `30a360d9`)
- `data/oil_swing_trader.py`, `data/oil_loop.py`, 2 test files.
  KEY: oil_loop's `fetch_live_prices` uses `portfolio.price_source`,
  NOT direct Binance HTTP (CL=F is not a Binance spot symbol). 25 tests
  pass.

### Batch C: Scheduled-task plumbing + crypto/oil hardening (commit
TBD — merge candidate)
- `scripts/win/crypto-loop.bat`, `oil-loop.bat`, install-*.ps1 ×3.
- crypto_loop + oil_loop now return EXIT_LOCK_CONFLICT (11) on dup
  instance, write heartbeat each cycle, wire telegram notify with
  config-absent fallback. 9 hardening tests pass.

### Batch D: Dashboard + observability (commit `fdb589b9`)
- `/api/oil` endpoint, `scripts/oil_loop_scorecard.py` mirroring MSTR.
- MSTR scorecard now reports time-to-Phase-A. 12 tests pass.

### Batch E: Documentation (this commit)
- `docs/OIL_LOOP_NOTES.md`, CHANGELOG entry, this SESSION_PROGRESS.

## What's next (user-driven, NOT this PR)

- Run `scripts/win/install-crypto-loop-task.ps1` + `install-mstr-loop-task.ps1`
  + `install-oil-loop-task.ps1` to register the scheduled tasks.
- Run `Start-ScheduledTask -TaskName 'PF-CryptoLoop' / 'PF-MstrLoop' /
  'PF-OilLoop'` to start them in DRY_RUN/shadow mode.
- Optionally probe the oil warrant catalog with a live Avanza session
  via `.venv/Scripts/python.exe -u data/oil_loop.py --once --debug`.
- Watch the scorecards (`scripts/{mstr,oil}_loop_scorecard.py`) for the
  live-flip readiness gates to clear.

## Restart needed

Yes — `data/crypto_loop.py` was edited (exit-code-11 + heartbeat). If
PF-CryptoLoop is already running, restart it after merge:
`schtasks /run /tn PF-CryptoLoop`. Same for any oil_loop instance.
metals_loop and main loop are NOT touched by this PR.

## Out of scope (explicit)

- Fishtrader greenfield system stays blocked on 6 user design questions.
- New oil-specific signals (cross_asset / supply_demand / term_structure)
  deferred — `oil_precompute.py` already produces the deep context the
  swing trader needs.
- DRY_RUN → live flips remain user decisions in separate future commits.
