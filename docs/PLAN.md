# PLAN — Multi-Asset Subsystem Completion (midfinance)

**Date:** 2026-05-01
**Branch:** `feat/midfinance-2026-05-01`
**Goal:** Bring oil to parity with crypto+MSTR (paper-mode swing subsystem) and complete the operational plumbing (scheduled tasks, heartbeats, install scripts) for crypto and MSTR loops so they can run unattended in DRY_RUN/shadow without flipping any live trading switches.

---

## Context (where we landed)

| Asset | Status as of 2026-05-01 |
|---|---|
| **Gold (XAU)** | Already production-traded by `data/metals_loop.py` + 64 XAU warrants in `data/metals_warrant_catalog.json`. **No work needed.** |
| **BTC + ETH** | Subsystem code shipped in merge `ae4f8705` (`feat/crypto-mstr-swing`). `data/crypto_loop.py` is autonomous + has CLI args (`--loop`/`--once`/`--report`). DRY_RUN=True default. **Missing**: scheduled-task wrapper (`scripts/win/crypto-loop.bat`), install script, exit-code-11 on lock conflict, heartbeat, telegram notify wiring. |
| **MSTR** | `portfolio/mstr_loop/` package complete, runs in shadow phase. `scripts/win/mstr-loop.bat` exists. **Missing**: install script (`install-mstr-loop-task.ps1`), scorecard time-to-90-days metric. |
| **Oil** | `portfolio/oil_precompute.py` (1085 lines, runs every 2h, comprehensive WTI+Brent context). 5 OLJA warrants found in `data/avanza_instruments_live.json` (MINI L/S OLJA + BEAR OLJAB X3) but missing barriers + stale prices. **No swing trader / no loop / no warrant catalog / no dashboard endpoint.** |

---

## What this PR does

### Oil — net new (mirrors crypto+MSTR pattern, NOT fishtrader greenfield)

The fishtrader plan (`docs/plans/2026-04-30-fishtrader.md`) is awaiting user input on 6 design questions. We don't gate on that. Instead we ship oil parity with the proven crypto+MSTR pattern:

| File | Purpose |
|---|---|
| `data/oil_swing_config.py` | Tunables, DRY_RUN=True default, mirrors `crypto_swing_config.py` shape. WTI-only universe in v1 (Brent deferred). |
| `data/oil_warrant_catalog.json` | Empty scaffold `{"refreshed_ts": null, "ttl_hours": 6, "warrants": {}}` — populated on first refresh. |
| `data/oil_warrant_refresh.py` | Avanza search for "BULL OLJA"/"BEAR OLJA"/"MINI L OLJA"/"MINI S OLJA"/"OIL TRACKER"/"BRENT" prefixes. Mirrors `crypto_warrant_refresh.py` API verbatim. |
| `data/oil_swing_trader.py` | Paper-mode trader. Reuses generic logic by parameterizing `CryptoSwingTrader` if practical, or copy-edit if structurally divergent. |
| `data/oil_loop.py` | 60s loop with embedded fast-tick monitor. Mirrors `crypto_loop.py` CLI exactly. Singleton lock at `data/oil_loop.lock`. Returns exit code 11 on lock conflict. |

### Crypto + MSTR — operational plumbing

| File | Purpose |
|---|---|
| `scripts/win/crypto-loop.bat` | Wrapper mirroring `metals-loop.bat`: cd, redirect logs to `data/crypto_loop_out.txt`, restart on crash, abort on exit code 11. |
| `scripts/win/oil-loop.bat` | Same pattern for oil. |
| `scripts/win/install-crypto-loop-task.ps1` | Registers `PF-CryptoLoop` scheduled task. **Does not start it** — user runs `Start-ScheduledTask` when ready. |
| `scripts/win/install-mstr-loop-task.ps1` | Registers `PF-MstrLoop` scheduled task (shadow phase by default — sets `MSTR_LOOP_PHASE=shadow` env var). |
| `scripts/win/install-oil-loop-task.ps1` | Registers `PF-OilLoop` scheduled task. |

### Loop hardening (touches existing crypto_loop.py)

- Return exit code 11 from `main()` on singleton lock conflict (mirrors metals).
- Write `data/crypto_loop.heartbeat` JSON on each successful cycle.
- Wire telegram `notify=` callback in `run_loop()` (loads `portfolio.telegram_notifications.send_telegram` with config). Falls back to no-op if config unavailable so unit tests stay isolated.
- Same hardening to oil_loop.py from day one.

### Dashboard

| File | Change |
|---|---|
| `dashboard/app.py` | Add `/api/oil` endpoint mirroring `/api/crypto`. Returns oil_swing_state + oil_value_history + oil_deep_context. |

### Observability

| File | Change |
|---|---|
| `scripts/mstr_loop_scorecard.py` | Add `time_to_phase_a` field: days of shadow data so far, days remaining to 90-day threshold. |
| `scripts/oil_loop_scorecard.py` | New: same shape as MSTR scorecard, for oil swing decisions. |

### Tests

| File | Coverage |
|---|---|
| `tests/test_oil_swing_config.py` | Constants exist, types correct, DRY_RUN=True. |
| `tests/test_oil_warrant_refresh.py` | Catalog round-trip, fallback to scaffold, TTL handling. |
| `tests/test_oil_swing_trader.py` | Entry gates, exit gates, position state. |
| `tests/test_oil_loop.py` | Singleton lock, exit code 11, run_one_cycle with mocked prices/signals. |
| `tests/test_crypto_loop_hardening.py` | Exit code 11, heartbeat written, notify called. |
| `tests/test_dashboard_oil.py` | `/api/oil` endpoint shape. |

### Documentation

| File | Change |
|---|---|
| `docs/OIL_LOOP_NOTES.md` | New: operator runbook mirroring `MSTR_LOOP_NOTES.md`. |
| `docs/CHANGELOG.md` | Entry for this multi-asset completion. |
| `docs/SESSION_PROGRESS.md` | Session record for next-session pickup. |
| `CLAUDE.md` | Add `/api/oil` to dashboard endpoint list, note the three new install scripts. |

---

## What this PR does NOT do

- **Does not flip DRY_RUN to False anywhere.** All new and existing loops stay in paper/shadow.
- **Does not auto-register scheduled tasks.** Provides `install-*.ps1` scripts that the user runs.
- **Does not start any new loops.** User runs `Start-ScheduledTask` when ready.
- **Does not modify signal weights or thresholds in production config.**
- **Does not touch live metals_loop or main.py loop behavior.** Oil is a separate process.
- **Does not implement the fishtrader greenfield system** — that plan stays awaiting user input.
- **Does not add new oil signals** (oil_cross_asset, oil_supply_demand, oil_term_structure). The existing `oil_precompute.py` already produces the deep context. Adding new signal modules is feature creep — out of scope for parity.

---

## Risks

1. **Avanza session contention.** Three loops (metals, crypto, oil) plus mstr_loop could compete for BankID auth. Mitigation: each loop holds its own singleton lock; auth is shared via `portfolio/avanza_session.py` (thread-safe singleton). DRY_RUN=True for crypto/oil means no actual order placement during the Avanza session window.

2. **Empty oil warrant catalog at startup.** On first `oil_loop --once` without a Playwright session, the catalog will be `{}` and the trader will skip all entries. This is the documented behavior of crypto_loop's first run. Mitigation: oil_loop logs an explicit "no warrants — skipping entries" line each cycle.

3. **DRY_RUN doesn't completely sandbox external state.** The trader writes to `data/oil_swing_decisions.jsonl`, `data/oil_swing_trades.jsonl`, etc. — those are paper-mode logs, no Avanza orders. Mitigation: tests assert `DRY_RUN=True` causes zero `place_order` calls.

4. **Scheduled-task install scripts run as user, not SYSTEM.** Same as PF-MetalsLoop and PF-DataLoop. Documented in install script comments.

5. **Heartbeat staleness detection not yet wired.** Adding heartbeat writes is necessary but not sufficient — a separate watchdog reads them. Out of scope for this PR; the install script comments document the future watchdog.

---

## Execution order

| Batch | Files | LOC | Tests | Commit message prefix |
|---|---|---|---|---|
| **A** | `oil_swing_config.py` + `oil_warrant_catalog.json` + `oil_warrant_refresh.py` + 2 test files | ~700 | yes | `feat(oil): swing config + warrant scaffold (Batch A)` |
| **B** | `oil_swing_trader.py` + `oil_loop.py` + 2 test files | ~1300 | yes | `feat(oil): swing trader + 60s loop (Batch B)` |
| **C** | `crypto-loop.bat` + `oil-loop.bat` + 3 install-*.ps1 + crypto_loop hardening (exit 11, heartbeat, notify) + test_crypto_loop_hardening | ~400 | yes | `feat(loops): scheduled-task plumbing + crypto hardening (Batch C)` |
| **D** | dashboard `/api/oil` + scorecard updates + test_dashboard_oil | ~250 | yes | `feat(dashboard,obs): /api/oil + scorecard time-to-phase (Batch D)` |
| **E** | `OIL_LOOP_NOTES.md` + CHANGELOG + SESSION_PROGRESS + CLAUDE.md edit | docs | n/a | `docs: oil loop notes + changelog (Batch E)` |

After Batch E:
1. **Codex adversarial review** on the worktree branch SHA.
2. Address P1/P2 findings; document P3 deferrals.
3. **Full pytest** with `-n auto`.
4. **Merge** to main.
5. **Push** via Windows git: `cmd.exe /c "cd /d Q:\finance-analyzer && git push"`.
6. **Clean up** worktree + branch.

---

## Why this design

1. **Mirror the proven pattern.** Crypto+MSTR shipped 2026-04-30 using exactly this paper-mode pattern. Oil gets parity with zero new architecture.
2. **No live-trading switch flips.** All flips remain user decisions, made in separate commits after paper-mode validation.
3. **Install scripts, not auto-registration.** The user controls when each loop starts. Same pattern as `install-metals-loop-task.ps1`.
4. **Operational plumbing first, observability second, documentation third.** Dashboard endpoint and scorecard land in Batch D so they reflect real running behavior.
5. **Codex review before merge.** Catches edge cases the unit tests miss (race conditions, error paths, config drift).
