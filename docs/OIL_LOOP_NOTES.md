# Oil Loop — Operator Notes

**Status (as of 2026-05-01):** DRY_RUN paper mode. `DRY_RUN = True` in
`data/oil_swing_config.py`. The loop logs decisions to
`data/oil_swing_decisions.jsonl` but places no Avanza orders.

## Quick start

The loop is **registered but not started** by default. After running the
install script, the user starts it explicitly:

```powershell
# Register the scheduled task (one-time)
powershell -ExecutionPolicy Bypass -File scripts\win\install-oil-loop-task.ps1

# Start it
Start-ScheduledTask -TaskName 'PF-OilLoop'
```

Manual one-shot (single cycle, no scheduling):

```bash
.venv/Scripts/python.exe -u data/oil_loop.py --once --debug
```

## What the loop does

60-second cycle, parallel to `data/metals_loop.py`:

1. Acquire singleton lock at `data/oil_loop.lock`.
2. Fetch live WTI price via `portfolio.price_source.fetch_klines("CL=F", ...)`.
   The router prefers Binance FAPI for real-time CL=F with yfinance
   fallback (per `oil_precompute.py` 2026-04-14 routing).
3. Read latest Layer-1 signal snapshot from
   `data/agent_summary_compact.json`.
4. Run `OilSwingTrader.evaluate_and_execute()`. Decisions log to
   `data/oil_swing_decisions.jsonl`. In DRY_RUN: no Avanza orders.
5. Embedded fast-tick monitor: every 10 seconds for the rest of the
   cycle, re-checks WTI for sharp-dip alerts (-2.5%) and velocity-flush
   alerts (+1.5% in 3 minutes).
6. Heartbeat written to `data/oil_loop.heartbeat` each cycle.

## Files this loop owns

| File | Purpose |
|---|---|
| `data/oil_swing_state.json` | Positions, cash, cycle counter |
| `data/oil_swing_decisions.jsonl` | Append-only decision log |
| `data/oil_swing_trades.jsonl` | Closed round-trips |
| `data/oil_value_history.jsonl` | Per-cycle value snapshot for dashboard |
| `data/oil_signal_log.jsonl` | Signal snapshots at decision time |
| `data/oil_signal_outcomes.jsonl` | Outcome backfill |
| `data/oil_risk.json` | Per-position barrier checks, drawdown |
| `data/oil_warrant_catalog.json` | Live OLJA warrant universe (refreshed every 6h) |
| `data/oil_loop.lock` | Singleton lock |
| `data/oil_loop.heartbeat` | External-watchdog liveness check |
| `data/oil_loop_out.txt` | Stdout/stderr from the bat wrapper |

## Why the trades log may be missing

`data/oil_swing_trades.jsonl` is created on first closed round-trip, not
at loop startup. If the file is absent, the loop is either:

1. **Outside the macro/news window** for the trader's gates — RSI in
   neutral zone, voters below threshold, regime confirmation pending.
   This is the normal idle state. Check `oil_swing_decisions.jsonl` for
   per-cycle reasons.
2. **DRY_RUN with no qualifying entries yet** — the entry gates require
   2+ cycles of consistent confidence and a passing MACD-decay ratio,
   which weeds out most single-spike triggers.
3. **Not running** — check
   `cmd.exe /c "schtasks /query /tn PF-OilLoop"`. If absent, the install
   script hasn't been run.

## Phase transition (DRY_RUN → live)

DRY_RUN → live requires:

- `DRY_RUN = False` in `data/oil_swing_config.py`.
- At least 30 days of paper-mode equity curve data showing positive
  expectancy net of simulated fees, ≥15 closed round-trips, ≥55% win
  rate. Verify via:

  ```bash
  .venv/Scripts/python.exe scripts/oil_loop_scorecard.py
  ```

  Live-flip gates are listed in the scorecard output. All four must
  PASS before flipping.

- Live warrant catalog must be populated. The refresh script
  (`data/oil_warrant_refresh.py`) needs a Playwright session with a
  logged-in Avanza state to fill in barriers and current prices for the
  seed warrants in `oil_swing_config.WARRANT_CATALOG_FALLBACK`. Until
  then, the catalog file at `data/oil_warrant_catalog.json` will stay
  empty and the trader will skip all entries.

- Manual user approval. **Do NOT flip the DRY_RUN flag unattended.**

The flip is a single-line config change in a separate commit, reversible
by reverting that commit. Existing positions (if any) are held to their
natural exits — the flag only affects new orders.

## Universe

v1: WTI only. Brent (BZ=F) is deferred until the Avanza warrant universe
for Brent specifically is verified. Most Avanza "OLJA" certificates
track Brent under the hood, so the LONG/SHORT exposure roughly aligns,
but the risk gates assume the WTI underlying for now.

The seed warrant catalog includes 5 OLJA instruments scraped from
`data/avanza_instruments_live.json` on 2026-04-30:

| Name | OB ID | Direction | Leverage |
|---|---|---|---|
| MINI L OLJA AVA 624 | 2370189 | LONG | (probe) |
| MINI L OLJA AVA 479 | 1329405 | LONG | 1.52 |
| MINI S OLJA AVA 699 | 2368945 | SHORT | 2.27 |
| MINI S OLJA AVA 701 | 2368906 | SHORT | 2.55 |
| BEAR OLJAB X3 AVA 2 | 2367789 | SHORT | 3.0 |

Barriers are missing on all 5 — the refresh script fills them in on
first probe with a live Avanza session.

## Watchdog (TBD)

`data/oil_loop.heartbeat` is updated each cycle (~60s). An external
watchdog reading the heartbeat age and restarting the loop if stale
> 180s is the next operational improvement. Until then, monitor manually
via the dashboard `/api/oil` endpoint, which surfaces the heartbeat as
part of its response.
