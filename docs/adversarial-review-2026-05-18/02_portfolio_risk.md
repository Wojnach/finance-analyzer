# Portfolio-Risk Review — subagent result (caveman:cavecrew-reviewer)

Totals: 2 P1 (🔴), 9 P2 (🟡)

## P1 / 🔴

1. **risk_management.py:761-774** — Concentration check returns `None` silently if `total_value<=0` (all-cash or negative). Blind spot, no WARN log.
2. **equity_curve.py:366-369** — SELL without matching BUY (missing ticker in buy_queues) silently skipped → orphan SELL records, incomplete P&L.
3. **portfolio_mgr.py:154-159** — Backup rotated AFTER mutate_fn(); if mutate crashes mid-execution, backup is from previous cycle, not current broken state. Recovery hides recent failures.

## P2 / 🟡

- risk_management.py:765 — `proposed_alloc = min(total_value * alloc_pct, cash)` uses stale avg_cost_usd fallback prices; underestimates position size when feed lags.
- equity_curve.py:423 — `remaining_shares <= 1e-10` arbitrary epsilon; floating-point accumulation may keep tiny residuals, orphan matches.
- trigger.py:376-379 — Clock-skew cooldown reset never persists; repeats log warnings every cycle until clock advances.
- warrant_portfolio.py:237-246 — Averaging-in: only updates `underlying_entry_price_usd` if BOTH prices >0; partial adds leave stale entry, breaks SL underlying levels.
- trade_guards.py:286-288 — Profitable SELL clears `consecutive_losses` but leaves `last_loss_ts` stale; direct access reports wrong streak duration.
- risk_management.py:752-758 — Concentration falls back to `avg_cost_usd` silently when live missing → 5-20% exposure blind spots after-hours.
- cost_model.py:44 — Courtage rounding at return time accumulates >0.05 SEK error over 100+ legs.
- equity_curve.py:413 — Shares rounded to 8 decimals; for warrants with units of 100, 8-decimal precision may truncate.
- trade_guards.py:318-328 — Timestamp pruning at `>= cutoff` boundary; microsecond-precision oscillation at exact 24h cutoff.
- warrant_portfolio.py:80-89 — `warrant_pnl()` returns None on `underlying_entry_price=0` without logging; caller can't distinguish missing vs zero.
- monte_carlo.py:394-402 — `_get_directional_probability` falls back to 0.5 when pre-computed missing; masks disabled focus_probabilities silently.
- risk_management.py:764 — Alloc % hardcoded (30% bold, 15% patient); no config override for vol regime adapt.
