# Metals-Core Review — subagent result (caveman:cavecrew-reviewer)

Totals: 3 P1 (🔴), 10 P2 (🟡), 1 P3 (🔵)

## P1 / 🔴

1. **grid_fisher.py:129-135, 1545-1552** — `eod_sell_order_id` never cleared on fill; next cycle sees filled position + non-null id → skips EOD flat → position carried overnight (against EOD-flat invariant). Only cleared in `roll_session_if_new_day` (line 467).
2. **metals_loop.py:251-268** — `_TRADING_MINUTES`=820 (warrant session 08:15-21:55). Used in `exit_optimizer.simulate_intraday_paths` regardless of evaluation time. Pre-market (06:00 CET) evals scale annualized vol wrong.
3. **metals_loop.py:1099-1131** — `_silver_fetch_xag` falls back to `_underlying_prices.get("XAG-USD")` on network failure; no timestamp/staleness check. Fast-tick reads stale prices for 5+ min during Binance hiccup → false entry/exit signals. Memory rule `feedback_live_prices_first.md` violated.

## P2 / 🟡

- grid_fisher.py:90-106 — Knockout proximity via constant-leverage approximation; ignores funding decay; positions at hidden risk across sessions.
- metals_loop.py:1023-1080 — Fast-tick sub-loop runs network calls sequentially with no timeout → 9 ticks × 10s sleep + 2-3 hangs = 110s+ cycle overrun. Loop reports false-low heartbeat.
- grid_fisher.py:1599-1612 — EOD market-sell bid×0.99 limit misses fills in 21:50-21:55 auction (wide spreads); positions stay overnight; next session orphans.
- grid_fisher.py:576-581 — `global_planned_notional` includes inventory @ stale `avg_entry`; cap overestimated by 50-100 SEK per buy-tier fill.
- metals_loop.py:2419-2448 — Stop-loss `_update_stop_orders` never validates ≥3% above barrier (memory rule violated). `TRAIL_TIGHTEN_ACCEL` 2% on 5x with 17% cushion → 2.5% above barrier knockout risk.
- iskbets.py:139-177 — `_evaluate_entry` conjunctive AND of bigbet + buy_votes; missing bigbet module silently vetos.
- orb_predictor.py:358-361 — Percentile `idx = int(len * pct/100)` floor — off-by-one, 1 rank too low.
- metals_loop.py:402-408 — `EOD_HOUR_CET=17.0` triggers summary 4.9h before warrant close at 21:55; agent sees premature EOD report.
- grid_fisher.py:880-924 — `_effective_global_cap` cached 60s; rapid fills across 5 instruments use stale cache, overdraw account 5000+ SEK (P0-13 2026-05-13 incident).
- metals_loop.py:1145-1193 — `_write_momentum_candidate` no compare-and-swap; concurrent writes clobber other tickers.
- grid_fisher.py:1165-1264 — `rotate_on_buy_fill` stop price not revalidated vs barrier (3% rule); 3.5% stop on 4.76x is 16.7% underlying.
- metals_loop.py:478-494 — Silver alert thresholds hardcoded for 4.76x BULL; BEAR or 5x MINI uses wrong magnitude.
- grid_fisher.py:814-869 — `_fetch_buying_power_sek` 30s timeout + 300s stale grace; 3 consecutive slow ticks = 3min stale cap; new fills cause overdraw.
- exit_optimizer.py:177-186 — `_estimate_volatility` 20% annual default if missing; no log when default used → operator blind.
- oil_grid_signal.py:88-89 — Confidence cap 0.8 vs GRID_MIN_SIGNAL_CONFIDENCE 0.56; oil arms at 56% while metals require 60%; asymmetric.

## P3 / 🔵

- metals_loop.py:1065-1079 — Fast-tick exceptions to `_safe_print` not `logger.error`; invisible to observability.
- metals_loop.py:6254-6257 — EOD hour gate `(h_cet%1)<0.05` cycle-jitter sensitive; inconsistent triggering.
