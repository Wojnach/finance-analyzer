# Adversarial Review — Portfolio Risk Subsystem

**Date:** 2026-05-26
**Scope:** portfolio_mgr.py, portfolio_validator.py, trade_guards.py,
trade_risk_classifier.py, trade_validation.py, risk_management.py,
equity_curve.py, monte_carlo.py, monte_carlo_risk.py, circuit_breaker.py,
warrant_portfolio.py, exposure_coach.py, decision_outcome_tracker.py,
cost_model.py.

Prior review (2026-05-24) cited where applicable. Re-checked still-open
items; flagged any that have been resolved.

---

## Critical (P0, severity 90-100)

portfolio/warrant_portfolio.py:96: P0: warrant_pnl hard-codes `implied_pnl_pct = underlying_change * leverage`, correct only for LONG. 41 SHORT certs in `metals_warrant_catalog.json` report INVERTED P&L → telegram/dashboard/journal/stop-levels all wrong on any held SHORT/BEAR cert. Fix. Read direction from holding/catalog (`+1` LONG / `-1` SHORT) and multiply: `implied_pnl_pct = underlying_change * leverage * sign`. [REPEAT]
portfolio/warrant_portfolio.py:42-48,265: P0: `save_warrant_state` + `record_warrant_transaction` have NO cross-process lock and NO threading lock — pure load → mutate → save. metals_loop, grid_fisher, fin_snipe, iskbets, Layer 2 all mutate `holdings[key]["units"]` concurrently → silently dropped BUY/SELL units and avg-cost corruption. Fix. Wrap the read-modify-write in `file_utils.jsonl_sidecar_lock(WARRANT_STATE_FILE)`. [REPEAT]
portfolio/portfolio_mgr.py:108-159: P0: `_save_state_to` and `update_state` use a process-local `threading.Lock` keyed by `str(path)`. Layer 2 subprocess + main loop + dashboard write the same `portfolio_state.json` / `portfolio_state_bold.json` from SEPARATE processes — the lock is invisible across them. Last-writer-wins drops trades and breaks cash_sek reconciliation. Fix. Switch to `jsonl_sidecar_lock(STATE_FILE)` around the entire read-rotate-write cycle. [REPEAT]
portfolio/trade_guards.py:32,47,126,264: P0: `_state_lock` is a single process-local `threading.Lock`; `_save_state` is raw `atomic_write_json` with no cross-process protection. main loop + Layer 2 + bigbet/iskbets all call `record_trade()` — concurrent BUY records on separate processes silently drop one cooldown timestamp, BYPASSING the per-ticker cooldown and enabling the double-buy bug. Fix. Use `jsonl_sidecar_lock(STATE_FILE)` around `_load_state` + `_save_state`. [REPEAT]
portfolio/risk_management.py:374,382,465,484,897: P0: LONG-only stop math hard-coded across `compute_stop_levels`, `compute_probabilistic_stops`, and `check_atr_stop_proximity` — `stop_price = entry * (1 - 2*atr_pct/100)` and `triggered = current_price < stop_price`. A SHORT/BEAR position with profitable downside move trips a "triggered" stop on green → forced sell into strength. Fix. Plumb a `direction_sign` (LONG=+1, SHORT=-1) through holdings schema; `stop = entry * (1 - 2*atr_pct/100*sign)`; `triggered = (current - stop)*sign < 0`; `_first_hit_times` direction parameter must vary by sign. [REPEAT]
portfolio/monte_carlo_risk.py:204,228: P0: `_trading_days` hardcoded to 365 for ALL positions including MSTR. `volatility_from_atr` correctly uses 252 for MSTR (line 428), but `T = horizon_days / 365` on line 228 means the 1-day VaR sigma is annualized on 252 then re-divided by 365 → ~20% systematic understatement of 1-day VaR for stock positions, directly underestimating the loss tail Layer 2 uses for sizing. Fix. Per-position `td_i` looked up via `trading_days_for_ticker(ticker)`; rescale T per asset inside the marginal loop (line 267), or normalize all vols to a single calendar-day basis before `simulate_correlated_returns`. [REPEAT]
portfolio/monte_carlo_risk.py:408: P0: raw `agent_summary.get("fx_rate", FX_RATE_FALLBACK)` bypasses the `_resolve_fx_rate` cached-fallback chain that risk_management.py added precisely because `agent_summary` can carry `fx_rate=1.0` (or missing) during early cycle / rotation. SEK VaR is then 10x off and false-circuit-breakers downstream. Fix. Import `_resolve_fx_rate` (or its move-target `fx_rates._resolve_fx_rate`) and call it here too. [REPEAT]
portfolio/portfolio_validator.py:1-300: P0: NO check that any single transaction left cash_sek negative at the moment it was recorded. `validate_portfolio` only validates the END state — `cash_sek >= 0` (line 57). A BUY that overdrew cash and a subsequent SELL that restored it both pass. The Bold strategy already lost 35K SEK by overtrading; there is no audit-trail invariant catching the moment of overdraft. Fix. Replay transactions chronologically, assert `running_cash >= -tolerance` after each, and append the offending tx index to errors.

---

## Important (P1, severity 80-89)

portfolio/risk_management.py:217-270: P1: drawdown circuit breaker falls back to cash-only value when `agent_summary` is empty. Layer 2 still receives `breached=False`. With holdings underwater the breaker is asleep. Fix. Return `{"breached": False, "stale": True}` and gate all BUY decisions on `not stale`. [REPEAT]
portfolio/risk_management.py:285-286: P1: when `current_value > peak_value` (line 285) the peak is updated even when `current_value` came from the cash-only stale fallback (line 270). A subsequent valid feed can be below the true peak — the breaker records a downward-revised peak. Mitigation. Don't overwrite peak with `current_value` when `summary` was empty (track an `is_stale` flag). [REPEAT]
portfolio/risk_management.py:373: P1: `atr_pct = min(atr_pct, 15.0)` caps ATR-derived stop at 15% on the UNDERLYING — but warrant catalog leverage means a 15% underlying drop = 75% warrant drop at 5x. The 15% cap is described in CLAUDE.md as the warrant-cert minimum stop width, yet here it's the maximum. Wrong cap direction biases stops TOO TIGHT for warrant tickers, contradicting `memory/feedback_mini_stoploss.md`. Fix. Apply a floor of 15% for warrant-class tickers, only cap at 15% for spot crypto/stocks.
portfolio/risk_management.py:870-919: P1: `check_atr_stop_proximity` repeats the same LONG-only formula as `compute_stop_levels` and is called with `action == "CHECK"` (line 948) on HOLD positions every cycle. SHORT warrant holdings get a spurious "danger zone" flag every loop. Fix. Direction-aware sign as in P0. [REPEAT]
portfolio/equity_curve.py:494-495: P1: `wins = [t for t in trips if t["pnl_pct"] > 0]; losses = [t for t in trips if t["pnl_pct"] <= 0]` — break-even (`pnl_pct == 0`) bucketed as loss, inflating loss_count and `max_consecutive_losses`. Fix. Use `< 0` for losses, separate `== 0` bucket. [REPEAT]
portfolio/equity_curve.py:23: P1: `ANNUALIZATION_DAYS = 365` is applied uniformly to BOTH patient and bold Sharpe/Sortino. Bold may hold MSTR (stock, 252) and patient holds crypto (365); mixing one annualization on a blended equity curve overstates stock-heavy Sharpe by `sqrt(365/252) ≈ 1.20`. Fix. Either (a) document as crypto-biased system-wide convention (system is mostly crypto/metals), or (b) compute per-asset-class annualized vol and aggregate.
portfolio/equity_curve.py:415-417: P1: `pnl_sek` and `fee_sek` are rounded to 2 decimals BEFORE being summed into `total_pnl_sek` on line 532. Round-trip rounding bias accumulates — over 1000 trades the cumulative error can exceed 100 SEK against the running cash. Fix. Sum unrounded values inside `_pair_round_trips` (store an internal float), only round at the final `compute_trade_metrics` boundary. [REPEAT]
portfolio/equity_curve.py:550-558: P1: Calmar uses `equity = [initial_value]; equity.append(equity[-1] + t["pnl_sek"])`. This rebuilds an equity curve from realized round-trip P&L only — ignores mark-to-market drawdown while holding (which is usually larger). Calmar therefore understates risk. Fix. Either source from `portfolio_value_history.jsonl` (true MTM), or document the limitation. [REPEAT]
portfolio/monte_carlo.py:305: P1: `simulate_ticker` computes `stop_price = price * (1 - 2 * atr_pct / 100)` (LONG only) and reports `p_stop_hit_{h}d` via `probability_below(stop_price)`. SHORT/BEAR positions get nonsensical stop-hit probability that Layer 2 reads as risk. Fix. Direction-aware sign + `probability_above` for shorts. [REPEAT]
portfolio/monte_carlo.py:150: P1: `seed` is consumed once at `np.random.default_rng(self.seed)` per MonteCarloEngine construction, but `simulate_all` increments seed per ticker (`ticker_seed = seed + i`). When the same ticker is simulated at multiple horizons inside `simulate_ticker` (lines 309-321), all horizons use the SAME seed → terminal prices are perfectly correlated across horizons. `p_stop_hit_1d` and `p_stop_hit_3d` carry hidden duplicate randomness. Fix. Either pass `seed=None` to MonteCarloEngine, or derive `seed + h * 7919` per horizon.
portfolio/monte_carlo_risk.py:188: P1: `[t for t, p in positions.items() if p.get("shares", 0) != 0]` accepts NEGATIVE shares (shorts). Downstream `total_value = sum(shares * price)` (line 473) then sums negative * positive = negative exposure, and `drawdown_probability` line 376 divides by `total_value <= 0` → returns 0 even if a synthetic short is losing money fast. Fix. Use `abs(shares) * price` for exposure denominator, and handle short-side P&L direction (currently `pnl += shares * price * (exp(r) - 1)` is correct for shorts IF shares is negative, so most of this is sound — but the exposure denominator is wrong).
portfolio/trade_guards.py:78-96: P1: `LOSS_ESCALATION = {0:1, 1:1, 2:2, 3:4, 4:8}` then time-decay uses `base >> halvings`. For `base = 2` after 1 halving → `2 >> 1 = 1` (jumps from 2x straight to base). Bit-shift floors any non-power-of-2 escalation to 1. Fragile to tuning the table. Fix. Use float arithmetic: `base = max(1.0, base / (2 ** halvings))`. [REPEAT]
portfolio/trade_guards.py:138-167: P1: `ticker_trades` cooldown key is `f"{strategy}:{ticker}"` — a BUY on the same ticker in the OTHER strategy doesn't block. Two Layer 2 invocations spawned 5s apart (one for patient, one for bold) on the same XAG spike fire simultaneously, doubling exposure on a correlated bet. Fix. For tickers in `CORRELATED_PAIRS` consider strategy-agnostic cooldown, or read `holdings` from the partner portfolio in `check_concentration_risk`.
portfolio/portfolio_validator.py:144-145: P1: `Holdings contains X with N shares but no matching transactions` fires on manually-seeded recovery positions (live Avanza-only reflection without synthesized transactions). Misleading error on first cycle. Fix. Add a `seed_marker` field or accept a `seed_transactions` list. [REPEAT]
portfolio/portfolio_validator.py:216-243: P1: avg_cost_usd check `expected_avg = total_cost / total_bought` doesn't account for partial sells — if a position was scaled in/out repeatedly, the weighted average of ALL historical BUYs is NOT the cost basis of the remaining shares. Triggers false 1% mismatch errors on positions that scaled out. Fix. Use FIFO-remaining cost (mirror `equity_curve._pair_round_trips` but stop matching when only buys remain in queue).
portfolio/decision_outcome_tracker.py:76: P1: bare `except Exception` swallows network/rate-limit errors silently — decision is dropped, never backfilled, accuracy biased upward toward outcomes that did fetch. Fix. Log WARNING with ticker+ts, increment a fail counter on health endpoint. [REPEAT]
portfolio/monte_carlo_risk.py:225-281: P1: drift_term `(mu - 0.5*sigma^2)*T` uses the same `T` as the volatility term but the drift `mu` was derived from `drift_from_probability(p_up, vol, trading_days=td)` per ticker (td may be 252) — then the simulator's `T` uses 365 (line 228). Inconsistent time-scaling between mu and sigma for stocks. Lockstep with P0 fx fix. [REPEAT]
portfolio/cost_model.py:49-51: P1: `total_cost_pct()` ignores `min_fee_sek`. A 1000 SEK stock trade has courtage = max(0.069, 1) = 1 SEK = 0.1% but `total_cost_pct` reports `(6.9 + 5.0 + 2.0)/100 = 0.139%`. For warrants `min_fee_sek = 0` so accurate, but for stock cost projections the reported number is wrong on small orders. Fix. Take an optional `trade_value` argument and include the floor. [REPEAT]
portfolio/cost_model.py:64-69: P1: WARRANT_COSTS `spread_bps = 40` (0.40% half-spread) is a flat constant. Real Avanza silver MINI half-spreads in `metals_warrant_catalog.json` range from 0.04% to 3% depending on liquidity. Overstates cost on liquid certs, understates on illiquid. Fix. Read per-cert `spread_pct` from catalog and pass at construction. [REPEAT]

---

## Latent (P2, severity 60-79)

portfolio/monte_carlo.py:174-177: P2: odd-`n_paths` branch reuses the same RNG (line 175) immediately after consuming `n_half` draws. Correct but introduces a 1-path asymmetry that breaks antithetic variance reduction guarantees on `n_paths=1`. Test-only; production uses 10K.
portfolio/monte_carlo.py:201,217,233: P2: `price_quantiles` returns rounded floats from `np.percentile`. Tail percentiles (5, 95) are NOT variance-reduced by antithetic Z (only the mean is). `p_stop_hit` precision rounded to 0.001 (line 324) overstates accuracy — true MC standard error is ~1/sqrt(10K) ≈ 1%. Document; or bump to 50K for tail probability columns.
portfolio/monte_carlo_risk.py:88-106: P2: Higham `_nearest_psd` clips eigenvalues at 1e-8 but doesn't validate the input is symmetric to begin with. `np.linalg.eigh` requires symmetric — and a `_get_prior_correlation` lookup that returns asymmetric values (e.g., one direction missing) silently produces complex eigenvalues. Mitigation. Symmetrize before eigh: `matrix = 0.5*(matrix + matrix.T)`.
portfolio/trade_validation.py:30-32: P2: hardcoded defaults `max_spread_pct=2.0`, `min_order_sek=1000.0` not instrument-aware. Silver MINIs commonly exceed 2% spread; CLAUDE.md says warrants need 5x leverage with wider stops. Plumb instrument type. [REPEAT]
portfolio/trade_validation.py:69: P2: `order_value > cash_available` uses strict `>`, so an exactly-equal allocation passes (cash drops to 0 with no buffer for the fee). Combined with no fee deduction in `validate_trade` (the function knows about courtage neither in `min_order_sek` nor in the cash-check), a 100% cash BUY can leave the portfolio with negative cash once the fee is debited. Fix. Use `>= cash_available - estimated_fee`, or accept `fee_sek` parameter.
portfolio/portfolio_mgr.py:162-180: P2: `portfolio_value` skips holdings with `price <= 0` or `price is None`, returning cash-only valuation. Same blind-spot pattern as risk_management's stale-feed fallback — silent optimistic value. Fix. Surface a `partial_valuation: True` flag in return.
portfolio/risk_management.py:81-89: P2: `_streaming_max` opens file outside the lock (line 85) — file_size was snapshotted under lock but the read itself races with appenders. If a new record extends the file between the cache read and the actual open, the offset is stale by milliseconds. Low-impact (peak only goes up; we just may rescan), but the lock semantics are misleadingly tight.
portfolio/equity_curve.py:182-189: P2: annualized return uses `(last / first) ** (1/years)` where first_val is the FIRST nonzero entry — for a portfolio that started with 0 holdings (cash 500K), first_val=500K is reasonable, but if `portfolio_value_history.jsonl` starts mid-cycle after a system restart, first_val may be a temporary low (mid-drawdown). Annualized return then explodes. Fix. Always anchor first_val to `INITIAL_VALUE` constant.
portfolio/equity_curve.py:184: P2: `days_elapsed >= 1` gate — for a 23-hour curve the annualized return stays None, silently dropping it from the dashboard. Edge-case during fresh installs. Document or relax to `>= 0.5`.
portfolio/circuit_breaker.py:64-72: P2: HALF_OPEN → OPEN transition doubles `self.recovery_timeout` (line 65), capped at `_max_recovery_timeout`. But `record_success` resets to `_base_recovery_timeout` (line 51) only on HALF_OPEN→CLOSED — if `record_success` fires while CLOSED (line 52), the timeout is NOT reset. Long-running breakers that flap will accumulate timeout, never returning to base. Fix. Always reset timeout in `record_success`, not just on state transition.
portfolio/exposure_coach.py:42-117: P2: `compute_exposure_recommendation` accepts `portfolio_concentration` but never gates `new_entries_allowed` on it — only zone+regime. A portfolio with 90% concentration in XAG still gets `new_entries_allowed=True` if market is healthy. Fix. Block new entries when concentration > 0.5 regardless of zone.
portfolio/trade_guards.py:78-79: P2: `if consecutive_losses >= 4: base = LOSS_ESCALATION[4]` — table only goes to 4. A 5th consecutive loss caps at 8x and stays at 8x. Intentional, but consider an absolute trade lockout after N losses (e.g., return `"block"` for N>=5) rather than just a wider cooldown.
portfolio/decision_outcome_tracker.py:81-83: P2: `correct = (outlook == "bullish" and change_pct > 0) or (outlook == "bearish" and change_pct < 0)` — `change_pct == 0` is bucketed as INCORRECT for both directions, biasing accuracy down on perfectly-flat outcomes (rare but happens overnight on stocks). Fix. Add a `flat` bucket or treat 0 as neutral-and-skip.

---

## TOP 3

1. **Warrant book has zero concurrency protection** (`warrant_portfolio.py`).
   `record_warrant_transaction` is pure read → mutate → save with neither a
   threading lock nor a cross-process lock. metals_loop, grid_fisher,
   fin_snipe, iskbets, and Layer 2 all hit it. The fix is one `with
   jsonl_sidecar_lock(WARRANT_STATE_FILE)` wrap and an hour of work — it
   prevents silently dropped units that the validator can't even detect
   because there's no per-cert reconciliation. [REPEAT from 2026-05-24]

2. **LONG-only direction in every stop, P&L, and MC path**
   (`warrant_portfolio.warrant_pnl`, `risk_management.compute_stop_levels` /
   `compute_probabilistic_stops` / `check_atr_stop_proximity`,
   `monte_carlo.simulate_ticker`). 41/99 catalog certs are SHORT. The system
   currently inverts P&L for any held BEAR cert and "triggers" stops on
   profitable moves. Adding a `direction_sign` field to the holdings schema
   and threading it through six modules is the single highest-impact
   correctness fix.

3. **No per-transaction overdraft invariant** (`portfolio_validator.py`).
   The validator only checks END-state `cash_sek >= 0`. A BUY that overdrew
   cash followed by a SELL that restored it passes silently. Combined with
   `validate_trade` using strict `>` for cash sufficiency AND not deducting
   the fee from the cash check (`trade_validation.py:69`), a 100%-cash BUY
   can leave negative cash post-fee — and the next cycle's validator can't
   tell. Walk transactions chronologically and assert running balance per
   step.

— end of report —
