# Adversarial Review — Agent 3: Portfolio & Risk Subsystem

**Subsystem:** Portfolio state, validation, guards, risk math, Monte Carlo, cost model, Kelly sizing, equity curve, backtester
**Files reviewed:** 18 (5,570 lines)
**Date:** 2026-05-19

## Severity counts
- P0 (data loss / silent money loss): 4
- P1 (real bug): 11
- P2 (latent / edge case): 9
- P3 (minor): 4
- **Total: 28 findings**

---

## P0 — Data loss / silent money loss

### F01 — `portfolio_mgr.py:108-113` Lock is per-process, not cross-process. Layer 2 subprocess + dashboard race.
**P0.** `_save_state_to()` uses `threading.Lock`, which only serializes within ONE Python process. Layer 2 is spawned as a separate `claude -p` subprocess (see `agent_invocation.py`). The dashboard runs as a third process. Two processes calling `update_state()` concurrently will both load the state, both mutate, both write — last-writer-wins, lost transactions. The "atomic" rename is atomic at the filesystem level but does NOT serialize read-modify-write across processes. There is no file-level `flock`/`msvcrt.locking` lock. CLAUDE.md flags this exact concern; CRITICAL Rule 4 calls atomic_write_json sufficient but it isn't for RMW. Recommend `portalocker` or an OS file lock around the RMW window.

### F02 — `risk_management.py:84-99` `_streaming_max` opens raw `open()` for reading a file that `atomic_append_jsonl` is concurrently appending to.
**P0.** The streaming peak reader uses `with open(history_path, encoding="utf-8") as f:` and iterates lines. `atomic_append_jsonl` appends entries non-atomically with respect to line boundaries on Windows (POSIX append-atomicity is OS-specific). A partial-line read silently fails the inner `json.loads` (continued, no log). Result: the latest peak can be silently missed for one cycle. Worse: if the writer is mid-flush during the reader's `f.tell()`, the cached offset stored in `_peak_cache` points into the middle of the next line, and the next call resumes mid-line. Subsequent JSON decode errors are swallowed and the peak silently freezes. The drawdown circuit breaker depends on accurate peak — a frozen peak can mask a real drawdown.

### F03 — `warrant_portfolio.py:42-49` `save_warrant_state` has NO concurrency lock at all.
**P0.** `portfolio_mgr.py` learned the lock lesson (per-file `threading.Lock`). `warrant_portfolio.py:42-49` writes the same critical state file `portfolio_state_warrants.json` without ANY lock — neither process nor thread. The metals loop, dashboard, and ad-hoc scripts all call `record_warrant_transaction()` (line 182) which load→mutate→save. Concurrent calls will lose transactions silently. The warrants file holds real leverage positions; a lost SELL means phantom holdings; a lost BUY means under-counted leverage. This is the highest-stakes state file in the system.

### F04 — `risk_management.py:653-654, 676` Fee accounting picks `max(state_fees, computed_fees)` which silently hides double-bookkeeping bugs.
**P0.** `total_fees = max(total_fees_from_state, computed_fees)` masks any divergence between the running `total_fees_sek` counter and the per-transaction `fee_sek` sum. If state-side fees were ever double-incremented (a common bug class), this returns the inflated number with no warning. Conversely if some transactions lack `fee_sek` fields, fees are under-reported. The right behaviour is to compare and log a discrepancy; the validator does this (`portfolio_validator.py:165-178`) but `transaction_cost_analysis` actively hides it. Operators reading the dashboard will see plausible-looking numbers even when accounting is broken.

---

## P1 — Real bugs

### F05 — `equity_curve.py:443-446, 469, 471` Profit factor and total_pnl use loss bucket `pnl_pct <= 0` for losses but `pnl_sek > 0` for wins. Zero-PnL trips are double-counted.
**P1.** `gross_profit = sum(t["pnl_sek"] for t in trips if t["pnl_sek"] > 0)` (line 468) but `losses = [t for t in trips if t["pnl_pct"] <= 0]` (line 495). The first uses strict `> 0`, the second uses `<= 0` — a trip with exactly zero pnl_pct is counted as a "loss" for win_loss_ratio, max_consecutive_losses, expectancy. Worse, the buckets disagree (pnl_sek vs pnl_pct) — net of fees a tiny win can be a SEK loss but `pnl_pct` (gross) is +. A trade can show up in `wins` (pnl_pct>0) AND contribute 0 to `gross_profit` (pnl_sek≤0). Inconsistent classifier silently corrupts Sharpe-comparable metrics.

### F06 — `equity_curve.py:232, 249, 256` Annualization uses 365 for crypto/24-7 but Sharpe still uses risk-free rate / 365. Mixing assumptions inflates Sharpe.
**P1.** `ANNUALIZATION_DAYS = 365` is justified for 24/7 crypto. But `daily_rf = RISK_FREE_RATE_ANNUAL / ANNUALIZATION_DAYS` (line 236) = 3.5%/365 = 0.0096%/day. The Swedish risk-free rate is a *business-day* yield (252 BD). For an 8-month sample with mixed regimes you'll under-subtract rf, biasing Sharpe slightly upward. Minor but documentable.

### F07 — `equity_curve.py:248-249` Sharpe recomputes `daily_std_dec` from decimal returns but `daily_vol` was computed from percent returns. Two different stds → confused annualization.
**P1.** Line 232 computes `daily_vol = sqrt(variance)` from `daily_rets` (in percent). Line 244 recomputes `daily_std_dec = sqrt(...)` from `daily_rets_dec` (in decimal). They differ by exactly 100x, but the `volatility_annual_pct` result is from the percent path and the Sharpe uses the decimal path. This is actually correct math but the redundant computation invites future drift. More important: `mean_excess` (line 239) is `sum(r - daily_rf)/n` in decimals; the BUG-225 comment claims it was extracted "to avoid O(n²) recomputation" but `mean_dec` (line 243) is NOT the same as `mean_excess` — `mean_excess` has rf subtracted, `mean_dec` doesn't. The std-from-mean-dec formula is technically the right denominator (returns std, not excess-return std), so this is right; but the comment is misleading and the two means look interchangeable on grep, inviting future "simplification" that breaks Sharpe silently.

### F08 — `equity_curve.py:252-257` Sortino downside deviation divides by total observations N, not by downside count. Result is the population formula but with denominator-mismatch comment "standard formula".
**P1.** Sortino has TWO conventions:
1. divide by N (standard, biased low for high-skew, used by Sortino's original paper)
2. divide by n_downside (used in many texts)
The choice matters; using N when most days are positive understates downside_dev and inflates Sortino. The comment claims "standard formula" but doesn't acknowledge which convention. Add doctring with citation; current dashboards may compare these numbers across strategies that use the other convention.

### F09 — `equity_curve.py:104-107` `_daily_returns` returns 0.0 for any day where `prev_val <= 0`. Silently injects fake "flat" days, biasing volatility and Sharpe.
**P1.** If a portfolio briefly hits zero (or near-zero) value due to corrupted snapshot, the daily-return calc inserts `0.0` instead of skipping. This silently smooths volatility and biases Sharpe upward. Should `continue` instead of appending 0.

### F10 — `monte_carlo.py:154, 226-227, 274-276` GBM uses hardcoded `T = horizon_days / 252.0` but `volatility_from_atr` annualizes with 252 trading days. For crypto (365-day market), this is wrong.
**P1.** Line 154: `T = self.horizon_days / 252.0` — converts horizon days to years assuming a 252-day year. But ATR for BTC is computed on 24/7 candles; volatility is "annualized" with 252 even though there's no weekend gap. The same bug exists in `monte_carlo_risk.py:227, 275`. Net effect: for BTC, you scale volatility by sqrt(252/14) but a 1-day horizon is treated as 1/252 of a year. For a stock-only world this is consistent; for the *mixed* portfolio (crypto + metals + MSTR), the time-and-volatility-base mismatch produces VaR that's off by ~22% (sqrt(365/252) ≈ 1.20). VaR is the critical risk number — being 20% off is real.

### F11 — `monte_carlo_risk.py:251-252, 265-272` t-copula → Gaussian marginals path silently produces wrong tail dependence if df changes elsewhere.
**P1.** The C9 comment claims fixed, but the actual line `T_samples = W * scale` (line 252) and then `U = t_dist.cdf(T_samples, df=self.df)` (line 256) are correct ONLY when `W = Z @ L.T` where Z is N(0,I). After scaling by `sqrt(df/S)`, the marginals of `T_samples` are univariate Student-t(df) — good. CDF gives uniforms — good. `norm.ppf(U)` gives Gaussian marginals — good. **BUT** the correlation structure embedded in `W = Z @ L.T` is *Gaussian*; multiplying by the chi-scale produces a t-COPULA but the linear-correlation matrix of the resulting Gaussian marginals is NOT the input `corr`. The actual Kendall's tau is preserved but Pearson is scaled by 2/π. So a `corr=0.7` prior produces empirical Pearson ~0.5 between the simulated returns. VaR for correlated portfolios will be understated.

### F12 — `monte_carlo_risk.py:407` `compute_portfolio_var` uses raw `agent_summary.get("fx_rate", FX_RATE_FALLBACK)` — bypasses the `_resolve_fx_rate` chain.
**P1.** `risk_management.py` learned the lesson (P1-15 in code comments) and added cached fallback. `monte_carlo_risk.py:407` still uses the raw `.get("fx_rate", FX_RATE_FALLBACK)`. If `agent_summary["fx_rate"]` is a stale 1.0 (the bug the cached-chain was added to fix), VaR-in-SEK is off by 10×. Result: dashboard shows VaR_95 = -1.2k SEK when reality is -12k SEK.

### F13 — `risk_management.py:373` ATR cap at 15% silently softens stop-loss for true high-vol instruments.
**P1.** `atr_pct = min(atr_pct, 15.0)` caps ATR. For a 5x silver MINI warrant on a 6% silver-spot day, true vol is ~30%; cap pulls it to 15%. Stop-loss distance is then `2 * 15 = 30%` below entry, much closer than appropriate for the actual vol. Worse, MINI warrants have **knockout barriers** (financing levels) — the 30% wide stop guarantees you'll get knocked out before stop triggers. CLAUDE.md memory notes "Never place a stop-loss within 3% of current bid" and "Never place stop-losses near MINI warrant barriers". This cap creates exactly that condition. Recommend: warn or refuse to publish stops for ATR > 15% rather than silently clamp.

### F14 — `risk_management.py:382` `triggered = current_price < stop_price` — no flag-direction check for SELL trades / BEAR warrants.
**P1.** The stop logic assumes long position (price below stop = triggered). For a held SHORT/BEAR warrant where you profit when underlying falls, the stop should fire when current price goes ABOVE stop. The function doesn't know direction. Today this is OK because the simulated portfolios only go long, but `warrant_portfolio.py` mixes BULL and BEAR certificates — when `compute_stop_levels` is fed a BEAR cert's underlying, `triggered` reports the wrong direction and risk dashboards lie.

### F15 — `warrant_portfolio.py:92-101` Warrant P&L treats leverage as symmetric — no distinction between BULL and BEAR direction.
**P1.** `implied_pnl_pct = underlying_change * leverage` (line 96). For a BEAR warrant (e.g. MINI SHORT SILVER), a positive underlying_change should produce a *negative* P&L. The code doesn't read direction from holding — `leverage` is treated as a pure scalar. A BEAR position is mis-marked symmetrically. Recommend `direction = holding.get("direction", "LONG")` and multiply by `+1` or `-1`. Currently relies on caller passing negative leverage for shorts (no convention documented). High-impact silent error.

### F16 — `kelly_sizing.py:49-52` Kelly fraction clamps to `[0, 1]` but does not warn on `kelly > 0.25`, despite half-Kelly being the recommendation everywhere else.
**P1.** A degenerate win_prob of 0.99 with avg_win=10, avg_loss=1 gives Kelly ≈ 0.989. `recommended_size` then takes `half_kelly * cash_sek * exposure_ceiling` — could allocate 49% of cash in one position. Combined with the 30% bold ceiling at line 269, the `min(..., max_alloc)` saves it for now — but if `max_alloc_sek` is widened or strategy reorganized, full-Kelly territory becomes reachable. Kelly should hard-cap at 0.25 (quarter-Kelly safety) regardless of inputs, or at least warn when `kelly > 0.5`.

### F17 — `kelly_metals.py:243-245` `daily_log_growth` uses `log(max(1e-10, 1 - f * cert_loss_frac))` to dodge log(0). The 1e-10 floor lets `f * cert_loss_frac ≥ 1` (total ruin) silently pass.
**P1.** If `f` (position_fraction) is 0.95 and `cert_loss_frac` is 1.1 (110% loss on a leveraged knock-out), `1 - 1.045 = -0.045` — the floor saves the log but the resulting "growth" number is `log(1e-10) ≈ -23`, which then drives `monthly_growth = exp(-23*22) = 0` — looks fine. But the underlying situation is "Kelly says ruin is on the table" — should refuse to size, not silently clip. The `MAX_POSITION_FRACTION = 0.95` saves it within this function, but the `cert_loss_frac` line (215) doesn't cap leverage * avg_loss / 100; a 10x cert with 12% avg_loss gives `cert_loss_frac = 1.2`, mathematically saying any position size is ruin.

---

## P2 — Latent / edge case

### F18 — `portfolio_mgr.py:60` `_rotate_backups` does shallow rotation. If both `.bak` and `.bak2` are corrupt, `_load_state_from` returns fresh defaults — wipes the portfolio.
**P2.** Defense in depth is good, but `load_state` returning a *default 500K SEK fresh portfolio* on triple-corruption is the silent worst case. Should raise an exception so Layer 2 stops trading, not pretend the portfolio is a fresh seed. The CRITICAL log line (line 102) doesn't propagate.

### F19 — `portfolio_validator.py:122-132` 1% relative tolerance for share remainders is too loose for warrants.
**P2.** Comment says "small remainder from rounding". 1% of bought warrant units on a 5x leveraged MINI position is 5× equity error — easily a few hundred SEK invisible drift per cycle. Tighten to 0.1% or use absolute 1-unit tolerance.

### F20 — `trade_guards.py:135` `effective_cooldown = base_cooldown * multiplier` — no upper bound.
**P2.** With `LOSS_ESCALATION[4]=8` plus base 30 min = 240 minutes (4 hours). OK. But if base is configured higher (e.g. 480 via config), cooldown becomes 64 hours — strategy effectively gets benched for 3 days. Worth capping at e.g. 24h, or at least logging when cooldown exceeds 1 day.

### F21 — `trade_guards.py:97` Time-decay halving silently goes below 1.
**P2.** `base = max(1, base >> halvings)` — guards against multiplier going below 1. But because `>>` is bit-shift on int, `8 >> 4 = 0`, then `max(1, 0) = 1`. OK in this branch. But if `LOSS_ESCALATION` is later changed to floats (e.g. 1.5x escalation), `>> halvings` fails with TypeError. Fragile; prefer `base / (2 ** halvings)` with `max(1, ...)`.

### F22 — `monte_carlo.py:88` `p_up = max(0.01, min(0.99, p_up))` clamp throws away high-conviction signals.
**P2.** A 0.99 cap means: even when the entire signal stack screams BUY with 99.5% confidence, the simulation drift is capped. For tail-risk simulations this is fine; for "what's my expected exit price?" computations this systematically biases recommendations toward HOLD.

### F23 — `monte_carlo_risk.py:188-189` Position filter is `shares != 0`, then matrix-extraction line 195 uses `all_tickers.index(t)` — O(n²) and silently wrong if `positions` keys are reordered (Py 3.7+ preserves order but the algorithm shouldn't rely on it).
**P2.** Use enumerate over `positions.items()` to get the index directly. Bug latent if anyone wraps positions in a class with different iteration order.

### F24 — `risk_management.py:756-770` Concentration computes `existing_value` using current price BUT proposes new `proposed_alloc` from `total_value * alloc_pct` — mixes cost basis and live mark.
**P2.** `existing_value = existing_shares * existing_price * fx_rate` uses CURRENT price (line 770) — that's correct for "what's my concentration". But the `proposed_alloc = min(total_value * alloc_pct, cash)` is a future commitment in cash; adding them and comparing to `total_value` over-counts by `(price_now - price_at_buy) * shares`. For appreciated holdings this overstates concentration; for depreciated, understates. Subtle but real.

### F25 — `risk_management.py:842-844` Correlation check fires on any held correlated ticker regardless of size.
**P2.** Holding 1 share of MSTR triggers a "correlated with BTC" warning even at <0.1% portfolio weight. Threshold should be size-weighted, e.g. flag only if combined weighted exposure > 10%.

### F26 — `backtester.py:92-95` `_build_accuracy_data` uses full-history blend (TODO comment acknowledges look-ahead).
**P2.** Documented in code, but ANY backtest result from this module is contaminated by look-ahead bias. As long as this module is gated behind dev-only use, it's P2. If anything reads the output for live decisions, escalate to P0.

---

## P3 — Minor

### F27 — `cost_model.py:50-51` `total_cost_pct()` ignores min_fee_sek silently — comment acknowledges but doesn't expose it.
**P3.** `total_cost_pct() = (courtage_bps + spread_bps + slippage_bps) / 100`. For STOCK_COSTS, min_fee_sek=1.0 — at 1000 SEK trade that's an extra 10 bps (10% added cost). Doc-only "excluding min fee" is silently misleading callers using this as a quick estimate. Either add a `trade_value_sek` parameter or rename to `total_cost_pct_no_min`.

### F28 — `exposure_coach.py:108-117` `compute_exposure_recommendation` does not validate `score` field type — passes through whatever `market_health.get("score", 50)` returns.
**P3.** If `score` is a string or None due to upstream bug, returned dict carries the bad value and consumers crash later. Coerce to float here.

### F29 — `decision_outcome_tracker.py:69` `now < target_dt` skip is correct, but no logging when nothing is backfilled — operator can't distinguish "no data ready" from "tracker silently broken".
**P3.** Add INFO-level summary `"processed=N skipped=K ready=R"` so dashboards can confirm liveness.

### F30 — `cumulative_tracker.py:79` Bare `except (OSError, IndexError)` swallows IndexError from `lines[-1]` — fine — but does not log. If snapshots file is truncated by external process, tracker silently degrades to "no last_ts" and starts producing duplicate snapshots.
**P3.** Add a warning log on the IndexError branch.

---

## Summary

Most critical: F01-F04 are real money exposure paths. F01 (cross-process lock) is the same class of bug as the March 3 stop-loss-API incident — coordination assumption that doesn't hold in a multi-process world. F03 (warrants no lock at all) is the biggest single risk because warrant positions are leveraged. F15 (BULL/BEAR sign) is silent P&L corruption today.

Drawdown/Sharpe math (F05-F10) has multiple inconsistencies that erode confidence in the dashboards' performance numbers. Monte Carlo (F10-F12) has a 252-vs-365 mismatch and copula correlation under-binding that quietly underestimates VaR for the correlated crypto+metals+MSTR book.

Recommendation: prioritize F01, F03, F15, F12, F10 first — these directly affect money-at-risk numbers the user sees.
