# Adversarial Review — Portfolio-Risk Subsystem

Scope: portfolio_mgr, portfolio_validator, risk_management, equity_curve, monte_carlo, monte_carlo_risk, kelly_sizing, circuit_breaker, trade_guards, trade_validation, trade_risk_classifier, cost_model, warrant_portfolio.

---

[P0] warrant_portfolio.py:215 — `record_warrant_transaction` does NOT use `update_state`/lock; concurrent BUY+SELL across threads (metals_loop fast-tick + main loop) read-modify-writes `holdings` racy → lost units, miscounted entries. Also writes via `atomic_write_json` but the read-modify in between is unprotected. | FIX: wrap entire load→mutate→save under a per-file `threading.Lock` like `portfolio_mgr.update_state`; or use `update_state` pattern.

[P0] warrant_portfolio.py:182-214 — Transaction record has NO `reason` field, violating CLAUDE.md "Log every trade with a reason in the transaction record." Warrant trades go to journal without rationale. | FIX: add required `reason` parameter to `record_warrant_transaction` and persist it on the txn dict.

[P0] kelly_sizing.py:269-323 — Position sizing uses `cash_sek * alloc_frac` and `half_kelly * cash_sek` instead of `total_portfolio_value * frac`. After drawdown, "cash * 30%" of a portfolio that's 80% deployed is much smaller than 30% of total value, but after a series of SELLs it explodes (cash spikes). Concentration check in `risk_management.check_concentration_risk:751-764` correctly uses `total_value`, but Kelly sizing doesn't — the two are inconsistent and Kelly will recommend more than the concentration cap allows. | FIX: compute total portfolio value (cash + holdings × price × fx) before applying alloc_frac, and use min(cash_sek, total_value × alloc_frac).

[P0] kelly_sizing.py:326 — `if rec_sek < 500: rec_sek = 0` — but the project rule (CLAUDE.md / portfolio-risk.md) is **min 1000 SEK per leg** (Avanza minimum courtage threshold). Kelly will recommend 500-999 SEK trades that fail validation in `trade_validation.validate_trade` (default 500) or below Avanza minimum. | FIX: change threshold to 1000; align with `trade_validation.min_order_sek` default (also 500 — bump that too).

[P0] trade_validation.py:32 — `min_order_sek: float = 500.0` default contradicts portfolio-risk rule of 1000 SEK Avanza minimum. Trades 500-999 SEK pass validation and get rejected by Avanza. | FIX: default `min_order_sek=1000.0`.

[P0] risk_management.py:367-369 — ATR stop placement: `stop_price = entry_price * (1 - 2*atr_pct/100)` with `atr_pct = min(atr_pct, 15.0)`. For a 5x silver MINI warrant the **knockout barrier is the binding constraint**, not 2×ATR — placing a stop near a 2×ATR distance can be ABOVE the knockout barrier (worse) or place a stop-loss within the explicit 3% prohibited zone documented in MEMORY (`feedback_mini_stoploss.md`). The code has zero awareness of warrant barriers. | FIX: for tickers in warrant config, compute distance to knockout barrier and clamp `stop_price` to max(barrier × 1.03, 2×ATR). Cross-reference `data/oil_warrant_catalog.json` / silver warrants to fetch barrier per holding.

[P0] risk_management.py:233 — `load_json(portfolio_path, default={})` returns empty dict on corrupt file → `initial_value = 500_000`, `holdings = {}`, `current_value = 500_000` (cash_sek default), drawdown = 0%. **Corrupt file silently bypasses the circuit breaker.** Compare with portfolio_mgr.py:90 which logs CRITICAL on corruption — risk_management never sees that path because it uses `load_json` directly with a benign default. | FIX: use `_load_state_from()` from portfolio_mgr (which returns CRITICAL log + backup recovery) instead of raw `load_json(default={})`.

[P0] equity_curve.py:495 — `losses = [t for t in trips if t["pnl_pct"] <= 0]` and `wins = [t for t in trips if t["pnl_pct"] > 0]`. Streak counter at line 511-519 uses `t["pnl_pct"] > 0` for wins, else loss — **a flat (pnl_pct == 0.0) round-trip counts as a LOSS in streaks but is excluded from losses for win_loss_ratio**. Inconsistent classification skews max_consecutive_losses upward and `expectancy_pct` down. Plus a true 0% net trade after fees is rare but possible. | FIX: pick one convention (`pnl_pct > 0`/`< 0`/`== 0` separately) and use uniformly.

[P0] equity_curve.py:262-282 — `compare_strategies` decides "leader" based on `total_return_pct` strict `>`. With identical zero-trade portfolios (both at initial value), tie defaults to "bold" silently. Cosmetic but propagates to dashboard mis-attribution. | FIX: explicit tie-breaker or "tie" string.

[P0] monte_carlo_risk.py:431 — `fx_rate = agent_summary.get("fx_rate", FX_RATE_FALLBACK)` — does NOT route through `_resolve_fx_rate` (the validated fallback chain in risk_management.py:121). If `agent_summary["fx_rate"]` is the legacy `1.0` literal (still seen in stale summaries), VaR/CVaR SEK is reported 10× too small — a 50K SEK risk reads as 5K. The whole point of P1-15 fix in risk_management was to defeat this; here it's regressed. | FIX: import and use `risk_management._resolve_fx_rate(agent_summary)`.

[P0] portfolio_mgr.py:166-180 — `portfolio_value` accepts `prices_usd` dict but does not apply leverage for warrant tickers. If `holdings` contains a MINI warrant key, `shares × price × fx` ignores the leverage factor entirely → completely wrong portfolio value. Warrants are tracked separately in warrant_portfolio.py but if any code path stuffs warrants into the main holdings (and grep shows trackers like `XBT-TRACKER` in CLAUDE.md Tier 3), the value is misvalued. | FIX: detect leveraged instruments (lookup config) or document hard separation; raise on unexpected ticker.

---

[P1] risk_management.py:286-298 — NaN/Inf guard returns `breached: True` with `peak_value: 0.0` and `current_value: 0.0` when both non-finite. But if `current_value` alone is NaN/Inf and `peak_value` is finite, the round-to-zero on the bad side still returns 100% — correct intent but **the rounded-to-0 numbers in the dict are lossy and downstream alert messages report `current=0`** which an oncall would read as "portfolio went to zero". | FIX: include the original raw values in a `raw` field, or send `current_value=None`.

[P1] risk_management.py:301-302 — Drawdown formula uses `peak_value > 0` guard, but if `peak_value < 0` (impossible normally, but if `_streaming_max` reads garbage with negative `bold_value_sek`) the breaker silently returns 0% drawdown. | FIX: also check `peak_value > 0` AND `current_value >= 0`; treat `current_value < 0` as breached.

[P1] risk_management.py:43-110 — `_streaming_max` uses cached byte-offset; if a writer truncates and rewrites the file (e.g., manual recovery / log rotation that copies to .bak then writes fresh), `file_size >= cached["offset"]` is True (file rewritten to similar size) and seek skips the new content entirely → peak stuck at old value, drawdown circuit breaker reads stale peak. | FIX: also detect modification time (`stat().st_mtime`) or first-line hash; invalidate cache on mismatch.

[P1] risk_management.py:243-264 — `agent_summary` stale path falls back to **cash-only** value but the comment says "NOT truly conservative". With holdings underwater, drawdown shows tiny → circuit breaker never trips. The warning is logged but the function still returns "looks fine". | FIX: when feed is stale AND holdings exist, return `breached: True` (fail-safe) OR persist last-known prices for fallback valuation.

[P1] risk_management.py:367-368 — `atr_pct = min(atr_pct, 15.0)` — silent cap. A 20% ATR is real (silver capitulation, BTC -30% day) and capping makes the stop tighter than ATR mathematics implies → premature exit on real volatility expansion. | FIX: log when capped; or use a regime-conditional cap (high-vol regime → 25%).

[P1] risk_management.py:898-905 — `distance_in_atr = (current_price - stop_price) / atr_value`. By construction `stop_price = entry_price × (1 - 2×atr_pct/100)`, so `current - stop = (current - entry) + 2×atr_pct/100 × entry`. When current ≈ entry, distance is ~2 ATR (correct). But this is computed using `entry_price`, then divided by `atr_value` based on `current_price` — mismatched bases. If current diverges from entry by 30%, the ATR-distance metric is biased ~30%. | FIX: use entry_price-based atr_value for consistency.

[P1] risk_management.py:546 — `age_days = age_hours / 24` — if `tx.get("timestamp")` parses but is far in future (corrupt log), `age_delta` is negative → negative age. | FIX: assert `age_hours >= 0`; clamp/log warning.

[P1] equity_curve.py:243-249 — Sharpe calculation duplicates std-dev computation (one at line 230 then again at line 243). The second one re-uses `daily_rets_dec` / `mean_dec` correctly, but the first `daily_vol` (line 231) is computed with `(r - mean_ret)` where `mean_ret` is in **percentage units** but `daily_rets_dec` is in decimal. The variance/vol at line 230 is in pct² units; volatility_annual_pct correctly stays as %. **The Sharpe at line 249 is correct (decimal/decimal), volatility_annual_pct is in %.** No bug per se, but the dual computation is fragile. | FIX: consolidate; compute once in decimals then convert for display.

[P1] equity_curve.py:236 — Risk-free conversion `daily_rf = RISK_FREE_RATE_ANNUAL / ANNUALIZATION_DAYS = 0.035/365`. Daily rf is a **simple-rate** approximation; correct geometric daily-rf is `(1+r)^(1/365)-1`. For 3.5% the error is sub-bp, but for 10%+ rates it grows. Compounding correctness matters when annualizing back. | FIX: `daily_rf = (1 + RISK_FREE_RATE_ANNUAL) ** (1/365) - 1` for principle.

[P1] equity_curve.py:217-220 — `if dd > 0.01:` — meaningfully below peak threshold of 0.01% (1bp) is too tight; floating-point noise on a 500K portfolio is ~1 SEK = 0.0002%, so this is OK, but the comment says "Meaningfully below peak" which is misleading. Cosmetic. | FIX: comment clarification or raise threshold to 0.1%.

[P1] equity_curve.py:404 — Sell fee allocated as `(sell_fee × matched / sell_shares)` — but for a SELL that closes only PART of a buy lot AND part of the sell is matched against a different buy, `matched < sell_shares` and the allocation is fractional. Correct. But if the SAME sell_tx is processed multiple times (multiple iterations of the inner while loop), each iteration accrues `sell_fee × matched/sell_shares` — and `matched` sums to `sell_shares` total over all iterations → total allocated sell fee equals `sell_fee`. Correct after all iterations. **However, if the loop breaks early (insufficient buy shares), `sell_fee × matched_so_far/sell_shares < sell_fee` → unallocated fee residue is silently dropped.** | FIX: assert post-loop that fees are fully allocated, or warn on residual.

[P1] equity_curve.py:494-495 — `wins = [t for t in trips if t["pnl_pct"] > 0]` uses `pnl_pct` (gross price-%, not net of fees). But P0-6 (line 405) made `pnl_sek` net of fees. `expectancy_pct` and `win_loss_ratio` use `pnl_pct` → expectancy is **gross**, but `total_pnl_sek` and `profit_factor` are **net**. Misleading: a strategy can show positive `expectancy_pct` while `total_pnl_sek` is negative due to fees. | FIX: document the gross/net split clearly OR derive a net `pnl_pct_net`.

[P1] kelly_sizing.py:106 — `if len(pnl_list) < 2: return None` — but with only 1 historical round-trip, the function cascades into ATR-based estimates. Also: line 109 uses `pnl_list` (which holds gross pnl_pct based on price diff with ZERO fee deduction at line 103: `(sell_price_per_share - avg_buy_price)/avg_buy_price`). Kelly inputs are **gross**, biasing avg_win up and avg_loss down → Kelly fraction over-allocates by fee×2 worth. | FIX: deduct fees from each pnl_pct entry using transaction fee_sek field.

[P1] kelly_sizing.py:91-95 — Weighted avg buy price uses **all BUYs**, not just BUYs preceding the SELL — anti-FIFO. A SELL early in the history paired against later BUYs gives nonsense P&L (zero or negative when actual was positive). | FIX: process transactions chronologically with FIFO matching like equity_curve._pair_round_trips.

[P1] monte_carlo.py:62-65 — `volatility_from_atr` annualizes using `sqrt(252/14)`. But ATR is on **hourly** candles per the docstring, so the right factor is `sqrt(252×24/14)` for hourly→annual, or `sqrt(252/14)` only if ATR is daily. The system mixes timeframes; the function takes any ATR%-period and naively scales by 252. If ATR comes from 1h candles (BTC/ETH primary), volatility is understated by ~sqrt(24)≈4.9× → MC stop probabilities massively under-report tail risk. | FIX: pass candle period (hours/days) explicitly and scale `sqrt(annual_periods/period)` correctly.

[P1] monte_carlo.py:68-97 — `drift_from_probability` clamps p_up to [0.01, 0.99] but uses `mu = sigma × z × sqrt(252) + 0.5×sigma²`. The horizon T is fixed at 1 (year) here implicitly — the formula assumes annual drift = sigma × z(p_up) × sqrt(252) which is wrong because z(p_up) measures probability over 1 day, not 1 year. To recover annual mu from a 1-day p_up: `mu_annual = (sigma × z × sqrt(1/252) + 0.5×sigma²) × 252`. The current formula says `mu = sigma × z × sqrt(252) + 0.5×sigma²` which over-amplifies the drift by sqrt(252)² = 252×. | FIX: derive correctly: for horizon T (years), `mu = z×sigma/sqrt(T) + 0.5×sigma²`. With T=1/252, `mu = z×sigma×sqrt(252) + 0.5×sigma²` — actually consistent. **Reverify**: maybe correct. MAYBE.

[P1] monte_carlo.py:152 — `rng = np.random.default_rng(self.seed)`. With `seed=None`, `default_rng` uses fresh entropy each call — antithetic variates only span half of `n_paths` per `simulate_paths` call, but **multi-call consistency** requires storing the rng. Currently `simulate_paths` is idempotent only because `_terminal_prices` is cached after first call — second call would create new rng → different output. Not a bug if guarded, but the API is footgun. | FIX: store rng on self; only re-init on explicit reset.

[P1] monte_carlo_risk.py:212 — `[t for t, p in positions.items() if p.get("shares", 0) != 0]` — `shares != 0` includes negative shares (short positions). Portfolio system has no shorting per CLAUDE.md, but if a transaction-recording bug ever produced negative shares (validator checks for it but ad-hoc state writes might bypass), VaR sims silently include them with reversed P&L sign. | FIX: filter `shares > 0` consistent with rest of system; warn on negative.

[P1] monte_carlo_risk.py:217-220 — `all_tickers = list(positions.keys())` then `indices = [all_tickers.index(t) for t in self._tickers]` — `positions` was filtered first (line 213) so `all_tickers == self._tickers` always; the index permutation is identity but the input correlation_matrix `corr` was built for the ORIGINAL `positions` ordering (passed in by `compute_portfolio_var` which iterates `holdings`). **Side-effect: only consistent because `compute_portfolio_var` filters tickers identically before building corr.** If a caller passes in positions with zero-share entries AND a corr built for filtered list, indexing is wrong. Fragile contract. | FIX: rebuild correlation sub-matrix from input ticker list explicitly.

[P1] monte_carlo_risk.py:393-400 — `total_value = sum(shares × price)` for `drawdown_probability`, then compares P&L to `-total_value × threshold/100`. But `total_value` is **gross long exposure**, not portfolio NAV (which is cash + holdings). For a portfolio with 100K cash and 100K holdings, a 5% drawdown on holdings is `-5K`, threshold is `-100K × 0.05 = -5K` → matches. But the user might want 5% of NAV (200K) = -10K. Misleading. | FIX: pass NAV or document "% of holdings exposure".

[P1] portfolio_mgr.py:35-41 — `_get_lock` uses `_locks_lock` to gate creation, but two threads can both pass `if key not in _state_locks` because they hold the same `_locks_lock` — actually NO, the inner block is under the lock, so creation is atomic. Correct. **However**: `_state_locks` keyed by `str(path)` — different absolute/relative paths (e.g. `data/portfolio_state.json` vs `Q:/finance-analyzer/data/portfolio_state.json`) hash to different keys → two locks, race regained. | FIX: normalize via `path.resolve()` before stringifying.

[P1] portfolio_mgr.py:108-113 — `_save_state_to` rotates backups then atomic-writes. **`_rotate_backups` uses `shutil.copy2` which is NOT atomic** — between copy and the atomic_write_json's rename, a crash leaves the .bak file mid-copy. Less critical than the main file but recovery code reads .bak; corrupt .bak masks the issue. | FIX: use `os.replace` chain on backup files (rename is atomic on same FS) instead of copy2.

[P1] portfolio_validator.py:243 — `avg_diff_pct > 1.0` tolerance (1%) for avg_cost_usd; for warrants where each unit is ~10 SEK and avg_cost_usd is small, 1% is a loose check. Also: weighted avg uses **all BUYs ever**, ignoring SELLs that should reduce the cost basis (FIFO would re-set avg). | FIX: account for FIFO-consumed lots when computing expected avg.

[P1] trade_guards.py:283-285 — `consecutive_losses[strategy] += 1` happens on `pnl_pct < 0`. But `pnl_pct == 0` (break-even after fees) decrements via the else branch (`= 0`). Inconsistent with equity_curve P0 above; semantic ambiguity. | FIX: choose: <= 0 counts as loss, or strictly < 0; document.

[P1] trade_guards.py:297-310 — Pruning happens INSIDE BUY branch only; `new_position_timestamps` accumulates indefinitely if SELL-only sequences run. After a recovery from drawdown (only SELLs), the BUY rate-limit window has stale timestamps. Eventually `recent` filter at line 200-211 drops them — but the on-disk list grows unbounded. | FIX: prune in record_trade unconditionally, not only on BUY.

[P1] trade_guards.py:148 — `last_trade = datetime.fromisoformat(last_trade_str)` — no `.replace("Z", "+00:00")` like at line 89. If `now_str` was written via `datetime.now(UTC).isoformat()` (Python 3.12+) it has `+00:00` not `Z`, so OK; but if any external producer writes Z-suffixed, this raises ValueError → caught silently → cooldown bypassed. | FIX: apply Z-to-+00:00 substitution consistently.

[P1] circuit_breaker.py:97-99 — On OPEN→HALF_OPEN transition, `allow_request` returns True AND sets `_half_open_probe_sent = True`. But the **probe doesn't actually call `record_success` or `record_failure`** automatically — caller has to. If caller crashes/forgets, breaker stays HALF_OPEN forever blocking further requests at line 106. | FIX: HALF_OPEN with stale probe should auto-revert to OPEN after 2× recovery_timeout; or document caller contract loudly.

[P1] cost_model.py:36-47 — `total_cost_sek` returns `courtage + spread + slippage`. `min_fee_sek=1.0` for stocks but **Avanza Mini courtage is actually 39 SEK minimum** for many tiers; 1 SEK is unrealistic and underestimates real costs for sub-50K SEK trades. CLAUDE.local.md says 1000 SEK min order; if courtage floor is 1 SEK that means 0.1% min fee — 39 SEK / 1000 SEK = 3.9%! | FIX: verify Avanza tier fees; for 1000 SEK trade a 39 SEK floor is unsurvivable. Set `min_fee_sek=39.0` for STOCK_COSTS or per actual broker tier.

[P1] warrant_portfolio.py:99-103 — `current_implied_sek = entry_price_sek × (1 + implied_pnl_pct)`. With 5x leverage and -25% underlying move, `implied_pnl_pct = -1.25` → `current_implied_sek = entry × -0.25` = NEGATIVE. **Warrant prices cannot be negative; they hit knockout barrier first.** This silently reports negative warrant valuations on big underlying moves. | FIX: floor at 0 with explicit "knocked out" flag; integrate barrier from warrant config to detect actual knockout.

[P1] warrant_portfolio.py:80 — `if not holding or not current_underlying_usd or not fx_rate` — uses falsy check. `current_underlying_usd = 0.0` is "missing data" (correct skip), but a legitimate underlying price near zero (silver crash) would also be skipped. Silver below 1 USD is impossible currently but the pattern is fragile. | FIX: explicit `is None` check.

---

[P2] portfolio_mgr.py:21-26 — `_DEFAULT_STATE` has no `total_fees_sek` field; `portfolio_validator` then reports "Missing or null field: total_fees_sek" on a brand-new portfolio. | FIX: include `total_fees_sek: 0` in defaults.

[P2] portfolio_mgr.py:65-75 — `_validated_state` does NOT validate `cash_sek` is numeric/non-negative. A corrupt save with `cash_sek: "five hundred thousand"` propagates. | FIX: add type/range checks aligned with portfolio_validator.

[P2] portfolio_validator.py:191-193 — `required_tx_fields = [..., "reason"]` enforces reason on every txn. Good. But warrant_portfolio.record_warrant_transaction does NOT add a reason (P0 above). Validator would never run on warrant file though, so the gap is unobserved. | FIX: add validate_warrant_state covering the warrant transaction format.

[P2] risk_management.py:543 — `first_buy_ts.replace(tzinfo=datetime.UTC)`. This shadows an awareness check: if `tx_str` is naive, code applies UTC arbitrarily — but transactions in this system are all UTC by convention (CLAUDE.md). MAYBE OK in practice. | FIX: log if naive timestamp encountered.

[P2] equity_curve.py:537-558 — Calmar uses round-trip-only equity curve, ignoring unrealized drawdowns. A position held through a 50% drawdown then recovered shows 0 round-trip drawdown → infinite Calmar. | FIX: combine round-trip equity with marked-to-market portfolio_value_history.

[P2] equity_curve.py:65-110 — `_daily_returns` groups by date taking last value per day. With irregular logging (gaps, multiple per minute, etc.), "last per day" is sensitive to the timezone of `dt.date()`. UTC date for crypto is fine but for stocks an EOD value at 22:00 CET is the next UTC date. Mixed-timezone bias. | FIX: align bucket boundary to a fixed market-anchored hour.

[P2] monte_carlo.py:154 — `T = self.horizon_days / 252.0` — using 252 for crypto which trades 365 days/year underestimates time. Volatility scaling should use the same trading-day basis as drift derivation. | FIX: pass annualization basis (252 vs 365) per asset class.

[P2] monte_carlo.py:217-219 — `if threshold <= 0: return 0.0` for `probability_below`. But `probability_above(threshold <= 0)` returns 1.0 — symmetric and OK in spirit, but `probability_below(0)` = 0 is wrong if any simulated price went sub-zero (which GBM can't, but log-normal floor isn't checked). MAYBE harmless. | FIX: drop the early return and use the np.mean correctly.

[P2] monte_carlo_risk.py:114-129 — `CORRELATION_PRIORS` has 8 NVDA/AMD/AMZN-class pairs but per CLAUDE.md, all those tickers were REMOVED Apr 9 2026 ("Removed Apr 09: PLTR, NVDA, MU, SMCI, TSM, TTWO, VRT" + earlier removals). Dead code; misleads anyone reading. | FIX: remove obsolete priors, keep only active tickers.

[P2] monte_carlo_risk.py:96 — `eigenvalues = np.maximum(eigenvalues, 1e-8)` — clip floor at 1e-8 is fine for 4-asset matrices; for a 20+ asset matrix this can leave negligible negative-noise leakage producing tiny imaginary parts in downstream computations. | FIX: use `max(1e-8, eigenvalues.max() × 1e-12)`.

[P2] kelly_sizing.py:39-42 — `if win_prob <= 0 or win_prob >= 1: return 0.0`. A win_prob of exactly 1.0 (riskless arb) returns 0 — should return full Kelly = 1. Edge case but mathematically wrong. | FIX: if `win_prob >= 1` return 1.0; if `win_prob <= 0` return 0.0.

[P2] kelly_sizing.py:323 — `min(half_kelly × cash_sek × exposure_ceiling, max_alloc)`. `exposure_ceiling` from agent_summary scales the BET, but `max_alloc` is unscaled. If exposure_ceiling=0.5 (half-defensive), the rec_sek scales correctly but max_alloc doesn't reflect the ceiling. Conceptual confusion, not a math bug. | FIX: scale max_alloc too if intent is "exposure ceiling caps everything".

[P2] trade_guards.py:50-69 — `_portfolios_have_transactions` reads three files synchronously every call to `get_all_guard_warnings`. ~3 disk hits per signal cycle × 60s loop = wasteful. | FIX: cache result with a 60s TTL or trigger only on first dry call.

[P2] trade_guards.py:96-97 — `halvings = int(elapsed_hours // LOSS_DECAY_HOURS)`, then `base = max(1, base >> halvings)`. For `base=8, halvings>=3`, result is 1. For `halvings=10`, `base >> 10 = 0`, max(1, 0) = 1 — fine. **But `base` is an int from LOSS_ESCALATION; if config injects a float, `>>` raises TypeError.** | FIX: explicit int conversion / use division.

[P2] trade_validation.py:32 — `max_cash_pct: float = 50.0` — for the **bold** strategy, `alloc_frac = 0.30` (kelly_sizing.py:269) so a single trade is 30% of cash, well under 50%. But the trade_validation isn't strategy-aware; bold could pass through 50% via direct call, exceeding its policy. | FIX: pass strategy or explicit `max_cash_pct` per strategy at call site.

[P2] trade_risk_classifier.py:69-77 — `position_pct > 20: score += 3` — positioning thresholds ignore strategy. Bold's 30% allocation is "large" by the classifier (3 points), but bold's stated policy is 30%. Dual-system inconsistency. | FIX: thresholds parametrized by strategy.

[P2] cost_model.py:64-70 — Warrant `spread_bps=40` half-spread. CLAUDE.md / memory says MINI silver spread is 0.6-1.0%, half ≈ 30-50bps — OK. But oil warrants and 5x XAU products have wider spreads (50-80 bps half). Single-warrant cost model masks instrument variability. | FIX: per-warrant cost overrides via config.

[P2] warrant_portfolio.py:32-33 — `if state is None: return _DEFAULT_STATE.copy()`. **Shallow copy of a dict containing dicts/lists** — mutations to `state["holdings"]` mutate the module-level default, polluting subsequent loads. `holdings` is `{}` initially so first mutation creates a new key (no aliasing), but if a future change adds nested defaults, this is a footgun. | FIX: `copy.deepcopy(_DEFAULT_STATE)`.

[P2] warrant_portfolio.py:80 — `if not holding or not current_underlying_usd or not fx_rate: return None` — silently swallows. No log. Operators have no breadcrumb for "warrant pnl unavailable". | FIX: logger.warning on each missing input.

---

[P3] portfolio_mgr.py:108-113 — `_save_state_to` is the only writer that rotates backups, but `update_state` (line 136) also rotates AFTER its own mutate — two consecutive saves would create two rotations and could lose the genuine pre-mutation backup. Cosmetic. | FIX: factor a `_persist_with_backup` helper, share between paths.

[P3] equity_curve.py:104 — `if prev_val > 0:` — divide-by-zero guard, but `prev_val == 0` is structurally impossible (filtered at line 91 `if value > 0`). Dead branch. | FIX: assert or remove.

[P3] equity_curve.py:198 — `peak = values[0]` — initializes peak to first observed value rather than initial portfolio value. If first datapoint is post-loss, max_drawdown understates. | FIX: peak = max(values[0], INITIAL_VALUE).

[P3] monte_carlo.py:193-203 — `np.percentile(self._terminal_prices, percentiles)` — uses default linear interpolation. For 10K paths this is fine; documenting choice would help. | FIX: comment.

[P3] monte_carlo_risk.py:280 — `U = t_dist.cdf(T_samples, df=self.df)` followed by `Z_marginal = norm.ppf(U[:, i])` per asset. **If U contains exactly 0.0 or 1.0 (rare but possible at heavy tails), `norm.ppf(0)=-inf, norm.ppf(1)=+inf` → NaN GBM returns → NaN portfolio P&L → NaN VaR.** | FIX: clip U to (eps, 1-eps).

[P3] kelly_sizing.py:319-320 — `exposure_ceiling = exposure_rec.get("exposure_ceiling", 1.0)`. Default 1.0 means "no limit" — fail-open. If agent_summary missing the key, no de-risking applied. | FIX: log warning when missing; default 0.7 fail-safe.

[P3] circuit_breaker.py:64-72 — Backoff doubles on each HALF_OPEN failure, capped at max_recovery_timeout. Fine, but **resets to `_base_recovery_timeout` only on `record_success`** — never on extended sleep. A breaker that hits max once stays at max until success, even after a calm month. | FIX: optional decay on long quiet period.

[P3] trade_guards.py:78-81 — `consecutive_losses >= 4: base = LOSS_ESCALATION[4]`. Caps at 8x (24h decay halving), but for a strategy in deep loss streak this **lifts the cooldown to weeks** (8x × 30min = 4h base). For a market-making style, 4h is huge. Cosmetic / by design. | FIX: document.

[P3] trade_validation.py:96-102 — `max_price_deviation_pct: float = 5.0` — for crypto in capitulation, a 10% move in 60s is real. Validation rejects → trade blocked. Defensible default but worth surfacing. | FIX: comment / regime-aware override.

[P3] trade_risk_classifier.py:20-26 — `_REGIME_SCORES` has no entry for "trending-up-strong" or other regime variants. Unknown regimes silently score 0 (line 81: `_REGIME_SCORES.get(regime_lower, 0)`). | FIX: log unknown regime.

[P3] cost_model.py:50-55 — `total_cost_pct` excludes min_fee, but `total_cost_sek` includes it — divergent semantics. Caller comparing pct to sek can get inconsistent numbers for small trades. | FIX: docstring spelling out the exclusion.

[P3] warrant_portfolio.py:225-227 — `avg_price = (old_units × old_price + units × price_sek) / new_units`. **No fee inclusion in average cost.** When summing entry costs you also paid spread/courtage at entry; ignoring it understates cost basis → realized P&L on close looks better than reality. | FIX: add fee allocation to entry_price_sek (or maintain separate `cost_basis_sek` field).

[P3] warrant_portfolio.py:142 — Returns `{"positions": {}, "total_value_sek": 0, "total_pnl_sek": 0}` with no error indication when a holdings entry has missing underlying ticker. Silent skip via line 150-151 `if not underlying: continue`. | FIX: log.

[P3] portfolio_validator.py:106-137 — Holdings vs transactions reconciliation tolerates 1% relative diff for closed positions but not open ones. An open position with 1% rounding error escapes detection. | FIX: same tolerance for both states.

---

COUNT: P0=11, P1=27, P2=20, P3=14, total=72
