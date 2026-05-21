# Portfolio-Risk Adversarial Review — 2026-05-21

Baseline: `604f0ef1` on `review/2026-05-21-fgl`.

Files in scope:
- portfolio/portfolio_mgr.py
- portfolio/portfolio_validator.py
- portfolio/trade_guards.py
- portfolio/risk_management.py
- portfolio/equity_curve.py
- portfolio/monte_carlo.py
- portfolio/monte_carlo_risk.py
- portfolio/exit_optimizer.py
- portfolio/price_targets.py
- portfolio/warrant_portfolio.py
- portfolio/cost_model.py
- portfolio/trade_risk_classifier.py

## Findings

### Critical (P0)

portfolio/equity_curve.py:346-405: P0: FIFO round-trip P&L double-counts buy fees. `price_per_share = total_sek / shares` is computed from the BUY's `total_sek`, which per `portfolio_validator.py:71-72` is "full allocation (including fee)". So `buy_price` already embeds the fee per share. Then `pnl_sek = (sell_price_per_share - buy_price) * matched - buy_fee_share - sell_fee_share` subtracts the buy fee a second time. On a 10K SEK BUY with 50 SEK fee, the per-share price is inflated and an additional ~50 SEK is deducted from pnl_sek — double-charging the buy fee. Symmetric on the sell side is fine because sell `total_sek` is net-of-fee per the same comment. Fix: derive `buy_price_per_share = (total_sek - fee_sek) / shares` so `(sell_per_share - buy_per_share) * matched` already represents gross-of-fee move, then subtract both `buy_fee_share + sell_fee_share` once. Tests `TestPnlSekNetOfFees` / `TestProfitFactorNetOfFees` claim coverage but appear to verify the present (buggy) arithmetic.

portfolio/warrant_portfolio.py:96-103: P0: warrant_pnl produces NEGATIVE implied value past 100% loss. With 5x leverage on a 25% underlying drop, `implied_pnl_pct = -1.25` and `current_implied_sek = entry_price_sek * (1 + -1.25) = -0.25 * entry_price_sek`. The function returns a NEGATIVE `total_value_sek`, inverting the portfolio value calc and silently understating the loss (or even showing fictional gain in summary aggregation). MINI warrants knock out at 0, never below. Fix: `current_implied_sek = max(0.0, entry_price_sek * (1 + implied_pnl_pct))` matching the clamp in `exit_optimizer._compute_pnl_sek:323`. Also surface a `knocked_out: True` flag so callers can detect dead positions instead of seeing a tiny positive value after a clamp.

portfolio/warrant_portfolio.py:1-266: P0: warrant_portfolio has NO concept of `financing_level` / knock-out barrier. The CLAUDE.local.md hard rule "NEVER place a stop-loss within 3% of current bid" / "near MINI warrant barriers" is enforced only in `exit_optimizer._compute_risk_flags:373-379` and `_apply_risk_overrides:430-436`. But the warrant *portfolio* model never carries the barrier price. A holding dict has `leverage` but no `financing_level`. So any consumer that imports `warrant_portfolio.warrant_pnl()` to make stop-loss decisions has zero awareness of the barrier and could place a stop right at it. Fix: persist `financing_level_usd` per holding in `record_warrant_transaction` and expose it from `get_warrant_summary`; add a `barrier_distance_pct` to each position's returned dict; refuse any stop within 3% of barrier at the API boundary (not just the exit optimizer).

portfolio/portfolio_mgr.py:21-26: P0: `_DEFAULT_STATE` omits `total_fees_sek`. `portfolio_validator.py:50-52` flags any state missing the field as an error, and `transaction_cost_analysis` / `journal.py:193` read it. A fresh portfolio created via `update_state` / `load_state` therefore validates as broken from cycle 1 until something writes the key. Fix: add `"total_fees_sek": 0` to `_DEFAULT_STATE` alongside `cash_sek`.

portfolio/portfolio_mgr.py:136-159: P0: `update_state` lock is BYPASSED by every legacy `load_state`/`save_state` caller. The `threading.Lock` only serializes within the `update_state` body. `portfolio/main.py:783` runs `save_state(state)` outside any lock; `daily_digest.py:220` runs `load_state()` outside any lock; Layer 2 subprocess reads via `load_json` directly. Two `update_state` calls cannot race each other, but `update_state` AND a legacy `save_state` CAN race and the latter wins, losing the mutation. Fix: route all writes through `update_state`; deprecate raw `save_state` / `load_state` for mutators; or make `save_state` acquire the same per-file lock. (Already filed as H-A1 in docs/AGENT_REVIEW_PORTFOLIO_RISK.md.)

portfolio/equity_curve.py:337-353: P0: `_pair_round_trips` assumes transactions list is chronological. The validator does NOT enforce timestamp ordering — duplicate-timestamp check is the only ordering invariant (`portfolio_validator.py:181-188`). If Layer 2 ever inserts a backdated transaction (e.g. recovery from a journal), FIFO matches a SELL against a later-dated BUY, inflating apparent hold time or pairing the wrong BUY with the SELL, producing wrong realized P&L. Fix: sort transactions by `timestamp` before building queues; or assert monotonic non-decreasing ts at load time.

portfolio/monte_carlo.py:154,217-218: P0: Crypto/metals annualization uses 252 trading days but those instruments trade 365 days/year. `T = horizon_days / 252.0` for crypto inflates implied T by 365/252 = 1.45x, raising the vol-term by sqrt(1.45) = 1.20x. Every BTC/ETH/XAG/XAU Monte Carlo stop-hit probability and price band is biased outward by ~20%. The equity-curve module already documents the correct convention (`equity_curve.py:23: ANNUALIZATION_DAYS = 365`) but monte_carlo.py and price_targets.py disagree. Fix: parameterize annualization on instrument class (252 for stocks, 365 for crypto/metals); reuse the existing constant.

portfolio/exit_optimizer.py:173,217-218: P0: Same 252 vs 365 mismatch for crypto in `simulate_intraday_paths`. `_TRADING_DAYS_PER_YEAR = 252` is multiplied by `_TRADING_MINUTES["crypto"] = 1440` giving 362,880 min/yr, but crypto's true minutes/yr is 525,600. dt is too large by ~45%, inflating the per-step variance. This contaminates `stop_hit_prob`, `expected_hit_time_min`, and every exit candidate's `fill_prob` for crypto. Fix: use 365 trading days for crypto; ideally derive `min_per_year` from `_TRADING_MINUTES[type] * (365 if type in {"crypto","warrant"} else 252)`.

### Important (P1)

portfolio/monte_carlo_risk.py:39,202: P1: t-copula df hardcoded to `DEFAULT_DF = 4`, never fit from data. The review brief explicitly asked. df=4 implies tail dependence λ ≈ 0.18 — reasonable for liquid majors, but too thin for stressed crypto (empirical df estimates 2.5–3.5). df is not even configurable through `compute_portfolio_var`. Fix: accept a `df` argument; if `historical_returns` has enough samples, estimate per-pair df via MLE or moment-matching kurtosis and clip to [3, 10].

portfolio/monte_carlo_risk.py:62-85: P1: correlation defaults to ZERO for tickers with <20 historical observations, then `_nearest_psd` keeps it at zero. For BTC/ETH (known >0.7 correlation per CLAUDE.md), missing history silently produces independent assumption and *understates* portfolio VaR. Fix: when historical data is insufficient, fall back to `correlation_priors` (already done in `build_correlation_matrix` at line 146) — but `estimate_correlation_matrix` is called standalone elsewhere; either deprecate it or have it consult priors as a floor.

portfolio/risk_management.py:373: P1: ATR cap at 15% silently inverts behaviour for warrants. `atr_pct = min(atr_pct, 15.0)` is intended to "prevent meaninglessly wide stops for warrants". But a true 25% ATR (silver warrant in a violent move) gets clamped to 15%, putting the stop *closer* to spot than reality justifies and triggering whiplash exits. Fix: cap should be PER-INSTRUMENT (3% for liquid stocks, 8% for metals underlying, 25% for leveraged warrants), not a global 15%.

portfolio/risk_management.py:374: P1: ATR stop computed off `entry_price`, never trailed. `stop_price = entry_price * (1 - 2 * atr_pct / 100)`. If a position runs +30%, the stop is still anchored to entry and gives back the entire gain. CLAUDE.md mentions "ATR-based trailing stop" but the implementation isn't trailing — it's static. Fix: track `position.trailing_peak_usd` (already exists in `exit_optimizer.Position`) and compute `stop_price = trailing_peak * (1 - 2 * atr_pct / 100)`.

portfolio/risk_management.py:217-317: P1: `check_drawdown` re-arms instantly after a recovery. There's no hysteresis or cool-down. A 25% drawdown that briefly recovers to <20% un-breaches in one cycle, then re-breaches the next; the consumer must dedupe or the system whipsaws between guarded and unguarded states. Fix: persist a `breach_armed` flag with a recovery requirement (e.g. drawdown below threshold for 24h before re-arming).

portfolio/risk_management.py:282-286: P1: Peak value computation: when current_value > historical peak, it bumps `peak_value = current_value` (line 285). Good. But when the file IS the cache source, the next save in `log_portfolio_value` writes the *just-bumped* value, so peak gets re-discovered on next call as the same number, which is fine. However the cache in `_peak_cache` only holds the *file* peak; if `current_value` overshoots, the cache returns a stale peak on the next call. `_peak_cache[cache_key] = {"peak": peak, "offset": end_offset}` is written with the file-derived peak, not the bumped one. Minor: cache freshness is OK because the next cycle's `log_portfolio_value` appends, then `_streaming_max` reads. But if `log_portfolio_value` hasn't run yet, the bumped value isn't persisted and the cycle reports a non-deterministic peak. Fix: bump cache when `current_value` exceeds cached peak; or ensure `log_portfolio_value` runs strictly before `check_drawdown`.

portfolio/risk_management.py:217-238: P1: `check_drawdown` reads `portfolio_path` then `agent_summary_path` non-atomically. Between the two reads, Layer 2 may write a new transaction (cash changes) while the agent_summary reflects old prices, or vice versa. Result: drawdown computed against inconsistent snapshots. Fix: snapshot agent_summary into memory first, then load portfolio; or pass both as in-memory dicts from the caller.

portfolio/risk_management.py:259-270: P1: When `agent_summary` is empty, the function falls back to `cash_sek` as the entire portfolio value. For a portfolio that is 80% in holdings, this looks like a 80% drawdown — false breach. The WARNING is logged but the numeric output `current_value` is wrong and breached. Fix: return a special `feed_stale: True` marker and SKIP the breach check, rather than producing a falsely-low value.

portfolio/portfolio_validator.py:130-132: P1: tolerance allows 1% of bought shares to silently vanish on close-out. "small remainder from rounding, ticker removed -- acceptable" — but at 1% of 10 BTC bought, that's 0.1 BTC ≈ $8K worth of unaccounted shares slipping through validation. Tolerance is way too loose for crypto. Fix: tighten to 1e-6 absolute, or compute the maximum dust allowed in *SEK terms* (e.g. 1 SEK).

portfolio/portfolio_validator.py:174-178: P1: "No fee_sek fields" message uses `total_fees_sek == 0` as the signal. If `total_fees_sek` is NULL but transactions exist, the earlier check (line 50) sets it to 0, and *this* check then fires. Coherent, but the message says "Fees may not be tracked" when the real issue is missing total_fees_sek. Confusing diagnosis. Fix: distinguish "no fee_sek on tx and no aggregate" from "aggregate is 0 but transactions exist".

portfolio/portfolio_validator.py:75-90: P1: cash reconciliation uses `total_sek` directly — assumes the BUY total INCLUDES fee and SELL total EXCLUDES fee. If Layer 2 ever inverts this (e.g. records BUY total without fee), the reconciliation passes silently while the actual cash is off. There's no per-transaction validation of `total_sek` arithmetic vs `shares × price_usd × fx_rate + (-)fee_sek`. Fix: cross-check `tx.total_sek` against `tx.shares * tx.price_usd * fx_rate +/- tx.fee_sek` (within a tolerance) per transaction.

portfolio/trade_guards.py:51-69: P1: `_portfolios_have_transactions()` reads three portfolio files with raw `load_json`. No lock, no retry on partial reads. If `portfolio_state.json` is mid-replace, `load_json` returns default `{}` and the function reports no transactions, suppressing the C4 wiring alarm. The atomic-write in portfolio_mgr is safe against torn reads (os.replace is atomic), but `load_json` returns `default` on `JSONDecodeError` rather than retrying — so a (rare) split-second window where the OS hasn't completed the rename gives a false-negative. Fix: retry once on default-fallback.

portfolio/trade_guards.py:95-99: P1: `_get_cooldown_multiplier` halves via `base >> halvings`. After ~3 days the multiplier is 1 regardless of the original 8x. That's intentional decay. But there's NO floor on `consecutive_losses` — a long losing streak (say 8) is read from `LOSS_ESCALATION.get(consecutive_losses, 1)` for any value ≥4 (returns 8). Then 8 >> 4 = 0 after 96h, and `max(1, base >> halvings)` floors to 1. The 12h → "effectively disabled" claim in CLAUDE.md is correct only if decay halvings count 96h+. Fix: nothing broken here, but document that the curve in CLAUDE.md (30m → 2h → 12h) refers to fix_agent dispatcher, not trade_guards.

portfolio/trade_guards.py:103-231: P1: `check_overtrading_guards` releases lock at line 128, then makes decisions on stale state. Another thread could call `record_trade` between the load and the decision, making the cooldown stale by the time it's used. Window is small but real. Fix: hold the lock until the warnings list is built (decision is read-only relative to state, so this is just longer critical section, not a correctness issue beyond the staleness).

portfolio/trade_guards.py:282: P1: consecutive_losses incremented BEFORE persisting last_loss_ts. If process crashes between increment and `_save_state`, both fields are lost — but if the crash is between `state["consecutive_losses"][strategy] = X+1` and `state["last_loss_ts"][strategy] = now_str`, the load on restart sees old values (intermediate dict mutation is in-process only, not persisted). OK — no bug. But also note: there's no compensating decrement if the SELL turns out to be a wash trade. Minor.

portfolio/equity_curve.py:494-495: P1: `wins/losses` split uses `pnl_pct > 0` for wins and `pnl_pct <= 0` for losses. A trade with EXACTLY 0% gross move (post-fee small loss) counts as a "loss" by pct but `pnl_sek` could be slightly negative (fees) — counted as a loss in *both* the pct stat and the SEK stat. Consistent enough. But `gross_profit/gross_loss` (line 468-469) uses `pnl_sek`. So a tiny-gain-net-of-fees-loss trade is a "win by pct" and a "loss by SEK". `profit_factor` and `win_rate` disagree on what counts. Fix: pick one denomination (SEK is the real one) consistently.

portfolio/equity_curve.py:557: P1: `years = days / 365.25; (1+total_return)**(1/years)`. With `years < 1` (sample <1y), the annualization extrapolates wildly. After 3 days with +5%, this reports `(1.05)^(365/3) - 1 ≈ 287,000%` annualized. The Calmar ratio inherits this fiction. Fix: require `years >= 0.25` (3 months) or report `None` for shorter samples; or use a non-extrapolating "average daily return × 365".

portfolio/monte_carlo.py:362-368: P1: `_get_directional_probability` derivation `p_up = 0.5 + conf * 0.3` for BUY (so 0.5-0.8 range). When `conf` exceeds 1.0 (which `weighted_confidence` can technically hit), `p_up > 0.8` and the `drift_from_probability` clamp at 0.99 bites. But there's no input validation. Fix: clamp `conf` to [0, 1] before scaling.

portfolio/monte_carlo.py:251: P1: skewness uses biased moment estimator `np.mean(((returns - mean_pct) / std_pct) ** 3)` rather than the sample-corrected Fisher-Pearson formula. For n=10,000 paths the bias is negligible, but the report claims "skew" which suggests population skewness; document or correct.

portfolio/monte_carlo.py:399: P1: `seed + i` for per-ticker independence — if `seed=0`, ticker 0 uses seed=0 which `np.random.default_rng(0)` accepts but produces identical streams across tickers for n=0. Minor since the next ticker uses seed=1. Comment notes `seed=None` was fixed; this remaining edge case is benign.

portfolio/exit_optimizer.py:54: P1: `usdsek: float = 10.85` hardcoded fallback in MarketSnapshot. If a caller forgets to pass `usdsek` (e.g. from a unit test or new code path), it silently uses 10.85, which is stale (the codebase's other fallback is `FX_RATE_FALLBACK = 10.50`). Two SEK fallbacks disagree — pick one and import. Fix: default to `FX_RATE_FALLBACK` from `portfolio.fx_rates`.

portfolio/exit_optimizer.py:185: P1: vol fallback to 0.20 (20% annual) when ATR is missing. For an XAG warrant with 60%+ realized vol, this systematically understates the variance of session paths, underestimating `stop_hit_prob` and overpromising fill probabilities. Fix: per-instrument-class default that mirrors `monte_carlo._ATR_DEFAULT_BY_CLASS`.

portfolio/exit_optimizer.py:430-436: P1: `_apply_risk_overrides` knock-out check uses `position.financing_level > 0`. For a BEAR (short) MINI, the financing level is ABOVE the underlying, and "distance < 3%" inverts: `(market.price - financing_level) / market.price * 100` is NEGATIVE. The early-return logic still fires correctly only if `distance_pct < 3` (a large negative number passes). But the WARNING message reports "%.1f%% from barrier" with a negative number, misleading operators. Fix: use `abs(market.price - financing_level) / market.price * 100`; differentiate BULL vs BEAR warrants.

portfolio/risk_management.py:868-919: P1: `check_atr_stop_proximity` uses `entry_price * (1 - 2 * atr_pct / 100)` — the SAME static stop as `compute_stop_levels`, no trailing logic. Same critique as P1 above. Compounds the risk: a position that has rallied is reported as far from stop, while the *new* sensible stop (trailed up) is much closer.

portfolio/price_targets.py:39-41: P1: 252-day annualization for 24/7 assets — same bug as monte_carlo.py. `T = hours / (252.0 * 24.0)` inflates T for crypto/metals. Fix: 365 for 24/7.

portfolio/price_targets.py:124: P1: `rng = np.random.default_rng(42)` — fixed seed. Every call to `running_extremes` produces deterministic identical paths across the entire process lifetime. Not a bug per se (reproducibility is nice), but if the price moves, you keep getting the same z-draws and the resulting "MC quantiles" effectively become a deterministic function of price/vol/drift. The Monte Carlo character is lost. Fix: thread seed through from caller, default to a random one.

portfolio/cost_model.py:64-70: P1: WARRANT_COSTS spread_bps=40 (0.40% half-spread). Avanza MINI warrants on silver routinely show 0.6-1.5% half-spread depending on size, and the codebase elsewhere uses ELONGIR_COSTS spread_bps=40 with explicit comment "0.40% half-spread". The simulated portfolio doesn't actually use this cost model for booking trades (per code search) — but `compute_exit_plan` does, and the EV computation will systematically over-promise fill values. Fix: per-instrument spread (XAG bid-ask is 60-100 bps, XAU 30-60 bps), or pull live spread from Avanza orderbook snapshot.

portfolio/cost_model.py:107-116: P1: `get_cost_model` defaults unknown types to STOCK_COSTS (6.9 bps courtage). If a typo in `instrument_type` (e.g. "warrants" vs "warrant") routes a warrant trade through stock costs, EV is computed against the wrong cost basis. Fix: raise on unknown type, or warn loudly.

portfolio/trade_risk_classifier.py:69-77: P1: position_size scoring caps at 3 points for >20%. A 90% position scores identically to a 21% position. For a portfolio aiming at <40% concentration, the 20%-cap loses signal in exactly the band where it matters most. Fix: tiered above 20% (e.g. >40% = +4, >60% = +5).

portfolio/trade_risk_classifier.py:80-84: P1: `_REGIME_SCORES.get(regime_lower, 0)` returns 0 (least risky) for ANY unknown regime, including typos. Silent fail. Fix: log on unknown regime; consider scoring unknown as "ranging" (2) rather than "no risk" (0).

portfolio/risk_management.py:763-787: P1: `check_concentration_risk` computes `proposed_alloc = min(total_value * alloc_pct, cash)` but doesn't account for OTHER pending BUYs in the same cycle. If patient is about to BUY BTC at 15% and ETH at 15%, each one alone is "under threshold" but the combined post-cycle position is 30%. Fix: aggregate across all proposed BUYs in this cycle before deciding.

portfolio/exit_optimizer.py:557-562: P1: `terminal_pnls = np.array([_compute_pnl_sek(position, float(p), market, costs) for p in np.percentile(terminal, [10, 25, 50, 75, 90])])` — `hold_ev` is the *mean of 5 quantile P&Ls*, not the mean of all path P&Ls. With a skewed distribution the mean-of-quantiles can be far from the true mean. Fix: `hold_ev = np.mean([_compute_pnl_sek(position, p, ...) for p in terminal])` (full vector, ~5K evals per call — acceptable inside a 60s loop).

portfolio/monte_carlo_risk.py:88-106: P1: `_nearest_psd` uses eigenvalue clipping to 1e-8. Cholesky then succeeds, but if the input is highly singular (e.g. duplicate ticker rows from a misordered `positions` dict), the rescaled correlation can still have near-singular eigenstructure and the variance scale shifts. Fix: deduplicate tickers explicitly before building the matrix; assert `cond(corr) < 1e10`.

portfolio/equity_curve.py:50-54: P1: `load_equity_curve` sorts by `ts` string, not parsed datetime. ISO-8601 lexicographic sort works for fully-qualified UTC strings, but if any entry has a different format (e.g. naive datetime without TZ suffix), sort order is wrong. Fix: parse to datetime then sort.

portfolio/portfolio_mgr.py:50: P1: `_rotate_backups` skips backup if `path.stat().st_size == 0`. A legitimate empty state file (e.g. after a corrupt-recovery scenario) is then not backed up, but neither is corruption manifesting as an empty file. Less critical: the backup would be empty anyway. OK.

portfolio/portfolio_mgr.py:54-58: P1: backup rotation copies `path.bak2 → path.bak3` etc but uses `shutil.copy2`, not atomic. If a power loss hits mid-copy, `.bak3` is corrupted. Then on recovery `_load_state_from` walks backups in order and would still find `.bak` or `.bak2` intact. Acceptable.

### P2 (Brittleness / Performance)

portfolio/risk_management.py:43-110: P2: `_streaming_max` cache invalidation is keyed by `(str(path), value_key)`. Two callers with different `floor` values but same key will see one another's cached peak. Floor is the only varying input not in cache key. Risk: a buggy caller passing floor=0 caches peak=actual then a correct caller passing floor=500000 reads the cached peak (correct). The reverse: caller passing floor=0 then floor=500000 — both work because cached peak ≥ floor in either case. OK in practice, document.

portfolio/risk_management.py:84-99: P2: `_streaming_max` does NOT take the `jsonl_sidecar_lock`. The history file is appended by `atomic_append_jsonl` (which holds the lock) but `_streaming_max` reads without the lock. On Windows with msvcrt the writer's lock is advisory — the reader could see a partially-written final line. Currently caught by `try/except json.JSONDecodeError: continue` (line 95), so a partial line is silently skipped. The valid prior peak is unaffected. OK, but document that the reader is lock-free by design.

portfolio/monte_carlo_risk.py:262-280: P2: per-marginal inverse CDF loops in Python (`for i, ticker in enumerate(self._tickers)`). For n_paths=10K and n_assets=5, this is 50K norm.ppf calls — fast enough, but vectorizable as `norm.ppf(U)` element-wise. Minor.

portfolio/monte_carlo_risk.py:289-307: P2: `portfolio_pnl` does `pnl += shares * price * (np.exp(returns) - 1)` per ticker. For large portfolios this is fine; for a 50-asset portfolio it's still O(N×paths). Acceptable.

portfolio/exit_optimizer.py:618-621: P2: `terminal_pnls` computed via list comprehension over 5 quantiles — see P1 above; same issue.

portfolio/equity_curve.py:480-491: P2: `trade_frequency_per_week` divides by `span_days` even for a single round trip on the same day, leading to division by 0 if both sell and buy on identical timestamp. Guarded by `if span_days > 0`. OK.

portfolio/exit_optimizer.py:617-620: P2: When `terminal` array is small (very short remaining session), the 5-percentile sample severely under-represents the distribution.

portfolio/portfolio_validator.py:181-188: P2: duplicate-tx key includes timestamp at full ISO precision. Two genuine trades at the same microsecond would falsely flag. Unlikely but possible if Layer 2 burst-writes. Fix: also include a tx index in the key.

portfolio/trade_guards.py:265-330: P2: `record_trade` holds the lock through a full read-modify-prune-save cycle. If `_save_state` (atomic_write_json) is slow (Windows AV interference), other callers block. Acceptable but consider releasing before the save and re-acquiring just for the actual replace.

portfolio/trade_guards.py:381-386: P2: C4 wiring warning fires only when `all_warnings == []`. If even one trivial warning is present (e.g. consecutive_losses informational), the C4 broken-wiring detector goes silent. Fix: separate the wiring check from the warning aggregation.

portfolio/risk_management.py:373: P2: The ATR 15% cap is applied AFTER `min(atr_pct, 15.0)` but the rest of the function uses the capped value to also compute `distance_to_stop_pct`. So the reported "distance to stop" can disagree with reality. Document or fix.

portfolio/monte_carlo.py:35-40: P2: `_ATR_DEFAULT_BY_CLASS` for crypto=3.5 mixes assets with very different vols (BTC ~3%, ETH ~4%, alts much higher). For a multi-asset crypto portfolio, this is too generic. Acceptable as default.

portfolio/exit_optimizer.py:557: P2: `target = float(target)` from `np.quantile(session_max, q)` — the quantile-based candidate construction means if session_max is bimodal, the candidate prices cluster around quantiles rather than spanning the realistic exit space.

portfolio/exit_optimizer.py:568: P2: `if target <= market.price * 0.999`: skips candidates within 0.1% of current price. For a position with razor-thin gain expectation (e.g. metals scalp), this can drop the *only* candidate above price. Soft.

portfolio/price_targets.py:106-108: P2: `fill_probability_buy` uses `price ** 2 / target` as a flip — works for BSM symmetry only when drift=0. With nonzero drift the symmetry is broken; the flip is approximate. Document.

portfolio/cost_model.py:36-47: P2: `total_cost_sek` returns 0 for `trade_value_sek <= 0`. But `min_fee_sek` is always charged on Avanza even on a tiny trade. The current logic correctly applies `max(courtage_value, min_fee_sek)` only when `trade_value > 0`, so a 0-SEK trade returns 0 (no min fee). That's correct for the boundary, but if a caller passes `trade_value_sek=1` they pay max(0.000069, 1.0) = 1 SEK minimum. OK.

portfolio/portfolio_mgr.py:60-62: P2: `_rotate_backups` swallows OSError as warning. If the FS is full and the rotation fails silently, the next atomic_write may succeed and overwrite the only good state. Fix: raise so the caller knows backups are broken.

## Patterns

- **Annualization inconsistency**: 252 vs 365 days. `monte_carlo.py`, `monte_carlo_risk.py`, `exit_optimizer.py`, and `price_targets.py` all use 252. `equity_curve.py` uses 365 for crypto. For 24/7 assets (BTC, ETH, XAG, XAU — i.e. ALL Tier-1 instruments per CLAUDE.md), the right value is 365. This biases every MC-derived quantity (stop probability, price bands, fill probabilities, VaR) by ~20%. Fix this in one place (a central `annualization_days(ticker)` function) and route every consumer through it.

- **Static stops, no trailing**: `compute_stop_levels`, `check_atr_stop_proximity`, and `compute_probabilistic_stops` all anchor to entry_price. Position runs +30% → static stop gives back the win on a -2% wobble. The `Position` dataclass already has a `trailing_peak_usd` field used by exit_optimizer alone; the rest of risk_management ignores it.

- **Lock bypass**: `portfolio_mgr.update_state` holds a per-file lock but legacy `load_state` / `save_state` callers bypass it, and the lock is in-process (`threading.Lock`) only — not cross-process. Layer 2 subprocess writing the same file is uncoordinated.

- **Per-instrument missing**: WARRANT_COSTS, ATR caps, vol defaults, knockout barriers are all global constants rather than per-instrument. The system treats XAG MINI 5x and stock-replacement warrants with the same cost/risk parameters.

- **Reconciliation tolerances too loose**: 1% share dust, 1 SEK cash tolerance, 1% avg_cost_usd drift. At 500K starting capital these tolerances permit substantial silent drift before the validator screams.

- **Live-price dependency without explicit handling**: When `agent_summary` is empty (loop just started, file rotating, fx_rates crashed), `_compute_portfolio_value` and `check_drawdown` fall back to cash-only, which trips false drawdown breach for held portfolios. The system already has a `_resolve_fx_rate` cache chain — extend the same pattern to prices.

- **Static seeds and identity transforms**: `price_targets.running_extremes` uses fixed seed=42, making "Monte Carlo" deterministic per (price,vol,drift) triple. The t-copula sim correctly uses `norm.ppf` (post C9 fix) but the comment history shows this exact identity-transform mistake was in production.

## Out of Scope but Spotted

- portfolio/main.py:783: `save_state(state)` runs only when `STATE_FILE.exists()` is False — odd inversion; comment suggests it's a first-run bootstrap. If the file exists but state is corrupted (recovered from defaults), this skip means the recovered defaults are never persisted. Fix: always persist after a successful load_state, idempotent.
- portfolio/journal.py: total_fees_sek defaulted to 0 — if portfolio_mgr defaults add the field per P0 above, journal stays consistent.
- portfolio/autonomous.py:835-846: `_load_bold_state_safe` has its own DEFAULT including `total_fees_sek: 0`. So autonomous mode bootstraps correctly while normal mode (via portfolio_mgr) does not.
- dashboard/app.py:_read_json caches portfolio_state reads with TTL; combined with the live writer using atomic os.replace, the dashboard never sees torn JSON but CAN serve up-to-TTL-stale snapshots — fine for a UI but worth documenting.
- portfolio/correlation_priors.py is imported by risk_management.py and monte_carlo_risk.py but not in scope; ensure priors cover BTC/ETH (0.7+) and XAG/XAU (0.8+) given correlation is the only barrier between independent-assets VaR and reality when historical data is sparse.
- portfolio/fx_rates.py provides `FX_RATE_FALLBACK = 10.50` and `FX_RATE_MIN/MAX` — exit_optimizer.MarketSnapshot defaults to 10.85, a third number; align.
- portfolio/session_calendar.remaining_session_minutes used by compute_probabilistic_stops without import-failure handling beyond a single try/except that silently returns `{}` — the silent failure mode is acceptable but the absence of a "probabilistic stops are unavailable" indicator means consumers may treat the empty dict as "no risk".

