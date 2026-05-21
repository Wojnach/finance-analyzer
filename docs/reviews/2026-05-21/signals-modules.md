# signals-modules — adversarial review 2026-05-21

Baseline: 604f0ef1
Worktree: Q:\finance-analyzer\worktrees\review-2026-05-21
Scope: 64 modules under `portfolio/signals/` plus LLM-signal wrappers (`portfolio/ministral_signal.py`, `portfolio/qwen3_signal.py`, `portfolio/qwen3_trader.py`) and `portfolio/signal_utils.py`. Read in full: all 17 active + most pending-validation. Spot-checked the rest.

Findings are formatted as `path:line: <P0|P1|P2|P3>: <problem>. Fix: <suggestion>.`

## Critical (P0)

`portfolio/signals/gold_real_yield_paradox.py:285-291`: P0: gold/yield calendar alignment is broken. `gold_daily_returns` (chronological, includes weekends from Binance) is paired with `yield_daily_changes` derived from FRED business-day series and reversed. Indexing by relative position not date — weekend bars in gold get matched to stale-Friday yields, producing a biased 30d correlation and spurious correlation-break vote. Fix: align both into a single DataFrame on a date index (DatetimeIndex.intersection), drop unmatched days, then compute returns/changes on the aligned frame.

`portfolio/signals/metals_vrp.py:169-176`: P0: VRP z-score math is wrong. `rv_for_vrp = gvz_hist_for_z - rv_mean` subtracts a single scalar (mean of recent RV) from the entire GVZ history — the resulting series is just GVZ shifted by a constant, not a VRP time series. The `vrp_mean`, `vrp_std`, and `vrp_median` are therefore properties of GVZ alone, not VRP. The z-score and level votes are dimensionally GVZ z-scores rebranded as VRP. Fix: build a proper VRP series `gvz_t - rv_t` aligned on common dates, then z-score that series.

## Important (P1)

`portfolio/signals/momentum_factors.py:332-362`: P1: `_apply_seasonality` reuses the same compounding-bug pattern that was explicitly fixed in `mean_reversion.py:477-487` (P1-6, 2026-05-02 follow-ups). The loop reads `detrended["close"].iloc[i - 1]` after mutating earlier rows, so detrending compounds geometrically. Fix: capture `original_close = detrended[_close_col].astype(float).copy()` before the loop and rebuild each row from `original_close.iloc[i - 1] * (1 + adj_ret)`, mirroring the fix in mean_reversion.py.

`portfolio/signals/hurst_regime.py:284-285,301-302`: P1: duplicate-vote inflation. In trending regime, `sub_signals["hurst_regime"]` and `sub_signals["trend_direction"]` are BOTH set to the same `trend_vote`. Same pattern in MR regime for `hurst_regime`/`mr_extreme`. The majority vote then sees 2 identical votes plus `hurst_momentum`, so a single underlying trend direction casts 2 of 4 votes — confidence is structurally inflated. Fix: collapse to 3 independent voters or weight each regime's vote 0.5.

`portfolio/signals/hurst_regime.py:200-206`: P1: `_hurst_momentum` interprets rising Hurst as BUY and falling Hurst as SELL, but rising H means "trend strengthening" without specifying direction — a strengthening downtrend should be SELL, not BUY. Fix: combine sign(roc) with the current trend direction (e.g., rising H + uptrend → BUY, rising H + downtrend → SELL).

`portfolio/signals/cross_asset_tsmom.py:130-145,148-158`: P1: duplicate-vote inflation. For XAU-USD, `_compute_cross_pair` reads TLT's `ret_63d` and `_compute_bond_momentum` reads the EXACT same value — both vote the same direction. For BTC-USD, `_compute_cross_pair` (SPY) and `_compute_equity_momentum` (SPY) duplicate. 4 voters but only 3 effective directions. Fix: drop one of the duplicates or use a different lookback for the cross_pair vs the asset-class proxy.

`portfolio/signals/cross_asset_tsmom.py:228`: P1: `indicators["gld_ret_63d"] = safe_float(_yf_ret("GLD"))` — but GLD is not in `_YF_TICKERS` (line 51 has `GC=F` instead). Indicator always None. Fix: change to `_yf_ret("GC=F")` or label as `gold_futures_ret_63d`.

`portfolio/signals/network_momentum.py:356-357`: P1: duplicate-vote pattern. `corr_regime` is rewritten to `net_div`'s direction when both fire, so 2 of 3 sub-signals are by construction identical. Fix: keep `corr_regime` independent — it should be a magnitude gate that suppresses, not a directional copy.

`portfolio/signals/ovx_metals_spillover.py:104-126`: P1: `_ovx_level_signal`, `_ovx_momentum_signal`, `_ovx_zscore_signal` all vote SELL when OVX rises sharply and BUY when it falls sharply — three votes from the same underlying OVX direction. Net effect: 4-voter majority is really one OVX direction signal. Fix: keep only one of (level z-score) or (level percentile rank) and one of (momentum) or (z-score), and let `_ovx_reversion_signal` be the contrarian counterweight.

`portfolio/signals/vix_term_structure.py:48-49`: P1: yfinance fetches `^VIX` / `^VIX3M`. These quotes are US-cash-session-only — outside RTH the cached values from previous close are returned. When the loop runs at 03:00 CET (US closed), this signal votes against a stale 13-hour-old quote that the crypto/metals markets have already absorbed. Fix: stamp the data freshness from yfinance and force HOLD when last_close is more than `N` hours stale; or use VIXY/VXX intraday ETF prices as proxy.

`portfolio/signals/vix_term_structure.py:141-197`: P1: signal is asset-agnostic — same vote for crypto, metals, MSTR. VIX backwardation should mean different things for safe-haven (BUY gold) vs risk-on (SELL BTC). Fix: gate direction by asset class similar to credit_spread.py.

`portfolio/signals/vol_ratio_regime.py:256-258`: P1: signal contract violation. `sub_signals["gk_cc_regime"]`, `vr_regime`, `er_regime`, `composite_regime` are strings "ranging"/"trending"/"neutral" — not BUY/SELL/HOLD. Engine and dashboard expect vote strings; these will not aggregate correctly via `majority_vote`. Fix: rename to indicator keys and add BUY/SELL/HOLD sub_signals separately, or wrap regime labels under `indicators` (where they belong).

`portfolio/signals/sentiment_extremity_gate.py:65,199,227`: P1: returns `"PASS"` instead of HOLD. Engine and accuracy tracker only recognize BUY/SELL/HOLD; "PASS" will be silently ignored or miscounted in downstream voting and accuracy backfill. Fix: emit `HOLD` (gating only) and signal "passthrough" via an indicator flag.

`portfolio/signals/tsi_chop_mr.py:142-145,174`: P1: sub_signals contain literals `"HOLD (trending)"`, `"ranging"`, `"transition"` — not BUY/SELL/HOLD. Engine will not vote on these strings. Fix: use plain HOLD and move regime label to `indicators`.

`portfolio/signals/connors_rsi2.py:144-151`: P1: sub_signals format is a nested dict `{"value": float, "signal": str}` rather than the standard `{name: vote}`. Most downstream consumers (`accuracy_stats`, dashboard) iterate `sub_signals.values()` expecting strings, so this signal's sub-votes are unparseable. Fix: emit flat `{name: "BUY|SELL|HOLD"}` and move values into `indicators`.

`portfolio/signals/adx_regime_switch.py:152-158`: P1: same nested-dict sub_signals contract violation as connors_rsi2. Same fix.

`portfolio/signals/drift_regime_gate.py:38-58`: P1: signal direction inverted vs cited source. arxiv:2511.12490 describes drift regimes ACTIVATING momentum signals — so >60% positive-days regime should bias BUY (continuation), not SELL (mean reversion). The code emits SELL on >60% positive days (line 55-56). Fix: re-read source — the regime is a GATE that opens momentum, the directional vote should come from a momentum sub-signal, not from contrarian inversion of the drift fraction.

`portfolio/signals/williams_vix_fix.py:218-222`: P1: 3 of 4 sub-indicators are BUY-only (bb_spike, percentile, rsi_confirm) and 1 is SELL-only (complacency). The composite has 3:1 BUY-bias just from architecture — even pure noise tends to produce BUY signals. Fix: add bear-market sub-signals or rebalance to a symmetric 2 BUY-capable + 2 SELL-capable design.

`portfolio/signals/intraday_seasonality.py:110-129`: P1: `_hour_alpha_vote` and `_dow_vote` BOTH return only `BUY` or `HOLD`, never SELL. Structural BUY bias during all favorable hours. Fix: emit SELL when multipliers are very low (e.g., mult <= 0.5 should vote SELL for risk-on assets, BUY for safe havens) or rename to "gate" sub-signals that abstain rather than vote.

`portfolio/signals/gold_overnight_bias.py:118-140`: P1: `_fix_proximity_vote` is BUY-only. Both branches (near AM fix, near PM fix) return BUY. Combined with `_session_phase_vote` that emits BUY during 19.5h-of-24 overnight window, this signal is BUY-biased by construction. Fix: emit HOLD near AM fix (overnight effect dissipating), reserve BUY only for near-PM-fix setup.

`portfolio/signals/news_event.py:281-303`: P1: `_sentiment_shift` defaults non-`normal`-severity headlines to `neg += 1` when no positive keyword matches and no "cut" phrase matches. Combined with the SELL-only `_keyword_severity_vote` and `_pre_event_risk`-style sister signals in econ_calendar, news-driven votes lean SELL even when sentiment is mixed. Fix: add an explicit `else` branch that votes neutral when no directional keywords are present, OR widen `_POSITIVE_KEYWORDS` to cover the common positive-but-non-critical lexicon (e.g., "deal", "expand", "growth", "win", "milestone").

`portfolio/signals/news_event.py:396`: P1: comment at line 387 says dissemination threshold "score >= 2.0" but the code at line 396 uses `score < 1.5`. Mismatch. Fix: pick one threshold and align the comment.

`portfolio/signals/econ_calendar.py:140-146`: P1: `_post_event_relief` returns BUY any time the next event is more than 72h away. During Fed pause periods (often weeks between high-impact events), this means a permanent BUY vote on this sub-signal — directly contradicting the "post-event relief" semantics. Fix: gate the BUY on `relief_events` being non-empty (recent high-impact event passed), and treat the "event-free calm" as HOLD or as a confidence dampener.

`portfolio/signals/crypto_macro.py:188-192`: P1: `_expiry_proximity` returns BUY for both quarterly AND regular expiry on day 0-1, despite the docstring warning that expiry day is a "volatility warning". Structural BUY bias every option expiry. Fix: emit HOLD on expiry day and BUY only the day AFTER expiry (post-expiry relief).

`portfolio/signals/cot_positioning.py:144`: P1: `_compute_cot_index` accepts as few as 10 history samples, but COT Index percentile is canonically a 156-week (3yr) measure. With 10 samples a single outlier dominates the range. The signal can therefore vote BUY/SELL with extreme confidence based on very thin history. Fix: raise minimum to 52 (1 year) and return HOLD until threshold is met.

`portfolio/signals/cot_positioning.py:173-178,241-246`: P1: current `nc_net` is prepended to history before computing percentile/z-score; mean/std/min/max include the current value. The "percentile of current value vs history" is therefore biased toward 50 (the current value is always inside its own range, regardless of how extreme it really is). Fix: compute mean/std/min/max from history EXCLUDING the current value, then compute percentile.

`portfolio/signals/cot_positioning.py:158-194`: P1: `_sub_cot_index` returns `(vote, confidence, indicators)` but the caller at line 376 only captures `(vote, conf, ind)` and never uses the confidence — meanwhile `compute_cot_positioning_signal` returns its own composite confidence at line 403. The per-sub-signal confidence (0.4-0.7 based on intensity) is discarded. Fix: either remove the unused return value or propagate per-sub-signal confidence into the composite.

`portfolio/signals/cot_positioning.py:354-358`: P1: no staleness check on COT report. CFTC publishes weekly (Friday); if the loop has been running without precompute refresh, `cot_data` could be 1-3+ weeks stale and the signal votes as if last week's positioning is current. Fix: parse `report_date`, force HOLD when older than 14 days.

`portfolio/signals/residual_pair_reversion.py:276-302`: P1: frequency mismatch between target and driver. `df["close"]` is the loop's per-ticker OHLCV (hourly or finer in production), but `_fetch_driver_closes` always pulls daily yfinance. `_rolling_ols_beta` then computes rolling cov/var assuming matching cadence; the alignment via `intersection` drops most of the target's intraday bars and the OLS regresses what's left (sparse, irregular). Beta and residuals are meaningless on a frequency-mismatched series. Fix: resample target to daily before regressing, or fetch driver at the matching frequency (`yf.download(..., interval="1h")`).

`portfolio/signals/residual_pair_reversion.py:145-152`: P1: half-life of a negative-theta AR(1) residual is taken via `abs(theta)`, but the OU model requires theta>0; negative theta describes oscillating residuals, not mean reversion. `_compute_half_life` should return NaN for theta<=0 rather than fabricating a half-life from absolute value. Fix: return NaN for theta<=0.

`portfolio/signals/crypto_evrp.py:204-242`: P1: docstring of `_evrp_percentile_signal` says "where current eVRP sits in 90-day distribution" but the implementation falls through to using `dvol_history` percentile directly (line 231-236), ignoring the realized-vol component entirely. The sub-signal is therefore DVOL percentile, not eVRP percentile — different signal than advertised. Fix: build a historical eVRP series by aligning DVOL and RV by date, then percentile that.

`portfolio/signals/vwap_zscore_mr.py:88-92`: P1: `vol_vote` is set to `vwap_z_vote` whenever volume is high — pure duplicate vote. 2 of 3 sub-signals are forced into the same direction when active. Fix: make volume an independent gate (suppress if low) rather than a directional copy.

`portfolio/signals/vwap_zscore_mr.py:104-105`: P1: confidence cap is 0.85 — violates the project's `_MAX_CONFIDENCE = 0.7` convention used by most metal/macro signals. Fix: lower cap to 0.7 for parity.

`portfolio/signals/trend_slope_momentum.py:155-161`: P1: confidence cap is 1.0, not 0.7. Combined with the high collinearity of its 4 sub-signals (all derived from slope + 50d momentum), this signal can output `confidence=1.0` from what is effectively a 2-voter aggregate. Fix: cap at 0.7.

`portfolio/signals/statistical_jump_regime.py:240-241`: P1 (ACTIVE SIGNAL): confidence not capped at 0.7. `confidence = raw_confidence * (0.5 + 0.3 * persistence_factor + 0.2 * jump_recency)` followed by `min(max(confidence, 0.0), 1.0)`. Active signal with 54.4% accuracy: confidence inflation directly degrades consensus quality. Fix: add `confidence = min(confidence, 0.7)`.

`portfolio/signals/statistical_jump_regime.py:191-208`: P1 (ACTIVE SIGNAL): in low_vol regime, `vol_vote` uses SMA slope to vote BUY/SELL — but `trend_vote` (line 217) also uses SMA slope. Two of three votes are identical in the low_vol regime — confidence inflation by collinearity. Fix: make vol_vote a gate (HOLD-only when high vol) rather than a directional voter that duplicates trend.

`portfolio/signal_utils.py:90-127`: P1: `majority_vote(votes, count_hold=False)` returns confidence = `winner / active_voters`. With a single BUY vote and 7 HOLDs, active=1, confidence=1.0. Every sub-signal module that wraps `majority_vote(..., count_hold=False)` without subsequently clamping at 0.7 produces full-conviction directional votes from a single sub-indicator firing. Module-level caps are inconsistently applied (some 0.7, some 0.75, some 0.85, some 1.0). Fix: either (a) change the default to `count_hold=True` for sub-signal aggregation, or (b) enforce a project-wide post-cap convention (e.g., `confidence *= active/total`) inside `majority_vote`.

`portfolio/qwen3_trader.py:188-190`: P1: regex fallback `re.search(r"\b(BUY|SELL|HOLD)\b", text.upper())` extracts the first BUY/SELL/HOLD literal from the entire reasoning text when JSON parse fails. Reasoning text often contains phrases like "I would BUY only if..." or "this is not a HOLD setup" — false directional votes get attributed to the model. The `2026-04-30` and `2026-05-15` notes acknowledge the parse fragility but the regex still extracts the first token regardless of context. Fix: scope the regex to a small window near the end of the response (e.g., last 200 chars) and prefer JSON-codefence extraction over freeform regex.

## Important (P2 — also reported because they would activate on enable)

`portfolio/signals/futures_basis.py:38-39,71-72`: P2: docstring/comment thresholds say -1.8 / 1.8 but constants are -1.5 / 1.5. Cosmetic but signals confusing intent. Fix: align comment to constant.

`portfolio/signals/futures_basis.py:210-217`: P2: `_SYMBOL_MAP` maps XAU-USD and XAG-USD to Binance FAPI symbols (PAXGUSDT, etc) for `premiumIndexKlines` — but Binance FAPI does not publish premium index for synthetic metals like XAU. The fetch returns empty data → signal silently votes HOLD with `n_klines=0` rather than detecting that the signal is inapplicable. Fix: gate the signal to crypto tickers only, or detect "no premium index available" via explicit FAPI symbol whitelist.

`portfolio/signals/futures_basis.py:83`: P2: z-score includes current value in mean/std (`mean = np.nanmean(basis_values)` over entire 168-bar window including the latest bar). Standard practice is to z-score against history excluding the current bar. Bias is small at 168 samples but compounds with the same pattern across many signals (cot_positioning, metals_vrp). Fix: `mean = np.nanmean(basis_values[:-1])`.

`portfolio/signals/realized_skewness.py:49-180`: P2: all 4 sub-indicators derive from the same rolling skewness function. Z-score, momentum (5-bar diff of skewness), kurtosis (mathematically related to skewness via 4th-moment ratio), regime divergence (short window minus long window of the same skewness) — votes are not orthogonal. CLAUDE.md notes this signal was KILLED at 33.3% recent accuracy; the collinearity here likely explains the failure to generalize.

`portfolio/signals/copper_gold_ratio.py:43`: P2: module-level `_CACHE: dict = {}` mutated without lock by 8 worker threads from ThreadPoolExecutor. Race condition can produce a torn dict during the rare write window. Same pattern recurs across at least 5 signal modules.

`portfolio/signals/copper_gold_ratio.py:251-265`: P2: `sub_signals` returns pre-inversion votes while `action` is post-inversion. For metals tickers the displayed `ratio_zscore` BUY accompanies a composite SELL. Confusing for log analysis. Fix: invert sub_signals in lockstep with action, OR add an `inverted: true` indicator and clearly note the displayed direction is pre-inversion.

`portfolio/signals/copper_gold_ratio.py:195`: P2: same `is_metals` inversion applied to XAU-USD and XAG-USD, but silver is BOTH industrial and safe haven — inverting copper/gold for silver may produce the wrong sign in industrial-demand regimes. Fix: split is_metals into is_silver vs is_gold or weight inversion strength by ticker.

`portfolio/signals/treasury_risk_rotation.py:174-179`: P2: 4 sub-signals (direction, momentum, z-score, persistence) are all derived from the same `spread_series`. In most regimes they vote identically — effective voter count is 1-2, confidence 1.0 collapses to actual ~0.5. Fix: drop direction OR z-score (highly redundant), and let persistence be a gate that suppresses rather than votes.

`portfolio/signals/complexity_gap_regime.py:141`: P2: `avg_corr = np.mean(np.abs(corr[mask]))` — uses absolute pairwise correlation. RMT-based crisis detection in the cited Kritzman/Li paper uses signed correlation (sign matters for synchronization). Taking abs erases anti-correlated pairs that are stabilizing the system. Fix: use raw `np.mean(corr[mask])` and threshold appropriately.

`portfolio/signals/complexity_gap_regime.py:248-251`: P2: 3 sub-signals all derived from the same gap series (z-score, slope, level). Collinear voting.

`portfolio/signals/mahalanobis_turbulence.py:193-204`: P2: `_Z_HIGH=2.0` and `_Z_ELEVATED=1.5` both produce the same vote direction — the elevated branch is dead code that never disagrees with HIGH. Either remove or differentiate confidence by tier.

`portfolio/signals/cubic_trend_persistence.py:118-149`: P2: all 3 sub-signals are deterministic functions of phi: trend_direction = sign(phi), cubic_expected = sign(b*phi+c*phi^3), trend_exhaustion = inverted sign(phi) when |phi|>threshold. Only 2 distinct behaviors. Voting is not orthogonal.

`portfolio/signals/cubic_trend_persistence.py:23-26`: P2: paper-calibrated constants B_DAILY=0.0129, C_DAILY=-0.0062 are not asset-class specific. The arxiv 2501.16772 paper calibrates these from S&P 500; crypto and metals likely have different b, c. Hard-coded constants on disabled signal — would matter if enabled.

`portfolio/signals/absorption_ratio_regime.py:204-209`: P2: 3 sub-signals (z-score, delta, percentile) all derived from same AR series — strongly collinear.

`portfolio/signals/amihud_illiquidity_regime.py:114-125`: P2: 3 sub-signals: illiq_z_score, illiq_trend (5-bar slope of z-score), volume_confirm. Z-score and trend are mechanically derived from the same ILLIQ series. Volume vote conflicts with ILLIQ vote (high volume = BUY, but high ILLIQ = SELL implies low effective dollar liquidity = often low volume) — they can ANTI-correlate, producing whipsaw HOLDs. Fix: drop trend OR z-score, treat volume as gate not voter.

`portfolio/signals/adx_regime_switch.py:58-113`: P2: all 3 sub-signals derived from ADX/+DI/-DI. In strong trending regime all 3 vote BUY (or SELL) in unison — effectively 1 voter, confidence inflated.

`portfolio/signals/xtrend_equity_spillover.py:218-251`: P2: SPY_RSI and QQQ_RSI historically >0.9 correlated. SPY_MACD and SPY_TREND are both SPY-derivatives. 4 sub-signals collapse to ~1-2 effective directions. P1-borderline given how aggressively this is BUY-biased on equity uptrends.

`portfolio/signals/xtrend_equity_spillover.py:59-72`: P2: yfinance SPY/QQQ are US-cash-session only. Same staleness issue as VIX during 24/7 crypto/metals trading.

`portfolio/signals/ttm_squeeze.py:171-191`: P2: at squeeze release moment, all 3 sub-signals vote SAME direction (sign of `mom_current`). Voting is unanimous by construction — `majority_vote` returns confidence 1.0, capped at 0.7. The "3 voters" provide no statistical lift over a single momentum vote.

`portfolio/signals/treasury_risk_rotation.py:80-84`: P2: `_compute_spread_series` uses `pct_change(_SLOPE_LOOKBACK)` then `_sub_slope_momentum` takes a 21-day diff of that 65-day pct_change series. Overlapping rolling windows produce autocorrelated momentum signal. Fix: use non-overlapping or shorter overlaps.

`portfolio/signals/copper_gold_ratio.py:43`, `portfolio/signals/metals_cross_asset.py:53`, `portfolio/signals/credit_spread.py:53`, `portfolio/signals/breakeven_inflation_momentum.py:50`, `portfolio/signals/hash_ribbons.py:51`, `portfolio/signals/gold_real_yield_paradox.py:39`, `portfolio/signals/metals_vrp.py:36`, `portfolio/signals/sentiment_extremity_gate.py:34`: P2: module-level mutable caches. Most acquire a lock (`_fred_cache_lock`, `_gvz_lock`, `_yield_cache_lock`) but `copper_gold_ratio._CACHE`, `hash_ribbons._hash_cache`, `sentiment_extremity_gate._fg_cache`, `breakeven_inflation_momentum._bei_cache`, `credit_spread._oas_cache`, `crypto_evrp._DVOL_CACHE`, `crypto_evrp._DVOL_HISTORY_CACHE` do NOT — ThreadPoolExecutor with 8 workers can race them. Fix: wrap mutations in a `threading.Lock` consistent with the other modules.

`portfolio/signals/credit_spread.py:284-292`: P2: when context lacks FRED key, the signal falls back to reading `config.json` from a relative path. If CWD is not the repo root, the read fails silently and the signal abstains. The cot_positioning module fixed this exact bug at SM-P1-4 (2026-05-02). Fix: use the absolute repo-rooted path pattern.

`portfolio/signals/hash_ribbons.py:99`: P2: `pd.Timestamp.fromtimestamp(v["x"], tz="UTC").normalize()` is deprecated in newer pandas — emits FutureWarning and behavior may change. Fix: `pd.Timestamp(v["x"], unit="s", tz="UTC").normalize()` or `pd.to_datetime(v["x"], unit="s", utc=True).normalize()`.

`portfolio/signals/hash_ribbons.py:207-211`: P2: `curr_above = sma30 >= sma60` and `prev_below = sma30 < sma60`. When sma30 == sma60 (rare but possible at flat regimes), both conditions can hold for adjacent bars on a single touch, falsely firing the recovery signal. Fix: use strict inequalities consistently.

`portfolio/signals/shannon_entropy.py:262-267,287-292`: P2: composite-vote design uses 4 sub_signals but each is set to the SAME `trend_dir` (or HOLD) — the dict gives an illusion of a 4-voter majority while in fact one signal (EMA crossover) drives all 4 entries. Confidence is computed independently via `base_confidence` arithmetic, but the dashboard's sub_signals diagnostic suggests independent votes that don't exist. Fix: emit a single composite vote and move the per-sub-signal labels into `indicators`.

`portfolio/signals/oscillators.py` and `portfolio/signals/momentum.py:189-191`: P2: Williams %R `_williams_r` SELL condition `val > -20` is correct (overbought zone -20 to 0) but the code does not gate on RVOL/regime — momentum.py signal is ACTIVE and these sub-votes are unconditional. Per CLAUDE.md memo, oscillators are disabled at 35% accuracy across all tickers — momentum.py's stochastic/cci/williams_r are essentially the same oscillator family. P1-borderline: confirm whether momentum.py is collinear-with-disabled-oscillators.

`portfolio/signals/momentum.py:80-117`: P2: Stochastic crossover detection uses `k_prev/d_prev` from `iloc[-2]`. On the FIRST poll of a fresh ticker after restart, `len(k)>=2` may be true but the value at -2 could be NaN warm-up — handled with `np.isnan` check, OK.

`portfolio/signals/mean_reversion.py:462-489`: P2: detrending compounding bug is fixed but the same logic is duplicated in momentum_factors.py:332-362 (NOT fixed). Refactor opportunity: share via `portfolio/seasonality.py` or a shared helper.

`portfolio/signals/forecast.py`: P2: disabled signal carrying significant Kronos/Chronos subprocess plumbing. CLAUDE.md notes 25.6% recent accuracy. Confirm full disable before re-enabling — Kronos-shadow accounting has historically polluted Chronos composites (see line 52-70 comment block).

`portfolio/signals/network_momentum.py:54-58`: P2: `_YF_PEERS` mapping defines XAU→GC=F and XAG→SI=F but the OWN_close passed to `_compute_network_divergence` is the BINANCE FAPI close for XAU/XAG. Two slightly-different price series for "the same" asset → small basis error in own_mom_20d vs the peer GC=F momentum. Edge case but real.

`portfolio/signals/mahalanobis_turbulence.py:64`: P2: `_YF_TICKERS = ["BTC-USD", "ETH-USD", "GC=F", "SI=F", "SPY"]` — 5 assets, but the live universe includes MSTR as a separate Tier-1 ticker. MSTR exposure (leveraged BTC) is collapsed into BTC, so turbulence on MSTR-specific moves is invisible. Acceptable for crisis detection but a noted limitation.

## Minor (P3 — noted but not blocking)

`portfolio/signals/crypto_macro.py:228`: P3: `OPTIONS_TTL` is referenced before defined at module level (line 281). Works due to lazy resolution in function body, but the order is fragile to refactoring.

`portfolio/signals/copper_gold_ratio.py:135-139`: P3: SMA window fallback when len(ratio) < 200 uses 20/50 instead of 50/200. Comment in docstring (1) implies "structural direction" — 20/50 isn't structural at hourly cadence.

`portfolio/signals/connors_rsi2.py:136-139`: P3: confidence cap is 0.75, slightly above 0.7 project convention. Minor.

`portfolio/signals/futures_basis.py:138-147`: P3: `_sustained_regime` confidence based on `n_backwardation/len(valid)` is always 0.875 or 1.0 for a 7/8 or 8/8 condition. Single-step confidence quantization rather than continuous.

`portfolio/signals/shannon_entropy.py:69`: P3: `returns = close.pct_change().dropna().values` — uses pandas `pct_change()` without `fill_method=None`. Newer pandas emits FutureWarning. Fix in conjunction with other signals using same pattern.

`portfolio/signals/network_momentum.py:227`: P3: `total_weight = sum(abs(c) for c in peer_corrs.values())` uses absolute correlation as weight — defensible but unusual; standard approach is positive weights only or signed weights with normalization.

`portfolio/signals/intraday_seasonality.py:91`: P3: fallback to `datetime.datetime.now(UTC)` when df.index has no `.hour` attr — production loop always has DatetimeIndex but defensive fallback drops alignment.

`portfolio/qwen3_signal.py:107-112`: P3: 240s timeout for Qwen3 subprocess inside a 60s loop. If the GPU is contested with Plex, this timeout can cascade through ticker pool timeouts.

`portfolio/qwen3_signal.py:121-157`: P3: batch mode timeout `60 + 30 * len(contexts)` can be very long with 5 tickers (210s). Acceptable but worth monitoring.

`portfolio/signals/gold_real_yield_paradox.py:282`: P3: edge case when len(close) < 30 — `close.iloc[-len(close)]` returns the first row. For 50-row df gold_30d_return becomes a 49-bar lookback. Minor mislabel.

## Patterns (systemic issues across modules)

1. **Duplicate-vote inflation.** Repeatedly across pending-validation modules, the same underlying indicator is split into multiple "sub-signals" that vote in lockstep: `hurst_regime` (regime+trend_direction), `cross_asset_tsmom` (cross_pair+bond/equity_momentum), `network_momentum` (net_div+corr_regime), `treasury_risk_rotation` (4 spread derivatives), `ttm_squeeze` (squeeze_state+mom_direction+mom_accel at release), `vwap_zscore_mr` (vwap_z+vol_confirm), `adx_regime_switch` (regime+momentum+spread), `cubic_trend_persistence`, `absorption_ratio_regime`, `complexity_gap_regime`. The composite's `majority_vote` then sees 2-4 "identical" votes and produces inflated confidence. Recommendation: instrument signal_engine to detect sub-signal collinearity at registration time (or by tracking per-cycle agreement rates) and warn when sub_signals routinely agree >90% — the signal is effectively a 1-voter masquerading as multi-voter.

2. **Confidence cap inconsistency.** Project convention appears to be `_MAX_CONFIDENCE = 0.7` in most external-data signals (econ_calendar, news_event, metals_cross_asset, futures_basis, credit_spread, cot_positioning, etc.) but several signals don't apply it: trend_slope_momentum (1.0), connors_rsi2 (0.75), vwap_zscore_mr (0.85), statistical_jump_regime (1.0 active!), dxy_cross_asset (1.0), shannon_entropy (0.7 internal but composed via base_confidence math). Recommendation: enforce a global cap inside signal_engine.py at signal-output boundary (already partially done via accuracy gates, but make it explicit).

3. **BUY-only / SELL-only sub-signal asymmetry.** `williams_vix_fix` (3 BUY-only + 1 SELL-only), `econ_calendar` (3 SELL-capable + 1 BUY-capable), `news_event` (sentiment_shift unmatched defaults to neg), `intraday_seasonality` (hour/dow BUY-only), `gold_overnight_bias` (fix_proximity BUY-only), `crypto_macro` (expiry both BUY), `hash_ribbons` (BUY-only by design). These produce structural directional bias even on noise. Recommendation: balance every voter so it can emit BUY, SELL, or HOLD with symmetric thresholds; if a signal is inherently directional (e.g., hash_ribbons), document the bias and let the engine assign it a one-sided weight.

4. **Module-level mutable caches without locks.** copper_gold_ratio._CACHE, hash_ribbons._hash_cache, sentiment_extremity_gate._fg_cache, breakeven_inflation_momentum._bei_cache, credit_spread._oas_cache, crypto_evrp._DVOL_CACHE/_DVOL_HISTORY_CACHE. ThreadPoolExecutor has 8 workers — these race. Recommendation: standardize on `shared_state._cached` pattern (already used by some) or wrap with a module-level threading.Lock.

5. **yfinance US-RTH staleness.** Many cross-asset signals (vix_term_structure, ovx_metals_spillover, xtrend_equity_spillover, treasury_risk_rotation, mahalanobis_turbulence, complexity_gap_regime, absorption_ratio_regime, copper_gold_ratio, cross_asset_tsmom, network_momentum) fetch ^VIX/^VIX3M/^OVX/SPY/QQQ/TLT/IEF/GC=F/SI=F/HG=F via yfinance. These quotes are US-cash-session only; during off-hours (e.g., the 03:00 CET loop), yfinance returns the prior close — possibly 13+ hours old. The 24/7 crypto/metals loop votes on stale macro. Recommendation: attach a `last_quote_age_hours` indicator and force HOLD when staleness exceeds an asset-class threshold (e.g., 4h for crypto, 8h for metals).

6. **Sub_signals contract violations.** `vol_ratio_regime` returns string labels ("ranging"/"trending"/"neutral"), `sentiment_extremity_gate` returns "PASS", `tsi_chop_mr` returns "HOLD (trending)"/"ranging"/"transition", `connors_rsi2` and `adx_regime_switch` return nested dicts `{value, signal}` rather than flat `{name: vote}`. These break downstream aggregation in signal_engine and accuracy_stats. Recommendation: enforce contract — every value in `sub_signals` MUST be one of BUY/SELL/HOLD; move auxiliary labels and values into `indicators`. Add a unit test that walks every registered signal and asserts this.

7. **Current-value-included z-score bias.** cot_positioning, futures_basis, metals_vrp, gold_real_yield_paradox (subtly), absorption_ratio_regime all compute z-scores where the rolling mean/std INCLUDE the current value. The bias is small for long windows but systematic — and exactly the wrong direction (z-score is dampened toward 0 when the most extreme current value is in the window). Recommendation: when computing z of value-at-time-t vs lookback, exclude t from mean/std.

8. **Detrending compounding bug pattern.** Fixed in mean_reversion.py:477-487, still present in momentum_factors.py:332-362. Recommendation: extract shared helper into `portfolio/seasonality.py`.

9. **Threshold magic-number drift.** Many comments reference threshold values different from the implemented constants (futures_basis comment -1.8 vs code -1.5, news_event dissemination >=2.0 vs >=1.5, cot_positioning percentile minimum-10 vs canonical 52+). Recommendation: when adjusting thresholds, update both code and comments in the same commit; consider extracting thresholds to a constants module to reduce drift surface.

## Likely-Broken Disabled Signals (would surprise on enable)

These signals are currently in `DISABLED_SIGNALS` (force-HOLD). If a future operator removes them from the disable list without rereading this review, the following bugs will activate immediately:

- `gold_real_yield_paradox.py` — P0 calendar alignment bug between gold (weekend-included) and FRED yields (business days). Correlation vote is mathematically corrupted.
- `metals_vrp.py` — P0 VRP z-score is actually GVZ z-score; signal name vs math don't match.
- `hurst_regime.py` — P1 duplicate-vote inflation and direction-blind momentum interpretation.
- `cross_asset_tsmom.py` — P1 cross_pair/bond_momentum duplicate vote for XAU+XAG; missing `GLD` indicator.
- `network_momentum.py` — P1 corr_regime duplicates net_div direction.
- `vix_term_structure.py` — P1 stale-quote risk during off-hours; asset-agnostic vote ignores risk-on vs safe-haven asymmetry.
- `vol_ratio_regime.py` — P1 sub_signals contain regime LABEL strings, not vote strings — engine will mis-aggregate.
- `tsi_chop_mr.py` — P1 sub_signals contain literal "HOLD (trending)" — same engine mis-aggregation.
- `connors_rsi2.py`, `adx_regime_switch.py` — P1 sub_signals are nested dicts; engine/dashboard expect flat string votes.
- `sentiment_extremity_gate.py` — P1 emits "PASS" instead of HOLD.
- `williams_vix_fix.py` — P1 structurally BUY-biased (3 BUY-only + 1 SELL-only).
- `drift_regime_gate.py` — P1 inverted direction vs cited source (mean-revert vs momentum continuation).
- `intraday_seasonality.py`, `gold_overnight_bias.py` — P1 BUY-only gates create structural BUY bias.
- `residual_pair_reversion.py` — P1 frequency mismatch between hourly target and daily yfinance driver corrupts beta/residuals.
- `crypto_evrp.py` — P1 percentile sub-signal is DVOL percentile mislabeled as eVRP.
- `vwap_zscore_mr.py` — P1 vol_confirm sub-signal duplicates vwap_z, confidence cap 0.85.
- `trend_slope_momentum.py` — P1 confidence cap 1.0, high collinearity across 4 sub-signals.

Lower-impact disabled signals to revisit before enable: `realized_skewness` (4 sub-signals from same series — already KILLED in production at 33%), `mahalanobis_turbulence`, `absorption_ratio_regime`, `complexity_gap_regime`, `treasury_risk_rotation`, `cubic_trend_persistence`, `metals_vrp`, `breakeven_inflation_momentum`, `ttm_squeeze`, `amihud_illiquidity_regime`, `ovx_metals_spillover`, `xtrend_equity_spillover`, `copper_gold_ratio`, `dxy_cross_asset`, `futures_basis`, `hash_ribbons` — all have moderate-to-high collinearity in their sub-signals (Patterns #1) that will likely cap their attainable accuracy.

## Active signals — explicit verdict

Of the 17 active signals reviewed:
- `mean_reversion.py` — clean, well-instrumented, prior P1 fixes documented inline (P1-6).
- `momentum.py` — clean OHLCV implementation; no lookahead.
- `momentum_factors.py` — P1 detrending compounding bug (same fix that landed in mean_reversion needs to land here).
- `news_event.py` — P1 sentiment_shift default-to-negative, P1 dissemination threshold comment mismatch.
- `econ_calendar.py` — P1 post_event_relief structural BUY during event-free calm windows.
- `crypto_macro.py` — P1 expiry sub-signal BUY-biased.
- `metals_cross_asset.py` — clean overall, FRED key extraction in `_get_fred_key` is convoluted (line 100-102) but functional.
- `cot_positioning.py` — multiple P1 (min history too low, current-in-history z-score bias, unused per-sub-signal confidence, no staleness check). User note in CLAUDE.md "100% 1d, 5 sam" suggests it's untested; bugs will surface as samples accumulate.
- `credit_spread.py` — P2 module-level cache without lock; otherwise clean.
- `statistical_jump_regime.py` — P1 confidence cap 1.0, P1 vol_vote/trend_vote collinearity in low_vol regime. ACTIVE signal — these directly degrade production consensus.
- `ministral_signal.py`, `qwen3_signal.py` — Plex VRAM abstention is good; GPU gate behavior is solid. P1 regex fallback in `qwen3_trader._parse_response` (line 188-190) can misattribute votes from freeform reasoning text.
- `fear_greed` (in `portfolio/fear_greed.py`, not in signals/) — streak-tracking H26 fix is good; no issues found.

## Out of Scope but Spotted

- `portfolio/signal_engine.py` and `portfolio/signal_utils.py:90-127` — `majority_vote` default `count_hold=False` is the root enabler for confidence inflation across the codebase. Worth a separate refactor PR considering changing the project-wide default.
- `portfolio/signal_engine.py:35-36` — `_adx_cache` is keyed by `(len, first_close, last_close)` which can collide across tickers (e.g., two assets with identical first/last close at the same length). Astronomically unlikely but worth keying on a ticker prefix.
- `portfolio/qwen3_trader.py:182-187` — confidence scale rescue (raw>1 → /100) is good defensive engineering; consider applying the same defensive rescue to ministral_trader.
- `portfolio/tickers.py:65-222` — `DISABLED_SIGNALS` contains 49 entries with inline rationale comments; consider extracting to `data/disabled_signals.yaml` with structured fields (date, samples_at_disable, accuracy_at_disable, follow_up_action) to make accuracy backfill scripted.
- `portfolio/signals/__init__.py` — empty plugin discovery package; consider adding a smoke test that imports every signal module and asserts each exposes a `compute_*_signal(df, context)` callable with the standard return shape `{action, confidence, sub_signals, indicators}` where action ∈ {BUY,SELL,HOLD} and every sub_signal value ∈ {BUY,SELL,HOLD}. Catches at least 6 of the contract-violation P1s above at registration time.
