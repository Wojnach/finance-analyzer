# Signals-Modules Review — subagent result (pr-review-toolkit:code-reviewer)

Totals: 9 P1 (🔴), 19 P2 (🟡). Confidence floor 80.

## P1 / 🔴

1. **mahalanobis_turbulence.py:99** — `_cached("...", _do_fetch, ttl=_CACHE_TTL)` mis-orders args. `_cached(key, ttl, func, *args)` signature → `_do_fetch` passed as `ttl` (function not number) AND `ttl=_CACHE_TTL` kwarg → duplicate binding → TypeError. **Module dead-on-arrival. Never voted.** Fix: `_cached("...", _CACHE_TTL, _do_fetch)`.

2. **complexity_gap_regime.py:92** — Identical `_cached` arg-order bug. Module dead. Same fix.

3. **vol_ratio_regime.py:256-260** — Sub_signals emit `"ranging"/"trending"/"neutral"` not canonical `BUY/SELL/HOLD`. Contract violation; downstream consumers misclassify or drop.

4. **tsi_chop_mr.py:33-42, 142-145** — Same contract violation: `"HOLD (trending)"`, `"trending"` etc.

5. **vwap_zscore_mr.py:90** — `vol_vote = vwap_z_vote if vwap_z_vote != "HOLD" else "HOLD"` duplicates the same vote when vol >1.2x avg. Bypasses `if active_votes < 2: HOLD` gate.

6. **hurst_regime.py:284-285, 301-302** — `sub_signals["hurst_regime"] = trend_vote; sub_signals["trend_direction"] = trend_vote` — same vote written twice under different keys; `majority_vote` double-counts.

7. **network_momentum.py:356-357** — `corr_regime` echoed to same direction as `net_div`; 2 votes from 1 source.

8. **crypto_evrp.py:204-242** — `_evrp_percentile_signal` and `_evrp_momentum_signal` compute DVOL rank/momentum, NOT eVRP (DVOL − RV). `rv_hist` computed at line 215-227 then never used. Indicator exports `"evrp_percentile"` misnamed.

9. **trend_slope_momentum.py:113-150** — 4 sub-signals all derived from 2 scalars (z_clipped, momentum). Common case: all 4 vote same direction → 4 votes from 1 signal, `majority_vote` reports 100% confidence on a single source.

## P2 / 🟡

- vix_term_structure.py:37 — `_Z_THRESHOLD=0.0` votes on every sign-of-deviation; filter doesn't filter. Plus backwardation_flag + contango_depth overlap → 3 of 4 sub-signals fire together.
- news_event.py:272-295 — Default-bearish for moderate-severity headlines not matching positive whitelist; SELL bias. Plus `_keyword_severity_vote` is SELL-only.
- intraday_seasonality.py:110-129 — `_hour_alpha_vote` and `_dow_vote` BUY-only, never SELL. Hour-of-day is direction-agnostic. Structural BUY bias.
- gold_overnight_bias.py:35-41, 118-140 — `_AM_FIX_HOUR=10, _AM_FIX_MIN=30` UTC year-round, but LBMA AM fix is 10:30 London local → 09:30 UTC during BST (~7 mo/yr). Plus `_fix_proximity_vote` BUY-only.
- cross_asset_tsmom.py:148-171, 228 — (a) Metals get same direction logic as equities; positive SPY momentum = risk-on = NEGATIVE for safe havens, but voted BUY for XAU/XAG. (b) Line 228 fetches `"GLD"` but `_YF_TICKERS = ["TLT","SPY","GC=F","BTC-USD"]` — GLD never fetched; indicator always None.
- copper_gold_ratio.py:248-265 — Final `action` inverted for metals but `sub_signals` left pre-inversion. Same pattern in `treasury_risk_rotation.py:184-198` and `xtrend_equity_spillover.py:228-251`. Invariant `majority_vote(sub_signals)==action` violated in 3 modules.
- breakeven_inflation_momentum.py:189-211 — 3 sub-signals (momentum/level/acceleration) all gate on same trend; triple-weighted trend voter.
- realized_skewness.py:55-69 — `skew_val` over `lookback=252`; `rolling_skew` window same; consecutive rolling values share 251/252 obs → tiny std → z-score explodes on small drifts.
- mean_reversion.py:141-158 — `_consecutive_days` borderline bug; flat-bar mid-streak break logic fragile (P2/P3 boundary).
- futures_basis.py:38-39, 71-72 — Doc says z±1.8 thresholds; code uses ±1.5. Stale docstring or stale code.
- metals_vrp.py:169-176 — Historical VRP z-score subtracts single `rv_mean` (scalar) from each GVZ point instead of contemporaneous RV[i]; reduces to "GVZ z-score" not VRP. Also `sqrt(252)` annualization for 24/7 metals (should be sqrt(365)).
- cubic_trend_persistence.py:125-146 — 3 sub-signals share `phi` input; moderate-trend regime gets 2 of 3 votes correlated.
- econ_calendar.py:48-68, 96-111 — `_event_proximity` (hours<=4) and `_pre_event_risk` (events_within_hours(4)) overlap; same event drives both sub-signals SELL. Plus `_event_type_info` SELL on FOMC/CPI within 24h. 3/5 SELL composite from 1 event.
- xtrend_equity_spillover.py:136-172 — SPY RSI + QQQ RSI ~0.9 correlated → 2 votes from 1 source. Plus MACD threshold ±0.5 absolute assumes SPY ~$500 — silently breaks on ticker config change.
- news_event.py:137-147 — `_SECTOR_REP_TICKER` map: NVDA, AAPL, LMT, PLTR, TTWO, VRT — ALL removed from Tier 1 (Mar 15 + Apr 09). Sector-peer fetches burn NewsAPI quota on dead tickers.
- crypto_macro.py:228 vs 281 — `OPTIONS_TTL` used at 228, defined at 281. Works today (module-level execution order) but fragile to refactor.
- gold_real_yield_paradox.py:120-144 — `_paradox_spread` BUY-only voter; structural BUY tilt.
- cot_positioning.py:197-223 — `_sub_commercial_change` reads `noncomm_net_change` and inverts; name misleads. If `comm_net_change` added upstream, code silently keeps using inverse.
- cryptotrader_lm.py:162 + finance_llama.py:218 — LLM confidence not clamped to [0,1]; hallucinated values propagate.

## Cross-cutting (signals-modules-specific)

1. **Sub-signal independence violated** in 8+ modules. `majority_vote` weighs correlated voters as independent. Fix at architecture level: collapse redundant voters, use agreement as confidence multiplier.
2. **Metals-inversion pattern leaves sub_signals pre-inversion** in 3 modules. Helper function for `_metals_invert(result)` would fix all three.
3. **Non-canonical sub_signal strings** in 3 modules. Add `_validate_signal_result` strict check in signal_engine to reject non-canonical.
4. **BUY-only / SELL-only voters** in 5 modules. Accuracy gate detects hit rate, not directional fairness.
5. **2 dead modules** (mahalanobis_turbulence, complexity_gap_regime). Pending validation but TypeError on import-time call — never voted, never could have produced data. Grep `_cached\([^,]+,\s*[^,]+,\s*ttl=` to lint.
