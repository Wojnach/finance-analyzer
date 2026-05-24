# Signals Modules — Adversarial Review (2026-05-24)

Scope: 67 modules in `portfolio/signals/*.py` against worktree
`Q:/finance-analyzer/finance-analyzer-reviews/2026-05-24`
(branch `review/fgl-2026-05-24`). Empty-baseline /fgl audit.

Files audited in depth: `connors_rsi2`, `mean_reversion`, `momentum`,
`momentum_factors`, `news_event`, `econ_calendar`, `crypto_macro`,
`metals_cross_asset`, `cot_positioning`, `credit_spread`, `statistical_jump_regime`,
`gold_real_yield_paradox`, `copper_gold_ratio`, `realized_skewness`,
`drift_regime_gate`, `williams_vix_fix`, `sentiment_extremity_gate`,
`choppiness_regime_gate`, `hurst_regime`, `vwap_zscore_mr`, `ttm_squeeze`,
`autotune_adaptive_cycle`, `trend_slope_momentum`, `adx_regime_switch`,
`amihud_illiquidity_regime`, `cubic_trend_persistence`, `intraday_seasonality`,
`hash_ribbons`, `crypto_evrp`, `network_momentum`, `xtrend_equity_spillover`,
`mahalanobis_turbulence`, `absorption_ratio_regime`, `complexity_gap_regime`,
`smart_money`, `heikin_ashi`, `trend`, `fibonacci`, `volatility`, `structure`,
`vol_ratio_regime`, `btc_etf_flow`, `breakeven_inflation_momentum`,
`gold_overnight_bias`, `residual_pair_reversion`.
Lighter scan of remaining files via grep patterns (silent except, bfill,
config.json relative load).

## Top 5

| # | Severity | File:line | Issue |
|---|----------|-----------|-------|
| 1 | P1 | `portfolio/signals/vwap_zscore_mr.py:124-125` and `portfolio/signals/autotune_adaptive_cycle.py:187-188` | Top-level `except Exception: return {HOLD,...}` swallows ALL errors without logging. Identical pattern to the prior `silent exception` audit P0s — masks math bugs, missing columns, type errors. |
| 2 | P1 | `portfolio/signals/credit_spread.py:285` and `portfolio/signals/gold_real_yield_paradox.py:265` | `load_json("config.json", ...)` uses RELATIVE path inside config-fallback branch. This is the exact failure mode fixed in batch 2 (`be4273d3`) and the P1 batch (`4adeec2d`) — when CWD ≠ repo root (PF-DataLoop launched from `C:\Windows`), the load returns None and the FRED-dependent signal silently degrades to HOLD. Apply absolute-path pattern from `cot_positioning.py:33` (`_DATA_DIR = Path(__file__).resolve().parent.parent.parent / ...`). |
| 3 | P1 | `portfolio/signals/btc_etf_flow.py:53,106-107` | Module is broken: function is named `compute(ticker, indicators, context)` not `compute_btc_etf_flow_signal(df, context)`. Module is NOT registered in `signal_registry.py`. Docstring (line 23) confirms "currently discovered but not registered". Dead code; also `except Exception: pass` silently swallows config-load failures. |
| 4 | P2 | `portfolio/signals/vwap_zscore_mr.py:90`, `portfolio/signals/autotune_adaptive_cycle.py:158-159`, `portfolio/signals/network_momentum.py:356-357`, `portfolio/signals/hurst_regime.py:284-302,333` | Sub-signal vote *copying* inflates the majority. `vol_vote = vwap_z_vote`, `corr_strength_vote = trend_vote`, `corr_regime = net_div`, `sub_signals["hurst_regime"] = trend_vote` AND `sub_signals["trend_direction"] = trend_vote` — all count the same direction TWICE in `majority_vote(...)`, breaking documented vote independence and biasing confidence upward. |
| 5 | P2 | `portfolio/signals/momentum.py:355-428` and `portfolio/signals/oscillators.py:509-578` | 8 sub-signal `try/except Exception` blocks each set `sub_signals[x] = "HOLD"` with **no logger call**. Silent NaN propagation through indicators or scipy errors mask permanent breakage of an active core signal (`momentum`). Adopt the `mean_reversion.py:504-506` pattern (`logger.debug("rsi2_mr failed", exc_info=True)`). |

## Findings (severity grouped)

### Critical (P1)

`portfolio/signals/vwap_zscore_mr.py:124-125` | P1 | silent_exception | Top-level `except Exception:\n    return {HOLD, 0.0, ...}` swallows every failure mode without logging. Masks math errors (`safe_float(z)` returning string, etc.), missing OHLCV columns, indicator regression bugs. | Replace with `logger.exception("vwap_zscore_mr failed")` and return HOLD only after logging.

`portfolio/signals/autotune_adaptive_cycle.py:187-188` | P1 | silent_exception | Same top-level pattern. The signal runs Ehlers high-pass + supersmoother + bandpass — any of which can NaN-cascade silently. | Same fix.

`portfolio/signals/credit_spread.py:285` | P1 | relative_path | `load_json("config.json", default={})` is the relative-CWD silent-fail pattern. PF-DataLoop CWD isn't guaranteed to be repo root; if FRED key is only in config.json (not env), the fallback never fires and credit_spread permanently returns empty. | Use absolute path: `_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.json"`.

`portfolio/signals/gold_real_yield_paradox.py:265` | P1 | relative_path | Same `load_json("config.json")` pattern in the config fallback. | Same fix.

`portfolio/signals/btc_etf_flow.py:53` | P1 | wrong_signature | Function is `compute(ticker, indicators, context=None)` but registry contract is `compute_X_signal(df, context=None)`. If ever added to `signal_registry.py`, the engine call site at `signal_engine.py:3614+` would crash on positional args. | Rename to `compute_btc_etf_flow_signal(df, context=None)` and refactor signature; register in `signal_registry.py`.

`portfolio/signals/btc_etf_flow.py:106-107` | P1 | silent_exception | `try: load_json(...)\nexcept Exception: pass` with no logging. Masks corrupt cache file, missing data dir, etc. | Add `logger.warning("flow cache load failed", exc_info=True)`.

### Important (P2)

`portfolio/signals/momentum.py:355-428` | P2 | silent_exception_no_log | 8 `try/except` blocks for sub-signals; each silently sets `"HOLD"` on exception. No logging. Module is an ACTIVE Tier-1 signal (53% 1d accuracy, 27K samples) — silent bugs invisible. | Add `logger.debug("<sub> failed", exc_info=True)` before HOLD fallback, matching `mean_reversion.py:504-506`.

`portfolio/signals/oscillators.py:509-578` | P2 | silent_exception_no_log | 8 sub-signals same pattern. Disabled but still computed if not in DISABLED_SIGNALS list. | Same.

`portfolio/signals/vwap_zscore_mr.py:90` | P2 | vote_double_count | `vol_vote = vwap_z_vote if vwap_z_vote != "HOLD" else "HOLD"` copies the z-score vote into the volume slot, then both are appended to `votes`. Same direction counted twice; majority becomes guaranteed when z-vote is non-HOLD. Confidence inflation. | Make `vol_vote` represent an independent direction or treat volume strictly as a confidence-gate that *suppresses* rather than amplifies the directional vote.

`portfolio/signals/autotune_adaptive_cycle.py:157-159` | P2 | vote_double_count | `corr_strength_vote = trend_vote` makes both vote in lockstep. `[bp_vote, trend_vote, corr_strength_vote]` becomes `[bp_vote, trend_vote, trend_vote]` whenever min_corr<-0.5. | Drop `corr_strength_vote` and use `min_corr` purely as a confidence modulator.

`portfolio/signals/network_momentum.py:356-357` | P2 | vote_double_count | `if corr_regime == "BUY" and net_div != "HOLD": corr_regime = net_div` — explicit synchronization. Comment at line 353-355 admits the sub-signal answers "trustworthy?" not direction — so it should be a confidence multiplier, not a vote. | Drop `corr_regime` from the vote list; use `corr_conf` to scale final confidence.

`portfolio/signals/hurst_regime.py:284-302,333` | P2 | vote_double_count | When regime == "trending", `sub_signals["hurst_regime"] = trend_vote` AND `sub_signals["trend_direction"] = trend_vote` — same value written to two keys. Final `votes = list(sub_signals.values())` includes both. Identical for mean_reverting path with `mr_extreme`. | Remove the duplicate key or vote only the regime aggregator.

`portfolio/signals/gold_real_yield_paradox.py:285-291` | P2 | date_alignment | `gold_daily_returns` is from intraday OHLCV; `yield_values` is from FRED daily series. The code aligns them by length (`gold_daily_returns[-min_len:]` vs reversed yield_daily_changes) WITHOUT date alignment. FRED has Mon-Fri data with US holidays; gold has 24/7 data. Resulting correlation computed on mis-aligned vectors. | Reindex both into a common date index (e.g., daily gold close at NY close time joined on FRED date) before computing `np.corrcoef`.

`portfolio/signals/sentiment_extremity_gate.py:1-17,159-167` | P2 | applicability_drift | Docstring states "Crypto-only" (alt.me F&G is crypto-specific), but `compute_sentiment_extremity_gate_signal` does NOT gate on ticker / asset_class. It computes and returns BUY/SELL for XAU, XAG, MSTR alike, applying F&G crypto sentiment to non-crypto assets. | Add `if ticker not in {"BTC-USD","ETH-USD"}: return HOLD with feature_unavailable=True`, mirroring `connors_rsi2.py:111-119`.

`portfolio/signals/amihud_illiquidity_regime.py:76-112` | P2 | bias_misdirection | Sub 1 maps ILLIQ z-score >2 → SELL and z<-1 → BUY (thin=SELL, thick=BUY). This is a regime *gate*, not directional. Amihud has no directional content — high ILLIQ predicts higher *risk premium* (positive returns on average) and low ILLIQ predicts the opposite. Direction mapping reverses the academic finding. Sub 3 (volume) similarly conflates "high volume" with "BUY". | Either restrict to confidence-gate semantics (multiply other signals' confidence by liquidity regime) or invert direction.

`portfolio/signals/intraday_seasonality.py:110-129` | P2 | misleading_sub_signal | `_hour_alpha_vote` returns the string "BUY" when `mult >= 1.2` regardless of trend direction. The string is reported back in `sub_signals["hour_alpha"]` as if it were a directional vote, but the final action at line 191-197 only uses `trend_vote`. Downstream consumers reading `sub_signals` see false directional signal. | Rename hour_alpha vote outputs to "ACTIVE"/"GATED" (regime states) not BUY/SELL.

`portfolio/signals/crypto_evrp.py:204-242` | P2 | misleading_sub_signal | `_evrp_percentile_signal` computes `rv_hist` (line 216) but never uses it; final percentile is computed against DVOL only (line 230-236). Sub-signal name says "evrp_percentile" but logic is DVOL percentile. Sub-signal label misrepresents what's measured. | Either compute the real eVRP rolling series and percentile-rank it, or rename to `dvol_percentile`.

`portfolio/signals/ttm_squeeze.py:69-70` | P2 | lookahead_window_internal | `y = pd.Series(y).ffill().bfill().values` inside `_linreg_value`. The `bfill()` uses later-bar values to fill earlier NaNs *within the regression window*. Because `y` is a snapshot of the most recent `period` values, this means the regression's earlier inputs are influenced by its later inputs. | Replace with `y = pd.Series(y).interpolate(method='linear', limit_direction='backward').ffill().values`, or skip NaN rows entirely.

`portfolio/signals/absorption_ratio_regime.py:112` and `portfolio/signals/mahalanobis_turbulence.py:138` | P2 | lookahead_window_internal | `valid_cols = valid_cols.ffill().bfill()` inside the rolling window. Each window is past-only (`returns.iloc[i-window:i]`), so the bfill stays inside the window — not a forward leak across the bar boundary. Still, it lets a later bar's data fill an earlier bar's NaN, which can systematically inflate correlation values on sparse data. | Drop the bfill; use only ffill so missing data on bar t uses bar t-1's value, not t+1's.

`portfolio/signals/trend_slope_momentum.py:98-99` | P2 | lookahead_minor | `raw_close = df["close"].ffill().bfill()`. `raw_close.iloc[-MOMENTUM_LOOKBACK]` could be backfilled from a later bar if the lookback bar's close was NaN. | Use ffill-only; if `iloc[-50]` is still NaN after ffill, return HOLD.

`portfolio/signals/intraday_seasonality.py:89` and `portfolio/signals/gold_overnight_bias.py:54` | P2 | silent_exception | `try: ... except Exception: pass` in tz/timestamp extraction. Wall-clock fallback masks data feed corruption (e.g., timezone-naive index when expected timezone-aware). | Add `logger.debug("timestamp extraction failed, using wallclock", exc_info=True)`.

### Low-impact (P3)

`portfolio/signals/xtrend_equity_spillover.py:122` | P3 | silent_default | `_compute_rsi` returns `50.0` (neutral) on NaN. Quietly masks broken data feed; downstream sees normal RSI value but data was bad. | Raise or return `None`, then map upstream to HOLD with `feature_unavailable=True`.

`portfolio/signals/cot_positioning.py:147,173-177` | P3 | minor_index_overlap | `nc_net_history = [nc_net]` then `for h in historical: ...append(val)`. If `historical` happens to include the current report (cot_history.jsonl can contain the latest week), the current value is double-counted in min/max. Effect on percentile: negligible (one extra duplicate value in a 156-week window). | Filter `historical` to exclude entries newer than the current `cot_data['report_date']`.

`portfolio/signals/sentiment_extremity_gate.py:38-51` | P3 | cache_key_design | `_fg_cache` is keyed only on time, not ticker. OK since alt.me F&G is global, but if `get_fear_greed(ticker)` ever becomes ticker-specific the cache will return stale data for other tickers. | Key cache by ticker, or document the global assumption.

`portfolio/signals/crypto_macro.py:228,281` | P3 | style_forward_reference | `OPTIONS_TTL` used in function body at line 228 but defined at module bottom line 281. Works correctly (module-level binding resolves at call time) but is fragile. | Move constants to top of module.

`portfolio/signals/cubic_trend_persistence.py:105` | P3 | not_lookahead | `returns_norm = (log_ret / sigma).dropna()` divides by a rolling std that includes the current bar. Standard in-sample z-scoring; not strictly lookahead since no future bar is used. Flagged here to confirm I checked and decided NO-FIX.

## Patterns observed

1. **Silent except is endemic.** `momentum.py`, `oscillators.py`, `vwap_zscore_mr.py`, `autotune_adaptive_cycle.py`, `btc_etf_flow.py`, `intraday_seasonality.py:89`, `gold_overnight_bias.py:54` all swallow exceptions without logging. The good pattern is `mean_reversion.py:504-506` which uses `logger.debug("...", exc_info=True)`.

2. **Vote inflation by copying** is structural. Several signals deliberately or accidentally write the same direction into multiple `sub_signals.values()` then majority-vote the list. Because `majority_vote` (called with `count_hold=False` in most cases) computes confidence as agreement-rate, doubling the same vote drives confidence up artificially. The legitimate "confidence modulator" pattern is in `vol_ratio_regime` (regime classification first, then per-regime directional logic with genuinely independent sub-votes).

3. **Relative path config loading** keeps reappearing despite batches 2 (`be4273d3`) and the P1 batch (`4adeec2d`) explicitly fixing it. `credit_spread.py` and `gold_real_yield_paradox.py` still use `load_json("config.json", ...)` in their fallback branches.

4. **Ticker applicability is enforced inconsistently.** `connors_rsi2`, `cot_positioning`, `hash_ribbons`, `metals_cross_asset`, `crypto_macro`, `gold_real_yield_paradox`, `crypto_evrp`, `btc_etf_flow`, `breakeven_inflation_momentum` gate on `_APPLICABLE_TICKERS`. `sentiment_extremity_gate`, `gold_overnight_bias` (partial), `intraday_seasonality` rely on engine-level filtering. The engine does filter via `_TICKER_DISABLED_SIGNALS` and per-asset-class exclusions, so practical impact is limited — but defense-in-depth is missing.

5. **Lookahead bias is mostly absent** in the actively-used signals. The few cases (ttm_squeeze linreg bfill, absorption_ratio bfill, trend_slope_momentum bfill) are confined to window-internal NaN handling rather than cross-bar leaks. None of the active 17 signals show a structural forward-data leak in their primary computation. Good.

6. **No bare divisions-by-zero or sqrt-of-negative remained** in spot-checks; the codebase uses `.replace(0, np.nan)` or explicit guards consistently. Good.

## Out-of-scope notes

The Connors RSI(2) ticker check (`connors_rsi2.py:111-113`) uses
`ticker.startswith(t.split("-")[0]) for t in _APPLICABLE_TICKERS` which
also matches `BTC-USDT`, `BTC123`, etc. Likely intentional (matches both
spot and perp variants) but worth confirming with the connors_rsi2 fix
author from batch 1 (`be4273d3`).

## Files NOT deeply audited

Of the 67 files, the following received only pattern-grep coverage rather
than full line-by-line read: `calendar_seasonal.py`, `candlestick.py`,
`claude_fundamental.py`, `cross_asset_tsmom.py`, `cryptotrader_lm.py`,
`dxy_cross_asset.py`, `finance_llama.py`, `forecast.py`, `futures_basis.py`,
`futures_flow.py` (partial), `macro_regime.py` (partial), `meta_trader.py`,
`metals_vrp.py`, `orderbook_flow.py`, `ovx_metals_spillover.py`,
`shannon_entropy.py`, `tsi_chop_mr.py`, `treasury_risk_rotation.py`,
`trend.py` (partial), `volume_flow.py`, `vix_term_structure.py`.

If any of these are promoted from shadow/pending to active status, a
follow-up focused review is warranted.
