# Adversarial Review — signals-modules (Claude-independent)

32 plugin signal modules, called by signal_engine for BUY/SELL/HOLD + confidence + rationale.

## Critical Findings

### P0-1: futures_flow NaN slips past `or 0` defense
**File:** `portfolio/signals/futures_flow.py:64-65, 92, 204`
`recent_oi = [d.get("oi", 0) or 0 for d in oi_history[-_MIN_HISTORY:]]` — if `d["oi"]` is `np.nan`, `0 or np.nan` returns `np.nan` (NaN is truthy-false... actually `0 or NaN` returns NaN; and `NaN or 0` returns 0). Either way, `math.isnan()` guards are inconsistent between `_oi_trend` and `_oi_divergence`. Division by NaN silently produces NaN signal.
**Fix:** explicit `math.isnan()` check at buffer entry; replace with 0.0.

### P0-2: Candlestick pattern-direction depends on trend but trend can degrade to "flat"
**File:** `portfolio/signals/candlestick.py:259-272, 343-354`
Hammer in uptrend = hanging man (SELL). If trend detection falls to "flat" (div-by-zero or low data), hammer returns HOLD; if the NEXT pattern is bullish engulfing, composite votes BUY despite bearish intent. No per-pattern backtest validation by asset class.
**Fix:** require trend state != "flat" for pattern vote to count; add per-asset-class backtest gate.

### P0-3: forecast.py stale 30-min accuracy cache
**File:** `portfolio/signals/forecast.py:491-506, 652`
`_load_forecast_accuracy()` caches per-ticker with TTL=1800s. If accuracy degrades 55% → 48% in the 25-min window, gate still greenlights raw vote.
**Fix:** TTL → 300s; log ticker accuracy every cycle.

## P1

### 4. crypto_macro.py OPTIONS_TTL undefined at first call
**File:** `portfolio/signals/crypto_macro.py:228-230, 281`
`OPTIONS_TTL` used on line 228 but defined at line 281 — NameError or undefined cache TTL → fallback HOLD forever.
**Fix:** move `OPTIONS_TTL = 900` to constants section (line 25).

### 5. calendar_seasonal.py hardcoded holiday dates — broken for 2026+
**File:** `portfolio/signals/calendar_seasonal.py:210-220`
MLK Day hardcoded `(1, 20)` but it's 3rd Monday (Jan 19 in 2026). Memorial Day hardcoded `(5, 26)` but May 25 2026. Juneteenth moves on weekends.
**Fix:** use `python-holidays` library: `holidays.US(years=year).items()`.

### 6. econ_calendar.py timezone-naive date extraction
**File:** `portfolio/signals/econ_calendar.py:30-36`
`_get_current_date(df)` does `.replace(tzinfo=UTC)` on an already-aware datetime → wrong local time. DST spring-forward (2AM→3AM) misses events.
**Fix:** `pd.Timestamp.tz_convert('UTC')` explicit.

### 7. claude_fundamental.py ticker string unescaped in prompts
**File:** `portfolio/signals/claude_fundamental.py:309-346`
`f"{ticker}: ${price} RSI={rsi_val}..."`. If ticker adversarial (synthetic pair, user-custom), could inject prompt. Headlines embedded similarly in news_event.
**Fix:** `json.dumps(ticker)` to escape.

## P2

### 8. cross_asset_tsmom.py GLD vs GC=F ticker mismatch
**File:** `portfolio/signals/cross_asset_tsmom.py:50-51, 227-228`
`_YF_TICKERS` has "GC=F" (gold futures). Code at 227 tries `_yf_ret("GLD")` — different symbol. `cross_pair_ret_63d` always None for gold.
**Fix:** use "GLD" consistently; unit test yfinance supports all tickers.

### 9. futures_basis.py bps vs % scale confusion
**File:** `portfolio/signals/futures_basis.py:245`
`basis_pct = safe_float(current_basis * 100)` — but threshold `_SUSTAINED_MIN_ABS = 0.0002` is basis fraction. Comparing 0.01 bps against 0.02 bps — wrong order of magnitude.
**Fix:** use basis fractions consistently; scale only for display.

### 10. dxy_cross_asset.py hardcoded 0.15% threshold — regime-blind
**File:** `portfolio/signals/dxy_cross_asset.py:28-30`
Same threshold on XAU (low DXY beta) and XAG (high DXY beta). No regime scaling.
**Fix:** scale threshold by regime vol multiplier (trending 2x, high-vol 1.5x).

### 11. volatility.py GARCH ratio > 1.2 threshold uncalibrated
**File:** `portfolio/signals/volatility.py:264`
Magic number; no backtest evidence attached.
**Fix:** validate ratio > 1.2 achieves >55% accuracy before using.

### 12. momentum.py NaN in indicator fields propagates to rationale JSON
**File:** `portfolio/signals/momentum.py:96-100, 134-141`
`_stochastic` returns `float("nan")` on NaN input; serialization path is fragile.
**Fix:** return `0.0` with explicit HOLD action.

## P3

### 13. cot_positioning.py 156-week percentile ignores regime shift
Mixing 2022 inflation + 2026 disinflation → "normal" reading maps to extreme percentile.
**Fix:** adaptive lookback with recent-window weighting in high-vol.

### 14. calendar_seasonal.py `_MIN_WINNING_VOTES = 2` too lenient
With 8 sub-signals, 2/8 quorum = 25% — permanent BUY tilt from structural sub-signals (month-end, Santa, pre-holiday).
**Fix:** raise to 3; weight by backtested accuracy.

## Per-Module One-Liners
1. **calendar_seasonal** — 8-sub majority; hardcoded dates broken 2026.
2. **candlestick** — 5 patterns × trend; trend-degradation risk.
3. **claude_fundamental** — 3-tier LLM cascade; ticker not escaped.
4. **cot_positioning** — CFTC contrarian; 156w lookback regime-blind.
5. **credit_spread** — HY OAS z-score; thresholds calibrated on old regimes.
6. **cross_asset_tsmom** — 4-sub cross-asset mom; GLD/GC=F mismatch.
7. **crypto_macro** — 5-sub crypto-only; `OPTIONS_TTL` undefined crash.
8. **dxy_cross_asset** — 1h DXY move threshold; not regime-scaled.
9. **econ_calendar** — 4-sub proximity; timezone-naive boundary.
10. **fibonacci** — 5-sub retracement/extension; 50-bar min; OK.
11. **forecast** — Kronos+Chronos; 30min accuracy cache stale.
12. **futures_basis** — 4-sub basis; bps vs % unit confusion.
13. **futures_flow** — 6-sub OI/LS/funding; NaN leak via `or 0`.
14. **gold_real_yield_paradox** — inverse gold/real-yield; not fully reviewed.
15. **heikin_ashi** — smoothed OHLC; not fully reviewed.
16. **hurst_regime** — R/S analysis; not fully reviewed.
17. **macro_regime** — 6-sub macro; not fully reviewed.
18. **mean_reversion** — BB + ATR oversold; not fully reviewed.
19. **metals_cross_asset** — gold/silver cross; not fully reviewed.
20. **momentum** — 8-sub RSI/stoch/CCI/etc; NaN in indicators.
21. **momentum_factors** — linear regression; not fully reviewed.
22. **news_event** — 5-sub headline; JSON not escaped.
23. **orderbook_flow** — bid/ask imbalance; not fully reviewed.
24. **oscillators** — 8-sub Awesome/Aroon/etc; NaN handling OK.
25. **shannon_entropy** — info-theoretic; not fully reviewed.
26. **smart_money** — whale flow + top-trader; not fully reviewed.
27. **structure** — HTF support/resistance; not fully reviewed.
28. **trend** — MA crosses + slope; not fully reviewed.
29. **vix_term_structure** — VIX curve; equity-only; not fully reviewed.
30. **volatility** — 7-sub BB squeeze/GARCH/etc; GARCH ratio uncalibrated.
31. **volume_flow** — OBV + MFI; not fully reviewed.
32. **structure.py (dup)** — see #27.

## Reviewer confidence
0.70 — ~50% coverage in depth. P0/P1 findings confidence 0.95; P2/P3 0.70. Disabled signals (crypto_macro, cot_positioning, credit_spread) reviewed but not audited for overhead (still fetch data on force-HOLD).
