# Adversarial Review: Signal Modules (2026-05-16)

Date: 2026-05-16. Scope: 18 signal modules (momentum, mean_reversion, momentum_factors, news_event, econ_calendar, crypto_macro, metals_cross_asset, cot_positioning, credit_spread, statistical_jump_regime, futures_basis, hurst_regime, shannon_entropy, realized_skewness, mahalanobis_turbulence, orderbook_flow, futures_flow, smart_money).

---

## P1 Bugs

### crypto_macro.py:228 - OPTIONS_TTL used before definition
File: portfolio/signals/crypto_macro.py:228
Bug: Line 228 passes OPTIONS_TTL to _cached() but constant defined at line 281 (after function). Module loads correctly but poor code organization.
Fix: Move OPTIONS_TTL = 900 to module top with other thresholds.

### orderbook_flow.py:155 - Threshold 0.01 discards weak z-scores
File: portfolio/signals/orderbook_flow.py:155
Bug: Line 155 checks `if abs(ofi_z) > 0.01` to decide if z-score is available. Z-scores between -0.01 and 0.01 treated as cold-start, fallback to absolute OFI. Valid weak signals discarded.
Why: Weak z-scores (-0.005) always trigger fallback, masking the metric.
Fix: Change to `if ofi_z != 0.0` OR define _OFI_ZSCORE_MIN_VALID with documentation.

### futures_flow.py:70,98 - NaN guards correct but verbose
File: portfolio/signals/futures_flow.py:70,98
Bug: Checks `not math.isnan(price_start) and ... and price_start > 0` which is correct but redundant (NaN guard + value guard).
Fix: Simplify to single guard pattern or document defensive strategy.

---

## P2 Risks

### cot_positioning.py:363-367 - Silent CFTC API quota exhaustion
File: portfolio/signals/cot_positioning.py:363-367
Bug: When local COT history < 20 entries, code fetches from CFTC SOCRATA (public, rate-limited ~1000/min). Busy day exhausts quota. Only debug logging, not warning.
Why: Silent API failure -> all COT signals return empty -> all tickers vote HOLD for days, no user warning.
Fix: Log .warning() on first CFTC fetch; implement http_retry backoff; cache 24h TTL.

### metals_cross_asset.py:220 - Intraday degradation no confidence penalty
File: portfolio/signals/metals_cross_asset.py:220
Bug: Requires 3 of 4 intraday sources (>= 3). Single API failure downgrades to 75% voting power. No confidence adjustment.
Why: API transient failures reduce signal quality invisibly.
Fix: Apply degradation_factor when intraday_ok < 4.

### econ_calendar.py:59 - Stale dates return HOLD silently
File: portfolio/signals/econ_calendar.py:59
Bug: If all events passed, next_event() returns None. Code handles it (HOLD) but doesn't warn except in one sub-call. Outdated econ_dates.json means signal dead for weeks.
Why: Hidden staleness; system appears running but signals are dead code.
Fix: Log .critical() if all sub-calls get next_event()==None.

### news_event.py:602-604 - Variable voter count breaks majority_vote
File: portfolio/signals/news_event.py:602-604
Bug: Line 603 only appends thesis_action if `enabled and action != HOLD`. Variable voters: 7 when active, 6 when disabled. Majority threshold shifts (5/6=83%, 4/7=57%).
Why: Same market condition has different confidence based on thesis availability.
Fix: Always append thesis_action if enabled, even if HOLD.

### hurst_regime.py:189 - pandas fill_method deprecated in 2.0+
File: portfolio/signals/hurst_regime.py:189
Bug: Line 189 `close.pct_change(fill_method=None)` uses parameter removed in pandas 2.0.
Why: Crashes on upgrade.
Fix: Remove parameter.

### shannon_entropy.py:204 - pandas fill_method deprecation
File: portfolio/signals/shannon_entropy.py:204
Bug: Same deprecation as above.
Fix: Remove parameter.

### realized_skewness.py:204 - pandas fill_method deprecation
File: portfolio/signals/realized_skewness.py:204
Bug: Same deprecation as above.
Fix: Remove parameter.

### momentum.py:189 - Williams R off-by-one: > not >=
File: portfolio/signals/momentum.py:189
Bug: Line 189 `if val > -20:` for overbought. Should be `>= -20` per tech analysis. Value -20.0 exactly produces HOLD, not SELL.
Why: Signal delays 1 bar when %R spikes to exactly -20.
Fix: Change to `>= -20`.

### momentum.py:110 - Stochastic asymmetric operators
File: portfolio/signals/momentum.py:110
Bug: Line 110 `k_prev <= d_prev and k_val > d_val and d_val < 20` has asymmetric operators (prev uses <=, curr uses >). Also d_val < 20 when standard is <= 20.
Why: Signals on bar AFTER d crosses 20, not on crossing bar. Timing latency.
Fix: Consistent operators; use `d_val <= 20` for standard oversold.

### futures_basis.py:225 - All-NaN basis data returns HOLD silently
File: portfolio/signals/futures_basis.py:225
Bug: If basis_values all NaN (Binance API down), code still votes HOLD without indicating failure mode.
Why: Silent data quality failure. User assumes neutral, system is blind.
Fix: Detect all-NaN early; log .warning(); return empty.

---

## P3 Nits

### crypto_macro.py:211 - Ticker normalization assumption
File: portfolio/signals/crypto_macro.py:211
Bug: Assumes ticker = "BTC-USD". If caller passes "BTCUSD", silently returns HOLD.
Fix: Add comment documenting normalization assumption.

### cot_positioning.py:27-28 - Commit reference lacks explanation
File: portfolio/signals/cot_positioning.py:27-28
Bug: Comment references commit c5b78210 without explaining why absolute paths matter.
Fix: Expand comment with explanation of scheduled task CWD issues.

### mahalanobis_turbulence.py:43 - Cache TTL 1h for daily data
File: portfolio/signals/mahalanobis_turbulence.py:43
Bug: _CACHE_TTL = 3600 for data updating daily. 60s loop = yfinance call every hour = 6MB/hr waste.
Fix: Increase to 86400 (24h).

### futures_basis.py:41-42 - Naming: threshold vs window confusion
File: portfolio/signals/futures_basis.py:41-42
Bug: _SUSTAINED_THRESHOLD=7 (count) and _SUSTAINED_WINDOW=8 (span) named inconsistently.
Fix: Rename _SUSTAINED_THRESHOLD to _SUSTAINED_MIN_COUNT.

### smart_money.py:349 - Supply zone variable naming inverted
File: portfolio/signals/smart_money.py:349
Bug: Sets zone_low=open (top of body), zone_high=high. Semantically backwards.
Fix: Use zone_ceil/zone_floor or supply_top/supply_bot for clarity.

### momentum_factors.py:356 - Seasonality detrending iloc fragile
File: portfolio/signals/momentum_factors.py:356
Bug: Uses `.iloc[i, col_idx]` which breaks if columns reordered before call.
Fix: Add assertion or use .loc with label.

---

## Summary

Severity | Count
---------|------
P1       | 3
P2       | 9
P3       | 6

Total: 18 findings

CRITICAL: 3 pandas deprecations (hurst, shannon, skewness) will crash on pandas 2.0+ upgrade. Urgent.
KEY RISKS: Silent API quota exhaustion (cot_positioning), stale economic data (econ_calendar), dynamic voter pools (news_event).
