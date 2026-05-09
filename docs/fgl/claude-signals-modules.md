# Adversarial review — portfolio/signals/ modules

Scope: `portfolio/signals/*.py` (49 files). Focus on active modules
(not in `tickers.py::DISABLED_SIGNALS`); also a scan pass over the rest.
Disabled set (as of 2026-05-09 commit aa493aec):
ml, fibonacci, futures_basis, hurst_regime, shannon_entropy,
vix_term_structure, gold_real_yield_paradox, cross_asset_tsmom,
copper_gold_ratio, network_momentum, ovx_metals_spillover,
xtrend_equity_spillover, complexity_gap_regime, realized_skewness,
mahalanobis_turbulence, crypto_evrp, hash_ribbons, drift_regime_gate,
vol_ratio_regime, residual_pair_reversion, williams_vix_fix,
treasury_risk_rotation, intraday_seasonality, calendar, futures_flow,
trend, macd, orderbook_flow, oscillators, smart_money,
claude_fundamental, sentiment.

Active modules deep-dived: momentum, mean_reversion, heikin_ashi,
structure, candlestick, volatility, volume_flow, momentum_factors,
news_event, macro_regime, metals_cross_asset, dxy_cross_asset,
credit_spread, cot_positioning, statistical_jump_regime, forecast,
crypto_macro, crypto_cross_asset, econ_calendar.

---

## Findings

[P0] crypto_cross_asset.py:257 — return dict uses `"signal"` key while every other module returns `"action"`; module is also NOT registered in `signal_registry.py::_register_defaults()` so `compute_crypto_cross_asset_signal` is never called. Dead code, plus would crash callers expecting `result["action"]`. | FIX: register the signal in `signal_registry.py` AND change return key from `"signal"` to `"action"` to match the rest of the module schema.

[P1] credit_spread.py:78-82 — `_Shim` class defines `__call__` as a `@staticmethod`; instances of a class with `staticmethod __call__` do NOT delegate properly (Python looks up `__call__` on the type, not the instance). If `portfolio.http_retry` import ever fails, `fetch_with_retry(url, **kwargs)` raises `TypeError: 'Shim' object is not callable`. | FIX: make it a normal `__call__` instance method (drop `@staticmethod`) or just assign `fetch_with_retry = requests.get`.

[P1] credit_spread.py:53,113-115 — module-level `_oas_cache` mutated without a lock under ThreadPoolExecutor (signal_engine runs 8 concurrent workers). Compare with metals_cross_asset.py:87 which DOES guard its FRED caches with `_fred_cache_lock`. | FIX: add `threading.Lock()` and wrap reads/writes, mirroring metals_cross_asset.

[P1] structure.py:79-82 — `_highlow_breakout` votes BUY when close within 2% of period high but votes SELL when close within 2% of period LOW. The latter is "sell at support" — has no breakdown check (no `current_close < period_low`); any pullback to range bottom triggers SELL. Mirror inversion of the BUY side: a momentum thesis sells AT new lows, not 2% above the low. | FIX: SELL only when `current_close < period_low` (true breakdown) or when `pct_from_low <= 0.02 AND prior_close > prior_period_low` (just-breached support); otherwise HOLD.

[P1] econ_calendar.py:142-145 — `_post_event_relief` votes BUY whenever `next_event["hours_until"] > 72`, i.e. during the empty-calendar majority of any week. With three SELL-only sub-signals + this one BUY-only sub-signal, the composite has structural BUY bias during calm windows that compounds when high-impact events are sparse. This is the same bias profile that just got `calendar` killed at 29.3% accuracy 2026-05-09. | FIX: gate the event-free BUY behind a regime-confirming check (RSI<70 or DXY weakening) so empty-calendar weeks don't auto-BUY; or convert the BUY to HOLD and only fire post-event relief 4-24h after a JUST-passed high-impact event.

[P2] news_event.py:199,270,330,402,473 — naive substring matching: `"raise"`, `"beat"`, `"cut"`, `"approval"` etc. `"Fed expected to raise rates"` matches `"raise"` and votes POS for stocks; `"Apple beat down by lawsuit"` matches `"beat"`. Multiple sub-signals (sentiment_shift, source_weight, dissemination, thesis_alignment) all share this primitive substring-match pattern — failures compound. | FIX: use `news_keywords.score_headline()` which already has phrase-level matching, or build a small `_BEARISH_RAISE_PHRASES = ("rate raise", "raises rates", "rate hike")` list mirroring the `_BEARISH_CUT_PHRASES` fix already done for "cut".

[P2] news_event.py:235-238 — `_keyword_severity_vote` votes SELL on critical/high but never votes BUY (no positive-severity branch). Combined with persist BUY-only sub-signals elsewhere, asymmetric across the composite. | FIX: add explicit symmetric "high positive severity" branch using `score_headline` returning a positive sign, OR document that asymmetry is intentional and remove this sub-signal from composite when no BUY-positive ever fires.

[P2] momentum_factors.py:81-85 — `_time_series_momentum` votes BUY/SELL on ANY positive/negative move with NO threshold. A 0.01% drift over the lookback flips the vote. Inflates noise; combined with the other 6 sub-signals that DO have thresholds, this sub always votes directionally. | FIX: add a threshold (e.g., abs(ts_mom) > 0.5 = HOLD) like every other sub-indicator in the file.

[P2] momentum_factors.py:158-162 — `_high_proximity` votes SELL when `ratio <= 0.80` (i.e., 20%+ below 52-week high). For an asset in a normal corrective pullback, this votes SELL constantly. Combined with `_low_reversal` which only votes BUY at the EXTREME (within 5% of low), there's a wide structural SELL band [25%-80%-of-high to 105%-of-low]. | FIX: tighten the threshold (e.g., 0.50 = "broken trend") or split into a momentum gate that only fires on freshly broken trend (close < SMA200 within last 20 bars).

[P2] volatility.py:86-93 — `_bb_breakout` compares same-bar `close` to same-bar Bollinger upper/lower (BB recomputed every bar including current close). Slight lookahead: if current close pushes up the BB upper, "breakout" detection is partially circular. Mild bias against legitimate breakouts because the BB widens to accommodate the close. | FIX: compare close to PRIOR-bar BB: `close.iloc[-1] > bb_upper.iloc[-2]`, mirroring how `_donchian_channel` already shifts (line 214/221).

[P2] crypto_macro.py:187-192 — `_expiry_proximity` returns BUY (not just informational HOLD as docstring claims) whenever `days <= 1` for ANY expiry (line 190-192 — non-quarterly fallback). Persistent BUY bias every Friday before crypto weekly expiry. | FIX: make non-quarterly expiry day return HOLD (only quarterly post-expiry-relief stays BUY), or convert the entire sub-signal to HOLD-only and document it as a confidence-reducer rather than a voter.

[P2] cot_positioning.py:102 — synchronous `requests.get` to CFTC API with NO retry, NO caching, every cycle when local history < 20 entries. On a fresh deploy or wiped data dir, every cycle does a 15s synchronous CFTC call inside the signal compute path (under per-ticker thread). | FIX: wrap in `_cached(...)` with 24h TTL since COT is a weekly data source, and use `http_retry.fetch_with_retry`.

[P2] volume_flow.py:323-324 — `price_up = price_change > 0 if not np.isnan(price_change) else True` defaults `price_up=True` on NaN. Bullish bias when last bar has missing data. | FIX: default to `False` and HOLD when price direction is unknown, mirroring how `_vote_volume_rsi` already gates on NaN VRSI.

[P2] news_event.py:542-544 — `_persist_headlines()` writes a JSON file to disk inside compute path on every cycle (every 60s × 5 tickers × 7 timeframes = 35 disk writes/min). Marshalls and atomic-writes. | FIX: persist only once per 5-min window (TTL guard) since headline cache TTL is 300s anyway; the disk file is for the fish monitor which reads it on demand.

[P2] metals_cross_asset.py:166-177 (`_compute_zscore`) and credit_spread.py:148-163 (`_oas_zscore_signal`) — z-score uses `history = values[:n]` which INCLUDES the current value being z-scored (`values[0]` is part of `history`). With `n=20` (the minimum) the current value contributes 5% to the mean/std, biasing z toward 0 (under-detecting extremes). For n=252 the bias is negligible. | FIX: exclude the current value: `mean = sum(values[1:n+1]) / n`, or accept the bias when n >= 100 only.

[P2] crypto_cross_asset.py:67,124 — daily ratios computed as `(close[-1] / close[-2] - 1) * 100`, but yfinance daily series has weekend gaps for stocks (DXY, SPY, gold) while crypto has none. On Monday, BTC's last-bar pair is BTC=Mon vs DXY=Friday — a 3-day gap framed as 1d. (Moot: module is dead per P0 above, but the same alignment bug should be checked elsewhere.)

[P3] macro_context.py:48-50 — DXY 5-day pct change uses `close.iloc[-5]` which is 4 periods back, not 5. Off-by-one (`iloc[-5]` is the 5th-from-last, i.e., 4 periods before iloc[-1]). Field name says `change_5d_pct`. | FIX: `close.iloc[-6]` for a true 5-day change.

[P3] structure.py:77 — `pct_from_low = (current_close - period_low) / period_low if period_low != 0 else np.inf`. If `period_low` is positive but very small (e.g., asset crashed), `pct_from_low` could be enormous; OK behavior. But `period_low == 0` returning `np.inf` is fine, however the SELL gate at line 81 `pct_from_low <= 0.02` means an `inf` would not trigger SELL. | FIX: just guard with `if period_low <= 0: return "HOLD", indicators` for clarity.

[P3] volatility.py:160 — `np.sqrt(365)` for HV annualization in `_historical_volatility`, but `_garch_signal` uses `np.sqrt(252)` (line 264, 268). Inconsistent across same module. Stocks should be 252, crypto/metals 365. | FIX: pick one convention (or branch on `ticker` if available); only matters for absolute level interpretation, not for the signal direction.

[P3] cot_positioning.py:217 — `indicators["comm_net_change"] = -change` stores a negated copy of `noncomm_net_change` and labels it `comm_net_change`. Confusing for debugging — a future schema refactor that adds a real `comm_net_change` field would silently shadow this. | FIX: rename to `derived_comm_change` or compute the actual commercial change from `comm_net` history.

[P3] candlestick.py:224-229 — `_is_green: close >= open` and `_is_red: close < open`: a flat doji (close == open) is "green" by this rule; combined with `_check_engulfing`, a doji-flat prior bar engulfed by a current red bar is labeled "bearish_engulfing" though pattern requires a real green prior. | FIX: add an explicit `_is_doji` branch returning False for both `_is_green` and `_is_red`, and skip engulfing when prior is doji.

[P3] crypto_macro.py:228 — `OPTIONS_TTL` referenced inside `compute_crypto_macro_signal` but defined at module-bottom (line 281). Works because the function only runs after import completes, but creates a load-order trap if anything ever moves the call to import time. | FIX: move the constant above the function.

[P3] crypto_macro.py:156 — `_exchange_netflow_signal` SELL gate is `consecutive_neg == 0`. If the producer ever emits `None` instead of `0`, the comparison raises `TypeError`. Wrapped in nothing — propagates. | FIX: coerce `consecutive_neg = int(netflow_data.get("consecutive_negative", 0) or 0)` at top of function.

[P3] statistical_jump_regime.py:197-208 — `vol_vote` only directional in low_vol regime; high_vol forces HOLD. With `jump_vote` and `trend_vote` both also derived from same close prices, the 3 sub-signals are highly correlated. Composite confidence is over-stated since votes are not independent. | FIX: drop `vol_vote` (it's a regime gate, not a directional voter) and apply it as a confidence multiplier in the post-vote step instead of an equal voter.

[P3] heikin_ashi.py:317-320 — Williams Alligator forward-shifts SMMA by N bars (8/5/3). Last bar's Lips comes from `lips_raw.shift(3).iloc[-1]` = `lips_raw.iloc[-4]` (4 bars old). For very fast moves, 4-bar lag may be misleading on intraday. Standard Alligator behavior; documenting only.

[P3] forecast.py:866 — `kronos_input = kronos_candles if kronos_candles and len(kronos_candles) >= 50 else (candles or [])` — falls back to 1h `candles` when Kronos-specific 5m fetch fails, but if `candles` is None *and* `df` has less than 50 rows, `kronos_input = []` → Kronos subprocess gets empty input. Subprocess error path returns HOLD; no crash. | FIX: explicit `if not kronos_input: skip Kronos` to avoid the wasted subprocess fork.

[P3] news_event.py:46-49 — `_HEADLINES_PATH` derived via `os.path.dirname` chain — works but fragile if the file moves. The other modules use `Path(__file__).resolve().parent.parent.parent / "data"`. Cosmetic. | FIX: standardize on the Path form.

[P3] mean_reversion.py:198-203 — `_low_reversal` BUY (ratio<=1.05 AND last_3_green) overlaps with SELL (ratio<=1.01). First-match wins so behavior is BUY when both fire (ratio in [1.0, 1.01] AND last_3_green) — that's the right call but the logic depends on ordering, not explicit guards. | FIX: make breakdown check `ratio <= 1.01 AND NOT last_3_green` to make the disjoint condition explicit.

[P3] news_event.py:330-331 — `_source_weight_vote` POS keyword check is again primitive: `kw in title for kw in ("beat","upgrade",...)` plus a `"rate cut" in title` literal substring. Inherits the same false-positive surface as P2 above.

[P3] structure.py:117-139 — RSI centerline cross uses absolute thresholds (>60 BUY, <40 SELL). For chronically overbought assets (BTC during a bull run), this generates persistent BUY votes regardless of whether RSI is ACTUALLY crossing. Same pattern in many places: name says "cross" but logic is just level. | FIX: rename to `_rsi_level` or add a true cross check using `iloc[-2]`.

[P3] metals_cross_asset.py:101 — `getattr(getattr(cfg, "golddigger", None), "fred_api_key", "") if hasattr(cfg, "fred_api_key") or hasattr(cfg, "golddigger") else ""` — convoluted; the outer ternary causes a `getattr` on `None` if neither attr exists (None.fred_api_key blocked by ternary, but `getattr(None, "fred_api_key", "")` returns ""). Works but unreadable. | FIX: refactor to explicit if/elif/else.

[MAYBE-P2] heikin_ashi.py:148-166 — `_ha_trend_signal` sets `all_red_no_upper_wick = True` and unconditionally inspects every candle in the tail; for a tail of mixed-color candles the tracking flags both flip to False legitimately. Logic appears correct but the loop tests both sides on every candle (not short-circuit). Inefficient but not a bug. | FIX (optional): split into two passes for clarity.

[MAYBE-P3] orderbook_flow.py — disabled but still computes `get_microstructure_context` if accidentally re-enabled per ticker via overrides; the function does multiple network calls inside the signal cycle (`get_orderbook_depth`, `get_recent_trades`) per timeframe per ticker. | FIX: cache lookups per (ticker, cycle).

---

## Counts

- Total findings: 30
- P0: 1 (`crypto_cross_asset` orphan + key-name mismatch)
- P1: 4 (Shim TypeError, no-lock cache, structure SELL-at-support, econ calm-window BUY)
- P2: 12
- P3: 12
- MAYBE: 2

Active-module concentration: news_event (4), momentum_factors (2), structure (2), volatility (2), crypto_macro (2), metals_cross_asset (2), credit_spread (2), cot_positioning (2), econ_calendar (1).

The single highest-impact fix is `crypto_cross_asset.py` — it's a registered-but-broken signal that would compound the macro-regime-shift bias the engine is currently fighting. Second highest: `econ_calendar` post-event-relief BUY, which has the same structural bias profile (`6/8 BUY-only sub-signals`) that just killed `calendar` at 29.3% accuracy this morning.
