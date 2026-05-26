# Signals Modules — Adversarial Review (2026-05-26)

Scope: 70 modules in `portfolio/signals/*.py`. Plugin contract verified:
each module exposes `compute_<name>_signal(df, [context])` returning
`{action, confidence, sub_signals, indicators}`. Registry at
`portfolio/signal_registry.py` lazy-loads on demand; `sub_signals` are
informational only (the engine consumes top-level `action` / `confidence`).

Focus this pass: modules NOT deeply audited on 2026-05-24 — `news_event`,
`calendar_seasonal`, `macro_regime`, `dxy_cross_asset`, `econ_calendar`,
`cross_asset_tsmom`, `treasury_risk_rotation`, `metals_vrp`,
`orderbook_flow`, `volume_flow`, `forecast`, `hash_ribbons`,
`bocpd_regime_switch`, `btc_gold_correlation_regime`, `drift_regime_gate`,
`futures_basis`, `shannon_entropy`. Also confirmed status of 2026-05-24
findings.

## Top 3

| # | Severity | File:line | Issue |
|---|----------|-----------|-------|
| 1 | P1 | `portfolio/signals/cross_asset_tsmom.py:148-171` | `_compute_bond_momentum` and `_compute_equity_momentum` return the same direction (`BUY`/`SELL`) for every ticker, ignoring asset class. TLT up = risk-off = SELL crypto / BUY metals; SPY up = risk-on = BUY crypto / SELL metals. Code returns `BUY` whenever the return is positive regardless. Polarity is reversed for half the ticker universe. |
| 2 | P1 | `portfolio/signals/treasury_risk_rotation.py:182-185` | Inverts `action` for safe-haven tickers (XAU/XAG) but leaves the `sub_signals` dict unchanged — downstream consumers, accuracy backfill, and correlation analysis see sub-signals voting BUY while the composite returns SELL. Telemetry is permanently inconsistent for metals. |
| 3 | P1 | `portfolio/signals/econ_calendar.py:44` | `last_time.to_pydatetime().replace(tzinfo=UTC)` reinterprets the wall-clock of a tz-aware Timestamp as UTC without converting. A 10:00 CET timestamp becomes 10:00 UTC, silently shifting by 1-2h — directly corrupts the `hours_until_event` math used by FOMC/CPI proximity sub-signals. Use `.tz_convert("UTC")` instead. |

## Findings (severity grouped)

### Critical (P1)

`portfolio/signals/cross_asset_tsmom.py:148-171` | P1 | direction_polarity | `_compute_bond_momentum` returns `BUY` when TLT > 0 (line 154-157), `_compute_equity_momentum` returns `BUY` when SPY > 0 (line 167-170). Both apply uniformly across all tickers. For risk assets (BTC/ETH/MSTR), TLT-up = risk-off = bearish, but code votes BUY. For safe havens (XAU/XAG), SPY-up = risk-on = bearish for metals, but code votes BUY. Sub-signals are systematically miscalibrated for ~3 of 5 tickers each. | Apply asset-class polarity table similar to `treasury_risk_rotation._invert`.

`portfolio/signals/treasury_risk_rotation.py:182-185` | P1 | telemetry_inconsistency | After computing `action` from sub-signals via `majority_vote`, the code calls `_invert(action)` for XAU/XAG but `sub_signals` dict still reports the pre-invert votes. The journal/backfill sees `slope_direction=BUY, slope_momentum=BUY, ... action=SELL`. This breaks accuracy correlation analysis and any downstream "vote agreement" metrics. | Invert sub-signal directions too when `is_safe_haven`, or invert spread sign before scoring sub-signals.

`portfolio/signals/econ_calendar.py:44` | P1 | tz_corruption | `last_time.to_pydatetime().replace(tzinfo=UTC)` does NOT convert — it replaces tzinfo on the naive wall-clock. A CET Timestamp `2026-05-26 10:00+02:00` becomes `datetime(2026,5,26,10,0, tzinfo=UTC)` — 2h shift. `_event_proximity` / `_pre_event_risk` then compute wrong `hours_until`, mis-firing the SELL-before-FOMC gate by hours. | Replace with `last_time.tz_convert("UTC").to_pydatetime()` (or `astimezone(UTC)` after to_pydatetime).

`portfolio/signals/news_event.py:555-589` | P1 | silent_exception_active | Active Tier-1 signal (50.6% 1d) has 7 try/except blocks, none of which log. `_sentiment_shift` and `_dissemination_vote` both perform keyword scoring that can fail on malformed headlines (None title etc) — failures collapse the sub-signal to HOLD silently. With 7 silent-collapses, the composite can degrade from BUY to HOLD invisibly. | Add `logger.debug("<sub> failed", exc_info=True)` in each except, following `econ_calendar.py:217,223,229,235,241` pattern.

`portfolio/signals/hash_ribbons.py:262,166-179` | P1 | timeframe_mismatch | Module fetches DAILY hashrate from blockchain.info but applies `PRICE_FAST=10`, `PRICE_SLOW=20` bars of the OHLCV `close` series which can be 15m/1h/3h/12h/1d depending on which timeframe signal_engine dispatches. The cited "9 signals 89% win rate" research used daily bars; on a 15m bar this is a 2.5-hour SMA, not a 10-day. Different signal than advertised, fired ~96x more frequently. | Either gate to daily timeframe only (`if context.get("timeframe") != "1d": return hold`) or resample close to daily before applying `_price_momentum_filter`.

`portfolio/signals/credit_spread.py:285` | P1 | relative_path | [REPEAT 2026-05-24] `load_json("config.json", default={})` in config fallback. CWD-dependent. Still unfixed. | Use absolute `_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.json"`.

`portfolio/signals/gold_real_yield_paradox.py:265` | P1 | relative_path | [REPEAT 2026-05-24] Same relative `load_json("config.json")` in FRED-key fallback. Still unfixed. | Same fix.

`portfolio/signals/vwap_zscore_mr.py:124-125` | P1 | silent_exception_active | [REPEAT 2026-05-24] Top-level bare `except Exception: return HOLD` with no logger. Still unfixed. Hides indicator computation failures on what is supposed to be an active mean-reversion voter. | Add `logger.exception("vwap_zscore_mr failed")` before return.

`portfolio/signals/autotune_adaptive_cycle.py:187-188` | P1 | silent_exception_active | [REPEAT 2026-05-24] Same pattern. Still unfixed. Ehlers bandpass / supersmoother chain is fragile to NaN. | Same fix.

`portfolio/signals/btc_etf_flow.py:53,106-107` | P1 | wrong_signature_dead_code | [REPEAT 2026-05-24] Function `compute(ticker, indicators, context)` (wrong name + signature) and not registered. Silent except at 106-107 still present. Module is dead code; the registry doesn't load it but it still ships. | Either rename and register, or delete the file from the repo to remove confusion.

### Important (P2)

`portfolio/signals/metals_vrp.py:168-176` | P2 | math_error_disguised_as_signal | `rv_for_vrp = gvz_hist_for_z - rv_mean` subtracts a single SCALAR `rv_mean` (mean of last 60 RV values) from a 60-element GVZ array. The "VRP distribution" thus has `std = std(gvz_history)` and `mean = mean(gvz_history) - rv_mean`. `vrp_z = (current_gvz - current_rv - vrp_mean) / vrp_std` reduces algebraically to `(current_gvz - mean_gvz)/std_gvz + (rv_mean - current_rv)/std_gvz`. The "VRP" label is misleading — the signal is dominated by GVZ z-score, not the realized-vol differential. | Construct historical RV series aligned by date and compute element-wise `vrp_hist = gvz_hist - rv_hist` before z-scoring.

`portfolio/signals/metals_vrp.py:206-217` | P2 | vote_inflation | Sub-signal 3 `vrp_momentum` requires both `gvz_change > 2.0` AND `vrp_z > 0` to vote BUY (and the inverse for SELL). By construction this can only fire in the same direction as `vrp_z` (sub-signal 1). Two sub-votes are perfectly correlated. | Drop the `vrp_z` agreement gate, or remove the redundant sub-signal entirely.

`portfolio/signals/calendar_seasonal.py:210-220` | P2 | wrong_holiday_dates | `_US_HOLIDAYS` is a hardcoded `(month, day)` tuple list using approximate dates for floating holidays — MLK Day, Presidents' Day, Memorial Day, Labor Day, Thanksgiving. Comment at line 209 explicitly admits "does not handle observed-date shifts". In 2026, Thanksgiving is Nov 26, not the hardcoded Nov 27 — pre-holiday signal fires the day BEFORE the wrong date, hitting Nov 25 (and missing the real pre-Thanksgiving Wednesday Nov 25 in 2026 by coincidence). For other years the miss is by days. | Use the `holidays` Python package (already used elsewhere in the codebase via `simple-timeserver` MCP) or compute "nth Monday of month" properly.

`portfolio/signals/calendar_seasonal.py:73-108` | P2 | timeframe_misapplied | `_turnaround_tuesday` uses `df["close"].iloc[-2]` as "yesterday's Monday close" — but signal_engine calls this on every timeframe (15m, 1h, 3h, 12h, 1d). On 1h bars `iloc[-2]` is one hour ago, not yesterday. Same applies to `_day_of_week_effect` which uses the timestamp of the LAST bar as "today" — for 15m crypto data running 24/7 the "Monday close" doesn't exist. | Gate to daily timeframe only, or resample to daily before extracting dow/turnaround signals.

`portfolio/signals/cross_asset_tsmom.py:107-127` | P2 | timeframe_label_mismatch | `_compute_own_tsmom` claims "252d momentum" but uses `lookback = min(252, n - 1)` bars. On a 1h chart with 60 rows (MIN_ROWS), lookback degenerates to 59 bars (~2.5 days). The threshold `0.005` is calibrated for daily; on 5m bars a 0.5% move is hourly noise. Signal output diverges from the cited research. | Compute lookback in HOURS or DAYS and reject if df timeframe < daily.

`portfolio/signals/cross_asset_tsmom.py:228` | P2 | dead_indicator | `"gld_ret_63d": safe_float(_yf_ret("GLD"))` — but `_YF_TICKERS = ["TLT", "SPY", "GC=F", "BTC-USD"]`. GLD is never fetched. Indicator always None, misleading anyone debugging the signal. | Either fetch GLD or remove the indicator key.

`portfolio/signals/orderbook_flow.py:140-150` | P2 | vote_inflation | Sub-signal 3 `vpin` is gated on the SAME `tir` (trade-imbalance ratio) sign as sub-signal 2 `trade_flow`. When VPIN is high, sub-2 and sub-3 vote the same direction by construction. | Decouple: make VPIN a confidence multiplier, or use a different directional source (e.g., depth_imbalance sign) for VPIN's direction.

`portfolio/signals/treasury_risk_rotation.py:174-180` | P2 | sub_signal_collinearity | `v_direction = _sub_slope_direction(spread_current)` and `v_zscore = _sub_slope_zscore(spread_series)` are both monotonic functions of `spread_current`. When spread is positive and large, both vote BUY; when zero or noisy, both HOLD. Confidence inflation. | Drop one or use the z-score purely as a confidence weight on the direction vote.

`portfolio/signals/volume_flow.py:324` | P2 | default_bias | `price_up = price_change > 0 if not np.isnan(price_change) else True` — when last-bar diff is NaN, defaults to TRUE (BUY direction). With `vrsi > 70` this then votes BUY. The comment claims "default neutral bias" but True is not neutral; it's bullish. | Use `if np.isnan(price_change): return HOLD-vote` for `_vote_volume_rsi`.

`portfolio/signals/drift_regime_gate.py:38-58` | P2 | direction_vs_paper | Module docstring cites arxiv 2511.12490 (drift regimes for momentum trading). The paper's regime test gates MOMENTUM signals on drift. This implementation inverts: `frac > 0.60 → SELL (mean-reversion)`, `frac < 0.40 → BUY`. The signal is a mean-reverter, not a regime gate for momentum as cited. Recent accuracy (68.1%) may reflect mean-reversion working on these tickers but the design rationale doesn't match the citation. | Either rewrite to actually gate momentum (return HOLD when not in drift) or rewrite the docstring to honestly describe the mean-reversion design.

`portfolio/signals/news_event.py:555-617` | P2 | sub_signal_collinearity | Five of six sub-signals (`headline_velocity`, `keyword_severity`, `sentiment_shift`, `source_weight`, `dissemination`) derive direction from the SAME `keyword_severity()` classification of the SAME headline list, just with different filters. When news is dominantly negative, all 5 vote SELL by construction. `_sector_impact_vote` is the only structurally independent sub-signal. | Score sub-signals on orthogonal signals: velocity-of-volume vs sentiment vs source-credibility, not all on severity.

`portfolio/signals/btc_gold_correlation_regime.py:88-91` | P2 | fragile_tz_check | `if hasattr(target_close.index, "tz"):` is True for any DatetimeIndex even when `index.tz is None`. Then `tz_localize(None) if index.tz is None else tz_convert(None)` — `tz_convert(None)` on a non-tz-aware index raises `TypeError`. For a non-DatetimeIndex (e.g. integer RangeIndex), `tz_localize` doesn't exist at all → AttributeError. Falls through to the outer `except`? There's none — would propagate. | Guard with `isinstance(index, pd.DatetimeIndex)` and only call `tz_*` when warranted.

`portfolio/signals/momentum.py:355-428` | P2 | silent_exception_no_log_active | [REPEAT 2026-05-24] 8 try/except blocks for sub-signals, no logging, on an active Tier-1 signal (52.9% 1d, 27K sam). Still unfixed. | Add `logger.debug` per `mean_reversion.py:504-506` pattern.

`portfolio/signals/oscillators.py:509-578` | P2 | silent_exception_no_log | [REPEAT 2026-05-24] 8 sub-signals same pattern. Disabled globally but per-ticker active for some. Still unfixed. | Same.

`portfolio/signals/vwap_zscore_mr.py:90` | P2 | vote_double_count | [REPEAT 2026-05-24] `vol_vote = vwap_z_vote if vwap_z_vote != "HOLD" else "HOLD"`. Same direction counted twice. Still unfixed. | Decouple votes.

`portfolio/signals/autotune_adaptive_cycle.py:157-159` | P2 | vote_double_count | [REPEAT 2026-05-24] `corr_strength_vote = trend_vote`. Lockstep vote. Still unfixed. | Same.

`portfolio/signals/network_momentum.py:356-357` | P2 | vote_double_count | [REPEAT 2026-05-24] `corr_regime = net_div`. Still unfixed. | Same.

`portfolio/signals/hurst_regime.py:284,302,333` | P2 | vote_double_count | [REPEAT 2026-05-24] `hurst_regime = trend_vote` AND `trend_direction = trend_vote`. Still unfixed. | Same.

`portfolio/signals/gold_real_yield_paradox.py:285-291` | P2 | date_alignment | [REPEAT 2026-05-24] FRED daily yields aligned to intraday gold returns by length only. Still unfixed. | Reindex on common date.

`portfolio/signals/sentiment_extremity_gate.py:1-17,159-167` | P2 | applicability_drift | [REPEAT 2026-05-24] Crypto-only F&G applied to all tickers. Still unfixed. | Add ticker gate.

`portfolio/signals/amihud_illiquidity_regime.py:76-112` | P2 | direction_polarity | [REPEAT 2026-05-24] Sub-1 maps thin market → SELL, thick market → BUY — inverts Amihud's expected sign. Still unfixed. | Use as confidence-gate only.

`portfolio/signals/ttm_squeeze.py:69-70` | P2 | lookahead_window_internal | [REPEAT 2026-05-24] `ffill().bfill()` inside regression window — later bar values fill earlier NaNs. Still unfixed. | Drop bfill.

`portfolio/signals/intraday_seasonality.py:110-129` | P2 | misleading_sub_signal | [REPEAT 2026-05-24] `_hour_alpha_vote` returns "BUY" for high-multiplier hours regardless of direction. Still unfixed. | Rename to ACTIVE/GATED.

## Top patterns this pass

1. **Direction polarity bugs in cross-asset signals.** Three modules (`cross_asset_tsmom`, `treasury_risk_rotation`, `dxy_cross_asset`) wrestle with the same problem: a single signal needs INVERTED direction for safe-haven vs risk-on assets. `treasury_risk_rotation` does it but leaves sub_signals inconsistent. `cross_asset_tsmom` forgets entirely and returns the same direction for all tickers. `dxy_cross_asset` correctly maps direction. A shared `_invert_for_asset_class(action, ticker)` helper would prevent recurrence.

2. **Timeframe-blind signals applied across the 7-TF grid.** `hash_ribbons`, `calendar_seasonal`, `cross_asset_tsmom` all encode parameters for a specific timeframe (daily) but get dispatched on 15m/1h/3h/12h/1d alike. The engine should pass timeframe in context and modules should reject when timeframe doesn't match design.

3. **Vote inflation by reusing the same signal in multiple sub-signals.** Affects `news_event` (5 of 6 sub-signals are keyword severity classifications), `metals_vrp` (vrp_momentum gates on vrp_z), `orderbook_flow` (vpin direction = trade_flow direction), `treasury_risk_rotation` (zscore + slope direction both from same series). Same root cause as the prior audit's vote-double-count cluster.

4. **Silent exception is still endemic** despite the 2026-05-24 audit. `momentum.py`, `oscillators.py`, `vwap_zscore_mr.py`, `autotune_adaptive_cycle.py`, `news_event.py` all unfixed. None of the prior P1 silent-exception findings have been resolved. The `claude_fundamental.py:197` earnings-fetch except logs at debug level so it's at least visible.

5. **Look-ahead is still mostly absent** in active signals. The few flagged (ttm_squeeze, absorption_ratio, trend_slope_momentum) are window-internal bfills, not cross-bar leaks. No new lookahead bugs found this pass.

## Out-of-scope

- Built-in signals (RSI, BB, Fear&Greed, Ministral-8B, Qwen3-8B, ML, MACD, EMA, Volume, Funding, Sentiment, Forecast wrappers, Claude Fundamental inline) live inside `portfolio/signal_engine.py`, not `portfolio/signals/*.py`. Reviewing them needs separate scope.
- LLM scaffolds (`meta_trader.py`, `finance_llama.py`, `cryptotrader_lm.py`) intentionally return HOLD with `feature_unavailable=True` — not reviewable until inference is wired.
- `claude_fundamental.py` was lightly scanned; the three-tier cascade and earnings fetch look correct. Deeper review of the JSON-extraction parser and confidence calibration deferred.

## Status of 2026-05-24 findings

- [REPEAT] vwap_zscore_mr.py:124 silent except — UNFIXED
- [REPEAT] autotune_adaptive_cycle.py:187 silent except — UNFIXED
- [REPEAT] credit_spread.py:285 relative load_json — UNFIXED
- [REPEAT] gold_real_yield_paradox.py:265 relative load_json — UNFIXED
- [REPEAT] btc_etf_flow.py wrong signature + silent except — UNFIXED (still dead)
- [REPEAT] vwap_zscore_mr / autotune / network_momentum / hurst_regime vote double-count — ALL UNFIXED
- [REPEAT] momentum.py / oscillators.py silent except cluster — UNFIXED
- [REPEAT] sentiment_extremity_gate.py crypto-only drift — UNFIXED
- [REPEAT] amihud_illiquidity_regime bias misdirection — UNFIXED
- [REPEAT] intraday_seasonality misleading sub-signal — UNFIXED
- [REPEAT] ttm_squeeze bfill window internal — UNFIXED
- [REPEAT] gold_real_yield_paradox date alignment — UNFIXED
- [REPEAT] absorption_ratio_regime / mahalanobis_turbulence window-internal bfill — UNFIXED
- [REPEAT] trend_slope_momentum bfill — UNFIXED
- [REPEAT] intraday_seasonality / gold_overnight_bias silent tz extract — UNFIXED

Zero of the 14 prior-audit findings have been remediated. Recommend a dedicated cleanup PR before promoting any pending-validation signal to active.
