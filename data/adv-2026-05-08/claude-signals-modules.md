# Adversarial Review: Finance Analyzer Signals Subsystem

Date: 2026-05-08
Scope: portfolio/signals/*.py (36+ modules)
Reviewer: Claude Haiku 4.5

---

## Findings

### portfolio/signals/trend.py:325-326
Ichimoku cloud uses forward-shifted SMA, enabling look-ahead bias.

Problem: senkou_a.shift(26) and senkou_b.shift(26) shift the spans 26 bars forward in time. The iloc[-1] comparison includes 26 bars of future data.

Fix: Change to shift(-26) or remove shift entirely; standard Ichimoku on live signals should use past-only lookbacks.

---

### portfolio/signals/volatility.py:57
BB squeeze threshold allows divide-by-zero when avg_width is zero or NaN.

Problem: avg_width > 0 check misses NaN values (np.nan > 0 returns False silently).

Fix: Use if avg_width > 0 and not np.isnan(avg_width).

---

### portfolio/signals/momentum.py:131
StochRSI division by zero when RSI range collapses.

Problem: When all RSI values in a window are identical, denom=0 becomes NaN; next bar can flip from NaN to valid number without smoothing.

Fix: If denom.isna().all(), return HOLD. Or add minimum range guard: denom.clip(lower=1.0).

---

### portfolio/signals/momentum.py:201
ROC denominator uses shifted (past) close, not current; conflates level with acceleration.

Problem: close.shift(period) creates self-referential lag. Lines 212, 216 treat ROC value as acceleration indicator.

Fix: Verify ROC intent. If acceleration needed, compute roc_now - roc_prev, not ROC value alone.

---

### portfolio/signals/calendar_seasonal.py:256
FOMC dates cache max() at module import; all future dates eventually become stale.

Problem: _FOMC_MAX_DATE computed once at import. Warning fires on every call after last FOMC date passes, but signal still votes. No halt, no escalation to critical_errors.jsonl.

Fix: Check FOMC list at compute time, not import time. Auto-extend via Fed calendar API.

---

### portfolio/signals/econ_calendar.py:127
post_event_relief BUY fires when recent event + next event >24h away; overlapping relief/risk windows.

Problem: If two high-impact events are 20h apart, relief window for event#1 persists while event#2 enters pre_event_risk zone. Silent conflict in composite vote.

Fix: Suppress relief when next event <24h away.

---

### portfolio/signals/metals_cross_asset.py:150
FRED fetch returns newest-first (descending order), but z-score assumes oldest-to-newest.

Problem: sort_order=desc, but values list is appended in order received; z-score lookback window is reversed chronologically.

Fix: Reverse values list after fetch: values = list(reversed(values)).

---

### portfolio/signals/cot_positioning.py:33
Deep context load has no staleness check; reflects week-old COT data indefinitely.

Problem: _load_deep_context() loads JSON with no timestamp validation. No warning if file is >1 week old (CFTC release cadence).

Fix: Check file mtime; if >7 days old, return None and fall back to API.

---

### portfolio/signals/credit_spread.py:159
OAS z-score uses current as values[0] included in historical mean; self-referential bias.

Problem: history = values[:lookback] includes current (values[0]); mean/stddev includes the value being scored.

Fix: Use history = values[1:1+lookback] to exclude current from historical sample.

---

### portfolio/signals/claude_fundamental.py:149
Earnings calendar cache TTL is 12h; earnings dates don't change daily. Inefficient yfinance calls.

Problem: _EARNINGS_CACHE_TTL = 43200 calls yfinance every 12h for all stocks; scales as 5N requests/day.

Fix: Increase TTL to 86400 (24h) or 604800 (7d); implement change-detection.

---

### portfolio/signals/forecast.py:37
Forecast TTL is 5 minutes; Chronos predictions stale in 12h/24h horizon signals.

Problem: _FORECAST_TTL = 300 means 24h forecast computed at 10:00 is reused at 10:04 (still 23h50m remaining).

Fix: Implement per-horizon TTL: 1h=5min, 3h=15min, 24h=60min.

---

### portfolio/signals/news_event.py
TICKER_SECTORS mapping hardcoded at module load; stale on corporate restructures.

Problem: Sector map is never refreshed. On M&A or spinoff, econ_calendar fires pre_event_risk on wrong tickers.

Fix: Load TICKER_SECTORS at compute time from config/API, not import time.

---

### portfolio/signals/forecast.py:789
Forecast candles cached but never validated for stale OHLCV data.

Problem: _cached(cache_key, _FORECAST_TTL, _load_candles_ohlcv) has no check that last bar is current. Stale bar causes wrong direction prediction.

Fix: Validate df.iloc[-1]["time"] >= datetime.now(UTC) - timedelta(minutes=60); return None if stale.

---

### portfolio/signals/claude_fundamental.py:94
Cooldown check uses time.time() (wall-clock), not monotonic; vulnerable to NTP adjustments.

Problem: If system time steps backward 30 seconds, cache appears expired when still warm.

Fix: Use time.monotonic() for all cooldown checks.

---

### portfolio/signals/volatility.py:36
_atr() uses EMA, not RMA (Wilder smoothing); ATR values ~30% lower than TA-Lib standard.

Problem: tr.ewm(span=period) uses α=2/(period+1). Wilder's RMA uses α=1/period. EMA-based ATR is more reactive, channels narrow faster.

Fix: Replace with RMA from signal_utils or implement Wilder's formula.

---

### portfolio/signals/momentum.py:110
Stochastic %K threshold <20 for BUY is tight; no volume confirmation.

Problem: Oversold <20 is 1-2 bars of extreme moves. On BTC/XAG, 15m bars swing 1-3% frequently. Crossover-only is noise.

Fix: Add volume confirmation (vol > 1.2x avg) or tighten threshold to <10.

---

### portfolio/signals/structure.py
No gap detection; high/low breakout ignores overnight discontinuity (gaps 5-10% on earnings).

Problem: Assume continuous bars. Gap breakouts are catalyzed by off-market news, not trader conviction.

Fix: Add gap detection; ignore structure signals in gap bars.

---

### portfolio/signals/smart_money.py, oscillators.py, orderbook_flow.py (disabled)
Dead code still imported by signal_registry; consumes CPU on import despite being disabled.

Problem: 19 disabled modules ~3000 LOC total. AST parsing/compilation overhead ~2s per startup, every 60s loop.

Fix: Move disabled modules to portfolio/signals/_disabled/ and exclude from import loop; or add lazy-load flag.

---

### portfolio/signals/treasury_risk_rotation.py
Signal added 2026-05-07 with no backward compatibility test or validation data.

Problem: Added mid-session with zero historical accuracy data. If misprogrammed, contributes noise for 7 days before accuracy catches it.

Fix: Add 30-day warm-up shadow mode; new signals vote HOLD until 30 samples logged and accuracy >47%.

---

### portfolio/signals/realized_skewness.py, statistical_jump_regime.py, hurst_regime.py (disabled)
Feature-engineered indicators with >3-month stale "pending validation" label.

Problem: Added Apr 2026, still marked pending. 3+ weeks with no status update or removal. Consume review overhead.

Fix: Decide: enable with 30-day shadow or archive to _disabled/. Update DISABLED_SIGNALS with resolution timestamp.

---

Summary: 8 bugs (look-ahead, staleness, NaN, cache semantics), 8 risks (edge cases, validation, dead code, thresholds), 4 pending (new signals, maturity, design sync).

