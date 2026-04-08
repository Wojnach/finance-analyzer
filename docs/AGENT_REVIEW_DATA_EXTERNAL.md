# Adversarial Review: data-external (Agent Findings)

Reviewer: Code-reviewer subagent (feature-dev:code-reviewer)
Date: 2026-04-08

---

## CRITICAL

### CD1. earnings_calendar.py: Wrong config key — Alpha Vantage earnings path NEVER executes [97% confidence]
**File**: `portfolio/earnings_calendar.py:42`

`config.get("alpha_vantage_key", "")` reads a top-level key that doesn't exist. Actual
config uses nested `config["alpha_vantage"]["api_key"]`. Result: `api_key` is always `""`,
early return fires, Alpha Vantage earnings path is **permanently broken** since day one.
System falls back to yfinance for earnings dates silently.

**Fix**: `api_key = config.get("alpha_vantage", {}).get("api_key", "")`

### CD2. fear_greed.py: Unguarded `body["data"][0]` crashes on malformed API response [92% confidence]
**File**: `portfolio/fear_greed.py:99`

No guard on `body["data"][0]`. If alternative.me returns valid JSON with missing/empty `data`
field, `KeyError` or `IndexError` propagates uncaught. Crashes F&G for ALL 20 tickers in
the current cycle.

**Fix**: `entries = body.get("data") or []; if not entries: return None`

### CD3. funding_rate.py: Unguarded dict access crashes on Binance error response [90% confidence]
**File**: `portfolio/funding_rate.py:23`

`float(data["lastFundingRate"])` and `float(data["markPrice"])` with no `.get()` guards.
Binance error responses (`{"code": -1121, "msg": "Invalid symbol."}`) cause `KeyError`.
Called per crypto ticker every loop cycle.

---

## HIGH

### HD1. econ_dates.py: April 2026 NFP date is Good Friday — market holiday [88% confidence]
**File**: `portfolio/econ_dates.py:61`

`NFP_DATES_2026` includes `date(2026, 4, 3)` — Good Friday, US markets closed. BLS releases
data but markets react on April 6 (Monday). Signal gates trades on wrong day and misses
the actual reaction date.

### HD2. onchain_data.py: 24h stale on-chain data served at DEBUG level only [85% confidence]
**File**: `portfolio/onchain_data.py:210`

When BGeometrics token is missing, accepts 24h stale MVRV/SOPR data. During BTC capitulation,
stale "accumulation" signal misleads. Violates CLAUDE.md rule "Live prices first."

### HD3. forecast_signal.py: Chronos-2 column access may KeyError on float columns [83% confidence]
**File**: `portfolio/forecast_signal.py:218`

`row["0.5"]` assumes string column names. Chronos-2 with `quantile_levels=[0.1, 0.5, 0.9]`
may return float-typed columns (`row[0.5]`). Silently disables all forecasts.

### HD4. crypto_macro_data.py: Gold/BTC ratio reads stale summary file [82% confidence]
**File**: `portfolio/crypto_macro_data.py:210`

Reads from `agent_summary_compact.json` (previous cycle's data) instead of live prices.
After crash+restart, ratio could be arbitrarily stale. Violates CLAUDE.md rule #3.

### HD5. futures_data.py: Missing `oi_usdt` field breaks interface contract [88% confidence]
**File**: `portfolio/futures_data.py:36,50`

Docstring promises `{oi, oi_usdt, symbol, time}` but returns only `{oi, symbol, time}`.
Any consumer using `result["oi_usdt"]` gets KeyError.

---

## MEDIUM

### MD1. alpha_vantage.py: Daily budget counter resets on process restart [85% confidence]
**File**: `portfolio/alpha_vantage.py:157-168`

`_daily_budget_used` is module-level, not persisted. Each crash+restart resets to 0.
Three restarts/day → 75 calls attempted against 25/day limit. Could get key blocked.

### MD2. sentiment.py: Primary model subprocess hangs 2min without TimeoutExpired catch [82% confidence]
**File**: `portfolio/sentiment.py:248-259`

`_run_model` uses `subprocess.run(timeout=120)` but doesn't catch `TimeoutExpired`.
120s hang per ticker × 20 tickers = potential 40-min stall.

### MD3. data_collector.py: Error entries propagate as valid timeframe data [80% confidence]
**File**: `portfolio/data_collector.py:312`

Failed fetches return `(label, {"error": str(e)})` instead of `None`. Downstream code
checking `result.get("indicators")` gets None (ok) but `result["action"]` → KeyError.

### MD4. crypto_macro_data.py: Max pain initialized to -1 — correct by accident [80% confidence]
**File**: `portfolio/crypto_macro_data.py:130-165`

`max_pain_value = -1` should be `float('inf')`. Works because `total_pain >= 0` always,
but fragile and misleading.

---

## LOW

### LD1. fx_rates.py: 2h stale threshold too long for live trading [80% confidence]
**File**: `portfolio/fx_rates.py:44-49`

USD/SEK can move 1-3% in 2h during volatility. Portfolio valuations silently wrong.

### LD2. data_collector.py: Alpaca/yfinance feed mismatch with extended hours [80% confidence]
**File**: `portfolio/data_collector.py:261-269`

yfinance fallback includes pre/post-market candles that Alpaca doesn't. Same ticker
produces different OHLCV depending on time of day.

---

## Cross-Critique: Claude Direct vs Data-External Agent

### Agent found that Claude missed:
1. **CD1**: earnings_calendar wrong config key — **completely broken since day one** — total miss
2. **CD2**: fear_greed unguarded access — missed (I noted API failures generically)
3. **CD3**: funding_rate unguarded access — missed
4. **HD1**: NFP date is Good Friday — complete miss (I noted hardcoded dates generically but
   didn't check specific dates against calendar)
5. **HD3**: Chronos-2 float column names — complete miss
6. **HD4**: Gold/BTC ratio reads stale file — complete miss
7. **HD5**: futures_data missing oi_usdt — complete miss
8. **MD2**: sentiment subprocess TimeoutExpired — missed
9. **MD3**: Error entries as valid data — missed
10. **MD4**: Max pain -1 initialization — missed

### Claude found that agent confirmed/expanded:
1. **H16/MD1**: Alpha Vantage daily budget — both found (I noted it, agent added restart detail)
2. **M13**: yfinance stale data — agent confirmed as LD2 (feed mismatch angle)

### Claude found that agent didn't cover:
1. **H15**: Cache key collision fragility — not covered
2. **M12**: Circuit breaker oscillation — not covered
3. **M25**: NewsAPI daily count not persisted — not covered (related to MD1 but different API)

### Net assessment:
The data-external agent found **3 CRITICAL + 5 HIGH + 4 MEDIUM + 2 LOW = 14 net-new issues**.
The earnings_calendar wrong config key (CD1) is the most impactful discovery — a feature
that has been silently broken since it was written, with the system never fetching earnings
dates from Alpha Vantage. The NFP Good Friday issue (HD1) is a concrete data error that
directly affects trading decisions.

Agent was significantly stronger here — specific API response validation bugs require
line-by-line reading that the broad review didn't do.
