# Adversarial Code Review: Signals-Modules Subsystem (Claude Reviewer)

## Executive Summary

**6 P1 findings, 9 P2 findings, 10 P3 findings.**

Key themes: silent data staleness (cached data without expiry), structural BUY/SELL bias (econ_calendar can never BUY, funding_rate BUY threshold 3x easier), unvalidated external API data causing crashes, and memory leaks in module-level caches.

---

## P1 Findings

### 1: `onchain_data.py:102` -- Cache timestamp type mismatch bypasses staleness check
ISO string timestamp causes `TypeError`, exception caught silently, BTC on-chain voter disabled.

### 2: `futures_flow.py:112-113` -- Unvalidated dict key crashes on malformed Binance API
No try/except wrapping sub-signal calls. One KeyError crashes entire futures_flow signal.

### 3: `forecast_signal.py:33` -- `_prophet_cache` is dead code; Prophet refits every call
Wastes 10-25s per cycle across 5 tickers.

### 4: `econ_calendar.py` -- Structural SELL-only bias
All 4 sub-indicators can only produce SELL or HOLD. Can never vote BUY.

### 5: `funding_rate.py:27-29` -- Asymmetric thresholds create structural BUY bias
BUY threshold 3x easier to trigger than SELL. Creates structural BUY bias.

### 6: `crypto_macro.py:228` -- `OPTIONS_TTL` referenced before module-level definition

## P2 Findings (9)

7: `volatility.py:159` -- `sqrt(365)` for all assets, wrong for stocks
8: `calendar_seasonal.py:209-228` -- Pre-holiday detection uses hardcoded dates
9: `sentiment.py:51-68` -- `TICKER_CATEGORIES` contains removed tickers
10: `cot_positioning.py:384` -- Single sub-signal can produce 100% confidence
11: `fear_greed.py:139-148` -- VIX mapping conflates stock-specific with generic
12: `news_event.py:256-260` -- "rate cut" substring match ignores negation
13: `orderbook_flow.py:175-176` -- `spread_health` always votes HOLD (dead sub-signal)
14: `mean_reversion.py:460-475` -- Recursive in-place detrending compounds errors
15: `bert_sentiment.py:79-112` -- Hardcoded label maps not verified against model config

## P3 Findings (10)

16-25: Off-by-one risks, batch truncation, circuit breaker tuning, overlapping seasonal signals, Aroon argmax uses first not last occurrence, etc.
