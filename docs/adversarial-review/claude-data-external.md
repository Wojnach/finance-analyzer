# Adversarial Code Review: Data-External Subsystem (Claude Reviewer)

## Executive Summary

**3 P1 findings, 7 P2 findings, 6 P3 findings.**

The worst finding: OHLCV data from all three price sources (Binance, Alpaca, yfinance) undergoes zero validation for NaN, zero, or negative prices before feeding indicators and trading signals. A single corrupt candle generates false BUY/SELL signals.

---

## P1 Findings

### P1-1: No price validation on OHLCV data
**File:** `portfolio/data_collector.py:94-98` (Binance), `:145-158` (Alpaca), `:243-248` (yfinance)
NaN/zero/negative prices propagate directly to signal computation. Zero close causes division-by-zero in RSI, MACD, Bollinger.

### P1-2: FOMC dates in oil_precompute.py contradict canonical source
**File:** `portfolio/oil_precompute.py:924-927`
Multiple wrong dates (Mar 19 vs Mar 18, May 6 vs Apr 29, Sep 17 vs Sep 16). Oil risk-off gating fires on wrong days.

### P1-3: FX rate -- insane rate silently degrades to stale/hardcoded value
**File:** `portfolio/fx_rates.py:36-41`
All SEK portfolio valuations become wrong. No persistent alert. Stale threshold keeps ticking.

## P2 Findings (7)

P2-1: FRED API key leaked in retry log URLs -- `oil_precompute.py:624`
P2-2: Raw requests without retry/circuit breaker -- `oil_precompute.py:510,604`
P2-3: FX cache not thread-safe -- `fx_rates.py:16`
P2-4: Partial download silently overwrites complete data -- `data_refresh.py:40`
P2-5: No Reddit rate limiting, print() instead of logger -- `social_sentiment.py`
P2-6: AV daily budget bypass -- `earnings_calendar.py:48`
P2-7: Gold/BTC ratio reads stale precomputed data instead of live -- `crypto_macro_data.py:209`

## P3 Findings (6)

P3-1: Error dict masquerades as valid timeframe -- `data_collector.py:313`
P3-2: Wrong event release times (14:00 UTC for all) -- `econ_dates.py:155`
P3-3: Budget counter not persisted across restarts -- `alpha_vantage.py:31`
P3-4: Inconsistent column naming -- `data_refresh.py:52`
P3-5: Stale ticker mappings, no active ticker coverage -- `social_sentiment.py:17`
P3-6: Lockless file read during concurrent append -- `crypto_macro_data.py:275`
