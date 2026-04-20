# Agent Review: data-external (2026-04-20)

## P1 Critical
1. **No OHLCV zero/negative price validation (STILL UNRESOLVED)** — NaN fixed (BUG-87) but zero/negative propagate. A single zero-price candle poisons all 33 signals.
2. **fx_rates._fx_cache has NO thread safety** — Plain dict mutated from 8-worker ThreadPoolExecutor. CPython GIL mitigates individual ops but check-then-act pattern is not atomic.

## P2 High
1. Fear & Greed data has no staleness marker (can be hours old without visibility)
2. onchain_data._load_onchain_cache doesn't use _coerce_epoch() (ISO timestamps cause TypeError)
3. Binance truncated response (3 candles instead of 100) passes validation if >= 26 rows
4. Crypto headline API failure produces silent "unknown" sentiment with confidence=0

## P3 Medium
1. _fx_alert_telegram uses unnecessary `global` keyword
2. onchain_data rate-limits with time.sleep(1) serially (6-11s blocking)
3. news_keywords.py imports datetime inside function (micro-perf)
4. Alpha Vantage API key could appear in DEBUG logs (currently safe)

## Prior Finding Status
- No OHLCV price validation (NaN/zero/negative): **PARTIAL** (NaN fixed, zero/negative NOT)
- fx_rates thread safety: **NOT FIXED**
- Empty Binance response: **FIXED** (BUG-100)
