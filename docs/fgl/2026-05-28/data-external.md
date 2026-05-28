# Data-External Subsystem Adversarial Review

Date: 2026-05-28
Scope: 14 new files (4328 LOC) feeding loop's signal engine

## Summary

P0 findings: 4 | P1 findings: 9 | P2 findings: 3 | P3 findings: 2

## Key Findings

portfolio/sentiment.py:854: P0 - Empty headline list returns unknown sentiment. News API outage doesn't disable voter; dilutes consensus. Fix: return None.

portfolio/alpha_vantage.py:280: P0 - Budget counter race condition. Incremented inside lock, reset outside. Two threads racing cause budget to exceed limit.

portfolio/onchain_data.py:29: P0 - _coerce_epoch() returns 0.0 on parse failure. Crashes if cached ts is ISO string. Forced API call every restart.

portfolio/metals_cross_assets.py:56: P0 - price_source.fetch_klines() exception swallowed. Returns empty DataFrame. No staleness indication.

portfolio/fx_rates.py:44: P1 - Rate outside [7,15] logged but not cached. P&L silently uses fallback, 10-15% wrong for 2h+.

portfolio/crypto_macro_data.py:224: P1 - _load_ratio_history() loads entire JSONL. File >30M causes MemoryError.

portfolio/sentiment.py:131: P1 - _fetch_crypto_headlines() returns [] silently on all failures.

portfolio/price_source.py:214: P1 - silent yfinance fallback on Binance failure. Caller gets 10-min stale data without indication.

portfolio/alpha_vantage.py:232: P1 - Budget check returns 0 if exhausted. Failed refresh doesn't roll back quota.

data/crypto_data.py:63: P1 - _WARNED guard never resets. After first error, subsequent errors silent forever.

## Totals

P0: 4 | P1: 9 | P2: 3 | P3: 2
