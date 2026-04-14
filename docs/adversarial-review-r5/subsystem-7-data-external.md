# Subsystem 7: Data External — Round 5 Findings

## CRITICAL (P1)

**DE-R5-1** — _load_onchain_cache still has ISO timestamp crash bug on no-token fallback path.
`onchain_data.py:95-107`. A-DE-5 fix applied to get_onchain_data but not _load_onchain_cache.
Fix: Apply _coerce_epoch in _load_onchain_cache.

**DE-R5-2** — Gold/BTC ratio reads from agent_summary_compact.json (stale disk file, up to 1h old).
`crypto_macro_data.py:202-259`. Not live prices.
Fix: Fetch live spot prices from price_source.fetch_klines.

**DE-R5-3** — get_open_interest returns dict missing oi_usdt key despite docstring advertising it.
`futures_data.py:33-53`. Downstream callers reading oi_usdt get None.
Fix: Add oi_usdt calculation or remove from docstring.

## HIGH (P2)

**DE-R5-4** — On-chain fallback uses 24h stale tolerance with DEBUG-level logging (invisible).
**DE-R5-5** — fear_greed.py streak file uses relative Path("data/...") — CWD-dependent.
**DE-R5-6** — Earnings calendar Alpha Vantage calls bypass 25-call/day budget tracker.

## MEDIUM (P3)

**DE-R5-7** — social_sentiment.py Reddit errors go to stdout (print) instead of logger.
**DE-R5-8** — crypto_scheduler.py Telegram report has no staleness guard on prices.
