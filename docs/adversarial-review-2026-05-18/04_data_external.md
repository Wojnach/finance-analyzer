# Data-External Review — subagent result (caveman:cavecrew-reviewer)

Totals: 3 P1 (🔴), 16 P2 (🟡)

## P1 / 🔴

1. **alpha_vantage.py:231 & 281** — `_check_budget()` read (line 231) not under lock; `_daily_budget_used` increment (281) under lock. Race window between read+increment → two threads pass budget check, double-count quota.
2. **crypto_macro_data.py:165** — Max pain calc inverted. Code finds minimum; comment says "maximize pain for buyers". Output flipped → signal direction reversed.
3. **econ_dates.py:155** — Event time hardcoded to 14:00 UTC. FOMC actual 19:00 UTC (2pm ET); CPI 13:30 UTC (8:30 ET). All macro events off by ±5h.

## P2 / 🟡

- data_collector.py:330 — TimeoutError silently drops results; stuck futures break pool state.
- metals_orderbook.py:65 — Empty bids/asks list passes through; no `len()` guard before [0].
- metals_orderbook.py:82 — Bid/ask sign convention undocumented; trade sign for `isBuyerMaker=True` may be inverted.
- microstructure_state.py:216 — `load_persisted_state()` no `_buffer_lock`; restart + metrics fetch race → split-brain.
- microstructure_state.py:226 — Age threshold 120s > 60s cycle; two missed cycles = "fresh".
- crypto_macro_data.py:119 — Deribit response parsed without `isinstance(list)` guard.
- fear_greed.py:109 — `data_list[0]` unguarded after `.get("data")`; empty/None list crashes.
- fx_rates.py:33 — 15-min cache vs 2h stale threshold; 105-min drift no alert.
- fx_rates.py:70 — First-fetch failure logs ERROR but doesn't `_fx_alert_telegram()`.
- metals_cross_assets.py:89 — `_pct_change` returns NaN; downstream signals don't check, NaN propagates into voting.
- metals_cross_assets.py:147 — Common-index intersection silently narrows on holiday divergence.
- price_source.py:235 — Fallback ERROR doesn't distinguish timeout vs bad ticker; ~10 false alarms/day.
- macro_context.py:46 — `close.iloc[-5]==0 or NaN` (weekend sparse data) → inf/NaN unguarded.
- market_health.py:82 — `fetch_klines` empty df on weekend; no `len(df)>=10` guard.
- bert_sentiment.py:131 — `glob.glob()` unsorted; `snapshots[0]` may be stale checkpoint, not latest.
- sentiment_shadow_backfill.py:127 — Loads entire JSONL into in-memory set; 365K entries → memory leak on repeated invocation.
