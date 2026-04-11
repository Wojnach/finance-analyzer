# Agent Review: data-external — Round 5 (2026-04-11)

**Agent**: feature-dev:code-reviewer
**Files reviewed**: 25 (data_collector, alpha_vantage, fear_greed, sentiment, bert_sentiment,
social_sentiment, onchain_data, futures_data, funding_rate, fx_rates, crypto_macro_data,
forecast_signal, ministral_signal, qwen3_signal, ml_signal, ml_trainer, indicators,
feature_normalizer, meta_learner, microstructure, microstructure_state, llama_server,
llm_batch, macro_context)
**Duration**: ~184s

---

## Findings (7 total: 1 P0, 6 P1)

### P0

**DE-R5-1** onchain_data.py:101-102 — _load_onchain_cache STILL has raw ISO timestamp bug
- A-DE-5 fix applied _coerce_epoch to `get_onchain_data()` but missed the fallback path
- `_load_onchain_cache()` does `time.time() - ts` where ts can be ISO string → TypeError
- Caught by bare except → returns None silently
- On-chain BTC voter is dead when no BGeometrics token configured
- Fix: Apply `_coerce_epoch()` in `_load_onchain_cache()` too

### P1

**DE-R5-2** microstructure_state.py:191-199 — persist_state() double-appends OFI every 5th cycle
- persist_state() calls get_microstructure_state() which calls record_ofi()
- Main loop already called get_microstructure_state() this cycle
- Every 5th cycle, OFI appears in history twice
- Z-score distribution tightens around duplicates → blinds signal to flow anomalies
- Fix: Use read-only state snapshot in persist_state(), don't re-call record_ofi()

**DE-R5-3** ml_signal.py:12-154 — FEATURES_PATH defined but never loaded at inference
- Feature order at inference depends on insertion order in compute_features()
- No validation against training-time feature names
- Likely contributor to ML signal's 28.2% accuracy (worse than coin flip)
- Fix: Load and reorder features at inference using saved feature names

**DE-R5-4** macro_context.py:38,226 — Missing yfinance MultiIndex guard
- Same bug class as A-DE-4 (fixed in fear_greed.py) but in _fetch_dxy() and _fetch_treasury()
- yfinance can return MultiIndex columns → h["Close"] returns DataFrame not Series
- float(series) raises TypeError, caught silently → DXY and yield curve dead
- Fix: Apply same MultiIndex flatten pattern from fear_greed.py

**DE-R5-5** forecast_signal.py:218 — Chronos-2 pred_df length not validated
- pred_df.iloc[h-1] with no length check → IndexError if forecast returns fewer rows
- Caught by outer try/except → silent None for forecast
- Fix: Add `if len(pred_df) < max_h: return None`

**DE-R5-6** funding_rate.py:23 — Unguarded data["lastFundingRate"] KeyError
- Binance premiumIndex can return error dict → KeyError kills funding rate signal (74.2% at 3h)
- Fix: Check isinstance(data, dict) and key presence

**DE-R5-7** feature_normalizer.py:37-39 — _buffers dict not thread-safe
- check-then-set pattern in _ensure_buffer() races under ThreadPoolExecutor
- Two threads can both create deques, one overwrites the other's values
- Fix: Use threading.Lock() or dict.setdefault()
