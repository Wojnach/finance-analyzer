# Agent Review: data-external

## P1 Findings
1. **fear_greed KeyError crash** on empty API response (fear_greed.py:100)
2. **futures_data missing oi_usdt** — docstring promises it, code doesn't return it

## P2 Findings
1. **fetch_json silently drops kwargs** — retries=0 ignored for BGeometrics (http_retry.py:76, onchain_data.py:123)
2. **crypto_macro_data reads stale agent_summary** — violates live prices rule (crypto_macro_data.py:207-221)
3. **onchain_data _load_onchain_cache** ISO timestamp not coerced (onchain_data.py:95-107)
4. **sentiment.py wrong Python path** for subprocess fallback (sentiment.py:34)
5. **Alpha Vantage budget not persisted** — resets on restart (alpha_vantage.py:31,157-168)
6. **crypto_macro raw open()** for history files (crypto_macro_data.py:275,397)

## P3 Findings
1. fear_greed routes XAU/XAG to VIX proxy (semantically wrong)
2. History files grow without bound, O(n) reads
3. fx_rates.py hardcoded fallback 10.85 is stale (~10.4 current)
