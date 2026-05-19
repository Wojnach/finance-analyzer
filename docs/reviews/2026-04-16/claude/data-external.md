# Adversarial Review — data-external (Claude-independent)

## P0

### 1. earnings_calendar.py bypasses Alpha Vantage 25/day budget
**File:** `portfolio/earnings_calendar.py:48-60`
Comment admits: "earnings calls bypass alpha_vantage.py's `_daily_budget_used` counter because there is no public increment function exported". 12 tickers × daily refresh = 12 silent AV calls outside the visible budget. After 2 days, quota exhausted silently.
**Fix:** export `_budget_track()` from `alpha_vantage.py`; route all AV calls through a shared counter; Telegram alert on <2 calls remaining.

## P1

### 2. fomc_dates.py / econ_dates.py hardcoded through 2027 — 22-month time bomb
**Files:** `portfolio/fomc_dates.py:24-57`, `portfolio/econ_dates.py:38-103`
When 2028 arrives, FOMC/econ proximity gates return empty → signals stop gating binary-event risk silently.
**Fix:** fetch from Fed API or calendar_config.json with annual roll-forward; runtime assert current_date is within coverage.

### 3. funding_rate.py hardcoded thresholds not regime-scaled
**File:** `portfolio/funding_rate.py:24-32`
0.03% / -0.01% thresholds: in a prolonged bull market rates trail positive for months → SELL fires daily (not contrarian, just directional noise). In panic, BUY never fires.
**Fix:** gate by VIX regime; use 30d percentile (top 25% → SELL, bottom 10% → BUY).

### 4. feature_normalizer.py — online vs offline mismatch for ML features
**File:** `portfolio/feature_normalizer.py:43-71`
100-window running stats; never persisted → cold start on restart → raw values for first 20 samples. Meanwhile `ml_trainer.py` computes features offline with full history (no min_periods). Distribution drift between train and inference.
**Fix:** persist stats to JSON on update; reload at startup; OR unify: offline training matches online normalizer's min_periods=20.

### 5. meta_learner.py feature order not persisted with model
**File:** `portfolio/meta_learner.py:369-427`
Features built from dict → DataFrame. Python 3.7+ preserves insertion order → stable only if `SIGNAL_NAMES` is identical between train and predict. A merge/refactor reordering SIGNAL_NAMES silently remaps features (feature 0 → feature 2 in model input). Catastrophic silent failure.
**Fix:** persist `feature_order` alongside model in joblib dump; reindex DataFrame at predict; assert columns match.

### 6. ministral_signal / qwen3_signal — output not validated against {BUY,SELL,HOLD}
**Files:** `portfolio/ministral_signal.py:50-89`, `portfolio/qwen3_signal.py:60-100`
LLM outputs "action" field; no check it's in set. An adversarial/hallucinated action like "RECOMMEND BUY XYZ" propagates into voting. Prompt injection from news headlines or ticker is a real vector.
**Fix:** `if decision not in {"BUY","SELL","HOLD"}: return HOLD with log.error`.

## P2

### 7. bert_sentiment model load failures silently fall back to neutral
**File:** `portfolio/bert_sentiment.py:100-130`
Hardcoded cache paths; if missing, exception caught silently in sentiment.py wrapper; default neutral=1.0. News-driven trades proceed on un-analyzed sentiment.
**Fix:** log WARNING with model name and reason on fallback.

### 8. forecast accuracy gate at meta-learner is orthogonal to sub-signal gate
**File:** `portfolio/meta_learner.py:385-394`
Meta-learner has its own AUC gate; `forecast_signal.py` has per-subsignal gate. Both active but independent — partial gate confuses signal flow.
**Fix:** single gating decision; aggregator reads both and chooses one.

### 9. macro_context.py serves stale DXY on weekend/holiday without warning
**File:** `portfolio/macro_context.py:65-149`
3-min TTL; on weekends returns Friday's close as "current"; downstream signals see 0% 1h change → treat as flat market.
**Fix:** staleness guard: `if now - last_ts > 2h: return {"warning": "market_closed"}`.

### 10. news_keywords.py multi-word match fragile to whitespace
**File:** `portfolio/news_keywords.py:79-83`
`re.escape(kw)` escapes the literal space. "rate hike" won't match "rate  hike" (double space) or "rate\nhike".
**Fix:** replace spaces with `\s+` before escape.

### 11. onchain_data.py silent fallback on corrupted timestamp
**File:** `portfolio/onchain_data.py:29-57`
`_coerce_epoch()` returns 0 on parse failure → cache miss → API call → if API fails, silent signal loss.
**Fix:** log WARNING on coercion fallback.

### 12. data_collector.py counts empty Binance response as failure
**File:** `portfolio/data_collector.py:89-93`
Empty response = circuit breaker failure → can trip breaker for legitimate "no trades this interval" on illiquid pair.
**Fix:** treat empty as success (zero records), don't increment failure count.

## P3

### 13. indicators.py RSI cold-start NaN bars
**File:** `portfolio/indicators.py:42-50`
First 14 bars are NaN after ffill — fragile if any signal uses first N bars naively.
**Fix:** document and enforce "do not use first min_rows bars"; runtime validation.

### 14. fx_rates.py hardcoded SEK fallback 10.85
**File:** `portfolio/fx_rates.py:50-55`
Long-term drift; by 2027 could be 15% stale → portfolio valuations wrong.
**Fix:** config-driven with update date; error on >30d stale.

### 15. earnings_calendar.py timezone ambiguity
**File:** `portfolio/earnings_calendar.py:70-87`
Alpha Vantage `reportedDate` is ET calendar date; compared against `today = datetime.now(UTC).date()` → off-by-one possible.
**Fix:** convert to ET before comparison.

## Looked OK
- **circuit_breaker.py** — state machine, probe-once logic sound.
- **sentiment.py** — fallback chain defensive.
- **linear_factor.py** — Ridge + z-score mathematically sound.
- **social_sentiment.py** — Reddit API simple and unproblematic.
- **forecast_signal.py** — Chronos/Prophet fallback, GPU gate integration solid.

## Reviewer confidence
0.85 — did not run full execution path through voting; did not test LLM parsing with actual model output. Gaps: data_collector retry logic + circuit breaker integration test.
