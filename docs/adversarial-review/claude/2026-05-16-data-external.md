# Data-External Adversarial Review — 2026-05-16

## P1: Economic Event Time Hardcoding Off by 30-90 Minutes

**File:** portfolio/econ_dates.py:155
**Bug:** Code hardcodes 14:00 UTC as release time for CPI/NFP (8:30 AM ET), but actual ET/UTC offset varies by DST.
**Why it matters:** 
- 8:30 AM EST (Nov-Mar) = 13:30 UTC (30 min off)
- 8:30 AM EDT (Mar-Nov) = 12:30 UTC (90 min off)
- Code's 14:00 UTC is never correct. Proximity gates fire at wrong times, potentially allowing trading within event windows.
**Fix:** Replace hardcoded 14:00 UTC with dynamic conversion: `datetime(evt["date"].year, evt["date"].month, evt["date"].day, 8, 30, tzinfo=timezone(timedelta(hours=-5)))` converted to UTC (accounting for DST), or use `datetime(evt["date"], time(13, 30), tzinfo=UTC)` as a compromise approximation for the post-DST half of the year.

---

## P2: Sentiment Confidence Can Return NaN to JSON Without Guard

**File:** portfolio/sentiment.py:897, 990, 1012
**Bug:** `round(avg[overall], 4)` propagates NaN if sentiment model outputs NaN confidence score (possible on model inference failure or unusual input).
**Why it matters:**
- Sentiment model (Trading-Hero-LLM, FinBERT, CryptoBERT) can output NaN on OOM, CUDA error, or malformed input
- `round(nan, 4)` returns `nan`, which becomes JSON `null` in some serializers
- Signal engine's accuracy backfill and alert logic treats null confidence differently than explicit 0.0, creating silent signal degradation
- User sees "confidence: null" in logs but no explicit error
**Fix:** Guard before rounding: `confidence = round(max(0.0, min(avg[overall], 1.0)), 4) if not math.isnan(avg[overall]) else 0.0`. Alternatively, add validation in `_aggregate_sentiments()` before returning avg dict.

---

## P3: Cache TTL Unit Mismatch in Futures Data

**File:** portfolio/futures_data.py:83
**Bug:** `get_open_interest_history()` passes `_OI_TTL` (300 seconds = 5 min) to `_cached()` for `futures_oi_hist_{ticker}_{period}`, but comment on line 21 says "5 min", and the call pattern implies cache should survive the full 30-entry history request cycle (typically 2-3 min on slow network).
**Why it matters:**
- History cache TTL is same as point-in-time OI cache (both 300s)
- If network is slow (>5s per request) or rate-limited, 30-entry history refetch burns 30 API calls instead of 1 call every 5 min
- Over 24h loop, this can exhaust Binance free tier (1200 req/min → ~1440 req/day)
**Fix:** Increase `_OI_TTL` to 600 (10 min) or separate `_HISTORY_TTL = 1800` (30 min) for history endpoints. Document the choice.

---

## P2: Missing NaN Guard in Sentiment Aggregation Average Calculation

**File:** portfolio/sentiment.py:699-701
**Bug:** If sentiment model returns NaN confidence on any headline (rare but possible on compute error), the weighted average calculation `pos_sum / total_w` produces NaN, which then silently propagates to all three keys in `avg` dict.
**Why it matters:**
- Subsequent code calls `max(avg, key=avg.get)` on line 709 — with NaN values, this returns a NaN-valued key instead of the dominant sentiment label
- NaN != NaN, so comparisons fail silently, signal engine treats result as corrupted
- No explicit error logged; voting just stops working for that ticker
**Fix:** After computing avg dict, validate and replace any NaN: `avg = {k: (v if not math.isnan(v) else 0.33) for k, v in avg.items()}` to restore neutral baseline on corrupt input.

---

## P2: Missing Fallback for All-Zero Dissemination Score

**File:** portfolio/news_keywords.py:248
**Bug:** If all three factors (source_factor, diversity_factor, clustering_factor) are 1.0, final score is exactly 1.0. But if any of the three intermediate calculations returns 0 (malformed articles, empty timestamp list), the product becomes 0, bypassing the `min(..., 3.0)` cap and producing `score = 0`.
**Why it matters:**
- Sentiment aggregation uses `dissemination_mult` as a weight multiplier in `_compute_weights()` (line 659)
- If `dissemination_mult = 0`, all headline weights become 0, `total_w = 0` on line 691, and function returns hardcoded neutral
- User gets "neutral" verdict with 0.33/0.33/0.33 split instead of actual model output
- This happens silently when headline list contains articles with unparseable timestamps
**Fix:** Return `max(score, 1.0)` instead of `min(score, 3.0)` to enforce minimum 1.0 multiplier. Or add explicit guard: `return max(min(round(score, 2), 3.0), 1.0)`.

---

## P1: Econ Calendar Only Covers 2026-2027, Silent Failure After Dec 2027

**File:** portfolio/econ_dates.py:23-103
**Bug:** ECON_EVENTS hardcoded to end 2027-12-03 (NFP). Calls to `next_event()` or `events_within_hours()` after that date return `None` (line 167) instead of continuing with newer dates.
**Why it matters:**
- System assumes it will never run past 2027. After Dec 2027, econ signals go silent
- No warning logged; proximity gate just stops firing
- User sees "no econ events found" and trades through major CPI/FOMC/NFP without proximity warnings
**Fix:** Extend calendar to 2028 (or make it dynamic by generating dates from a rule: "first Friday of month = NFP" + fixed CPI/FOMC cycle). Add a runtime check: `if ref_date.year > 2027: logger.warning("econ_dates calendar expired")`.

---

## P2: Alpha Vantage Daily Budget Not Decremented on Earnings Calls

**File:** portfolio/earnings_calendar.py:49-50
**Bug:** `_fetch_earnings_alpha_vantage()` calls Alpha Vantage EARNINGS endpoint (costs 1 AV credit) but never increments `alpha_vantage._daily_budget_used`. Earnings fetches burn budget silently.
**Why it matters:**
- `alpha_vantage.py:281` increments budget only inside `refresh_fundamentals_batch()`, not in standalone earnings calls
- If earnings gate checks 100 tickers, that's 100 AV calls not tracked
- Budget tracker shows "10/25 used" but actual usage is "110/25 used", suppressing other signals
**Fix:** Export a function from `alpha_vantage.py` (e.g., `decrement_daily_budget(n)`) and call it from `_fetch_earnings_alpha_vantage()` after successful fetch. Or move earnings caching into the `alpha_vantage.py` module itself.

---

## P3: No Error Handling for Invalid Sentiment Model Output Schema

**File:** portfolio/sentiment.py:883-892
**Bug:** `details` list unpacks `sent["confidence"]` and `sent["scores"]` without guard. If model returns `{"sentiment": "positive"}` (missing "confidence" or "scores"), code crashes with KeyError.
**Why it matters:**
- Rare but possible if model subprocess returns truncated output or old schema version
- Exception propagates, full `get_sentiment()` call fails
- Signal engine catches and logs error, ticker sentiment goes silent for 15 min (SENTIMENT_TTL)
**Fix:** Use `.get("confidence", 0.0)` and `.get("scores", {})` with safe defaults. Validate model output schema before building details list.

---

## SUMMARY
P1 = 2 (econ time hardcoding, calendar expiry)
P2 = 4 (sentiment NaN confidence, aggregation NaN, dissemination zero, AV budget leak)
P3 = 2 (futures cache TTL, sentiment schema validation)

