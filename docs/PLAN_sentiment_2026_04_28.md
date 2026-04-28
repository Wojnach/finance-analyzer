# Plan — Fix sentiment relevance + aggregation collapse

**Branch:** `fix/sentiment-relevance-and-aggregation`
**Date:** 2026-04-28
**Author:** sentiment investigation session

## Problem

Investigation 2026-04-28 found shadow LLM accuracy is technically computable but
the active sentiment signal itself has collapsed. Five concrete failure modes:

1. **No relevance filter on input.** Random "Bitcoin price action this week"
   wire headlines flow into `_run_model()` alongside genuine catalysts. Models
   then label them `neutral` (correctly — they ARE neutral) and that neutral
   verdict drowns out the few decisive headlines.

2. **`_aggregate_sentiments()` averages probabilities with no decisiveness
   threshold** (`portfolio/sentiment.py:582`). 6 mostly-neutral headlines
   (0.30/0.30/0.40 each) + 2 decisive positive (0.10/0.10/0.80) average to
   ~0.25/0.25/0.50 → still neutral. A 0.34/0.33/0.33 split labels neutral
   even though the call is essentially random.

3. **CryptoBERT is misapplied.** Trained on crypto-twitter slang ("wagmi",
   "ngmi", emoji); we feed it CryptoCompare press wire headlines. Result:
   99.1% neutral output across 2,817 samples. The model is barely engaging.

4. **Score-averaging vs. label-majority.** A few decisive headlines are
   diluted by many tepid-neutral ones. The shadow-evaluation pipeline uses
   per-row `correct` (label-based), so the production aggregator should match.

5. **Critical errors (`data/critical_errors.jsonl`).** 6 unresolved
   `accuracy_degradation` entries from 2026-04-28 cite `sentiment` dropping
   75.3% → ~42% as the lead offender. This work directly addresses the
   regression.

## Why neutral peers drag down decisive models — explained

The aggregation in `portfolio/sentiment.py:555-583`:

```python
pos_sum = sum(s["scores"]["positive"] * w for s, w in zip(sentiments, weights))
...
overall = max(avg, key=avg.get)
```

Every headline contributes its full probability vector to a weighted average.
If 8 of 10 headlines score `(0.30 pos, 0.30 neg, 0.40 neu)` and 2 scream
`(0.05 pos, 0.05 neg, 0.90 pos)`, the *average* still leans neutral because
the unweighted mass dominates. This was correct under the original assumption
that headline weights would dampen noise (1.0 default → low weight). But
with no relevance filter, every headline gets `weight=1.0`, so the noise
isn't dampened — it's drowning the signal.

This is also why "decisive" models (Trading-Hero-LLM, 63/29/8) get drowned
when paired with permaneutral peers (FinGPT 93%, FinBERT 87.5%): in the
shadow-evaluation pipeline, each model is aggregated with the SAME
score-averaging function. A model that votes positive on 30% of headlines
loses to one that votes neutral on 90% of headlines — every time.

## Fixes (the user's requested 4)

### Fix 1: Headline relevance filter (NEW)

Add `is_relevant_headline(title, ticker)` to `portfolio/news_keywords.py`.

A headline is "relevant" if ANY of:
- Has matched keywords from `score_headline()` (weight > 1.0), OR
- Mentions the ticker symbol or known synonyms (BTC/Bitcoin, ETH/Ethereum,
  XAU/gold, XAG/silver, or the literal stock ticker), OR
- Comes from a credible source AND has title length > 25 chars (filters out
  bare price-tickers like "Bitcoin: $67,123").

Add `_filter_relevant_headlines(articles, ticker)` to `portfolio/sentiment.py`
that drops irrelevant headlines BEFORE `_run_model()` runs. Fallback: if the
filter would remove >90% of headlines, keep the most-recent N as a safety net
so we don't go silent on slow news days.

Apply to both primary and shadow inference paths so they see the same input.

### Fix 2: Decisiveness threshold in `_aggregate_sentiments`

After computing `avg`, check the margin between top label and second:
```python
top = max(avg, key=avg.get)
sorted_scores = sorted(avg.values(), reverse=True)
margin = sorted_scores[0] - sorted_scores[1]
if margin < 0.05 and top != "neutral":
    return "neutral", avg  # not decisive enough
```

A 0.34/0.33/0.33 split now correctly returns neutral (margin 0.01 < 0.05).
A 0.50/0.30/0.20 split still returns the winner (margin 0.20 > 0.05).

### Fix 3: Label-majority aggregation mode

Add a second aggregation mode `_aggregate_sentiments_majority()`:
- For each headline, classify its own label decisively (top score must beat
  second by ≥ 0.10, else "neutral")
- Weighted majority vote over per-headline labels
- Pick winner unless tied within `MIN_LABEL_MARGIN` of weight units

Wire it as the new default in `_aggregate_sentiments()` via a `mode=` arg
defaulting to `"majority"` (old behavior available as `mode="average"`).
Keep both because Fix 2's threshold-augmented average is still useful for
dissecting per-model verdicts.

### Fix 4: Replace CryptoBERT as crypto primary

Change `get_sentiment()` for crypto tickers:
- Primary (votes): `Trading-Hero-LLM`
- Shadow (logged only): `CryptoBERT` (kept so we still measure if it ever
  recovers, plus continuity for the existing 30d accuracy baseline)

Rationale: 99.1% neutral output on 2,817 samples is conclusive — it's not
the right model for press-wire input. Trading-Hero on crypto headlines at
least produces variance, even if biased. The 47% directional accuracy gate
will continue to act as a circuit breaker if Trading-Hero's bias makes it
worse.

## What could break

- **Sentiment vote distribution shift.** Trading-Hero replaces CryptoBERT
  for BTC/ETH; expect more BUY votes initially given Trading-Hero's
  permabull tendency. The 47% accuracy gate will catch sustained
  underperformance.
- **Fewer headlines processed per cycle.** Relevance filter may drop ~50% of
  wire headlines; verify the fallback (keep most-recent N if all dropped) is
  hit when news is genuinely slow.
- **Threshold may surface "neutral" more often initially**, then less often
  over time as the relevance filter feeds more decisive inputs upstream. Net
  effect: fewer noisy votes, more decisive ones when they appear.
- **Backward compat:** `_aggregate_sentiments()` keeps the same return
  signature `(label, avg_dict)`. Existing callers (FinGPT, FinBERT, primary)
  are unaffected at the call site.

## Execution batches

### Batch 1: Tests first (TDD)
- `tests/test_sentiment_relevance_filter.py` — new file, ~10 cases:
  - keyword-matched headline kept
  - ticker-symbol-mention kept
  - credible-source long headline kept
  - bare price-ticker dropped
  - all-irrelevant fallback keeps most-recent N
  - empty input returns empty
- `tests/test_portfolio.py` — add `TestSentimentDecisiveness`:
  - margin > 0.05 picks winner
  - margin < 0.05 returns neutral even if top != neutral
  - exact tie returns neutral
- `tests/test_portfolio.py` — add `TestSentimentMajority`:
  - 3 positive + 1 neutral → positive
  - 3 neutral + 1 weak-positive → neutral (per-headline threshold)
  - 2 positive + 2 negative → neutral (weight tie)
  - keyword-weighted majority wins over count
- Run targeted tests, watch them fail.

### Batch 2: Implementation
- `portfolio/news_keywords.py`: add `is_relevant_headline(title, ticker)` +
  per-ticker synonyms map
- `portfolio/sentiment.py`:
  - `_filter_relevant_headlines(articles, ticker, *, fallback_n=3)` helper
  - `_aggregate_sentiments(...)` adds `mode=` param and decisiveness threshold
  - `_aggregate_sentiments_majority()` new label-majority implementation
  - `get_sentiment()`:
    - swap crypto primary CryptoBERT → Trading-Hero
    - run CryptoBERT as new shadow for crypto, stash via existing pending-A/B
      buffer, picked up in `flush_ab_log()`
    - apply `_filter_relevant_headlines()` before `_run_model` calls
- Run targeted tests, watch them pass.

### Batch 3: Wire CryptoBERT shadow + flush
- `portfolio/sentiment.py:flush_ab_log()` — extend to include CryptoBERT
  shadow if present in stash
- Verify schema of `data/sentiment_ab_log.jsonl` stays additive (new shadow
  entry alongside fingpt + finbert; downstream backfill is by-model so it
  picks up the new entries automatically).

### Batch 4: Verify + ship
- Run full test suite parallel
- Update `docs/CHANGELOG.md`
- Commit + merge + push
- Append `resolution` entry to `data/critical_errors.jsonl` for the 6
  accuracy_degradation entries (sentiment regression directly addressed)
- Restart loops

## Files affected (estimate)

- `portfolio/news_keywords.py` (extended)
- `portfolio/sentiment.py` (modified)
- `tests/test_sentiment_relevance_filter.py` (new)
- `tests/test_portfolio.py` (extended)
- `docs/CHANGELOG.md` (entry added)

Total: ~5 files. Within batch limits; can do as 2 commit batches: "tests"
then "impl + wiring + docs".
