# PLAN: 5,476 missing-outcome rows in `data/llm_probability_log.jsonl`

## Status
- Branch: `fix/missing-backfill-outcomes-20260501`
- Worktree: `/mnt/q/finance-analyzer-missing-backfills`
- Date: 2026-05-01
- Author: Claude (deferred research)

## Background

Today's `local_llm_report_latest.json` flagged: "Backfill outcomes for `llm_probability_log.jsonl` (5,476 rows awaiting outcome join)." The dedup fix in `82258cbe` removed null-horizon duplicates from the outcomes file but did not address why outcomes are missing in the first place.

## Phase 1: Classification (read-only, complete)

Walked the entire 25,998-row probability log, joined against the 20,133-row outcome file. As of 2026-05-01 ~01:30 UTC, 5,863 rows are unmatched (slight increase from the 5,476 reported on 2026-04-30 02:08 as the loop kept logging).

Classifier: `scripts/_classify_missing_llm_outcomes.py` (one-off, NOT for commit).

| Bucket                     | Count | Recoverable? |
|----------------------------|------:|--------------|
| `too_recent` (1d horizon)  | 3,570 | No (legitimate skip — wait for horizon)  |
| `missing_target_only`      | 1,978 | Mostly yes (widen tolerance)            |
| `missing_entry_only`       | 126   | Mostly yes (widen tolerance)            |
| `both_missing`             | 96    | Mostly yes (widen tolerance, MSTR-heavy)|
| `should_have_been_written` | 81    | Yes (next backfill cron run)            |
| `bad_row_ticker`           | 12    | No (malformed — empty ticker)           |
| **Total unmatched**        | **5,863** | **2,221 prices recoverable, 3,582 wait, 12 bad, 6 NVDA stale, 81 trivially recoverable on next cron** |

## Phase 2: Root-cause analysis

### `too_recent` (3,570 rows, all 1d horizon)
Legitimate. These rows logged in the past 24h. Will be eligible after horizon elapses. The `min_age_hours` parameter on the backfill cron picks them up on subsequent runs. No fix needed.

### `missing_target_only` / `missing_entry_only` / `both_missing` (2,200 rows)
Root cause: `_lookup_price_at_time` in `portfolio/forecast_accuracy.py` has a hardcoded **2h tolerance**. Many target/entry times fall in snapshot gaps wider than 2h.

Snapshot gaps come from two sources:

1. **Loop downtime**: `cumulative_tracker.maybe_log_hourly_snapshot()` only fires when `portfolio/main.py --loop` is running. If the loop was down for 4-12h (Layer 2 outages, restarts, code merges, machine sleep), there's a gap. Affects ALL tickers symmetrically.
2. **MSTR overnight gap**: MSTR is an Alpaca stock. After-hours, Alpaca returns no quote → no `prices_usd['MSTR']` → no MSTR field in the snapshot. The 13:30-20:00 EDT trading window translates to roughly 13:30-19:30 UTC during DST. Outside that, MSTR has no snapshot for ~12-13h overnight (and ~60h Fri close → Mon open).

Distance distribution per ticker (target_time → nearest snap):

| Ticker   | n   | median | p90    | max    |
|----------|----:|-------:|-------:|-------:|
| BTC-USD  | 413 | 4.94h  | 6.84h  | 7.34h  |
| ETH-USD  | 385 | 4.84h  | 6.84h  | 7.34h  |
| XAU-USD  | 369 | 4.77h  | 6.84h  | 7.34h  |
| XAG-USD  | 358 | 4.84h  | 6.99h  | 7.34h  |
| MSTR     | 543 | 16.48h | 22.82h | 24.32h |
| NVDA     | 6   | 373h   | 373h   | 373h   |

NVDA is special: Alpaca was decommissioned for NVDA on 2026-04-09, but a few stale LLM rows logged on 2026-04-21/23. Those are outside the snapshot window — unrecoverable.

Recovery rates by tolerance per ticker (max of entry_distance and target_distance):

| ticker  | total |  2h |  4h |  8h | 12h | 24h | 48h |
|---------|------:|----:|----:|----:|----:|----:|----:|
| MSTR    |   669 |   0 | 144 | 144 | 168 | 591 | 669 |
| BTC-USD |   434 |  21 | 158 | 434 | 434 | 434 | 434 |
| ETH-USD |   403 |  18 | 151 | 403 | 403 | 403 | 403 |
| XAU-USD |   393 |  24 | 156 | 393 | 393 | 393 | 393 |
| XAG-USD |   376 |  18 | 142 | 376 | 376 | 376 | 376 |
| NVDA    |     6 |   0 |   0 |   0 |   0 |   0 |   0 |

(These per-ticker totals sum to more than 2,200 because the table comes from the broader missing-snapshot bucket — including the `entry_only` and `target_only` cases, which I want to cover with the same fix.)

### `should_have_been_written` (81 rows)
All entry_time = 2026-04-29T23:07:10..23:27:20, all 1d horizon. Latest backfill ran 2026-04-30T23:05:02 — 5 minutes before the target_time elapsed. These will be picked up automatically on the next cron tick.

### `bad_row_ticker` (12 rows)
Same `ts` (2026-04-21T15:00:35) across 12 rows — looks like one bad batch where `ticker` was an empty string. Currently the backfill silently skips with `skipped_bad_row`. We should write a `critical_errors.jsonl` entry the FIRST TIME we see this so the loop dispatcher can notice empty-ticker logging.

### `bad_row_ticker` and NVDA — combined
6 NVDA rows + 12 empty-ticker rows = 18 rows that no fix can recover. Documented as known loss.

## Phase 3: Fix design

### Fix 1: Configurable lookup tolerance (the main fix)

Add `tolerance_hours` parameter to `_lookup_price_at_time` (default 2.0 to preserve existing semantics). The LLM outcome backfill passes a wider tolerance.

But we don't want to use a flat 24h everywhere — that would over-attribute moves. Better: a **per-asset tolerance policy**.

Concrete policy after looking at the recovery matrix:

- **Crypto (24/7) and metals (24/7) tickers**: 8h tolerance. Rationale: these markets never close. A 4-8h gap is loop downtime; the price probably moved smoothly through it. p90 is 6.84h, max 7.34h — 8h gives 100% recovery on these tickers.
- **MSTR (US stock, regular hours only)**: 24h tolerance. Rationale: MSTR's overnight gap is structural (~12h on weekdays, ~60h Fri→Mon). Using the previous trading day's close to attribute a 1d return is the standard "next-trading-day close" approach. 24h gives 88% recovery; the remaining 78 rows are weekend-spanning where Fri close → Mon open is the right comparison.
- **Weekend-spanning MSTR**: 72h tolerance. Recovers the last ~12% (78 rows). The fundamental nature of a "1d ahead" prediction made Friday is "next-trading-day close", which is Monday — that's a Fri-Mon comparison even though the calendar gap is ~60h.
- **NVDA and other decommissioned tickers**: no tolerance helps; price snapshots stop at decommission. Document as known loss.

Implementation: I'll add a two-arg helper `_pick_tolerance(ticker)` in `llm_outcome_backfill.py` that returns 8h for crypto/metals, 72h for stocks. Actually simpler: put the tolerance map in the backfill module since `forecast_accuracy.py` is used by other consumers that should keep 2h.

### Fix 2: Critical-error journal entry on bad rows

When `skipped_bad_row` increments AND the row has empty ticker (not a JSON parse error), append to `data/critical_errors.jsonl`:
```json
{"ts": "...", "level": "warn", "category": "llm_probability_log_bad_row",
 "caller": "portfolio.llm_outcome_backfill", "message": "row missing ticker",
 "context": {"ts": "<row_ts>", "signal": "<signal>"}}
```
But: if there are 12 historical bad rows already, we'd spam 12 entries every backfill run. Better: only journal when we encounter a bad-row pattern that's NOT yet recorded as `resolves_ts` in the journal. For now, simpler approach: write ONE entry per backfill run summarizing "N bad rows seen this run". That avoids spam.

### Fix 3: Run the backfill once after merge

Trivial: just run `.venv/Scripts/python.exe scripts/backfill_llm_outcomes.py` after deploying the fix. This will sweep up:
- The 81 "should_have_been_written" rows.
- The newly-recoverable rows under the wider tolerance.

Expected outcome row count after fix: 20,135 (current) + ~2,200 (newly recovered) + 81 (post-cron stragglers) ≈ 22,400. The 3,570 too-recent rows still remain pending until their horizons elapse, and they are not a bug.

## Phase 4: TDD plan

### Test 1: `tests/test_forecast_accuracy.py`
Add a test that verifies `tolerance_hours=8.0` returns the price 5h before target.
Add a test that verifies `tolerance_hours=24.0` returns the price 16h before target (MSTR overnight case).
Add a test that verifies default `tolerance_hours=2.0` still returns None for 3h-stale (the existing `test_outside_tolerance` semantics).

### Test 2: `tests/test_llm_outcome_backfill.py`
Add a test asserting BTC-USD with snapshots 7h apart still backfills.
Add a test asserting MSTR with snapshots 16h apart still backfills.
Add a test asserting MSTR with snapshots 80h apart (Fri→Mon) DOES NOT backfill (out of policy).
Add a test asserting empty-ticker rows count toward `skipped_bad_row` (regression).

## Phase 5: Out of scope

- The structural problem of "MSTR has no after-hours snapshot" is real but not worth fixing — `1d ahead` for a stock SHOULD be measured at next-trading-day close. That's what the wider tolerance achieves.
- Improving the snapshot writer (e.g., extending hours, using yfinance after-hours) is bigger scope and risks polluting the snapshot file with low-quality after-hours quotes.
- Detecting and fixing the empty-ticker logging upstream — outside scope for this fix.

## Phase 6: Execution checklist

- [x] Worktree created
- [ ] Plan committed
- [ ] Tests written (4-6 new tests)
- [ ] Tests fail (TDD red phase)
- [ ] `forecast_accuracy._lookup_price_at_time` gains `tolerance_hours` parameter (default 2.0)
- [ ] `llm_outcome_backfill` uses per-asset tolerance map
- [ ] `sentiment_shadow_backfill` uses same per-asset tolerance map (parallel consumer)
- [ ] Tests pass
- [ ] Run full backfill in worktree to verify recovery count matches projection
- [ ] Adversarial review (codex) — TODO depending on time
- [ ] Commit
- [ ] DO NOT push or merge to main (per user instruction)

## Risk analysis

- **Risk**: Wider tolerance could attribute price moves from a different period to the wrong row. Concretely: an MSTR row entered Tuesday 22:00 UTC with target Wed 22:00 UTC, looked up via Wed 19:30 close, then comparison entry might be Tue 19:30 (not Tue 22:00). The actual return measured is Tue close → Wed close, not Tue 22:00 → Wed 22:00. **Verdict**: this is the correct semantics for a 1d-ahead prediction on a stock. The original 22:00 timestamp was an artifact of the LLM signal being computed during after-hours, not an indication that we should compare 22:00-to-22:00.
- **Risk**: Existing tests for `_lookup_price_at_time` may break. **Verdict**: keep default 2h, no breakage.
- **Risk**: Sentiment shadow backfill changes might surface different shadow accuracy numbers. **Verdict**: that's the goal — the shadow accuracy was missing many rows, which is exactly what the report flagged. The change makes more rows count, which is more correct.
- **Risk**: Mass-rerun could re-write some legitimately-old rows differently. **Verdict**: dedup keys are unchanged; old outcome rows stay the same. Only NEW outcome rows get written.

## Approval

- Per user instructions, this branch will NOT be pushed to origin or merged to main. Final state: a finding + branch + tests + commit on `fix/missing-backfill-outcomes-20260501`.
