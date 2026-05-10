# Claude critique of codex findings — signals-core

## Verdicts

- [P1] Silently discard unprocessed JSONL rows when max_entries reached — Q:\finance-analyzer\portfolio\forecast_accuracy.py:322-327
  Verdict: CONFIRMED
  Reason: Loop breaks when `updated >= max_entries` (line 322-323), but `modified_entries` only contains entries processed through the break point (line 320 appends inside loop). File is rewritten with only this prefix (line 327), permanently losing all remaining entries. Real data loss when predictions file has >500 entries with mature outcomes to backfill.

- [P1] Return stale SQLite when JSONL is ahead — Q:\finance-analyzer\portfolio\accuracy_stats.py:150-153
  Verdict: CONFIRMED
  Reason: Returns `db.load_entries()` as soon as snapshot_count() > 0 (line 150-153), ignoring that dual-write failures (caught by outcome_tracker.py at line 166) mean JSONL has more recent snapshots. Callers like `signal_accuracy()` will ignore fresher data and make decisions on stale accuracy tallies.

- [P2] Use daily close instead of intraday price for short-horizon outcomes — Q:\finance-analyzer\portfolio\outcome_tracker.py:269-276
  Verdict: CONFIRMED
  Reason: `t.history(start=..., end=...)` at line 269 uses default daily interval. Lines 272-276 extract the last close on or before `target_date`, which is EOD, not the actual price near `target_ts` (which could be 1h/3h/4h/12h before EOD). Poisons short-horizon outcome accuracy for stock tickers (MSTR).

## New findings (mine, not codex's)

None found.

## Summary
- Confirmed: 3
- Partial: 0
- False-positive: 0
- New from me: 0
