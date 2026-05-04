# Codex Review — 1-signals-core

## Summary

The forecast accuracy path is not working end-to-end: the writer and reader disagree on schema, and the backfill can delete historical predictions once it hits its cap. In addition, the new SQLite-first loader can silently serve stale history after any DB write hiccup, which undermines the accuracy-based logic across the subsystem.

Full review comments:

- [P1] Emit forecast votes in the schema the accuracy tracker reads — Q:\fa-review\portfolio\forecast_signal.py:365-372
  `compute_forecast_accuracy()` only scores votes from `entry["sub_signals"]` / `entry["raw_sub_signals"]`, but the rows written here only contain nested `chronos` and `prophet` payloads. As a result, every backfilled row in `forecast_predictions.jsonl` contributes zero scored votes, so forecast accuracy reports and `accuracy_degradation`'s forecast checks stay empty even though predictions are being logged.

- [P1] Preserve the unprocessed tail when `max_entries` stops backfill — Q:\fa-review\portfolio\forecast_accuracy.py:341-348
  If a run backfills at least `max_entries` horizon outcomes (500 by default), the `break` exits before the remaining predictions have been copied into `modified_entries`, and `_write_predictions()` then rewrites the JSONL with only that processed prefix. On any backlog large enough to hit the cap, this deletes the rest of `forecast_predictions.jsonl` while trying to catch up.

- [P1] Fall back to JSONL when the SQLite copy is incomplete — Q:\fa-review\portfolio\accuracy_stats.py:150-153
  This makes SQLite authoritative as soon as it has any rows, but the write side is explicitly best-effort: `outcome_tracker.log_signal_snapshot()` appends to JSONL before attempting `SignalDB.insert_snapshot()`, and `backfill_outcomes()` swallows `_db.update_outcome()` failures. After one transient SQLite write error, every caller here will keep reading the stale DB copy and permanently ignore the newer JSONL snapshot/outcome, so accuracy gating and degradation checks can run on incomplete history.
The forecast accuracy path is not working end-to-end: the writer and reader disagree on schema, and the backfill can delete historical predictions once it hits its cap. In addition, the new SQLite-first loader can silently serve stale history after any DB write hiccup, which undermines the accuracy-based logic across the subsystem.

## Full review comments

- [P1] Emit forecast votes in the schema the accuracy tracker reads — Q:\fa-review\portfolio\forecast_signal.py:365-372
  `compute_forecast_accuracy()` only scores votes from `entry["sub_signals"]` / `entry["raw_sub_signals"]`, but the rows written here only contain nested `chronos` and `prophet` payloads. As a result, every backfilled row in `forecast_predictions.jsonl` contributes zero scored votes, so forecast accuracy reports and `accuracy_degradation`'s forecast checks stay empty even though predictions are being logged.

- [P1] Preserve the unprocessed tail when `max_entries` stops backfill — Q:\fa-review\portfolio\forecast_accuracy.py:341-348
  If a run backfills at least `max_entries` horizon outcomes (500 by default), the `break` exits before the remaining predictions have been copied into `modified_entries`, and `_write_predictions()` then rewrites the JSONL with only that processed prefix. On any backlog large enough to hit the cap, this deletes the rest of `forecast_predictions.jsonl` while trying to catch up.

- [P1] Fall back to JSONL when the SQLite copy is incomplete — Q:\fa-review\portfolio\accuracy_stats.py:150-153
  This makes SQLite authoritative as soon as it has any rows, but the write side is explicitly best-effort: `outcome_tracker.log_signal_snapshot()` appends to JSONL before attempting `SignalDB.insert_snapshot()`, and `backfill_outcomes()` swallows `_db.update_outcome()` failures. After one transient SQLite write error, every caller here will keep reading the stale DB copy and permanently ignore the newer JSONL snapshot/outcome, so accuracy gating and degradation checks can run on incomplete history.
