# Cross-critique — signals-core

## Codex findings Claude missed

| Codex finding | Why Claude missed it |
|---|---|
| `forecast_signal.py:365-372` — forecast votes written in `chronos`/`prophet` nested payload, but `compute_forecast_accuracy()` only scores `entry["sub_signals"]`/`raw_sub_signals`. **Every backfilled row scores zero votes.** | Claude reviewed `accuracy_stats.py` and `signal_db.py` for accuracy math but did not cross-check the writer schema against the reader schema. This is a writer/reader contract bug — the kind that hides because each side looks correct in isolation. Codex caught it by tracing the data flow end-to-end. |
| `forecast_accuracy.py:341-348` — backfill `break` on `max_entries` cap exits before copying remaining unprocessed entries to `modified_entries`, then `_write_predictions()` rewrites with only the prefix. **Deletes the rest of `forecast_predictions.jsonl` on any backlog at cap.** | Claude focused on the accuracy gate logic (signal_engine.py) and atomic I/O (signal_decay_alert.py). Did not read `forecast_accuracy.py` line by line. This is a destructive bug; Claude missing it is a real gap. |
| `accuracy_stats.py:150-153` — SQLite-first reader makes DB authoritative whenever it has any rows, but writes are best-effort (errors swallowed in `outcome_tracker.log_signal_snapshot()` and `backfill_outcomes()`). One transient SQLite error → permanent silent staleness. | Claude saw the `signal_db.py` SQL accuracy math problem but missed the SQLite-vs-JSONL authority/fallback question. Codex caught the write-path silent failure that creates the staleness. |

## Claude findings Codex missed

| Claude finding | Why Codex missed it |
|---|---|
| `signal_decay_alert.py:35-36` — raw `open() + json.load()` on actively-written `accuracy_cache.json`. Torn read on Windows atomic-rename → silent `[]` return. | Codex didn't flag the I/O rule violation explicitly. May have considered it stylistic, but the silent `[]` is the core operational risk. |
| `signal_db.py:271, 302-303, 330-331, 370` — SQL accuracy methods omit the `_MIN_CHANGE_PCT` neutral-outcome filter that the Python path uses. Overstates accuracy for SQL consumers (notably `apply_confidence_penalties` Stage 6 for MSTR per-ticker at 47.8%). | Codex focused on the SQLite-vs-JSONL authority question (above) but didn't drill into the SQL methods' filtering math. Both bugs co-exist — the SQLite path can return stale data AND when it does have data the SQL math is wrong. |
| `signal_engine.py:3132-3139` — `btc_proxy` injected into MSTR votes outside `SIGNAL_NAMES`, bypasses accuracy gate while counting toward MIN_VOTERS=3. | Codex didn't review the cross-ticker consensus injection path. Project-specific structural concern that requires knowing the gate semantics. |
| `signal_engine.py:3475-3484` — utility boost can promote sub-47% signals above the accuracy gate (44% × 1.08 → 47.5%). | Codex focused on schema/I/O concerns; this is a pure semantics question about whether the boost should apply pre- or post-gate. |
| `accuracy_stats.py:1918-1929` — `write_ticker_accuracy_cache()` uses single shared `"time"` key for all horizons (vs. main cache's `"time_{horizon}"` pattern). | Codex noted the SQLite staleness but missed this independent JSON cache TTL bug. Both apply to ticker accuracy reads. |

## Disagreements

None directly. The reviews are operating at different levels:
- **Codex** focused on writer/reader schema contracts and the SQLite-first architectural decision.
- **Claude** focused on the accuracy gate semantics and the SQL math itself.

Both are real, and they compound — Codex's "SQLite serves stale rows" + Claude's "SQL math is wrong even when fresh" means the per-ticker confidence penalty system is double-broken.

## What both missed (likely)

- **Per-horizon accuracy tracking for `btc_proxy`** — neither reviewer asked whether `btc_proxy` should accumulate its own outcome record. If it should, both writer (`signal_log` snapshot) and `outcome_tracker` need updating.
- **Race between `_write_predictions()` and a concurrent `compute_forecast_accuracy()`** — Codex flagged the data-loss bug but didn't ask whether the rewrite is atomic.
- **`meta_learner.py` purge gap of 2 days against the actual 3h/12h/1d horizons** — 2 days is fine for ≤1d but neither reviewer checked whether longer horizons (3d, 5d outcomes) leak into the train set.

## Reconciled verdict

**P0 (must fix before next deploy):**
1. (Codex) Forecast schema mismatch — entire forecast accuracy path is silently dead. **High impact: forecast accuracy reports are blank.**
2. (Codex) Forecast backfill data loss at `max_entries` cap. **Destructive on large backlogs.**
3. (Claude) `signal_db.py` SQL accuracy methods skip neutral filter. **Per-ticker confidence penalty silently bypassed for sub-52% tickers.**
4. (Claude) `signal_decay_alert.py` raw open + decay alerts silenced on Windows.

**P1 (should fix):**
5. (Codex) SQLite-first reader stale on write failure.
6. (Claude) `btc_proxy` bypasses accuracy gate.
7. (Claude) Utility boost crosses gate threshold.
8. (Claude) Ticker accuracy cache shared time key.

**P2:**
9. (Claude) Duplicated `acc_horizon` definition.
10. (Claude) Unlocked persistence reads.
