# Plan — Accuracy P0/P1 Adversarial Follow-ups (2026-05-02)

## Context

Three deferred findings from
`docs/PLAN_adversarial_followups_20260502.md` carved off into a
focused worktree because all three live in the per-ticker
accuracy + sizing pipeline:

| ID | File | Severity |
|----|------|----------|
| P0-1 | `portfolio/ticker_accuracy.py:53-62` | Critical |
| P1-11 | `portfolio/kelly_sizing.py:139` | Important |
| SC-P1-3 | `portfolio/outcome_tracker.py:430-446` | Important |

## Findings

### P0-1: ticker_accuracy.py — no neutral filter

`accuracy_by_ticker_signal()` did `if (vote==BUY and chg>0) or
(vote==SELL and chg<0)` without invoking `_vote_correct()` or
respecting `_MIN_CHANGE_PCT` (0.05% noise floor). Every other
accuracy reducer in `accuracy_stats.py` (signal_accuracy,
per_ticker_accuracy, consensus_accuracy, accuracy_by_signal_ticker,
signal_accuracy_by_regime, signal_best_horizon_accuracy,
accuracy_by_ticker_signal in accuracy_stats.py) routes through
`_vote_correct`. ticker_accuracy was the sole outlier.

Per-ticker accuracy feeds `direction_probability()` (Mode B Telegram
notifications) and `kelly_sizing._get_ticker_signal_accuracy()`
(position sizing). Overstated accuracy → overstated win_prob →
oversized Kelly bets.

**Fix:** route through `accuracy_stats._vote_correct()`. 4-line
change in the hot loop.

**Tests added:** 6 in `tests/test_ticker_accuracy.py`
(`TestVoteCorrectNeutralFilter`):
- neutral positive outcomes skipped
- None change_pct skipped
- neutral negative outcomes skipped
- oracle equivalence with `_vote_correct`
- direction_probability propagates the filtered count
- signals dropped when every outcome is neutral/None

### P1-11: kelly_sizing.py — uses system-wide accuracy

`_get_ticker_signal_accuracy()` consulted only
`agent_summary["signal_accuracy_1d"]["signals"]` — system-wide per-signal
accuracy aggregated across every instrument. A signal that's 70%
accurate on XAG-USD but 30% on BTC-USD shows up as ~50% in that block,
under-sizing XAG-USD trades and over-sizing BTC-USD trades.

The data needed already exists upstream:
`accuracy_stats.accuracy_by_ticker_signal_cached()` returns
`{ticker: {signal: {accuracy, samples, ...}}}` and is consumed by
`signal_engine.py:3194` and `accuracy_degradation.py`. Just needed
plumbing into agent_summary and a lookup priority in kelly_sizing.

**Fix:** read from `agent_summary["per_ticker_signal_accuracy"]`
when present and >= 5 samples on this ticker; fall back to system-wide
otherwise; fall back to system-wide entirely when the per-ticker block
is missing (backwards compat with older agent_summary writers).
`recommended_size().source` advertises which path was used.

**Tests added:** 6 in `tests/test_kelly_sizing.py`
(`TestPerTickerAccuracyPlumbing`):
- per-ticker beats system-wide
- backwards compat: no per-ticker block → fall back
- ticker not in per-ticker block → fall back
- per-ticker low samples → fall back
- weighted average across multiple signals using per-ticker
- end-to-end source attribution in `recommended_size()`

### SC-P1-3: outcome_tracker.py — JSONL race

`backfill_outcomes()` did read-modify-rewrite of `SIGNAL_LOG`
(`data/signal_log.jsonl`) with no coordination against
`log_signal_snapshot()`'s `atomic_append_jsonl`. Multi-second
HTTP calls in the middle (`_fetch_historical_price` → Binance,
Alpaca, yfinance) gave Layer 1 ample window to append entries —
which were then clobbered by the final `os.replace`.

`atomic_append_jsonl` uses sidecar lockfile
`data/.signal_log.jsonl.lock`. backfill must respect the same lock.

**Fix:** acquire the sidecar lock in two windows:
1. Snapshot phase (lock held briefly) — record `snapshot_size`,
   parse tail entries.
2. Process phase (lock released) — slow HTTP calls run unblocked.
3. Rewrite phase (lock re-acquired) — re-stat file, copy any bytes
   past `snapshot_size` verbatim into tmp file after processed tail,
   `os.replace` under the lock.

Concurrent tail bytes are copied as raw bytes — never re-parsed
as JSON — preserving every byte the appender wrote.

`file_utils` is out of scope for this batch, so the lock helpers
(`_signal_log_lock_path`, `_hold_signal_log_lock` context manager) are
inlined in `outcome_tracker.py` using the same cross-platform pattern
(msvcrt on Windows, fcntl elsewhere). Refactoring to share with
`file_utils.atomic_append_jsonl` would be cleaner — punt to a future
batch when file_utils is in scope.

**Tests added:** 4 in `tests/test_outcome_tracker_backfill.py`
(`TestBackfillVsLiveAppendRace`):
- single concurrent append via `_fetch_historical_price` hook
- regression: no concurrent writer = unchanged behaviour
- 3 concurrent appends in the same processing window
- cross-thread concurrent append (realistic Layer 1 + backfill scenario)

## Worktree

`/mnt/q/finance-analyzer-acc-p0p1` on branch
`fix/acc-p0p1-followups-20260502`.

Branched off `40197785` (today's main).

## Commits

| # | Finding | Status | Commit |
|---|---------|--------|--------|
| 1 | P0-1 ticker_accuracy `_vote_correct` neutral filter | FIXED + 6 tests | `f7023395` |
| 2 | P1-11 kelly_sizing per-ticker accuracy plumbing | FIXED + 6 tests | `0b56bcb2` |
| 3 | SC-P1-3 outcome_tracker JSONL appender-lock coordination | FIXED + 4 tests | `48f50534` |

Total: 3 outstanding findings fixed, 16 new regression tests added
across 3 test files.

## Test status

- `test_ticker_accuracy.py`: 40/40 pass (34 existing + 6 new)
- `test_kelly_sizing.py`: 40/40 pass (34 existing + 6 new)
- `test_outcome_tracker_backfill.py`: 18/18 pass (14 existing + 4 new)
- Related suites unaffected: `test_kelly_metals.py` 26/26,
  `test_outcome_tracker_core.py` + `test_decision_outcome_tracker.py`
  + `test_per_ticker_accuracy_override.py` 113/113.

## Out of scope (deferred, intentionally left for parent or future batch)

- Refactoring the inlined lock helpers in outcome_tracker into a shared
  `file_utils.with_jsonl_lock()` context manager (file_utils not in this
  batch's scope).
- Plumbing `accuracy_by_ticker_signal_cached()` output into
  `agent_summary["per_ticker_signal_accuracy"]` upstream — the kelly_sizing
  fix uses it when present and falls back gracefully when not, so the
  upstream plumbing is a separate, non-regressing follow-up.
- Producing the `per_ticker_signal_accuracy` block in
  `portfolio/reporting.py` (writes agent_summary.json) — out of scope
  per the file allowlist.

## Branch handoff

- Branch: `fix/acc-p0p1-followups-20260502` (NOT pushed)
- Worktree: `/mnt/q/finance-analyzer-acc-p0p1`
- Tip SHA: see git log on the branch
- Per /fgl protocol: do NOT merge or push — return to parent.
