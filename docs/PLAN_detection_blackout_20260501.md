# Plan — Detection blackout 2026-04-21 → 2026-04-28 root cause + residual gap (2026-05-01)

## Context

Earlier session caught a 7-day blackout: sentiment regressed 75.3% → 42% across
2026-04-21 → 2026-04-28 but the first `accuracy_degradation` row hit
`critical_errors.jsonl` only at 2026-04-28T01:26 UTC. Manual fixes shipped on
2026-04-25 (`381283da`) and 2026-04-27 (`ce173f46`) prove a human noticed the
degradation; the system did not. Layer 2 was making suboptimal decisions for a
week with the user's "data-driven, not speculative" rule operating on stale
ground truth.

The 2026-04-28 batch (commits `ad7d9500`, `753febab`, `9e13060e`, `5ba9eae2`)
addressed:

1. **Writer bulletproofing** — verify `accuracy_snapshots.jsonl` actually grew
   before persisting `last_snapshot_date_utc`.
2. **Backfill recovery** — `scripts/backfill_accuracy_snapshots.py` rebuilt the
   missing 04-22 → 04-27 entries from the signal log.
3. **Binomial-SE significance gate** — block false positives from small samples.
4. **Silent-failure journal rate-limit** — keep `critical_errors.jsonl` from
   flooding when the writer recurs in failure.

This investigation confirms those four are correctly applied; tests in
`test_accuracy_degradation*.py` (42/42) pin the new behavior. **What remains is
the residual structural gap that allowed the blackout to last 7 days instead
of 1.**

## Root cause (high confidence)

The detector is gated by three sequential conditions that all must pass:

1. **Throttle** — once per ~hour.
2. **Econ blackout** — ±24h around FOMC/CPI/NFP.
3. **Baseline lookup** — `_find_snapshot_near(target=now-7d, max_delta_hours=36)`.
4. **Baseline age** — selected snapshot must be ≥6.0d old.

During 04-22 → 04-27 the snapshot writer silently no-op'd. The pre-existing
JSONL had only Apr 19, 20, 21 (plus the Feb 20 stub). On 04-22, the lookup
target is 04-15; the closest available snapshot is 04-19 at 96h delta — well
outside the 36h tolerance. Same on 04-23 (72h), 04-24 (48h), 04-25 (24h
*passes*), 04-26 (0h), 04-27 (0h). 04-25/26/27 had a viable baseline but no
write of "today's recent" snapshot, and the throttle's cached
`last_full_check_violations` was `[]` from the no-baseline branches on prior
days. Because the writer wasn't running, the post-write throttle invalidation
(`last_full_check_time = 0`) never fired, so the detector kept replaying `[]`
for hours after it would have actually had a baseline.

**Compounded by:** `BASELINE_MAX_DELTA_HOURS = 36.0` is too tight for the daily
writer's failure modes. A *single skipped day* can push the detector past
tolerance for the next 24h. A two-day gap pushes it past tolerance for ≥48h
even with a healthy resume.

## Evidence

* `data/accuracy_snapshots.jsonl` — every snapshot from 04-22 through 04-27
  carries `_backfilled: true` with note "Recreated 2026-04-28 from signal_log
  replay; per_ticker lifetime + forecast omitted." None have a microsecond
  timestamp; all are 06:00:00+00:00 (synthetic) — a fingerprint of the backfill
  script.
* `data/critical_errors.jsonl` — first `accuracy_degradation` row at
  2026-04-28T01:26:23, listing `sentiment 75.3%→40.1%`. Zero alerts on 04-21
  through 04-27.
* The pre-04-28 `MIN_SAMPLES_HISTORICAL/CURRENT = 100` and absent SE gate would
  have fired loudly on 04-25 (drop=40.6pp, both N≥200 in the actual data) IF a
  baseline existed within 36h. It didn't.
* Commit `106daf02` (2026-04-22 13:59) fixed `_vote_correct` raising on
  `change_pct=None`. That bug killed `signal_accuracy('1d')` mid-horizon, so
  `save_full_accuracy_snapshot` raised, was caught by
  `maybe_save_daily_snapshot`'s try/except as `WARNING` and returned False —
  state correctly NOT advanced. After the 04-22 fix, the writer should have
  resumed but didn't (the `_backfilled` flags on 04-22 onward confirm zero
  organic writes that week). Commit `ad7d9500` traced this to a separate
  state-without-write desync, but the underlying mechanism for that desync is
  still not pinned in any commit message.

The user's hypothesis #2 (dedup bug deflated SE → drops looked like noise) does
NOT apply to the blackout itself: the binomial-SE gate did not exist until
2026-04-28T13:14 UTC (commit `9e13060e`), AFTER the first alert had already
fired.

## Residual gap and proposed fix

**The 2026-04-28 batch handles the case where `maybe_save_daily_snapshot`
runs and silently no-ops. It does not handle:**

(a) **Loop crash / not running** — if `PF-DataLoop` is wedged or
   `maybe_save_daily_snapshot` is never called, no row appears in
   `critical_errors.jsonl`, and `check_critical_errors.py` (the STARTUP CHECK)
   stays silent because it only surfaces what's already journaled.

(b) **Tight baseline tolerance** — even with a healthy writer, a single
   day-skip pushes `_find_snapshot_near` outside `BASELINE_MAX_DELTA_HOURS=36`
   for the next 24-48h.

(c) **Visibility into "the detector is currently dark"** — there's no signal
   to operators that the gate is currently disabled by missing baseline. From
   the outside it looks identical to "no degradation."

**Proposed: add a `snapshot_freshness` invariant to `loop_contract.py` that
fires WARNING when `accuracy_snapshots.jsonl` mtime is >36h, escalating to
CRITICAL at 48h.** This is independent of `maybe_save_daily_snapshot`'s
self-check; it surfaces the desync via a separate code path that any cycle
can verify.

## Non-goals

* Re-implementing the 04-28 silent-failure detection. Already shipped.
* Loosening `BASELINE_MAX_DELTA_HOURS` — that's a real safety property; a
  3-day-old baseline is unsuitable for 7d delta detection. The fix is
  upstream (writer reliability), not the lookup tolerance.
* Backfill changes. The script is idempotent and works.
* New gates inside the detector. The gate cascade is correct; what's missing
  is independent visibility.

## Changes

### Batch 1 — Test (TDD)

`tests/test_loop_contract_snapshot_freshness.py` (new):

* `test_fresh_snapshot_no_violation` — `accuracy_snapshots.jsonl` modified <36h
  ago: no violation produced.
* `test_stale_36h_warning` — file mtime 36h+ stale: WARNING violation.
* `test_stale_48h_critical` — file mtime 48h+ stale: CRITICAL violation.
* `test_missing_jsonl_warning` — file does not exist (fresh deployment): WARNING
  with message "no snapshot file present" (not CRITICAL — could be day-1).
* `test_invariant_does_not_raise_on_io_error` — patch `Path.stat` to raise; the
  invariant returns `[]` and logs at WARNING (must never take down the
  contract framework).

### Batch 2 — Implementation

`portfolio/loop_contract.py`:

* Add `SNAPSHOT_FRESHNESS_INVARIANT = "snapshot_freshness"`.
* Add `check_snapshot_freshness_safe()` modeled after
  `check_signal_accuracy_degradation_safe`: try/except wraps a private
  `_check_snapshot_freshness()` that compares `ACCURACY_SNAPSHOTS_FILE`'s mtime
  to `now`.
* Wire into `verify_contract` after the existing degradation call (line 636).

`portfolio/loop_contract.py` (cont.):

* Mtime threshold constants: `SNAPSHOT_STALE_WARN_HOURS = 36`,
  `SNAPSHOT_STALE_CRITICAL_HOURS = 48`. Justification: the daily writer runs at
  06:00 UTC; 36h tolerance accommodates one missed day plus the morning before
  the next snapshot window. 48h means we've gone two cycles, enough to make the
  baseline lookup fail.

### Batch 3 — Wire to `CRITICAL_ERROR_DISPATCH_INVARIANTS`

Add `"snapshot_freshness"` to the existing frozenset at line 67 so that when
the invariant escalates to CRITICAL via `ViolationTracker`, the auto-fix-agent
dispatcher sees the row and engages.

## Test plan

* Run only `tests/test_loop_contract*.py` and
  `tests/test_accuracy_degradation*.py` (focus per /fgl protocol).
* Expect: existing degradation tests stay 42/42; new freshness tests
  ~5/5 added.

## Out of scope

* Investigating why `maybe_save_daily_snapshot` had the original
  state-without-write desync that started the blackout. The fingerprint
  evidence (every 04-22 to 04-27 entry is `_backfilled`) confirms the writer
  literally did not append, but the only non-`raise` no-op path through the
  code is `save_full_accuracy_snapshot` → no-op → state advance, which the
  04-28 size-check now defends against. If the underlying mechanism needs
  further investigation, that's a separate ticket.
