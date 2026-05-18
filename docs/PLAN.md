# Plan — grid_fisher silent-rejection loop fix

Date: 2026-05-18
Branch: `fix/grid-fisher-hours-gate`
Worktree: `.worktrees/grid-fisher-hours-gate`

## Problem

On 2026-05-18 18:30–18:40 UTC (= 20:30–20:40 CET) `grid_fisher` entered a
tight loop on OIL-USD (`BULL_OLJAB_X5_AVA_2`, ob 2367797):

1. `place_buy_order` returns success + `order_id` (e.g. 883727550).
2. ~55 s later (next reconcile tick) the `order_id` is absent from
   `get_open_orders`. `reconcile_against_live` sees the live position
   delta is 0 → classifies as `external_cancel_buy`, frees the tier.
3. Next tick re-arms the same tier. Loops forever.

Root cause: FNSE warrant orderbook for OLJAB is closed after ~17:25 CET.
Verified via `/_api/market-guide/certificate/2367797`:
- `quote.timeOfLast = 2026-05-18T06:15:01Z` (~12 h before the loop fired)
- `quote.totalVolumeTraded = 0`

Avanza accepts the submission and returns an `order_id` but the order
never lives in the book — auto-cancelled immediately. `grid_fisher` has
no after-hours gate per instrument and no rapid-cancel back-off, so it
spins until EOD sweep or signal flip.

## Goals

1. Stop the same failure mode for any warrant that is silently
   auto-cancelled (after-hours, halted, knocked out, instrument-specific
   reject). Two independent gates so neither is single-point.
2. Zero false skips during normal hours — quote-staleness threshold
   wide enough to absorb thin-trade lulls.
3. Cheap: no extra Avanza calls per tick beyond what's already cached.

## Design

### Gate A — quote-staleness pre-placement gate

Before placing a ladder for an instrument, check the cached `quote.timeOfLast`
for the orderbook. Skip placement if `now - timeOfLast >
GRID_QUOTE_STALENESS_THRESHOLD_S`.

* Threshold: 1800 s (30 min). FNSE certs trade actively every few
  seconds during open hours, so 30 min is comfortably above the noise
  floor and well under a half-day after-hours gap.
* Implementation: `place_buy_ladder` already calls `get_quote` as the
  bid fallback. Pull `timeOfLast` from the same response (ms epoch),
  compare to wall clock. To avoid double-calling when the caller
  supplied a bid, add a tiny per-instrument quote cache (60 s TTL).
* Decision log: `skip_quote_stale` with `ob_id`, `ticker`,
  `time_of_last_age_s`, `threshold_s`.

### Gate B — rapid-cancel back-off (post-reconcile)

When `reconcile_against_live` classifies a tier as `external_cancel_buy`,
record the cancel timestamp + age relative to that tier's `placed_ts`.
If the age is below `GRID_RAPID_CANCEL_THRESHOLD_S` (default 120 s),
increment a per-instrument counter `rapid_cancel_count`. When the count
reaches `GRID_RAPID_CANCEL_MAX_CONSECUTIVE` (default 2), set
`inst.cooldown_until` to `now + GRID_RAPID_CANCEL_COOLDOWN_S` (default
6 h — long enough to span the rest of the trading day).

* Reset counter on any successful fill or after the cooldown elapses.
* Decision log: `rapid_cancel_backoff` with `ob_id`, `ticker`, `count`,
  `cooldown_until`.

### Config (`portfolio/grid_fisher_config.py`)

```python
GRID_QUOTE_STALENESS_THRESHOLD_S = 1800        # 30 min
GRID_QUOTE_CACHE_SECS = 60                      # per-instrument cache
GRID_RAPID_CANCEL_THRESHOLD_S = 120              # cancel within 2 min = "rapid"
GRID_RAPID_CANCEL_MAX_CONSECUTIVE = 2            # 2 in a row → cooldown
GRID_RAPID_CANCEL_COOLDOWN_S = 6 * 3600          # 6 h
```

### State

Add to `InstrumentState`:

```python
rapid_cancel_count: int = 0
last_rapid_cancel_ts: Optional[str] = None
```

Reset by `roll_session_if_new_day`. `TierOrder` already carries
`placed_ts` so no schema bump on the tier dataclass.

State schema-version bumps 1 → 2; existing state files load cleanly
because `from_dict` already uses `.get(...)` for every field.

## Files to change

| File | Change |
|------|--------|
| `portfolio/grid_fisher_config.py` | Add 5 new constants. |
| `portfolio/grid_fisher.py` | (a) per-instrument quote cache + Gate A in `place_buy_ladder`; (b) rapid-cancel detection in `tick` after `reconcile_against_live`; (c) reset counters in `roll_session_if_new_day` / `record_fill`. New `InstrumentState` fields. |
| `tests/test_grid_fisher_hours_gate.py` | New: unit tests for Gate A + Gate B happy / edge / reset paths. |
| `docs/SESSION_PROGRESS.md` | Append the fix + restart instructions. |
| `docs/GRID_FISHER.md` | Document both gates. |

## Execution order

1. Branch + plan commit (this file).
2. Premortem agent → append to this file → commit.
3. Batch 1: config constants + state fields + helpers. Commit.
4. Batch 2: Gate A (quote-staleness). Commit.
5. Batch 3: Gate B (rapid-cancel back-off). Commit.
6. Batch 4: tests. Commit.
7. Adversarial review subagent on the branch. Fix P1/P2. Commit.
8. `pytest -n auto`. Commit any fixes. Merge → main → push.
9. Restart `PF-MetalsLoop` so new code loads.

## Risks (initial)

* **False skip in thin warrants**: a low-volume instrument may go 30 min
  between trades during real hours. Mitigation: 30 min is generous;
  config knob to tune; gate skips only placements, never cancels.
* **State migration**: `InstrumentState.from_dict` already uses `.get`
  for every field, so adding the two new fields is transparent for any
  state file written by v1.
* **Test fixture coupling**: existing grid_fisher tests build
  `InstrumentState` via constructor — new fields default to safe values,
  so no fixture churn expected. Will verify with full grid_fisher test
  module before commit.

## Premortem

(appended after fresh-agent run)
