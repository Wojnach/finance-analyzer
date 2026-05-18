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

(Generated 2026-05-18 by fresh general-purpose subagent.)

### N1. `timeOfLast` field shape assumption [silent-failure]
**Chain:** Premortem flagged that `get_quote` (calls
`/_api/market-guide/stock/{ob_id}/quote`) might not actually include
`timeOfLast`. **Verified empirically** before implementing — the field
is present on both certificate and stock orderbooks, in milliseconds.
Empirical sample at 2026-05-18 18:50 UTC for ob 2367797:
`{"buy":16.18,"last":14.46,"timeOfLast":1779084901283,"updated":...}`.
**Mitigation:** still defensive — if `timeOfLast` is missing or non-numeric
the gate treats it as "stale" (fail-safe — skip placement) and logs
`skip_quote_field_missing` once per instrument per process so we notice
if Avanza changes the shape.

### N2. 30-min threshold false-skip during real lulls [silent-failure / threshold]
**Chain:** A thin-trade lull on a real instrument could exceed 30 min
during open hours, surfacing as `skip_quote_stale` and starving the leg
of placements during a real dip.
**Mitigation:** threshold lives in config (`GRID_QUOTE_STALENESS_THRESHOLD_S`)
so it's a one-line tune. Add a counter on `/api/grid-fisher` payload
(`stale_skips_by_ob`) so we can audit false positives in dashboard.

### N3. Quote-cache thread-safety [concurrency]
**Chain:** A new mutable dict attribute on `GridFisher` is read/written
during `tick()` which runs on the metals_loop calling thread, while
`_safe_session_call` itself dispatches I/O to a worker thread. A dict
mutation racing iteration could `RuntimeError`.
**Mitigation:** wrap the cache in a `threading.Lock`. Cheap, idiomatic.

### N4. Rapid-cancel counter false-trips on direction-flip cancels [hidden coupling]
**Chain:** During a direction flip, `cancel_armed_buys` cancels every
armed tier; reconcile on the next tick could see ghost cancels.
**Mitigation/verification:** Verified by reading `cancel_armed_buys`
(grid_fisher.py:999) — it calls `cancel_buy_tier` which REMOVES the
tier from `inst.buy_ladder` after a SUCCESS response, so reconcile
won't see "armed but missing" later. `flip_direction` itself also
clears the ladder. Gate B counter will not increment on a flip. ACCEPT.
Issuer-side mass-cancel (instrument halt, barrier knockout) does
trigger Gate B, which is **the intended behavior** — those are exactly
the conditions we want a 6 h cooldown for.

### N5. Dashboard / Layer 2 hardcoded category enum [Layer 2 / dashboard]
**Chain:** New `skip_quote_stale` and `rapid_cancel_backoff` log
categories could break a consumer that filters on a known-categories
enum.
**Verification:** Grepped — `dashboard/app.py:865` just tails the file
with `load_jsonl_tail` (50 lines) and returns opaque entries. No enum.
`portfolio/agent_invocation.py` and `portfolio/reporting.py` don't read
`grid_fisher_decisions.jsonl`. ACCEPT.

### N6. Rapid-cancel counter not persisted between reconcile + persist [atomic I/O]
**Chain:** Counter is incremented in the reconcile branch of `tick`,
but `tick`'s `_persist()` call is at the end. If `PF-MetalsLoop` is
restarted mid-tick (which happens after every code merge per
`feedback_restart_loops.md`), the counter resets and the threshold is
never reached across restarts.
**Mitigation:** call `_persist()` immediately inside the reconcile loop
when `rapid_cancel_count` is mutated (single extra atomic write per
ghost cancel — cheap, the file is ~3 KB).

### N7. State schema v1→v2 silent-drift [test-passes-prod-differs]
**Chain:** Plan bumps version 1→2, but nothing checks. A future hand-
edited v3 state file would load as v2 with default zeros for any new
fields, silently dropping the older counter values.
**Mitigation:** at load time, if `state.version > GRID_STATE_SCHEMA_VERSION`,
log critical and bail. Existing `load_state` likely already handles
backward — review during Batch 1 and add forward-version assertion.

## Plan amendments from premortem

1. **Defensive shape check** in the new quote-fetch wrapper —
   missing/invalid `timeOfLast` → fail-safe skip, one-shot log.
2. **`threading.Lock` around the quote cache** — write helper
   `_get_cached_quote(ob_id)` that acquires the lock.
3. **`_persist()` inside reconcile** when `rapid_cancel_count`
   increments — never lose progress across a metals_loop restart.
4. **Forward-version assert** in `load_state` —
   `if state.version > GRID_STATE_SCHEMA_VERSION: critical + bail`.
5. **Stale-skips counter on dashboard payload** — add a small per-cycle
   counter dict so `/api/grid-fisher` can surface false positives.

