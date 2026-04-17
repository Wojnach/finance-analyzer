# Momentum-Entries Plan — 2026-04-17

## Goal

Close four specific gaps in the metals loop that together caused the system to
miss today's coordinated silver/BTC/ETH breakout even though signals flagged it
sustained. Ship one cohesive change: make the metals loop detect upside momentum
for entries (not just downside for exits), execute it at realistic conviction
thresholds, and do so with cash sizing that actually meets Avanza's courtage
floor.

## What was wrong (concrete, from code + runtime state at 08:28 CET 2026-04-17)

1. **Adaptive sizing missing on Kelly-fallback path.** `metals_swing_trader.py:1838`
   sets `alloc = cash * POSITION_SIZE_PCT / 100` with no `MIN_TRADE_SEK` floor.
   With current `cash=2822 SEK` and `POSITION_SIZE_PCT=30`, alloc becomes 847 SEK,
   then the next `if alloc < MIN_TRADE_SEK` (line 1846) rejects the trade outright.
   The Kelly-primary path (line 1805) already floors correctly, but when Kelly
   returns no-edge or errors, we silently skip even when cash is sufficient to
   meet the courtage floor.

2. **Silver fast-tick gated on existing position.** `metals_loop.py:933` inside
   `_sleep_for_cycle` short-circuits the 10s sub-loop when `_has_active_silver()`
   returns False. With zero silver positions held, the fast-tick machinery is
   dark — so the system cannot react in sub-cycle granularity to a breakout
   that would be an entry trigger.

3. **Entry-momentum detection does not exist.** All momentum primitives in
   `metals_loop.py` are downside-only: `SILVER_VELOCITY_ALERT_PCT = -0.8`
   (three-minute drop for warning), `MOMENTUM_MIN_VELOCITY = -0.5`, trailing
   stop tightens on negative acceleration. There is no `+0.8%/3min` or
   equivalent rising-velocity detector, and no path for it to feed into the
   swing trader's entry evaluator.

4. **Fish engine disable comment is stale and misleading.** `metals_loop.py:766`
   says `FISH_ENGINE_ENABLED = False  # 2026-04-15: disabled after 12 consecutive
   losses (-12,257 SEK session). Re-enable only after 6 known integration bugs
   fixed`. Memory file `project_fish_engine_live_test.md` confirms those 6 bugs
   ARE fixed (since 2026-04-09). The April-15 loss happened AFTER the fixes —
   the strategy itself has no edge, the integration works. Leaving the door
   open with a "re-enable after bug fix" comment invites a future operator to
   flip the switch back on under a false premise.

## Design

### Feature 1: Adaptive sizing (Kelly-fallback path floor + cash-feasibility check)

**File:** `data/metals_swing_trader.py`

Replace the fallback branch `alloc = cash * POSITION_SIZE_PCT / 100` with:

```python
alloc = cash * POSITION_SIZE_PCT / 100
# Floor at courtage minimum. A smaller order pays the fixed courtage which
# destroys EV on leveraged intraday plays; the floor is the hard minimum.
alloc = max(alloc, float(MIN_TRADE_SEK))
# Cap at 95% of cash so we keep a small courtage buffer.
alloc = min(alloc, cash * 0.95)
```

Add a pre-check: if `cash * 0.95 < MIN_TRADE_SEK`, log and skip (genuinely
cannot place an order of the required minimum size from this cash).

**Tests:**
- `test_fallback_alloc_floors_at_min_trade_sek` — cash=2822, POSITION_SIZE_PCT=30
  yields alloc=1000 (not 847), passes the subsequent min check.
- `test_fallback_alloc_caps_at_95_pct_cash` — cash=1500 with 30% pct floors to
  1000, caps at 1425 → final 1000.
- `test_fallback_alloc_skips_when_cash_below_floor` — cash=900: cash*0.95=855,
  no order placed.

### Feature 2: Entry-side fast-tick (always-on positive-velocity detector)

**File:** `data/metals_loop.py`

New config constants next to the existing silver fast-tick block:

```python
SILVER_ENTRY_FAST_TICK_ENABLED = True
SILVER_ENTRY_VELOCITY_WINDOW = 18          # 18 x 10s = 3 min
SILVER_ENTRY_VELOCITY_ALERT_PCT = 0.8      # +% rise threshold
SILVER_ENTRY_MIN_RVOL = 1.5                # require volume ratio above this
SILVER_ENTRY_DEDUP_WINDOW_SEC = 300        # at most one trigger per 5 min
GOLD_ENTRY_FAST_TICK_ENABLED = True
GOLD_ENTRY_VELOCITY_WINDOW = 18
GOLD_ENTRY_VELOCITY_ALERT_PCT = 0.4        # gold's typical range is ~half of silver's
GOLD_ENTRY_MIN_RVOL = 1.5
```

New module state:

```python
_xag_entry_prices = deque(maxlen=SILVER_ENTRY_VELOCITY_WINDOW)
_xau_entry_prices = deque(maxlen=GOLD_ENTRY_VELOCITY_WINDOW)
_entry_last_trigger_ts = {}   # {"XAG-USD": float, "XAU-USD": float}
```

New functions:

- `_xau_fetch_price()` — XAUUSDT lightweight fetch, mirror of `_silver_fetch_xag`.
- `_entry_fast_tick(ticker, fetch_fn, prices_deque, threshold_pct, min_rvol, dedup_sec)`:
  fetch live price, append, compute velocity when deque full. If >= threshold
  AND rvol>=min AND dedup window elapsed → write momentum state to
  `data/metals_momentum_state.json` and send Telegram.
- `_write_momentum_candidate(ticker, direction, velocity_pct, price, rvol)` —
  atomic write via `atomic_write_json`. File format:

```json
{
  "XAG-USD": {
    "direction": "LONG",
    "velocity_pct": 0.92,
    "price_at_trigger": 79.85,
    "rvol": 2.2,
    "triggered_at": "2026-04-17T08:30:12+00:00",
    "consumed_at": null,
    "ttl_sec": 300
  }
}
```

Gating change in `_sleep_for_cycle`: the existing short-circuit

```python
if not SILVER_FAST_TICK_ENABLED or not _has_active_silver():
```

becomes

```python
exit_tick_active = SILVER_FAST_TICK_ENABLED and _has_active_silver()
entry_tick_active = (
    SILVER_ENTRY_FAST_TICK_ENABLED or GOLD_ENTRY_FAST_TICK_ENABLED
)
if not (exit_tick_active or entry_tick_active):
    simple_sleep; return
```

Inside the sub-loop, dispatch both when applicable.

**Tests:** (`tests/test_metals_entry_fasttick.py`)
- `test_entry_fast_tick_writes_candidate_on_positive_velocity` — seed deque
  with rising prices, verify candidate file written with direction=LONG.
- `test_entry_fast_tick_skips_when_velocity_below_threshold` — rising +0.3%
  only → no candidate.
- `test_entry_fast_tick_dedup_window` — two triggers within 5 min → only first
  writes.
- `test_entry_fast_tick_requires_rvol` — +1% velocity but rvol=0.8 → skipped.

### Feature 3: Momentum entry path in swing trader

**Files:** `data/metals_swing_trader.py`, `data/metals_swing_config.py`

New config constants in `metals_swing_config.py`:

```python
MOMENTUM_ENTRY_ENABLED = True
MOMENTUM_MIN_BUY_CONFIDENCE = 0.50   # relaxed from 0.60
MOMENTUM_MIN_BUY_VOTERS = 2          # relaxed from 3
MOMENTUM_CANDIDATE_TTL_SEC = 300     # ignore candidates older than 5 min
MOMENTUM_STATE_FILE = "data/metals_momentum_state.json"
```

New helpers on the trader:

```python
def _check_momentum_candidate(self, ticker: str) -> dict | None:
    """Return fresh unconsumed LONG momentum candidate for ticker, else None."""

def _consume_momentum_candidate(self, ticker: str) -> None:
    """Mark a momentum candidate consumed so it doesn't retrigger."""
```

In `_evaluate_entry(sig, ticker)`:

```python
momentum = None
if MOMENTUM_ENTRY_ENABLED and direction == "LONG":
    momentum = self._check_momentum_candidate(ticker)

min_conf = MOMENTUM_MIN_BUY_CONFIDENCE if momentum else MIN_BUY_CONFIDENCE
min_voters = MOMENTUM_MIN_BUY_VOTERS if momentum else MIN_BUY_VOTERS

if confidence < min_conf:
    return False, f"confidence {confidence:.2f} < {min_conf}"
...
if majority < min_voters:
    return False, f"{direction}_count={majority} < {min_voters}"
```

Other gates unchanged. Momentum override only relaxes confidence and voter
count. RSI-zone, MACD-improving, regime-confirm, TF-alignment stay full-strength.

After a successful `_execute_buy`, call `_consume_momentum_candidate(ticker)`.

**Tests:** (`tests/test_metals_swing_momentum.py`)
- `test_evaluate_entry_uses_relaxed_gates_with_momentum`
- `test_evaluate_entry_rejects_below_relaxed_gates`
- `test_momentum_candidate_expires_after_ttl`
- `test_momentum_candidate_consumed_after_entry`
- `test_momentum_candidate_ignored_for_short`
- `test_evaluate_entry_unchanged_without_momentum` (regression guard)

### Feature 4: Fish engine permanent deprecation gate

**File:** `data/metals_loop.py`

Replace the current disable comment with a dated deprecation note and add a
startup hard-fail guard so anyone flipping the flag back on without removing
the guard is loudly surfaced:

```python
# Fish engine permanently deprecated 2026-04-17.
#
# The 6 integration bugs from project_fish_engine_live_test.md were fixed
# 2026-04-09, but the subsequent 2026-04-15 session lost 12,257 SEK across
# 12 consecutive trades. Integration works; the strategy itself has no
# measurable edge given current signal quality. The swing trader with the
# new momentum-entry path (feat/momentum-entries 2026-04-17) supersedes it
# for upside-breakout entries.
#
# Do NOT re-enable without running at least 30 walk-forward paper trades
# through a fresh backtest demonstrating positive expectancy.
FISH_ENGINE_ENABLED = False
```

Add an assertion near the entry points that use the flag so a mistaken enable
is caught at module import:

```python
assert not FISH_ENGINE_ENABLED, (
    "Fish engine is deprecated — see comment at declaration. "
    "Re-enable requires 30+ paper trades with positive expectancy."
)
```

**Tests:** (`tests/test_fish_engine_deprecated.py`)
- `test_fish_engine_flag_stays_false` — regression guard that
  `FISH_ENGINE_ENABLED` remains False.
- `test_fish_engine_assertion_catches_enable` — monkeypatch and verify the
  assertion fires.

## What could break

- **Loop cycle timing.** Entry fast-tick adds two HTTP GETs per 10s tick
  (XAG + XAU). ~12 extra requests/min to Binance FAPI. Well under rate limits.
- **State file contention.** Written by loop, read by swing trader (same
  process). Atomic writes via `atomic_write_json` plus fresh reads avoid
  partial-read risk.
- **False breakouts on thin-liquidity opens.** `RVOL >= 1.5` gate is the
  primary defense. If still too many false positives, raise velocity threshold
  to +1.0% silver / +0.6% gold.
- **Kelly path unaffected.** Adaptive sizing fix only changes the fallback
  branch. Kelly-primary math is unchanged.
- **Parallel session coordination.** Other active worktrees:
  `research/adversarial-round-2-20260417` touches meta_learner + trigger
  modules; `fix/tier-downshift-low-conviction` is doc-only so far. No file
  overlap with this branch.

## Execution order

1. Batch 0 (plan): commit this file.
2. Batch 1: adaptive sizing — swing trader fallback branch + tests.
3. Batch 2: entry-side fast-tick — metals_loop.py additions + tests.
4. Batch 3: swing trader momentum-entry path — entry evaluator + consume +
   config constants + tests.
5. Batch 4: fish engine deprecation gate + tests.
6. Batch 5: full test suite + SESSION_PROGRESS.md update.
7. Batch 6: merge to main (pull latest first to avoid clobber), push via
   Windows git, restart loops, CHANGELOG.

## Non-goals

- SHORT-side momentum entries. `SHORT_ENABLED=False` and canary warrants are
  empty — no production path yet.
- Backtesting the velocity thresholds. Chosen as symmetric mirrors of exit
  thresholds. Tune after live observation.
- Cross-asset conditional Monte Carlo. Scope is the momentum gap on metals;
  joint MC across BTC/ETH/XAG is a separate feature.
