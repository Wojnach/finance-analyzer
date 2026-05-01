# Plan — Metals Trader / Config Follow-ups (2026-05-02)

## Context

Six metals-trader and metals-config findings carved out of
`docs/PLAN_adversarial_followups_20260502.md` for surgical fixing on
worktree `fix/metals-p1-followups-20260502`.

`data/metals_swing_trader.py` is 3,337 lines (not 7,667 as the task
description suggested — line numbers in the description are off but
the targets are unique enough to find by content).

## Findings to Fix

### 1. P1-9: Hardcoded `usdsek=10.85` in exit optimizer

- **File**: `data/metals_swing_trader.py:2823`
- **Symptom**: `MarketSnapshot(asof_ts=..., price=..., bid=None, usdsek=10.85)`
  passes a hardcoded FX rate to the exit optimizer. If SEK has moved 5-10%
  since the hardcoded value was set, EV calculations are systematically biased.
- **Fix**: Replace with `fetch_usd_sek()` (existing helper in
  `portfolio.fx_rates` — already does live fetch + 15-min in-process cache +
  stale-cache fallback + hardcoded 10.85 last-resort). Wrap call in try/except
  so an FX module crash falls back to 10.85 (current behaviour).
- **Test**: Patch `fetch_usd_sek` to return 11.20, verify it flows into the
  `MarketSnapshot.usdsek` argument (use mock for `compute_exit_plan`).

### 2. MC-P1-1: Orphan stop computed from entry price, not bid

- **File**: `data/metals_swing_trader.py:2667-2669` (`_set_stop_loss`)
- **Symptom**: When `_set_stop_loss(pos_id)` is called from `ingest_position`
  for an orphan position, it reads `pos["entry_price"]` (the orphan's original
  Avanza purchase price). For a LONG orphan now down 20%, the stop computed
  from entry sits ABOVE current price → triggers immediately on placement →
  forced sell at unfavourable price.
- **Fix**: Add an optional `anchor_price` parameter to `_set_stop_loss`.
  When called from `ingest_position`, pass the current warrant bid as the
  anchor. The default `None` preserves existing behaviour for fresh buys
  (where entry IS the current price).
- **Test**:
  - Fresh buy path: anchor=None → uses entry → identical to current behaviour.
  - Orphan path: pass a current bid that is below entry → stop is computed
    from the bid, not entry → trigger price is below current price.

### 3. MC-P1-3: `pos_id` collision on same-second buys

- **File**: `data/metals_swing_trader.py:2528`
- **Symptom**: `pos_id = f"pos_{int(time.time())}"`. If two `_execute_buy`
  calls land in the same epoch second (rare but possible — momentum entry
  + scheduled cycle), the second overwrites the first in `state["positions"]`.
  The original pattern at line 1043 (orphan path) was already fixed with
  `_{ob_id_str}` suffix. Same fix applies here.
- **Fix**: Change to `f"pos_{int(time.time())}_{warrant['ob_id']}"`. Two
  buys for the same ob_id within one second is impossible (BUY_COOLDOWN +
  MAX_CONCURRENT gates), so this is collision-proof.
- **Test**: Mock `time.time()` to return the same value for two `_execute_buy`
  calls on different ob_ids → both positions persist.

### 4. MC-P1-4: Zero-price sell on price-fetch failure

- **File**: `data/metals_swing_trader.py:2786-2787` (`_check_exits`)
- **Symptom**: `warrant_data = fetch_price(...)`; `current_bid = warrant_data.get("bid", 0) if warrant_data else 0`.
  If `fetch_price` returns None or a dict missing "bid", `current_bid=0`.
  The downstream `_execute_sell(..., current_bid=0, ...)` then places a
  SELL at price=0 — the order will either reject at Avanza (best case) or
  fill at the bid (worst case if Avanza interprets price=0 as "any price").
- **Fix**: Inside `_check_exits`, BEFORE the `_execute_sell` call (line
  2973), if `exit_reason` is set AND `current_bid <= 0`, abort the sell
  with a log + Telegram alert. The position stays in state and will be
  re-evaluated next cycle when (hopefully) the price fetch succeeds.
- **Test**: Position with active exit_reason + `fetch_price` patched to
  return None → `_execute_sell` is NOT called, position remains in state,
  warning is logged.

### 5. P1-8 + MC-P1-2: HARD_STOP too tight + HW/SW stop coordination

- **Files**:
  - `data/metals_swing_config.py:177` — `HARD_STOP_UNDERLYING_PCT = 2.0`
  - `data/metals_swing_config.py:264` — `STOP_LOSS_UNDERLYING_PCT = 2.5`
- **Current state**:
  - SW stop at 2.0% underlying → 10% on 5x cert → too tight (intraday wicks
    on XAG can be 1.5-2.5% in minutes during macro events)
  - HW stop at 2.5% underlying → 12.5% on 5x cert → also too tight, AND
    only fires after SW (correct ordering)
- **User memory** ("Wider stop-losses"): "5x certs need -15%+ stops, not
  -8%, to survive intraday wicks"
- **Fix**:
  - Raise `HARD_STOP_UNDERLYING_PCT` from `2.0` → `3.0` (5x → 15%)
  - Raise `STOP_LOSS_UNDERLYING_PCT` from `2.5` → `3.5` (5x → 17.5%)
  - The 0.5% gap keeps HW above SW so SW always fires first; HW is the
    safety net for process-down scenarios. Per MC-P1-2: HW must be ≥ SW.
- **Risk**: This is a LIVE-TRADING config change. If an actual catastrophic
  move occurs, losses will be 50% larger per position than under the
  previous tight stop (15% vs 10% on 5x). Mitigation: position sizing
  (`POSITION_SIZE_PCT=30`) is unchanged, so worst-case dollar loss per
  position is still bounded. The user explicitly prefers this trade-off.
- **Revert path**: Set both values back to 2.0/2.5 in
  `data/metals_swing_config.py` and restart `PF-MetalsLoop`.
- **Test**: Verify constants import correctly; no behaviour test (config
  values, semantics unchanged).

### 6. (Bonus) P1-14: `_coerce_epoch` defensive logging

- **File**: `portfolio/onchain_data.py:29-57`
- **Symptom**: When `_coerce_epoch` receives an unparseable value (corrupt
  cache, future format change), it returns `0.0` silently. A cache with
  a corrupt ts will silently force a cache miss on every restart, burning
  the BGeometrics 15 req/day budget. Operators have no signal that this
  is happening.
- **Fix**: Add a `logger.debug` line inside the final fall-through branch
  (after both isoformat and float parses fail) logging the type and a
  truncated repr of the unparseable value. Still returns 0.0.
- **Test**: Pass an unparseable value (e.g. `{"foo": "bar"}` or `None`),
  verify the debug log fires AND the return is 0.0.

## Worktree

`/mnt/q/finance-analyzer-metals-p1s` on branch `fix/metals-p1-followups-20260502`.

## Implementation order

1. P1-14 (smallest, isolated change, builds confidence)
2. MC-P1-3 (1-line, no behaviour change beyond uniqueness)
3. MC-P1-4 (defensive guard, well-isolated)
4. P1-9 (FX swap, well-isolated)
5. MC-P1-1 (function signature change, ripples to one call site)
6. P1-8 + MC-P1-2 (config-only, last so prior fixes are tested in isolation)

## Commit cadence

One commit per finding:
- `fix(onchain_data): P1 — log unparseable timestamp values in _coerce_epoch`
- `fix(metals_swing_trader): MC-P1 — pos_id collision-proof with ob_id suffix`
- `fix(metals_swing_trader): MC-P1 — guard zero-price sell on fetch failure`
- `fix(metals_swing_trader): P1 — use live FX rate in exit optimizer (was hardcoded 10.85)`
- `fix(metals_swing_trader): MC-P1 — use current bid as stop anchor on orphan ingest`
- `fix(metals_swing_config): P1+MC-P1 — widen HARD_STOP to 3% (15% on 5x) + HW>SW coordination`

## Test plan

- One targeted test per fix in the matching `tests/test_metals_*.py` or
  `tests/test_onchain*.py` file.
- After all 6 fixes: run targeted suites in isolation, then full
  `tests/test_metals_*.py tests/test_onchain*.py` collection.
- Final: do NOT merge or push (per `/fgl` protocol, return branch only).
