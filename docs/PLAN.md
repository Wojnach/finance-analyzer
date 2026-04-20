# Plan — Warrant-side exit fix (2026-04-20 evening, v2 post-research)

## Today's incident (ground truth)

- **Entry**: 10:33:56 CET — MINI L SILVER AVA 331 (ob_id 2379768), 210u @ warrant 14.62 SEK, underlying silver $79.38
- **Peak**: ~16:00 CET — warrant 15.48 (+5.9%), underlying $80.38 (+1.26%)
- **Exit**: 21:50:16 CET — warrant 14.90 (+1.9%), `EXIT_OPTIMIZER: SESSION_END_IMMINENT, HOLD_TIME_EXTENDED (EV +58 SEK)`
- **Missed**: ~122 SEK of peak profit (sold at +1.9%, could have captured ~+4% with trailing)

## Why no rule fired at peak

- Underlying moved only +1.26% — below `TRAILING_START_PCT = 1.5%` → trailing never armed
- Warrant moved +5.9% (market-maker markup / spread games / momentum premium)
- **No exit rule watches warrant bid.** `MarketSnapshot.bid` in the optimizer is documented as the underlying USD bid (not the warrant SEK bid — see 2026-04-09 `-2430 SEK` fake-loss bug comment at `metals_swing_trader.py:2676-2682`)

## Exit hierarchy (complete map, confirmed via research)

In `_check_exits` at `data/metals_swing_trader.py:2565`:
1. **Layer 0**: `exit_optimizer` — runs first. Overrides rules only on `KNOCKOUT_DANGER`, `SESSION_END_IMMINENT`, or `stop_hit_prob > 0.30`. Otherwise advisory.
2. **Layer 1a**: Underlying take-profit (`und_change_pct >= 3%`)
3. **Layer 1b**: Underlying trailing (`und_change_pct >= 1.5%` arms, `from_peak_pct <= -1.0%` fires)
4. **Layer 2**: Hard stop (`und_change_pct <= -2%`)
5. **Layer 3**: Signal reversal (SELL consensus with ≥ MIN_BUY_VOTERS)
6. **Layer 4**: Time limit (24h safety net)
7. **Layer 5**: EOD exit (minutes_to_close ≤ 0)

All gated by `if not exit_reason` — first match wins.

**`trailing_active` flag is display-only** — set at line 2726, read only at line 3088 (Telegram summary). Not a decision gate. The actual trailing exit condition is `from_peak_pct <= -TRAILING_DISTANCE_PCT` checked directly, independent of the flag.

## Research findings (per user instruction: understand before changing)

- **Exit thresholds were repeatedly LOOSENED** (TP 2% → 3% on 2026-04-14, momentum hold 10m → 5m on 2026-04-17) because tighter thresholds caused false exits on noise.
- **No prior code tracks warrant-side price.** Zero mentions of `warrant_bid`, `peak_warrant`, etc. in main. Today is the first recognition of MM-markup pattern.
- **Exit optimizer is probabilistic-advisory only.** Its risk override saved us today at EOD but couldn't have caught the 16:00 peak because the optimizer also operates on underlying simulation.
- **`MarketSnapshot.bid` must stay underlying-only.** Passing warrant bid caused the -2430 SEK bug. Warrant-side handling must be OUTSIDE the optimizer.
- **mstr_loop has same architecture** but no optimizer — same blind spot. Out of scope for this branch but noted for follow-up.

## Fix (v2, design corrected per review findings)

Add parallel warrant-side take-profit + trailing rules in `_check_exits`, tracking `peak_warrant_bid` on the position dict. **Sticky trailing flag** (read back, not just written) — this is the HIGH bug that reviewer-2 caught in v1.

### Config (`data/metals_swing_config.py`)
```python
WARRANT_TAKE_PROFIT_PCT = 5.0          # exit when bid/entry >= 1.05
WARRANT_TRAILING_START_PCT = 3.0       # arm trailing at bid/entry >= 1.03
WARRANT_TRAILING_DISTANCE_PCT = 1.5    # exit on 1.5% retrace from warrant peak
```

### Code (`data/metals_swing_trader.py`, `_check_exits` block 2b)
Correct structure (v2):
```python
# STICKY: arm once, remain armed even if warrant retraces below +3%
if direction == "LONG" and current_bid > 0 and entry_warrant > 0:
    warrant_change_pct = (current_bid / entry_warrant - 1) * 100
    peak_bid = pos.get("peak_warrant_bid", entry_warrant)
    if current_bid > peak_bid:
        pos["peak_warrant_bid"] = current_bid
        peak_bid = current_bid
    from_peak_warrant_pct = (current_bid / peak_bid - 1) * 100 if peak_bid > 0 else 0

    # Arm trailing (sticky — once True, stays True)
    if warrant_change_pct >= WARRANT_TRAILING_START_PCT:
        pos["warrant_trailing_active"] = True

    # Take-profit (fires at +5%)
    if not exit_reason and warrant_change_pct >= WARRANT_TAKE_PROFIT_PCT:
        exit_reason = f"WARRANT_TAKE_PROFIT: warrant +{warrant_change_pct:.2f}% >= +{WARRANT_TAKE_PROFIT_PCT}%"
    # Trailing (fires once ARMED and retrace exceeds distance — even if warrant_change_pct back below +3%)
    elif (not exit_reason
            and pos.get("warrant_trailing_active")
            and from_peak_warrant_pct <= -WARRANT_TRAILING_DISTANCE_PCT):
        exit_reason = f"WARRANT_TRAILING_STOP: {from_peak_warrant_pct:.2f}% from warrant peak"
```

Key changes from v1:
- **Sticky flag read**: `pos.get("warrant_trailing_active")` gate on trailing — fires after retrace even if warrant_change dropped below +3% (the v1 HIGH bug)
- **Arm before check**: trailing activation happens unconditionally when warrant hits +3%, even if TP fires same cycle (non-destructive)
- **SHORT grep anchor**: inline `# TODO(short-reenable)` comment so future SHORT re-enablement grep-finds this skip (reviewer-1 P2)

### Tests (`data/test_metals_swing_trader.py`)

Replace v1 tests + add new coverage:
1. `test_warrant_take_profit_fires` — bid +7% → WARRANT_TAKE_PROFIT fires (existing, keep)
2. `test_warrant_trailing_fires_after_retrace_below_start` — **NEW**: bid rises to +5%, arms; drops to +2.5% (below +3% start); retrace from peak > 1.5% → must fire (this tests the HIGH bug fix)
3. `test_warrant_trailing_fires_in_arm_band` — **NEW**: bid between +3% and +5%, peak higher, retrace > 1.5% → fires (without TP interference)
4. `test_warrant_rules_noop_when_bid_zero` — bid=0 → no exit (existing)
5. `test_warrant_peak_persists_across_cycles` — **NEW**: verify `peak_warrant_bid` updates on rising bid, doesn't reset
6. `test_warrant_short_direction_skipped` — **NEW**: SHORT position → warrant rules silently skipped (document behavior until SHORT re-enabled)

### State persistence (reviewer-1 P2-1, noted but out of scope)
`peak_warrant_bid` mutations persist on the next `_save_state` call in `_check_exits` via the existing `dirty = True` path when any position is removed. On no-exit cycles, the peak is not flushed until another caller saves state. **This matches the pre-existing `peak_underlying` behavior** — accepted as a consistent convention, flagged for joint follow-up.

## Rollback of v1 bugs

Before writing v2, reset the three bugs in the current branch:
1. HIGH: elif structure missed trailing on retrace below +3% → **fixed by sticky flag**
2. MEDIUM: `test_warrant_trailing_activates_and_fires` was testing TP not trailing → **replaced by two new targeted tests**
3. MEDIUM: SHORT skip silent → **grep anchor comment added**

## Batches

### Batch 2 (on existing branch) — v2 rewrite
- Reset v1 additions (keep config, rewrite exit block + tests)
- Commit as `fix(metals_swing): warrant-side exit v2 — sticky trailing + grep anchor`

### Batch 3 — Re-review
- Run both pr-review-toolkit agents again on the new diff
- Codex if out of usage cap

### Batch 4 — Test + ship
- Full pytest (tolerate pre-existing 18-24 failures)
- Merge + push + restart `PF-MetalsLoop` (code is loaded from data/, loop import at startup only — restart required)

## Risk budget

- **Silver position is CLOSED** (sold via optimizer at 21:50). No live exposure.
- **EU + US markets closed** (22:24 CET at time of plan). Metals cert MM closed (21:55). Full ~10h before next market open tomorrow.
- **Change is additive** — existing underlying-side rules untouched, warrant-side runs in parallel.
- **Worst case on production bug**: trailing fires earlier than intended → locks in smaller profit. Equivalent to the existing underlying-side TP behavior at different thresholds.

## Not changing this session

- Underlying-side `trailing_active` dead-state (pre-existing, same pattern)
- Exit_optimizer internal logic (too large a change without incident data)
- mstr_loop warrant-side rules (same fix pattern, separate branch)
- State persistence for peak fields (pre-existing convention)
