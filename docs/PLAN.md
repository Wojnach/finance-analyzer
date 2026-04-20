# Plan — Warrant-side take-profit exit (2026-04-20 evening)

## Problem

Live incident: MINI L SILVER AVA 331 bought 10:33 at 14.62 SEK.
- Warrant peaked at **15.48 (+5.9%)** around 16:00
- Underlying silver peaked at only **+1.26%** — below the 3% `TAKE_PROFIT_UNDERLYING_PCT` threshold
- System held through two rejection peaks (15.48, 15.35) because exit rules are all underlying-indexed
- Now back at 15.11 (+2.0%), gave back ~120 SEK of peak profit

Root cause: market makers can mark the warrant up 5-6% on a 1-2% underlying move (spread widening, liquidity, momentum premium). The exit logic only watches underlying %.

## Fix — Add warrant-side take-profit and trailing

New config knobs in `data/metals_swing_config.py`:
- `WARRANT_TAKE_PROFIT_PCT = 5.0` — exit when warrant bid ≥ entry × 1.05
- `WARRANT_TRAILING_START_PCT = 3.0` — activate warrant-side trailing at +3%
- `WARRANT_TRAILING_DISTANCE_PCT = 1.5` — 1.5% retrace from warrant peak = exit

New exit-rule block in `_check_exits()` at `data/metals_swing_trader.py:2720` (parallel to existing take-profit / trailing rules). Runs ONLY if the existing underlying-side rules haven't already fired. Tracks `peak_warrant_bid` on the position dict.

Additive only — underlying-side rules stay intact.

## Why these thresholds

Warrant at 5x leverage: 3% underlying ≈ 15% warrant. Today's incident showed warrant +5.9% at underlying +1.26% = **4.7x actual leverage seen on warrant bid** (extra from MM spread dynamics). So:
- 5% warrant take-profit ≈ 1.0% underlying equivalent — conservative, captures the MM-spike pattern
- 3% warrant trail-start ≈ same as 0.6% underlying — earlier activation than the 1.5% underlying trail
- 1.5% warrant trail-distance ≈ snugger than 1% underlying trail

If we're too eager: lose late-rally upside (can rebuy).
If we're too slow: repeat today's miss (can't get back).

## What could break

- **Double-fire:** warrant take-profit and trailing trigger same cycle — fine, they produce the same exit, first match wins via `if not exit_reason`.
- **Peak-warrant tracking on corrupt state:** new field missing on legacy positions. Guard with `pos.setdefault("peak_warrant_bid", current_bid)` on each eval.
- **MM bid = 0 during session outages:** guarded by existing `current_bid = warrant_data.get("bid", 0)` — skip eval when bid ≤ 0.
- **Tests:** 4 new tests in `tests/test_metals_swing_trader.py` (take-profit fires, trailing arms+fires, bid=0 no-op, peak tracking persists).

## Batches

### Batch 1 — Config + exit rule + tests
- `data/metals_swing_config.py`: add 3 constants
- `data/metals_swing_trader.py`: add warrant-side TP/trailing in `_check_exits`, track `peak_warrant_bid`
- `tests/test_metals_swing_trader.py`: 4 new tests

### Batch 2 — Adversarial review
- `pr-review-toolkit:code-reviewer` + `silent-failure-hunter` on diff
- Codex if auth back (else skip, document)

### Batch 3 — Tests + ship
- Full pytest suite
- Merge + push + restart `PF-MetalsLoop`

## Order
1. Batch 1 (implement + unit tests)
2. Batch 2 (review)
3. Batch 3 (full suite + merge + push + restart loop)
