# Fishing Position: Auto-Trailing Stop + EOD Sell

## Date: 2026-04-08

## Problem

When the user places a fishing limit buy order manually and it fills, the metals
loop treats it as a regular swing position: wide trailing stop, no EOD sell. The
user wants intraday fishing to be: tight 5% cert trail, auto-sell at 21:55.

## What to build (3 gaps)

### 1. Tag fishing positions (`_fishing: true`)
When detect_holdings discovers a new position whose ob_id is in the WARRANT_CATALOG
(from fin_fish_config.py), tag it as `_fishing: true` in the POSITIONS dict.

### 2. Tighter trailing stop for fishing positions
The existing `compute_smart_trail_distance` uses 5% on the cert price as its base
distance — this already matches the user's desired 5% trail. But TRAIL_START_PCT
(1% gain before trailing begins) may be too conservative for fishing. Fishing
positions should start trailing immediately (from entry) since they're already
bought at a dip level.

### 3. EOD auto-sell at 21:55 for fishing positions
Add a check in the main loop: if hour_cet >= 21:50 and any `_fishing` position
is active, sell it via emergency_sell. Swing positions hold overnight; fishing
positions don't.

## Design

### Tagging (in detect_holdings, ~5 lines)
Build a set of fishing ob_ids from WARRANT_CATALOG at module load.
When a new position is added to POSITIONS, check if ob_id is in that set.
If yes, add `_fishing: true` to the position dict.

### Trailing (in update_smart_trailing_stops, ~3 lines)
For positions with `_fishing: true`, set TRAIL_START_PCT to 0 (trail immediately).
The base 5% distance is already correct.

### EOD sell (new function + hook in main loop, ~20 lines)
`_eod_sell_fishing_positions(page)`: at 21:50 CET, sell all `_fishing` positions.
Uses existing `emergency_sell()` function. Sends Telegram notification.

## Files changed
- `data/metals_loop.py` — all 3 changes
- `tests/test_fish_engine_integration.py` — tests for fishing tag + EOD

## What could break
- False tagging: an ob_id in WARRANT_CATALOG that isn't a fishing position (e.g.,
  the user buys a BULL X5 for a swing trade, not fishing). Mitigated: only new
  positions auto-detected by detect_holdings get tagged, not pre-existing ones.
- EOD sell at 21:50 could sell a position the user wants to hold overnight.
  Mitigated: only `_fishing: true` positions get sold. User can manually remove
  the flag by editing metals_positions_state.json.

## Execution order
1. Write tests first
2. Implement in metals_loop.py (single batch — all changes are in one file)
3. Run tests
4. Code review
5. Merge + push
