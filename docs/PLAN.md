# Plan: Metals Loop v7 — Cascading Stop-Loss + Derivative Exit

## Post-Mortem (Mar 2)
- Claude Layer 2 said HOLD 43 times, never SELL, even 2.27% from stop
- Monte Carlo P(stop) was static 0.42% — never updated — actual probability was 100%
- No rate-of-change detection — accelerating selloff went undetected
- Emergency sells failed for gold (Avanza "short sell not allowed" — already sold by broker stop)
- Chronos was blind all session (field mismatch bug, now fixed)

## Changes

### Batch 1: 3x Cascading Stop-Loss Orders (data/metals_loop.py)

Place 3 independent sell limit orders per position at startup, spread across 3 levels:
- S1 (33% of units): at configured stop price
- S2 (33% of units): at stop - 1% of stop
- S3 (remaining units): at stop - 2% of stop

This handles gap-through: if price gaps below S1, S2 and S3 may still fill.

Functions:
- `place_stop_loss_orders(page, positions)` — places 3 orders per active position
- `check_stop_order_fills(page, stop_state)` — monitors for fills
- `update_trailing_stops(page, positions, stop_state, prices)` — ratchets up when positions gain
- Stop order state persisted to `data/metals_stop_orders.json`

Trailing logic:
- When bid > entry * (1 + trail_start_pct), move stops up
- New S1 = max(current_S1, bid * (1 - trail_distance_pct))
- Minimum move of 1% to avoid excessive API calls
- Cancel old orders, place new ones at higher levels

### Batch 2: Derivative-Based Momentum Exit (data/metals_loop.py)

Monitor rate-of-change acceleration. When conditions met, auto-sell immediately:

Detection:
- Track price_history (already exists, 120-entry circular buffer)
- Compute: velocity = (bid_now - bid_N_checks_ago) / N  (1st derivative)
- Compute: acceleration = velocity_now - velocity_prev  (2nd derivative)
- If acceleration < threshold AND velocity < 0 AND abs(velocity) > min_velocity:
  → MOMENTUM EXIT: auto-sell via API

Parameters:
- `MOMENTUM_LOOKBACK = 5` (5 checks = ~7.5 min)
- `MOMENTUM_MIN_VELOCITY = -0.5` (must be dropping at least 0.5% per check)
- `MOMENTUM_ACCELERATION_THRESHOLD = -0.1` (must be accelerating)
- Only triggers when position is already L1+ (>8% from peak or <8% from stop)

### Batch 3: L2 Auto-Exit Override (data/metals_loop.py)

When position is <3% from stop, auto-sell instead of deferring to Claude.
Change: L2 ALERT (<5%) now triggers auto-sell when:
- Price trend is downward (last 3 bids declining)
- Position has been in L1+ zone for 5+ checks

This prevents the "43 HOLDs while approaching stop" pattern.

### Batch 4: Get Positions from Avanza API

Use `/_api/position-data/positions` endpoint to:
- Get actual holdings at startup (units, entry, P&L)
- Detect when broker stop-loss triggers independently

### Batch 5: Dynamic Monte Carlo

- Recalculate MC stop probability every 10 checks
- Use recent price_history volatility, not just historical daily ranges

## Files Changed
- `data/metals_loop.py` — main implementation (batches 1-4)
- `data/metals_risk.py` — dynamic MC updates (batch 5)
- `data/metals_stop_orders.json` — new, stop order tracking

## Execution Order
1. Batch 1 (stop orders) — highest priority, hardware protection
2. Batch 2 (momentum exit) — software early warning
3. Batch 3 (L2 override) — prevent Claude HOLD paralysis
4. Batch 4 (Avanza positions) — better state management
5. Batch 5 (dynamic MC) — better risk estimates
