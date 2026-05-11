# PLAN — Grid Market-Maker for Warrants (2026-05-11)

## Goal

Marja Folcke–style limit-order grid fishing applied to leveraged warrants. Stand up a
continuous market-maker that places multi-tier resting buy limits on a selected warrant
direction (BULL or BEAR), rotates each fill into an opposite-side sell limit + stop,
and reconciles continuously. Three underlyings: silver, gold, oil. One side active per
underlying at a time, signal-gated. Live from day one — user explicitly rejected DRY_RUN
phasing. Tiny size (1200 SEK per leg) is the risk control.

## Why

- Single-leg fish (`portfolio/fin_fish.py`) is disabled, never reached production parity
  with the deprecated `data/fish_engine.py`.
- `metals_swing_trader.py` is a directional one-position model; it cannot capture
  oscillation cash flow that mean-reversion-in-range provides.
- Marja's grid pattern (many small resting limits, capture spread+swing) maps cleanly to
  leveraged warrants if we accept smaller per-leg target (1.2-1.5%) than her 5-10% native
  microcap spread, because 5x leverage amplifies the underlying move.
- User wants 3 underlyings active simultaneously: XAG (silver), XAU (gold), WTI/Brent
  (oil) — needs first oil warrant catalog entry.

## Inspiration & Constraints from Memory

- `feedback_fishing_hold_time.md` (this session): hold MINUTES TO HOURS, never overnight.
- `feedback_stops_outside_volatility.md`: stops sit outside intraday volatility band.
- `feedback_min_order_size_1000_sek.md`: each leg ≥ 1000 SEK; 1200 SEK target clears it.
- `feedback_lead_with_probabilities.md`: every order placement logs P(fill within session).
- `reference_avanza_stops_orders_coexist.md`: stop-loss + sell limit coexist, full volume.
- `reference_turbo_mini_pricing.md`: pull live parity/barrier/strike before sizing.
- `project_fish_engine_live_test.md`: 12,257 SEK loss on 2026-04-15 — read before live.

## Interpretation of User Requirements

- **"1200 SEK per instrument"** → interpreting as **per LEG** (per individual limit
  order). Total per instrument-direction = 1200 × N_TIERS = 3600 SEK with default
  3 tiers. Per-instrument hard cap of 6000 SEK to allow oversize from rotation. Flagging
  in commit; user can redirect.
- **"Both sides" misunderstanding** → user is correct: hold ONE direction at a time per
  underlying, signal-gated. Drop the synthetic-MM-both-sides idea; only BULL XOR BEAR
  active at any time, switched on signal flip with a cooldown.
- **"Silver + Gold + Oil"** → XAG, XAU, WTI/Brent. Brent (OLJAB) warrants have tighter
  spreads than GSCI (0.08% vs 0.15%) → prefer Brent.
- **"Live immediately"** → no DRY_RUN gate. Per-leg 1200 SEK is the real safety budget.

## Catalog (Discovered)

| Underlying | Direction | Cert                  | Orderbook | Spread% | Last (SEK) |
|------------|-----------|-----------------------|-----------|---------|------------|
| XAG-USD    | LONG      | BULL_SILVER_X5_AVA_4  | 1650161   | (live)  | (live)     |
| XAG-USD    | SHORT     | BEAR_SILVER_X5_AVA_12 | 2286417   | (live)  | (live)     |
| XAU-USD    | LONG      | BULL_GULD_X5_AVA      | 738811    | (live)  | (live)     |
| XAU-USD    | SHORT     | BEAR_GULD_X5_VON4     | 1047859   | 2.2%    | (live)     |
| WTI/Brent  | LONG      | BULL_OLJAB_X5_AVA_2   | 2367797   | 0.08    | 24.73      |
| WTI/Brent  | SHORT     | BEAR_OLJAB_X5_AVA_2   | 2367803   | 0.11    | 9.20       |

Live spreads re-pulled at startup and every cycle; the table above is a cache snapshot.

## Architecture

### New files (5)

1. **`portfolio/grid_fisher.py`** — orchestrator. State machine per (instrument), tick
   driver, fill detection via open-orders diff, exit rotation, EOD sweep.
2. **`portfolio/grid_tiers.py`** — pure-function tier math. Inputs: bid, ATR, structural
   levels, knockout level, fill_probability. Output: list of (price, qty, expected_p_fill).
3. **`portfolio/grid_fisher_config.py`** — config constants. ENABLED flag, tier %s, sizes,
   cooldowns, per-instrument caps, EOD timing.
4. **`data/grid_fisher_state.json`** — runtime state. Atomic R/W via `file_utils`.
5. **`data/grid_fisher_decisions.jsonl`** — append-only decision log.

### Modified files (3)

- **`data/metals_loop.py`** — wire `GridFisher.tick()` into 60s cycle after swing trader.
- **`data/fin_fish_config.py`** — add PREFERRED_INSTRUMENTS entries for oil warrants.
- **`dashboard/app.py`** — new `/api/grid-fisher` endpoint exposing state file.

### New tests (3)

- `tests/test_grid_tiers.py` — tier math, knockout skip, min-order, per-cap.
- `tests/test_grid_fisher_state.py` — state transitions, atomic write, schema versioning.
- `tests/test_grid_fisher_reconcile.py` — open-orders diff with partial fills, gaps.

## State Schema (`data/grid_fisher_state.json`)

```json
{
  "version": 1,
  "session_id": "2026-05-11",
  "halted": false,
  "halt_reason": null,
  "global_session_pnl_sek": 0.0,
  "global_max_dd_sek": 0.0,
  "by_instrument": {
    "<ob_id>": {
      "ticker": "XAG-USD",
      "active_direction": "LONG",
      "cert_name": "BULL_SILVER_X5_AVA_4",
      "buy_ladder": [
        {"tier": 0, "order_id": "abc", "price": 42.50, "qty": 24,
         "placed_ts": "2026-05-11T10:00:00Z", "status": "ARMED",
         "p_fill_session": 0.68}
      ],
      "sell_ladder": [
        {"linked_buy_tier": 0, "order_id": "xyz", "price": 43.30, "qty": 24,
         "placed_ts": "...", "status": "ARMED"}
      ],
      "stop_loss_id": "stop123",
      "stop_loss_price": 41.50,
      "inventory_units": 24,
      "avg_entry_price": 42.50,
      "session_pnl_sek": -12.5,
      "fills_this_session": 1,
      "consecutive_losses": 0,
      "cooldown_until": null,
      "last_direction_flip_ts": null
    }
  }
}
```

## Order Lifecycle

```
[STARTUP]
  → load state, reconcile against live Avanza open_orders + positions
  → if direction flipped vs cached signal → cancel old ladder, rearm new direction
  → if EOD reached → cancel all, force flat

[TICK every 60s in metals_loop]
  1. Pull live signal consensus for ticker (BUY/SELL/HOLD + confidence)
  2. If HOLD or confidence < MIN_BUY_CONFIDENCE → skip placement, leave existing alone
  3. If direction differs from active_direction:
       - cancel all unfilled buy tiers on old direction
       - if inventory > 0 on old direction → leave sell ladder + stop, let it exit
       - mark cooldown to prevent flip-flop
  4. If new placement needed:
       - call grid_tiers.build(bid, ATR, knockout, range)
       - for each tier missing in state → place_buy_order via avanza_session
       - update state
  5. Poll open_orders + positions; diff against state:
       - missing order_id → mark FILLED → trigger rotation
       - rotation: place sell limit at fill_price * (1+TARGET_PCT)
       - rotation: place stop_loss at fill_price * (1-STOP_PCT)
  6. If session_pnl_sek <= -PER_SESSION_LOSS_LIMIT → halt instrument
  7. If global drawdown breached → halt all

[EOD at close - 10min]
  1. Cancel all unfilled buy tiers
  2. Tighten sell limits to bid + 0.5% to encourage close-of-day exit
  3. At close - 5min, if inventory > 0 → market sell remaining
  4. Reset session_pnl_sek for next day
```

## Risk Machinery (reused, not bypassed)

- `portfolio/risk_management.check_drawdown()` — global 20% halt
- `portfolio/exit_optimizer._compute_risk_flags()` — knockout proximity
- `portfolio/trade_guards.check_overtrading_guards()` — per-instrument cooldown
- `portfolio/price_targets.fill_probability_buy()` — tier ranking
- `file_utils.atomic_write_json()` / `atomic_append_jsonl()` — state I/O

## Configuration Defaults

```python
GRID_FISHER_ENABLED = True               # live from start
GRID_TIERS = 3
GRID_TIER_SPACING_PCT = [0.3, 0.8, 1.5]  # buy below bid
GRID_TARGET_PCT = 1.2                    # sell above fill (post-courtage clear)
GRID_STOP_PCT = 3.5                      # stop below fill (outside intraday vol)
GRID_LEG_SEK = 1200                      # user's stated cap, per-leg
GRID_MAX_LEGS_PER_INSTRUMENT = 3
GRID_PER_INSTRUMENT_MAX_SEK = 6000       # = ~5 × max-fills × leg-size
GRID_PER_SESSION_LOSS_LIMIT_SEK = 500
GRID_EOD_SWEEP_MINUTES_BEFORE = 10
GRID_EOD_MARKET_SELL_MINUTES_BEFORE = 5
GRID_DIRECTION_FLIP_COOLDOWN_MIN = 30
GRID_MIN_SIGNAL_CONFIDENCE = 0.56        # matches metals MIN_BUY_CONFIDENCE
GRID_ADX_TREND_FILTER = 35               # above this → only-with-trend mode
GRID_ACTIVE_INSTRUMENTS = {
    "XAG-USD": {"LONG": 1650161, "SHORT": 2286417},
    "XAU-USD": {"LONG": 738811,  "SHORT": 1047859},
    "OIL":     {"LONG": 2367797, "SHORT": 2367803},
}
```

## Trend-Filter Gate (prevent MM-in-trend disaster)

If ADX(14) > 35: only place ladder on the WITH-trend direction. In a sustained one-way
move the counter-trend ladder fills repeatedly and never reverts — classic market-maker
death spiral. Memory `project_fish_engine_live_test.md` 12K loss was partially this.

## Direction-Flip Logic

Signal flips BULL → BEAR on instrument X:
1. Cancel all unfilled BULL buy tiers.
2. Let existing BULL inventory exit via its sell ladder + stop. Do NOT force-flip.
3. After cooldown (30min default) → if signal still BEAR → start BEAR ladder.
4. This prevents whipsaw on noisy signal flips.

## Execution Order (Batches)

### Batch 1: Config + state schema + tier math (pure logic)
- `portfolio/grid_fisher_config.py`
- `portfolio/grid_tiers.py`
- `tests/test_grid_tiers.py`
- Update `data/fin_fish_config.py` PREFERRED_INSTRUMENTS with oil entries
**Commit:** `feat(grid-fisher): config, tier math, oil warrants`

### Batch 2: State machine + atomic I/O (no Avanza side effects yet)
- `portfolio/grid_fisher.py` — state load/save/transitions only, no order placement
- `tests/test_grid_fisher_state.py`
**Commit:** `feat(grid-fisher): state machine and persistence`

### Batch 3: Order placement + fill detection + rotation
- `portfolio/grid_fisher.py` — wire avanza_session order calls
- `tests/test_grid_fisher_reconcile.py`
**Commit:** `feat(grid-fisher): order lifecycle and rotation`

### Batch 4: EOD sweep + integration + dashboard
- `portfolio/grid_fisher.py` — EOD logic
- `data/metals_loop.py` — wire `grid.tick()` into cycle
- `dashboard/app.py` — `/api/grid-fisher` endpoint
**Commit:** `feat(grid-fisher): loop integration, eod sweep, dashboard`

### Batch 5: Enable + smoke test + docs
- Confirm `GRID_FISHER_ENABLED = True` in production config
- Add `docs/GRID_FISHER.md` with operator runbook
- Update `CLAUDE.md` Architecture section to mention grid fisher
**Commit:** `feat(grid-fisher): docs and enable`

## What Could Break (and mitigation)

- **Avanza order rate limit:** placing 3 instruments × 3 tiers = 9 orders per startup,
  plus rotation orders on each fill. Mitigation: spread placement over 5s, cap at 10
  orders/minute, reuse `avanza_orders.py` retry logic.
- **Order ID collision with swing trader:** swing trader places its own orders on the
  same orderbook. Mitigation: grid_fisher skips instruments where swing trader holds a
  position; check via `metals_swing_state.positions`.
- **State file race vs swing trader:** they write to separate files but read shared
  Avanza state. Mitigation: each reads its own snapshot per tick.
- **Knockout cliff:** XAG dumps through tier 3 AND through stop → full loss across all
  tiers. Mitigation: knockout proximity check skips tiers within 8% of barrier.
- **EOD missed:** if Avanza closing time differs from cached value (DST flip), we hold
  overnight. Mitigation: pull `todayClosingTime` from Avanza API every cycle.
- **Loop crash with open orders:** if metals_loop dies, orders stay live. Mitigation:
  reconcile-on-startup; cancel unfilled tiers on restart, leave inventory + sells alone.

## Verification Plan

- Unit tests for tier math, state transitions, reconcile diff under all corner cases.
- Smoke test: after merge, run with `GRID_FISHER_DRY_PROBE=1` env var which logs
  intended actions without placing; observe 1 cycle in live metals_loop.
- After merge: restart `PF-MetalsLoop` task, tail `data/metals_loop_out.txt` for
  `GridFisher init:` line, verify catalog count == 3, no exceptions in first 5 cycles.

## Deferred (backlog)

- BULL/BEAR rotation between instruments based on relative-strength signal
- Probability-weighted tier sizing (more SEK on higher-p_fill tiers)
- Cross-instrument inventory hedging
- Multi-leg laddered stops (vs single position stop)
- Backtest harness against historical tape
