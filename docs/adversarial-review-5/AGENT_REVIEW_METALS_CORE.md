# Agent Review: metals-core — Round 5 (2026-04-11)

**Agent**: feature-dev:code-reviewer
**Files reviewed**: 20 (metals_loop, metals_swing_trader, metals_swing_config,
metals_shared, metals_risk, metals_execution_engine, metals_signal_tracker,
metals_warrant_refresh, metals_llm, metals_avanza_helpers, metals_cross_assets,
metals_orderbook, metals_precompute, metals_ladder, exit_optimizer, price_targets,
orb_predictor, fin_fish, fin_snipe, iskbets)
**Duration**: ~476s (largest subsystem, ~19K lines)

---

## Findings (6 total: 2 CRITICAL, 2 IMPORTANT, 2 P2)

### CRITICAL (Money-Losing)

**MC-C1** [HIGHEST PRIORITY] metals_swing_trader.py:1717 — _execute_sell doesn't cancel stop first
- SwingTrader places SELL without cancelling hardware trailing stop
- Avanza rejects: stop volume + sell volume > position size → "short.sell.not.allowed"
- Position stuck open, no exit, stop may fire at worse price later
- Legacy fish engine HAS this fix (cancel-before-sell with rollback)
- SwingTrader was built without inheriting the pattern
- **Same class of defect as the March 3 incident**
- Fix: Cancel stop_order_id BEFORE place_order(SELL), rollback if cancel fails

**MC-C2** metals_swing_trader.py:1547 — usdsek=10.85 hardcoded in exit optimizer
- A-MC-2 fix (fetch_usd_sek) applied to fin_snipe but NOT to SwingTrader
- 10% FX error flips exit decisions near threshold
- Another instance of the incomplete-fix pattern
- Fix: Use fetch_usd_sek() with 10.85 fallback

### IMPORTANT (Correctness)

**MC-I1** metals_loop.py:2227-2248 — Fish engine uses naive datetime.now() for CET
- No timezone → wrong in CEST (summer), 1-hour systematic offset
- Rest of codebase uses get_cet_time() from metals_shared.py
- Affects EOD exit timing and fishing strategy gating
- Fix: Use get_cet_time() or ZoneInfo("Europe/Stockholm")

**MC-I2** metals_loop.py:2547 + orb_predictor.py:32-33 — ORB window hardcoded for winter
- Trigger: now_utc.hour >= 10 (winter CET→UTC). In CEST: 10 UTC = 12:00 Stockholm
- ORB "morning session" becomes midday consolidation in summer
- Both trigger condition and MORNING_START_UTC/END_UTC need DST awareness
- Fix: Use Stockholm timezone for ORB window, not hardcoded UTC hours

### P2 (Code Quality)

**MC-I3** metals_loop.py:6051 — Raw open() for Claude agent log (still present)
- File handle leak if invoke_claude raises between open and close
- Fix: Wrap in context manager

**MC-I4** metals_signal_tracker.py:190,340,419,660 — Raw open() reads entire JSONL
- Reads full file on every backfill call (every 10 cycles)
- Fix: Use load_jsonl_tail() for bounded reads

---

## Prior-Round Status

| ID | Status |
|----|--------|
| C12 (raw open in log_portfolio_value) | **FIXED** — metals_risk.py uses atomic_append_jsonl |
| C14 (naked position on stop-loss failure) | FIXED in fish engine, **NEW INSTANCE** in SwingTrader (MC-C1) |
| H31 (POSITIONS dict without lock) | **LOWER RISK** — LLM thread doesn't touch POSITIONS directly |
| HARD_STOP=2% | **REASONABLE** — 2.0% underlying with 2.5% stop, matches risk tolerance |
| ORB in CEST | **CONFIRMED OPEN** — both trigger and predictor constants wrong in summer |

## Non-Issues Verified
- metals_shared.py DST handling: correct via ZoneInfo
- metals_avanza_helpers.py: correct stop-loss API endpoint
- metals_warrant_refresh.py: no deadlock, correct merge logic
- metals_execution_engine.py: advisory-only, no order placement
- fin_fish.py, fin_snipe.py, iskbets.py: advisory-only
