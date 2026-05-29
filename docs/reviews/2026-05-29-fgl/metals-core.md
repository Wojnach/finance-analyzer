# FGL Adversarial Review — metals-core (2026-05-29)

Scope: data/metals_loop.py, metals_swing_trader.py, metals_execution_engine.py,
metals_risk.py, metals_shared.py, metals_llm.py, metals_signal_tracker.py,
metals_avanza_helpers.py, metals_swing_config.py, fish_engine.py,
silver_monitor.py; portfolio/grid_fisher.py, grid_tiers.py,
grid_fisher_config.py, metals_ladder.py, metals_orderbook.py, fin_snipe.py,
fin_fish.py, iskbets.py, oil_grid_signal.py.

Reviewed at HEAD 745dc577.

## Counts
- P0: 0
- P1: 4
- P2: 5
- P3: 3

The stop-loss endpoint, cert-direction selection, global/per-instrument cap
enforcement, EOD-flat double-sell guard, and swing/grid ob_id overlap
(FISHING_OB_IDS) all check out — no P0 found. The grid-active instruments are
all constant-leverage BULL/BEAR X5 certs with barrier=0, so the absence of a
knockout-proximity guard in the grid rotate-stop path is not currently a
live knockout risk (it would become one if a barrier'd MINI is ever added to
GRID_ACTIVE_INSTRUMENTS).

---

## P1 — incorrect math / race / missing guard

portfolio/grid_fisher.py:1538-1594: P1: rotate_on_buy_fill cancels the existing stop (line 1538-1543) BEFORE placing the replacement (1546-1568). If the cancel succeeds but place_stop_loss fails, the code sets stop_needs_rearm=True and KEEPS inst.stop_loss_id pointing at the now-cancelled stop (1576) — the position is naked until the next tick's re-arm block (~60s window). The critical log + atomic_append to critical_errors.jsonl document it but don't close the gap. → Place the new stop FIRST, confirm SUCCESS, then cancel the old one (place-before-cancel). If Avanza rejects a second concurrent stop on reserved volume, fall back to cancel→place but null out stop_loss_id immediately on cancel so state never claims protection that doesn't exist.

portfolio/grid_fisher.py:1939-1973: P1: eod_market_flat cancels the stop-loss (1940-1948, sets stop_loss_id=None) and the armed sell tiers (1932-1938) BEFORE attempting the market-flat sell. If place_sell_order returns None or a non-SUCCESS status (1966-1984), the code `continue`s, leaving the inventory with NO stop, NO sell ladder, and no EOD sell — fully unprotected until a future tick retries. During the close auction (illiquid) this is exactly when fills/rejects are most likely. → Place the aggressive EOD sell first and only cancel the stop/sells after the EOD sell is confirmed SUCCESS; on failure, leave the existing stop in place.

data/metals_swing_config.py:323 (consumed at metals_swing_trader.py:3073): P1: EOD_EXIT_MINUTES_BEFORE = 0 makes the swing-trader EOD-flat rule `minutes_to_close <= 0` — it can only fire AT or AFTER close, i.e. it never executes a sell inside trading hours. The config comment itself says "REVERT to 25 after current position closes." Combined with MAX_HOLD_HOURS=24, a swing position can be carried overnight with only HARD_STOP/TRAILING managing it — contradicting the "does NOT want to hold warrants for a full day" user preference and the subsystem's EOD-flat contract. → Restore EOD_EXIT_MINUTES_BEFORE to a positive value (25 per the DST-gap analysis in the same file) or wire dynamic todayClosingTime; at minimum confirm with the operator that overnight holds are intended.

data/metals_swing_trader.py:2447-2451 / metals_execution_engine.py:38,53: P1: market close is hardcoded to 21:55 CET in two live code paths (swing `close_cet = 21.0 + 55/60`; execution engine MARKET_CLOSE_CET / hours_to_metals_close). .claude/rules/metals-avanza.md explicitly says "Check API for todayClosingTime — do NOT hardcode 21:55. Varies with DST." During the Mar/Oct DST-gap weeks the real Avanza commodity close shifts to ~21:00 CET, so EOD logic anchored to 21:55 is 55 min late — the documented bleed-overnight trap. → Replace with a get_session_close_cet() lookup (the swing config comment already flags this TODO); until then the EOD buffer must be wide enough to cover the gap.

## P2 — robustness

portfolio/grid_fisher.py:1495-1551 + grid_tiers.py:208-223: P2: build_exit_levels / the rotate stop compute stop_price purely as fill_price*(1-stop_pct) with NO knockout-proximity guard, unlike build_buy_ladder which runs _tier_skip_for_knockout. Safe today only because every GRID_ACTIVE_INSTRUMENTS entry is a barrier=0 cert. If a barrier'd MINI is ever added to the grid map, the rotate stop (and the EOD-flat aggressive sell at bid*0.99) could sit past the barrier. → Add a barrier-aware clamp/skip in build_exit_levels mirroring the buy-ladder guard, gated on barrier metadata being present.

data/metals_swing_trader.py:2714-2786 (_set_stop_loss): P2: the hardware stop is computed as warrant-price * (1 - sl_warrant_pct/100) with no check that the implied trigger stays the documented ≥3% away from current bid (the legacy metals_loop path at 2456-2463 DOES enforce a 3% min distance; the swing path does not). For an orphan ingest anchored to a stale bid, or a MINI near its barrier, the trigger could land closer than intended. → Add the same "trigger must be ≥3% below current bid" guard before place_stop_loss, and a barrier-distance check for MINI api_type positions.

portfolio/grid_fisher.py:1677-1682: P2: when reconcile reports multiple filled_buys in a single tick, rotate_on_buy_fill is invoked once per fill; each call cancels-then-replaces the full-inventory stop. Two sequential cancel/place round-trips per tick widen the naked window described in the P1 above and double the Avanza order traffic. → Batch rotations: place per-fill sells individually but recompute/replace the inventory stop ONCE after all fills in the tick are processed.

data/metals_swing_trader.py:3210 / 3792 (metals_loop): P2: SELL exits place a NORMAL limit order at current_bid (swing) / bid (loop emergency). At a wide-spread or fast-moving moment a limit exactly at bid may not fill, and the swing path then enters SELL_FAILED_COOLDOWN (300s) — a stuck losing position waits 5 min before retry while the stop is the only backstop. → Consider bid*0.99 (aggressive limit, as the grid EOD path already does) for genuine exits, or shorten the failed-sell cooldown for HARD_STOP-class reasons.

data/metals_swing_trader.py:596-633 (_send_telegram): P2: config.json is re-read on every Telegram send (mute_all check) via load_json with no caching; under a tight exit cascade this is repeated disk I/O on the hot path. Minor, but the file is a symlink to an external location. → Cache mute_all with a short TTL.

## P3 — nits

data/metals_swing_config.py:45: P3: DRY_RUN = False is committed in source for a live-money subsystem. Intentional, but a single accidental revert flips the whole swing trader live silently. → Source the live/dry flag from config.json so it can't be toggled by an unrelated code edit.

portfolio/grid_fisher.py:1800-1806: P3: the ADX trend filter branch is a no-op (`pass`) with a comment explaining it accepts the signal at face value — dead code that reads as if it gates something. → Either remove the branch or implement the intended with-trend check.

data/metals_swing_trader.py:38 / metals_avanza_helpers.py: P3: two parallel Avanza helper layers exist (portfolio.avanza_control used by swing, metals_avanza_helpers used by metals_loop) with near-duplicate place_order/place_stop_loss. Drift risk between them (e.g. the courtage-threshold warning, lock op names). → Converge on one canonical helper module.
