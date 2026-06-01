# FGL Review — metals-core

Adversarial read-only review of the metals subsystem (real-money Avanza warrant/certificate
trading). Scope = clean-PR worktree `fgl/metals-core`:
`data/metals_loop.py`, `data/metals_swing_trader.py`, `portfolio/grid_fisher.py`,
`data/metals_risk.py`, `portfolio/oil_grid_signal.py`, `data/fish_engine.py`.
Cross-file context read from the live repo: `portfolio/avanza_session.py`,
`portfolio/avanza_control.py`, `data/metals_avanza_helpers.py`,
`portfolio/grid_fisher_config.py`, `portfolio/grid_tiers.py`.

Endpoint invariant verified GOOD on every stop-loss path:
- grid_fisher → `avanza_session.place_stop_loss` → `POST /_api/trading/stoploss/new` (avanza_session.py:806).
- cascade / swing → `avanza_control.place_stop_loss` → `metals_avanza_helpers.place_stop_loss`
  → `POST /_api/trading/stoploss/new` (metals_avanza_helpers.py:386).
No stop-loss is placed via the regular order API. The Mar-3 incident invariant holds.

---

## Critical (P0–P1)

- **[P1] data/metals_loop.py:5148-5168 (`update_trailing_stops`) + 2447-2519 (`_rebuild_stop_orders_for`)** —
  Naked-position window on stop-rebuild. Both functions cancel the existing
  cascade stop-loss orders FIRST (`_cancel_stop_orders`, lines 5152 / 2450), then place
  new ones. The re-place loop only commits state `if orders:` (2512) and tolerates
  per-level failures by logging and continuing. Causal chain: signal/price ratchets the
  stop up → old broker stops cancelled → every `place_stop_loss` for the new level is
  rejected (Avanza halt, transient 5xx, courtage-min reject, CSRF drift) → `orders` ends
  empty → no broker-side protection exists, yet the position is still held. A 5x cert with
  no stop through a downward wick is exactly the loss this module is built to prevent.
  Unlike the fish-sell path (which snapshots + re-arms on failure), the trailing-stop
  rebuild has no rollback. **Fix:** place the new stops BEFORE cancelling the old (or
  snapshot the old stop IDs and re-arm them if every new placement fails), mirroring
  `_rearm_stops_after_failed_sell`. At minimum, emit a `critical_errors.jsonl` entry +
  Telegram when a rebuild cancels old stops but commits zero new ones.

- **[P1] data/metals_loop.py:2936-3055 (`_fish_engine_execute_buy`)** — Fish-engine BUY
  opens a leveraged (5x) cert position and places NO broker-side stop-loss. `confirm_entry`
  (3048) only mutates in-memory engine state; protection is purely the software per-tick
  exit in `fish_engine._evaluate_exit`. Causal chain: fish BUY fills → metals_loop process
  dies / hangs / GPU-locks / loses Avanza session before the next tick → position sits with
  no hardware stop until the loop restarts and `_reconcile_fish_engine_position` runs. A
  5x silver cert gapping down during that gap is an uncapped loss. The swing trader
  (`_set_stop_loss`, metals_swing_trader.py:2776) and the trade-queue buy
  (place_stop_loss_orders / HW trailing) both place broker stops on entry; the fish path is
  the lone entry that does not. **Fix:** place a hardware stop (or HW trailing stop) right
  after a successful fish BUY, exactly as `_handle_buy_fill` does (metals_loop.py:4784-4827).

---

## Important (P2)

- **[P2] portfolio/grid_fisher.py:1664-1676 (`tick`) via avanza_session.py:653-669
  (`get_open_orders`)** — The degraded-fetch guard is defeatable on one specific failure
  mode. `tick` treats `open_orders_raw is None` as "degraded → skip", relying on
  `_safe_session_call` returning `None` on read failure. But `get_open_orders` swallows
  its own `RuntimeError` (non-401 API errors, plus the deals-and-orders fallback also
  failing) and returns `[]`, not raising. So a 5xx/shape-error on the open-orders endpoint
  yields `open_orders_raw == []` (not `None`) → guard passes → `open_order_ids` is empty →
  `reconcile_against_live` (690) sees every ARMED buy as "missing from live". With inventory
  unchanged (`delta == 0`) each resting buy is marked CANCELLED (738) and fed to Gate B
  (`_maybe_arm_rapid_cancel_cooldown`), which can arm a spurious 6 h cooldown
  (GRID_RAPID_CANCEL_COOLDOWN_S) and churn cancel/replace. Note `get_positions` raises on
  the same class of failure (api_get → RuntimeError, avanza_session.py:265) → `None` →
  guard catches; the asymmetry between the two readers is the bug. **Fix:** have
  `get_open_orders` raise (or return a sentinel) on read failure, OR have grid_fisher treat
  an empty open-orders list as suspicious when `inventory_units > 0` / armed tiers exist and
  skip reconcile that tick. Do not let a swallowed `[]` mean "everything cancelled".

- **[P2] data/metals_loop.py:5042-5060 (`check_stop_order_fills`)** — Stop-loss fill
  detection queries the regular-order endpoint
  `/_api/trading-critical/rest/order/{accountId}/{orderId}` for IDs that were created via
  the stop-loss endpoint (`/_api/trading/stoploss/new`). Stop-loss orders are a separate
  surface; this GET can return non-200 for a live stop ID, so `order["status"]` never
  flips to `filled` and the cascade keeps believing the stop is resting. Causal chain: a
  stop actually triggers and sells at the broker → fill never detected here → local
  `pos["units"]` not decremented → a later SELL/EOD path sizes against stale units → risk
  of short-sell-reject or double-sell. Mitigated (per the in-code note) by holdings
  reconciliation elsewhere, which is why this is P2 not P1. **Fix:** query the stop-loss
  status endpoint (`/_api/trading/stoploss/...`) for stop IDs, or detect stop fills purely
  from position-volume reconciliation rather than per-order GET.

- **[P2] data/metals_swing_trader.py:2796 (`_check_exits`) and 2450 (`_can_enter`);
   data/fish_engine.py:216; portfolio/grid_fisher.py:287** — Market close is hardcoded to
  21:55 CET rather than read from the API `todayClosingTime`, which the project rule
  (`.claude/rules/metals-avanza.md`) explicitly forbids: "Check API for `todayClosingTime`
  — do NOT hardcode 21:55. Varies with DST." `_cet_hour()` is DST-aware for the current
  wall-clock hour, and grid_fisher's `minutes_until_eod` uses zoneinfo for the *offset*, but
  the 21:55 close instant itself is a literal in all four places. Causal chain: any Avanza
  schedule change (half-day, holiday early close, DST-edge session) makes EOD-flat fire late
  or not at all → inventory carried past close against the user's explicit "no full-day
  hold" preference, or EOD market-sell submitted into a closed book. The exit optimizer
  already consults `session_calendar.get_session_info` (metals_swing_trader.py:2890-2891),
  so the authoritative close time is available in-process and is simply not used by the
  rule-based EOD branch. **Fix:** source the close instant from `get_session_info` /
  the Avanza `todayClosingTime` everywhere EOD is decided.

---

## Minor (P3)

- **[P3] portfolio/grid_fisher_config.py:50 (`GRID_STOP_PCT = 3.5`) vs
  `.claude/rules/metals-avanza.md`** — The grid market-maker's stop is a 3.5% *warrant*
  move; on a 5x cert that is ~0.7% underlying. The metals rule states "5x leverage
  certificates need -15%+ stops, not -8%, to survive intraday wicks." The grid is a
  fast-rotation MM with EOD-flat and its config comment justifies 3.5% as "outside the
  typical 15-minute volatility band," so this is a deliberate design divergence rather than
  a defect — but it directly contradicts the written rule and should be reconciled (either
  document the MM carve-out in the rule, or widen the stop). The swing trader, by contrast,
  correctly uses `STOP_LOSS_WARRANT_PCT = 30%` (≈ base × 5x), satisfying the rule.

- **[P3] portfolio/oil_grid_signal.py:112-148 (`compute_signal`) / 151-176
  (`get_cached_or_refresh`)** — The cache `ts` is stamped at compute time, not at kline-data
  time. There is no check that the latest Brent (BZ=F) bar is recent. Over a weekend / oil
  close, `fetch_klines` (yfinance fallback) can return stale 1h bars; `_signal_from_indicators`
  computes a confident LONG/SHORT from them, and the consumer sees a fresh `ts` < 300 s and
  treats it as live. Real-money exposure is bounded because grid_fisher's Gate A
  (`_is_quote_stale`, grid_fisher.py:1251) independently skips placement on a dead warrant
  orderbook, so the stale oil signal cannot by itself place an order. **Fix:** reject the
  signal (direction=None) when the last kline timestamp is older than a small multiple of
  the interval; surface data-age, not just compute-age.

- **[P3] data/metals_risk.py:171 & 197 (`simulate_warrant_risk`)** — The Itô drift
  correction `-0.5*vol_annual**2` is applied twice: once when building `drift` from
  `direction_prob` (171) and again inside the GBM log-return (197). This double-subtracts the
  variance drag, biasing simulated terminal prices slightly low (pessimistic `p_stop_hit`,
  conservative VaR). Analytics-only — `simulate_warrant_risk` feeds the risk summary, not an
  order path — so no money is mis-sized, but the MC numbers shown to the operator/Layer 2 are
  skewed. **Fix:** compute `drift = z * vol_annual` and let the single `-0.5σ²` in the
  log-return supply the convexity term.

- **[P3] data/fish_engine.py + data/metals_loop.py:3040 (fish BUY at `ask`)** — The fish
  entry crosses the spread (limit at ask) rather than resting a dip limit, which conflicts
  with the fishing rule "NEVER hit the ask. Place limits at computed dip levels." Defensible
  for a momentum/ORB breakout entry where immediacy matters, but it is the *fish* engine and
  the rule is phrased categorically. Worth an explicit carve-out comment or a marketable-limit
  cap to bound slippage.

---

## Summary

The stop-loss endpoint invariant (the highest-stakes rule) is satisfied on every path —
no stop is routed through the regular order API. Knockout/barrier proximity is correctly
guarded on grid buy-tier placement (`grid_tiers._tier_skip_for_knockout`,
GRID_KNOCKOUT_SAFETY_PCT = 8%), and the grid stop side cannot sit near a barrier given the
direction math. State I/O is atomic throughout (`file_utils.atomic_write_json` /
`atomic_append_jsonl`), the swing reconciler correctly distinguishes None (transient) from
{} (flat), and the fish-engine sell path has robust cancel-snapshot / re-arm / finally
rollback against naked windows.

Two P1 real-money exposures remain and should block merge until addressed:
1. The trailing-stop **rebuild** cancels old broker stops before placing new ones with no
   rollback, so an all-rejected re-place leaves a leveraged position naked
   (metals_loop.py:5148 / 2447).
2. The **fish-engine BUY places no broker stop**, so a process death after entry leaves a
   5x cert unprotected until restart (metals_loop.py:2936).

P2s worth fixing soon: the `get_open_orders → []` asymmetry that defeats grid_fisher's
degraded-fetch guard (can trigger false mass-cancellation + spurious 6 h cooldown); the
stop-fill check querying the wrong endpoint; and the hardcoded 21:55 EOD close that
violates the explicit `todayClosingTime` rule across swing/fish/grid.
