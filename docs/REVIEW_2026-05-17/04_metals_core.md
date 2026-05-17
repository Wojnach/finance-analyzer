# Adversarial Review — Metals Core Subsystem
**Date:** 2026-05-17
**Scope:** metals/oil/crypto swing loops, fishing intraday system, grid market-maker
**Reviewed files (29 .py files, ~28K LOC):**
metals_loop, metals_swing_trader, metals_swing_config, metals_signal_tracker, metals_risk, metals_shared, metals_execution_engine, metals_avanza_helpers, metals_history_fetch, metals_llm, metals_warrant_refresh, oil_loop, oil_swing_trader, oil_swing_config, oil_warrant_refresh, crypto_loop, crypto_swing_trader, crypto_swing_config, crypto_warrant_refresh, fish_engine, fish_monitor, exit_optimizer, price_targets, fin_snipe, fin_snipe_manager, fin_fish, grid_fisher, oil_grid_signal.

Issues reported with confidence ≥ 80. Findings grouped by severity.

---

## CRITICAL (confidence 90-100)

### C1. Hardware trailing stop call passes invalid kwargs — every trailing stop placement silently fails
**Confidence: 98**
**File:** `Q:/finance-analyzer/data/metals_loop.py:4787-4805`

```python
result = place_stop_loss(
    page, ACCOUNT_ID, ob_id_str,
    trigger_price=HARDWARE_TRAILING_PCT,
    sell_price=0,
    volume=vol,
    trigger_type="FOLLOW_DOWNWARDS",     # <-- not accepted
    value_type="PERCENTAGE",             # <-- not accepted
    valid_days=HARDWARE_TRAILING_VALID_DAYS,
)
if result.get("status") == "SUCCESS":    # <-- raises AttributeError
    hw_stop_id = result.get("stoplossOrderId", "?")
```

`place_stop_loss` is imported from `portfolio.avanza_control` (`metals_loop.py:341-349`); that function's signature (`portfolio/avanza_control.py:137-156`) is:

```python
def place_stop_loss(page, account_id, ob_id, trigger_price, sell_price,
                    volume, valid_days: int = 8):
```

It does NOT accept `trigger_type` or `value_type`. The call at line 4787 will raise `TypeError: place_stop_loss() got an unexpected keyword argument 'trigger_type'`, which is caught silently by the bare `except Exception as e:` at line 4808. The wider `result.get(...)` on the tuple `(success, stop_id)` returned by the helper would also fail.

`HARDWARE_TRAILING_ENABLED = True` (line 446) — so this code path runs on every BUY fill in the live loop and has **never** successfully placed a hardware trailing stop. Every swing-buy position relying on the hardware trailing-stop safety net has been running naked at the broker level. The data/adversarial_review_out.txt entry P1-12 already documented this issue but the bug is still present.

**Fix:** either (a) implement a `FOLLOW_DOWNWARDS`/`PERCENTAGE` variant of `place_stop_loss` in `metals_avanza_helpers.py` and route the call through it, or (b) replace this branch with a fixed-trigger stop computed from `cur_bid * (1 - HARDWARE_TRAILING_PCT/100)`, called with the actual `(page, account_id, ob_id, trigger_price, sell_price, volume, valid_days)` signature.

---

## IMPORTANT (confidence 80-89)

### I1. `fin_snipe_manager` places hard stops too tight on leveraged certs — 5% cert stop = 1% underlying on 5x
**Confidence: 88**
**File:** `Q:/finance-analyzer/portfolio/fin_snipe_manager.py:61, 529-563`

```python
HARD_STOP_CERT_PCT = 0.05            # 5% from entry price (cert side)
HARD_STOP_SELL_BUFFER_PCT = 0.01

trigger_price = _round_order_price(position_avg * (1.0 - HARD_STOP_CERT_PCT))
sell_price = _round_order_price(trigger_price * (1.0 - HARD_STOP_SELL_BUFFER_PCT))
```

On a 5x cert this stop fires after just 1% of underlying movement; on a typical 1-1.5% intraday silver wick the stop is **guaranteed to fire**. Memory `feedback_mini_stoploss.md` / grudges explicitly state: "5x certs need -15%+ stops, not -8%, to survive intraday wicks". The metals_swing_trader path (`_set_stop_loss`) was widened to `SL_BASE_UNDERLYING_PCT(6) × leverage(5) = 30%` cert stop precisely to avoid this wick-out — fin_snipe_manager still uses the legacy tight stop.

The hysteresis at line 545 only skips placement if the new stop is too close to current bid; it does not adjust the **width** of the stop on leveraged instruments.

**Fix:** Scale `HARD_STOP_CERT_PCT` by the instrument leverage (read from `snapshot["leverage"]`), e.g. `target_cert_stop_pct = MAX(0.15, BASE_UNDERLYING_PCT × leverage)`.

### I2. `fin_snipe_manager` stop plan ignores MINI barrier proximity
**Confidence: 82**
**File:** `Q:/finance-analyzer/portfolio/fin_snipe_manager.py:529-563, 606`

`_compute_stop_plan` produces a stop at `position_avg × 0.95` regardless of how close that lands to the warrant's `barrier_level`. The barrier metadata IS loaded (line 606 — `_summarize_market` records `barrier_level`), but the stop plan never consults it.

Grudges rule: "Never place stop-loss within 3% of MINI barrier price". For MINI L SILVER AVA series (barriers $32-$75), a 5% cert stop on a barrier-proximate position is effectively a "sell at zero" trigger — when underlying breaches barrier the cert collapses past the stop limit price and the hardware stop fires into no-bid.

**Fix:** Inside `_compute_stop_plan`, project the implied underlying price at the cert trigger (`trigger_und = current_und × (1 - cert_drop / leverage)`), then refuse placement (or widen stop) if `trigger_und < barrier × 1.03`.

### I3. `crypto_swing_trader._place_sell` LIVE path never updates `consecutive_losses` — loss cooldown escalation broken in live mode
**Confidence: 90**
**File:** `Q:/finance-analyzer/data/crypto_swing_trader.py:578-626`

```python
def _place_sell(self, pos_id, pos, current_underlying, current_warrant_bid, reason):
    if cfg.DRY_RUN or self.executor is None:
        ...
        if warrant_pnl_pct < 0:
            self.state["consecutive_losses"] = ... + 1
        else:
            self.state["consecutive_losses"] = 0
        return ...

    try:
        res = self.executor("sell", pos=pos, price=current_warrant_bid, reason=reason)
        if res and res.get("ok"):
            _log_trade(...)
            self.state["positions"].pop(pos_id, None)
            return {"executed": True, "dry_run": False, "result": res}
        # ^^^ no consecutive_losses update here
```

The LIVE branch (lines 615-626) removes the position and logs the trade but **never** computes P&L or updates `state["consecutive_losses"]`. The loss-escalation cooldown logic in `_cooldown_cleared` (lines 488-490) reads this counter, so in production:
1. Trader takes losing trades repeatedly without any cooldown extension.
2. `LOSS_ESCALATION = {0:1, 1:2, 2:4, 3:8}` never engages.
3. After 3 losses, no "8× cooldown" backoff applies — the very behaviour `LOSS_ESCALATION` was designed to prevent.

Crypto is `DRY_RUN=True` today so the bug is dormant, but the comment on the trader (line 19) and config explicitly mark this as ready to be flipped live; the moment `DRY_RUN=False` ships, loss escalation is silently disabled.

**Identical bug in `Q:/finance-analyzer/data/oil_swing_trader.py:575-620`** (verbatim parallel structure, same DRY_RUN-only update path).

**Fix:** Move the `warrant_pnl_pct` calculation and `consecutive_losses` update out of the DRY_RUN branch into a code path that runs after `self.state["positions"].pop(pos_id, None)` on both DRY_RUN and live success paths.

### I4. `fish_monitor.py` hardcodes EXIT_TIME at 21:00 — fires 55 minutes BEFORE Avanza warrant close
**Confidence: 85**
**File:** `Q:/finance-analyzer/data/fish_monitor.py:18, 163-165`

```python
EXIT_TIME = datetime.time(21, 0)
...
if now.time() >= EXIT_TIME:
    log("21:00 REACHED — position should be closed manually. Stopping monitor.")
    break
```

Avanza commodity warrants trade 08:15–21:55 CET (per `metals_shared.is_market_hours`, CLAUDE.md memory, and `.claude/rules/metals-avanza.md`). The monitor abandons positions at 21:00 — 55 minutes before close, surrendering nearly an hour of trailing-stop coverage during the most volatile US-close window. The log message even acknowledges it: "position should be closed manually" — i.e. the monitor stops protecting a still-open position.

Also: `EXIT_TIME` uses naive `datetime.time` with no timezone, so on a host whose system clock is not CET (e.g. UTC servers, WSL with default UTC) the exit fires at the wrong local time entirely.

**Fix:** Use `datetime.time(21, 55)` with explicit `Europe/Stockholm` zoneinfo comparison, mirroring `grid_fisher.minutes_until_eod`.

### I5. Grid fisher EOD market-flat assumes Avanza cancel succeeds — risks double-sell
**Confidence: 81**
**File:** `Q:/finance-analyzer/portfolio/grid_fisher.py:1553-1570`

```python
for tier in list(inst.sell_ladder):
    if tier.status == ORDER_ARMED and tier.order_id:
        self._safe_session_call(
            self.session.cancel_order, tier.order_id, default=None,
        )
        tier.status = ORDER_CANCELLED      # set unconditionally
# stop cancel — also unconditional:
if inst.stop_loss_id:
    cancel_stop_fn = getattr(self.session, "cancel_stop_loss", None)
    if cancel_stop_fn is not None:
        self._safe_session_call(cancel_stop_fn, inst.stop_loss_id, default=None)
    inst.stop_loss_id = None
```

The code marks each sell tier `CANCELLED` and clears `stop_loss_id` without checking whether Avanza actually cancelled the order. `_safe_session_call` returns `None` on timeout/exception/`SUCCESS`-with-no-result — there is no way to distinguish a successful cancel from a silent failure. The flow then proceeds to place a fresh full-volume EOD market sell (line 1583). If even one prior cancel was a no-op, the position now has the original sell + the EOD aggressive sell racing to fill — eventually short-selling the position once both fill (the very pattern the P0-9 `eod_sell_order_id` guard at line 1545 was added to prevent for the EOD sell itself).

**Fix:** Inspect the cancel result; only mark `ORDER_CANCELLED` / clear `stop_loss_id` on a `SUCCESS` response. On rejection, log + skip the EOD market sell for this instrument (consistent with the lines 1588-1606 retry-next-tick pattern).

### I6. `fin_snipe_manager` action executor wraps every order in bare `except Exception` — silent broker rejections
**Confidence: 80**
**File:** `Q:/finance-analyzer/portfolio/fin_snipe_manager.py:1380-1424`

```python
try:
    ...
    elif order_type == "stop_loss":
        ok, result = place_stop_loss_no_page(...)
    else:
        ok, result = place_order_no_page(...)
    results.append({"ok": ok, "result": result, **action})
except Exception as exc:
    logger.error("Action execution crashed ... %s", exc, exc_info=True)
    results.append({"ok": False, "result": {"error": str(exc)}, ...})
```

This catches the same TypeError class that masked the C1 hardware-stop bug for weeks. If any of `place_stop_loss_no_page`, `place_order_no_page`, `delete_stop_loss_no_page`, `delete_order_no_page` ever ships with a signature mismatch — or a transient `KeyError` on `result` — the failure becomes a silent `ok=False` log entry. The exit/stop pipeline keeps running with stale state, and `apply_execution_results_to_state` (line 1469-1488) increments the cancel-fail counter quietly, eventually marking the order "dead" without it actually being dead at Avanza.

Combined with C1, this exception class is the dominant silent-failure mode in the metals subsystem. The bare catch is too wide.

**Fix:** Narrow to specific recoverable exceptions (network, JSON decode, Avanza API errors) and let `TypeError` / `AttributeError` / `KeyError` propagate — those indicate code bugs that need to crash the cycle visibly.

### I7. `oil_loop.run_loop` and `crypto_loop.run_loop` declare `-> int` but fall off end implicitly returning `None`
**Confidence: 82**
**File:** `Q:/finance-analyzer/data/oil_loop.py:297-363`, `Q:/finance-analyzer/data/crypto_loop.py:284-352`

```python
def run_loop(notify: Any = None) -> int:
    """Forever loop. Returns an int status code so main() can propagate it.
    Returns:
        0  on graceful shutdown (SIGINT/SIGTERM)
        EXIT_LOCK_CONFLICT (11) if another instance holds the singleton lock
    """
    lock = acquire_singleton_lock()
    if lock is None:
        ...
        return EXIT_LOCK_CONFLICT
    ...
    try:
        ...
        while not stop["flag"]:
            ...
    finally:
        release_singleton_lock(lock)
        logger.info("oil_loop exited cleanly")
    # <-- no return statement: implicitly returns None
```

`main()` calls `return run_loop(notify=notify)` (line 432) and `sys.exit(main())` (line 439). `sys.exit(None)` exits 0, but the contract was that graceful shutdown returns explicit 0. Some shells / supervisors treat `None`-coerced exit differently from the documented contract, and the `metals-loop.bat` wrapper convention is to restart on non-11/0 exit — handing it a non-int return value via the docstring lie is asking for downstream regressions.

**Fix:** Add `return 0` after the `finally` block (or inside the `while` loop's exit path).

---

## P2 / Lower-confidence findings (not separately scored)

The following were observed but either fall below the 80 confidence threshold or describe acceptable risk per documented user preferences. Listed for completeness:

- **fish_monitor.py module-level state** (`high_water_mark`, `current_sl_trigger`) is not persisted; any restart loses the trailing high-water mark and resets `current_sl_trigger` to the hardcoded 24.67. Operational script, single-use per session.
- **Hardcoded warrant identifiers** in `fish_monitor.py:15, 17` (`CERT_ID="2304634"`, `ENTRY_PRICE=29.02`) — by design for a one-off operational monitor, but the file is checked into git; reusing it on the next position requires editing constants in-place.
- **MINI barrier check absent in `metals_swing_trader._set_stop_loss`** — same class as I2 but on a wider stop (-30% cert ≈ -6% underlying on 5x). User memory explicitly accepts 10-20% knockout risk, so this is closer to an accepted tradeoff than a bug.
- **`fish_engine.tick` hardcodes 21:55 CET session end** (lines 216-217) — same DST-gap-week risk as the metals_swing_trader's `close_cet = 21.0 + 55/60` at lines 2448, 2796. metals_swing_config explicitly documents and accepts the issue (lines 303-321).
- **Telegram message on swing buy reports the deprecated module constants** `TAKE_PROFIT_WARRANT_PCT` / `STOP_LOSS_WARRANT_PCT` (metals_swing_trader.py:2697) instead of the per-position `pos["tp_warrant_pct"]` / `pos["sl_warrant_pct"]` actually used by `_evaluate_exit` and `_set_stop_loss`. Cosmetic but misleading on 1x trackers or non-5x certs.

---

## Patterns explicitly verified clean

- **Stop-loss endpoint contract** (`/_api/trading/stoploss/new`) — `metals_avanza_helpers.place_stop_loss` (line 386), `fish_monitor.create_sl` (line 78), the JS in `metals_loop.py:5326`, and `grid_fisher.rotate_on_buy_fill` (via `session.place_stop_loss`) all use the correct endpoint. The legacy regular-order-API path documented in grudges has not regressed.
- **CLAUDECODE env cleanup** before spawning `claude` subprocesses — `metals_loop.py:6763` properly pops it; `agent_invocation`, `claude_gate`, `multi_agent_layer2`, `analyze`, and helpers also pop it (grep verified across `.py` files).
- **Atomic state I/O** — `grid_fisher.save_state`, `metals_swing_trader._save_state`, `crypto_swing_trader._save_state`, `oil_swing_trader._save_state`, `metals_risk` state writes all use `portfolio.file_utils.atomic_write_json` / `atomic_append_jsonl`.
- **No float `==` money math** — searched; comparisons in `metals_swing_trader._reconcile_orders` and `grid_fisher` use `_price_matches` helpers with tolerance, not `==`.
- **Singleton lock with stale-pid detection** in oil_loop, crypto_loop, metals_loop — `O_CREAT|O_EXCL` atomic creation + `_pid_alive` check is correct.
- **`avanza_order_lock` cross-process serialization** — wraps every `place_order`, `place_stop_loss`, `delete_order`, `delete_stop_loss` in `metals_avanza_helpers`. No order placement bypasses the lock.
- **Crypto/oil loop DRY_RUN default** — `crypto_swing_config.DRY_RUN = True` (line 65), `oil_swing_config.DRY_RUN = True` (line 104). Both correctly ship inert.
- **Grid fisher live-cash gate** (`_effective_global_cap`) fails closed when buying-power fetch fails and no fresh-enough cache exists, preventing the 2026-05-13 OLJAB-on-empty-cash incident from recurring.
- **`exit_optimizer.compute_exit_plan`** — checked `_TRADING_MINUTES["warrant"] = 820` matches 08:15-21:55 (13.67h × 60 ≈ 820), so the volatility annualization is consistent with the documented warrant session.
