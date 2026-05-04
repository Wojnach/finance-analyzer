# Claude Review — metals-core

## P0 (money-losing or data-corrupting)

- `data/metals_loop.py:4793-4802` — Hardware trailing stop calls `place_stop_loss` with `trigger_type`/`value_type` kwargs that the function signature does not accept
  ```python
  result = place_stop_loss(
      page, ACCOUNT_ID, ob_id_str,
      trigger_price=HARDWARE_TRAILING_PCT,   # 5.0 (a %)
      sell_price=0,
      volume=vol,
      trigger_type="FOLLOW_DOWNWARDS",
      value_type="PERCENTAGE",
      valid_days=HARDWARE_TRAILING_VALID_DAYS,
  )
  ```
  `portfolio/avanza_control.py:137-156` shows `place_stop_loss` signature is `(page, account_id, ob_id, trigger_price, sell_price, volume, valid_days=8)` — no `trigger_type` or `value_type`. The extra kwargs are silently swallowed by the underlying chain or raise `TypeError`. Either way, what gets sent to Avanza is a standard MONETARY LESS_OR_EQUAL stop at price 5.0 SEK with `sell_price=0`. **The "hardware trailing stop" described in config `HARDWARE_TRAILING_ENABLED=True` has never actually worked.** Every new fill from the queue path is silently left without broker-side trailing protection. Confidence 98.

- `data/metals_loop.py:4907-4915` — `place_stop_loss_orders` skips proximity guard if `fetch_price` returns None
  ```python
  cur_price_data = fetch_price(page, pos["ob_id"], pos.get("api_type", "warrant"))
  cur_bid = (cur_price_data or {}).get("bid", 0)
  if cur_bid > 0:
      distance_pct = (cur_bid - stop_base) / cur_bid * 100
      if distance_pct < 3.0:
          log(f"  SKIP stop for {key}: ...")
          continue
  ```
  When `fetch_price` returns None (transient session failure), `cur_bid==0`, the proximity guard is skipped, and stop placement proceeds with the stale `stop_base` from state. If the warrant has since moved dramatically, the stale `pos["stop"]` could be above or within the current spread — instant fill at bad price. Gated by `STOP_ORDER_ENABLED=False` so partial mitigation. Confidence 80.

## P1 (high-confidence bugs)

- `data/metals_swing_trader.py:142, 2738` — `metals_swing_trader` imports `place_stop_loss` from `portfolio.avanza_control`, but unpacks the return as `(success, stop_id)` — wrong shape
  ```python
  from portfolio.avanza_control import (
      ...
      place_stop_loss,
  )
  ...
  success, stop_id = place_stop_loss(...)
  ```
  `portfolio/avanza_control.py:137-156` returns `(ok: bool, result: dict)`. `data/metals_avanza_helpers.py:330-408` returns `(success: bool, stop_id: str)`. The swing trader picked the wrong import — `stop_id` receives a dict, gets stored in `pos["stop_order_id"]`, and later `_delete_stop_loss(page, ACCOUNT_ID, stop_id)` is called with a dict argument. **Hardware stops on swing trader positions are never cancelled on exit.** Confidence 95.

- `data/metals_swing_trader.py:3151` — `_execute_sell` does not cancel the stop-loss BEFORE placing the sell
  ```python
  success, result = place_order(self.page, ACCOUNT_ID, pos["ob_id"], "SELL", current_bid, units)
  ```
  No stop cancellation precedes this. Avanza may reject with `short.sell.not.allowed` if reserved stop volume + sell volume exceeds position size. The fish engine explicitly calls `_ensure_stops_cancelled_before_sell` before every sell. SwingTrader cancels at line 3179 only on the success path — failure leaves stop live and retries the sell next tick, looping forever. Confidence 88.

- `data/metals_swing_trader.py:537-538` — `_cet_hour()` fallback uses fixed UTC+1 — wrong from late March to late October
  ```python
  except ImportError:
      now = datetime.datetime.now(datetime.UTC)
      return ((now.hour + 1) % 24) + now.minute / 60
  ```
  Stockholm is UTC+2 during summer DST. If `zoneinfo` is missing (Windows without `tzdata`), the 21:55 EOD guard shifts to 22:55 (allows entries after real close), the 08:15 open check shifts to 09:15 (blocks the first hour). Production uses Python 3.10+ so usually safe, but no defensive fallback to `tzdata` install check. Confidence 82.

- `data/metals_swing_trader.py:2426, 2758` — `close_cet = 21.0 + 55/60` hardcoded across DST gap weeks
  ```python
  close_cet = 21.0 + 55 / 60  # 21:55 CET
  ```
  Documented in config but never fixed. During the EU+US DST mismatch (~3 weeks/year), real close is 21:00 CET. EOD branch fires too late. Currently harmless because `EOD_EXIT_MINUTES_BEFORE = 0` disables the EOD-exit path. Re-enabling it would re-introduce the bug. Confidence 82.

## P2 (concerns / smells)

- `data/fish_engine.py:654-655` — bare `except Exception: pass` on layer2 vote tactic
  ```python
  try:
      l2_vote = self._vote_layer2(state)
      ...
  except Exception:
      pass
  ```
  Silent swallow on layer2 vote failure. If the journal file shape changes, the vote contributes nothing rather than degrading predictably. Documented fish engine bug history makes this a real risk.

- `data/metals_loop.py:4903-4904` — `pos["stop"]` used as raw warrant trigger price without verifying SEK vs USD
  ```python
  stop_base = pos["stop"]
  ```
  `pos["stop"]` is populated by Layer 2 agent decisions from `metals_positions_state.json`. If Layer 2 writes a stop in underlying USD, the placement function treats it as a warrant SEK trigger — instant rejection or fire. Gated by `STOP_ORDER_ENABLED=False`.

## Did NOT find

1. Wrong stop-loss endpoint — all live trading paths correctly use `/_api/trading/stoploss/new`.
2. Stops placed near MINI warrant barrier — `MIN_BARRIER_DISTANCE_PCT = 10%` enforced in swing config; barrier distance checked before selection.
3. Order size below 1000 SEK — Kelly sizing floors at `MIN_TRADE_SEK = 1000`; both paths enforce.
4. Integer share rounding errors — `int(alloc / ask_price)` at line 2088 is correct.
5. silver_monitor + metals_loop race — silver_monitor was merged into metals_loop in v10; only one process runs silver fast-tick.
6. Stop placed inside bid/ask spread — all placements compute trigger from underlying percentage off warrant price, not from spread.
