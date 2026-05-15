# Adversarial Review — 4 metals-core (second-reviewer / codex-substitute)

> Codex CLI quota was exhausted at start of session. This review is produced by a
> second Claude subagent with isolated context as a substitute second opinion.

## P0 — money-losing or data-corrupting (must fix)

- `portfolio/exit_optimizer.py:60-74` + `:303-340` + `:741-750` — **Position struct
  has no direction; warrant PnL formula assumes LONG only**

  ```python
  @dataclass(frozen=True)
  class Position:
      symbol: str
      qty: float
      entry_price_sek: float
      entry_underlying_usd: float
      entry_ts: datetime
      instrument_type: str = "warrant"
      leverage: float = 1.0
      ...
  ```

  ```python
  # _compute_pnl_sek, warrant branch (no financing_level)
  pct_move = (exit_price_usd - position.entry_underlying_usd) / position.entry_underlying_usd
  warrant_move = pct_move * position.leverage
  exit_warrant_sek = position.entry_price_sek * (1 + warrant_move)
  ```

  For a BEAR cert (e.g. `BEAR SILVER X5 AVA 12`, ob `2286417`, leverage=5.0),
  when the underlying rises, `pct_move > 0` so `warrant_move > 0` so the
  function claims the BEAR cert is gaining value — but bears go DOWN when
  the underlying goes UP. EV ranking is inverted, the optimizer recommends
  worst-case exits, and `compute_exit_plan_from_summary` (line 741) /
  `fin_snipe_manager.py:467-477` never pass `direction`. Also breaks the
  candidate filter at `:568` (`target <= market.price * 0.999: continue`)
  which prunes all winning candidates for SHORTs. Production paths
  (`fin_snipe_manager._summarize_market`, `reporting.py:684`) silently
  produce LONG-only plans for any BEAR cert in the warrant book.

- `portfolio/grid_fisher.py:1217-1256` — **Stop-loss cancel-then-place is not
  rolled back on cancel failure → orphaned stop + new stop both live**

  ```python
  if inst.stop_loss_id and not self._probe_only:
      cancel_fn = getattr(self.session, "cancel_stop_loss", None)
      if cancel_fn is not None:
          self._safe_session_call(
              cancel_fn, inst.stop_loss_id, default=None,
          )
  new_stop_id: Optional[str] = None
  if not self._probe_only and inst.inventory_units > 0:
      ...
      result = self._safe_session_call(
          self.session.place_stop_loss,
          inst.ob_id, stop_price, stop_sell_price,
          inst.inventory_units,
          default=None,
      )
      ...
      if result.get("status") == "SUCCESS":
          new_stop_id = str(result.get("stoplossOrderId") or "") or None
      ...
  inst.stop_loss_id = new_stop_id
  ```

  `_safe_session_call` swallows exceptions and returns `default=None` — the
  return value of the cancel is never inspected. If Avanza returned 4xx /
  network blip / "stop already cancelled" but the actual Avanza state
  still has the stop live (which has happened — see
  `data/metals_loop.py:4994-5008` fall-back handling), we blow away
  `inst.stop_loss_id` and write the NEW stop's id. The old stop is now
  orphaned on Avanza but covers `inventory_units`. Next dip both stops
  fire — first sells inventory, second goes short. This is exactly the
  duplicate-sell pattern the P0-9 fix on the EOD path (`:1538-1552`)
  was added to prevent, but the rotation path doesn't have that guard.

- `portfolio/grid_fisher_config.py:50` + `portfolio/grid_tiers.py:208-223` —
  **3.5% cert stop on 5x leveraged warrants = 0.7% underlying move, violates
  user rule "5x certs need -15%+ stops"**

  ```python
  GRID_STOP_PCT = 3.5
  ```

  ```python
  def build_exit_levels(fill_price: float, target_pct: float, stop_pct: float):
      ...
      sell_limit = _round_price(fill_price * (1 + target_pct / 100.0))
      stop_loss = _round_price(fill_price * (1 - stop_pct / 100.0))
      return sell_limit, stop_loss
  ```

  `GRID_ACTIVE_INSTRUMENTS` (`:140-144`) targets BULL/BEAR X5 silver, X8
  gold, X3 oil — all leveraged certs. Per `.claude/rules/metals-avanza.md`
  ("5x leverage certificates need -15%+ stops, not -8%, to survive
  intraday wicks") and `memory/feedback_mini_stoploss.md` ("Never place
  a stop-loss within 3% of bid for silver/leveraged warrants"), a 3.5%
  warrant stop on a 5x cert is exactly the pattern that gets wicked out.
  The comment on `:50` claims "Stop sits outside the typical 15-minute
  volatility band so noise does not fire it" — but XAG 1min ATR commonly
  exceeds 0.3% on the underlying = 1.5% on a 5x cert, so a 3.5% cert stop
  is barely 2× ATR. `metals_swing_config.py:235` correctly uses
  `STOP_LOSS_WARRANT_PCT = 30.0` for the same family of instruments —
  grid_fisher is an order of magnitude tighter.

- `portfolio/iskbets.py:304-356` — **Layer 2 gate defaults to APPROVE on
  empty/garbage Claude output; only `auth_error` triggers SKIP**

  ```python
  approved = True
  reasoning = ""
  try:
      ...
      output, success, exit_code, status = invoke_claude_text(...)
      output = (output or "").strip()
      if status == "auth_error":
          approved, reasoning = False, "auth failure"
      elif success and output:
          approved, reasoning = _parse_gate_response(output)
      else:
          logger.warning("ISKBETS L2 GATE: claude failed status=%s exit=%s for %s", status, exit_code, ticker)
  except Exception as e:
      ...
      logger.warning("ISKBETS L2 GATE: error — %s", e)
  ```

  If Claude returns `success=True` but `output=""` (the
  March-2026-style "exit 0 with empty stdout" failure), neither the
  `auth_error` branch nor the `success and output` branch fires — we
  fall through and `approved` is still `True` from the initialiser. The
  CLAUDE.md preamble explicitly calls this out as the silent-failure
  pattern that produced the 3-week Layer 2 auth outage. `_parse_gate_response`
  (line 287) defaults to `approved=True` for the same reason — even the
  successful-output path defaults to APPROVE when the LLM gives a
  response that doesn't contain `DECISION:`. Real money positions get
  opened on Avanza off the back of an LLM that returned no text.

## P1 — high-confidence bugs (should fix)

- `data/fish_engine.py:213-232` + `data/metals_swing_trader.py:2796-2797` +
  `portfolio/fin_fish.py:196-197` — **21:55 CET close hardcoded; DST gap
  weeks (Mar 8–29 / Oct 25–Nov 1) silently break EOD logic**

  ```python
  # fish_engine.py
  if hour > 21 or (hour == 21 and minute >= 55):
      if self.position is not None:
          ...
          "reason": "session end 21:55 CET",
  ```

  ```python
  # metals_swing_trader.py
  close_cet = 21.0 + 55 / 60  # 21:55 CET
  minutes_to_close = (close_cet - h) * 60
  ```

  `metals_swing_config.py:303-323` documents this explicitly: during
  DST-gap weeks the real close shifts to 21:00 CET so the 21:55 hardcode
  fires 45 min after the venue is closed. The mitigation there (raise
  `EOD_EXIT_MINUTES_BEFORE` to 25) is currently set to **0** on
  `:323` ("REVERT to 25 after current position closes"), and fish_engine
  / fin_fish don't have that escape hatch at all. The `.claude/rules/metals-
  avanza.md` rule ("Check API for `todayClosingTime` — do NOT hardcode
  21:55. Varies with DST.") is being violated in three files.

- `portfolio/grid_fisher.py:1573-1582` — **EOD market-sell falls back to
  `0.01 SEK` floor when both `bid` and `avg_entry_price` are 0**

  ```python
  quote = self._safe_session_call(
      self.session.get_quote, inst.ob_id, default=None,
  )
  if quote is None:
      bid = inst.avg_entry_price
  else:
      bid = float((quote or {}).get("buy") or 0)
  if bid <= 0:
      bid = inst.avg_entry_price
  aggressive = round(max(bid * 0.99, 0.01), 2)
  ```

  If the quote API is unreachable AND state has a 0 avg_entry_price (which
  happens on a freshly-seeded instrument with an inventory drift from
  reconciliation), `aggressive = 0.01` and we market-sell the entire
  inventory at 1 öre. On 1000 units that's 10 SEK recovered vs whatever
  the cert was actually worth. The `if bid <= 0:` recheck after the
  fallback doesn't loop back if `avg_entry_price` is also 0. Add a third
  guard: skip the EOD market-sell instead of pricing at 1 öre.

- `data/metals_avanza_helpers.py:330-408` — **`place_stop_loss` returns
  `(False, "")` on Playwright exception with `exc_info=True`, but caller
  in `grid_fisher.rotate_on_buy_fill` ignores the `stop_id=""` empty string
  and writes `None` regardless**

  Combined with the cancel-then-place flow in `grid_fisher.py:1247-1255`:
  if the place itself fails, `result` is None, log emitted, and
  `inst.stop_loss_id = new_stop_id` (None). The previous stop_id (if any)
  is already cancelled — so on a partial failure during rotation the
  position is silently naked. There is no retry-next-tick logic that
  re-attempts placement (compare `metals_swing_trader._retry_deferred_stops`
  which DOES retry). Grid fisher's `rotate` is fire-and-forget.

- `portfolio/exit_optimizer.py:54` + `:719` — **USDSEK has a 10.85 default
  that's silently used on any failure mode**

  ```python
  @dataclass(frozen=True)
  class MarketSnapshot:
      ...
      usdsek: float = 10.85
  ```

  ```python
  fx_rate = agent_summary.get("fx_rate", 10.85)
  ```

  Mar–May 2026 spot has been 10.45–10.95 — 10.85 is roughly current but
  every 0.10 SEK is ~1% of every SEK calculation. The `A-MC-2 (2026-04-11)`
  comment in `fin_snipe_manager.py:424-433` confirms the same pattern
  caused a 10x error when `usdsek=1.0` was the fallback; the current
  `or 10.85` fallback only avoids that magnitude error, not the smaller
  ongoing 1-3% drift error on every PnL print.

- `data/metals_swing_config.py:323` + `data/crypto_swing_config.py:163` +
  `data/oil_swing_config.py:199` — **`EOD_EXIT_MINUTES_BEFORE = 0`
  effectively disables EOD exit for all three loops; crypto/oil swing
  traders have NO hardware stop-loss path**

  Metals_swing_trader DOES place a hardware stop via
  `place_stop_loss` (`metals_swing_trader.py:2776`), so even with
  `EOD_EXIT_MINUTES_BEFORE=0` a position has broker-side protection.
  Crypto_swing_trader and oil_swing_trader have software stops only
  (`crypto_swing_trader.py:412`: `if warrant_pct_change <= -sl_pct:`
  return exit). If the loop process dies, the position is unprotected
  for an arbitrary period. With `MAX_HOLD_HOURS=72` (crypto) /
  `MAX_HOLD_HOURS=48` (oil), a process crash during a weekend leaves an
  unbounded exposure to overnight crypto wicks (5-10% common). No
  hardware stop because the deployment is `DRY_RUN=True` today — but
  the rule "stops first" should ship before flipping live.

- `portfolio/oil_grid_signal.py:51-63` — **RSI is computed with pure
  EWM, not Wilder's SMA-then-EWM seed; gates `GRID_MIN_SIGNAL_CONFIDENCE`
  on a wrong value**

  ```python
  def _rsi(series: pd.Series, period: int = 14) -> float:
      ...
      avg_gain = gains.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
      avg_loss = losses.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
      ...
  ```

  Wilder's RSI seeds the first 14 values with SMA and then runs Wilder's
  smoothing. Pure EWM with alpha=1/14 produces a different (and noisier)
  value, particularly in the first ~30 bars of the series. Confidence
  threshold (`GRID_MIN_SIGNAL_CONFIDENCE = 0.56`) is then computed off
  this value (`oil_grid_signal.py:86-94`). Recommend `pandas_ta.rsi` or
  Wilder seed.

- `data/metals_risk.py:45` — **`MAX_TRADES_PER_SESSION` comment claims
  "08:00-17:25" session but Avanza commodity warrants trade 08:15-21:55**

  ```python
  MAX_TRADES_PER_SESSION = 6         # max trades in a single market session (08:00-17:25)
  ```

  Either the comment is stale or the rate-limit window math (not shown
  in scope, lives elsewhere in metals_risk.py) is keyed on equities
  hours rather than commodity-warrant hours. The CLAUDE.md note
  "Avanza commodity warrants: 08:15-21:55 CET (NOT 17:25)" is being
  violated. Likely silently allows more than 6 trades per actual
  warrant session, or stops counting trades after 17:25.

## P2 — concerns / smells (worth addressing)

- `data/fish_engine.py:647-655` — **Layer 2 weight-2 hack double-inserts
  the same vote into `votes` dict; misleading "N tactics agree" message
  + bypasses MIN_VOTES intent**

  ```python
  l2_vote = self._vote_layer2(state)
  if l2_vote:
      votes['layer2'] = l2_vote
      votes['layer2_w'] = l2_vote  # duplicate to get weight 2
  ```

  A single LLM-derived vote alone trips `MIN_VOTES = 2` (`:86`). The
  decision report then claims "2 tactics agree LONG: layer2, layer2_w"
  — there's only one real source. If Layer 2 had an auth failure that
  silently returned bullish + conviction>0.4 we'd open a position with
  no other confirmation.

- `portfolio/grid_fisher.py:266-296` — **EOD timing math uses
  `Europe/Stockholm` zoneinfo for DST but still hardcodes the venue's
  close at 21:55; doesn't consult Avanza `todayClosingTime`**

  Same DST-gap risk as P1: the wall-clock conversion is correct, but
  the venue's effective close varies seasonally even in CET. See the
  metals_swing_config.py:303-323 documentation of the DST-gap drift.

- `data/fish_engine.py:215-232` — **Session-end 21:55 SELL signal
  doesn't auto-clear `self.position`; next tick fires SELL again**

  Returning the SELL action is correct, but if the caller doesn't call
  `confirm_exit` (e.g. session-end after Avanza is closed), the position
  remains in engine state and every subsequent tick re-emits the SELL.
  Telegram floods with duplicate "session end" notifications.

- `portfolio/orb_predictor.py:36-46` + `:131-132` — **`_morning_window_utc()`
  is evaluated once in `__init__`; predictor instantiated in summer but
  processing winter history (or vice versa) uses wrong UTC bracket for
  every other day**

  ```python
  if morning_start_utc is None or morning_end_utc is None:
      _start, _end = _morning_window_utc()
      morning_start_utc = morning_start_utc or _start
      morning_end_utc = morning_end_utc or _end
  ```

  Backtest accuracy degrades silently across DST boundaries. Also
  `DAY_START_UTC=8 / DAY_END_UTC=22` (`:32-33`) is static — the
  Avanza warrant window of 08:15–21:55 CET is 06:15–19:55 UTC in
  summer, 07:15–20:55 UTC in winter; neither matches `8..22`.

- `data/fish_engine.py:467-492` — **MC-band TP/SL sign handling relies
  on caller-guaranteed `mc_bands['5'] < ep` and `mc_bands['75'] > ep`
  invariant; degenerate bands produce minimum 1% floors but the
  `tp_pct/sl_pct` enforcement masks real bad data instead of skipping**

  ```python
  tp_pct = max(tp_pct, 1.0)
  sl_pct = min(sl_pct, -1.0)
  ```

  If MC bands collapse (e.g. high-vol regime where MC produces a
  same-direction quantile pair), the floor silently picks 1%/-1%. The
  caller has no way to know the dynamic TP/SL was bogus and fell back —
  the `(MC)` tag (`:496`, `:505`) is appended either way. Better to
  abort to `EXIT_TP_PCT/EXIT_SL_PCT` and log the band-degeneracy.

- `data/metals_swing_trader.py:2760` — **Stop sell price computed as
  `trigger_price * 0.99`; for a low-priced cert the absolute 1% gap can
  be tighter than tick size, yielding sell_price = trigger_price after
  rounding to öre**

  ```python
  trigger_price = round(stop_anchor * (1 - warrant_drop_pct), 2)
  sell_price = round(trigger_price * 0.99, 2)
  ```

  E.g. trigger=1.50, 1% below is 1.485 → rounds to 1.49 OK. But for
  trigger=0.50: sell=0.495 → 0.50. Sell-price == trigger-price means the
  stop triggers but never fills (limit price = trigger). For sub-1-SEK
  certs (some MINI warrants), this is a real edge case.

- `data/fish_monitor.py:14-37` — **Hardcoded `CERT_ID = "2304634"`,
  `ENTRY_PRICE = 29.02`, `current_sl_trigger = 24.67` and a position-
  specific `TRAIL_SCHEDULE`; this is a one-off snapshot, not a service**

  Module is loaded by the per-position monitor; the constants don't get
  refreshed when a different cert is bought. Risk: stale stop schedule
  ratcheted against a different cert's entry price → wrong SL ladder
  fires.

## Did NOT find

1. **Silent failures**: found multiple (P0 iskbets gate, P1 grid_fisher
   place_stop_loss + cancel rollback) — these are the dominant pattern
   in the subsystem.
2. **Race conditions**: order placement is gated by `avanza_order_lock`
   in metals_avanza_helpers (`:382-393`, `:476-485`) and the grid_fisher
   `_safe_session_call` runs Playwright on a worker thread; cancel/place
   sequencing in `fin_snipe_manager._stage_replacements` (`:317-337`)
   uses two-phase cancel-then-place to avoid concurrent windows. No new
   race issues spotted.
3. **Money-losing bugs**: found multiple (P0 direction-less Position,
   P0 grid_fisher stop pct, P0 grid_fisher rotation cancel rollback,
   P1 EOD 1-öre floor).
4. **State corruption**: `atomic_write_json` / `atomic_append_jsonl` are
   used consistently across persisted state files; grid_fisher singleton
   lock pattern in oil/crypto/metals loops is sound.
5. **Logic errors that pass tests**: P0 LONG-only exit_optimizer is
   covered by tests of the LONG path; SHORT positions would silently
   degrade in production. The `tactics_agreed` duplication for
   Layer 2 (P2) is a documented "weight 2" hack but no test asserts the
   semantics.
6. **Resource leaks**: Playwright pages are reused via shared session;
   subprocess in oil_loop._pid_alive is imported at module top
   (`:81` comment notes the codex fix); requests calls have timeout=5.
   Not spotted.
7. **Time/timezone bugs**: documented all P1+P2 hardcoded-21:55 and DST
   issues.
8. **API misuse**: stop-loss endpoint is correctly `/_api/trading/stoploss/new`
   in `metals_avanza_helpers.place_stop_loss` (`:386`), `fish_monitor.create_sl`
   (`:78`), and `place_stoploss_once.py:147` (one-shot). `delete_stop_loss`
   uses the canonical 2-segment URL `/_api/trading/stoploss/{accountId}/{stopId}`
   (`:481`). The fallback in `metals_loop.py:4995-5005` tries stoploss-cancel
   first then falls back to regular order cancel on 404 — correct.
9. **Trust boundary violations**: warrant catalog data is loaded from
   atomic JSON, no eval/exec/shell injection paths spotted.
10. **Partial state assumptions**: there are many `.get(key, default)`
    patterns where the default silently masks missing data
    (`fish_engine._evaluate_exit:434-440` defaults RSI=50, MC=0.5,
    metals_action="HOLD", event_hours=999 — if upstream truncates
    state, the engine quietly evaluates as if everything is neutral
    and stays in position). Documented in P2 MC band degeneracy.
