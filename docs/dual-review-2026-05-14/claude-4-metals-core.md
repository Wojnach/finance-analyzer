# Adversarial Review — 4 metals-core (main-thread Claude, independent)

## P0 — money-losing or data-corrupting (must fix)

- `portfolio/iskbets.py:287, 313` + `portfolio/iskbets.py:335-338` — gate fails open to APPROVE on empty / malformed Claude output.
  ```python
  def _parse_gate_response(output):
      ...
      approved = True   # default
      ...
      if upper.startswith("DECISION:"):
          val = ...
          if "SKIP" in val:
              approved = False
          # APPROVE is the default, so only SKIP changes it
  ```
  And the call site at line 313 also defaults `approved = True`. Line 335: `elif success and output: approved, reasoning = _parse_gate_response(output)`. If `success=True` but `output=""` or `output="foo bar"` (no DECISION marker), `_parse_gate_response` returns `(True, "")` — silent approve. The auth-error override exists for one specific failure mode; every OTHER failure mode (network truncation, claude wrote to stderr, output buffer flush failed) silently approves the warrant entry. This is the exact 3-week silent Layer 2 outage pattern documented in CLAUDE.md. Fail closed: require an explicit `DECISION: APPROVE` token, default SKIP.

- `portfolio/exit_optimizer.py:58-73` — `Position` dataclass has no `direction` field; PnL math at line 95 (`pnl_sek`) assumes LONG. For BEAR warrants (BEAR SILVER X5, BEAR OLJAB X3), `(exit_price - entry_price)` has the wrong sign — every BEAR exit's EV is computed inverted. The `compute_exit_plan` then picks the "best" candidate which is actually the WORST. Combined with grid_fisher's BEAR placement (`GRID_ACTIVE_INSTRUMENTS["XAG-USD"]["SHORT"] = "2286417"`), every BEAR exit recommendation is structurally wrong.

- `portfolio/grid_fisher_config.py:50` — `GRID_STOP_PCT = 3.5` is too tight for 5x certs.
  ```python
  GRID_STOP_PCT = 3.5
  ```
  Comment at line 48-49 reads "stop sits outside the typical 15-minute volatility band". The rule file `.claude/rules/metals-avanza.md` says **"5x leverage certificates need -15%+ stops, not -8%, to survive intraday wicks"**. 3.5% on a 5x cert = 0.7% adverse underlying move → 15-min wicks fire it routinely. Result: every fill rotates into a sell+stop where the stop fires within minutes, locking a loss while the sell limit at +1.2% never fills. The net edge per cycle promised (~0.5%) becomes a guaranteed loss because the loss leg trips orders of magnitude more often than the win leg. Either raise stop_pct to 15% (matches rule), or refuse to deploy this strategy until backtested with realistic stop.

- `portfolio/grid_fisher.py:1576-1582` — EOD market-sell falls back to `avg_entry_price` if quote bid is 0, then to `0.01` floor.
  ```python
  if quote is None:
      bid = inst.avg_entry_price
  else:
      bid = float((quote or {}).get("buy") or 0)
  if bid <= 0:
      bid = inst.avg_entry_price
  aggressive = round(max(bid * 0.99, 0.01), 2)
  ```
  If both quote AND avg_entry_price are 0 (state-file corruption or fresh-replay scenario), `bid = 0`, `aggressive = max(0 * 0.99, 0.01) = 0.01`. We then `place_sell_order(ob_id, 0.01, inventory_units)` — a 1-öre limit sell on a 100+ unit position. Avanza will accept it; it fills against whatever the lowest existing bid is, which on a thin warrant could be ≤ 0.01 SEK matched. Result: liquidate the position at near-zero. Fix: refuse to place if both bid and avg_entry_price are non-positive.

## P1 — high-confidence bugs (should fix)

- `data/fish_engine.py:213-216` — hard-coded session-end at 21:55 CET.
  ```python
  if hour > 21 or (hour == 21 and minute >= 55):
      ...
  ```
  Avanza commodity warrants close 21:55 *Stockholm time*. During DST-gap weeks (late March, late October), the loop's `hour_cet` may be off by one if the calling code passes UTC-based hour. The grid_fisher uses `zoneinfo.ZoneInfo("Europe/Stockholm")` at line 285 — correct pattern. fish_engine should match. Test the late-October DST flip-back date.

- `portfolio/grid_fisher.py:1554-1561` — `eod_market_flat` cancels armed sell tiers and the stop, then places aggressive sell. If `cancel_stop_loss` fails (returns None via `_safe_session_call`), `inst.stop_loss_id = None` is set on line 1570 *anyway* — so the stop is forgotten by our state but may still be live on Avanza. The new aggressive sell goes through, and now both the orphaned stop AND the EOD sell are resting against the same inventory. When one fills, the other oversells (the system goes net short). Rollback the state mutation on cancel failure.

- `portfolio/iskbets.py:282-301` — `_parse_gate_response` is case-insensitive on the marker but case-strict on value parsing.
  ```python
  upper = line.upper()
  if upper.startswith("DECISION:"):
      val = line.split(":", 1)[1].strip().upper()
      if "SKIP" in val:
          approved = False
  ```
  `"DECISION: skipped"` matches → SKIP correctly. `"DECISION: SKIP-FALSE-POSITIVE"` also contains "SKIP" → matches as SKIP. `"DECISION: NO_SKIP"` also matches "SKIP" substring → SKIP. The substring check is too loose. Same on line 295 for "APPROVE" — but currently only SKIP is checked, so the false-positive direction is APPROVE-to-SKIP, which is fail-safe — except for the silent-APPROVE-default P0 above.

- `data/oil_swing_trader.py`, `data/crypto_swing_trader.py` — sister loops to metals_swing_trader. Both default `DRY_RUN=True` per CLAUDE.md. Need to verify the EOD logic copy matches metals_swing_trader's `close_cet = 21.0 + 55/60` (line 2796 of metals_swing_trader.py). Stocks-via-MSTR uses 21:55 close in the metals loop; if oil_swing_trader copies that, it's wrong (oil trades 23:00 CET on CME). Grep confirms only fish_engine hard-codes 21:55 in this subsystem (line 213); oil_swing_trader doesn't hit grep, but verify when it goes live.

- `portfolio/grid_fisher.py:1545-1552` — duplicate-sell guard via `inst.eod_sell_order_id`. If the previous-day sell ID is persisted in `data/grid_fisher_state.json` and the state file is reused across sessions WITHOUT clearing `eod_sell_order_id` on new-session start, EOD on day N+1 sees a non-None id and skips the sell entirely. Look for the clear path — line 463 mentions "Clear the EOD-sell flag so the new session can re-arm". Verify the session-reset path actually runs before tick on session start; if reset happens lazily (only on first explicit flip), an inherited id from yesterday blocks today's EOD sweep.

- `portfolio/oil_grid_signal.py` — direction signal for grid_fisher per CLAUDE.md. The CLAUDE.md notes "Brent BZ=F RSI+EMA, 5-min cache". RSI in many implementations seeds with EWM not Wilder's smoothing — produces ~5pp different values on the first 20 bars. If the RSI threshold gate is wrong, direction flips wrong. Verify Wilder's seeding in the RSI calc.

- `portfolio/metals_risk.py` — separate risk path for metals. Per subagent's hint, may use equity hours 09:00-17:25 instead of commodity 08:15-21:55. Investigate.

## P2 — concerns / smells (worth addressing)

- `portfolio/grid_fisher_config.py:25` — `GRID_FISHER_ENABLED = True` and `GRID_FISHER_PROBE_ONLY = False` mean live ordering. Combined with P0-3 (3.5% stop too tight), production is actively losing on every fill. Recommend setting PROBE_ONLY=True until P0-3 is addressed.

- `portfolio/grid_fisher.py:1607` — `inst.eod_sell_order_id = str(order_id)` assigned but never cleared on fill. Once EOD sell fills, the id stays in state, and a re-init that doesn't go through the "new session" reset path could see a stale id. Detect fill via `_on_order_filled` and clear there.

- `data/fish_engine.py:248` — `if len(self._mc_history) > 3: self._mc_history.pop(0)` — a rolling window of 3. With a 60s tick the MC history covers only 3 minutes. Mean reversion of a Monte Carlo P(up) reading should use ≥ 10 minutes for meaningful smoothing.

- `portfolio/orb_predictor.py` / `portfolio/orb_postmortem.py` / `portfolio/orb_backtest.py` — ORB (Opening Range Breakout) window. Stockholm opens 08:15 CET for warrants but the ORB convention is the *first N minutes* — verify N is set per asset class (5min for crypto vs 15min for warrants vs 30min for equities) and not a shared constant.

- `portfolio/iskbets.py:316-329` — `invoke_claude_text(..., timeout=30)` — 30s for "fast APPROVE/SKIP". If Claude is loaded with a heavy prompt context (CLAUDE.md auto-load), 30s may not be enough; the gate then times out → `success=False` → falls through to default APPROVE per P0 chain. Bump timeout and let the L2 cooldown handle storms.

- `data/metals_loop.py:401` — `EOD_HOUR_CET = 17.0` constant. Comment says "legacy summary trigger", but the variable is read in the main loop body — verify it's not being used as a market-close gate anywhere. If yes, it's the equity close (17:25), wrong for warrants.

## Did NOT find

1. **Silent failures**: P0 iskbets gate is the worst. Other paths (avanza_session calls wrapped in `_safe_session_call`) log on None return and skip the cycle — better than auto-approve.
2. **Race conditions**: grid_fisher state updates run on a single tick thread per session; no contention with metals_loop's separate thread observed in the code I read.
3. **Money-losing bugs**: P0-1 through P0-4 above; P1 stop-rollback gap.
4. **State corruption**: state files use atomic_write_json per the rule. P2 has stale-id concerns but not corrupted state.
5. **Logic errors that pass tests**: BEAR exit_optimizer EV math (P0) — tests probably only fixture LONG positions.
6. **Resource leaks**: Avanza session reuse OK; no per-cycle Playwright spawn.
7. **Time/timezone bugs**: P1 — fish_engine hardcoded 21:55 CET.
8. **API misuse**: stop-loss endpoint `/_api/trading/stoploss/new` correct per grep at metals_loop.py:4996. Min 1000 SEK / leg enforced via GRID_LEG_SEK=1200.
9. **Trust boundary violations**: ob_id is integer-string from internal config; not user input. Safe.
10. **Incorrect partial-state assumptions**: P0-4 (bid=0 + avg_entry=0 = 0.01 floor sell).
