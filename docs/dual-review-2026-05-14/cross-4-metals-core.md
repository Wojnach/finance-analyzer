# Cross-Critique — 4 metals-core

## Agreement — high-confidence findings (both reviewers)

- **`portfolio/exit_optimizer.py` — `Position` lacks direction, warrant PnL formula assumes LONG, P0.** Both reviewers identify the same dataclass and the same `pct_move = (exit - entry) / entry` math. Codex extends with `:303-340` `:741-750` call sites and the candidate-filter pruning bug at `:568`. Claude flags grid_fisher's BEAR placement (`"SHORT" = "2286417"`). **Independent rediscovery.** Production effect: every BEAR exit recommendation is structurally inverted; the optimizer picks the WORST candidate as "best".

- **`portfolio/grid_fisher_config.py:50` GRID_STOP_PCT=3.5 vs 5x certs (P0).** Both flag the rule violation (`.claude/rules/metals-avanza.md`: 5x certs need ≥15% stops). Codex extends with `grid_tiers.build_exit_levels` math and the comparison to `metals_swing_config.STOP_LOSS_WARRANT_PCT = 30.0` — same family of instruments, 10x looser stops elsewhere. **Strong cross-validation.** Action: raise to 15% or PROBE_ONLY mode until backtested.

- **`portfolio/iskbets.py` Layer 2 gate fails open to APPROVE on empty Claude output, P0.** Both flag the same default = True + missing `DECISION:` marker fallback. Codex additionally surfaces that `_parse_gate_response` itself defaults to APPROVE on missing marker, AND links it explicitly to the 3-week silent Layer 2 auth outage shape. **Independent rediscovery, very high confidence.**

- **`portfolio/grid_fisher.py:1554-1561` cancel-stop rollback gap (P1).** Both find the same pattern: cancel via `_safe_session_call` which swallows None, then `inst.stop_loss_id = None` written regardless of whether the actual cancel succeeded. Codex's framing extends to rotate_on_buy_fill (`:1217-1256`) with the same shape. **Same family, both right.** Action: gate the state mutation on confirmed cancel + retry-next-tick on place failure.

- **EOD market-sell 1-öre floor (P0/P1 split).** Claude flagged it as P0, Codex as P1. Both identify the `bid=0 + avg_entry=0 → 0.01 floor` path. Severity disagreement: Codex argues "happens only on freshly-seeded inventory drift", Claude argues "state-file corruption or fresh-replay scenario". Pragmatic: **P0 because the impact is catastrophic when it triggers** (liquidate 1000 units at 1 öre).

- **`data/fish_engine.py:213-232` 21:55 CET hardcode (P1).** Both flag. Codex extends with the full DST-gap timeline (Mar 8-29 / Oct 25-Nov 1), the explicit `metals_swing_config.py:303-323` documentation of the drift, the `EOD_EXIT_MINUTES_BEFORE=0` regression that disabled the mitigation, AND the rule file `.claude/rules/metals-avanza.md` ("Check API for `todayClosingTime`"). **Codex significantly stronger.** Action: read `todayClosingTime` from Avanza.

- **`portfolio/oil_grid_signal.py` RSI uses pure EWM not Wilder's (P1).** Both flag. Codex extends with line refs (`:51-63`, `:86-94`) and confidence-threshold consumer (`GRID_MIN_SIGNAL_CONFIDENCE`). Claude flags "Wilder's seeding" issue. **Same finding.**

## Codex found, Claude missed

- **`data/metals_avanza_helpers.py:330-408` — `place_stop_loss` returns `(False, "")` on Playwright exception; grid_fisher writes `None` regardless and there's no retry-next-tick (P1).** Compare to `metals_swing_trader._retry_deferred_stops` which DOES retry. Grid fisher's `rotate` is fire-and-forget. **Real silent-failure gap — Claude missed the "no retry path" piece.**

- **`portfolio/exit_optimizer.py:54` `usdsek = 10.85` hardcoded default (P1).** Claude didn't flag. Codex notes Mar-May 2026 spot has been 10.45-10.95 — 1-3% ongoing drift error on every PnL print, same family as the 10x error from `usdsek=1.0` (A-MC-2 fix). **Codex right.**

- **`data/metals_swing_config.py:323` `EOD_EXIT_MINUTES_BEFORE = 0` regression (P1).** Comment says "REVERT to 25 after current position closes" — never reverted. Effectively disables EOD exit for all three loops. Crypto/oil have software stops only (no hardware), so if process dies on weekend, exposure is unbounded. **Real ongoing regression Claude missed.**

- **`data/metals_risk.py:45` `MAX_TRADES_PER_SESSION` comment says "08:00-17:25" but commodity warrants are 08:15-21:55.** Codex flags both directions: either limits more trades than expected, or stops counting at 17:25. **Real, Claude missed.**

- **`data/fish_engine.py:647-655` `votes['layer2_w'] = l2_vote` duplicate to bypass MIN_VOTES (P2).** A single LLM vote with `MIN_VOTES=2` opens position with no other confirmation. If Layer 2 had a silent auth failure that returned bullish + conviction>0.4, position opens with no second source. **Codex right — direct hit, Claude missed.**

- **`portfolio/orb_predictor.py:36-46, 131-132` — `_morning_window_utc()` evaluated once in `__init__`, drifts across DST boundaries.** Backtest accuracy degrades silently. Also `DAY_START_UTC=8 / DAY_END_UTC=22` static — Avanza window is 06:15-19:55 UTC summer, 07:15-20:55 UTC winter. **Real silent ORB calibration bug — Claude flagged ORB at P2 generally but missed the DST drift.**

- **`data/metals_swing_trader.py:2760` — `sell_price = trigger * 0.99` rounds to `trigger_price` for sub-1-SEK certs.** Limit price = trigger price means stop triggers but never fills. Sub-1-SEK MINI warrants. **Real, narrow, Codex caught.**

- **`data/fish_monitor.py:14-37` — hardcoded `CERT_ID=2304634, ENTRY_PRICE=29.02, current_sl_trigger=24.67` and position-specific `TRAIL_SCHEDULE`.** One-off snapshot loaded by the per-position monitor; doesn't refresh for new cert. **Real, stale-state risk, Codex caught.**

## Claude found, Codex missed

- **`portfolio/iskbets.py:316-329` 30s timeout for "fast APPROVE/SKIP".** Claude flagged as P2: heavy CLAUDE.md auto-load makes 30s tight, gate-timeout chains into default APPROVE per P0. Codex flagged the parent P0 but missed the timeout-too-short contributor.

- **`portfolio/grid_fisher.py:1545-1552` duplicate-sell guard via `inst.eod_sell_order_id` persisted across sessions.** Claude's catch: if state file is reused and `eod_sell_order_id` not cleared on new session start, EOD day N+1 skips the sell entirely. Codex didn't drill into the cross-session reset path. **Real, narrow.**

- **`portfolio/grid_fisher.py:1607` — `inst.eod_sell_order_id = str(order_id)` assigned but never cleared on fill.** Once EOD fill happens, the id stays in state forever. Reset path through `_on_order_filled` doesn't clear. **Claude right.**

- **`data/fish_engine.py:248` — `_mc_history` rolling window of 3 (3-minute smoothing).** Too short for meaningful mean reversion on MC P(up). Claude flagged as P2. Codex didn't.

- **`portfolio/iskbets.py:282-301` — substring match on "SKIP" matches `"SKIP-FALSE-POSITIVE"` and `"NO_SKIP"`.** Claude's catch: too-loose substring check. The fail-safe direction here helps slightly (APPROVE→SKIP is conservative) — but combined with the P0 fail-open default, the parser is unreliable both ways. **Real P1.**

## Disagreements

**Severity disagreement on EOD 1-öre floor**: Claude P0, Codex P1. Both note the same impact (liquidate at 1 öre). The dispute is frequency — Codex assumes only fresh-seeded inventory drift; Claude assumes broader corruption scenarios. **Resolution: leave as P0** because the failure mode is catastrophic (full position at 1 öre) regardless of frequency; the cheap fix (refuse to place if bid<=0 AND avg_entry<=0) doesn't need a frequency justification.

## What BOTH missed (third pass)

- **`portfolio/grid_fisher.py` order-id state-vs-broker reconciliation.** Neither reviewer asked: after a process restart, how does grid_fisher detect that an order it placed yesterday is still live on Avanza vs filled vs cancelled by the broker? `_safe_session_call(get_open_orders)` lookup, if it exists, isn't audited. Without reconciliation, state.json's `eod_sell_order_id` can lie persistently in either direction.

- **`portfolio/exit_optimizer.py` interaction with `kelly_metals`.** Cross-subsystem: `kelly_metals.recommended_metals_size` (flagged P0 in #3 portfolio-risk for leverage saturation) sizes the position; `exit_optimizer.compute_exit_plan` exits it. If they disagree on the leverage applied (kelly assumes 5x, optimizer assumes 1x because `leverage=1.0` default), the planned exit is wrong even before BEAR direction enters. Neither reviewer cross-linked.

- **`data/metals_loop.py` 10s silver fast-tick.** Mentioned in CLAUDE.md as "embedded 10s silver fast-tick monitor" but neither reviewer audited whether this 10s tick respects the same EOD/DST math, or whether it has its own hardcoded thresholds.

- **`portfolio/grid_fisher.py` `GRID_LEG_SEK=1200` and `GRID_GLOBAL_CAP=6500`.** Codex notes the values. Neither reviewer asked: does the cap include broker-side resting orders that may have been forgotten in state? If yes, a stuck-state grid_fisher will eventually refuse all legs while still holding orphaned orders.

- **`data/fish_engine.py` MC bands consumer.** Both reviewers flagged P2 issues with bands. Neither audited whether MC P(up) computation uses the FX-rate-bypass path identified in cross-3-portfolio-risk (monte_carlo_risk.py:419). If yes, the MC bands themselves are 1-10x off, which would explain the band-degeneracy floor cases.

## Verdict

P0 list after cross: **5 confirmed** (Position direction, GRID_STOP_PCT 3.5%, iskbets fail-open, EOD 1-öre floor, exit_optimizer SHORT path).
P1 list after cross: **~10 confirmed** (cancel rollback, place_stop_loss retry gap, DST 21:55 hardcode, EOD_EXIT_MINUTES_BEFORE regression, RSI Wilder, usdsek hardcoded, MAX_TRADES_PER_SESSION, oil/crypto soft stops, _parse_gate_response substring, fish_engine layer2 weight hack).
P2 list after cross: ~9.

Metals-core has the most P0 + P1 findings of any subsystem reviewed — partly because of the dual-direction (BULL/BEAR) cert complexity that LONG-assuming code wasn't designed for.
