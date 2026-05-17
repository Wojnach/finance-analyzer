# Trading-Bots Adversarial Review — 2026-05-17

Scope: `portfolio/golddigger/*` (9 files), `portfolio/elongir/*` (9 files),
`portfolio/mstr_loop/*` (11 files incl. 5 strategies). 38 files, ~3,800 LOC.
Conventions checked: atomic state I/O, mstr_loop shadow-default, stop-loss
API, avanza_order_lock coordination, no concurrent-instrument orders.

Verdict: **shadow-default holds**, **stop-loss API correct**, **state I/O
atomic**, **order lock universally enforced via shared facades**. No P1
auto-promotes. Several P1 (other) and P2 issues — most concerning are
**three live cash mutations from a single shadow/paper code path**
(execution.py:246-253) and a **shadow-mode side-effect leakage that
silently hits the live Avanza API** (execution.py:74-86).

---

## Critical (90-100)

None auto-promote. Phase default is correctly `shadow`, golddigger uses
the canonical stop-loss endpoint, all state writes are atomic, and the
shared `avanza_order_lock` serialises every BUY/SELL/SL placement across
all three bots + metals_loop + grid_fisher.

---

## Important (80-89)

### I1 — mstr_loop shadow mode silently calls live Avanza `get_quote` on every BUY/SELL [conf 85]

`portfolio/mstr_loop/execution.py:74-87` — `_compute_shadow_cert_price`
is the cert-pricing helper for **shadow and paper** modes, but its
"prefer live quote" preference order issues a live Avanza HTTP GET via
`avanza_session.get_quote(ob_id)` on every shadow BUY:

```python
74    try:
75        ob_id = (
76            config.BULL_MSTR_OB_ID if direction == "LONG" else config.BEAR_MSTR_OB_ID
77        )
78        if ob_id:
79            from portfolio.avanza_session import get_quote
80            q = get_quote(ob_id)
```

Plus the symmetric path in `_estimate_cert_bid` (`execution.py:438-445`)
that fires on every shadow SELL and every cycle's
`update_trail_state` → `_estimate_cert_bid` chain.

Consequences:
1. **Shadow-mode requirement violation in spirit.** The 90-day shadow
   phase is supposed to be no-side-effects. Live `get_quote` calls
   touch the shared Avanza session (consumes rate budget, refreshes the
   BankID idle timer, can fail and leak warnings into the metals_loop /
   golddigger ops logs that share the same session).
2. **Reads, but still hits the order endpoint domain.** Not a stop-loss
   API confusion (it's `/_api/market-guide/stock/{ob}/quote`), but it
   is a side-effecting outbound call on every cycle when MSTR is open.
3. **Wrong endpoint family for the cert.** `get_quote` uses
   `/_api/market-guide/stock/{id}/quote` (`portfolio/avanza_session.py:667-673`)
   but BULL MSTR X5 SG4 is a cert (orderbook 2257847). The market-guide
   `stock` path frequently returns empty/null for certs — meaning the
   "live fidelity" rationale doesn't even pay off, while still firing
   the side-effects.

Fix: gate the live-quote branch on `config.PHASE == "paper"` (or new
`config.SHADOW_USE_LIVE_QUOTES = False`) so shadow mode is strictly
read-from-nothing. If shadow truly needs the price for log fidelity,
fetch via `data_collector` (Binance/Alpaca proxy) not `avanza_session`.

### I2 — mstr_loop `_handle_sell` mutates wins/losses/total_trades in shadow mode (state contamination) [conf 80]

`portfolio/mstr_loop/execution.py:245-253`:

```python
245    # Update aggregate stats + cooldown marker.
246    state.total_trades += 1
247    state.total_pnl_sek += pnl_sek
248    if pnl_sek > 0:
249        state.wins += 1
250    else:
251        state.losses += 1
252    state.last_exit_ts[decision.strategy_key] = _now_iso()
253    state.remove_position(decision.strategy_key)
```

These mutations run for **all phases including shadow**. Combined with
`_handle_partial_sell` (`execution.py:322` also runs in shadow), the
persistent `data/mstr_loop_state.json` accumulates simulated
trade-count, win-rate, and P&L across the entire 90-day shadow phase.
When the operator flips Phase A → paper or live, those shadow stats
become indistinguishable from live trades for `telegram_report.py:73-90`
("trades: X (YW/ZL, NN%)") and `mstr_loop_scorecard.py`.

This is technically "intentional" per the docstring at line 247 ("partials
count toward running P&L"), but the docstring on `execution.py:1-18`
explicitly states shadow "positions tracked in-memory only" — the stats
mutations contradict that contract.

Fix: gate `state.total_trades`/`state.total_pnl_sek`/`state.wins`/
`state.losses` mutations on `config.PHASE != "shadow"`. Keep the
shadow scorecard in `SHADOW_LOG` only (`_record_shadow` already
writes the full event for separate aggregation).

### I3 — golddigger naked position window if hardware stop-loss fails [conf 82]

`portfolio/golddigger/runner.py:190-209`:

```python
190    # Hardware stop-loss after BUY
191    if side == "BUY" and cfg is not None and getattr(cfg, 'hardware_stop_loss', False):
...
204                            else:
205                                logger.error("Failed to place hardware stop-loss!")
206                                _send_telegram("_GOLDDIGGER: Failed to place stop-loss!_", config)
```

When `place_stop_loss` returns `sl_ok = False`, the bot only sends a
Telegram alert and proceeds — the BUY is already filled by line 177.
The position is now naked until the next 5s poll cycle reaches
`_check_exit_conditions` (`bot.py:347`), which uses software stop
against the bid. If the loop crashes between the BUY response and the
next cycle, or Avanza session expires (session-check is every
`session_check_interval = 300s`), no hardware stop exists. With a 20x
leveraged cert this is a wipeout risk inside one cycle on a fast move.

The grudge memory (`memory/feedback_mini_stoploss.md`) is about stop
placement near barriers, not naked-position windows — but the operational
risk is the same shape: a leveraged position relying on bot-side checks
for survival.

Fix: roll back the BUY if SL placement fails. Either (a) immediately
issue a market SELL at the bid on SL failure, or (b) require SL
placement to succeed before persisting `state.open_position(...)` —
currently `state.open_position` ran in `bot.py:323-332` BEFORE
`_execute_order` even sees the action (the action dict is what comes
back to the runner). Restructure so SL placement is a pre-condition
for `state.open_position`.

### I4 — golddigger `_check_session_alive` 5-min recovery wait blocks ALL trading [conf 80]

`portfolio/golddigger/runner.py:293-339`:

```python
293                if not session_ok:
294                    logger.error("Avanza session expired after 3 checks — "
295                                 "waiting 5 min for possible renewal...")
...
300                    time.sleep(300)
```

The `time.sleep(300)` blocks the entire loop, including position
monitoring on any open BULL GULD X20 cert held during US session.
A 20x leveraged position with no live bid checking for 5 minutes can
move 30%+ on FOMC/CPI macro events. The bot DOES check the open
position via software stop in `_check_exit_conditions`, but only when
the loop ticks — which is blocked by the sleep.

Note that golddigger's session check is gated by `cfg.session_check_interval = 300s`
(config.py:119), so the worst case is a 5-min sleep triggered ~every 5
min cycle on persistent session failure — but each individual 5-min
window is enough to wipe out an open position.

Fix: if there is an open position, do a fast exit (software SELL via
the still-valid page if a quote can be fetched; if not, log critical
and alert) before sleeping. Or shorten the wait to a single poll cycle
and rely on `MAX_CONSECUTIVE_ERRORS` to halt after sustained failure.

---

## Medium (60-79)

### M1 — mstr_loop dropped HOLD votes inflate weighted scores (look-ahead-shaped bias) [conf 75]

`portfolio/mstr_loop/data_provider.py:99-138` — `_compute_weighted_scores`
excludes HOLDs from both numerator and denominator. Docstring justifies
this by historical-vs-live mismatch (compacted log drops HOLDs).
**But two voters (one BUY, one SELL) is enough to drive `long_score = 0.5`
or `short_score = 0.5`** — close to the configured 0.55 entry thresholds,
with no minimum-voters gate inside the helper.

A new MSTR signal that fires only "BUY" with single-voter activation
would silently dominate the score. Strategy layer should enforce a
`MIN_BUY_VOTERS` gate as the docstring promises (line 105-106 mentions
"MIN_BUY_VOTERS" as a separate strategy gate), but neither
`momentum_rider.py` nor `mean_reversion.py` enforce one — they only
check `weighted_score_long >= 0.55`.

Fix: add `MIN_ACTIVE_VOTERS` gate in `momentum_rider._evaluate_entry`
and `mean_reversion._evaluate_entry`. Suggest `>= 3` voters with
non-zero weight before honoring the score.

### M2 — golddigger uses `from_config` `equity_sek` but bot ignores it for risk sizing [conf 70]

`portfolio/golddigger/bot.py:302` — `sizing = self.risk.size_position(entry_price, self.state.cash_sek, ...)`.
`risk.py:87` — `def size_position(self, entry_ask, equity_sek, ...)`. The
bot passes `state.cash_sek` (current cash) as `equity_sek`. But
`record_trade_pnl` (`risk.py:46-56`) checks `self._daily_pnl <= -cfg.daily_loss_limit * cfg.equity_sek`
(line 50) — uses the **config equity**, not state cash. Inconsistent
basis: sizing uses live cash, halt-check uses static config. After a
losing trade that drops cash, sizing shrinks but halt threshold stays
at config baseline.

Fix: pick one. Recommend tracking equity peak as in mstr_loop and using
the rolling baseline for both sizing and halt.

### M3 — elongir trade message formats with hardcoded fx_rate fallback [conf 65]

`portfolio/elongir/runner.py:138`:

```python
138    equity = state.equity(action.get("silver_usd", 0), 10.5)  # approximate
```

The fx_rate is hardcoded to 10.5 for the trade-message equity calculation,
even though the snapshot has the real FX rate. The displayed equity in
the Telegram is wrong by the fx delta. Cosmetic for the message, but if
this equity ever feeds back into a decision via a future refactor it
becomes a bug.

Fix: pass `snapshot.fx_rate` through to the formatter.

### M4 — mstr_loop `_compute_shadow_cert_price` hardcoded 100.0 baseline distorts shadow P&L [conf 70]

`portfolio/mstr_loop/execution.py:87`:

```python
85    except Exception:
86        logger.debug("execution: live quote unavailable, using synthetic", exc_info=True)
87    return 100.0
```

When live quote fails (every shadow cycle if avanza_session unavailable,
which the docstring says is "tests, offline env, or pre-auth cold
start"), every shadow BUY is recorded at 100 SEK cert price. Combined
with the linear-leverage model in `_approx_cert_price_from_underlying`,
shadow P&L is anchored to a synthetic price that has no relation to the
real cert. The 90-day shadow scorecard built from this will overstate
or understate real P&L by 2-10x depending on actual cert price.

Fix: synthesize a more realistic baseline from
`bundle.price_usd × known_leverage_factor` or refuse to BUY in shadow
when live quote unavailable (skip cycle, log).

### M5 — elongir bot mutates state then calls log_trade after save [conf 65]

`portfolio/elongir/bot.py:211-242` — state mutations happen, then
`self.state.save(self.cfg.state_file)` at line 227, then `log_trade(...)`
at line 230. If log_trade raises (disk full, JSONL malformed), the state
file says the trade happened but the trade log has no record — the
trade is invisible to the daily report and to any scorecard that joins
state to trades.

Same shape exists in `_execute_sell` (lines 294-321).

Fix: log_trade before save (state file is the durable record; trade log
is observability — observability should land first so missed obs are
visible). Or wrap both in a single atomic transaction.

---

## Low (40-59)

### L1 — elongir runner f-string `{cfg.equity_sek:,.0f}` in logger.info uses % formatter, will TypeError on some inputs [conf 55]

`portfolio/elongir/runner.py:176-178`:

```python
176        logger.info(
177            "Elongir starting (poll: %ds, equity: %,.0f SEK)",
178            cfg.poll_seconds, cfg.equity_sek,
179        )
```

The `%,.0f` format specifier is **not valid for `%`-style formatting**
(it's an f-string idiom). At runtime this will raise `ValueError:
unsupported format character ','` when the log line tries to render.
Likely never noticed because the `INFO` log line might be silenced or
the test path uses f-strings everywhere. Verified by attempting
`"%,.0f" % 100.0` in Python — raises ValueError.

Fix: `"equity: %.0f SEK"` or switch to f-string.

### L2 — golddigger `bot.py` augmented signals refresh inside entry-check path bypasses session-window gate [conf 50]

`portfolio/golddigger/bot.py:275-284` — `self.augmented.refresh_if_needed()`
is called inside `_check_entry_conditions`. Outside session window,
`bot.step()` returns at line 134 before reaching this — so refresh only
fires during session. OK on session. But `refresh_if_needed` is invoked
on every entry attempt regardless of whether other gates have already
ruled out a BUY (e.g., spread too wide, or signal not yet at threshold).
Wasted Binance API calls on every entry-attempt cycle. Cosmetic at
60s refresh interval, but combines poorly with the 5s poll cadence.

Fix: move `augmented.refresh_if_needed()` to top of `step()` so its
self-throttle does the gating, not the call-site.

### L3 — mstr_loop `seconds_until_next_session` unused [conf 50]

`portfolio/mstr_loop/session.py:91-119` — `seconds_until_next_session`
is defined but never called by `loop.py` (which sleeps a fixed
`CYCLE_INTERVAL_SEC = 60` regardless). During weekends + after-hours
the loop wakes every 60s to compute `in_session_window() → False` and
sleep again. Wasted CPU + log entries.

Fix: in `run_forever`, use `seconds_until_next_session()` when outside
the window to sleep until open. Or delete the helper.

### L4 — mstr_loop `mean_reversion.py` reads `BEAR_MSTR_OB_ID` at module import time would freeze None [conf 45]

`portfolio/mstr_loop/config.py:33`:

```python
33  BEAR_MSTR_OB_ID: str | None = os.environ.get("MSTR_LOOP_BEAR_OB_ID") or None
```

Module-level frozen at import. If operator wants to enable mean_reversion
they must set the env var before mstr_loop starts and restart the bot.
The docstring says "or set MSTR_LOOP_BEAR_OB_ID env var" — operator
contract is unclear: can they edit config.py and HUP the process? No,
the loop won't pick it up either way without restart. Acceptable but
worth documenting in `docs/MSTR_LOOP_NOTES.md`.

### L5 — elongir bot doesn't track entry_fx_rate, MTM uses wrong FX after rate changes [conf 50]

`portfolio/elongir/bot.py:280-285`:

```python
278        proceeds = pos.quantity * w_bid
279        fee = proceeds * self.cfg.commission_pct
280        net_proceeds = proceeds - fee
281
282        # P&L calculation
283        pnl = net_proceeds - pos.cost_sek
```

`pos.cost_sek` is locked at entry-time fx_rate (via `warrant_price_sek`).
`w_bid` at exit uses current fx_rate. If USD/SEK moves 1% during a 3h
hold, the reported P&L includes 1% FX bleed without it being attributed.
Acceptable for SEK-denominated bookkeeping (the cert IS SEK-denominated)
but the silver_gain_pct in the trade log is computed off
`(snapshot.silver_usd - pos.entry_silver_usd) / pos.entry_silver_usd`
without the FX delta, so the reason-string under-/over-states the
underlying move responsible for the P&L. Cosmetic.

---

## Notes / Observations (not findings)

### N1 — Stop-loss API: COMPLIANT

golddigger's `place_stop_loss` (`runner.py:200`) → `avanza_control.place_stop_loss`
(`avanza_control.py:137-156`) → `metals_avanza_helpers.place_stop_loss`
(`data/metals_avanza_helpers.py:330`) — which uses the canonical
`/_api/trading/stoploss/new` endpoint per the Mar 3 incident rule.
Confirmed at `metals_avanza_helpers.py:330-390`.

### N2 — Cross-bot order coordination: COMPLIANT

All Avanza order paths (golddigger, metals_loop, mstr_loop Phase D,
fin_snipe_manager) flow through `avanza_session.place_buy_order`,
`avanza_session.place_sell_order`, `avanza_session.place_stop_loss`, or
`metals_avanza_helpers.place_order`/`place_stop_loss`. Every one of
those is wrapped in `with avanza_order_lock(...)` (verified at
`avanza_session.py:620, 644, 800, 915` and `metals_avanza_helpers.py:298, 383, 426, 477`
and `avanza_control.py:174, 222`). Cross-process file lock at
`data/avanza_order.lock` with 2s fail-fast timeout. Race-on-buying-power
scenario from `avanza_order_lock.py:6-13` is properly mitigated.

### N3 — mstr_loop PHASE default: COMPLIANT

`portfolio/mstr_loop/config.py:19`:

```python
19  PHASE: Phase = (os.environ.get("MSTR_LOOP_PHASE") or "shadow").strip()
```

Default is `"shadow"`. Operator must explicitly set `MSTR_LOOP_PHASE=live`
to enable orders. Single source-of-truth, no surprise live escalation
from import-time. **However see I1/I2/M4 for shadow side-effect
violations.**

### N4 — State atomic-write: COMPLIANT

All three bots use `portfolio.file_utils.atomic_write_json` /
`atomic_append_jsonl`:
- `golddigger/state.py:47` (BotState.save), `state.py:133, 183` (log_trade, log_poll)
- `elongir/state.py:122` (BotState.save), `state.py:173, 205` (log_trade, log_poll)
- `mstr_loop/state.py:199` (save_state), `execution.py:554, 579` (record helpers),
  `loop.py:66` (poll log)

One exception worth noting: `mstr_loop/telegram_report.py:33-38` reads the
state file with a raw `with open(...) as f: raw = json.load(f)` —
acceptable because it's read-only and the writer (`_save_ts_state` at
line 42) uses `atomic_write_json`. So readers see either old or new file,
never corrupt.

### N5 — No CLAUDECODE env leak: CONFIRMED CLEAN

Grepped all bot subprocess spawn paths. None of the three bots spawn
Claude subprocesses (only `_send_telegram` and Avanza HTTP calls).
Risk does not apply.

### N6 — Premium arb is a stub (never enabled)

`portfolio/mstr_loop/strategies/premium_arb.py:42-44` — `step()` returns
None unconditionally. `enabled = False` and `STRATEGY_TOGGLES["premium_arb"] = False`
in config. The MSTR-premium-mis-pricing risk called out in the brief
("Premium arb that mis-prices NAV premium") doesn't apply because no
calculation exists — the docstring acknowledges it as "scaffold only,
DO NOT enable" and lists the prerequisites. Same for `earnings_play`
and `overnight_gap`.

### N7 — Strategy registration cannot double-fire

`portfolio/mstr_loop/strategies/__init__.py:26-45` — strategies are
loaded into a list keyed by `key`, iterated once per cycle in
`loop.run_cycle:132-143`. Each strategy has its own position slot in
`state.positions` keyed by `strategy_key` — no double-fire on the same
position. Two strategies can hold positions in the SAME cert (e.g.,
both momentum_rider and mean_reversion in BULL MSTR if mis-configured)
because there is no per-instrument deduplication, but mean_reversion is
hardcoded to BEAR_MSTR_OB_ID so this is structurally prevented.

### N8 — Singleton locks present for all three bots

- golddigger: `data/golddigger.singleton.lock` via `process_lock`
  (`runner.py:98-113`).
- elongir: `data/elongir.singleton.lock` via `process_lock`
  (`runner.py:77-92`).
- mstr_loop: declared `SINGLETON_LOCK_FILE = "data/mstr_loop.singleton.lock"`
  in `config.py:199` BUT **not acquired anywhere in loop.py or
  __main__.py**. Verified: grepped `singleton_lock` in mstr_loop —
  zero usages. Two `python -m portfolio.mstr_loop` runs would coexist.
  Low impact because shadow is idempotent and paper/live use
  `state.cash_sek` which would over-deduct on race. Mark as P2 latent
  bug if Phase D ever activates.

---

## Summary

| Severity | Count | IDs |
|---|---|---|
| Critical (90-100) | 0 | — |
| Important (80-89) | 4 | I1, I2, I3, I4 |
| Medium (60-79) | 5 | M1, M2, M3, M4, M5 |
| Low (40-59) | 5 | L1, L2, L3, L4, L5 |

**Top fix priority**: I1 (shadow leaks live API calls) and I2 (shadow
contaminates persistent stats) violate the documented shadow contract
and would distort the 90-day evaluation that gates Phase A approval.
I3 (golddigger naked position on SL failure) is the highest absolute
financial risk of the four — a 20x cert without a stop is a single-tick
wipeout risk.

Files reviewed: 38 (all listed in brief).
Live state files: not opened per brief.
