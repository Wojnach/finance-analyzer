# Adversarial Code Review — Metals Core Subsystem
**Date:** 2026-05-24
**Branch:** `review/fgl-2026-05-24`
**Reviewer:** Claude Opus 4.7 (1M context, adversarial mode)
**Scope:** `data/metals_loop.py`, `data/fish_engine.py`, `data/crypto_loop.py`, `data/oil_loop.py`, `portfolio/grid_fisher*.py`, `portfolio/fin_snipe*.py`, `portfolio/golddigger/`, `portfolio/elongir/`, `portfolio/mstr_loop/`, `portfolio/oil_grid_signal.py`, `portfolio/iskbets.py`, `portfolio/bigbet.py`, `portfolio/analyze.py`, `portfolio/orb_predictor.py`, `portfolio/ministral_signal.py`, `portfolio/qwen3_signal.py`, `portfolio/metals_orderbook.py`, `portfolio/microstructure*.py`

This subsystem handles **real money flow** via Avanza warrants on 5x silver, gold, and oil certificates. Knockout barriers exist. Stop-loss math is life-or-death.

---

## TOP 5 (sorted by money-loss potential)

1. **`portfolio/grid_fisher.py:1655` | P0 | naked-position window**
   Stop-rearm retry uses `round(sp * 0.995, 2)` — a 0.5 % sell-price buffer below trigger. The original rotate-on-buy-fill path (`grid_fisher.py:1496`) does the same. On a 5x leveraged cert, a 0.5 % gap is roughly 1/7 of a normal 15-min ATR-band; a 2 % gap-down on the underlying = 10 % cert drop = trigger fires but the sell limit at `trigger*0.995` is already 9.5 % above the prevailing bid, will not fill, and the position falls another 5–10 % before manual intervention. Memory `feedback_mini_stoploss.md` mandates ≥3 % buffer. Fix: widen to `round(sp * 0.97, 2)` (3 %), matching the in-tree `STOP_ORDER_SPREAD_PCT` (1 % per level × 3 levels = 3 %) used by metals_loop's legacy stop ladder.

2. **`portfolio/fin_snipe_manager.py:62, 537` | P0 | wick bypasses stop sell**
   `HARD_STOP_SELL_BUFFER_PCT = 0.01` ⇒ stop sell limit sits **1 % below** the trigger. On 5x XAG / XAU certs, a single 0.2 % underlying spike past the trigger is enough to wick *through* the sell limit before any fill, leaving the position naked while Avanza shows "stop triggered, order resting unfilled". The same defect that bit the operator pre-2026-05-19. Companion `HARD_STOP_CERT_PCT = 0.05` (5 % below avg entry) is itself tight — only 1 % underlying for a 5x cert. Combined, the stop is too close to entry AND has too small a fill buffer. Fix in two steps:
   - `HARD_STOP_SELL_BUFFER_PCT = 0.03` (3 % below trigger)
   - Consider widening `HARD_STOP_CERT_PCT` to `0.08` (8 %, ≈ 1.6 % underlying — outside normal 15-min noise on silver). The prompt's claim that this was "now 0.03 per recent commit" is **false** — see commit `be4273d3` which changed `MIN_STOP_DISTANCE_PCT` (a different constant) from 1.0 → 3.0.

3. **`portfolio/grid_fisher.py:1885-1891` | P1 | EOD fire-sale fallback**
   `eod_market_flat()` first cancels the protective stop (line 1879), then fetches a quote. If `get_quote` returns a non-None dict with no `"buy"` key (a documented Avanza shape on illiquid hours), `bid` collapses to `0`, then falls through to `inst.avg_entry_price` (line 1890). If `avg_entry_price` is also unset (legacy state-file path, no test coverage), `aggressive = round(max(0 * 0.99, 0.01), 2) = 0.01` SEK — a 0.01 SEK sell on a 5+ SEK cert is a fire-sale and Avanza will fill it from the first bid down to 0.01. Stop already cancelled = naked while the limit hangs at 0.01. The probability of `avg_entry_price = 0` with `inventory_units > 0` is low (record_fill always sets it), but defence-in-depth says don't ship a code path that can sell at 0.01. Fix: abort EOD sweep with `critical_errors.jsonl` entry if `bid <= 0` and `avg_entry_price <= 0`; rearm a fresh stop instead.

4. **`data/metals_loop.py:2469, 4913` | P1 | legacy stop ladder sell buffer 1 %**
   Both `_rebuild_stop_orders_for` and `place_stop_loss_orders` compute `sell_price = round(trigger_price * 0.99, 2)` — same 1 % buffer issue as fin_snipe_manager. Mitigated by `STOP_ORDER_ENABLED = False` default + `HARDWARE_TRAILING_ENABLED = True`, but if an operator flips the toggle in an emergency (and it has a CLI/Telegram path), the same wick-bypass failure mode lights up. Fix: align with the corrected `HARD_STOP_SELL_BUFFER_PCT`.

5. **`data/layer2_invoke.py:46, 68` | P1 | partial fix from `4adeec2d`**
   Commit `4adeec2d` added absolute paths for `config.json` reads in the three `data/layer2_*.py` scripts, but `layer2_invoke.py` still has two relative-path *writes* — `data/layer2_journal.jsonl` (line 46) and `data/telegram_messages.jsonl` (line 68). `layer2_action.py` and `layer2_exec.py` were fully patched to use `BASE = pathlib.Path("Q:/finance-analyzer/data")`; layer2_invoke.py needs the same. Practical impact is bounded because `claude_gate.invoke_claude` sets `cwd=BASE_DIR`, so the writes currently land in the right place — but a manual `python data/layer2_invoke.py` from anywhere else drops a `data/` folder under that CWD and never updates the canonical journal. Fix: apply the same `BASE = pathlib.Path("Q:/finance-analyzer/data")` pattern.

---

## Critical (severity ≥ 90)

### `portfolio/grid_fisher.py:1496, 1655` | P0 | stop sell buffer too tight on 5x certs
**Path/lines:**
- `portfolio/grid_fisher.py:1496` (rotate_on_buy_fill)
- `portfolio/grid_fisher.py:1655` (stop_needs_rearm retry in tick)

```python
# 1496
stop_sell_price = round(stop_price * 0.995, 2)
# 1655
inst.ob_id, sp, round(sp * 0.995, 2),
```
**Cat:** stop-loss math / naked position risk on 5x leveraged certs.
**Why:** `0.995` = 0.5 % buffer. A normal 1-min volatility band on silver during US open is 0.3–0.6 % underlying = 1.5–3 % on a 5x cert. The stop trigger fires, the sell limit at `trigger * 0.995` is already inside that volatility band but at a stale price; if the price has gapped 1 %+ in the cert during the 100 ms between trigger and broker order receipt, the limit never fills. Position falls further naked. Memory `feedback_mini_stoploss.md` is explicit: minimum 3 % distance for silver MINIs. This is the same class of bug as the FGL-P0-2 fix (commit `be4273d3`) that widened `MIN_STOP_DISTANCE_PCT` 1.0 → 3.0 in fin_snipe_manager — but the grid_fisher pair was never widened.
**Fix:** `round(stop_price * 0.97, 2)` (3 % buffer). Sym-link constant via `grid_fisher_config.GRID_STOP_SELL_BUFFER_PCT = 0.03` so both call sites read the same value. Add a unit test that asserts the gap stays above a configurable floor.

### `portfolio/fin_snipe_manager.py:62, 537` | P0 | 1 % stop buffer + 5 % cert trigger combo
See top-5 #2 above. **Cat:** stop-loss math on 5x MINI certs. **Fix:** widen `HARD_STOP_SELL_BUFFER_PCT` to 0.03, consider widening `HARD_STOP_CERT_PCT` to 0.08. Add regression test asserting `(trigger - sell) / trigger >= 0.025` for any computed stop plan, and `(avg - trigger) / avg >= 0.05` after both knobs are set.

---

## Important (severity 80–89)

### `portfolio/grid_fisher.py:1885-1891` | P1 | EOD path can fire-sale at 0.01 SEK
See top-5 #3 above. **Cat:** order placement / fail-open. **Fix:** abort EOD sweep when both quote and avg_entry are missing; emit `critical_errors.jsonl` entry; rearm a fresh stop and retry next tick. Currently the path silently produces a 0.01 SEK sell on the failure branch.

### `data/metals_loop.py:2469, 4913` | P1 | legacy stop ladder 1 % buffer
See top-5 #4 above. **Cat:** stop-loss math. **Fix:** parameterize buffer; unify with the fin_snipe_manager constant; cover both code paths in tests.

### `data/layer2_invoke.py:46, 68` | P1 | relative-path writes after partial fix
See top-5 #5 above. **Cat:** atomic-I/O / CWD-sensitivity. **Fix:** absolute paths via `pathlib.Path(__file__).resolve().parent.parent / "data" / …`.

### `data/metals_loop.py:1592-1596` | P2 | signal-summary fallback shape masks stale data
```python
path = "data/agent_summary.json"
if not os.path.exists(path):
    path = "data/agent_summary_compact.json"
```
**Cat:** silent fail / stale signal use. If the full summary stops being written (Layer 1 crash), the loop silently swallows the compact file. The compact file has fewer tickers / fewer timeframes, but `read_signal_data()` returns the dict either way without flagging the downgrade. Gate Z stale detection further down only checks `signal_age_sec`, not which file produced it. An operator looking at "fresh signal" log lines won't know they got the degraded variant. **Fix:** add a `data_source` field to the returned dict, surface it in the structured status print line near loop heartbeat.

### `portfolio/grid_fisher.py:1885-1888` | P2 | quote with empty dict returns 0 bid
```python
quote = self._safe_session_call(self.session.get_quote, inst.ob_id, default=None)
if quote is None:
    bid = inst.avg_entry_price
else:
    bid = float((quote or {}).get("buy") or 0)
if bid <= 0:
    bid = inst.avg_entry_price
aggressive = round(max(bid * 0.99, 0.01), 2)
```
**Cat:** stale-data fallback at EOD. When Avanza returns an empty dict (no "buy" field on a halted instrument), bid becomes 0 then falls back to avg_entry_price. The EOD sell then prices at *avg_entry × 0.99*, leaving 49–100 % of any unrealised gains on the table — and if the cert had moved up significantly (a barrier-crossing rally on the BEAR side), the limit sits so far below market it'll fill instantly to anyone willing to take it. **Fix:** use the most-recent `live_volume` or `last_seen_price` from the reconcile data instead of stale entry.

### `portfolio/fin_snipe_manager.py:1500-1502` | P2 | `place_stop_loss_no_page` ID extraction is fragile
```python
placed_id = (
    str((result.get("result") or {}).get("stop_id") or "")
    or str((((result.get("result") or {}).get("parsed") or {}).get("stoplossOrderId")) or "")
)
```
**Cat:** stop-loss tracking. The double-fallback means if Avanza ever changes their response shape (or returns under a third key), the ID is silently lost. Line 1499 just logs a warning ("untracked"), then the next cycle the reconcile path treats the position as if no stop existed and places ANOTHER one. Stops stack up at Avanza until inventory * stops > position size, then the place call returns `short.sell.not.allowed`. **Fix:** raise a `_notify_critical` if both fallbacks return empty so an operator sees it within 30 min instead of via the side effect of stacked stops.

### `portfolio/oil_grid_signal.py:35` | P2 | relative `SIGNAL_FILE` path
```python
SIGNAL_FILE = "data/oil_grid_signal.json"
```
**Cat:** CWD-sensitive write. Same anti-pattern as elsewhere. Currently safe because metals_loop sets `os.chdir(BASE_DIR)` at startup, but if anyone imports this from a different entry point (`python -m portfolio.oil_grid_signal` from some other CWD), it'll write to `<cwd>/data/oil_grid_signal.json` and the metals_loop will keep reading the old empty/stale value at the canonical path. **Fix:** `SIGNAL_FILE = str(Path(__file__).resolve().parent.parent / "data" / "oil_grid_signal.json")`.

### `data/fish_engine.py:120, 998-1000` | P2 | relative trade-log path + DEBUG silent failure
```python
TRADE_LOG = "data/fish_trades.jsonl"
...
try:
    atomic_append_jsonl(self._trade_log_path, entry)
except Exception:
    logger.debug("Failed to log trade to %s", ...)
```
**Cat:** silent failure on the strategy that lost 12 257 SEK on 2026-04-15. Even though FISH_ENGINE_ENABLED=False today, if it's ever re-enabled per the documented revival checklist, a logging failure inside a hot trade-execute path will be `DEBUG`-level invisible. **Fix:** when re-enabling fish_engine, escalate the except to `WARNING` and route to `critical_errors.jsonl` with `category="fish_engine_log_drop"`.

### `portfolio/grid_fisher.py:1655` stop_loss_price | P3 | stop_loss_price stale across reset
Stop retry uses `inst.stop_loss_price` which is the price set at original `rotate_on_buy_fill`. If the underlying has moved substantially since the initial buy fill, the retry places the stop at the *original* level — could be very near current bid or already past it. Probably wanted is "use original stop_loss_price unless bid moved significantly away, then re-anchor to current peak − stop_pct%". **Cat:** trailing-stop hygiene. **Fix:** at retry, also check `bid >= stop_loss_price * (1 + GRID_STOP_PCT/100 + safety)` and re-anchor if not.

---

## Notes (severity < 80, flagged for awareness)

### data/metals_warrant_catalog.json staleness
The cache file's `refreshed_ts` is **3 days old** (2026-05-21). TTL is 6h per `metals_warrant_refresh.py`. Code path: if refresh fails, falls back to stale catalog with WARNING. Acceptable today (barriers only change weekly), but if Avanza ever rebases a cert weekly, a 3-day-old `barrier` and `barrier_dist_pct` value could drive a buy decision past the new barrier. Suggest tightening behaviour: refuse to place on stale catalog when staleness > 24h, or force a Playwright probe to refresh.

### data/metals_loop.py "race" between 10s fast-tick and 60s cycle — NOT actually a race
The prompt cites lines 1407-1503 (silver fast-tick) racing the main cycle on module globals. I checked `_sleep_for_cycle` (lines 1023-1080): the fast-tick is invoked **serially inside the main cycle's sleep window** in the same thread. No `threading.Thread` use anywhere in metals_loop. There is no true race — but module-global mutation by `_silver_fast_tick` is still a code-smell because:
- A future addition of a worker thread would silently corrupt state.
- The shared `_silver_alerted_levels` set grows unbounded until session reset (also called from main, single-threaded today).

If multithreading is ever added, the buffer + alert set need a lock. For now: documentation comment, no code change.

### portfolio/microstructure_state.py | thread-safety is correct
The state buffers ARE protected by `_buffer_lock`. `get_microstructure_state` appends to history once per cycle via `persist_state`. The orderbook_flow signal reads via `load_persisted_state` (file-based, no shared memory). Clean. False-alarm noted in earlier review.

### portfolio/golddigger/runner.py:201 | 2 % stop sell buffer
`sell_price = round(stop_price * 0.98, 2)` — 2 % buffer below trigger. Better than 1 % but still tight on a 20x gold cert (cfg suggests `leverage = 20.0`). Suggest 3 %. Lower severity because GoldDigger has a software-side `check_stop_loss` (line 145) that fires a market-equivalent sell *before* the hardware stop fills — the hardware stop is just defence-in-depth for a process crash.

### data/metals_loop.py:413, 424, 1592 | relative paths under `os.chdir(BASE_DIR)`
The chdir at line 209 protects all the relative paths. Acceptable today but fragile against a future refactor that imports metals_loop modules into another loop with a different chdir.

### portfolio/ministral_signal.py + qwen3_signal.py | subprocess fallback timeout = 240s
240s timeout on a Layer 1 cycle (60s cadence) means a single LLM subprocess can stall the loop's ticker pool. Mitigated by `_invoke_lock` serialisation, but worth noting: if Plex is transcoding AND the llama-server died AND the cold-start fallback hangs for 240s, the entire metals signal pipeline misses 4 cycles. Memory `gpu_loop_reliability.md` says GPU models MUST complete every loop cycle — this falls short. Lower severity because Plex coordination + abstention sentinel already prevents the worst case.

### portfolio/grid_fisher.py:1419-1422 | `_order_delay_s = 0.6` between placements
6 placements / 10 instruments × tier x 1 cycle could stack up. `GRID_MAX_ORDERS_PER_MIN=10` rate-limit absorbs it. Just be aware: a full-restart that places 6 ladders (3 instruments × 2 tiers) costs 3.6 s of sleep inside one cycle. That overruns the 60 s budget combined with the grid tick's reconcile (1-2 s) plus all other work. Sub-P3.

### data/chronos_server.py:18 | hardcoded `Q:/finance-analyzer`
Acceptable for single-user single-host setup but flagged for future cross-machine portability.

---

## Verification of prior P0s from the prompt

| Prompt claim | Verified state | Comment |
|---|---|---|
| grid_fisher.py:1493 stop sell-price floor stop_price*0.995 = 0.5% (too tight) | **STILL PRESENT** at lines 1496 + 1655 | This review's top-5 #1 |
| fin_snipe_manager.py:537 HARD_STOP_CERT_PCT = 0.05 (now 0.03 per recent commit) | **STILL 0.05** | Prompt is incorrect — commit `be4273d3` only changed `MIN_STOP_DISTANCE_PCT` 1.0 → 3.0 |
| fin_snipe_manager HARD_STOP_SELL_BUFFER_PCT = 0.01 (1% below trigger, wicks bypass) | **STILL 0.01** | This review's top-5 #2 |
| metals_loop.py:1407-1503 silver fast-tick races slow loop on module globals | **NO RACE** | Single-threaded; fast-tick runs serially inside main-cycle sleep. Future code-hazard only |
| Layer 2 kill wedge (fixed batch 3) | Confirmed fixed in commit `81e0c78e` | Not re-reviewed here |
| grid_fisher stop-rearm (fixed batch 3 + P1 commit) | Mostly fixed in `81e0c78e` + `289e5030` | Retry path still uses 0.995 buffer (top-5 #1) and stale `stop_loss_price` (P3 above) |

---

## Recommended remediation order

1. **Today, before next metals trade**: top-5 items #1 + #2. Both are one-line constant changes plus regression tests.
2. **This week**: top-5 #3 (EOD fire-sale guard), #4 (metals_loop legacy ladder), #5 (layer2_invoke paths).
3. **Next sprint**: P2 items above, especially `place_stop_loss_no_page` ID extraction (silent stop-stacking) and the catalog staleness gate.
4. **Doc-only**: add a comment block in `_silver_fast_tick` reminding future contributors that the function is single-threaded today and that adding any worker thread requires a lock on the module globals.

End of review.
