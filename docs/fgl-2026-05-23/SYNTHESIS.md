# FGL Adversarial Review — Synthesis (2026-05-23)

Eight-subsystem empty-baseline adversarial pass on finance-analyzer at
`b006520a` (post auto-improvement merge). One review subagent per subsystem,
plus an independent self-review pass cross-critiquing the agent output.

Subsystem reports:
- [signals-core.md](signals-core.md) — 5 P0 / 7 P1 / 9 P2
- [orchestration.md](orchestration.md) — 4 P0 / 9 P1
- [portfolio-risk.md](portfolio-risk.md) — 4 P0 / 6 P1
- [metals-core.md](metals-core.md) — 5 P0 / 6 P1
- [avanza-api.md](avanza-api.md) — 5 P0 / 8 P1
- [signals-modules.md](signals-modules.md) — 1 P0 / 3 P1 / 2 P2
- [data-external.md](data-external.md) — 6 P0 / 8 P1
- [infrastructure.md](infrastructure.md) — 4 P0 / 6 P1

**Total raw counts:** 34 P0, 53 P1. After dedup + cross-critique below:
**~22 distinct P0** worth resolving, **~40 distinct P1**, plus a structural
backlog.

---

## Top P0 — Fix-Order (money / silent-failure)

Ordered by blast radius × likelihood, not by file. Highest first.

### 1. `kelly_metals.recommended_metals_size` size is 50–100× too large

`portfolio/kelly_metals.py:207-221`. Half-Kelly is divided by `cert_loss_pct/100`
which introduces a `100/avg_loss_pct` over-scaling. At typical inputs
(`p=0.52, win=3.09%, loss=2.43%, leverage=5`) the function returns ~0.58 of
buying power; the correct trade-percentage Kelly is ~0.008. The 0.95
`MAX_POSITION_FRACTION` cap is the only thing keeping the bot from
recommending all-in on every metals BUY. Consecutive-loss reduction kicks in
only after 4 in a row.

**Action:** rewrite using `f* = (p*win - q*loss)/(win*loss)`, divide by
leverage. Pin regression test at the typical-input band [0.005, 0.020].

### 2. `fin_snipe_manager` places stops 1% from bid — direct violation of 3% memory rule

`portfolio/fin_snipe_manager.py:64` sets `MIN_STOP_DISTANCE_PCT = 1.0`.
Memory invariant `feedback_mini_stoploss.md` + `.claude/rules/metals-avanza.md`
both say 3%. Sibling paths in metals_loop.py and `_update_stop_orders_for`
already enforce 3%. Single noise tick on a 5x cert can trip the stop
instantly into a thin bid → ~9% slippage.

**Action:** raise to 3.0, apply gate to the keep-existing-stop branch
(line 555-563) as well.

### 3. `grid_fisher.rotate_on_buy_fill` leaves naked position when stop-rearm fails

`portfolio/grid_fisher.py:1424-1522`. Sequence: (1) place new sell limit, (2)
cancel old stop, (3) place new stop. If step 3 returns `None` (timeout /
broker reject / auth glitch), `inst.stop_loss_id = new_stop_id` unconditionally
writes `None` and the position has *increased* inventory with no broker-side
stop. `place_buy_ladder` never re-arms, so the gap persists until next fill
event.

**Action:** don't overwrite `inst.stop_loss_id` when `new_stop_id is None`,
set `stop_needs_rearm` flag, attempt re-arm before next ladder placement,
log critical-error entry for fix-agent.

### 4. Multi-agent specialist failure cascades into "success" synthesis

`portfolio/agent_invocation.py:967-971`. `wait_for_specialists` returns
`{name: bool}` with auth-error detection overriding success=True → False.
But: `success_count = sum(1 for v in results.values() if v)` is computed
and not gated — synthesis runs even when 0/3 specialists succeeded. Layer
2 invocation row is logged `status="success"` (synthesis itself exits 0)
hiding three upstream auth errors. Re-opens the March–April silent auth
outage class.

**Action:** if `success_count == 0` (or `< quorum_floor`), skip synthesis
and short-circuit to `_log_trigger(..., "specialist_quorum_fail", ...)`
with critical_errors.jsonl entry.

### 5. `signal_engine.py:4205` accuracy gate config-overrideable below floor

```python
accuracy_gate = sig_cfg.get("accuracy_gate_threshold", ACCURACY_GATE_THRESHOLD)
```

`.claude/rules/signals.md` floor is 47% / 50% (tiered). No clamp on the
config-supplied value. YAML typo or a research experiment can promote
30%-accurate noise into the consensus.

**Action:** `accuracy_gate = max(ACCURACY_GATE_THRESHOLD, float(cfg.get(...)))`,
or raise `ConfigError` on values below the documented floor.

### 6. `signal_decay_alert.py` uses relative `"data/..."` paths

`portfolio/signal_decay_alert.py:27,148`. Same failure pattern as the
2026-05-02 ic_computation.py P0 fix. PF-OutcomeCheck Task Scheduler launch
CWD ≠ repo root → `load_json` returns `None` → `check_signal_decay`
returns `[]` → operator sees "no decay detected" while detector is
silently broken. Same silent-failure class as the March–April auth outage.

**Action:** mirror `ic_computation.py:25`:
```python
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ACCURACY_CACHE_FILE = DATA_DIR / "accuracy_cache.json"
DECAY_ALERTS_FILE = DATA_DIR / "signal_decay_alerts.jsonl"
```

### 7. `api_utils.get_binance_config` reads `apiKey`, validator requires `key`

`portfolio/api_utils.py:60` reads `ex.get("apiKey", "")`. `config_validator.py`
documents creds live under `exchange.key`. Every signed-API caller silently
gets `("", "")`. Currently latent (only unsigned spot/klines used) but the
first futures-account read will look like 401-rate-limited rather than
mis-configured. Silent-failure class.

**Action:** change `apiKey` → `key`. Add startup assertion that
`get_binance_config()` returns non-empty creds.

### 8. `connors_rsi2` ticker guard absorbs `context=` into `**kwargs`

`portfolio/signals/connors_rsi2.py:103`. Function signature is
`(df, ticker="", **kwargs)`. Registry sets `requires_context=True` so
dispatcher calls `compute_fn(df, context=context_data)`. `context=`
falls into `**kwargs`; `ticker` stays `""`; crypto-only guard at line 110
short-circuits to False. Signal computes on every ticker including
XAU/XAG/MSTR. Currently in DISABLED_SIGNALS / shadow → doesn't vote
but DOES emit shadow predictions polluting per-ticker accuracy used by
the promotion pipeline.

**Action:** signature → `(df, context=None)`, `ticker = (context or {}).get("ticker", "")`.

### 9. Promoted shadow signal silently HOLDs forever

`portfolio/signal_engine.py:3601-3685`. `_promoted_override = True` lets a
DISABLED signal pass the disabled-gate, then the throttle at 3663-3685
reads `get_status(sig_name) == "shadow"` and force-HOLDs. If shadow_registry
has both `promoted=true` and `status="shadow"` (the promotion script
doesn't atomically prevent), the signal "is promoted" but never votes.
Only trace is `_throttled=True` in extra_info.

**Action:** in the throttle block, if `_promoted_override`, set
`_throttle_skip = False`. Or fix the promotion workflow to atomically
flip `status` to `"active"` and assert
`not (status == "shadow" and promoted)`.

### 10. Drawdown circuit breaker mixes historical vs current FX rates

`portfolio/risk_management.py:270-317` + `561-624`. `log_portfolio_value`
writes `patient_value_sek` using the FX rate at write-time; `check_drawdown`
compares today's SEK value (today's FX) against the cached SEK peak. An 8%
USD/SEK move alone (well within the 7–15 sanity band, ±20%) can trip or hide
the 20% gate. `_streaming_max` is monotonic, so a SEK weakening run inflates
peaks permanently.

**Action:** record USD values alongside SEK in history, compute drawdown on
USD. Or store FX rate per row and recompute on the fly. Or emit dual breakers
(SEK + USD) trip if either crosses.

### 11. Trigger price baseline stales across multi-day quiet periods

`portfolio/trigger.py:496-509`. `state["last"]["prices"]` is rewritten ONLY
when `triggered=True`. After a 24h+ quiet period (e.g. accuracy gate stuck
HOLD), a normal 2% intraday move fires as a "moved X% up" trigger comparing
against price from 24h ago. False positives wear the Layer 2 token budget
during otherwise quiet markets.

**Action:** refresh price baseline every cycle when no consensus changes,
keep `last_trigger_time` updated only on actual triggers.

### 12. Layer 2 kill+wait wedge — `_agent_proc` never cleared

`portfolio/agent_invocation.py:647-682`. `_kill_overrun_agent` keeps
`_agent_proc` non-None when `wait(15)` times out even after a successful
`taskkill /F`. `_agent_proc.poll() is None` then forever (Popen tracks the
dead PID), so future `invoke_agent` calls see "already running" and refuse
to spawn. Layer 2 permanently wedged until loop restart.

**Action:** if `taskkill` returned 0/128 and `wait(15)` times out, force
`_agent_proc = None` anyway — the OS has already killed the process, the
Popen object's internal handle is stale. Log critical_errors entry.

### 13. `portfolio/avanza/trading.place_order` lacks whitelist + size cap + lock

`portfolio/avanza/trading.py:38-102`. Session path enforces three guards
(ALLOWED_ACCOUNT_IDS, MAX_ORDER_TOTAL_SEK=50_000, avanza_order_lock). The
unified-package path enforces NONE. `__init__.py` re-exports `place_order`,
`place_stop_loss`, `modify_order`, `cancel_order` — any new caller importing
from `portfolio.avanza` bypasses every guard the ISK-only invariant depends
on.

**Action:** create `portfolio/avanza/_guards.py` with the whitelist + size
cap + lock; have ALL four order paths (session/client/control/trading) call
the shared guard.

### 14. `avanza/account.get_buying_power` returns zero on miss (regresses Bug C7)

`portfolio/avanza/account.py:87-94`. On account-not-found returns
`AccountCash(buying_power=0.0, total_value=0.0, own_capital=0.0)` with only
a WARNING log. Callers doing `available = cash.total_value - locked` will
read `0.0` and conclude "no positions, free to allocate elsewhere" —
silently wrong. session-path version (avanza_session.py:385-539) was
explicitly fixed for this in 2026-04-09. Also only tries `accountId`/`id`;
session path tries `accountId`/`id`/`accountNumber`/`number`.

**Action:** return `None` on miss; try all four ID-field shapes; share
the lookup helper between the two paths.

### 15. `avanza/account.get_positions(account_id=None)` leaks pension account

`portfolio/avanza/account.py:27-61`. Default returns every position across
every account — including pension `2674244`. `portfolio/avanza_client.get_positions`
hard-filters; new package path does not. New dashboard / report code calling
`from portfolio.avanza import get_positions` will leak pension holdings,
breaking `memory/feedback_isk_only.md`.

**Action:** default-filter to module-level `ALLOWED_ACCOUNT_IDS`, require
explicit kwarg to bypass. Same for `get_transactions`.

### 16. EOD 21:55 hardcoded, ignores `todayClosingTime` + DST

`portfolio/grid_fisher.py:277-279`, `data/metals_loop.py:1574,2032`. Avanza
warrant close time is DST-dependent; `.claude/rules/metals-avanza.md` says
query `todayClosingTime` from the API. Hardcoded constant misses by 1h
across DST transitions = positions held past close OR liquidated early.

**Action:** query `/_api/market-guide/<inst>/<ob_id>` once per session,
persist `eod_local_time` to grid_fisher state. Fall back to constant only
when the API call fails.

### 17. Loop crash inside 21:50–21:55 EOD window leaves inventory unswept

`data/metals_loop.py:7585-7591` + `7850-7872`. No `atexit`/signal handler
runs `grid_fisher.eod_market_flat()` + `_eod_sell_fishing_positions()`.
Crash inside the EOD window → overnight inventory exposure breaks the
"EOD-flat" guarantee in CLAUDE.md.

**Action:** register `atexit` + SIGTERM handler that calls the EOD sweep
during the window. Persist a "EOD pending sweep" flag so next process boot
can recover.

### 18. `data_collector._fetch_klines` yfinance path unlocked for direct callers

`portfolio/data_collector.py:265,280-294`. The yfinance call at line 265
runs outside `_yfinance_lock` for callers that bypass `_fetch_one_timeframe`
(5+ callers including golddigger, oil_grid_signal). yfinance is documented
thread-unsafe (cookie/session race), 8-worker ThreadPoolExecutor → silent
data corruption / empty DataFrames in past incidents.

**Action:** move lock acquisition into `_fetch_klines` (or into
`yfinance_klines` itself) so ALL paths are serialized.

### 19. `fx_rates.fetch_usd_sek` silently returns stale rate

`portfolio/fx_rates.py:28-71`. Cached return value is indistinguishable
from a fresh one. Staleness only logs + 4h-throttled Telegram. Callers
performing portfolio valuation cannot refuse stale rates. A 12h-stale FX
during a real SEK shock would mis-value the entire portfolio. Violates
live-prices-first rule.

**Action:** return `(rate, is_stale, age_seconds)` tuple (or sibling
`fetch_usd_sek_with_metadata()`). Have `risk_management._resolve_fx_rate`
refuse rates older than N hours.

### 20. `signal_engine.py:4019,4252` core-gate uses pre-persistence votes

`core_active = sum(1 for s in CORE_SIGNAL_NAMES if votes.get(s) == "BUY|SELL")`
is computed on the **unfiltered** `votes` dict. `_apply_persistence_filter`
mutes votes that haven't held for 2 cycles. If the filter flushes the entire
core slate, `core_active >= 1` is still True (pre-filter), and weighted
consensus runs on enhanced-only votes — breaching the documented contract
"enhanced signals can strengthen/weaken but never create consensus alone."

**Action:** compute `core_active_post_persistence` from `consensus_votes`,
gate against that at line 4252.

### 21. Warrant P&L ignores knockout barrier — drawdown breaker reads paper recovery

`portfolio/warrant_portfolio.py:52-113`. No `financing_level` / `barrier`
fields. Model lets warrant value go negative through the barrier and
"recover" if underlying rallies back. MINI certs are barriered: cross →
zero permanent. `risk_management.check_drawdown` reads the wrong value.
`exit_optimizer.Position` has the correct `financing_level` + `_compute_pnl_sek`
floor at 0, but that path doesn't feed the drawdown breaker.

**Action:** extend holding schema with `financing_level_usd` +
`knockout_observed_ts`; floor `current_implied_sek` at 0; mark dead
permanently when underlying historically crossed the barrier.

### 22. `journal.load_recent` + `journal_index.retrieve_relevant_entries` linear-scan full journal every Tier 2/3

`portfolio/journal.py:23-40` + `journal_index.py:351-399`. 1,282 lines today,
~50K projected after a year (60-day rotation × ~800/day). Linear parse +
ISO parse on every Layer 2 invocation; BM25 indexer rebuilds from scratch
every call. Today ~100-500ms overhead; in 6-12 months → seconds, eating
T1 (180s) budget.

**Action:** switch `load_recent` to `load_jsonl_tail` (already exists).
Cap `retrieve_relevant_entries` indexed window at `entries[-2000:]`. Cache
the fitted `BM25` between invocations, rebuild only when journal mtime
changes.

---

## High-impact P1s — Worth fixing this week

Grouped, not numbered, because order within P1 is less important.

**Concurrency / races:**
- `microstructure_state.persist_state` not lock-protected at iteration
  (data-external P1-7).
- `warrant_portfolio.record_warrant_transaction` no lock; concurrent
  metals_loop + fin_snipe calls can lose writes (portfolio-risk P1-2).
- Patient/Bold `load_state()` + `save_state()` direct callers bypass the
  new `update_state` lock (portfolio-risk P1-1).
- `grid_fisher.cancel_armed_buys` fall-through on `cancel_failed` leaves
  ZOMBIE ARMED tier forever (metals-core P1-4).
- `grid_fisher.arm_direction` direction-flip can leak orders if the cancel
  is best-effort (metals-core P1-5).

**Stop-loss + EOD:**
- `grid_fisher.eod_market_flat` uses `bid*0.99` (only 1% slippage budget),
  falls back to `avg_entry_price` (wildly wrong for deep losers) — leaks
  positions overnight on illiquid days (metals-core P1-2).
- `_handle_buy_fill` legacy fallback places stops with NO 3% gate; dead
  path right now, single config flip re-enables (metals-core P0-5).
- `cancel_all_stop_losses_for` only matches `orderbook.id`, ignores
  `orderBookId` / `orderbookId` Avanza shape drift (avanza-api P1-2).
- `place_trailing_stop` packing `trail_percent` into `trigger_price` +
  `sell_price=0` only safe because of `BUG-223` guard in the session
  path; unified path has NO equivalent guard, a `MONETARY`+`sell_price=0`
  call goes straight to broker as a market-sell (avanza-api P0-2).

**Auth / silent failure class:**
- `verify_session` returns False on 5xx, calls `close_playwright()`,
  forces TOTP fallback for the rest of the process — TOTP path doesn't
  share `ALLOWED_ACCOUNT_IDS` enforcement (avanza-api P0-5).
- `health.heartbeat()` writes fresh `last_heartbeat` even when the main
  loop is wedged — daemon thread ticks independently of cycle progress.
  Watchdog never fires on a hung main thread (infrastructure P1-1).
- `_check_telegram_confirm` doesn't reject bot senders (avanza-api P1-4).

**Data / signal correctness:**
- `accuracy_stats.py:937-943` fabricates `correct` count as
  `int(round(blended * max(at_samples, rc_samples)))` — fake number
  presented to dashboard / `--accuracy` / Layer 2 as if it were a real
  success count (signals-core P1-5).
- `statistical_jump_regime` regime transition: opposing-direction jump
  doesn't reset count → wrong regime locked in (signals-modules P1.1).
- `news_event` has structurally-SELL bias (no BUY branch in
  `_keyword_severity_vote`, `_sentiment_shift` defaults moderate-severity
  to negative) (signals-modules P1.2).
- `funding_rate` BUY at -0.01% vs SELL at +0.03% — 3x more sensitive on
  BUY side, structural long bias (data-external P1-8).
- `econ_dates.is_macro_window` hardcodes 14:00 UTC for ALL event types;
  CPI/NFP release at 12:30/13:30 UTC, FOMC at 18:00/19:00 UTC. Window
  offset by 1–6 hours from reality (data-external P1-4).
- `sentiment.fetch_newsapi_with_tracking` undercounts quota on empty
  responses → 429s before local counter realizes (data-external P1-1).
- `onchain_data._fetch_all_onchain` persists partial dict on partial
  failure, 12h TTL prevents retry of failed metrics (data-external P1-2).
- `futures_data.get_open_interest` reads `data["openInterest"]` without
  `.get()` → KeyError on malformed Binance response → circuit breaker
  doesn't fire → loop retries every cycle burning quota (data-external P1-3).

**Atomic I/O / journaling:**
- `data/metals_loop.py:1978-1996` raw `open() + json.load()` for
  `metals_warrant_catalog.json` — violates project rule #4
  (metals-core P1-1).
- `process_lock.acquire_lock_file` writes metadata AFTER acquiring lock
  — windowed corruption if reader peeks between (infrastructure P1-2).
- `file_utils.atomic_write_json` directory fsync gap on Windows — docstring
  claims "guarantees durability on power loss" but `os.replace` is
  metadata-journaled separately (infrastructure P0-2 cross-listed).

---

## Cross-Critique — Where Agents Got It Wrong / Overstated

For honesty I list places where I disagree with my subagents or where the
finding needs scoping:

1. **infrastructure agent P0-2 (directory fsync gap)** — Marked P0; I think
   P1. On NTFS the rename is metadata-journaled so the failure window is
   tiny. Real Windows boxes use a UPS or accept the risk. Docstring should
   be loosened (not a P0).

2. **portfolio-risk agent P0-4 ("trailing stop is static")** — Module
   docstring vs. implementation mismatch is real, but `compute_stop_levels`
   is called from a context that re-runs every cycle on the current
   `entry_price = avg_cost`. The stop never trails because the input
   never changes — but it ALSO never gets wider on a profit-run. P1
   (missing feature), not P0 (broken safety).

3. **signals-core agent P0-5 (SignalDB `check_same_thread`)** — Latent,
   not active. Module-level `SignalDB()` doesn't exist today. P2/backlog,
   not P0.

4. **signals-modules agent P0.1 (connors_rsi2)** — Confidence 95 is right
   but impact statement overstates: signal is in shadow + DISABLED,
   doesn't vote in consensus. The pollution is to the promotion pipeline's
   accuracy data (still a real bug, still worth fixing). P0 because of the
   class-of-bug (dispatcher contract drift goes unnoticed), not the
   immediate trading impact.

5. **MIN_VOTERS_METALS=2 vs documented MIN_VOTERS=3 rule** — Not flagged
   by signals-core agent. I checked: documented + rationalized in code
   (line 1016-1020). Not a violation.

6. **metals-core agent P1-2 (`bid*0.99` only 1% slippage)** — The 1% might
   be intentional to favor non-fill (re-list at deeper level) over forced
   bad fill. Worth re-evaluating but not obviously wrong.

7. **data-external agent P0-3 (Deribit silent kill on `len(parts)!=4`)**
   — Real silent failure, but the affected signal (`crypto_evrp`,
   `crypto_macro`) currently ACTIVE → P0 is correct.

8. **orchestration agent P0-3 (Popen exception leaves stale `_agent_start`)**
   — Real but small blast radius: subsequent invocation hits the existing
   reentrancy block at line 761 which catches `_agent_proc != None`. P1.

---

## Structural Findings (not in any specific subsystem)

These don't fit a single P0/P1 line but are worth flagging.

### `signal_engine.py` is 4,416 lines — god file

That's roughly 10× the typical Python module size. The accuracy gate logic,
the dispatch loop, the persistence filter, the dynamic correlation
computation, the per-ticker disabled lists, the dead-zone helpers, the
weighted consensus math — every change here has a 4-of-100 chance of
silently breaking one of the other concerns. The 4 P0s found in this file
during the review (P0-2, P0-3, P0-4, plus the swallowed `is_promoted`
exception at 3602-3606) all relate to the per-signal dispatch state machine
being too complex to read in one sitting.

**Backlog:** carve out `signal_engine/dispatch.py`,
`signal_engine/accuracy_gate.py`, `signal_engine/persistence.py`,
`signal_engine/correlation.py`, `signal_engine/consensus.py`.

### Three parallel Avanza order paths with diverging guards

`portfolio/avanza_session.py` (Playwright), `portfolio/avanza_client.py`
(TOTP `avanza-api`), `portfolio/avanza_control.py` (cmd-and-control RPC),
and now `portfolio/avanza/` (unified typed package). Each has its own
guard set; the typed package's guards are the weakest. Order-lock + size
cap + account whitelist need to be a SHARED dependency, not copied four
times. The avanza-api P0s (P0-1 to P0-4) all stem from this fragmentation.

**Backlog:** one `portfolio/avanza/_guards.py` module, all four paths
import it. Add CI assertion that every public `place_*`/`modify_*`/
`cancel_*` function references the guards module.

### Manual `try/except: pass` masking inside cyclic dispatch

Counted 18 `except (...): pass` blocks across signal_engine.py,
agent_invocation.py, grid_fisher.py, accuracy_stats.py, equity_curve.py,
etc. (`grep -nE "except [^:]*: *(\\n[\\s]+)pass"`). Of those:
- 2 are correct (best-effort cleanup with no caller observability).
- 11 swallow real errors that would help debug silent failures.

The `is_promoted` swallow at `signal_engine.py:3602-3606` is the cleanest
example: if shadow_registry.json is corrupted, promotion silently reverts
to disabled with no telemetry. The 3-week silent auth outage was the same
class — exit 0 + no telemetry.

**Backlog:** narrow each `except` to a specific exception class and add
a `logger.debug(...)` minimum.

### CLAUDE.md drift

`credit_spread_risk` is listed in the 17 ACTIVE signals but `tickers.DISABLED_SIGNALS`
has it disabled since 2026-05-21. Will compound: next time someone reads
CLAUDE.md for a Layer 2 prompt, they'll reason about a signal that doesn't
vote. Mentioned in signals-modules.md.

**Backlog:** one-line fix to CLAUDE.md to reflect the current 16-active set.

---

## What the Loop is Currently Doing Right

For balance: every subagent except metals-core noted the active core path is
substantially hardened. Specifically:

- `file_utils.atomic_write_json` + `atomic_append_jsonl` are real (the small
  directory-fsync gap notwithstanding) and used by ~20 JSONL writers; no
  raw `json.dump` to portfolio state.
- Auth-error detection (`claude_gate.detect_auth_failure`) is robust against
  the echo / feedback-loop class that the 2026-04-16 echo fix added.
- Accuracy data has been hardened against poisoned numeric fields (Codex
  rounds 10/11/12/13 documented in `signal_engine.py:2233-2255`).
- Drawdown breaker has NaN/Inf fail-safe (`risk_management.py:288-303`).
- `valid_actions` validation prevents arbitrary action strings from
  entering the vote (`_validate_signal_result`).
- Multi-agent specialist auth-error scan EXISTS (just not gated to block
  synthesis — the bug above is one missing branch, not absent telemetry).
- Singleton lock is cross-platform now (msvcrt + fcntl) — no silent
  no-op on non-Windows.
- 11 bug fixes from this morning's auto-improvement session are already
  in main (commit `b006520a`).

The system is closer to "complex with edges" than "fragile". The P0s above
are real and worth fixing, but none of them indicate a structural
miscarriage — they're the residue of a fast-moving codebase.

---

## Methodology / Reproducibility

- One Claude `pr-review-toolkit:code-reviewer` subagent per subsystem,
  spawned in parallel (background) from main thread.
- Each subagent given: file scope, project context excerpts from CLAUDE.md,
  relevant `.claude/rules/*.md` invariants, instruction to find P0/P1/P2/P3
  with file:line citations.
- Main thread ran independent grep/Read pass while subagents worked, then
  cross-critiqued each output against verifying reads of cited files.
- Worktree: `Q:\finance-analyzer-fgl-2026-05-23` on branch
  `review/fgl-2026-05-23`. No code modifications; review docs only.
- Cleanup: docs committed to `main`, branch + worktree removed.

Commit SHA reviewed: `b006520a` (post auto-improvement merge 2026-05-23).
