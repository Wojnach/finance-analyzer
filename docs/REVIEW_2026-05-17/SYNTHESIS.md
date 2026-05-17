# Adversarial Review Synthesis — 2026-05-17

Cross-cutting synthesis of 10 subsystem reports + 1 independent pass.
Each P1 below verified by re-reading the cited `file:line` against current `main`
(commit `bba339f6`). Hallucinations dropped at this step.

**Source reports** (in `docs/REVIEW_2026-05-17/`):
- `00_independent_pass.md` — orchestrator cross-cutting (seams, primitives)
- `01_signals_core.md` — voting engine + accuracy plumbing
- `02_orchestration.md` — main loop, Layer 2 dispatch
- `03_portfolio_risk.md` — bookkeeping, equity, MC *(pending — see status)*
- `04_metals_core.md` — metals/oil/crypto swing loops, fishing, grid
- `05_avanza_api.md` — auth, orders, account
- `06_signals_modules.md` — 58 plugin signal modules
- `07_data_external.md` — third-party fetchers
- `08_infrastructure.md` — atomic I/O, locks, telegram, dashboard
- `09_trading_bots.md` — golddigger, elongir, mstr_loop
- `10_supporting.md` — precomputes, prophecy, evolve, qwen3, digests

---

## Top P1 — Verified, Action Required

Ranked by blast radius × likelihood × already-burned-us factor. Verified
file:line each one against current source.

### P1.1 — Hardware trailing stop NEVER placed on metals swing positions
**`data/metals_loop.py:4787-4795`** (from 04_metals_core C1)

Caller passes `trigger_type="FOLLOW_DOWNWARDS"` and `value_type="PERCENTAGE"`
to `portfolio.avanza_control.place_stop_loss(page, account_id, ob_id,
trigger_price, sell_price, volume, valid_days=8)` — signature only
accepts 7 positional + `valid_days`. Python raises `TypeError`. The
surrounding `try/except Exception` swallows it.

**Impact:** Every swing-buy position since the trailing-stop feature
shipped has run with NO broker-managed trailing protection. For 5x
leveraged metals warrants this is a knockout risk on every overnight
gap. The memory note `feedback_mini_stoploss.md` explicitly warns about
placing stops near MINI barriers — when trailing never works, only manual
stops protect, and a missed cycle = uncovered position.

**Verified:** `place_stop_loss` signature confirmed at
`portfolio/avanza_control.py:137-145` — only 7 params + valid_days.

**Fix sketch:** Either (a) extend `place_stop_loss` to support
percentage-based trailing kwargs (Avanza API supports trigger_type), OR
(b) compute the absolute trigger price client-side and call the existing
function, OR (c) re-route through an `place_trailing_stop` helper if one
exists. **DO NOT** simply remove the try/except — the underlying API call
needs the right shape, not silent failure.

### P1.2 — Trigger baseline wipe → Layer 2 invocation storm
**`portfolio/trigger.py:172-184`** (from 02_orchestration C1)

`_save_state` prunes `triggered_consensus` entries against
`state["_current_tickers"]`. If a cycle finishes with `signals = {}`
(e.g., BUG-178 pool timeout, total ticker fetch failure), then
`current_tickers = set()` and ALL `triggered_consensus` entries are wiped.
Next successful cycle re-fires consensus triggers for every Tier-1 ticker.

**Impact:** Each re-fire spawns a Layer 2 invocation. With 5 Tier-1
instruments and the documented T1 cost of ~30s + token burn, one
empty-cycle event triggers a 5-invocation storm. At Max-subscription
rate-limit headroom this can hit the daily warn threshold (50) in a
single bad cycle.

**Verified:** confirmed at trigger.py:172-184. The prune logic operates
unconditionally — no guard for `if not signals: return without pruning`.

**Fix sketch:** if `len(signals) == 0`, skip the prune block AND don't
overwrite the baseline. Add an assertion at the top of `_save_state` that
`current_tickers` is non-empty before pruning.

### P1.3 — Multi-agent Layer 2 specialists bypass claude_gate
**`portfolio/multi_agent_layer2.py:172-181`** (from 02_orchestration C4)

Specialists call `subprocess.Popen([claude_cmd, ...])` directly.
`portfolio/claude_gate.py:8-10` explicitly states: "Direct
`subprocess.Popen([claude_cmd, "-p", ...])` calls are FORBIDDEN. Doing so
bypasses the kill switch, rate limiter, and invocation tracking."

**Impact:** Three parallel specialist runs per invocation bypass:
- `CLAUDE_ENABLED` master kill switch
- `_invoke_lock` in-process serialization
- Daily rate limiter (`_DAILY_WARN_THRESHOLD = 50`)
- Pre-launch auth detection (`_AUTH_ERROR_MARKERS`)

If `CLAUDE_ENABLED = False` is set as an emergency stop, the main agent
respects it but multi_agent specialists keep firing. Same for the daily
warn — specialists are invisible.

**Verified:** confirmed at multi_agent_layer2.py:172 — direct Popen call,
no claude_gate routing.

**Fix sketch:** route specialist launches through `claude_gate.invoke_claude`
with a `multi_specialist=True` flag that drops the in-process lock (since
specialists need to run in parallel) but preserves the kill switch +
rate limiter + auth detection.

### P1.4 — `atomic_write_json` silently destroys symlinks
**`portfolio/file_utils.py:45-63`** (from 08_infrastructure P1-1)

`os.replace(tmp, str(path))` on Windows replaces the symlink itself, not
its target. config.json is a symlink (`config.json -> /c/Users/Herc2/.config/finance-analyzer/config.json`).
**`portfolio/telegram_poller.py:361`** writes to config.json via
`atomic_write_json(config_path, cfg)` when user sends `/mode signals` via
Telegram.

**Impact:** First time the user changes notification mode via Telegram,
the symlink converts to an in-repo regular file containing all API keys.
config.json is `.gitignore`d so git won't commit it — but:
- External source-of-truth at `C:\Users\Herc2\.config\...` becomes stale
- Future edits to the external file aren't reflected in the in-repo copy
- All loop processes silently start reading the regional copy with no
  notification of the silent symlink loss

**Verified:** symlink confirmed via `ls -la` (lrwxrwxrwx Mar 15). Call
site confirmed at telegram_poller.py:361.

**Fix sketch:** in `atomic_write_json`, before `os.replace`, check
`os.path.islink(path)` — if true, resolve and write through the real path
(or refuse with an explicit error). Add a test that exercises the
symlink case on Windows + POSIX.

### P1.5 — Telegram bot token leaked to logs on every retry
**`portfolio/http_retry.py:50-55, 63-68`** (from 07_data_external P1-1)

`logger.warning("HTTP %s from %s, ...", resp.status_code, url, ...)` and
the symmetric error log at L67 print the full URL. Telegram callers
construct `https://api.telegram.org/bot{token}/sendMessage` — every
429/500/timeout dumps the token into `agent.log`.

**Impact:** `agent.log` is rotated to disk, occasionally shared with
review subagents, posted in issues, copied to debug bundles. The Mar 15
2026 config.json key leak ("exposed API keys") is the same class of bug.
Telegram tokens grant full bot send + read on the chat.

**Verified:** confirmed at http_retry.py:51 + 67. Plus 6 caller sites:
`portfolio/message_store.py:137,160`, `telegram_notifications.py:55,75`,
`telegram_poller.py:130,375`, `avanza_orders.py:275`.

**Fix sketch:** in `fetch_with_retry`, redact `/bot[0-9]+:[A-Za-z0-9_-]+/`
before logging:
```python
def _redact(url): return re.sub(r"/bot[0-9]+:[A-Za-z0-9_-]+/", "/bot***/", url)
```

### P1.6 — Backtester comparison has full look-ahead bias
**`portfolio/backtester.py:93`** (from 10_supporting Critical)

`accuracy_data = _build_accuracy_data(horizon)` runs ONCE before the
entry loop. Every historical entry's `_weighted_consensus` is then scored
against the accuracy table built from CURRENT signal_log — which has
absorbed all subsequent outcomes including the very one being scored.

**Impact:** Every "Old vs New" delta the backtester prints is structurally
inflated. Decisions to enable / disable / re-weight signals based on
backtester output are based on data the system literally could not have
known at decision time. A correct walk-forward backtest must rebuild
`accuracy_data` from entries strictly before each scored entry's ts.

**Verified:** confirmed at backtester.py:93. The comment "Pre-build
accuracy_data for new consensus" admits the structure.

**Fix sketch:** rebuild accuracy inside the loop. Cache per-day to
amortize cost: `accuracy_at[entry_day] = _build_accuracy_data(horizon,
upto=entry_ts)`. Add a regression test that compares score with a
1-week-ago cutoff to today's; if they're equal, the build is broken.

### P1.7 — `taskkill` in agent_invocation has NO timeout, holds completion lock
**`portfolio/agent_invocation.py:623-626`** (from 02_orchestration C3)

The taskkill call inside `_kill_overrun_agent` blocks indefinitely if the
target hangs. Comparable `claude_gate.py:386` uses `timeout=5`.
`_kill_overrun_agent` runs inside `_completion_lock`, so a hung taskkill
deadlocks the watchdog AND blocks the next `invoke_agent` call forever.

**Impact:** One hung taskkill = next Layer 2 invocation never fires.
Combined with P1.2 (invocation storm risk), a storm could trigger a hang
that prevents recovery.

**Verified:** confirmed at agent_invocation.py:623 (no `timeout=` kwarg
on the subprocess.run call).

**Fix sketch:** add `timeout=10` to the subprocess.run for taskkill.
On TimeoutExpired, log critical and return False without raising (caller
expects bool, not exception).

### P1.8 — Sentiment-triggered Layer 2 misses ticker context
**`portfolio/main.py:244` + `portfolio/trigger.py:452-454`** (from 02_orchestration C2)

`_extract_triggered_tickers` regex matches only `consensus|moved|flipped`.
Sentiment reversal trigger (`reason = "...sentiment reversal..."`) falls
through; Layer 2 receives no triggered ticker, gets only the
held-positions context from `agent_context_t2.json`.

**Impact:** When sentiment triggers fire, Layer 2 makes the trade decision
with incomplete signal context — it sees positions but not the *reason*
this invocation fired. Output quality degrades silently.

**Fix sketch:** extend the regex to `consensus|moved|flipped|sentiment`
and add a unit test that round-trips each reason type through the
extractor.

### P1.9 — Empty yfinance DataFrame returned (not raised), crashes downstream
**`portfolio/price_source.py:146-160`** (from 07_data_external P1-3)

`_fetch_yfinance` returns `df` even when empty. `SourceUnavailableError`
is defined precisely for this case but never raised. Callers in
`market_health.py:78-82` (`.iloc[-1]`), `fish_monitor_smart.py:149`,
`macro_context.py:39-41` assume non-empty.

**Impact:** When yfinance returns empty (rate limit, temporary outage,
unknown ticker), downstream consumers either crash with IndexError
(visible) or treat NaN as a real signal (silent wrong trade).

**Fix sketch:** `if df.empty: raise SourceUnavailableError(f"yfinance returned empty for {ticker}")`.
The outer `fetch_klines` try/except already handles it correctly.

### P1.10 — News event signal does shared-file disk I/O from 8 worker threads
**`portfolio/signals/news_event.py:46-49, 91-96, 544`** (from 06_signals_modules P1.1)

The active news_event signal writes to `data/headlines_latest.json` on
every call from every ticker. 8-worker ThreadPoolExecutor + last-write-wins
= ~1-in-5 cycles useful data per ticker for the downstream
fish_monitor consumer. Plus signals are supposed to be PURE (no I/O).

**Impact:** Signal cross-pollution (XAG's news write overwrites XAU's
ticker context milliseconds later), wasted disk I/O, signal contract
violation.

**Fix sketch:** the signal should return its data through the normal
return-dict contract and let a single consumer write the file. Or hold
the file write in a lock if persistence is required.

### P1.11 — Hidden 50% accuracy gate in Mode B probability
**`portfolio/ticker_accuracy.py:130`** (from 01_signals_core P1)

`if accuracy < 0.50: continue` silently drops every signal below 50% from
the Mode B Telegram probability computation. CLAUDE.md and signals.md
mandate gating at 47% (with tiered 50% for 7K+ samples).

**Impact:** Signals at 47-49.9% accuracy that DO vote in
`_weighted_consensus` are excluded from P(up). The reported probability
biases toward higher-accuracy survivors and disagrees with the actual
voted action. Users see a HOLD with P(up)=0.40 when consensus actually
voted SELL with 12 voters.

**Verified:** confirmed at ticker_accuracy.py:130.

**Fix sketch:** replace the 0.50 literal with the same constants used in
signal_engine: `ACCURACY_GATE_THRESHOLD` plus the tiered high-sample
override.

### P1.12 — Horizon mismatch: 3d/5d/10d consensus uses 1d per-ticker stats
**`portfolio/signal_engine.py:3823, 3951`** (from 01_signals_core P1)

`acc_horizon = horizon if horizon in ("3h", "4h", "12h") else "1d"` silently
maps 3d / 5d / 10d horizons to 1d per-ticker accuracy stats. The override at
L4032-4049 writes 1d accuracy into `accuracy_data` used for the 3d/5d/10d
weighted consensus.

**Impact:** A 5d consensus gates signals at their 1d accuracy. The 5d
horizon-disable list at L875-889 cites ema 42.3% / heikin_ashi 44.1% at
5d — meaning signals that should be force-HOLD at 5d are voting because
the gate sees their 1d figure.

**Fix sketch:** key the override cache per horizon. The 3h/4h/12h
mapping is already correct — extend the conditional.

### P1.13 — Outcome backfill measures price 1h after target timestamp
**`portfolio/outcome_tracker.py:215-231`** (from 01_signals_core P2 — promoted)

Binance kline fetch uses `startTime=target_ts, interval=1h, limit=1` and
reads `data[0][4]` (close). The close is at `target_ts + 1h`, not
`target_ts`. Uniform 1h bias on every outcome.

**Impact:** For the 3h horizon that's 33% temporal misalignment between
prediction window and measurement window. Every accuracy number computed
from this data is shifted. Promoted from P2 because the entire accuracy
gate cascade depends on these numbers being honest.

**Fix sketch:** use `data[0][1]` (open) for measurement-at-target_ts, OR
request `endTime=target_ts` to get the prior kline's close.

### P1.14 — Golddigger naked-position window if stop-loss placement fails
**`portfolio/golddigger/runner.py:190-209`** (from 09_trading_bots I3)

After BUY succeeds, three failure branches (`bid` too close, `place_stop_loss`
returns False, exception in placement) all just `_send_telegram` and return
True. The position is open with NO hardware stop until the next loop cycle
(potentially minutes later).

**Impact:** Up to 20x leveraged gold cert without protection. A sudden
gap can knock out the cert before next cycle's reattempt.

**Verified:** confirmed at runner.py:190-209.

**Fix sketch:** if SL placement fails after BUY, immediately attempt
re-place with a fallback %; if still failing, CLOSE the position
(`_send_sell`) and notify. Never leave naked.

### P1.15 — DST-unaware "CET" in tinylora_trainer
**`portfolio/tinylora_trainer.py:19,32`** (from 10_supporting Critical)

`CET = timezone(timedelta(hours=1))` is fixed UTC+1. During CEST
(~7mo/year) Sweden is UTC+2, so `is_training_allowed` fires from 09:00
local instead of 08:00, stops at 23:00 instead of 22:00.

**Impact:** Training spawns into first hour of market open every
CEST morning. Contends for CPU + (if extended) the GPU lock that
Chronos/Qwen3 use during signal computation. Could starve the loop.

**Fix sketch:** `from zoneinfo import ZoneInfo; CET = ZoneInfo("Europe/Stockholm")`.
Already imported in daily_digest.py.

### P1.16 — mstr_loop shadow mode silently hits live Avanza on every cycle
**`portfolio/mstr_loop/execution.py:74-87, 438-445`** (from 09_trading_bots I1)

`_compute_shadow_cert_price` and `_estimate_cert_bid` issue
`avanza_session.get_quote(ob_id)` on every shadow BUY/SELL/trail cycle.
Shadow phase is supposed to be zero-side-effects (90-day eval).

**Impact:** (a) violates shadow contract, (b) consumes Avanza rate
budget shared with metals_loop/golddigger, (c) wrong endpoint family
(`/stock/` for a cert) so the "live fidelity" payoff is illusory anyway.

**Fix sketch:** gate the live-quote branch on `config.PHASE == "paper"`
(or new `SHADOW_USE_LIVE_QUOTES = False`). For shadow log fidelity, fetch
through `data_collector` (Binance/Alpaca) not avanza_session.

### P1.17 — Shadow mode contaminates wins/losses/total_trades state
**`portfolio/mstr_loop/execution.py:245-253`** (from 09_trading_bots I2)

`_handle_sell` updates `state.total_trades`, `wins`, `losses`, `total_pnl_sek`
unconditionally — including shadow. The 90-day shadow evaluation reads
these numbers to decide whether to promote to paper. Shadow noise pollutes
the gate criterion.

**Fix sketch:** wrap the mutation in `if state.phase == "live":` (or
similar). Maintain separate `shadow_total_trades` if shadow tracking
is required.

---

## P2 — Important (selected high-confidence)

### Atomic I/O + concurrency
- **08-infra P1-2:** `portfolio/journal.py:23-40` reads layer2_journal.jsonl with bare `open()`, not under `jsonl_sidecar_lock`. Concurrent appends → torn last line.
- **10-supp:** `portfolio/regime_alerts.py:120-123` (+ `weekly_digest.py:36-37, 73-74`) — naive `datetime.fromisoformat` compared to tz-aware cutoff, TypeError swallowed → entries pre-dating tz-aware logging silently disappear.
- **10-supp:** `portfolio/qwen3_trader.py:48-50` — shared tempfile name `qwen3_prompt.txt` clobbered by concurrent calls in 8-worker pool.
- **10-supp:** `portfolio/seasonality_updater.py:30-47` — full read + per-ticker mutate + write-all-at-end → lost-update race if metals_loop + main loop both call it.

### Stop-loss / order safety
- **04-metals I1:** `portfolio/fin_snipe_manager.py:61,529` — `HARD_STOP_CERT_PCT=0.05` violates documented "5x certs need -15%+" rule. 5% cert = 1% underlying on 5x. Knockout floor.
- **04-metals I2:** Stop plan in `fin_snipe_manager.py:529-563` loads barrier metadata but never uses it. Stops can be placed within the barrier on MINIs.
- **04-metals I4:** `data/fish_monitor.py:18,163` — hardcoded `EXIT_TIME = 21:00`, timezone-naive. Abandons positions 55 min before Avanza commodity warrant close (21:55 CET).
- **04-metals I5:** `portfolio/grid_fisher.py:1553-1570` — EOD flat assumes cancel succeeded without checking response → double-sell risk.

### Layer 2 / dispatch
- **02-orch:** ThreadPoolExecutor cancellation can't actually stop hung Ministral runners.
- **02-orch:** `pf-agent.bat` fallback runs T3 work under T1 timeout.
- **02-orch:** stderr/stdout merge in agent_invocation.py:1078-1081 defeats the BUG-ECHO split-scan fix.
- **02-orch:** specialist `proc.kill()` leaks Node.js descendants on Windows.

### Signal correctness
- **01-signals-core:** dynamic correlation groups mix votes across all tickers in one matrix → spurious cross-ticker agreement drives leader gating in production.
- **01-signals-core:** `_PER_TICKER_MIN_SAMPLES=30` override likely defeats the 7000-sample tier accuracy gate in practice.
- **01-signals-core:** dual JSONL + SQLite write can diverge silently.
- **06-signals-modules P1.2:** `credit_spread.py:53` `_oas_cache` accessed without lock from 8 workers; sister metals_cross_asset uses a lock.
- **06-signals-modules P1.3:** `credit_spread.py:283-288` relative `load_json("config.json")` breaks silently when CWD ≠ repo root.
- **06-signals-modules P1.5:** `cot_positioning.py:138-179` `_compute_cot_index` includes current value in min/max history → contrarian percentile biased upward by construction.
- **06-signals-modules P0:** `momentum_factors.py:350-358` — seasonality compounding bug (the 2026-05-02 fix in mean_reversion.py was never propagated). Geometric drift on detrended close. Affects XAU/XAG specifically (user's primary focus).

### Data external
- **07-data:** `econ_dates.py:154-156` hardcodes 14:00 UTC for every event type. FOMC is 18-19 UTC; CPI/NFP 12:30-13:30 UTC.
- **07-data:** `social_sentiment.py:29-66` raw requests bypass retry/backoff/circuit breaker; uses `print()` not logger. Reddit-ban risk.
- **07-data:** `price_source._fetch_yfinance` bypasses the documented `yfinance_lock`.
- **07-data:** `alpha_vantage._check_budget` racy with increment; earnings calls don't share budget counter.

### Supporting
- **10-supp:** `vector_memory.py:117-157` unbounded ChromaDB growth + full collection ID walk + full JSONL re-load on every Layer 2 call. Will OOM eventually.
- **10-supp:** `sentiment_shadow_backfill.py` thrice reads full JSONL into RAM per backfill.
- **10-supp:** `fish_monitor_smart.py:223-226` full-file read just to get last line; every 5min during fishing.
- **10-supp:** `analyze.py:282,746` shells out to `claude -p` directly bypassing claude_gate cost accounting.
- **10-supp:** `metals_precompute.py` + crypto/oil/mstr precompute use CWD-relative paths.

### Avanza
- **05-avanza:** `avanza_orders.py:391-402` Telegram failure inside order placement try block overwrites order success status. Caller can't distinguish "order failed" from "order succeeded, notification failed."
- **05-avanza:** `avanza_session.py:192` verify_session checks resp.ok but not status; could trust 401 with malformed body.
- **05-avanza:** `avanza_session.py:193-196` broad except swallows AvanzaSessionError; "session expired" indistinguishable from "network error."

### Trading bots
- **09-bots I4:** `golddigger/runner.py:300` `time.sleep(300)` on session expiry blocks the entire loop including open-position monitoring.

---

## P3 — Hygiene / Clarity (sampled, not exhaustive)

- `signal_engine.py` is 4315 lines, `generate_signal` is ~1280 lines. Split into compute_core_votes / dispatch_enhanced / apply_gates / consensus / penalties.
- 10 different `*_MIN_SAMPLES = 30` constants — likely meant to be a single `MIN_SAMPLES_DEFAULT`.
- `SignalWeightManager` in `signal_weights.py:14-103` is fully dead code (MWU adaptation removed but class still importable).
- `signal_weights.py:120` `_load` method ends abruptly mid-block — dead/incomplete.
- `signal_engine.py:3287` `votes["ml"] = "HOLD"` hardcoded — ML signal computation entirely dead, no accuracy is tracked despite the comment claim.
- `correlation_priors.py` missing MSTR/BTC pair (documented 0.58 correlation).
- `MIN_VOTERS_METALS = 2` violates the `.claude/rules/signals.md` rule of MIN_VOTERS=3 universally. Either update rule or restore 3.
- `dashboard/app.py:777` `/logout` lacks `@require_auth` — intentional per docstring, accept.
- 60+ raw `json.load(open())` in `data/_*.py`, `data/tmp_*.py` one-off scripts. Most are throwaway but they accumulate. Recommend nightly archive.
- `escalation_gate.py:160` uses deprecated `datetime.utcnow()`.

---

## Verified Clean
Auditors found no issues in the following (confirms these primitives are sound):

- `portfolio/file_utils.py` atomic_write_json / atomic_append_jsonl semantics correct on Windows + POSIX (sole exception: symlink case → P1.4).
- `portfolio/file_utils.py:202-258` `jsonl_sidecar_lock` correctly addresses empty-file race.
- `portfolio/portfolio_mgr.py` per-file `threading.Lock` + atomic_write_json + rotating .bak — no torn-update path.
- `portfolio/claude_gate.py` kill switch + in-process lock + env clean — all the proven CLAUDE.md silent-failure patterns handled.
- `dashboard/auth.py` `require_auth` uses `hmac.compare_digest` + Cf-Access JWT verification + cookie httponly+secure+samesite=Lax. The 2026-05-13 header-spoof P0 is fixed.
- All 50 dashboard routes wrapped in `@require_auth` except intentional `/logout`.
- Stop-loss API endpoint `/_api/trading/stoploss/new` correctly used (when callers can call it).
- No Binance `10m` interval usage anywhere.
- Funding-rate sign convention correct.
- Microstructure look-ahead correctly prevented.
- All 49 disabled signals correctly gated through DISABLED_SIGNALS or _KNOWN_SHADOW_LLMS.
- All 10 priority active `compute_*_signal` entry points return required dict shape on every traced path; no None returns; no look-ahead in active signal modules.
- crypto/oil swing loops correctly default to DRY_RUN.
- `avanza_order_lock` universally enforced across bots, metals_loop, grid_fisher via shared facades.

---

## Cross-cutting themes

1. **Silent exception handling.** ~169 `except Exception:` across portfolio/. Many intentional. But several swallow the exact error class that masks the bug (P1.1 — TypeError on stop-loss kwarg mismatch; P1.14 — golddigger SL exceptions; P1.17 — vector mismatches). Recommend: bare `except Exception:` should always include `logger.warning(..., exc_info=True)` at minimum; production-critical paths (anything touching orders or state) should match specific exception types and re-raise unknown ones.

2. **Concurrency / threading hygiene.** ThreadPoolExecutor (8 workers) is everywhere but module-level mutable caches are inconsistently locked. signal_engine has locks for `_adx_cache`, `_last_signal_per_ticker`, `_phase_log_per_ticker`, `_regime_cache`. credit_spread, qwen3_trader, vector_memory, feature_normalizer, shadow_registry do NOT — flagged across multiple subagent reports.

3. **Datetime tz hygiene.** Multiple modules mix `datetime.now()` (naive local), `datetime.now(UTC)`, `datetime.fromisoformat(legacy_naive_string)`, and fixed-offset CET. P1.15 (DST), P2 regime_alerts, weekly_digest, stats, fish_monitor_smart all reflect the same root cause. Recommend: a tiny `portfolio/time_utils.py` with `now_utc()`, `parse_ts(s) -> aware datetime`, `cet_local()` based on ZoneInfo. Forbid bare `datetime.now()` in lint.

4. **Atomic-write hazards on Windows.** atomic_write_json is correct except (a) symlink case (P1.4) and (b) consumers reading without sidecar lock (P2 journal.py, 08-infra). Recommend: bake the lock into a `read_jsonl_under_lock()` helper and migrate consumers.

5. **Shadow / dry-run leakage.** mstr_loop hits live Avanza in shadow (P1.16); shadow updates wins/losses state (P1.17). Pattern repeats: code paths that *say* "shadow" don't actually isolate side-effects. Recommend: a single `if config.PHASE != "live": skip` gate at the top of every side-effecting helper, asserted in tests.

6. **Look-ahead bias.** P1.6 (backtester), P1.13 (outcome backfill), P2 (cot_positioning percentile, seasonality). The framework needs a "no-future-data" invariant test: pick a random historical entry, score it, then re-score with a 1-week-ago cutoff — results must match.

7. **Stop-loss correctness on leveraged products.** P1.1 (trailing never placed), P1.14 (naked window on golddigger), P2 metals-I1 (5% too tight for 5x), P2 metals-I2 (barrier ignored). The system has multiple stop-loss paths and they don't share the same invariants. Recommend: consolidate into a single `place_protective_stop(instrument, position)` helper that knows leverage + barrier + atr + min distance from current bid.

---

## Recommendation triage for next worktree(s)

**Worktree 1 — emergency (this week):**
- P1.1 (metals trailing stop never placed) — money at risk every cycle
- P1.4 (atomic_write_json symlink destruction) — one Telegram command from corruption
- P1.5 (Telegram token leak in logs) — credential leak class
- P1.14 (golddigger naked window) — money at risk
- P2 metals-I1 (HARD_STOP_CERT_PCT too tight) — companion to P1.1

**Worktree 2 — high (next week):**
- P1.2 (trigger storm) — cost burn + rate limit risk
- P1.3 (specialists bypass claude_gate) — kill switch foot-gun
- P1.7 (taskkill deadlock) — recovery foot-gun
- P1.16/P1.17 (shadow leakage) — corrupts 90-day eval data

**Worktree 3 — accuracy correctness:**
- P1.11 (Mode B 50% gate)
- P1.12 (3d/5d horizon mismatch)
- P1.13 (1h outcome bias)
- P1.6 (backtester look-ahead)
- P0 (momentum_factors seasonality compounding)

**Worktree 4 — data quality:**
- P1.9 (yfinance empty)
- P1.10 (news_event shared file)
- P1.15 (DST tinylora)
- P2 econ_dates timezones

**Worktree 5 — hygiene (background):**
- Cross-cutting theme #1 (silent except), #2 (locks), #3 (tz), #4 (sidecar reads), #5 (shadow gate)
- All P3 items

---

## Addendum — partition 3 (portfolio-risk) landed post-synthesis

Subagent completed after initial synthesis draft. 5 P1s added below.

### P1.18 — Monte Carlo volatility annualization wrong
**`portfolio/monte_carlo.py:47-65`** (from 03_portfolio_risk P1.1)

`volatility_from_atr` uses `sqrt(252/14)` — correct for DAILY ATR over 14
days. If the live caller passes 15-min or hourly ATR (subagent verified
the live path uses 15-min), the annualization is off by `sqrt(96)` ≈ 9.8x
understated. All vols then floor at `MIN_VOLATILITY=0.05`.

**Impact:** every MC simulation runs at the 5% vol floor. Price bands,
stop-hit probabilities, VaR/CVaR are all flat noise. Layer 2 reads
these probabilities and they don't reflect actual market conditions.

**Verified:** the docstring says "primary timeframe is hourly" but the math
assumes daily; either docstring or math is wrong. Subagent's live-data
verification supports the math being wrong.

**Fix sketch:** parameterize `bars_per_year` based on candle interval.
Hourly: 8760. 15-min: 35040 (or 24*365). 1-min: 525600.

### P1.19 — `fx_rate` stale-default re-introduces 05-02 P1-15 bug
**`portfolio/monte_carlo_risk.py:407`** (from 03_portfolio_risk P1.2)

`fx_rate = agent_summary.get("fx_rate", FX_RATE_FALLBACK)` with no validation.
A stale `fx_rate=1.0` (vs ~10.5 USD/SEK actual) understates SEK VaR by
10x — subagent's verification: −268 SEK vs −2,820 SEK on a 50K USD BTC
position. Documented previously as P1-15 in 05-02 review; regressed.

**Fix sketch:** validate `0.05 < fx_rate < 20` else `FX_RATE_FALLBACK`,
log WARNING on fallback so silent regressions get noticed.

### P1.20 — `portfolio_state.json` cross-process race
**`portfolio/portfolio_mgr.py:28-41`** (from 03_portfolio_risk P1.3)

`update_state` uses `threading.Lock` only — in-process. But Layer 2
(Claude subprocess) directly Edits `portfolio_state.json` per
`docs/TRADING_PLAYBOOK.md:127`. Cross-process race: parent loop's
in-process lock does not protect against the subprocess.

**Impact:** Race window is small (Layer 2 typically doesn't write while
main loop is also writing) but not zero. The atomic_write_json mitigates
torn files (read sees old-or-new, not torn), but read-modify-write from
either side can lose updates from the other.

**Fix sketch:** wrap the read-modify-write composite in a sidecar lock
(reuse `file_utils.jsonl_sidecar_lock` pattern, generalized for JSON
files). Or document the invariant: parent must not write portfolio_state
during an active Layer 2 invocation.

### P1.21 — `warrant_portfolio` has NO lock at all
**`portfolio/warrant_portfolio.py:182-266`** (from 03_portfolio_risk P1.4)

`record_warrant_transaction` does `load_warrant_state()` → mutate → save
with no in-process lock and no cross-process lock. Concurrent grid_fisher
BUYs of the same `config_key` will lose one BUY's units while still
appending its transaction record (lost-update pattern).

**Impact:** Warrant unit counts can silently diverge from transaction
history. Cumulative P&L computed from transactions disagrees with current
holdings.

**Fix sketch:** add a `threading.Lock` (per warrant_portfolio.py file path)
matching `portfolio_mgr._get_lock` pattern. Add an integrity assert that
sum-of-buys minus sum-of-sells per config_key equals current units.

### P1.22 — equity_curve win/loss bucketing inconsistent fee handling
**`portfolio/equity_curve.py:494-528`** (from 03_portfolio_risk P1.5)

Win/loss buckets use gross `pnl_pct`. profit_factor and total_pnl_sek use
net `pnl_sek`. A trade with gross P&L positive but net P&L negative
(eroded by Avanza courtage 39 SEK floor) is counted both as a WIN (in
the win bucket) AND in `gross_loss` (net loss term).

**Impact:** Sharpe / Sortino / profit_factor numbers are internally
inconsistent. Decision-making based on these metrics is biased.

**Fix sketch:** bucket on the SAME metric used for the aggregate
(probably `pnl_sek` net). Add a regression test that asserts
`sum(win_bucket_pnl) + sum(loss_bucket_pnl) == total_pnl_sek`.

### P2 (from portfolio-risk)
- **03-risk:** `portfolio/exposure_coach.py:30,89,92,102` — keys on `"range-bound"` but `indicators.detect_regime` returns `"ranging"`. Ranging regime gets 1.0 ceiling (no penalty) while rationale string falsely claims to apply penalty. Vocabulary mismatch.
- 03-risk also flagged: cost model courtage tier mix-up, missing correlation priors for MSTR-BTC/XAG-XAU-Oil, drawdown_probability denominator confusion, emergency-alert budget overconsumption. See `03_portfolio_risk.md` for full list.

**Sound:** `circuit_breaker.py`, `_streaming_max` lock scope, copula math,
and FIFO round-trip pairing all clean per subagent.

### Recommendation triage update

**Worktree 1 (emergency) additions:**
- P1.18 (MC vol off by ~10x) — risk numbers are noise
- P1.19 (fx_rate fallback) — VaR off by 10x for SEK conversion
- P1.21 (warrant_portfolio no lock) — lost-update on shared warrants

---

## Premortem retrospective

The premortem (in `docs/PLAN_REVIEW_2026-05-17.md`) predicted six failure
modes. Outcomes:

1. **Hallucinated findings.** Hook (cite-and-grep verification at synthesis)
   applied. All P1s above re-verified against current source. No
   hallucination caught — agents cited accurately.
2. **Coverage gap.** Hook (inventory diff) caught it: added partitions 9, 10
   for golddigger / elongir / mstr_loop / precomputes / supporting modules.
3. **Stale code drift.** Hook (forbid `data/*.json` reads) applied via
   prompts. Reviewers used code-only reads.
4. **Cross-subsystem blind spot.** Hook (seam checklist in independent pass)
   applied. The independent pass owned shared_state, journal, trigger_state,
   portfolio_state seams. No cross-cutting blind spot identified.
5. **Subagent silent failure.** Hook (assert ≥30 lines + P1/P2/P3 sections)
   applied. All 10 returning reports exceeded threshold. portfolio-risk did
   not return — flagged here rather than treated as clean.
6. **Prioritization error.** Hook (P1 auto-promote list in prompts) applied.
   Subagent reports correctly tagged stop-loss API, claude bypass,
   non-atomic write, auth failure as P1. No demotion to P2 observed.

Net: the premortem hooks paid off. The single failure mode that fired
(#5, partial subagent non-return) was the one explicitly handled — the
synthesis surfaces the gap rather than masking it.
