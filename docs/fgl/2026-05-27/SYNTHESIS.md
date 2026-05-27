# FGL Adversarial Review — Synthesis (2026-05-27)

Eight-subsystem empty-baseline adversarial pass on finance-analyzer at
`03b5cb3f` (post auto-improvement merge). One Claude Code review subagent
per subsystem (`pr-review-toolkit:code-reviewer`) plus an independent
main-thread pass cross-cutting the subsystem boundaries.

Previous round: `docs/fgl-2026-05-23/SYNTHESIS.md` (34 P0, 53 P1). This
round runs against the codebase AFTER those fixes landed (commits
2026-05-23 → 2026-05-26).

## Subsystem reports

| Subsystem | P0 | P1 | P2 | P3 | File |
|-----------|----|----|----|----|------|
| signals-core | 3 | 9 | 6 | 4 | [signals-core-findings.md](signals-core-findings.md) |
| orchestration | 4 | 7 | 6 | 3 | [orchestration-findings.md](orchestration-findings.md) |
| portfolio-risk | 2 | 6 | 5 | 3 | [portfolio-risk-findings.md](portfolio-risk-findings.md) |
| metals-core | 3 | 6 | 4 | 2 | [metals-core-findings.md](metals-core-findings.md) |
| avanza-api | 3 | 6 | 5 | — | [avanza-api-findings.md](avanza-api-findings.md) |
| signals-modules | 1 | 7 | 6 | 4 | [signals-modules-findings.md](signals-modules-findings.md) |
| data-external | — | 7 | 9 | 6 | [data-external-findings.md](data-external-findings.md) |
| infrastructure | 1 | 7 | 8 | 4 | [infrastructure-findings.md](infrastructure-findings.md) |
| independent (cross-cutting) | — | 1 | 4 | 1 | [independent-findings.md](independent-findings.md) |
| **Total raw counts** | **17** | **56** | **53** | **27** | |

After dedup + cross-critique below: **~15 distinct P0** worth resolving
this round, **~40 distinct P1**, plus a structural backlog.

---

## Top P0 — Fix-Order (money / silent-failure / observability)

Ordered by blast-radius × likelihood. Sources cited so the synthesis
is auditable against the underlying reports.

### 1. Stage-4 `_dynamic_min_voters_for_regime` silently defeats `MIN_VOTERS_METALS = 2` in ranging regime

**Source:** signals-core P0 #2 (`signal_engine.py:2008-2027` + `3067-3073` + `4090`)

`MIN_VOTERS_METALS = 2` was added 2026-05-11 to fix "metals 3-voter floor
produced 0 trades in 20 days." Downstream `apply_confidence_penalties`
Stage-4 calls `_dynamic_min_voters_for_regime(regime)` which returns 5
for `ranging` / `unknown` / `None` — the most common metals regime per
the codebase commentary. Result: the metals carve-out is a no-op in the
dominant regime; the bug it was written to fix is unfixed; the user's
*primary focus* (per CLAUDE.md and the silver_prophecy memo) emits zero
trades during ranging tape regardless of signal strength.

**Action:** Make `_dynamic_min_voters_for_regime` asset-class aware OR
short-circuit Stage-4 for tickers whose configured `min_voters` is below
the dynamic floor. Regression test: 2-voter metals consensus in ranging
regime must emit BUY/SELL not HOLD.

### 2. Drawdown circuit breaker blinded by a single missing ticker price

**Source:** portfolio-risk P0 #1 (`risk_management.py:201-212`)

`_compute_portfolio_value` falls back to `pos.get("avg_cost_usd", 0)`
when `signals.get(ticker, {}).get("price_usd")` is missing/zero. The
position is then valued at *entry* price, not market. If silver craters
30% but the XAG signal block is stale for one cycle (FAPI hiccup), the
portfolio value looks flat, peak-cache never updates downward, and the
50% drawdown block in `agent_invocation` *cannot* trip during the exact
data-stale conditions where it most needs to fire. Compounded by
`_peak_cache` being monotonically non-decreasing: a mis-valued spike
during a stale-price window becomes the all-time peak forever.

**Action:** Drop the `avg_cost_usd` fallback. Propagate `stale_value=True`
and fail-safe (block downstream) OR substitute last-good market price
from `portfolio_value_history.jsonl` with an age cap.

### 3. `monte_carlo_risk` and `exit_optimizer` bypass the 2026-05-02 FX-fallback gate

**Source:** portfolio-risk P0 #2 (`monte_carlo_risk.py:408`,
`exit_optimizer.py:718`)

The P1-15 fix in `risk_management.py` added `_resolve_fx_rate()` that
rejects the legacy `1.0` literal via the sanity band [7, 15]. Two
production paths still call `agent_summary.get("fx_rate", DEFAULT)`
directly. A stale `fx_rate: 1.0` understates SEK valuations by ~10x in
those modules — exactly the failure mode the P1-15 fix was meant to
eliminate.

**Action:** Route both call-sites through `_resolve_fx_rate`. Search
for `agent_summary.get("fx_rate"` across `portfolio/` and audit every
hit; promote `_resolve_fx_rate` to `portfolio/fx_rates.py` so neither
module has to import `risk_management`.

### 4. `metals_avanza_helpers.place_order` accepts caller-supplied `account_id` with no allow-list check

**Source:** avanza-api P0 #1 (`data/metals_avanza_helpers.py:253` +
`portfolio/avanza_control.py:130` + `place_stop_loss`)

The page-based `place_order` wrapper used by GoldDigger / iskbets paths
does NOT enforce `ALLOWED_ACCOUNT_IDS`. The `_place_order` in
`avanza_session.py` does. A caller bug — or a config typo — that
propagates the pension `account_id` (2674244) through this wrapper
will place real orders on the pension account. The `feedback_isk_only`
memory is the only thing standing between this code path and a
filed-and-settled pension trade.

**Action:** Mirror `_place_order`'s allow-list guard in
`metals_avanza_helpers.place_order` and `avanza_control.place_order` AND
in `place_stop_loss`. Add startup assertion that the only account ids
in any wrapper's call path are 1625505.

### 5. `place_stoploss_once.api_delete` missing `account_id`; 404 treated as success

**Source:** avanza-api P0 #2 (`data/place_stoploss_once.py:115`)

`api_delete(f"/_api/trading/stoploss/{stop_id}")` omits the account
parameter. Combined with `avanza_session:377` treating 404 as success,
the "delete old, place new" cascade silently double-encumbers the
position (old stop still alive on the server side; new stop placed on
top). Documented in `ADVERSARIAL_REVIEW_2026-04-29.md` and
`PLAN_avanza_followups_20260502.md`; not yet fixed.

**Action:** Add the explicit account id to the delete URL. Treat 404 as
failure, not success, in the delete path (success requires confirmed
removal).

### 6. `avanza_client._place_order` TOTP path bypasses ceiling/min guards

**Source:** avanza-api P0 #3 (`portfolio/avanza_client.py:327`)

The TOTP-confirm flow used by `avanza_orders.py` (Telegram-confirm
window) does NOT enforce the BUG-211 50K SEK ceiling and the H8
min-1000-SEK guard that the canonical `avanza_session._place_order`
enforces. A confirmed-via-Telegram order can exceed the per-order cap
that the other path enforces — defeating the cap entirely for the path
the user actually uses for size confirmation.

**Action:** Apply the same ceiling/min check in the TOTP path. Extract
the guard into a `_validate_order_size` helper so all order paths
share one source of truth.

### 7. `escalation_gate` single-worker pool wedges permanently on first runner hang

**Source:** orchestration P0 #1 (`escalation_gate.py:32, 203-210`)

`_RUNNER_EXECUTOR = ThreadPoolExecutor(max_workers=1)`; on
`_fut.result(timeout=10)` TimeoutError, `_fut.cancel()` is a no-op
because the task has already started. The single worker thread stays
inside `query_llama_server` forever (no inner timeout). Every
subsequent `should_escalate` call queues behind the wedged worker,
times out, and fails open — silently escalating every trigger to
Claude. Multi-week silent cost overrun once the first Ministral hang
fires.

**Action:** Per-call daemon thread + result queue with `queue.get(
timeout=10)`. Defense-in-depth: hard inner timeout on
`query_llama_server`. On the third consecutive `runner_timeout`, open
a module-level circuit-breaker for N minutes that returns a fixed
`(escalate=True, reason="ministral_circuit_open")`.

### 8. Multi-agent specialist `proc.kill()` is single-PID — Node.js helpers orphan on Windows

**Source:** orchestration P0 #2 (`multi_agent_layer2.py:180-191, 218-221`)

Popen without `creationflags=CREATE_NEW_PROCESS_GROUP`; on timeout
`proc.kill()` kills only the direct Node child. MCP servers, the
actual API client process, and any local-LLM helpers Claude spawned
stay alive as orphans. This is the *exact* failure mode that
`claude_gate.A-IN-2 (2026-04-11)` fixed for the main agent path —
the fix was never propagated to `multi_agent_layer2`.

**Action:** Reuse `claude_gate._popen_kwargs_for_tree_kill()` and
`_kill_process_tree()` here. One-line import + two-line call-site
change.

### 9. `_agent_log_start_offset` set outside `_completion_lock` — wrong slice scanned for auth errors

**Source:** orchestration P0 #3 (`agent_invocation.py:1116-1117`)

The byte offset that the watchdog uses to scan agent.log for auth
markers is captured *outside* `_completion_lock`. A watchdog tick
between two consecutive `invoke_agent` calls in the unhappy path can
read the NEW offset while scanning the OLD subprocess's log — either
finding nothing (offset past EOF) or finding a fake marker in another
invocation's slice. Reopens the March-April silent auth outage class
in a more subtle way.

**Action:** Move the entire spawn block (lines 1110-1232) inside
`_completion_lock`. The lock is already a serialisation point by
design; the larger critical section is harmless.

### 10. `invoke_agent` returns `None` on `specialist_quorum_fail` — main.py mis-logs as `skipped_busy`; status missing from completion-rate metric

**Source:** orchestration P0 #4 (`agent_invocation.py:1046`)

Bare `return` after the quorum-fail trigger log writes `None`. Caller
in `main.py:951` treats `None` as falsy and writes a SECOND
`_log_trigger` row with status `skipped_busy_<why>`. Both
`specialist_quorum_fail` and `skipped_busy_*` are missing from
`get_completion_stats.tracked_statuses` so the dashboard
`/api/loop_health` undercounts the failure as zero. Operators don't
see Claude budget being burned on quorum failures.

**Action:** Change line 1046 from `return` to `return False`. Add
`specialist_quorum_fail` to the `tracked_statuses` tuple. Route the
critical_errors append through `claude_gate.record_critical_error`
for dedup.

### 11. `llama_server._start_server` spawns `llama-server.exe` without a Job Object — orphan on parent crash

**Source:** infrastructure P0 #1 (`portfolio/llama_server.py:456`)

Heaviest subprocess in the system (~5 GB VRAM, exclusive port 8787),
spawned with bare `subprocess.Popen`. Every other expensive child uses
`subprocess_utils.popen_in_job()` so the OS auto-kills on parent
death. The orphan reaper at `subprocess_utils.kill_orphaned_llama`
matches the *legacy* name `llama-completion.exe` only — it will never
find a stranded `llama-server.exe`. Hard parent kill leaves the LLM
holding port + VRAM + GPU lock indefinitely.

**Action:** Use `subprocess_utils.popen_in_job()`. Update
`kill_orphaned_llama` to also match `llama-server.exe`, OR add
`kill_orphaned_llama_server` and call it on every loop startup.

### 12. `_with_browser_recovery` retries mutating POSTs — duplicate orders on transient browser death

**Source:** metals-core P0 #1 (`portfolio/avanza_session.py:212-232`)

`_with_browser_recovery` wraps every `api_post` including
`order/new` and `stoploss/new`. On `TargetClosedError` / browser-dead
detection, it tears down Playwright and **retries the same POST**.
Browser-dead errors cannot distinguish "browser died before POST sent"
from "POST landed at Avanza but response read failed." In the latter
case, Avanza creates the order, the retry creates a second identical
order, and the caller logs only the retry's `orderId`. Up to 2×
position size on a single user-intended order — bypasses every per-
instrument and per-strategy notional cap. Worst case at 50K SEK ceiling
= 100K SEK warrant exposure on a single signal.

**Action:** Pass an `idempotent: bool` flag from `api_get` (True) and
`api_post` (False) into `_with_browser_recovery`. Skip retry for
mutating endpoints — or after relaunch, read open orders to detect
whether the original request actually landed before retrying.

### 13. Swing-trader stop-loss anchor never validated against warrant barrier

**Source:** metals-core P0 #2 (`data/metals_swing_trader.py:2714-2786`)

`_set_stop_loss` computes `trigger_price = stop_anchor * (1 - sl_pct/100)`.
It does NOT consult the warrant barrier. On a 5x LONG MINI XAG with
barrier=30.5 and underlying=33: a 30% warrant drop ≈ 6% underlying →
underlying at 31.02. A second 1.7% tick wipes the warrant. The stop
never fires because it's below the post-knockout price (=0).
`MIN_BARRIER_DISTANCE_PCT = 10` is enforced only at `_select_warrant`
— once a warrant is in inventory, broker stop placement is barrier-
blind. Directly contradicts `memory/feedback_mini_stoploss.md` which
is RED-FLAGGED in MEMORY.md.

**Action:** In `_set_stop_loss`, after computing `trigger_price`,
convert back to implied underlying via leverage, compute
`barrier_distance_pct`, require ≥ 5%. If breached, raise the stop
towards entry so the implied underlying sits at `barrier × (1 +
buffer/100)` for LONG. Telegram critical alert if no safe stop fits.

### 14. EOD swing exits permanently disabled — overnight gap risk on every swing trade

**Source:** metals-core P0 #3 (`data/metals_swing_config.py:323`)

`EOD_EXIT_MINUTES_BEFORE = 0` with comment "REVERT to 25 after current
position closes" (dated 2026-04-13). The consumer at
`metals_swing_trader.py:3073` checks `if minutes_to_close <=
EOD_EXIT_MINUTES_BEFORE` — so EOD exit only fires *past* close,
effectively never. Combined with `MAX_HOLD_HOURS = 24`, every swing
position carries overnight gap risk. User explicitly: *"Does NOT want
to hold warrants for a full day"*. A silver gap-down on Monday open
wipes 30-50% of a 5x position before any stop can react during EU
pre-open.

**Action:** Set `EOD_EXIT_MINUTES_BEFORE = 25`. Add a config-drift
guard: Telegram alert when the value becomes 0 again.

### 15. `_apply_persistence_filter` loses all state on process restart — 2-cycle stocks confirmation silently violated

**Source:** signals-core P0 #3 (`signal_engine.py:598-688`)

`_persistence_state` is a module-level dict with no disk persistence.
Every restart (auto-restart-on-crash, code merge, daily Task Scheduler
relaunch) wipes it. First cycle for each ticker hits the cold-start
branch and returns votes unfiltered — even for stocks where
`min_cycles=2`. The "stocks need 2 consecutive same-direction votes"
policy is silently bypassed on every restart. Combined with
exponential backoff crash recovery, every crash window emits one
"trusted" cycle of unfiltered votes before the filter re-engages.

**Action:** Persist `_persistence_state` via `atomic_write_json` after
every update (batched once per cycle like sentiment state). Load on
first call. Alternative: treat first-cycle votes as `cycles=1` for
stocks so they still need a confirmation cycle before voting.

---

## High-impact P1 (sample, not exhaustive)

Compact one-liners; full detail in the subsystem files.

### Concurrency / atomic-I/O gaps

- **`health.py` last-write-wins cross-process race** between main loop and metals loop on `health_state.json` (infrastructure P1 — `_health_lock` is in-process; both processes do read-modify-write of the same file via `atomic_write_json`).
- **`prophecy.py` read-modify-write race**: all mutators do `load → mutate → save` with no lock; concurrent Layer 2 + metals can silently drop belief updates. (infrastructure P1)
- **`journal.py:load_recent` reads `layer2_journal.jsonl` with raw `open()` — no sidecar lock.** Concurrent rotation `os.replace` mid-iteration produces torn reads. (infrastructure P1)
- **`signal_history.jsonl` lock is process-local** but ≥5 long-running processes (PF-DataLoop, PF-MetalsLoop, PF-CryptoLoop, PF-MstrLoop, PF-OilLoop) all write the file. (signals-core P1)
- **Persisted monotonic clock across process restarts** breaks `trigger.py` sustained-flip duration gate. (independent P1)
- **`_PROMOTED_CACHE` cache race** in `shadow_registry.is_promoted` — no lock around TTL-check-and-write; 8-worker × 80-signal hot path = duplicate disk reads under contention. (signals-core P1)

### Silent-failure / observability

- **AlertBudget is dead code** — class implemented, never instantiated. Telegram-ban risk unmitigated; a misbehaving signal can blast 1000+ messages. (infrastructure P1)
- **`health_state.last_invocation_tier` written at spawn, never cleared at completion** — T1 silent failures take 20 min to alert instead of 12 because the stale T3 tier inflates the grace window. (orchestration P1)
- **`telegram_poller._handle_mode_command` writes `config.json` (symlinked API-key file) from any sender matching the chat_id** — no `sender.id` allow-list, no command auth, no rate limit on the `/mode` path. (infrastructure P1)
- **CFTC SOCRATA fetcher has no retry / no rate-limit / no circuit breaker** — a single 503 aborts the whole precompute and the next-success window isn't recorded so the 7d COT interval restarts from the failed timestamp. (data-external P1)
- **`crypto_precompute` and `mstr_precompute` bypass `_binance_limiter`** by calling `requests.get` directly against api.binance.com — invisible quota burn against the shared budget; 429 silently nulls BTC fields. (data-external P1)
- **`bigbet` text path lacks auth-error scan** beyond the auto-detect helper — auth_failure can land as plain `text=""` with no critical_errors row. (orchestration P1)

### Risk-gate enforcement

- **MINI barrier-proximity rule isn't enforced pre-trade** anywhere. `memory/feedback_mini_stoploss.md` calls it CRITICAL; only `exit_optimizer` has any check, and only for held positions. (portfolio-risk P1)
- **Concentration / correlation / regime-mismatch / barrier-proximity flags are `severity:"warning"` only** — no `should_block_risk()` analog wired into agent invocation. (portfolio-risk P1)
- **Per-strategy concentration check misses cross-strategy aggregation** — XAG can be 39% Patient + 39% Bold + 100% Warrants without any flag firing. (portfolio-risk P1)
- **`warrant_pnl` and `compute_stop_levels` are long-only** — BEAR/SHORT certs (catalog includes them) get inverted P&L and upside-down stops if state is ever written for them. Dormant but latent. (portfolio-risk P1)
- **Swedish market holiday calendar exists but is never consulted** — Avanza warrant trading proceeds on Midsummer Eve / Whit Monday / etc. (orchestration P1)
- **`avanza_client._with_browser_recovery` retries mutating POSTs with no client-side idempotency token** — `order/new` and `stoploss/new` can be double-placed on browser-dead recovery. (avanza-api P1)
- **`cancel_order` / `cancel_stop_loss` / `delete_order_live` / `delete_stop_loss` skip `ALLOWED_ACCOUNT_IDS` check.** (avanza-api P1)
- **Pre-flight bid/ask sanity is absent from every order path** — a 10× mispricing upstream reaches Avanza unchallenged. (avanza-api P1)
- **5-minute Telegram CONFIRM window allows stale-price execution** on SELL orders for volatile warrants. (avanza-api P1)
- **EOD detection hardcodes 21:55 in three places** (grid_fisher, metals_swing_trader, metals_execution_engine) — half-day closures would over-hold 8.5h past close if calendar handling slips. (metals-core P1)
- **`_fetch_warrant_catalog_prices` computes `barrier_distance_pct` LONG-only** — SHORT certs with barrier get negative distance; silently filtered out of execution rec. Latent today, first SHORT silver scalp surfaces it. (metals-core P1)
- **`write_context` fallback `hours_remaining` uses legacy 17:25 close** when `metals_execution_engine` import fails — Layer 2 mis-estimates remaining window by 4.5h. (metals-core P1)
- **Sell + stop-loss volume not cross-checked against position size** in `grid_fisher.rotate_on_buy_fill` — multi-fill burst can place overlapping sells. (metals-core P1)
- **`_select_warrant` doesn't re-check `tradable` at place-time** — stale catalog entry can drive a BUY POST on a knocked-out warrant. (metals-core P1)
- **Dashboard `/api/golddigger` reports `market_close_cet: "21:30"`** — inconsistent with every other path's 21:55. (metals-core P1)

### Signal correctness

- **`crypto_evrp.py` direction contradicts its own docstring** AND the percentile sub-signal is partial dead code (parameter ignored). One of 16 active voters on BTC/ETH; whichever way the fix goes (flip code or update doc) needs a pinned regression test. (signals-modules P0)
- **`btc_etf_flow.py` registered nowhere, wrong signature, wrong return schema** — dead today, crashes at first invocation if wired. (signals-modules P1)
- **`cot_positioning._fetch_cot_historical` uses raw `requests.get`** with no retry/CB; fires inside the per-ticker dispatch loop on cold cache. (signals-modules P1)
- **`sentiment_extremity_gate` module-level F&G cache shares state across all tickers** — latent silent cross-pollination if `fear_greed.py` ever becomes ticker-aware. (signals-modules P1)
- **`intraday_seasonality` and `gold_overnight_bias` silently fall back to wall-clock when df has RangeIndex** — every backtest row gets the same hour; backtest results are meaningless for these two. (signals-modules P1)
- **`news_event._thesis_alignment_vote` is structurally bias-confirming** — when prophecy is bullish and news is bearish, returns HOLD ("don't vote against belief"). (signals-modules P1)
- **`_BIAS_MIN_ACTIVE = 30` compared against `total` (incl. HOLD) not active vote count** — bias penalty fires on essentially no evidence for rare-activation signals. (signals-core P1)
- **`_dynamic_min_voters_for_regime` returns the strict 5-voter floor for `None`/`"unknown"` regime** — every restart produces a HOLD-only window for metals/stocks. Pairs with the persistence-filter cold start (P0 #12) to make immediate-post-restart behavior incoherent. (signals-core P1)
- **IC `_rolling_ic` uses overlapping 50-sample windows** — ICIR std drastically underestimated; the `_IC_STABILITY_MIN = 0.10` gate is broken. (signals-core P1)
- **`train_signal_weights._load_signal_history` reads JSONL only, never the SQLite signal_db that production writes** — linear-factor model silently stops adapting after a JSONL rotation. (signals-core P1)
- **`cusum_accuracy_monitor` docstring claims 3-7 observation detection latency; actual MIN_OBSERVATIONS=20 + 10-obs cooldown = ~30 observations** — no faster than the daily batch check; operators misinformed. (signals-core P1)
- **`signal_accuracy` uses `outcome.get("change_pct", 0)` default** — missing key silently treated as flat outcome; corrupt-outcome detector blind to schema migrations. (signals-core P1)
- **`direction_probability_with_forecast` collapses `chronos_24h_pct=None` to 0** — Chronos OOM/abstain is silently indistinguishable from "Chronos predicted flat"; multi-day outage undetectable. (signals-core P1)

### Misc orchestration / infra

- **`analyze.py:_clean_env` does NOT pop `CLAUDE_CODE_ENTRYPOINT`** — every other path does; nested-session error possible when invoked from a Claude shell. (orchestration P1)
- **Perception gate auto-bypasses on `"consensus"` keyword regardless of confidence** — undermines the whole budget filter; the most common low-value invocation is sub-confidence consensus. (orchestration P1)
- **`_acquire_file_lock` busy-loops with 1s sleep + tasklist storm** under contention — cycle latency tail grows to 5-10 s. (infrastructure P1)
- **`gpu_gate._GPU_LOCK_FILE = Path("Q:/models/.gpu_lock")` hardcoded path** — silent fallback to subprocess on missing drive; CI/dev systems can't test the lock path. (infrastructure P1)
- **MSTR holdings/debt/shares hardcoded in `mstr_precompute`** — stale by ~1 month; NAV-premium signal drifts continuously since no config override path. (data-external P1)
- **Deribit `_parse_expiry` is locale-dependent** — `strptime("%d%b%y")` silently mis-parses MAR on non-English Windows locale, corrupting max-pain. (data-external P1)
- **`_no_position_skip` ignores warrant holdings** — could skip Layer 2 while warrants are open if the gate is enabled. (independent P2)
- **`trigger.py` `None` confidence raises TypeError in ranging dampening** — single cycle loss on signals returning explicit `confidence=None`. (independent P2)

---

## Themes (cross-cutting)

### Theme 1 — Concurrency invariants are correct in the helpers, broken at the call sites

`file_utils.atomic_write_json` and `atomic_append_jsonl` are correct.
`jsonl_sidecar_lock` is correct. Multiple `threading.Lock` patches landed
to fix specific 8-worker races (`_peak_cache_lock`, `_completion_lock`,
`_last_signal_lock`, `_phase_log_lock`).

But the *invariants* — "use atomic helpers, hold sidecar lock for read-
modify-write" — are not enforced at the call sites. `journal.py` reads
with raw `open()`. `health.py`, `prophecy.py`, `digest.py`, `daily_digest.py`
do read-modify-write without locks. `signal_history.jsonl` lock is
process-local while 5+ processes write it.

**Recommendation:** Make `load_jsonl`/`load_jsonl_tail` take the sidecar
lock internally so readers can't bypass it accidentally. Promote the
helpers + lock contract into a lint rule (a one-shot grep for raw
`open(...)`/`json.load(`/`json.loads(...read()` against tracked data
files would catch new offenders).

### Theme 2 — "Fail open" is the wrong direction for safety-critical gates

Orchestration P1 noted that drawdown is the only Layer 2 gate that
fails closed. Escalation_gate, no_position_skip, trade_guards,
perception_gate, claude_gate config-check all fail open — when the
gating logic itself misbehaves, the system defaults to invoking Claude.
The audit notes 47/47 self-heal timeouts in the past 7d. CLAUDE.md cost
narrative is "minimise unnecessary invocations." The defaults are
fighting each other.

**Recommendation:** Audit every `except Exception:` block inside a gate
function. Decide explicitly: fail-open or fail-closed. Document the
choice next to the except clause. The current default-fail-open across
the board is a coding habit, not a deliberate safety choice.

### Theme 3 — State written at spawn but not cleared at completion

`_agent_tier`, `_agent_log_start_offset`, `last_invocation_tier`,
`_agent_reasons` are all written when invoke_agent spawns and not
consistently cleared on completion. The grace-window calculation in
loop_contract uses the stale value. This is the same class of bug as
the persisted monotonic clock in trigger.py — module-state lifecycle is
inconsistent.

**Recommendation:** Single helper `_clear_agent_state()` called from
both happy and unhappy completion paths AND from `_kill_overrun_agent`.
Add a unit test that spawns, simulates completion, and asserts the
module-state dict is fully reset.

### Theme 4 — Hard-coded reference files (cooldown tables, holiday lists, MSTR fundamentals) decay silently

- FOMC/CPI/NFP dates hardcoded through Dec 2027 (data-external P3).
- MSTR holdings/debt/shares hardcoded (data-external P1).
- Swedish holiday calendar defined but unreferenced (orchestration P1).
- `_PER_TICKER_CONSENSUS_GATE` justified for tickers removed from the
  universe months ago (independent P2).
- Per-ticker disabled signal blacklists with date stamps going back to
  2026-04-15 (signals-core).

**Recommendation:** Add a `scripts/check_static_data_freshness.py` that
inspects all hardcoded date-bounded references and warns when within 90
days of expiry. Wire it into the daily PF-OutcomeCheck so it ends up in
critical_errors.jsonl when something is about to lapse.

### Theme 5 — Avanza account-id allow-listing is not uniformly enforced

`_place_order` enforces. Several wrappers + the delete path + the
TOTP-confirm path do not. The user's `feedback_isk_only` memo is
explicit ("ONLY show ISK account 1625505"); the code holds that line
in one path and leaks in five.

**Recommendation:** Decorator `@require_isk_account` on every function
that takes `account_id` as a parameter. Decorator raises if the value
isn't in the allow-list. Add a startup assertion that all decorated
functions are reachable only via the allow-list path.

---

## What this round did NOT find that prior rounds did

Cross-check against `docs/fgl-2026-05-23/SYNTHESIS.md`:

- **`kelly_metals` 50-100× sizing bug** — fix landed (verified in
  current code; kelly_metals.py uses the corrected formula).
- **`fin_snipe_manager` 1% stop-loss vs 3% rule** — fix landed.
- **`grid_fisher.rotate_on_buy_fill` naked-position window** — addressed
  via `stop_needs_rearm` flag.
- **Multi-agent specialist `success_count == 0` not gated** — quorum
  gate landed (`agent_invocation.py:1023-1030`); this round found a new
  bug downstream of that gate (P0 #10 above) but the original quorum
  check is correct.
- **`signal_engine.py:4205` accuracy gate config-overrideable below
  floor** — gate now clamps to base floor (verified at signal_engine
  call sites).
- **`signal_decay_alert.py` relative `"data/..."` paths** — converted to
  module-relative `Path(__file__).resolve().parent.parent / "data"`.

The prior round's fixes held. The new round surfaces a new class of
silent-failure (escalation_gate wedge, llama_server orphan, drawdown
blinded by stale price) plus the per-ticker / per-regime interaction
bugs in `_dynamic_min_voters` and `_apply_persistence_filter`.

---

## Recommended fix order for the next implementation pass

Ordered by money-at-risk × likelihood, with cheap fixes first within
each tier.

1. **P0 #14 — EOD_EXIT_MINUTES_BEFORE = 0** (metals_swing_config.py:323)
   — single-character change (0 → 25). Reinstates the user's "no
   overnight" rule. Telegram alert on config drift.
2. **P0 #13 — Swing-trader stop ignores barrier** (metals_swing_trader.py)
   — re-uses the existing `MIN_BARRIER_DISTANCE_PCT` constant; lifts the
   `_select_warrant` check into `_set_stop_loss`. Directly defends the
   #1 grudge memory.
3. **P0 #1 — Metals MIN_VOTERS in ranging regime** (signal_engine.py)
   — directly defeats user-stated primary focus; small code change.
4. **P0 #2 — Drawdown blinded by stale price** (risk_management.py:201)
   — safety-critical; the 50% block depends on this.
5. **P0 #3 — FX-fallback bypass** (monte_carlo_risk.py, exit_optimizer.py)
   — single line per call-site; high impact on risk math accuracy.
6. **P0 #4, #5, #6 — Avanza account-id leaks** (metals_avanza_helpers,
   place_stoploss_once, avanza_client._place_order) — money-loss path.
   Build the `@require_isk_account` decorator once, apply to all
   wrappers.
7. **P0 #12 — `_with_browser_recovery` duplicate-order retry**
   (avanza_session.py:212) — gate retry on idempotent flag; mutating
   POSTs become "no retry, surface failure".
8. **P0 #7 — escalation_gate wedge** — silent multi-week cost overrun
   once it fires; small fix.
9. **P0 #11 — llama-server orphan** — recovers GPU + port + lock on
   parent crash.
10. **P0 #8, #9 — Specialist tree-kill + auth-scan offset** — same fix
    pattern, both inside `claude_gate`/orchestration.
11. **P0 #10, #15 — `return None` mis-log + persistence-filter cold
    start** — cleanup work that prevents the next silent regression.

P1 batch after P0 lands; structural themes (locks at call sites, fail-
open audit, hard-coded freshness checker, `@require_isk_account`
decorator extension to all wrappers) can be split out as separate PRs
once the bug fixes are in.
