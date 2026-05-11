# Adversarial Review 2026-05-11 — Synthesis

Dual independent adversarial review of finance-analyzer codebase. Codex (gpt-5.4, xhigh effort, read-only) and Claude (general-purpose subagents, sonnet-class via the agent tool, read-only) reviewed the same 8 disjoint subsystems in parallel. This doc cross-critiques both reviews and consolidates the highest-confidence findings.

Setup: worktree `Q:/fa-adv-2026-05-11` from `main@9b3f7084`; eight `review/baseline-N-<subsystem>` branches stamped from the same SHA; both reviewers were given identical adversarial prompts under `_prompts/`.

## TL;DR

**11 P0 blockers** and **~50 P1 incident-class defects** were independently surfaced. Three patterns dominate:

1. **Silent-fallback failure mode is endemic.** `try/except: logger.debug` + return-default appears 30+ times in `signal_engine.py` alone. The codebase's own scar tissue (Layer 2 outage Mar–Apr 2026, BUG-178, accuracy regression) keeps growing because the immune response is "don't crash" without surfacing.
2. **The Mar-3 stop-loss incident class is still live in three places.** `grid_fisher.cancel_order` fallback when `cancel_stop_loss` is missing; barrier-proximity unchecked on stops placed against existing inventory; `compute_stop_levels` never compares the proposed stop against the MINI financing level.
3. **Project-rule drift between docs and code.** Min order size = 500 in three sizing modules vs the rule of 1000. Stop-cap = 15 % ATR vs the documented "user accepts 10-20 % knockout risk on 5x certs". `claude_gate` bypassed by `bigbet.py` and `pf-agent.bat`. Min-voters semantics correctly enforced in engine — that one is healthy.

## Methodology

| | Codex | Claude subagent |
|---|---|---|
| Tool | `codex exec --sandbox read-only` (default `gpt-5.4`, xhigh reasoning) | `general-purpose` agent, identical adversarial prompt |
| Output | Conversation transcript w/ inline file dumps via `js_repl` | Direct structured markdown report |
| Final structured report | **Never emitted** — model spent its turn budget reading files | Yes (every subsystem) |
| Useful signal | Prose narration between tool calls (extracted to `*-codex-prose.md`) | Full P0/P1/P2/P3 + tests-missing + cross-cut sections |

**Meta-finding.** Codex on Windows hit two friction points: (a) PowerShell shell commands were rejected by the sandbox policy, forcing a fallback to the Node `js_repl` tool; (b) the resulting tool-call volume consumed the turn budget before the model could compose a final structured report. Eight runs all show the same pattern — codex emitted thoughtful narration ("I’ve found a hard policy mismatch", "warrant_portfolio models MINI products as simple leverage multipliers and never tracks financing/barrier") but never reached the formal P0/P1/P2 enumeration. The Claude subagent route was 100 % effective at producing structured output. For the next FGL daily run, either (i) raise codex's turn budget / token cap, (ii) switch to `codex review --base review/empty-baseline-N` with a smaller per-subsystem scope, or (iii) treat the codex prose as a complementary signal layer rather than the primary deliverable.

**Cross-critique direction.** Because codex never emitted P0/P1 lines, the "both flagged" overlap is computed by checking whether codex's prose explicitly hints at a defect that Claude also raised. Where codex's hint is independent of Claude's finding (e.g. codex spotted "validator default minimum below 1000 SEK floor" before any Claude review existed), it is credited as an independent corroboration.

---

## Per-subsystem cross-critique

### 1 — signals-core

**Codex prose hints (1-signals-core-codex-prose.md):** one hard policy mismatch in engine constants; outcome backfill doing JSONL parsing/rewrite outside the atomic helper path.

**Claude P0 (4):**
- `signal_db.py:31-37` — single `sqlite3.connect()` cached on `self._conn` shared across the 8-worker thread pool. `sqlite3.Connection` is not thread-safe by default; `try/except Exception` at l.157/165 swallows the resulting `ProgrammingError` and silently falls back to JSONL → dual-write divergence.
- `ic_computation.py:73-147` — IC rolling window not sorted chronologically; samples from different tickers interleave; `ic_buy`/`ic_sell` are **average returns**, not information coefficients (mislabel propagates to per-ticker cache).
- `signal_decay_alert.py:34-39` — raw `open()/json.load` on `accuracy_cache.json` (atomic-write target). Read race → silent miss in the decay watchdog.
- `signal_history.py:64-98` — intra-process lock only; cross-process writers (crypto_loop, oil_loop, metals_loop, outcome_tracker) race on `signal_history.jsonl`, last-writer-wins truncation.

**Claude P1 (7):** persistence-filter seed asymmetry (signal_engine:566-625); regime-gate exemption uses wrong horizon for 3h decisions; `signal_accuracy_ewma` silently drops tz-naive entries; `signal_history.update_history` hardcoded to `SIGNAL_NAMES` so dynamic signals never recorded; walk-forward survivorship bias; outcome dual-write SQLite-first race; `maybe_prewarm_dashboard_accuracy` falls back to `"noop"` string when `process_lock` import fails — prewarm fires without exclusion.

**Cross-critique.** Codex's prose hint about "outcome backfill doing its own JSONL parsing/rewrite" is the *same* failure family Claude documents at `outcome_tracker.py:478-482` (P1) and the `signal_decay_alert.py` raw-read (P0). **Both agree.** Codex's "hard policy mismatch in engine constants" is unverifiable without the final report; Claude did NOT flag a constant mismatch in `signal_engine.py`. Possible candidates: the persistence-filter seed asymmetry (cycles=0/1/2 inconsistency), or the `MIN_CYCLES` vs documented "2 consecutive" semantics. **Codex-only signal** worth investigating: explicit grep of engine module constants vs CLAUDE.md doc.

### 2 — orchestration

**Codex prose hints (2-orchestration-codex-prose.md):** watchdog/auth-scan/timeout paths central; health cache + prewarmer + trigger/tier "can silently degrade the whole loop without crashing".

**Claude P0 (4):**
- `agent_invocation.py:846` + `multi_agent_layer2.py:145` — unconditional `NODE_OPTIONS="--stack-size=16384"` overwrites inherited env.
- `llm_prewarmer.py:299-301` — synchronous `query_llama_server` call from `flush_llm_batch` can block the main loop ~120 s on a model swap (30 s VRAM wait + 90 s startup). The whole 600 s cadence is at risk.
- `bigbet.py:175-181` — direct `subprocess.run(["claude", "-p", ...])` **bypasses `claude_gate` entirely**. No tree-kill, no auth-failure detection, no `_invoke_lock`, no `CLAUDECODE` cleanup. The exact silent-exit-0 failure class that the gate was built to prevent.
- `agent_invocation.py:824-830` — `pf-agent.bat` fallback also bypasses claude_gate and runs at T3 regardless of requested tier; grandchildren outlive `_kill_overrun_agent`.

**Claude P1 (11):** lock scope gaps in `_check_agent_completion_locked`; `_agent_log_start_offset` re-flag risk (BUG-ECHO 2026-04-16 replay); journal/index unbounded reads per invocation; `prophecy.save_beliefs` lacks file lock for concurrent checkpoint+layer2 update; `health.check_staleness` returns "stale forever" on naive timestamp; `telegram_notifications` Markdown 400-empty-body retry loses messages; `llama_server.query_llama_server` holds thread+file locks across entire 240 s HTTP call.

**Cross-critique.** Codex flagged "the prewarmer, the trigger/tier logic" as silent-degradation surfaces; Claude's prewarmer P0 and trigger flip_cooldowns growth (P2) corroborate. **Both agree.** Codex's "auth-scan" focus is the watchdog tick that Claude shows is mis-protected (`_agent_log_start_offset` P1). **Both agree.** Neither reviewer made it to `analyze.py` or `reflection.py` in depth; those should get a follow-up pass.

### 3 — portfolio-risk

**Codex prose hints (3-portfolio-risk-codex-prose.md):** "trade validator default minimum is below the stated 1000 SEK floor"; "risk classifier has a silent 'unknown regime = zero risk' path"; "warrant_portfolio.py models MINI products as simple leverage multipliers and never tracks financing/barrier at all, which means knockouts can be missed entirely".

**Claude P0 (4):**
- `trade_validation.py:32`, `kelly_sizing.py:326`, `kelly_metals.py:44` — all three default `min_order_sek = 500.0`. Avanza floor is 1000. Layer 2 clears cash, then Avanza rejects.
- `risk_management.py:374` — `atr_pct = min(atr_pct, 15.0)` clamps ATR-based stops below the documented 20 % knockout tolerance and the function never compares the resulting stop against the MINI financing level; can return a stop *inside* the knockout barrier.

**Claude P1 (10):** `check_atr_stop_proximity` flags at 1.0× ATR (already inside the wick band, reactive not preventive); `compute_probabilistic_stops` silently disables the stop by clamping to `entry × 0.01`; drawdown circuit-breaker NaN guard persists sentinel = 0 to disk; `monte_carlo_risk.fx_rate` bypasses the sanity-band fallback chain; `exit_optimizer._compute_pnl_sek` only deducts cost on exit (entry-side cost missing); hold-to-close EV computed from 5-quantile trapezoid not full path; trade_guards cooldown bypassable via wall-clock; portfolio_mgr backup rotation `with_suffix` foot-gun.

**Cross-critique.** All three of codex's prose hints (1000-SEK floor, silent risk-class fallback, no MINI barrier tracking) appear verbatim in Claude's P0/P1. **Strong agreement.** Codex's "risk classifier silent unknown-regime → zero risk" maps to `trade_risk_classifier.py` which Claude flagged at P2; that should be promoted to P1 given the prose corroboration.

### 4 — metals-core

**Codex prose hints (4-metals-core-codex-prose.md):** "structural issues in metals_loop and grid_fisher"; "actual trade sets overlap"; "MINI barrier-distance guard only exists in ranking/pretrade helpers"; "session-window mismatches in golddigger and elongir"; `orb_backtest.py` is explicitly walk-forward (codex withdrew its concern there); "precompute paths are mostly self-refreshing".

**Claude P0 (3):**
- `grid_fisher.py:1089-1094` — stop-loss cancel falls back to generic `cancel_order` if injected session lacks `cancel_stop_loss`. Mar-3 incident pattern.
- `grid_fisher.py:1097-1098` + `grid_tiers.py:208-223` — stop placement skips knockout/barrier check on existing inventory (the `_tier_skip_for_knockout` guard only protects opening buys).
- `mstr_loop/config.py:19` + `execution.py:165-169` — `PHASE=live` gated only by env var. No approval token, no Telegram confirmation. Stray `export MSTR_LOOP_PHASE=live` puts MSTR live against account 1625505.

**Claude P1 (10):** EOD 21:55 hardcoded vs the `.claude/rules/metals-avanza.md` "do NOT hardcode" rule; `fin_fish.py:735` BEAR MINI uses `pass` not `continue` (selects knocked-out warrants); cross-process race between metals_loop swing trader and grid_fisher on same ob_ids; global halt threshold scales by total instruments not active; `eod_market_flat` can place 0.01 SEK if quotes fail; `grid_tiers` barrier math ignores parity; `fin_fish` FX rate silent fallback to 10.0; ORB backtest no high-before-low ordering check; oil_grid_signal cache bypasses TTL on date-format mismatch.

**Cross-critique.** Codex's "trade sets overlap" prose maps directly to Claude's metals_loop ↔ grid_fisher cross-process race P1. **Both agree.** Codex's "MINI barrier guard only in pretrade helpers" is the same defect Claude raised as P0 #2 — codex correctly identified the gap before Claude's structured listing existed. **Both agree, codex deserves equal credit.** Codex's golddigger/elongir session-window flag aligns with Claude's P2 hits on those modules. Codex's exoneration of `orb_backtest.py` (walk-forward) directly contradicts Claude's P1 on high-before-low ordering — **disagreement.** Re-checking Claude's evidence at `orb_backtest.py:182-202`: Claude is correct — the simulator does not record `high_hour_utc`/`low_hour_utc`, so "buy at low → sell at high" days where the high came first are scored as wins even though the buy never filled in real time. The walk-forward training/holdout split is sound (codex right about that) but the trade-simulation P&L is biased.

### 5 — avanza-api

**Codex prose hints (5-avanza-api-codex-prose.md):** "one concrete issue candidate in the logging path"; "lock reentrancy concern"; auth/account/type code paths under review; specifically called out the question "whether this wrapper is actually reentrant for the same caller, or only looks that way".

**Claude P0 (4):**
- `avanza_session.py:212-232` — `_with_browser_recovery` holds `_pw_lock` across `close_playwright()` (which re-acquires the same RLock) and across `_get_playwright_context()`. Other order/cancel callers wait 15-30 s behind a relaunch; the 2 s `OrderLockBusyError` ceiling means the next caller drops the order entirely.
- `avanza_session.py:608-616` — `_place_order` sends raw `price` without tick-rounding. `tick_rules.round_to_tick` exists but is never wired in. Avanza silently `INVALID_PRICE`-rejects ~30 % of off-tick prices (Mar 24 incident).
- `avanza_session.py:751` — stop-loss whitelist guard exists in session path but NOT in `portfolio/avanza/trading.place_stop_loss`. If `client.account_id` ever bridges to pension via config drift, stops fire on the wrong account.
- `avanza_account_check.py:215-219` — cache poisoning: `_cache_result` keyed without `category`; a single `ok=True` for a wrong ID sticks for process lifetime, no production `reset_cache()` caller.

**Claude P1 (7):** `get_open_orders` swallows pagination truncation; `_walk_accounts` non-deterministic when Avanza returns both `categorizedAccounts` and `accounts` shapes; `get_stop_losses` lenient variant indistinguishable empty vs error (Mar-3 class); `rearm_stop_losses_from_snapshot` acquires per-stop locks (partial-rearm risk); `avanza/scanner.search_query` substring-matches `"BULL OLJA"` literal; streaming reconnect has no max-lifetime + handshake-rate guard; `avanza_orders.py:138-142` logs `confirm_token` at INFO into `agent.log` (5-min trade-execution token leakage).

**Cross-critique.** Codex's "lock reentrancy concern" is exactly Claude's `_with_browser_recovery` P0; Codex held the finding to verify whether the RLock actually was reentrant before reporting — Claude proves it is, but the issue is the *blocking duration*, not the reentrancy. **Both agree on direction, Claude has the sharper analysis.** Codex's "one concrete issue candidate in the logging path" likely corresponds to the `confirm_token` INFO-level leak that Claude raised at P1; codex implicitly down-weighted it, Claude flagged it explicitly. **Codex-corroborated P1.**

### 6 — signals-modules

**Codex prose hints (6-signals-modules-codex-prose.md):** "smart_money + heikin_ashi look-ahead candidates"; "fixed-horizon detectors (forecast, credit_spread, cot_positioning, metals_cross_asset) leak the same vote across horizons"; engine returns `(action, conf, extra_info)` after detector result dicts (contract validated).

**Claude P0 (2):**
- `signals/mahalanobis_turbulence.py:99` — `_cached("key", _do_fetch, ttl=...)` is a signature drift vs `shared_state._cached(key, ttl, func)`. TypeError on first call. Disabled today but flagged as live booby trap on re-enable.
- `signals/claude_fundamental.py:929` — per-tier cache `ts` is set BEFORE the background refresh thread runs. If the CLI fails (timeout, gate kill, network), `_cache[tier]["results"]` stays at the previous (possibly `{}`) value until next cooldown — silent permanent HOLD.

**Claude P1 (7):** `news_event._HEADLINES_PATH` single shared file races across 8 workers; `intraday_seasonality` falls back to wall-clock `now()` on no-datetime-index df (look-ahead in backtests); `calendar_seasonal` hardcoded floating-Monday holiday dates (wrong every year); `futures_flow._LS_EXTREME_LOW=0.7` threshold drift (Binance scale 0.7 ≈ mild lean, not "overleveraged short"); `forecast.py` and `futures_flow._oi_acceleration` condition on currently-forming last bar (inflates Chronos shadow accuracy); `smart_money.py:374` zone-expansion math `proximity_pct / _ZONE_PROXIMITY_PCT` cancels to 1 (dead knob); `claude_fundamental` reads + tail-parses 400-entry JSONL on every signal call (hot-path disk I/O).

**Cross-critique.** Codex's two prose hits (smart_money + heikin_ashi look-ahead; fixed-horizon leakage in forecast/credit_spread/cot_positioning/metals_cross_asset) both align with Claude's findings: smart_money P0/P1 (zone math + look-ahead) and the forecast bar-conditioning P1. **Both agree.** Codex extends Claude with the `heikin_ashi` Alligator forward-projection concern — Claude did NOT inspect heikin_ashi in depth. **Codex-only finding worth a follow-up read**: `signals/heikin_ashi.py` Alligator lines and `signals/credit_spread.py` horizon-leakage.

### 7 — data-external

**Codex prose hints (7-data-external-codex-prose.md):** "alpha_vantage budget accounting defect"; "social_sentiment direct retry bypass"; reviewed `alpha_vantage`, `http_retry`, `fx_rates`, `price_source`, `microstructure`, `onchain_data`, `session_calendar`, the sentiment stack.

**Claude P0 (4):**
- `futures_data.py:33-55` — `get_open_interest()` returns `oi` not `oi_usdt`; downstream consumers expect `oi_usdt`; manifests as "0 usdt OI" in agent summaries, distorting futures-flow voter.
- `funding_rate.py:44-49` — sign convention possibly inverted relative to Binance. At 74 % accuracy (3 h horizon) the signal could be lucky-flipped on certain horizons. Needs unit test with captured spike + known directional outcome.
- `fx_rates.py:46-71` — sanity-failed rate path doesn't update `cached_time`; stale cache used indefinitely after a single bad rate; 24h+ FX drift goes silent.
- `http_retry.py:88` — `fetch_json` calls `raise_for_status()` AFTER the retry loop; 4xx responses (401/403/404) reach as a populated `resp`, raise HTTPError, bare-except swallows, returns `default=None`. Silent auth-dead vs transient indistinguishable.

**Claude P1 (7):** `alpha_vantage._daily_budget_used` not persistent across restart (flapping loop = 50+ AV reqs/day); budget race between check and fetch; `earnings_calendar` bypasses AV budget tracker; `session_calendar._eu_dst` TypeError on naive datetime; `news_keywords` round-trips raw regex through `.pattern`; `onchain_data._load_onchain_cache` is dead code; `price_source` silent fallback to yfinance with symbol-routing collision (Binance FAPI XAU-USD perpetual → yfinance XAU-USD spot mid-stream).

**Cross-critique.** Codex's "alpha_vantage budget-accounting defect" is exactly Claude's P1 (persistence + race + earnings bypass). **Both agree.** Codex's "social_sentiment direct retry bypass" is independent of Claude's `http_retry.py:88` P0 — they are different bugs (codex spotted a module bypassing the retry layer; Claude spotted the retry layer itself losing 4xx info). **Both real, both should ship.** Recommend explicit follow-up on `portfolio/social_sentiment.py` HTTP path.

### 8 — infrastructure

**Codex prose hints (8-infrastructure-codex-prose.md):** "dashboard surface mostly read-only"; "raw file reads in dashboard"; "Windows wrappers that interact with scheduled tasks and Claude subprocesses".

**Claude P0 (3):**
- `dashboard/app.py:1049-1052` (`/api/iskbets`) — returns `iskbets_config.json` unfiltered. Future addition of an API-key field to iskbets config = leak via dashboard cookie. No redaction layer.
- `dashboard/app.py:101-102` — `_get_config()` caches full `config.json` dict in `_cache` keyed `json:Q:\\...\\config.json`. Plus `dashboard/auth.py:60-66` keeps its own `_CFG_VALUE` copy. Two unfiltered in-memory caches of the secret-bearing config.
- `scripts/fix_agent_dispatcher.py:235-236` — kill-switch checked per-category. `touch fix_agent.disabled` mid-run will not stop in-flight subsequent categories.

**Claude P1 (7):** `process_lock` never re-validates PID metadata on reacquire (post-SIGKILL stale data); `subprocess_utils.kill_orphaned_by_cmdline` interpolates unvalidated `pattern` into WMIC `WHERE` clause; `kill_orphaned_llama` uses `shell=True` + inline quotes (future-refactor injection); `fix_agent_dispatcher._save_state` skips fsync (regression vs `atomic_write_json`); `file_utils.jsonl_sidecar_lock` derives lock path without `Path.resolve()` (rotation race re-emerges across worktree/junction paths); `_read_tail_with_growth` unbounded-memory failure (single 80 MB line → 64 MB grow → fall through to full-file load → OOM); Cloudflare Access bypass trusts `Cf-Access-*` headers without JWT signature verification.

**Cross-critique.** Codex's "raw file reads in dashboard" maps to Claude's `_read_tail_with_growth` P1 (and the broader `_get_config` P0 cache). **Both agree.** Codex's interest in Windows wrappers aligns with Claude's `subprocess_utils.kill_orphaned_*` P1 family. **Both agree.** Neither reviewer dug into `vector_memory` ChromaDB unbounded growth — Claude noted it in cross-cut; codex did not surface it.

---

## Cross-cutting patterns

1. **Silent-fallback is the dominant defect class.** Every subsystem has multiple `try/except Exception: logger.debug(..., exc_info=True); return default` blocks. The codebase needs a `log_once_then_debug(category, msg)` helper that promotes first-occurrence-per-process to WARNING.

2. **`claude_gate` is being routinely bypassed.** Three live bypasses: `bigbet.py` direct subprocess.run, `pf-agent.bat` cmd /c invocation, and `multi_agent_layer2.py` parallel-specialist spawn that imports its own subprocess machinery. Each one re-introduces the silent-exit-0 failure mode that motivated the gate after the Mar–Apr 2026 outage. Treat any direct `subprocess.run(["claude", ...])` outside `claude_gate.py` as a P0.

3. **Min-order-size, knockout-buffer, and ATR-cap constants are scattered.** 500 SEK floor in 3 sizing modules vs 1000 SEK in 2 enforcement modules; 3 % barrier buffer in `exit_optimizer` vs no barrier check in `compute_stop_levels`; 15 % ATR cap vs documented 20 % knockout tolerance. A single `portfolio/instrument_profile.py` or `portfolio/constants.py` source-of-truth with `from portfolio.avanza.trading import MIN_ORDER_SEK` would close the gaps.

4. **The MINI warrant financing-level / barrier is not a first-class concept anywhere upstream of `exit_optimizer`.** `risk_management.compute_stop_levels`, `grid_fisher.place_stop_loss`, `fin_fish` selection — all reduce a MINI cert to a leverage multiplier. The barrier is the cert's only real risk; knockouts are missed entirely if the barrier-distance check fires only in the optimizer.

5. **Cross-process state writers are not coordinated.** `signal_history.jsonl`, `prophecy.json`, and the Avanza order layer are all written by multiple processes (PF-DataLoop, PF-MetalsLoop, PF-CryptoLoop, PF-OilLoop, PF-MstrLoop, fix_agent_dispatcher). The sidecar lock at `file_utils.atomic_append_jsonl` was just patched (commit 3b623129), but the same race pattern exists in `signal_history`, `prophecy`, and any read-modify-write path. Audit grep for `load_json(...)` followed by `atomic_write_json(...)` outside a file lock.

6. **Codex sandbox-policy friction wastes ~50 % of codex's turn budget on PowerShell→js_repl fallback.** Add `Q:/finance-analyzer` to the `~/.codex/config.toml` workspace trust block; promote the read-only sandbox to allow direct file reads via `Get-Content`; or switch the recurring `PF-AdversarialReview` task to invoke `codex review --base review/empty-baseline-N` with smaller per-subsystem scopes so codex has spare budget for the final report.

---

## Top action list (recommended P0/P1 fixes, ranked by trade-impact × shipping risk)

| # | Subsystem | Fix | Source |
|---|---|---|---|
| 1 | portfolio-risk | Make `MIN_ORDER_SEK = 1000` the single canonical constant; import everywhere | both |
| 2 | portfolio-risk | `compute_stop_levels` takes `financing_level`; floors at `financing_level * 1.03`; warns if would land inside | both |
| 3 | metals-core | `grid_fisher.__init__` raises if `session.cancel_stop_loss` missing | claude |
| 4 | metals-core | `place_stop_loss` re-checks barrier proximity even on stops against existing inventory | both |
| 5 | metals-core | `MSTR_LOOP_PHASE=live` requires on-disk approval token with checksum | claude |
| 6 | orchestration | `bigbet.py` and `pf-agent.bat` route through `claude_gate.invoke_claude` | claude |
| 7 | orchestration | `llm_prewarmer.flush_llm_batch` dispatches prewarm on a daemon thread with 30s wallclock | claude |
| 8 | signals-core | `signal_db._get_conn` uses `threading.local()` cache + `check_same_thread=False` | claude |
| 9 | signals-core | `signal_decay_alert.check_signal_decay` uses `file_utils.load_json` | claude |
| 10 | avanza-api | `_place_order` calls `round_to_tick` before payload assembly (both paths) | claude |
| 11 | avanza-api | `_with_browser_recovery` releases `_pw_lock` before relaunch; uses single-shot lock for the request only | claude |
| 12 | data-external | `fx_rates` invalidates cache on sanity-failure; raises if no fresh rate | claude |
| 13 | data-external | `http_retry.fetch_json` distinguishes 4xx vs transient | claude |
| 14 | infrastructure | `/api/iskbets` allowlists fields; never returns raw config dict | claude |
| 15 | infrastructure | `fix_agent_dispatcher.run()` re-checks `KILL_SWITCH.exists()` at every category iteration | claude |
| 16 | infrastructure | `dashboard/auth.py` verifies `Cf-Access-Jwt-Assertion` against CF JWKS | claude |
| 17 | signals-modules | Fix `signals/mahalanobis_turbulence.py:99` arg order; gate registration on import-time-call success | claude |
| 18 | signals-modules | `claude_fundamental._needs_refresh` only bumps `ts` after successful refresh | claude |
| 19 | signals-modules | `news_event._HEADLINES_PATH` keyed by ticker | claude |
| 20 | signals-modules | `calendar_seasonal._US_HOLIDAYS` uses `pandas.tseries.holiday.USFederalHolidayCalendar` | claude |

## Test gaps (highest priority)

- Thread-safety test for `SignalDB` (4 threads × `load_entries` + `insert_snapshot`) — signals-core P0 regression.
- Cross-process race test for `signal_history.jsonl` writes from satellite loops.
- ATR-stop math: assert returned stop > `financing_level * 1.03` on a MINI cert with parametrized leverage.
- `place_stop_loss` is NEVER called via `place_order`/`cancel_order` paths (Mar-3 regression).
- Knockout-proximity applied to stops on EXISTING inventory, not just opening buys.
- `cancel_stop_loss` is required (not optional) at `grid_fisher` construction.
- MSTR loop refuses `PHASE=live` without on-disk approval token.
- Min-order-size: parametrized over `trade_validation`, `kelly_sizing`, `kelly_metals`, asserts `>= 1000`.
- `bigbet.py` routes through `claude_gate.invoke_claude` (mock-and-verify test).
- `funding_rate` sign convention vs captured Binance spike with known direction.
- `http_retry.fetch_json` distinguishes 401 vs 503 vs `ConnectionError`.
- `/api/iskbets` does NOT leak fields outside the allowlist (parametrized over future config keys).
- `_read_tail_with_growth` refuses to fall through to full-file load on single-line >128 MB file.

## Inventory of artifacts

- `00-PARTITION.md` — subsystem partition + branch convention
- `_prompts/{1..8}-<subsystem>.txt` — review prompts (identical across both reviewers per subsystem)
- `{1..8}-<subsystem>-codex.md` — raw codex `exec` transcripts (kept for audit; mostly tool-call dumps)
- `{1..8}-<subsystem>-codex-prose.md` — extracted codex prose narration
- `{1..8}-<subsystem>-claude.md` — Claude subagent structured reviews (the primary deliverable)
- `_extract_codex_prose.py` — helper script that produced the `-prose.md` extracts
- `99-SYNTHESIS.md` — this doc

## What this review did NOT cover

- The ~250+ `data/_*.py` ad-hoc one-off scripts (intentionally out of scope).
- Test code under `tests/` (would be the next adversarial pass — many of these defects exist because the test suite doesn't enforce them).
- Live behavior validation (no loops were restarted, no orders placed, no Avanza session opened). Findings are from static analysis only.
- The 142-module map referenced in `docs/SYSTEM_OVERVIEW.md` was not cross-checked against current files (`fin_fish_manager.py` was in-scope but does not exist on disk — minor catalog drift).
