# Orchestration Adversarial Review — 2026-05-27

Scope: `portfolio/{main,agent_invocation,trigger,trigger_buffer,market_timing,loop_contract,loop_health,loop_processes,process_lock,circuit_breaker,escalation_gate,escalation_router,claude_gate,multi_agent_layer2,autonomous,analyze,bigbet,iskbets,perception_gate}.py`

## Summary

- **P0:** 4 findings (silent fail-open of escalation gate after first hang, missing tree-kill on specialist timeout, `last_invocation_tier` stickiness extends T1 silent-failure detection by 17 minutes, `invoke_agent` returns `None` on `specialist_quorum_fail` causing main.py to mis-log as `skipped_busy`).
- **P1:** 7 findings (race window setting `_agent_tier` outside `_completion_lock`, no auth-error scan in `bigbet` text path beyond auto-detect, perception-gate's `_BYPASS_KEYWORDS` bypasses regardless of confidence, multi_agent specialist Popen lacks `creationflags=CREATE_NEW_PROCESS_GROUP`, Swedish holiday calendar defined but never consulted, `_load_state` race when corrupt JSON, `_check_recent_trade` updates baseline only when one strategy file readable).
- **P2:** 6 findings (Stack-size NODE_OPTIONS missing from claude_gate paths, `analyze.py` direct subprocess does not strip `CLAUDE_CODE_ENTRYPOINT`, price baseline reset every cycle defeats long-quiet slow-drift triggers, `_acquire_singleton_lock` non-atomic write of metadata, post-cycle `signal_postmortem` writes `_epoch` into the result dict the dashboard reads, trigger.py `_startup_grace_active` is a process-global thread-unsafe flag).
- **P3:** 3 findings.

Top 3 themes:
1. **Hot subprocess lifecycle gaps that pre-date 2026-04-16 hang fix are still alive in adjacent paths** — multi_agent specialist `proc.kill()` is single-PID (not tree), and escalation_gate's single-worker ThreadPoolExecutor leaks the runner thread on timeout (the `_fut.cancel()` is a no-op once the worker has started).
2. **State globals leak across invocations** — `_agent_tier`, `_agent_log_start_offset`, `last_invocation_tier` in `health_state.json` are written at spawn but never cleared on completion. A T3 followed by T1 makes loop_contract use the T3 grace window for the T1, hiding silent failures up to 17 minutes longer than designed.
3. **Fail-open on every Layer 2 gate** — drawdown circuit breaker is the only block; escalation_gate fails open, no_position_skip fails open, trade_guards fails open, perception_gate fails open. When the gating stack misbehaves, the system DEFAULTS to invoking Claude — which the CLAUDE.md cost narrative + 7d audit (47/47 self-heal timeouts) suggests is the wrong direction.

**Biggest risk:** A single `query_llama_server` hang (no inner timeout) permanently wedges `escalation_gate._RUNNER_EXECUTOR` (max_workers=1, _fut.cancel cannot stop a running task on CPython 3.12). Every subsequent escalation_gate call queues behind the wedged worker, returns "fail open" after 10s, and burns Claude budget — invisible because each call logs `ministral_unavailable` once and looks like a transient miss.

---

### [P0] escalation_gate single-worker pool wedges permanently on first runner hang

**File:** `Q:/finance-analyzer/portfolio/escalation_gate.py:32`, used at `:203-210`
**Issue:** `_RUNNER_EXECUTOR = ThreadPoolExecutor(max_workers=1)` and the call site does `_fut = _RUNNER_EXECUTOR.submit(call, prompt); raw = _fut.result(timeout=10)`. On TimeoutError, `_fut.cancel()` is called but Python's Future.cancel only succeeds if the task has not yet started. By definition a 10-second-overrun task is already running. The cancel is a no-op. The single worker thread stays inside `query_llama_server` indefinitely (no inner timeout — see `portfolio/llama_server.py` query helper which uses HTTP `requests.post` without a connect/read timeout in some code paths).
**Impact:** First time the local Ministral server hangs (which CLAUDE.md memory notes happens — `Chronos-2`, `Kronos` GPU lock issues), the executor is dead for the lifetime of `portfolio.main`. Every subsequent trigger that reaches escalation_gate queues a new submit() that waits behind the wedged worker; after 10s each returns `ministral_unavailable` fail-open. The audit trail shows transient `ministral_unavailable` rows — looks normal. Real cost: every triggered cycle now escalates to Claude instead of being filtered. Multi-week silent cost overrun.
**Fix:** (a) Use a fresh thread per call rather than a pool — `threading.Thread(target=..., daemon=True)` + a result queue with `queue.get(timeout=10)`; the daemon thread won't block process exit and won't block other calls. (b) Pass a hard inner timeout to `query_llama_server` (defense in depth). (c) On the third consecutive `runner_timeout`, set a module-level circuit-breaker flag that short-circuits `should_escalate` to `(True, 0.0, "ministral_circuit_open")` for N minutes — same pattern as agent_invocation's auth-cooldown.
**Confidence:** high

### [P0] Multi-agent specialist timeout uses single-PID kill, leaks Node.js helpers on Windows

**File:** `Q:/finance-analyzer/portfolio/multi_agent_layer2.py:180-191` (Popen), `:218-221` (kill on timeout)
**Issue:** `subprocess.Popen(cmd, ...)` for specialists is launched WITHOUT `creationflags=CREATE_NEW_PROCESS_GROUP` (compare to `claude_gate._popen_kwargs_for_tree_kill()` which sets the flag for exactly this reason). On timeout, `proc.kill()` terminates only the direct child Node.js process. The Node child spawns MCP servers, the actual API client process, and any local-LLM helpers Claude uses — all of these stay alive as orphans. This is the exact "VRAM leak / zombie subprocess" failure mode documented in claude_gate.py:357-365 (`A-IN-2 2026-04-11`) — the fix was applied to claude_gate but NOT to multi_agent_layer2.
**Impact:** Every specialist timeout on Windows leaks 2-4 Node helper processes, each holding ~150MB and any GPU VRAM allocations. After a few timeouts the system hits OOM or GPU lock contention silently — the next signal generation cycle fails with `gpu_gate` lock timeout and signals_failed climbs without root cause. Specialists run T2+ which is 5-7 min each, and 3 specialists per invocation, so a timed-out invocation under multi_agent=true can leak 12 orphan processes.
**Fix:** Mirror `claude_gate._popen_kwargs_for_tree_kill()` and `_kill_process_tree()` here. Add the creationflags to the Popen call and replace `proc.kill()` with a taskkill `/T` invocation. The helpers already exist in claude_gate — import and reuse.
**Confidence:** high

### [P0] `_agent_log_start_offset` is set outside `_completion_lock` — wrong slice scanned for auth errors

**File:** `Q:/finance-analyzer/portfolio/agent_invocation.py:1116-1117`
**Issue:** `_agent_log_start_offset = agent_log_path.stat().st_size if ...` is assigned at module-global scope from inside `invoke_agent()` but the corresponding `with _completion_lock:` block ends at line 839 (kill-or-skip decision). The size capture at 1117 is OUTSIDE the lock. If the completion watchdog tick (running every 30s) lands between two consecutive `invoke_agent` calls in the unhappy path where the first call's `_agent_proc` was just killed and the second is about to be spawned, the watchdog reads the OLD `_agent_log_start_offset` (from this new spawn's stat()) while scanning the OLD subprocess's log content. The auth-error scanner then either (a) finds nothing because the offset is past the EOF of the old write, or (b) finds a fake auth marker in another invocation's slice.
**Impact:** Asymmetric: most of the time the offset is approximately correct, but in the exact failure mode the auth detection was built for (silent 0-exit auth error) the wrong byte range is scanned, the auth_error status is not assigned, and the 3-week silent outage reoccurs.
**Fix:** Move the entire spawn block (lines 1110-1232) inside `_completion_lock`. The lock is already a single thread of contention by design — the small added critical section is harmless.
**Confidence:** medium-high (the race window is small but real; verified by reading the watchdog tick at line 100-111 which calls `_check_agent_completion_locked` which reads `_agent_log_start_offset` via `_scan_agent_log_for_auth_failure`)

### [P0] `invoke_agent` returns `None` on specialist quorum failure — main.py mis-logs as `skipped_busy`

**File:** `Q:/finance-analyzer/portfolio/agent_invocation.py:1046`
**Issue:** Inside the multi_agent path, when `success_count == 0`, the code does `_log_trigger(reasons, "specialist_quorum_fail", tier=tier)` then `return` (line 1046, bare return → returns `None`). The function signature otherwise returns `bool`. Caller in `main.py:951-952`:
```
result = invoke_agent(reasons_list, tier=tier)
_log_trigger(reasons_list, f"invoked_{why}" if result else f"skipped_busy_{why}", tier=tier)
```
With `result=None` → falsy → writes a SECOND `_log_trigger` row with status `skipped_busy_<why>`. Two rows in `invocations.jsonl` for one event, the second incorrectly suggesting "busy" (the actual issue was specialist failure). This corrupts the `get_completion_stats` `total/success/failed/timeout/auth_error` rollup at lines 1672-1714 (because `skipped_busy_*` is not in `tracked_statuses` → silently dropped, but `specialist_quorum_fail` is also not in that set, so BOTH rows go missing from completion stats).
**Impact:** Completion-rate health metric on dashboard `/api/loop_health` undercounts specialist-quorum failures as zero. Operators don't see Claude budget being burned on quorum failures. Also: the auto-fix-agent dispatcher's category=`specialist_quorum_fail` critical_errors row (written at line 1033-1043) is now the only durable signal, and it's written via raw `atomic_append_jsonl` bypassing `claude_gate.record_critical_error` which means no dedup against repeat failures.
**Fix:** Change line 1046 from `return` to `return False`. Add `"specialist_quorum_fail"` to the `tracked_statuses` tuple at line 1681 (and route through `claude_gate.record_critical_error` for dedup, line 1033-1043).
**Confidence:** high

### [P1] `_agent_tier` / `_agent_reasons` set BEFORE `subprocess.Popen` outside completion lock — watchdog can observe wrong tier on old PID

**File:** `Q:/finance-analyzer/portfolio/agent_invocation.py:1158-1170`
**Issue:** Comment at 1152-1157 claims setting timing/tier BEFORE Popen avoids a watchdog race, but in fact this introduces a different race. Between line 1162 (`_agent_reasons = list(reasons)`) and line 1163 (Popen), the watchdog can fire, take the lock, read `_agent_proc` (which still points at a possibly-just-completed previous instance or None), and see `_agent_tier` mismatched with `_agent_proc`. If the previous invocation completed but `_agent_proc=None`, the watchdog's `if _agent_proc is None: return None` saves it. But if the previous invocation's `_agent_proc.poll() is None` was True (still running — should not happen because reentrancy check passed, but consider Popen failing AND the previous proc being kept), the watchdog now reads `_agent_timeout` and `_agent_start` from the NEW invocation and may decide the OLD process is in timeout when it isn't.
**Impact:** Edge case: Popen raises (the bare `except Exception` at 1233 catches it), `_agent_proc` was never reassigned, but `_agent_start/_agent_tier/_agent_reasons` ARE set for the new invocation, polluting the watchdog's view of the previous (still alive but unreferenced because Popen raised) subprocess. Comment at 1153 acknowledges this risk ("If Popen fails, `_agent_proc` stays None and the watchdog ignores the stale metadata") but assumes `_agent_proc` is None — true only if there was no previous invocation OR completion was processed.
**Fix:** Move the entire spawn block (including the global assignments) inside a single `_completion_lock` block. Symmetric with the reentrancy check that already holds the lock at 815-839.
**Confidence:** medium

### [P1] Perception gate force-bypasses on `consensus` keyword regardless of confidence — undermines budget gating

**File:** `Q:/finance-analyzer/portfolio/perception_gate.py:26, 52-55`
**Issue:** `_BYPASS_KEYWORDS = ("consensus", "F&G crossed", "post-trade")`. Any reason string containing the substring `consensus` causes `should_invoke` to return `(True, "bypass: 'consensus' in trigger")` BEFORE the `min_signal_strength` check. But the perception_gate's whole purpose is to filter low-value invocations, and the most common low-value invocation is exactly "consensus BUY at 32%" — sub-confidence consensus crossings that should be filtered.
**Impact:** Token budget waste. Trigger.py already has a 40%-confidence ranging-regime dampener and a configurable consensus_min_pct floor, so by the time perception_gate sees a "consensus" reason it has already passed those upstream gates. The bypass here is redundant at best and at worst overrides downstream gating. Compounded with the autonomous-first router (escalation_router.py), the bypass list lets the perception gate filter only F&G_crossed and post-trade — which are exactly the cases that should ALWAYS escalate. Net effect: perception_gate gates almost nothing.
**Fix:** Remove `"consensus"` from `_BYPASS_KEYWORDS`. The min_signal_strength check below it (lines 67-86) correctly handles consensus-with-real-confidence by reading the actual weighted_confidence from the compact summary.
**Confidence:** high (this is a direct logic error vs the gate's stated purpose)

### [P1] `health_state.last_invocation_tier` is written at spawn, never cleared at completion — T1 silent failure stays undetected for 20 min

**File:** `Q:/finance-analyzer/portfolio/agent_invocation.py:1206-1208` (writer); `portfolio/loop_contract.py:215-230, 312` (reader)
**Issue:** `invoke_agent` writes `health["last_invocation_tier"] = effective_tier` at spawn. Nothing in `check_agent_completion_locked` clears it. The loop_contract `check_layer2_journal_activity` uses this tier to choose the grace window: T1=12min, T2=12min, T3=20min. After any T3 run, the next several T1 invocations inherit the T3 grace window (20m) until a fresh T1 invocation overwrites the value at the START of its spawn. Catch: agent_invocation publishes `effective_tier` only at spawn, so a T1 that gets SKIPPED upstream (drawdown block, perception gate, no_position_skip, auth_cooldown) never overwrites the stale T3 value. Many consecutive skips leave the T3 grace in place.
**Impact:** A T1 silent failure that should fire `layer2_journal_activity` violation at 12min now waits 20min — 8 extra minutes of silent failure with no alert. This was the exact class of bug the 2026-04-17 dynamic-grace work was meant to fix; the bookkeeping is incomplete.
**Fix:** In `_check_agent_completion_locked` (line 1632-1649) cleanup block, also clear `health["last_invocation_tier"]` and `health["last_invocation_tier_ts"]` via `atomic_write_json`. Alternatively, gate the loop_contract grace lookup by `_agent_proc is not None` (in-flight) — if no invocation is currently in flight, use a base default grace.
**Confidence:** high

### [P1] `analyze.py` subprocess does NOT strip `CLAUDE_CODE_ENTRYPOINT` — "nested session" error possible

**File:** `Q:/finance-analyzer/portfolio/analyze.py:23-37`
**Issue:** `_clean_env()` pops `CLAUDECODE` but NOT `CLAUDE_CODE_ENTRYPOINT`. Every other Claude invocation path (`agent_invocation._clean_env`, `claude_gate._clean_env`, `multi_agent_layer2.launch_specialists`) pops both. MEMORY.md explicitly notes: "`CLAUDECODE` env var — If inherited by loop/Task Scheduler, `claude -p` fails ('nested session' error). Fix: `set CLAUDECODE=` in bat file. Caused 34h outage Feb 18-19." The same class of bug applies to `CLAUDE_CODE_ENTRYPOINT` — at minimum it's inconsistent with the documented pattern.
**Impact:** When `analyze.py` is invoked from within a Claude Code interactive session (which is the documented usage: user runs `.venv/Scripts/python.exe portfolio/main.py --analyze ETH-USD` while their parent shell has Claude active), the subprocess inherits CLAUDE_CODE_ENTRYPOINT and may fail with the nested-session error. Returns blank output, prints "Claude returned code N" — silent breakage.
**Fix:** Add `env.pop("CLAUDE_CODE_ENTRYPOINT", None)` to `_clean_env`. One line.
**Confidence:** high

### [P1] Swedish market holiday calendar defined but NEVER consulted — Avanza warrants trade on Swedish holidays

**File:** `Q:/finance-analyzer/portfolio/market_timing.py:198-241`
**Issue:** `swedish_market_holidays()` and `is_swedish_market_holiday()` exist. A grep across the entire `portfolio/` tree shows the only references are the function definitions themselves — NO production caller. By contrast `is_us_market_holiday` is referenced at lines 256, 290, 336 inside `_is_agent_window`, `is_us_stock_market_open`, and `get_market_state`. The metals subsystem trades Avanza warrants 08:15-21:55 CET (per CLAUDE.md), but `_is_agent_window` only excludes US holidays. On Swedish-only holidays (Midsummer Eve, Whit Monday, National Day, Epiphany, May Day, Boxing Day, Ascension), warrant trading is suspended on Avanza — but the Layer 2 / metals loop will still fire signals and try to invoke the agent. Avanza orders will reject; iskbets/metals_loop will silently leave alerts that the user cannot act on.
**Impact:** Wasted Layer 2 invocations on days the user cannot trade Swedish warrants. Worse: signal triggers fire, alerts get sent, but Avanza API order placements fail with HTTP errors — the metals system may crash-loop on order placement during a Swedish holiday until the operator notices. CLAUDE.md memory specifically notes "Avanza commodity warrants: 08:15-21:55 CET" — but DST shift handling in metals-avanza.md says "Check API for `todayClosingTime`" — yet the per-cycle agent_window check uses `_is_agent_window` which only looks at US holidays.
**Fix:** Either (a) add `is_swedish_market_holiday(now)` check to a new `_is_metals_window()` function consulted by `data/metals_loop.py` and warrant-trading paths, or (b) add Avanza API liveness check before warrant order placement. (a) is cheaper and matches the existing pattern.
**Confidence:** high

### [P1] `_check_recent_trade` updates baseline only when at least one portfolio file readable — false trade-detection on next cycle after IO error

**File:** `Q:/finance-analyzer/portfolio/trigger.py:199-227`
**Issue:** `_check_recent_trade` iterates Patient + Bold, catches `KeyError`/`AttributeError` per portfolio (line 221) and skips. The new baseline `new_tx_counts[label] = current_count` (line 217) is set INSIDE the try, so a portfolio with an IO error doesn't get its baseline updated. `last_checked_tx_count` is updated at line 225 only when `new_tx_counts` is non-empty — so a complete IO failure on both portfolios LEAVES the old baseline in place. Next cycle, when files are readable again, the txn count delta will be >0 because the baseline is stale → fires `post-trade reassessment` for a trade that already triggered last cycle.
**Impact:** False "post-trade reassessment" trigger after any disk hiccup. Spawns a Tier 2 invocation (per `classify_tier`) for a non-existent trade. Costs Claude budget; loop_contract `layer2_journal_activity` then expects a journal entry for a fake trigger.
**Fix:** Catch `OSError`/`Exception` (not just KeyError/AttributeError) — and on IO failure for ALL portfolios, return `False` WITHOUT updating baseline. Add a log.warning so operators see the IO hiccup.
**Confidence:** medium

### [P1] `bigbet` direct `import subprocess` is unused but a residue of the pre-claude_gate era — risk of regression

**File:** `Q:/finance-analyzer/portfolio/bigbet.py:8`
**Issue:** `import subprocess` at line 8 but only `claude_gate.invoke_claude_text` is used (line 176). The import is dead. Same in `iskbets.py:9` (`import subprocess` but uses `invoke_claude_text` at line 323). Both files used to spawn Claude directly — the migration to `claude_gate` for cost tracking left the import. Dead imports themselves are P3 but they're a regression footgun: a future PR could re-introduce a `subprocess.Popen(["claude", ...])` call inside these modules without tripping linter "unused import" warnings since the name is bound.
**Impact:** Future regression to the pre-claude_gate cost-tracking bypass. Each non-gate call bypasses the auth-error scan, the rate limiter, and the daily cost rollup.
**Fix:** Delete the bare `import subprocess` from both files. Make new direct invocations fail import.
**Confidence:** medium

### [P1] `_acquire_singleton_lock` writes PID metadata AFTER acquiring the lock but BEFORE writing — partial corruption risk

**File:** `Q:/finance-analyzer/portfolio/main.py:73-92`
**Issue:** `fh = open(_SINGLETON_LOCK_FILE, "a+")` opens in append mode (line 74), then `fh.seek(0); msvcrt.locking(...)` (line 78-80) takes the lock, then `fh.seek(0); fh.truncate(); fh.write(f"{os.getpid()}\n")` (line 87-89) replaces content. If the loop process is SIGKILLed between truncate (line 88) and write+flush (line 89-90), the lock file is left zero-byte. The OS releases the lock on process exit so a new instance can take it — fine — but the lock file content is empty. Other tooling that reads the file for PID inspection (e.g., `loop_processes.scan`, monitoring scripts) sees empty content and assumes "stale lock with no owner" → may try to forcibly remove it.
**Impact:** Low frequency, but the lock file is the singleton guard — any tooling that mis-handles an empty lock file could spawn a duplicate loop.
**Fix:** Use `portfolio/process_lock.py:acquire_lock_file` which already exists and does this correctly via `_write_lock_metadata` that opens, locks, writes atomically through a single flush. Replace `_acquire_singleton_lock` with a thin wrapper that calls `acquire_lock_file(... owner="main_loop", metadata={...})`.
**Confidence:** medium

### [P2] `invoke_claude` paths in `claude_gate` do NOT set `NODE_OPTIONS=--stack-size=16384`

**File:** `Q:/finance-analyzer/portfolio/claude_gate.py:164-177` (_clean_env), used at `:602, 737`
**Issue:** `agent_invocation.invoke_agent` and `multi_agent_layer2.launch_specialists` set `NODE_OPTIONS=--stack-size=16384` to prevent Claude CLI stack overflow (`_STACK_OVERFLOW_EXIT_CODE = 3221225794`, file `agent_invocation.py:1125-1128`). The auto-disable after 5 consecutive stack overflows confirms this is a real production failure mode. However, `claude_gate._clean_env()` (used by `invoke_claude`/`invoke_claude_text` for bigbet, iskbets, claude_fundamental, self-heal, analyze) does NOT set the stack size. Same Claude binary, same risk of stack overflow on long contexts — but no auto-disable counter in those paths either.
**Impact:** `bigbet` / `iskbets` / `claude_fundamental` invocations can stack-overflow silently. The `auto-disable after 5` counter is per-module — claude_gate has none. A stack overflow in `bigbet` won't be caught by the agent_invocation counter.
**Fix:** Move `NODE_OPTIONS` augmentation into `claude_gate._clean_env()` so every gated invocation gets the protection. Remove the duplication from `agent_invocation` and `multi_agent_layer2` (they call into the same env-builder pattern).
**Confidence:** medium

### [P2] Price baseline reset every cycle defeats slow-drift price_move triggers

**File:** `Q:/finance-analyzer/portfolio/trigger.py:496-501`
**Issue:** `state["last"]["prices"] = dict(prices_usd)` runs unconditionally every cycle (comment at 496-498 says this prevents stale-comparison false positives after quiet periods). But the 2% price_move threshold compares `current vs prev_prices = prev.get("prices", {})` which is captured at line 272 BEFORE this update — so the baseline for next cycle's comparison is THIS cycle's current price. A 1.8% move per 10-min cycle for 8 cycles = 15% cumulative move never triggers, because each cycle resets the comparison floor. Combined with the 10-min `INTERVAL_MARKET_OPEN`, a slow grinding move during a "ranging" regime won't trigger price_move.
**Impact:** Slow trending moves (the kind that matter for swing positioning) don't fire `price_move` triggers — the system relies entirely on sustained flip + consensus crossing + F&G for trend detection. F&G updates daily; consensus + flip both require 3+ cycles. A 15% grinding move that takes 80 minutes is invisible to the trigger system until consensus eventually flips.
**Fix:** Two-baseline pattern: one "last_trigger_baseline_prices" that's updated only when a trigger fires (preserving comparison from the LAST trigger), and one "last_cycle_prices" for the stale-comparison guard. The current code conflates them.
**Confidence:** medium

### [P2] `_startup_grace_active` is a module global mutated by `check_triggers` — non-reentrant

**File:** `Q:/finance-analyzer/portfolio/trigger.py:127, 231-271`
**Issue:** `_startup_grace_active = True` at module scope, `global` declared in `check_triggers`. Pattern only works if `check_triggers` is called from one thread. The main loop calls it from the main thread (line 808 in main.py) — fine in production. But the test harness or any future caller (signal_engine? trigger_buffer?) that calls `check_triggers` concurrently could observe the grace flag inconsistently. The startup-grace logic only matters on the FIRST call after a process start anyway; thread safety is not a real bug today but it's a footgun.
**Impact:** Minor. Risk is more about future regressions than current behavior.
**Fix:** Either inline the grace check into per-call state (compare `os.getpid()` against stored PID, no module global needed), or wrap with `threading.Lock`. The PID-compare path already exists at line 239 — `_startup_grace_active` is redundant once PID compare is the source of truth.
**Confidence:** low

### [P2] `loop_health.write_heartbeat` swallows ALL exceptions including writability — silent loop liveness loss

**File:** `Q:/finance-analyzer/portfolio/loop_health.py:179-227`
**Issue:** `write_heartbeat` catches bare `Exception` (line 223) and returns False. The docstring says "best-effort by design" but it doesn't log at WARNING level — only `logger.debug` (line 225). Default log level in production is INFO; debug logs are dropped. So a permanent disk-full / permission-denied condition silently disables heartbeat writes for ALL loops that use this helper (crypto, oil, mstr, metals, golddigger per `DEFAULT_HEARTBEAT_FILES`). The watchdog then alerts on stale heartbeat — but the operator's only visible signal is "loop stale" with no indication that the loop itself is alive and the failure is in telemetry.
**Impact:** Misdiagnosis. Operator restarts a healthy loop because telemetry is broken; restart doesn't fix the underlying write failure; alert continues to fire.
**Fix:** Log at WARNING with rate-limit (e.g., log once per minute on consecutive failures). `logger.debug` → `logger.warning` with a counter.
**Confidence:** medium

### [P2] `signal_postmortem` writes `_epoch` into the JSON dict that downstream consumers read

**File:** `Q:/finance-analyzer/portfolio/main.py:382-397`
**Issue:** Lines 388-391: `result = generate_postmortem(); result["_epoch"] = time.time(); _awj(POSTMORTEM_FILE, result)`. The `_epoch` field is mutation of the generator's return dict for cache-staleness checks (line 386). Any downstream consumer of `data/postmortem.json` (likely the dashboard at `/api/postmortem` or `/api/signal_postmortem`) now gets a `_epoch` field it didn't ask for. If the dashboard JSON-schema-validates the response, this leaks an internal timestamp marker as if it were data.
**Impact:** Minor — JSON pollution. But if any consumer iterates keys (e.g., a UI table), it'll show `_epoch` as a label.
**Fix:** Wrap: `staleness_marker = {"_epoch": time.time(), **result}` and write a separate `data/postmortem.meta.json` for the staleness check OR keep the staleness check in-memory only and just compare `time.time() - file_mtime(POSTMORTEM_FILE)`.
**Confidence:** medium

### [P3] `agent_invocation.py:42-66` module-level mutable globals make per-test isolation expensive

**File:** `Q:/finance-analyzer/portfolio/agent_invocation.py:42-87`
**Issue:** 16+ module-level mutable globals (`_agent_proc, _agent_log, _agent_log_start_offset, _agent_start, _agent_start_wall, _agent_timeout, _agent_tier, _agent_reasons, _journal_ts_before, _telegram_ts_before, _journal_count_before, _telegram_count_before, _patient_txn_count_before, _bold_txn_count_before, _consecutive_stack_overflows, _watchdog_thread, _watchdog_stop`). Each test that exercises this module must reset all of them. Comment at line 144-158 acknowledges the test-only `_stop_completion_watchdog` exists for xdist hermeticity — a clear signal that the global pattern is leaking into tests.
**Fix:** Encapsulate in an `AgentState` dataclass instance. Reduces test boilerplate and makes the lock semantics explicit (one state object → one lock).
**Confidence:** low (refactor risk; existing tests pass)

### [P3] `escalation_router._ticker_held` opens portfolio JSON twice per call

**File:** `Q:/finance-analyzer/portfolio/escalation_router.py:136-146`
**Issue:** For each ticker in reasons, calls `_ticker_held` which calls `load_json` on BOTH `portfolio_state.json` and `portfolio_state_bold.json`. With N reasons, that's 2N file reads per `should_escalate_to_claude` call. Per-cycle cost is small (KB-sized state files) but the function is on the hot path.
**Fix:** Pre-load both portfolios once at the top of `should_escalate_to_claude` and pass holdings dict down. Bonus: aligns with the `held_positions` parameter that the sister `escalation_gate.should_escalate` already accepts.
**Confidence:** low

### [P3] `loop_contract.check_journal_uniqueness_safe` reads entire JSONL into memory

**File:** `Q:/finance-analyzer/portfolio/loop_contract.py:1244-1265`
**Issue:** `with open(...) as f: lines = f.readlines()` reads the full journal into memory to take the last 50 entries. Comment at line 1252-1255 acknowledges this and says "if this ever grows, swap to seek-from-end". Journal grows ~hundreds of entries per year — not a crisis, but the comment itself flags it.
**Fix:** Use `portfolio/file_utils.load_jsonl_tail` (already exists, used in agent_invocation.py:476).
**Confidence:** low

---

## Cross-cutting observations (not findings, for context)

- **No P0 finding on auth-detection** — the 2026-04-13 work on `detect_auth_failure` + the BUG-ECHO 2026-04-16 fix + per-stream scan + line-anchor check are thorough. The ONLY auth-detection gap is the `_agent_log_start_offset` race (P0 above).
- **`PF_HEADLESS_AGENT` env var is correctly set in all 4 Claude-spawning paths** (`agent_invocation`, `claude_gate._clean_env`, `multi_agent_layer2`, `analyze`). The CLAUDE.md hang failure pattern is well-defended.
- **Crash backoff is robust** — `_safe_crash_recovery` wraps with a guaranteed floor sleep (line 1207-1243 main.py), and the counter persists across restarts (`crash_counter.json`).
- **The completion watchdog at agent_invocation.py:90-141 is well-designed** — daemon thread, shared lock, swallows tick exceptions so the watchdog can't self-kill.
- **`process_lock.acquire_lock_file` exists and is correct** but is NOT used by main.py's singleton guard (which reimplements the pattern inline — see P1 finding above).
- **Multi-agent layer 2 has a `cleanup_reports` call at the START of `launch_specialists` (line 135) but NOT at the END after synthesis** — comment at line 246 says "after synthesis" but no caller invokes it post-synthesis. The next launch's pre-cleanup catches it. Minor confusion only.
