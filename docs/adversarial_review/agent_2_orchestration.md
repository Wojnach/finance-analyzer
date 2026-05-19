# Adversarial Review — Orchestration Subsystem

**Subsystem:** Orchestration (main loop, Layer 2 invocation, triggers, autonomous fallback, gates)
**Files reviewed:** 18
**Reviewer:** agent_2_orchestration
**Date:** 2026-05-19

## Files

- portfolio/main.py
- portfolio/agent_invocation.py
- portfolio/trigger.py
- portfolio/trigger_buffer.py
- portfolio/autonomous.py
- portfolio/market_timing.py
- portfolio/session_calendar.py
- portfolio/claude_gate.py
- portfolio/gpu_gate.py
- portfolio/shared_state.py
- portfolio/process_lock.py
- portfolio/loop_contract.py
- portfolio/loop_health.py
- portfolio/loop_processes.py
- portfolio/escalation_gate.py
- portfolio/escalation_router.py
- portfolio/multi_agent_layer2.py
- portfolio/perception_gate.py

## Findings by severity

- **P0 (silent failure / data loss / bad trade risk):** 2
- **P1 (real bug, hits production within a week):** 9
- **P2 (latent bug):** 5
- **P3 (minor):** 2

Total: 18 findings.

---

## P0 — Critical

### P0-1 — singleton lock not released on Linux/WSL; stale lock file may block restart
**File:** `portfolio/main.py:95-109` (`_release_singleton_lock`)

`_acquire_singleton_lock` calls `fcntl.flock(fh, fcntl.LOCK_EX|LOCK_NB)` on POSIX (line 82), but `_release_singleton_lock` only handles the Windows `msvcrt.LK_UNLCK` path (line 101-103). On Linux/WSL the `fcntl` branch is silently skipped — release relies entirely on the process-exit kernel cleanup that follows. If `atexit` runs but the process lingers (e.g., subprocess inherited the fd via `subprocess.Popen` without `close_fds=True` — the metals loop / dashboard may), the lock can be held longer than expected and the next loop restart hits `_DUPLICATE_EXIT_CODE = 11`. On Windows the metadata file is also not truncated/cleared, leaving stale PID hints behind. **Fix:** mirror the acquire branch — call `fcntl.flock(fh, fcntl.LOCK_UN)` in the else branch before closing. Use `portfolio/process_lock.py` (which already does this correctly) instead of the duplicate inline implementation.

### P0-2 — `future.result()` exception in ticker pool propagates out, crashes whole cycle
**File:** `portfolio/main.py:616-617`

```
for future in as_completed(futures, timeout=_TICKER_POOL_TIMEOUT):
    name, result = future.result()
```

`_process_ticker` does have its own try/except (line 558), but it returns `(name, None)` on exception. However, **if `_process_ticker` itself raises BEFORE the inner try-block** (Python module-level errors, an import error inside `_process_ticker`'s lazy imports, or an unhandled error in `future.result()` due to a dead thread), `future.result()` re-raises into the for-loop. The only `except` is `TimeoutError`, not generic `Exception`. A single bad ticker can crash the entire cycle, triggering `_safe_crash_recovery` + Telegram alert spam + losing all OTHER tickers that already completed successfully. **Fix:** wrap `future.result()` in try/except — log ticker name + exception, mark as failed, continue iterating remaining futures.

---

## P1 — Important

### P1-1 — Layer 2 "off-hours skip" silently disables ALL decision-making (no autonomous fallback)
**File:** `portfolio/main.py:972-979`

```
elif layer2_cfg.get("enabled", True):
    if _is_agent_window():
        ...invoke_agent(...)
    else:
        logger.info("Layer 2: outside market window, skipping")
        _log_trigger(reasons_list, "skipped_offhours", tier=tier)
```

When layer2 is enabled but `_is_agent_window()` is False (weekend, US holiday, or before EU open / after NYSE close), the trigger is logged as `skipped_offhours` but autonomous_decision is NOT called. The only path that falls back to autonomous is `else:` (layer2 disabled, line 980) OR `escalation_router` path with `autonomous_first_enabled`. Crypto and metals trade 24/7, so off-hours triggers on those (e.g., XAG-USD F&G crossing on Saturday morning) get fully dropped — no journal, no Telegram. **Fix:** in the "outside agent window" branch, also call `autonomous_decision` to maintain the recommendation pipeline.

### P1-2 — Autonomous fallback NOT triggered when `invoke_agent` returns False (skipped_busy / blocked_drawdown / blocked_trade_guards / skipped_auth_cooldown / skipped_gate)
**File:** `portfolio/main.py:973-976` + `agent_invocation.py:684-936`

`invoke_agent` can return False for 6+ different gating reasons (stack-overflow auto-disable, auth cooldown, drawdown block, trade-guards block, no-position-skip, perception gate skip, multi-agent failure, drawdown unavailable). All these paths just log `skipped_busy` (line 976) — autonomous never runs. A drawdown block specifically means the trader most needs feedback yet receives none. **Fix:** distinguish "skipped because busy" (the original intent — agent already running) from "blocked / gated", and route gated paths to autonomous.

### P1-3 — Rate limiter uses wall-clock `time.time()` — vulnerable to NTP jumps
**File:** `portfolio/shared_state.py:269-279` (`_RateLimiter.wait`)

```
now = time.time()
elapsed = now - self.last_call
...
self.last_call = self.last_call + self.interval if wait_time > 0 else now
```

`last_call` is wall-clock. If NTP adjusts the clock forward by 30s (common on Windows after sleep/wake), elapsed = 30s + interval, all subsequent calls bypass throttling — burst could exceed Binance/Alpaca per-minute caps and trigger 429. If NTP jumps backward, callers sleep for impossible durations. Most modules in this repo use `time.monotonic()` for elapsed (see `_safe_elapsed_s`, `loop_contract`); rate limiter should too. **Fix:** switch to `time.monotonic()`.

### P1-4 — `_cached_or_enqueue` calls `enqueue_fn` while holding `_cache_lock` — blocks all cache reads
**File:** `portfolio/shared_state.py:157-214`

The entire body of `_cached_or_enqueue` runs inside `with _cache_lock:` (line 157). `enqueue_fn` is invoked at line 205. If `enqueue_fn` does any blocking work (queue.put with bound, network call, file I/O, llama_server check), it holds `_cache_lock` and freezes the other 7 ThreadPoolExecutor workers waiting on cache reads. The dogpile prevention design assumes `enqueue_fn` is O(1) push but the codebase enqueues into shared queues that may have locks of their own — deadlock potential if `enqueue_fn` acquires a lock that other cache callers transitively hold. **Fix:** release `_cache_lock` before calling `enqueue_fn`; or atomically reserve the loading state, release, then enqueue.

### P1-5 — `trigger_buffer.add()` / `flush_due()` are not atomic across processes — concurrent writes lose data
**File:** `portfolio/trigger_buffer.py:121-138, 160-196`

Both `add()` and `flush_due()` do load → mutate → save with no file lock. If two callers (main loop reset + metals loop / dashboard) call concurrently, one's append is silently overwritten when the other saves. The "Remove flushed atomically" comment in the docstring is misleading — `atomic_write_json` is atomic at the OS level (rename) but the read-modify-write sequence is NOT atomic. **Fix:** wrap reads + writes with `portfolio/process_lock.py` advisory lock on a sentinel file, or move buffer state to SQLite WAL where this is solved.

### P1-6 — `escalation_gate` thread-pool leaks on success path (always `shutdown(wait=False, cancel_futures=True)` in finally)
**File:** `portfolio/escalation_gate.py:202-219`

```
_ex = _cf.ThreadPoolExecutor(max_workers=1)
try:
    _fut = _ex.submit(call, prompt)
    try:
        raw = _fut.result(timeout=10)
    except _cf.TimeoutError:
        ...
        _ex.shutdown(wait=False, cancel_futures=True)
        return ...
    finally:
        _ex.shutdown(wait=False, cancel_futures=True)
```

`shutdown(wait=False)` returns immediately. On timeout, the inner ministral call is still running (you can't cancel a running thread in Python). Successive failed invocations spawn a new `ThreadPoolExecutor` each time AND leave a hung thread holding GPU/llama_server resources — over hours/days this leaks threads. Each ThreadPoolExecutor also allocates a queue and metadata structure. **Fix:** reuse a single module-level executor; or use a daemon thread with a `threading.Event` cancellation signal.

### P1-7 — `_no_position_skip` reads agent_context_t1.json — never reads agent_context_t2 / compact, so T2/T3 invocations also fall under the T1 lens
**File:** `portfolio/agent_invocation.py:355-374`

```
ctx = load_json(DATA_DIR / "agent_context_t1.json", default={}) or {}
```

`_no_position_skip` is called from `invoke_agent` regardless of tier, but it only reads the T1 context file. T2/T3 use larger summaries with more signals; using T1 here means the gate evaluates a degraded signal view and can produce false positives (skipping a T2 invocation that the T2 context would show as a strong-entry signal). **Fix:** select context file by tier, or use the full agent_summary_compact.json.

### P1-8 — Completion watchdog runs at 30s interval but T1 tier timeout = 180s — silent agent could run up to 210s past budget
**File:** `portfolio/agent_invocation.py:81, 1311-1374`

`_COMPLETION_WATCHDOG_INTERVAL_S = 30` and `_check_agent_completion_locked` checks `elapsed > _agent_timeout`. Worst case: agent finishes its actual work at t=180s but watchdog's last tick was at t=178s, next tick at t=208s. That's 28s of wasted budget on the happy path; on the kill path the agent is allowed to hang up to 30s past its tier timeout. For T1 (180s) that's a 16% over-budget; for T2 (600s) it's 5%; for T3 (900s) it's 3%. Not catastrophic but conflicts with the documented "30s of real budget" claim. **Fix:** lower interval (10s); or compute deadline = start + timeout and have watchdog wake at deadline; OR document the slack in TIER_CONFIG.

### P1-9 — `multi_agent_layer2.wait_for_specialists` checks log file AFTER closing log_fh but `proc.kill()` does not wait for stdout buffer flush
**File:** `portfolio/multi_agent_layer2.py:206-237`

```
proc.kill()
proc.wait(timeout=5)
results[name] = False
finally:
    log_fh = getattr(proc, "_log_fh", None)
    if log_fh:
        log_fh.close()

# Auth-error scan AFTER close
try:
    log_path = ...
    text = log_path.read_text(...)
    if detect_auth_failure(...): ...
```

After `kill()`, the OS may not have flushed the killed process's `stderr=STDOUT` redirection to the log file. The scan can miss auth-error output that the dying CLI just printed. Also, the inner `proc.wait(timeout=5)` after kill may itself time out (Claude CLI Node.js teardown can exceed 5s on Windows per BUG-189), leaving a zombie that subsequently writes to a now-closed log_fh — UB. **Fix:** use longer wait (15s, matching `_kill_overrun_agent`); explicitly fsync log_fh before close.

---

## P2 — Latent

### P2-1 — `_run_post_cycle` uses `config` loaded ONCE at `loop()` startup — config edits not picked up
**File:** `portfolio/main.py:1314, 1328, 1363`

```
config = _load_config()        # line 1314
...
_run_post_cycle(config, ...)   # 1328 (initial) and 1363 (each cycle)
```

`run()` itself reloads config each cycle (line 447), but `_run_post_cycle` uses the boot-time config. Edits to `notification.mode`, `claude_budget.*`, `bigbet.enabled`, `gpu_signals.*`, `layer2.*` only take effect on next process restart. The system runs 24/7 with crash-recovery — restart is not guaranteed within hours of an edit. **Fix:** reload `config` inside the while-loop (cheap if `_load_config` is mtime-cached, which it is per the trigger.py comment).

### P2-2 — Auth-cooldown scan stops on first non-skip entry — `auth_error` w/ empty `ts` field bypasses cooldown
**File:** `portfolio/agent_invocation.py:701-721`

```
for entry in reversed(recent[-50:]):
    status = entry.get("status", "")
    if status.startswith("skipped"):
        continue
    ts = entry.get("ts", "")
    if status == "auth_error" and ts:
        ... cooldown check ...
    break
```

If the most recent non-skip entry is `auth_error` but `ts` is empty (truncated row, partial write during crash), the inner `if` is False, code falls through the empty body, and `break` exits without firing cooldown — defeating the recovery mechanism that prevents the next loop from spawning another doomed Claude. **Fix:** treat missing ts on auth_error as "block invocation" (fail-safe), not "fall through".

### P2-3 — `_make_session_end` produces negative `utc_hour` for sessions starting just after midnight CET
**File:** `portfolio/session_calendar.py:82-89`

`utc_hour = cet_hour - offset`. With CEST (offset=2) and `cet_hour=1`, `utc_hour=-1`. `now.replace(hour=-1, ...)` raises ValueError. Currently safe because all `SESSIONS` use `open_cet >= 8`, but a future session with an early CET open (e.g., Asian-hour metals) would crash session_calendar. **Fix:** normalize `(cet_hour - offset) % 24` and adjust the date when the result wraps past midnight.

### P2-4 — `_check_recent_trade` doesn't initialize counts on first run — every restart fires "post-trade reassessment" trigger
**File:** `portfolio/trigger.py:193-221`

```
prev_count = last_checked_tx.get(label, current_count)
```

On a fresh `trigger_state.json` (post-restart or first install), `last_checked_tx` is empty, so `prev_count = current_count` and `current_count > prev_count` is False — looks safe. BUT the startup grace block (line 234) updates baselines and returns early WITHOUT updating `last_checked_tx`. The very next cycle: `prev_count = last_checked_tx.get(label, current_count)` is still defaulting to current_count. Looks correct for "no trades yet", but the moment a trade is recorded by a separate process (e.g., metals_loop, GoldDigger, or manual Avanza order) between cycles, every restart re-fires post-trade. **Fix:** record current tx counts during the startup grace block.

### P2-5 — `loop_processes.scan` exception on `psutil.process_iter` access kills the entire scan (logs warning but inside the per-process try, so OK) — but `_iter_processes` references `info` outside the try in the `except` clause
**File:** `portfolio/loop_processes.py:91-107`

```
for p in psutil.process_iter(["pid", "cmdline", "create_time", "name"]):
    try:
        info = p.info
        out.append(...)
    except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
        logger.debug("loop_processes: skip pid=%s err=%s", info.get("pid"), exc)
```

If `p.info` itself raises (psutil.NoSuchProcess on the first attribute read), `info` is never bound and the except clause raises `UnboundLocalError`. Wraps an exception in another exception, which then escapes the try entirely. **Fix:** assign `info = {}` before the try, or use `getattr(p, 'info', {})`.

---

## P3 — Minor

### P3-1 — `escalation_gate._log_decision` uses `datetime.utcnow()` (deprecated since Python 3.12)
**File:** `portfolio/escalation_gate.py:160`

```
"ts": _dt.datetime.utcnow().isoformat() + "Z",
```

`datetime.utcnow()` is deprecated since Python 3.12 and returns a naive datetime. Inconsistent with the rest of the codebase that uses `datetime.now(UTC)`. **Fix:** `datetime.now(UTC).isoformat()`.

### P3-2 — `_TICKER_BLOCKLIST` differs between `trigger_buffer` and `escalation_router` — same logical concept, divergent lists
**File:** `portfolio/trigger_buffer.py:28` vs `portfolio/escalation_router.py:150-153`

`trigger_buffer` blocklist: `{"BUY", "SELL", "HOLD", "USD", "EUR", "SEK"}`.
`escalation_router` blocklist: `{"BUY", "SELL", "HOLD", "USD", "EUR", "SEK", "ATR", "RSI", "BB", "MA", "TF"}`.

The shorter list will treat "ATR", "RSI" etc. as tickers, leading to wrong reason_type classification and wrong ticker-keyed buckets. **Fix:** consolidate into a single shared module-level constant (e.g., `portfolio/ticker_utils.py`).

---

## Cross-cutting observations (not findings — context)

1. **Duplicate singleton lock implementations.** `portfolio/main.py:42-110` reimplements what `portfolio/process_lock.py` already does correctly. Migrating saves ~70 lines AND fixes P0-1 in one stroke.
2. **Telegram alert suppression after 5 crashes (`_MAX_CRASH_ALERTS = 5`)** does include a 100-crash summary cadence, so visibility is preserved — this is NOT a finding, the design is intentional and correct.
3. **`PF_HEADLESS_AGENT=1`** is correctly set in both `agent_invocation.py:1053` and `multi_agent_layer2.py:160` and `claude_gate.py:_clean_env`. Good.
4. **`CLAUDECODE` env var pop** happens in all three subprocess paths — good.
5. **GPU lock stale recovery** (sweeper daemon, `_try_break_stale_lock`) is robust — the 2026-05-03 fix closed the 25-hour wedge hole correctly.
6. **DST handling** in `market_timing.py` is correct for spring-forward / fall-back (uses date arithmetic and UTC comparisons, not local-time hour math). The 09:30 ET open is correctly translated to 13:30 UTC EDT / 14:30 UTC EST.

---

## Recommended remediation order

1. **P0-2** (ticker pool crash) — silently loses cycle data; fix immediately.
2. **P0-1** (singleton release) — only bites on Linux/WSL, but the codebase explicitly supports WSL per CLAUDE.md.
3. **P1-1** (off-hours no fallback) — affects 24/7 crypto/metals coverage.
4. **P1-2** (gated paths no fallback) — drawdown blocks specifically need autonomous output.
5. **P1-3** (rate limiter wall clock) — NTP-jump prone, post-laptop-sleep.
6. **P1-4** (cache lock blocks enqueue) — deadlock risk under heavy llama_server contention.
7. **P1-5** (trigger_buffer races) — opt-in feature (batch_window_s=0 default), low immediate risk.
8. Remainder by severity.
