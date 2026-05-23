# orchestration review — 2026-05-23

Scope: portfolio/{main,agent_invocation,trigger,trigger_buffer,market_timing,
multi_agent_layer2,autonomous,claude_gate,gpu_gate,loop_health,loop_contract,
loop_processes,process_lock,shared_state,escalation_gate,escalation_router,
perception_gate}.py — 10,525 LOC total.

## P0

### P0-1 — Multi-agent specialist log read after exit-0 auth failure is suppressed by stale exit code (silent auth bug class)
File: `portfolio/multi_agent_layer2.py:214-242`.

```python
success = proc.returncode == 0
results[name] = success
...
# 2026-04-13: Auth-error scan — specialist log is truncated per run
text = log_path.read_text(...)
if detect_auth_failure(text, caller=f"layer2_specialist_{name}", ...):
    results[name] = False
```

`launch_specialists` opens the specialist log with mode `"w"` at line 179 BEFORE
spawning the subprocess. That truncates the file but the file handle is then
written to by the Popen child. The auth scan at 233 re-opens the log path AFTER
`log_fh.close()` at 226, which is correct on the happy path. However the
synthesis prompt is then built and the parent (`agent_invocation.py:961-980`)
falls through to `_build_tier_prompt(tier, reasons)` if `procs` was empty, but
when `procs` is non-empty it builds `prompt = build_synthesis_prompt(...)` and
proceeds to spawn the synthesis Claude regardless of whether any specialist
returned `auth_error`. Quote `agent_invocation.py:967-972`:

```python
results = wait_for_specialists(procs, timeout=specialist_timeout)
success_count = sum(1 for v in results.values() if v)
logger.info("Specialists complete: %d/%d succeeded", success_count, len(results))
# Even if some fail, proceed with synthesis using available reports
prompt = build_synthesis_prompt(ticker, reasons)
```

`success_count` is logged but never gated on. If all three specialists hit auth
failure (the canonical `--bare`/expired-OAuth scenario), `wait_for_specialists`
records each via `detect_auth_failure` → `critical_errors.jsonl`, but the
synthesis agent still spawns with empty/missing `_specialist_*.md` reports.
The synthesis Claude has `--allowedTools Edit,Read,Bash,Write` and will happily
write a journal/decision with no data because the prompt's `Read <path>.` lines
silently produce empty files. This produces a "journal_written=true /
telegram_sent=true / status=success" invocation row that hides three upstream
auth_error events. The user gets a routine "Layer 2 invoked" Telegram with a
content-free trade decision.

Fix: when ≥1 specialist returned False AND any of the three logs trip
`detect_auth_failure`, short-circuit the synthesis launch and treat the parent
invocation as auth_error.

### P0-2 — `_kill_overrun_agent` clears `_agent_proc=None` before completion-watchdog's NEXT tick, but the killed PID can still be alive on Windows past wait timeout
File: `portfolio/agent_invocation.py:647-682`.

```python
try:
    _agent_proc.wait(timeout=15)
except subprocess.TimeoutExpired:
    if kill_ok:
        logger.error("Agent pid=%s did not exit after kill+15s wait", pid)
    kill_ok = False
...
if kill_ok:
    _agent_proc = None
else:
    logger.error("Kill failed for pid=%s — keeping _agent_proc to block respawn", pid)
```

This protects against double-spawn for the immediate next call but `kill_ok`
goes False on the 15s wait timeout EVEN IF the taskkill itself succeeded
(line 650-652: `if kill_ok: logger.error(...)` then `kill_ok = False`). The
zombie agent then permanently blocks all future Layer 2 invocations until
process restart, because `_agent_proc` is non-None and `_agent_proc.poll()`
will return None forever (PID gone, Popen tracks it as alive). At line 761
`if _agent_proc and _agent_proc.poll() is None:` infinitely returns the
"still running" branch. There is no recovery path other than loop restart.
Combined with the singleton lock in `main.py:_acquire_singleton_lock`, the
operator must kill the entire main loop process to unwedge Layer 2.

Fix: if taskkill returned 0/128 BUT `wait()` timed out, re-poll PID via
`psutil.pid_exists(pid)` — if the OS no longer has the PID, clear `_agent_proc`
anyway (Popen object is stale). Otherwise mark a circuit-breaker file and exit
the loop process so the supervisor restart can clear state.

### P0-3 — `_safe_elapsed_s` falls back to wall-clock when monotonic is "poisoned", but `_agent_start_wall = 0.0` is the sentinel — if Popen succeeds and then a code path resets monotonic state without resetting wall, the timeout logic permanently disarms
File: `portfolio/agent_invocation.py:487-527, 1562-1563`.

Cleanup at `_check_agent_completion_locked` does:

```python
_agent_start = 0
_agent_start_wall = 0.0
```

Both reset to 0. The next `invoke_agent` sets both at line 1080-1081 before
Popen. But on Popen FAILURE (the inner `except Exception as e:` at line 1155)
the locals are NOT reset — `_agent_start` and `_agent_start_wall` still point
at the previous successful invocation's timestamps. `_agent_proc` remains the
PRIOR object (the `_agent_proc = subprocess.Popen(...)` on line 1085 raised
before assignment, but the prior `_agent_proc` from the previous run was
already cleared to None at line 1560 in the prior completion path). Net
effect: usually safe. But the comment at line 1074-1079 explicitly worries
about Popen-failure leaving stale `_agent_start` — and the actual code path
in the exception handler at 1155-1159 only closes `log_fh`. No reset of
`_agent_start`, `_agent_start_wall`, `_agent_tier`, `_agent_reasons`,
`_agent_timeout`. If Popen fails AND a future code path observes
`_agent_proc=None` but reads `_agent_start_wall` directly (none in current
code, but the comment at 49-52 promises the invariant), the elapsed calc
would be unbounded-large.

Fix: in the Popen exception path at line 1155, reset all six module globals
to their idle values before returning False.

### P0-4 — Trigger baselines `state["last"]` are updated ONLY when `triggered=True` — a stuck no-trigger loop will perpetually compare to ancient prices
File: `portfolio/trigger.py:496-509`.

```python
if triggered:
    state["last_trigger_time"] = time.time()
    state["last"] = {
        "signals": {...},
        "prices": dict(prices_usd),
        ...
        "time": time.time(),
    }
```

If `check_triggers` returns `(False, [])` for 24 hours (e.g. all signals stuck
in HOLD because the accuracy gate is force-HOLD), `state["last"]["prices"]` is
never refreshed. Then a normal 2% intraday move that would normally NOT trigger
because cumulative drift was already absorbed by yesterday's baseline now
triggers on a cumulative move from 24h ago. The `last_trigger_time` and
`last["prices"]` are intertwined — the price-move trigger at line 410-432 uses
`prev.get("prices", {})` which is `state["last"]["prices"]` — so a long quiet
period followed by any cumulative drift past 2% from the last actual trigger
fires an "X moved 5% up" trigger on a slow 24h drift, not a meaningful move.

This is a missed false-positive class for high-volatility instruments
(silver/oil) during multi-day low-trigger periods. The "Ranging dampening" at
line 304-317 doesn't help — it only updates `triggered_consensus`, not
`state["last"]["prices"]`.

Fix: refresh the price baseline on every cycle when no consensus changes, but
keep `last_trigger_time` only for actual triggers.

## P1

### P1-1 — `_record_new_trades` race against concurrent portfolio writes
File: `portfolio/agent_invocation.py:1280-1313, 1097-1108`.

Between snapshot at 1100-1104 (`_patient_txn_count_before = len(...)` before
Popen) and `_record_new_trades` at 1478 (after subprocess exit), the
Layer 2 subprocess writes to `portfolio_state.json` via its own atomic-write
path. The completion handler then does its own non-atomic read at 1295:

```python
state = load_json(pf_path, default={}) or {}
txns = state.get("transactions", [])
if len(txns) <= count_before:
    continue
new_txns = txns[count_before:]
```

If the Layer 2 agent has trimmed/rotated transactions (e.g. via a separate
cleanup path), `len(txns) - count_before` slice can include transactions that
already existed under different indices. `record_trade` then double-counts.
There is no transaction ID matching — purely positional. Acceptable today
because no cleanup path exists, but the comment "BUG-219: snapshot transaction
counts" assumes append-only, which is not enforced anywhere.

Fix: snapshot a hash of the last txn record, not just the count.

### P1-2 — `_completion_watchdog` polls `_check_agent_completion_locked` every 30s but the same lock is held during the FULL completion handler including auth scan + telegram send. The watchdog can be blocked behind a slow `send_or_store` (no timeout on telegram path)
File: `portfolio/agent_invocation.py:90-111, 1485-1504`.

```python
def _completion_watchdog() -> None:
    while not _watchdog_stop.is_set():
        if _watchdog_stop.wait(_COMPLETION_WATCHDOG_INTERVAL_S):
            return
        try:
            with _completion_lock:
                _check_agent_completion_locked()
```

`_check_agent_completion_locked` at 1485-1494 calls `send_or_store(...)` while
holding `_completion_lock`. `send_or_store` ultimately hits Telegram HTTP. If
Telegram is slow (90s timeout per requests in `http_retry`), the lock is held
for 90s and the main loop's `run()` → `check_agent_completion` at `main.py:437`
blocks for 90s waiting on the lock. Cycle time inflates by 90s on every L2
completion + slow Telegram. Worse, `try_invoke_agent` also takes the lock at
line 760, so concurrent triggers block.

Fix: release the lock before doing Telegram/file I/O. Build the result dict +
clear state under the lock, send notifications after release.

### P1-3 — Singleton lock byte-range race: `_acquire_singleton_lock` uses `LK_NBLCK,1` on byte 0, but every process opens with "a+" which seeks to EOF — then `fh.seek(0)` happens AFTER `LK_NBLCK`
File: `portfolio/main.py:73-92`.

```python
fh = open(_SINGLETON_LOCK_FILE, "a+", encoding="utf-8")
try:
    fh.seek(0)
    if msvcrt is not None:
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
```

Actually the seek IS before the locking call here — code is correct, despite
the comment fearing the order. BUT the second `fh.seek(0); fh.truncate();
fh.write(...)` at 87-90 occurs AFTER the lock is acquired and does not
re-lock. On Windows, msvcrt locks span byte ranges, so `truncate()` past byte
0 is unprotected by the lock — another process could observe a half-written
PID file. Low impact (it's just a PID display), but the design comment claims
"single-instance lock" which is correct, just the PID file content is racy.

### P1-4 — `process_lock.acquire_lock_file` returns the file handle but `_unlock_file` uses `msvcrt.LK_UNLCK,1` on byte 0 — on Windows, locks held by a process release at close() anyway, but the explicit unlock can fail silently if the file pointer is not at byte 0
File: `portfolio/process_lock.py:77-83`.

```python
def _unlock_file(fh: IO[str]) -> None:
    fh.seek(0)
    if msvcrt is not None:
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
```

OK, seek is present. But `release_lock_file` swallows `OSError` silently —
if the unlock fails (process already released, FD closed), the next process
trying to acquire could conceivably see the OS think a lock exists.
Low-likelihood on Windows. Worth a logger.warning.

### P1-5 — `claude_gate._is_real_auth_marker_line` rejects lines starting with `[` to filter JSON-log echoes — but a real CLI auth error printed as `[ERROR] Not logged in` would be filtered as false positive
File: `portfolio/claude_gate.py:206-231`.

```python
_AUTH_MARKER_PREFIX_REJECT = ("'", '"', "`", "(", ">", "[", " ", "\t")

def _is_real_auth_marker_line(line: str, marker: str) -> bool:
    ...
    if line[0] in _AUTH_MARKER_PREFIX_REJECT:
        return False
    return line.startswith(marker)
```

The function checks `line[0] in _AUTH_MARKER_PREFIX_REJECT`, but then asks
whether the line STARTS WITH the marker (`Not logged in`, etc.). The first
guard already rejects `[ERROR] Not logged in` because `line[0]=='['`. The
function never reaches the marker check for lines starting with `[`. If the
Claude CLI ever changes its auth error format to be prefixed by a tag
(common for newer Node CLIs — `[anthropic] Not logged in`), this falls back
to silent exit-0 — the original bug class. The current CLI behavior is
known but future versions could regress.

Fix: scan the line for the marker after stripping a possible bracketed prefix
like `\[[A-Z]+\]\s*`.

### P1-6 — Multi-agent specialist log opened mode `"w"` overwrites between specialist runs of different invocations, but the auth-scan reads it AFTER kill — if the kill path raised, the auth scan never runs
File: `portfolio/multi_agent_layer2.py:218-242`.

```python
except subprocess.TimeoutExpired:
    logger.warning("Specialist %s timed out, killing", name)
    proc.kill()
    proc.wait(timeout=5)
    results[name] = False
finally:
    log_fh = getattr(proc, "_log_fh", None)
    if log_fh:
        log_fh.close()
...
try:
    log_path = DATA_DIR / f"_specialist_{name}.log"
    if log_path.exists():
        text = log_path.read_text(...)
        if detect_auth_failure(text, ...):
            results[name] = False
```

If `proc.wait(timeout=5)` after kill itself raises (still alive on Windows),
the `finally` runs but the auth-scan block at 232-240 is still in the same
function — actually it's a sibling try-block AFTER the finally, so reachable.
OK on second read. But `proc.kill()` on Windows uses `TerminateProcess` on
the direct child only — not the tree. Unlike `claude_gate._kill_process_tree`,
specialist kills leak the Claude CLI's spawned MCP/Node helpers. Repeated
specialist timeouts will accumulate orphan Node processes.

Fix: route specialist kill through `claude_gate._kill_process_tree`.

### P1-7 — `escalation_gate.should_escalate` uses `concurrent.futures.ThreadPoolExecutor(max_workers=1)` per call and `shutdown(wait=False)` on timeout — leaks threads if Ministral hangs forever
File: `portfolio/escalation_gate.py:202-219`.

```python
_ex = _cf.ThreadPoolExecutor(max_workers=1)
try:
    _fut = _ex.submit(call, prompt)
    try:
        raw = _fut.result(timeout=10)
    except _cf.TimeoutError:
        ...
        _ex.shutdown(wait=False, cancel_futures=True)
        return True, 0.0, "ministral_unavailable"
    finally:
        _ex.shutdown(wait=False, cancel_futures=True)
```

`cancel_futures=True` cancels QUEUED futures, but the running future (the
hung Ministral call) cannot be cancelled — it runs to completion. The
ThreadPoolExecutor object is GC'd but the worker thread continues blocking
in `llama_server.query_llama_server` (which has its own internal timeout but
no caller-visible kill). Each escalation_gate timeout leaks one thread. Over
a week of degraded llama_server, the process accumulates hundreds of stuck
threads holding HTTP sockets / GPU lock attempts.

Fix: hold a module-level executor (`max_workers=1`, reused), so concurrent
timeouts don't leak. Or implement a hard timeout in `query_llama_server` and
let it raise.

### P1-8 — `trigger.check_triggers` writes `state["last"]` and prunes baselines in one atomic write but reads PORTFOLIO files via `_check_recent_trade` without coordinating with the agent's own portfolio writes
File: `portfolio/trigger.py:199-227, 489-518`.

A T2/T3 Layer 2 invocation can be writing to `portfolio_state.json` while the
NEXT main-loop cycle's `check_triggers` reads `transactions` count to detect a
"post-trade reassessment". If the agent has appended a transaction but not yet
committed `cash_sek` (multi-step write — actually portfolio_mgr uses atomic
write, so the file replacement is atomic, but the trigger reads can observe a
state where transactions[+1] but cash_sek has not been debited if the agent
made multiple atomic writes mid-decision). Net effect: `post-trade
reassessment` fires twice per trade in rare timing windows.

Low impact (just one extra T2 invocation per trade), but the user has burned
budget on this exact failure mode (see auth_error cooldown at line 706-734).

### P1-9 — `_kill_overrun_agent` writes `_log_trigger(..., "timeout", ...)` BEFORE `_scan_agent_log_for_auth_failure` returns — auth_error never overrides the "timeout" status in invocations.jsonl
File: `portfolio/agent_invocation.py:661-682`.

```python
auth_label = f"layer2_t{_agent_tier}_timeout" if _agent_tier else "layer2_timeout"
_scan_agent_log_for_auth_failure(auth_label)

# BUG-91: Log the timed-out invocation before returning
_log_trigger(
    _agent_reasons or fallback_reasons or [],
    "timeout",
    tier=_agent_tier or fallback_tier,
)
```

The auth scan WRITES to `critical_errors.jsonl` (good) but doesn't return its
result here — the timeout status is logged unconditionally. So when a hung
agent printed "Not logged in" then froze, the `invocations.jsonl` shows
`status=timeout` not `status=auth_error`. The auth cooldown at line 706-734
walks `invocations.jsonl` looking for `status == "auth_error"` — it will miss
this case and immediately re-spawn another doomed agent. This recreates the
exact "20 doomed spawns in 30 min" pattern the cooldown was added to prevent.

Fix: capture `auth_detected = _scan_agent_log_for_auth_failure(...)` and log
`status="auth_error"` instead of `"timeout"` when True.

### P1-10 — `trigger.py` startup grace check uses module-level `_startup_grace_active` boolean, persists `last_loop_pid` to disk. If two main loop processes race on startup (singleton lock failure window), both observe `_startup_grace_active=True` and both write a different PID
File: `portfolio/trigger.py:127, 230-269`.

The singleton lock in `main.py:_acquire_singleton_lock` runs before
`trigger.check_triggers` is ever called, so in practice the second process
exits at the singleton check. But if `_acquire_singleton_lock` returns True on
both (file-locking race on a network filesystem, or Windows OneDrive sync
toggling), both processes silently update baselines and skip triggering. Low
likelihood on local Q: drive but worth noting.

### P1-11 — `loop_contract.check_layer2_journal_activity` uses `last_invocation_tier` from health_state to pick grace, but `agent_invocation.invoke_agent` writes that field BEFORE Popen on line 1128 — if Popen fails, health_state has stale T3 tier from a prior run
File: `portfolio/agent_invocation.py:1118-1132`, `portfolio/loop_contract.py:215-230`.

```python
effective_tier = 3 if not claude_cmd else tier
health_path = DATA_DIR / "health_state.json"
health = load_json(health_path, default={}) or {}
health["last_invocation_tier"] = effective_tier
health["last_invocation_tier_ts"] = datetime.now(UTC).isoformat()
atomic_write_json(health_path, health)
```

Written pre-Popen. If Popen raises (line 1155), the field is now T3 (or
whatever) but no agent ran. The next contract check will use T3 grace (20min)
for what was never actually invoked. False-negative: real journal silence
goes unalerted because the grace window says "still in flight".

### P1-12 — `agent_invocation.invoke_agent` at 906-928 reads `should_block_trade(guard_result)` and only blocks if BOTH strategies blocked. The `for w in guard_result["warnings"]` list iteration uses `w.get("strategy") or w.get("details", {}).get("strategy")` — case sensitivity (`"patient"` vs `"Patient"`) is not normalized
File: `portfolio/agent_invocation.py:910-921`.

```python
blocked_strategies = {
    w.get("strategy") or w.get("details", {}).get("strategy")
    for w in guard_result["warnings"]
    if w.get("severity") == "block"
    ...
}
blocked_strategies.discard(None)
if {"patient", "bold"}.issubset(blocked_strategies):
    ...
```

`trade_guards.record_trade` (called at 1307) writes `strategy=strategy`
verbatim from the caller. The caller at 1291-1294 passes literal lowercase
`"patient"` / `"bold"`. But `reporting.py`'s guard-warning writer (separate
module) may emit different casing. If it ever emits `"Patient"` or `"BOLD"`,
the set membership check fails and the block never fires. Trade gets through.

Fix: normalize `.lower()` on both sides.

## P2

### P2-1 — `_no_position_skip` reads `agent_context_t1.json` for entry-strong-signal check, but tier 2/3 invocations don't generate t1 context — the gate fails open on T3
File: `portfolio/agent_invocation.py:355-377`.

Acceptable (fail-open), but the comment claims this is a Claude-budget saving
measure. On T3 the context file might be stale or missing entirely.

### P2-2 — `multi_agent_layer2.SPECIALISTS["risk"].timeout=90` but the global `specialist_timeout` at `agent_invocation.py:966` defaults to 30s, then `wait_for_specialists(procs, timeout=specialist_timeout)` enforces 30s — the per-spec timeout field is dead
File: `portfolio/multi_agent_layer2.py:43-66`, `portfolio/agent_invocation.py:966-967`.

```python
"risk": {..., "timeout": 90, "max_turns": 8},
```

vs. main caller:

```python
specialist_timeout = config.get("layer2", {}).get("specialist_timeout_s", 30)
results = wait_for_specialists(procs, timeout=specialist_timeout)
```

`wait_for_specialists` uses a SHARED deadline (`deadline = time.time() + timeout`)
for all three specialists in a single linear loop — so the slowest specialist
gets whatever's left after the others. Per-spec timeouts in SPECIALISTS are
unused. The comment at 962-964 says "30s to avoid blocking the main loop"
but with three Claude subprocesses each needing 30-60s for tool turns, 30s is
guaranteed to time out at least one. Multi-agent mode is effectively broken.

### P2-3 — `escalation_router._ticker_held` re-reads portfolio JSON files on every reason in every cycle — under a 20-reason trigger storm, 40 file reads
File: `portfolio/escalation_router.py:136-146`.

Cheap on SSD but inefficient. Memoize per call.

### P2-4 — `gpu_gate._release_lock` uses `unlink(missing_ok=True)` but ignores `OSError` from filesystem busy — if Windows has the file open by a dying process, the unlink fails silently and the next acquire hits FileExistsError, retrying until the 60s deadline
File: `portfolio/gpu_gate.py:98-100, 211-237`.

The `_try_break_stale_lock` path handles dead-PID, but not "file unlinkable
because Windows handle still open". Could deadlock GPU contention for full
60s after a crash. The sweeper daemon at 137 will reap eventually (after
mtime > 300s), but the immediate acquirer waits the full timeout.

### P2-5 — `_safe_crash_recovery` uses `min(2 ** min(n, 16), 30)` cap — for n>=5 the cap is always 30s. After 5 crashes in a row, every subsequent crash waits exactly 30s — no further escalation
File: `portfolio/main.py:1235-1242`.

```python
floor = min(2 ** min(n, 16), _CRASH_FLOOR_SLEEP_CAP)
```

`_CRASH_FLOOR_SLEEP_CAP = 30`. After crash 5, this caps at 30s forever. The
main backoff (`_crash_sleep`) caps at 300s (`_MAX_CRASH_BACKOFF`). So when
`_crash_sleep` SUCCEEDS, we get 10-300s with jitter. When it FAILS (the
"floor sleep" path), we get 30s. The floor is LESS than the main backoff —
not really a safety floor for sustained failure. Crash 1000 with disk full
gives 30s spins.

### P2-6 — `loop_contract.check_layer2_journal_activity` returns `[]` (contract passes) when ANY precondition file is unreadable — the canonical silent-fail pattern
File: `portfolio/loop_contract.py:292-302`.

```python
cfg = load_json(CONFIG_FILE, default={}) or {}
if not cfg.get("layer2", {}).get("enabled", True):
    return []

health = load_json(HEALTH_STATE_FILE)
if not health:
    return []
```

If health_state.json is corrupt mid-write, returns `[]`. The doc comment at
286-288 explicitly defends this ("false-positive alerts would erode trust"),
but it's the same class as the auth-failure exit-0 bug — silent fall-through
when the input is unparseable. Worth a logger.warning at minimum.

### P2-7 — `trigger.SUSTAINED_DURATION_S=900` (15 min) combined with `INTERVAL_MARKET_OPEN=600` (10 min) means the duration gate fires after 2 cycles — but `SUSTAINED_CHECKS=3` requires 3 cycles. The OR-debounce is effectively just the duration gate
File: `portfolio/trigger.py:105-106`, `portfolio/market_timing.py:20`.

Not a bug — but the count gate is redundant now. Worth removing for clarity.

### P2-8 — `agent_invocation.py:1018-1022` pins model to `claude-sonnet-4-6` — hardcoded, no config override
Quote:

```python
cmd = [
    claude_cmd, "-p", prompt,
    "--model", "claude-sonnet-4-6",
    ...
```

If Anthropic deprecates that exact model ID, all Layer 2 invocations 404 with
no fallback. Better: pin via config.layer2.model with sonnet-4-6 default.

### P2-9 — `loop_processes.scan` uses substring matching on cmdline. Patterns like `"portfolio.mstr_loop"` would match a process running `python -c "import portfolio.mstr_loop"` (e.g. an investigative shell)
File: `portfolio/loop_processes.py:54-67`.

False-positive duplicate detection. Low impact.

### P2-10 — `trigger_buffer.flush_due` returns merged reasons but `_save_entries(path, remaining)` is non-atomic with `_load_entries(path)` — concurrent flushers can double-fire
File: `portfolio/trigger_buffer.py:160-196`.

The function `_save_entries` uses `atomic_write_json` (safe). But the
load-then-save sequence is not locked — two main-loop cycles racing on the
same file (unlikely with the singleton, but possible if loop iteration N+1
starts before N finished trigger-buffer logic) would each see the same
buffered entries and both emit. The trigger buffer is currently disabled by
default (`batch_window_s=0`) so dormant risk.

## P3

### P3-1 — `_load_stack_overflow_counter` returns 0 on missing file but the counter is only decremented to 0 on a non-overflow completion (`agent_invocation.py:1547-1552`). After 5 consecutive overflows L2 is auto-disabled and stays disabled across loop restarts.
File: `portfolio/agent_invocation.py:163-176, 691-697`.

By design — but `_MAX_STACK_OVERFLOWS=5` and the counter persists to disk.
After a process restart, the counter is loaded from disk and L2 stays
disabled with no operator alert beyond the original. Recovery requires
manual deletion of `data/stack_overflow_counter.json`. Should be documented.

### P3-2 — `agent_invocation._extract_ticker` defaults to `"XAG-USD"` if no regex matches
File: `portfolio/agent_invocation.py:269-285`.

A non-ticker trigger like "startup" or "F&G crossed 20" will produce
`feedback_ticker = "XAG-USD"` and dump XAG decision history into the prompt
context unrelated to the actual trigger. Misleads the agent.

### P3-3 — `claude_gate.invoke_claude_text` and `invoke_claude` both retry on timeout but no cap on TOTAL retries — a chronically slow Claude wedges per-cycle work
File: `portfolio/claude_gate.py:602-635, 736-771`.

Single attempt per call — no retry. But callers (silver_monitor, fundamentals)
may call repeatedly on each cycle. Without rate limiting beyond the daily
threshold of 50, a single bad day could blow through OAuth limits.

### P3-4 — `escalation_gate._log_decision` uses `_dt.datetime.utcnow().isoformat() + "Z"` instead of `datetime.now(UTC).isoformat()` — `utcnow()` is deprecated in Python 3.12+
File: `portfolio/escalation_gate.py:160`.

Cosmetic / future deprecation warning.

### P3-5 — `multi_agent_layer2.build_synthesis_prompt` references `data/trading_insights.md` with no existence check (unlike the T1/T2/T3 prompts in agent_invocation)
File: `portfolio/multi_agent_layer2.py:108-119`.

If the file is missing, the synthesis Claude burns a Read tool turn on
nothing. Minor budget waste.

### P3-6 — `autonomous._consensus_accuracy` caches for 5 minutes but never invalidates on signal-engine reload — stale accuracy data drives autonomous decisions for up to 5 min after a model reload event
File: `portfolio/autonomous.py:46-73`.

Low impact for paper trading.

### P3-7 — `loop_health.write_heartbeat` "best-effort, swallows all exceptions" — if heartbeat is the canary for the watchdog, swallowing failures defeats the canary
File: `portfolio/loop_health.py:208-226`.

By design but worth a CRITICAL log on failure so the operator notices.

### P3-8 — `shared_state._cached` returns None on exception when stale data exceeds `ttl*3` — but no critical_errors.jsonl entry. Cycle silently falls back to None forever
File: `portfolio/shared_state.py:109-125`.

Acceptable for data signals but stack overflow detection here would surface
the canonical "silent failure cascade" pattern.

### P3-9 — `agent_invocation.invoke_agent` config double-load (commit 4c58628e was supposed to consolidate, see commit msg) but line 1145 inside `try` block reloads config to escape MarkdownV1
File: `portfolio/agent_invocation.py:1144-1146`.

```python
try:
    config = _load_config()
    reason_str = escape_markdown_v1(", ".join(reasons[:3]))
```

The outer config was loaded at 738. Reloading at 1145 wastes the I/O the
commit claimed to remove. Probably a missed reuse.

## Coverage notes

Read in full: agent_invocation.py (1646 LOC), trigger.py (657), main.py
(1532, lines 1-1000), multi_agent_layer2.py (255), trigger_buffer.py (202),
market_timing.py (342), escalation_gate.py (229), escalation_router.py (269),
perception_gate.py (95), process_lock.py (107), gpu_gate.py (266), claude_gate.py
(841), loop_health.py (235), loop_processes.py (173), shared_state.py (388),
autonomous.py (first 200 of 846).

Spot-grepped: loop_contract.py (2442 LOC) — sampled `_get_layer2_grace_s`,
`check_layer2_journal_activity`, `_dispatch_critical_errors_for_degradation`,
`verify_and_act`. Did NOT read the remaining 2000 LOC in detail.

Stdin/input audit across portfolio/: only `ministral_trader.py:262` and
`qwen3_trader.py:240` read stdin, both are short-lived helpers spawned with
piped stdin by `llama_server`/`llm_batch`, not by L2 paths. No `input()` or
`getpass()` reachable from L2 subprocess code paths.

The canonical "exit 0 + Not logged in" bug class is reasonably defended in
`detect_auth_failure` + the cooldown at `invoke_agent:706-734`, but P0-1
(specialist auth failure doesn't gate synthesis) and P1-9 (timeout path
loses auth status) re-open the same failure mode in adjacent code paths.

## 5-line summary

P0-1: multi_agent specialist auth failures don't gate synthesis spawn — empty-context decision possible (silent re-open of March-April outage class).
P0-2: `_kill_overrun_agent` permanently wedges Layer 2 when taskkill succeeds but `wait()` times out — only loop restart recovers.
P0-3: Popen-failure exception path leaks all six agent module globals — `_agent_start`/`_agent_start_wall` stay populated until next successful invoke.
P0-4: `state["last"]["prices"]` only refreshes on TRIGGER, so quiet 24h periods produce false "X moved 5%" triggers on cumulative drift from yesterday.
P1-9 critical: `_kill_overrun_agent` logs `status="timeout"` even when auth marker detected, so the auth_error cooldown at line 706 misses it and respawns immediately — the exact "20 doomed spawns" loop the cooldown was added to stop.
