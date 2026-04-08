# Adversarial Review: infrastructure (Agent Findings)

Reviewer: Code-reviewer subagent (feature-dev:code-reviewer)
Date: 2026-04-08
Confidence scores included per finding.

---

## CRITICAL

### CI1. GPU lock file descriptor leaked on write failure — 5-min deadlock [92% confidence]
**File**: `portfolio/gpu_gate.py:126-128`

`os.open(O_CREAT|O_EXCL)` creates the lock file and returns an fd. If `os.write(fd, ...)`
raises (disk full), the fd is never closed and the lock file remains. Next caller sees
the file, `_read_lock()` returns `{}` (empty parts), PID=0, `_pid_alive(0)` returns False.
But `_is_stale()` checks mtime — fresh file won't be stale for 300s. Result: GPU locked
for 5 minutes with no owner + fd leaked for process lifetime.

**Fix**: Wrap write+close in `try/finally` with `os.close(fd)`, and on write failure
call `_release_lock()` before re-raising.

### CI2. journal.py context file written non-atomically — Layer 2 reads partial data [90% confidence]
**File**: `portfolio/journal.py:568,580`

`write_context()` uses `Path.write_text()` directly, not `atomic_write_json`. The context
file is the primary input Layer 2 reads for historical decisions. If the 60s loop writes
while a Claude subprocess reads, Layer 2 gets partial/empty context. Violates CLAUDE.md
rule #4: "Atomic I/O only."

**Fix**: Use tempfile+replace pattern or `atomic_write_json`.

---

## HIGH

### HI1. health.py fromisoformat timezone mismatch — kills dashboard health endpoint [88% confidence]
**File**: `portfolio/health.py:66`

If `last_invocation_ts` was written by older code using `datetime.utcnow().isoformat()`
(no `+00:00`), `fromisoformat()` returns naive datetime. `datetime.now(UTC) - naive_dt`
raises `TypeError`. Kills `get_health_summary()` on every dashboard request.

### HI2. claude_gate.py scans entire invocations.jsonl on every call — O(N) unbounded [86% confidence]
**File**: `portfolio/claude_gate.py:97-104`

`_count_today_invocations()` calls `load_jsonl()` with no limit. After months (3000+ records),
this becomes measurable on every 60s cycle. On Windows, holding the file open can block
concurrent `atomic_append_jsonl`.

**Fix**: Use `load_jsonl_tail()` with max_entries=200.

### HI3. update_module_failures no-ops on empty list — stale failures never cleared [85% confidence]
**File**: `portfolio/health.py:122-129`

When a cycle succeeds with zero failures, `update_module_failures([])` returns immediately
without clearing the old failure list. Dashboard shows old failures indefinitely — false positive.

**Fix**: Always write, setting to `None` when clean.

### HI4. `_loading_keys` not cleared on BaseException — permanent dogpile block [83% confidence]
**File**: `portfolio/shared_state.py:79-104`

`SystemExit` or `GeneratorExit` propagates without removing key from `_loading_keys`.
That signal is permanently dead for the process lifetime with no log entry.

**Fix**: Use `try/finally` instead of `try/except KeyboardInterrupt / except Exception`.

### HI5. http_retry returns 4xx responses as "success" objects — callers confused [82% confidence]
**File**: `portfolio/http_retry.py:36-49`

`fetch_with_retry()` returns the Response object for any non-retryable status (401, 403, 400).
Callers checking `if r is not None` treat auth failures as success. Structural ambiguity.

### HI6. claude_gate subprocess timeout doesn't kill child — orphaned processes accumulate [80% confidence]
**File**: `portfolio/claude_gate.py:219-221`

`subprocess.run()` raises `TimeoutExpired` but doesn't kill the child. Claude process
continues consuming resources. Next invocation spawns another — unbounded accumulation.

---

## MEDIUM

### MI1. compute_indicators min_rows doesn't account for BB(20) requirement [80% confidence]
**File**: `portfolio/indicators.py:29`

Min rows check uses `macd_slow` (17 or 26), but BB needs 20 rows. With 17-19 rows,
BB produces all-NaN → `price_vs_bb` always returns "inside" — suppresses BB signals.

**Fix**: `min_rows = max(macd_slow, 20)`.

### MI2. ThreadPoolExecutor uncaught thread exceptions invisible in logs [80% confidence]
**File**: `portfolio/logging_config.py:43-47`

`setup_logging()` only configures `portfolio` logger. Thread pool exceptions in
fire-and-forget patterns vanish silently.

**Fix**: Configure `threading.excepthook` or ensure all Future results are checked.

### MI3. message_store mojibake dict has shadowed duplicate keys [80% confidence]
**File**: `portfolio/message_store.py:55-60`

Unicode lookalike characters in dict keys — later entries silently overwrite earlier ones.
Some mojibake patterns never repaired, appearing as garbage in Telegram notifications.

---

## Cross-Critique: Claude Direct vs Infrastructure Agent

### Issues agent found that Claude missed entirely:
1. **CI1**: GPU lock fd leak on write failure — complete miss (I noted the GPU gate was
   "well designed" — agent found the edge case I overlooked)
2. **CI2**: journal.py non-atomic write — complete miss (violates CLAUDE.md rule #4)
3. **HI1**: health.py timezone mismatch (same pattern as trade_guards CR1 but different file)
4. **HI2**: claude_gate O(N) scan — complete miss
5. **HI3**: Stale module failures in health — complete miss
6. **HI4**: `_loading_keys` BaseException leak — complete miss
7. **HI5**: http_retry 4xx ambiguity — complete miss
8. **HI6**: claude_gate subprocess orphan — complete miss
9. **MI1**: indicators.py min_rows vs BB — complete miss
10. **MI2**: Thread exception invisibility — complete miss

### Issues Claude found that agent confirmed:
1. **H17/MI-overlap**: Health file read-modify-write race — both found
2. **M14**: atomic_append_jsonl partial writes — agent didn't re-raise but similar concern
3. **M16**: Telegram rate limiting — agent didn't cover this specifically

### Issues Claude found that agent didn't cover:
1. **M15**: load_jsonl_tail boundary line skip
2. **L3**: prune_jsonl memory usage

### Net assessment:
The infrastructure agent found **10 net-new issues** that the independent review missed.
Particularly impactful: the GPU lock fd leak (CI1), the journal non-atomic write (CI2),
and the `_loading_keys` BaseException leak (HI4). The independent review praised the GPU
gate as "well designed" — the agent correctly found the edge case that breaks it.

**Agent win rate for this subsystem: ~80%** — agent found most of the important issues
while Claude's direct review missed critical implementation details.
