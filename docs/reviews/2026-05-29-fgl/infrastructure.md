# Adversarial Code Review: Infrastructure Subsystem
2026-05-29 | finance-analyzer

## Severity Summary
- P0 (State Corruption): 3
- P1 (Race Condition): 4
- P2 (Robustness): 3

---

## P0 Findings (State Corruption / Hard Failures)

### portfolio/file_utils.py:269 | P0: Windows sidecar lock never unlocked on crash
Windows msvcrt.locking() does NOT auto-release on process exit. If atomic_append_jsonl acquires the lock and crashes before explicit LK_UNLCK, the sidecar lock file remains locked forever. POSIX fcntl.flock releases on close (correct). Windows lock exhaustion halts all JSONL writers.
- Risk: Stale lock blocks signal_log, critical_errors, telegram_messages indefinitely.
- Fix: Add psutil-based stale-lock cleanup to jsonl_sidecar_lock (same pattern as gpu_gate). Check mtime > 5min AND PID dead before auto-reap.

### portfolio/telegram_poller.py:361 | P0: Config.json symlink write destroys external secrets
config.json is a symlink to external file with API keys. _handle_mode_command() does load_json -> modify -> atomic_write_json. If symlink target becomes unreachable between read and write, the real file can be corrupted or overwritten with partial JSON. Race window: load_json (L345) to atomic_write_json (L361).
- Risk: Loss of API keys if symlink target deleted during write window.
- Fix: Enforce read-then-write atomicity. Better: forbid config writes at runtime; use transient state file for notification mode.

### portfolio/http_retry.py:76 | P0: Silent error swallow - no fatal-error distinction
fetch_with_retry() returns None after all retries exhausted. Caller cannot distinguish transient (503, timeout) from fatal (401 bad token, 400 malformed). Bad token masquerades as transient; loop retries forever, wasting resources, never alerts user.
- Risk: Telegram auth failure silent forever.
- Fix: Return tuple (response, error_type) or result object with {success, retryable, status_code}. Caller distinguishes retry-later from fail-fast.

---

## P1 Findings (Race Conditions / Missing Guards)

### portfolio/shared_state.py:89 | P1: Dogpile prevention lacks max-retry bounds
When _cached() finds key in _loading_keys, returns stale without retry. Lines 68-74 evict stuck keys >120s. BUT: if loading thread crashes at L92, exception path attempts retry; if that also fails, re-enters exception path. All 8 workers pile on, filling error log.
- Risk: Broken signal causes retry storm.
- Fix: Add _max_error_retries counter; if >3 failures in 30s, force-evict key and return None.

### portfolio/health.py:64 | P1: Heartbeat read-modify-write not atomic
heartbeat() and update_health() call load_health (read) -> modify dict -> atomic_write_json (write). Between read and write, another thread's update is lost. Thread A reads, Thread B reads, Thread A writes, Thread B writes (overwrites A's changes).
- Risk: Concurrent error counts lost.
- Fix: _health_lock exists (L17) but only wraps write. Extend to wrap read-modify-write in heartbeat() L83-86.

### portfolio/process_lock.py:39 | P1: Lock file first-write TOCTOU race
acquire_lock_file() opens file in 'a+' mode. If two processes call simultaneously on brand-new lock file (size 0), both may succeed before _lock_file() is called. Windows msvcrt.locking() with LK_NBLCK can let both lock the same byte if timing overlaps.
- Risk: Two processes both believe they acquired singleton lock.
- Fix: Use os.open(lock_path, os.O_CREAT | os.O_EXCL) for atomic create; fall back to current locking only if file exists.

### portfolio/shared_state.py:208 | P1: Enqueue-retry race when should_enqueue_fn raises
When should_enqueue_fn() raises exception (L200), code defaults to should_enq=True (L206). But if key already in _loading_keys, retry is skipped (L208). Caller gets None instead of waiting for refresh. If exception repeats, key stuck in _loading_keys forever (or 120s eviction).
- Risk: Broken rotation gate stops permanently.
- Fix: On exception, either force-enqueue or evict key BEFORE checking _loading_keys.

---

## P2 Findings (Robustness / Coverage)

### portfolio/logging_config.py:27 | P2: Root logger named 'portfolio', not root
Line 27: getLogger('portfolio'). Child loggers like getLogger('portfolio.file_utils') work, but true root (getLogger()) misses portfolio.* messages. RotatingFileHandler only attaches to 'portfolio' logger, so third-party logs (requests, urllib3) fill stdout but dont rotate.
- Risk: Third-party library messages not captured in portfolio.log.
- Fix: Attach handlers to true root logger, or set propagate=True on root.

### portfolio/log_rotation.py:379 | P2: Rotation fsync-to-replace transient crash window
Lines 379-507: with jsonl_sidecar_lock wraps rotation. After writing kept lines and fsync (L503), os.replace (L504) happens. On Windows, tiny window between fsync completion and actual metadata flush. If process dies between fsync and replace, kept file lost (temp orphaned).
- Risk: Rare data loss if rotation crashes between fsync and replace.
- Fix: Add second fsync after os.replace() to guarantee metadata flushed.

### portfolio/telegram_notifications.py:105 | P2: No escalation on Telegram outage
_do_send_telegram() returns True/False; send_or_store() logs accordingly. If Telegram down (503, timeout), loop retries every 60s but message logged 'sent:false'. No escalation to alert user (e.g. 'Telegram unreachable 2h').
- Risk: Silent bot disconnection; user unaware.
- Fix: Add _telegram_outage_tracker; if >10 consecutive failures over 10min, write to critical_errors.jsonl.

---

## Summary

3 P0 (state corruption), 4 P1 (race conditions), 3 P2 (robustness).

Top 5:
1. Windows sidecar lock hangs (file_utils.py:269)
2. Config.json symlink write destroys secrets (telegram_poller.py:361)
3. Silent Telegram failure (http_retry.py:76 + telegram_notifications.py:105)
4. Health state lost (health.py:64)
5. Lock TOCTOU (process_lock.py:39)
