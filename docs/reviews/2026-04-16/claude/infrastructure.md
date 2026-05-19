# Adversarial Review — infrastructure (Claude-independent)

## P0 — Critical

### 1. `atomic_append_jsonl` not concurrency-safe on Windows
**File:** `portfolio/file_utils.py:155-167`
Standard `open(path, "a")` + fsync. On Windows with concurrent threads, O_APPEND + Python buffering does NOT guarantee non-interleaved writes. Corrupted JSONL breaks Layer-2 context recovery.
**Fix:** file-level `threading.Lock` OR switch to `atomic_write_jsonl` (rewrite entire file) for critical appends.

### 2. shared_state cache TTL read races with clear/write
**File:** `portfolio/shared_state.py:50-56, 166-172`
Line 109 `max_stale = ttl * _MAX_STALE_FACTOR` crashes with TypeError if another thread cleared TTL field → inconsistent cache state; stale returns None instead of last-known value.
**Fix:** validate TTL is int/float before arithmetic; default to constant if missing.

### 3. http_retry retries POST on timeout — breaks idempotency
**File:** `portfolio/http_retry.py:31-32, 58-65`
Retries POST on 500-504 and timeout. POSTs are not idempotent. Telegram sendMessage timeout → retry → duplicate message.
**Fix:** don't auto-retry POSTs on timeout; require caller to supply idempotency key.

## P1

### 4. health heartbeat uses wall clock — vulnerable to NTP jump-back
**File:** `portfolio/health.py:25`
`datetime.now(UTC).isoformat()` for heartbeat; `check_staleness` subtracts → negative age on clock jump → false "healthy".
**Fix:** `time.monotonic()` for staleness; wall clock only for logs; guard `age >= 0`.

### 5. telegram_poller offset — duplicate command replay on crash
**File:** `portfolio/telegram_poller.py:53-75`
`self.offset` updated in memory only after processing. Crash between execute and persist → restart reprocesses. `_startup_time` filter is 60s; crash + restart within 60s still replays.
**Fix:** atomically persist offset to disk before executing (or immediately after each update).

### 6. message_store.py — no dedup on rapid re-send
**File:** `portfolio/message_store.py:87-102`
`log_message()` appends without duplicate check. Same alert re-fired (retry, race) logs twice → Telegram duplicate.
**Fix:** `(category, text_hash, timestamp_bucket)` dedup before append.

### 7. alert_budget.py — check-then-append race
**File:** `portfolio/alert_budget.py:36-46`
`should_send()` and `_prune_old()` not atomic. Two threads both pass `len(sent) < max_per_hour` check → both append → budget overflow.
**Fix:** introduce `_lock` and wrap check-append under it.

### 8. Rate limiter vulnerable to clock jump
**File:** `portfolio/shared_state.py:247-262`
`elapsed = now - self.last_call` with wall clock; NTP jump forward makes elapsed huge → burst through rate limit.
**Fix:** cap elapsed at 1 interval; use `time.monotonic()`.

### 9. subprocess_utils Job Object assignment race
**File:** `portfolio/subprocess_utils.py:114-149`
`AssignProcessToJobObject()` called without checking process exit status (line 132-140). Suppressed silently. Child can survive parent death as orphan.
**Fix:** check `proc.poll()` before assignment; reap immediately if already exited.

### 10. file_utils.atomic_write_json temp file leak on FileExistsError
**File:** `portfolio/file_utils.py:27`
On Windows with AV active, `os.replace(tmp, target)` can fail FileExistsError if target is locked. Temp NOT cleaned up; OSError catch doesn't cover FileExistsError subclass. Repeated writes exhaust temp dir.
**Fix:** explicit cleanup on FileExistsError; retry briefly; log failure.

## P2

### 11. logging rotation not atomic with write
**File:** `portfolio/logging_config.py:43-47`
Size check before each write; two threads both under limit → both write → file exceeds maxBytes before rollover. RotatingFileHandler partial serialization.
**Fix:** explicit rotation at cycle start; guard size check under handler lock.

### 12. reporting.agent_summary partial failure silently skipped
**File:** `portfolio/reporting.py:739`
`_atomic_write_json(AGENT_SUMMARY_FILE, summary)` atomic; but if summary-construction raises (e.g., line 326 fundamentals), function exits without writing → Layer 2 reads stale. Per-module warnings collected but not reliably surfaced.
**Fix:** catch per-module exceptions, record in `_module_warnings`, ensure full summary is always written.

### 13. config_validator.py doesn't check types or extra keys
**File:** `portfolio/config_validator.py:33-47`
Presence-only; `config["telegram"]["token"] = 123` (int) passes, crashes later.
**Fix:** jsonschema or custom type validation.

### 14. backup.py no integrity check post-copy
**File:** `portfolio/backup.py:36`
`shutil.copy2()` no hash verification. Corrupt backup invisible until restore.
**Fix:** SHA256 before and after; abort on mismatch.

### 15. journal_index.py O(n) full scan per query
**File:** `portfolio/journal_index.py:370-381`
Loads entire `layer2_journal.jsonl` (68MB) per call. GC pressure; main loop slowdown.
**Fix:** in-memory incremental index updated on append; cache in shared_state.

## P3

### 16. prophecy.py no belief conflict detection
**File:** `portfolio/prophecy.py:200-270`
Contradictory beliefs (silver_bull + silver_bear) both evaluate independently. Layer 2 gets conflicting macro signals.
**Fix:** conflict detector; auto-disable contradictories.

### 17. NewsAPI budget reset UTC-only, DST misalignment
**File:** `portfolio/shared_state.py:312-320`
UTC midnight reset; active hours CET. Resets drift by 1h at DST boundaries.
**Fix:** reset at market-open timezone.

### 18. reflection.py reads portfolio_state.json without lock
**File:** `portfolio/reflection.py:149-151, 182-183`
Informational only but TOCTOU between Layer-2 write and reflection read.
**Fix:** use `atomic_write_json` consistently for writes; read lock for reflection.

## Looked OK

- **atomic_write_json** — tempfile + os.replace + fsync, handles cleanup (except P1 #10 FileExistsError).
- **load_json / require_json** — TOCTOU handled via try/except.
- **load_jsonl_tail** — efficient, truncation-safe.
- **http_retry backoff + jitter** — correct for GET.
- **http_retry Telegram retry_after** — 429 response honored.
- **health.update_health** — locked; ring buffer of last 20 errors correct.
- **journal.build_context** — formatting correct.
- **message_store.send_or_store** — category routing, mute gates correct.
- **message_store Markdown fallback** — retries without formatting on parse error.
- **telegram_notifications.escape_markdown_v1** — escapes safely.

## Reviewer confidence
0.87 for P0/P1 (patterns observable in live logs: stale cache timeouts, duplicate messages possible). 0.70 for P2 (requires specific conditions). 0.50 for P3 (edge cases).
