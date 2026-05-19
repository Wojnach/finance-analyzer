# Adversarial Review: Infrastructure Subsystem (2026-05-16)

Scope: portfolio/file_utils.py, http_retry.py, health.py, gpu_gate.py, shared_state.py, telegram_notifications.py, telegram_poller.py, message_store.py, journal.py, journal_index.py, daily_digest.py, weekly_digest.py, digest.py, notification_text.py, loop_health.py, market_health.py, logging_config.py, config_validator.py.

---

## [P1] Windows rename race after fsync in atomic_write_json
**File:** portfolio/file_utils.py:59
**Bug:** `os.replace(tmp, str(path))` is invoked after fsync of the file contents, but the *directory* is not fsynced. On Windows + ReFS/NTFS power-loss windows, the tmp file may be synced while the directory entry rename is not durable.
**Why it matters:** Critical journal files (`layer2_journal.jsonl`, `signal_log.jsonl`) are audit trails. A power-loss between fsync and rename can leave the target stale while the tmp is renamed but undurable.
**Fix:** On POSIX, `os.fsync(dir_fd)` after replace. On Windows, leave as-is but document the limitation — Windows `os.replace` is atomic at the FS level but not necessarily durable across an immediate crash. Consider preferring NTFS transactional APIs for the highest-criticality files, or accept the trade-off and document.

---

## [P1] Dogpile-prevention race in shared_state stale-return path
**File:** portfolio/shared_state.py:79-89
**Bug:** Thread A detects key in `_loading_keys`, decides to return stale, releases the lock. Thread B re-adds the key. Thread A returns stale without removing its loading marker — the key now sits in `_loading_keys` indefinitely (until the 120s timeout), causing subsequent threads to return stale instead of refreshing.
**Why it matters:** Local-LLM signal accuracy (Ministral/Qwen3/Chronos) degrades because votes stay stale 2 minutes longer than designed.
**Fix:** Remove the loading-key entry under the same lock that decided to return stale; or use an event with proper cleanup.

---

## [P1] Naive vs aware datetime subtraction in check_agent_silence
**File:** portfolio/health.py:206
**Bug:** Computes `now = datetime.now(UTC)` at L206 but parses naive timestamp from invocations.jsonl at L202. `now - naive_dt` raises TypeError.
**Why it matters:** `/api/loop_health` and dashboard health endpoint crash and return 500; ops loses visibility of agent silence.
**Fix:** Normalize the parsed timestamp to UTC via `datetime.fromisoformat(...).replace(tzinfo=UTC)` if naive, or treat any naive parse as a bug and raise visibly.

---

## [P1] gpu_gate thread lock not released if _release_lock raises
**File:** portfolio/gpu_gate.py:265-266
**Bug:** `__exit__` calls `_release_lock()` (L264) before releasing the in-process thread lock (L266). If `_release_lock` raises (filesystem error), the thread lock is never released. Subsequent acquirers wait forever.
**Why it matters:** A single transient filesystem error wedges all LLM inference for the rest of the process lifetime. Loop appears alive but every GPU-bound signal HOLDs.
**Fix:** Wrap `_release_lock()` in try/finally so the thread lock release is unconditional.

---

## [P1] Telegram token in error paths
**File:** portfolio/telegram_notifications.py:55 and portfolio/telegram_poller.py:374
**Bug:** URL with token is built at L54-81 and may appear in tracebacks/log lines. Poller logs `r.text[:200]` which can include the token after server-side echoes.
**Why it matters:** API credential leak to logs that may be shipped to ops dashboards.
**Fix:** Build URL via session adapter that masks the bot token in repr; log only HTTP status and shortened response body with explicit `token=***` redaction.

---

## [P1] journal_index full-file rescan on every Layer 2 invocation
**File:** portfolio/journal_index.py:367-383
**Bug:** `retrieve_relevant_entries()` opens and reads the entire JSONL on every query. After 30 days the file is ~30 MB; on every Layer 2 invocation it reads the whole file.
**Why it matters:** Context loading eats 30-60s of the T1 (180s) budget. Repeated subprocesses thrash the disk; perf cliff at month boundary.
**Fix:** Maintain an in-memory tail index (offsets/dates) and seek to the relevant date range; or use the existing `load_jsonl_tail` helper.

---

## [P2] http_retry jitter applied before retry_after override
**File:** portfolio/http_retry.py:40-52
**Bug:** Jitter is added to base backoff at L41, *then* `retry_after` (server-supplied) may overwrite the wait at L44. If retry_after=300 and jitter was already added, the override is correct but the order makes the override misleading; worse, if the path computes `wait = max(wait, retry_after)` the jitter inflates the wait.
**Why it matters:** Telegram bans on repeated 429 violations. The function may violate `retry_after` contract.
**Fix:** Compute `retry_after` first, then jitter — and never exceed `retry_after` itself; use `wait = retry_after + small_jitter`.

---

## [P2] health.py error_count unbounded vs errors[] capped at 20
**File:** portfolio/health.py:40
**Bug:** Errors list is bounded to last 20; counter is not. Dashboard reports `error_count: 200, errors=[20 items]`.
**Why it matters:** Operators see catastrophic-looking counter while only 20 recent errors are inspectable. Decisions on alerts become noisy.
**Fix:** Display `errors_recent_24h` (counted from list) plus `errors_total` (running count). Or cap the counter to the visible list size and surface a separate metric.

---

## [P2] logging_config duplicate handlers on re-import
**File:** portfolio/logging_config.py:24-47
**Bug:** Module-level `_configured` flag persists across reloads. In subprocesses spawned via `multiprocessing`-like mechanisms, the flag is inherited but handlers are not — leading to repeated handler attachment in some paths.
**Why it matters:** Every WARNING logged twice; portfolio.log inflates, slowing rotation, increasing disk I/O.
**Fix:** Check `logger.handlers` before adding; remove existing matching handlers idempotently.

---

## [P2] message_store dedupe key collision
**File:** portfolio/message_store.py:87-102
**Bug:** Dedupe key `(timestamp, text, category)` allows two identical messages within the same second to both be stored if a sub-second race fires `send_telegram` twice.
**Why it matters:** Audit trail polluted by duplicates; downstream consumers double-count alerts.
**Fix:** Include a UUID v4 generated at the *first* call site (before send) and dedupe on that.

---

## [P2] config_validator errors list silently ignored
**File:** portfolio/config_validator.py:43-84
**Bug:** `validate_config()` returns an errors list but only `validate_config_file()` raises on non-empty. Direct callers of `validate_config()` receive errors silently.
**Why it matters:** A bad config edit intended to disable trading or reduce risk may not raise; process continues with old/wrong settings.
**Fix:** Have `validate_config` raise on first call site error; provide a separate `collect_config_errors` helper for tests/UI that want non-raising behavior.

---

## [P2] Digest timing tz mismatch (daily vs 4h)
**File:** portfolio/daily_digest.py:73 vs portfolio/digest.py:62
**Bug:** Daily digest uses local time, 4h digest uses UTC. At DST boundaries, daily digest fires at wrong wall-clock time and may arrive after the next 4h digest, breaking the expected order.
**Why it matters:** User receives morning briefing late or out of order on DST day; confidence in scheduled notifications drops.
**Fix:** Choose a single tz (UTC preferred) for all scheduled digests; document the wall-clock equivalent in the digest header.

---

## [P2] journal.load_recent miscounts on corrupted lines
**File:** portfolio/journal.py:28-40
**Bug:** Skips malformed JSON lines but the skipped lines count toward `max_entries`, so the returned list may be shorter than requested when corruption is present.
**Why it matters:** Layer 2 context sees fewer entries than expected on a day with a torn append.
**Fix:** Read entries until `max_entries` *valid* entries are collected; or scan in reverse, parse, and discard malformed.

---

## [P3] loop_health naive timestamp acceptance
**File:** portfolio/loop_health.py:99-100
**Bug:** Accepts a naive timestamp without normalizing; downstream subtraction with `datetime.now(UTC)` crashes (see also health.py finding above).
**Why it matters:** Same TypeError class. Low severity because the call sites currently always provide aware timestamps, but defensive.
**Fix:** Coerce to UTC at function entry: `if ts.tzinfo is None: ts = ts.replace(tzinfo=UTC)`.

---

## SUMMARY
P1=6 P2=7 P3=1
