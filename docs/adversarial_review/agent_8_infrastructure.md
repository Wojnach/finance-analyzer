# Agent 8 — Infrastructure Adversarial Review

**Subsystem:** infrastructure  
**Files reviewed:** 32 (file_utils, http_retry, api_utils, subprocess_utils, health, log_rotation, journal, journal_index, reporting, telegram_notifications, telegram_poller, message_store, message_throttle, alert_budget, notification_text, digest, daily_digest, weekly_digest, config_validator, tickers, instrument_profile, stats, others)  
**Findings:** 27 total — **P0: 4 | P1: 6 | P2: 10 | P3: 7**  
**Agent:** caveman:cavecrew-reviewer

---

## P0 (Data Loss / Silent Failure)

- `portfolio/journal.py:28`: **race condition**. `load_recent()` reads JOURNAL_FILE without acquiring `jsonl_sidecar_lock()`. Concurrent `atomic_append_jsonl()` writes can interleave mid-read, causing torn lines or incomplete entries parsed as valid JSON (KeyError silenced). Missing "ts" field crashes downstream code silently.
- `portfolio/health.py:29`: **DST vulnerability**. `uptime_seconds` uses `time.time() - state["start_time"]`. If system clock jumps (DST, NTP correction, manual set), uptime becomes negative or false. On restart, `start_time` is stale from prior session, inflating uptime indefinitely. No monotonic clock guard.
- `portfolio/api_utils.py:28-31`: **config cache deadlock**. If `config_path.stat().st_mtime` raises OSError (antivirus lock, permission denied), `_config_cache` is never cleared. Cache stays stale forever; even after file becomes readable, the check `mtime != _config_mtime` fails silently and stale secrets/credentials are used for all subsequent API calls.
- `portfolio/telegram_poller.py:290`: **audit trail loss**. `_log_inbound()` catches all exceptions at WARNING and returns silently. If `atomic_append_jsonl()` raises (disk full, permission denied), the user's command is processed (state mutated, trade executed) but never logged. Audit contract broken; trade cannot be reconciled if JSONL write fails.

## P1 (Real Bugs)

- `portfolio/health.py:49`: uptime accumulation. `load_health()` returns `{"start_time": time.time(), ...}` with fresh wall-clock time. On subsequent calls from the same process, the freshly-computed start_time is minutes/hours old (from cold boot), so uptime_seconds inflates. `reset_session_start()` exists but is not called at every loop startup — only on explicit invocation.
- `portfolio/http_retry.py:51-57`: retry-after header ignored for transient errors. 429 (rate-limit) honors retry_after header, but 502, 503, 504 (transient server errors) ignore it and retry on exponential backoff.
- `portfolio/log_rotation.py:270-364`: JSONL rotation holds lock for entire read→write→replace sequence. Under high signal_log write rate (1000s lines/60s), the lock blocks all appends for hundreds of ms.
- `portfolio/config_validator.py:46-57`: silent OPTIONAL_KEYS validation. Missing optional keys log WARNING only, not ERROR. If Telegram token is absent from config, validation passes and main loop crashes later during send_telegram().
- `portfolio/subprocess_utils.py:290`: shell injection in `kill_orphaned_llama()`. Uses `shell=True` with ps_cmd containing backtick-escaped user input. Must use subprocess array form, not shell=True.
- `portfolio/log_rotation.py:363`: fsync after replace. Rotation writes tmp, then `os.replace()`, then `os.fsync(f.fileno())` on still-open handle. fsync after replace does not guarantee directory entry durability. Call fsync BEFORE replace.

## P2 (Latent)

- `portfolio/file_utils.py:74-95`: corrupt JSON crash-prone. Callers that do `cfg = load_json(...)` and access keys directly will crash with TypeError if file corrupt.
- `portfolio/message_throttle.py:64`: clock-dependent cooldown. `state["last_analysis_sent"] = time.time()` uses wall-clock without timezone. NTP backward adjustment leaves cooldown stuck.
- `portfolio/health.py:161-166`: future timestamp false-negative. `check_staleness()` — if heartbeat is from future time, age becomes negative, never stale.
- `portfolio/alert_budget.py:38-49`: emergency budget spam — empty PRIORITY_EMERGENCY messages consume budget without utility.
- `portfolio/daily_digest.py:43-47`: corrupt digest state loss — re-init loses all prior digest history.
- `portfolio/api_utils.py:28-31`: mtime cache race on sub-second writes — two writes within same second have identical mtime.
- `portfolio/telegram_poller.py:220-221`: 1-hour post-restart bypass window. If user traded manually in meantime, stale command mutates wrong positions.
- `portfolio/file_utils.py:290-292`: fsync on open handle unreliable on network shares / USB drives.
- `portfolio/message_store.py:138`: token exposure in logs. Telegram token embedded in f-string URL.
- `portfolio/journal.py:28-40`: performance cliff on large files — `load_recent()` iterates entire file.

## P3 (Minor)

- `portfolio/config_validator.py:49`: validation loop type error — non-dict at non-leaf position causes TypeError instead of clear message.
- `portfolio/telegram_poller.py:307-308`: multiline command injection — newline-embedded text silently ignored.
- `portfolio/digest.py:143`: hardcoded stale FX rate (10.5, current ~10.7).
- `portfolio/health.py:294-296`: signal_health list race — concurrent batch updates can lose entries.
- `portfolio/file_utils.py:24-29`: broken symlink validation — `os.replace()` silently fails on dangling link.
- `portfolio/log_rotation.py:277`: BOM not stripped on JSONL rotation.
- `portfolio/message_throttle.py:61-66`: duplicate queue not detected — same message queued repeatedly replaces itself.
