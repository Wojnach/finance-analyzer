# Agent Adversarial Review: infrastructure

**Agent**: feature-dev:code-reviewer
**Subsystem**: infrastructure (5,721 lines, 26 files)
**Duration**: ~352 seconds
**Findings**: 14 (3 P0, 6 P1, 5 P2)

---

## P0 Findings

### A-IN-1: Log Rotation Non-Atomic Archive Write — Data Loss on Crash [P0]
- **File**: `portfolio/log_rotation.py:222,235-242`
- **Description**: `gzip.open(gz_path, "wt")` writes archive directly (no tempfile). If process crashes mid-write, archive is corrupt AND entries have been removed from the working file. Permanent data loss of log entries.
- **Fix**: Use tempfile for gz write, rename into place. Mirror `file_utils.atomic_write_json` try/except pattern.

### A-IN-2: claude_gate.invoke_claude Leaves Zombie on TimeoutExpired [P0]
- **File**: `portfolio/claude_gate.py:207-224`
- **Description**: `subprocess.run(timeout=)` raises `TimeoutExpired` but does NOT kill the child process. The zombie `claude -p` continues running (up to 900s for T3). Combined with no concurrency lock, two Claude processes can overlap. Caused the 34h outage Feb 18-19.
- **Fix**: Use `Popen` + `proc.communicate(timeout=)` + `proc.kill()` in exception handler.

### A-IN-3: No Concurrency Lock in claude_gate — Simultaneous Invocations [P0]
- **File**: `portfolio/claude_gate.py:112-243`
- **Description**: No mutex, no in-flight tracking. Multiple callers can spawn concurrent `claude -p` processes. Rate-limit check is warning-only, not a gate.
- **Fix**: Add `threading.Lock` for in-process + file-based lock for cross-process protection.

---

## P1 Findings

### A-IN-4: health.check_staleness Crashes on Naive Timestamps [P1]
- **File**: `portfolio/health.py:72-74`
- **Description**: `datetime.fromisoformat(hb)` may return naive datetime. Subtraction from aware `datetime.now(UTC)` raises TypeError. Dashboard `/api/health` returns 500.
- **Fix**: Normalize: `if last.tzinfo is None: last = last.replace(tzinfo=UTC)`.

### A-IN-5: _count_today_invocations O(N) Full JSONL Scan [P1]
- **File**: `portfolio/claude_gate.py:97-105`
- **Description**: Reads entire invocations log on every Claude call. File grows unboundedly (no rotation policy).
- **Fix**: Use `load_jsonl_tail(max_entries=200)`. Add to ROTATION_POLICIES.

### A-IN-6: rotate_jsonl Reads Entire 68MB File Into RAM [P1]
- **File**: `portfolio/log_rotation.py:156-179`
- **Description**: Loads complete signal_log.jsonl into two Python lists. Under GPU memory pressure (Ministral + Chronos loaded), could cause pagefile thrash or OOM.

### A-IN-7: message_throttle._send_now TOCTOU Race [P1]
- **File**: `portfolio/message_throttle.py:92-113`
- **Description**: No lock between should_send_analysis check and _send_now call. Concurrent threads bypass cooldown, sending duplicate Telegram messages.
- **Fix**: Add module-level threading.Lock.

### A-IN-8: telegram_poller Writes config.json Without Validation [P1]
- **File**: `portfolio/telegram_poller.py:150-165`
- **Description**: Raw `json.load()` (violates project rule), read-modify-write race on config.json. Concurrent writes silently discard other processes' changes.
- **Fix**: Use `load_json()`. Add config write lock.

### A-IN-9: shared_state._cached Suppresses KeyboardInterrupt [P1]
- **File**: `portfolio/shared_state.py:98-101`
- **Description**: `except KeyboardInterrupt: return None` — suppresses Ctrl+C. Also `_loading_timestamps` not cleaned up on interrupt.
- **Fix**: Re-raise after cleanup. Add `_loading_timestamps.pop(key, None)`.

---

## P2 Findings

### A-IN-10: gpu_gate._write_lock Is Dead Code [P2]
- **File**: `portfolio/gpu_gate.py:83-87`
- **Description**: Non-atomic `write_text()` lock write — would break O_EXCL guarantee if used. Never called.
- **Fix**: Remove dead function.

### A-IN-11: crypto_scheduler Uses Local-Timezone Timestamp [P2]
- **File**: `portfolio/crypto_scheduler.py:310`
- **Description**: `datetime.now().astimezone().isoformat()` produces CET offset, not UTC.
- **Fix**: `datetime.now(UTC).isoformat()`.

### A-IN-12: journal.write_context Non-Atomic Write [P2]
- **File**: `portfolio/journal.py:568,580`
- **Description**: `CONTEXT_FILE.write_text(md)` is non-atomic. Crash mid-write leaves truncated context.
- **Fix**: Tempfile + os.replace pattern.

### A-IN-13: backup.py No Error Handling for Failed Copies [P2]
- **File**: `portfolio/backup.py:36`
- **Description**: Single `shutil.copy2` failure aborts entire backup run. No try/except per file.

### A-IN-14: rotate_text Truncates Live Log While Writer Active [P2]
- **File**: `portfolio/log_rotation.py:316-318`
- **Description**: Truncates file in-place with `open("w")` while bat redirect has handle open. May cause PermissionError or data loss.
