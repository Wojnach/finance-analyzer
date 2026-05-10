# Adversarial Review: infrastructure subsystem (2026-05-08)

[P0] portfolio/http_retry.py:36
**Retries on all 4xx errors instead of only retryable 5xx/429.**
Problem: Auth failures (401/403) and bad-request (400) get retried, masking real
client-side bugs and bypassing the retry gate's intent.
Fix: Restrict retry policy to 5xx + 429 (and 408); never retry other 4xx.

[P0] dashboard/app.py:1039
**`POST /api/validate-portfolio` lacks CSRF token validation.**
Problem: Only the auth cookie is checked. Cross-origin form post can mutate state
without user consent. Same surface used for any other POST endpoint.
Fix: Require an explicit anti-CSRF header (SameSite=lax cookie is not enough), or a
short-lived token tied to the session cookie.

[P1] portfolio/shared_state.py:276
**Rate limiter thundering herd at `wait_time=0`.**
Problem: When wait time computes to zero, multiple threads see the same prior-call
timestamp and all skip the sleep — limit defeated.
Fix: Hold a lock around the read+update, or use a token-bucket primitive.

[P1] portfolio/file_utils.py:173-180
**`load_jsonl_tail` boundary detection assumes `\n` is always inside ASCII.**
Problem: Multibyte UTF-8 codepoints can include bytes that look like `\n`; boundary
search splits a codepoint, producing decode errors that look like data corruption.
Fix: `errors="replace"` or seek backward to a valid codepoint boundary before decode.

[P1] portfolio/gpu_gate.py:130-142
**Stale-lock break races with concurrent acquirer.**
Problem: `exists()` then `release()` is a TOCTOU; another process acquires after the
check but before the release; this process then releases the new live lock, allowing
two-process VRAM contention.
Fix: Use a real file lock (`msvcrt.locking` / `fcntl`) plus PID check inside the lock.

[P1] portfolio/process_lock.py:62-73
**`msvcrt.locking` unlock without `seek(0)` on the error path.**
Problem: Unlock targets the wrong byte range; lock is held permanently from Windows'
view, blocking subsequent process starts until reboot.
Fix: Always seek(0) before lock/unlock; wrap in try/finally that restores fp position.

[P1] portfolio/telegram_notifications.py:52, portfolio/message_store.py:119
**Token / chat_id leak via Markdown parse-error logs.**
Problem: When Telegram returns a parse error, the full request payload (including
chat_id and sometimes a token snippet) ends up in log output.
Fix: Redact tokens before logging; truncate chat_id; never include the bot token.

[P1] portfolio/subprocess_utils.py:140-143
**Timeout uses `proc.kill()` without Job Object guarantee on Windows.**
Problem: On Windows `proc.kill()` does not propagate to child processes; a claude CLI
sub-subprocess can keep running after the parent kill. This was the silent-auth outage
class.
Fix: Use a Windows Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`; assign the
process tree to it on spawn.

[P1] portfolio/log_rotation.py:326-327
**`os.replace()` without `fsync()`.**
Problem: Write-back cache loss on crash leaves a half-written rotated log with
truncated tail; loss of recent error context exactly when needed.
Fix: `flush() + os.fsync(fileno())` before replace; consider directory fsync on POSIX.

[P1] portfolio/prophecy.py:70-73
**Concurrent `save_beliefs()` overwrites without merge.**
Problem: Layer 1 + Layer 2 writing prophecies in the same cycle clobber each other.
Fix: Read-modify-write under a per-file lock, or version optimistic concurrency.

[P2] portfolio/shared_state.py:94-123
**Cache exception handler mutates `_tool_cache` outside the lock.**
Problem: Lock released before cleanup; concurrent reader sees partially-mutated state.
Fix: Hold the lock through the cleanup branch.

[P2] dashboard/app.py:155-168
**Adaptive `tail_bytes` growth unbounded on corrupted JSONL.**
Problem: Bad line keeps doubling read size; up to 64MB allocation before fallback.
Fix: Cap growth at a sane max (8MB); on cap miss, fail-safe to "no records".

[P2] portfolio/alert_budget.py:42-44
**After-empty deque returns stale `last_call`.**
Problem: First message after deque pruned to empty passes; next caller sees stale
`last_call`, computes a huge wait. Alerts can be silenced for hours.
Fix: Reset `last_call` when deque empties.

[P2] portfolio/subprocess_utils.py:289-299
**PowerShell JSON parsing trusts stdout is valid JSON.**
Problem: A non-JSON PS error string raises in `json.loads`; caller gets crash instead
of context.
Fix: `try: json.loads ... except json.JSONDecodeError: log + return None`.

[P2] portfolio/journal.py:28-40
**TOCTOU between log_rotation.move and reader buffered read.**
Problem: Reader skipping lines (or hitting EOF early) when rotation runs mid-iteration.
Fix: Open with `O_RDONLY`, accept partial reads; or coordinate via shared rotation
lock.

[P3] dashboard/app.py:175-181
**Integer parsing silently truncates floats (`200.5 -> 200`).**
Problem: Bounds check is the only guard; sloppy fractional input slides through.
Fix: Validate digits-only before `int()`, return 400 otherwise.

[P3] portfolio/file_utils.py:56
**`ensure_ascii=True` bloats every JSON file with `\uXXXX` escapes.**
Problem: Bigger files, harder diffs.
Fix: `ensure_ascii=False` with explicit `encoding="utf-8"`.

[P3] portfolio/shared_state.py:68-75
**Eviction log warns "stuck key" on normal 2-min batch flushes.**
Problem: Alert fatigue.
Fix: Raise threshold to 5 min, or whitelist known batch keys.

[P3] dashboard/app.py:226-242
**GoldDigger payload duplicates fields via shallow copy.**
Problem: API surface confusion.
Fix: Pick one canonical field name; remove the duplicate.

[P3] portfolio/message_store.py:37-49
**Mojibake substitution incomplete; rare double-encoded cases survive.**
Fix: Use `ftfy` if available; document unsupported corners.

[P3] portfolio/gpu_gate.py:33
**Hardcoded `Q:/models` Windows-only path.**
Problem: Non-Windows installs never lock; concurrent VRAM loads possible.
Fix: Read path from config or compute relative to repo.

[P3] portfolio/prophecy.py:323
**Division-by-zero guard misses missing-field condition.**
Problem: Missing field falls through; consumer sees None where a number is expected.
Fix: `if field in data and data[field] != 0:`.

## Summary

2 P0 + 8 P1 + 5 P2 + 7 P3 = 22 findings. Themes: CSRF gap on dashboard POSTs, retry
policy retries 4xx, secrets in error logs, Windows process-tree kill not enforced,
concurrent writers without locks (prophecy, log rotation, GPU gate), rate-limiter
thundering herd at wait=0. Boundary handling weak in JSONL tail and adaptive
tail-bytes growth.
