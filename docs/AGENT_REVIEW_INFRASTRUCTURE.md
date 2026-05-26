# Agent Review — Infrastructure (2026-05-26)

Reviewer: caveman:cavecrew-reviewer
Tools: Read, Grep, Bash (no Write — summary returned inline; main thread saved to file)

## Findings

**P0**

dashboard/auth.py:175: P0: Bearer token path returns without `_refresh_cookie()`. Cookie expires after 1 year regardless of activity. Call `_refresh_cookie(make_response(f(*args, **kwargs)), expected)` before return to match cookie / query-param paths.

portfolio/health.py:165: P0: `datetime.fromisoformat(hb)` returns naive datetime. `datetime.now(UTC) - last` (aware − naive) crashes with `TypeError`. Fix: `fromisoformat(hb).replace(tzinfo=UTC)` or assume UTC.

**P1**

portfolio/http_retry.py:55: P1: Jitter added AFTER `retry_after` cap. `wait += random.uniform(0, wait)` doubles wait on Telegram 429s — "retry_after=10" becomes 10–20 s instead of 10 s. Bound jitter (`min(wait, 60)`) or add before cap.

portfolio/shared_state.py:88-89: P1: `_loading_keys.add(key)` and `_loading_timestamps[key]` set OUTSIDE try block (line 91). BaseException between 88–90 leaves orphaned key for 120 s. Wrap in try/finally.

**P2**

portfolio/health.py:40: P2: `error_count` unbounded — incremented per error cycle, can overflow 2 B over 1 year. Combined with separate `errors[-19:]` list, field can drift. Cap at 10 000.

portfolio/log_rotation.py:455: P2: Temp file with `filepath.with_suffix(".tmp")` does NOT resolve symlinks like `atomic_write_json` (uses `_resolve_write_path`). If target is a symlink, tmp lands on wrong volume, `os.replace` fails. Use `tempfile.mkstemp(dir=str(path.parent))`.

portfolio/claude_gate.py:343: P2: `_count_today_invocations()` calls `load_jsonl(INVOCATIONS_LOG)` — full-file scan on every Layer 2 invocation (lines 487, 554). File grows unbounded (no rotation policy). ~5400 calls/month = 200 ms by month 3. Use tail read + count.

## Top 3

1. **dashboard/auth.py:175** — Bearer auth path skips cookie refresh. Real-world impact: CLI scripts hold a working token, never refresh, then silently lose access after a year. Trivial fix.
2. **portfolio/health.py:165** — Naive datetime subtraction is a guaranteed `/api/health` crash the moment any heartbeat lands without `+00:00` suffix (pre-Python-3.11 ISO write paths). Single-line fix; immediate ship.
3. **portfolio/http_retry.py:55** — Telegram's documented `retry_after` is honored at 2× by the jitter — under sustained rate-limit the system hits 30–60 s gaps when the API only asked for 15. Slows the loop visibly and pointlessly.

## totals: 7 findings (2 P0, 2 P1, 3 P2)
