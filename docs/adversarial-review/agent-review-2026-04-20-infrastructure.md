# Agent Review: infrastructure (2026-04-20)

## P1 Critical
1. **Dashboard token timing attack (STILL UNRESOLVED)** — `==` comparison + CORS `*`. Brute-forceable from LAN.
2. **Telegram poller can wipe ALL API keys** — Corrupt read → cfg={} → atomic_write_json overwrites real config.
3. **journal.py write_context uses non-atomic write_text** — Crash mid-write → empty context → Layer 2 operates without decision memory.

## P2 High
1. shared_state._loading_timestamps leaks on success path (grows unbounded, scanned inside hot lock)
2. log_rotation.py lacks fsync before os.replace (power loss → truncated signal_log)
3. weekly_digest reads entire 68MB+ signal_log.jsonl into memory (OOM risk)
4. Dashboard /api/signal-log reads 68MB for 50 entries (blocks Flask thread)
5. CORS allows any origin with wildcard (compounds timing attack risk)

## P3 Medium
1. journal.py load_recent reads entire journal file
2. file_utils.py atomic_append_jsonl doesn't handle disk-full gracefully (torn line)
3. Dashboard api_mstr_loop reads full JSONL for last entry
4. http_retry returns None on all failures (no distinction for callers)
5. _RateLimiter sleeps while holding lock (serializes all waiting threads)

## Prior Finding Status
- Dashboard timing attack: **UNRESOLVED**
- Telegram poller config wipe: **UNRESOLVED**
- journal.py non-atomic write: **UNRESOLVED**
- _loading_timestamps leak: **PARTIAL** (_update_cache pops, _cached doesn't)
- 4h digest OOM: **FIXED** (uses load_jsonl_tail)
- Dashboard re-reading files: **FIXED** (5s cache)
