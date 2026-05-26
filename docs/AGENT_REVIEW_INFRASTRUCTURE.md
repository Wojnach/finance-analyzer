# Adversarial Review: infrastructure (Agent Findings)

Reviewer: caveman-ultra
Date: 2026-05-26
Style: caveman one-liner

---

## Critical Findings

dashboard/auth.py:175: Bearer token auth returns without calling `_refresh_cookie()`. Cookie expires after 1 year regardless of active use. Call `_refresh_cookie(make_response(f(*args, **kwargs)), expected)` before return to match cookie/query paths.

portfolio/health.py:165: `datetime.fromisoformat(hb)` returns naive datetime (no tzinfo). Line 165 does `datetime.now(UTC) - last` (aware - naive) → TypeError crashes dashboard /api/health. Fix: `fromisoformat(hb).replace(tzinfo=UTC)` or assume UTC.

---

## P1 Findings (Race, Resource Leak, Loop Throughput)

portfolio/http_retry.py:55: Jitter added AFTER `retry_after` cap. Line 55 does `wait += random.uniform(0, wait)`, doubling wait on Telegram 429s. Telegram says "retry_after=10" → wait becomes 10-20s instead of 10s. Bound jitter or add before cap: `wait = min(60, backoff) + jitter`.

portfolio/shared_state.py:88–89: `_loading_keys.add(key)` at line 88, `_loading_timestamps[key] = time.time()` at line 89, both OUTSIDE the try block (line 91). BaseException between 88–90 leaves orphaned key for 120s (no eviction). Wrap in try/finally or move try earlier.

---

## P2 Findings (Edge Case, Perf)

portfolio/health.py:40: `error_count` unbounded, no saturation cap. Incremented every cycle with error. Over 1 year can overflow 2B. Combined with separate `errors[-19:]` list, field can drift. Cap at 10,000 or use `deque(maxlen=100)`.

portfolio/log_rotation.py:455: Temp file created with `filepath.with_suffix(".tmp")` does NOT resolve symlinks like `atomic_write_json` does (line 59 `_resolve_write_path`). If config.json symlink, tmp lands on wrong volume → os.replace fails. Use `tempfile.mkstemp(dir=str(path.parent))` like file_utils.

portfolio/claude_gate.py:343: `_count_today_invocations()` calls `load_jsonl(INVOCATIONS_LOG)` — full-file scan. Called on lines 487, 554. File grows unbounded (no rotation policy). ~5400 invocations/month → 200ms scan by month 3. Use tail read + binary search or append-only count file.

---

## Prior Review Status (2026-05-24)

✅ [RESOLVED] portfolio/journal.py:580 — write_context() uses atomic_write_text
❓ [SAFE] portfolio/gpu_gate.py:217–219 — fd leak claim: except block closes fd on error, actually safe
⏳ [REPEAT] portfolio/claude_gate.py:343 — See P2 findings above

---

## Summary

2 P0 findings (security + crash), 2 P1 findings (race + jitter), 3 P2 findings (saturation + perf + symlink).
All tier-1 infrastructure files audited. atomic_write_json family is correct (fsync + replace). Rotation and logging are safe. Main issues: bearer auth missing refresh, health datetime crash, jitter math, thread-unsafe cache bake, unbounded counters, symlink mismatch, full-file scan on invocation count.

