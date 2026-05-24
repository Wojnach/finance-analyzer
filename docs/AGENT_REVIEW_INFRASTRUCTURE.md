# Adversarial Review: infrastructure (Agent Findings)

Reviewer: caveman:cavecrew-reviewer
Date: 2026-05-24
Style: caveman one-liner

---

## Top findings

- `portfolio/dashboard/auth.py:175` — 🔴 P1 bug: Bearer token auth path skips `_refresh_cookie()`. Session expires after 1 year regardless of active use. **Fix:** Call `_refresh_cookie(make_response(f(*args, **kwargs)), expected)` before return on the bearer branch.
- `portfolio/http_retry.py:55` — 🟡 P2 risk: Jitter added after `retry_after` cap. Adds 100% overhead to Telegram 429 waits, causing excessive delays. **Fix:** Add jitter before cap or bound to ±60s.
- `portfolio/shared_state.py:88–89` — 🟡 P2 risk: Dogpile key added to `_loading_keys` before timestamp write. `BaseException` between can leave orphaned key for 120 seconds (no eviction). **Fix:** Reverse order or wrap in try/finally.
- `portfolio/health.py:37–40` — 🟡 P2 risk: `error_count` unbounded, no saturation cap. Can grow to 2B+ in long-running process. Combined with `error_count` field independent of error list, can drift. **Fix:** Cap at 10,000 or use `deque(maxlen=1000)`.

## Totals
1 P1 bug, 3 P2 risks.

## Not re-flagged but may still apply (from 2026-04-08 prior review — verify before fix)
- gpu_gate.py:126-128 GPU lock fd leak on write failure → 5-min deadlock
- journal.py:568,580 write_context() uses Path.write_text(), not atomic
- health.py:66 fromisoformat timezone mismatch crashes dashboard
- claude_gate.py:97-104 _count_today_invocations() scans full file every call

Agent did not re-confirm these in 2026-05-24 pass — main thread cross-cut should verify.
