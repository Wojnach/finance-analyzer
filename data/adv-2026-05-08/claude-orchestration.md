# Adversarial Review: orchestration subsystem (2026-05-08)

[P0] portfolio/agent_invocation.py:1142
**Timeout-enforcement dead code when `_agent_timeout == 0`.**
Problem: Truthiness check on `_agent_timeout` is falsy at 0, so the timeout branch is skipped and the subprocess runs forever. Silent no-op on timeout.
Fix: Use explicit `if _agent_timeout > 0 and elapsed > _agent_timeout`.

[P0] portfolio/agent_invocation.py:839
**Byte-offset capture for auth-error scan is racy.**
Problem: Offset captured before file open creates a window where early subprocess output is missed by the silent-auth-failure detector — exactly the failure mode of the Mar–Apr 2026 outage.
Fix: Capture offset *after* opening the log file handle.

[P0] portfolio/trigger.py:231
**Consensus baseline consumed even when ranging dampening suppresses trigger.**
Problem: Baseline updated regardless of whether Layer 2 was invoked. Next valid crossing is missed because baseline already shifted.
Fix: Only update baseline when trigger fires; keep stale baseline through dampened cycles.

[P1] portfolio/agent_invocation.py:580
**Module-global `_agent_timeout` stale during reentrancy.**
Problem: New trigger overwrites timeout of an old running agent. T1 trigger arriving while T3 agent runs replaces 900s with 120s, and the still-running T3 is killed prematurely.
Fix: Use `{pid: timeout}` dict keyed by spawned process.

[P1] portfolio/agent_invocation.py:1323
**`_agent_log_start_offset` not cleared on completion.**
Problem: Inconsistent cleanup; stale offset persists into next invocation, distorting the auth-failure scan window.
Fix: Clear offset alongside other globals in the completion block.

[P1] portfolio/agent_invocation.py:544
**Stack-overflow auto-disable is sticky.**
Problem: Counter reaching 5 disables agent spawn permanently — Layer 2 never recovers without manual reset. No decay path.
Fix: Add 24h decay on the failure counter.

[P1] portfolio/trigger.py:189
**Startup grace period flag ignored on in-process loop restart.**
Problem: Module-level grace flag not reset on restart-without-reimport; spurious triggers fire immediately after a soft restart.
Fix: Tie grace to current process PID.

[P1] portfolio/agent_invocation.py:293
**Decision-feedback loop does O(N) full-journal scans per invocation.**
Problem: Journal grows unbounded over weeks (10K+ entries); each invocation linearly scans the whole file. Cumulative drag.
Fix: Cap to last 100 entries, or use journal_index for tail reads.

[P1] portfolio/main.py:658
**`ThreadPoolExecutor.cancel_futures` is best-effort only.**
Problem: Running threads not killed; resources leak (open files, sockets, VRAM if Chronos in-flight).
Fix: Add 5s grace period + log orphans; consider hard-kill for known-runaway tasks.

[P1] portfolio/agent_invocation.py:1148
**Timeout-kill failure not persisted.**
Problem: `kill_ok=False` lost; hung agent indistinguishable from killed agent in `invocations.jsonl`.
Fix: Include `kill_status` field in journal record.

[P1] portfolio/trigger.py:434
**`classify_tier()` uses UTC weekday for market-hours decisions.**
Problem: Wrong day at UTC/CET boundary (e.g., 23:00 UTC Thursday = 00:00 CET Friday). Tier misclassified at the day rollover.
Fix: Convert to CET before computing weekday.

[P1] portfolio/health.py:140
**Heartbeat keepalive daemon killed abruptly on loop exit.**
Problem: 2s grace insufficient for daemon mid-write to `health_state.json`. Corruption risk on graceful shutdown.
Fix: Drain daemon (signal + join with longer timeout) before main thread exits.

[P1] portfolio/agent_invocation.py:809
**Broad `except Exception:` masks `ImportError`.**
Problem: All exceptions swallowed; import failure silently falls through to bat-file invocation, hiding broken Python paths.
Fix: Catch specifically `FileNotFoundError`/`OSError`; re-raise `ImportError` so it surfaces.

## Summary

3 P0 + 10 P1. Themes: silent timeout failures, baseline poisoning under dampening,
stale module-globals across reentrant invocations, UTC vs CET drift, broad except
masking real errors. Several echo prior outage patterns (Mar–Apr silent auth) — the
detector is still racy.
