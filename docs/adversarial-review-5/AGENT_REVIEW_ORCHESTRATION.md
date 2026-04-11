# Agent Review: orchestration — Round 5 (2026-04-11)

**Agent**: feature-dev:code-reviewer
**Files reviewed**: 13 (main.py, agent_invocation.py, trigger.py, market_timing.py,
autonomous.py, multi_agent_layer2.py, loop_contract.py, process_lock.py, health.py,
claude_gate.py, gpu_gate.py, crypto_scheduler.py, session_calendar.py)
**Duration**: ~249s

---

## Findings (8 total: 3 P0, 5 P1)

### P0

**OR-R5-1** agent_invocation.py:302 — Layer 2 uses bare Popen, bypasses claude_gate
- No CREATE_NEW_PROCESS_GROUP, no tree-kill, no _invoke_lock
- claude_gate.py explicitly states "Direct Popen calls are FORBIDDEN"
- A-IN-2 fix never migrated to the PRIMARY Layer 2 launcher
- On each T3 timeout, zombie Node.js processes accumulate
- Fix: Route through claude_gate.invoke_claude() or apply same Popen kwargs

**OR-R5-2** main.py:949-950 — fromisoformat() crash on Python ≤3.10 with UTC offset
- Heartbeat uses datetime.now(UTC).isoformat() → "+00:00" suffix
- fromisoformat on Python ≤3.10 doesn't support this → ValueError
- Caught by bare except → crash-detection alerting silently broken
- Fix: Normalize timestamp or use compatible parsing

**OR-R5-3** (STILL OPEN — C6, 3 rounds) main.py — check_drawdown() never called
- Confirmed yet again: zero production callers, function is fully tested
- Financial safety control completely inert

### P1

**OR-R5-4** agent_invocation.py:241-270 — Shared deadline starves later specialists
- wait_for_specialists uses single 30s deadline for all specialists
- Slow first specialist reduces all subsequent budgets
- Fix: Independent timeout per specialist

**OR-R5-5** claude_gate.py:334 + loop_contract.py:640 — Self-heal blocks main loop 180s
- Self-heal acquires _invoke_lock for up to 180s during T3 agent run
- Main loop stalls, next cycle at 780s instead of 600s
- Fix: Run self-heal in background thread

**OR-R5-6** main.py:740,743 — classify_tier and update_tier_state do double state load
- M10/NEW-4 added state= parameter but caller never passes it
- Double disk I/O per trigger, stale state risk
- Fix: Load once, pass to both calls

**OR-R5-7** multi_agent_layer2.py:153-168 — File handle leak on specialist launch failure
- log_fh opened before Popen, not closed if Popen raises
- Fix: try/finally on Popen to close log_fh

**OR-R5-8** crypto_scheduler.py:310 — Local timezone instead of UTC
- datetime.now().astimezone() emits CET/CEST, not UTC
- Inconsistent with all other timestamps in the system
- Fix: Use datetime.now(UTC).isoformat()

---

## Regression Check
- A-IN-2 (tree kill): FIXED in claude_gate — but NOT in agent_invocation (OR-R5-1)
- A-IN-3 (invoke lock): FIXED in claude_gate — not applicable to agent_invocation
- C4 (first-of-day T3): FIXED
- C3 (specialists 150→30s): PARTIAL (still synchronous)
- C6 (check_drawdown): STILL OPEN
