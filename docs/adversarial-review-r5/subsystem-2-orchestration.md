# Subsystem 2: Orchestration — Round 5 Findings

## CRITICAL (P1)

**OR-R5-1** — bigbet.py bypasses _invoke_lock and kill switch. `subprocess.run(["claude", ...])` directly.
`bigbet.py:169-174`. Kill switch, rate limiter, CLAUDECODE env stripping all bypassed.
Fix: Route through claude_gate.invoke_claude_text().

**OR-R5-2** — multi_agent_layer2.py launches 3 specialists bypassing _invoke_lock. `Popen(["claude", ...])` directly.
`multi_agent_layer2.py:154-168`. 4 concurrent Claude processes with no serialization.
Fix: Route through claude_gate or add multi-agent semaphore.

**OR-R5-3** — Specialist report files never cleaned up. Synthesis reads stale prior-run data.
`agent_invocation.py:248-266`. `cleanup_reports()` exists but is never called.
Fix: Call cleanup_reports() before launch_specialists().

## HIGH (P2)

**OR-R5-4** — analyze.py direct subprocess calls bypass claude_gate. 2 call sites.
**OR-R5-5** — _startup_grace_active is dead logic once PID check fires.
**OR-R5-6** — classify_tier triple disk reads — documented optimization never activated.
**OR-R5-7** — _consecutive_stack_overflows reset on auth_error (should only reset on success).
**OR-R5-8** — wait_for_specialists sequential drain starves later specialists (shared 30s budget).

## MEDIUM (P3)

**OR-R5-9** — perception_gate reads prior-cycle agent_summary_compact (stale data).
**OR-R5-10** — bigbet.py doesn't strip CLAUDECODE env var (nested session risk).
