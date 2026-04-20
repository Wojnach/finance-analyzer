# Agent Review: orchestration (2026-04-20)

## P1 Critical
1. **agent_invocation.py STILL bypasses claude_gate** — Direct Popen (line 451). No kill switch, rate limiter, concurrency lock, or tree-kill. bigbet.py and iskbets.py also bypass.
2. **multi_agent_layer2.py blocks main loop** — 30s synchronous wait during multi-agent mode. No heartbeat, no health update during wait.
3. **trigger_state.json TOCTOU race** — Load/mutate/save without file lock. Telegram poller thread could race.

## P2 High
1. No process group isolation for Layer 2 agent (no CREATE_NEW_PROCESS_GROUP)
2. Specialist process log_fh leaked on launch failure
3. bigbet.py/iskbets.py bypass claude_gate (6 concurrent Claude procs possible)
4. _agent_proc global state not thread-safe (bare module globals, no lock)

## P3 Medium
1. Crash recovery backoff can be bypassed by Task Scheduler restart (mitigated by persisted counter)
2. Singleton lock uses byte-range locking (fragile on network filesystems)
3. _startup_grace_active never re-armed (benign in production)
4. autonomous.py throttle file has no locking (single-threaded in practice)
5. loop_contract.py has two independent read-modify-write cycles on same file

## Prior Finding Status
- Agent invocation bypasses claude_gate: **UNRESOLVED**
- Trigger state TOCTOU: **UNRESOLVED**
- Digest concurrent access: **PARTIALLY RESOLVED**
- Subprocess governance (zombies): **PARTIALLY RESOLVED** (claude_gate callers fixed, primary path not)
