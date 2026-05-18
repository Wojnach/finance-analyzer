# Orchestration Review — subagent result (caveman:cavecrew-reviewer)

Totals: 3 P1 (🔴), 17 P2 (🟡)

## P1 / 🔴 Critical

1. **agent_invocation.py:1059-1062** — `_journal_count_before` and `_telegram_count_before` assigned as locals in `invoke_agent()`, referenced in `_check_agent_completion_locked()` without `global` declaration. Will UnboundLocalError on first completion check or use stale module-scope values. Silent journal_written/telegram_sent misdetection. **NEEDS VERIFICATION** — check actual code.

2. **agent_invocation.py:1024-1028** — `_agent_log_start_offset` set inside try block; if `agent_log_path.stat()/.exists()` raises, offset stays stale from prior invocation; watchdog reads wrong file region.

3. **agent_invocation.py:677-684** — Stack overflow counter persistence: loaded at module import, persisted globally, but never reloaded between invocations. Process exit without explicit save loses history.

4. **main.py:1273-1276** — Singleton lock acquired before atexit.register; loop() raise before atexit registration leaves lock held; subsequent restart hits exit code 11.

5. **trigger.py:389-400** — Flip reason division by zero if prev_price=0 (corrupted baseline) → crash.

## P2 / 🟡

- main.py:611-661 — pool.shutdown(wait=False) leaks daemon threads blocked in network I/O across cycles.
- main.py:615-661 — TimeoutError path cancels futures but in-progress network I/O threads survive; cross-cycle thread leakage.
- main.py:607 — `_TICKER_POOL_TIMEOUT` wall-clock based via as_completed(), not monotonic. NTP jump / suspend breaks timing.
- main.py:974-976 — invoke_agent inside heartbeat_keepalive context; gate-skipped invocations still trigger __exit__ on keepalive; logs misleading.
- main.py:839-843 — `triggered=True` with empty reasons_list path muddled; should be `if triggered and reasons_list:`.
- main.py:839-989 — `should_escalate_to_claude` reads portfolio JSON while Layer 2 rewrites; except swallows → dd=0.0 silent bypass.
- trigger.py:225-263 — `_startup_grace_active` cleared on first call; if signal fetches all timeout, never activates → re-fires baseline triggers on restart.
- trigger.py:233-241 — PID-based grace; rapid restarts with overlapping PIDs duplicate triggers.
- trigger.py:172-190 — `_save_state()` mid-write failure on Windows truncates triggered_consensus, wipes baseline.
- trigger.py:174-190 — atomic_write_json failure under concurrent read (warrant loop / dashboard) → not persisted → stale triggers next cycle.
- trigger.py:376-387 — Flip cooldown uses `time.time()` elapsed; NTP backward jump → elapsed<0 → resets cooldown, unblocks flip immediately.
- trigger.py:256-262 — Empty signals during grace → empty triggered_consensus → all signals fire as "new consensus" next cycle.
- agent_invocation.py:694-715 — Auth cooldown lookup catches Exception fail-open; corrupted invocations.jsonl → bypass cooldown silently.
- agent_invocation.py:702 — Last-50-entry scan misses recent auth_error surrounded by successes in large file.
- agent_invocation.py:856-871 — Trade-guard block-both-strategies logic ambiguous; partial blocks may proceed unintentionally.
- agent_invocation.py:950-951 — `_log_trigger()` after invoke_agent() returns False uses 'why' from escalation_router → misleading reason recorded.
- agent_invocation.py:1144-1148 — Popen exception handler leaves `_agent_start`/`_agent_timeout`/`_agent_tier` set → watchdog kills next legitimate subprocess.
- agent_invocation.py:745-769 — Race between main-thread kill+new-invoke and watchdog tick observing stale start/timeout.
- claude_gate.py:96-110 — `_parse_claude_json_stdout` brace-depth walker brittle on escaped backslashes in JSON strings (Windows paths).
- market_timing.py:258-260 — `_is_agent_window()` hour comparison without wraparound handling; DST bugs slip through.
