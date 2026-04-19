# Agent Review: orchestration

## P2 Findings
1. **Stale config in _run_post_cycle** — config loaded once at startup, never reloaded (main.py:1081,1128)
2. **Agent subprocess inherits stdin** — risk of interactive blocking (agent_invocation.py:451). Fix: stdin=subprocess.DEVNULL
3. **classify_tier/update_tier_state race** — 3 separate reads of trigger_state.json per cycle (main.py:803,806)
4. **Signal-flip trigger fires on HOLD** — spurious triggers from stale prev_triggered (trigger.py:226-236)
5. **wait_for_specialists zombie leak** — second TimeoutExpired uncaught (multi_agent_layer2.py:207-208)
6. **_crash_sleep timing** — initial run crash adds extra full interval to recovery

## P3 Findings
1. classify_tier/update_tier_state redundant disk reads
2. _run_post_cycle inconsistent error tracking for signal_postmortem
3. _FADE_FLIP_RE tightly coupled to reason format
4. _extract_triggered_tickers misses sentiment reasons

## P4 Findings
1. logging_config non-portfolio loggers silently dropped
2. process_lock metadata format fragile
