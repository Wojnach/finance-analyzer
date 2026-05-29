# Orchestration Subsystem — Adversarial Code Review (2026-05-29-fgl)

Subsystem: Layer 1 loop lifecycle, Layer 2 Claude subprocess invocation, trigger detection, journal/contract.
Files reviewed (HEAD 745dc577): main.py, agent_invocation.py, trigger.py, trigger_buffer.py, market_timing.py,
loop_contract.py, loop_health.py, loop_processes.py, autonomous.py, multi_agent_layer2.py, claude_gate.py,
reporting.py(refs), journal.py, journal_index.py(refs), escalation_router.py, reflection.py, analyze.py, bigbet.py,
data/layer2_invoke.py, data/layer2_action.py, data/layer2_exec.py.

## Counts
- P0: 1
- P1: 4
- P2: 8
- P3: 6
- Total: 19

---

## P0 — loop crash / data corruption / silent or undetected Layer 2 failure

`portfolio/multi_agent_layer2.py:181-198 + portfolio/agent_invocation.py:1071-1110`: P0: In multi-agent mode the specialist Claude subprocesses are launched with a bare `subprocess.Popen([claude_cmd, "-p", ...])` (launch_specialists) and the synthesis agent then runs through invoke_agent's normal Popen — BOTH bypass `claude_gate.invoke_claude`, which claude_gate.py documents as "the ONLY approved way to invoke Claude" and which owns the kill switch, the per-day rate limiter, AND the in-process `_invoke_lock`. Because specialists skip `_invoke_lock`, three specialist subprocesses + the main Layer 2 + any bigbet/iskbets gate calls can all run Claude concurrently (each ~500MB+ Node), defeating the very serialization claude_gate exists to enforce; and they skip CLAUDE_ENABLED, so the master kill switch does NOT stop specialist spawns. The auth-failure scan IS done post-hoc (wait_for_specialists:235-243), but the concurrency/kill-switch bypass is a genuine reliability hole on the primary trade path. → Route specialist + synthesis spawns through claude_gate (or at minimum check `CLAUDE_ENABLED`/`check_claude_gates` AND acquire `_invoke_lock` around the parallel batch).

---

## P1 — incorrect logic / race / missing guard

`portfolio/agent_invocation.py:1101`: P1: On specialist-quorum failure invoke_agent does a bare `return` (returns None) after already calling `_log_trigger(reasons, "specialist_quorum_fail")`. Back in main.py:963-964/988-989 the caller then does `_log_trigger(reasons_list, "invoked" if result else "skipped_busy")` — None is falsy so it logs a SECOND row `skipped_busy` for the same trigger. Two invocations.jsonl rows per trigger corrupts get_completion_stats() counts and the contract's "latest L2 invocation" precondition reads `skipped_busy` (not the real `specialist_quorum_fail`). → Return an explicit `False` (or a sentinel the caller maps to a non-spammy status) and skip the duplicate `_log_trigger` in the quorum-fail path.

`portfolio/loop_health.py:215`: P1: `write_heartbeat()` hardcodes `"status": "ok"` in the payload regardless of the `ok=False` argument. The `ok` bool is written to a separate field, but any consumer keying on `status` (the documented field, see read_loop_status semantics) will NEVER see a failed cycle — a loop reporting failed cycles still advertises status=ok. Silent masking of loop-failure telemetry. → Set `"status": "ok" if ok else "error"` (or drop the redundant hardcoded field and standardize on `ok`).

`portfolio/main.py:832-851 + portfolio/trigger.py:503-516`: P1: When the trigger buffer defers (batch_window_s>0), main sets `triggered=False`/`reasons_list=[]` for this cycle, but `check_triggers` already advanced `state["last_trigger_time"]` and the `state["last"]` signal/price/F&G baselines (it returned triggered=True). The buffered reasons are re-emitted on a later flush, but the trigger baselines have moved past the signals that produced them — so any NEW trigger condition that arose in the same cycle as a buffered one is silently consumed (baseline updated, never fired) and can't re-trigger. Default batch_window_s=0 makes this dormant, but it is a correctness landmine if the feature is enabled. → Only advance trigger baselines when reasons are actually emitted, not when they are buffered; or have the buffer own the baseline-advance.

`portfolio/main.py:890-984`: P1: The autonomous-first escalation path (`autonomous_first_enabled`) calls `autonomous_decision(...)` whose journal write is wrapped in a swallow-all `try/except` (autonomous.py:83-89). If `_autonomous_decision_inner` raises before `atomic_append_jsonl(JOURNAL_FILE, ...)` (autonomous.py:151), NO journal entry is written, yet main logs status `autonomous_{why}` — which is NOT in the contract's `_LEGITIMATE_SKIP_STATUSES`. The contract then fires a `layer2_journal_activity` CRITICAL that is indistinguishable from a real silent Layer 2 failure, polluting critical_errors.jsonl (the exact 39-unresolved-entry symptom). → Write a minimal journal stub in autonomous_decision's exception handler (mirroring the timeout/incomplete stubs in agent_invocation), so a failed autonomous run still leaves a contract-satisfying marker tagged as a failure.

---

## P2 — robustness with correctness impact

`portfolio/loop_contract.py:344-387`: P2: Precondition 4b suppresses the `layer2_journal_activity` contract violation when the latest L2 invocation status is in `_KNOWN_FAILURE_STATUSES = {incomplete, auth_error}` (and newer than the trigger). This is safe ONLY because check_agent_completion writes a stub journal for incomplete runs — but it means a genuine repeating auth_error storm is silently absorbed by the contract (the contract relies entirely on the inline record_critical_error in detect_auth_failure to surface it). If the auth scan ever misses (rotated log, offset past EOF), the contract that was built specifically to catch the Mar-Apr auth outage is muted. → Keep the suppression but emit at least an INFO-severity contract note when suppressing on a failure status, so the silence is observable.

`portfolio/agent_invocation.py:1564-1571`: P2: Completion status precedence is `auth_error → failed(exit!=0) → success(journal&telegram) → incomplete`. An agent that exits 0, writes a journal, but never sends Telegram (telegram_sent=False due to prune-race or a real send failure) is classed `incomplete` and fires an `*L2 INCOMPLETE*` Telegram + a stub journal — even though the decision WAS journaled. Conversely a journal-written-but-telegram-genuinely-failed run is indistinguishable from a true silent stall. The `_detect_append` count-delta + ts fallback (640-662) mitigates the prune race but not a real Telegram outage. → Separate "decision made (journal written)" from "notification delivered" in the status taxonomy so a notification-only failure doesn't masquerade as a no-decision incomplete.

`portfolio/loop_processes.py:106`: P2: In `_iter_processes`, the `except` handler references `info.get("pid")` but `info` is assigned only inside the `try`. If `p.info` (the first statement) raises NoSuchProcess/AccessDenied, `info` is unbound and line 106 raises UnboundLocalError, escaping the except and crashing `scan()` (the `/api/loop-processes` endpoint, used for duplicate-loop detection). → Reference `p.pid` (available on the proc object) in the handler instead of `info.get("pid")`, or initialize `info=None` before the try.

`portfolio/analyze.py:746-757`: P2: The `--watch` loop's Claude call does NOT run `detect_auth_failure` on the output (unlike run_analysis at 282-300). A `claude` exit-0 "Not logged in" response is parsed as an empty/HOLD watch response with returncode 0 and no auth alert — the exact silent-auth class the system is hardened against, reintroduced on this path. → Add the same `detect_auth_failure(stdout+stderr)` guard before parsing the watch response.

`portfolio/analyze.py:282-289, 746-753`: P2: Both analyze paths call `subprocess.run(["claude", ...])` directly, bypassing `claude_gate.invoke_claude` (forbidden by claude_gate's module docstring) — they skip the in-process `_invoke_lock` and the daily rate limiter. These are manual CLI commands (`--analyze`/`--watch`) so concurrency with the loop is unlikely, but they can still race the loop's Layer 2 if run while the loop is live. → Route through `claude_gate.invoke_claude_text`.

`portfolio/agent_invocation.py:870-894`: P2: The reentrancy/timeout block holds `_completion_lock` for the read-decide-kill path, but `_kill_overrun_agent` does `_agent_proc.wait(timeout=15)` plus a `taskkill` `subprocess.run(timeout=10)` INSIDE the lock. The 30s completion watchdog also takes `_completion_lock` each tick. A wedged taskkill (logged as "taskkill hung") can hold the lock up to ~25s, during which the watchdog tick blocks — and `_ensure_completion_watchdog` also grabs the lock, so a new invocation's watchdog-arm can stall. Not fatal (daemon thread) but couples the kill latency to watchdog liveness. → Perform the blocking kill/wait outside the critical section (snapshot proc under lock, kill outside, re-acquire to clear state).

`portfolio/trigger.py:230-269`: P2: The startup-grace path keys on `os.getpid()` vs persisted `last_loop_pid`. PIDs are reused by the OS; if a restarted loop happens to get the same PID as the previous run, `saved_pid == current_pid` and the grace period is SKIPPED, firing a spurious T3 full review on first cycle after restart (the thing grace exists to prevent). Low probability but non-zero on a busy box. → Combine PID with process start time (or a monotonic boot token) for the grace key.

`portfolio/main.py:960`: P2: Escalate-to-claude is gated by `_is_agent_window()` but when escalate=True AND outside the agent window, the code falls through to `autonomous_decision` and logs `autonomous_{why}`. That's reasonable, but the drawdown/top5-split escalation reasons (a real risk event) are silently downgraded to a recommendation-only autonomous HOLD with no Telegram emphasis that an escalation was suppressed. A 5%+ drawdown jump on a weekend gets no Claude review and no distinct alert. → Emit a distinct notification when a genuine escalation is suppressed by the agent window.

---

## P3 — nits

`data/layer2_invoke.py, data/layer2_action.py, data/layer2_exec.py`: P3: Stale agent-authored example scripts (Feb 2026 timestamps, removed tickers AAPL/MU/TSM, absolute `Q:/finance-analyzer/data` paths) that write the journal/telegram JSONL with raw `open(..., "a")` instead of `file_utils.atomic_append_jsonl`. If the Layer 2 agent copies these as a template (they look like canonical examples), journal writes bypass atomic I/O. → Delete or clearly mark as non-canonical; replace raw writes with atomic_append_jsonl.

`portfolio/analyze.py:23-36`: P3: `analyze._clean_env()` pops `CLAUDECODE` and sets `PF_HEADLESS_AGENT=1` but does NOT pop `CLAUDE_CODE_ENTRYPOINT` (claude_gate._clean_env and agent_invocation both pop it). A stray entrypoint marker could re-trigger nested-session detection. → Also pop `CLAUDE_CODE_ENTRYPOINT`.

`portfolio/claude_gate.py:690`: P3: `invoke_claude_text` return type annotation is `tuple[str, bool, int]` (3-tuple) but the function returns a 4-tuple `(text, success, exit_code, status)` and the docstring documents 4. Annotation is wrong. → Fix annotation to `tuple[str, bool, int, str]`.

`portfolio/market_timing.py:244-260 + portfolio/loop_contract.py:376`: P3: `_is_agent_window` returns False on weekends/US-holidays, so Layer 2 never runs for held 24/7 crypto/metals positions over a weekend (logged `skipped_offhours`, correctly not alerted). Intentional, but a held leveraged position gets zero Layer 2 risk review for ~60h. → Consider a reduced "positions-held-only" weekend agent window for crypto/metals.

`portfolio/loop_health.py:35-38`: P3: Docstring/comment says "loops cycle every 60s" and bases the 300s stale threshold on it; the main data loop is 600s (market_timing INTERVAL_*). Main is intentionally excluded here, and the registered loops (crypto/oil/mstr/metals/golddigger) are 60s, so the threshold is fine — but the comment is stale and misleading. → Update the comment to reflect per-loop cadence.

`portfolio/agent_invocation.py:1413-1446`: P3: `_record_new_trades` slices `txns[count_before:]` assuming transactions are append-only and counts only grow. If a portfolio state file is restored/rewound (count shrinks), the slice is empty and new trades silently aren't recorded for guards. Edge case. → Guard with `if len(txns) <= count_before: continue` already exists (line 1430) — but a rewind-then-regrow within one cycle is unhandled; acceptable, note only.
