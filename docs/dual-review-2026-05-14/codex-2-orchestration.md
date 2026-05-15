# Adversarial Review — 2 orchestration (second-reviewer / codex-substitute)

> Codex CLI quota was exhausted at start of session. This review is produced by a
> second Claude subagent with isolated context as a substitute second opinion.

## P0 — money-losing or data-corrupting (must fix)

- `portfolio/agent_invocation.py:572` — `_kill_overrun_agent` clears `_agent_proc=None` even when kill failed
  The function declares in its docstring (lines 507-511): *"If the kill command itself failed — caller must NOT spawn a replacement in that case because the old process may still be holding resources."* The caller at line 662-666 obeys this for the **current** invocation. But line 572 unconditionally runs:
  ```python
  _agent_proc = None
  return kill_ok
  ```
  Because `_agent_proc` is now None, the NEXT cycle's `invoke_agent` will pass the
  `if _agent_proc and _agent_proc.poll() is None` reentrancy gate at line 649 and spawn a SECOND `claude -p` while the FIRST is still alive. With the kill failure mode commonly being a Node.js process tree where `taskkill /T /F` returns 0 for the parent but children survive, this leaks claude grandchildren AND races two live claude processes writing to:
  - `data/agent.log` (opened in plain `"a"` append mode at line 914 — NOT atomic, bytes interleave)
  - `_agent_log_start_offset` is rewritten by the new spawn, so any `"Not logged in"` printed by the *old* process between old-offset and new-offset is invisible to `_scan_agent_log_for_auth_failure` — the exact failure mode CLAUDE.md and `claude_gate.detect_auth_failure` were built to defeat.

  Fix: only clear `_agent_proc = None` when `kill_ok` is True; otherwise leave it pointing at the (still-alive) Popen so the next cycle's reentrancy check fires correctly.

- `portfolio/multi_agent_layer2.py:163-185, 210` — specialists bypass `claude_gate`, no tree kill, no rate limiter, no auth scan-on-completion
  `launch_specialists` spawns three `claude -p` Popens directly with `subprocess.Popen([claude_cmd, "-p", prompt, ...])`. This bypasses every gate in `portfolio.claude_gate`:
  1. No `_invoke_lock` serialization — three specialists race against any other concurrent Claude invocation (claude_fundamental from signal engine, bigbet, iskbets, the main Layer 2 agent itself).
  2. No `_popen_kwargs_for_tree_kill()` — when `wait_for_specialists` times out and calls `proc.kill()` at line 210, only the direct Claude child dies. The Node.js descendants (MCP servers, local-LLM helpers Claude may have spawned) stay alive, leaking file handles, sockets, and on a 30s-timeout retry storm, GPU VRAM. This is exactly the leak `claude_gate._kill_process_tree` exists to prevent (see comment at gate.py:354-365).
  3. No entry in `claude_invocations.jsonl`, so the daily rate limiter / cost tracking is blind to these spawns.
  4. Auth-error scan runs only AFTER `proc.wait` returns (line 222-230), so a specialist hung on a `Not logged in`/retry loop produces auth_error markers nobody scans until well past the 30s timeout, and the synthesis agent's prompt at line 113 is built from `report_paths` regardless of whether the specialists ran — letting a silent-auth-failure cascade into a "fall through to normal agent launch with synthesis prompt" (agent_invocation.py:850), now reading partial/missing specialist reports.

## P1 — high-confidence bugs (should fix)

- `portfolio/agent_invocation.py:914` — `agent.log` opened in non-atomic append mode
  ```python
  log_fh = open(agent_log_path, "a", encoding="utf-8")
  ```
  This is fine for one process but combined with the P0 above and with multi_agent_layer2 also writing specialist log files in `data/_specialist_*.log` (line 169-170 in multi_agent — fresh "w" mode each spawn so OK), any second concurrent claude that arises from the kill-failure path will interleave bytes into the same agent.log. The downstream `_scan_agent_log_for_auth_failure` slices by byte offset (`f.seek(_agent_log_start_offset)`) and decodes UTF-8 — interleaved writes produce decode errors that get swallowed by `errors="replace"`, hiding any auth marker that fell on the splice.

- `portfolio/agent_invocation.py:1411-1423` — `_check_agent_completion_locked` cleanup forgets `_agent_timeout`
  The cleanup block clears `_agent_proc, _agent_log, _agent_start, _agent_start_wall, _agent_tier, _agent_reasons, _journal_ts_before, _telegram_ts_before, _patient_txn_count_before, _bold_txn_count_before` but NOT `_agent_timeout`. `_agent_timeout` was last set by the most-recent invocation. In `_check_agent_completion_locked` line 1226 the still-running check uses `if _agent_timeout and elapsed > _agent_timeout`. If a T3 (900s) invocation finishes, completion runs, `_agent_timeout` stays at 900. If the next trigger fires invoke_agent for T1, line 943 overwrites `_agent_timeout = timeout = 180`. So in the happy path the new tier's timeout wins on the next spawn. BUT the watchdog tick can fire BETWEEN cleanup and the next spawn — `_agent_proc is None` short-circuits at line 1214 so it's harmless TODAY. Still, leaving _agent_timeout uncleared is a footgun for any future refactor that moves the watchdog check higher or makes `_agent_proc` re-assertable from another path. Fix by adding `global _agent_timeout` and `_agent_timeout = 0` to the cleanup block.

- `portfolio/agent_invocation.py:589-624` — auth_error cooldown shadows config-load failures
  ```python
  recent = load_jsonl(INVOCATIONS_FILE)
  for entry in reversed(recent[-50:]):
      ...
  ```
  Walks all 50 latest entries. The loop logic is correct: walks back over skip-statuses, the FIRST non-skip entry decides cooldown. But `load_jsonl` will silently return `[]` if the file is unreadable (Windows lock during pruning), which means we cannot detect a real auth-error cooldown and will spawn a new doomed Claude. The fallthrough comment at line 619-624 says this is intentional ("Better to attempt and fail loudly than silently skip"), but combined with the 20-spawn-burst pattern the comment itself describes (2026-05-03→05-10), letting the spawn proceed on read failure re-enables the exact storm pattern this gate was built to break. Should either propagate the file-read failure as `skipped_auth_cooldown` (fail-closed) OR cache the most recent auth_error in module state so transient file-lock races don't unblock the cooldown.

- `portfolio/main.py:608-661` — `pool.shutdown(wait=False)` may strand zombie ticker threads
  ```python
  pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ticker")
  futures = {pool.submit(_process_ticker, name, source): name for ...}
  try:
      for future in as_completed(futures, timeout=_TICKER_POOL_TIMEOUT):
          ...
  except TimeoutError:
      ...
      for f in futures:
          f.cancel()
      signals_failed += len(timed_out)
  finally:
      pool.shutdown(wait=False, cancel_futures=True)
  ```
  The comment at line 608-609 acknowledges `shutdown(wait=True)` blocks the loop. But `wait=False, cancel_futures=True` only cancels futures that haven't started running; an in-flight `_process_ticker` thread blocked on a slow yfinance or Avanza HTTP call keeps running, holds the rate limiter mutex, and possibly writes to caches the next cycle's pool re-reads. Over 24h this stacks: 5 stuck threads per stuck cycle × N stuck cycles per day = unbounded thread accumulation. No hard cap. The "5 zombie threads finishing 330-525s into the cycle" pattern documented at line 581-583 *is* this bug. Need either: (a) a per-thread timeout inside `_process_ticker` for the data-fetch + signal calls so a single ticker can be abandoned without leaking, or (b) a hard cap on live ticker threads via a semaphore around `_process_ticker`.

- `portfolio/trigger.py:165-199` — startup grace fires on `_load_state` returning `{}` from corruption
  When `trigger_state.json` is corrupted, `load_json` logs a WARNING and returns `{}`. `saved_pid = state.get(_GRACE_PERIOD_KEY)` is None. `current_pid != None`, so grace fires. Update writes a fresh state. This silently masks state-file corruption — a watch-process monitoring `data/critical_errors.jsonl` will never see it. After ~3 weeks of corruptions producing 3-week silent state loss, every restart looks "fine" because the grace path papers over it. Should detect "state loaded but had unexpected schema after corruption" (e.g. `_load_state` returned {} but file existed with non-zero size) and record_critical_error("trigger_state_corrupted").

- `portfolio/agent_invocation.py:1338-1342` — fishing context derived from possibly-stale journal tail
  ```python
  if journal_written:
      with suppress(Exception):
          new_journal_entry = last_jsonl_entry(JOURNAL_FILE)
          if new_journal_entry:
              _write_fishing_context(new_journal_entry)
  ```
  `journal_written` was computed by comparing tail-timestamps BEFORE the cleanup of `_agent_log`. Between that comparison and `last_jsonl_entry` here, another writer (metals_loop, autonomous) could append a different entry — `last_jsonl_entry` is a 4KB tail read, so we may read the WRONG strategy's entry as if it were the agent's. `_write_fishing_context` would then set `direction_bias` from the unrelated entry, poisoning XAG fish engine direction decisions. The fishing_context.json is consumed by the grid_fisher (CLAUDE.md describes "places 2-tier buy ladders" with a "tactic_weight=2.0" — strong vote). Wrong direction means real-money-equivalent buys/sells on the wrong side. Fix: capture the new entry from the same diff that set `journal_written=True`, not a fresh tail read.

- `portfolio/analyze.py:282-289` — `subprocess.run(["claude", "-p", ...])` bypasses `claude_gate`
  ```python
  result = subprocess.run(
      ["claude", "-p", prompt, "--max-turns", "1"],
      capture_output=True, text=True, timeout=120,
      env=_clean_env(), stdin=subprocess.DEVNULL,
  )
  ```
  CLAUDE.md gate.py:7-9 says: *"Direct `subprocess.Popen([claude_cmd, "-p", ...])` calls are FORBIDDEN. Doing so bypasses the kill switch, rate limiter, and invocation tracking."* This includes timeout-killing the process TREE — the direct child dies on `subprocess.run` timeout, but Node.js helpers Claude spawns are zombied (same root issue as the multi_agent_layer2 finding). The auth check at line 296-300 IS run, but uses concat-with-newline (`f"{output}\n{result.stderr or ''}"`) which the gate.py:587-604 comment explicitly warns against ("Concat-without-newline could merge the marker into the last stdout line"). Newline-with-newline IS safe, but the marker would still get appended at line >16 and `_AUTH_SCAN_LINE_LIMIT` would skip it. Same applies to `_clean_env` at analyze.py:23-37: pops only CLAUDECODE, not CLAUDE_CODE_ENTRYPOINT — a subset of the gate's `_clean_env` (gate.py:164-177) and may still produce nested-session errors.

## P2 — concerns / smells (worth addressing)

- `portfolio/agent_invocation.py:830-857` — multi-agent path blocks main loop synchronously
  After `launch_specialists` returns, `wait_for_specialists(procs, timeout=specialist_timeout)` blocks the main thread for up to `specialist_timeout` (default 30s) inside `run()`. With T2/T3 triggers firing during a busy market open, this can add 30s to the cycle budget on top of the ticker pool's 360s ceiling, pushing total runtime past the 480s `MAX_CYCLE_DURATION_S` and tripping a `cycle_duration` violation every multi-agent invocation. The TODO at line 843-845 acknowledges this. Resolve by either: (a) actually moving specialist coordination to a background thread that completes before the synthesis-prompt phase, or (b) gating multi-agent to T3 only where the 900s budget already absorbs it.

- `portfolio/main.py:846-861` — keepalive heartbeat covers `invoke_agent` but not perception_gate / drawdown gate
  `with heartbeat_keepalive():` wraps the `invoke_agent(...)` and autonomous-fallback calls. But `invoke_agent` itself does perception-gate (line 688-695) + drawdown gate (724-755) + trade-guard gate (777-820) + multi-agent specialists (829-856) BEFORE the heartbeat-keepalive even matters because the keepalive context already entered. If any of those gates raise an unhandled exception or block (e.g. trade_guards import failure on a partial deploy), the keepalive thread is still running so heartbeat stays fresh; that's correct. But the keepalive context manager only wraps invoke_agent's call — meaning if perception_gate raises and is caught at line 694-695 (just logged), the keepalive enters and exits immediately. No issue. OK.

- `portfolio/loop_contract.py:1004` — `_JOURNAL_UNIQUENESS_WINDOW_S = 600` is shorter than T3 timeout
  `check_journal_uniqueness` flags duplicate `trigger_id`s within 10 min. T3 timeout is 900s = 15 min. A T3 that genuinely times out and respawns with the same trigger_id will write two journal entries 15 min apart and NOT trigger this invariant. Tighten the window OR widen it to T3 + slack (~20 min) to catch genuine retry storms.

- `portfolio/digest.py:217-233` — bold portfolio failure swallows TypeError but not ZeroDivisionError correctly
  ```python
  except (KeyError, TypeError, ZeroDivisionError) as e:
      logger.warning("Bold state read failed: %s", e)
  ```
  But the divide-by-zero would only happen on `bold["initial_value_sek"]==0`, and the preceding guard `if bold and bold.get("initial_value_sek"):` already short-circuits when initial is zero/None. So ZeroDivisionError can't reach this except. Minor cleanup; the real concern is the comment in main.py:778 (`# BUG-103: Guard against zero/missing initial_value_sek`) suggests the broader code remembers this class of bug — keep all sites consistent.

- `portfolio/market_timing.py:14-22` — INTERVAL_MARKET_OPEN bumped to 600s while comments still claim 60s elsewhere
  The orchestration design assumes "Layer 1 runs on a 60s cadence" (CLAUDE.md), but `market_timing.py` says `INTERVAL_MARKET_OPEN = 600`. Other timing assumptions in agent_invocation comments still reference "60s cadence" (e.g. line 70 onwards). The watchdog's 30s tick (line 76) was sized for 60s cadence. With 600s cadence the watchdog ticks ~20 times per cycle, fine. But `check_agent_completion` is also called from `run()` at the top of each 600s cycle, meaning the wall-clock-timeout check is enforced effectively only via the 30s watchdog. Acknowledged in comments; verify the watchdog is actually being spawned by `_ensure_completion_watchdog` (only called from inside the success path of `invoke_agent` at line 1003 — if Popen ever fails before reaching it, no watchdog ever arms).

- `portfolio/reflection.py:149-151` — load_json positional arg vs kwarg inconsistency
  ```python
  patient = load_json(PORTFOLIO_FILE, {})
  bold = load_json(BOLD_FILE, {})
  ```
  `load_json` accepts `(path, default=None)` so positional `{}` works, but every other caller in the codebase uses `default=` keyword. Trivial style issue but easy to misread as "always default to {}, even on parse error" which is what it does — fine, just notable.

## Did NOT find

1. Silent failures: extensively guarded — `detect_auth_failure` is wired into both happy and timeout paths, `record_critical_error` boolean is checked before claiming dedup slots, `_track_module_outcome` escalates after 10 consecutive cycles.
2. Race conditions: `_completion_lock` correctly serializes watchdog and main-loop access to `_agent_proc`; `atomic_append_jsonl` uses a sidecar lock. The kill-failure shared-state bug (P0) is the only race I surfaced.
3. Money-losing math: orchestration layer doesn't price trades. Drawdown circuit breaker thresholds (20%/50%) match the documented `feedback_risk_tolerance.md` policy.
4. State corruption (other than `_agent_proc=None` after failed kill): trigger_state and contract_state writes are all atomic; JSONL appends go through `atomic_append_jsonl`.
5. Logic errors that pass tests: I didn't run the tests but the trigger downshift logic at trigger.py:393-429 is uniquely tested at module level and the regex anchoring rationale (lines 379-388) is sound.
6. Resource leaks: the ticker-pool zombie pattern (P1) is acknowledged in code; the multi_agent specialist tree-kill gap (P0) is new.
7. Time/timezone bugs: `time.monotonic()` is used consistently for elapsed checks; the `_safe_elapsed_s` fallback handles wall-clock poisoning. DST math in `market_timing.py` is correct for 2026 (verified second Sunday of March / first Sunday of November / last Sundays of March/October).
8. API misuse: this subsystem doesn't call Avanza or Binance directly; that's signals-core / metals-core.
9. Trust boundary violations: no eval/exec; subprocess argv uses list form (no shell=True); the `_extract_ticker` regex uses `\b` anchors and is safe.
10. Incorrect assumptions about partial state: `_load_state` returns `{}` defaults, and the rest of trigger.py is mostly safe to `.get(...)` chains. The startup-grace corruption-masking (P1) is the only finding here.
