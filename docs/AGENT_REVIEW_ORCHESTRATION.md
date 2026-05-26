# Orchestration Subsystem — Adversarial Code Review (2026-05-26)

**Scope:** `portfolio/main.py`, `portfolio/agent_invocation.py`, `portfolio/trigger.py`, `portfolio/trigger_buffer.py`, `portfolio/market_timing.py`, `portfolio/claude_gate.py`, `portfolio/gpu_gate.py`, `portfolio/loop_contract.py`, `portfolio/loop_health.py`, `portfolio/loop_processes.py`, `portfolio/process_lock.py`, `portfolio/escalation_gate.py`, `portfolio/escalation_router.py`, `portfolio/autonomous.py`, `portfolio/multi_agent_layer2.py`, `portfolio/subprocess_utils.py`, `portfolio/health.py`

**Method:** Whole-file audit. Cross-checked against `docs/AGENT_REVIEW_ORCHESTRATION.md` (2026-05-24) — prior findings marked [REPEAT] (unfixed) or [RESOLVED] (verified fixed since).

---

## Findings

Format: `path:line | severity | description. fix.`

### Critical (P0, 90-100)

`portfolio/main.py:973-980` | P0 | layer2/off-hours-silent-skip [REPEAT] — Bare-elif Layer 2 path still has NO autonomous fallback when `_is_agent_window()` returns False. Weekends + EU/US off-hours → XAU/XAG/BTC/ETH triggers logged as `skipped_offhours`, no journal, no Telegram, no decision. Canonical "loop runs but produces nothing" failure class. Fix: invoke `autonomous_decision(...)` under `heartbeat_keepalive()` instead of the bare `_log_trigger(..., "skipped_offhours")`.

`portfolio/agent_invocation.py:402-423` | P0 | layer2/broken-skip-gate [REPEAT] — `_no_position_skip` reads `signals` from `agent_context_t1.json`; `reporting._write_tier1_summary` (line 1191-1247) writes only `held_positions`, `all_prices`, `macro_headline`. Field is always absent → `ctx.get("signals")` returns None → gate ALWAYS returns `(True, "no_position_no_entry")` whenever no positions held. Config flag default off; flipping `no_position_skip_enabled=true` silently kills every Layer 2 entry trigger. Fix: read `weighted_confidence` from `agent_summary_compact.json` OR populate `signals.<ticker>.weighted_confidence` in `_write_tier1_summary`.

`portfolio/agent_invocation.py:1021` | P0 | multi-agent/specialist-timeout-default [REPEAT] — `specialist_timeout_s` default is 30s. Each specialist's own `SPECIALISTS[*].timeout` is 90-120s. With 30s budget every specialist gets killed at line 220 of multi_agent_layer2.py → `success_count==0` → quorum_fail → main.py:951 logs duplicate `skipped_busy_<why>`. Multi-agent mode unusable with default config. Fix: raise default to `max(SPECIALISTS[*].timeout)+30 = 150s` and align with intrinsic specialist budgets.

`portfolio/main.py:1051, 1086` | P0 | cycle-id-restart-aliasing [REPEAT] — Safeguard checks (`% 100 == 0`) and IC refresh (`% 60 == 30`) gated on `_run_cycle_id`, an in-memory counter reset on every process restart. With PF-DataLoop auto-restart `30s` and 600s cycle cadence, real cadence is "every 10h IF no restart" → effectively dead. IC cache silently stale; safeguards never fire. Fix: gate on monotonic wall-clock ts in `shared_state` (same pattern already used by `_last_log_rotation_ts` at line 404).

`portfolio/loop_contract.py:335` | P0 | autonomous-first/false-positive-storm [REPEAT] — In-flight suppression matches `status == "invoked"` exactly, but main.py:952 writes `f"invoked_{why}"` (e.g. `"invoked_drawdown_+5.5pct"`) when autonomous-first path escalates. Once `autonomous_first_enabled=true`, every escalation gets a non-matching status → in-flight check fails → `layer2_journal_activity` fires false-positive violations → critical_errors flood → fix-agent dispatcher wakes against nothing. Fix: `status == "invoked" or status.startswith("invoked_")`.

`portfolio/agent_invocation.py:1117 vs portfolio/log_rotation.py:62-67` | P0 | auth-detector-blind-to-log-rotation — `_agent_log_start_offset` captured pre-Popen. `agent.log` rotates at 10MB (log_rotation hourly post-cycle). During a long T3 (900s budget), if `rotate_all()` fires mid-invocation the offset becomes meaningless: `_scan_agent_log_for_auth_failure` (line 610-612) seeks to a stale offset in the *new* rotated file → reads either truncated tail or empty bytes → "Not logged in" lines that lived in the rotated-away segment go undetected. Re-introduces the exact silent-auth class from Mar-Apr 2026. Fix: capture log inode/path at spawn, detect mismatch at scan time, fall back to reading the rotated `.1.gz` for the missed window or treat scan-failure as auth-suspect.

### Important (P1, 80-89)

`portfolio/multi_agent_layer2.py:209-227` | P1 | wait-for-specialists-sequential [REPEAT] — `proc.wait(timeout=remaining)` iterated sequentially; third specialist receives diminishing budget after first two consume theirs. Not true parallel wait despite parallel launch. Fix: `concurrent.futures.wait(FIRST_COMPLETED)` loop.

`portfolio/multi_agent_layer2.py:220` | P1 | specialist-kill-no-tree-kill — `proc.kill()` then `proc.wait(timeout=5)` terminates only direct child. Claude CLI is Node.js spawning helper processes (MCP servers, claude API client). Orphaned grandchildren leak file handles + GPU VRAM. `claude_gate._kill_process_tree` already exists with taskkill /T; specialists ignore it. Fix: route through `claude_gate._kill_process_tree(proc, label=f"specialist_{name}")`.

`portfolio/loop_contract.py:1254-1265` | P1 | journal-uniqueness-readlines [REPEAT] — `f.readlines()` over `layer2_journal.jsonl` (up to 5000 pruned entries) per cycle. Same anti-pattern fixed by BUG-109/190 elsewhere. Also `except OSError` doesn't catch `UnicodeDecodeError` on a half-written entry. Fix: `load_jsonl_tail(LAYER2_JOURNAL_FILE, max_entries=_JOURNAL_UNIQUENESS_TAIL)` and `except (OSError, UnicodeDecodeError)`.

`portfolio/agent_invocation.py:762-783` | P1 | auth-cooldown-bounded-lookback [REPEAT] — `recent[-50:]` may push a real auth_error out of view during a burst of skipped/error entries → cooldown bypassed → fresh doomed spawn → another auth_error storm. Fix: filter by ts cutoff (last 30min) first, then look for auth_error.

`portfolio/trigger.py:233` | P1 | set-not-json-serializable [REPEAT] — `state["_current_tickers"] = set(...)` is not JSON-serializable. Line 195 pops before `atomic_write_json`, but a raise between assignment and pop crashes the next save and silently disables the prune. Fix: store list, or pass via sentinel kwarg rather than embedded.

`portfolio/autonomous.py:100` | P1 | full-jsonl-load-per-cycle [REPEAT] — `load_jsonl(JOURNAL_FILE, limit=5)` deque-trims to last 5 but reads full file from start. O(N) per cycle for a 5K-entry file. Fix: `load_jsonl_tail(JOURNAL_FILE, max_entries=5)`.

`portfolio/escalation_router.py:140-146` | P1 | per-reason-portfolio-reload [REPEAT] — `_ticker_held` opens both portfolio JSON files per ticker. N reasons × 2 files → 2N reads. Fix: cache loaded states in `should_escalate_to_claude` scope.

`portfolio/escalation_router.py:228-251` | P1 | early-return-shadows-criteria [REPEAT] — Loop returns on first match per reason. `held_sell_flip` on reason[0] returns before `top5_split` on reason[1] is evaluated. Documented as equal-weight criteria but precedence is reason-order-dependent. Fix: collect all matches, return highest-priority (or document precedence).

`portfolio/claude_gate.py:610-611` | P1 | timeout-path-skips-auth-scan — When `_run_with_tree_kill` returns `timed_out=True`, the function sets `status="timeout"` and never scans stdout/stderr for auth markers. If the CLI printed "Not logged in" early then hung on network retry, status is recorded as `timeout` not `auth_error` → critical_errors.jsonl skips the auth_failure category → fix-agent never wakes. Same asymmetry that motivated P1-3 fix in agent_invocation.py. Fix: run `detect_auth_failure(stdout, ...)` and `detect_auth_failure(stderr, ...)` on the timeout branch too; override status to `auth_error` if either hits.

`portfolio/claude_gate.py:685-690` | P1 | invoke_claude_text-return-type-mismatch [REPEAT] — Declared return is `tuple[str, bool, int]` but body returns 4-tuple `(text, success, exit_code, status)`. Callers (`iskbets.py:324`, `bigbet.py:177`, `signals/claude_fundamental.py:502`) unpack 4 successfully because Python doesn't enforce annotations, but mypy/IDE/refactor risk is real. Fix: annotate as `tuple[str, bool, int, str]`.

`portfolio/loop_health.py:213-218` | P1 | heartbeat-status-field-hardcoded-ok — `write_heartbeat(ok=False, ...)` still writes `payload["status"]="ok"`. Caller's `ok` arg lands in `payload["ok"]`, but any reader keyed on `status` would mistakenly see a failed cycle as healthy. The current watchdog reads `state` (derived from age), so this is latent — but the field name "status":"ok" is straight-up misleading and a one-line refactor of the watchdog flips it into a silent-fail. Fix: write `"status": "ok" if ok else "fail"`.

`portfolio/process_lock.py:64-66, 86-105` | P1 | lock-byte-range-release-on-truncate — `msvcrt.locking(fileno, LK_NBLCK, 1)` locks 1 byte at fh position 0. `_write_lock_metadata` then `fh.seek(0); fh.truncate(); fh.write(...)`. On Windows, truncating a file containing a byte-range lock at the locked offset is documented as unspecified behavior; some configurations release the lock. Second process can then acquire its own lock against the now-zero-length file. Singleton guarantee breaks. Fix: write metadata BEFORE acquiring the lock, or lock a high byte (e.g. byte 1024) and reserve byte 0 for metadata.

`portfolio/trigger_buffer.py:166-196` | P1 | buffer-read-modify-write-race — `flush_due` loads entries, computes remaining, then writes. No file lock around the cycle. Concurrent `add()` from another process (bigbet/iskbets/subprocess) → entries appended between load and save → lost on save. Atomic-write only protects the *write*, not the cycle. Fix: take the same singleton/process lock used by main loop, or merge read entries with current-on-disk before writing.

`portfolio/health.py:30-35, 195` | P1 | last-invocation-ts-misnamed-fires-false-silent — `update_health(last_trigger_reason=...)` writes `last_invocation_ts = last_trigger_time`. That's the TRIGGER time, not the actual subprocess completion. `check_agent_silence` then alerts "silent agent" if no triggers fire >2h, but a healthy market with no triggers is the expected case, not failure. Risk: false-positive silent alerts ranging market days. Fix: track `last_invocation_ts` from `invocations.jsonl` completion entries only (caller status==success/incomplete/failed/timeout), not from trigger logs.

`portfolio/subprocess_utils.py:129-132` | P1 | job-object-assigned-after-popen-race — `subprocess.Popen(cmd, ...)` returns when CreateProcess succeeds; the child can already be running and spawning grandchildren before `AssignProcessToJobObject` (line 132). Any grandchildren spawned in that window are NOT in the job → KILL_ON_JOB_CLOSE doesn't terminate them on parent death. Same "orphan helper" class subprocess_utils was built to prevent. Fix: spawn with `CREATE_SUSPENDED` flag, assign to job, then `ResumeThread` — standard Windows job-object pattern.

`portfolio/gpu_gate.py:73-83` | P1 | pid-reuse-blocks-stale-lock-break — `_pid_alive` uses `psutil.pid_exists(pid)`; if the original holder died and the OS recycled that PID to an unrelated process, `_pid_alive` returns True → `_try_break_stale_lock` declines → GPU lock held forever (until manual delete). The exact wedge story documented for chronos 13152 (2026-05-02) is still reachable if the recycled PID lands on something live. Fix: also verify process command line / image name matches expected llama-related binary before honoring "alive"; OR record a stable per-acquisition UUID in the lock file and verify against the holder.

### Lower priority (P2, 70-79)

`portfolio/loop_processes.py:97-106` | P2 | psutil-info-binding [REPEAT] — `info = p.info` inside try; except handler references `info.get("pid")` which is unbound if `process_iter` itself raises before assignment. Fix: init `info = {}` before try.

`portfolio/agent_invocation.py:1097` | P2 | model-pinned-no-fallback-tier-aware — `--model claude-sonnet-4-6` hardcoded; pf-agent.bat fallback at line 1107 is unconditionally T3. If Sonnet 4.6 is deprecated/removed by Anthropic, every L2 invocation fails. Fix: read model+tier from config, build with fallback ladder.

`portfolio/agent_invocation.py:1107-1108` | P2 | bat-fallback-prompt-tier-mismatch [REPEAT] — Fallback to `pf-agent.bat` is hardcoded T3 but the prompt was built for the requested tier (T1/T2). Fix: rebuild prompt as T3 when bat path is used, or skip fallback for non-T3.

`portfolio/main.py:282` | P2 | post-cycle-untracked-task [REPEAT] — `_maybe_send_digest(config)` not wrapped in `_track` like sibling tasks. Failures don't appear in `report.post_cycle_results`. Fix: `_track("digest", _maybe_send_digest, config)`.

`portfolio/multi_agent_layer2.py:188-189` | P2 | popen-attr-patching [REPEAT] — `proc._log_fh = log_fh; proc._name = name` patches private-namespaced attrs on Popen. Works on CPython, fragile to refactor / not type-checkable. Fix: wrap in a dataclass keyed by pid.

`portfolio/multi_agent_layer2.py:228-240` | P2 | post-kill-log-race [REPEAT] — Auth scan reads `_specialist_<name>.log` after `proc.kill()`; with no fsync of the log_fh before close, the last buffered chunk may be missing the "Not logged in" line. Rare. Fix: `os.fsync(log_fh.fileno())` before close, OR scan only after a small bounded wait.

`portfolio/gpu_gate.py:147` | P2 | sweeper-sleeps-before-checking — `_sweeper_loop` calls `time.sleep(_SWEEPER_INTERVAL_SECONDS)` BEFORE first `_try_break_stale_lock()` call. On freshly-restarted loop with a pre-existing stale lock, first sweep happens 30s after process start. Cosmetic. Fix: check first, then sleep.

`portfolio/subprocess_utils.py:209` | P2 | os-import-via-dunder — `__import__("os").getpid()` works but `os` is already imported indirectly via `subprocess`. Just `import os` at top. Hygiene only.

`portfolio/subprocess_utils.py:142-143` | P2 | timeout-kill-no-tree-kill — `_run_with_job_object` on TimeoutExpired calls `proc.kill()` (direct child only) then `proc.communicate()`. The Job Object will kill the tree at `CloseHandle(job)` in the finally — relies on KILL_ON_JOB_CLOSE flag firing. Fine in current flow but fragile if `_create_job_object` ever drops that flag. Defense-in-depth: explicitly close the job on timeout BEFORE communicate-drain.

`portfolio/escalation_gate.py:202-215` | P2 | thread-pool-per-call [REPEAT] — `ThreadPoolExecutor(max_workers=1)` created and destroyed per gate call. ~50/cycle in autonomous-first mode. Fix: module-level singleton executor.

`portfolio/market_timing.py:244-260` | P2 | misleading-window-name [REPEAT] — `_is_agent_window` returns False on weekends entirely. Name implies general agent layer; actually US-business-hours gate. Confused future maintainers (see P0 #1). Fix: rename to `_is_us_claude_window`.

`portfolio/health.py:188-196` | P2 | last_invocation_ts-cache-write-skips-lock — Line 193-196 takes the lock but the early read at 183 doesn't. Two callers can both pass the `if not last_ts:` check on a missing key, both parse, both write. Idempotent so harmless, but lock is meant to serialize. Fix: enter lock once, read+write under it.

### Top 3 Summary

1. **`portfolio/main.py:973-980` off-hours silent skip [REPEAT, 5th review unfixed]** — Two months of reviews have flagged this; the canonical "loop runs but produces nothing" failure mode the project's own CLAUDE.md warns against (the Mar-Apr 2026 auth-silent outage). 24/7 instruments (XAU/XAG/BTC/ETH) lose all decision coverage every weekend and overnight. Fix is ~5 lines: mirror the autonomous fallback already present in the autonomous-first branch.

2. **`portfolio/agent_invocation.py:1117` auth-detector blind to log rotation [NEW]** — `_agent_log_start_offset` captured pre-Popen against a file that rotates hourly at 10MB. A 900s T3 invocation crossing the rotation boundary leaves the offset pointing at a now-empty or shorter file → scan reads nothing → "Not logged in" lines in the rotated-away segment never detected. Exact silent-auth class from Mar-Apr 2026 reachable via a single log rotation timing. The fix-agent dispatcher relies on this scan landing a row in critical_errors.jsonl; lose the scan and we're back to "exit 0 with no work done, no alert."

3. **`portfolio/main.py:1051, 1086` cycle-id-restart aliasing [REPEAT unfixed]** — IC cache refresh and safeguard checks gated on a counter reset every process restart. With auto-restart-30s on PF-DataLoop and 600s cadence, these branches are effectively never executing. Silent data degradation: IC weighting goes stale; outcome-staleness alerts never fire; dead-signal detection never fires. The wall-clock pattern already exists at line 404 for `_last_log_rotation_ts` — copy-paste fix.

Both #1 and #3 are flagged for the 5th consecutive monthly review. Per CLAUDE.md "System reliability is #1": fix these BEFORE shipping any new feature. The two-line nature of #1's fix vs the multi-month miss is the loudest signal in the codebase right now.
