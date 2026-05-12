# Claude adversarial review: orchestration (2026-05-12)

## Summary

All four prior P0s from 2026-05-11 are STILL PRESENT verbatim — none of
the recommended fixes landed in the past 24h. The most worrying class
is the persistent set of subprocess sites that bypass `claude_gate`
entirely: `bigbet.py:175`, `iskbets.py:322`, `analyze.py:282` &
`:746`, the `pf-agent.bat` fallback at `agent_invocation.py:824-829`,
and the direct `subprocess.Popen` paths in `agent_invocation.py:869`
and `multi_agent_layer2.py:168`. Five of those bypass the
`_invoke_lock`/tree-kill/daily-rate-limit/auth-detection guarantees
documented at `claude_gate.py:8` ("Direct subprocess.Popen calls are
FORBIDDEN"). A new wrinkle: `agent_invocation.invoke_agent` itself —
the canonical Layer 2 path — does NOT actually route through
`invoke_claude` either; it manages its own `_agent_proc` state and
`_completion_lock`, which is correct for the tier-based async model
but means the "everyone routes through claude_gate" rule is aspirational
even for the primary call site.

Secondary themes: (1) several O(N) full-journal reads on every Layer 2
invocation (`journal.load_recent`, `journal_index.retrieve_relevant_entries`,
`agent_invocation._build_decision_feedback`, `claude_gate._count_today_invocations`)
remain unchanged from yesterday and now compound now that
prewarmer+watchdog+specialist threads all also walk the same files;
(2) `flip_cooldowns` and `triggered_consensus` are pruned per-cycle in
`trigger._save_state` (good!) but `flip_cooldowns` itself is NOT
pruned, only `triggered_consensus`; (3) several module-level globals
still have racy access patterns — `_agent_log_start_offset`,
`AlertBudget._sent_timestamps`, `prophecy.save_beliefs` — and the
`_completion_lock` doesn't cover the read paths it should.

## P0 — Blockers

- `portfolio/agent_invocation.py:847` — **UNCHANGED FROM 2026-05-11**.
  `agent_env["NODE_OPTIONS"] = "--stack-size=16384"` still
  unconditionally overwrites any inherited `NODE_OPTIONS`. Same issue
  at `portfolio/multi_agent_layer2.py:145`. The interaction with the
  stack-overflow auto-disable (`agent_invocation.py:544`) is what
  makes this P0 not P1: if the user ever exports any unrelated
  `NODE_OPTIONS`, the Layer 2 stack-overflow workaround silently
  drops, the agent crashes 5 times in a row, Layer 2 auto-disables,
  and the user has no audit trail of why. Fix:
  `agent_env["NODE_OPTIONS"] = f"--stack-size=16384 {os.environ.get('NODE_OPTIONS','')}".strip()`.

- `portfolio/llm_prewarmer.py:299-301` — **UNCHANGED FROM 2026-05-11**.
  `query_llama_server(server_slot, "test", n_predict=1, temperature=0.0)`
  is still called synchronously from `flush_llm_batch` at
  `llm_batch.py:312` on the loop thread. The call grabs
  `llama_server._thread_lock` AND a 300s-timeout file lock, then runs
  `_stop_server` → `_wait_for_vram_reclaim` (up to 30s plex-safe) →
  `subprocess.Popen` → 90s startup poll. Worst-case 120s on the main
  thread right after a 25-40s batch. The wrapper at `llm_batch.py:310-314`
  only catches exceptions, not slow paths. Fix: dispatch on a daemon
  thread with a strict 30s guard, or pre-flight with
  `model_load_safe()` + `_query_free_vram_mb()` and skip if VRAM is
  tight.

- `portfolio/bigbet.py:175-181` — **UNCHANGED FROM 2026-05-11**.
  Raw `subprocess.run(["claude", "-p", prompt, "--max-turns", "1"], ..., timeout=30)`
  bypasses `claude_gate.invoke_claude_text` entirely. No tree-kill on
  timeout (grandchild leak class documented at `claude_gate.py:282-291`),
  no `_invoke_lock` so the 8-worker ticker pool can fan out N parallel
  bigbets, no daily rate-limit counter, no kill-switch honored. The
  env-cleanup at `bigbet.py:173-174` strips only `PF_HEADLESS_AGENT`
  setup — `CLAUDECODE` is NOT removed (only added by the gate's
  `_clean_env`), so a metals-loop-spawned bigbet inherits the parent's
  Claude Code session marker and the CLI errors with "nested session".
  Fix: route through `invoke_claude_text(prompt, caller="bigbet",
  model="haiku", timeout=30)`.

- `portfolio/agent_invocation.py:824-829` — **UNCHANGED FROM 2026-05-11**.
  `cmd = ["cmd", "/c", str(agent_bat)]` then `Popen(cmd, ...)` falls
  back to `pf-agent.bat` when `claude` is not on PATH. This path:
  (1) bypasses `claude_gate._invoke_lock`, (2) uses tier-1 timeout
  (`_agent_timeout=120` at line 866) but the bat runs as Tier 3
  (max_turns 40 / playbook full-review), causing systematic
  watchdog-kills mid-flight, (3) `cmd /c` may not propagate the
  cleaned env on Windows depending on the bat file's local `set`
  calls, (4) `_kill_overrun_agent` uses `taskkill /F /T /PID` on
  `_agent_proc.pid` — but that pid is `cmd.exe`'s, not the actual
  claude grandchild, leaving orphans. Codex 2026-04-17 added a
  partial fix to publish `effective_tier=3` to health_state (line
  908) but the `_agent_timeout` itself was not aligned.

## P1 — High

- `portfolio/agent_invocation.py:869` — `_agent_proc = subprocess.Popen(cmd, ...)`
  on the happy path ALSO does not route through `claude_gate`. The
  module owns its own `_completion_lock`/`_invoke_lock`-equivalent
  state, which is correct for the async tier model, but means
  `claude_gate.invoke_claude*`'s `_DAILY_WARN_THRESHOLD` counter
  (`claude_gate.py:454`) misses all Layer 2 invocations entirely.
  `get_invocation_stats()` at line 651 reports a number that is
  systematically under the truth by `_log_trigger`'s count. Fix:
  either call `_log_invocation` from `agent_invocation` too, or
  document that Layer 2 doesn't count toward the gate's quota
  (mention in the gate docstring).

- `portfolio/agent_invocation.py:428` — `f.seek(_agent_log_start_offset)`
  reads the module global without holding `_completion_lock`.
  `_kill_overrun_agent` (line 446) is sometimes called by the watchdog
  inside the lock, but `_scan_agent_log_for_auth_failure` is also
  callable directly. If a previous invocation crashed mid-cleanup
  between line 838 (`global _agent_log_start_offset` assignment) and
  successful cleanup at line 1336, the offset is stale. Next
  invocation's scan re-reads agent.log starting at a stale byte
  position — possibly re-flagging the previous run's auth markers
  (the exact echo-feedback pattern that `_AUTH_MARKER_PREFIX_REJECT`
  was built to defeat). Fix: snapshot the offset into a local before
  any read, and reset it to `0` in cleanup so a stale offset never
  leaks.

- `portfolio/claude_gate.py:271-279` — `_count_today_invocations`
  does a full re-parse of `claude_invocations.jsonl` on EVERY
  invocation. Capped at 5000 entries by the prune at `main.py:362`
  but still O(N) on a check that should be O(1). Fix: cache today's
  count in `shared_state` keyed by UTC date, reset at midnight.

- `portfolio/multi_agent_layer2.py:207` — `proc.kill()` on specialist
  Popen does NOT tree-kill. Specialist claude processes spawn Node
  helpers + MCP servers like the main agent does. The cleanup at
  line 211-213 closes the log_fh but the grandchild still holds
  the file handle on Windows. Fix: import and use
  `claude_gate._kill_process_tree(proc, label=f"specialist_{name}")`.

- `portfolio/multi_agent_layer2.py:127-182` — `launch_specialists` also
  spawns `subprocess.Popen` directly with `claude_cmd, "-p", prompt,
  ...`. Same `claude_gate` bypass class as `bigbet.py`. The three
  parallel specialists hit Claude in lockstep — no `_invoke_lock`
  to serialize. Combined with the main agent that may launch in the
  same tick, that's 4 concurrent claude CLI processes, each ~500MB
  RAM. Fix: either serialize via `_invoke_lock` (kills the
  multi-agent latency win) or document that the four spawns are an
  exception requiring explicit memory accounting.

- `portfolio/prophecy.py:70-73` — `save_beliefs` still has NO lock.
  `evaluate_checkpoints` runs every cycle on the loop thread;
  `add_belief` / `update_belief` / `remove_belief` (lines 76, 105, 128)
  all do read-modify-write via this unprotected save. Layer 2 can
  call these via its prompt's tool budget (Bash+Edit) — a concurrent
  checkpoint evaluation will silently overwrite the belief update.
  Fix: module-level `threading.Lock` around load+save.

- `portfolio/journal.py:23-40` — `load_recent` still opens the
  whole layer2_journal.jsonl with plain `open()` and iterates every
  line. Called from `journal.write_context` on every Layer 2
  invocation (`agent_invocation.py:607`). Capped at 5000 lines but
  pays O(N) every time. Fix: `load_jsonl(JOURNAL_FILE, limit=50)` or
  `load_jsonl_tail` — same fix the 2026-05-11 review recommended.

- `portfolio/journal_index.py:370-383` — `retrieve_relevant_entries`
  also reads the entire journal via plain `open()`. Called from
  `journal.write_context` when `smart_retrieval=true`, doubling the
  cost. Fix: replace with `load_jsonl(..., limit=500)` and let BM25
  rank the bounded subset.

- `portfolio/agent_invocation.py:293` — `_build_decision_feedback`
  calls `load_jsonl(JOURNAL_FILE)` (full read) on EVERY
  `invoke_agent` (line 800). Third full-journal read per invocation,
  after `load_recent` and `retrieve_relevant_entries`. Fix:
  `load_jsonl(JOURNAL_FILE, limit=100)` — most-recent 5 ticker
  matches almost certainly fall inside the last 100.

- `portfolio/health.py:152-166` — `check_staleness` calls
  `datetime.fromisoformat(hb)` and then `datetime.now(UTC) - last`.
  If `hb` is tz-naive, the subtract raises `TypeError`. The except
  at line 162 catches `ValueError, TypeError`, returns `(True,
  float("inf"), state)` — "stale forever" until someone manually
  edits the file. Same pattern in `check_agent_silence` at line 202
  (catches both, but `last_ts` source from health_state.json may
  also be naive from older builds). `update_health` writes
  `datetime.now(UTC).isoformat()` (line 25), so newly-written
  values are aware — but if a historical write was naive, the file
  is permanently stale-flagged. Fix: `last.replace(tzinfo=UTC) if
  last.tzinfo is None else last`.

- `portfolio/main.py:1147` — heartbeat staleness check at startup:
  `last_beat = datetime.fromisoformat(heartbeat_file.read_text().strip())`
  then `(datetime.now(UTC) - last_beat).total_seconds()`. If the
  prior loop crashed before writing a tz-aware heartbeat, this
  raises TypeError, caught at line 1159, no restart notification
  fires. Fix as health.py above.

- `portfolio/main.py:1129-1137` — Singleton lock still acquired
  BEFORE `atexit.register(_release_singleton_lock)`. If
  `_acquire_singleton_lock` raises (not just returns False — the
  `RuntimeError` at line 69-71 for "no msvcrt/no fcntl" path),
  atexit is never registered. The lock file persists across the
  crash. Fix: register atexit first, then acquire — the release is
  a no-op when nothing was acquired.

- `portfolio/llama_server.py:553-568` — `_thread_lock` + 300s file
  lock both held for the entire `_ensure_model` (which can take 90s
  startup + 30s vram reclaim) + `_query_http` (up to 240s). If a
  query wedges below the timeout on a connection-establish (Windows
  doesn't always honor `requests.post(timeout=240)` for socket-level
  hangs), every in-process caller of `query_llama_server` blocks for
  up to ~6 minutes. Same finding as 2026-05-11. Fix: wrap the HTTP
  call in `concurrent.futures.ThreadPoolExecutor.submit(...).result(timeout=240)`
  with hard cancel.

- `portfolio/telegram_notifications.py:60-81` — Markdown-failure
  retry path still has the silent-loss class. `r2.json()` is not
  called on the no-parse_mode retry, so a 400 with empty body on
  the retry just returns False. The first 400 has a parse-keyword
  guard (line 72) so only matching errors retry — a 400 with a
  non-parse-related error description (rate limit, chat blocked,
  message too long edge cases) is silently dropped on the first
  attempt. Fix: retry once without parse_mode on ANY 400, then log
  if the retry fails.

- `portfolio/agent_invocation.py:1244-1257` — `atomic_append_jsonl(INVOCATIONS_FILE, log_entry)`
  is wrapped in try/except but on failure (`logger.warning`), the
  completion path still claims success at line 1272. `_record_new_trades`
  runs (line 1270), `_consecutive_stack_overflows` is reset (line 1327),
  module state is cleared (line 1336-1345). But the invocation row
  is missing — `get_completion_stats` permanently under-reports.
  Fix: retry once, then `record_critical_error("invocation_log_write_failed",
  ...)`.

## P2 — Medium

- `portfolio/trigger.py:282` — `flip_cooldowns` dict grows unbounded
  per ticker. `_save_state` (line 117-126) prunes `triggered_consensus`
  based on `_current_tickers` but does NOT prune `flip_cooldowns`.
  Stale tickers accumulate forever. With XBT-TRACKER /
  ETH-TRACKER / MINI-SILVER aliases plus any historical delistings
  in the `_current_tickers` set, the dict could carry 100+ stale
  keys after months of running. Fix: same prune treatment as
  `triggered_consensus` — `pruned = {k: v for k, v in flip_cooldowns.items()
  if k in current_tickers}` at line 124.

- `portfolio/trigger.py:160-199` — Startup grace branch still does an
  unlocked read-modify-write of `trigger_state.json`. A concurrent
  reader (e.g., dashboard `/api/triggers`) sees half-built state
  between load (line 162) and save (line 198).

- `portfolio/circuit_breaker.py:82-106` — The HALF_OPEN race is
  unchanged. `allow_request` returns True only on the OPEN→HALF_OPEN
  transition; if the caller crashes between line 97 and `record_success`
  / `record_failure`, state is HALF_OPEN with `probe_sent=True`
  forever, and `allow_request` always returns False at line 106 —
  the breaker is wedged. Fix: wrap the transition + probe in a
  try/except that reverts to OPEN if the caller raises, OR add a
  recovery_timeout-on-HALF_OPEN that promotes back to OPEN if no
  result was recorded within the timeout.

- `portfolio/llama_server.py:178-180` — `logger.debug("Failed to kill server pid=%s", pid)`
  references `pid` which is unbound if (a) `os.path.exists(_PID_FILE)`
  returned False (the inside-the-if branch never runs), (b) the
  `with open()` raised, or (c) `content` was empty (`if content:`
  skipped). All three paths land in the except handler. UnboundLocalError
  on the logging call gets caught by the bare `except:` so it's
  silent — but the debug message is useless on those error paths.
  Fix: initialize `pid = None` at the top of the try block.

- `portfolio/multi_agent_layer2.py:196-210` — `wait_for_specialists`
  uses `remaining = max(1, deadline - time.time())`. Once the deadline
  passes, every still-running proc gets exactly 1s instead of 0s. With
  3 specialists × 1s = 3s extra after the explicit timeout. Combined
  with the no-tree-kill issue above, a hung specialist with N
  grandchildren can extend the wait by up to 3s × cleanup latency
  per grandchild.

- `portfolio/alert_budget.py:36-46` — `_sent_timestamps` (deque) and
  `_buffer` (list) mutated without a lock. The class is imported into
  telegram paths that can be hit from the 8-worker ticker pool +
  poller thread + watchdog thread. Race causes occasional
  double-increment of the budget or buffer corruption. Fix: add a
  `threading.Lock` and wrap `should_send` + `flush_buffer`.

- `portfolio/perception_gate.py:60-65` — On cold start (both
  `agent_summary_compact.json` AND `agent_summary.json` missing,
  `_load_compact_summary` returns None), the gate returns
  `(True, "no summary available, pass through")` — gate effectively
  DISABLED at the moment the system is most uncertain. Fix:
  combine with a "wait for ≥1 successful cycle" guard, OR return
  `(False, "cold start, no signal data yet")` to fail-closed.

- `portfolio/decision_outcome_tracker.py:43` — `decisions[-max_entries:]`
  truncates from the END. Old unfilled entries (decisions made >3d
  ago that never got an outcome record because the price fetch
  failed at the time) are NEVER retried. The horizon-3d window is
  silently lossy. Fix: backfill OLDEST unfilled first, or scan all
  decisions and skip only those that already have outcomes.

- `portfolio/regime_alerts.py:39-43` — `_get_last_regime` does
  `load_jsonl(REGIME_HISTORY_FILE)` (full read) for every ticker
  per cycle. With 5 Tier-1 tickers × every cycle, and the file
  growing unbounded (no prune in `main.py:357-364`), this is the
  next file to blow past 50MB. Fix: use `load_jsonl_tail` and
  iterate in reverse.

- `portfolio/digest.py:48-52` — `_set_last_digest_time` writes state
  via `_atomic_write_json` but loads via `load_json(...default={})`.
  `load_json` returning `{}` on missing file means the `if not state`
  branch always evaluates to True (because `{}` is falsy). Cosmetic —
  ultimately the state IS written. But the migration check at line
  39-40 (read `trigger_state.json` to find legacy timestamp) ALSO
  runs every call until the new file is written, doubling the I/O
  on bootstrap.

- `portfolio/digest.py:217-233` — Bold portfolio load uses bare
  `except (KeyError, TypeError, ZeroDivisionError)` — missing
  `OSError`/`FileNotFoundError`. A partial-file read (file removed
  between `exists()` check at line 217 and `load_json` at line 219)
  raises something not caught, crashing the digest entirely.

- `portfolio/autonomous.py:88-90` — `except Exception:` swallows
  errors via `logger.exception`. If the autonomous path is the only
  Layer 2 path (`layer2.enabled=false`), chronic exceptions here
  mean NO journal/decision/telegram for an unknown duration, and
  the loop reports `_log_trigger(..., "autonomous", ...)` (line 251
  in `agent_invocation.py`) as if it succeeded. Fix: include the
  outcome in the invocation log status (e.g. status="autonomous_error").

- `portfolio/health.py:182-196` — `check_agent_silence` reads
  `state["last_invocation_ts"]` from `load_health()` (line 182,
  outside lock) then later does a read-modify-write at line 193-196.
  Concurrent `update_health` calls between the unlocked load and
  the locked write can drop fields. Fix: read inside `_health_lock`.

- `portfolio/llm_batch.py:300` — `_ss._full_llm_cycle_count += 1` is
  not atomic across processes. The metals loop runs its own LLM
  batch in a separate Python process and bumps a different counter.
  The prewarmer trusts the main loop's counter only. If a future
  integration shares the counter via shared_state file, the
  off-by-one cascades into incorrect slot prewarming. Document or
  add a file-based counter.

- `portfolio/telegram_poller.py:262-265` — `should_persist` settles
  on `(processed=True) OR drop_reason in _SETTLED_DROP_REASONS`. The
  `_save_offset` failure at line 92-101 is best-effort — a transient
  disk error during ack leaves the in-memory offset advanced but
  not persisted. Next restart re-fetches that update, the stale
  filter at startup_time-60s drops it (because msg_date is old),
  and the user's queued command is permanently lost. This is the
  exact bug round-7 was supposed to fix; the bypass at
  `RESTART_BYPASS_MAX_AGE_S` (line 39) helps but only within 1h.
  Fix: hard-fail the loop iteration if `_save_offset` fails AFTER
  processing — better to re-process than to drop.

- `portfolio/agent_invocation.py:578-579` — `_safe_elapsed_s()` is
  called outside `_completion_lock` for the reentrancy check, but
  the read of `_agent_start` / `_agent_start_wall` is unprotected.
  The watchdog could be mid-clearing those globals at line 1338-1339
  when this read happens. Most likely outcome: `raw < 0`, caught
  by the wall-clock fallback. Defensive but not robust. Fix: take
  `_completion_lock` for the entire reentrancy check (extends the
  existing critical section).

## P3 — Low

- `portfolio/main.py:1115-1126` — `_sleep_for_next_cycle` warns on
  overrun every cycle without rate-limiting. Sustained 300s+ cycles
  emit one WARNING per cycle to agent.log. Fix: rate-limit the
  warning to once per minute, OR include a count summary every
  10 overruns.

- `portfolio/main.py:991` — `_consecutive_crashes = _load_crash_counter()`
  at module import. If the counter was last bumped weeks ago (e.g.
  `data/crash_counter.json` left at count=4 after a recovery that
  forgot to reset), the next crash triggers alert suppression
  immediately at threshold 5. Fix: check `updated` timestamp from
  the counter file, reset if older than 24h.

- `portfolio/reflection.py:166-172` — `last_ts = datetime.fromisoformat(last["ts"])`
  followed by `datetime.now(UTC) - last_ts` raises TypeError if
  `last_ts` is naive. Caught by `except (KeyError, ValueError)` —
  which does NOT include `TypeError`. Result: the function raises
  out of the caller. Fix as `health.py`.

- `portfolio/agent_invocation.py:1018` — `text = f"...CRASH LOOP SUMMARY..."`
  passes raw markdown to `send_or_store`; if the latest error
  message contains markdown specials (`*`, `_`, `[`), the Telegram
  render fails and the message is dropped via the unparse retry
  path at `telegram_notifications.py:74` — but if the body also
  has unescaped chars the second send may fail silently.

- `portfolio/llm_prewarmer.py:230-235` — Pytest detection via
  `PYTEST_CURRENT_TEST` env var. If a subprocess inherits the var
  from a wrapping pytest run (rare in CI), the prewarmer never
  fires for the duration of that subprocess. Document the assumption
  in the docstring (the override `PF_PREWARM_FORCE_RUN=1` exists,
  but undocumented).

- `portfolio/regime_alerts.py:39-43` — see P2 entry above —
  classified P3 if file is regularly pruned by log_rotation; P2
  otherwise. Currently no prune entry in `main.py:360`.

- `portfolio/journal_index.py:60-65` — `BM25._idf` returns 0 for
  unseen terms (line 64). The math at line 86 then multiplies by 0
  inside the score sum; harmless. But the avg_doc_len math at
  line 50 (`sum(self.doc_lens) / self.doc_count if self.doc_count else 1`)
  is defensive — if `doc_count == 0` the loop at line 76-87 doesn't
  execute. Code is correct, but the defense is for an unreachable
  state. Worth documenting that BM25 requires `fit(documents)` with
  N ≥ 1 documents.

- `portfolio/agent_invocation.py:153-168` — `_load_stack_overflow_counter`
  and `_save_stack_overflow_counter` use `load_json` /
  `atomic_write_json` correctly, but the read at module import
  (line 171) means a corrupted file makes Layer 2 think it has 0
  recent overflows when it might really have 5 — re-enables a
  broken claude binary. Fix: log WARNING if `load_json` returned
  None for an existing file.

## Status of prior P0s (2026-05-11)

- **`agent_invocation.py:846` NODE_OPTIONS overwrite** — **STILL
  PRESENT.** Line is now 847; code identical. No fix applied.
- **`multi_agent_layer2.py:145` NODE_OPTIONS overwrite** — **STILL
  PRESENT.** Line 145 identical.
- **`llm_prewarmer.py:299-301` synchronous query_llama_server** —
  **STILL PRESENT.** Lines 299-301 identical. Still called
  synchronously from `flush_llm_batch` at `llm_batch.py:312`.
- **`bigbet.py:175-181` direct subprocess bypasses claude_gate** —
  **STILL PRESENT.** Lines 175-181 identical. Auth detection was
  added inline (line 191-193) but the gate bypass remains. The
  `CLAUDECODE` env strip is also still missing.
- **`agent_invocation.py:824-830` pf-agent.bat fallback** — **STILL
  PRESENT.** Lines 824-829 identical. Codex 2026-04-17 publishes
  `effective_tier=3` to health_state (line 908) so the loop_contract
  uses the right grace window, but `_agent_timeout=120` for a
  T1-requested-via-bat fallback is still set from `tier_cfg["timeout"]`
  at line 563, causing watchdog kill mid-flight without tree-kill.

The 4 P0s are 4/4 unaddressed. Of the 11 P1s from 2026-05-11, the only
visibly addressed one is `_completion_watchdog` arming (line 925 now
calls `_ensure_completion_watchdog()` — looks complete). The other 10
remain unchanged or only partially patched.

## Tests missing

- No test covers the silent exit-0 auth-failure path through the
  `pf-agent.bat` fallback (the bypass).
- No test covers `_kill_overrun_agent` failing to tree-kill the
  Claude CLI Node grandchildren.
- No test for `multi_agent_layer2.launch_specialists` failing to
  tree-kill grandchildren on timeout.
- No test for concurrent `prophecy.save_beliefs` write-races.
- No test that `prewarm_next_model` does NOT block the main loop
  > 30s (no synchronous deadline guard exists).
- No test that `_completion_watchdog` does NOT spawn duplicates
  under concurrent `invoke_agent` reentry attempts.
- No test for `bigbet.invoke_layer2_eval` race with `claude_gate._invoke_lock`
  (the bypass means there is no serialization).
- No assert that `NODE_OPTIONS` from environment is preserved
  through Layer 2 / specialist spawns.
- No test that `flip_cooldowns` dict is pruned of stale tickers.
- No test that `check_staleness` / `check_agent_silence` handle
  tz-naive last_heartbeat values without falsely flagging stale.
- No test for `telegram_poller._save_offset` failure resulting in
  the queued-command-loss class (round-7 bug regression).
