# Independent Reviewer Pass — 2026-05-17

Hand-written cross-cutting review by the orchestrator. Targets seams between
subsystems and recurring hazard categories. Read in parallel with the 10
subagent reports; this file is the cross-check.

## Methodology
Read at full depth:
- portfolio/file_utils.py (atomic I/O primitives)
- portfolio/shared_state.py (concurrent cache, rate limiters)
- portfolio/portfolio_mgr.py (state read-modify-write)
- portfolio/trigger.py (first 120 lines + spot checks)
- portfolio/agent_invocation.py (auth failure detection, CLAUDECODE clear, kill path)
- portfolio/signal_engine.py (voting math, HOLD dilution, gate cascade)
- portfolio/claude_gate.py (kill switch, env hygiene)
- portfolio/health.py (heartbeat)
- dashboard/auth.py (CF/cookie/query/bearer)
- dashboard/app.py (route auth grep)

Skim greps: bare except: across portfolio (169 occurrences), raw `json.load(open())`
across data/ scripts (60+ occurrences), stop-loss endpoint usage.

## Cross-cutting findings

### P1
None new. The subsystem agents flag the in-scope P1s; my pass confirms the
high-leverage primitives are sound:
- `portfolio/file_utils.py:24-63` atomic_write_text / atomic_write_json use
  `tempfile.mkstemp(dir=path.parent)` + `os.fsync` + `os.replace` — correct on
  Windows (same volume, atomic rename).
- `portfolio/file_utils.py:202-258` `jsonl_sidecar_lock` correctly addresses
  the "two first-writers race on empty file" failure mode with a pre-seeded
  sidecar `\0` byte, blocking msvcrt locking on Windows, blocking flock on POSIX.
- `portfolio/file_utils.py:261-284` `atomic_append_jsonl` wraps the append inside
  the sidecar lock — torn-line risk addressed.
- `portfolio/portfolio_mgr.py:35-41` per-file `threading.Lock` resolved C8.
  `update_state(mutate_fn)` (L136-159) holds the lock across the full
  read-modify-write — no torn-update race.
- `portfolio/agent_invocation.py:1033-1047` CLAUDECODE pop + PF_HEADLESS_AGENT=1
  + stdin=subprocess.DEVNULL — three of the proven CLAUDE.md silent-failure
  patterns all handled correctly.
- `portfolio/agent_invocation.py:527-577` `_scan_agent_log_for_auth_failure`
  scans agent.log slice on BOTH the happy-completion AND the timeout-kill path
  for the "Not logged in" marker — directly addresses the 3-week Mar-Apr 2026
  silent auth outage.
- `dashboard/auth.py:103-200` `require_auth` uses `hmac.compare_digest`,
  verifies Cf-Access JWT via `dashboard.cf_access.verify_cf_jwt`, sets cookie
  httponly+secure+samesite=Lax — correct.
- Stop-loss endpoint hits `/_api/trading/stoploss/new` (avanza_session.py:801),
  matching CLAUDE.md's mandatory path.

### P2 (real)

**P2-1: `data/` is a junk drawer of one-off scripts with non-atomic JSON I/O.**
Greppable: 60+ occurrences of `json.load(open(...))` and `json.dumps(..., f)` in
`data/_*.py`, `data/layer2_*.py`, `data/tmp_*.py`, `data/_check_*.py`,
`data/_send_*.py`, etc.
- These appear to be transient agent-spawned scripts (likely written by past
  Layer 2 sessions). They violate the CLAUDE.md atomic I/O rule.
- Risk: if any of them write to a state file that the loop reads concurrently
  (`portfolio_state.json`, `metals_swing_state.json`, etc.), they can produce
  torn writes that crash the loop or corrupt state.
- Fix suggestion: nightly job to archive `data/_*.py` and `data/tmp_*.py` to
  `data/archive/_one_shots/` rather than letting them accumulate, and explicit
  policy that one-shot scripts must use file_utils.

**P2-2: `data/layer2_execute.py:81`, `data/layer2_invoke.py:75`, `data/layer2_exec.py:77`,
`data/layer2_send_now.py:92` all do `json.load(open("config.json"))`.**
- config.json is the symlink to the API-key file. Reading it is fine; the issue
  is these are duplicate ad-hoc scripts that bypass `portfolio.api_utils.load_config`
  and risk drift (e.g., if config schema changes).
- Lower risk than P2-1 because reads can't corrupt the file; flagged for cleanup.

**P2-3: agent_invocation.py heartbeat tier-publish swallows all errors.**
`portfolio/agent_invocation.py:1107-1121` writes `last_invocation_tier` to
health_state.json inside `try: ... except Exception as e: logger.warning(...)`.
If the write silently fails (disk full, file locked by antivirus), the
loop_contract grace window uses stale tier — for a T1 invocation that ran 3min,
the loop_contract may still apply T3's 20m grace, masking a real silent failure.
Suggestion: count consecutive failures and surface to critical_errors.jsonl
after N misses.

**P2-4: `portfolio/file_utils.py:283-284` `atomic_append_jsonl` does write+flush+fsync
inside the lock, but does not check disk-full / ENOSPC.**
If the disk fills, the append fails and the caller's try/except (usually
`except Exception:`) swallows it. The journal silently stops growing — every
Layer 2 invocation looks like "no journal entry", which `agent_invocation.py`'s
count-delta heuristic (L1059) reports as `journal_written=false` for every
subsequent run. Suggestion: bubble ENOSPC explicitly so health_state notes it.

**P2-5: `portfolio/agent_invocation.py:1087-1097` patient/bold txn count
snapshot swallows broad `Exception` and falls back to 0.**
If `portfolio_state.json` is mid-write (rare with atomic_write but possible
during corruption recovery), load_json returns None, then `.get("transactions", [])`
on None raises AttributeError, caught here as `_patient_txn_count_before = 0`.
Downstream record_trade detects "new trades" wherever 0 != current count.
False-positive trade-recording risk on the first cycle after corruption.
Suggestion: use `load_json(default={})` and verify dict shape.

**P2-6: `data/metals_loop.py:618` `atomic_write_json(POSITIONS_STATE_FILE, ...)`
inside `_save_positions` is correct, but `_save_positions` has callers that
modify the position dict and call save without holding any lock.**
metals_loop has the singleton lock so only one process writes — OK. But the
embedded silver fast-tick monitor runs on its own thread and also reads
positions. The read can be torn from the perspective of the fast-tick if it
reads the dict mid-mutation. (Spot check needed; subagent rev-metals-core
should confirm.)

### P3 (style / clarity)

**P3-1: `dashboard/app.py:777` `/logout` has no `@require_auth`.**
- Intentional per docstring at L787: "an unauthenticated visitor hitting
  /logout still gets a useful clear-cookie response." Accept.

**P3-2: `portfolio/signal_engine.py` is 4315 lines.**
- Voting math, gates, accuracy, persistence, regime detection are all in one
  file. Logic is clearly commented and well-organized but file is approaching
  the "no one can hold it all in their head" threshold. Suggest extracting
  gate cascade (Stages 1-7) into `portfolio/signal_gates.py`. Not urgent.

**P3-3: 169 `except Exception:` occurrences across portfolio/.**
- Most are intentional fail-safes (e.g., heartbeat, telegram, optional metric).
- Counter-sample (`portfolio/agent_invocation.py:1095`): silently sets txn
  counts to 0 — flagged as P2-5 above.
- Recommendation: spot-audit per subsystem rather than blanket fix.

## Seam checklist (premortem hook #4)

### (a) shared_state.py consumers
`_tool_cache` mutated under `_cache_lock` (L22), but read by every signal
worker via `_cached()`. Lock is correctly held for all mutations.
`_loading_keys` + `_loading_timestamps` cleaned up in success, KeyboardInterrupt,
and generic Exception paths (L100-125). No leak path observed in fast scan.

### (b) Journal writers vs dashboard readers
- Writers use `atomic_append_jsonl` (sidecar locked).
- Dashboard endpoints use `load_jsonl_tail` (offset-based read of last 512KB).
- The tail reader does not take the sidecar lock — it can race with a writer
  mid-fsync. Worst case: the tail decode drops a partial last line (`json.loads`
  fails, line skipped). No state corruption, just one stale dashboard read.
  Accept as P3.

### (c) trigger_state producers vs autonomous consumers
- `portfolio/trigger.py:88` STATE_FILE → atomic_write_json.
- `portfolio/autonomous.py` consumes via `load_json` (default empty dict).
- No torn-state path observed.

### (d) portfolio_state.json — main loop, Layer 2 subprocess, dashboard
- Layer 2 subprocess shells out to its own Claude session, which may write
  transactions via record_trade. The parent loop reads via load_state →
  per-file lock. The subprocess runs in a separate process, so the in-process
  lock does not protect it.
- Mitigation: atomic_write_json (separate OS-level rename) means a read can
  never see a torn file, only an old-or-new file. Read-modify-write from the
  subprocess can lose a concurrent write from the parent (last-writer-wins),
  but in practice the parent only writes outside Layer 2 invocations.
- **Latent risk** if the parent ever decides to write portfolio_state during
  a Layer 2 run (e.g., a stop-loss fill detected on the parent side). Worth
  documenting the invariant in `portfolio_mgr.py`. P3.

### (e) GPU lock (Q:/models/gpu_lock.py)
Outside scope (separate venv). The subagent for orchestration should not
double-flag.

## Verification of subagent findings (running)
As each subagent report lands, I will re-grep its cited file:line + code
quotes against current source. Hallucinations get dropped at synthesis.

## What this independent pass did NOT cover
- Anything inside `portfolio/signals/*.py` (delegated to rev-signals-modules)
- Anything inside `portfolio/golddigger/`, `elongir/`, `mstr_loop/` (rev-trading-bots)
- Anything inside `data/metals_*` (rev-metals-core)
- Most of `data/*_loop.py` (rev-metals-core)
- Precomputes, prophecy, fin_evolve (rev-supporting)
- Most data fetcher modules (rev-data-external)
- 95% of dashboard endpoints (rev-infrastructure)

Each subagent owns its slice. This file owns the seams.
