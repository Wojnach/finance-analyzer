# Codex adversarial review: infrastructure
## Summary
I found 1 blocker, 5 high-severity issues, 4 medium issues, and 1 low issue. The biggest problem is a direct dashboard auth bypass on the LAN-facing listener. I did not find the requested fixture-cleanup bug in `tests/conftest.py:16-97`; those fixtures use `yield`/restore correctly.

## P0 — Blockers
- `dashboard/auth.py:125-135` trusts any request that presents both `Cf-Access-Authenticated-User-Email` and `Cf-Access-Jwt-Assertion`; it does not verify the JWT or restrict trust to a known proxy. `dashboard/app.py:2058-2092` simultaneously binds the dashboard to `[::]:5055` for direct IPv4/IPv6/LAN access. Any client that can hit Flask directly can spoof those headers, bypass local auth, and receive a 1-year `pf_dashboard_token` cookie via `_refresh_cookie()`.

## P1 — High
- `scripts/fix_agent_dispatcher.py:144-151` reimplements atomic state writes with a fixed `fix_agent_state.json.tmp`, `write_text()`, and no `fsync()`. Concurrent invocations can stomp the same temp file, and a crash/power loss can roll back `blocked_until` / `consecutive_failures`, violating the 30m → 2h → 12h backoff contract.
- `scripts/check_critical_errors.py:31-35,59-60,81-82` and `scripts/fix_agent_dispatcher.py:71-76,160,175-176,239-250` compare `datetime.fromisoformat(...)` results directly against UTC-aware cutoffs. A single naive `ts` or `blocked_until` raises `TypeError` instead of degrading gracefully, so one bad row can disable the startup check or the dispatcher.
- `portfolio/subprocess_utils.py:129-133,168-176` launches the child before job assignment and ignores `AssignProcessToJobObject`'s BOOL return. On a failed assignment or parent death in that window, the child escapes the job and survives parent termination, defeating the module’s advertised orphan-kill guarantee.
- `scripts/win/silver-monitor.bat:1-31`, `scripts/win/golddigger-loop.bat:1-24`, `scripts/win/golddigger.bat:1-10`, `scripts/win/mstr-loop.bat:1-7`, and `scripts/win/pf-local-llm-report.bat:1-25` are registered as scheduled-task entrypoints by `scripts/win/install-market-tasks.ps1:23-25,53-55`, `scripts/win/install-mstr-loop-task.ps1:21-23`, and `scripts/win/install-local-llm-report-task.ps1:30-32`, but none clear `CLAUDECODE`. That regresses the explicit post-Feb-18/19 operating rule for task-launched processes.
- `scripts/win/golddigger-loop.bat:5-24` and `scripts/win/golddigger.bat:5-10` restart unconditionally after every exit. Unlike `crypto-loop.bat` / `oil-loop.bat` / `metals-loop.bat`, they never stop on singleton-lock conflicts, so a schtasks restart that collides with a live/orphaned instance churns forever instead of exiting cleanly.

## P2 — Medium
- `portfolio/shadow_registry.py:93-104,118-126` does unlocked read-modify-write. `atomic_write_json()` only makes the replace atomic; it does not serialize writers. Two producers adding or resolving different shadows can overwrite each other and silently drop registry state.
- `portfolio/vector_memory.py:129-154,242-244,267-281` rereads the entire `layer2_journal.jsonl`, materializes all stored IDs with `collection.get()`, and never prunes either side. Memory and latency therefore grow without bound with journal size.
- `dashboard/house_blueprint.py:108-109,288-289,387-389` and `dashboard/app.py:898-914` bypass the canonical `file_utils` layer with raw `json.loads(...read_text())` and manual `open()` scans on live GET endpoints. That violates the repo I/O rule and reintroduces torn-read behavior; `api_run()` currently turns a partial `_manifest.json` read into a 500.
- `portfolio/file_utils.py:349-392` rewrites JSONL without `jsonl_sidecar_lock()`. Any append that lands between the read phase and `os.replace()` is lost, so the log-rotation race fixed in `atomic_append_jsonl()` can still recur via pruning.

## P3 — Low
- `dashboard/app.py:777-801` clears the auth cookie via GET `/logout`. With `SameSite=Lax`, a cross-site top-level navigation can log a user out without intent. Impact is low, but it is still state mutation on GET.

## Tests missing
- No auth test covers the `Cf-Access-*` bypass path or enforces trusted-proxy / JWT-validation behavior; current dashboard tests only exercise query-token, cookie, and Bearer flows.
- No test feeds naive timestamps into `scripts/check_critical_errors.py` or `scripts/fix_agent_dispatcher.py`.
- `tests/test_io_safety_sweep.py:52-104` only scans `portfolio/**/*.py`, which is why the raw JSON reads in `dashboard/` and `scripts/` escaped the sweep.
- No concurrency test covers `shadow_registry` read-modify-write races or `fix_agent_dispatcher._save_state()`'s fixed-temp-file behavior.
- No scheduled-task regression test checks that installed BAT wrappers clear `CLAUDECODE=` and stop on lock-conflict exits.
- No scale test exercises `vector_memory` with a large journal / large persisted collection.
- No concurrent-append test covers `prune_jsonl()` the way `atomic_append_jsonl()` is covered.