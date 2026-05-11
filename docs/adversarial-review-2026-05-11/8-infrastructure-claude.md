# Claude adversarial review: infrastructure

## Summary

Infrastructure surface is mostly solid — `atomic_write_json`, the sidecar-lock
`atomic_append_jsonl`, and the dashboard auth chain are well thought through.
But several real bugs hide in the seams: dashboard endpoints leak the
production loop's `config.json` through `/api/iskbets`, `kill_orphaned_by_cmdline`
uses `shell=False` WMIC with an untrusted-substring pattern, the
`fix_agent_dispatcher` has a state-corruption regression that fires when the
state file is malformed, `process_lock` writes the PID to the lock file but
never validates it against the current process on re-acquisition, and the
new `_read_tail_with_growth` has an unbounded-memory failure mode that can
OOM the dashboard. The 7-day rolling check in `check_critical_errors` does
the comparison naively against ISO strings, which works only as long as
journal writers stay disciplined. The `vector_memory` module ignores
embedding-dim mismatches and grows ChromaDB unboundedly.

## P0 — Blockers

- **dashboard/app.py:1049-1052 (`/api/iskbets`) — Why it bites:** This endpoint
  returns `_read_json(DATA_DIR / "iskbets_config.json")` to anyone holding
  the dashboard cookie. `iskbets_config.json` contains the iskbets section
  copied out of `config.json` — Binance/Alpaca keys are not in it today,
  but the project rule says "config.json (symlink to external file with API
  keys) must never log it" and this endpoint funnels arbitrary config-like
  JSON to clients with zero filtering. Compounding this: there is no
  redaction layer anywhere in `dashboard/app.py`. If iskbets_config ever
  picks up a `telegram` or `exchange` key — easy refactor mistake — the
  bot's full API keys leak to anyone with the cookie. **Fix:** allowlist
  fields explicitly: `return jsonify({"config": {k: v for k, v in (config
  or {}).items() if k in {"min_bid_sek", "max_bid_sek", "cooldown_s",
  ...}}, "state": state})`.

- **dashboard/app.py:101-102 (`_read_json` cache TTL on config.json) — Why
  it bites:** `_get_config()` calls `_read_json(CONFIG_PATH, ttl=60)` and
  caches the full config dict in `_cache` keyed `json:Q:\\...\\config.json`.
  Any future endpoint that returns its `_cache` keys, or anyone debugging
  via Flask shell access, has direct access to the full config including
  API keys. `dashboard/auth.py:60-66` keeps its own copy in `_CFG_VALUE`
  separately, again storing the full config dict in memory indefinitely
  (TTL only governs refresh). **Fix:** strip secrets at load time — read
  the file, hand only `dashboard_token`, `house_root`, `iskbets` (filtered)
  to the cache.

- **scripts/fix_agent_dispatcher.py:235-236 — Why it bites:** Kill switch
  check is `if KILL_SWITCH.exists(): return GateDecision(False, ...)`. The
  CLAUDE.md project rule states `touch data/fix_agent.disabled` disables
  auto-spawn entirely. But the check runs only at `check_gates()` call
  time, AFTER `_load_state()`. If `_load_state()` raises (corrupt JSON
  during a partial write) the corruption-handler at line 117-131 will
  call `_append_critical(...)` — which imports `portfolio.file_utils` and
  through that potentially pulls in modules with side effects. More
  critically: `KILL_SWITCH.exists()` is not checked in `run()` BEFORE the
  category loop — categories with a corrupt-state `blocked_until_global`
  block correctly, but a category whose state was already healthy will
  still proceed to `invoke_claude_fn(...)` because the gate is per-
  category. Edge case: kill switch created mid-run between two category
  iterations — second category still fires. **Fix:** check the kill switch
  at `run()` entry, before any work.

## P1 — High

- **portfolio/process_lock.py:83-104 — Why it bites:** `_write_lock_metadata`
  truncates the lock file and writes `pid=<getpid()>` plus owner/metadata.
  But there is no PID-reuse defense at acquire time. On Windows after a
  reboot, PIDs are reused aggressively. `acquire_lock_file` calls `_lock_file`
  which uses `msvcrt.locking(LK_NBLCK, 1)` — this succeeds if no LIVE
  process holds the lock. Correct. **But:** the metadata written into the
  lock file is never read back to verify the current process actually owns
  the lock. If a stale `.lock` file from a crashed run remains (because
  `release_lock_file` was bypassed by SIGKILL/power loss), the file lock
  on disk is released by the OS but the file *contents* still show the
  dead PID. Any operator reading the lock file to "see who has the loop"
  will see false data after a crash + restart. The `pf-loop.bat` exit-11
  branch suggests this is being relied on for orphan detection. **Fix:**
  on successful lock acquisition, re-truncate and re-write metadata
  unconditionally; or compare-and-set against the stored PID.

- **portfolio/subprocess_utils.py:214-220 — Why it bites:**
  `kill_orphaned_by_cmdline(pattern, ...)` interpolates `pattern` into a
  WMIC `where` clause: `f"CommandLine like '%{pattern}%'"`. If `pattern`
  contains a single quote (e.g., a caller passes a path containing one),
  WMIC parses it as a SQL-style WHERE clause and could match wider than
  intended — or, with carefully crafted input, match nothing and silently
  fail to kill orphans. Callers in this repo pass static strings, but
  there is no input validation and the function has no docstring caveat.
  The follow-on `subprocess.run([...wmic..., ...where..., ...])` is
  parameterized as a list so shell injection is blocked at the OS level,
  but WMIC's own filter language is the attack surface. **Fix:** validate
  `pattern` matches `^[A-Za-z0-9_.\\-]+$` or escape single quotes.

- **portfolio/subprocess_utils.py:267-284 — Why it bites:**
  `kill_orphaned_llama` builds a PowerShell command with `shell=True` and
  inline embedded quotes (`\\\"Name='llama-completion.exe'\\\"`). The
  process name is hardcoded so injection is moot, but the
  `subprocess.run(ps_cmd, ..., shell=True)` pattern means *any future
  refactor* that lets a caller pass a process name reopens shell-injection.
  Also: `kernel32.TerminateProcess(h_proc, 1)` doesn't check return value,
  so a permission failure (running as non-admin) is silent.

- **scripts/fix_agent_dispatcher.py:144-151 — Why it bites:** `_save_state`
  writes the temp file with `tmp.write_text(json.dumps(state, indent=2),
  encoding="utf-8")` and then `os.replace(tmp, STATE_FILE)`. There is no
  `f.flush() + os.fsync()` between the write and the rename, unlike
  `file_utils.atomic_write_json`. On power loss between write and replace,
  the new `.tmp` file can be replaced into place but contain partial
  buffered bytes — exactly the corruption case `_load_state()` is trying
  to defend against. **Fix:** use `file_utils.atomic_write_json` directly.

- **portfolio/file_utils.py:202-258 (`jsonl_sidecar_lock`) — Why it bites:**
  The lock file is `path.parent / f".{path.name}.lock"`. If two processes
  invoke `atomic_append_jsonl` against the same logical file via different
  resolved paths (e.g. through a Windows junction, or the worktree symlink
  noted at `Q:/finance-analyzer/.worktrees/`) the sidecar locks are
  DIFFERENT files and the mutual exclusion is bypassed. The rotation race
  the docstring claims to fix re-emerges silently. **Fix:** resolve
  `path` via `Path.resolve()` before deriving the lock path.

- **dashboard/app.py:133-168 (`_read_tail_with_growth`) — Why it bites:**
  On a misbehaving log (e.g. one with no `\n` characters at all, or one
  line larger than `tail_bytes`) the loop doubles `tail_bytes` up to
  64 MB, but the `len(rows) < limit` retry condition fires when
  `_load_jsonl_tail_impl` can't parse rows. If a single 80 MB line exists
  it'll consume 64 MB of RAM, then fall through to `_load_jsonl_impl`
  which reads the WHOLE file (potentially gigabytes) into a deque. Under
  a flood (multiple dashboard tabs polling `/api/telegrams?limit=5000`)
  Flask workers will OOM. **Fix:** cap final `_load_jsonl_impl` fallback
  on `Path(path).stat().st_size <= 128*1024*1024` or stream.

- **dashboard/auth.py:131-135 (Cloudflare header trust) — Why it bites:**
  Trust chain accepts any client that sends BOTH `Cf-Access-Authenticated-
  User-Email` AND `Cf-Access-Jwt-Assertion` headers — *without verifying
  the JWT signature*. The comment says CF "strips inbound Cf-Access-*
  headers at its edge", but that's only true if the deployment is gated
  by CF Access. If the dashboard is ever directly exposed (port-forwarded,
  ngrok tunnel for debugging, future LAN access enablement) any client
  can forge both headers and bypass auth completely. The bearer token is
  cryptographically verified via `hmac.compare_digest`; the CF header
  path is not. **Fix:** verify `Cf-Access-Jwt-Assertion` against CF's
  JWKS endpoint, or at minimum gate on `request.headers.get(
  "Cdn-Loop", "").startswith("cloudflare")` as a secondary check.

- **dashboard/app.py:1067-1088 (`/api/validate-portfolio`) — Why it bites:**
  `request.get_json(silent=True)` with no size limit. Flask default
  `MAX_CONTENT_LENGTH` is unset → unbounded request body. A malicious
  cookie holder can POST a 1 GB JSON object and OOM the worker.

## P2 — Medium

- **scripts/check_critical_errors.py:81 — Why it bites:** `parsed < cutoff`
  uses an aware datetime from `_parse_ts(ts)`. If an entry's ts string is
  naive (missing tz info, e.g. a future writer forgets `+00:00`),
  `datetime.fromisoformat` returns a naive datetime and `parsed < cutoff`
  raises `TypeError: can't compare offset-naive and offset-aware
  datetimes` — which is swallowed by the surrounding `_load_entries`
  exception handler... no, actually it's NOT swallowed; line 81 is in
  `find_unresolved` with no try/except, so the whole script crashes
  printing a stack trace and exits non-zero with no output the user can
  act on. **Fix:** force tz-aware on parse: `if parsed.tzinfo is None:
  parsed = parsed.replace(tzinfo=UTC)`.

- **portfolio/shared_state.py:88 — Why it bites:** Dogpile path: when
  `key in _loading_keys` and no stale data available, returns `None`
  inside the lock. Callers that don't handle None correctly (e.g.
  signal modules expecting a dict) will crash. The "return None rather
  than pile on" docstring is correct in principle but the contract is
  silent — callers across the codebase need to understand that any
  `_cached()` call can return None on a cold cache during contention.

- **portfolio/shadow_registry.py:74-78 — Why it bites:** `save_registry`
  calls `atomic_write_json` with the full dict. `add_shadow`, `resolve_
  shadow`, `seed_defaults` all do read-modify-write without locking.
  Two concurrent producers (e.g. `seed_defaults()` called from main
  loop startup race with a `resolve_shadow()` from a script) → last
  write wins, the other's update is silently lost.

- **portfolio/vector_memory.py:46-54 — Why it bites:** ChromaDB
  collection is created with default embedding (`all-MiniLM-L6-v2`, 384
  dim). If the user ever swaps embedding models, the existing collection
  schema mismatches and `collection.add` raises. There is no embedding-
  dim validation. Worse: there is no bound on collection size — every
  journal entry ever written gets embedded forever, the chromadb dir
  grows unboundedly, and `embed_entries` re-reads the entire `existing`
  ID set on every call (line 130-133, `collection.get()` loads ALL ids
  into memory).

- **portfolio/message_store.py:46-49 (mojibake replacements) — Why it
  bites:** Several dict keys are identical (`"â"` appears 5+ times). In
  a Python dict literal, duplicate keys silently drop earlier values —
  only the LAST mapping survives. So only the final `"â": ""` actually
  runs, and the intended em-dash / right-quote / arrow replacements are
  dead code. Mojibake repair is half-broken.

- **dashboard/app.py:894-918 (`/api/mstr_loop`) — Why it bites:** Raw
  `open(poll_path, encoding="utf-8")` then `_json.loads(line)` — exactly
  the pattern CLAUDE.md says to avoid. The `last_poll` loop reads every
  line, parses each, and only keeps the last. On a 100 MB jsonl this is
  an O(N) scan per request — pair with the 5 s `_DEFAULT_TTL` cache and
  this is fine, but the I/O pattern is wrong on principle.

- **dashboard/house_blueprint.py:108-110 — Why it bites:** `slugs =
  json.loads(manifest.read_text())` — same raw read pattern. No
  encoding specified, so on Windows it picks up cp1252 and trips on
  non-ASCII slugs.

- **scripts/win/adversarial-review.bat:5-7 — Why it bites:** Uses
  `set CLAUDECODE=` correctly. But it appends to `data\adversarial_
  review_out.txt` with `2>&1`, and `claude -p` may print "Not logged
  in" or other auth-failure output that gets silently appended. No
  exit-code check — the user can't tell if the run succeeded except
  by reading the file. The Feb 18-19 outage pattern.

- **dashboard/app.py:2080-2084 (dual-stack bind) — Why it bites:**
  `sock.bind(("::", port))` with `IPV6_V6ONLY=0`. On hosts where
  another process already holds `::1:5055` (e.g. an IDE preview server)
  the bind fails with EADDRINUSE — `SO_REUSEADDR` set, but that
  flag's semantics on Windows let TWO processes accept on the same
  port and load-balance randomly. Requests sometimes hit a stranger.

## P3 — Low

- **portfolio/file_utils.py:79-80** — bare except `OSError` swallows
  permission errors. Returns default. The H35 comment says corruption
  should be observable; permission-denied isn't logged at WARNING.

- **portfolio/subprocess_utils.py:182-195** — `close_job` swallows
  `Exception`, masking double-close bugs.

- **dashboard/app.py:1075-1077** — `request.get_json(silent=True)` returns
  None on parse error → `if not data` triggers a 400. Empty `{}` body also
  trips this — legitimate empty validation requests fail with the same
  error.

- **portfolio/shared_state.py:265-280** — `_RateLimiter.wait` reserves
  `last_call = last_call + interval` even when the lock contention is
  high; this can drift `last_call` arbitrarily into the future,
  starving later callers indefinitely under sustained load.

- **scripts/win/metals-loop.bat:11** — no `START /B /WAIT` like
  pf-loop.bat uses. A Ctrl+C on the parent kills the python child but
  the bat-script restart loop is also killed. Asymmetric with main
  loop.

- **portfolio/notification_text.py:13** — `humanize_ticker(None)`
  returns `""`. Callers that branch on truthiness fall through silently.

## Tests missing

- `process_lock` PID-reuse: simulate stale lock + dead PID, ensure
  re-acquisition writes fresh metadata.
- `atomic_append_jsonl` with symlinked path: assert both writers
  resolve to the same sidecar lock.
- `_read_tail_with_growth` OOM defense: huge single-line jsonl file
  must not load whole file.
- `kill_orphaned_by_cmdline` injection: pattern containing single
  quote should be rejected or escaped.
- Cloudflare header spoof: request with forged CF headers (no actual
  CF JWT) should return 401.
- `fix_agent_dispatcher` kill-switch race: touching
  `fix_agent.disabled` between categories must stop subsequent
  invocations.
- `message_store._COMMON_MOJIBAKE_REPLACEMENTS` duplicate keys —
  assert all intended replacements are reachable.
- `check_critical_errors` with naive-tz entry — assert it doesn't crash.

## Cross-cut observations

The single largest infrastructure risk is **auth header trust** — the
Cloudflare bypass at `dashboard/auth.py:131-135` is the right idea but
the implementation accepts forged headers because it doesn't verify the
JWT signature. The second-largest is **config leakage via dashboard
endpoints** — `/api/iskbets` returns config-like JSON unfiltered, and
the in-memory caches in both `dashboard/app.py` and `dashboard/auth.py`
hold the full config dict containing API keys. Both are reachable by
anyone with the dashboard cookie, and Cloudflare Access policy is the
only thing standing between a stolen cookie and the API key set.

`file_utils.py` is the canonical I/O layer and most of the codebase uses
it correctly, but `scripts/fix_agent_dispatcher.py:144-151` bypasses it
for state writes, and `dashboard/app.py:898-916` plus
`dashboard/house_blueprint.py:108` use raw open+json.loads. Those
exceptions should be migrated.

The `pf-loop.bat` / `metals-loop.bat` CLAUDECODE clearing is correct —
the 34h Feb 18-19 outage is not a current risk for these wrappers. But
`scripts/win/adversarial-review.bat` exists and follows the same
pattern, so the discipline is holding.
