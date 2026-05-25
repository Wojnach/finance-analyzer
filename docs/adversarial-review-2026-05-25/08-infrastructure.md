# Infrastructure Adversarial Review

Worktree: `Q:/finance-analyzer-worktrees/review-infrastructure` (read-only).
Reviewer: claude (opus 4.7 1M).
Scope: file_utils, journal, journal_index, health, logging_config, log_rotation, gpu_gate, process_lock, message_store, message_throttle, telegram_notifications, telegram_poller, alert_budget, subprocess_utils, dashboard/app, dashboard/auth, dashboard/cf_access, dashboard/export_static, dashboard/house_blueprint, dashboard/system_status, dashboard/trading_status.

The hot paths in this subsystem have been hardened over the last six months — the sidecar-lock contract for JSONL append+rotate is solid, CF-Access JWT verification was added in May, dual-stack bind is explicit, every API route I enumerated (44 routes in `dashboard/app.py` + 8 in `house_blueprint.py`) has `@require_auth`, and Telegram bot tokens are redacted by `http_retry._redact_url` before any log line. What's left are smaller correctness/availability bugs and a handful of design-time risks worth surfacing.

## P0 findings

(none with confidence ≥ 91 — no live auth bypass, no plaintext-secret leak, no torn-write on the canonical append path)

## P1 findings

### 1. `atomic_append_jsonl` does not `O_APPEND` — interleaving risk on Windows if the sidecar lock ever fails to acquire

`portfolio/file_utils.py:269-292` — `atomic_append_jsonl` opens with `"ab"` (append mode at the libc level) then writes/flushes/fsyncs inside the sidecar lock. `open("ab")` on CPython/Windows does **not** set `O_APPEND` semantics in a kernel-atomic way (libc seeks-to-end on write); correctness depends entirely on the sidecar lock serialising writers. If `jsonl_sidecar_lock` ever fails to acquire (the path at `file_utils.py:240-247` — sidecar create raises OSError → logged but `yield` still runs without a lock held), two concurrent appenders can interleave inside a single line, producing a torn JSONL row.

The fallback path is explicit:

```
if not lock_path.exists():
    try:
        with open(lock_path, "ab") as lf:
            ...
    except OSError as exc:
        logger.warning("sidecar lock creation failed for %s: %s", path, exc)

with open(lock_path, "rb+") as lock_f:   # this will also raise if lock_path absent
```

…but if the first `open("ab")` for the sidecar succeeds and the *second* `open("rb+")` fails (race with another process unlinking the sidecar, AV scan, etc.), the contextmanager re-raises and `yield` never runs — so the append never happens, the entry is lost, and the caller sees the OSError. Caller handling for atomic_append_jsonl is rare: `journal.py`, `message_store.log_message`, `telegram_poller._log_inbound` all let the exception bubble up. Fix: catch OSError on lock acquire, log critical, and fall back to a process-local `threading.Lock` so the write still happens (durability > strict ordering for telemetry).

### 2. `last_jsonl_entry` / `load_jsonl` / `load_jsonl_tail` do not take the sidecar lock — readers can observe rotation mid-flight

`portfolio/file_utils.py:112-207, 316-354`. `rotate_jsonl` (`log_rotation.py:240-367`) does:

1. acquire lock
2. read full file
3. write keep-lines to tmp
4. `os.replace(tmp, filepath)`
5. release lock

On Windows, `os.replace` of an open file fails (`PermissionError`). A reader (`load_jsonl`, dashboard `_read_jsonl`) that opens the source between step 2 and step 4 will hold an exclusive read handle long enough to make step 4 raise `OSError`, which `rotate_jsonl` propagates and the caller (`rotate_all`) records as `"status": "error"` for that file. The lock file the writer holds doesn't help — readers don't take it.

Result: every dashboard poll while rotation is in flight has a small chance of breaking the rotation pass. The 2026-05-11 silver invariant escalations noted in the docstring may have a sibling: rotation periodically failing for `signal_log.jsonl` (68MB) when dashboard polling is hot. Fix: have heavy readers (rotation+truncation candidates) acquire the sidecar lock in shared/read mode. Lightweight readers can keep the existing best-effort path.

### 3. `process_lock` does not validate that the holder PID is actually the expected executable

`portfolio/process_lock.py`. `acquire_lock_file` does a `flock`/`msvcrt.locking` for non-blocking exclusive access — if held, returns `None`. But the metadata written into the lock file (`pid=…`, `started=…`) is informational only; no caller validates that the PID belongs to the right process. Combined with PID reuse on Windows (PIDs are recycled quickly), this means stale-lock detection elsewhere in the codebase (`gpu_gate`, plus any caller that reads the metadata to decide whether to break the lock) can be fooled by an unrelated process that inherited the same PID. Recommend including `executable_name` + `cmdline_hash` in the metadata and validating in stale-break paths.

### 4. `gpu_gate._pid_alive` returns `False` when psutil is missing — risk of breaking a valid GPU lock

`portfolio/gpu_gate.py:73-83`. The comment ("assuming PID %d is dead to prevent stale GPU lock deadlock") acknowledges the trade-off but the consequence is severe: a Chronos forecast that's legitimately running for 200s on the GPU can have its file lock reaped by a second process if psutil isn't installed there. Both processes then load models simultaneously, exceed VRAM, and one OOMs (or worse — corrupted model loads). Fix: require psutil in the LLM venv's requirements (`Q:/models/.venv-llm`) and FAIL CLOSED (`return True` = "assume alive") if it's missing, matching defensive defaults elsewhere.

### 5. `gpu_gate._try_break_stale_lock` is vulnerable to PID reuse

`portfolio/gpu_gate.py:103-134`. The predicate is `mtime > 300s AND _pid_alive(owner_pid) == False`. If the original owning process (e.g. chronos pid 13152, per the comment) died and Windows recycled PID 13152 to an unrelated process (Steam, browser tab, antivirus), `_pid_alive(13152) == True` → lock NOT broken → the loop wedges forever again. The 2026-05-02 25-hour wedge described in the docstring would repeat after a PID-recycle collision. Mitigation: persist `executable_name` in the lock file metadata and confirm match (via psutil's `Process(pid).name()`) before honoring an "alive" verdict.

### 6. `health.update_health` rewrites the entire `health_state.json` blob inside a lock, but `get_health_summary` (dashboard) reads via `load_json` with no lock

`portfolio/health.py:20-41, 338-368`. The atomic-write-via-replace makes torn-read impossible on POSIX, and on Windows `os.replace` of an open-file target fails noisily — meaning the dashboard's `load_json` can intermittently return `default={}` (because `load_json` swallows `OSError` and returns the default with a WARNING). `get_health_summary` then shows `cycle_count=0`, `error_count=0`, `signals_ok=0`, `agent_silent=True` — the dashboard flips to "RED" for a few seconds whenever update_health races a dashboard poll. The 5s TTL cache reduces the impact but doesn't eliminate it. Fix: retry the read once after a short sleep when `load_json` falls back to default.

### 7. `telegram_poller` persists offset on `unrecognized` drops — user typos can advance past unprocessed legitimate messages

`portfolio/telegram_poller.py:151-155, 261-265`. `_SETTLED_DROP_REASONS` includes `"unrecognized"` so any message whose first word isn't `bought|sold|cancel|status|/mode` advances the persisted offset. Picture: user is typing a multi-line trade confirmation; the first short message ("hey claude") is unrecognized and acks; the second message ("bought MSTR 5 @ 312") arrives after a restart. Telegram's getUpdates filters by offset > last_acked, but if the persistence path raced ahead, the real command can still be processed — so this isn't strictly a lost-message bug. The real concern: anyone in the chat (this is gated to `self.chat_id` so probably just the operator) typing miscellaneous text contributes to offset advancement, and the persisted state is therefore tied to the operator's typing cadence rather than to dispatchable work. Cosmetic but operationally surprising — recommend separating "ack offset" from "settled action" so the persisted file only advances on processed commands.

### 8. `house_blueprint.api_run` raises `abort(500)` on JSON decode failure — leaks internal error path

`dashboard/house_blueprint.py:387-398`. If `_manifest.json` is malformed, `api_run` calls `abort(500)` which renders Flask's default 500 page (HTML, includes stack trace if `DEBUG=True`). The other routes use `abort(404)`. Replace with a JSON error response. Minor — but it's the only divergence from the rest of the house blueprint's error-handling discipline.

### 9. `export_static` writes API JSON into `dashboard/static/api-data/` which is auto-served WITHOUT auth

`dashboard/export_static.py`. The module is marked DEPRECATED but is still importable. Anyone who runs `python -m dashboard.export_static` (or imports it for tests and triggers `export_all()`) writes 20 files under `dashboard/static/api-data/` — Flask's `static_folder` serves the entire `dashboard/static/` tree with NO `@require_auth`. The docstring warns about this and `.gitignore` excludes the directory, but the failure mode is one stray `make export` or test run away. Recommend deleting `export_static.py` entirely (or moving the write target to `dashboard/_export/` outside the static handler) and removing the call site in `tests/test_dashboard_export_static.py` if it still exists.

### 10. `dashboard/auth.require_auth` redundantly refreshes the cookie even after CF-Access authentication

`dashboard/auth.py:152-153`. When CF Access has just JWT-verified a request, the wrapper re-sets `pf_dashboard_token=<expected>` with `httponly + secure + samesite=Lax`. If a user authenticates via CF Access from a fresh device, they then have BOTH a CF-Access session cookie AND a `pf_dashboard_token` cookie whose value is the literal config secret. Once that cookie is in the browser, the dashboard accepts it bypassing CF Access (so the operator-only Cloudflare Access policy gives way to a 1-year cookie). If the CF Access scope ever shrinks (e.g. operator's email removed from the Access app) but the cookie hasn't expired, that account still has dashboard access for up to 365 days. Recommend: only set the cookie on token-/bearer-auth success, NOT on CF-Access. (At minimum, document.)

### 11. `_read_config_uncached` caches an empty `{}` if config.json is unreadable; `_get_config` then never refreshes

`dashboard/auth.py:60-80`. On a transient read failure, `_read_config_uncached` returns `{}`, but `_get_config` only overwrites `_CFG_VALUE` if `fresh` is truthy or `_CFG_VALUE is None`. So an empty result is only cached at cold start. If config.json is briefly unreadable on first dashboard request after process start (symlink to external file + AV scan, per the BUG-210 note), `_get_config` caches `{}` for 60s — and during that window `_get_dashboard_token()` returns `None`, which makes `require_auth` ALLOW ALL REQUESTS (the "no token configured = open access" fallback at `auth.py:124-125`). Window is 60 seconds at process startup. Fix: on cold-start failure, leave `_CFG_VALUE = None` (force re-read on next request) AND fail closed if token can't be determined.

## P2 findings

### 12. `subprocess_utils.kill_orphaned_llama` uses `shell=True` with hardcoded args

`portfolio/subprocess_utils.py:278-291`. The command string is a literal — no user input — so it's not currently exploitable. But the pattern is a footgun if the function is ever extended. Use the list form (`["powershell.exe", "-NoProfile", "-Command", ...]`) with `shell=False`.

### 13. `dashboard/app.add_cors_headers` sets `Access-Control-Allow-Credentials: false` but the auth model is cookie-based

`dashboard/app.py:52-65`. Cookies aren't sent on cross-origin requests when `Credentials: false`, so the whitelisted dev origins (localhost:3000) can't authenticate against the dashboard via cookie at all — only via `?token=` URL params or Bearer header (which exposes the token in browser history / referer leaks). For a dev environment this is mostly fine, but consider documenting it so contributors don't waste an hour debugging "why doesn't auth work from my React dev server."

### 14. `cf_access._JWKS_CLIENT_CACHE` lookup outside the lock, then verified inside — TOCTOU is benign but `PyJWKClient` may be created twice on cold start

`dashboard/cf_access.py:67-96`. The cache check at line 73-76 takes the lock, fine. But after a cache miss the network fetch at line 78-91 runs OUTSIDE the lock, and only THEN reacquires the lock to insert. Two concurrent cold-start requests both fetch JWKs and write the cache. The duplicate fetch is harmless (~50ms each) and resolves immediately. Cosmetic.

### 15. `log_rotation.rotate_jsonl` swallows malformed lines into `keep_lines` — they accumulate forever

`portfolio/log_rotation.py:296-299`. A line that fails `json.loads` is kept "to avoid data loss." Reasonable, BUT subsequent rotations will keep re-processing it and re-keeping it. If the file gets corrupted (e.g. one partial line from a non-locked writer), that line will exist forever in production. A "quarantine" file (`<name>.corrupt.jsonl`) for lines that fail to parse N times would be safer.

### 16. `health.check_outcome_staleness` iterates outcomes with `for t in outcomes:` then re-iterates `for t_outcomes in outcomes.values():` — redundant + bug

`portfolio/health.py:389-406`. The `has_any` block iterates over keys to check 1d outcomes. Then the `if has_any:` body re-iterates `outcomes.values()`. The two iterations could be one — but more importantly, the first block's `has_any = any(... for t in outcomes)` is True for ANY ticker with a 1d outcome, but the iteration that follows runs over ALL tickers including ones without outcomes. `h_data.get("ts")` returns `None` for non-dict values, and `isinstance(h_data, dict)` filters the None case. So no crash, but the intent is unclear and one tick of accidental complexity.

### 17. `dashboard.app._iter_latest_dict_entries` reads `read_limit` entries but downstream filter loops only consume `limit` — large read_limit values amplify TTL-cache memory

`dashboard/app.py:189-194, 909-933`. `/api/telegrams` calls `_iter_latest_dict_entries(..., read_limit=5000)`. The cache key is `jsonl_tail_v2:{path}:5000`. Each cached entry holds ~5000 dicts × ~500 bytes each = ~2.5 MB. Total cache footprint across endpoints can hit ~50 MB. Not critical but worth a memory cap on `_cache`.

### 18. `dashboard/app._serve_dual_stack` binds with `SO_REUSEADDR` but not `SO_EXCLUSIVEADDRUSE` on Windows

`dashboard/app.py:2337-2341`. On Windows, `SO_REUSEADDR` allows another process to bind to the SAME port (rather than POSIX semantics of reusing TIME_WAIT). If a malicious process on the same host binds 5055 first, it can hijack connections. Real risk requires local admin to install a malicious service, so impact is limited. Use `SO_EXCLUSIVEADDRUSE` on Windows to get POSIX-like exclusive bind semantics.

### 19. `journal_index` loads the ENTIRE `layer2_journal.jsonl` on every `retrieve_relevant_entries` call

`portfolio/journal_index.py:367-399`. The function is invoked by `journal.write_context()`, which runs every Layer 2 invocation. As the journal grows, BM25 index rebuild grows O(N) per invocation. At 60 invocations/day × 1000 entries = 60K parse/tokenize ops per call. Rotation caps at 60d but a healthy week's journal already approaches 1000 entries. Recommend caching the index by file mtime — if mtime hasn't changed, reuse the index.

### 20. `dashboard.app._normalize_metals_decisions` and `_build_metals_context_fallback` swallow keys without validation

`dashboard/app.py:425-668`. These functions accept arbitrary nested dicts from disk and project them onto the dashboard schema. A malformed `metals_decisions.jsonl` entry where `positions["MINI-SILVER"]` is a string instead of a dict would crash `dict(payload)`. The outer try/excepts mostly hide this. Recommend explicit `isinstance(... , dict)` guards at each level.

### 21. `dashboard.app.api_avanza_account._AVANZA_REQ_TIMEOUT_SECONDS = 25` but the cookie cache TTL is 30s — request can outlive the cache

`dashboard/app.py:1932`. Minor coordination issue: a request that takes 25s and writes to `_AVANZA_CACHE` will set `at=time.monotonic()` AFTER the snapshot, but a concurrent request that started 26s ago would have already considered it "fresh." Race window is ~5s — operationally benign.

### 22. `house_blueprint` returns `_render_markdown(text)` unsanitized — markdown library output may contain raw HTML

`dashboard/house_blueprint.py:211-216`. `markdown.markdown(..., extensions=[...])` does NOT sanitize HTML in the input by default — any `<script>` tag inside a `_summary.md` file would render as JS. The blueprint is gated by `require_auth` and the markdown files are sourced from a trusted user-owned directory (`Q:\househunting`), but the trust boundary is fragile. If `findapartments_scan` ever scrapes attacker-controlled HTML and writes it into a thesis file, the operator's browser will execute it as same-origin to the dashboard cookie. Mitigation: enable `markdown.Extension` with the `markdown-strict` or `bleach` post-filter.

## P3 findings

### 23. `portfolio/logging_config.py:43-47` — RotatingFileHandler is not thread-safe across processes

Two processes both opening `data/portfolio.log` will race on rotation (each thinks it owns the rename). Not currently an issue because only the main loop logs there, but if dashboard/golddigger ever import `setup_logging`, they'll fight over rotation. Use `ConcurrentRotatingFileHandler` (`concurrent-log-handler` package) if multi-process logging is ever needed.

### 24. `portfolio/health.update_signal_health_batch` writes the whole health_state.json on every batch — chatty

Each Layer 1 cycle calls it once with all 15 active signals — but `signal_health` accumulates across many cycles and grows the JSON blob. After 6 months: ~50K calls × per-signal rolling window. The file is currently ~30KB. Recommend periodic prune (drop signals with 0 calls in last 30 days).

### 25. `message_store._normalize_message_whitespace` strips trailing whitespace from non-code lines — could break intentional Markdown line-break syntax (two trailing spaces = `<br>`)

`portfolio/message_store.py:62-71`. Markdown v1 uses two trailing spaces for a hard break. The normalization at `.strip()` removes that. Telegram's MarkdownV1 doesn't honor the trailing-space convention anyway (uses `\n`), so this is currently safe. Note only.

### 26. `dashboard.app.api_validate_portfolio` returns 500 with `f"Validation error: {e}"`

`dashboard/app.py:1063`. Stack-leak via exception message. Operator-only via require_auth, so impact is limited; still recommend logging the exception server-side and returning a generic message client-side.

### 27. `gpu_gate._sweeper_loop` swallows all exceptions at debug level

`portfolio/gpu_gate.py:137-152`. A genuine bug in the sweeper would be invisible. Bump the inner `except` to `logger.warning` so silent breakage is observable.

### 28. `process_lock._write_lock_metadata` silently truncates the file then writes — torn read possible

`portfolio/process_lock.py:101-107`. `seek(0) + truncate() + write()` is not atomic; a reader between truncate and write sees an empty lock file. The lock semantics (msvcrt range lock) still hold, but `_read_lock`-style consumers can intermittently see `{}`. Fix: write to tmp + replace, or write fixed-width padded.

## Cross-cutting observations

* **Read-side lock discipline is the weakest link.** The append/rotate write path takes `jsonl_sidecar_lock`; readers don't. For an append-only telemetry log this is fine, but for files that get rewritten (rotation, `prune_jsonl`, `atomic_write_jsonl`), readers can observe a rename-in-flight on Windows and force the writer to fail.
* **`load_json` returning `default={}` on OSError is load-bearing for graceful degradation but can mask the absence of a real config.** The `require_auth` open-allow when `dashboard_token is None` (`auth.py:124-125`) is the most dangerous instance — combined with `_get_config()` caching `{}` for 60s on cold-start read failure, there's a real "60s open dashboard at process start" window.
* **Stale-lock detection across the codebase keys off PID liveness, not PID identity.** Windows PID reuse can defeat every "is the holder still alive?" check (`gpu_gate`, any future `process_lock` stale-break path). Add executable-name validation.
* **The pickup handler whitelist (`scripts/process_pending_pickups._HANDLERS`) is the right pattern** — explicit guard against CWE-706 from a tampered `data/pending_pickups.json`. Worth highlighting as the example for any future dynamic-dispatch system in the codebase.
* **Telegram bot token redaction is centralised in `http_retry._redact_url`** — solid. The only place a token could still leak is `r.text[:200]` in `telegram_poller._send_reply` (line 385), but Telegram's API doesn't echo the token in error responses.
* **CF Access JWT verification is correctly implemented** (signature, exp/iat/aud requires, email-header-must-match-claim). The fail-closed posture matches the security-critical role.
* **Dual-stack bind is correct** (`IPV6_V6ONLY=0`, manual socket setup, `ThreadedWSGIServer(fd=…)`). The only missing piece is `SO_EXCLUSIVEADDRUSE` on Windows.
* **All 52 enumerated dashboard endpoints have `@require_auth`** (44 in app.py + 8 in house_blueprint.py). `/logout` is intentionally open (P2 docstring covers the rationale). The grep `@app.route` followed by `@require_auth` decorator-on-next-line scan finds zero gaps.

## Files reviewed

* `Q:/finance-analyzer-worktrees/review-infrastructure/portfolio/file_utils.py`
* `Q:/finance-analyzer-worktrees/review-infrastructure/portfolio/journal.py`
* `Q:/finance-analyzer-worktrees/review-infrastructure/portfolio/journal_index.py`
* `Q:/finance-analyzer-worktrees/review-infrastructure/portfolio/health.py`
* `Q:/finance-analyzer-worktrees/review-infrastructure/portfolio/logging_config.py`
* `Q:/finance-analyzer-worktrees/review-infrastructure/portfolio/log_rotation.py`
* `Q:/finance-analyzer-worktrees/review-infrastructure/portfolio/gpu_gate.py`
* `Q:/finance-analyzer-worktrees/review-infrastructure/portfolio/process_lock.py`
* `Q:/finance-analyzer-worktrees/review-infrastructure/portfolio/message_store.py`
* `Q:/finance-analyzer-worktrees/review-infrastructure/portfolio/message_throttle.py`
* `Q:/finance-analyzer-worktrees/review-infrastructure/portfolio/telegram_notifications.py`
* `Q:/finance-analyzer-worktrees/review-infrastructure/portfolio/telegram_poller.py`
* `Q:/finance-analyzer-worktrees/review-infrastructure/portfolio/alert_budget.py`
* `Q:/finance-analyzer-worktrees/review-infrastructure/portfolio/subprocess_utils.py`
* `Q:/finance-analyzer-worktrees/review-infrastructure/portfolio/http_retry.py` (corroborating — token-redaction regex)
* `Q:/finance-analyzer-worktrees/review-infrastructure/scripts/process_pending_pickups.py` (corroborating — handler whitelist)
* `Q:/finance-analyzer-worktrees/review-infrastructure/dashboard/app.py`
* `Q:/finance-analyzer-worktrees/review-infrastructure/dashboard/auth.py`
* `Q:/finance-analyzer-worktrees/review-infrastructure/dashboard/cf_access.py`
* `Q:/finance-analyzer-worktrees/review-infrastructure/dashboard/export_static.py`
* `Q:/finance-analyzer-worktrees/review-infrastructure/dashboard/house_blueprint.py`
* `Q:/finance-analyzer-worktrees/review-infrastructure/dashboard/system_status.py`
* `Q:/finance-analyzer-worktrees/review-infrastructure/dashboard/trading_status.py`
