# Infrastructure Adversarial Review — 2026-05-17

Scope per spec: atomic I/O, locks, telegram, journals, dashboard auth/endpoints,
config validator, subprocess utils, shared state, message store, process lock.

## Files audited (code only — no data/*.json read)

- `portfolio/file_utils.py`
- `portfolio/health.py`
- `portfolio/shared_state.py`
- `portfolio/process_lock.py`
- `portfolio/subprocess_utils.py`
- `portfolio/api_utils.py`
- `portfolio/config_validator.py`
- `portfolio/telegram_notifications.py`
- `portfolio/telegram_poller.py`
- `portfolio/message_store.py`
- `portfolio/journal.py`
- `dashboard/app.py` (2246 lines, ~all routes + helpers)
- `dashboard/auth.py`
- `dashboard/cf_access.py`
- `dashboard/system_status.py`
- `dashboard/trading_status.py`
- `dashboard/house_blueprint.py`
- `dashboard/export_static.py`

Routes accounted for: 41 `@app.route` + 9 `@bp.route`. Every API/HTML route is
either wrapped in `@require_auth` or intentionally bare (`/logout`,
`/static/*`). No auth-bypass holes found.

---

## P1 (must fix)

### P1-1 — `atomic_write_json` on the `config.json` symlink replaces the symlink, not the target
**File:** `portfolio/file_utils.py:45-63`, attack surface via
`portfolio/telegram_poller.py:312-369` (`_handle_mode_command`) and
`portfolio/iskbets.py:61-63` (`_save_config` on a different file, only relevant
if it were ever pointed at config.json).

```python
# file_utils.py:53-59
fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str, ensure_ascii=ensure_ascii)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, str(path))
```

`os.replace(tmp, str(path))` on Windows calls `MoveFileExW(... ,
MOVEFILE_REPLACE_EXISTING)`. Per the Win32 docs (and confirmed CPython
behavior), when the destination is a symbolic link the call replaces the
**symlink itself**, not the link's target. CLAUDE.md is explicit:

> Config: Symlink `config.json` → `C:\Users\Herc2\.config\finance-analyzer\config.json` (OUTSIDE repo). NEVER commit config.json — exposed API keys on Mar 15, 2026.

Trigger path:
1. User sends `/mode probability` to Telegram.
2. `telegram_poller._handle_mode_command` (lines 332, 345, 361) loads
   `config.json` via `load_json`, mutates `notification.mode`, then calls
   `atomic_write_json(config_path, cfg)`.
3. After the call, `config.json` is no longer a symlink — it's a regular
   file living inside the repo with a copy of the contents that were resolved
   through the symlink. Now there are TWO copies of the secrets (the
   original external file plus the new in-repo file) and the next `git add`
   that catches it commits API keys.

The BUG-210 size guard (lines 350-355) does NOT mitigate this — the loaded
config has hundreds of keys after a successful symlink read; the guard only
fires when the read returned `{}`. The symlink replacement happens on the
SUCCESS path, exactly when the keys are present.

Secondary downstream effect: every `atomic_write_json` call site that
targets a path that might be a symlink has the same exposure. Today only
`config.json` is symlinked, so the user-triggered `/mode` command is the
real-world reproducer.

**Fix sketch:** in `atomic_write_json`/`atomic_write_text`, resolve the
destination through `path.resolve()` (or `os.path.realpath`) before computing
the temp directory and before the `os.replace` target. Alternatively, detect
`path.is_symlink()` and write through the resolved target. Add a test that
asserts symlink survival after atomic_write_json.

---

### P1-2 — `journal.load_recent` reads `layer2_journal.jsonl` with no sidecar lock — can race with concurrent `atomic_append_jsonl`
**File:** `portfolio/journal.py:23-40`

```python
def load_recent(max_entries=10, max_age_hours=8):
    if not JOURNAL_FILE.exists():
        return []
    cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
    entries = []
    with open(JOURNAL_FILE, encoding="utf-8") as f:
        for line in f:
            ...
            try:
                entry = json.loads(line)
                ts = datetime.fromisoformat(entry["ts"])
                if ts >= cutoff:
                    entries.append(entry)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
```

Two related issues:

1. **Bypasses the project I/O convention.** CLAUDE.md rule 4 says "Atomic
   I/O only" — the existing `load_jsonl` helper handles the same failure
   modes and is the canonical reader.
2. **No `jsonl_sidecar_lock`.** `atomic_append_jsonl` (file_utils.py:261-284)
   takes the sidecar lock for every append. A concurrent `load_recent`
   read does NOT take that lock, so on Windows the reader can observe a
   torn last line if the appender is mid-write. The `try/except
   json.JSONDecodeError` masks the corruption — the agent silently sees a
   journal with the latest entry missing.

In practice the Layer 2 agent invocation reads `load_recent` to build its
context immediately AFTER appending its own entry (decision flow in
`agent_invocation.py`). If the immediately-prior cycle's append is still in
flight (rare but observable on a slow disk under metals_loop + main loop
load), the agent sees a stale context.

The same pattern appears in `dashboard/app.py:875-919` (`api_mstr_loop`)
which uses raw `open()` to scan `mstr_loop_poll.jsonl` and
`mstr_loop_trades.jsonl` line-by-line — for a 2-line lookup this is
acceptable (last-entry semantics), but it still skips the lock.

**Fix:** swap `load_recent` to use `load_jsonl` or `load_jsonl_tail`, and
optionally hold `jsonl_sidecar_lock(JOURNAL_FILE)` for the read window. The
existing readers/writers already share the sidecar pattern; the new code
just needs to participate.

---

## P2 (should fix)

### P2-1 — `acquire_lock_file` leaks the file handle if `_lock_file` raises something other than `OSError`
**File:** `portfolio/process_lock.py:25-47, 63-74`

```python
fh = path.open("a+", encoding="utf-8")
try:
    _lock_file(fh)
except OSError:
    fh.close()
    return None
```

`_lock_file` (lines 63-74) raises `RuntimeError("No file locking mechanism
available...")` when neither msvcrt nor fcntl is importable. That branch is
**not** caught by the `except OSError`, so the open file handle leaks and
`acquire_lock_file` re-raises. Caller signatures are `fh | None`, so callers
treat exceptions as fatal — they don't get a leak, but they get a hard crash
with the lock file still pinned open. Symmetrize the exception handler:
`except (OSError, RuntimeError):` (or `BaseException` with re-raise) and
close `fh` before returning/re-raising.

### P2-2 — `subprocess.run` in `kill_orphaned_llama` uses `shell=True`
**File:** `portfolio/subprocess_utils.py:284-291`

```python
result = subprocess.run(
    ps_cmd,
    capture_output=True,
    text=True,
    timeout=15,
    shell=True,
)
```

`ps_cmd` is a hardcoded string with no user input, so today this is not
exploitable. But the function is the only place in `subprocess_utils.py`
that uses `shell=True`, and the surrounding `kill_orphaned_by_cmdline`
demonstrates the correct argv-list form (line 227-230). Future maintainers
will pattern-match on this and may add an interpolated argument. Convert
to the argv-list form (same as the sibling function) for defense in depth.

### P2-3 — `kill_orphaned_by_cmdline` escape table is incomplete — backtick is the PowerShell escape char itself
**File:** `portfolio/subprocess_utils.py:214-225`

```python
safe_pattern = (
    pattern.replace("'", "''")
    .replace("[", "``[")
    .replace("]", "``]")
    .replace("*", "``*")
    .replace("?", "``?")
)
```

The escape is `` `` `` (two backticks → one literal backtick + escape next
char). The implementation handles `'[]*?` but does NOT escape the backtick
itself — so a pattern containing a literal `` ` `` ends up unescaping the
following character in the PowerShell `-like` operator. All current callers
pass internal constants (no user input), so this is hardening, not a live
bug. Add `.replace("`", "``")` first in the chain so the substitution sees
already-escaped backticks.

### P2-4 — `_full_llm_cycle_count` / `_newsapi_daily_count` rate-limit state resets to zero on every process restart
**File:** `portfolio/shared_state.py:217-223, 304-307, 324-347`

```python
_full_llm_cycle_count = 0
...
_newsapi_daily_count = 0
_newsapi_daily_reset = 0.0
```

Module-level counters that persist only for the lifetime of the process.
PF-DataLoop has a 30s restart delay — a crash + restart resets the daily
NewsAPI counter, meaning the 90-call daily budget can be silently exceeded.
The `_newsapi_daily_reset < today_start` check (line 334) does the right
thing within a single process, but a restart at 09:00 UTC followed by 90
more calls before midnight UTC overshoots the 100/day free tier.

This is documented as "in-memory only" for `_full_llm_cycle_count` but NOT
for `_newsapi_daily_count`. Either persist the counter to disk (atomic
write every N calls) or document that the budget is per-process-uptime.

### P2-5 — `_loading_keys` eviction may not fire when no fresh request lands during stuck window
**File:** `portfolio/shared_state.py:68-89`

The stuck-key eviction (lines 68-74) only runs when a NEW request enters
`_cached`. If a key gets stuck in `_loading_keys` and no other thread asks
for the same or any other key inside `_cached` for >120s, the key stays
stuck. Subsequent first-asker is a `dogpile prevention return None` path
because `key in _loading_keys` is checked BEFORE the eviction sweep runs —
actually re-reading: the sweep runs at lines 68-74 unconditionally before
the dogpile check, so the next caller does evict the stuck key and proceed.
**Verified safe**, marking down to P3 noise — leaving here for future
maintainers as confirmation that the eviction is reachable.

### P2-6 — `dashboard/auth.py` `_get_config` cache uses `time.monotonic` with no invalidation hook
**File:** `dashboard/auth.py:54-79`

```python
_CFG_TTL = 60.0
...
if _CFG_VALUE is not None and (now - _CFG_AT) < _CFG_TTL:
    return _CFG_VALUE
```

60-second cache means a config rotation (rotating `dashboard_token`) is
honored within 60s — acceptable, but worth a doc string note. If the user
ever needs immediate rotation after a leak, the dashboard keeps accepting
the old token for up to 60s. Considered intentional; consider documenting
the 60s window explicitly so an operator knows when to wait. P3-ish but
worth flagging because the token cache is the only thing standing between
a leaked token and read-only access to every dashboard endpoint.

---

## P3 (nice to have)

### P3-1 — `dashboard/app.py` hardcodes port 5055 in `__main__`
**File:** `dashboard/app.py:2245-2246`

```python
if __name__ == "__main__":
    _serve_dual_stack(port=5055)
```

CLAUDE.md documents port 5055 — fine, but every port reference should
read from one constant (or config). Currently 5055 appears in 4 places
(app.py 2240/2246, the CORS allowlist 45-48). Single-source the constant.

### P3-2 — `dashboard/system_status.py` and `dashboard/trading_status.py` mix timezones
**File:** `dashboard/system_status.py:49` defines `UTC = timezone.utc`,
`dashboard/trading_status.py:54-58` uses `ZoneInfo("Europe/Stockholm")`.
Both correct in isolation; the home view payload combines both
without labeling, so a future "show last-trade time" tooltip will be
ambiguous. Add a docstring + tag each timestamp with its zone in the JSON.

### P3-3 — `portfolio/message_store.py:37-49` mojibake table has visually-duplicate keys
**File:** `portfolio/message_store.py:37-49`

The dict literals look like duplicate keys (`"â"` appearing 5 times),
but each is actually a distinct multi-byte UTF-8 sequence — verified by
loading the dict at runtime and printing `repr(key)` for each. The table
loads with 11 unique keys, so behavior is correct. Cosmetic: use explicit
`\u`-escapes or `"\\xe2\\x80\\x99"`-style byte literals so the source
reflects the actual key bytes — would also stop the `Edit` tool from
seeing them as ambiguous matches.

### P3-4 — `portfolio/api_utils.py:21-36` `load_config` uses raw `open()` not `load_json`
**File:** `portfolio/api_utils.py:21-36`

```python
with open(config_path, encoding="utf-8") as f:
    _config_cache = json.load(f)
```

CLAUDE.md rule 4: "Never `json.loads(open(...).read())` — use the provided
utilities." This is the file other callers go through to get API keys.
Switch to `load_json(config_path, default=None)` — raise if None (current
behavior on JSONDecodeError is to re-raise; load_json swallows it and
returns the default, so the call site needs to keep its existing error
handling). Same patch was applied to `telegram_poller._handle_mode_command`
on 2026-05-02 (per its own comment) but never to `api_utils`.

### P3-5 — `portfolio/file_utils.py:275` orphan "Unxfails" word in docstring
**File:** `portfolio/file_utils.py:275`

```
system-wide. Unxfails
``tests/test_fix_agent_dispatcher.py::test_concurrent_append_does_not_corrupt_jsonl``.
```

Looks like a merge artifact ("Unfails" or "Unxfails" — probably should be
"Unxfails" → "Unblocks"/"Un-x-fails" the test). Cosmetic.

---

## Verification — what I confirmed is sound

The following high-leverage primitives were stress-tested against the spec
items and pass:

- **`atomic_write_json` / `atomic_write_text` / `atomic_write_jsonl`** all use
  `tempfile.mkstemp(dir=path.parent)` (same volume → atomic rename) +
  `f.flush()` + `os.fsync()` + `os.replace()`. Windows semantics correct
  (P1-1 caveat: the symlink case is the only failure mode).
  `portfolio/file_utils.py:24-63, 287-305`.

- **`atomic_append_jsonl`** wraps the append inside
  `jsonl_sidecar_lock(path)`, which on Windows uses `msvcrt.locking(...,
  LK_LOCK, 1)` (blocking) and on POSIX uses `fcntl.flock(..., LOCK_EX)`
  (blocking). Lock byte is pre-seeded so empty-file races can't interleave.
  `portfolio/file_utils.py:202-284`.

- **`process_lock.acquire_lock_file`** is non-blocking
  (`LK_NBLCK` / `LOCK_EX | LOCK_NB`). Returns `None` on contention, so the
  exit-11 caller pattern in main.py works. P2-1 covers the
  RuntimeError-leak corner.

- **Dashboard auth ordering** (auth.py:103-200) correctly validates the
  CF-Access JWT against published JWKs via `verify_cf_jwt` BEFORE trusting
  the `Cf-Access-Authenticated-User-Email` header — fixes the 2026-05-13
  P0 header-spoofing finding. Cookie and bearer paths use
  `hmac.compare_digest`. Query-param path same.

- **`/logout`** is intentionally unauthenticated (auth.py:777-802 comment)
  so an expired-cookie client can still self-clear. The cookie is wiped
  with `max_age=0` matching the original flags. No auth surface.

- **CF-Access JWT verification** (`dashboard/cf_access.py:99-161`)
  fail-closes on missing config, bad signature, wrong aud, expired,
  email/claim mismatch. Each failure logs and returns `None`. Caller
  treats `None` as "fall through to next auth method" — correct.

- **Telegram sends** use `fetch_with_retry(..., timeout=30)` with a finite
  retry count (verified in `http_retry.py:17-19`). Never blocks the calling
  loop indefinitely. `message_store._do_send_telegram` (lines 105-167) and
  `telegram_notifications.send_telegram` (lines 35-81) both honor the
  timeout and have Markdown-fallback retry on HTTP 400.

- **Telegram poller offset persistence** (`telegram_poller.py:64-106`) is
  fail-soft, clamps negative values, and only persists offset after a
  command settles (success or intentional drop). On dispatch failure the
  persisted offset stays behind so restart re-fetches and retries.
  RESTART_BYPASS_MAX_AGE_S (line 39) caps the post-restart catch-up at 1
  hour so a multi-day outage doesn't fire every queued `bought MSTR ...`.

- **All 41 `@app.route` + 9 `@bp.route` handlers** are wrapped in
  `@require_auth` except `/logout` (intentional). Verified via `grep`.

- **`dashboard/static/*`** is served by Flask's default static handler
  without auth — confirmed only frontend assets there (CSS / JS /
  manifest / icons). `export_static.py` is deprecated and its target
  directory is gitignored (per its own docstring) — no `static/api-data/`
  leak today.

- **No dashboard endpoint exposes config.json, API keys, or Avanza session
  cookies.** `/api/iskbets` returns `data/iskbets_config.json` which is a
  user-set trading-params file (enabled flag, expiry, ticker) per the
  writes in `portfolio/iskbets.py:61-63`, not config.json. `/api/avanza_account`
  returns position/order/stop-loss data but not session cookies.

- **Process orphan reaping** (`subprocess_utils.py`) uses Windows Job
  Objects with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` for new spawns and
  a safety-net PowerShell scan for already-orphaned llama-completion.exe.
  Both paths handle Job Object assignment failure gracefully (log + fall
  through, line 134-137 / 177-179).

- **Health heartbeat** writes every 60s via `heartbeat_keepalive` (a daemon
  thread bounded by Event.wait) and is wrapped in `try/except` with WARNING
  level so a disk-full never aborts an in-flight Layer 2 decision
  (`portfolio/health.py:96-149`). `update_health` / `heartbeat` /
  `update_module_failures` / `update_signal_health_batch` all hold
  `_health_lock` across read-modify-write — no torn updates.

- **`dashboard/house_blueprint.py`** validates `run_id` (regex) and `slug`
  (regex + `secure_filename`) before using as path components, so the
  /house viewer cannot be path-traversed despite reading from an
  attacker-influenceable directory listing. Test asserts every route has
  `@require_auth` (line 21-26 comment).

---

## Summary

Two P1 issues to action:

- **P1-1** `atomic_write_json` blows away the `config.json` symlink when
  the Telegram `/mode` command writes config — secrets duplicate into the
  repo working tree.
- **P1-2** `journal.load_recent` reads journal lines without the
  `jsonl_sidecar_lock` other writers use; torn-line corruption is masked
  by `json.JSONDecodeError` swallow.

Five P2s worth scheduling: lock-handle leak on RuntimeError (P2-1),
defense-in-depth `shell=True` removal (P2-2), backtick escape (P2-3),
NewsAPI daily counter persistence (P2-4), config-cache TTL doc note (P2-6).

Five P3s for hygiene.

Cross-cutting confirmation:
the 50 route handlers are all behind `@require_auth` (except intentional
`/logout`), CF-Access JWT is properly verified, atomic primitives are
correct on Windows, telegram never blocks, heartbeat never crashes the
loop, locks are released on exception (with the one P2-1 corner). The
infrastructure layer is in good shape.

---

## File pointers

- `Q:/finance-analyzer/portfolio/file_utils.py:24-63` — atomic_write_json/_text
- `Q:/finance-analyzer/portfolio/file_utils.py:202-284` — sidecar lock + atomic_append_jsonl
- `Q:/finance-analyzer/portfolio/journal.py:23-40` — `load_recent` raw open (P1-2)
- `Q:/finance-analyzer/portfolio/telegram_poller.py:312-369` — `_handle_mode_command` (P1-1 trigger)
- `Q:/finance-analyzer/portfolio/process_lock.py:25-74` — acquire_lock_file (P2-1)
- `Q:/finance-analyzer/portfolio/subprocess_utils.py:198-291` — kill_orphaned_* (P2-2, P2-3)
- `Q:/finance-analyzer/portfolio/shared_state.py:304-347` — NewsAPI counter (P2-4)
- `Q:/finance-analyzer/dashboard/auth.py:54-200` — _get_config + require_auth
- `Q:/finance-analyzer/dashboard/cf_access.py:67-161` — JWT verification
- `Q:/finance-analyzer/dashboard/app.py:755-2204` — all routes
- `Q:/finance-analyzer/dashboard/house_blueprint.py:73-89` — slug/run_id validation
