# Claude adversarial review: infrastructure (2026-05-12)

## Summary

Infrastructure is mostly disciplined: `file_utils` consistently uses
`tempfile + fsync + os.replace`, sidecar locking on JSONL appends, a
session-scoped fixture redirects production caches in tests, and dashboard
auth uses `hmac.compare_digest` everywhere. The atomic-I/O contract is the
load-bearing piece and it holds.

That said, several blocking and high-severity defects remain:

- **P0 command-injection** in `subprocess_utils.kill_orphaned_by_cmdline()`:
  the `pattern` argument is f-string-spliced into a PowerShell `-like` mask
  with no sanitisation. Any caller that ever passes user/config-derived text
  achieves remote PowerShell execution. Callers today look hard-coded but
  this is a foot-gun primitive sitting unguarded in a library module.
- **P0 process-lock metadata corruption window**: `acquire_lock_file()`
  calls `fh.seek(0); fh.truncate()` *after* locking but *before* writing
  metadata. A crash between truncate and write leaves an empty lock file —
  the bat-wrapper exit-11 check still works (the lock byte is still held),
  but every diagnostic that parses the PID out of the file silently gets
  nothing, and `pf-restart.ps1` cannot identify orphans by CommandLine in
  combination with the lock holder.
- **P0 atomic_append_jsonl + os.fsync(0) tradeoff is fine; but
  `atomic_write_json` does not fsync the parent directory** after
  `os.replace`. On a power-loss or hard reset, the new inode can survive
  while the directory entry rolls back to the prior name (POSIX) or to
  nothing (some Windows ReFS configurations). This invalidates the
  documented H34 ("durability on power loss") claim.
- **P1**: `_avanza_worker_loop` runs a single Playwright thread but the
  request queue has no bound. A wedged Avanza session (BankID expired)
  combined with a Cloudflare-tunnel polling client produces a 25s-per-
  request backlog that grows without limit.
- **P1**: Cookie auth is HttpOnly + SameSite=Lax but **no CSRF token**
  on POST `/api/validate-portfolio`. A LAN attacker who guesses the
  attack origin can submit a forged portfolio JSON cross-site — currently
  read-only but the endpoint is mutating in spirit (returns validation
  results that gate further actions).
- **P1**: `_load_jsonl_tail_impl` decodes UTF-8 with `errors="replace"`,
  then `json.loads`. A truncated last line with multi-byte UTF-8 gets a
  `�` and gets silently dropped. For the loop-health watchdog this
  means a fresh entry can vanish from `_load_last_n_hours` exactly when
  the watchdog needs to see it.

The remainder are P2/P3 cleanups around CSRF, Cookie `Secure` behind
plaintext local proxy, and observability around the auto-spawn fix agent.

## P0 — Blockers

### P0-1. PowerShell command injection in `kill_orphaned_by_cmdline()`

`portfolio/subprocess_utils.py:214-218`:

```python
ps_cmd = (
    "Get-CimInstance Win32_Process "
    f"| Where-Object {{ $_.CommandLine -like '*{pattern}*' }} "
    "| Select-Object -ExpandProperty ProcessId"
)
```

`pattern` is f-spliced directly into a PowerShell string literal.
Any caller passing untrusted text (e.g. a ticker name from `config.json`,
a JSONL trigger field, or even a poorly-validated log path) can break out
of the single quotes and execute arbitrary PowerShell. Today's call sites
in `subprocess_utils` itself are hard-coded, but this is a library helper
and `kill_orphaned_by_cmdline()` has no docstring warning against
attacker-controlled input.

Fix: pass the pattern as a positional CIM argument or use
`-CommandLine -match [regex]::Escape($pattern)`, or build via
`subprocess.list2cmdline` + a parameter file. At minimum, reject any
pattern containing `'`, `"`, `;`, `|`, `&`, `` ` ``, `$`, or `\n`.

### P0-2. process_lock.py truncate-before-write window

`portfolio/process_lock.py:98-103`:

```python
fh.seek(0)
fh.truncate()
fh.write(" ".join(f"{key}={value}" for key, value in payload.items()) + "\n")
fh.flush()
```

The truncate happens before the write. A crash, power loss, or any
exception during `fh.write` leaves a zero-byte lock file. The OS lock
itself still survives (held on the handle), so `pf-loop.bat`'s exit-11
detection still works — but `pf-restart.ps1` and every "who holds the
loop?" diagnostic that reads the file gets nothing. The whole point of
writing PID/started/owner metadata is to recover the holder identity on
crash; this path destroys it.

Worse, the catch is `except Exception: pass` (line 103-104), which
silently swallows the failure. Combined with the load_json calling
convention elsewhere in the codebase, this is exactly the silent-degradation
pattern that caused the March-April auth outage.

Fix: write to `lock_path + ".tmp"` then `os.replace`, OR write atomically
in one `f.write(content + "\n")` after the read-side has been refactored to
tolerate a stale-content-plus-trailing-newline.

### P0-3. atomic_write_json does not fsync parent directory

`portfolio/file_utils.py:45-63`. Sequence is:

```python
os.fdopen(fd, "w") → write → flush → fsync(file)
os.replace(tmp, str(path))
```

This fsyncs the **file data** but not the **directory inode**. On Linux,
POSIX permits the rename to be persisted in the inode cache but lost from
the directory metadata on power failure. On Windows NTFS this is usually
safe because `MoveFileEx` (the underlying call) flushes the MFT — but on
ReFS, and on networked drives (which the user's Q: drive may be), the
guarantee is weaker. The docstring at line 49 claims "Fsyncs before
replace to guarantee durability on power loss (H34)" which over-promises
the actual guarantee.

Fix: on POSIX, after `os.replace`, open `path.parent` and fsync the dir
fd. Windows: the only true fix is `FlushFileBuffers` on the directory
handle, which Python doesn't expose — accept the limitation in docs but
remove the H34 claim, or call into `ctypes.windll.kernel32`.

### P0-4. JSONL sidecar lock release uses absolute byte 0 — race with concurrent unlock

`portfolio/file_utils.py:240-258`:

```python
with open(lock_path, "rb+") as lock_f:
    lfd = lock_f.fileno()
    ...
    if _msvcrt is not None:
        os.lseek(lfd, 0, os.SEEK_SET)
        _msvcrt.locking(lfd, _msvcrt.LK_LOCK, 1)  # blocking
        ...
    finally:
        if win_locked and _msvcrt is not None:
            try:
                os.lseek(lfd, 0, os.SEEK_SET)
                _msvcrt.locking(lfd, _msvcrt.LK_UNLCK, 1)
```

If the `yield` body raises, the finally runs. But if the file descriptor
was already closed by an external thread (e.g. via signal handler or
fork-exec inheritance), the `os.lseek` raises `OSError` which is
unhandled at the function boundary — the original exception is replaced
by the unlock failure. More importantly, `_msvcrt.locking(LK_LOCK, 1)`
is **blocking** with no timeout, so any wedged appender on Windows can
hang every loop and every dashboard request that touches a JSONL file.

The earlier `acquire_lock_file()` uses `LK_NBLCK` (non-blocking) for
this exact reason. The sidecar lock should grow a timeout-or-fallback
path; on Windows there is no `LK_LOCK` analog with deadline, but
spinning on `LK_NBLCK` with a small sleep is the standard workaround.

### P0-5. fix_agent_dispatcher invokes claude via claude_gate, but claude_gate's blocking gate is config.layer2.enabled

`scripts/fix_agent_dispatcher.py:398-415` imports
`portfolio.claude_gate.invoke_claude`, which honours
`config.json[layer2.enabled]` (claude_gate.py:438-451). If the user
disables Layer 2 (e.g. during research, during a known-broken state),
the fix agent **also stops firing** — but the auto-spawn task still runs
every 10 min, recording `fix_attempt_skipped` rows. These rows are level
"info" so check_critical_errors won't surface them. The user has no way
to learn that the fix agent has been silently neutralised, and the
30m-2h-12h backoff *does* still tick because the dispatcher state writes
happen regardless.

Net effect: a user who disables Layer 2 for an unrelated reason
permanently disables the auto-fix system without any warning. Either:

- The dispatcher should use its own gate (`PF_FIX_AGENT_DISABLED` env
  or `fix_agent.disabled` kill switch — the latter exists), and
  *bypass* the Layer 2 config check.
- Or it should surface a critical row when `config.layer2.enabled=false`.

This is also why the CLAUDE.md docs about `fix_agent.disabled` and
`config.layer2.enabled` interact in a way users don't expect.

## P1 — High

### P1-1. Avanza worker queue has no bound

`dashboard/app.py:1713-1778`. `_AVANZA_REQ_Q: queue.Queue[dict] =
queue.Queue()` is unbounded. The worker's `_avanza_snapshot_impl()` can
spend the full 25s budget per request. With three concurrent dashboard
clients polling at 30s cadence the queue fills, every request times out
at `_AVANZA_REQ_TIMEOUT_SECONDS = 25.0`, and the queue grows forever
because the worker drains at one-per-25s while requests arrive at
~one-per-10s.

Fix: bound the queue (`queue.Queue(maxsize=8)`) and return a 503 or a
cached snapshot when full. Also consider a global rate-limit so a single
client can't starve the worker.

### P1-2. No CSRF protection on POST /api/validate-portfolio

`dashboard/app.py:1067-1088`. Cookie auth with `SameSite=Lax` blocks
top-level cross-site GETs but not all POSTs (form-data POST is still
permitted under Lax). Combined with the CORS handler at
`app.py:52-61` that allows `Access-Control-Allow-Credentials = false`
explicitly — fine — and rejects unknown origins for the header reflection
— also fine.

But a *same-origin* page (e.g. `/house/` content reflecting user input)
or a tunneled subdomain (`raanman.lol`) could host a form that POSTs to
`/api/validate-portfolio` and the cookie rides along. Today the endpoint
only validates and returns — not state-changing — but the contract is
"portfolio validator" and the next iteration could plausibly trigger a
side effect.

Fix: require a CSRF token (HMAC of session token) on the POST path, OR
require `Bearer` auth for POSTs (so a same-origin XHR with no cookie
must supply the token explicitly).

### P1-3. load_jsonl_tail silently drops a truncated UTF-8 line

`portfolio/file_utils.py:177` decodes with `errors="replace"` then
`json.loads`. A torn last line that ends mid-multi-byte UTF-8 codepoint
becomes `...�` and `json.loads` raises which gets `continue`d. For
short polling windows (loop-health watchdog reading the last 24h with
adaptive growth), this means the *most recent* entry — the one the
watchdog most needs — can silently vanish until the next loop flushes
another line.

Fix: when the last line fails to decode, retry the read with
`tail_bytes *= 2` (mirror the dashboard's `_read_tail_with_growth`).
Or skip the trailing partial-line in the seek-from-end path explicitly
by looking for the last `\n` boundary.

### P1-4. Cookie set with `secure=True` will be silently dropped on plain-HTTP localhost

`dashboard/auth.py:97`. `secure=True` is set unconditionally on all
issued cookies. Browsers refuse to set/send a Secure cookie on
non-HTTPS origins. The dashboard binds dual-stack at `[::]:5055`
(app.py:2058-2092) and the README still documents `http://localhost:5055`
as the access pattern. Any user who hits `http://localhost:5055/?token=X`
gets a 401 cycle: `require_auth` sets the cookie, the browser silently
drops it on the next request, `require_auth` 401s. The token query trick
hides this on the first request only.

The Cloudflare tunnel path works because it terminates TLS at the edge
and the cookie sails. The bug bites local debugging.

Fix: detect `request.is_secure` (or check `X-Forwarded-Proto`) and only
set `secure=True` when actually behind TLS. Or document explicitly that
local access requires `https://localhost:5055` (which the app doesn't
support without a self-signed cert).

### P1-5. Recursion depth check in fix_agent_dispatcher can be bypassed by clearing env

`scripts/fix_agent_dispatcher.py:243-244, 362, 406`. The recursion guard
reads `os.environ[PF_FIX_AGENT_DEPTH]` at dispatcher startup and
temporarily mutates it for each invoke_claude. But the spawned Claude
agent runs with `allowed_tools = "Read,Edit,Bash"` — Bash can run
`unset PF_FIX_AGENT_DEPTH` then `python scripts/fix_agent_dispatcher.py`,
re-entering depth-0. The MAX_RECURSION_DEPTH=1 guard does not survive a
child shell.

Practical risk is low because (a) the agent is told not to spawn loops
and (b) Bash invocation requires user-style env discipline, but the
"defense in depth" intent of the env flag fails if the child shell can
reset it. Better: use a sentinel file under `/tmp` or `data/` that the
dispatcher creates on entry, tests for at start, and cleans on exit.

### P1-6. vector_memory has no lock; ChromaDB singleton races on first init

`portfolio/vector_memory.py:34-54`. `_get_collection()` lazy-inits a
module-level singleton without a lock. Two threads calling at the same
time both hit `_collection is None`, both enter `chromadb.PersistentClient`
on the same path, and ChromaDB's persistence layer (Sqlite under the
hood as of recent versions) may end up with two clients writing to the
same `data/chromadb/chroma.sqlite3`. Today the only call site is
`get_semantic_context()` from Layer 2 which is invoked single-threaded,
but the dashboard's `house_blueprint` could grow a feature that hits it.

Fix: add `_init_lock = threading.Lock()` around the lazy init.

### P1-7. message_store mojibake table replaces multiple distinct bad inputs with the same output

`portfolio/message_store.py:38-49`. The `_COMMON_MOJIBAKE_REPLACEMENTS`
dict has multiple identical-key entries (Python dict-literal will keep
the last one):

```python
"â": "—",
...
"â": "'",
"â": "'",
'â': '"',
'â': '"',
"â": "→",
"â": "↑",
"â": "↓",
```

Several of those are literally the same `"â"` key with different
replacement values — only the last (`"↓"`) wins at runtime. The intent
was probably to handle the multi-byte sequences `â€“`, `â€™`, `â€œ`,
`â†'` etc. but the source-code editing chopped them down to bare `â`. So
every mojibake input that contains a `â` ends up replaced with `↓`,
which actively *damages* messages that contain legitimate `â` (e.g. a
French ticker name or a copy-pasted French news source).

Fix: rewrite the table with explicit multi-byte sequences and remove
ambiguous bare-`â` keys.

## P2 — Medium

### P2-1. config_validator does not validate `dashboard_token` or `house_root`

`portfolio/config_validator.py:23-30`. Required keys are telegram,
alpaca, exchange. `dashboard_token` is missing from both REQUIRED and
OPTIONAL — the dashboard auth path tolerates `None` (no token = no auth)
which is fine for dev but is a leak in prod. The validator should at
least *warn* when running with no `dashboard_token` since the dashboard
will be wide-open.

`house_root` (referenced from `house_blueprint.py:59`) also is not
declared anywhere; defaulting to `r"Q:\househunting"` works but a
documented optional key would prevent surprises.

### P2-2. check_critical_errors.py treats resolution lines as level-agnostic

`scripts/check_critical_errors.py:62-67, 75-76`. A "resolution" record is
detected purely by the presence of `resolves_ts`. There is no level check
on the *resolver* line. An attacker (or a misbehaving agent) writing
`{"resolves_ts": "<any prior ts>"}` with `level: "critical"` would mark
that prior critical as resolved — and *itself* would be filtered out
because `level == "critical"` AND `resolves_ts` AND `resolution is None`
means line 79 catches it as resolved-by-self.

In practice the journal writer is local code and not an attack surface,
but the symmetry break with `system_status._errors_unresolved()` (which
checks `category == "resolution"`, dashboard/system_status.py:148) means
the two consumers disagree about what's resolved. A row with
`resolves_ts` but `category != "resolution"` resolves on the CLI tool
but NOT on the dashboard. Pick one rule.

### P2-3. _read_json cache uses object identity for path → string keying

`dashboard/app.py:101-102`. `_read_json(path)` keys the cache by
`f"json:{path}"`. `Path` objects' `__str__` is OS-normalised — fine. But
a caller that passes a string vs Path of the same target gets two cache
entries. Not a correctness bug but a memory-growth foot-gun in the
60s polling cycle.

Also, `_cache` (line 79) is **unbounded**. Every distinct path read by
every endpoint accumulates indefinitely. The 5s TTL prevents *staleness*
but does not evict entries. After weeks of uptime this is hundreds of
MB if file payloads are large. Add an LRU cap.

### P2-4. dashboard `/api/avanza_account` and friends do not enforce per-IP rate limits

The Avanza worker thread is a singleton; if the queue is bounded
(per P1-1) requests get 503'd. But there is no IP-level rate limit on
the dashboard endpoints overall. A misbehaving polling client (or a
compromised LAN device under Cloudflare Access) can hammer
`/api/system_status` and force constant computation. The 30s TTL helps,
but force=1 bypasses it.

### P2-5. fix_agent_dispatcher: cooldown writes survive a failed atomic_append_jsonl

`scripts/fix_agent_dispatcher.py:380-389, 408-441`. If
`_append_critical()` for `fix_attempt_started` succeeds but the
`fix_attempt_completed` append fails (disk full, lock contention),
`update_state_after_attempt(state, category, success)` still runs and
`_save_state(state)` at line 443 still persists the cooldown. Next
dispatcher tick sees a clean state and skips the category for 30m —
but the user's audit trail has a `started` with no `completed`.

Add an unconditional `_append_critical` for the completion event even
when the prior one failed, and on persistent failure record a
`fix_agent_dispatcher_io_error` row.

### P2-6. _violation_resolved: cross-stream dedup hides genuine new violations

`dashboard/system_status.py:391-399`. Any contract violation whose
identity matches an *unresolved* critical_errors row is hidden under
"violations" because the errors panel already shows it. That's correct
for de-dup but breaks observability: if 6 distinct new violations of
the same invariant arrive in 24h, only one critical_errors row was
written (cooldown) and all 6 are hidden. The user sees no escalation
even though something is firing repeatedly.

Fix: hide identical violations but expose a count badge ("6×
contract_violations within 24h, see errors panel").

### P2-7. Process-lock fcntl path leaks file descriptor on lock-failure exception

`portfolio/process_lock.py:36-44`. The path `fh = path.open(...); try:
_lock_file(fh); except OSError: fh.close(); return None;` is correct
for `OSError`. But `_lock_file` can also raise `RuntimeError` (line 68)
when neither msvcrt nor fcntl is available — that bubbles up without
closing the handle. Niche but real on exotic Python builds.

### P2-8. Dashboard `/api/oil`, `/api/crypto`, etc. embed config-derived paths without escaping

Not a current bug — paths are hard-coded under `DATA_DIR`. But the
pattern of `_read_json(DATA_DIR / "foo.json")` is repeated so often that
a future refactor exposing one of those filenames via a query parameter
would instantly become a path-traversal. Add a `_safe_data_path()`
helper that asserts `.resolve().is_relative_to(DATA_DIR)`.

### P2-9. atomic_append_jsonl uses ensure_ascii=False; atomic_write_jsonl too

`portfolio/file_utils.py:279, 298`. Fine — but `atomic_write_json` at
line 56 uses `ensure_ascii=True` by default (caller-overridable). The
asymmetry means `agent_summary.json` written via `atomic_write_json` is
ASCII-escaped but the same content fed back into a JSONL line via
`atomic_append_jsonl` becomes UTF-8. A reader doing strict-equality
checks on round-tripped data will mismatch.

### P2-10. SafeJSONProvider does not strip NaN inside nested numpy arrays

`dashboard/app.py:20-37`. `_json_safe` handles `float`, `dict`, `list`,
`tuple`. A `numpy.ndarray` or `pandas.Series` falls through to the
default encoder and re-introduces `NaN`/`Infinity` into the JSON output.
The portfolio_state / agent_summary writers are pure Python so this is
latent, but `dashboard/system_status._signal_aggregate` and the metals
context fallback could plausibly hand back numpy data through a future
refactor.

## P3 — Low

### P3-1. shadow_registry: status mutation does not validate "shadow" stays valid

`portfolio/shadow_registry.py:107-128`. `resolve_shadow(status="shadow")`
is allowed (line 116-117 allows all 3 valid values). That's a no-op when
the entry is already "shadow" but a *regression* when the entry is
already "promoted" — it demotes silently. Add a transition matrix or at
least a logger.warning on backwards transitions.

### P3-2. process_lock metadata `started` uses datetime.now(UTC) but not monotonic

If the system clock moves backward (NTP correction), older lock files
appear newer. Diagnostics that sort by `started=` get confused. Low
severity but worth noting.

### P3-3. notification_text._THESIS_STATUS_LABELS only covers 4 statuses

`portfolio/notification_text.py:5-10`. If thesis_status grows a 5th
value (e.g. "EVOLVING") the fallback at line 20 title-cases the raw key.
Probably fine but flag any new thesis_status to require this table
update.

### P3-4. dashboard/auth.py COOKIE_MAX_AGE is "just under 400-day cap"

The 365 * 24 * 3600 value (line 46) assumes Chrome's 400-day cap is the
limit. Safari and Firefox have their own caps (Safari ITP can clamp to
7 days for some first-party cookies under certain heuristics). Document
that "1-year rolling" is best-effort and the user may need to
re-authenticate on Safari periodically.

### P3-5. fix_agent_dispatcher's BACKOFF_SCHEDULE_S after exhaustion sets blocked_until = +3650 days

`scripts/fix_agent_dispatcher.py:275-279`. "Effectively disabled" via
`now + 3650 days`. This *is* a kill switch in practice, but the
recovery path (line 277 comment "user must manually reset by editing
state file or adding a resolution line") has no scripted helper. Add
a `scripts/reset_fix_agent_state.py` or document the JSON shape in
CLAUDE.md.

### P3-6. shared_state._cached: error path returns stale, then resets time to (now - ttl + 60)

`portfolio/shared_state.py:124`. Resetting the cache time to a value
that makes the next call retry in 60s is clever but means an
unrelated thread reading `_tool_cache[key]["time"]` for staleness
sees a *future-shifted* timestamp. The TTL math is internally
consistent but external observers (diagnostics, the dashboard's
cache-age UI) will be off by ~ttl seconds.

### P3-7. backtester.py imports `portfolio.signal_engine` at function-call time

`portfolio/backtester.py:71-78`. Lazy import avoids circular imports
but a misconfigured environment (`SIGNAL_NAMES` empty) gives no error
until backtest runs. Add a smoke test (`tests/test_backtester_smoke.py`)
that imports + asserts non-empty.

### P3-8. scripts/win/*.bat: `set CLAUDECODE=` after `cd /d` works, but only for the wrapper's spawn

The pattern in pf-loop.bat (line 11), metals-loop.bat (line 8),
crypto-loop.bat (line 11), oil-loop.bat (line 9), pf-agent.bat
(line 10) is correct. But scripts that spawn nested cmd shells (e.g.
`pf-restart.bat` calling pf-restart.ps1 which uses Get-CimInstance)
inherit CLAUDECODE again if the *parent* of the .bat had it set —
the .bat's own `set CLAUDECODE=` is fine, but Task Scheduler may
inject it on relaunch. There's no test that asserts this is clean at
the python.exe level. Add `tests/test_subprocess_env_hygiene.py` that
launches portfolio/main.py via the .bat and asserts os.environ.get
("CLAUDECODE") is None.

golddigger-loop.bat (lines 4-7) does **NOT** clear CLAUDECODE — could
matter if golddigger ever invokes claude_gate.

### P3-9. message_store sanitize_message_text strips ALL control chars including \x0B (vertical tab)

`portfolio/message_store.py:36`. `_CONTROL_CHAR_RE` covers most of
the C0/C1 set but a few legitimate-but-rare control chars (ASCII
record separators used by some logging frameworks) are removed. Low
impact for Telegram messages but worth noting.

### P3-10. dashboard `_serve_dual_stack` does not catch port-bind failure

`dashboard/app.py:2080-2092`. If port 5055 is already bound (e.g. a
prior dashboard didn't exit), the socket bind raises `OSError` and the
process exits non-zero. PF-Dashboard's task action does not auto-
restart. Add a friendly error message and an instruction to
`pf-restart.ps1 dashboard` (which doesn't exist).

### P3-11. shadow_registry seed_defaults() always writes, even if no entries are added

`portfolio/shadow_registry.py:231-239`. `save_registry(reg)` is called
unconditionally at line 239. If nothing changed, atomic_write_json
still rewrites the file (new mtime, new content identical). Minor I/O
waste but breaks change-detection systems.

## Tests missing

- **test_subprocess_utils_injection**: feed `kill_orphaned_by_cmdline(
  "foo'; Remove-Item -Recurse Q:\\finance-analyzer; '*")` and assert
  PowerShell does NOT execute the embedded payload (mock subprocess.run
  and inspect the actual command line passed to PowerShell).
- **test_process_lock_metadata_persists_on_crash**: simulate a
  KeyboardInterrupt between `fh.truncate()` and `fh.write()`; the
  remaining lock file should still contain a parseable PID line OR be
  absent — never zero bytes.
- **test_atomic_write_json_durability_on_dir**: hard to write
  hermetically; at minimum assert `os.replace` is the last syscall and
  no `os.unlink` of the target happens.
- **test_jsonl_sidecar_lock_timeout**: spawn two processes, hold the
  lock in one, assert the second blocks AND has a configurable
  timeout. Currently it blocks indefinitely on Windows.
- **test_load_jsonl_tail_truncated_utf8**: write a file that ends with
  half a multi-byte codepoint (e.g. `b"\xe2\x82"` mid-€), tail-read it,
  assert the *complete* prior entries are returned and the partial line
  is silently dropped — NOT that an entry is corrupted to U+FFFD and
  silently parses to a different dict.
- **test_dashboard_auth_no_secure_on_plain_http**: hit
  `http://localhost:5055/?token=X` with a stock Flask test client,
  assert Set-Cookie does NOT include `Secure` when `request.is_secure
  is False`. (This test will fail today.)
- **test_dashboard_validate_portfolio_requires_csrf**: POST without a
  CSRF token, assert 403.
- **test_avanza_worker_queue_bounded**: enqueue 100 requests rapidly,
  assert that beyond the bound the queue refuses with a structured
  error rather than growing.
- **test_fix_agent_dispatcher_respects_layer2_disabled**: set
  `config.json[layer2.enabled] = false`, run dispatcher, assert it
  surfaces a critical row about "fix agent disabled by Layer 2 config"
  instead of silently logging `fix_attempt_skipped`.
- **test_fix_agent_dispatcher_recursion_resistant_to_env_clear**:
  spawn a dispatcher that clears `PF_FIX_AGENT_DEPTH` before
  re-invoking; assert the second invocation is blocked by a sentinel
  file mechanism, not just by the env flag.
- **test_message_store_mojibake_table_no_duplicate_keys**: assert
  `len(_COMMON_MOJIBAKE_REPLACEMENTS) == n_distinct_keys`; today the
  literal dict silently dedupes and the test would catch the bug at
  P1-7.
- **test_check_critical_errors_vs_system_status_resolution_agreement**:
  build a fixture journal where a row has `resolves_ts` but
  `category != "resolution"`; assert the two consumers agree on
  unresolved count.
- **test_shadow_registry_resolve_does_not_allow_backwards_transition**:
  promote a shadow, then call `resolve_shadow(status="shadow")`,
  assert ValueError or no-op (per design choice).
- **test_dashboard_cache_bounded**: hit 10,000 distinct paths,
  assert `len(_cache) <= some_cap`.
- **test_golddigger_loop_bat_clears_CLAUDECODE**: parse the .bat,
  assert `set CLAUDECODE=` is present before the first python.exe
  invocation.
