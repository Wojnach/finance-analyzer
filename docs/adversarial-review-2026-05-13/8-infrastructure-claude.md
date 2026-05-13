# Claude adversarial review: infrastructure

## Summary

Reviewed the data-integrity / process / IPC / dashboard tier. The atomic
I/O primitives in `portfolio/file_utils.py` are mostly competent (fsync
+ tempfile + os.replace), but the sidecar-lock primitive that backs
every JSONL writer has a real-but-subtle creation race and the
`atomic_append_jsonl` path never validates the line it just wrote, so
torn-line corruption is observable instead of prevented. The dashboard
auth layer is the weakest surface: the Cloudflare bypass trusts
`Cf-Access-*` headers with no IP allow-list, so anything that can reach
port 5055 directly (LAN, misconfigured tunnel, mis-routed DNS, host
header attack) can forge those headers and bypass the token entirely.
Several other route-level defects compound: a CSRF-naive POST
endpoint, an order-sensitive cache TTL, an Avanza worker thread with
unbounded queue growth on stalls, and tail-read helpers that silently
under-deliver on adaptive shrinking. process_lock has no stale-PID
detection — a host crash leaves the lock owned by a dead process and
the next loop launch refuses to start. subprocess_utils' Job Object
plumbing uses `proc._handle`, a private attribute that's wrong on
PyPy / future CPython. fix_agent_dispatcher mutates `os.environ`
across siblings without holding any lock, and stores tens-of-thousands
of cooldowns into a single JSON file that's atomically replaced every
run. Backtester has look-ahead via close-included slicing and ships
zero fee/slippage model — printed deltas are not portfolio-realistic.

P0 (data integrity / external exposure): 4 — auth bypass via header
forgery; sidecar-lock TOCTOU; cross-volume `os.replace` failure mode
silently leaks tmp files; `atomic_append_jsonl` writes raw bytes
without per-line validation so a partial fsync on disk-full produces a
torn line that survives.

P1 (correctness, would cause real outages): 12.
P2 (degraded behaviour, surfacing): 14.
P3 (cosmetic / hardening): 8.

---

## P0 — Blockers

### P0-1. Cloudflare Access bypass — header forgery from LAN
**Files:** `dashboard/auth.py:132-135`, `dashboard/app.py:2058-2092`

```python
cf_email = request.headers.get("Cf-Access-Authenticated-User-Email")
cf_jwt = request.headers.get("Cf-Access-Jwt-Assertion")
if cf_email and cf_jwt:
    return _refresh_cookie(make_response(f(*args, **kwargs)), expected)
```

The auth decorator trusts ANY request that presents both `Cf-Access-*`
headers, with the reasoning "CF strips inbound Cf-Access-* headers at
its edge and re-injects them only after successful Access policy
evaluation." That is only true for traffic that actually traverses
Cloudflare. The dashboard binds at `[::]:5055` (dual-stack, all
interfaces, see `_serve_dual_stack`). Any host on the LAN, any
container on the box, any process that resolves the public hostname
to a private IP (DNS rebinding), or anyone who gets the tunnel
configuration wrong can connect directly and send:

```
GET /api/avanza_account HTTP/1.1
Host: dashboard.example
Cf-Access-Authenticated-User-Email: attacker@evil.com
Cf-Access-Jwt-Assertion: anything-non-empty
```

— and bypass the dashboard_token entirely. The `cf_jwt` value is
never verified. The comment "but a LAN attacker spoofing headers
would need to know about and forge both" is wrong — both are trivially
forged once you read this comment. Real CF Access integrations
**validate the JWT signature** against Cloudflare's published JWK set
(or do mTLS / require the request to come from the cf-connecting-ip
range). Neither is done here.

Compounding factor: this route exposes `/api/avanza_account`, which
returns LIVE positions, cash, open orders, and stop-losses — i.e. the
attacker's reconnaissance to know exactly when to front-run the
warrant ladder.

**Fix:** Either verify the CF Access JWT against the team's JWKS, OR
require `request.remote_addr` to be in CF's published IP ranges, OR
bind the server to `127.0.0.1` and rely solely on the tunnel. Drop the
unconditional trust path.

### P0-2. `jsonl_sidecar_lock` TOCTOU on lock-file creation
**File:** `portfolio/file_utils.py:229-258`

```python
lock_path = path.parent / f".{path.name}.lock"
if not lock_path.exists():
    try:
        with open(lock_path, "ab") as lf:
            if lf.tell() == 0:
                lf.write(b"\0")
    except OSError as exc:
        logger.warning("sidecar lock creation failed for %s: %s", path, exc)

with open(lock_path, "rb+") as lock_f:
    ...
    _msvcrt.locking(lfd, _msvcrt.LK_LOCK, 1)
```

Two concurrent first-writers race the `exists()` check; both open in
append mode; both could find `tell() == 0`; both write `\0`. Result:
two `\0` bytes in the lock file. Each subsequent caller locks **byte
0**. That works for serialization, but the seed-creation path is
still a write race on the very file we depend on for mutual exclusion
of *every* JSONL writer in the system — `signal_log.jsonl`,
`critical_errors.jsonl`, `telegram_messages.jsonl`,
`claude_invocations.jsonl`, `layer2_journal.jsonl`,
`accuracy_snapshots.jsonl`, and ~14 others.

Worse: on Windows, opening the file `"ab"` while another process holds
`msvcrt.locking(..., LK_LOCK, 1)` on byte 0 of the SAME file via a
different handle may raise `PermissionError`. The exception path falls
through to `with open(lock_path, "rb+")` immediately afterward —
which will also fail. The except-OSError path **logs at WARNING and
proceeds with the open**, so we end up holding no lock and silently
appending unsynchronized.

The "first-byte sentinel" file is also created with `"ab"` (append)
rather than `"x"` (exclusive create), so the exclusive-create-or-fail
semantics aren't enforced.

**Fix:** Open with `O_CREAT | O_EXCL` for the seed path, fall through
to a retry loop if file exists. Or: use `os.open(..., O_CREAT)` once
at module import to pre-seed the lock files for known paths.

### P0-3. `atomic_write_*` cross-volume `os.replace` silently leaks tmp on failure
**File:** `portfolio/file_utils.py:45-63, 286-305, 380-391`

```python
fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
try:
    ...
    os.replace(tmp, str(path))
except BaseException:
    with suppress(OSError):
        os.unlink(tmp)
    raise
```

The `tempfile.mkstemp(dir=str(path.parent))` call is correct — same
volume. **BUT** `path.parent.mkdir(parents=True, exist_ok=True)` runs
before mkstemp, and if the parent directory is a junction / symlink /
mount point that crosses to a different volume on Windows (which Q:
on this user's setup can be — it's listed as a separate disk), the
tmp file lands on a different volume than the target, and
`os.replace` raises `OSError: [WinError 17]` (cannot rename across
volumes). The except path tries `os.unlink(tmp)` — fine — but the
**re-raised exception in `atomic_write_json` propagates up to
callers** including `prophecy.py`, `portfolio_mgr.py`,
`accuracy_stats.write_accuracy_cache`, `golddigger`, `iskbets`,
`shadow_registry.save_registry`, etc. The vast majority of those
callers wrap the write in a broad `except Exception` and log-and-go,
so the failure becomes a silent skip. Net effect: state file is never
updated, but the system reports "saved" because the exception was
swallowed upstream.

Compounding: on Windows, `os.replace` to a file that another process
has open with `share_mode != FILE_SHARE_DELETE` raises `PermissionError`.
Same silent-skip behavior. The dashboard's `_cached_read` keeps a 5s
TTL file-handle-free cache, but if any external editor or AV scanner
has `agent_summary.json` open for read, **the writer silently no-ops**.

**Fix:** Either (a) raise a typed `AtomicWriteFailed` exception that
callers cannot silently swallow, (b) record critical_error on
ENOENT/EXDEV/EACCES from `os.replace`, (c) add a self-test on writer
init that verifies `os.replace` works on the target dir.

### P0-4. `atomic_append_jsonl` allows torn lines on disk-full / SIGTERM
**File:** `portfolio/file_utils.py:261-284`

```python
def atomic_append_jsonl(path, entry):
    path = Path(path)
    data = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
    with jsonl_sidecar_lock(path):
        with open(path, "ab") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
```

The header docstring says the function provides "atomic semantics
across threads and processes", but appending to a regular file via
`f.write(...)` is **NOT atomic** above the PIPE_BUF size (4096 bytes
on Linux, undefined on Windows). For most entries (<4KB) this is
fine in practice, but:

1. The `signal_log.jsonl` entries are routinely larger than 4KB (tens
   of tickers × ~30 signal votes + metadata). A snapshot regularly
   exceeds 8KB.
2. The function holds a sidecar lock, so concurrent writers don't
   interleave bytes — good. But if the process is SIGTERM'd or the
   disk hits ENOSPC mid-`write()`, the file ends with a partial line
   and **no trailing newline**. The next writer's append (after the
   process restart) lands directly on the partial line, producing a
   malformed JSON line that load_jsonl skips silently.
3. The fsync **happens after** the lock context could theoretically
   release if `fsync` raises — but it's still inside the `with` block,
   so this is OK.

The real failure: there's no read-back-verify, no length-prefix, no
truncate-on-failure. `signal_log.jsonl` is the source-of-truth for
accuracy stats; a single torn line per week shows up as silent
sample-count drift in `accuracy_cache.json`.

`prune_jsonl` (line 349-393) does validate JSON during read but does
NOT preserve the offending malformed lines — it silently drops them
on next prune, **erasing the audit trail of the torn write**.

**Fix:** Write to `path + ".tmp.<pid>"`, fsync, then concat-append by
locking + read-tmp + write to target + fsync. Or: verify the line
parses post-write, truncate back on failure. Or: use a SQLite WAL
journal for hot logs.

---

## P1 — High

### P1-1. `process_lock` has no stale-lock / dead-PID recovery
**File:** `portfolio/process_lock.py:25-47, 86-107`

```python
fh = path.open("a+", encoding="utf-8")
try:
    _lock_file(fh)
except OSError:
    fh.close()
    return None
```

`acquire_lock_file` writes `pid=<getpid()> started=<iso>` into the
lock file *after* successfully acquiring it (line 86-107). The actual
lock is held via `msvcrt.locking(fh.fileno(), LK_NBLCK, 1)` /
`fcntl.flock(LOCK_EX | LOCK_NB)`. Both of those release automatically
when the holding process dies — Windows kernel and POSIX file-table
cleanup. So the lock itself is OK across crash.

**BUT** if the host loses power, on Windows the lock state on a
network/junction volume may persist briefly after reboot until the
kernel cleans the handle. Worse, the metadata-only PID/started lines
are never validated by the next acquirer. There's no `is_pid_alive()`
fallback — if the locking primitive returns "still locked" for any
reason (e.g. another user's process owns it, antivirus scanner has it
open, or a sibling Python process orphaned a handle), the loop refuses
to start and there is **no diagnostic surfacing** beyond
`return None`.

The PF-DataLoop scheduled task uses `auto-restart 30s` (per
`memory/MEMORY.md`), and the comment in CLAUDE.md says "system
reliability is #1, the loop must run 100% of the time". If
`acquire_lock_file` returns None on transient OS noise, the loop
**will not restart**.

**Fix:** After `acquire_lock_file` returns None, read the lock file's
`pid=` payload and check if that PID is alive via
`OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)` on Windows. If dead,
delete the lock and retry once. Surface the decision to
`critical_errors.jsonl` either way.

### P1-2. subprocess_utils: `proc._handle` is a private attribute
**File:** `portfolio/subprocess_utils.py:132, 175`

```python
kernel32.AssignProcessToJobObject(job, int(proc._handle))
```

`subprocess.Popen._handle` is implementation-private to CPython on
Windows. On PyPy it doesn't exist. On future CPython releases it may
become a `weakref`-managed handle that isn't an integer-castable
`HANDLE`. The fix-agent dispatcher and Claude CLI invocations live
behind this — if Python 3.13+ changes the attribute (it already
moved from `_handle` to `_handle.handle` in some 3.12 patch builds),
every subprocess silently runs without Job Object protection, and the
fallback path in `popen_in_job` (which logs at `debug` level) hides
the breakage.

**Fix:** Use the public `proc.pid` + `kernel32.OpenProcess(...)` to
get a fresh handle, or use `psutil` which abstracts the platform
detail. Log at WARNING (not DEBUG) when Job Object assignment fails —
this is a known orphan-risk path for llama-completion.exe (see
`kill_orphaned_llama`).

### P1-3. subprocess_utils: stdin not closed before communicate → deadlock risk
**File:** `portfolio/subprocess_utils.py:125-143`

```python
input_data = popen_kwargs.pop("input", None)
if input_data is not None and "stdin" not in popen_kwargs:
    popen_kwargs["stdin"] = subprocess.PIPE

proc = subprocess.Popen(cmd, **popen_kwargs)
...
stdout, stderr = proc.communicate(input=input_data, timeout=timeout)
```

`run_safe` is the path used to invoke `claude -p` from Layer 2.
`communicate(input=None)` on Windows DOES close stdin, so most cases
are fine. **But** the assignment between `Popen` and `communicate`
spans `kernel32.AssignProcessToJobObject(...)` which can take >10ms.
If the child process (Claude CLI) reads stdin before the parent calls
`communicate`, AND the child's read returns a 0-length read because
the parent hasn't closed yet (stdin=PIPE) — Claude CLI's reported
behavior in CLAUDE.md ("nested session" error on inherited CLAUDECODE
env) is consistent with this pattern.

Separately: there's no explicit `CLAUDECODE` env scrubbing in
`run_safe`. `memory/MEMORY.md` explicitly warns "If [CLAUDECODE env
var is] inherited by loop/Task Scheduler, `claude -p` fails ('nested
session' error). Fix: `set CLAUDECODE=` in bat file. Caused 34h
outage Feb 18-19." The fix has been pushed to the .bat layer, but
`run_safe` itself doesn't enforce it — if anyone bypasses the .bat,
the outage recurs.

**Fix:** In `run_safe`, when invoking `claude -p`-like commands,
default `env=` to a copy of `os.environ` with CLAUDECODE/CLAUDE_CODE
stripped. Or: detect command basename `claude` and scrub.

### P1-4. fix_agent_dispatcher mutates `os.environ` without lock
**File:** `scripts/fix_agent_dispatcher.py:405-423`

```python
prior_env = os.environ.get(RECURSION_ENV)
os.environ[RECURSION_ENV] = str(caller_recursion_depth + 1)
try:
    success, exit_code = invoke_claude_fn(...)
finally:
    if prior_env is None:
        os.environ.pop(RECURSION_ENV, None)
    else:
        os.environ[RECURSION_ENV] = prior_env
```

The dispatcher iterates categories serially in one thread, so the
mutation pattern itself is OK. The latent bug: `invoke_claude_fn`
inside the try-block may *spawn its own threads* via the Layer 2
multi-agent specialist tools (per CLAUDE.md, there are parallel agent
spawns). If any of those threads read `os.environ` while the parent
holds the bumped RECURSION_ENV, they will inherit `depth+1` and
prematurely block, regardless of whether they are descendants of the
fix-agent or not.

Compounding: `os.environ` in CPython is process-global but NOT
thread-safe on every platform (Linux glibc 2.32 had a documented
TOCTOU between getenv and putenv). On Windows, mutating environ
between thread spawns can leak the bumped value to siblings if the
threading model uses `_Py_NewInterpreter` (unlikely here, but
possible via embedded tooling).

**Fix:** Pass the recursion depth as an explicit prompt-prefix /
argv flag to `invoke_claude_fn` (and have it forward to subprocess
env via `Popen(env=...)`), not via mutating the parent's environ.

### P1-5. fix_agent_dispatcher state-file growth unbounded
**File:** `scripts/fix_agent_dispatcher.py:144-151, 256-282`

`update_state_after_attempt` writes a `by_category[<name>]` entry for
every category seen. Categories include `accuracy_degradation`,
`fix_agent_state_corrupt`, `signal_log_reconciliation`,
`layer2_journal_activity`, plus open-vocab category names from
`record_critical_error` callers across the codebase. There is **no
purge** of stale category entries. After a few years of operation,
`fix_agent_state.json` will accumulate hundreds of dead categories
each with a `blocked_until` 10 years in the future. The whole file is
re-serialized and re-fsync'd every dispatcher run (every 10 minutes).

Less severe but real: an attacker who can append to
`critical_errors.jsonl` (e.g. via a compromised loop) can populate
arbitrary categories and stuff `fix_agent_state.json` until it's
megabytes.

**Fix:** Purge categories whose `blocked_until` is past + 30 days and
have no recent entries.

### P1-6. fix_agent_dispatcher cooldown bypass when state load fails twice
**File:** `scripts/fix_agent_dispatcher.py:99-141`

When `_load_state` hits a corrupt file, it (correctly) writes a
1-hour global block AND records a critical error of category
`fix_agent_state_corrupt`. **But** the conservative default it
returns:

```python
return {
    "by_category": {},
    "recursion_counter": 0,
    "blocked_until_global": (_now() + timedelta(hours=1)).isoformat(),
    "_corrupt_loaded_at": _now().isoformat(),
}
```

is then **persisted** at line 443 via `_save_state(state)`. Next
dispatcher run loads it fine, sees `blocked_until_global` is in the
past after 1h, and proceeds normally — losing the audit of the prior
corruption beyond the single jsonl entry. Worse: between `_load_state`
returning the conservative default and `_save_state` persisting it,
the dispatcher iterates categories and may spawn fix agents if the
`blocked_until_global` is exactly at the boundary (no monotonic
clock).

Worse still: the dispatcher's own write of
`fix_agent_state_corrupt` via `_append_critical` becomes the next
dispatcher run's unresolved entry. The fix agent for that category
will... fail to fix the file (it's already been overwritten), the
attempt is recorded as `fix_agent_failed`, backoff escalates, and
eventually that category enters the 3650-day-disabled state. Self-
healing rather than self-defeating, but only barely.

### P1-7. fix_agent_dispatcher BACKOFF_SCHEDULE_S off-by-one disable
**File:** `scripts/fix_agent_dispatcher.py:46-47, 273-281`

```python
BACKOFF_SCHEDULE_S = [1800, 7200, 43200]  # 30m → 2h → 12h, then disabled
...
idx = min(new_count - 1, len(BACKOFF_SCHEDULE_S) - 1)
if new_count > len(BACKOFF_SCHEDULE_S):
    entry["blocked_until"] = (now + timedelta(days=3650)).isoformat()
```

CLAUDE.md documents the backoff as "30m → 2h → 12h → effectively
disabled". That is what `new_count > len(BACKOFF_SCHEDULE_S)`
(i.e. `new_count > 3`) gives — the 4th failure trips disable. So the
behavior matches documentation. **But** there is no
"deeply-disabled" path that re-engages — a category that hit the
disable threshold once and was then resolved manually still carries
`consecutive_failures=4` (no decay on resolution). A future
unrelated failure of the same category increments to 5 and stays
disabled. The "stop re-firing on the same error" comment in the
prompt at line 323-325 assumes the resolution line will be appended;
the consecutive_failures counter is independent of the journal.

**Fix:** Reset `consecutive_failures` to 0 on any
`resolves_ts`-pointing entry that lands in the journal, OR add a
`fix_agent_state.json` self-decay (1 failure unit per day) so a
once-disabled category recovers.

### P1-8. message_store dedup absent — same message stored N times
**File:** `portfolio/message_store.py:87-103, 170-219`

Despite the review brief noting "Message store: dedup by hash, atomic
append", **there is no dedup**. `log_message` unconditionally appends
to `telegram_messages.jsonl`. `send_or_store` calls `log_message`
twice for muted paths (lines 196 and 204) — once for the per-category
mute, once... actually only once per code path, OK. But if the same
trade decision is sent on a retry (`_do_send_telegram` returns False
on first call, caller retries upstream), `send_or_store` is called
twice and produces two journal entries plus two Telegram messages.
Telegram sees the same `text` body and renders it as two separate
messages.

The brief also mentions "hash includes timestamp (defeats dedup)" —
not applicable because dedup doesn't exist. The fix should add it.

The mojibake table at line 37-49 has **duplicate keys**:

```python
"â": "—",
"â€“": "–",
"â": "'",  # collision with previous "â":
"â": "'",  # another collision
'â': '"',  # another collision
...
```

Python dict literal semantics: the LAST assignment wins. So all of
`â` → `—`, `â` → `'`, `â` → `'`, `â` → `"`, `â` → `"`, `â` →
`→`, `â` → `↑`, `â` → `↓` collapse into whatever the source bytes
of the *final* `â` literal are. This is silently broken mojibake
repair.

### P1-9. message_store Markdown injection from user-controlled fields
**File:** `portfolio/message_store.py:105-167`

`_do_send_telegram` sends `parse_mode=Markdown`, then on parse failure
retries WITHOUT parse_mode. The retry path is fine, but the primary
path will happily render any `*bold*`/`[link](url)` from the message
body, and the body is built from signal reasoning, news headlines
(headline text from NewsAPI / CryptoCompare), and Claude Layer 2
output. A news headline like
`Bitcoin tanks [click here](https://evil.com/grab?token=xxx)` would
render as a clickable link in Telegram. The control character
sanitizer (`_CONTROL_CHAR_RE`) does NOT strip `[`/`]`/`(`/`)` or `*`.

Layer 2 messages going through `send_or_store` are particularly
vulnerable because they include LLM-generated text that may include
attacker-influenced content from any news source the LLM saw.

**Fix:** Either escape Markdown metacharacters in news-derived fields
before formatting (preferred — keep Markdown for known-safe layout),
or switch all sends to plain text. Telegram's `MarkdownV2` requires
explicit escaping of `[]()_*~>#+-=|{}.!\` so the current `Markdown`
mode is the lenient legacy parser, but link rendering is still on.

### P1-10. shadow_registry write-then-read race
**File:** `portfolio/shadow_registry.py:88-104, 107-127`

`add_shadow` does load → mutate → save. `resolve_shadow` does the
same. There is no lock — two processes calling concurrently will
last-writer-wins. Less critical because the registry is updated
infrequently from a small number of writers, but the
`atomic_write_json` it uses (per save) is followed by a fresh
`load_registry` in the next call, with no version/etag, so a
concurrent re-entry of `shadow:fingpt` immediately after a manual
`resolved` flip would resurrect the shadow entry.

### P1-11. dashboard `_cached_read` returns stale data after underlying file deleted
**File:** `dashboard/app.py:79-94`

```python
def _cached_read(key, ttl, read_fn):
    now = time.monotonic()
    with _cache_lock:
        entry = _cache.get(key)
        if entry and (now - entry[1]) < ttl:
            return entry[0]
    result = read_fn()
    with _cache_lock:
        _cache[key] = (result, now)
    return result
```

`_cache` is a plain dict with no eviction. The dashboard runs for
months at a time. Every JSONL file read with a unique `limit` value
creates a new cache key. After enough requests with varying `limit`
or `tail_bytes` parameters, the cache grows unbounded. `_read_jsonl`
generates `f"jsonl_tail_v2:{path}:{limit}"` keys — if a UI bug or
attacker sends 10,000 different `?limit=` values, the dashboard
caches 10,000 entries.

Also: there is no negative-result caching distinction. If `read_fn()`
returned `None` (file missing), that `None` is cached for 5 seconds.
A file that the writer just created is invisible for up to 5 seconds —
fine for stationary data, not fine for `health.heartbeat`-style
liveness signals where the dashboard's own threshold is 120s.

### P1-12. backtester: look-ahead via outcome lookup against entry's own ts
**File:** `portfolio/backtester.py:113-198`

```python
for entry in entries:
    ts = entry.get("ts", "")
    ...
    outcomes = entry.get("outcomes", {})
    tickers_data = entry.get("tickers", {})
    for ticker, tdata in tickers_data.items():
        ...
        outcome = outcomes.get(ticker, {}).get(h)
```

The "outcome" for each entry is read from the SAME entry. The
backtest's accuracy_data (line 93) is `_build_accuracy_data(horizon)`
which loads the **current** accuracy cache — i.e. accuracy computed
using outcomes that include the test entries themselves. This is
in-sample look-ahead: the weights used to decide BUY/SELL in the new
consensus are derived from the very outcomes being evaluated.

The brief notes "walk-forward, no look-ahead" — the code does no
walk-forward. There is no time-split, no train/test partition, no
expanding window. The `--days` parameter only restricts the
evaluation set; the accuracy weights used by `_weighted_consensus`
have already been trained on all-time data including the future
relative to each entry being tested.

Result: `print_report` ALWAYS shows the new system beating the old
system, because the new system is being scored on the same outcomes
it was trained on. This is a textbook overfit confirmation.

Compounding: zero fee model, zero slippage model, zero stop-loss
modeling. The "% accuracy improvement" cannot be translated to P&L.

**Fix:** Implement true walk-forward — split entries into train/test
chronologically, fit accuracy on train, score on test. Or at minimum
prepend a 30-day "warmup" carve-out from the evaluation set.

---

## P2 — Medium

### P2-1. file_utils `load_jsonl_tail` first-line-drop heuristic mis-fires on UTF-8 boundary
**File:** `portfolio/file_utils.py:163-183`

The "peek byte at offset-1 to check for newline" works for ASCII but
the seek-by-byte may land **mid-codepoint** for UTF-8 (e.g. mid `→`
which is `0xE2 0x86 0x92`). The `data.decode("utf-8", errors="replace")`
substitutes a `�` for the leading half-codepoint, the first
"line" starts with a replacement char, and the heuristic at line 180
treats the replacement-char-starting line as "drop me, mid-line."
This is correct in spirit but the byte-peeked `prior == b"\n"`
check is unaware of encoding boundaries — if `offset` lands one byte
after a 3-byte codepoint (so `f.read(1)` returns `b"\x92"`),
`prior == b"\n"` is False, the line is dropped — fine. But if
`offset` lands right after a `\n` *inside* a multi-byte char in the
prior line... that can't happen in UTF-8 (newline is never inside a
multi-byte char). So the heuristic is OK *for UTF-8 specifically*.
The brittleness is the implicit dependency on encoding.

More concerning: the dashboard's `_read_tail_with_growth` (app.py
133-168) uses this helper for `telegram_messages.jsonl` which contains
non-ASCII (Swedish, emoji). On large files where the doubling halts
at `max_retry_bytes == 64MB`, the function falls through to a
full-scan `load_jsonl(path, limit=limit)` — which on an 80MB+ file
defeats the entire purpose. Logs grow.

### P2-2. `load_jsonl` reads entire file before deque trimming
**File:** `portfolio/file_utils.py:104-133`

```python
container = deque(maxlen=limit) if limit else []
...
with f:
    for line in f:
        ...
        container.append(json.loads(line))
```

For an 80MB log with `limit=100`, this still parses every line.
`deque(maxlen=limit)` keeps only the last N **after parsing all N**
of every entry. The deque comment in the docstring promises "much
more efficient than load_jsonl_tail" — actually it's the opposite,
and the dashboard's full-scan fallback in `_read_tail_with_growth`
makes that the hot path on big logs.

### P2-3. dashboard `/api/mstr_loop` reads JSONL line-by-line with no tail
**File:** `dashboard/app.py:876-919`

```python
poll_path = DATA_DIR / "mstr_loop_poll.jsonl"
if poll_path.exists():
    try:
        with open(poll_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        out["last_poll"] = _json.loads(line)
                    except _json.JSONDecodeError:
                        pass
```

This iterates every line in the file (potentially 100K+ poll
snapshots) and **overwrites** `last_poll` each time. O(N) scan to
get the last entry. Use `last_jsonl_entry` from `file_utils.py:308`
which is what it exists for.

Also: same broken pattern at line 907-917 for `mstr_loop_trades.jsonl`.

### P2-4. dashboard `/api/validate-portfolio` no CSRF / origin check
**File:** `dashboard/app.py:1067-1088`

```python
@app.route("/api/validate-portfolio", methods=["POST"])
@require_auth
def api_validate_portfolio():
    data = request.get_json(silent=True)
    ...
```

The dashboard accepts the auth cookie set on
`pf_dashboard_token`. A malicious LAN page (or an XSS in any other
cookie-sharing site if `samesite=Lax` doesn't apply — Lax does block
this for cross-origin POST, so we're OK *for that vector*). However:

- The CORS handler at line 44-61 allows
  `Access-Control-Allow-Origin: http://localhost:3000` for any
  request with that Origin — including state-changing POST. If an
  attacker can get a victim's browser to load a page from
  `http://localhost:3000` (a common dev port), they can POST to
  `/api/validate-portfolio` cross-origin.
- More importantly, the validator could be a vector for
  resource-exhaustion: it accepts arbitrary JSON via `get_json` with
  no size cap. A 100MB payload validates "all errors" path slowly
  and DoS the Flask thread pool.

**Fix:** Add `request.content_length` cap. Strip the
`http://localhost:3000` allow-list entry from production.

### P2-5. config_validator masks key names in exception but logs raw
**File:** `portfolio/config_validator.py:81-84`

```python
errors = validate_config(config)
if errors:
    for err in errors:
        logger.error("config validation: %s", err)
    raise ValueError(f"config.json validation failed: {'; '.join(errors)}")
```

The error string includes the key path, e.g. `"missing required key:
exchange.secret"`. Acceptable — it does NOT leak the secret value
itself. The empty-value check at line 55-56 also stays well-behaved.

**However**, optional-key warnings at line 75 print the key path
with `logger.warning` — `golddigger.fred_api_key` shows up in logs
verbatim. Not the value, just the path. Low risk.

The bigger issue: `validate_config` has NO type or value validation
beyond "key exists and isn't empty string". A `telegram.chat_id`
that is the string `"my chat"` instead of an int passes validation,
and the actual Telegram API call dies far downstream with a 400.
Similarly `alpaca.key` accepted as a list, a dict, an int — anything
that isn't an empty string passes.

### P2-6. config_validator config-path is symlink, no symlink-target validation
**File:** `portfolio/config_validator.py:13`

```python
CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.json"
```

CLAUDE.md states `config.json` is a symlink to
`C:\Users\Herc2\.config\finance-analyzer\config.json` and warns
`NEVER commit config.json`. `Path.resolve()` on a symlink resolves
the link. If the symlink is dangling (target moved/renamed),
`load_json` returns None and the loop refuses to start (line 67-68).
The error message ("config.json not found or unreadable") doesn't
tell the user the target is missing — the user will check the
symlink itself, see it exists, and be confused. Logging
`os.readlink(CONFIG_FILE)` in the error path would save 30 minutes
of debugging.

### P2-7. shared_state `_RateLimiter` race on first call
**File:** `portfolio/shared_state.py:255-280`

```python
def __init__(self, max_per_minute, name=""):
    self.interval = 60.0 / max_per_minute
    self.last_call = 0.0  # epoch-0 → "elapsed" is huge → first call free
```

`last_call = 0.0` means the first call ever sees `elapsed = now - 0`
which is ~1.7 billion seconds — far more than `interval`. So the
first call is always free, and `self.last_call = now`. That's the
intended behavior. **But** in a multi-threaded burst, 8 threads can
all enter on `last_call == 0.0`, all compute `wait_time = 0.0` (no
sleep), and all set `last_call = now` to the same wall clock —
firing 8 simultaneous Alpaca / Binance API calls in the first cycle
before the limiter ever engages.

The fix comment claims atomicity ("the next thread to enter will see
this and wait longer"), but the **first 8 threads** all see
`last_call ≤ now - interval` and bypass the limiter. After that the
limiter spaces them out correctly.

The Binance rate is 1200/min, sustained burst of 8 in 1ms is fine.
Alpaca 200/min — also fine. Alpha Vantage 5/min — **NOT fine** if 8
threads hit it in burst. The limiter is constructed with `5`. The
first 8 calls go through; tokens 6, 7, 8 trip Alpha Vantage's
HTTP 429 immediately.

**Fix:** Init `last_call = time.time()` so the first call respects
the interval and bursts are serialized.

### P2-8. shared_state `_loading_keys` leak on `_cached_or_enqueue` exception in stale_available check
**File:** `portfolio/shared_state.py:158-215`

```python
with _cache_lock:
    if key in _tool_cache and now - _tool_cache[key]["time"] < ttl:
        return _tool_cache[key]["data"]
    ...
    if should_enq and enqueue_fn and context is not None and key not in _loading_keys:
        _loading_keys.add(key)
        _loading_timestamps[key] = time.time()
        try:
            enqueue_fn(key, context)
        except Exception as e:
            _loading_keys.discard(key)
            _loading_timestamps.pop(key, None)
            ...
```

The cleanup is only in the `try/except` around `enqueue_fn`. The
**eviction of stuck loading keys** in `_cached` (line 68-75) tries to
reclaim these via `_LOADING_TIMEOUT`, but that path is ONLY in
`_cached`, NOT in `_cached_or_enqueue`. So if `_cached_or_enqueue`
adds a key that never gets resolved (e.g. flush_llm_batch crashes
between `_loading_keys.add` and `_update_cache`), the key stays
loading forever in this codepath. A subsequent `_cached_or_enqueue`
call sees `key in _loading_keys` is True, but the early-return on
`stale_available` path makes that benign — the system just always
returns stale. Subtle bug class: stale-forever instead of refresh.

### P2-9. shared_state cache eviction may discard fresh entries
**File:** `portfolio/shared_state.py:54-66`

```python
if len(_tool_cache) > _CACHE_MAX_SIZE:
    expired = [k for k, v in _tool_cache.items()
               if now - v["time"] > v.get("ttl", 3600) * _MAX_STALE_FACTOR]
    for k in expired:
        del _tool_cache[k]
    if len(_tool_cache) > _CACHE_MAX_SIZE:
        sorted_keys = sorted(_tool_cache, key=lambda k: _tool_cache[k]["time"])
        evict_count = len(sorted_keys) // 4 or 1
        for k in sorted_keys[:evict_count]:
            del _tool_cache[k]
```

The LRU fallback evicts the OLDEST 25% by `time` (which is set on
write). For caches with short TTL (e.g. `VOLUME_TTL = 300`), all
entries are oldest within minutes; LRU semantics here are actually
"longest-cached". A fresh write to a low-TTL key still gets evicted
if it's `len // 4` oldest. The effect: high-frequency low-TTL
entries (volume, fear/greed) get aggressively re-fetched, defeating
the cache.

**Fix:** Track `last_access` separately from `time` (write time) and
LRU-evict by `last_access`.

### P2-10. `_violations_recent` cross-stream resolution has comparison bug
**File:** `dashboard/system_status.py:381-385`

```python
if entry.get("invariant") == "layer2_journal_activity":
    details = entry.get("details") or {}
    trig = details.get("trigger_time")
    if trig and latest_l2_journal_ts and str(latest_l2_journal_ts) >= str(trig):
        return True
```

String-compare on ISO timestamps is correct IFF both strings use the
same timezone offset format. `trig` could be `2026-05-13T12:30:00Z`
while `latest_l2_journal_ts` is `2026-05-13T12:30:00+00:00`.
Lexicographic compare: `+00:00` vs `Z` — `+` (0x2B) < `Z` (0x5A) <
`0`-`9` digits region, so `"...+00:00" < "...Z"`. The result:
"...+00:00" timestamps compare unequal to "...Z" timestamps even
when they represent the same instant. Layer 2 silence violations
will silently fail to clear when one source writes Z-suffix and
another writes offset-suffix.

`_parse_ts` (line 730-733) does the right thing but isn't used here.

**Fix:** Compare parsed `datetime` objects, not strings.

### P2-11. dashboard SafeJSONProvider strips NaN but not Decimal
**File:** `dashboard/app.py:20-37`

`_json_safe` handles `float` NaN/Infinity but doesn't handle
`Decimal`, `numpy.float64`, `pandas.Timestamp`, or custom numeric
types. Avanza session returns `Decimal` for prices in some code
paths (per `avanza_session.py` patterns). When those flow through
unguarded, Flask's default JSON encoder raises `TypeError`, the
dashboard returns 500, the user sees "Internal server error" with no
detail.

`jsonify` then catches and re-raises; the route catch-all logs to
`logger.exception` but the user-facing response is opaque.

### P2-12. dashboard `_serve_dual_stack` race on socket close
**File:** `dashboard/app.py:2058-2092`

```python
sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("::", port))
sock.listen(128)
server = ThreadedWSGIServer("::", port, app, fd=sock.fileno())
server.serve_forever()
```

`SO_REUSEADDR` on Windows is **dangerous** — it doesn't have the
POSIX TIME_WAIT semantic; instead it allows MULTIPLE processes to
bind the same port simultaneously, with traffic going to whichever
got the most recent accept. If the dashboard restarts while another
instance is still alive, both listeners coexist and the user sees
randomly-stale data.

Use `SO_EXCLUSIVEADDRUSE` on Windows (`0x80000000` — the right
constant for the "I'm the only one" semantic). Linux's `SO_REUSEADDR`
already does the TIME_WAIT thing correctly; the Windows path needs
the exclusive flag.

### P2-13. dashboard Avanza worker queue unbounded
**File:** `dashboard/app.py:1713-1778`

`_AVANZA_REQ_Q = queue.Queue()` has no maxsize. If the underlying
Playwright session hangs or BankID auth requires re-prompt, every
incoming `/api/avanza_account` request enqueues a future, the worker
processes them slowly, and the queue grows. Each enqueued future
waits `_AVANZA_REQ_TIMEOUT_SECONDS = 25.0` then gives up — but the
worker still has the queue full of pending requests it will
eventually serve. Meanwhile clients see 25s timeouts and retry,
piling more on.

**Fix:** `queue.Queue(maxsize=5)`. Reject `503 Service Unavailable`
on `queue.Full` instead of letting the queue grow.

### P2-14. backtester: per-signal break uses old `volotal` shape silently
**File:** `portfolio/backtester.py:188-197`

```python
sig_data = accuracy_data.get(sig_name, {})
sig_acc = sig_data.get("accuracy", 0.5)
sig_samples = sig_data.get("total", 0)
from portfolio.signal_engine import ACCURACY_GATE_MIN_SAMPLES
if sig_samples >= ACCURACY_GATE_MIN_SAMPLES and sig_acc < ACCURACY_GATE_THRESHOLD:
    pass  # gated out — doesn't count in new system
else:
    sig_results[sig_name]["new_total"] += 1
```

The import inside the loop is per-iteration overhead — re-imports
are cached but the module attribute lookup happens every entry. With
~100K entries the cumulative cost matters for what's already a
multi-minute backtest.

More important: the gate check is shape-coupled to `accuracy_data`
schema. If `blend_accuracy_data` returns `accuracy` outside `0..1`
(e.g. percentage form), every signal looks gated.

---

## P3 — Low

### P3-1. file_utils.atomic_write_text/json — no `parents=True` race
**File:** `portfolio/file_utils.py:31-32, 52-53`

`path.parent.mkdir(parents=True, exist_ok=True)` then
`tempfile.mkstemp(dir=str(path.parent), ...)`. If the parent dir is
deleted by a sibling process between the two calls, mkstemp raises
ENOENT. Tiny window but real on Windows where AV scanners can
transiently revoke directory access.

### P3-2. vector_memory swallows ALL exceptions silently
**File:** `portfolio/vector_memory.py:259-264`

```python
except ImportError:
    logger.debug("chromadb not installed, skipping vector memory")
    return []
except Exception as e:
    logger.warning("vector memory error: %s", e)
    return []
```

The bare `except Exception` masks ChromaDB corruption (disk full,
index rebuild needed, version mismatch). Logger.warning is the right
level but the user has to actively look for it. A degraded mode
(zero results returned) silently coexists with the live system that
the Layer 2 prompt assumes is working.

`_entry_id` uses `hashlib.md5` on `ts` — if two entries share a
timestamp (sub-second collisions are common in burst loops), they
get the same ID, and `collection.add` with duplicate IDs raises in
some chromadb versions (silently in others).

### P3-3. notification_text format-only — no validation
**File:** `portfolio/notification_text.py`

Pure formatting helpers. `format_portfolio_context` does
`f"{patient_total / 1000:.0f}K SEK ({patient_pnl:+.0f}%)"`. If
`patient_total = NaN` (float bug upstream), this renders as `"nanK
SEK"` and Telegram shows it. Acceptable for diagnostics; the
upstream should be the gate.

### P3-4. shadow_registry timezone-naive iso parse fallback
**File:** `portfolio/shadow_registry.py:140-146`

If the stored `entered_shadow_ts` is timezone-naive (legacy entries
or hand-edited), `fromisoformat` returns a naive datetime, and the
function patches it to UTC. Fine for forward-compat, but it means
days-in-shadow calculation could be off by up to 23h if the
legacy entry was in a non-UTC timezone.

### P3-5. system_status `_color` rank-based bump uses string compare via dict
**File:** `dashboard/system_status.py:661-665`

```python
def bump(level: str) -> None:
    nonlocal severity
    rank = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    if rank[level] > rank[severity]:
        severity = level
```

If `level` or `severity` is None or `"UNKNOWN"`, KeyError. The caller
controls `level` to known values, so it's defensive only.

### P3-6. fix_agent_dispatcher AGENT_ALLOWED_TOOLS string typo
**File:** `scripts/fix_agent_dispatcher.py:58`

`AGENT_ALLOWED_TOOLS = "Read,Edit,Bash"` — passed to invoke_claude
as a string. If invoke_claude expects a list, a string is iterable
(per-character). Worth verifying against `portfolio.claude_gate`
signature.

### P3-7. dashboard `/logout` does not invalidate server-side
**File:** `dashboard/app.py:777-802`

Logout sets `max_age=0` on the cookie but does not invalidate any
server-side session token (there is none — token is config-static).
A user who logs out on phone A is still logged in on phone B with
the same cookie. Not a bug per se, but the user might assume the
behavior matches a typical webapp.

### P3-8. trading_status `_in_session` docstring lies
**File:** `dashboard/trading_status.py:277-289`

Docstring says "15:30–21:55 inclusive of open, exclusive of close",
but `SESSION_OPEN = dtime(8, 30)` and `SESSION_CLOSE = dtime(21, 30)`.
The docstring was not updated when the constants were unified (per
the 2026-05-11 comment at line 47). Confusing to a future
reader / debugger.

---

## Tests missing

1. **file_utils**: no test exercising `os.replace` failure across
   volumes (EXDEV / WinError 17). Add a test that monkeypatches
   `os.replace` to raise OSError and asserts the tmp file is
   unlinked + the original is unmodified.
2. **file_utils**: no test for `jsonl_sidecar_lock` first-creation
   race. Add a test spawning N threads that all try to acquire fresh
   lock against a non-existent lock path simultaneously and assert
   none corrupt the seed file.
3. **process_lock**: no test for dead-PID recovery (because the
   feature doesn't exist).
4. **subprocess_utils**: no test that asserts `CLAUDECODE` is
   scrubbed before invoking claude (because the scrub doesn't
   exist).
5. **auth.py**: no test that a `Cf-Access-Authenticated-User-Email`
   header from a non-CF source IP is rejected — because that check
   doesn't exist. **Add it as a failing test** first, then implement.
6. **fix_agent_dispatcher**: test the cooldown bypass scenario where
   the state file is corrupted twice in a row; assert global block
   doesn't lift.
7. **backtester**: no test for walk-forward semantics. Add a test
   that asserts the accuracy weights used for entry at time T do
   NOT see outcomes for any entry with ts >= T.
8. **dashboard**: no test for query-string `?token=` value being
   logged in Flask's request log. Add a test verifying the token is
   not in `werkzeug` access log output, or — better — disable the
   query-token path and require Bearer.
9. **message_store**: no test for Markdown injection via news
   headlines. Add a test that crafts a `[evil](url)` headline and
   asserts the rendered Telegram body escapes the brackets.
10. **shared_state**: no test for `_RateLimiter` first-burst
    behavior — add a test that spawns 8 threads against a `5/min`
    limiter at process start, asserts at most 1 call goes through
    in the first ~12 seconds.

---

## Cross-cutting observations

1. **Defense-in-depth missing in atomic I/O.** `file_utils.py` is the
   bedrock for every other module's state. It correctly does
   tempfile + fsync + os.replace for whole-file writes, but it
   does NOT verify the written content parses back, and it does NOT
   record the failure in any way visible to the operator. Callers
   `except Exception` and continue. The system has no observable
   metric for "atomic writes that silently failed". Add a counter
   in `health_state.json` and a contract invariant in
   `loop_contract.py`.

2. **The dashboard is an externally-reachable service AND a debugging
   tool, but treats them the same.** The dashboard binds to `[::]`
   so it's reachable from the LAN by anyone with the host's IP.
   `dashboard_token` is config-static — there's no token rotation,
   no scope, no audit trail. Combined with the Cloudflare-header
   bypass, a single misconfiguration on the tunnel turns this into
   a remote pivot point. Recommend splitting into two: a
   `127.0.0.1`-bound debug surface and a CF-tunnel-only public
   surface with proper JWT verification.

3. **Auth surface inconsistency between dashboard.app and house_blueprint.**
   `house_blueprint` uses the same `require_auth` decorator, which is
   correct, but adds `send_file` of arbitrary paths under the
   configured `house_root`. `_validate_run_id` and `_validate_slug`
   use regex matchers that look tight but `_validate_slug` calls
   `secure_filename(slug)` first, then checks `cleaned == slug` —
   that means any slug that passes the regex AND is unchanged by
   `secure_filename` is accepted. The combo is OK. But the regex
   `^[a-z0-9][a-z0-9-]{2,200}$` permits names like `con` /
   `aux` / `nul` which are reserved on Windows; `secure_filename`
   does NOT block those — a request to `/house/runs/2026-05-13/con`
   on Windows will hit an unexpected error path on the file open.

4. **Backtester ships zero realism.** No fees, no slippage, no
   stop-loss, no spread, no rebate. Treats every BUY/SELL flag as
   instantly fillable at the close price. The "+X.X% accuracy
   improvement" headline number is therefore not a P&L claim — but
   the report doesn't say that. Any reader will translate "1d
   horizon: +4.2pp" to "should be +4.2% portfolio return", which is
   wrong by an order of magnitude. Add explicit "ACCURACY ≠ P&L"
   disclaimer to `print_report`, OR add a parallel P&L simulation
   layer.

5. **Critical-error journal has no integrity envelope.** Anyone with
   write access to `data/critical_errors.jsonl` can spoof resolution
   entries that silence real failures. The startup check in
   `check_critical_errors.py` trusts `resolves_ts` references
   verbatim. A trivial poisoning: tail one line with
   `{"resolves_ts": "<oldest critical's ts>", "category": "resolution",
   "level": "info"}` and the startup gate goes green forever. Acceptable
   threat model (anyone with file write access has already won) — but
   worth noting that the auto-spawn fix agent operates on the same
   trust assumption.

6. **`_RateLimiter` (P2-7) and shared_state cache are mutually
   exclusive paradigms.** The dogpile prevention in `_cached`
   correctly returns stale during refresh, but it interacts
   awkwardly with rate limiting: a thread that finds `key in
   _loading_keys` returns stale and proceeds; the loading thread is
   blocked in the rate limiter's `wait()`. If the rate limit window
   is long (Alpha Vantage 12s), every requester gets stale for 12s
   regardless of whether the loading thread is making progress.
   Acceptable, but the design intent (one fetch serves many) is
   weakened by the limiter.

7. **`shadow_registry`, `vector_memory`, `backtester`** are all
   read-mostly singletons that load state, mutate in memory, and
   write back without versioning. None use ETags or content hashes
   to detect concurrent third-party edits. The `shadow_registry`
   in particular is updated both by automated audit scripts AND by
   manual hand-edits ("seed_defaults" suggests this) — a manual
   edit between read and write is silently overwritten.

8. **fix_agent_dispatcher writes are not audit-protected.** The
   dispatcher's own writes to `critical_errors.jsonl` use
   `atomic_append_jsonl` — same primitive as the source of truth.
   So the dispatcher's `fix_attempt_skipped` /
   `fix_attempt_completed` / `fix_agent_failed` lines are
   indistinguishable in format from any source. A bug in the
   dispatcher that mistakes a `fix_attempt_skipped` line for a real
   critical (because `level: info` is the only distinguisher and
   `_find_unresolved` filters on it) — works today but if the level
   ever changes, the dispatcher could chase its own tail. Add a
   `producer` field stamped at write time to make provenance
   explicit.

9. **No log-rotation policy is visible** for the JSONL files this
   review covered. `critical_errors.jsonl`, `signal_log.jsonl`,
   `telegram_messages.jsonl`, `claude_invocations.jsonl`, all grow
   unbounded. `prune_jsonl` exists in `file_utils.py` but is not
   wired to a scheduled task. After 12 months of operation, the
   dashboard's tail-read paths and the dispatcher's full-scan paths
   degrade in O(N) — observable as creeping latency before any
   single explicit failure.

10. **Process lock metadata is purely cosmetic.** `_write_lock_metadata`
    writes pid + started + owner + custom metadata to the lock file,
    but no consumer reads it (verified by grep). The intent (per the
    P1-1 finding) is presumably for stale-detection, which doesn't
    exist. Either implement the consumer or remove the metadata
    write — write-only data is debt.
