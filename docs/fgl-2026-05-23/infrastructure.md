# Infrastructure Subsystem — Adversarial Review (2026-05-23)

Empty-baseline read-only review. The infrastructure layer is the system's
safety floor — every other subsystem trusts `file_utils`, `health`, `journal`,
`telegram_notifications`, and `log_rotation` to not lose data or silently fail.
This review found a small number of P0/P1 issues sitting under hardened
machinery, primarily around a directory-fsync gap, an unbounded journal
load on every Layer 2 invocation, a config key mismatch that silently
returns empty Binance creds, and a heartbeat lie that flips a dead loop
to "healthy" after a crash.

Files reviewed: `portfolio/file_utils.py`, `journal.py`, `journal_index.py`,
`telegram_notifications.py`, `message_store.py`, `message_throttle.py`,
`telegram_poller.py`, `notification_text.py`, `alert_budget.py`,
`logging_config.py`, `log_rotation.py`, `health.py`, `subprocess_utils.py`,
`api_utils.py`, `config_validator.py`, `process_lock.py`, `loop_health.py`,
`reporting.py`, `http_retry.py`.

---

## P0 — Critical (silent data loss or core safety property broken)

### P0-1. `api_utils.get_binance_config` reads the wrong key — returns empty creds silently

**File:** `portfolio/api_utils.py:56-60`

```python
def get_binance_config():
    config = load_config()
    ex = config.get("exchange", {})
    return ex.get("apiKey", ""), ex.get("secret", "")
```

The 2026-05-11 fix in `config_validator.py` explicitly documents that
Binance creds live under `exchange.key` / `exchange.secret`:

```python
# 2026-05-11 fix: Binance creds actually live under ``exchange.key`` /
# ``exchange.secret`` (freqtrade-style config layout...) — they have
# never been under a ``binance`` top-level section.
REQUIRED_KEYS = [
    ...
    ("exchange", "key"),
    ("exchange", "secret"),
]
```

But `get_binance_config()` reads `exchange.apiKey` (camelCase, capital `K`)
which has never existed. Result: every caller of `get_binance_config()`
silently receives `("", "")` and any Binance API call requiring auth
(signed endpoints — futures account, FAPI position queries, anything
private) will fail with HTTP 401, get re-tried, and eventually log a
warning. The unsigned spot endpoints (price/klines) keep working, which
is exactly why this has gone undetected.

Validator passes (it requires `exchange.key`); production code reads
`exchange.apiKey`; the two are decoupled. This is the same class of bug
as the March–April 2026 Layer 2 auth outage that ran undetected for
~3 weeks because the failure mode kept signing exit code 0.

**Severity:** P0. The signed-API blast radius depends on whether anything
beyond spot price polling actually uses these creds today — but the
classification is correct because the bug is invisible until the dependent
code actually exercises it. Worse: a future module that needs signed
auth will appear to "work" (compile and run) but silently get rate-limited
or rejected.

**Fix:** change `ex.get("apiKey", "")` → `ex.get("key", "")` to match the
documented config layout, and add a startup assertion in
`validate_config_file()` that `get_binance_config()` actually returns
non-empty creds when REQUIRED_KEYS are present.

---

### P0-2. `atomic_write_json` / `atomic_write_text` skip directory fsync — durable on POSIX, NOT on Windows power-loss

**File:** `portfolio/file_utils.py:32-71`

The docstring claims: *"Fsyncs before replace to guarantee durability on
power loss (H34)."* This is **partially false**. The code fsyncs the
**file** before `os.replace`, but it never fsyncs the **directory** that
contains the file. POSIX semantics require both for guaranteed durability
of the rename. On Linux/ext4 this is well-documented; on Windows NTFS,
`os.replace` is metadata-journaled so the rename itself is durable
**after** it commits, but the *fsync of file before rename* on Windows
does not flush the NTFS metadata log, so a power loss between
`f.flush()`/`os.fsync()` and `os.replace` returning can leave the
**target file** truncated or missing while the **tempfile is gone**.

This matters because every critical write in the system runs through
this — `portfolio_state.json`, `portfolio_state_bold.json`,
`health_state.json`, `crash_counter.json`, `pending_telegram.json`,
`agent_summary.json`, plus every snapshot the loop emits per cycle.

Observable failure mode: after a hard power cut, `portfolio_state.json`
could be missing entirely (tmpfile unlinked, original replaced but not
flushed to disk metadata). Recovery would land on `load_json` returning
default `{}` and the system happily booting with a wiped portfolio.

**Severity:** P0 for portfolio state files (silent total loss of cash
balance + holdings). Less critical for `health_state.json` (regenerated
on next cycle) but still wrong.

**Fix:** after `os.replace(tmp, str(path))`, on POSIX call
`fd = os.open(str(path.parent), os.O_RDONLY); os.fsync(fd); os.close(fd)`.
On Windows there is no portable directory fsync — the closest is
`FlushFileBuffers` on the volume handle which is privileged. Best
mitigation on Windows: `os.replace` is metadata-journaled and durable
after it commits, so the bigger gap is actually the *file* fsync not
covering NTFS metadata. Use `ctypes` to call `FlushFileBuffers` on the
**parent directory** handle opened with `GENERIC_READ |
FILE_FLAG_BACKUP_SEMANTICS` (requires admin on some volumes; document
as best-effort with a TODO and add an opt-out env var for tests).

At minimum: update the docstring so callers don't assume the guarantee
holds.

---

### P0-3. `journal.load_recent` reads the entire journal on every Layer 2 invocation

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
```

`layer2_journal.jsonl` is currently 1,282 lines and growing ~10-100/day
under normal Layer 2 activity. `load_recent` runs **every Layer 2
invocation** via `write_context()` (and also via `journal_index.py` which
loads the entire file again for BM25 indexing — see P0-4). Recent commits
mention the loop has had a journal load fix, but it didn't reach
`load_recent`: the function still does linear scan + JSON-parse + ISO
parse of every line, every time, regardless of `max_age_hours`.

The 60-day rotation policy in `log_rotation.py:106-112` keeps a 60-day
window; assume a steady-state journal of 10K-50K lines after a year of
operation. On every Tier 1 invocation (180s budget) this scan now eats
non-trivial wall time — and the journal index does the same work
separately.

The recent commit `63e214a8` (`fix(agent): stale reports, stdin leak,
overflow gate, journal load`) **claims** to address this but I don't see
a tail-read in `load_recent`. There is `file_utils.load_jsonl_tail`
available — it's not used here.

**Severity:** P0 — not a correctness bug today but a slow strangulation
that will degrade Layer 2 timing as the journal grows. The same code
path runs inside `heartbeat_keepalive` so a slow read here directly steals
budget from the trade decision.

**Fix:** switch `load_recent` to use `load_jsonl_tail(JOURNAL_FILE,
max_entries=max_entries * 4, tail_bytes=256_000)` and apply the
`max_age_hours` filter to the already-tailed entries. The 4x oversample
guards against the case where the last N lines are mostly out of age
window.

---

### P0-4. `journal_index.retrieve_relevant_entries` reads the entire journal — every Tier 2/3 invocation

**File:** `portfolio/journal_index.py:351-399`

```python
def retrieve_relevant_entries(signals, held_tickers, regime, prices, k=8):
    ...
    entries = []
    try:
        with open(JOURNAL_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    ...
    index = JournalIndex()
    index.build(entries)  # tokenize EVERY entry, compute BM25 IDF over all docs
```

Same problem as P0-3 but worse: this is called whenever
`config.journal.smart_retrieval` is true (default), which is **every
Tier 2/3 Layer 2 invocation**. The function reads the entire journal AND
tokenizes every entry AND computes BM25 over all documents. At 50K
entries this is several seconds of CPU per invocation.

There is no caching: each invocation rebuilds the index from scratch.

**Severity:** P0. Combined with P0-3, the per-invocation overhead today
might be 100-500ms; in 6-12 months it's seconds. The Layer 2 invocation
budget is tight (T1=180s, T2=600s) and this overhead is invisible until
it pushes a tier into timeout.

**Fix:** cap the indexed window — `entries = entries[-2000:]` after load.
And use `load_jsonl_tail` instead of full read. Even better: persist the
fitted `BM25` object between invocations and only rebuild when the
journal mtime changes.

---

## P1 — Important (latent failure modes)

### P1-1. `health.heartbeat()` writes a fresh `last_heartbeat` even when the loop is wedged

**File:** `portfolio/health.py:64-86` + `loop_health.py`

The `heartbeat_keepalive` context manager spawns a daemon thread that
writes `last_heartbeat = now()` every 60s — purely from the keepalive
thread, **independently of whether the main loop thread is actually
making progress**. Quoting the docstring:

> Layer 2 invocation can block up to 600s (T2) or 900s (T3), but
> update_health() only runs at end-of-cycle (AFTER Layer 2 returns).
> Without periodic touches the dashboard /api/health endpoint flips
> fresh→stale every triggering cycle...

The intent is reasonable, but the implementation lies: if the **main
loop thread** crashes inside the Layer 2 subprocess call (or wedges in
a C extension, or deadlocks on a mutex), the keepalive daemon thread
keeps ticking heartbeats. The thread is `daemon=True` so it dies when
the process dies — but if the process **stays alive** (hung subprocess
waitpid, GIL contention, or the parent waiting on a child that never
exits), the heartbeat continues to advance and `check_staleness` returns
`is_stale=False`, indicating health.

This is the exact silent-failure mode CLAUDE.md flags: *"the ~3-week
silent Layer 2 auth outage of March–April 2026 that went undetected
because `claude -p --bare` exited 0 while printing 'Not logged in'."*
Heartbeats that lie about liveness are the same class of problem.

**Severity:** P1. The watchdog/auto-restart relies on staleness detection;
a heartbeat that ticks when the loop is wedged means the watchdog never
fires.

**Fix:** the keepalive should write a *counter* (cycle id from the main
thread captured at `__enter__`) along with the timestamp, and
`check_staleness` should also verify that the **main-loop cycle counter**
has advanced within a longer window (e.g. 30 minutes). If the heartbeat
ticks but the cycle counter is stuck, the loop is wedged. Today that's
indistinguishable from healthy.

---

### P1-2. `process_lock.acquire_lock_file` writes metadata **after** acquiring lock — windowed corruption

**File:** `portfolio/process_lock.py:39-47`

```python
fh = path.open("a+", encoding="utf-8")
try:
    _lock_file(fh)
except OSError:
    fh.close()
    return None

_write_lock_metadata(fh, owner=owner, metadata=metadata)
```

`_write_lock_metadata` does `fh.seek(0); fh.truncate(); fh.write(...);
fh.flush()` but never `os.fsync`. If the holder crashes between truncate
and write, the lock file is empty — observers (`scripts/check_*`)
reading the lock file get no useful metadata about who's holding it.

Also: `fh.open("a+")` then `seek(0); truncate()` — on Windows the file
position semantics for `a+` mode are platform-quirky (writes always go
to end regardless of seek). The seek/truncate pattern works in practice
but is fragile.

**Severity:** P1 — diagnostic, not functional. The lock itself works
fine; recovery diagnostics suffer.

**Fix:** add `os.fsync(fh.fileno())` after the flush, and document that
lock metadata is best-effort.

---

### P1-3. `_do_send_telegram` retries WITHOUT `parse_mode=Markdown` — but doesn't escape Markdown-special chars in the user-facing dynamic strings

**File:** `portfolio/message_store.py:147-164` and `telegram_notifications.py:60-80`

When Telegram returns 400 with a Markdown parse error, the code retries
without `parse_mode`. Two problems:

1. The **unformatted retry path doesn't log that fallback formatting was
   used** beyond a single WARNING line. If the original message contained
   intentional Markdown (e.g. `*ALERT*`), the user receives raw asterisks.
   This is a UX issue: alerts arrive but look broken.

2. The retry-without-formatting code path treats this as success (returns
   `True`) — but the JSONL store records `sent=True` for the **original
   Markdown-formatted text**, not what was actually sent. Truth-in-store
   is violated.

**Severity:** P1. Low blast radius but the JSONL is consumed by dashboard
endpoints (`/api/telegrams`) and treated as a fidelity log.

**Fix:** when falling back, append a sentinel field like
`{"fallback": "no_parse_mode"}` to the entry so the dashboard can mark it.
Or pre-escape the text on the retry and store the actually-sent string.

---

### P1-4. Crash-counter suppression cliff: `_consecutive_crashes > _MAX_CRASH_ALERTS` (=5) then nothing until #100

**File:** `portfolio/main.py:1145-1163`

```python
if _consecutive_crashes > _MAX_CRASH_ALERTS:
    logger.error("Crash #%d (alerts suppressed after %d)", ...)
    if _consecutive_crashes % _CRASH_SUMMARY_INTERVAL == 0:  # =100
        try:
            ...
            send_or_store(text, config, category="error")
```

User explicitly asked: *"What happens at crash #6?"* — Answer: it goes to
`logger.error` only. No Telegram. The next Telegram is at crash #100. At
the typical exponential backoff (10s → 5 min cap) that's roughly 6 hours
of silence in a wedge-and-restart loop.

This is too conservative. If the loop is crashing at the floor cap (5min)
× 95 crashes = ~8 hours of total silence. Combined with `PF-FixAgentDispatcher`
which has its own exponential backoff (30m → 2h → 12h → disabled per
CLAUDE.md), there is a window where both the alert path AND the auto-fix
path go quiet simultaneously.

**Severity:** P1. The system can be down for hours with zero notification.

**Fix:** lower the summary interval to 20-30, OR escalate to a `critical_errors.jsonl`
entry on crash #6 so the startup check picks it up on the next session.

---

### P1-5. `atomic_append_jsonl` releases the lock BEFORE the file write returns to the OS — torn-line risk under stress

**File:** `portfolio/file_utils.py:269-292`

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

This is mostly correct — the sidecar lock guards the write window. But
one edge case: `open(path, "ab")` on Windows with FILE_APPEND_DATA
guarantees atomic append per the docs **only if the write is a single
syscall AND smaller than 4KB**. Layer 2 journal entries are routinely
larger than 4KB (full debate field, multi-ticker thesis). When the
write exceeds 4KB, NTFS doesn't guarantee atomicity, but the
`jsonl_sidecar_lock` does serialize concurrent writers — so the
**inter-process** torn-line risk is mitigated.

The **intra-process / out-of-band** risk remains: if any code path
appends to a JSONL file via raw `open(..., "a")` without taking the
sidecar lock, it can write between a locked writer's chunks. Grep
confirms ~45 files use `atomic_append_jsonl` — there are probably some
that don't. A defensive audit is warranted.

**Severity:** P1.

**Fix:** add a CI check that flags raw `open(*.jsonl, "a")` or
`with open(*.jsonl, "a")` patterns outside `file_utils.py`.

---

## P2 — Minor

### P2-1. `message_store._COMMON_MOJIBAKE_REPLACEMENTS` has duplicate keys (silent overwrite)

**File:** `portfolio/message_store.py:37-49`

```python
_COMMON_MOJIBAKE_REPLACEMENTS = {
    "Â·": "·",
    "â": "—",
    "â€“": "–",
    "â": "'",       # duplicate key — overwrites previous
    "â": "'",       # duplicate
    'â': '"',       # duplicate
    'â': '"',       # duplicate
    "â": "→",       # duplicate
    "â": "↑",       # duplicate
    "â": "↓",       # duplicate
    "Â": "",
}
```

The literal `"â"` appears 7 times as a key. Python dict literal
semantics: last write wins, and previous mappings are silently lost. The
mojibake repair table effectively only has 3-4 entries.

**Severity:** P2. Cosmetic — Telegram messages get the wrong unicode
substitution sometimes.

**Fix:** the source bytes for these replacements were apparently lost
when the file was saved as UTF-8 and re-edited; reconstruct from the
actual mojibake examples in `telegram_messages.jsonl` archives. The
intended distinctions (em-dash, en-dash, left/right quotes, arrows) need
different source byte sequences. Probably each was originally `Ã¢â\x80\x94`,
`Ã¢â\x80\x93`, etc. encoded into a single `â` character through repeated
UTF-8/Latin-1 round-trips.

---

### P2-2. `config_validator.validate_config_file` accepts missing chat_id as long as the *key* is present (whitespace not stripped before validation)

**File:** `portfolio/config_validator.py:43-57`

The empty-string check uses `not obj.strip()` but only when `obj` is a
string. The `telegram.chat_id` in CLAUDE.md / production is often an
integer literal in config.json. If a user accidentally sets
`"chat_id": 0`, validator accepts it (0 is not an empty string), and
every Telegram send fails with HTTP 400 "chat not found".

**Severity:** P2. Localized to ops error.

**Fix:** add a type-aware emptiness check: integers must be non-zero,
strings must be non-empty.

---

### P2-3. `telegram_notifications.send_telegram` is the wrong abstraction for the wrong audience

**File:** `portfolio/telegram_notifications.py:35-81`

Two near-identical send functions exist: `send_telegram` (here) and
`message_store._do_send_telegram`. The former is gated by
`telegram.layer1_messages` config; the latter is the canonical send
path. Per the module comments, `send_telegram` is disabled by default
(layer1_messages=false) — but it remains called from a few legacy
paths. The risk: future contributors will use whichever they discover
first and skip the per-category routing in `send_or_store`.

**Severity:** P2 / housekeeping.

**Fix:** delete `send_telegram` and have callers go through `send_or_store`.
Or rename it to `_legacy_send_telegram` and add a deprecation comment.

---

### P2-4. `log_rotation.rotate_text` truncates the source file with `open(filepath, "w")` — race window vs writers

**File:** `portfolio/log_rotation.py:431-440`

```python
if compress:
    _gzip_file(filepath, rotation_1_gz)
else:
    shutil.copy2(filepath, rotation_1)

# Truncate the original file (creates a fresh empty file)
with open(filepath, "w", encoding="utf-8") as f:
    f.write("")
```

`agent.log`, `loop_out.txt`, `golddigger_out.txt` are continuously
written by long-running processes. Between `_gzip_file` (which reads
all bytes) and `open(filepath, "w")` (which truncates), any writer
appending to the file will lose those bytes — they were written to the
file but not picked up by the gzip pass. Tiny window in practice (ms),
but for `loop_out.txt` (very chatty), data loss is essentially
guaranteed per rotation.

JSONL rotation uses the sidecar lock to prevent this; text rotation
does not.

**Severity:** P2 — diagnostic logs only, no trading impact.

**Fix:** use a shared sidecar lock pattern, OR accept the loss and
document it.

---

### P2-5. `subprocess_utils._run_with_job_object` doesn't decode stdout/stderr — caller-visible bytes/str surprise

**File:** `portfolio/subprocess_utils.py:113-153`

`_run_with_job_object` constructs `subprocess.CompletedProcess(args, returncode,
stdout, stderr)` returning whatever `proc.communicate` returned. That is
**bytes** unless the caller passed `text=True` (which lands in `popen_kwargs`).
Most callers pass `capture_output=True, text=True` — fine. But callers that
pass only `capture_output=True` get bytes here, while the plain
`subprocess.run` fallback respects the absence of `text` and also returns
bytes. So consistent — but the helper docstring claims it's a "drop-in
replacement" which it almost is. Document the edge case where `input=...`
is bytes vs str: there is no validation here.

**Severity:** P2.

---

### P2-6. `journal._detect_warnings` indents `warnings.append` one level too deep — only matches some whipsaws

**File:** `portfolio/journal.py:247-254`

```python
for i in range(len(actions) - 2):
    a1, t1 = actions[i]
    a3, t3 = actions[i + 2]
    if t1 and t3 and t1 == t3 and ((a1 == "BUY" and a3 == "SELL") or (a1 == "SELL" and a3 == "BUY")):
            warnings.append(
                f"{strat}: whipsaw on {t1} ({a1}→{a3} within 3 entries)"
            )
```

The `warnings.append(...)` is indented at 16 spaces — one extra level
beyond the `if`. Python tolerates this (inside the `if` block, extra
indent is allowed for a single statement) but it's a code smell. Confirmed
it parses and runs; the logic is intact. No bug, just lint-grade.

**Severity:** P2 / style.

---

## P3 — Nits

- **`file_utils.atomic_append_jsonl` docstring typo** (line 283): "Unxfails" should be "Un-x-fails" or similar — clearly a stray character.
- **`logging_config.setup_logging`** uses a module-level `_configured` flag without a lock — racy if first call happens from two threads simultaneously. In practice setup is called once at import; not an issue.
- **`api_utils.load_config`** uses `with open(...) as f: _config_cache = json.load(f)` — violates CLAUDE.md rule 4 ("never raw json.loads(open(...).read())"). Should use `load_json` from `file_utils` for consistency.
- **`http_retry.fetch_with_retry`** retries 429s using the response's `retry_after`, but then adds `random.uniform(0, wait)` jitter — doubling effective backoff. Probably intentional but the 429-specific path should respect Telegram's exact `retry_after` instead of jittering it.
- **`process_lock.acquire_lock_file`** opens with `"a+"` then seeks to 0 and truncates — the `a+` mode on POSIX has well-defined seek-then-write semantics that ignore seek for writes. Works on Windows by accident. Use `"r+"` or `"w+"` to be explicit.
- **`reporting._track_module_outcome`** holds a module-level `threading.Lock` but `should_escalate` is mutated outside the lock — minor race; the worst case is a duplicate critical_errors entry, not a missed one.

---

## Summary (5 lines)

The atomic-write floor (`file_utils.py`) is solid in principle but has
two gaps: a missing directory fsync that breaks the power-loss durability
claim, and a sidecar-lock contract that's not enforced on all JSONL
writers. P0-1 (Binance config key mismatch: `apiKey` vs `key`) silently
returns empty signed-API creds. The journal subsystem reads the full
1,282-line file on every Layer 2 invocation (P0-3) and BM25-indexes it
again on every smart-retrieval invocation (P0-4) — both should use
`load_jsonl_tail` instead. Heartbeat liveness can lie when the main loop
is wedged because the keepalive daemon ticks independently of cycle
progress (P1-1). Crash-alert cliff (#5→#100) creates multi-hour silent
windows after suppression triggers (P1-4).
