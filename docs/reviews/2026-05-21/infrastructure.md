# Infrastructure Adversarial Review — 2026-05-21

Baseline: 604f0ef1. Worktree: `Q:\finance-analyzer\worktrees\review-2026-05-21`.
Scope: portfolio/file_utils.py, http_retry.py, gpu_gate.py, shared_state.py,
message_throttle.py, message_store.py, api_utils.py, logging_config.py,
feature_normalizer.py, telegram_notifications.py, telegram_poller.py,
dashboard/app.py, portfolio/data_refresh.py (plus auth.py, cf_access.py,
log_rotation.py which they pull in).

Adversarial mindset: data corruption, auth bypass, secret leak, silent data loss
on every line.

---

## Critical / P0

`portfolio/api_utils.py:30-31`: P0: `load_config()` reads `config.json` with raw
`open()` + `json.load(f)`, bypassing `load_json()`/atomic semantics. This is
the single global config (Binance, Alpaca, Avanza, Telegram, NewsAPI, FRED keys
plus `dashboard_token`). If an external `atomic_write_json` (e.g. the
telegram_poller `/mode` handler at line 361) renames the new file across this
read on Windows, this open can hit ERROR_SHARING_VIOLATION or read a
half-written intermediate (Windows symlink target also gets replaced).
`telegram_poller.py:343` even references this exact race ("config.json is a
symlink to an external file; raw open() can race against an external
atomic_write_json rename mid-read on Windows (we've seen partial-byte reads in
agent.log)"). Inside `api_utils.load_config` itself the same bug persists.
**Fix:** Use `load_json(config_path)` from file_utils, raising explicitly when
the result is empty (the rest of the system depends on the contents).

`portfolio/api_utils.py:33-35`: P0: On any read exception the cached `_config_cache`
is preserved silently — and on first-ever load failure, the exception bubbles
only when `_config_cache is None`. **Combined with the raw-`open` race above,
a partial read that yields a non-dict (rare) would be cached and served for
the rest of process lifetime.** No mtime is updated when the read fails inside
the cached path, so the same broken cache survives mtime ticks too.

`portfolio/file_utils.py:269-292`: P0 (latent, partial mitigation present):
`atomic_append_jsonl` writes via `open(path,"ab"); f.write(data); fsync`. On
Windows, after the rename in `rotate_jsonl` (line 364) the original `f`
handle that another process held would have been invalidated *before* the
rotation got the lock — but the function unconditionally opens, writes, and
fsyncs *after* acquiring the sidecar lock, so this is correct. **However**:
file open mode `"ab"` does NOT guarantee atomic append on Windows under
concurrent processes — only POSIX O_APPEND guarantees that. The sidecar lock
saves us; if the sidecar lock file is ever deleted (an admin "cleanup" on
`data/.signal_log.jsonl.lock`) the lock degenerates into a fresh-creation
race exactly as the docstring on line 224 warns. **Fix:** Re-create the
sidecar inside the lock with `O_CREAT|O_EXCL` semantics if absent, and warn
on creation; or move to a permanent `data/locks/` directory not in the same
glob as the JSONL files.

`portfolio/file_utils.py:117-141`: P0: `load_jsonl` silently swallows
`json.JSONDecodeError` (line 138-140) and returns the parsed-so-far list. A
single corrupted line in the middle of `signal_log.jsonl` produces a quietly
truncated tail (every line *after* the corrupt one is still parsed correctly,
so this is OK), BUT in `prune_jsonl` at line 403 the corrupt line is dropped
on rewrite ("skipping malformed line") — meaning a half-written line caused
by a (different-process) crash mid-write will be permanently erased on the
next prune. The sidecar lock protects writes by `atomic_append_jsonl` and
`rotate_jsonl`, but `prune_jsonl` (line 379) does NOT take the sidecar lock
before reading lines. **Fix:** `prune_jsonl` must acquire
`jsonl_sidecar_lock(path)` for the entire read→write-tmp→replace sequence;
same divergence pattern the rotate_jsonl docstring (line 235) calls out.

`dashboard/auth.py:154-159`: P0 (mitigated upstream, but worth surfacing):
Cookie validated via `hmac.compare_digest` — good. But the cookie token is
the **raw `dashboard_token` from config.json**, the same secret used for
Bearer auth, query auth, and printed in any URL the user clicks once. If
`?token=…` ever appears in browser history, referer headers (CSRF risk
into a 3rd-party site embedding a dashboard image), or webserver access
logs upstream of Cloudflare, that secret is compromised and grants
indefinite 365-day access (line 46). **Fix:** issue a separate
HMAC-signed cookie value (e.g. `HMAC(server_secret, user_id + expiry)`)
rather than echoing the master token. At minimum, log a warning when a
client connects via `?token=`, recommending immediate logout/relogin.

`dashboard/app.py:1043-1064`: P0: `/api/validate-portfolio` is a **POST**
mutator-shaped endpoint guarded only by `require_auth`. The endpoint
itself doesn't currently mutate (it validates a body), so it's safe today —
**but** there is **no CSRF token check** anywhere in the dashboard. If the
authenticated user visits a malicious page in another tab, that page can
POST cross-origin (Content-Type `application/json` is normally a preflighted
"non-simple" request, so the actual POST is preflight-blocked — good). The
CORS handler at `dashboard/app.py:53-61` only allow-lists the 4 same-host
origins, so the preflight will fail for attackers. This *probably* prevents
exploitation today, but `Access-Control-Allow-Credentials: false` means the
cookie shouldn't be sent cross-origin anyway. **Concrete risk:** if anyone
later adds a state-mutating POST (e.g. `/api/cancel-order`) and follows the
same shape, they get an unprotected CSRF surface. **Fix:** introduce a
SameSite=Strict cookie (currently Lax — line 98), or add a `X-CSRF-Token`
header check for all POSTs.

`dashboard/auth.py:98`: P0: `samesite="Lax"` on the auth cookie permits
top-level GET cross-site sends. Combined with the `/logout` endpoint
(`app.py:778`) being unauthenticated and accepting cross-site GET, this is
a CSRF logout primitive — any link a user clicks can sign them out. Minor
in itself, but Strict is the only correct value for an auth cookie that
holds a non-rotating long-lived shared secret. **Fix:** set
`samesite="Strict"`, and either keep the legacy `?token=…` first-visit flow
or document the workflow.

`portfolio/telegram_poller.py:345-361`: P0 (mitigated by guard, but
subtle): `_handle_mode_command` writes to **`config.json`** — the symlinked
external file containing all API keys — based on an inbound Telegram
message. The chat_id filter at `_handle_update` line 180 is the only
authentication. **The BUG-210 guard (line 350) only checks `len(cfg) < 5`** —
a malicious or corrupted config with 5+ junk keys would still be written
back, destroying API keys. The risk model assumes the Telegram chat is not
compromised; if it is, the attacker can flip notification.mode but cannot
plant new keys (mode_arg is hard-validated to "signals"/"probability" on
line 339). Acceptable, but **the `len(cfg) < 5` heuristic is fragile** —
add a content schema check (verify all of `telegram`, `alpaca`, `exchange`,
`dashboard_token` keys are present and non-empty before rewriting).

`portfolio/file_utils.py:74-94`: P0 design: `load_json` returns `default`
on `OSError` (line 86). On Windows during an `atomic_write_json` rename,
the reader can hit `PermissionError` and silently get the default `None`
or `{}`. **For trade state files this means a transient AV scan or another
process holding a handle silently degrades to "empty portfolio".** Several
hot paths use `load_json(...) or {}` (e.g. `dashboard/app.py:1561`,
`1562`), masking the failure. **Fix:** distinguish "missing" (FileNotFound)
from "transient OS error" by adding a small retry loop (3 × 50ms) inside
`load_json` for OSError, OR raise a typed exception on OSError and let
critical callers (portfolio_mgr, trade_guards) decide. The lossy default
behavior is exactly the `signal_log_reconciliation` divergence pattern
that the codebase has already burned itself on (see file_utils.py:234).

`portfolio/log_rotation.py:439-440`: P0: After rotating, the original file
is **truncated by re-opening with `"w"`**, not via the atomic tempfile
swap. There is a window between `_gzip_file(filepath, rotation_1_gz)`
(line 434) and the `open(filepath, "w")` truncation where a concurrent
`atomic_append_jsonl` from another process could write to the original —
that data is then truncated to zero on line 439. The `rotate_text` path
does **not** acquire the sidecar lock the way `rotate_jsonl` does
(`rotate_jsonl` correctly holds `jsonl_sidecar_lock` for the full
read→write→replace). For `loop_out.txt`, `agent.log`, `golddigger_out.txt`
(line 62-84) this is a data-loss race. **Fix:** rotate_text needs an
equivalent lock, or use `os.rename(filepath, rotation_temp)` + create new
empty file (rename is atomic so any in-flight write goes to the old file
which becomes the rotated copy).

---

## Important / P1

`portfolio/http_retry.py:55,67`: P1: `wait += random.uniform(0, wait)`
is NOT AWS full jitter — full jitter is `wait = random.uniform(0, wait)`.
The current formula leaves wait in `[wait, 2*wait]`, providing only
50% decorrelation. After commit fd64c7cd the comment claims "full-delay
jitter" but the math is half-jitter-plus-base. **Fix:**
`wait = random.uniform(0.5 * wait, 1.5 * wait)` for symmetric or
`wait = random.uniform(0, wait)` for true full jitter. The current
implementation is fine for production but the comment is misleading and
sets up future readers to assume full jitter has been done.

`portfolio/http_retry.py:51,54`: P1: For Telegram 429 `retry_after` is
**multiplied** with jitter (line 55 `wait += random.uniform(0, wait)`
runs after retry_after replaces wait). So if Telegram says "retry in
30s", we wait between 30s and 60s. Telegram's `retry_after` is a hard
floor — exceeding it is fine, but doubling it doubles user-perceived
latency. **Fix:** apply jitter only as `+ random.uniform(0, 1)` (sub-
second) on top of Telegram's retry_after, not a multiplicative range.

`portfolio/http_retry.py:42`: P1: PUT/DELETE/PATCH methods fall through
to `requester.request(method, ...)` which does NOT pass `json_body`.
Any caller using a non-GET/POST method with a JSON body loses the body
silently. **Fix:** `resp = requester.request(method, url, headers=headers,
params=params, json=json_body, timeout=timeout)`.

`portfolio/shared_state.py:262-279`: P1 (subtle correctness bug): The
`_RateLimiter.wait()` reservation logic on line 277:
`self.last_call = self.last_call + self.interval if wait_time > 0 else now`.
If wait_time == 0 (first call ever, or after a long idle period), we set
`last_call = now`. Good. But after a burst where threads queue up, each
sees `wait_time > 0` and adds `interval` — so after N queued threads,
`last_call = old_last_call + N*interval`. The Nth thread sleeps until
`old_last_call + N*interval`. If `old_last_call` was already in the past,
threads sleep for the right amount of time relative to *each other*, but
relative to wall-clock the first thread may *over-sleep*. Not a bug, just
worth noting: this is a token-bucket-style emulation, not a true token
bucket. The "tokens" never refill while idle; a long-idle limiter
suddenly hit by 8 threads issues only 1 immediate then 7 spaced — fine,
but it doesn't deliver the "burst credit" advantage of a real bucket.

`portfolio/shared_state.py:50-52`: P1: Cache lookup checks
`now - _tool_cache[key]["time"] < ttl` but `ttl` is the caller-passed TTL,
not the TTL stored on the entry. Mismatched callers with different TTLs
for the same key (unlikely, but possible) would get inconsistent freshness
assessments. **Fix:** trust the stored TTL: `now - entry["time"] < entry.get("ttl", ttl)`.

`portfolio/shared_state.py:54-66`: P1: Cache eviction triggers only on
size > `_CACHE_MAX_SIZE` (512). With `MIN_VOTERS = 3` × 7 timeframes ×
5 tickers × ~17 signals × multiple sub-keys, 512 is borderline tight,
and the LRU fallback evicts only `len(sorted_keys) // 4` per pass —
under a sustained miss storm the cache can repeatedly hit 512 → 384 →
512 → 384 in a hot loop, *each pass holding the cache lock for the full
sort+delete*. **Fix:** raise to 2048 or evict 50% on overflow.

`portfolio/shared_state.py:88-89,200-203`: P1: `_loading_keys.add(key)`
+ `_loading_timestamps[key] = time.time()` is correct, but the dual
data structure invites desync. If a future contributor only updates
`_loading_keys` and forgets `_loading_timestamps`, the stuck-key
eviction at lines 69-74 won't fire. **Fix:** wrap in a small dataclass
or replace with a single `_loading: dict[str, float]` where presence
implies in-flight.

`portfolio/gpu_gate.py:80-83`: P1: `_pid_alive(pid)` falls back to
returning `True` when `psutil` isn't installed — meaning **stale locks
can never be reaped** if psutil is missing. The warning fires once, but
the loop then wedges forever (the 25h outage of 2026-05-02 referenced in
the module docstring is exactly this failure mode). **Fix:** on Windows
use `ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)`
fallback when psutil is unavailable; never return `True` unconditionally.

`portfolio/gpu_gate.py:213-222`: P1: `os.open(..., O_CREAT|O_EXCL|O_WRONLY)`
is atomic for the create itself, but the write of the metadata
(`f"{model_name}|{os.getpid()}|{time.time()}|{threading.get_ident()}"`)
on line 216 is a *separate* syscall. If the process is SIGKILL'd between
the `os.open` and the `os.write`, `_read_lock()` on the next caller
parses an empty string and returns `model="unknown", pid=0, ts=0`. With
`pid=0`, `_pid_alive(0)` returns False (line 75), so the lock IS broken
correctly on the next sweep. Good — but the staleness check `_is_stale()`
uses mtime, which on Windows might still be very recent for the newly
created (then crashed) file. So we wait `_STALE_SECONDS = 300` (line 35)
before reaping. **Fix:** if `_read_lock()` returns `pid=0` or empty
fields, treat as stale immediately regardless of mtime.

`portfolio/gpu_gate.py:137-152`: P1: `_sweeper_loop()` has no shutdown
hook. It's a daemon thread, so on process exit it dies. But if the
sweeper thread itself catches an exception inside `time.sleep` (e.g.
KeyboardInterrupt on Windows), the loop exits silently because
`KeyboardInterrupt` is **not** caught by `except Exception` (line 149).
With `_start_sweeper`'s singleton check (`is_alive()`), the next
`gpu_gate()` call will respawn it — but only if `gpu_gate()` is called.
In a metals_loop process that doesn't use the GPU after startup, the
sweeper dies and never respawns. **Fix:** catch BaseException
explicitly *only* for the sweeper, since `daemon=True` will still let
ctrl-C terminate the process.

`portfolio/message_throttle.py:39-41,107-111`: P1: `should_send_analysis`,
`queue_analysis`, `flush_and_send`, and `_send_now` all read+write
`pending_telegram.json` non-atomically across the check-and-send. Two
threads can both pass the cooldown gate and both call `_send_now`,
double-sending. Lock the file (or use a threading.Lock) around
`should_send_analysis` + state update. **Fix:** wrap in
`threading.Lock()` at module level.

`portfolio/telegram_notifications.py:35-81`: P1: `send_telegram` logs the
token via `_redact_url` from `http_retry.py:20` — but the URL
`f"https://api.telegram.org/bot{token}/sendMessage"` is passed straight
in. `_redact_url` uses a regex `/bot[0-9]+:[A-Za-z0-9_-]+/` (line 17 of
http_retry.py). If a token ever does NOT match `\d+:[A-Za-z0-9_-]+`
(e.g. someone tests with a fake token "TEST"), the regex misses and the
token leaks into the logs. Production Telegram tokens always match this
shape, so the risk is small. **Fix:** redact by hostname match (any
URL containing `api.telegram.org` + `/bot...` segment) rather than by
token shape.

`portfolio/telegram_notifications.py:22-29`: P1: `_MD_V1_SPECIAL`
regex covers `_*\`\[\]` — but Telegram Markdown v1 also breaks on `(` `)`
in link syntax `[text](url)`. If a reason string contains `]` followed
by `(` the parser interprets as a link; escaping `]` alone isn't enough.
**Fix:** include `(` and `)` in the escape class, or switch the entire
codebase to MarkdownV2 (which has a clearer escape set).

`portfolio/message_store.py:36-49`: P1: `_COMMON_MOJIBAKE_REPLACEMENTS`
has multiple entries with identical keys (e.g. lines 41, 42, 43 all map
the same `"â"` bytes to different glyphs depending on order, but Python
dict literal keeps only the last value). **Fix:** confirm intent — the
literal `"â"` on lines 41-47 are visually the same byte sequence in
this file; dict-literal de-dup yields only one mapping. Use byte
sequences (`b"\xc3\xa2"` etc.) and a function-based mapping, not a
literal dict.

`portfolio/message_store.py:135-141`: P1: After Markdown parse failure,
the unformatted retry uses `text` with no escape. If the message
contains characters that *could* parse as Markdown (e.g. `_` mid-word),
the retry succeeds — but the user sees raw underscores. Not a bug, just
worth noting: the retry preserves the readability win.

`portfolio/logging_config.py:43-46`: P1: `RotatingFileHandler` rotates
based on file size, but on Windows the open log file is held with
share-deny semantics. Other processes appending via the **separate**
`atomic_append_jsonl` path (not via the logger) won't conflict. But a
second `setup_logging` call from a child process would attempt to add
*another* `RotatingFileHandler` for the same file — fortunately
guarded by `_configured` module-level flag (line 14). **Caveat**: each
subprocess has its own `_configured`, so each spawns its own handler.
With multiple Python processes (main + golddigger + metals + crypto loops)
writing to the same `portfolio.log` via `RotatingFileHandler`, **rotation
events race** — process A renames `.log` → `.log.1` while process B is
mid-write to the old file (now `.log.1`). Both processes then open a
fresh `.log`, but Process B has a stale handle on `.log.1`. **Fix:** use
process-name-suffixed log files (`portfolio.{pid}.log`) or switch to
syslog/`WatchedFileHandler`.

`portfolio/logging_config.py:38`: P1: No `StreamHandler.setLevel(...)` —
inherits from root. If a future contributor sets a lower level on the
root logger globally, stdout floods. Minor.

`dashboard/app.py:53-61`: P1: CORS only includes localhost:5055/3000.
Mobile clients (the 2026-05-03 mobile-first redesign at app.py:771)
accessing via Cloudflare tunnel hit a different origin and won't get
CORS headers. If the dashboard ever moves to a separate static-frontend
hosted elsewhere, this breaks silently (preflight 200 but no headers).
**Fix:** allowlist `cf_access_team_domain` + tunnel hostname from
config.json.

`dashboard/app.py:1043`: P1: `/api/validate-portfolio` accepts arbitrary
JSON via `request.get_json(silent=True)`. Flask's default body size limit
is unlimited unless `MAX_CONTENT_LENGTH` is set. A malicious
authenticated client (or a leaked token) can POST a 4GB JSON body and
OOM the dashboard process. **Fix:** `app.config["MAX_CONTENT_LENGTH"] =
1 * 1024 * 1024`.

`dashboard/app.py:155-160`: P1: `_read_tail_with_growth` doubles
`tail_bytes` until it has `limit` entries OR `capped >= file_size` OR
`capped >= max_retry_bytes`. For corrupted JSONL with `limit=5000` and a
file of 100MB where 99% of lines are unparseable, the loop reads the full
64MB cap, finds <5000 valid rows, then falls back to `_load_jsonl_impl`
(full scan, line 168) which reads the **entire** file again. Double-read
of a 100MB file under a per-request lock means a slow request can starve
other dashboard polls. **Fix:** track parse failure rate during the
tail scan; abort early into the full-scan path if failure rate >50%.

`dashboard/app.py:80-95`: P1: `_cached_read` lock-protects writes, but
the cached return value is shared by reference. If a caller mutates a
returned dict in-place (e.g. `state["foo"] = "bar"`), every future cached
read sees the mutation. Several call sites do `_read_json(...) or {}`
which is fine, but `_normalize_golddigger_state(state, ...)` at line 297
calls `state = dict(state or {})` immediately — defensive copy is the
contract callers expect, but it's nowhere documented. **Fix:** return
`copy.deepcopy(result)` from the cache; or document the immutability
contract.

`portfolio/telegram_poller.py:113-121`: P1: `_poll_loop` catches
`Exception` and sleeps 5s. If `_get_updates` raises **before** any
response (e.g. DNS failure for `api.telegram.org`), the loop retries
forever with no exponential backoff and no rate-limit awareness. A 5s
poll = 17,280 requests/day → potential Telegram block. **Fix:**
exponential backoff on consecutive failures (5s → 30s → 5min cap).

`portfolio/telegram_poller.py:160-161`: P1: `self.offset = max(self.offset,
update_id + 1)` updates **in-memory** offset before processing. If the
process crashes during dispatch (line 247), `_save_offset` is not called,
which is the documented behavior. But if the same code path is hit by an
**incoming** Telegram update during a long-running `_handle_mode_command`
(synchronous config rewrite), the next `_get_updates` returns an empty
list (in-memory offset is already past the in-flight update). Acceptable,
just confirming the design.

`dashboard/app.py:1052-1054`: P1: `request.get_json(silent=True)` returns
`None` on parse failure; we then jsonify with 400. No size limit, no
content-type validation. With `silent=True`, Flask swallows the parse
error — so callers can't distinguish "no body" from "malformed JSON".
**Fix:** use `silent=False` + try/except, return distinguished error
codes.

`dashboard/app.py:1916-1949`: P1: The Avanza worker thread queue
(`_AVANZA_REQ_Q`) is unbounded. A burst of dashboard polls during a slow
Avanza response can queue dozens of pending snapshots; only the first
gets processed, but the others time out at 25s each AND the worker
keeps churning through them serially even after the HTTP client has
abandoned. **Fix:** if `future["done"]` is set but result is unread,
drain the queue / skip. Or use a single-slot reduction queue.

`portfolio/shared_state.py:330-336`: P1: `newsapi_quota_ok` uses
`datetime.now(UTC).replace(hour=0,...)` to compute "today_start", but
the counter is reset based on `now` (line 335). Race: thread A and
thread B both call after midnight, both call `_newsapi_daily_count = 0`,
then both increment via `newsapi_track_call`. Result is one over-count
(harmless). More concerning: `_newsapi_daily_reset = now` on line 335
is set to wall-clock `now`, but the comparison on line 333 uses
`today_start`. After the next midnight, `_newsapi_daily_reset` (=
yesterday's `now`) is < `today_start` so counter resets. That works,
but the variable name suggests it stores "the reset boundary" when it
actually stores "when we last reset". Confusing.

---

## Patterns

1. **Atomic I/O compliance is mostly enforced** — the in-scope modules
   correctly use `atomic_write_json`/`atomic_append_jsonl`/`load_json`
   except for two notable holes: (a) `api_utils.load_config` reads raw
   open() — see Atomic I/O Audit below; (b) `rotate_text` truncates with
   `open(..., "w")` rather than via the tempfile swap.

2. **Cookie/auth surface is layered but treats the master secret as the
   long-lived auth token.** Cookie SameSite=Lax permits cross-site GET;
   if a future endpoint mutates state via GET (none currently does), it's
   exploitable. Acceptable today but fragile.

3. **GPU lock recovery is well-architected** — dual-layer (thread+file),
   reactive + sweeper daemon, shared predicate. Two minor gaps: psutil
   import fallback returns True (line 82-83), and the metadata-write
   gap window between os.open and os.write.

4. **Race-condition discipline is uneven.** `shared_state._RateLimiter`,
   `_cached`, and `_cached_or_enqueue` are carefully thought through.
   `message_throttle` and `gpu_gate` sweeper are less careful.

5. **Error masking via silent default returns is the dominant data-loss
   risk.** `load_json` returns default on OSError. `_read_json` returns
   `None or {}`. `load_jsonl_tail` returns `[]` on UnicodeDecodeError.
   Each call site adds another `or {}` defensive default. Net effect:
   transient I/O errors degrade silently to empty data, which the
   trading logic then trusts as "no position". The `signal_log_reconciliation`
   contract invariant exists precisely because of this pattern.

6. **Markdown injection surface is bounded but present.** `escape_markdown_v1`
   doesn't escape parentheses; some message paths bypass escape entirely
   (e.g. `send_or_store` accepts arbitrary text with no sanitisation
   beyond mojibake repair).

7. **Telegram bot tokens are mostly redacted (commit 4ea8c46e), but the
   redaction is regex-shaped and brittle.** Tokens not matching
   `\d+:[A-Za-z0-9_-]+` would leak.

8. **dashboard/app.py is 2334 lines and houses 30+ endpoints inline.**
   Caching, normalization, business logic, and route handlers are
   intermingled. Refactor candidate: split per-bot (mstr, oil, metals,
   crypto, gold) into blueprints like the existing `house_blueprint`.

---

## Atomic I/O Compliance Audit

Files in scope that bypass `file_utils`:

- `portfolio/api_utils.py:30-31` — `open(config_path); json.load(f)`.
  Critical config file. **Must move to `load_json`.**
- `portfolio/log_rotation.py:439-440` — `open(filepath, "w")` truncate.
  No sidecar lock around `rotate_text`. **Race with in-flight appenders
  on text logs.**
- `portfolio/log_rotation.py:359-364` — tmp+fsync+replace, but inside
  the sidecar-lock the operation IS correct. Just note the manual
  re-implementation rather than calling `file_utils.atomic_write_jsonl`.
- `dashboard/auth.py:62-63` — `open(CONFIG_PATH); json.load(fp)`. Same
  config.json read race as api_utils. The auth module deliberately
  avoids importing portfolio.file_utils to keep dependency cycle clean,
  but this means **dashboard auth bypasses the atomic-read discipline**.
  A `_get_config()` race during a `/mode` rewrite could yield `{}`,
  causing `_get_dashboard_token()` → None, causing `require_auth` to
  **let the request through unauthenticated** (line 122 returns the
  unwrapped function when token is None).

Spotted while reading these files — out-of-scope but worth flagging:

- 31 files use raw `json.dump(...)` based on the grep at the start of
  the review. Several are test fixtures (acceptable). The rest:
  `scripts/write_research_outputs.py`, `scripts/pf.py`,
  `scripts/iskbet*.py`, `scripts/signal_correlation_audit.py`,
  `data/silver_monitor.py`, `data/metals_history_fetch.py`,
  `data/crypto_monitor.py`, `dashboard/export_static.py` — most are
  one-shot scripts so atomicity less critical, but
  `data/silver_monitor.py` is part of the metals subsystem which
  CLAUDE.md flags as the primary trade target. **Recommend audit pass
  on `data/*.py` writers.**

- `portfolio/api_utils.py` is imported by 14+ files including
  `http_retry`, `data_collector`, dashboard endpoints. Every load_config
  call traverses the unsafe read path.

---

## Dashboard Auth Surface

**Endpoints reviewed (33 routes, per CLAUDE.md):**

- All `/api/*` routes are decorated with `@require_auth` — confirmed via
  grep. `/api/validate-portfolio` (POST) is the only mutating-shaped
  endpoint and it's read-only today (line 1057 just calls
  `validate_portfolio` which returns errors).
- `/logout` (line 779) intentionally unauthenticated to allow cookie
  clear without auth (fine).
- `/` and `/legacy` require auth (line 757, 769).
- Static assets (`send_from_directory("static", "index.html")`) only
  reached via authenticated index route — but `app.static_folder="static"`
  means `/static/*` is served by Flask WITHOUT `@require_auth`. **P1:**
  any JS/CSS in `static/` is publicly accessible. If config-bearing data
  is ever embedded into a static file by the build process, it leaks.
  **Fix:** disable Flask's default static route and serve via auth'd
  `send_from_directory` calls only.

**Auth layers (in order, per `require_auth` at auth.py:120):**

0. `Cf-Access-Authenticated-User-Email` + `Cf-Access-Jwt-Assertion`
   header — **was P0 spoofable before commit 4ea8c46e; now JWT-verified
   against CF JWKs**. Correct.
1. Cookie `pf_dashboard_token` — hmac.compare_digest, 365-day Lax. See
   P0 above re: token == master secret.
2. `?token=` query param — first-visit flow. **Token-in-URL risk** as
   noted in P0.
3. `Authorization: Bearer` header — CLI clients. Same secret.

**Bypass risks:**

- If `dashboard_token` is unset/empty in config.json,
  `_get_dashboard_token()` returns None and `require_auth` lets the
  request through with no auth (auth.py:121-123). Combined with the
  config-read race in `_read_config_uncached` (auth.py:60), a transient
  Permission denied during a config write could open the dashboard
  publicly for the duration of the 60-second cache TTL. **P0-adjacent.**
  **Fix:** treat "no token configured" as a hard 401 rather than a
  pass-through; require explicit opt-in via `dashboard.allow_anon: true`
  flag.
- IPv6 dual-stack bind on `[::]:5055` (line 2318-2322) is intentional.
  Local NoOpsec assumption is Cloudflare Access fronts it. If the local
  host is on a LAN where another machine resolves to a routable IPv6
  address, the dashboard is exposed. No firewall rule enforces
  loopback-only listen. **P2:** document the assumption.
- CORS allowlist (app.py:44-49) is closed-list; OK.

---

## Out of Scope but Spotted

- `portfolio/data_refresh.py:30-48`: writes to `user_data/data/binance/futures/*.feather`
  via pandas `to_feather`. No atomicity. A crash mid-write leaves
  corrupt feather files. Not loaded by hot paths but downstream
  backtests will fail confusingly.
- `portfolio/data_refresh.py:31-39`: hardcodes `time.sleep(0.2)` per
  batch to space out Binance calls — bypasses the `_binance_limiter`
  in shared_state. Should use the central limiter for consistent
  rate management.
- `portfolio/feature_normalizer.py:_buffers` is module-level mutable
  state with NO lock. If `update()` is called from the 8-worker
  ThreadPoolExecutor (very likely — signals run per-ticker in parallel),
  `deque.append` is atomic but the `if key not in _buffers: _buffers[key] = deque(...)`
  on line 38-40 is a check-then-act race. Two threads can both create
  buffers and one overwrites the other, losing accumulated samples.
  **P1.** Add a lock or use `_buffers.setdefault(key, deque(maxlen=_DEFAULT_WINDOW))`.
- `portfolio/feature_normalizer.py:`: training-serving skew risk —
  the rolling buffer is in-memory only. Process restart loses all
  state. The `_MIN_SAMPLES = 20` cold-start guard means after each
  restart, 20 cycles run with raw thresholds (not z-scored). On a 60s
  loop that's 20 minutes of degraded signal logic. **P2.** Persist
  buffers to disk on shutdown OR load historical samples from
  `signal_log.jsonl` on startup.
- `portfolio/feature_normalizer.py:62-71`: `arr.std()` with NumPy's
  default `ddof=0` (population std). Z-scores using sample mean +
  population std bias slightly. Pedantic; doesn't affect signal logic.
- `portfolio/message_throttle.py:122`: `mark_trade_sent()` is a
  documented no-op. Dead code — either delete or implement.
- `portfolio/shared_state.py:330,361`: imports `datetime` and `zoneinfo`
  inside function bodies. Fine for hot-reload safety but slows
  per-call by microseconds.
- `dashboard/app.py:2289`: `from dashboard.house_blueprint import bp`
  at module-init means any failure inside house_blueprint crashes the
  dashboard process. The "house" project is unrelated to trading.
  **P2.** Wrap registration in try/except so house failures don't
  blank `/api/health`.
- `dashboard/app.py:14`: imports Flask, jsonify, etc. as top-level.
  Cold-start cost is acceptable. But `import functools, hmac, math,
  threading, time` from line 3-9 plus deferred imports inside many
  endpoints (e.g. line 1278, 1281, 1633, 1825) suggest someone
  hand-tuned import time. Not a bug, just a note.
- `portfolio/api_utils.py:8-13`: API base URLs are module-level
  constants — good. But `BINANCE_BASE` is hardcoded; no override via
  env or config. A regional failover or test endpoint can't be wired
  without code change. **P2.**
- `portfolio/telegram_poller.py:48-49`: stores `self.token` and
  `self.chat_id` on the instance. Token is in memory for the process
  lifetime; if the dashboard module ever dumps `vars(self)` for
  diagnostics (e.g. `/api/health`), token leaks. No such dump exists
  today.
- `portfolio/gpu_gate.py:33-34`: `_GPU_LOCK_DIR = Path("Q:/models")` is
  hardcoded to the user's drive. Untestable on CI without
  monkey-patching. **P2.** Move to config.

---

End of review.
