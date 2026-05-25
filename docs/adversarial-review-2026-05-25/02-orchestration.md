# Orchestration Adversarial Review

Scope: portfolio/main.py, agent_invocation, autonomous, trigger,
trigger_buffer, claude_gate, loop_contract, loop_health, loop_processes,
market_timing, shared_state, multi_agent_layer2, perception_gate,
escalation_gate, escalation_router, subprocess_utils, process_lock.

## P0 findings

portfolio/agent_invocation.py:1163: Layer 2 Claude subprocess spawned
WITHOUT tree-kill flags. The Popen call passes neither
`creationflags=CREATE_NEW_PROCESS_GROUP` nor `start_new_session=True`,
unlike `claude_gate._popen_kwargs_for_tree_kill()` which exists for
exactly this reason. The kill path uses `taskkill /F /T /PID` which on
Windows follows the kernel parent-child tree, so it *usually* works —
but only if the process tree relationships are intact. On Unix
`_agent_proc.kill()` only kills the direct child. The Claude CLI spawns
Node.js helpers (MCP servers, llama-completion.exe via Bash tool, etc.);
a hung tree leaves orphans that `kill_orphaned_llama` has to mop up
reactively. The 3-week silent auth outage cited in CLAUDE.md was the
same class of bug — subprocess invariants assumed without verification.
Spawn with the same kwargs claude_gate uses.

portfolio/agent_invocation.py:629-737 (`_kill_overrun_agent`): When
`taskkill` fails AND `_agent_proc.wait(15)` also times out, kill_ok is
False and the code intentionally KEEPS `_agent_proc` set to "block
respawn". This is correct for spawn blocking but creates an INFINITE
TIMEOUT LOG LOOP: the completion watchdog runs every 30s, calls
`_check_agent_completion_locked`, sees `poll() is None`, sees elapsed >
timeout, calls `_kill_overrun_agent` again — which logs another
"timeout" entry to invocations.jsonl and runs another 15s wait, all
inside `_completion_lock`. The main loop's `run()` is also blocked from
making any new invocations because the same lock is held. Outcome on an
unkillable Node process: invocations.jsonl bloats with duplicate
timeout rows every ~45s indefinitely, Layer 2 is wedged forever, no
alert because `_log_trigger` doesn't fire on repeat timeouts. Need:
(a) `_kill_overrun_agent` should record the wedged-pid once and refuse
to re-attempt taskkill for that pid within a cooldown; (b) emit a
critical_errors.jsonl entry on the first kill failure so the fix-agent
sees it; (c) suppress duplicate timeout rows in the watchdog tick when
pid is unchanged.

portfolio/agent_invocation.py:716-731 (timeout path): `_log_trigger`
writes status="timeout" but if `_scan_agent_log_for_auth_failure`
returned True, the only record of auth failure is in
critical_errors.jsonl. The auth-error cooldown logic in `invoke_agent`
(line 766-783) checks invocations.jsonl for `status == "auth_error"`
and skips spawns for 30 minutes. After a timed-out auth-error, this
cooldown does NOT engage — the next trigger immediately re-spawns
another Claude that will also auth-fail. The fix is to upgrade the
log status to "auth_error" (or "auth_error_timeout") when the scan hits,
so the cooldown gate sees it. The whole point of the 2026-05-13
cooldown gate is to stop the "20-bursts-in-30-min" auth_error storm; a
timed-out auth failure defeats it.

portfolio/main.py:1108-1130 (crash counter): `_consecutive_crashes` is
loaded from disk at module import time and only reset in
`_reset_crash_counter()` which runs after a successful `run()`. If the
loop crashes BEFORE reaching `run()` (config validator throws, singleton
lock acquire returns False non-zero exit, Telegram poller construction
explodes, initial heartbeat write fails), Task Scheduler restarts the
process and the counter only grows. After 5 such pre-`run()` crashes,
all alerts go silent — exactly the failure mode the comment at line 35
says we must NEVER have. Summaries every 100 crashes (line 1151) help
but the 100-cycle gap between alerts could be 100 * (10s..5min backoff
= 16min..8h) of silent failure. The CLAUDE.md outage was exactly this
class of "loop appears to run, nothing surfaces" issue. Fix: reset
counter on `_acquire_singleton_lock()` success (acquired a fresh
instance = healthy process boot) OR check whether last persisted
counter is from a "fresh process" and decay it.

portfolio/multi_agent_layer2.py:127-242 (specialist spawn + kill):
Specialists are launched in parallel (3 concurrent Claude subprocesses)
WITHOUT tree-kill flags, then killed on timeout with bare `proc.kill()`
(line 220). On both Windows and Unix this leaves grandchild Node
processes orphaned. Combined with the `proc._log_fh = log_fh` /
`proc._name = name` attribute-poke pattern (line 188-189) which relies
on Popen accepting arbitrary attributes — works on CPython, not
guaranteed elsewhere — and the absence of any process-tree cleanup, a
single multi-agent T2/T3 timeout can leak 3-6 Node processes holding
~500MB RAM each. Over a day of T2 timeouts this exhausts RAM. The
existing `kill_orphaned_llama()` reaper specifically targets
llama-completion.exe, not Claude CLI Node processes. Use
`_popen_kwargs_for_tree_kill()` and call a taskkill /T /F on each
specialist pid on timeout.

## P1 findings

portfolio/claude_gate.py:805 (return tuple mismatch): `invoke_claude_text`
docstring promises `(text, success, exit_code, status)` — 4 values —
and the return statement at line 805 returns 4 values. OK. But line 690
declaration `def invoke_claude_text(... ) -> tuple[str, bool, int]:` is
annotated as 3 values. Callers reading the type annotation will unpack
3, getting a TypeError or worse a wrong status assignment. Annotation
must be `tuple[str, bool, int, str]`.

portfolio/agent_invocation.py:805-839 (reentrancy check under lock): The
`_completion_lock` is held during the 15-second `_agent_proc.wait(15)`
inside `_kill_overrun_agent`. While the lock is held, the completion
watchdog cannot run, the main loop's `check_agent_completion` cannot
run, and any new `invoke_agent` call cannot start. So the entire
orchestration freezes for 15s every time a timeout fires. With
overlapping triggers this can chain to multiple 15s freezes per cycle.
Drop the lock around the `wait()` call, or use a separate
"kill_in_progress" sentinel so other callers can early-return without
acquiring the lock.

portfolio/agent_invocation.py:1163 (Layer 2 Popen ignores
`PF_HEADLESS_AGENT` race): Lines 1158-1170 set `_agent_start`,
`_agent_start_wall`, `_agent_timeout`, `_agent_tier`, `_agent_reasons`
BEFORE `subprocess.Popen` and then `_agent_proc` immediately after.
Comment at 1152 says this prevents the watchdog from observing a live
`_agent_proc` with stale metadata. But there is NO lock around this
block. The watchdog runs in a separate thread, takes `_completion_lock`
on each tick. Between Popen returning and the assignment
`_agent_proc = subprocess.Popen(...)`, the watchdog could already be
checking — but it would see `_agent_proc` from the PREVIOUS invocation
(or None). Mostly safe, but `_agent_log = log_fh` is on line 1171 AFTER
`_agent_proc`. If the watchdog fires between 1163 and 1171, it sees
the new `_agent_proc` but stale `_agent_log` — `_kill_overrun_agent`
will close the old log handle, and `_agent_log = None`, leaving the
new spawn writing to a leaked handle. Take `_completion_lock` for the
whole spawn block.

portfolio/main.py:1163 (`_agent_proc` access not under lock at spawn):
The `_agent_proc` variable is mutated outside `_completion_lock` in the
spawn path (line 1163), while the watchdog and `check_agent_completion`
both read it under the lock. Python attribute assignment is atomic, but
the read-modify-write of `_agent_log` immediately after is not. Risk:
watchdog tick interleaves between `_agent_proc=Popen` and
`_agent_log=log_fh`, sees a live proc but no log handle, calls
`_scan_agent_log_for_auth_failure` which reads from `_agent_log_start_offset`
on disk — that's fine, but then on subsequent kill `_agent_log` is
None so the close path is skipped. Minor leak, but the broader race
exists.

portfolio/escalation_gate.py:202-219 (per-call ThreadPoolExecutor): Every
call to `should_escalate` creates a fresh `ThreadPoolExecutor(max_workers=1)`
and never shuts it down with `wait=True`. On timeout the executor is
shut down with `wait=False, cancel_futures=True`, but the actual
in-flight ministral query thread (`_default_runner`) is NOT cancelled —
Python can't cancel a running thread. So every timeout leaks one thread
that may still hold the llama-server GPU lock until it actually returns.
Combined with `should_escalate` being called from `main.run()` on every
trigger, a stuck ministral server bleeds threads continuously. Reuse a
module-level executor or guard with a global "in-flight" sentinel.

portfolio/escalation_gate.py:160 (deprecated `utcnow`):
`_dt.datetime.utcnow().isoformat() + "Z"`. `utcnow()` is deprecated in
Python 3.12 and slated for removal. Use `datetime.now(UTC).isoformat()`.

portfolio/trigger.py:264 (state silently corrupted by startup): On the
startup grace iteration, the code rewrites
`state["triggered_consensus"][ticker] = sig["action"]` for every ticker
including HOLDs. After grace, a ticker showing HOLD is recorded as HOLD
in `triggered_consensus`. Then later when the same ticker reaches BUY
from HOLD, the consensus crossing fires correctly. That's fine. But the
grace path at line 248 also writes `state["last"]["signals"] = {...}`,
`state["last"]["prices"]`, `state["last"]["fear_greeds"]`,
`state["last"]["sentiments"]` and skips updating `state["last_full_review_time"]`.
Then `classify_tier` sees no `last_trigger_date` for today and returns
3 ("first real trigger of the day") on the very next trigger. So
restarting the loop during market hours effectively forces a Tier 3 on
the next trigger. Whether that's intentional is unclear from the
comments; it does spend Claude budget on what may be a routine trigger.

portfolio/trigger.py:179-196 (`_save_state` prune): Pop of
`_current_tickers` happens regardless of branch (line 195), but
`triggered_consensus` is only pruned when `current_tickers is not None
and len(current_tickers) > 0`. If `_current_tickers` is set to an empty
set, we skip prune and proceed to save. Good. But if a caller calls
`_save_state` directly without setting `_current_tickers`, the prune is
skipped entirely (since `current_tickers is None`). No production
caller does this, but `check_triggers` is the only entry point — if
that contract changes, baseline pruning silently regresses. Add a
comment or guard.

portfolio/subprocess_utils.py:132 (Job Object race): `AssignProcessToJobObject`
runs AFTER `Popen` returns, meaning the child has already begun
executing. If the child spawns grandchildren BEFORE the assignment
lands (very fast Node startup hitting a script that immediately exec's
something), the grandchildren escape the Job Object. The standard fix
is `CREATE_SUSPENDED`, assign, then `ResumeThread`. Probably rare in
practice (Node initialization takes ~100ms before any user code) but
the comment claiming "automatically killed if the parent dies" is too
strong as written.

portfolio/subprocess_utils.py:320-325 (`kill_orphaned_llama` PID-reuse):
"parent_alive" check is `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION,
False, ppid) → handle`. This returns a handle for any reachable
process, including a freshly recycled PID that happens to match. So a
genuinely orphaned llama-completion.exe whose parent died and whose PID
got recycled by an unrelated process appears "alive" and is NOT
killed. Use `OpenProcess` + `WaitForSingleObject(handle, 0) ==
WAIT_TIMEOUT` to verify the parent is actually running and didn't exit
recently. Alternately compare parent creation time.

portfolio/process_lock.py:25-47 (`acquire_lock_file` truncation under
lock): The flow is `path.open("a+")` → `_lock_file` → `_write_lock_metadata`
which calls `seek(0); truncate(); write(payload); flush()`. The
truncate+write happens INSIDE the held byte-0 lock, so concurrent
acquirers will fail at `_lock_file`. Good. But there is no atomic
guarantee that the metadata write survives a crash mid-write — a
half-written PID like `pi` is readable on next acquire attempt by the
process inspecting the lock file. Since the consumers just compare PID
strings (no schema), readers see garbage. Minor issue; affects only
diagnostic readers.

portfolio/main.py:1130 (`_consecutive_crashes` is module-global): Loaded
once at import and persisted on every change. If two Python processes
ever run simultaneously (e.g., singleton lock failure plus a second
instance that managed to bypass), both write to the same counter file
and race. Atomic write protects against torn writes but the counter
becomes non-monotonic. Singleton lock is the primary defense; if it
ever leaks (CLAUDE.md notes "duplicate detection" tile exists), this
file races.

portfolio/agent_invocation.py:746-752 (auto-disable counter persistence):
`_consecutive_stack_overflows` is read at module import (line 179) and
auto-disables Layer 2 after 5 consecutive stack-overflow crashes. The
counter resets only on a non-stack-overflow completion (line 1628).
There is no manual reset path other than editing
`stack_overflow_counter.json` by hand. If a transient Claude CLI bug
causes 5 stack overflows once and then resolves, Layer 2 stays
disabled forever until a non-SO completion succeeds — but no
completion can happen because Layer 2 is disabled. Deadlock. The
auto-disable should also reset after N days, or expose a reset CLI.

portfolio/loop_contract.py:2272-2326 (self-heal infinite spawn risk):
`_trigger_self_heal` calls `claude_gate.invoke_claude` which acquires
`_invoke_lock`, runs a 180s subprocess. Self-heal is gated by
`tracker.can_self_heal()` (cooldown), gated by config flag (default
off), and the audit notes 47/47 sessions timed out. If config flips
on, every CRITICAL violation triggers a 180s synchronous claude call
holding `_invoke_lock`, which blocks every other claude caller in the
process. With overlapping CRITICAL violations the loop wedges for
multiple 180s waits per cycle. Recommendation: keep default off as
documented, but if enabled, run self-heal in a background thread or
limit to one in-flight session globally.

portfolio/escalation_router.py:140-146 (`_ticker_held` reads disk per
ticker): For each candidate ticker in `reasons`, `_ticker_held` loads
BOTH portfolio_state.json AND portfolio_state_bold.json from disk. With
N triggered reasons that's 2N file reads per `should_escalate_to_claude`
call. Called on every trigger from `main.run()`. Reads are uncached,
non-atomic, and may catch atomic_write_json's temp-file rename
mid-flight (returns None / default). Cache the holdings dict per
invocation.

portfolio/claude_gate.py:343-351 (`_count_today_invocations` O(N) per
call): Loads the entire invocations.jsonl on every `invoke_claude` /
`invoke_claude_text` call to count today's entries. With 5K-entry prune
threshold and 50+ invocations per day, this is a full scan on every
call. Holds the file open during read, blocking the atomic_append. Use
a cached counter that resets at UTC midnight.

portfolio/agent_invocation.py:179 (`_load_stack_overflow_counter` race):
Module init reads the counter, but `_save_stack_overflow_counter` is
called only from inside `_check_agent_completion_locked`. If the loop
restarts mid-completion, the counter may be stale. Minor — restart
already implies trouble.

## P2 findings

portfolio/market_timing.py:13 (`MARKET_OPEN_HOUR = 7` legacy constant):
"kept at 7 (summer value)" — dead code/data per the comment. Either
remove it or replace its callers with the DST-aware function. Leaving
it invites future devs to use the wrong value.

portfolio/market_timing.py:54-64 (`_eu_market_open_hour_utc` docstring):
The docstring conflates BST and CEST as if they're equivalent. They're
not: CEST = UTC+2, BST = UTC+1. The values returned (7 in summer, 8
in winter) are actually correct for Stockholm market (09:00 CET/CEST
local), but the docstring's "08:00 local time" and "BST=UTC+1" framing
are misleading enough that a future contributor will "fix" the
working code. Rewrite to say "Stockholm market open at 09:00 local".

portfolio/shared_state.py:295 (NewsAPI counter daily reset bug-prone):
`_newsapi_daily_reset` is updated to `now` (line 335) when `<
today_start`. So between 23:00 UTC today and 01:00 UTC tomorrow, if
the function is called at 23:55 it sets reset=now=23:55. Then at
00:05 UTC `_newsapi_daily_reset (23:55) < today_start (00:00)` is
False, so the counter does NOT reset. Effectively the daily counter
only resets on the FIRST call of each new UTC day if the previous
call's `now` was BEFORE `today_start` at that earlier call. Edge case:
if `newsapi_quota_ok` isn't called between today_start of day N+1 and
some time in day N+1, when it IS called, the comparison
`_newsapi_daily_reset (prev day) < today_start (day N+1)` IS True and
reset happens. So the bug only manifests if `now` happens to fall on
a daylight boundary in a way I'm not seeing. Re-examine.

portfolio/multi_agent_layer2.py:188 (Popen attribute poking): Setting
`proc._log_fh` and `proc._name` on a Popen instance works in CPython
but is non-portable. Use a dataclass wrapper.

portfolio/autonomous.py:51 (lazy import of time inside function): Inside
`_consensus_accuracy`, `import time` is done lazily but `time` is also
the top-level `import time` elsewhere. Moves an import into a function
body unnecessarily. Cosmetic.

portfolio/main.py:1130 (singleton lock + crash counter ordering): The
singleton lock acquire happens in `loop()` at line 1273. The crash
counter is loaded at MODULE IMPORT (line 1130) which precedes loop()
by the entire import chain. If two processes start nearly
simultaneously, both load the counter, one fails singleton lock and
exits, but the second process's counter is now whatever was on disk —
which may be stale from an earlier successful run. Re-load the
counter after acquiring the singleton lock.

portfolio/agent_invocation.py:1095-1100 (model hardcoded): `--model
claude-sonnet-4-6` is hardcoded. T3 full reviews might want Opus.
Config-driven would be safer. Note: model selection is in the prompt,
so this is a real operational lever buried in code.

portfolio/trigger.py:599-648 (`classify_tier` reads disk under hot path):
`load_state()` is called at line 610 each time tier classification runs.
With a sustained trigger, this read happens per cycle. Minor — caller
can pass state.

portfolio/trigger_buffer.py:98-118 (full read+write per add): `add()`
reads the entire buffer JSON, appends one entry, writes the entire
buffer back. O(N) per call. With high trigger rates the buffer file
churns. Switch to JSONL append-only with periodic compaction during
flush.

portfolio/loop_processes.py:127-138 (substring match too loose):
`KNOWN_LOOPS["telegram_poller"] = "telegram_poller"` is just the bare
word. Any process whose command line happens to contain "telegram_poller"
(grep, log tail, this very review script) matches. Use a tighter
pattern with leading slash/separator.

portfolio/process_lock.py:39 (open mode "a+" on existing lock file):
"a+" mode positions at EOF on existing files. The metadata write does
`seek(0); truncate(); write(...)` which is correct, but the `seek(0)`
in `_lock_file` (line 64) just before `msvcrt.locking(...)` doesn't
need to be done both places. Cosmetic.

portfolio/perception_gate.py:90-95 (silent compact-summary fallback):
On missing compact, falls back to full agent_summary.json (~64KB).
Loading a 64KB JSON per trigger when the compact is missing is
expensive and there's no warning. Log a warning.

portfolio/loop_health.py:114 (`fromisoformat` Python version): Pre-3.11
`fromisoformat` doesn't accept Z suffix. Line 114 replaces "Z" with
"+00:00" which is the right fix. OK on 3.11+.

portfolio/multi_agent_layer2.py:178-180 (log file open mode "w"):
Specialist logs are truncated each spawn. If a specialist times out
and the previous run had captured useful auth-failure context, that
history is gone. Use rotation or "a" mode.

portfolio/main.py:1187 (jitter random.random() is not seeded
deterministically): Crash backoff uses `random.random()` for jitter
without seeding. Acceptable. Cosmetic.

portfolio/agent_invocation.py:805-839 (`_kill_overrun_agent` and
`subprocess_utils.run_safe` not used): The kill path uses raw
`subprocess.run(["taskkill", ...])` instead of going through
`subprocess_utils.run_safe`. So the taskkill subprocess itself is not
protected by a Job Object. Minor — taskkill exits quickly.

portfolio/shared_state.py:264 (rate limiter wait outside lock): Comment
at line 263 explains the design. The implementation is correct: each
caller reserves a future slot. Acceptable, but the math
`self.last_call = self.last_call + self.interval if wait_time > 0 else
now` can push `last_call` arbitrarily far into the future under heavy
contention. There's no upper bound. Under a burst of 8 threads with
1ms apart, the 8th thread's reserved slot is 7*interval in the future
— it sleeps that long even though throughput is well within budget.
Probably acceptable for finance, may not be for HFT.

portfolio/escalation_router.py:149-153 (`_TICKER_BLOCKLIST` incomplete):
Only blocks 5 currency codes and a handful of TA names. Patterns like
"ATR" "RSI" appear, but "MACD" "EMA" "SMA" "ADX" "OBV" etc. don't.
If a reason string says "MACD flipped BUY->SELL for XAU-USD", regex
matches "MACD" as a ticker. Then the `_ticker_held` check looks for
MACD in holdings → False → no escalation when there should be. Expand
blocklist.

## P3 findings

portfolio/agent_invocation.py:67-69 (Windows-specific exit code as
module constant): `_STACK_OVERFLOW_EXIT_CODE = 3221225794` is
hardcoded. Comment explains it's `0xC00000FD`. Move to a named
constant or platform-conditional.

portfolio/claude_gate.py:46 (`CLAUDE_ENABLED = True` module constant):
Documented as "master kill switch" but lives in source code, not config.
Toggling requires a code edit + restart. Move to config.json.

portfolio/main.py:39-49 (try/except ImportError): The msvcrt/fcntl import
fallback duplicates `portfolio/process_lock.py` which does the same.
DRY: main.py could use process_lock.acquire_lock_file directly.

portfolio/trigger.py:271 (`_startup_grace_active = False` redundant set):
Already cleared at line 267 just before. Line 271 is dead.

portfolio/loop_contract.py:1909 (`hashlib.sha256` for dedup key): SHA-256
is overkill for dedup. md5 or blake2s-16 would be faster and the
collision risk is irrelevant for non-security use.

portfolio/escalation_gate.py:117 (`_JSON_RE = re.compile(r"\{[^{}]*\}")`):
This regex does NOT match nested JSON objects (`[^{}]` excludes both
braces). If ministral returns `{"escalate": true, "obj": {"nested": 1}}`
the regex matches `{"nested": 1}` first (or fails). The fallback
`json.loads(candidate)` is tried first which usually handles it, but
on malformed output the regex path returns wrong JSON. Use a depth
counter (claude_gate.py does this correctly at line 100-127).

## Cross-cutting observations

- Two different "claude subprocess" code paths exist:
  `claude_gate.invoke_claude` (centralized, tree-killable, serialized,
  logged) and `agent_invocation.invoke_agent` (direct Popen, no tree
  kill, separate lock). The CLAUDE_GATE module docstring explicitly
  forbids direct Popen but `agent_invocation` ignores it. Either route
  Layer 2 through claude_gate (preferred — fixes all the spawn issues
  in one go) or replicate `_popen_kwargs_for_tree_kill` and
  `_run_with_tree_kill` in agent_invocation.

- Auth-failure detection lives in three places: claude_gate's `invoke_claude`
  (subprocess stdout/stderr), agent_invocation's
  `_scan_agent_log_for_auth_failure` (agent.log slice), and
  multi_agent_layer2's per-specialist scan. The first uses the
  in-process `_invoke_lock`, the second uses `_completion_lock`, the
  third has no synchronization. A simultaneous L2 + specialist auth
  failure could write 4+ records to critical_errors.jsonl for one
  underlying credential issue. dedup at the consumer (check_critical_errors)
  is needed but not verified.

- `_completion_lock` (agent_invocation) and `_invoke_lock` (claude_gate)
  are independent. A Layer 2 spawn from `try_invoke_agent` is NOT
  serialized against a concurrent claude_fundamental
  `invoke_claude_text` call. Two Claude processes can coexist —
  ~1GB RAM, 2x rate-limit usage. Probably OK on this machine but the
  invariant claimed by `_invoke_lock`'s docstring is not actually
  enforced for the highest-volume caller.

- The crash counter (main.py) and stack overflow counter
  (agent_invocation.py) and self-heal cooldown (loop_contract.py) all
  persist to JSON files with the same atomic_write_json mechanism but
  no shared schema. Adding a unified ResilienceState dataclass would
  cut duplicate counter management.

- Multiple subprocess timeouts use raw `subprocess.run(timeout=...)`
  rather than `subprocess_utils.run_safe`. The latter exists precisely
  for this. Audit and convert.

- DST handling in `market_timing.py` is correct for Stockholm
  (despite a misleading docstring) and correct for NYSE. No bugs
  found here. Could be cleaner using `zoneinfo` directly instead of
  recomputing DST boundaries.

- `loop_contract.py` is 2442 lines — by far the largest file in scope.
  Multiple responsibilities (contract definition, violation tracking,
  alerting, dispatching, self-healing). Split into modules: contract
  definitions, tracker, alerter, dispatcher.

- The trigger buffer (`trigger_buffer.py`) is documented as "added
  2026-05-15 — not yet wired into main.py". main.py:819 actually DOES
  wire it now (`if triggered and _batch_window > 0`). The doc string
  is stale.

- `_TICKER_PAT` in main.py and `_TICKER_RE` in escalation_router.py and
  `_TICKER_RE` in trigger_buffer.py and the regex in
  agent_invocation._extract_ticker are all slightly different patterns
  for the "what's a ticker?" question. Centralize.

- The escalation gate's 10s ministral timeout (line 207) is enforced
  by a per-call ThreadPoolExecutor that leaks threads on hang. Combined
  with the gate being called on every trigger, a slow llama-server
  could leak indefinitely. The gate is currently disabled by default
  (`ministral_pregate_enabled = False`).

- Telegram alert suppression for crashes "first 5 only" is documented
  in the user's CLAUDE.md memory but the implementation in
  `_crash_alert` increments BEFORE checking. So crash #5 gets
  suppressed (because `_consecutive_crashes > _MAX_CRASH_ALERTS` is
  `5 > 5` = False on crash #5, but `6 > 5` = True on crash #6). So
  alerts are sent for crashes 1-5 and crash #6 gets the
  "_Further crash alerts suppressed_" suffix. The behavior matches
  the comment "first 5 only" but the boundary semantics are subtle.

## Files reviewed

- portfolio/main.py (1532 lines) — orchestration loop, crash recovery,
  singleton lock, trigger routing, post-cycle housekeeping
- portfolio/agent_invocation.py (1724) — Layer 2 spawn, completion
  watchdog, kill path, auth-error detection
- portfolio/autonomous.py (846) — Layer 3 fallback decision engine
- portfolio/trigger.py (661) — change detection, sustained debouncing,
  tier classification, downshift
- portfolio/trigger_buffer.py (202) — 5-min reason buffer
- portfolio/claude_gate.py (841) — centralized Claude CLI gate,
  tree-kill helpers, rate-limit warnings, auth-failure detection
- portfolio/loop_contract.py (2442) — runtime invariants, violation
  tracker, alerter, dispatcher, self-heal
- portfolio/loop_health.py (235) — heartbeat rollup across loops
- portfolio/loop_processes.py (173) — duplicate-process detection
- portfolio/market_timing.py (342) — DST-aware market hours, holidays
- portfolio/shared_state.py (388) — cache, rate limiters, dogpile
  prevention, NewsAPI quota
- portfolio/multi_agent_layer2.py (255) — parallel specialist spawn
  and wait
- portfolio/perception_gate.py (95) — rule-based pre-invocation filter
- portfolio/escalation_gate.py (229) — ministral pre-gate classifier
- portfolio/escalation_router.py (269) — autonomous-vs-claude routing
- portfolio/subprocess_utils.py (337) — Windows Job Object helpers,
  orphan reaper
- portfolio/process_lock.py (107) — cross-platform file lock
