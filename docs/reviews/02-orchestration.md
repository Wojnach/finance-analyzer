# Orchestration Review

Adversarial whole-file review of the Layer 1 loop + Layer 2 `claude -p` subprocess
orchestration. Read-only from worktree `Q:/fa-rev-0531`. Focus: silent stalls,
the daily `contract_violation` ("trigger fired but no journal entry") seeded failure,
DST bugs, trigger races, crash-recovery masking, concurrency.

Files reviewed:
`portfolio/main.py`, `portfolio/agent_invocation.py`, `portfolio/loop_contract.py`,
`portfolio/trigger.py`, `portfolio/trigger_buffer.py`, `portfolio/market_timing.py`,
`portfolio/autonomous.py`, `portfolio/claude_gate.py`, `portfolio/reporting.py` (skim),
`portfolio/multi_agent_layer2.py`, `portfolio/escalation_gate.py` (skim),
`portfolio/escalation_router.py`, `portfolio/health.py` (relevant fns),
`docs/TRADING_PLAYBOOK.md` (journal format).

---

## ROOT-CAUSE HYPOTHESIS for the daily `contract_violation`

The violation fires when `check_layer2_journal_activity()` (loop_contract.py:277)
sees `health_state.last_trigger_time` newer than the newest `layer2_journal.jsonl`
entry by more than the per-tier grace window, AND none of preconditions 4/4b/6
suppress it.

The dominant cause is **timestamp provenance mismatch, not a true silent agent
failure**. The contract trusts two independently-clocked values:

1. `health.last_trigger_time` — written by `update_health()` (health.py:32) at the
   *end* of the spawning cycle, `datetime.now(UTC)`.
2. journal `ts` — written *by the Layer 2 Claude subprocess itself*, per
   `docs/TRADING_PLAYBOOK.md:212` (`"ts": "ISO-8601 UTC"`). The agent is a free-form
   LLM: it frequently stamps the entry with the **signal-snapshot time from
   `agent_summary`** (30–90s *before* the trigger) rather than wall-clock-at-write,
   and may emit a **timezone-naive** string.

This produces two distinct failure modes that both surface as the same
`contract_violation`:

- **(A) journal ts < trigger ts** even on a fully successful run. The 2026-05-30
  Precondition 6 patch (loop_contract.py:413-419) is a targeted band-aid for exactly
  this, but it only suppresses when the *latest invocation row* is
  `status=="success" && journal_written` AND `inv_ts >= last_trigger-2s`. Any run that
  lands `status="auth_error"`, `status="stack_overflow"`, a specialist `invoked` row
  (see P1-#4), or whose invocation row is older than the journal-snapshot skew slips
  through and fires.
- **(B) naive/aware comparison TypeError** (see P1-#2) — when the agent writes a
  naive `ts`, the comparison raises, the whole contract for that cycle is dropped,
  and on the *next* cycle the still-unmatched trigger re-fires the violation.

Secondary contributor: the **`auth_error` completion path writes NO journal stub**
(P0-#1). Every other terminal status (timeout/failed/incomplete) writes a stub that
satisfies the journal check; `auth_error` relies solely on the contract's status-based
suppression (precondition 4b), which is fragile to row-ordering and to the auth scan
itself missing the marker (P1-#3).

Net: the contract is correct in spirit but is comparing a machine clock against an
LLM-authored, possibly-naive, possibly-back-dated timestamp. Until the journal `ts`
is stamped by trusted host code (or the contract compares against the invocation-row
ts instead of the journal-entry ts), this will keep firing near-daily.

---

## CRITICAL (P0)

**P0-#1 — `auth_error` completion writes no journal stub; relies entirely on
status-suppression.**
`agent_invocation.py:1569-1731` (`_check_agent_completion_locked`).
Status branches `failed` (1628) and `incomplete` (1655) and the timeout path
(`_kill_overrun_agent`, 769-786) each append a journal stub so the contract sees an
entry. The `auth_error` branch does NOT. Causal chain: silent auth outage (the exact
Mar–Apr 2026 failure mode) → agent exits printing "Not logged in" → status set to
`auth_error` → no journal entry written → contract suppression depends on the
`auth_error` invocation row being the newest L2 row AND its ts being within 2s of
`last_trigger`. If a later writer appends to `invocations.jsonl` (a specialist row,
a subsequent `invoked`/`skipped_busy`, P1-#4) or the auth scan misses the marker
(P1-#3), the contract fires `contract_violation` — masking that the real category is
`auth_failure`, and worse, if suppression *does* hold there is no durable journal
record of the auth-failed trigger.
→ In the `auth_error` branch, write a journal stub identical to the
`failed`/`incomplete` stubs (`decisions: NO_DECISION`, `status:"auth_error"`,
tier/duration/exit_code). This makes the journal the single source of truth for every
terminal status and removes the dependency on row-ordering for the most dangerous
failure class.

**P0-#2 — Layer 2 auth-failure detection scans only a rotation-fragile byte slice of
a shared log, with a 16-line / start-of-line matcher that real CLI output can evade.**
`agent_invocation.py:576-637` (`_scan_agent_log_for_auth_failure`) +
`claude_gate.py:274-327` (`detect_auth_failure`) + `claude_gate.py:190-231`.
Three compounding weaknesses on the primary trading path:
(a) The main L2 subprocess runs default streaming text (no `--output-format json`,
agent_invocation.py:1156-1161) into `data/agent.log` with `stderr=STDOUT`. The scan
reads `agent.log[start_offset:EOF]`. Hourly `log_rotation.rotate_text()` runs in
`_run_post_cycle` *while a T2/T3 agent runs for minutes* (main.py:406-417). The 05-28
guard (agent_invocation.py:611-621) resets `start=0` only when `cur_size <
start_offset`; if rotation truncates and the fresh file then grows *past* the old
offset before the scan, the slice misses the early "Not logged in" preamble entirely.
(b) `_AUTH_SCAN_LINE_LIMIT=16` + `_is_real_auth_marker_line` require the marker at the
absolute start of one of the first 16 lines. Any CLI version that prints a banner,
spinner, or a blank/whitespace-prefixed line before the marker, or pushes it past
line 16, defeats detection → status silently becomes `success`/`incomplete` instead of
`auth_error` → reverts to the original 3-week-outage failure mode.
(c) The scan is `except Exception: return False` (635-637) — a transient log read
error is indistinguishable from "no auth failure".
→ Run the main L2 agent with `--output-format json` and parse the structured envelope
for an auth/error signal (as `claude_gate.invoke_claude` already does, claude_gate.py:625),
instead of regex-scanning a free-text log slice. At minimum: capture this invocation's
output to a *dedicated per-invocation* file (not the shared, rotated `agent.log`) so the
scan is exact and rotation-immune, and treat a scan *exception* as "unknown — do not
downgrade to success".

---

## IMPORTANT (P1)

**P1-#1 — `check_layer2_journal_activity` is documented "never raises" but is not
exception-guarded; a naive-vs-aware journal `ts` throws TypeError and silently drops
the ENTIRE contract for the cycle.**
`loop_contract.py:277-574` (comparisons at 442, 382, 388, 401, 418) called unguarded
from `verify_contract` (772). `last_trigger`/`inv_ts` are aware (host code writes
`datetime.now(UTC).isoformat()`), but the journal `ts` is LLM-authored (PLAYBOOK:212)
and may be naive (`...T12:00:00`, no offset). `journal_ts >= last_trigger - timedelta(...)`
then raises `TypeError: can't compare offset-naive and offset-aware`. The only catch is
in main.py:1508 (`except Exception as e_contract: logger.warning`), which discards
*all* invariant results for that cycle (cycle-duration, success-rate, signal-stability,
journal-activity) — a silent loss of the entire safety net, intermittently, on exactly
the cycles the journal check matters.
→ Normalize every `_parse_iso` result to aware UTC (assume UTC when naive), and wrap
the function body in `try/except` returning `[]` with a logged warning, to honor the
docstring contract.

**P1-#2 — Trigger lost between detection and dispatch when `trigger_buffer` defers.**
`main.py:832-851` + `trigger.py:296-343,503-516`. With `claude_budget.batch_window_s>0`,
`run()` calls `trigger_buffer.add()` then, if `flush_due()` returns nothing, sets
`triggered=False`. But `check_triggers()` has *already* advanced the persistent
`triggered_consensus[ticker]=action` baseline (trigger.py:335) and set
`state.last_trigger_time`. When the buffer later flushes that same reason, the
consensus crossing will NOT re-detect (baseline already moved to the new action), so the
deferred trigger can be silently swallowed — Layer 2 never sees the consensus event.
Default `batch_window_s=0` (disabled) limits blast radius today.
→ Do not advance the consensus baseline until the trigger is actually dispatched, or
have the buffer carry enough state to re-emit; alternatively gate the baseline write on
`flushed`.

**P1-#3 — `auth_error` row suppression vs `incomplete` mislabel: an undetected auth
failure that's also missing a journal becomes `incomplete`, not `auth_error`.**
`agent_invocation.py:1570-1577`. Status precedence is auth → exit!=0 → success →
incomplete. When `detect_auth_failure` misses the marker (P0-#2 (a)/(b)) AND the agent
wrote nothing, `exit_code==0` so status falls to `incomplete`. The contract's
`_KNOWN_FAILURE_STATUSES` includes `incomplete` (loop_contract.py:372-377) so it's
suppressed — meaning a real auth outage is recorded only as a benign-looking
`incomplete` with no `auth_failure` critical-error row and no Telegram auth alert. The
near-daily `contract_violation` is the only symptom that something is wrong, and it
points the operator at the wrong category.
→ Once P0-#2 hardens detection this shrinks; additionally, treat repeated `incomplete`
runs (N consecutive with `journal_written=False`) as an escalation category in
`get_completion_stats`/health so a silent-incomplete streak surfaces distinctly from
one-off incompletes.

**P1-#4 — Multi-agent specialists write `status:"invoked"` rows into the SAME file the
contract trusts as the Layer 2 invocation log, with no `tier`, and block the loop
synchronously.**
`multi_agent_layer2.py:39,201-207` writes to `data/invocations.jsonl`
(== `loop_contract.LAYER2_INVOCATIONS_FILE`, loop_contract.py:102) with
`caller="multi_agent_specialist_*"`, `status:"invoked"`, no `tier`. The contract's
preconditions 4/4b/6 all read `last_jsonl_entry(LAYER2_INVOCATIONS_FILE)` and key off
`status`/`tier`. A specialist `invoked` row can become the newest entry and either (a)
spuriously suppress the journal-activity check within grace, or (b) cause the real L2
completion row to no longer be "latest", defeating the success/auth suppression in
P0-#1/Precondition 6. Separately, `wait_for_specialists` (multi_agent_layer2.py:218-265)
blocks the main loop thread up to `specialist_timeout_s` (default 30s) and kills
under-budget specialists, then the synthesis prompt still tells the agent to "Read the
reports" that may be empty.
→ Write specialist invocation rows to a *separate* file (e.g. `specialist_invocations.jsonl`),
not the L2 contract log. Multi-agent currently only runs when `layer2.multi_agent=true`,
so confirm config before prioritizing the blocking concern.

**P1-#5 — `_consecutive_stack_overflows` auto-disable is effectively permanent; Layer 2
stays dark with no recurring alert.**
`agent_invocation.py:179,801-807,1684-1712`. After 5 consecutive stack-overflow exits
Layer 2 is auto-disabled and `invoke_agent` returns False on every future call
(`skipped_stack_overflow`). The counter only resets on a *non-stack-overflow
completion* (1707-1712) — but once disabled, no subprocess ever runs again, so that
reset path is unreachable. The disable is persisted to disk (survives restart). The
alert fires exactly once at the moment of disable (1698-1704). `skipped_stack_overflow`
is in `_LEGITIMATE_SKIP_STATUSES` (loop_contract.py:365-371), so the contract is
permanently suppressed too → Layer 2 is silently and permanently off with no recurring
signal.
→ Add a time-based half-open retry (e.g. allow one probe invocation every N hours while
disabled) and a periodic "Layer 2 still disabled" reminder; reset the counter when the
disabled state is entered so a single later success can re-enable.

**P1-#6 — Crash-alert suppression after 5 crashes can hide a multi-hour outage for up
to 100 crashes.**
`main.py:1121-1199,1219-1255`. After `_MAX_CRASH_ALERTS=5`, alerts are suppressed and a
summary is sent only every `_CRASH_SUMMARY_INTERVAL=100` crashes. With backoff capped at
`_MAX_CRASH_BACKOFF=300s`, 100 crashes ≈ up to ~8h of silence before the next operator
ping. The counter is persisted, so a loop that crash-loops, gets restarted by Task
Scheduler, and crash-loops again accumulates across restarts and stays in the suppressed
band. A persistent crash (bad config, unwritable disk) is exactly when you most want an
alert.
→ Add a wall-clock floor to alerting (e.g. always alert at most once per 30 min
regardless of crash count) instead of a pure modulo-100 gate; reset the persisted
counter on a clean successful cycle (already done via `_reset_crash_counter`, but verify
it runs before the next crash can re-suppress).

---

## MINOR (P2 / P3)

**P2-#1 — `update_tier_state` does an independent disk read-modify-write of
`trigger_state.json`, racing/clobbering `check_triggers`' write within the same cycle.**
`trigger.py:651-661` calls `_load_state()` fresh then `_save_state()`, re-running the
prune logic with `_current_tickers` absent (so prune is skipped, OK) but overwriting any
state written by a concurrent metals/other process. Single main thread per cycle makes
intra-cycle corruption unlikely, but `trigger_state.json` is read by other components;
the load-modify-write is not atomic across processes. → Pass the already-loaded `state`
dict through `classify_tier`/`update_tier_state` (the API already supports `state=`) so
there's one write per cycle.

**P2-#2 — Layer 2 never runs for 24/7 crypto/metals on weekends or Swedish holidays.**
`market_timing.py:244-262` (`_is_agent_window`) returns False on weekends, US holidays,
and Swedish holidays. main.py:986-992 then logs `skipped_offhours` for any crypto/metals
trigger fired Sat/Sun — so BTC/ETH/XAU/XAG (which trade 24/7 and are the stated primary
focus) get zero Layer 2 analysis for ~2 days every week. Contract correctly suppresses,
so this is not a silent stall, but it is a coverage gap that contradicts the 24/7 asset
model. → If intended, document it; if not, gate the window on asset class (crypto/metals
always in-window).

**P2-#3 — `get_market_state` checks US holidays but not Swedish holidays, while
`_is_agent_window` checks both.** `market_timing.py:329-344` vs 244-262. Inconsistent
calendars: on a Swedish-only holiday the loop reports state `"open"` and processes
stocks, but the agent window is closed. Minor, but the two functions should share one
holiday predicate to avoid divergence. → Factor a single `is_trading_day(asset_class, dt)`.

**P2-#4 — `_safe_elapsed_s` wall-clock fallback can over- or under-estimate after an
NTP step, weakening the timeout it's meant to enforce.** `agent_invocation.py:533-573`.
When monotonic is "poisoned" it falls back to `time.time() - _agent_start_wall`, which is
itself NTP-jump-prone (the very reason monotonic was chosen). Acceptable as a
last-resort, but a large backward NTP step on the wall clock could compute a negative
`wall_elapsed` → clamp to 0 → timeout never fires this cycle (the watchdog retries next
tick, so bounded). Low severity given the 30s watchdog re-check. → Note only.

**P3-#1 — `detect_auth_failure` records a `critical_errors.jsonl` row on every scan hit;
the same agent.log slice can be scanned twice (completion path + a watchdog/main race is
prevented by the lock, but the kill path at 760 and completion path at 1564 scan
different offsets of the same run).** Duplicate `auth_failure` rows are possible across
the timeout-then-detect sequence. Dedup is handled downstream by the fix-agent
dispatcher cooldown, so impact is cosmetic. → Note only.

**P3-#2 — `_extract_ticker` regex `\b([A-Z]{2,5}-USD)\b` and the stock pattern can
misclassify multi-ticker triggers, sending the wrong ticker to multi-agent specialists
and decision-feedback.** `agent_invocation.py:315-331`. First-match-wins on a
multi-ticker reason list picks an arbitrary ticker. Fail-open (specialists fall back to
single-agent), low impact. → Note only.

---

## What is solid (no action)

- `_completion_lock` correctly serializes the main-thread `check_agent_completion` and
  the 30s watchdog; the post-clear `return None` prevents double-logging and
  double-`record_trade` (agent_invocation.py:1480-1494,1714-1731).
- Timeout-kill path writes both an `invocations.jsonl` `timeout` row AND a journal stub
  (agent_invocation.py:763-786) — the right pattern; replicate it for `auth_error`
  (P0-#1).
- DST math in `market_timing.py` uses aware UTC datetimes throughout; EU/US transition
  dates and `_observed`/`_nth_weekday`/Easter are correct. No hardcoded-offset bug found.
- Crash recovery guarantees a sleep floor even if both alert and backoff helpers raise
  (`_safe_crash_recovery`, main.py:1219-1255).
- Trigger prune correctly skips on empty ticker set to avoid an invocation storm
  (trigger.py:181-194).
- `--bare` correctly absent from all three spawn sites with prominent warnings.
