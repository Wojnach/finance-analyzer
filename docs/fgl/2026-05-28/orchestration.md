# Orchestration subsystem — adversarial review

Scope: loop lifecycle, Layer 2 invocation, trigger detection, loop contract.
Files reviewed (one diff, worktree `Q:\fa-fgl\orchestration`):
`portfolio/main.py`, `agent_invocation.py`, `trigger.py`, `loop_contract.py`,
`autonomous.py`, `market_timing.py`, `claude_gate.py`, `reporting.py`,
`trigger_buffer.py`, `loop_processes.py`.

## Count summary

| Severity | Count |
|----------|-------|
| P0       | 2     |
| P1       | 3     |
| P2       | 5     |
| P3       | 3     |

## ROOT CAUSE OF THE LIVE INCIDENT (`layer2_journal_activity` firing many times/day)

**It is NOT a silent Layer 2 failure. It is a false-positive contract check.**

Evidence (live data, 2026-05-21 → 2026-05-28):
- 233 `layer2_journal_activity` CRITICAL entries in `data/critical_errors.jsonl`.
- Last 400 invocations have **zero** `timeout` and **zero** `failed` statuses.
  Every violation's `last_invocation_status` = **`success`** with
  `journal_written=True`, `telegram_sent=True`.
- For every violation, the journal entry that the successful run wrote is
  timestamped **23–87 s BEFORE** `health.last_trigger_time`. Example
  (2026-05-28 10:30): trigger detected 10:30:39.038, `invoked` row logged
  10:30:38, but the agent's journal entry for that exact trigger
  (`trigger="XAU-USD flipped HOLD->BUY (sustained)"`) is stamped
  `2026-05-28T10:30:15Z` — 24 s earlier than `last_trigger_time`.

Mechanism: the Claude subprocess stamps the journal `ts` from the cycle
summary timestamp (cycle start / rounded whole-second `...Z`), not from
wall-clock-now after the trigger fired. `last_trigger_time`
(`portfolio/health.py:32`, written by `update_health()` after trigger
detection, microsecond precision) is therefore always slightly AFTER the
journal `ts`. `loop_contract.check_layer2_journal_activity` line 410
requires `journal_ts >= last_trigger` with **zero tolerance**, so a
perfectly successful, journaled, telegrammed run still trips the CRITICAL
contract. Net effect: 233 spurious CRITICAL alerts that (a) spam Telegram,
(b) pollute `critical_errors.jsonl` (which forces every future Claude
session's startup check to surface them and arms the fix-agent
dispatcher), and (c) train operators to ignore the one contract whose job
is to catch a genuine silent Layer 2 outage. This is alert-fatigue
masking — the highest-priority orchestration bug.

---

## P0 — money / crash / data-corruption / silent-exit-0

`portfolio/loop_contract.py:410`: P0 false-positive-contract: `check_layer2_journal_activity` requires `journal_ts >= last_trigger` with zero clock-skew tolerance. The agent stamps the journal `ts` from the cycle-summary timestamp (whole-second `...Z`, set at cycle start), while `last_trigger_time` is written later in the cycle with microsecond precision, so a fully successful run's journal entry is reliably 23–87 s earlier than the trigger. Result: 233 spurious CRITICAL `layer2_journal_activity` violations in 7 days, all on `inv_status=success`. Fix: pass the check when a journal entry exists whose `trigger` field matches `health.last_trigger_reason` AND whose ts is within a tolerance window (e.g. `journal_ts >= last_trigger - 180s`), mirroring the 2 s tolerance already used in preconditions 4b/5b (lines 380, 386). Simplest robust fix: compare journal ts to the matching invocation's `completed_at` / `ts` in `LAYER2_INVOCATIONS_FILE`, or match on the trigger-reason string rather than on a cross-clock timestamp ordering. Until fixed, the contract is worse than useless — it hides real failures behind noise.

`portfolio/agent_invocation.py:1444-1457`: P0 silent-exit-0 / contract-blind-spot: the timeout-kill path in `_check_agent_completion_locked` returns a result dict but **never appends a journal stub**, unlike the `incomplete` path (lines 1583-1598) which writes a `NO_DECISION` stub. A timed-out invocation therefore leaves NO journal entry for its trigger. The contract's `_KNOWN_FAILURE_STATUSES` (line 372-375) contains only `{incomplete, auth_error}` — **`timeout` and `failed` are absent**, so a timeout does not suppress the journal-activity check either. A genuine T2/T3 hang (currently rare in the data, but the exact failure class this subsystem must survive) thus produces a real silent gap with no journal record. Fix: in the timeout branch write the same `NO_DECISION`/`status:"timeout"` stub to `JOURNAL_FILE` that the incomplete branch writes, and add `"timeout"`/`"failed"` to `_KNOWN_FAILURE_STATUSES` so the contract treats a recorded terminal failure as "explained, not silent".

---

## P1 — wrong under realistic conditions

`portfolio/main.py:855`: P1 trigger-loss / heartbeat-reset: `update_tier_state(tier)` is called BEFORE `invoke_agent`, and for T3 it sets `last_full_review_time = now` (`trigger.py:659-660`) unconditionally. If `invoke_agent` then returns False (agent busy `skipped_busy`, drawdown block, trade-guard block, perception-gate skip, stack-overflow auto-disable), the T3 periodic-review clock is reset even though no full review ran. The next T3 "heartbeat" is pushed out by the full 4 h interval. Fix: call `update_tier_state(tier)` only after a successful invocation/decision, or pass the invocation result into it and skip the `last_full_review_time` bump when the run did not actually execute.

`portfolio/trigger.py:296-343` + `main.py:842-855`: P1 trigger-consumed-without-action: `check_triggers` persists the consensus/flip baseline (`triggered_consensus[ticker]=action`, `state["last"]`) via `_save_state` on the SAME cycle it returns the reason — before `invoke_agent` runs. If the subsequent invocation is skipped/blocked/`skipped_busy`, the baseline has already advanced, so the consensus crossing will NOT re-fire next cycle. A real BUY/SELL crossing can be silently dropped whenever the agent is busy or a gate blocks. Fix: only advance `triggered_consensus` for a ticker once its trigger has actually been dispatched (or re-derive crossings from a separately-persisted "last acted" baseline rather than "last seen").

`portfolio/agent_invocation.py:1095-1100`: P1 detection-gap: the Layer-2 `claude -p` command (built here directly, bypassing `claude_gate.invoke_claude` despite that module declaring direct Popen "FORBIDDEN") omits `--output-format json`. Unlike `claude_gate`, completion detection here cannot see the CLI's `result`/`usage`/turn-limit envelope — it infers success purely from exit code + journal-line-count delta + an auth-marker scan. A max-turns-exhausted run that writes a partial/garbage journal line still counts as `journal_written=True` → `success`. Fix: add `--output-format json`, parse the envelope (reuse `claude_gate._parse_claude_json_stdout`), and treat a turn-limit/empty-`result` envelope as `incomplete` even when exit==0 and a journal line appeared.

---

## P2 — latent

`portfolio/market_timing.py:244-260`: P2 metals-coverage-gap: `_is_agent_window` opens at EU equity open (07:00 UTC summer / 08:00 UTC winter = 09:00 CET). Avanza commodity warrants — the primary metals focus — trade from **08:15 CET**. Layer 2 is therefore skipped (`skipped_offhours`, main.py:982) for the first ~45 min that XAG/XAU warrants are tradable each morning. Fix: widen the agent window's lower bound for metals/crypto-driven triggers to cover 08:15 CET, or gate the window per asset class rather than on EU equity hours.

`portfolio/agent_invocation.py:88-89` (autonomous) and `portfolio/autonomous.py:88-89`: P2 silent-no-journal-in-fallback: `autonomous_decision` wraps `_autonomous_decision_inner` in a blanket `except Exception: logger.exception(...)` and the journal append is at line 151 — any exception in `_classify_tickers`/`_ticker_prediction`/`_build_*` before line 151 swallows the error AND writes no journal entry, leaving the contract violated in L2-disabled mode with only a log line. Fix: write a minimal journal stub in the `except` handler (mirroring the L2 incomplete stub) so the fallback path can never go journal-silent.

`portfolio/agent_invocation.py:179` + `1602-1604`: P2 cross-process counter race: `_consecutive_stack_overflows` is a module global loaded once at import and persisted with `atomic_write_json`. The main loop spawns the agent in-process so this is fine within one process, but the watchdog thread and the main-thread completion call both mutate it under `_completion_lock` (OK) — however the bigbet/iskbets/metals processes that also spawn `claude` do NOT share this counter, so a CLI stack-overflow crash storm originating outside agent_invocation is invisible to the auto-disable. Fix: if cross-process stack-overflow protection is intended, move the counter behind a file lock read-modify-write rather than an in-memory global seeded at import.

`portfolio/loop_contract.py:304`: P2 trigger-clock-monotonicity: the precondition computes `trigger_age_s` from `health.last_trigger_time` which `update_health` rewrites to `now()` every cycle that has any trigger reason. A rapid succession of distinct triggers keeps moving `last_trigger_time` forward, so the grace window effectively never elapses for the FIRST trigger and the journal-for-trigger-N check is always evaluated against trigger-N+k. This is the same clock-coupling that produces the P0 false positive; even after the tolerance fix, the contract conflates "a trigger fired" with "this specific trigger was journaled". Fix: track per-trigger identity (reason + ts) and verify a journal entry whose `trigger` field matches, rather than a global last-trigger timestamp.

`portfolio/trigger.py:172-196` (`_save_state`) + `main.py`: P2 trigger_state read-modify-write race: `check_triggers` does `_load_state()` → mutate → `_save_state()` (`atomic_write_json`). `classify_tier`/`update_tier_state` each independently `_load_state()`/`_save_state()` later in the same cycle (main.py:852-855). The write in `update_tier_state` reloads from disk and can clobber concurrent updates from the trigger pass if any other writer (dashboard, a second loop instance) touches `trigger_state.json` between the two reads. Atomic-replace prevents corruption but not lost updates. Fix: thread a single `state` dict from `check_triggers` through `classify_tier(state=...)`/`update_tier_state(state=...)` (the params already exist) and write once.

---

## P3 — minor

`portfolio/loop_processes.py:106`: P3 unbound-name-on-error: the `except (NoSuchProcess, AccessDenied)` handler references `info.get("pid")`, but `info = p.info` is the statement that can raise (psutil populates `.info` lazily). If `p.info` raises, `info` is unbound and the handler throws `NameError` (not caught by the clause), crashing `scan()` / the `/api/loop-processes` endpoint. Fix: use `p.pid` (available without `.info`) in the log call, or set `info = {}` before the try.

`portfolio/agent_invocation.py:1046`: P3 inconsistent-return-type: the multi-agent specialist-quorum-fail branch does `return` (None) whereas every other early exit in `invoke_agent` returns `False`. Callers (`main.py:954`) treat the result as a bool for the `invoked`/`skipped_busy` log label; `None` is falsy so it works today, but the type inconsistency is a latent trap. Fix: `return False`.

`portfolio/agent_invocation.py:182-194`: P3 stale-doc-vs-config drift: the module docstring/CLAUDE.md describe T1 timeout as 120 s in several comments while `TIER_CONFIG` now uses 180 s, and `loop_contract.LAYER2_JOURNAL_GRACE_S_BY_TIER[1]` is 720 s "until the deeper bug is traced" — that deeper bug is the P0 above (T1 grace was widened to paper over the false positive). Fix: after the P0 timestamp-tolerance fix, restore T1 grace to a tight value (timeout + slack ≈ 240 s) so the contract can again catch real T1 silent failures, and reconcile the comments.

## Notes on things that are CORRECT (checked, not findings)

- CLAUDECODE / CLAUDE_CODE_ENTRYPOINT env scrub IS present before Popen
  (`agent_invocation.py:1121-1123`, `claude_gate.py:174-175`) — the 34 h
  nested-session outage class is guarded.
- `--bare` is NOT present and is explicitly documented as forbidden.
- Auth-marker scan covers both the happy-completion and timeout-kill paths
  (`_scan_agent_log_for_auth_failure` shared helper).
- `layer2_context.md` IS produced (journal.write_context, invoke_agent:848)
  so the T1/T2/T3 `cat` chain does not starve the agent.
- Crash-recovery backoff has a guaranteed floor sleep
  (`_safe_crash_recovery`) even if alert/counter writes fail.
- DST math in `market_timing.py` (EU + US) and NYSE/Swedish holiday
  calendars are correct.
