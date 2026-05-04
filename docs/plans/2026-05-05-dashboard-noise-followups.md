# Dashboard noise — open follow-ups

**Date:** 2026-05-05
**Predecessor PRs:** `f3c120ce` (dashboard violations filter + trades SEK column),
  `3fb5b39c` (claude review P1: context fallback + ESCALATED strip).

The two PRs above cleared the home dashboard's "Errors & violations"
panel from 6 cv → 1 cv. Three open items remain. Documented here so they
don't get lost across sessions.

---

## (1) Live `accuracy_degradation` regression — DEFERRED, parallel work in flight

**Status:** the home page now shows exactly 1 cv row:
`ESCALATED (3x consecutive): 2 signal(s) dropped >15pp...`. That alert is
*real* — a different signal pair than the one disabled by the May 3
research session. Resolving it requires either:

- adding more entries to `DISABLED_SIGNALS` in `portfolio/tickers.py`, OR
- moving the accuracy_snapshot baseline forward (the 7-day-old snapshot
  is increasingly stale and recurrence amplifies the drift), OR
- re-weighting the recurring offenders.

**Why deferred:** worktree
`.worktrees/dashboard-accuracy-2026-05-05` (branch
`fix/dashboard-accuracy-disabled-signals-2026-05-05`) is *currently*
modifying `portfolio/tickers.py`, `portfolio/accuracy_stats.py`,
`tests/test_accuracy_stats.py` plus dashboard JS for showing
disabled-signal status. Touching the same files here would conflict.
Re-pickup once that branch merges.

**Pickup checklist when the parallel branch merges:**

1. Confirm what the parallel work shipped (UI visibility vs actual
   `DISABLED_SIGNALS` additions).
2. If the live regression still surfaces on the dashboard, pull the
   audit script: `.venv/Scripts/python.exe scripts/audit_accuracy_drops.py`.
3. Decide per-signal: disable, accept, or rebaseline.
4. After merge, restart `PF-DataLoop` so the new `DISABLED_SIGNALS`
   set is loaded.

---

## (2) Stale `critical_errors.jsonl` rows from the May 3 outage — DATA CLEANUP

**Status:** 7 `layer2_journal_activity` entries from
`2026-05-03T23:03:51 → 23:04:01` are still showing under the home page's
`errors.unresolved` count (red rows). They were written when the May 3
ETH-USD trigger genuinely hit a Layer 2 outage — but the agent has been
running fine for >24h and the underlying triggers were eventually
journaled. No explicit `resolves_ts` rows were ever appended, so the
existing `_errors_unresolved` filter still counts them.

**Fix:** append 7 resolution rows to `data/critical_errors.jsonl` —
one per stale entry, each with `category: "resolution"` and
`resolves_ts` pointing at the original row's `ts`. Per
`scripts/check_critical_errors.py:53-85` and the protocol documented in
the CLAUDE.md startup-check block, this is the correct mechanism.

**Why now (not via PR):** `data/critical_errors.jsonl` is gitignored
(it's an append-only runtime journal). The cleanup is a single run-time
file edit, not a code change. The deeper fix would be: have the contract
check itself emit a resolution row when it detects implicit resolution
on the next cycle. Out of scope here — see (3a) below.

**Pickup checklist:**

1. Run a script (or one-liner) that, for each unresolved
   `layer2_journal_activity` critical-level row in the last 7 days,
   verifies a `layer2_journal.jsonl` entry exists with
   `ts >= context.trigger_time`.
2. For verified-resolved rows, append a resolution entry with
   `{"ts": <now>, "level": "info", "category": "resolution",
    "resolves_ts": <original_ts>, "message": "auto-resolved …"}`.
3. Confirm the dashboard `errors.unresolved` count drops accordingly
   on next 30s refresh.

---

## (3) T1 grace window mismatch — UNDIAGNOSED

**Status:** observed in `data/invocations.jsonl` for 2026-05-04: every
T1 invocation succeeded but ran 397-538 s. T1's nominal timeout is 120 s
(`portfolio.agent_invocation.TIER_CONFIG[1] = {"max_turns": 15,
"timeout": 120}`). T2 (600 s) and T3 (900 s) invocations also clustered
around 480-540 s. So either:

- (3a) the timeout-enforcement loop in `_check_agent_completion` is
  not being driven often enough between checks (compare actual elapsed
  to `_agent_timeout` poll cadence — the comment on
  `agent_invocation.py:1015` already notes a prior incident: "T1
  timeout=120s ran 603s"); OR
- (3b) tier escalation is happening internally without updating
  `_agent_tier` / `_agent_timeout`; OR
- (3c) the recorded `tier` in `invocations.jsonl` reflects the
  trigger's tier label, while the running invocation actually uses a
  different effective timeout.

**User-visible noise has already been mitigated** by the dashboard
cross-stream filter — once a journal entry lands, the CV row is hidden.
But Telegram still fires the violation alert in the gap between the
180 s grace expiry and the journal write, so the deeper bug is worth
tracing.

**Pickup checklist:**

1. Read `_check_agent_completion` and trace where `_agent_timeout` is
   compared against `_safe_elapsed_s()`. Verify the polling cadence in
   the main loop is short enough (≤30 s) and that the check actually
   fires while the subprocess is alive.
2. Sample a few T1 entries from `data/invocations.jsonl` and check
   `data/agent.log` for the corresponding T1 subprocess: does the
   process actually run under the T1 120 s cap, or does it run under
   the T3 900 s cap?
3. Either fix the timeout enforcement (3a/3b) or correct the recorded
   tier in invocations.jsonl (3c).
4. If the deeper fix is too large, the minimal hotfix is to widen
   `LAYER2_JOURNAL_GRACE_S_BY_TIER[1]` from 180 → 600 s so the
   contract check matches reality and Telegram noise stops.

---

## How to pick this up next session

```bash
cat docs/plans/2026-05-05-dashboard-noise-followups.md
git worktree list  # check whether dashboard-accuracy-2026-05-05 has merged
```

Item (2) is a 5-minute fix and can be done right after reading this.
Item (1) gates on the parallel branch landing. Item (3) needs the
real-time investigation in steps 1-2 above.
