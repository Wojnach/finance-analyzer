# Independent Adversarial Pass — FGL 2026-05-29 (orchestrator's own review)

Written WITHOUT reading the 8 subagent outputs first, to stay unanchored. Focus:
cross-subsystem seams + the live production incident + meta-process — the failure
classes a per-subsystem reviewer structurally cannot see (each owns only its slice).

## A. THE LIVE INCIDENT — 39 unresolved `layer2_journal_activity` violations (P0-class, meta)

**Evidence:** `check_critical_errors.py` reports 39 unresolved critical entries over
2026-05-22 → 05-29. The dominant category (28 of 39) is `contract_violation /
layer2_journal_activity`: "Layer 2 trigger fired Nm ago … but no journal entry has
been written since. Agent may be failing silently." Plus 9× `accuracy_degradation`
(12 signals >15pp) and 4× `avanza_account_mismatch` (session expired 05-23).

**Trace (orchestration seam):**
- `loop_contract.check_layer2_journal_activity()` (`portfolio/loop_contract.py:277`)
  fires when `health.last_trigger_time` is within 6h, past the per-tier grace, AND the
  newest `layer2_journal.jsonl` entry predates the trigger — UNLESS the newest L2
  invocation status is `invoked` / a legitimate-skip / `incomplete` / `auth_error`.
- The completion path `agent_invocation.check_agent_completion()`
  (`portfolio/agent_invocation.py:1490`) DOES write an `incomplete` stub when the
  subprocess exits 0 without journaling (`:1642`), and the timeout path writes a
  `timeout` stub (`:769`). So a *completed* invocation should never leave a contract gap.
- Therefore the chronic violations come from one of: (a) `invoke_agent` returning
  False → `skipped_busy` (main.py:964/989), which is **deliberately NOT suppressed**
  (comment at loop_contract.py:353 — "couldn't kill old agent / no agent binary" must
  not be masked); or (b) genuine silent non-journaling on a path that bypasses
  completion processing (e.g. `_agent_proc` lost, loop restarted mid-invocation).

**The real P0 is meta, not the individual entry:** `critical_errors.jsonl` is
append-only and "resolved" only by manually appending a resolution line. Nobody is
appending resolutions, so entries accumulate. The system's #1 silent-failure tripwire —
the exact mechanism built to catch the next 3-week auth outage — now shows 39 standing
alerts. That is **alert fatigue by design**: when the real outage arrives it will be
entry #40 in a list the operator has learned to ignore. Either L2 is genuinely failing
to journal on a recurring path (real silent failure) OR the contract over-fires
(false-positive flood) — **both outcomes destroy the detector's trust value, which is
worse than having no detector.**

**Recommended (not implemented — review only):**
1. Add an auto-resolution path: when a later cycle observes the contract PASS for a
   trigger that previously fired a violation, append a `resolution` line automatically
   instead of requiring a human. Closes the append-only hygiene gap.
2. Instrument the actual split: log a per-violation `root_cause` derived from the
   newest invocation status (`skipped_busy` vs `no-invocation-at-all` vs
   `completed-but-no-journal`). 28 undifferentiated entries hide which failure mode is live.
3. The avanza session has been expired since 05-23 (6 days). Whatever cron is supposed
   to keep it fresh is not running or not alerting loudly enough — the daily
   `avanza_account_mismatch` re-fire is itself proof the tripwire works but is ignored.

## B. Cross-subsystem seams (between two reviewers' slices — coverage-gap risk)

- **B1 (orchestration ↔ infrastructure), P1:** `agent_invocation._scan_agent_log_for_auth_failure`
  (`:610`) seeks into `agent.log` from a captured byte offset, with a guard for
  rotation-truncation (offset > size → scan whole file). But `log_rotation.rotate_text()`
  runs hourly inside `_run_post_cycle` *while a T2/T3 agent runs for minutes*. If rotation
  fires and the new file grows PAST the old offset before the scan, the scan reads only
  the tail beyond the stale offset and can miss a "Not logged in" line that landed in the
  rotated-away portion. The size-shrink guard only covers the shrink case, not
  rotate-then-regrow. Detection: tag each agent.log line with the invocation id and scan by
  id, not byte offset.
- **B2 (orchestration ↔ portfolio-risk), P1:** completion path calls `_record_new_trades()`
  (`agent_invocation.py:1615`) only on the happy path after `check_agent_completion`. If
  the loop restarts between the agent writing a trade to portfolio_state and
  `_record_new_trades()` running, the overtrading guards (cooldown/loss-escalation in
  `trade_guards`) never see that trade → next cycle can over-trade. Detection: derive
  recorded-trades from a persisted high-water mark on transaction index, not from
  in-process invocation lifecycle.
- **B3 (signals-core ↔ signals-modules), P1:** the regime gates (drift/adx/amihud/chop/
  bocpd/vol_ratio) are documented as "suppress only" but feed the same voting array as
  directional signals. If any gate ever emits a directional vote (BUY/SELL) on an error
  default instead of HOLD, it silently changes consensus. The seam: signals-core trusts
  modules to self-classify as non-directional; nothing in the engine enforces it.
  Detection: an engine-level assertion that registered regime-gate modules can only
  contribute HOLD or a confidence multiplier, never a directional vote.
- **B4 (data-external ↔ everything), P0-class if present:** a cache that serves
  last-known-good silently on live-fetch failure poisons every downstream signal AND the
  Layer 2 prompt simultaneously — a single stale-price source corrupts both layers and
  looks healthy. This is the highest-leverage single point; flagged to data-external
  reviewer as its primary hunt.

## C. Spot-checks of my own (independent confirmations)

- `file_utils.atomic_write_json` (`:53`) / `atomic_write_text` (`:32`): **correct** —
  mkstemp in the resolved real-file parent dir (same FS → atomic `os.replace`), fsync
  before replace, temp cleanup on BaseException, symlink resolution so it never clobbers
  the link inode. No finding. (Note: `_resolve_write_path` resolving symlinks means a
  caller passing the `config.json` symlink path WOULD write through to the external
  secret file — safe today because no caller does, but there is no guard preventing it.)
- `agent_invocation._detect_append` (`:640`): correct handling of the prune-race
  (count-delta trusted only when positive, else fall back to newest-ts). Good defensive
  code; no finding.

## D. Stale-state / repo hygiene (P2)

- Numerous abandoned worktrees under `.worktrees/` and `.claude/worktrees/` (10+), each a
  full checkout. These waste disk (repo working tree is ~40MB of code; data dir is 1.5G
  but excluded) and produce confusing grep hits across stale copies of `loop_contract.py`
  etc. GUIDELINES rule 9 mandates worktree cleanup after merge — it is not happening.
  Detection: a weekly `git worktree prune` + stale-branch reaper.
- `config.json` correctly external symlink; `_resolve_write_path` honors it. OK.

## E. Severity tally (my pass)
P0(meta): 1 (the standing-violation alert-fatigue trap) · P1: 4 (B1, B2, B3, +avanza
session-expired-unremediated) · P2: 2 (worktree hygiene, config-symlink write guard) ·
plus B4 flagged to data-external as P0-if-confirmed.
