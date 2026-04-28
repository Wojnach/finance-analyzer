# Plan — Accuracy degradation alert: root-cause + statistical rigor (2026-04-28)

## Context

Yesterday we shipped a Telegram-cooldown fix for the `accuracy_degradation`
contract alert (merge `7d507748`). Today's diagnostic confirmed the cooldown
is working but surfaced that **the alert itself is firing on a system that's
not in a healthy state to evaluate degradation**: the daily accuracy
snapshot writer has been silently failing for a week, the detector is
comparing against an anomalously high baseline computed on a small window
(N=223), and the cooldown hash drifts every hour because percentages drift
between snapshots.

The user said: *"the spam wasn't the problem, the problem was what it was
reporting — it's a red flag that something is broken in the system and
THAT needs fixing."* They're right. The spam was the symptom; the *thing
the smoke detector keeps reporting* is what we need to fix.

## What's broken (confirmed via investigation)

### 1. The snapshot writer hasn't actually written for 7 days

`data/accuracy_snapshots.jsonl` had only 4 entries before today's manual
trigger:

```
2026-02-20  signals_recent={}   (pre-feature)
2026-04-19  sentiment_recent=83.4% / 217 samples
2026-04-20  sentiment_recent=76.7% / 236 samples
2026-04-21  sentiment_recent=75.3% / 223 samples
[no entries Apr 22..27]
```

But `data/accuracy_snapshot_state.json` says
`last_snapshot_date_utc: 2026-04-28`. The state file's mtime is
`06:10 UTC` (matches yesterday's operational fixup) — so something updated
the state-as-if-the-snapshot-was-written without actually appending to the
JSONL. This is a silent-failure mode: in `maybe_save_daily_snapshot`
(`portfolio/accuracy_degradation.py:658`), the gating only checks
`state["last_snapshot_date_utc"] == today_str`. Once that's set, the
function returns False and never tries again — even when the JSONL is
empty for today.

The natural daily writer should fire after `06:00 UTC` each day. Either it
hasn't been firing (loop down at the wrong hour repeatedly) or the
operational fixup yesterday wrote the state without writing the JSONL,
poisoning today's natural run.

### 2. The "baseline" is statistical noise

The Apr 21 snapshot has `sentiment_recent: 75.3% / 223 samples`. But
`signals.sentiment` (lifetime over 39k samples) is **46%**. The 75% reading
was a 7-day window where sentiment happened to be on a hot streak — a
small-sample anomaly, not the signal's true performance.

Today's recent: `sentiment_recent: 43.3% / 187 samples`. That's roughly
*at* lifetime (46%). So the "32pp drop" the alert reports is largely
**regression to the mean from an anomalously good week**.

The detector at `accuracy_degradation.py:556-582` requires:
- Drop ≥ 15pp
- New accuracy < 50%
- Both old and new sample sizes ≥ 100

`MIN_SAMPLES = 100` on a 7-day window with binomial-noise SE of
`±√(0.25/100) ≈ 5%` produces frequent spurious 15pp drops when comparing
two random 7-day samples of the same underlying signal.

### 3. The cooldown hash drifts every cycle

The current hash includes percentage strings like `33.7%`, `33.2%` from
the violation message. These drift each cycle as new samples land,
producing a fresh hash on every hour-ish interval and bypassing the
multi-hash dedup. Result: hourly Telegram spam.

This is the same pattern we already fixed for `layer2_journal_activity`
(folded `details["trigger_time"]` into a stable identity hash). For
`accuracy_degradation` the stable identity is the **set of violating
signal keys**, not the rendered percentages.

### 4. Real signal weakness exists but is mixed in with noise

Looking at lifetime accuracy:
- `sentiment` lifetime 46%, recent 43% — **at lifetime, no real drop**
- `momentum_factors` lifetime 54%, recent 31% — **real 23pp gap, worth investigating**
- `BTC-USD::structure`, `BTC-USD::futures_flow`, `XAG-USD::trend`, etc. —
  per-ticker drops on smaller windows, mix of noise and real

The MSTR cluster's "90% → 41%" is on N=156 baseline → fundamentally a
small-sample artifact. `MSTR::sentiment` was re-enabled Apr 16
(commit `fd504d44`) — three weeks of signals, then a small hot streak
got recorded as the "baseline".

## Goals

Fix the infrastructure so the alert is *meaningful* when it fires:

1. **Snapshot writer**: bulletproof against state-without-write desync.
   Verify JSONL append before persisting state.
2. **Snapshot backfill**: regenerate snapshots for Apr 22-27 from
   historical signal log, so the detector has a real 7-day baseline.
3. **Detector statistical rigor**: raise minimum sample sizes; require
   the drop to be statistically significant (e.g., binomial-test p < 0.01
   or absolute drop > 2 standard errors) instead of a flat 15pp.
4. **Stable cooldown hash**: hash on sorted violating signal keys, not
   on rendered message text. Mirror the `layer2_journal_activity` fix.
5. **Verify the real degradation that remains** after noise-filtering.
   For any signal that *still* fires post-fix, check whether it's
   regime-shift, outcome-pollution, or a code bug.

## Non-goals (deferred)

- Per-signal disable decisions (`MSTR::sentiment` etc.) — that's a config
  tuning session, not a code fix. Surface candidates in the journal.
- Outcome backfill bug investigation — recent signal_log entries having
  empty outcomes is normal backfill timing (need 24h elapsed for 1d
  horizon).
- Rebuilding the accuracy pipeline from scratch.

## Implementation batches

### Batch 1 — Snapshot writer bulletproofing

**Files:** `portfolio/accuracy_degradation.py`,
`tests/test_accuracy_degradation_snapshot.py` (new)

- In `maybe_save_daily_snapshot` (line 658), record `accuracy_snapshots.jsonl`
  size before calling `save_full_accuracy_snapshot()`. After the call,
  verify the file grew by at least 1 line. If it didn't grow, log
  `critical_errors.jsonl` entry and refuse to update state.
- Add an end-to-end test: stub `save_accuracy_snapshot` to a no-op and
  verify state is NOT updated.
- Add a test for the legitimate path: real append → state updated.

**Acceptance:** if the JSONL doesn't grow, the state isn't updated, and a
critical_errors entry is written so the dispatcher can engage.

### Batch 2 — Backfill missing snapshots

**Files:** `scripts/backfill_accuracy_snapshots.py` (new), one-shot.

- Iterate dates Apr 22 through Apr 28.
- For each date, replay `save_full_accuracy_snapshot(at=date)` using
  signal_log entries with `ts <= date`.
- Existing infra (`save_full_accuracy_snapshot`) doesn't take a "now"
  parameter; need to thread one through, or use a dated cutoff filter.
- Append to `data/accuracy_snapshots.jsonl` in chronological order.

**Acceptance:** `data/accuracy_snapshots.jsonl` has entries for every
date Apr 22-28; each entry's `signals_recent.sentiment.total` is in a
plausible range (~150-250).

### Batch 3 — Detector statistical rigor

**Files:** `portfolio/accuracy_degradation.py`,
`tests/test_accuracy_degradation_significance.py`

- Raise `MIN_SAMPLES_HISTORICAL` and `MIN_SAMPLES_CURRENT` from 100 → 300
  for the recent-window check. (Lifetime checks unchanged.)
- Add a significance gate: instead of just `drop_pp >= 15.0`, require
  `drop_pp >= max(15.0, 2 * SE)` where
  `SE = sqrt(p_old*(1-p_old)/N_old + p_new*(1-p_new)/N_new) * 100`.
- Tests: synthesize known-noise samples (N=200 each, true p=0.5, two
  draws) — assert detector does NOT fire 95% of runs. Synthesize a
  real degradation (N=500 each, p=0.55 → p=0.30) — assert detector DOES
  fire.

**Acceptance:** synthetic noise stops firing; synthetic real degradation
still fires.

### Batch 4 — Stable cooldown hash

**Files:** `portfolio/loop_contract.py`,
`tests/test_loop_contract_alert_cooldown.py`

- Extend `_hash_violation_identity` (already exists for
  `layer2_journal_activity`) to handle `accuracy_degradation`: hash on
  sorted `details["alerts"][*]["key"]` (the signal+scope identifiers),
  ignoring percentages.
- Test: same set of violating signals with drifting percentages → same
  hash, dedup works. Different set of violating signals → different
  hash, fresh send.

**Acceptance:** a 4-hour run with hourly accuracy_degradation re-fires on
the same signal set produces 1 Telegram, not 4.

### Batch 5 — Verify what's left after noise-filtering

**One-shot script** in `scripts/audit_accuracy_drops.py`:

- For each signal/scope flagged today, compute:
  - Lifetime accuracy
  - Each of the last 7 weekly-window accuracies
  - Whether current is > 2 SE below lifetime (real)
  - Whether the change vs 7d ago is > 2 SE (real)
- Output a markdown table to `docs/accuracy_audit_20260428.md`.

**Acceptance:** a list of signals that are *still* statistically below
their lifetime expectation, separated from regression-to-mean noise.
This output informs follow-up gating decisions (config, not code).

## Verification (system-wide, after all batches)

1. `pytest tests/ -n auto` — full suite green; no new flakes.
2. Manual: trigger `maybe_save_daily_snapshot()` after deleting
   accuracy_snapshot_state.json — confirm JSONL grew by 1 line and state
   was updated.
3. Manual: run with mocked save_accuracy_snapshot raising — confirm state
   NOT updated and critical_errors row appears.
4. Manual: with the post-fix detector, the next contract cycle should
   either (a) not fire (most signals are noise) or (b) fire on a smaller,
   higher-confidence set of real degraders.
5. Restart loops via `schtasks /run /tn "\PF-DataLoop"` and
   `\PF-MetalsLoop`.
6. Watch `data/portfolio.log` for the next 30 min — expect at most 1
   accuracy_degradation Telegram even if the signal set drifts.

## Risks

- **Backfill produces wrong numbers** if outcomes were polluted before
  BUG-220 was fixed (Apr 24). Mitigation: log per-snapshot sample sizes
  and skip if anomalous.
- **Significance gate suppresses real signal regime shifts** in the
  short term until enough data accumulates. Acceptable trade — we'd
  rather miss a 1-week regime shift than spam every cycle.
- **Hash change rolls existing cooldown state**. Old hashes on disk
  don't match the new identity. First post-fix cycle will fire once,
  then cool down. Acceptable.

## Out of scope

- Restarting loops mid-batch. Restart only after all batches merged.
- Changes to ViolationTracker / contract framework architecture.
- Layer 2 trading logic.
- Any signal *implementation* changes (we're auditing, not editing
  signals).
