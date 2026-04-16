# Session in-flight snapshot — 2026-04-16 accuracy degradation tracker

**LAST UPDATED:** 2026-04-16 ~14:20 UTC (running ~1h 35min)
**BRANCH:** feat/accuracy-degradation in worktree /mnt/q/finance-analyzer-degradation
**APPROVED PLAN:** /root/.claude/plans/zazzy-strolling-candy.md
**FULL PLAN DOC (committed):** docs/plans/2026-04-16-accuracy-degradation-tracker.md

## Status by phase

| # | Phase                                 | Status      | Last commit |
|---|---------------------------------------|-------------|-------------|
| 1 | Pre-flight worktree + plan            | DONE        | 33e5847     |
| 2 | Codex pre-impl adversarial review     | DONE        | 16c8128 (4 P1/P2 fixes folded into plan) |
| 3 | Batch 1 — snapshot infra              | DONE        | bb8bba5     |
| 4 | Batch 2 — degradation tracker module  | DONE        | 2381dc2     |
| 5 | Batch 3 — loop integration            | DONE        | 1637960     |
| 6 | Batch 4 — docs + e2e verification     | DONE        | 007ddc6     |
| 7 | Codex post-impl adversarial review    | PARTIAL     | Codex hit usage limit → substituted with pr-review-toolkit:code-reviewer agent. Found 2 P1, 4 P2/P3 issues. |
| 8 | P1/P2/P3 fixes from review            | IN PROGRESS | (uncommitted on branch) |
| 9 | Final test suite + merge + push       | PENDING     |             |
|10 | Restart PF-DataLoop                   | PENDING     |             |
|11 | Worktree cleanup                      | PENDING     |             |

## Commits on branch (ahead of main)

```
33e5847 docs(plan): accuracy degradation tracker
16c8128 docs(plan): address Codex pre-impl adversarial findings
bb8bba5 feat(accuracy): batch 1 — snapshot infra for degradation tracker
2381dc2 feat(accuracy): batch 2 — degradation tracker module + tests
1637960 feat(loop): batch 3 — wire degradation tracker into the main loop
007ddc6 docs: batch 4 — degradation tracker session notes + e2e verification
```

## Uncommitted (post-impl review fixes — pending tests-pass + commit)

```
modified: portfolio/accuracy_degradation.py
modified: portfolio/accuracy_stats.py
modified: portfolio/econ_dates.py
modified: tests/test_accuracy_degradation.py
modified: tests/test_accuracy_snapshot_extras.py
```

### Findings being addressed

- **P1 #1** (perf disaster): `_per_ticker_recent` was re-loading the
  50,000-row signal log 41× per call. Refactored to share a single
  `load_entries()` across all four diff scopes. `accuracy_by_signal_ticker`
  now accepts `entries=` kwarg. `_per_ticker_recent` accepts `entries=`
  kwarg. `_diff_against_baseline` and `save_full_accuracy_snapshot`
  load entries once and thread through.
- **P1 #2** (stale summary): `maybe_send_degradation_summary` now checks
  that the latest snapshot's date matches today, refuses to ship a
  summary built on yesterday's data + today's "Δ vs prev 7d" label.
- **P2 #3** (throttle stale baseline): `maybe_save_daily_snapshot` on
  success now sets `last_full_check_time = 0.0` and clears
  `last_full_check_violations` so the next contract cycle re-runs the
  full check against the fresh baseline.
- **P2 #4** (blackout replays stale): `check_degradation` now returns
  `[]` and clears the cached violation list when blackout is active,
  rather than replaying potentially stale alerts that ViolationTracker
  would escalate.
- **P3 #7** (forecast filter): replaced negative filter
  (`not k.startswith('ministral')`) with explicit allowlist
  (`startswith('chronos') or startswith('kronos')`) in daily summary.
- **P3 #10** (clock dependency): `recent_high_impact_events` now accepts
  `ref_time=None` injection point. The 4 brittle `test_recent_high_impact*`
  tests in test_accuracy_snapshot_extras.py rewritten to pin ref_time.

### Test status (after fixes)

- 38/38 degradation tests pass (test_accuracy_degradation +
  test_accuracy_snapshot_extras + test_loop_contract_accuracy)
- Full neighbor suite not yet re-run — next step

## Files touched on this branch

| File                                                                        | Change                                                                              |
|-----------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| portfolio/accuracy_degradation.py                                           | NEW — full module                                                                   |
| portfolio/accuracy_stats.py                                                 | extras kwarg + entries kwarg on accuracy_by_signal_ticker, consensus_accuracy(days)|
| portfolio/forecast_accuracy.py                                              | cached_forecast_accuracy + invalidate                                               |
| portfolio/econ_dates.py                                                     | recent_high_impact_events with ref_time                                             |
| portfolio/loop_contract.py                                                  | invariant #12 + safe wrapper                                                        |
| portfolio/main.py                                                           | two _track calls in post-cycle                                                      |
| tests/test_accuracy_degradation.py                                          | NEW — ~600 LOC, 22 tests after fixes                                                |
| tests/test_accuracy_snapshot_extras.py                                      | NEW — 15 tests                                                                      |
| tests/test_loop_contract_accuracy.py                                        | NEW — 4 tests                                                                       |
| scripts/_e2e_degradation_check.py                                           | NEW — e2e verification recipe                                                       |
| docs/plans/2026-04-16-accuracy-degradation-tracker.md                       | NEW                                                                                 |
| docs/SESSION_PROGRESS.md                                                    | appended Session 2026-04-16 entry                                                   |
| docs/SESSION_PROGRESS_2026-04-16_degradation_in_flight.md (this file)       | NEW — crash-recovery snapshot                                                       |

## Recovery steps if Claude Code crashes

1. `cd /mnt/q/finance-analyzer-degradation`
2. `git status` to confirm uncommitted changes match the list above
3. Run `cmd.exe /c 'cd /d Q:\finance-analyzer-degradation && Q:\finance-analyzer\.venv\Scripts\python.exe -m pytest tests/test_accuracy_degradation.py tests/test_accuracy_snapshot_extras.py tests/test_loop_contract_accuracy.py -q'`
4. If green, commit with: `git add portfolio/accuracy_degradation.py portfolio/accuracy_stats.py portfolio/econ_dates.py tests/test_accuracy_degradation.py tests/test_accuracy_snapshot_extras.py && git commit -m "fix(accuracy): post-impl review — perf, stale guard, blackout cache clear, throttle invalidation"`
5. Run full suite: `cmd.exe /c 'cd /d Q:\finance-analyzer-degradation && Q:\finance-analyzer\.venv\Scripts\python.exe -m pytest tests/ -n auto'`
6. Pre-existing failure count should be ~24-26 (per CLAUDE.md known-failures)
7. Retry Codex post-impl review: `cd /mnt/q/finance-analyzer-degradation && codex review --base main` (check usage limit reset; was scheduled for 15:46 UTC)
8. Address any new Codex findings, re-test, commit
9. Merge: `cd /mnt/q/finance-analyzer && git merge --no-ff feat/accuracy-degradation -m "Merge feat/accuracy-degradation: signal accuracy degradation tracker"`
10. Push: `cmd.exe /c "cd /d Q:\finance-analyzer && git push"`
11. Restart loop: `cmd.exe /c "Q:\finance-analyzer\scripts\win\pf-restart.bat loop"`
12. Tail `data/portfolio.log` to confirm new "Loop started" + "accuracy_degradation" entry in next cycle's contract output
13. Cleanup: `git worktree remove /mnt/q/finance-analyzer-degradation && git branch -d feat/accuracy-degradation`

## Codex usage limit

Hit the OpenAI usage cap mid post-impl review at ~14:00 UTC. Reset at
15:46 UTC. Retry once available — may surface findings the
substitute Claude reviewer missed.

## What user originally asked for

> what are the runtime contracts we have for our loops. I'm thinking
> perhaps we should add more. We should perhaps have a accuracy and
> probability tracker for our signals. If they severly degrade over
> time then i need to be notified about it for example perhaps

Then:

> use Codex plugin as much as you can when implementing changes. Perhaps
> have codex and you follow guidelines in /fgl

Then (this turn):

> make sure to doucment everything you are doing locally in case you or
> any subagents crash, we've been going strong for 1h 35 min now
