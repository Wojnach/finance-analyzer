# PLAN — Git Tracking Hygiene 2026-05-02

**Branch:** `chore/git-tracking-hygiene-20260502`
**Worktree:** `/mnt/q/finance-analyzer-git-hygiene`
**Base:** `main` @ `9cdaa23a`

## Trigger

Commit `89214641` (May 1 2026) added `data/critical_errors.jsonl` to git as a
brand-new 2-line file. The live runtime journal that `check_critical_errors.py`
appends to has since grown to **104 lines** (102-line divergence). Any future
`git pull` that brings down a different version of that file would clobber the
running loop's append-only state, hiding active critical errors.

The suspicion: other runtime files may have slipped into tracking the same way.

## Audit findings

### Method
1. `git ls-files data/` → 75 tracked files
2. `git status --short data/` → 21 currently DIRTY (uncommitted divergence) — strong "runtime" signal
3. Cross-reference each DIRTY file against `git check-ignore`, against
   `.gitignore` line-by-line grep, and against grepping `portfolio/` and
   `scripts/` for writers.
4. Heuristic: file's mtime < 24h AND multiple `atomic_write_*` callers ⇒ runtime.

### Catalog of all currently-tracked-AND-dirty files

| File | Verdict | Writer(s) | Already in .gitignore? | Action |
|------|---------|-----------|------------------------|--------|
| `data/critical_errors.jsonl` | RUNTIME journal (append-only) | `accuracy_degradation.py`, `agent_invocation.py`, `claude_gate.py`, `bigbet.py`, `loop_contract.py` | NO | `gitignore` + `git rm --cached` |
| `data/forecast_health.jsonl` | RUNTIME journal (770KB) | `forecast_accuracy.py`, `forecast.py`, `health_check.py`, `log_rotation.py` | NO | `gitignore` + `git rm --cached` |
| `data/forecast_predictions.jsonl` | RUNTIME journal (2.8MB) | `forecast_accuracy.py`, `forecast_signal.py`, `fish_monitor_smart.py`, `local_llm_report.py` | NO | `gitignore` + `git rm --cached` |
| `data/sentiment_ab_log.jsonl` | RUNTIME journal (3.9MB) | `llm_batch.py`, `main.py`, `sentiment.py`, `sentiment_shadow_backfill.py` | NO | `gitignore` + `git rm --cached` |
| `data/accuracy_snapshots.jsonl` | RUNTIME journal (500KB) | `accuracy_degradation.py`, `accuracy_stats.py`, `loop_contract.py` | YES (line 117) but tracked | `git rm --cached` (gitignore already covers) |
| `data/auto-improve-progress.json` | RUNTIME progress state (overwritten by `auto-improve.bat`) | `scripts/auto-improve.bat` | NO | `gitignore` + `git rm --cached` |
| `data/after-hours-research-progress.json` | RUNTIME progress state | `scripts/after-hours-research.bat` | NO | `gitignore` + `git rm --cached` |
| `data/models/meta_learner_1d.joblib` | RUNTIME ML model (PF-MLRetrain task) | `portfolio/meta_learner.py` | NO (`models/*.joblib` in line 130 only matches root `models/`) | `gitignore` + `git rm --cached` |
| `data/models/meta_learner_3d.joblib` | RUNTIME ML model | same | NO | `gitignore` + `git rm --cached` |
| `data/models/meta_learner_3h.joblib` | RUNTIME ML model | same | NO | `gitignore` + `git rm --cached` |
| `data/models/meta_learner_5d.joblib` | RUNTIME ML model | same | NO | `gitignore` + `git rm --cached` |
| `data/models/meta_learner_1d_metrics.json` | RUNTIME ML metrics | same | NO | `gitignore` + `git rm --cached` |
| `data/models/meta_learner_3d_metrics.json` | RUNTIME ML metrics | same | NO | `gitignore` + `git rm --cached` |
| `data/models/meta_learner_3h_metrics.json` | RUNTIME ML metrics | same | NO | `gitignore` + `git rm --cached` |
| `data/models/meta_learner_5d_metrics.json` | RUNTIME ML metrics | same | NO | `gitignore` + `git rm --cached` |
| `data/morning_briefing.json` | INTENTIONAL daily snapshot | `scripts/after-hours-research.bat` | NO | **LEAVE TRACKED** — user commits these |
| `data/daily_research_macro.json` | INTENTIONAL daily snapshot | same | NO | **LEAVE TRACKED** |
| `data/daily_research_quant.json` | INTENTIONAL daily snapshot | same | NO | **LEAVE TRACKED** |
| `data/daily_research_review.json` | INTENTIONAL daily snapshot | same + `data/_write_review.py` | NO | **LEAVE TRACKED** |
| `data/daily_research_signal_audit.json` | INTENTIONAL daily snapshot | same + `signal_correlation_audit.py` | NO | **LEAVE TRACKED** |
| `data/daily_research_ticker_deep_dive.json` | INTENTIONAL daily snapshot | `data/_ticker_deep_dive_writer.py` | NO | **LEAVE TRACKED** |

### Distinction: RUNTIME vs SNAPSHOT

- **RUNTIME** = appended/overwritten on each loop cycle by Layer 1 / Layer 2 /
  metals loop / scheduled tasks. Not human-maintained. Will diverge from git
  every minute.
- **SNAPSHOT** = overwritten at most once per day by an intentional research
  session. Committed by the user as part of a daily research commit
  (`docs+data: after-hours research session YYYY-MM-DD`). Git history of these
  files is the deliberate research archive.

The `daily_research_*` and `morning_briefing.json` files have a documented
commit cadence (e.g. `689395ba`, `b4abf091`, `180679e5`). They're loud because
they're DIRTY now, but the *next* research-session commit will sweep them up.
Untracking would break the user's workflow.

### Other currently-clean tracked-runtime files (not on dirty list, but
should be ignored to prevent next-cycle divergence)

| File | Verdict | Action |
|------|---------|--------|
| `data/prophecy.json` | RUNTIME (rewritten via `atomic_write_json`) but currently identical to HEAD | LEAVE TRACKED — file is part of bootstrap state, see SOPs |
| `data/metals_swing_state.json` | RUNTIME (every metals loop tick) | LEAVE TRACKED — same reason; bootstrap seed |
| `data/metals_positions_state.json` | RUNTIME (fin_fish writes) | LEAVE TRACKED — bootstrap seed |
| `data/temporal_patterns.json` | STATIC config (one commit ever) | LEAVE TRACKED — config |
| `data/shadow_registry.json` | STATIC config (one commit ever) | LEAVE TRACKED — config |
| `data/consensus_replay_20260416.json` | one-off snapshot | LEAVE TRACKED |
| `data/liberation_day_playbook.json` | static research note | LEAVE TRACKED |

**Decision rationale for "currently-clean tracked-runtime":** these files have
been managed this way for months without divergence pain. Untracking them now
would (a) be out of trigger scope, (b) require coordination with the loop's
bootstrap path that reads them at startup. We fix what's actively broken (the
DIRTY runtime files), not files that work today.

## Plan of action

### Batch 1 — Fix critical_errors.jsonl divergence (the trigger)

1. Add `data/critical_errors.jsonl` to `.gitignore`.
2. `git rm --cached data/critical_errors.jsonl` — the live 104-line file stays
   on disk; only the index entry is removed.
3. Verify `scripts/check_critical_errors.py` still finds and reads the live
   file (it uses an absolute path computed from `__file__`, so untracking has
   no effect on runtime behavior).
4. Commit: `chore(git): untrack data/critical_errors.jsonl runtime journal`.

### Batch 2 — Untrack other runtime journals (broader fix)

Add to `.gitignore`:
- `data/forecast_health.jsonl`
- `data/forecast_predictions.jsonl`
- `data/sentiment_ab_log.jsonl` (already accidentally NOT in gitignore even
  though `data/signal_log.jsonl` etc. are)
- `data/auto-improve-progress.json`
- `data/after-hours-research-progress.json`
- `data/models/meta_learner_*.joblib`
- `data/models/meta_learner_*_metrics.json`

Run `git rm --cached` on each. Run `git rm --cached
data/accuracy_snapshots.jsonl` (gitignored already, just untrack).

Commit: `chore(git): untrack runtime journals & ML model artifacts`.

### Batch 3 — Audit `.gitignore` organization (light touch)

Don't refactor — preserving exact exclusion semantics matters more than tidy.

I'll only:
- Group the new entries with the existing "Runtime data" / "Scratch/debug
  scripts" sections (don't scatter them).
- Add a short comment block explaining the RUNTIME vs SNAPSHOT distinction
  for future maintainers, so they know not to gitignore `daily_research_*`.
- Note that `models/*.joblib` (root) does NOT cover `data/models/*.joblib`.

I will NOT consolidate the 40+ specific entries into globs in this PR. The
specific entries are valuable: someone added them one at a time after each
incident. A glob like `data/*_state.json` could over-match
(e.g. `metals_swing_state.json` is intentionally tracked) and silently
re-untrack files we want tracked. Out of scope.

## What could break

1. **Loop reads the staged-but-deleted file** — No. `git rm --cached` only
   removes the index entry. The on-disk file is untouched.
2. **Future merge from main brings down the deletion** — Yes, this is the
   point. Other working trees will see the file untracked but their on-disk
   copy preserved. New clones will not have the file at all, but the loop
   creates it on first append (`atomic_append_jsonl` handles missing files).
3. **CI/tests reference the file** — Need to check. `accuracy_snapshots.jsonl`
   has a backfill script (`scripts/backfill_accuracy_snapshots.py`); that
   script writes the file, doesn't depend on tracking. Tests in
   `tests/test_*` may fixture-mock the path. Will run focused tests.
4. **Daily research commits get noisier** — No, those files stay tracked.
5. **`.gitignore` semantics change for unrelated patterns** — No, only adding
   lines, not modifying.

## Test plan

1. `python scripts/check_critical_errors.py --days 30` before and after the
   `git rm --cached`. Output must be identical.
2. `pytest tests/ -k "critical_errors or accuracy_snapshots or forecast_health
   or forecast_predictions or sentiment_ab"` — focused tests on the affected
   paths.
3. Spot check: the live loop processes will NOT be touched. The worktree is
   isolated from `Q:\finance-analyzer\data\` only at the file-tracking level
   (the actual files in this worktree are HEAD copies, not the live ones).

## Out of scope

- Refactoring `.gitignore` patterns into globs (preserves semantics).
- Untracking the `daily_research_*` snapshots (intentional).
- Untracking `prophecy.json` / `metals_*_state.json` (working as intended,
  bootstrap seed).
- Modifying any code in `portfolio/` or `scripts/`.
- The `data/.*.lock` lockfile noise in `git status` — those are already
  ignored implicitly by the loop's own gitignore (or should be); separate
  follow-up if not.
