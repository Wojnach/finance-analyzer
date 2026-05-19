# 2026-03-17 Merge / Port Plan

## Goal

Convert the current branch review into a safe, staged integration plan for `main` without
blindly merging stale branches or generated runtime artifacts.

## Current State

- `feat/model-upgrades` is already merged into `main`.
- The current checkout still has a small verified local follow-up patch plus a large amount of
  generated/runtime noise in `data/` and `dashboard/static/api-data/`.
- Several unmerged branches are not safe to merge wholesale because they are older than current
  `main`, contain machine-local workflow files, or need fixes first.

## Guardrails

- Do not commit generated/runtime files from `data/` or dashboard snapshot JSON unless explicitly wanted.
- Keep the current local follow-up patch separate from branch integrations.
- Prefer porting/cherry-picking focused changes over merging older branches wholesale.
- Re-run targeted tests after each meaningful integration step.

## Phase 1 — Land The Current Local Follow-Up Patch

Scope:
- `portfolio/qwen3_trader.py`
- `portfolio/message_store.py`
- `portfolio/telegram_notifications.py`
- `docs/CHANGELOG.md`
- `tests/test_message_store.py` if included with the patch

Why:
- This patch is already understood and verified.
- It should not get mixed with unrelated branch work.

Validation:
- `.venv\Scripts\python.exe -m pytest -q tests\test_message_store.py tests\test_model_upgrades.py tests\test_digest.py`

Status:
- Done on 2026-03-18:
  - `f92fe16 fix(notifications): add mute gate and qwen3 asset checks`

## Phase 2 — Merge The Best Immediate Candidate

Branch:
- `feat/price-targets-accuracy`

Why:
- Small, focused scope.
- Strong targeted test signal.
- Additive reporting/price-target improvements.

Validation already run:
- `Q:\finance-analyzer\.venv\Scripts\python.exe -m pytest -q tests\test_price_targets.py`
- Result: `82 passed`

Expected files:
- `portfolio/price_targets.py`
- `portfolio/reporting.py`
- `tests/test_price_targets.py`

Recommendation:
- No longer needed: by 2026-03-18 this branch was patch-equivalent to current `main`.

## Phase 3 — Port The Duplicate-Loop Fix, Don’t Merge The Whole Branch

Branch:
- `prevent-loop-duplicates`

Why:
- The useful part is the singleton lock in `data/metals_loop.py` plus its tests.
- The branch also carries assistant/workflow files that should stay out of `main`.

Use:
- Port/cherry-pick only:
  - `data/metals_loop.py`
  - `tests/test_unified_loop.py`

Do not merge:
- `AGENTS.md`
- `CLAUDE.md`
- `WORKTREE_NOTES.md`

Validation already run:
- `Q:\finance-analyzer\.venv\Scripts\python.exe -m pytest -q tests\test_unified_loop.py`
- Result: `27 passed`

Recommendation:
- No longer needed: by 2026-03-18 the singleton-lock work was already present on current `main`
  in a slightly better form.

## Phase 4 — Split The Useful Test Harness Changes

Branch:
- `improve/auto-session-2026-03-09`

Why:
- Some of the test-harness changes are useful.
- The automation script is machine-specific and uses dangerous bypass flags.

Safe subset candidate:
- `tests/conftest.py`
- `tests/integration/test_strategy.py`
- `docs/TESTING.md`

Review carefully before keeping:
- `scripts/auto-improve-codex.bat`
- `scripts/auto-improve.bat`
- `docs/auto-improve-prompt-codex.md`

Notes:
- `scripts/auto-improve-codex.bat` hardcodes `Q:\finance-analyzer`.
- The Codex runner also uses dangerous approval-bypass flags and should be parameterized or kept local.

Validation:
- Partial only. `tests/test_portfolio.py::TestIntegrationHerc2::test_full_report` was skipped in the worktree.

Recommendation:
- Split this branch; do not merge it wholesale.

## Phase 5 — Deliberate Review Of The Larger Accuracy Branch

Branch:
- `local-llm-accuracy-inrepo`

Why:
- Good targeted test signal and useful functionality.
- But it changes model-vote gating and dashboard/reporting behavior, so this is not a blind merge.

Validation already run:
- `Q:\finance-analyzer\.venv\Scripts\python.exe -m pytest -q tests\test_local_llm_report.py tests\test_local_llm_accuracy.py tests\test_dashboard.py tests\test_dashboard_export_static.py tests\test_forecast_accuracy_gating.py`
- Result: `155 passed`

Expected files include:
- `portfolio/local_llm_report.py`
- `portfolio/signals/forecast.py`
- `portfolio/signal_engine.py`
- `dashboard/app.py`
- related tests/docs/task scripts

Recommendation:
- Review once more for operational impact, then merge if still desired.
- This is a second-tier candidate after Phases 1-4.

## Phase 6 — Fix Before Merge

### `golddigger-signal-upgrade`

Reason to hold:
- Adds useful overlays and tests, but changes live trading behavior.
- Event-risk timing logic appears wrong for some release types.

Known issue:
- Event-risk window logic hardcodes a single UTC release time and needs correction before merge.

Validation already run:
- `Q:\finance-analyzer\.venv\Scripts\python.exe -m pytest -q tests\test_golddigger.py`
- Result: `105 passed`

Recommendation:
- Fix the event window logic first, then re-evaluate merge readiness.

### `worktree-fix-l3-trailing-stop`

Reason to hold:
- Not merge-ready as-is.
- No dedicated branch-local tests found.

Known issue:
- Trigger-path escalation can still emit `L3 EMERGENCY` reasons even when the later sell path blocks the actual sell.

Validation:
- `Q:\finance-analyzer\.venv\Scripts\python.exe -m py_compile data\metals_loop.py`

Recommendation:
- Rework on top of current `main` after the singleton-lock port is done.

### `feat/fin-silver-command`

Reason to hold:
- Useful idea, but incomplete and untested.
- Depends on files/workflows that are not cleanly present in the branch.

Validation:
- `Q:\finance-analyzer\.venv\Scripts\python.exe -m py_compile portfolio\silver_precompute.py`

Recommendation:
- Rebase and finish later; do not merge as-is.

## Phase 7 — Optional / Skip

### Optional
- `improve/auto-session-2026-03-05`
  - Docs-only.
  - Safe but low priority.

### Skip
- `chore/persist-operator-directive`
  - Workflow/instruction files only.
- `gh-pages`
  - Do not merge into `main`.

## Recommended Order

1. Commit the current local verified follow-up patch.
2. Merge `feat/price-targets-accuracy`.
3. Port/cherry-pick only the singleton-lock/test pieces from `prevent-loop-duplicates`.
4. Split and optionally keep the test-harness subset from `improve/auto-session-2026-03-09`.
5. Do a deliberate merge decision on `local-llm-accuracy-inrepo`.
6. Fix `golddigger-signal-upgrade`.
7. Rework or shelve `worktree-fix-l3-trailing-stop` and `feat/fin-silver-command`.

## Resume Notes

If credits run out, resume from this file and today’s memory entry:
- `docs/plans/2026-03-17-merge-triage-plan.md`
- `memory/2026-03-17.md`

## 2026-03-18 Re-Triage After Local-LLM Merge

Status update:
- `local-llm-accuracy-inrepo` has now been merged onto `main` as an explicit ancestry merge.
- Current `main` already contained the branch's useful local-LLM behavior in newer form, so the merge intentionally kept current `main` semantics.
- Focused validation after that review passed:
  - `.venv\Scripts\python.exe -m pytest -q tests\test_local_llm_accuracy.py tests\test_local_llm_report.py tests\test_forecast_accuracy_gating.py tests\test_dashboard.py tests\test_dashboard_export_static.py`
  - `161 passed`

Remaining branch classification:

### Already effectively present on `main`

- `feat/price-targets-accuracy`
  - `git cherry -v main feat/price-targets-accuracy` reports the commit as patch-equivalent.
  - No merge work remains.

- `prevent-loop-duplicates`
  - The useful singleton-lock idea is already represented on current `main` in a better form.
  - Do not merge the whole branch.

### Manual port only, not safe to merge wholesale

- `golddigger-signal-upgrade`
  - Reviewed again on 2026-03-18 and merged as explicit ancestry only.
  - Current `main` already contained the useful GoldDigger behavior in newer form, so no tree changes were taken from the old branch.
  - Validation after review:
    - `.venv\Scripts\python.exe -m pytest -q tests\test_golddigger.py`
    - `107 passed`

- `feat/fin-silver-command`
  - Real feature commit exists (`bc89216`) and only introduces:
    - `.claude/commands/fin-silver.md`
    - `.gitignore`
    - `portfolio/silver_precompute.py`
  - Current `main` already has a compatibility-wrapper `portfolio/silver_precompute.py` delegating to `portfolio.metals_precompute`.
  - Treat this branch as a redesign/port task, not a merge.

- `improve/auto-session-2026-03-09`
  - Useful commit exists (`2f0e3e5`) for test harness portability.
  - Port only:
    - `docs/TESTING.md`
    - `tests/conftest.py`
    - `tests/integration/test_strategy.py`
    - maybe the related `tests/test_portfolio.py` hunk
  - Do not merge the whole branch.

### Defer / skip

- `worktree-fix-l3-trailing-stop`
  - Still not merge-ready. Needs a fresh fix on top of current `main`.

- `improve/auto-session-2026-03-05`
  - Old docs branch. Not worth merging wholesale.

- `chore/persist-operator-directive`
  - Workflow/instruction branch. Keep out of `main`.

- `gh-pages`
  - Never merge into `main`.
