# Cleanup Triage — 2026-05-19

Untracked files audit. Goal: classify each as **COMMIT-READY**, **NEEDS-WORK**,
or **STALE** so the working tree can be cleaned without losing in-progress
work. Generated after `.gitignore` was extended to cover runtime state — these
are the files that remained untracked and need human judgement.

Status legend:
- **COMMIT-READY** — finished, in use, low-risk; should be committed.
- **NEEDS-WORK** — incomplete, has TODOs, deferred, or references missing.
- **STALE** — obsolete, dated, or already past its purpose; defer/delete.

---

## High priority — broken-import or active references

| File | Mtime | Status | Reason | Recommendation |
|------|-------|--------|--------|----------------|
| `dashboard/static/js/views/cost.js` | 2026-05-13 | **COMMIT-READY** | Imported by `dashboard/static/js/main.js:37` (`import "./views/cost.js"`). Without it the bundle is broken in CI / fresh checkout. Wraps `/api/claude_cost` endpoint already implemented in `dashboard/app.py:2236`. | **Commit immediately** to restore dashboard build. |

---

## Slash commands (`.claude/commands/`)

| File | Mtime | Status | Reason | Recommendation |
|------|-------|--------|--------|----------------|
| `.claude/commands/avanza-search.md` | 2026-04-01 | COMMIT-READY | Functional `/avanza-search <query>` slash command. Calls `api_post('/_api/search/filtered-search', ...)`. Self-contained. | Commit. |
| `.claude/commands/digest-project.md` | 2026-04-09 | COMMIT-READY | Functional `/digest-project` slash command. References `.claude/skills/digest-project/SKILL.md`. | Commit. |
| `.claude/commands/time.md` | 2026-03-26 | COMMIT-READY | `/time` slash command — reports current time + open markets. | Commit. |

---

## Active infrastructure scripts

| File | Mtime | Status | Reason | Recommendation |
|------|-------|--------|--------|----------------|
| `scripts/win/rc-keepalive.ps1` | 2026-03-25 | COMMIT-READY | Soft-recycles idle RC servers (Anthropic ~20min TTL workaround). Referenced from `docs/SESSION_PROGRESS.md:648`. Live infra. | Commit. |
| `scripts/win/rc-watchdog.ps1` | 2026-03-24 | COMMIT-READY | Companion to `rc-keepalive`. Referenced from `docs/SESSION_PROGRESS.md:130,645`. | Commit. |
| `scripts/win/pf-loop-ensure.ps1` | 2026-03-18 | COMMIT-READY | Idempotent loop launcher — used on logon/wake. Referenced in `docs/adversarial-review-2026-05-12/`. | Commit. |
| `scripts/win/add-cloudflared-path.ps1` | 2026-03-12 | COMMIT-READY | One-time PATH setup for cloudflared. Small. Reviewed in adversarial-review 2026-05-11. | Commit. |
| `scripts/sysmon.py` | 2026-03-25 | COMMIT-READY | Lightweight system monitor — used by `sysmon` slash command per memory. | Commit. |

---

## Fin Snipe thin launchers (paired with `portfolio/fin_snipe[_manager].py`)

| File | Mtime | Status | Reason | Recommendation |
|------|-------|--------|--------|----------------|
| `scripts/fin_snipe.py` | 2026-03-12 | COMMIT-READY | Thin wrapper → `portfolio.fin_snipe:main`. Trivial. | Commit. |
| `scripts/fin_snipe_manager.py` | 2026-03-12 | COMMIT-READY | Thin wrapper → `portfolio.fin_snipe_manager:main`. Trivial. | Commit. |
| `scripts/win/fin-snipe.bat` | 2026-03-12 | COMMIT-READY | `-m portfolio.fin_snipe` launcher. Trivial. | Commit. |
| `scripts/win/fin-snipe-manager.bat` | 2026-03-12 | COMMIT-READY | `-m portfolio.fin_snipe_manager` launcher. Trivial. | Commit. |
| `scripts/avanza_metals_ladder.py` | 2026-03-12 | COMMIT-READY | Backward-compat wrapper → `portfolio.fin_snipe:main`. Trivial. | Commit. |
| `scripts/avanza_metals_check.py` | 2026-03-10 | COMMIT-READY | Quick Avanza metals position check (JSON stdout). Self-contained. | Commit. |

---

## Signal research tooling (auditable, used during recent research sessions)

| File | Mtime | Status | Reason | Recommendation |
|------|-------|--------|--------|----------------|
| `scripts/signal_research_extract.py` | 2026-04-28 | COMMIT-READY | Extract step of signal-research pipeline. Referenced in adversarial review. | Commit. |
| `scripts/signal_research_phase34.py` | 2026-04-23 | COMMIT-READY | Phase 3/4 of signal-research pipeline. | Commit. |
| `scripts/signal_correlation_audit.py` | 2026-04-25 | COMMIT-READY | Referenced in `docs/PLAN_git_tracking_hygiene_20260502.md:50` (writes `daily_research_signal_audit.json` — a tracked daily snapshot). The script that emits it must be tracked too. | Commit (critical for snapshot reproducibility). |
| `scripts/backtest_new_signals.py` | 2026-04-13 | COMMIT-READY | Backtest harness for new signals (RSI/BB/momentum/etc). | Commit. |
| `scripts/tune_new_signals.py` | 2026-04-13 | COMMIT-READY | Companion tuner for `backtest_new_signals`. | Commit. |
| `scripts/write_research_outputs.py` | 2026-04-26 | COMMIT-READY | Emits research output bundles. | Commit. |

---

## ML / training tooling

| File | Mtime | Status | Reason | Recommendation |
|------|-------|--------|--------|----------------|
| `scripts/benchmark_gpu_models.py` | 2026-03-17 | COMMIT-READY | Benchmarks LLM models (Ministral / Chronos / Qwen3). Useful infra. | Commit. |
| `scripts/prepare_kronos_training_data.py` | 2026-03-27 | COMMIT-READY (verify) | Training-data prep for Kronos. Pairs with `data/kronos_training/` runtime dir (now gitignored). | Commit, or move to `training/kronos/` namespace if matured. |
| `scripts/setup_wsl_claude.ps1` | 2026-03-22 | COMMIT-READY | One-time WSL bootstrap for Claude CLI. | Commit. |

---

## Auto-improve subsystem (Codex)

| File | Mtime | Status | Reason | Recommendation |
|------|-------|--------|--------|----------------|
| `scripts/auto-improve-codex.bat` | 2026-03-05 | COMMIT-READY | Codex auto-improve runner. Pairs with `docs/auto-improve-prompt-codex.md`. Runtime data already gitignored (`data/auto-improve-codex-*`). | Commit (script + doc together). |
| `docs/auto-improve-prompt-codex.md` | 2026-03-11 | COMMIT-READY | The prompt the .bat feeds Codex. Required pair. | Commit. |

---

## Adversarial review outputs (2026-05-16)

| File | Mtime | Status | Reason | Recommendation |
|------|-------|--------|--------|----------------|
| `docs/adversarial-review/SYNTHESIS-2026-05-16.md` | 2026-05-16 | COMMIT-READY | Dual review synthesis (claude + codex sides over 8 subsystems, baseline `main @ 27cb7c79`). | Commit. |
| `docs/adversarial-review/claude/` (8 files) | 2026-05-16 | COMMIT-READY | Per-subsystem review reports from claude side. | Commit. |
| `docs/adversarial-review/codex/` (8 files) | 2026-05-16 | COMMIT-READY | Per-subsystem review reports from codex side. | Commit. |

---

## Long-form design docs

| File | Mtime | Status | Reason | Recommendation |
|------|-------|--------|--------|----------------|
| `docs/GOLDDIGGER_FINAL.md` | 2026-03-09 | COMMIT-READY | 48KB design report for "BULL GULD X20 AVA" intraday bot. Companion to `portfolio/golddigger/`. | Commit. |
| `docs/PLUGINS_AND_SKILLS.md` | 2026-03-31 | COMMIT-READY | Plugin + skill catalog. | Commit. |
| `docs/SYSTEM_HEALTH_CONTRACT.md` | 2026-03-31 | COMMIT-READY | Invariants spec (companion to `data/contract_state.json` runtime). | Commit. |
| `docs/UNSLOTH_RUNTIME_LEARNINGS_2026-03-17.md` | 2026-03-18 | COMMIT-READY (dated) | Captured learnings from Unsloth runtime experiment. Historical, but useful. | Commit. |
| `docs/oil-deep-research-report.md` | 2026-03-19 | COMMIT-READY | Oil deep research (paired with `Oil-research.pdf`). | Commit (md + pdf). |
| `docs/oil-deep-research-report-2.md` | 2026-03-19 | COMMIT-READY | v2 of above (paired with `Oil-research-2.pdf`). | Commit. |
| `docs/Oil-research.pdf` | 2026-03-19 | COMMIT-READY | Source PDF for `oil-deep-research-report.md`. ~106KB. | Commit. |
| `docs/Oil-research-2.pdf` | 2026-03-19 | COMMIT-READY | Source PDF for `oil-deep-research-report-2.md`. ~141KB. | Commit. |

---

## Implementation plans (`docs/plans/`)

| File | Mtime | Status | Reason | Recommendation |
|------|-------|--------|--------|----------------|
| `docs/plans/2026-03-17-merge-triage-plan.md` | 2026-03-18 | STALE | Old merge-triage plan from March, work long shipped. | Commit as historical record OR delete. |
| `docs/plans/2026-03-18-unsloth-finetune-plan.md` | 2026-03-18 | STALE | Unsloth finetune plan; experiment shelved (5GB `.venv-unsloth/` is its leftover). | Commit as historical OR delete. |
| `docs/plans/2026-04-16-gemma4-loop-plan.md` | 2026-04-16 | NEEDS-WORK? | Gemma4 loop plan; not yet shipped per memory. Check status. | Verify status with user before commit/delete. |
| `docs/plans/2026-04-18-ic-weighting-integration.md` | 2026-04-18 | NEEDS-WORK? | IC-weighting integration plan; check if shipped. | Verify status. |

---

## Superpowers plans + specs

| File | Mtime | Status | Reason | Recommendation |
|------|-------|--------|--------|----------------|
| `docs/superpowers/plans/2026-03-27-3h-signal-optimization.md` | 2026-03-27 | STALE | 3h-signal optimization plan. | Commit as historical OR delete. |
| `docs/superpowers/plans/2026-03-30-househunting.md` | 2026-03-30 | STALE (in this repo) | 84KB Stockholm househunting plan. Belongs to `/mnt/q/househunting/`, not finance-analyzer. | Move to househunting repo OR delete. |
| `docs/superpowers/plans/2026-03-31-metals-microstructure-signals.md` | 2026-03-31 | NEEDS-WORK? | 65KB metals microstructure plan; some shipped (OFI, VPIN per memory). Verify scope completed. | Verify; commit as historical if done. |
| `docs/superpowers/plans/2026-04-04-strategy-orchestrator-merge.md` | 2026-04-04 | NEEDS-WORK? | 43KB strategy-orchestrator merge plan; verify ship status. | Verify. |
| `docs/superpowers/plans/2026-04-28-dashboard-ops-board.md` | 2026-04-28 | NEEDS-WORK? | 77KB dashboard ops board plan; check status (recent). | Verify. |
| `docs/superpowers/specs/2026-03-30-househunting-design.md` | 2026-03-30 | STALE | Househunting design — same misplacement as plan above. | Move to househunting repo OR delete. |
| `docs/superpowers/specs/2026-04-28-dashboard-ops-board-design.md` | 2026-04-28 | NEEDS-WORK? | Pairs with the 2026-04-28 plan. | Verify with plan. |

---

## PLAN-trigger-noise (explicitly deferred)

| File | Mtime | Status | Reason | Recommendation |
|------|-------|--------|--------|----------------|
| `docs/PLAN-trigger-noise.md` | 2026-04-17 | NEEDS-WORK | Status: **DEFERRED — awaiting user decision** (per file header). | Decide: ship, defer further (commit as draft), or drop. |

---

## Codex docs (loose)

| File | Mtime | Status | Reason | Recommendation |
|------|-------|--------|--------|----------------|
| `docs/Configure Codex to reduce command permission prompts for my development….md` | unknown | NEEDS-WORK | Filename has Unicode ellipsis + spaces (broken on some shells). | Rename to ASCII slug (`docs/codex-permission-prompts-config.md`) before commit. |
| `docs/codex guidelines.md` | unknown | NEEDS-WORK | Filename has space. | Rename to `docs/codex-guidelines.md` before commit. |
| `docs/deep research/` (dir) | unknown | NEEDS-WORK | Directory with space in name. | Rename to `docs/deep-research/` before commit (or move contents to `docs/research/`). |

---

## Reviews dir

| Dir | Mtime | Status | Reason | Recommendation |
|-----|-------|--------|--------|----------------|
| `docs/reviews/` (6 files) | 2026-04-16 | NEEDS-WORK (verify) | Unknown contents. Likely review outputs. | Inspect contents, commit if final reports; otherwise gitignore. |

---

## Stale / dated one-shots

| File | Mtime | Status | Reason | Recommendation |
|------|-------|--------|--------|----------------|
| `scripts/monitor_silver_exit.py` | 2026-04-23 | **STALE** | One-shot monitor for *2026-04-23* silver exit. Hard-coded date past. | Delete. |
| `scripts/cleanup_settings_20260508.sh` | 2026-05-01 | **STALE** | One-shot scheduled for 2026-05-08 09:00 CEST. Already run. | Delete (header even says "After this runs successfully, the task and this script can be deleted"). |
| `scripts/win/settings-cleanup-20260508.bat` | 2026-05-01 | **STALE** | Companion to above. | Delete. |
| `scripts/win/metals-arm-stop-once.bat` | 2026-03-05 | **STALE (BROKEN)** | Invokes `data/arm_stop_orders_once.py` which **does not exist**. | Delete. |

---

## Bucket-2 watch items (gitignored but maybe should be tracked)

These were added to `.gitignore` in this sweep, but might be intentional
scaffolds. Confirm before next push:

| File | Why might be scaffold |
|------|----------------------|
| `data/metals_warrant_catalog.json` | Per `.gitignore` comment, `crypto_warrant_catalog` + `oil_warrant_catalog` are "deliberately tracked as committed scaffolds." Metals likely same pattern. |
| `data/seasonality_profiles.json` | Could be a static profile, not runtime. |
| `data/signal_weights.json` | Could be calibrated weights checkpoint, not runtime. |
| `data/system_lessons.json` | Lessons file — accumulates over time but might be intentional. |

If any should stay tracked, remove the gitignore entry and `git add` it.

---

## Summary counts

- **COMMIT-READY**: 32 files (slash commands × 3, infra scripts × 5, fin_snipe launchers × 6, signal research × 6, ML × 3, auto-improve × 2, adversarial review × 17, long-form docs × 8, dashboard cost.js × 1)
- **NEEDS-WORK**: 12 files (plans needing ship-status verification, codex docs needing rename, reviews dir needing inspection, PLAN-trigger-noise deferred)
- **STALE**: 6 files (dated one-shots + misplaced househunting + old triage plans)

## Recommended commit order

1. **Hot fix** — commit `dashboard/static/js/views/cost.js` first (dashboard build is broken without it).
2. **Slash commands** — commit `.claude/commands/{avanza-search,digest-project,time}.md`.
3. **Infra scripts** — commit `scripts/win/rc-{keepalive,watchdog}.ps1`, `pf-loop-ensure.ps1`, `add-cloudflared-path.ps1`, `scripts/sysmon.py`.
4. **fin_snipe launchers** — small batch.
5. **Signal research tooling** — needed for `daily_research_signal_audit.json` reproducibility.
6. **Adversarial review outputs** — commit as a single review-evidence drop.
7. **Long-form design docs + plans** — bundle the COMMIT-READY ones; verify NEEDS-WORK items first.
8. **Rename + commit codex docs** with ASCII filenames.
9. **Stale deletes** last (after verifying nothing depends on them).
