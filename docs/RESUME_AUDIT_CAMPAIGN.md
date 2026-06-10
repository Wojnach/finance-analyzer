# RESUME — Audit-Fix Campaign (read this INSTEAD of /digest-project)

> For the next session: do NOT re-digest the project or re-run the audit. Everything you
> need is in this file + the two linked docs. Written 2026-06-11 ~01:00 CET when the
> session hit its usage limit mid-campaign.

## What this campaign is

A 2026-06-10 multi-agent audit produced `docs/IMPROVEMENT_AUDIT_2026-06-10.md` —
138 findings (36 P1, 68 P2, 34 P3, 3 refuted), every P0/P1 skeptic-verified, full
descriptions + suggested fixes + file:line. User said: **"fix all of it, batch by
batch"**, following the **/fgl protocol** (explore → plan → premortem → implement →
adversarial review → test → merge → push).

The execution plan with per-batch specs lives in **`docs/PLAN.md` on branch
`fix/audit-batches-20260610`** (worktree `.worktrees/audit-fixes`). Its `## Premortem`
section has 14 failure narratives with **BINDING hooks** per batch — read it before
implementing any batch.

## Where things stand (2026-06-11 early AM)

| Batch | Status |
|---|---|
| B1 ops-automation | DONE, merged `31df4c77`, retro-reviewed, **pushed** |
| B2 live-incident signals | DONE, merged `0d0d2e94`, retro-reviewed, **pushed** |
| plan + premortem + retro fixes | DONE, merged `bb72e445`, **pushed** |
| B3 prophecy | Commit `68546e7d` ON BRANCH (pushed), **NOT merged to main** — review fixes were IN FLIGHT when the session died (see below) |
| B4-B12 | Not started — specs in PLAN.md |

Both `main` and the branch are pushed to origin.

### B3 — first thing to do next session

A background agent was applying 5 adversarial-review fixes to `68546e7d` when usage ran
out. Check `git -C .worktrees/audit-fixes log --oneline -3` and `status --short`:

- If a commit like "fix(prophecy): review fixes for 68546e7d" exists → merge branch to
  main, push, proceed to B4.
- If instead there are UNCOMMITTED changes (last seen: `M scripts/prophecy-daily.bat`,
  new `prophecy/write_guard.py`) → finish the work yourself, guided by this fix list:
  1. **Write-scoping doubt (security):** reviewer claims claude CLI `--allowedTools`
     does not honor `Write(data/prophecy_runs/**)` path syntax (tool names only) → Write
     would be unrestricted for the headless web-researching prophecy agent. Regardless
     of how the syntax question resolves, add a post-run integrity guard to
     `scripts/prophecy-daily.bat`: snapshot `git status --porcelain` before the claude
     call, compare after; any modified tracked file or new file outside
     `data/prophecy_runs/` → rate-limited critical (category `prophecy_write_breach`),
     exit non-zero, skip publish. (`prophecy/write_guard.py` was being created for
     this.) Add "verify Write denial outside data/prophecy_runs/" to the unfreeze smoke
     checklist in the .bat header.
  2. Fix misleading fail-closed ordering comment in the .bat (code exits before
     alerts.py runs; sentinel recreation is belt-and-braces, not a strict ordering).
  3. `prophecy/cost.py` ~339: move soft-cap-warning rationale comment outside the `if`.
     (Last-seen transcript suggests this one was already applied.)
  4. `prophecy/outcomes.py` ~666: comment documenting daily-bar weekend scoring latency
     (MSTR/oil Fri-targeting predictions score Monday+1).
  5. `tests/test_prophecy_pipeline.py` ~1232: `len(missing) == 12` →
     `len(pcfg.enabled_instruments()) - 1`.
  Then: run `tests/test_prophecy_pipeline.py`, commit
  ("fix(prophecy): review fixes for 68546e7d"), merge to main, push.

## Per-batch flow (repeat for B4 → B12)

1. Read the batch spec in `docs/PLAN.md` (worktree) + its binding premortem hooks.
2. Spawn ONE implementer agent (general-purpose) working in
   `/mnt/q/finance-analyzer/.worktrees/audit-fixes`. Prompt it with: the PLAN.md batch
   section, the matching audit-report section path, the conventions block (atomic I/O
   via portfolio.file_utils; dated rationale comments; never read config.json; tests via
   `/mnt/q/finance-analyzer/.venv/Scripts/python.exe -m pytest` from the worktree), and
   the prepared commit message. See `.remember/remember.md` for the established pattern.
3. Spawn `caveman:cavecrew-reviewer` on the batch commit. Fix P1/P2 (SendMessage back to
   the implementer is cheapest — it has context). Document P3 in PLAN.md.
4. Commit → merge to main → push main + branch
   (`cmd.exe /c "cd /d Q:\finance-analyzer && git push"` — this WORKED on 2026-06-11,
   despite older memory saying the classifier blocks it).
5. Append one line to `.remember/remember.md`.

**Usage discipline (user constraint):** one implementer + one reviewer per batch, no
extra agents, terse prompts, no /digest-project, no re-audit. Batch order can be
re-prioritized if usage is tight — B4 (real money) and B5 (orchestration) matter most;
B10-B12 are cheapest.

## Remaining batch one-liners (full specs in PLAN.md)

- **B4 metals real-money** — Playwright single-thread executor pinning, get_open_orders
  error propagation (sweep ALL call sites — metals_loop:5497, dashboard app.py:2142,
  grid_fisher `_safe`), grid_fisher halt-EOD-sweep + halt clear + todayClosingTime
  (fallback 21:55, hook 14), min-order exemption for closing SELLs, trailing-stop
  cancel-before-place with naked-stop critical (hook 1), getUpdates single consumer.
- **B5 orchestration** — autonomous `_regime` key fix (production path during freeze!),
  invoke_agent completion-before-spawn + fail-closed enable gate + claude_gate master
  check + PID persist/reap (psutil cmdline guard, hook 7), trigger cooldown order +
  monotonic fix, skipped_busy double-log.
- **B6 signal-core** — btc_proxy MSTR disabled-vote leak, metals MIN_VOTERS floor
  (voter_count logging + revert trigger, hook 11), seasonal-vs-cap order, applicable
  count, backfill rotation lock (chunked, hook 5), blend double-count, accuracy_cache
  pre/post snapshot + gate-flip diff (hook 10).
- **B7 portfolio-risk** — _DEFAULT_STATE deepcopy, stops branches, drawdown peak bound,
  fx band, _streaming_max robustness, trade_guards corrupt-state alert + exits bypass
  cooldown.
- **B8 swing-loops** — MSTR SHORT priced off BEAR cert, live cash sync or fail-loud,
  EOD backstop, PHASE validation, crypto 0.0 guard, oil `bar_ts`/`fetched_ts` SPLIT
  (hook 4 — do NOT just restamp ts), fast-tick 10s→60s.
- **B9 infra** — msvcrt bounded retry + Windows contention test (hook 8), retry_after
  cap, poller ack-after-success + explicit-drop Telegram reply (hook 12), prune_jsonl
  recovery decoder, atomic_write_jsonl sidecar lock (mind hook 5 lock-order), fetch_json
  kwargs.
- **B10 dashboard** — markdown XSS sanitize, token/slug log redaction, cookie secure
  per-scheme, MAX_CONTENT_LENGTH, JWKS negative cache, avanza_account queue bound.
- **B11 docs** — CLAUDE.md fact sweep (15 active signals not 21, MIN_VOTERS metals=2,
  counts/routes/tests/cadence), README, TRADING_PLAYBOOK, SYSTEM_OVERVIEW header.
- **B12 hygiene** — .gitignore + debris removal, orphaned worktrees prune (~326 MB,
  NOT audit-fixes), track RC_DISABLED doc, move data/test_metals_swing_trader.py with
  tmp_path audit (hook 9), archive data/ debug scripts.

## End-of-campaign ops (after B12, main checkout)

1. Full suite `-n auto`; baseline = 42 known pre-existing failures (recorded in B2).
2. Append resolution entries for the 16 `accuracy_degradation` criticals in
   `data/critical_errors.jsonl` (verdict: genuine June 1-6 regime crash; structural
   crypto_macro long bias fixed in B2 `0d0d2e94`).
3. Inspect `scripts/pickups/` handler for token-freeze safety, then force-run:
   `.venv/Scripts/python.exe scripts/process_pending_pickups.py --force LLM-CRYPTOTRADER-72H`.
4. USER admin (Windows, elevated): re-run `scripts/win/install-pending-pickups-task.ps1`
   (action line changed in B1); decide PF-FixAgentDispatcher (flag
   `data/fix_agent.disabled` is in place).
5. Restart PF-DataLoop + PF-MetalsLoop (schtasks via cmd.exe; fallback taskkill+
   Start-Process per memory 2026-06-03).
6. Post-unfreeze smoke (premortem hook 6): one manual `claude -p` per re-enabled path
   (layer2, prophecy) before trusting schedules; verify prophecy Write denial outside
   data/prophecy_runs/.
7. Update docs/SESSION_PROGRESS.md; delete this file and the campaign worktree
   (`git worktree remove .worktrees/audit-fixes && git branch -d fix/audit-batches-20260610`).

## Key paths

- Audit report: `docs/IMPROVEMENT_AUDIT_2026-06-10.md` (main, committed)
- Plan + premortem: `docs/PLAN.md` (branch `fix/audit-batches-20260610`)
- Worktree: `.worktrees/audit-fixes`
- Handoff log: `.remember/remember.md` (campaign entries 2026-06-10/11)
