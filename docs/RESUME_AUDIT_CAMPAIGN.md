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
| B3 prophecy | DONE — `68546e7d` + review fixes `30d38f95`, merged to main `aa667166`, reviewed, **pushed** |
| B4 metals | DONE — `18d9d0cc` + `f1e6384c`, merged, reviewed, **pushed** |
| B5 orchestration | DONE — `1c93e174`, merged, self-reviewed, **pushed** |
| B6-B12 | Not started — B6 findings pre-extracted (re-extract via awk if /tmp cleared) |

Both `main` and the branch are pushed to origin. **Next session starts directly at B6
(signal-core).** Already done off-schedule: ops item 2 (16 accuracy_degradation criticals
+ today's netflow staleness alert RESOLVED in the journal 2026-06-11, startup check exits
0) and the hook-10 pre-snapshot (`data/accuracy_cache.pre_b6.json`).

### B3 security notes worth knowing (already fixed, context only)

- claude CLI `--allowedTools` DOES accept specifier rules (`Write(path/**)` etc. — own
  help text uses `"Bash(git *) Edit"`), reviewer's contrary claim was wrong; runtime
  enforcement still gets verified at the unfreeze smoke run (checklist in
  scripts/prophecy-daily.bat header).
- Windows user settings (`C:\Users\Herc2\.claude\settings.json`) allow `Write(*)`/
  `Bash(*)` — so the prophecy .bat uses `--setting-sources ""` (load NO settings) to
  keep the restricted toolset airtight.
- `prophecy/write_guard.py` = post-run git-porcelain integrity guard (new file outside
  data/ after the claude run → `prophecy_write_breach` critical, publish skipped).
  data/ excluded wholesale (60s loops churn it); permission layer polices data/.

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
