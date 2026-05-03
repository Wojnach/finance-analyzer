# PLAN — Codex review followups for 2026-05-04 loop-infra session

**Date:** 2026-05-04
**Branch:** `fix/codex-review-followups-2026-05-04`
**Worktree:** `.worktrees/codex-fixes`

**Goal:** Address the 4 codex-review findings on the recent merged commits
(`8558fb5a..faaa32e6`). Two P2, two P3 — all real, none rollback-worthy.

---

## Findings (from codex agent, ranked by severity)

### P2-1: `_read_jsonl` 4MB tail cap can under-deliver entries
**Where:** `dashboard/app.py:_read_jsonl` (commit `faaa32e6`)
**Risk:** Large-window callers like `/api/telegrams` (read_limit=5000)
where individual rows can be up to 4096 chars (per `portfolio/message_store.py`).
With `tail_bytes = max(512_000, min(4_000_000, limit * 1024))` and
`limit=5000`, we cap at 4 MB. 5000 × 4096 ≈ 20 MB of data — we silently
return fewer entries than asked, missing older category/search matches.
**Fix:** When the parsed row count is short, retry with a larger
`tail_bytes` (doubling, up to file size). If still short after the
file-size attempt, fall through to the full-scan path.

### P2-2: Persistent prewarm protected only by `threading.Lock`
**Where:** `portfolio/accuracy_stats.py:maybe_prewarm_dashboard_accuracy`
(commit `9d5e5328`)
**Risk:** Two processes (e.g., main loop + a manual prewarm trigger,
or two main-loop instances during a botched restart) can both load the
same stale ts, both pass the gate, and both run the full prewarm before
either persists the new ts. Theoretical today (only main loop calls
this), but the cost of fixing now is small.
**Fix:** Wrap the lazy-load + gate-check + persist sequence in a file
lock (`portfolio.process_lock` already exists for golddigger / metals
singletons; reuse the helper).

### P3-1: `load_jsonl_tail` boundary bug
**Where:** `portfolio/file_utils.py:load_jsonl_tail` (NOT introduced by
this session but newly exposed by commit `faaa32e6`)
**Risk:** When `offset > 0`, the function unconditionally drops the
first decoded line. If the seek lands exactly on a `\n` boundary (no
truncation), the first line is valid and loses one entry. Off-by-one
that probably has been latent in this function since it was added.
**Fix:** Read one extra byte before `offset` and check whether the
prior byte is `\n`. If yes, the seek is on a boundary and the first
line is intact — keep it. If no, drop the first line as today.

### P3-2: Heartbeat wrapper coercion outside try/except
**Where:** `data/crypto_loop.py:write_heartbeat`,
`data/oil_loop.py:write_heartbeat` (commit `e9d5e0d1`)
**Risk:** The migration moved `dict(extra or {})`, `int(extra.pop(...))`,
`bool(extra.pop(...))` outside the `try` block. Pre-refactor, all errors
in the heartbeat path were swallowed at the `try`. Now if a caller
ever passes `cycle="N/A"` or `extra` is a non-dict, the wrapper raises.
Live trading loops cannot raise here.
**Fix:** Move the coercion inside the `try` block. If conversion fails,
default to safe values (`cycle=0`, `n_positions=0`).

---

## Out-of-scope

- The 63 pre-existing test failures from the parallel suite run match the
  documented config-drift list. Reconciling them is its own session.
- Worktree-with-codex-review-before-merge is not applicable retroactively;
  this PR backfills the verification step.

## Execution

1. Worktree created ✅
2. Commit this plan
3. Batch 1: fix P2-1 (`_read_jsonl` retry + fallback) — biggest user impact
4. Batch 2: fix P3-1 (`load_jsonl_tail` boundary)
5. Batch 3: fix P3-2 (wrapper coercion safety)
6. Batch 4: fix P2-2 (cross-process file lock for prewarm)
7. Tests after each batch
8. Merge, push, restart dashboard + main loop
9. Cleanup worktree
