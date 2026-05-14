# Plan: Layer 2 Tier Performance Pass

**Branch:** `perf/layer2-tier-perf`
**Started:** 2026-05-14
**Scope:** Reduce wasted tool-call roundtrips across T1/T2/T3, give every tier
budget headroom that compensates for unavoidable sequential reads.

## Context

Three commits already shipped to main in this work:
- `89aa6f68` — cost-tracking infra (claude_gate json output, bigbet/iskbets routing, scripts/claude_cost_report.py)
- `aec5ad19` — stdin=DEVNULL + auth-error cooldown
- `68b99d2e` — T1 timeout 120 → 150
- `9991a0e5` — T1 prompt: 3 Reads → 1 Bash cat

T1 work confirmed reduces wall time ~25-30s per invocation. T2 and T3 still
do 5+ sequential Reads each cycle.

## Audit numbers (3d, n=45)

```
T1 duration: min=0  p25=104  p50=114  p75=122  p90=135  p95=139  max=145
T2 duration: p50=172  p95=247  max=284 (already 600s budget, healthy)
T3 duration: p50=240  p95=390  max=390 (already 900s budget, healthy)
```

T2 and T3 are not currently timing out at the budget edge, but they're
spending ~50-80s per invocation on sequential file Reads that can collapse
into one Bash cat. Same fix, same savings, just less urgent than T1.

## What to ship

### Batch 1 — T2/T3 prompt collapse + budget headroom

Files:
- `portfolio/agent_invocation.py`
  - `_build_tier_prompt` T2 branch — replace 5 sequential Reads of
    `trading_insights.md` (optional), `TRADING_PLAYBOOK.md`,
    `layer2_context.md`, `agent_context_t2.json`, both portfolio_state
    files into single Bash cat. Use `[ -f data/trading_insights.md ] && cat
    ... ; cat ...` so the optional read stays optional.
  - `_build_tier_prompt` T3 branch — same treatment for
    `TRADING_PLAYBOOK.md`, `layer2_context.md`, `agent_summary_compact.json`,
    both portfolio_state files.
  - TIER_CONFIG — bump T1 budget another 30s (150 → 180s) to leave
    real headroom on top of the Bash-cat optimization, in case other
    work creeps back in. T2/T3 stay (already plenty).

### Batch 2 — docs + tests

- `CLAUDE.md` — update T1 budget reference (150 → 180).
- `.claude/rules/infrastructure.md` — same.
- `tests/test_agent_invocation.py` — update T1 timeout assertion (150 → 180).
- Add a regression test that the T1/T2/T3 prompts contain "cat" and do
  NOT instruct the agent to Read those files individually.

### Batch 3 — adversarial review (fresh Claude Code agent, not Codex)

Spawn a `feature-dev:code-reviewer` agent on the worktree branch. Provide:
- Diff range (`git log main..perf/layer2-tier-perf`)
- Focus areas: prompt correctness, optional-file handling for
  trading_insights.md, broken instructions if the cat command fails
  partially.
- Output contract: file:line findings, severity.

Fix any P1/P2 finding before merging.

### Batch 4 — test, merge, push

- `.venv/Scripts/python.exe -m pytest tests/ -n auto` (full suite)
- Fix any new failures (NOT the 26 known pre-existing).
- Merge `perf/layer2-tier-perf` into `main`.
- `cmd.exe /c git push`.
- Restart `PF-DataLoop` via schtasks.
- Clean up worktree.

## What could break

1. **trading_insights.md optional read** — if the cat partial-fails on a
   missing optional file, the agent might bail. Use `[ -f X ] && cat X`
   pattern so missing file is silent success.
2. **agent_summary_compact.json size** — ~1400 lines, ~64KB. Cat'ing it
   into one tool result is fine size-wise (claude tool results easily
   handle this).
3. **Bash tool availability** — claude CLI is invoked with
   `--allowedTools "Edit,Read,Bash,Write"` so Bash is on the allowlist.
   Confirmed in `portfolio/agent_invocation.py` cmd build.
4. **Prompt-following** — agents sometimes ignore "do NOT Read X" and
   read anyway. If observed post-deploy, we don't lose anything (just
   the savings). Worst case is same-as-before.

## Verification

After merge + restart, watch `data/invocations.jsonl` for:
- T1 p50 drop from 114s → ~85-95s (already in flight from `9991a0e5`)
- T2 p50 drop from 172s → ~120-140s
- T3 p50 drop from 240s → ~180-200s
- Zero new timeouts at the budget edges
- Zero status="incomplete" rows linked to truncated prompts

Compare 3d before/after via `scripts/claude_cost_report.py --days 3`.
