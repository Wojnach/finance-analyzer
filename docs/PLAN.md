# PLAN — FGL Adversarial Codebase Review (2026-05-29)

## Goal
Full adversarial review of finance-analyzer, partitioned into 8 subsystems. Each
subsystem reviewed by a fresh Claude Code review subagent (independent context) +
my own independent adversarial pass. Cross-critique both, synthesize into one doc.
Commit all docs to main, push via Windows git, clean up review refs.

This is a **review-only deliverable** (docs, no code changes).

## Partition (8 subsystems)

1. **signals-core** — voting engine, weighting, accuracy infra
2. **orchestration** — Layer 1 loop, Layer 2 subprocess, triggers, journal
3. **portfolio-risk** — portfolio state, risk, sizing, guards, exits
4. **metals-core** — metals loop, swing trader, grid fisher, fishing
5. **avanza-api** — Avanza session/orders/auth (cavecrew-reviewer)
6. **signals-modules** — active signal modules + LLM signals
7. **data-external** — collectors, external APIs, precompute
8. **infrastructure** — IO, health, locks, notifications, dashboard (cavecrew-reviewer)

(Full file lists embedded in each subagent prompt.)

## Mechanic — empty-baseline diff (deviation from literal "8 worktrees")
data/ is 1.5G → 8 full worktrees ≈ 12GB + slow Windows checkouts. The whole-subsystem-diff
requirement is satisfied by diffing against the empty tree, no working-tree copy needed:
`git diff fgl-empty-baseline HEAD -- <files>` renders each file as a full-add diff.
Created one `fgl-empty-baseline` orphan ref + 8 `review/fgl-2026-05-29-<subsystem>` refs (free).
GUIDELINES.md explicitly permits "git worktree **or** branch." Reviewers are read-only on
source and emit findings as their result → no collision, no isolation needed beyond refs.

## Severity scale
- **P0** — live trading / loop crash / data corruption / auth outage / silent-failure (exit 0 wrong) / wrong-direction trade
- **P1** — incorrect signal/accuracy/risk math, race, missing guard, security
- **P2** — robustness, error handling, maintainability with correctness impact
- **P3** — style/nits (documented, not necessarily fixed)

## Adversarial themes (from known failure history)
silent failures (subprocess exit 0 while broken — the 3-week auth outage), atomic-I/O
violations, direction-blindness (SHORT path half-wired), Layer 2 journal contract violations
(39 unresolved in critical_errors.jsonl as of this session), accuracy-gate inversions,
concurrency races on shared JSON state, Avanza stop-loss API misuse (instant-fill incident).

## Execution
1. Plan + premortem (below), commit.
2. Spawn 8 background review subagents (6× pr-review-toolkit:code-reviewer for broad
   subsystems, 2× caveman:cavecrew-reviewer for avanza-api + infrastructure). Each writes
   `docs/reviews/2026-05-29-fgl/<subsystem>.md` + returns compact summary.
3. While they run: write my own independent adversarial pass → `_my-independent-pass.md`.
4. Collect subagent results.
5. Cross-critique (my pass vs subagents; flag agreements, contradictions, likely false positives).
6. Synthesis → `SYNTHESIS.md` (deduped, severity-ranked, theme-clustered).
7. Commit all docs to main, push via Windows git, delete review refs.

## Premortem (review-failure modes — adapted; this deliverable changes no code)
Standard production-incident premortem assumes code changes. This is docs-only, so the real
risk is **review quality**, not a prod incident. Failure modes:

1. **Hallucinated findings** — agent reports a bug in code it didn't read (cited line wrong /
   behavior misread). *Detection:* synthesis requires every P0/P1 to cite `file:line`; my
   independent pass + cross-critique spot-check high-severity claims.
2. **False-positive flood drowns real P0s** — low-confidence noise erodes trust. *Detection:*
   severity discipline; synthesis dedups and demotes uncorroborated P2/P3.
3. **Coverage gap at partition seams** — a cross-subsystem bug (trigger → agent_invocation →
   journal contract) falls between two reviewers. *Detection:* my pass targets cross-subsystem
   flows; synthesis has a "seam risks" section.
4. **Stale-code findings** — agent flags an issue fixed in a recent commit. *Detection:*
   reviewers diff current HEAD; cross-critique checks recent git-log themes.
5. **Agent silent-failure (meta)** — a background reviewer exits without writing its file or
   returns empty (the very theme we hunt). *Detection:* verify all 8 output files exist +
   non-trivial before synthesis; re-run any no-show.

ACCEPT: no live-trading risk since no code merges; only docs land on main.
