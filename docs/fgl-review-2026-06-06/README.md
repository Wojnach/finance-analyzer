# FGL Adversarial Review — 2026-06-06

Full-codebase adversarial review, 8 subsystems. Read **`01-SYNTHESIS.md`** first — it
cross-critiques every subagent finding (3 rejected/re-graded), ranks survivors by money/
reliability impact, and names the 4 systemic themes.

## Method
- Partition into 8 subsystems → 8 `review/sub-*` git branches in worktree
  `Q:/finance-analyzer-review`, each committed against `review/empty-baseline` (empty tree) so the
  whole subsystem renders as one additive diff (`git diff review/empty-baseline review/sub-<name>`).
- One fresh review subagent per subsystem, run in parallel in the background:
  6× `pr-review-toolkit:code-reviewer` (broad), 2× `caveman:cavecrew-reviewer` (tight: avanza, infra).
- Independent orchestrator pass (`00-own-pass.md`) on the foundational + integration-seam files,
  written before collecting subagent output, used to establish ground truth and catch false positives.
- Cross-critique: orchestrator read the cited code for high-severity claims; 2 infra false positives
  + 1 re-grade overturned.

## Files
| File | Subsystem | Reviewer | Headline |
|---|---|---|---|
| `00-own-pass.md` | cross-cutting | orchestrator | kill-switch asymmetry, lock-contract gaps, ground truth |
| `01-SYNTHESIS.md` | all | orchestrator | cross-critique + ranked P0/P1 + systemic themes |
| `02-signals-core.md` | signals-core | code-reviewer | no hot-path P0; seasonal post-cap inflation (P1) |
| `03-orchestration.md` | orchestration | code-reviewer | sequential specialist wait, no L2 timeout |
| `04-portfolio-risk.md` | portfolio-risk | code-reviewer | unlocked warrant book, fail-toward-trading guards |
| `05-metals-core.md` | metals-core | code-reviewer | **naked overnight exposure** (EOD stop strip) |
| `06-avanza-api.md` | avanza-api | cavecrew-reviewer | **silent failed stop-delete → overfill** (confirmed) |
| `07-signals-modules.md` | signals-modules | code-reviewer | crypto_evrp direction vs docstring |
| `08-data-external.md` | data-external | code-reviewer | http_retry unbounded sleep, stale-as-live cache |
| `09-infrastructure.md` | infrastructure | cavecrew-reviewer | 2 false positives caught; config-write coupling |

## Headline tally (post-cross-critique)
**7 P0** (1 confirmed by orchestrator, 3 verify-recommended, 3 clear): grid EOD stop-strip,
session-roll abandon, avanza silent stop-delete, http_retry stall, stale-as-live cache, sequential
specialist wait, no L2 wall-clock timeout. **~12 P1**, **~10 P2**, P3 maintainability.
**2 reviewer findings rejected as false positives** (infra `seek(-1)` UB, infra `shared_state` deadlock).

## Scope
Review-only. No code changed. Fixes are a follow-up session — see `01-SYNTHESIS.md` → "Suggested fix
order". No live config/weight/threshold changes without human approval (per `docs/GUIDELINES.md`).
