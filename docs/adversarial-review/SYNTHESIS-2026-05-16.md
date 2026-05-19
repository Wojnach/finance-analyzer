# Dual Adversarial Review — 2026-05-16

## Overview

Two independent adversarial code reviews of the finance-analyzer codebase, one
per side, partitioned into 8 subsystems:

| Subsystem | Files in scope |
| --- | --- |
| signals-core | 9 |
| orchestration | 14 |
| portfolio-risk | 11 |
| metals-core | 17 |
| avanza-api | 17 |
| signals-modules | 18 |
| data-external | 13 |
| infrastructure | 18 |
| **Total** | **117** |

Both sides operated on the same empty-baseline branches off `main @ 27cb7c79`.

| Side | Engine | Sandbox |
| --- | --- | --- |
| Claude (Opus 4.7 1M) | `caveman:cavecrew-reviewer` subagents, 8 parallel | read-only via Read/Grep/Bash |
| Codex (gpt-5) | `codex exec -s read-only` non-interactive, 8 parallel | read-only |

Each side then critiqued the other's review (cross-critique) and findings were
synthesized into per-subsystem and master summaries.

## Top P1 Findings (cross-validated by both sides)

_Filled after both sides complete and cross-critique runs. See per-subsystem
files in `docs/adversarial-review/`._

## Per-Subsystem Sources

| Subsystem | Claude review | Codex review | Cross-critique |
| --- | --- | --- | --- |
| signals-core | `claude/2026-05-16-signals-core.md` | `codex/2026-05-16-signals-core.md` | `crit/2026-05-16-signals-core-*.md` |
| orchestration | `claude/2026-05-16-orchestration.md` | `codex/2026-05-16-orchestration.md` | `crit/2026-05-16-orchestration-*.md` |
| portfolio-risk | `claude/2026-05-16-portfolio-risk.md` | `codex/2026-05-16-portfolio-risk.md` | `crit/2026-05-16-portfolio-risk-*.md` |
| metals-core | `claude/2026-05-16-metals-core.md` | `codex/2026-05-16-metals-core.md` | `crit/2026-05-16-metals-core-*.md` |
| avanza-api | `claude/2026-05-16-avanza-api.md` | `codex/2026-05-16-avanza-api.md` | `crit/2026-05-16-avanza-api-*.md` |
| signals-modules | `claude/2026-05-16-signals-modules.md` | `codex/2026-05-16-signals-modules.md` | `crit/2026-05-16-signals-modules-*.md` |
| data-external | `claude/2026-05-16-data-external.md` | `codex/2026-05-16-data-external.md` | `crit/2026-05-16-data-external-*.md` |
| infrastructure | `claude/2026-05-16-infrastructure.md` | `codex/2026-05-16-infrastructure.md` | `crit/2026-05-16-infrastructure-*.md` |

## Aggregated Severity Counts

_To be filled after both sides complete._

## Methodology

1. **Worktree branching.** 8 baseline branches off `main @ 27cb7c79`, one per
   subsystem, in `Q:/fa-adversarial-wt/`. Codex was given each worktree as
   working dir via `codex exec -C`.
2. **Codex prompt.** Read-only adversarial review of named files only, with
   severity tags (P1/P2/P3) and file:line citations. See
   `Q:/fa-adversarial-wt/PROMPT_TEMPLATE.txt`.
3. **Claude prompt.** `caveman:cavecrew-reviewer` subagent with same scope and
   format, but allowed to use Read/Grep/Bash on the primary worktree directly.
4. **Cross-critique.** Each side's findings were passed back to the other side
   as the "original review" with instructions to grade each finding for
   validity, severity, and fix correctness, plus add up to 5 missed findings.
5. **Synthesis.** This document aggregates and de-duplicates across sources.

## Outcome

This review is documentation-only. No code changes were made — findings will
be triaged into `docs/adversarial-review/IMPROVEMENT_BACKLOG.md` follow-ups.

Baseline branches were deleted after review docs were committed to `main`.
