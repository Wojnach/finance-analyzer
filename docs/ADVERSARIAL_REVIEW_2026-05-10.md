# Dual Adversarial Review — 2026-05-10

## Context

Full dual adversarial review of `finance-analyzer` codebase. Two independent
reviewers (Claude Opus 4.7, Codex GPT-5.4 xhigh) audit each subsystem
separately, then critique each other's findings. Final synthesis prioritises
findings confirmed by both reviewers and surfaces high-conviction unique
findings from either side.

## Subsystem Partition (8)

| Subsystem | Coverage |
|---|---|
| signals-core | `signal_engine.py`, `signal_registry.py`, `accuracy_stats.py`, `outcome_tracker.py`, `forecast_accuracy.py` |
| orchestration | `main.py`, `agent_invocation.py`, `trigger.py`, `market_timing.py`, `autonomous.py`, `claude_gate.py`, `gpu_gate.py`, `session_calendar.py` |
| portfolio-risk | `portfolio_mgr.py`, `trade_guards.py`, `risk_management.py`, `equity_curve.py`, `monte_carlo.py`, `monte_carlo_risk.py`, `accuracy_degradation.py`, `loop_contract.py` |
| metals-core | `data/metals_loop.py`, `exit_optimizer.py`, `price_targets.py`, `orb_predictor.py`, `iskbets.py`, `fin_snipe.py`, `fin_fish.py`, `metals_orderbook.py`, `metals_cross_assets.py` |
| avanza-api | `avanza_session.py`, `avanza_orders.py`, `avanza/scanner.py` |
| signals-modules | `portfolio/signals/*.py` (49 modules) |
| data-external | `data_collector.py`, `fear_greed.py`, `sentiment.py`, `alpha_vantage.py`, `futures_data.py`, `onchain_data.py`, `fx_rates.py`, `market_health.py` |
| infrastructure | `file_utils.py`, `http_retry.py`, `health.py`, `shared_state.py`, `journal.py`, `telegram_notifications.py`, `message_store.py`, `microstructure.py`, `microstructure_state.py` |

## Worktree Layout

8 worktrees rooted at `.worktrees/adv-<subsystem>/` on branches
`review/2026-05-08-<subsystem>` (off `empty-baseline` for diff scope). Reviews
ran with `codex review --base empty-baseline` per worktree. Path issue with the
git index in worktrees forced codex to fall back to filesystem inspection;
findings are still real because reviewers operate on file content, not diff
metadata.

## Artefact Layout

```
data/adv-2026-05-08/
  claude-<subsystem>.md       # Claude independent reviews (8)
  codex-<subsystem>.md        # Codex independent reviews (8)
  cross-claude-on-codex-<subsystem>.md   # Claude critiques codex (8)
  cross-codex-on-claude-<subsystem>.md   # Codex critiques Claude (8)
  synthesis-<subsystem>.md    # Per-subsystem dual synthesis (8)
  findings.json               # machine-readable consolidated list
docs/
  ADVERSARIAL_REVIEW_2026-05-10.md       # this doc
  ADVERSARIAL_SYNTHESIS_2026-05-10.md    # top-level synthesis + priority queue
```

## Method (per subsystem)

1. **Independent reviews** — Claude and Codex separately audit the worktree,
   no awareness of the other reviewer. Each emits a Markdown finding list with
   severity (P1/P2/P3), file:line citation, problem statement, fix sketch.
2. **Cross-critique (both directions)** — each reviewer reads the other's
   findings list and labels each finding `confirmed / partial / false-positive`
   with reasoning. Disagreements are logged, not suppressed.
3. **Per-subsystem synthesis** — merge the two findings lists, dedupe, mark
   convergence (both saw it) vs divergence (only one saw it). Re-rank
   severity by conviction: convergent P1 stays P1; lone P1 → drops to P2
   unless cross-critique upgrades it back.
4. **Top-level synthesis** — aggregate across all 8 subsystems into a
   prioritised remediation queue, grouped by theme (consensus logic, persistence
   filter, accuracy gate, MC risk, broker session, etc.).

## Why dual

Single-reviewer LLM passes hallucinate findings ("ghost bugs") about 15-25% of
the time at this depth. Dual-review with cross-critique cuts that rate hard
because false positives rarely survive scrutiny from a second model with full
file access. Convergent findings are very high-confidence; lone-reviewer
findings get a probabilistic discount in the synthesis.

## Output of this session

This doc is the index. The synthesis lives in
`docs/ADVERSARIAL_SYNTHESIS_2026-05-10.md`. The raw evidence (claude/codex/cross
files per subsystem) lives under `data/adv-2026-05-08/`.

## Status

- Independent reviews: complete (16 docs, 68k lines codex + ~860 lines claude).
- Cross-critique: written this session.
- Synthesis: written this session.
- Implementation of findings: NOT in scope of this session — handed to the
  remediation queue in the synthesis doc.

## Not in scope

Code fixes. The pipeline is review → cross-critique → synthesis → remediation
queue. Acting on findings is a separate session (per CLAUDE.md: never deploy
untested changes; review work alone is high-leverage).
