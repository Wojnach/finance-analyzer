# PLAN — Reduce Claude CLI Layer 2 invocations 40/day → ~6/day

Branch: `reduce-claude-invocations`
Date: 2026-05-15

## Context

Layer 2 (Claude CLI) currently fires ~40 times/day, ~20 successful runs, mostly HOLD-HOLD outcomes. Most triggers are low-conviction flickers (consensus 16–30%, sustained-flip on noise). Local LLMs (Ministral-8B, Qwen3-8B, Chronos-2) + 33 signals already do the hard work — Claude should be the rare arbiter, not the default. Sonnet pin just shipped (cost cap), but the right fix is to call less.

User picked options 1, 2, 4, 5, 6, 7, 8 from the brainstorm. Skipped option 3 (per-ticker cooldown bump — felt like a band-aid).

## Scope — 7 changes

| # | Change | File(s) | Risk |
|---|---|---|---|
| 1 | Drop sub-40% consensus triggers entirely | `portfolio/trigger.py` | low |
| 2 | No-position-no-entry skip in `invoke_agent` | `portfolio/agent_invocation.py`, reuse `portfolio_mgr.load_state/load_bold_state` | low |
| 4 | Raise `SUSTAINED_CHECKS` 3→5 for low-density triggers | `portfolio/trigger.py` | low |
| 5 | Confidence floor + ATR floor on every trigger reason | `portfolio/trigger.py` | low |
| 6 | Autonomous-first: Layer 3 handles routine; Claude only on escalation criteria | `portfolio/main.py`, `portfolio/autonomous.py`, `portfolio/agent_invocation.py` | **med** |
| 7 | Ministral pre-gate classifier ("should_escalate?") | new `portfolio/escalation_gate.py`, hook in `portfolio/main.py` | med |
| 8 | 5-min trigger batching aggregator | new `portfolio/trigger_buffer.py`, hook in `portfolio/main.py` | low-med |

All thresholds become config keys under new section `config.claude_budget`.

## Existing primitives to reuse

- `portfolio/perception_gate.py` — already a gate point (currently disabled). Item 5 piggybacks.
- `portfolio/portfolio_mgr.py:load_state/load_bold_state` — item 2 holdings check.
- `portfolio/trigger.py:_save_state` — item 8 buffer persistence.
- `portfolio/llm_batch.py` (Ministral runner) — item 7 classifier.
- `portfolio/autonomous.py:autonomous_decision` — item 6 routine path.
- `portfolio/signal_engine.py` `weighted_confidence` output — item 5 floor.
- `portfolio/file_utils.atomic_write_json / atomic_append_jsonl` — required for any state I/O.

## Config additions (`config.claude_budget` section in `config.json`)

```jsonc
{
  "claude_budget": {
    "consensus_min_pct": 40,
    "sustained_checks_low_density": 5,
    "sustained_density_threshold": 0.40,
    "min_weighted_confidence": 0.55,
    "min_atr_multiple": 1.5,
    "no_position_skip_enabled": true,
    "autonomous_first_enabled": true,
    "escalate_drawdown_pct": 5.0,
    "escalate_top5_disagree": true,
    "ministral_pregate_enabled": true,
    "ministral_pregate_min_score": 0.5,
    "batch_window_s": 0
  }
}
```

All keys default to current behavior if absent (e.g. `consensus_min_pct: 0`, `autonomous_first_enabled: false`) so the change is opt-in via config flip. `batch_window_s` default 0 = disabled; set to 300 to enable 5-min batching.

## Implementation batches

### Batch A — Trigger gates (items 1, 4, 5)
`portfolio/trigger.py` only.
- Load `claude_budget` config at module init.
- Item 1: in consensus block (lines 212–250), require `buy_conf*100 ≥ consensus_min_pct` before emitting reason.
- Item 4: in sustained-flip block (lines 253–287), use `sustained_checks_low_density` when signal density < `sustained_density_threshold`. Density = active_voters / applicable_signals.
- Item 5: each trigger emission carries `(reason, weighted_conf, atr_mult)`. Emit only if `conf ≥ min_weighted_confidence` OR `atr_mult ≥ min_atr_multiple` OR reason_type in {"first_of_day","periodic_review","F&G_extreme","post_trade"}.
- Tests: `tests/test_trigger.py` — add cases for each new gate; ensure existing trigger paths still fire when thresholds met.

### Batch B — No-position skip (item 2)
`portfolio/agent_invocation.py` only.
- New helper `_no_position_skip(reasons) → (bool, str)`:
  - Load both portfolios via `portfolio_mgr.load_state` / `load_bold_state`.
  - Parse tickers via existing `_extract_triggered_tickers`.
  - If every ticker has zero shares in both portfolios AND no reason carries `weighted_conf ≥ 0.65` (read from buffered trigger context, see Batch C state file or re-read latest `agent_context_t1.json`) → return (True, "no_position_no_entry").
- Insert gate before subprocess spawn (around current line 800). Status `skipped_no_position`.
- Tests: `tests/test_agent_invocation.py` — held vs unheld vs entry-strong-conf cases.

### Batch C — Trigger batching (item 8)
New file `portfolio/trigger_buffer.py`.
- 5-min sliding window keyed on (ticker, reason_type).
- API: `buffer.add(reasons_with_meta, ts)`, `buffer.flush_due(now) → list[merged_reason]`.
- State: `data/trigger_buffer.json` (atomic write).
- Dedupe identical reasons; concat distinct ones within window. Flush on window expiry or T3 escalation reason present.
- Hook in `portfolio/main.py` between `check_triggers()` (line 807) and `invoke_agent()` (line 848): triggers go to buffer; only flushed reasons reach `invoke_agent`.
- Tests: new `tests/test_trigger_buffer.py`.

### Batch D — Autonomous-first (item 6)
`portfolio/main.py` + `portfolio/autonomous.py` + `portfolio/agent_invocation.py`.
- Master switch: `claude_budget.autonomous_first_enabled`. When true, default routing is `autonomous_decision(...)`.
- Escalation criteria — call `invoke_agent` ONLY if ANY true:
  1. `tier == 3` (F&G extreme / first-of-day / periodic 4h)
  2. Drawdown change > `escalate_drawdown_pct` since last decision
  3. Top-5 reliable signals split BUY vs SELL on triggered ticker (use `accuracy_stats.top_n_for_ticker`)
  4. Held position + SELL-side flip toward exit
  5. Post-trade trigger
- All others → `autonomous_decision`. Journal entry written. Telegram sent.
- Tests: extend `tests/test_autonomous.py` + new escalation-criteria cases in `tests/test_main_escalation.py`.

### Batch E — Ministral pre-gate (item 7)
New `portfolio/escalation_gate.py`.
- Function `should_escalate(reasons, tier, signals, prices, held_positions) → (bool, float, str)`:
  - Builds short structured prompt (trigger reasons + top-5 signal posture + held positions).
  - Calls `llm_batch.run_ministral(prompt, schema)` returning JSON `{escalate: bool, confidence: float, why: str}`.
  - Falls open (return `(True, 0.0, "ministral_unavailable")`) on any error — never silently swallow.
- Hook in `main.py` AFTER Batch D escalation criteria, as the final guard before `invoke_agent`. If `escalate=False` AND `confidence ≥ ministral_pregate_min_score` → route to autonomous instead. Log gate decision to `data/escalation_gate.jsonl`.
- Tests: mock Ministral output in `tests/test_escalation_gate.py`.

## Order of execution

Wave 1 (parallel): A, B, C — fully independent.
Wave 2: D — depends on A's gate config wired in.
Wave 3: E — sits between D's criteria and `invoke_agent`.

Each wave: implement → run targeted tests → commit. Final: integration test + Claude Code review subagent (NOT codex, per user direction).

## What could break

- **False negatives.** Stricter gates can swallow real triggers. Mitigation: T3 first-of-day path always on; log `skipped_*` reasons with full context in `invocations.jsonl` for weekly audit.
- **Autonomous-first wrong call.** `autonomous.py` writes HOLD-only today. Risk of holding through degradation. Mitigation: escalation criteria explicitly include held-position SELL-side flip and drawdown move.
- **Ministral classifier drift.** Could systematically under-escalate. Mitigation: log every gate decision to `data/escalation_gate.jsonl` + post-hoc weekly review.
- **Config file race.** Use existing `load_config()` (atomic). Already safe.
- **Worktree missing config.json symlink.** Targeted tests only inside worktree; full suite runs after merge to main.

## Verification

1. Targeted tests per batch:
   `.venv/Scripts/python.exe -m pytest tests/test_trigger.py tests/test_agent_invocation.py tests/test_autonomous.py tests/test_trigger_buffer.py tests/test_escalation_gate.py tests/test_main_escalation.py -v`
2. Full suite on main after merge: `.venv/Scripts/python.exe -m pytest tests/ -n auto` — must not regress (allow 26 known pre-existing failures per `docs/TESTING.md`).
3. **Claude Code review agent** (NOT codex). Subagent reads branch diff, reports issues, no fix authority.
4. Dry run: replay last 24h of `data/invocations.jsonl` through new gates (write `scripts/simulate_budget_gates.py`) — count what would have been skipped. Target ≥70% reduction.
5. Post-merge: monitor `data/invocations.jsonl` for 24h. Target ~6 invocations/day. Audit `skipped_*` entries weekly.

## Rollback

Each config key has a safe default that disables the new behavior. Flip switches off via `config.json` without redeploy. Code-level: revert merge commit on main.
