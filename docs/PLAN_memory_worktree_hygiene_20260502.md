# Memory + Worktree Hygiene Audit — 2026-05-02

Worktree branch: `chore/memory-worktree-hygiene-20260502`
Worktree path: `/mnt/q/finance-analyzer-memory-hygiene`
Memory dir audited: `/root/.claude/projects/-mnt-q-finance-analyzer/memory/`
Scope: `project_*.md`, `feedback_*.md`, `reference_*.md` (NOT `user_*.md`)

## Summary

- 49 in-scope memory files reviewed
- 3 memories updated (mark resolved / refresh state)
- 2 memories already ahead of schedule (touched yesterday, no action)
- 0 memories deleted (all retain reference value)
- 44 memories kept as-is (still valid or codifying durable rules)
- 1 MEMORY.md index entry refreshed
- Worktree audit: 3 non-main worktrees, all unmerged, all in-flight — no cleanup

## Part A — Memory dispositions

### Updated (state changed since memory was last written)

#### 1. `project_metals_deferred_20260409.md` — UPDATED
- **Why stale**: 22 days old; 4 of 6 follow-ups are objectively done in code, but memory still lists them as "pending follow-up".
- **Verifications run**:
  - `scripts/fingpt_daemon.py` removed: `ls` returns "No such file or directory"
  - `scripts/win/pf-restart.bat` exists (used in `feedback_restart_loops.md`, current procedure)
  - `portfolio/avanza_session.py` — `get_buying_power` (#2 in memory) now handles BOTH the legacy `categorizedAccounts` AND new `categories` shape (lines 385-520), with all four ID fields tried (`accountId`, `id`, `accountNumber`, `number`). Bug C7 from memory is FIXED in `avanza_session.py`, not just `metals_avanza_helpers.py`.
  - `data/metals_swing_trader.py` — `SHORT_ENABLED = False`, `SHORT_CANARY_WARRANTS = frozenset()` still as documented. Item 3 still pending.
  - `data/metals_loop.py` — 7667 lines (was 6500 in memory). ARCH-18 noted in `docs/SYSTEM_OVERVIEW.md:287`. Still deferred. Item 4 still pending.
  - `portfolio/golddigger/data_provider.py` — DGS10 FRED fallback in place (item 1 resolved by parallel work).
  - `log()`/`print()` in metals_loop.py — item 5 still pending (would require dedicated session).
- **Action**: Mark items 1, 2, 6 resolved; keep 3, 4, 5 as the actual pending list.

#### 2. `project_chronos_vram_contention.md` — UPDATED
- **Why stale**: 21 days old; the "next session" sub-task at the bottom (BUG-178 dogpile fix in `accuracy_stats.py`) was almost certainly handled in subsequent accuracy-pipeline work (`project_accuracy_pipeline_20260428.md` shipped 47b4d474 which raised MIN_SAMPLES, added binomial SE gates, etc.). The original VRAM-contention investigation is complete — premise rejected — but the closing "Next session: wrap the cache reads in `shared_state._cached()`" note is now hanging unresolved in the memory.
- **Action**: Add resolution note pointing at the accuracy-pipeline merges; keep the body intact (the "premise rejected" diagnosis is still load-bearing reference for future sessions).

#### 3. `MEMORY.md` index — UPDATED
- Refresh project_metals_deferred entry to reflect that some items shipped.
- Refresh project_chronos_vram_contention entry to reflect the cross-reference.

### Already up-to-date (touched yesterday — 2026-05-01)

#### `project_known_bugs.md`
- 1 day old. CRITICAL-2 already marked RESOLVED 2026-04-17, end-to-end verified 2026-05-01.
- `tests/test_signal_engine.py:651` regression pin still valid.
- No action.

#### `project_accuracy_degradation_20260416.md`
- 1 day old. Already carries the 2026-05-01 misattribution correction at the top.
- Recommendations remain as historical context.
- No action.

### Kept as-is (still valid; codifies durable rules or paths)

These memories remain accurate and load-bearing. None require update.

**Trading rules / behavioral feedback (durable)**:
- `feedback_avanza_order_volume.md` — Avanza sell + stop-loss volume constraint still applies
- `feedback_be_decisive.md` — durable behavioral rule
- `feedback_check_buying_power.md` — durable budgeting rule
- `feedback_comment_for_future_sessions.md` — durable commenting standard
- `feedback_concise_responses.md` — durable response style
- `feedback_config_grep_not_read.md` — durable security rule (paired with `feedback_no_api_keys_in_claude_settings.md`)
- `feedback_fishing_20260331.md` — historical session record + lessons; preserved
- `feedback_fishing_direction.md` — durable trend-following discipline
- `feedback_fishing_lessons_20260330.md` — historical session record
- `feedback_fishing_patience.md` — durable limit-order discipline
- `feedback_live_price_every_query.md` — Binance FAPI endpoints still current
- `feedback_log_everything.md` — durable logging philosophy
- `feedback_min_order_size_1000_sek.md` — Avanza fee schedule unchanged
- `feedback_no_api_keys_in_claude_settings.md` — recent (1 day old)
- `feedback_no_panic_sell.md` — durable exit discipline
- `feedback_no_signal_inversion.md` — durable, codified at signal_engine.py:34-36
- `feedback_powershell_bash_quoting.md` — durable quoting rule
- `feedback_restart_loops.md` — `pf-restart.bat` confirmed exists
- `feedback_same_day_watch.md` — recent (3 days old)
- `feedback_signal_confidence_threshold.md` — durable ≥60% rule
- `feedback_stop_loss_spread.md` — durable MINI/cert wisdom
- `feedback_trading_rules.md` — account 1625505, durable
- `feedback_understand_before_proposing.md` — durable behavioral rule
- `feedback_wider_stop_losses.md` — durable leveraged-cert sizing rule

**Househunting (project memories — separate codebase, still active)**:
- `feedback_househunting_accuracy.md`
- `feedback_househunting_methodology.md`
- `feedback_househunting_priorities.md`
- `project_househunting.md`
- `project_househunting_lib.md`

**SHIPPED projects (kept as historical record + paths reference)**:
- `project_accuracy_pipeline_20260428.md` — 3 days old, shipped, key file paths still valid
- `project_contract_alert_architecture_20260428.md` — 3 days old, shipped
- `project_fingpt_llmbatch_session_20260409.md` — daemon retired (verified), shipped
- `project_fingpt_parser_defaulting_neutral.md` — shipped
- `project_fish_engine_live_test.md` — `data/fish_engine.py` still exists with `FISH_ENGINE_ENABLED = False`
- `project_llama_swap_reduction.md` — shipped, `portfolio/llama_server.py` and `portfolio/bert_sentiment.py` exist
- `project_loop_contract.md` — shipped, `portfolio/loop_contract.py` exists, 75 tests
- `project_macro_window_gating_20260428.md` — 3 days old, shipped, `is_macro_window` exists
- `project_metals_catalog_staleness.md` — shipped, `data/metals_warrant_refresh.py` exists, `MIN_BARRIER_DISTANCE_PCT=10`, `MIN_BUY_CONFIDENCE=0.60`, `REGIME_CONFIRM_CHECKS=2` all confirmed in `metals_swing_config.py`
- `project_quant_signal_improvements.md` — shipped, `portfolio/feature_normalizer.py` exists (still intentionally unwired)
- `project_research_improvements.md` — plan only, `docs/superpowers/plans/2026-04-01-research-improvements.md`
- `project_sentiment_models_cpu.md` — shipped
- `project_session_continuity.md` — durable
- `project_signal_research_agent.md` — deployed, PF-SignalResearch task running

**Planned-not-started (still valid as project specs)**:
- `project_adaptive_limit_fishing.md` — feature not built; spec valid
- `project_daily_gambit_scanner.md` — feature not built (verified `scripts/daily_gambit.py` does not exist); spec valid

**Reference docs (paths confirmed)**:
- `reference_avanza_fast_trading.md` — `portfolio/avanza_session.py` API unchanged
- `reference_avanza_new_package.md` — `portfolio/avanza/` package exists with all 10 modules + `scanner.py`
- `reference_avanza_trading_hours.md` — durable DST guidance
- `reference_hw_monitoring.md` — durable hardware reference (CPU has been further nerfed since, but memory acknowledges this in body)
- `reference_wsl_git_remote.md` — durable

## Part B — Worktree audit

`git worktree list` reports 4 worktrees (one was missed in the task brief):

| Path | Branch | Last commit | Merged into main? | Status |
|---|---|---|---|---|
| `/mnt/q/finance-analyzer` | `main` | `9cdaa23a` | n/a | active main |
| `/mnt/q/finance-analyzer/.worktrees/dashboard-ops-board` | `feat/dashboard-ops-board` | `4eeb2267` 2026-04-28 16:11 | NO (10 unmerged commits) | in-flight, leave alone |
| `/mnt/q/fishtrader-wt` | `feature/fishtrader-2026-05-01` | `f685c44b` 2026-05-01 02:29 | NO (1 unmerged commit, batch 0 scaffolding) | in-flight, leave alone |
| `/mnt/q/finance-analyzer.followups` | `feat/midfinance-followups-2026-05-02` | `a46cf092` 2026-05-02 00:53 | NO | in-flight, leave alone |

All three non-main worktrees:
- Have unmerged commits (`git branch --no-merged main` confirms all three appear in unmerged list)
- Belong to other in-flight sessions (different agents, different scopes)
- Are not stale by date (newest is today, oldest 2026-04-28)

**Disposition**: NO cleanup. Per task spec: "DO NOT remove other people's in-flight worktrees."

The `/mnt/q/finance-analyzer.followups` worktree (4th one) was not in the task brief but appears in the list — it is also in-flight (committed today) and out of scope.

## Files to commit in this worktree

Only this finding doc (`docs/PLAN_memory_worktree_hygiene_20260502.md`).

Memory updates land in `/root/.claude/projects/...` which is OUTSIDE the repo and not version-controlled by this worktree.
