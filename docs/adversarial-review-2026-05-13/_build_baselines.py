"""Build 8 orphan empty-baseline branches for adversarial review 2026-05-13.

For each subsystem the branch is a single root commit = snapshot of main@HEAD
with every file in the subsystem REMOVED. `codex review --base <branch>` from
the current worktree then sees the subsystem files as a fresh "added" diff.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

WORKTREE = Path(r"Q:/fa-adv-2026-05-13")
BASE_BRANCH = "adv-review-2026-05-13"
BRANCH_PREFIX = "adv-2026-05-13/baseline-"

# (number, name, paths) — paths can be files or directories.
SUBSYSTEMS: list[tuple[int, str, list[str]]] = [
    (1, "signals-core", [
        "portfolio/signal_engine.py",
        "portfolio/signal_registry.py",
        "portfolio/signal_utils.py",
        "portfolio/signal_weights.py",
        "portfolio/signal_weight_optimizer.py",
        "portfolio/signal_history.py",
        "portfolio/signal_state_since.py",
        "portfolio/signal_decay_alert.py",
        "portfolio/signal_postmortem.py",
        "portfolio/signal_db.py",
        "portfolio/accuracy_stats.py",
        "portfolio/accuracy_degradation.py",
        "portfolio/ticker_accuracy.py",
        "portfolio/outcome_tracker.py",
        "portfolio/forecast_accuracy.py",
        "portfolio/ic_computation.py",
        "portfolio/train_signal_weights.py",
        "portfolio/linear_factor.py",
        "portfolio/feature_normalizer.py",
        "portfolio/short_horizon.py",
    ]),
    (2, "orchestration", [
        "portfolio/main.py",
        "portfolio/agent_invocation.py",
        "portfolio/autonomous.py",
        "portfolio/trigger.py",
        "portfolio/market_timing.py",
        "portfolio/claude_gate.py",
        "portfolio/gpu_gate.py",
        "portfolio/health.py",
        "portfolio/alert_budget.py",
        "portfolio/llm_prewarmer.py",
        "portfolio/llm_calibration.py",
        "portfolio/llm_batch.py",
        "portfolio/llm_outcome_backfill.py",
        "portfolio/llm_probability_log.py",
        "portfolio/llama_server.py",
        "portfolio/multi_agent_layer2.py",
        "portfolio/perception_gate.py",
        "portfolio/focus_analysis.py",
        "portfolio/reporting.py",
        "portfolio/journal.py",
        "portfolio/journal_index.py",
        "portfolio/telegram_notifications.py",
        "portfolio/telegram_poller.py",
        "portfolio/digest.py",
        "portfolio/daily_digest.py",
        "portfolio/weekly_digest.py",
        "portfolio/reflection.py",
        "portfolio/regime_alerts.py",
        "portfolio/analyze.py",
        "portfolio/bigbet.py",
        "portfolio/prophecy.py",
        "portfolio/qwen3_signal.py",
        "portfolio/circuit_breaker.py",
        "portfolio/cumulative_tracker.py",
        "portfolio/decision_outcome_tracker.py",
        "portfolio/loop_contract.py",
    ]),
    (3, "portfolio-risk", [
        "portfolio/portfolio_mgr.py",
        "portfolio/portfolio_validator.py",
        "portfolio/trade_guards.py",
        "portfolio/trade_validation.py",
        "portfolio/trade_risk_classifier.py",
        "portfolio/risk_management.py",
        "portfolio/monte_carlo.py",
        "portfolio/monte_carlo_risk.py",
        "portfolio/equity_curve.py",
        "portfolio/exit_optimizer.py",
        "portfolio/kelly_sizing.py",
        "portfolio/kelly_metals.py",
        "portfolio/exposure_coach.py",
        "portfolio/warrant_portfolio.py",
        "portfolio/cost_model.py",
        "portfolio/instrument_profile.py",
        "portfolio/stats.py",
        "portfolio/iskbets.py",
        "portfolio/strategies",
    ]),
    (4, "metals-core", [
        "data/metals_loop.py",
        "data/crypto_loop.py",
        "data/oil_loop.py",
        "portfolio/grid_fisher.py",
        "portfolio/grid_fisher_config.py",
        "portfolio/grid_tiers.py",
        "portfolio/oil_grid_signal.py",
        "portfolio/fin_fish.py",
        "portfolio/fin_fish_manager.py",
        "portfolio/fin_snipe.py",
        "portfolio/fin_snipe_manager.py",
        "portfolio/fish_instrument_finder.py",
        "portfolio/fish_monitor_smart.py",
        "portfolio/gold_precompute.py",
        "portfolio/silver_precompute.py",
        "portfolio/oil_precompute.py",
        "portfolio/crypto_precompute.py",
        "portfolio/price_targets.py",
        "portfolio/orb_predictor.py",
        "portfolio/orb_backtest.py",
        "portfolio/orb_postmortem.py",
        "portfolio/mstr_loop",
        "portfolio/elongir",
        "portfolio/golddigger",
    ]),
    (5, "avanza-api", [
        "portfolio/avanza_session.py",
        "portfolio/avanza_client.py",
        "portfolio/avanza_orders.py",
        "portfolio/avanza_tracker.py",
        "portfolio/avanza_control.py",
        "portfolio/avanza_account_check.py",
        "portfolio/avanza_order_lock.py",
        "portfolio/avanza_resilient_page.py",
        "portfolio/avanza",
    ]),
    (6, "signals-modules", [
        "portfolio/signals",
    ]),
    (7, "data-external", [
        "portfolio/data_collector.py",
        "portfolio/fear_greed.py",
        "portfolio/sentiment.py",
        "portfolio/bert_sentiment.py",
        "portfolio/alpha_vantage.py",
        "portfolio/futures_data.py",
        "portfolio/funding_rate.py",
        "portfolio/fx_rates.py",
        "portfolio/onchain_data.py",
        "portfolio/news_keywords.py",
        "portfolio/social_sentiment.py",
        "portfolio/crypto_macro_data.py",
        "portfolio/crypto_scheduler.py",
        "portfolio/earnings_calendar.py",
        "portfolio/econ_dates.py",
        "portfolio/fomc_dates.py",
        "portfolio/seasonality.py",
        "portfolio/seasonality_updater.py",
        "portfolio/session_calendar.py",
        "portfolio/price_source.py",
        "portfolio/http_retry.py",
        "portfolio/api_utils.py",
        "portfolio/data_refresh.py",
        "portfolio/forecast_signal.py",
        "portfolio/indicators.py",
        "portfolio/metals_orderbook.py",
        "portfolio/microstructure.py",
        "portfolio/microstructure_state.py",
        "portfolio/metals_cross_assets.py",
        "portfolio/tickers.py",
    ]),
    (8, "infrastructure", [
        "portfolio/file_utils.py",
        "portfolio/shared_state.py",
        "portfolio/process_lock.py",
        "portfolio/subprocess_utils.py",
        "portfolio/config_validator.py",
        "portfolio/notification_text.py",
        "portfolio/message_store.py",
        "portfolio/shadow_registry.py",
        "portfolio/vector_memory.py",
        "portfolio/backtester.py",
        "dashboard",
        "scripts/check_critical_errors.py",
        "scripts/fix_agent_dispatcher.py",
        "conftest.py",
    ]),
]


def run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=WORKTREE, capture_output=True, text=True, **kw)


def main() -> int:
    # Confirm we're on the worktree branch.
    cp = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    print(f"current branch: {cp.stdout.strip()}")
    base_sha = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    print(f"baseline SHA: {base_sha}")

    for n, name, paths in SUBSYSTEMS:
        branch = f"{BRANCH_PREFIX}{name}"
        print(f"\n=== subsystem {n}: {name} -> {branch} ===")

        # Clean up if branch exists.
        run(["git", "branch", "-D", branch])

        # Create branch at HEAD (same content as adv-review-2026-05-13).
        # We do NOT use --orphan because we want codex to see the subsystem files
        # as added vs the baseline. The baseline = main HEAD with subsystem files
        # removed. So branch-from-HEAD, remove subsystem, commit, then `codex review
        # --base <branch>` will diff back the subsystem.
        cp = run(["git", "checkout", "-b", branch, base_sha])
        if cp.returncode != 0:
            print(f"  ERROR creating branch:\n{cp.stderr}")
            return 1

        # Remove subsystem files/dirs.
        existing = []
        for p in paths:
            full = WORKTREE / p
            if full.exists():
                existing.append(p)
            else:
                print(f"  WARN path does not exist: {p}")
        if not existing:
            print(f"  ERROR no existing paths to remove for {name}")
            return 1
        cp = run(["git", "rm", "-rf"] + existing)
        if cp.returncode != 0:
            print(f"  ERROR git rm:\n{cp.stderr[:500]}")
            return 1

        # Commit.
        cp = run(["git", "commit", "-m", f"baseline-{n}-{name}: remove subsystem files for adversarial review"])
        if cp.returncode != 0:
            print(f"  ERROR commit:\n{cp.stderr}")
            return 1
        head = run(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()
        print(f"  baseline branch HEAD: {head}, {len(existing)} paths removed")

        # Return to adv-review branch.
        cp = run(["git", "checkout", BASE_BRANCH])
        if cp.returncode != 0:
            print(f"  ERROR checkout back:\n{cp.stderr}")
            return 1

    print("\nAll 8 baseline branches created.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
