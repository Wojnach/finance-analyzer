"""Launch 8 codex *exec* adversarial reviews in parallel.

`codex review --base` returned empty diffs because the baseline branches
share the same commit as HEAD as their merge-base. Switch to `codex exec`
with the file list embedded in the prompt and `--output-last-message` to
capture only the final report.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WORKTREE = Path(r"Q:/fa-adv-2026-05-13")
DOCS = WORKTREE / "docs/adversarial-review-2026-05-13"
OUT_DIR = DOCS / "_codex_out"
PROMPTS = DOCS / "_prompts"
FINAL_DIR = DOCS  # codex final markdown lands directly in the docs dir

SUBSYSTEMS: list[tuple[int, str, list[str]]] = [
    (1, "signals-core", [
        "portfolio/signal_engine.py", "portfolio/signal_registry.py",
        "portfolio/signal_utils.py", "portfolio/signal_weights.py",
        "portfolio/signal_weight_optimizer.py", "portfolio/signal_history.py",
        "portfolio/signal_state_since.py", "portfolio/signal_decay_alert.py",
        "portfolio/signal_postmortem.py", "portfolio/signal_db.py",
        "portfolio/accuracy_stats.py", "portfolio/accuracy_degradation.py",
        "portfolio/ticker_accuracy.py", "portfolio/outcome_tracker.py",
        "portfolio/forecast_accuracy.py", "portfolio/ic_computation.py",
        "portfolio/train_signal_weights.py", "portfolio/linear_factor.py",
        "portfolio/feature_normalizer.py", "portfolio/short_horizon.py",
    ]),
    (2, "orchestration", [
        "portfolio/main.py", "portfolio/agent_invocation.py",
        "portfolio/autonomous.py", "portfolio/trigger.py",
        "portfolio/market_timing.py", "portfolio/claude_gate.py",
        "portfolio/gpu_gate.py", "portfolio/health.py",
        "portfolio/alert_budget.py", "portfolio/llm_prewarmer.py",
        "portfolio/llm_calibration.py", "portfolio/llm_batch.py",
        "portfolio/llm_outcome_backfill.py", "portfolio/llm_probability_log.py",
        "portfolio/llama_server.py", "portfolio/multi_agent_layer2.py",
        "portfolio/perception_gate.py", "portfolio/focus_analysis.py",
        "portfolio/reporting.py", "portfolio/journal.py",
        "portfolio/journal_index.py", "portfolio/telegram_notifications.py",
        "portfolio/telegram_poller.py", "portfolio/digest.py",
        "portfolio/daily_digest.py", "portfolio/weekly_digest.py",
        "portfolio/reflection.py", "portfolio/regime_alerts.py",
        "portfolio/analyze.py", "portfolio/bigbet.py",
        "portfolio/prophecy.py", "portfolio/qwen3_signal.py",
        "portfolio/circuit_breaker.py", "portfolio/cumulative_tracker.py",
        "portfolio/decision_outcome_tracker.py", "portfolio/loop_contract.py",
    ]),
    (3, "portfolio-risk", [
        "portfolio/portfolio_mgr.py", "portfolio/portfolio_validator.py",
        "portfolio/trade_guards.py", "portfolio/trade_validation.py",
        "portfolio/trade_risk_classifier.py", "portfolio/risk_management.py",
        "portfolio/monte_carlo.py", "portfolio/monte_carlo_risk.py",
        "portfolio/equity_curve.py", "portfolio/exit_optimizer.py",
        "portfolio/kelly_sizing.py", "portfolio/kelly_metals.py",
        "portfolio/exposure_coach.py", "portfolio/warrant_portfolio.py",
        "portfolio/cost_model.py", "portfolio/instrument_profile.py",
        "portfolio/stats.py", "portfolio/iskbets.py",
        "portfolio/strategies/",
    ]),
    (4, "metals-core", [
        "data/metals_loop.py", "data/crypto_loop.py", "data/oil_loop.py",
        "portfolio/grid_fisher.py", "portfolio/grid_fisher_config.py",
        "portfolio/grid_tiers.py", "portfolio/oil_grid_signal.py",
        "portfolio/fin_fish.py", "portfolio/fin_snipe.py",
        "portfolio/fin_snipe_manager.py", "portfolio/fish_instrument_finder.py",
        "portfolio/fish_monitor_smart.py", "portfolio/gold_precompute.py",
        "portfolio/silver_precompute.py", "portfolio/oil_precompute.py",
        "portfolio/crypto_precompute.py", "portfolio/price_targets.py",
        "portfolio/orb_predictor.py", "portfolio/orb_backtest.py",
        "portfolio/orb_postmortem.py", "portfolio/mstr_loop/",
        "portfolio/elongir/", "portfolio/golddigger/",
    ]),
    (5, "avanza-api", [
        "portfolio/avanza_session.py", "portfolio/avanza_client.py",
        "portfolio/avanza_orders.py", "portfolio/avanza_tracker.py",
        "portfolio/avanza_control.py", "portfolio/avanza_account_check.py",
        "portfolio/avanza_order_lock.py", "portfolio/avanza_resilient_page.py",
        "portfolio/avanza/",
    ]),
    (6, "signals-modules", [
        "portfolio/signals/",
    ]),
    (7, "data-external", [
        "portfolio/data_collector.py", "portfolio/fear_greed.py",
        "portfolio/sentiment.py", "portfolio/bert_sentiment.py",
        "portfolio/alpha_vantage.py", "portfolio/futures_data.py",
        "portfolio/funding_rate.py", "portfolio/fx_rates.py",
        "portfolio/onchain_data.py", "portfolio/news_keywords.py",
        "portfolio/social_sentiment.py", "portfolio/crypto_macro_data.py",
        "portfolio/crypto_scheduler.py", "portfolio/earnings_calendar.py",
        "portfolio/econ_dates.py", "portfolio/fomc_dates.py",
        "portfolio/seasonality.py", "portfolio/seasonality_updater.py",
        "portfolio/session_calendar.py", "portfolio/price_source.py",
        "portfolio/http_retry.py", "portfolio/api_utils.py",
        "portfolio/data_refresh.py", "portfolio/forecast_signal.py",
        "portfolio/indicators.py", "portfolio/metals_orderbook.py",
        "portfolio/microstructure.py", "portfolio/microstructure_state.py",
        "portfolio/metals_cross_assets.py", "portfolio/tickers.py",
    ]),
    (8, "infrastructure", [
        "portfolio/file_utils.py", "portfolio/shared_state.py",
        "portfolio/process_lock.py", "portfolio/subprocess_utils.py",
        "portfolio/config_validator.py", "portfolio/notification_text.py",
        "portfolio/message_store.py", "portfolio/shadow_registry.py",
        "portfolio/vector_memory.py", "portfolio/backtester.py",
        "dashboard/", "scripts/check_critical_errors.py",
        "scripts/fix_agent_dispatcher.py",
    ]),
]


TEMPLATE = """You are doing an ADVERSARIAL code review of the **{NAME}** subsystem of finance-analyzer (a quantitative trading system at Q:\\fa-adv-2026-05-13).

Sandbox: read-only. Do NOT modify any files. Read each file fully.

**Files in scope** (review these and ONLY these):
{FILE_LIST}

**Project rules from CLAUDE.md**:
- Atomic I/O only via `file_utils.atomic_write_json` / `load_json` / `atomic_append_jsonl`. Raw `json.loads(open(...).read())` is a defect.
- MIN_VOTERS = 3 (consensus = active BUY+SELL voters, not total).
- Accuracy gate: signals <47% accuracy with 30+ samples are force-HOLD (NOT inverted — inversion causes whiplash).
- Recency-weighted: 70% recent (7d) + 30% all-time.
- Regime penalties: ranging 0.75x, high-vol 0.80x confidence multipliers.
- NEVER commit config.json (symlink to external API keys file).
- Stop-loss API: use `/_api/trading/stoploss/new`, NOT regular order API (Mar-3 incident).
- Min order size: 1000 SEK (Avanza floor).
- User accepts 10-20% knockout risk on 5x certs; de-risk only at 50%+.
- Never place stop-loss within 3% of current bid; never near MINI warrant barriers.
- Layer 2 invocation must go through `claude_gate` (auth-scan, tree-kill, CLAUDECODE cleanup).
- CLAUDECODE env var must be cleared before invoking `claude -p` (caused 34h outage Feb 18-19).

**Adversarial bug classes — hunt for ALL of these:**
1. Silent failures: try/except that swallow errors or return defaults without logging/alerting.
2. Race conditions: shared state across threads/processes, JSONL append/rotate races, missing locks, mutable defaults, single sqlite connection shared.
3. Float / numeric: division by zero, NaN propagation, off-by-one, integer-vs-float comparison, log/sqrt of 0/negative, pct_change on length-1.
4. Stale data / cache: TTL not honored, mixed-tz time math, "now" inconsistent across modules.
5. Accuracy/weighting math: sample-size guards missing, biased estimators, mixing horizons, look-ahead bias, survivorship bias.
6. Signal-inversion or weight rule violations (no inverting <50% — force HOLD instead).
7. Atomic I/O violations (raw open() on canonical state files).
8. Resource leaks: open files / sqlite connections / playwright pages not closed.
9. Security: SQL injection, unsafe deserialization, eval/exec, secrets in logs, dashboard auth bypass.
10. Dead code / unused parameters masking intent.
11. Project-rule drift: hardcoded constants contradicting CLAUDE.md.
12. Stop-loss / barrier-proximity bugs (Mar-3 class).
13. Subprocess management: tree-kill failure, orphan grandchildren, stdin not closed (deadlock), env var clobber.
14. Telegram message delivery: dedup race, markdown 400 lost, formatting injection.

**Output format — STRICT MARKDOWN. This is your final deliverable; everything else is scratch:**

# Codex adversarial review: {NAME}

## Summary
(2-3 sentence overall verdict)

## P0 — Blockers (production breakage / data loss / silent wrong trades)
- `path:line` — short description. Why it bites: ... Fix: ...

## P1 — High (will cause incidents)
- `path:line` — ...

## P2 — Medium (correctness / robustness)
- `path:line` — ...

## P3 — Low (style / dead code / minor)
- `path:line` — ...

## Tests missing
- `path:line` — what scenario isn't covered.

## Cross-cutting observations
- patterns spanning multiple files

**Rules:**
- Cite exact file paths and line numbers — no vague claims.
- No false-positive padding. If uncertain, label as "(uncertain)".
- Read full files; don't pad with grep snippets.
- If a subsystem area is clean, say so explicitly — don't invent findings.
- Quality > quantity. ~20-40 real findings is typical.

Spend your full token budget reading and analysing. Emit the final markdown report as your final agent message — do NOT print it earlier as a tool call.
"""


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pids: dict[str, int] = {}
    codex_exe = r"C:\Users\Herc2\AppData\Roaming\npm\codex.cmd"

    for n, name, files in SUBSYSTEMS:
        prompt = TEMPLATE.replace("{NAME}", name).replace(
            "{FILE_LIST}", "\n".join(f"- {f}" for f in files)
        )
        prompt_path = PROMPTS / f"{n}-{name}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        out_path = OUT_DIR / f"{n}-{name}.stdout"
        err_path = OUT_DIR / f"{n}-{name}.stderr"
        final_path = FINAL_DIR / f"{n}-{name}-codex.md"

        cmd = [
            codex_exe, "exec",
            "--sandbox", "read-only",
            "--cd", str(WORKTREE),
            "--skip-git-repo-check",
            "--output-last-message", str(final_path),
            "-",  # read prompt from stdin
        ]

        with open(out_path, "wb") as fout, open(err_path, "wb") as ferr:
            proc = subprocess.Popen(
                cmd,
                cwd=str(WORKTREE),
                stdout=fout,
                stderr=ferr,
                stdin=subprocess.PIPE,
                shell=False,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            proc.stdin.write(prompt.encode("utf-8"))
            proc.stdin.close()
        pids[name] = proc.pid
        print(f"launched {name} pid={proc.pid} -> {final_path.name}")

    state_path = OUT_DIR / "_pids.txt"
    state_path.write_text(
        "\n".join(f"{name}={pid}" for name, pid in pids.items()) + "\n",
        encoding="utf-8",
    )
    print(f"\npids saved -> {state_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
