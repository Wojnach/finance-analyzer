"""Pickup handler: verify cryptotrader_lm v2 LoRA after 72h shadow data.

Replays the manual verification logic from
`docs/IMPROVEMENT_BACKLOG.md#LLM-CRYPTOTRADER-72H`:

1. Load `data/llm_probability_log.jsonl` + `data/llm_probability_outcomes.jsonl`.
2. Filter for `signal == "cryptotrader_lm"` rows produced since the merge
   (default 2026-05-18T20:00 CET, overridable in the pickup's
   `context.merged_at`).
3. Use the shared `portfolio.llm_probability_log.is_directional_prediction`
   filter so the count matches the auto-promotion gate (and the dashboard).
4. Join to outcomes, compute accuracy.
5. Apply the decision tree from the backlog item:
     * n_directional >= min AND accuracy >= promote-bar -> verdict="promote"
     * n_directional >= min AND accuracy <= retire-bar  -> verdict="retire"
     * n_directional < min                              -> verdict="defer"

Verdict is a recommendation only. The handler does NOT mutate
`data/shadow_registry.json` -- promotion/retirement still needs a human
sign-off run of `scripts/review_shadow_signals.py --promote` because the
voter actively trades on consensus. Telegram alert + SESSION_PROGRESS
entry surface the recommendation for next-session pickup.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
import traceback
from pathlib import Path


def _parse_iso(raw: str | None) -> _dt.datetime | None:
    if not raw:
        return None
    try:
        dt = _dt.datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.UTC)
        return dt
    except (TypeError, ValueError):
        return None


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def run(pickup: dict, repo_root: Path) -> dict:
    """Compute cryptotrader_lm verdict; never raises."""
    try:
        sys.path.insert(0, str(repo_root))
        from portfolio.llm_probability_log import is_directional_prediction

        ctx = pickup.get("context") or {}
        thresholds = ctx.get("decision_thresholds") or {}
        min_dir = int(thresholds.get("min_directional", 50))
        promote_bar = float(thresholds.get("min_accuracy_for_promote", 0.60))
        retire_bar = float(thresholds.get("max_accuracy_for_retire", 0.55))

        cutoff = _parse_iso(ctx.get("merged_at"))
        if cutoff is None:
            cutoff = _dt.datetime(2026, 5, 18, 18, 0, tzinfo=_dt.UTC)

        log_path = repo_root / "data" / "llm_probability_log.jsonl"
        out_path = repo_root / "data" / "llm_probability_outcomes.jsonl"

        outcomes: dict = {}
        for row in _iter_jsonl(out_path):
            if not isinstance(row, dict) or row.get("signal") != "cryptotrader_lm":
                continue
            key = (row.get("ts"), row.get("ticker"), row.get("horizon"))
            outcomes[key] = row.get("outcome")

        stats = {
            "n_total": 0,
            "n_directional": 0,
            "n_matched": 0,
            "correct": 0,
            "buy_predictions": 0,
            "sell_predictions": 0,
        }
        per_ticker: dict = {}
        for row in _iter_jsonl(log_path):
            if not isinstance(row, dict) or row.get("signal") != "cryptotrader_lm":
                continue
            ts = _parse_iso(row.get("ts"))
            if ts is None or ts < cutoff:
                continue
            stats["n_total"] += 1
            if not is_directional_prediction(row):
                continue
            stats["n_directional"] += 1
            chosen = row.get("chosen")
            if chosen == "BUY":
                stats["buy_predictions"] += 1
            elif chosen == "SELL":
                stats["sell_predictions"] += 1
            ticker = row.get("ticker")
            t = per_ticker.setdefault(
                ticker, {"n_directional": 0, "n_matched": 0, "correct": 0},
            )
            t["n_directional"] += 1
            key = (row.get("ts"), ticker, row.get("horizon"))
            actual = outcomes.get(key)
            if actual is None:
                continue
            stats["n_matched"] += 1
            t["n_matched"] += 1
            if chosen == actual:
                stats["correct"] += 1
                t["correct"] += 1

        accuracy = (
            stats["correct"] / stats["n_matched"]
            if stats["n_matched"]
            else None
        )

        if stats["n_directional"] < min_dir:
            verdict = "defer"
            summary = (
                f"Insufficient directional rows: {stats['n_directional']} < "
                f"{min_dir}. Extend window 7d and re-check."
            )
        elif accuracy is None:
            verdict = "defer"
            summary = (
                f"{stats['n_directional']} directional rows but zero matched "
                "outcomes -- outcome backfill not caught up. Defer."
            )
        elif accuracy >= promote_bar:
            verdict = "promote"
            summary = (
                f"Accuracy {accuracy:.3f} on {stats['n_matched']} matched "
                f"rows >= {promote_bar}. Recommend promotion. Manual "
                "confirmation required via scripts/review_shadow_signals.py "
                "--promote."
            )
        elif accuracy <= retire_bar:
            verdict = "retire"
            summary = (
                f"Accuracy {accuracy:.3f} on {stats['n_matched']} matched "
                f"rows <= {retire_bar}. Recommend retirement. Update "
                "data/shadow_registry.json status=retired and consider "
                "removing the GGUF from llama_server rotation."
            )
        else:
            verdict = "defer"
            summary = (
                f"Accuracy {accuracy:.3f} on {stats['n_matched']} matched "
                f"rows -- in the gap ({retire_bar} < acc < {promote_bar}). "
                "Extend window 7d and re-check."
            )

        telegram_lines = [
            f"LLM-CRYPTOTRADER-72H verdict: {verdict.upper()}",
            f"  directional: {stats['n_directional']} (BUY={stats['buy_predictions']}, SELL={stats['sell_predictions']})",
            f"  matched: {stats['n_matched']}",
            (
                f"  accuracy: {accuracy:.3f}"
                if accuracy is not None
                else "  accuracy: n/a"
            ),
            "  " + summary,
        ]

        return {
            "verdict": verdict,
            "summary": summary,
            "details": {
                "accuracy": accuracy,
                "stats": stats,
                "per_ticker": per_ticker,
                "cutoff_iso": cutoff.isoformat(),
                "thresholds": {
                    "min_directional": min_dir,
                    "promote_bar": promote_bar,
                    "retire_bar": retire_bar,
                },
            },
            "telegram_lines": telegram_lines,
        }
    except Exception as e:
        return {
            "verdict": "error",
            "summary": f"Handler crashed: {e!r}",
            "details": {"traceback": traceback.format_exc()},
            "telegram_lines": [
                "LLM-CRYPTOTRADER-72H pickup handler crashed.",
                f"  error: {e!r}",
            ],
        }
