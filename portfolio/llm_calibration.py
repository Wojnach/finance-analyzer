"""Brier score and log-loss for LLM probability distributions.

Why this is a separate module
-----------------------------
`portfolio.accuracy_stats` is already ~40 KB of tight per-signal argmax
accuracy logic with thread-locks, TTL caches, and horizon fanout. Mixing
probabilistic calibration math into that file would bloat an already
critical-path module. This separate module keeps the calibration analysis
lazy-loadable — it's read by the daily `local_llm_report` and any ad-hoc
analyst script, never from the hot ticker loop.

What it provides
----------------
* `brier_score(probs, outcome)` — single-row multi-class Brier score.
* `log_loss(probs, outcome, eps=1e-12)` — single-row log-loss.
* `outcome_from_return(pct_change, buy_threshold, sell_threshold)` — map a
  realized return to the {BUY, HOLD, SELL} ground-truth class used by the
  probability log.
* `compute_metrics(log_path, outcomes_by_key, days=30)` — aggregate Brier
  and log-loss by signal. `outcomes_by_key` is a `{(ts, ticker): outcome}`
  lookup; the caller owns outcome backfill and passes in whatever source is
  available (snapshot-based lookup, outcome_tracker, or a mocked dict in
  tests). This decoupling avoids recomputing outcomes every call and keeps
  the math pure.

What it does NOT do (deliberately)
----------------------------------
* Does not backfill outcomes on the probability log. That's a scheduled
  job belonging to `portfolio.outcome_tracker` (follow-up).
* Does not cache results. Compute is cheap; cache at the caller if needed.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger("portfolio.llm_calibration")

_VALID_ACTIONS = frozenset({"BUY", "HOLD", "SELL"})
_BASE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_LOG = _BASE_DIR / "data" / "llm_probability_log.jsonl"


def outcome_from_return(
    pct_change: float,
    buy_threshold: float = 0.005,
    sell_threshold: float = -0.005,
) -> str:
    """Map a realized return (fractional, e.g. 0.012 = +1.2 %) to the
    ground-truth class {BUY, HOLD, SELL}. Thresholds are asymmetric-safe:
    `sell_threshold` must be negative, `buy_threshold` must be positive.

    Default dead-band is ±0.5 %, matching the existing argmax accuracy
    layer's tolerance for "flat" moves (see `accuracy_stats.signal_accuracy`
    which treats |pct| < 0.005 as HOLD-correct).
    """
    if pct_change >= buy_threshold:
        return "BUY"
    if pct_change <= sell_threshold:
        return "SELL"
    return "HOLD"


def brier_score(probs: dict[str, float], outcome: str) -> float:
    """Multi-class Brier score for a single prediction.

    `probs` must have the three keys {BUY, HOLD, SELL} summing to 1.
    `outcome` must be one of those three. Returns a float in [0, 2] where
    lower is better; 0 = perfect, 0.667 = uniform guess (1/3 each), 2 =
    maximally wrong.

    Formula: `Σ_c (P(c) - I[outcome = c])^2`.
    """
    if outcome not in _VALID_ACTIONS:
        raise ValueError(f"outcome must be in {_VALID_ACTIONS}, got {outcome!r}")
    total = 0.0
    for c in _VALID_ACTIONS:
        p = float(probs.get(c, 0.0))
        indicator = 1.0 if c == outcome else 0.0
        total += (p - indicator) ** 2
    return total


def log_loss(probs: dict[str, float], outcome: str, eps: float = 1e-12) -> float:
    """Log-loss (cross-entropy) for a single prediction.

    `-log(P(outcome))` clipped by `eps` to keep log-loss finite when a
    probability is reported as exactly 0 (which can happen in the confidence-
    split fallback with perfect confidence).
    """
    if outcome not in _VALID_ACTIONS:
        raise ValueError(f"outcome must be in {_VALID_ACTIONS}, got {outcome!r}")
    p = float(probs.get(outcome, 0.0))
    p = max(p, eps)
    p = min(p, 1.0)
    return -math.log(p)


def _read_probability_log(
    log_path: Path | str | None = None,
    days: int | None = 30,
):
    """Yield parsed rows from the probability log, optionally filtering by
    age. Invalid rows are skipped silently; the logger side already
    validates schema on write."""
    path = Path(log_path) if log_path else _DEFAULT_LOG
    if not path.exists():
        return
    cutoff = None
    if days is not None:
        cutoff = datetime.now(UTC) - timedelta(days=days)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if cutoff is not None:
            try:
                ts = datetime.fromisoformat(row.get("ts", ""))
            except (TypeError, ValueError):
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if ts < cutoff:
                continue
        yield row


def compute_metrics(
    outcomes_by_key,
    *,
    log_path: Path | str | None = None,
    days: int | None = 30,
) -> dict[str, dict]:
    """Aggregate Brier + log-loss by signal, over the probability log.

    `outcomes_by_key` is a callable `(ts: str, ticker: str, horizon: str) →
    Optional[str]` returning the ground-truth class or None if outcome is
    not yet available (too recent, no snapshot, etc.). Rows without an
    available outcome are skipped and tallied under `missing_outcome`.

    Returns:
      `{signal: {"samples": N, "brier_mean": float, "log_loss_mean": float,
                 "missing_outcome": K, "buckets": {pred_class: {"n": ...,
                                                                "hit_rate":
                                                                float}}}}`.

    The `buckets` slot holds a per-chosen-action empirical hit-rate table
    useful for the calibration histogram downstream — not the full
    reliability diagram, but enough to spot the "confidently wrong"
    signature without pulling `sklearn`.
    """
    brier_sum: dict[str, float] = defaultdict(float)
    log_loss_sum: dict[str, float] = defaultdict(float)
    samples: dict[str, int] = defaultdict(int)
    missing: dict[str, int] = defaultdict(int)
    bucket_hits: dict[tuple[str, str], int] = defaultdict(int)
    bucket_total: dict[tuple[str, str], int] = defaultdict(int)

    for row in _read_probability_log(log_path=log_path, days=days):
        sig = row.get("signal")
        probs = row.get("probs") or {}
        if not sig or not isinstance(probs, dict):
            continue
        if set(probs.keys()) != _VALID_ACTIONS:
            continue
        try:
            outcome = outcomes_by_key(
                row.get("ts", ""),
                row.get("ticker", ""),
                row.get("horizon", "1d"),
            )
        except Exception:
            outcome = None
        if outcome is None:
            missing[sig] += 1
            continue
        if outcome not in _VALID_ACTIONS:
            continue
        try:
            brier_sum[sig] += brier_score(probs, outcome)
            log_loss_sum[sig] += log_loss(probs, outcome)
        except ValueError:
            continue
        samples[sig] += 1
        chosen = row.get("chosen")
        if chosen in _VALID_ACTIONS:
            bucket_total[(sig, chosen)] += 1
            if chosen == outcome:
                bucket_hits[(sig, chosen)] += 1

    result: dict[str, dict] = {}
    for sig, n in samples.items():
        sig_buckets = {}
        for cls in _VALID_ACTIONS:
            total = bucket_total.get((sig, cls), 0)
            if total:
                sig_buckets[cls] = {
                    "n": total,
                    "hit_rate": bucket_hits.get((sig, cls), 0) / total,
                }
        result[sig] = {
            "samples": n,
            "brier_mean": brier_sum[sig] / n,
            "log_loss_mean": log_loss_sum[sig] / n,
            "missing_outcome": missing.get(sig, 0),
            "buckets": sig_buckets,
        }
    # Report signals that had rows but all missing outcome.
    for sig, m in missing.items():
        if sig not in result:
            result[sig] = {
                "samples": 0,
                "brier_mean": None,
                "log_loss_mean": None,
                "missing_outcome": m,
                "buckets": {},
            }
    return result
