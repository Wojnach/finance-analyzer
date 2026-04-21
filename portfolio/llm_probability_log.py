"""Per-vote probability logger for LLM-family signals.

Why this exists
---------------
The existing accuracy layer (`portfolio/accuracy_stats.py`) records argmax
verdicts only: when Ministral votes "BUY" and the next day's close is higher,
that is stored as "correct". We lose the model's full predicted distribution
(e.g. `{BUY: 0.62, HOLD: 0.28, SELL: 0.10}`), which means we cannot tell
whether a wrong vote was made confidently or uncertainly. That is precisely
the distinction needed to decide whether a shadow-mode LLM is ready for
promotion — accuracy alone does not expose calibration.

This module provides an append-only JSONL log of per-call probability
distributions keyed by `(signal, ticker, horizon)`. Backfilled outcomes
are added by `portfolio.forecast_accuracy` / `portfolio.outcome_tracker` on
their regular cadence, which enables Brier-score and log-loss computation
downstream.

Schema of one row
-----------------
```json
{
  "ts": "2026-04-21T13:40:00+00:00",
  "signal": "ministral",
  "ticker": "BTC-USD",
  "horizon": "1d",
  "probs": {"BUY": 0.12, "HOLD": 0.55, "SELL": 0.33},
  "chosen": "HOLD",
  "confidence": 0.55,
  "tier": null
}
```

`tier` is populated for cascade signals like `claude_fundamental`
(values `haiku` / `sonnet` / `opus`), null for single-model signals.
`probs` must sum to 1.0 ± 1e-6; the logger enforces this.
"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path

from portfolio.file_utils import atomic_append_jsonl

logger = logging.getLogger("portfolio.llm_probability_log")

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_PROB_LOG = _DATA_DIR / "llm_probability_log.jsonl"

# Set of signals that emit probability distributions worth logging. Signals
# outside this set (e.g. pure technicals like rsi/macd) are silently no-op'd
# by `log_vote` so callers can call unconditionally.
_LLM_SIGNALS = frozenset(
    {
        "ministral",
        "qwen3",
        "sentiment",
        "news_event",
        "forecast",
        "claude_fundamental",
    }
)

_VALID_ACTIONS = frozenset({"BUY", "HOLD", "SELL"})


def log_vote(
    signal: str,
    ticker: str,
    probs: dict[str, float],
    *,
    horizon: str = "1d",
    chosen: str | None = None,
    confidence: float | None = None,
    tier: str | None = None,
    log_path: Path | None = None,
) -> bool:
    """Append one probability distribution entry to the log.

    Returns True if written, False if the row was rejected (not an LLM signal,
    bad probability distribution, etc.). Never raises — the logger is fire-
    and-forget so a malformed vote never aborts a signal cycle.

    Callers should invoke this unconditionally after an LLM signal produces
    a vote; non-LLM signal names are silently no-op'd via `_LLM_SIGNALS`.

    `probs` must:
      - have all three keys {BUY, HOLD, SELL}
      - be numeric, non-negative, sum to 1.0 ± 1e-6
      - `chosen` (if provided) must be in {BUY, HOLD, SELL}
      - `confidence` (if provided) must be in [0.0, 1.0]

    Any violation → False return and a debug log line, not an exception.
    """
    if signal not in _LLM_SIGNALS:
        return False

    if not isinstance(probs, dict):
        logger.debug("log_vote rejected: probs is not a dict for %s/%s", signal, ticker)
        return False
    if set(probs.keys()) != _VALID_ACTIONS:
        logger.debug(
            "log_vote rejected: probs keys %s != %s for %s/%s",
            sorted(probs.keys()), sorted(_VALID_ACTIONS), signal, ticker,
        )
        return False

    try:
        buy_p = float(probs["BUY"])
        hold_p = float(probs["HOLD"])
        sell_p = float(probs["SELL"])
    except (TypeError, ValueError):
        logger.debug("log_vote rejected: non-numeric probs for %s/%s", signal, ticker)
        return False

    if buy_p < 0 or hold_p < 0 or sell_p < 0:
        logger.debug("log_vote rejected: negative prob for %s/%s", signal, ticker)
        return False

    total = buy_p + hold_p + sell_p
    if abs(total - 1.0) > 1e-6:
        logger.debug(
            "log_vote rejected: probs sum %.6f not 1.0 for %s/%s", total, signal, ticker,
        )
        return False

    if chosen is not None and chosen not in _VALID_ACTIONS:
        logger.debug("log_vote rejected: bad chosen=%s for %s/%s", chosen, signal, ticker)
        return False

    if confidence is not None:
        try:
            conf_f = float(confidence)
        except (TypeError, ValueError):
            logger.debug("log_vote rejected: non-numeric confidence for %s/%s", signal, ticker)
            return False
        if conf_f < 0.0 or conf_f > 1.0:
            logger.debug("log_vote rejected: confidence %s out of range for %s/%s", conf_f, signal, ticker)
            return False
    else:
        conf_f = None

    entry = {
        "ts": _dt.datetime.now(_dt.UTC).isoformat(),
        "signal": signal,
        "ticker": ticker,
        "horizon": horizon,
        "probs": {"BUY": buy_p, "HOLD": hold_p, "SELL": sell_p},
        "chosen": chosen if chosen is not None else max(
            _VALID_ACTIONS, key=lambda k: probs[k],
        ),
        "confidence": conf_f,
        "tier": tier,
    }

    try:
        atomic_append_jsonl(log_path or _PROB_LOG, entry)
        return True
    except Exception as e:
        logger.debug("log_vote write failed for %s/%s: %s", signal, ticker, e)
        return False


def is_llm_signal(name: str) -> bool:
    """Public helper: check whether a signal name belongs to the LLM set this
    module tracks. Useful for gating integration points without having to
    re-import the private frozenset."""
    return name in _LLM_SIGNALS


def llm_signals() -> frozenset[str]:
    """Public snapshot of the LLM signal set. Returned frozenset is immutable
    so callers cannot mutate state."""
    return _LLM_SIGNALS


def _clamp_confidence(conf: float) -> float:
    """Clamp confidence to [0, 1]."""
    if conf < 0.0:
        return 0.0
    if conf > 1.0:
        return 1.0
    return conf


def derive_probs_from_result(
    signal_name: str,
    action: str,
    confidence: float,
    indicators: dict | None = None,
) -> dict[str, float] | None:
    """Derive a {BUY, HOLD, SELL} probability distribution from a signal's
    native output shape.

    Preference order:

    1. **Rich shape** — if `indicators["avg_scores"]` exists with
       positive/negative/neutral keys (sentiment family), map directly:
       BUY := positive, SELL := negative, HOLD := neutral. Preserves the
       model's calibrated distribution.
    2. **Confidence-split fallback** — assign `confidence` to the chosen
       action and split `(1 - confidence) / 2` equally between the other
       two. Not perfectly calibrated but gives a coherent log for signals
       that emit only `{action, confidence}`.

    Returns None when input is malformed (non-LLM signal, bad action,
    confidence out of range). Never raises — callers can invoke
    unconditionally and skip logging when None is returned.

    Does NOT call log_vote() itself; callers compose the two steps so the
    log site owns timing and metadata.
    """
    if signal_name not in _LLM_SIGNALS:
        return None
    if action not in _VALID_ACTIONS:
        return None

    try:
        conf = _clamp_confidence(float(confidence))
    except (TypeError, ValueError):
        return None

    indicators = indicators or {}
    avg_scores = indicators.get("avg_scores")
    if isinstance(avg_scores, dict):
        pos = avg_scores.get("positive")
        neg = avg_scores.get("negative")
        neu = avg_scores.get("neutral")
        if all(isinstance(x, (int, float)) for x in (pos, neg, neu)):
            total = float(pos) + float(neg) + float(neu)
            if total > 0:
                return {
                    "BUY": float(pos) / total,
                    "SELL": float(neg) / total,
                    "HOLD": float(neu) / total,
                }

    # Confidence-split fallback. Floor at 1/3 — if the model ACTIVELY chose
    # an action, its probability for that action must be at least the random
    # baseline, otherwise the distribution contradicts the argmax.
    conf = max(conf, 1.0 / 3.0)
    if conf >= 1.0:
        probs = {"BUY": 0.0, "HOLD": 0.0, "SELL": 0.0}
        probs[action] = 1.0
    else:
        remainder = (1.0 - conf) / 2.0
        probs = {"BUY": remainder, "HOLD": remainder, "SELL": remainder}
        probs[action] = conf

    total = sum(probs.values())
    if abs(total - 1.0) > 1e-9:
        # Numerical drift from rounding — normalise.
        probs = {k: v / total for k, v in probs.items()}
    return probs
