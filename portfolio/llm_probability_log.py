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
        # Currently active voters and the previously-tracked LLM cohort.
        "ministral",
        "qwen3",
        "sentiment",
        "news_event",
        "forecast",
        "claude_fundamental",
        # 2026-05-15 LLM shadow-enrollment: split sentiment sub-models out
        # so per-model accuracy/Brier can be measured independently of the
        # aggregate `sentiment` voter (currently 46% on 40K samples). The
        # ensemble masks which sub-model is actually carrying signal. These
        # names are emitted by portfolio/sentiment.py.fan_out_sub_models()
        # when the aggregate is computed; if the fan-out site is absent
        # for any reason, log_vote() is a no-op so this is safe to seed.
        "cryptobert",
        "finbert",
        "trading_hero",
        "fingpt",
        # 2026-05-15 LLM shadow-enrollment: scaffold entries for three
        # unused models on disk (finance-llama-8b-gguf, cryptotrader-lm,
        # custom-meta-trader). Wrappers in portfolio/signals/* register
        # with the signal engine and emit log_vote() abstention rows until
        # real inference is wired up. Registering the name here so log_vote
        # accepts the row instead of silently dropping it.
        "finance_llama",
        "cryptotrader_lm",
        "meta_trader",
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


def is_directional_prediction(row: dict) -> bool:
    """Return True iff `row` (one entry from ``llm_probability_log.jsonl``)
    represents a directional prediction worth scoring against the outcome.

    A row is directional when BOTH:

    * ``confidence > 0`` — abstain rows emitted by the canonical
      ``_abstain()`` helper in scaffold/error paths set ``confidence=0``
      and ``chosen="HOLD"``. Counting them in the accuracy denominator
      inflates HOLD-biased shadows because outcome backfill labels
      ~64% of 1d windows as HOLD; a model that always picks HOLD/conf=0
      then "scores" 64% on garbage and would auto-promote.
    * ``chosen in {"BUY","SELL"}`` — HOLD is non-information for trading
      direction. The existing ``data/accuracy_cache.json`` pipeline
      already excludes HOLD predictions from its denominator
      (``total = total_buy + total_sell``); this helper enforces the
      same methodology when reading the probability log.

    Used by:
    * ``scripts/review_shadow_signals.py`` (auto-promotion gate)
    * ``dashboard/app.py:_compute_llm_leaderboard`` (LLM scorecard)

    Both consumers MUST share this filter or they will disagree about
    which shadows are eligible for promotion — exactly the dashboard /
    gate drift documented as premortem narrative #1 in the
    2026-05-18 plan (worktree ``fix/shadow-gate-lora-20260518``).

    Why a row-shape check vs schema validation: the log is append-only
    and historical rows are immutable. Some legacy rows pre-date the
    ``confidence`` field — those return False (treated as abstain) so
    only post-2026-04 rows count toward accuracy, which is the
    conservative choice for a calibration metric.
    """
    if not isinstance(row, dict):
        return False
    conf = row.get("confidence")
    if conf is None:
        return False
    try:
        if float(conf) <= 0.0:
            return False
    except (TypeError, ValueError):
        return False
    chosen = row.get("chosen")
    return chosen in ("BUY", "SELL")


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

    # 2026-05-29 (llm-confidence-calibration): for signals that emit only a
    # scalar confidence (ministral, qwen3), that confidence is anti-calibrated
    # — high self-reported confidence on directionally-wrong calls drove the
    # logged Brier WORSE than uniform (ministral 0.785, qwen3 0.793 vs 0.667).
    # Replace the raw confidence with the empirically-fitted P(correct) from
    # data/llm_confidence_calibration.json before splitting. calibrate() is
    # identity (returns conf unchanged) when no map exists, the signal is
    # absent, or anything goes wrong — so this can never make Brier worse than
    # the synthetic baseline, and the hot loop never raises. Applied ONLY here
    # in the confidence-split fallback; the avg_scores branch above (sentiment
    # family) already carries real per-class scores and is left untouched. See
    # portfolio/llm_confidence_calibration.py for why we do not read real token
    # logprobs on the batched ministral/qwen3 inference path.
    try:
        from portfolio.llm_confidence_calibration import calibrate as _calibrate
        conf = _clamp_confidence(float(_calibrate(signal_name, action, conf)))
    except Exception:
        # Never let calibration break the logger — keep the raw conf.
        pass

    # Confidence-split fallback. Three regimes:
    #   - conf >= 1.0: model fully certain
    #   - 0 < conf < 1.0: standard split, action gets conf, others split (1-conf)/2
    #   - conf <= 0:    "soft preference" — model committed to argmax but
    #                   didn't quantify confidence (or quantification was lost).
    #                   Use {action=0.5, others=0.25} to honor the argmax
    #                   without overstating certainty.
    #
    # 2026-04-30 (incident: qwen3+ministral all-uniform probabilities):
    # The pre-fix `max(conf, 1/3)` floor was wrong. With conf=0:
    #   conf = max(0, 1/3) = 1/3
    #   remainder = (1 - 1/3) / 2 = 1/3
    #   probs[action] = 1/3, others = 1/3 — UNIFORM, contradicts argmax.
    # qwen3 + ministral logged 100% uniform for 9+ days because of this,
    # invalidating Brier-score calibration analysis (probability_log entries
    # had `chosen=BUY` but `probs=[1/3, 1/3, 1/3]` — the cardinal sin of
    # calibration data: distribution disagrees with the argmax).
    if conf >= 1.0:
        probs = {"BUY": 0.0, "HOLD": 0.0, "SELL": 0.0}
        probs[action] = 1.0
        return probs

    if conf <= 0.0:
        probs = {"BUY": 0.25, "HOLD": 0.25, "SELL": 0.25}
        probs[action] = 0.5
        return probs

    # Standard split. If conf is very small (e.g. 0.05) the action would
    # still be below the others (0.475 each), contradicting the argmax.
    # Floor the action just above 1/3 in that regime so the argmax holds.
    if conf < 1.0 / 3.0 + 0.05:
        conf = 1.0 / 3.0 + 0.05
    remainder = (1.0 - conf) / 2.0
    probs = {"BUY": remainder, "HOLD": remainder, "SELL": remainder}
    probs[action] = conf

    total = sum(probs.values())
    if abs(total - 1.0) > 1e-9:
        # Numerical drift from rounding — normalise.
        probs = {k: v / total for k, v in probs.items()}
    return probs
