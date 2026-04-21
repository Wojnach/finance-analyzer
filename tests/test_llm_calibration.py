"""Tests for portfolio.llm_calibration."""
from __future__ import annotations

import json
import math

import pytest

from portfolio import llm_calibration as mod


# --- outcome_from_return ----------------------------------------------------


def test_outcome_default_buy_threshold():
    assert mod.outcome_from_return(0.01) == "BUY"
    assert mod.outcome_from_return(0.005) == "BUY"  # inclusive


def test_outcome_default_sell_threshold():
    assert mod.outcome_from_return(-0.01) == "SELL"
    assert mod.outcome_from_return(-0.005) == "SELL"


def test_outcome_default_dead_band():
    assert mod.outcome_from_return(0.0) == "HOLD"
    assert mod.outcome_from_return(0.004) == "HOLD"
    assert mod.outcome_from_return(-0.004) == "HOLD"


def test_outcome_custom_thresholds():
    # +1 %/-1 % thresholds
    assert mod.outcome_from_return(0.009, buy_threshold=0.01, sell_threshold=-0.01) == "HOLD"
    assert mod.outcome_from_return(0.011, buy_threshold=0.01, sell_threshold=-0.01) == "BUY"


# --- brier_score ------------------------------------------------------------


def test_brier_perfect_prediction():
    probs = {"BUY": 1.0, "HOLD": 0.0, "SELL": 0.0}
    assert mod.brier_score(probs, "BUY") == 0.0


def test_brier_uniform_guess():
    probs = {"BUY": 1 / 3, "HOLD": 1 / 3, "SELL": 1 / 3}
    # For uniform: Σ (1/3 - I)^2 = (1/3-1)^2 + 2*(1/3)^2 = 4/9 + 2/9 = 6/9 = 2/3
    assert mod.brier_score(probs, "BUY") == pytest.approx(2 / 3, rel=1e-6)


def test_brier_maximally_wrong():
    probs = {"BUY": 0.0, "HOLD": 0.0, "SELL": 1.0}
    # outcome=BUY: (0-1)^2 + (0-0)^2 + (1-0)^2 = 2.0
    assert mod.brier_score(probs, "BUY") == pytest.approx(2.0)


def test_brier_rejects_bad_outcome():
    with pytest.raises(ValueError):
        mod.brier_score({"BUY": 0.5, "HOLD": 0.3, "SELL": 0.2}, "MAYBE")


# --- log_loss ---------------------------------------------------------------


def test_log_loss_perfect_prediction():
    probs = {"BUY": 1.0, "HOLD": 0.0, "SELL": 0.0}
    # -log(1) = 0
    assert mod.log_loss(probs, "BUY") == pytest.approx(0.0)


def test_log_loss_uniform_guess():
    probs = {"BUY": 1 / 3, "HOLD": 1 / 3, "SELL": 1 / 3}
    # -log(1/3) = log(3) ≈ 1.0986
    assert mod.log_loss(probs, "BUY") == pytest.approx(math.log(3), rel=1e-6)


def test_log_loss_zero_probability_clamped():
    """Zero probability on the true class is clamped by eps so log-loss
    stays finite (not infinity)."""
    probs = {"BUY": 0.0, "HOLD": 0.5, "SELL": 0.5}
    ll = mod.log_loss(probs, "BUY")
    # -log(1e-12) ≈ 27.63
    assert 25 < ll < 30


def test_log_loss_rejects_bad_outcome():
    with pytest.raises(ValueError):
        mod.log_loss({"BUY": 0.5, "HOLD": 0.3, "SELL": 0.2}, "YES")


# --- compute_metrics --------------------------------------------------------


def _write_prob_log(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_compute_metrics_aggregates_by_signal(tmp_path):
    log_path = tmp_path / "llm_probability_log.jsonl"
    _write_prob_log(log_path, [
        {"ts": "2026-04-21T12:00:00+00:00", "signal": "ministral", "ticker": "BTC-USD",
         "horizon": "1d", "probs": {"BUY": 0.7, "HOLD": 0.2, "SELL": 0.1}, "chosen": "BUY", "confidence": 0.7},
        {"ts": "2026-04-21T12:05:00+00:00", "signal": "ministral", "ticker": "BTC-USD",
         "horizon": "1d", "probs": {"BUY": 0.1, "HOLD": 0.2, "SELL": 0.7}, "chosen": "SELL", "confidence": 0.7},
        {"ts": "2026-04-21T12:00:00+00:00", "signal": "qwen3", "ticker": "BTC-USD",
         "horizon": "1d", "probs": {"BUY": 0.3, "HOLD": 0.4, "SELL": 0.3}, "chosen": "HOLD", "confidence": 0.4},
    ])
    # All outcomes hit (ministral: BUY correct, SELL correct; qwen3: HOLD correct)
    outcomes = {
        ("2026-04-21T12:00:00+00:00", "BTC-USD", "1d"): "BUY",
        ("2026-04-21T12:05:00+00:00", "BTC-USD", "1d"): "SELL",
    }
    qwen3_outcomes = {("2026-04-21T12:00:00+00:00", "BTC-USD", "1d"): "HOLD"}
    all_outcomes = {**outcomes, **qwen3_outcomes}

    def _lookup(ts, tkr, h):
        # Prefer explicit ticker+horizon match; ministral overrides qwen3 for same key.
        return all_outcomes.get((ts, tkr, h))

    # But ministral's first row needs BUY, qwen3's needs HOLD — same (ts, tkr, h) conflict.
    # Use a richer lookup keyed on signal.
    rows = [
        ("2026-04-21T12:00:00+00:00", "BTC-USD", "ministral", "BUY"),
        ("2026-04-21T12:05:00+00:00", "BTC-USD", "ministral", "SELL"),
        ("2026-04-21T12:00:00+00:00", "BTC-USD", "qwen3", "HOLD"),
    ]
    # Since compute_metrics signature is (ts, tkr, h) → outcome, and all three
    # rows above collide at ts+tkr+h, we rewrite two qwen3 rows with different ts.
    log_path.write_text("")
    _write_prob_log(log_path, [
        {"ts": "2026-04-21T12:00:00+00:00", "signal": "ministral", "ticker": "BTC-USD",
         "horizon": "1d", "probs": {"BUY": 0.7, "HOLD": 0.2, "SELL": 0.1}, "chosen": "BUY", "confidence": 0.7},
        {"ts": "2026-04-21T12:05:00+00:00", "signal": "ministral", "ticker": "BTC-USD",
         "horizon": "1d", "probs": {"BUY": 0.1, "HOLD": 0.2, "SELL": 0.7}, "chosen": "SELL", "confidence": 0.7},
        {"ts": "2026-04-21T12:10:00+00:00", "signal": "qwen3", "ticker": "BTC-USD",
         "horizon": "1d", "probs": {"BUY": 0.3, "HOLD": 0.4, "SELL": 0.3}, "chosen": "HOLD", "confidence": 0.4},
    ])
    map_ = {
        ("2026-04-21T12:00:00+00:00", "BTC-USD", "1d"): "BUY",
        ("2026-04-21T12:05:00+00:00", "BTC-USD", "1d"): "SELL",
        ("2026-04-21T12:10:00+00:00", "BTC-USD", "1d"): "HOLD",
    }

    out = mod.compute_metrics(lambda ts, t, h: map_.get((ts, t, h)),
                               log_path=log_path, days=None)

    assert "ministral" in out and "qwen3" in out
    assert out["ministral"]["samples"] == 2
    assert out["qwen3"]["samples"] == 1
    # Ministral both right with 0.7 conf → brier = (0.3^2 + 0.2^2 + 0.1^2) per row = 0.14
    assert out["ministral"]["brier_mean"] == pytest.approx(0.14, rel=1e-6)
    # Chosen action hit rate: ministral BUY=1/1, SELL=1/1; qwen3 HOLD=1/1
    assert out["ministral"]["buckets"]["BUY"]["hit_rate"] == 1.0
    assert out["ministral"]["buckets"]["SELL"]["hit_rate"] == 1.0
    assert out["qwen3"]["buckets"]["HOLD"]["hit_rate"] == 1.0


def test_compute_metrics_reports_missing_outcome(tmp_path):
    log_path = tmp_path / "llm_probability_log.jsonl"
    _write_prob_log(log_path, [
        {"ts": "2026-04-21T12:00:00+00:00", "signal": "ministral", "ticker": "BTC-USD",
         "horizon": "1d", "probs": {"BUY": 0.7, "HOLD": 0.2, "SELL": 0.1}, "chosen": "BUY", "confidence": 0.7},
    ])
    out = mod.compute_metrics(lambda *_: None, log_path=log_path, days=None)
    assert out["ministral"]["samples"] == 0
    assert out["ministral"]["missing_outcome"] == 1
    assert out["ministral"]["brier_mean"] is None


def test_compute_metrics_skips_malformed_rows(tmp_path):
    log_path = tmp_path / "llm_probability_log.jsonl"
    log_path.write_text(
        "\n".join([
            # Malformed: missing keys
            json.dumps({"ts": "2026-04-21T12:00:00+00:00", "signal": "ministral",
                         "probs": {"BUY": 0.5, "HOLD": 0.5}}),
            # Malformed: JSON garbage
            "not json",
            # Good
            json.dumps({
                "ts": "2026-04-21T12:00:00+00:00", "signal": "ministral",
                "ticker": "BTC-USD", "horizon": "1d",
                "probs": {"BUY": 0.7, "HOLD": 0.2, "SELL": 0.1},
                "chosen": "BUY", "confidence": 0.7,
            }),
        ]) + "\n"
    )
    out = mod.compute_metrics(
        lambda ts, t, h: "BUY",
        log_path=log_path, days=None,
    )
    # Only the good row counted
    assert out["ministral"]["samples"] == 1


def test_compute_metrics_returns_empty_when_log_missing(tmp_path):
    log_path = tmp_path / "nonexistent.jsonl"
    out = mod.compute_metrics(lambda *_: "BUY", log_path=log_path, days=None)
    assert out == {}
