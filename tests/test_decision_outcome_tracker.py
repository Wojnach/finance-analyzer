"""Tests for portfolio.decision_outcome_tracker."""

import datetime as dt
import json

import pytest


def test_backfill_skips_neutral_outlook(tmp_path, monkeypatch):
    """Neutral outlook predictions are not scored."""
    from portfolio import decision_outcome_tracker as mod

    decisions_file = tmp_path / "decisions.jsonl"
    outcomes_file = tmp_path / "outcomes.jsonl"
    monkeypatch.setattr(mod, "DECISIONS_FILE", decisions_file)
    monkeypatch.setattr(mod, "OUTCOMES_FILE", outcomes_file)

    ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2)).isoformat()
    decision = {
        "ts": ts,
        "predictions": {"BTC-USD": {"outlook": "neutral", "conviction": 0.0}},
        "prices": {"BTC-USD": 67000},
        "fx_rate": 10.0,
    }
    decisions_file.write_text(json.dumps(decision) + "\n")

    monkeypatch.setattr(
        "portfolio.outcome_tracker._fetch_historical_price",
        lambda ticker, target_ts: 68000.0,
    )

    count = mod.backfill_decision_outcomes()
    assert count == 0


def test_backfill_scores_bullish_correctly(tmp_path, monkeypatch):
    """Bullish outlook + positive change = correct."""
    from portfolio import decision_outcome_tracker as mod

    decisions_file = tmp_path / "decisions.jsonl"
    outcomes_file = tmp_path / "outcomes.jsonl"
    monkeypatch.setattr(mod, "DECISIONS_FILE", decisions_file)
    monkeypatch.setattr(mod, "OUTCOMES_FILE", outcomes_file)

    ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2)).isoformat()
    decision = {
        "ts": ts,
        "predictions": {
            "MU": {"outlook": "bullish", "conviction": 0.7, "recommendation": "BUY"},
        },
        "prices": {"MU": 100.0},
        "fx_rate": 10.0,
    }
    decisions_file.write_text(json.dumps(decision) + "\n")

    monkeypatch.setattr(
        "portfolio.outcome_tracker._fetch_historical_price",
        lambda ticker, target_ts: 105.0,
    )

    count = mod.backfill_decision_outcomes()
    assert count >= 1

    outcomes = [json.loads(line) for line in outcomes_file.read_text().strip().split("\n")]
    assert any(o["correct"] is True and o["ticker"] == "MU" for o in outcomes)


def test_backfill_scores_bearish_correctly(tmp_path, monkeypatch):
    """Bearish outlook + negative change = correct."""
    from portfolio import decision_outcome_tracker as mod

    decisions_file = tmp_path / "decisions.jsonl"
    outcomes_file = tmp_path / "outcomes.jsonl"
    monkeypatch.setattr(mod, "DECISIONS_FILE", decisions_file)
    monkeypatch.setattr(mod, "OUTCOMES_FILE", outcomes_file)

    ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=4)).isoformat()
    decision = {
        "ts": ts,
        "predictions": {
            "ETH-USD": {"outlook": "bearish", "conviction": 0.5, "recommendation": "SELL"},
        },
        "prices": {"ETH-USD": 2000.0},
        "fx_rate": 10.0,
    }
    decisions_file.write_text(json.dumps(decision) + "\n")

    monkeypatch.setattr(
        "portfolio.outcome_tracker._fetch_historical_price",
        lambda ticker, target_ts: 1900.0,
    )

    count = mod.backfill_decision_outcomes()
    assert count >= 1

    outcomes = [json.loads(line) for line in outcomes_file.read_text().strip().split("\n")]
    bearish_outcomes = [o for o in outcomes if o["ticker"] == "ETH-USD"]
    assert all(o["correct"] is True for o in bearish_outcomes)


def test_backfill_scores_wrong_prediction(tmp_path, monkeypatch):
    """Bullish outlook + negative change = incorrect."""
    from portfolio import decision_outcome_tracker as mod

    decisions_file = tmp_path / "decisions.jsonl"
    outcomes_file = tmp_path / "outcomes.jsonl"
    monkeypatch.setattr(mod, "DECISIONS_FILE", decisions_file)
    monkeypatch.setattr(mod, "OUTCOMES_FILE", outcomes_file)

    ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2)).isoformat()
    decision = {
        "ts": ts,
        "predictions": {
            "NVDA": {"outlook": "bullish", "conviction": 0.6, "recommendation": "BUY"},
        },
        "prices": {"NVDA": 200.0},
        "fx_rate": 10.0,
    }
    decisions_file.write_text(json.dumps(decision) + "\n")

    monkeypatch.setattr(
        "portfolio.outcome_tracker._fetch_historical_price",
        lambda ticker, target_ts: 190.0,
    )

    count = mod.backfill_decision_outcomes()
    assert count >= 1

    outcomes = [json.loads(line) for line in outcomes_file.read_text().strip().split("\n")]
    assert any(o["correct"] is False and o["ticker"] == "NVDA" for o in outcomes)


def test_backfill_deduplicates(tmp_path, monkeypatch):
    """Running backfill twice does not create duplicate outcomes."""
    from portfolio import decision_outcome_tracker as mod

    decisions_file = tmp_path / "decisions.jsonl"
    outcomes_file = tmp_path / "outcomes.jsonl"
    monkeypatch.setattr(mod, "DECISIONS_FILE", decisions_file)
    monkeypatch.setattr(mod, "OUTCOMES_FILE", outcomes_file)

    ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=4)).isoformat()
    decision = {
        "ts": ts,
        "predictions": {
            "ETH-USD": {"outlook": "bearish", "conviction": 0.5, "recommendation": "SELL"},
        },
        "prices": {"ETH-USD": 2000.0},
        "fx_rate": 10.0,
    }
    decisions_file.write_text(json.dumps(decision) + "\n")

    monkeypatch.setattr(
        "portfolio.outcome_tracker._fetch_historical_price",
        lambda ticker, target_ts: 1900.0,
    )

    count1 = mod.backfill_decision_outcomes()
    count2 = mod.backfill_decision_outcomes()
    assert count1 >= 1
    assert count2 == 0  # no new outcomes on second run


def test_backfill_skips_future_horizons(tmp_path, monkeypatch):
    """Decisions too recent for any horizon are skipped."""
    from portfolio import decision_outcome_tracker as mod

    decisions_file = tmp_path / "decisions.jsonl"
    outcomes_file = tmp_path / "outcomes.jsonl"
    monkeypatch.setattr(mod, "DECISIONS_FILE", decisions_file)
    monkeypatch.setattr(mod, "OUTCOMES_FILE", outcomes_file)

    # Decision from 1 hour ago -- too soon for 1d or 3d horizon
    ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)).isoformat()
    decision = {
        "ts": ts,
        "predictions": {
            "BTC-USD": {"outlook": "bullish", "conviction": 0.8, "recommendation": "BUY"},
        },
        "prices": {"BTC-USD": 67000},
        "fx_rate": 10.0,
    }
    decisions_file.write_text(json.dumps(decision) + "\n")

    fetch_called = []
    monkeypatch.setattr(
        "portfolio.outcome_tracker._fetch_historical_price",
        lambda ticker, target_ts: fetch_called.append(1) or 68000.0,
    )

    count = mod.backfill_decision_outcomes()
    assert count == 0
    assert len(fetch_called) == 0


def test_backfill_skips_missing_price(tmp_path, monkeypatch):
    """Predictions without a base price in the prices dict are skipped."""
    from portfolio import decision_outcome_tracker as mod

    decisions_file = tmp_path / "decisions.jsonl"
    outcomes_file = tmp_path / "outcomes.jsonl"
    monkeypatch.setattr(mod, "DECISIONS_FILE", decisions_file)
    monkeypatch.setattr(mod, "OUTCOMES_FILE", outcomes_file)

    ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2)).isoformat()
    decision = {
        "ts": ts,
        "predictions": {
            "XAG-USD": {"outlook": "bullish", "conviction": 0.7, "recommendation": "BUY"},
        },
        "prices": {},  # no price for XAG-USD
        "fx_rate": 10.0,
    }
    decisions_file.write_text(json.dumps(decision) + "\n")

    monkeypatch.setattr(
        "portfolio.outcome_tracker._fetch_historical_price",
        lambda ticker, target_ts: 35.0,
    )

    count = mod.backfill_decision_outcomes()
    assert count == 0


def test_backfill_handles_empty_file(tmp_path, monkeypatch):
    """Empty decisions file returns 0."""
    from portfolio import decision_outcome_tracker as mod

    decisions_file = tmp_path / "decisions.jsonl"
    outcomes_file = tmp_path / "outcomes.jsonl"
    monkeypatch.setattr(mod, "DECISIONS_FILE", decisions_file)
    monkeypatch.setattr(mod, "OUTCOMES_FILE", outcomes_file)

    # Don't create the file — load_jsonl should handle missing file
    count = mod.backfill_decision_outcomes()
    assert count == 0


def test_backfill_handles_fetch_failure(tmp_path, monkeypatch):
    """If _fetch_historical_price raises, that outcome is skipped."""
    from portfolio import decision_outcome_tracker as mod

    decisions_file = tmp_path / "decisions.jsonl"
    outcomes_file = tmp_path / "outcomes.jsonl"
    monkeypatch.setattr(mod, "DECISIONS_FILE", decisions_file)
    monkeypatch.setattr(mod, "OUTCOMES_FILE", outcomes_file)

    ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2)).isoformat()
    decision = {
        "ts": ts,
        "predictions": {
            "BTC-USD": {"outlook": "bullish", "conviction": 0.8, "recommendation": "BUY"},
        },
        "prices": {"BTC-USD": 67000},
        "fx_rate": 10.0,
    }
    decisions_file.write_text(json.dumps(decision) + "\n")

    def failing_fetch(ticker, target_ts):
        raise ConnectionError("API down")

    monkeypatch.setattr(
        "portfolio.outcome_tracker._fetch_historical_price",
        failing_fetch,
    )

    count = mod.backfill_decision_outcomes()
    assert count == 0
