from datetime import UTC, datetime
from unittest.mock import patch

from portfolio.accuracy_stats import accuracy_by_signal_ticker
from portfolio.signal_engine import _gate_local_model_vote
from portfolio.signals.forecast import _gate_subsignal_votes_by_accuracy


def _now():
    return datetime.now(UTC).isoformat()


class TestAccuracyBySignalTicker:
    def test_aggregates_per_ticker_accuracy_for_one_signal(self):
        entries = [
            {
                "ts": _now(),
                "tickers": {
                    "BTC-USD": {"signals": {"ministral": "BUY"}},
                    "ETH-USD": {"signals": {"ministral": "SELL"}},
                },
                "outcomes": {
                    "BTC-USD": {"1d": {"change_pct": 1.0}},
                    "ETH-USD": {"1d": {"change_pct": -1.0}},
                },
            },
            {
                "ts": _now(),
                "tickers": {
                    "BTC-USD": {"signals": {"ministral": "BUY"}},
                },
                "outcomes": {
                    "BTC-USD": {"1d": {"change_pct": -1.0}},
                },
            },
        ]

        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = accuracy_by_signal_ticker("ministral", horizon="1d")

        assert result["BTC-USD"]["samples"] == 2
        assert result["BTC-USD"]["correct"] == 1
        assert result["BTC-USD"]["accuracy"] == 0.5
        assert result["ETH-USD"]["samples"] == 1
        assert result["ETH-USD"]["accuracy"] == 1.0


class TestGateLocalModelVote:
    def test_holds_low_accuracy_vote(self):
        acc = {"BTC-USD": {"accuracy": 0.40, "samples": 40}}
        with patch("portfolio.signal_engine._load_local_model_accuracy", return_value=acc):
            vote, info = _gate_local_model_vote("ministral", "BUY", "BTC-USD", config={})

        assert vote == "HOLD"
        assert info["gating"] == "held"
        assert info["accuracy"] == 0.4
        assert info["samples"] == 40

    def test_keeps_vote_when_accuracy_is_good(self):
        acc = {"BTC-USD": {"accuracy": 0.61, "samples": 40}}
        with patch("portfolio.signal_engine._load_local_model_accuracy", return_value=acc):
            vote, info = _gate_local_model_vote("ministral", "SELL", "BTC-USD", config={})

        assert vote == "SELL"
        assert info["gating"] == "raw"
        assert info["accuracy"] == 0.61

    def test_returns_raw_vote_when_samples_are_insufficient(self):
        acc = {"BTC-USD": {"accuracy": 0.10, "samples": 5}}
        with patch("portfolio.signal_engine._load_local_model_accuracy", return_value=acc):
            vote, info = _gate_local_model_vote("ministral", "BUY", "BTC-USD", config={})

        assert vote == "BUY"
        assert info["gating"] == "insufficient_data"
        assert info["samples"] == 5


class TestGateForecastSubsignals:
    def test_gates_bad_ticker_subsignal_but_keeps_good_one(self):
        accuracy = {
            "1h": {
                "chronos_1h": {
                    "accuracy": 0.7,
                    "total": 40,
                    "by_ticker": {"BTC-USD": {"accuracy": 0.4, "correct": 4, "total": 10}},
                }
            },
            "24h": {
                "chronos_24h": {
                    "accuracy": 0.8,
                    "total": 40,
                    "by_ticker": {"BTC-USD": {"accuracy": 0.8, "correct": 16, "total": 20}},
                },
                "kronos_24h": {
                    "accuracy": 0.65,
                    "total": 30,
                    "by_ticker": {},
                },
            },
        }

        with patch("portfolio.signals.forecast._load_forecast_subsignal_accuracy", return_value=accuracy):
            gated, info = _gate_subsignal_votes_by_accuracy(
                {
                    "chronos_1h": "BUY",
                    "chronos_24h": "SELL",
                    "kronos_24h": "BUY",
                },
                "BTC-USD",
                config_forecast={},
            )

        assert gated["chronos_1h"] == "HOLD"
        assert gated["chronos_24h"] == "SELL"
        assert gated["kronos_24h"] == "BUY"
        assert info["chronos_1h"]["gating"] == "held"
        assert info["chronos_1h"]["source"] == "ticker"
        assert info["kronos_24h"]["gating"] == "raw"
        assert info["kronos_24h"]["source"] == "global"
