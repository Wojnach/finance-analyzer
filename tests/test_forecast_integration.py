"""Tests for forecast integration — enriched probabilities, thesis alignment, compact summary."""

import json
import math
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# TestDirectionProbabilityWithForecast
# ---------------------------------------------------------------------------

class TestDirectionProbabilityWithForecast:
    """Tests for ticker_accuracy.direction_probability_with_forecast()"""

    def _make_base_result(self, probability=0.5, direction="neutral", signals_used=2,
                          total_samples=100, details=None):
        """Helper to build a fake direction_probability() return value."""
        if details is None:
            details = [
                {"name": "rsi", "vote": "BUY", "accuracy": 0.6, "samples": 50,
                 "p_up": 0.6, "weight": math.sqrt(50)},
                {"name": "ema", "vote": "BUY", "accuracy": 0.55, "samples": 50,
                 "p_up": 0.55, "weight": math.sqrt(50)},
            ]
        return {
            "direction": direction,
            "probability": probability,
            "signals_used": signals_used,
            "total_samples": total_samples,
            "signal_details": details,
        }

    @patch("portfolio.ticker_accuracy.direction_probability")
    def test_no_forecast_data(self, mock_dp):
        """No forecast data -> forecast_blended=False, base result unchanged."""
        from portfolio.ticker_accuracy import direction_probability_with_forecast
        mock_dp.return_value = self._make_base_result(0.6, "up")
        result = direction_probability_with_forecast("BTC-USD", {"rsi": "BUY"})
        assert result["forecast_blended"] is False
        assert result["probability"] == 0.6

    @patch("portfolio.ticker_accuracy.direction_probability")
    def test_chronos_not_ok(self, mock_dp):
        """Forecast data with chronos_ok=False -> not blended."""
        from portfolio.ticker_accuracy import direction_probability_with_forecast
        mock_dp.return_value = self._make_base_result(0.6, "up")
        fd = {"chronos_ok": False, "chronos_24h_pct": 1.5, "chronos_24h_conf": 0.8}
        result = direction_probability_with_forecast("BTC-USD", {"rsi": "BUY"},
                                                      forecast_data=fd)
        assert result["forecast_blended"] is False

    @patch("portfolio.ticker_accuracy.direction_probability")
    def test_valid_forecast_blended(self, mock_dp):
        """Valid forecast data with positive pct_move -> probability pushed up."""
        from portfolio.ticker_accuracy import direction_probability_with_forecast
        mock_dp.return_value = self._make_base_result(0.5, "neutral")
        fd = {"chronos_ok": True, "chronos_24h_pct": 2.0, "chronos_24h_conf": 0.7}
        result = direction_probability_with_forecast("XAG-USD", {"rsi": "BUY"},
                                                      forecast_data=fd)
        assert result["forecast_blended"] is True
        assert result["probability"] > 0.5  # should push up
        assert result["forecast_pct_move"] == 2.0
        assert result["forecast_confidence"] == 0.7

    @patch("portfolio.ticker_accuracy.direction_probability")
    def test_negative_pct_move_pushes_down(self, mock_dp):
        """Negative forecast pct_move -> probability pushed down."""
        from portfolio.ticker_accuracy import direction_probability_with_forecast
        mock_dp.return_value = self._make_base_result(0.5, "neutral")
        fd = {"chronos_ok": True, "chronos_24h_pct": -3.0, "chronos_24h_conf": 0.8}
        result = direction_probability_with_forecast("BTC-USD", {"rsi": "BUY"},
                                                      forecast_data=fd)
        assert result["forecast_blended"] is True
        assert result["probability"] < 0.5

    @patch("portfolio.ticker_accuracy.direction_probability")
    def test_very_small_pct_move_not_blended(self, mock_dp):
        """pct_move < 0.1 -> not blended even with high confidence."""
        from portfolio.ticker_accuracy import direction_probability_with_forecast
        mock_dp.return_value = self._make_base_result(0.5, "neutral")
        fd = {"chronos_ok": True, "chronos_24h_pct": 0.05, "chronos_24h_conf": 0.9}
        result = direction_probability_with_forecast("BTC-USD", {"rsi": "BUY"},
                                                      forecast_data=fd)
        assert result["forecast_blended"] is False
        assert result["probability"] == 0.5

    @patch("portfolio.ticker_accuracy.direction_probability")
    def test_zero_confidence_not_blended(self, mock_dp):
        """Zero confidence -> not blended."""
        from portfolio.ticker_accuracy import direction_probability_with_forecast
        mock_dp.return_value = self._make_base_result(0.6, "up")
        fd = {"chronos_ok": True, "chronos_24h_pct": 2.0, "chronos_24h_conf": 0.0}
        result = direction_probability_with_forecast("BTC-USD", {"rsi": "BUY"},
                                                      forecast_data=fd)
        assert result["forecast_blended"] is False

    @patch("portfolio.ticker_accuracy.direction_probability")
    def test_horizon_mapping_1d(self, mock_dp):
        """1d horizon -> uses chronos_24h keys."""
        from portfolio.ticker_accuracy import direction_probability_with_forecast
        mock_dp.return_value = self._make_base_result(0.5, "neutral")
        fd = {
            "chronos_ok": True,
            "chronos_24h_pct": 5.0,
            "chronos_24h_conf": 0.8,
            "chronos_1h_pct": 1.0,
            "chronos_1h_conf": 0.5,
        }
        result = direction_probability_with_forecast("BTC-USD", {"rsi": "BUY"},
                                                      forecast_data=fd, horizon="1d")
        assert result["forecast_pct_move"] == 5.0
        assert result["forecast_confidence"] == 0.8

    @patch("portfolio.ticker_accuracy.direction_probability")
    def test_horizon_mapping_3h(self, mock_dp):
        """3h horizon -> uses chronos_1h keys."""
        from portfolio.ticker_accuracy import direction_probability_with_forecast
        mock_dp.return_value = self._make_base_result(0.5, "neutral")
        fd = {
            "chronos_ok": True,
            "chronos_24h_pct": 5.0,
            "chronos_24h_conf": 0.8,
            "chronos_1h_pct": 1.0,
            "chronos_1h_conf": 0.6,
        }
        result = direction_probability_with_forecast("BTC-USD", {"rsi": "BUY"},
                                                      forecast_data=fd, horizon="3h")
        assert result["forecast_pct_move"] == 1.0
        assert result["forecast_confidence"] == 0.6

    @patch("portfolio.ticker_accuracy.direction_probability")
    def test_direction_updated_after_blending(self, mock_dp):
        """After blending, direction should reflect the blended probability."""
        from portfolio.ticker_accuracy import direction_probability_with_forecast
        # Start neutral, strong upward forecast should push to "up"
        base = self._make_base_result(0.5, "neutral")
        base["signal_details"] = [
            {"name": "rsi", "vote": "BUY", "accuracy": 0.5, "samples": 10,
             "p_up": 0.5, "weight": math.sqrt(10)},
        ]
        mock_dp.return_value = base
        fd = {"chronos_ok": True, "chronos_24h_pct": 5.0, "chronos_24h_conf": 0.9}
        result = direction_probability_with_forecast("XAG-USD", {"rsi": "BUY"},
                                                      forecast_data=fd)
        assert result["forecast_blended"] is True
        assert result["direction"] == "up"
        assert result["probability"] > 0.52

    @patch("portfolio.ticker_accuracy.direction_probability")
    def test_none_pct_move_handled(self, mock_dp):
        """None values in forecast data don't cause errors."""
        from portfolio.ticker_accuracy import direction_probability_with_forecast
        mock_dp.return_value = self._make_base_result(0.5, "neutral")
        fd = {"chronos_ok": True, "chronos_24h_pct": None, "chronos_24h_conf": None}
        result = direction_probability_with_forecast("BTC-USD", {"rsi": "BUY"},
                                                      forecast_data=fd)
        assert result["forecast_blended"] is False
        assert result["forecast_pct_move"] == 0
        assert result["forecast_confidence"] == 0


# ---------------------------------------------------------------------------
# TestThesisAlignmentVote
# ---------------------------------------------------------------------------

class TestThesisAlignmentVote:
    """Tests for news_event._thesis_alignment_vote()"""

    def test_config_gate_disabled(self):
        """When prophecy.news_alignment is not set, returns HOLD."""
        from portfolio.signals.news_event import _thesis_alignment_vote
        action, ind = _thesis_alignment_vote([], "XAG-USD", {})
        assert action == "HOLD"
        assert ind["enabled"] is False

    def test_config_gate_enabled_no_beliefs(self):
        """Enabled but no active beliefs -> HOLD."""
        from portfolio.signals.news_event import _thesis_alignment_vote
        config = {"prophecy": {"news_alignment": True}}
        with patch("portfolio.prophecy.get_active_beliefs", return_value=[]):
            action, ind = _thesis_alignment_vote([], "XAG-USD", config)
        assert action == "HOLD"
        assert ind["enabled"] is True

    def test_bullish_belief_positive_headlines(self):
        """Bullish belief + positive headlines -> BUY."""
        from portfolio.signals.news_event import _thesis_alignment_vote
        config = {"prophecy": {"news_alignment": True}}
        belief = {"id": "silver_bull", "direction": "bullish", "conviction": 0.8}
        # Use keywords that register as non-"normal" severity in news_keywords.py
        # "upgrade" (moderate 1.5), "earnings beat" (moderate 1.5), "buyback" (moderate 1.5)
        headlines = [
            {"title": "Analysts upgrade silver outlook amid strong demand"},
            {"title": "Silver ETF approved by regulators, upgrade expected"},
            {"title": "Silver buyback program announced by major producer"},
        ]
        with patch("portfolio.prophecy.get_active_beliefs", return_value=[belief]):
            action, ind = _thesis_alignment_vote(headlines, "XAG-USD", config)
        assert action == "BUY"
        assert ind["alignment"] == "confirmed"
        assert ind["belief_id"] == "silver_bull"

    def test_bearish_belief_negative_headlines(self):
        """Bearish belief + negative headlines -> SELL."""
        from portfolio.signals.news_event import _thesis_alignment_vote
        config = {"prophecy": {"news_alignment": True}}
        belief = {"id": "btc_bear", "direction": "bearish", "conviction": 0.7}
        headlines = [
            {"title": "Bitcoin crashes amid tariff fears and recession risk"},
            {"title": "Crypto market faces crash as war escalates"},
            {"title": "Major exchange bankruptcy feared"},
        ]
        with patch("portfolio.prophecy.get_active_beliefs", return_value=[belief]):
            action, ind = _thesis_alignment_vote(headlines, "BTC-USD", config)
        assert action == "SELL"
        assert ind["alignment"] == "confirmed"

    def test_bullish_belief_negative_headlines_contradicted(self):
        """Bullish belief + negative headlines -> HOLD (contradicted)."""
        from portfolio.signals.news_event import _thesis_alignment_vote
        config = {"prophecy": {"news_alignment": True}}
        belief = {"id": "silver_bull", "direction": "bullish", "conviction": 0.8}
        headlines = [
            {"title": "Silver crashes on tariff escalation"},
            {"title": "Metals market crash amid global recession fears"},
            {"title": "Silver falls as war fears grip markets"},
        ]
        with patch("portfolio.prophecy.get_active_beliefs", return_value=[belief]):
            action, ind = _thesis_alignment_vote(headlines, "XAG-USD", config)
        assert action == "HOLD"
        assert ind["alignment"] == "contradicted"

    def test_neutral_belief_returns_hold(self):
        """Neutral belief direction -> HOLD regardless of headlines."""
        from portfolio.signals.news_event import _thesis_alignment_vote
        config = {"prophecy": {"news_alignment": True}}
        belief = {"id": "btc_range", "direction": "neutral", "conviction": 0.5}
        headlines = [
            {"title": "Bitcoin surges on massive rally"},
            {"title": "Crypto market upgrade expected"},
        ]
        with patch("portfolio.prophecy.get_active_beliefs", return_value=[belief]):
            action, ind = _thesis_alignment_vote(headlines, "BTC-USD", config)
        assert action == "HOLD"

    def test_highest_conviction_belief_used(self):
        """When multiple beliefs exist, highest conviction is used."""
        from portfolio.signals.news_event import _thesis_alignment_vote
        config = {"prophecy": {"news_alignment": True}}
        beliefs = [
            {"id": "low", "direction": "bearish", "conviction": 0.3},
            {"id": "high", "direction": "bullish", "conviction": 0.9},
        ]
        # Use keywords from news_keywords.py that register as non-"normal" severity
        headlines = [
            {"title": "Silver upgrade expected after strong earnings beat"},
            {"title": "Silver buyback announced by major producers"},
        ]
        with patch("portfolio.prophecy.get_active_beliefs", return_value=beliefs):
            action, ind = _thesis_alignment_vote(headlines, "XAG-USD", config)
        assert ind["belief_id"] == "high"
        assert ind["belief_direction"] == "bullish"

    def test_insufficient_headline_count(self):
        """Fewer than 2 keyword-bearing headlines -> HOLD."""
        from portfolio.signals.news_event import _thesis_alignment_vote
        config = {"prophecy": {"news_alignment": True}}
        belief = {"id": "silver_bull", "direction": "bullish", "conviction": 0.8}
        headlines = [
            {"title": "Silver prices steady today"},  # no severity keyword
            {"title": "Markets flat overnight"},      # no severity keyword
        ]
        with patch("portfolio.prophecy.get_active_beliefs", return_value=[belief]):
            action, ind = _thesis_alignment_vote(headlines, "XAG-USD", config)
        assert action == "HOLD"

    def test_prophecy_import_failure_returns_hold(self):
        """If prophecy module import fails, returns HOLD gracefully."""
        from portfolio.signals.news_event import _thesis_alignment_vote
        config = {"prophecy": {"news_alignment": True}}
        with patch("portfolio.prophecy.get_active_beliefs", side_effect=ImportError("no module")):
            action, ind = _thesis_alignment_vote([], "XAG-USD", config)
        assert action == "HOLD"
        assert ind["enabled"] is True


# ---------------------------------------------------------------------------
# TestForecastSignalsInSummary
# ---------------------------------------------------------------------------

class TestForecastSignalsInSummary:
    """Tests for forecast_signals section in agent_summary."""

    def _make_signals(self, tickers_with_forecast=None):
        """Build a minimal signals dict with forecast indicators."""
        signals = {}
        for ticker in (tickers_with_forecast or []):
            signals[ticker] = {
                "action": "HOLD",
                "confidence": 0.0,
                "indicators": {
                    "close": 100.0,
                    "rsi": 50.0,
                    "macd_hist": 0.0,
                    "macd_hist_prev": 0.0,
                    "ema9": 100.0,
                    "ema21": 100.0,
                    "price_vs_bb": "inside",
                    "atr": 2.0,
                    "atr_pct": 2.0,
                },
                "extra": {
                    "forecast_action": "BUY",
                    "forecast_confidence": 0.6,
                    "forecast_indicators": {
                        "chronos_ok": True,
                        "kronos_ok": False,
                        "chronos_1h_pct": 0.5,
                        "chronos_1h_conf": 0.55,
                        "chronos_24h_pct": 1.2,
                        "chronos_24h_conf": 0.7,
                    },
                    "_buy_count": 0,
                    "_sell_count": 0,
                    "_voters": 0,
                },
            }
        return signals

    def test_forecast_signals_added_when_chronos_data_present(self):
        """forecast_signals section added when Chronos data present in extra."""
        from portfolio.reporting import write_agent_summary

        signals = self._make_signals(["XAG-USD"])

        # Minimal required args for write_agent_summary
        state = {
            "cash_sek": 500000,
            "initial_value_sek": 500000,
            "holdings": {},
            "transactions": [],
        }

        # Mock out everything to isolate the forecast_signals logic
        with patch("portfolio.reporting._atomic_write_json") as mock_write, \
             patch("portfolio.reporting._write_compact_summary"), \
             patch("portfolio.reporting._cached", return_value=None), \
             patch("portfolio.reporting.portfolio_value", return_value=500000), \
             patch("portfolio.reporting.detect_regime", return_value="ranging"), \
             patch("portfolio.reporting.get_enhanced_signals", return_value={}), \
             patch("portfolio.reporting.AGENT_SUMMARY_FILE", Path("/tmp/fake_summary.json")):

            # Patch the AGENT_SUMMARY_FILE.exists() to return False
            with patch.object(Path, "exists", return_value=False):
                summary = write_agent_summary(
                    signals, {"XAG-USD": 30.0}, 10.0, state, {}, []
                )

        assert "forecast_signals" in summary
        assert "XAG-USD" in summary["forecast_signals"]
        fs = summary["forecast_signals"]["XAG-USD"]
        assert fs["chronos_ok"] is True
        assert fs["chronos_24h_pct"] == 1.2
        assert fs["action"] == "BUY"

    def test_forecast_signals_not_added_when_no_chronos_data(self):
        """No forecast_indicators in extra -> no forecast_signals section."""
        from portfolio.reporting import write_agent_summary

        # Signals without forecast_indicators
        signals = {
            "BTC-USD": {
                "action": "HOLD",
                "confidence": 0.0,
                "indicators": {
                    "close": 67000.0,
                    "rsi": 50.0,
                    "macd_hist": 0.0,
                    "macd_hist_prev": 0.0,
                    "ema9": 67000.0,
                    "ema21": 67000.0,
                    "price_vs_bb": "inside",
                    "atr": 1500.0,
                    "atr_pct": 2.2,
                },
                "extra": {
                    "_buy_count": 0,
                    "_sell_count": 0,
                    "_voters": 0,
                },
            }
        }

        state = {
            "cash_sek": 500000,
            "initial_value_sek": 500000,
            "holdings": {},
            "transactions": [],
        }

        with patch("portfolio.reporting._atomic_write_json"), \
             patch("portfolio.reporting._write_compact_summary"), \
             patch("portfolio.reporting._cached", return_value=None), \
             patch("portfolio.reporting.portfolio_value", return_value=500000), \
             patch("portfolio.reporting.detect_regime", return_value="ranging"), \
             patch("portfolio.reporting.get_enhanced_signals", return_value={}), \
             patch("portfolio.reporting.AGENT_SUMMARY_FILE", Path("/tmp/fake_summary.json")):
            with patch.object(Path, "exists", return_value=False):
                summary = write_agent_summary(
                    signals, {"BTC-USD": 67000.0}, 10.0, state, {}, []
                )

        assert "forecast_signals" not in summary

    def test_correct_fields_extracted(self):
        """Verify all expected fields are present in forecast_signals entries."""
        from portfolio.reporting import write_agent_summary

        signals = self._make_signals(["BTC-USD"])
        state = {
            "cash_sek": 500000,
            "initial_value_sek": 500000,
            "holdings": {},
            "transactions": [],
        }

        with patch("portfolio.reporting._atomic_write_json"), \
             patch("portfolio.reporting._write_compact_summary"), \
             patch("portfolio.reporting._cached", return_value=None), \
             patch("portfolio.reporting.portfolio_value", return_value=500000), \
             patch("portfolio.reporting.detect_regime", return_value="ranging"), \
             patch("portfolio.reporting.get_enhanced_signals", return_value={}), \
             patch("portfolio.reporting.AGENT_SUMMARY_FILE", Path("/tmp/fake_summary.json")):
            with patch.object(Path, "exists", return_value=False):
                summary = write_agent_summary(
                    signals, {"BTC-USD": 67000.0}, 10.0, state, {}, []
                )

        fs = summary["forecast_signals"]["BTC-USD"]
        expected_keys = {"action", "chronos_1h_pct", "chronos_1h_conf",
                         "chronos_24h_pct", "chronos_24h_conf", "kronos_ok", "chronos_ok"}
        assert expected_keys == set(fs.keys())

    def test_forecast_signals_propagated_to_compact(self):
        """forecast_signals is propagated to compact summary."""
        from portfolio.reporting import _write_compact_summary
        from portfolio.reporting import _atomic_write_json

        summary = {
            "signals": {},
            "timeframes": {},
            "fear_greed": {},
            "forecast_signals": {
                "XAG-USD": {
                    "action": "BUY",
                    "chronos_1h_pct": 0.5,
                    "chronos_1h_conf": 0.55,
                    "chronos_24h_pct": 1.2,
                    "chronos_24h_conf": 0.7,
                    "kronos_ok": False,
                    "chronos_ok": True,
                }
            },
        }

        with patch("portfolio.reporting._atomic_write_json") as mock_write, \
             patch("portfolio.reporting._get_held_tickers", return_value=set()):
            _write_compact_summary(summary)

        # Check that forecast_signals was included in compact output
        assert mock_write.called
        compact = mock_write.call_args[0][1]
        assert "forecast_signals" in compact
        assert "XAG-USD" in compact["forecast_signals"]


# ---------------------------------------------------------------------------
# TestProphecyInSummary
# ---------------------------------------------------------------------------

class TestProphecyInSummary:
    """Tests for prophecy section in compact summary."""

    def test_prophecy_propagated_to_compact(self):
        """Prophecy section is propagated to compact summary."""
        from portfolio.reporting import _write_compact_summary

        summary = {
            "signals": {},
            "timeframes": {},
            "fear_greed": {},
            "prophecy": {
                "beliefs": [
                    {"id": "silver_bull_2026", "ticker": "XAG-USD",
                     "direction": "bullish", "conviction": 0.8}
                ],
                "total_active": 1,
            },
        }

        with patch("portfolio.reporting._atomic_write_json") as mock_write, \
             patch("portfolio.reporting._get_held_tickers", return_value=set()):
            _write_compact_summary(summary)

        compact = mock_write.call_args[0][1]
        assert "prophecy" in compact
        assert compact["prophecy"]["total_active"] == 1

    def test_prophecy_excluded_when_empty(self):
        """Empty prophecy (0 active beliefs) not propagated."""
        from portfolio.reporting import _write_compact_summary

        summary = {
            "signals": {},
            "timeframes": {},
            "fear_greed": {},
            # prophecy not in summary at all
        }

        with patch("portfolio.reporting._atomic_write_json") as mock_write, \
             patch("portfolio.reporting._get_held_tickers", return_value=set()):
            _write_compact_summary(summary)

        compact = mock_write.call_args[0][1]
        assert "prophecy" not in compact


# ---------------------------------------------------------------------------
# TestForecastAccuracyInSummary
# ---------------------------------------------------------------------------

class TestForecastAccuracyInSummary:
    """Tests for forecast_accuracy in compact summary."""

    def test_forecast_accuracy_propagated_to_compact(self):
        """forecast_accuracy section is propagated to compact."""
        from portfolio.reporting import _write_compact_summary

        summary = {
            "signals": {},
            "timeframes": {},
            "fear_greed": {},
            "forecast_accuracy": {
                "health": {"chronos": {"success_rate": 0.92}},
                "accuracy": {"XAG-USD": {"chronos_24h": 0.76}},
            },
        }

        with patch("portfolio.reporting._atomic_write_json") as mock_write, \
             patch("portfolio.reporting._get_held_tickers", return_value=set()):
            _write_compact_summary(summary)

        compact = mock_write.call_args[0][1]
        assert "forecast_accuracy" in compact
        assert compact["forecast_accuracy"]["health"]["chronos"]["success_rate"] == 0.92

    def test_forecast_accuracy_excluded_when_absent(self):
        """No forecast_accuracy in summary -> not in compact."""
        from portfolio.reporting import _write_compact_summary

        summary = {
            "signals": {},
            "timeframes": {},
            "fear_greed": {},
        }

        with patch("portfolio.reporting._atomic_write_json") as mock_write, \
             patch("portfolio.reporting._get_held_tickers", return_value=set()):
            _write_compact_summary(summary)

        compact = mock_write.call_args[0][1]
        assert "forecast_accuracy" not in compact


# ---------------------------------------------------------------------------
# TestSignalEngineIndicators
# ---------------------------------------------------------------------------

class TestSignalEngineIndicators:
    """Tests for signal_engine storing indicators in extra_info."""

    def test_enhanced_signal_indicators_stored(self):
        """Enhanced signal indicators dict is stored in extra_info when present."""
        from portfolio.signal_engine import generate_signal
        import pandas as pd
        import numpy as np

        # Create a minimal DF and indicators
        n = 50
        df = pd.DataFrame({
            "open": np.random.uniform(90, 110, n),
            "high": np.random.uniform(100, 120, n),
            "low": np.random.uniform(80, 100, n),
            "close": np.random.uniform(90, 110, n),
            "volume": np.random.uniform(1000, 5000, n),
        })
        ind = {
            "rsi": 50.0,
            "macd_hist": 0.0,
            "macd_hist_prev": 0.0,
            "ema9": 100.0,
            "ema21": 100.0,
            "price_vs_bb": "inside",
            "close": 100.0,
            "atr": 2.0,
            "atr_pct": 2.0,
        }

        # Mock the enhanced signal registry to return a single signal with indicators
        fake_result = {
            "action": "BUY",
            "confidence": 0.6,
            "sub_signals": {"sub1": "BUY"},
            "indicators": {"test_indicator": 42, "another": True},
        }
        fake_entry = {
            "module": "portfolio.signals.forecast",
            "function": "compute_forecast_signal",
            "requires_context": True,
        }

        with patch("portfolio.signal_engine.get_enhanced_signals",
                    return_value={"test_sig": fake_entry}), \
             patch("portfolio.signal_engine.load_signal_func",
                    return_value=lambda df, context=None: fake_result), \
             patch("portfolio.signal_engine._cached", side_effect=lambda k, *a, **kw: None):
            action, conf, extra = generate_signal(
                ind, ticker="BTC-USD", config={}, df=df
            )

        assert "test_sig_indicators" in extra
        assert extra["test_sig_indicators"]["test_indicator"] == 42
        assert extra["test_sig_indicators"]["another"] is True

    def test_enhanced_signal_no_indicators_no_key(self):
        """Enhanced signal without indicators dict -> no _indicators key in extra."""
        from portfolio.signal_engine import generate_signal
        import pandas as pd
        import numpy as np

        n = 50
        df = pd.DataFrame({
            "open": np.random.uniform(90, 110, n),
            "high": np.random.uniform(100, 120, n),
            "low": np.random.uniform(80, 100, n),
            "close": np.random.uniform(90, 110, n),
            "volume": np.random.uniform(1000, 5000, n),
        })
        ind = {
            "rsi": 50.0,
            "macd_hist": 0.0,
            "macd_hist_prev": 0.0,
            "ema9": 100.0,
            "ema21": 100.0,
            "price_vs_bb": "inside",
            "close": 100.0,
            "atr": 2.0,
            "atr_pct": 2.0,
        }

        # Result without indicators key
        fake_result = {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
        }
        fake_entry = {
            "module": "portfolio.signals.trend",
            "function": "compute_trend_signal",
        }

        with patch("portfolio.signal_engine.get_enhanced_signals",
                    return_value={"trend": fake_entry}), \
             patch("portfolio.signal_engine.load_signal_func",
                    return_value=lambda df: fake_result), \
             patch("portfolio.signal_engine._cached", side_effect=lambda k, *a, **kw: None):
            action, conf, extra = generate_signal(
                ind, ticker="BTC-USD", config={}, df=df
            )

        assert "trend_indicators" not in extra


# ---------------------------------------------------------------------------
# TestNewsEventThesisIntegration
# ---------------------------------------------------------------------------

class TestNewsEventThesisIntegration:
    """Tests for thesis_alignment integration in compute_news_event_signal."""

    def test_thesis_alignment_in_sub_signals(self):
        """thesis_alignment is included in sub_signals dict."""
        from portfolio.signals.news_event import compute_news_event_signal
        import pandas as pd
        import numpy as np

        df = pd.DataFrame({"close": [100] * 30})
        result = compute_news_event_signal(df, context=None)
        assert "thesis_alignment" in result["sub_signals"]
        assert result["sub_signals"]["thesis_alignment"] == "HOLD"

    def test_thesis_alignment_included_in_vote_when_enabled(self):
        """When thesis alignment is enabled and votes, it's included in majority."""
        from portfolio.signals.news_event import compute_news_event_signal
        import pandas as pd

        df = pd.DataFrame({"close": [100] * 30})
        context = {
            "ticker": "XAG-USD",
            "config": {"prophecy": {"news_alignment": True}},
        }

        belief = {"id": "silver_bull", "direction": "bullish", "conviction": 0.8}

        # Use keywords from news_keywords.py that register as non-"normal" severity
        # AND match positive keywords in _thesis_alignment_vote
        headlines = [
            {"title": "Analysts upgrade silver to strong buy target"},
            {"title": "Silver ETF approved amid earnings beat for miners"},
            {"title": "Major buyback program announced for silver producers"},
        ]

        with patch("portfolio.signals.news_event._fetch_headlines", return_value=headlines), \
             patch("portfolio.prophecy.get_active_beliefs", return_value=[belief]):
            result = compute_news_event_signal(df, context=context)

        assert result["sub_signals"]["thesis_alignment"] == "BUY"
        assert result["indicators"].get("thesis_enabled") is True
        assert result["indicators"].get("thesis_alignment") == "confirmed"
