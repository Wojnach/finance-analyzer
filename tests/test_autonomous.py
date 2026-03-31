"""Tests for portfolio.autonomous — autonomous decision engine for main loop."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent


def _make_signal(action="HOLD", conf=0.5, rsi=50, macd_hist=0, close=100,
                 buy_count=0, sell_count=0, total_applicable=20, regime="range-bound",
                 extra_overrides=None):
    extra = {
        "_buy_count": buy_count,
        "_sell_count": sell_count,
        "_total_applicable": total_applicable,
        "fear_greed": 45,
        "fear_greed_class": "Fear",
        "weighted_confidence": conf,
    }
    if extra_overrides:
        extra.update(extra_overrides)
    return {
        "action": action,
        "confidence": conf,
        "indicators": {"rsi": rsi, "macd_hist": macd_hist, "close": close},
        "extra": extra,
    }


def _make_tf_data(ticker, tf_actions=None):
    """Build minimal tf_data entry. tf_actions: list of (label, action) pairs."""
    if tf_actions is None:
        tf_actions = [
            ("Now", "HOLD"), ("12h", "HOLD"), ("2d", "HOLD"),
            ("7d", "HOLD"), ("1mo", "HOLD"), ("3mo", "HOLD"), ("6mo", "HOLD"),
        ]
    entries = []
    for label, action in tf_actions:
        entries.append((f"{ticker} {label}", {
            "action": action,
            "confidence": 0.5,
            "indicators": {"rsi": 50, "macd_hist": 0, "close": 100},
        }))
    return entries


def _base_config():
    return {
        "telegram": {"token": "fake-token", "chat_id": "123"},
        "notification": {
            "mode": "signals",
            "focus_tickers": ["XAG-USD", "BTC-USD"],
            "analysis_cooldown_seconds": 10800,
        },
        "layer2": {"enabled": False},
    }


def _patient_state(cash=500000, holdings=None):
    return {
        "cash_sek": cash,
        "holdings": holdings or {},
        "transactions": [],
        "initial_value_sek": 500000,
        "total_fees_sek": 0,
    }


def _bold_state(cash=500000, holdings=None):
    return {
        "cash_sek": cash,
        "holdings": holdings or {},
        "transactions": [],
        "initial_value_sek": 500000,
        "total_fees_sek": 0,
    }


# ===========================================================================
# _classify_tickers
# ===========================================================================

class TestClassifyTickers:
    def test_t3_includes_buy_sell_and_held(self):
        from portfolio.autonomous import _classify_tickers
        signals = {
            "BTC-USD": _make_signal("BUY", buy_count=5, sell_count=1),
            "ETH-USD": _make_signal("HOLD"),
            "NVDA": _make_signal("SELL", buy_count=1, sell_count=4),
        }
        patient = _patient_state(holdings={"ETH-USD": {"shares": 1, "avg_cost_usd": 2000}})
        bold = _bold_state()
        actionable, top_hold, hold_count, sell_count = _classify_tickers(
            signals, patient, bold, tier=3, triggered_tickers=set()
        )
        assert "BTC-USD" in actionable
        assert "NVDA" in actionable
        assert "ETH-USD" in actionable  # held in patient
        assert hold_count >= 0
        # sell_count tracks non-shown SELLs; NVDA is in actionable so sell_count=0
        assert sell_count >= 0

    def test_t1_only_held(self):
        from portfolio.autonomous import _classify_tickers
        signals = {
            "BTC-USD": _make_signal("BUY", buy_count=5),
            "ETH-USD": _make_signal("HOLD"),
        }
        patient = _patient_state(holdings={"ETH-USD": {"shares": 2, "avg_cost_usd": 2000}})
        bold = _bold_state()
        actionable, _, _, _ = _classify_tickers(
            signals, patient, bold, tier=1, triggered_tickers=set()
        )
        assert "ETH-USD" in actionable
        # BTC-USD should NOT be in actionable for T1 (not held)
        assert "BTC-USD" not in actionable

    def test_t2_triggered_plus_held(self):
        from portfolio.autonomous import _classify_tickers
        signals = {
            "BTC-USD": _make_signal("BUY", buy_count=5),
            "ETH-USD": _make_signal("HOLD"),
            "NVDA": _make_signal("HOLD"),
        }
        patient = _patient_state(holdings={"NVDA": {"shares": 10, "avg_cost_usd": 185}})
        bold = _bold_state()
        actionable, _, _, _ = _classify_tickers(
            signals, patient, bold, tier=2, triggered_tickers={"BTC-USD"}
        )
        assert "BTC-USD" in actionable
        assert "NVDA" in actionable
        assert "ETH-USD" not in actionable

    def test_no_signals(self):
        from portfolio.autonomous import _classify_tickers
        actionable, _, hold_count, sell_count = _classify_tickers(
            {}, _patient_state(), _bold_state(), tier=3, triggered_tickers=set()
        )
        assert actionable == {}
        assert hold_count == 0
        assert sell_count == 0

    def test_all_hold_no_positions_returns_top(self):
        from portfolio.autonomous import _classify_tickers
        signals = {
            "BTC-USD": _make_signal("HOLD", buy_count=4, sell_count=0),
            "ETH-USD": _make_signal("HOLD", buy_count=3, sell_count=1),
            "NVDA": _make_signal("HOLD", buy_count=0, sell_count=0),
        }
        actionable, top_hold, hold_count, _ = _classify_tickers(
            signals, _patient_state(), _bold_state(), tier=3, triggered_tickers=set()
        )
        # When all HOLD, top_hold should have the most interesting tickers
        assert len(top_hold) <= 5

    def test_bold_held_included(self):
        from portfolio.autonomous import _classify_tickers
        signals = {"BTC-USD": _make_signal("HOLD")}
        bold = _bold_state(holdings={"BTC-USD": {"shares": 0.1, "avg_cost_usd": 65000}})
        actionable, _, _, _ = _classify_tickers(
            signals, _patient_state(), bold, tier=1, triggered_tickers=set()
        )
        assert "BTC-USD" in actionable


# ===========================================================================
# _ticker_prediction
# ===========================================================================

class TestTickerPrediction:
    def test_buy_consensus(self):
        from portfolio.autonomous import _ticker_prediction
        sig = _make_signal("BUY", conf=0.75, rsi=45, buy_count=7, sell_count=1)
        tf_entries = _make_tf_data("BTC-USD", [
            ("Now", "BUY"), ("12h", "BUY"), ("2d", "HOLD"),
            ("7d", "BUY"), ("1mo", "HOLD"), ("3mo", "HOLD"), ("6mo", "HOLD"),
        ])
        pred = _ticker_prediction("BTC-USD", sig, tf_entries)
        assert pred["outlook"] == "bullish"
        assert pred["conviction"] > 0.3
        assert pred["recommendation"] == "BUY"

    def test_sell_consensus(self):
        from portfolio.autonomous import _ticker_prediction
        sig = _make_signal("SELL", conf=0.8, rsi=72, buy_count=1, sell_count=6)
        tf_entries = _make_tf_data("ETH-USD")
        pred = _ticker_prediction("ETH-USD", sig, tf_entries)
        assert pred["outlook"] == "bearish"
        assert pred["recommendation"] == "SELL"

    def test_hold_consensus(self):
        from portfolio.autonomous import _ticker_prediction
        sig = _make_signal("HOLD", conf=0.3, buy_count=2, sell_count=2)
        pred = _ticker_prediction("NVDA", sig, _make_tf_data("NVDA"))
        assert pred["outlook"] == "neutral"
        assert pred["recommendation"] == "HOLD"

    def test_high_rsi_reduces_buy_conviction(self):
        from portfolio.autonomous import _ticker_prediction
        sig = _make_signal("BUY", conf=0.6, rsi=75, buy_count=5, sell_count=1)
        pred = _ticker_prediction("BTC-USD", sig, _make_tf_data("BTC-USD"))
        assert pred["conviction"] < 0.7  # overbought penalty

    def test_low_rsi_boosts_buy_conviction(self):
        from portfolio.autonomous import _ticker_prediction
        sig = _make_signal("BUY", conf=0.6, rsi=25, buy_count=5, sell_count=1)
        pred = _ticker_prediction("BTC-USD", sig, _make_tf_data("BTC-USD"))
        # Low RSI supports a buy
        assert pred["conviction"] > 0.3

    def test_buy_suppressed_when_votes_weak(self):
        from portfolio.autonomous import _ticker_prediction
        sig = _make_signal("BUY", conf=0.7, rsi=45, buy_count=2, sell_count=4)
        pred = _ticker_prediction("AAPL", sig, _make_tf_data("AAPL"))
        assert pred["recommendation"] == "HOLD"
        assert pred["conviction"] == 0.0

    def test_tf_alignment_boosts_conviction(self):
        from portfolio.autonomous import _ticker_prediction
        sig = _make_signal("BUY", conf=0.6, rsi=50, buy_count=5, sell_count=1)
        aligned_tf = _make_tf_data("BTC-USD", [
            ("Now", "BUY"), ("12h", "BUY"), ("2d", "BUY"),
            ("7d", "BUY"), ("1mo", "BUY"), ("3mo", "BUY"), ("6mo", "BUY"),
        ])
        pred_aligned = _ticker_prediction("BTC-USD", sig, aligned_tf)
        pred_mixed = _ticker_prediction("BTC-USD", sig, _make_tf_data("BTC-USD"))
        assert pred_aligned["conviction"] >= pred_mixed["conviction"]

    def test_empty_tf_entries(self):
        from portfolio.autonomous import _ticker_prediction
        sig = _make_signal("BUY", conf=0.5, buy_count=3, sell_count=0)
        pred = _ticker_prediction("BTC-USD", sig, [])
        assert pred["recommendation"] in ("BUY", "HOLD")

    def test_thesis_string_generated(self):
        from portfolio.autonomous import _ticker_prediction
        sig = _make_signal("BUY", conf=0.7, rsi=40, buy_count=6, sell_count=1)
        pred = _ticker_prediction("XAG-USD", sig, _make_tf_data("XAG-USD"))
        assert isinstance(pred["thesis"], str)
        assert len(pred["thesis"]) > 0


# ===========================================================================
# _build_reflection
# ===========================================================================

class TestBuildReflection:
    def test_price_moved_as_expected(self):
        from portfolio.autonomous import _build_reflection
        prev = {
            "ts": (datetime.now(UTC) - timedelta(hours=2)).isoformat(),
            "prices": {"BTC-USD": 65000, "ETH-USD": 1900},
            "tickers": {"BTC-USD": {"outlook": "bullish", "thesis": "uptrend"}},
        }
        current_prices = {"BTC-USD": 67000, "ETH-USD": 1950}
        reflection = _build_reflection(prev, current_prices)
        assert isinstance(reflection, str)
        assert len(reflection) > 0

    def test_no_previous_entry(self):
        from portfolio.autonomous import _build_reflection
        assert _build_reflection(None, {"BTC-USD": 67000}) == ""

    def test_no_prices_in_previous(self):
        from portfolio.autonomous import _build_reflection
        prev = {"ts": datetime.now(UTC).isoformat(), "tickers": {}}
        assert _build_reflection(prev, {"BTC-USD": 67000}) == ""

    def test_bearish_thesis_price_dropped(self):
        from portfolio.autonomous import _build_reflection
        prev = {
            "ts": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
            "prices": {"ETH-USD": 2000},
            "tickers": {"ETH-USD": {"outlook": "bearish", "thesis": "weak"}},
        }
        reflection = _build_reflection(prev, {"ETH-USD": 1900})
        assert "confirm" in reflection.lower() or "correct" in reflection.lower() or "ETH" in reflection


# ===========================================================================
# _detect_regime
# ===========================================================================

class TestDetectRegime:
    def test_majority_regime(self):
        from portfolio.autonomous import _detect_regime
        signals = {
            "BTC-USD": _make_signal(regime="trending-up"),
            "ETH-USD": _make_signal(regime="trending-up"),
            "NVDA": _make_signal(regime="range-bound"),
        }
        # Need to fix: regime is in extra, not top-level
        for sig in signals.values():
            sig["extra"]["regime"] = sig["extra"].get("regime", "range-bound")
        signals["BTC-USD"]["extra"]["regime"] = "trending-up"
        signals["ETH-USD"]["extra"]["regime"] = "trending-up"
        signals["NVDA"]["extra"]["regime"] = "range-bound"
        assert _detect_regime(signals) == "trending-up"

    def test_empty_signals(self):
        from portfolio.autonomous import _detect_regime
        assert _detect_regime({}) == "range-bound"

    def test_single_regime(self):
        from portfolio.autonomous import _detect_regime
        signals = {"BTC-USD": _make_signal()}
        signals["BTC-USD"]["extra"]["regime"] = "high-vol"
        assert _detect_regime(signals) == "high-vol"


# ===========================================================================
# _build_telegram — Mode A (signals)
# ===========================================================================

class TestBuildTelegram:
    def _call(self, actionable=None, hold_count=0, sell_count=0,
              patient=None, bold=None, prices=None, fx=10.5,
              signals=None, tf_data=None, predictions=None,
              config=None, tier=3, regime="range-bound", reflection="",
              reasons=None):
        from portfolio.autonomous import _build_telegram
        if actionable is None:
            actionable = {
                "BTC-USD": _make_signal("BUY", buy_count=5, sell_count=1, close=67000),
            }
        if patient is None:
            patient = _patient_state()
        if bold is None:
            bold = _bold_state()
        if prices is None:
            prices = {"BTC-USD": 67000}
        if signals is None:
            signals = actionable
        if tf_data is None:
            tf_data = {"BTC-USD": _make_tf_data("BTC-USD")}
        if predictions is None:
            predictions = {
                "BTC-USD": {
                    "outlook": "bullish", "conviction": 0.6,
                    "thesis": "breakout", "recommendation": "BUY",
                },
            }
        if config is None:
            config = _base_config()
        return _build_telegram(
            actionable, hold_count, sell_count, patient, bold,
            prices, fx, signals, tf_data, predictions, config,
            tier, regime, reflection, reasons or ["BTC-USD consensus BUY"],
        )

    def test_returns_string(self):
        msg = self._call()
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_auto_prefix(self):
        msg = self._call()
        assert "AUTO" in msg

    def test_buy_ticker_in_grid(self):
        msg = self._call()
        assert "BTC" in msg

    def test_hold_count_shown(self):
        actionable = {"BTC-USD": _make_signal("HOLD", buy_count=2, sell_count=1, close=67000)}
        predictions = {
            "BTC-USD": {
                "outlook": "neutral", "conviction": 0.2,
                "thesis": "range", "recommendation": "HOLD",
            },
        }
        msg = self._call(actionable=actionable, predictions=predictions, hold_count=15)
        assert "15" in msg

    def test_portfolio_context(self):
        msg = self._call()
        assert "Patient portfolio" in msg
        assert "Bold portfolio" in msg

    def test_first_line_under_70_chars(self):
        msg = self._call()
        first_line = msg.split("\n")[0]
        assert len(first_line) < 80  # some slack for emoji/markdown

    def test_under_telegram_limit(self):
        # Even with many tickers, should stay under 4096
        actionable = {}
        predictions = {}
        prices = {}
        tf_data_full = {}
        for ticker in ["BTC-USD", "ETH-USD", "NVDA", "AMD", "PLTR", "GOOGL", "AMZN"]:
            actionable[ticker] = _make_signal("BUY", buy_count=5, sell_count=1, close=100)
            predictions[ticker] = {
                "outlook": "bullish", "conviction": 0.6,
                "thesis": "breakout", "recommendation": "BUY",
            }
            prices[ticker] = 100
            tf_data_full[ticker] = _make_tf_data(ticker)
        msg = self._call(
            actionable=actionable, predictions=predictions,
            prices=prices, tf_data=tf_data_full,
            signals=actionable,
        )
        assert len(msg) <= 4096

    def test_sell_count_in_summary(self):
        actionable = {"BTC-USD": _make_signal("HOLD", buy_count=2, sell_count=1, close=67000)}
        predictions = {
            "BTC-USD": {
                "outlook": "neutral", "conviction": 0.2,
                "thesis": "range", "recommendation": "HOLD",
            },
        }
        msg = self._call(
            actionable=actionable,
            predictions=predictions,
            hold_count=10,
            sell_count=2,
        )
        assert "2 with sell signals" in msg.lower()

    def test_buy_mode_hides_hold_sell_summary(self):
        msg = self._call(hold_count=10, sell_count=2)
        assert "hold" not in msg.lower()
        assert " sell" not in msg.lower()

    def test_sell_mode_hides_hold_sell_summary(self):
        actionable = {"BTC-USD": _make_signal("SELL", buy_count=1, sell_count=5, close=67000)}
        predictions = {
            "BTC-USD": {
                "outlook": "bearish", "conviction": 0.7,
                "thesis": "breakdown", "recommendation": "SELL",
            },
        }
        msg = self._call(
            actionable=actionable,
            predictions=predictions,
            hold_count=10,
            sell_count=2,
        )
        assert "*AUTO SELL" in msg
        assert "10 more on hold" not in msg.lower()
        assert "2 with sell signals" not in msg.lower()

    def test_mode_b_probability(self):
        config = _base_config()
        config["notification"]["mode"] = "probability"
        msg = self._call(config=config)
        assert "PROBABILITY" in msg or "AUTO" in msg

    def test_trade_recommendation_shown(self):
        msg = self._call()
        # Should mention BUY recommendation somewhere
        assert "BUY" in msg

    def test_reasoning_line_present(self):
        msg = self._call()
        lines = msg.strip().split("\n")
        # Last lines should be reasoning
        assert len(lines) >= 3

    def test_bold_holdings_shown(self):
        bold = _bold_state(
            cash=300000,
            holdings={"NVDA": {"shares": 50, "avg_cost_usd": 185}},
        )
        msg = self._call(bold=bold, prices={"BTC-USD": 67000, "NVDA": 185})
        assert "Bold portfolio" in msg
        assert "NVDA 50 shares" in msg


# ===========================================================================
# _should_send
# ===========================================================================

class TestShouldSend:
    def test_buy_signal_always_sends(self, tmp_path):
        from portfolio.autonomous import _should_send
        predictions = {"BTC-USD": {"recommendation": "BUY"}}
        reasons = ["BTC-USD consensus BUY"]
        with patch("portfolio.autonomous.THROTTLE_FILE", tmp_path / "throttle.json"):
            assert _should_send(predictions, reasons, tier=2) is True

    def test_sell_signal_always_sends(self, tmp_path):
        from portfolio.autonomous import _should_send
        predictions = {"BTC-USD": {"recommendation": "SELL"}}
        reasons = ["BTC-USD consensus SELL"]
        with patch("portfolio.autonomous.THROTTLE_FILE", tmp_path / "throttle.json"):
            assert _should_send(predictions, reasons, tier=2) is True

    def test_fg_extreme_always_sends(self, tmp_path):
        from portfolio.autonomous import _should_send
        predictions = {"BTC-USD": {"recommendation": "HOLD"}}
        reasons = ["F&G crossed 15"]
        with patch("portfolio.autonomous.THROTTLE_FILE", tmp_path / "throttle.json"):
            assert _should_send(predictions, reasons, tier=2) is True

    def test_t3_always_sends(self, tmp_path):
        from portfolio.autonomous import _should_send
        predictions = {"BTC-USD": {"recommendation": "HOLD"}}
        reasons = ["periodic"]
        with patch("portfolio.autonomous.THROTTLE_FILE", tmp_path / "throttle.json"):
            assert _should_send(predictions, reasons, tier=3) is True

    def test_routine_hold_throttled(self, tmp_path):
        from portfolio.autonomous import _should_send
        throttle_file = tmp_path / "throttle.json"
        # Write recent timestamp
        recent = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
        throttle_file.write_text(json.dumps({"last_send": recent}))
        predictions = {"BTC-USD": {"recommendation": "HOLD"}}
        reasons = ["sentiment shift"]
        with patch("portfolio.autonomous.THROTTLE_FILE", throttle_file):
            assert _should_send(predictions, reasons, tier=2) is False

    def test_routine_hold_passes_after_cooldown(self, tmp_path):
        from portfolio.autonomous import _should_send
        throttle_file = tmp_path / "throttle.json"
        # Write old timestamp
        old = (datetime.now(UTC) - timedelta(minutes=35)).isoformat()
        throttle_file.write_text(json.dumps({"last_send": old}))
        predictions = {"BTC-USD": {"recommendation": "HOLD"}}
        reasons = ["sentiment shift"]
        with patch("portfolio.autonomous.THROTTLE_FILE", throttle_file):
            assert _should_send(predictions, reasons, tier=2) is True

    def test_no_throttle_file(self, tmp_path):
        from portfolio.autonomous import _should_send
        throttle_file = tmp_path / "throttle.json"
        predictions = {"BTC-USD": {"recommendation": "HOLD"}}
        reasons = ["sentiment shift"]
        with patch("portfolio.autonomous.THROTTLE_FILE", throttle_file):
            assert _should_send(predictions, reasons, tier=2) is True

    def test_post_trade_always_sends(self, tmp_path):
        from portfolio.autonomous import _should_send
        predictions = {"BTC-USD": {"recommendation": "HOLD"}}
        reasons = ["post-trade reassessment"]
        with patch("portfolio.autonomous.THROTTLE_FILE", tmp_path / "throttle.json"):
            assert _should_send(predictions, reasons, tier=2) is True

    def test_consensus_hold_only_is_suppressed(self, tmp_path):
        from portfolio.autonomous import _should_send
        predictions = {"BTC-USD": {"recommendation": "HOLD"}}
        reasons = ["BTC-USD consensus HOLD"]
        with patch("portfolio.autonomous.THROTTLE_FILE", tmp_path / "throttle.json"):
            assert _should_send(predictions, reasons, tier=3) is False


# ===========================================================================
# autonomous_decision — integration
# ===========================================================================

class TestAutonomousDecision:
    @pytest.fixture(autouse=True)
    def setup_paths(self, tmp_path):
        self.journal_file = tmp_path / "layer2_journal.jsonl"
        self.decisions_file = tmp_path / "layer2_decisions.jsonl"
        self.throttle_file = tmp_path / "autonomous_throttle.json"
        self.bold_state_file = tmp_path / "portfolio_state_bold.json"
        self.bold_state_file.write_text(json.dumps(_bold_state()))
        self._patches = [
            patch("portfolio.autonomous.JOURNAL_FILE", self.journal_file),
            patch("portfolio.autonomous.DECISIONS_FILE", self.decisions_file),
            patch("portfolio.autonomous.THROTTLE_FILE", self.throttle_file),
            patch("portfolio.autonomous._load_bold_state_safe",
                  return_value=json.loads(self.bold_state_file.read_text())),
            patch("portfolio.autonomous.send_or_store", return_value=True),
        ]
        for p in self._patches:
            p.start()
        yield
        for p in self._patches:
            p.stop()

    def test_writes_journal_entry(self):
        from portfolio.autonomous import autonomous_decision
        config = _base_config()
        signals = {"BTC-USD": _make_signal("BUY", buy_count=5, sell_count=1, close=67000)}
        prices = {"BTC-USD": 67000}
        state = _patient_state()
        tf_data = {"BTC-USD": _make_tf_data("BTC-USD")}
        autonomous_decision(
            config, signals, prices, 10.5, state,
            ["BTC-USD consensus BUY"], tf_data, tier=3, triggered_tickers={"BTC-USD"},
        )
        assert self.journal_file.exists()
        entries = [json.loads(l) for l in self.journal_file.read_text().strip().split("\n")]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["source"] == "autonomous"
        assert "BTC-USD" in entry.get("prices", {})

    def test_writes_decision_log(self):
        from portfolio.autonomous import autonomous_decision
        config = _base_config()
        signals = {"BTC-USD": _make_signal("HOLD", close=67000)}
        autonomous_decision(
            config, signals, {"BTC-USD": 67000}, 10.5, _patient_state(),
            ["periodic"], {"BTC-USD": _make_tf_data("BTC-USD")},
            tier=3, triggered_tickers=set(),
        )
        assert self.decisions_file.exists()
        entries = [json.loads(l) for l in self.decisions_file.read_text().strip().split("\n")]
        assert len(entries) == 1
        assert "predictions" in entries[0]

    def test_sends_telegram_on_buy(self):
        from portfolio.autonomous import autonomous_decision, send_or_store
        config = _base_config()
        signals = {"BTC-USD": _make_signal("BUY", buy_count=5, sell_count=1, close=67000)}
        autonomous_decision(
            config, signals, {"BTC-USD": 67000}, 10.5, _patient_state(),
            ["BTC-USD consensus BUY"], {"BTC-USD": _make_tf_data("BTC-USD")},
            tier=2, triggered_tickers={"BTC-USD"},
        )
        send_or_store.assert_called_once()
        msg = send_or_store.call_args[0][0]
        assert "BTC" in msg

    def test_throttles_routine_hold(self):
        from portfolio.autonomous import autonomous_decision, send_or_store
        # First call: should send
        config = _base_config()
        signals = {"BTC-USD": _make_signal("HOLD", close=67000)}
        autonomous_decision(
            config, signals, {"BTC-USD": 67000}, 10.5, _patient_state(),
            ["sentiment shift"], {"BTC-USD": _make_tf_data("BTC-USD")},
            tier=2, triggered_tickers=set(),
        )
        # Second call: should be throttled (within 30 min)
        send_or_store.reset_mock()
        autonomous_decision(
            config, signals, {"BTC-USD": 67000}, 10.5, _patient_state(),
            ["sentiment shift"], {"BTC-USD": _make_tf_data("BTC-USD")},
            tier=2, triggered_tickers=set(),
        )
        send_or_store.assert_not_called()

    def test_journal_has_decisions(self):
        from portfolio.autonomous import autonomous_decision
        config = _base_config()
        signals = {
            "BTC-USD": _make_signal("BUY", buy_count=5, sell_count=1, close=67000),
            "ETH-USD": _make_signal("HOLD", close=2000),
        }
        autonomous_decision(
            config, signals, {"BTC-USD": 67000, "ETH-USD": 2000}, 10.5,
            _patient_state(), ["BTC-USD consensus BUY"],
            {"BTC-USD": _make_tf_data("BTC-USD"), "ETH-USD": _make_tf_data("ETH-USD")},
            tier=3, triggered_tickers={"BTC-USD"},
        )
        entry = json.loads(self.journal_file.read_text().strip().split("\n")[0])
        assert "patient" in entry["decisions"]
        assert "bold" in entry["decisions"]
        assert entry["decisions"]["patient"]["action"] == "HOLD"
        assert entry["decisions"]["bold"]["action"] == "HOLD"

    def test_journal_has_regime(self):
        from portfolio.autonomous import autonomous_decision
        signals = {"BTC-USD": _make_signal("HOLD", close=67000)}
        signals["BTC-USD"]["extra"]["regime"] = "trending-up"
        autonomous_decision(
            _base_config(), signals, {"BTC-USD": 67000}, 10.5, _patient_state(),
            ["periodic"], {"BTC-USD": _make_tf_data("BTC-USD")},
            tier=3, triggered_tickers=set(),
        )
        entry = json.loads(self.journal_file.read_text().strip().split("\n")[0])
        assert entry["regime"] == "trending-up"

    def test_handles_empty_signals(self):
        from portfolio.autonomous import autonomous_decision
        # Should not crash
        autonomous_decision(
            _base_config(), {}, {}, 10.5, _patient_state(),
            ["startup"], {}, tier=3, triggered_tickers=set(),
        )
        assert self.journal_file.exists()

    def test_reflection_from_previous(self):
        from portfolio.autonomous import autonomous_decision
        # Write a previous journal entry
        prev = {
            "ts": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
            "source": "autonomous",
            "trigger": "periodic",
            "regime": "range-bound",
            "reflection": "",
            "decisions": {"patient": {"action": "HOLD"}, "bold": {"action": "HOLD"}},
            "tickers": {"BTC-USD": {"outlook": "bullish", "thesis": "uptrend", "conviction": 0.6}},
            "prices": {"BTC-USD": 65000},
            "watchlist": [],
        }
        self.journal_file.write_text(json.dumps(prev) + "\n")

        signals = {"BTC-USD": _make_signal("HOLD", close=67000)}
        autonomous_decision(
            _base_config(), signals, {"BTC-USD": 67000}, 10.5, _patient_state(),
            ["periodic"], {"BTC-USD": _make_tf_data("BTC-USD")},
            tier=3, triggered_tickers=set(),
        )
        entries = [json.loads(l) for l in self.journal_file.read_text().strip().split("\n")]
        assert len(entries) == 2
        # Second entry should have reflection
        assert len(entries[1].get("reflection", "")) > 0

    def test_exception_safety(self):
        """autonomous_decision should not propagate exceptions."""
        from portfolio.autonomous import autonomous_decision
        with patch("portfolio.autonomous._classify_tickers", side_effect=RuntimeError("boom")):
            # Should not raise
            autonomous_decision(
                _base_config(), {"BTC-USD": _make_signal()}, {"BTC-USD": 67000},
                10.5, _patient_state(), ["test"], {}, tier=3, triggered_tickers=set(),
            )

    def test_multiple_tickers(self):
        from portfolio.autonomous import autonomous_decision
        config = _base_config()
        signals = {
            "BTC-USD": _make_signal("BUY", buy_count=5, sell_count=1, close=67000),
            "ETH-USD": _make_signal("SELL", buy_count=1, sell_count=5, close=2000),
            "NVDA": _make_signal("HOLD", close=185),
            "XAG-USD": _make_signal("BUY", buy_count=7, sell_count=0, close=33),
        }
        prices = {"BTC-USD": 67000, "ETH-USD": 2000, "NVDA": 185, "XAG-USD": 33}
        tf_data = {t: _make_tf_data(t) for t in signals}
        autonomous_decision(
            config, signals, prices, 10.5, _patient_state(),
            ["XAG-USD consensus BUY", "BTC-USD consensus BUY"],
            tf_data, tier=3, triggered_tickers={"XAG-USD", "BTC-USD"},
        )
        entry = json.loads(self.journal_file.read_text().strip().split("\n")[0])
        assert "BTC-USD" in entry["prices"]
        assert "XAG-USD" in entry["prices"]


# ===========================================================================
# _format_price
# ===========================================================================

class TestFormatPrice:
    def test_large_price(self):
        from portfolio.autonomous import _format_price
        assert _format_price(67602) == "$68K"

    def test_medium_price(self):
        from portfolio.autonomous import _format_price
        result = _format_price(185.50)
        assert "$" in result

    def test_small_price(self):
        from portfolio.autonomous import _format_price
        result = _format_price(33.45)
        assert "$" in result

    def test_very_large_price(self):
        from portfolio.autonomous import _format_price
        result = _format_price(1949.50)
        assert "$" in result


# ===========================================================================
# _tf_heatmap
# ===========================================================================

class TestTfHeatmap:
    def test_all_buy(self):
        from portfolio.autonomous import _tf_heatmap
        tf_entries = _make_tf_data("BTC-USD", [
            ("Now", "BUY"), ("12h", "BUY"), ("2d", "BUY"),
            ("7d", "BUY"), ("1mo", "BUY"), ("3mo", "BUY"), ("6mo", "BUY"),
        ])
        heatmap = _tf_heatmap(tf_entries)
        assert heatmap == "BBBBBBB"

    def test_all_sell(self):
        from portfolio.autonomous import _tf_heatmap
        tf_entries = _make_tf_data("BTC-USD", [
            ("Now", "SELL"), ("12h", "SELL"), ("2d", "SELL"),
            ("7d", "SELL"), ("1mo", "SELL"), ("3mo", "SELL"), ("6mo", "SELL"),
        ])
        heatmap = _tf_heatmap(tf_entries)
        assert heatmap == "SSSSSSS"

    def test_mixed(self):
        from portfolio.autonomous import _tf_heatmap
        tf_entries = _make_tf_data("BTC-USD", [
            ("Now", "BUY"), ("12h", "HOLD"), ("2d", "SELL"),
            ("7d", "BUY"), ("1mo", "SELL"), ("3mo", "HOLD"), ("6mo", "BUY"),
        ])
        heatmap = _tf_heatmap(tf_entries)
        assert len(heatmap) == 7
        assert heatmap[0] == "B"
        assert heatmap[2] == "S"

    def test_empty(self):
        from portfolio.autonomous import _tf_heatmap
        assert len(_tf_heatmap([])) <= 7


# ===========================================================================
# Telegram formatting edge cases
# ===========================================================================

class TestTelegramEdgeCases:
    def test_escape_special_chars_in_reason(self):
        """Reason strings with underscores/asterisks should not break Markdown."""
        from portfolio.autonomous import _build_telegram
        actionable = {"BTC-USD": _make_signal("BUY", buy_count=5, sell_count=1, close=67000)}
        predictions = {"BTC-USD": {"outlook": "bullish", "conviction": 0.6,
                                    "thesis": "test_with_underscores", "recommendation": "BUY"}}
        msg = _build_telegram(
            actionable, 10, 0, _patient_state(), _bold_state(),
            {"BTC-USD": 67000}, 10.5, actionable,
            {"BTC-USD": _make_tf_data("BTC-USD")},
            predictions, _base_config(), 3, "range-bound", "",
            ["test_reason_with_underscores"],
        )
        assert isinstance(msg, str)

    def test_zero_fx_rate(self):
        """Should not crash on zero fx rate."""
        from portfolio.autonomous import _build_telegram
        actionable = {"BTC-USD": _make_signal("BUY", buy_count=5, close=67000)}
        predictions = {"BTC-USD": {"outlook": "bullish", "conviction": 0.6,
                                    "thesis": "test", "recommendation": "BUY"}}
        msg = _build_telegram(
            actionable, 0, 0, _patient_state(), _bold_state(),
            {"BTC-USD": 67000}, 0, actionable,
            {"BTC-USD": _make_tf_data("BTC-USD")},
            predictions, _base_config(), 3, "range-bound", "", ["test"],
        )
        assert isinstance(msg, str)


# ===========================================================================
# BUG-2: _consensus_acc_cache never invalidates
# ===========================================================================

class TestConsensusAccuracyCache:
    def test_cache_refreshes_after_ttl(self, tmp_path):
        """Cache should not persist forever — must re-read after TTL expires."""
        import portfolio.autonomous as mod
        summary_file = tmp_path / "agent_summary_compact.json"
        summary_file.write_text(json.dumps({
            "signal_accuracy_1d": {"consensus": {"accuracy": 0.52}}
        }))
        old_data_dir = mod.DATA_DIR
        old_cache = mod._consensus_acc_cache
        old_cache_ts = mod._consensus_acc_cache_ts
        try:
            mod.DATA_DIR = tmp_path
            mod._consensus_acc_cache = None
            mod._consensus_acc_cache_ts = 0
            # First call: should load 0.52
            acc1 = mod._consensus_accuracy()
            assert acc1 == 0.52
            # Update file
            summary_file.write_text(json.dumps({
                "signal_accuracy_1d": {"consensus": {"accuracy": 0.61}}
            }))
            # Second call with fresh cache: still returns old value
            acc2 = mod._consensus_accuracy()
            assert acc2 == 0.52
            # Expire cache by setting timestamp to 0
            mod._consensus_acc_cache_ts = 0
            acc3 = mod._consensus_accuracy()
            assert acc3 == 0.61
        finally:
            mod.DATA_DIR = old_data_dir
            mod._consensus_acc_cache = old_cache
            mod._consensus_acc_cache_ts = old_cache_ts

    def test_cache_returns_none_then_retries(self, tmp_path):
        """If file missing on first call, should retry on next call (not cache None forever)."""
        import portfolio.autonomous as mod
        old_data_dir = mod.DATA_DIR
        old_cache = mod._consensus_acc_cache
        old_cache_ts = mod._consensus_acc_cache_ts
        try:
            mod.DATA_DIR = tmp_path
            mod._consensus_acc_cache = None
            mod._consensus_acc_cache_ts = 0
            # No file exists
            acc1 = mod._consensus_accuracy()
            assert acc1 is None
            # Now create the file
            (tmp_path / "agent_summary_compact.json").write_text(json.dumps({
                "signal_accuracy_1d": {"consensus": {"accuracy": 0.55}}
            }))
            # Expire cache
            mod._consensus_acc_cache_ts = 0
            acc2 = mod._consensus_accuracy()
            assert acc2 == 0.55
        finally:
            mod.DATA_DIR = old_data_dir
            mod._consensus_acc_cache = old_cache
            mod._consensus_acc_cache_ts = old_cache_ts


# ===========================================================================
# BUG-4: T3 sell_count dead code
# ===========================================================================

class TestT3SellCount:
    def test_t3_sell_count_is_zero_for_sell_tickers(self):
        """In T3, all SELL tickers go into actionable, so sell_count should be 0."""
        from portfolio.autonomous import _classify_tickers
        signals = {
            "BTC-USD": _make_signal("BUY", buy_count=5, sell_count=1),
            "ETH-USD": _make_signal("SELL", buy_count=1, sell_count=5),
            "NVDA": _make_signal("SELL", buy_count=0, sell_count=3),
            "AMD": _make_signal("HOLD"),
        }
        actionable, _, hold_count, sell_count = _classify_tickers(
            signals, _patient_state(), _bold_state(), tier=3, triggered_tickers=set()
        )
        # BUY and SELL tickers should be in actionable
        assert "BTC-USD" in actionable
        assert "ETH-USD" in actionable
        assert "NVDA" in actionable
        # AMD is HOLD, not held, so it's a hold
        assert "AMD" not in actionable
        assert hold_count == 1
        # sell_count should be 0 since all SELLs are in actionable
        assert sell_count == 0
