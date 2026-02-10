"""
Unit tests for trading signal logic.
Tests the trigger + guards entry model and sell signal generation.
"""

import pytest
import pandas as pd
import numpy as np


class TestBuySignals:
    """Test trigger + guards buy signal model."""

    def test_single_trigger_enters(self):
        """A single trigger with all guards true should enter."""
        from user_data.strategies.signals import evaluate_buy_conditions

        result = evaluate_buy_conditions(
            triggers={"rsi_cross": True, "macd_hist_cross": False, "bb_touch": False},
            guards={"ema_uptrend": True, "volume_spike": True, "trend_1h": True},
        )
        assert result["should_enter"] is True
        assert result["active_triggers"] == ["rsi_cross"]
        assert result["failed_guards"] == []

    def test_no_trigger_no_entry(self):
        """No triggers firing should not enter even with all guards true."""
        from user_data.strategies.signals import evaluate_buy_conditions

        result = evaluate_buy_conditions(
            triggers={"rsi_cross": False, "macd_hist_cross": False, "bb_touch": False},
            guards={"ema_uptrend": True, "volume_spike": True, "trend_1h": True},
        )
        assert result["should_enter"] is False

    def test_trigger_with_failed_guard(self):
        """Trigger + failed guard should not enter."""
        from user_data.strategies.signals import evaluate_buy_conditions

        result = evaluate_buy_conditions(
            triggers={"rsi_cross": True, "macd_hist_cross": True, "bb_touch": False},
            guards={"ema_uptrend": True, "volume_spike": False, "trend_1h": True},
        )
        assert result["should_enter"] is False
        assert "volume_spike" in result["failed_guards"]

    def test_multiple_triggers_still_enters(self):
        """Multiple triggers firing should still enter (OR logic)."""
        from user_data.strategies.signals import evaluate_buy_conditions

        result = evaluate_buy_conditions(
            triggers={"rsi_cross": True, "macd_hist_cross": True, "bb_touch": True},
            guards={"ema_uptrend": True, "volume_spike": True, "trend_1h": True},
        )
        assert result["should_enter"] is True
        assert len(result["active_triggers"]) == 3

    def test_all_guards_must_pass(self):
        """Every guard must be true for entry."""
        from user_data.strategies.signals import evaluate_buy_conditions

        result = evaluate_buy_conditions(
            triggers={"rsi_cross": True},
            guards={"ema_uptrend": False, "volume_spike": False, "trend_1h": False},
        )
        assert result["should_enter"] is False
        assert len(result["failed_guards"]) == 3


class TestSellSignals:
    """Test sell signal generation."""

    def test_rsi_overbought_generates_sell(self):
        from user_data.strategies.signals import evaluate_sell_conditions

        conditions = {
            "rsi": 75,
            "macd_bearish_cross": False,
            "ema_bearish_cross": False,
            "volume_spike": False,
        }
        result = evaluate_sell_conditions(conditions)
        assert result["rsi_signal"] is True

    def test_macd_crossover_generates_sell(self):
        from user_data.strategies.signals import evaluate_sell_conditions

        conditions = {
            "rsi": 50,
            "macd_bearish_cross": True,
            "ema_bearish_cross": False,
            "volume_spike": False,
        }
        result = evaluate_sell_conditions(conditions)
        assert result["macd_signal"] is True

    def test_ema_crossover_generates_sell(self):
        from user_data.strategies.signals import evaluate_sell_conditions

        conditions = {
            "rsi": 50,
            "macd_bearish_cross": False,
            "ema_bearish_cross": True,
            "volume_spike": False,
        }
        result = evaluate_sell_conditions(conditions)
        assert result["ema_signal"] is True


class TestShouldExecuteTrade:
    """Test trade execution decision."""

    def test_trigger_guard_signal_enters(self):
        from user_data.strategies.signals import should_execute_trade

        signal = {"should_enter": True, "active_triggers": ["rsi_cross"]}
        assert should_execute_trade(signal) is True

    def test_trigger_guard_signal_rejects(self):
        from user_data.strategies.signals import should_execute_trade

        signal = {"should_enter": False, "failed_guards": ["volume_spike"]}
        assert should_execute_trade(signal) is False

    def test_confidence_signal_threshold(self):
        from user_data.strategies.signals import should_execute_trade

        weak = {"confidence": 0.2}
        strong = {"confidence": 0.7}
        assert should_execute_trade(weak, min_confidence=0.5) is False
        assert should_execute_trade(strong, min_confidence=0.5) is True


class TestSignalTiming:
    """Test signal timing and cooldown."""

    def test_no_duplicate_signals(self):
        from user_data.strategies.signals import SignalTracker

        tracker = SignalTracker(cooldown_periods=3)
        assert tracker.can_signal("BTC/USDT", "buy") is True
        tracker.record_signal("BTC/USDT", "buy")
        assert tracker.can_signal("BTC/USDT", "buy") is False

        for _ in range(3):
            tracker.advance_period()
        assert tracker.can_signal("BTC/USDT", "buy") is True

    def test_different_pairs_independent(self):
        from user_data.strategies.signals import SignalTracker

        tracker = SignalTracker(cooldown_periods=3)
        tracker.record_signal("BTC/USDT", "buy")
        assert tracker.can_signal("ETH/USDT", "buy") is True

    def test_buy_sell_independent(self):
        from user_data.strategies.signals import SignalTracker

        tracker = SignalTracker(cooldown_periods=3)
        tracker.record_signal("BTC/USDT", "buy")
        assert tracker.can_signal("BTC/USDT", "sell") is True
