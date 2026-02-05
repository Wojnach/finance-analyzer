"""
Unit tests for trading signal logic.
These tests verify that buy/sell signals are generated correctly.
"""

import pytest
import pandas as pd
import numpy as np


class TestBuySignals:
    """Test buy signal generation."""

    def test_rsi_oversold_generates_buy(self):
        """RSI below threshold should contribute to buy signal."""
        from user_data.strategies.signals import evaluate_buy_conditions

        conditions = {
            "rsi": 25,  # Below 30 threshold
            "macd_bullish_cross": False,
            "ema_bullish_cross": False,
            "volume_spike": False,
        }

        result = evaluate_buy_conditions(conditions)
        assert result["rsi_signal"] == True

    def test_macd_crossover_generates_buy(self):
        """MACD bullish crossover should contribute to buy signal."""
        from user_data.strategies.signals import evaluate_buy_conditions

        conditions = {
            "rsi": 50,  # Neutral
            "macd_bullish_cross": True,
            "ema_bullish_cross": False,
            "volume_spike": False,
        }

        result = evaluate_buy_conditions(conditions)
        assert result["macd_signal"] == True

    def test_ema_crossover_generates_buy(self):
        """EMA bullish crossover should contribute to buy signal."""
        from user_data.strategies.signals import evaluate_buy_conditions

        conditions = {
            "rsi": 50,
            "macd_bullish_cross": False,
            "ema_bullish_cross": True,
            "volume_spike": False,
        }

        result = evaluate_buy_conditions(conditions)
        assert result["ema_signal"] == True

    def test_combined_signals_stronger(self):
        """Multiple confirming signals should produce higher confidence."""
        from user_data.strategies.signals import evaluate_buy_conditions

        # Single signal
        single = evaluate_buy_conditions(
            {
                "rsi": 25,
                "macd_bullish_cross": False,
                "ema_bullish_cross": False,
                "volume_spike": False,
            }
        )

        # Multiple signals
        multiple = evaluate_buy_conditions(
            {
                "rsi": 25,
                "macd_bullish_cross": True,
                "ema_bullish_cross": True,
                "volume_spike": True,
            }
        )

        assert multiple["confidence"] > single["confidence"]

    def test_volume_confirmation(self):
        """Volume spike should increase signal confidence."""
        from user_data.strategies.signals import evaluate_buy_conditions

        without_volume = evaluate_buy_conditions(
            {
                "rsi": 25,
                "macd_bullish_cross": True,
                "ema_bullish_cross": False,
                "volume_spike": False,
            }
        )

        with_volume = evaluate_buy_conditions(
            {
                "rsi": 25,
                "macd_bullish_cross": True,
                "ema_bullish_cross": False,
                "volume_spike": True,
            }
        )

        assert with_volume["confidence"] > without_volume["confidence"]


class TestSellSignals:
    """Test sell signal generation."""

    def test_rsi_overbought_generates_sell(self):
        """RSI above threshold should contribute to sell signal."""
        from user_data.strategies.signals import evaluate_sell_conditions

        conditions = {
            "rsi": 75,  # Above 70 threshold
            "macd_bearish_cross": False,
            "ema_bearish_cross": False,
            "volume_spike": False,
        }

        result = evaluate_sell_conditions(conditions)
        assert result["rsi_signal"] == True

    def test_macd_crossover_generates_sell(self):
        """MACD bearish crossover should contribute to sell signal."""
        from user_data.strategies.signals import evaluate_sell_conditions

        conditions = {
            "rsi": 50,
            "macd_bearish_cross": True,
            "ema_bearish_cross": False,
            "volume_spike": False,
        }

        result = evaluate_sell_conditions(conditions)
        assert result["macd_signal"] == True

    def test_ema_crossover_generates_sell(self):
        """EMA bearish crossover should contribute to sell signal."""
        from user_data.strategies.signals import evaluate_sell_conditions

        conditions = {
            "rsi": 50,
            "macd_bearish_cross": False,
            "ema_bearish_cross": True,
            "volume_spike": False,
        }

        result = evaluate_sell_conditions(conditions)
        assert result["ema_signal"] == True


class TestSignalThresholds:
    """Test signal threshold configurations."""

    def test_configurable_rsi_thresholds(self):
        """RSI thresholds should be configurable."""
        from user_data.strategies.signals import evaluate_buy_conditions

        # With default threshold (30)
        default = evaluate_buy_conditions(
            {
                "rsi": 28,
                "macd_bullish_cross": False,
                "ema_bullish_cross": False,
                "volume_spike": False,
            },
            rsi_oversold=30,
        )

        # With custom threshold (25)
        custom = evaluate_buy_conditions(
            {
                "rsi": 28,
                "macd_bullish_cross": False,
                "ema_bullish_cross": False,
                "volume_spike": False,
            },
            rsi_oversold=25,
        )

        assert default["rsi_signal"] == True  # 28 < 30
        assert custom["rsi_signal"] == False  # 28 > 25

    def test_minimum_confidence_threshold(self):
        """Signals below minimum confidence should not trigger."""
        from user_data.strategies.signals import should_execute_trade

        weak_signal = {"confidence": 0.2, "rsi_signal": True, "macd_signal": False}
        strong_signal = {"confidence": 0.7, "rsi_signal": True, "macd_signal": True}

        assert should_execute_trade(weak_signal, min_confidence=0.5) == False
        assert should_execute_trade(strong_signal, min_confidence=0.5) == True


class TestSignalTiming:
    """Test signal timing and cooldown."""

    def test_no_duplicate_signals(self):
        """Should not generate same signal on consecutive candles."""
        from user_data.strategies.signals import SignalTracker

        tracker = SignalTracker(cooldown_periods=3)

        # First signal should be allowed
        assert tracker.can_signal("BTC/USDT", "buy") == True
        tracker.record_signal("BTC/USDT", "buy")

        # Immediate repeat should be blocked
        assert tracker.can_signal("BTC/USDT", "buy") == False

        # After cooldown, should be allowed again
        for _ in range(3):
            tracker.advance_period()
        assert tracker.can_signal("BTC/USDT", "buy") == True

    def test_different_pairs_independent(self):
        """Signals for different pairs should be independent."""
        from user_data.strategies.signals import SignalTracker

        tracker = SignalTracker(cooldown_periods=3)

        tracker.record_signal("BTC/USDT", "buy")

        # Different pair should not be affected
        assert tracker.can_signal("ETH/USDT", "buy") == True

    def test_buy_sell_independent(self):
        """Buy and sell signals should have independent cooldowns."""
        from user_data.strategies.signals import SignalTracker

        tracker = SignalTracker(cooldown_periods=3)

        tracker.record_signal("BTC/USDT", "buy")

        # Sell signal should not be affected by recent buy
        assert tracker.can_signal("BTC/USDT", "sell") == True
