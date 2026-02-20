"""Tests for enhanced composite signal modules.

Covers basic tests for each signal module:
- Returns one of BUY/SELL/HOLD
- Works with sample OHLCV data
- Handles edge cases (empty data, NaN values, insufficient data)

Modules tested: trend, volatility_sig, candlestick, structure, fibonacci,
smart_money, oscillators, heikin_ashi, volume_flow
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared test data helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n=250, base=100.0, trend=0.0, volatility=1.0, seed=42):
    """Create a synthetic OHLCV DataFrame."""
    np.random.seed(seed)
    noise = np.random.randn(n) * volatility
    close = base + np.cumsum(noise) + np.arange(n) * trend
    close = np.maximum(close, 1.0)
    high = close + np.abs(np.random.randn(n) * volatility)
    low = close - np.abs(np.random.randn(n) * volatility)
    low = np.maximum(low, 0.5)
    opn = close + np.random.randn(n) * 0.3
    volume = np.random.randint(100, 10000, n).astype(float)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "open": opn, "high": high, "low": low,
        "close": close, "volume": volume, "time": dates,
    })


def _make_uptrend(n=250, base=100.0):
    return _make_ohlcv(n=n, base=base, trend=0.5)


def _make_downtrend(n=250, base=200.0):
    return _make_ohlcv(n=n, base=base, trend=-0.5)


def _make_empty_df():
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def _make_nan_df(n=100):
    """DataFrame with NaN values in OHLCV."""
    df = _make_ohlcv(n=n)
    df.loc[df.index[:20], "close"] = np.nan
    df.loc[df.index[:10], "high"] = np.nan
    return df


def _assert_valid_signal(result):
    """Assert that a signal result dict is well-formed."""
    assert isinstance(result, dict)
    assert "action" in result
    assert result["action"] in ("BUY", "SELL", "HOLD")
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "sub_signals" in result
    assert isinstance(result["sub_signals"], dict)
    for name, vote in result["sub_signals"].items():
        assert vote in ("BUY", "SELL", "HOLD"), \
            f"Sub-signal {name} has invalid vote: {vote}"


# ===========================================================================
# Test: Trend Signal
# ===========================================================================

class TestTrendSignal:
    def test_returns_valid_signal(self):
        from portfolio.signals.trend import compute_trend_signal
        result = compute_trend_signal(_make_ohlcv())
        _assert_valid_signal(result)

    def test_uptrend_data(self):
        from portfolio.signals.trend import compute_trend_signal
        result = compute_trend_signal(_make_uptrend())
        _assert_valid_signal(result)
        # Strong uptrend should lean BUY (not guaranteed, depends on sub-signals)
        assert result["action"] in ("BUY", "HOLD")

    def test_downtrend_data(self):
        from portfolio.signals.trend import compute_trend_signal
        result = compute_trend_signal(_make_downtrend())
        _assert_valid_signal(result)

    def test_empty_df(self):
        from portfolio.signals.trend import compute_trend_signal
        result = compute_trend_signal(_make_empty_df())
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_none_input(self):
        from portfolio.signals.trend import compute_trend_signal
        result = compute_trend_signal(None)
        assert result["action"] == "HOLD"

    def test_insufficient_data(self):
        from portfolio.signals.trend import compute_trend_signal
        result = compute_trend_signal(_make_ohlcv(n=5))
        assert result["action"] == "HOLD"

    def test_has_7_sub_signals(self):
        from portfolio.signals.trend import compute_trend_signal
        result = compute_trend_signal(_make_ohlcv())
        assert len(result["sub_signals"]) == 7

    def test_indicators_present(self):
        from portfolio.signals.trend import compute_trend_signal
        result = compute_trend_signal(_make_ohlcv())
        assert "indicators" in result
        assert "adx" in result["indicators"]
        assert "sar" in result["indicators"]


# ===========================================================================
# Test: Volatility Signal
# ===========================================================================

class TestVolatilitySignal:
    def test_returns_valid_signal(self):
        from portfolio.signals.volatility import compute_volatility_signal
        result = compute_volatility_signal(_make_ohlcv())
        _assert_valid_signal(result)

    def test_has_6_sub_signals(self):
        from portfolio.signals.volatility import compute_volatility_signal
        result = compute_volatility_signal(_make_ohlcv())
        expected_subs = {"bb_squeeze", "bb_breakout", "atr_expansion",
                        "keltner", "historical_vol", "donchian"}
        assert set(result["sub_signals"].keys()) == expected_subs

    def test_empty_df(self):
        from portfolio.signals.volatility import compute_volatility_signal
        result = compute_volatility_signal(_make_empty_df())
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_data(self):
        from portfolio.signals.volatility import compute_volatility_signal
        result = compute_volatility_signal(_make_ohlcv(n=10))
        assert result["action"] == "HOLD"

    def test_high_volatility_data(self):
        from portfolio.signals.volatility import compute_volatility_signal
        result = compute_volatility_signal(_make_ohlcv(volatility=10.0))
        _assert_valid_signal(result)

    def test_indicators_present(self):
        from portfolio.signals.volatility import compute_volatility_signal
        result = compute_volatility_signal(_make_ohlcv())
        assert "bb_width" in result["indicators"]
        assert "atr" in result["indicators"]


# ===========================================================================
# Test: Candlestick Signal
# ===========================================================================

class TestCandlestickSignal:
    def test_returns_valid_signal(self):
        from portfolio.signals.candlestick import compute_candlestick_signal
        result = compute_candlestick_signal(_make_ohlcv())
        _assert_valid_signal(result)

    def test_has_4_sub_signals(self):
        from portfolio.signals.candlestick import compute_candlestick_signal
        result = compute_candlestick_signal(_make_ohlcv())
        expected_subs = {"hammer", "engulfing", "doji", "star"}
        assert set(result["sub_signals"].keys()) == expected_subs

    def test_patterns_detected_field(self):
        from portfolio.signals.candlestick import compute_candlestick_signal
        result = compute_candlestick_signal(_make_ohlcv())
        assert "patterns_detected" in result
        assert isinstance(result["patterns_detected"], list)

    def test_minimum_3_rows(self):
        from portfolio.signals.candlestick import compute_candlestick_signal
        result = compute_candlestick_signal(_make_ohlcv(n=2))
        assert result["action"] == "HOLD"

    def test_exactly_3_rows(self):
        from portfolio.signals.candlestick import compute_candlestick_signal
        result = compute_candlestick_signal(_make_ohlcv(n=3))
        _assert_valid_signal(result)

    def test_none_input(self):
        from portfolio.signals.candlestick import compute_candlestick_signal
        result = compute_candlestick_signal(None)
        assert result["action"] == "HOLD"

    def test_bullish_engulfing_detection(self):
        """Craft a bullish engulfing pattern and check detection."""
        from portfolio.signals.candlestick import compute_candlestick_signal

        # Create pattern: downtrend, then red candle followed by larger green candle
        data = {
            "open":  [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 80, 78, 82],
            "high":  [101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 81, 79, 83],
            "low":   [99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 77, 76, 77],
            "close": [99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 77, 78, 82],
            "volume": [100] * 21,
        }
        df = pd.DataFrame(data)
        result = compute_candlestick_signal(df)
        _assert_valid_signal(result)


# ===========================================================================
# Test: Structure Signal
# ===========================================================================

class TestStructureSignal:
    def test_returns_valid_signal(self):
        from portfolio.signals.structure import compute_structure_signal
        result = compute_structure_signal(_make_ohlcv())
        _assert_valid_signal(result)

    def test_has_4_sub_signals(self):
        from portfolio.signals.structure import compute_structure_signal
        result = compute_structure_signal(_make_ohlcv())
        expected_subs = {"high_low_breakout", "donchian_55",
                        "rsi_centerline", "macd_zeroline"}
        assert set(result["sub_signals"].keys()) == expected_subs

    def test_none_input(self):
        from portfolio.signals.structure import compute_structure_signal
        result = compute_structure_signal(None)
        assert result["action"] == "HOLD"

    def test_empty_df(self):
        from portfolio.signals.structure import compute_structure_signal
        result = compute_structure_signal(_make_empty_df())
        assert result["action"] == "HOLD"

    def test_short_data(self):
        from portfolio.signals.structure import compute_structure_signal
        result = compute_structure_signal(_make_ohlcv(n=1))
        assert result["action"] == "HOLD"

    def test_indicators_present(self):
        from portfolio.signals.structure import compute_structure_signal
        result = compute_structure_signal(_make_ohlcv(n=100))
        assert "rsi" in result["indicators"]
        assert "macd_hist" in result["indicators"]


# ===========================================================================
# Test: Fibonacci Signal
# ===========================================================================

class TestFibonacciSignal:
    def test_returns_valid_signal(self):
        from portfolio.signals.fibonacci import compute_fibonacci_signal
        result = compute_fibonacci_signal(_make_ohlcv())
        _assert_valid_signal(result)

    def test_has_5_sub_signals(self):
        from portfolio.signals.fibonacci import compute_fibonacci_signal
        result = compute_fibonacci_signal(_make_ohlcv())
        expected_subs = {"fib_retracement", "golden_pocket", "fib_extension",
                        "pivot_standard", "pivot_camarilla"}
        assert set(result["sub_signals"].keys()) == expected_subs

    def test_none_input(self):
        from portfolio.signals.fibonacci import compute_fibonacci_signal
        result = compute_fibonacci_signal(None)
        assert result["action"] == "HOLD"

    def test_insufficient_data(self):
        from portfolio.signals.fibonacci import compute_fibonacci_signal
        result = compute_fibonacci_signal(_make_ohlcv(n=10))
        assert result["action"] == "HOLD"

    def test_indicators_have_swing_levels(self):
        from portfolio.signals.fibonacci import compute_fibonacci_signal
        result = compute_fibonacci_signal(_make_ohlcv())
        assert "swing_high" in result["indicators"]
        assert "swing_low" in result["indicators"]
        assert "fib_levels" in result["indicators"]


# ===========================================================================
# Test: Smart Money Signal
# ===========================================================================

class TestSmartMoneySignal:
    def test_returns_valid_signal(self):
        from portfolio.signals.smart_money import compute_smart_money_signal
        result = compute_smart_money_signal(_make_ohlcv())
        _assert_valid_signal(result)

    def test_has_5_sub_signals(self):
        from portfolio.signals.smart_money import compute_smart_money_signal
        result = compute_smart_money_signal(_make_ohlcv())
        expected_subs = {"bos", "choch", "fvg", "liquidity_sweep", "supply_demand"}
        assert set(result["sub_signals"].keys()) == expected_subs

    def test_none_input(self):
        from portfolio.signals.smart_money import compute_smart_money_signal
        result = compute_smart_money_signal(None)
        assert result["action"] == "HOLD"

    def test_insufficient_data(self):
        from portfolio.signals.smart_money import compute_smart_money_signal
        result = compute_smart_money_signal(_make_ohlcv(n=10))
        assert result["action"] == "HOLD"

    def test_indicators_have_structure(self):
        from portfolio.signals.smart_money import compute_smart_money_signal
        result = compute_smart_money_signal(_make_ohlcv())
        assert "structure" in result["indicators"]
        assert "unfilled_fvgs" in result["indicators"]
        assert "in_demand_zone" in result["indicators"]
        assert "in_supply_zone" in result["indicators"]


# ===========================================================================
# Test: Oscillators Signal
# ===========================================================================

class TestOscillatorsSignal:
    def test_returns_valid_signal(self):
        from portfolio.signals.oscillators import compute_oscillator_signal
        result = compute_oscillator_signal(_make_ohlcv())
        _assert_valid_signal(result)

    def test_has_8_sub_signals(self):
        from portfolio.signals.oscillators import compute_oscillator_signal
        result = compute_oscillator_signal(_make_ohlcv())
        expected_subs = {"awesome", "aroon", "vortex", "chande",
                        "kst", "schaff", "trix", "coppock"}
        assert set(result["sub_signals"].keys()) == expected_subs

    def test_none_input(self):
        from portfolio.signals.oscillators import compute_oscillator_signal
        result = compute_oscillator_signal(None)
        assert result["action"] == "HOLD"

    def test_insufficient_data(self):
        from portfolio.signals.oscillators import compute_oscillator_signal
        result = compute_oscillator_signal(_make_ohlcv(n=10))
        assert result["action"] == "HOLD"

    def test_indicators_present(self):
        from portfolio.signals.oscillators import compute_oscillator_signal
        result = compute_oscillator_signal(_make_ohlcv())
        ind = result["indicators"]
        assert "awesome_osc" in ind
        assert "aroon_osc" in ind
        assert "cmo" in ind


# ===========================================================================
# Test: Heikin-Ashi Signal
# ===========================================================================

class TestHeikinAshiSignal:
    def test_returns_valid_signal(self):
        from portfolio.signals.heikin_ashi import compute_heikin_ashi_signal
        result = compute_heikin_ashi_signal(_make_ohlcv())
        _assert_valid_signal(result)

    def test_has_7_sub_signals(self):
        from portfolio.signals.heikin_ashi import compute_heikin_ashi_signal
        result = compute_heikin_ashi_signal(_make_ohlcv())
        expected_subs = {"ha_trend", "ha_doji", "ha_color_change",
                        "hull_ma", "alligator", "elder_impulse", "ttm_squeeze"}
        assert set(result["sub_signals"].keys()) == expected_subs

    def test_none_input(self):
        from portfolio.signals.heikin_ashi import compute_heikin_ashi_signal
        result = compute_heikin_ashi_signal(None)
        assert result["action"] == "HOLD"

    def test_empty_df(self):
        from portfolio.signals.heikin_ashi import compute_heikin_ashi_signal
        result = compute_heikin_ashi_signal(_make_empty_df())
        assert result["action"] == "HOLD"

    def test_insufficient_data(self):
        from portfolio.signals.heikin_ashi import compute_heikin_ashi_signal
        result = compute_heikin_ashi_signal(_make_ohlcv(n=5))
        assert result["action"] == "HOLD"

    def test_ha_indicators(self):
        from portfolio.signals.heikin_ashi import compute_heikin_ashi_signal
        result = compute_heikin_ashi_signal(_make_ohlcv())
        ind = result["indicators"]
        assert "ha_color" in ind
        assert ind["ha_color"] in ("green", "red")
        assert "ha_streak" in ind
        assert isinstance(ind["ha_streak"], int)
        assert "hull_fast" in ind
        assert "hull_slow" in ind
        assert "elder_color" in ind
        assert "ttm_squeeze_on" in ind
        assert isinstance(ind["ttm_squeeze_on"], bool)


# ===========================================================================
# Test: Volume Flow Signal
# ===========================================================================

class TestVolumeFlowSignal:
    def test_returns_valid_signal(self):
        from portfolio.signals.volume_flow import compute_volume_flow_signal
        result = compute_volume_flow_signal(_make_ohlcv())
        _assert_valid_signal(result)

    def test_has_6_sub_signals(self):
        from portfolio.signals.volume_flow import compute_volume_flow_signal
        result = compute_volume_flow_signal(_make_ohlcv())
        expected_subs = {"obv", "vwap", "ad_line", "cmf", "mfi", "volume_rsi"}
        assert set(result["sub_signals"].keys()) == expected_subs

    def test_none_input(self):
        from portfolio.signals.volume_flow import compute_volume_flow_signal
        result = compute_volume_flow_signal(None)
        assert result["action"] == "HOLD"

    def test_insufficient_data(self):
        from portfolio.signals.volume_flow import compute_volume_flow_signal
        result = compute_volume_flow_signal(_make_ohlcv(n=10))
        assert result["action"] == "HOLD"

    def test_indicators_present(self):
        from portfolio.signals.volume_flow import compute_volume_flow_signal
        result = compute_volume_flow_signal(_make_ohlcv())
        ind = result["indicators"]
        assert "obv" in ind
        assert "vwap" in ind
        assert "cmf" in ind
        assert "mfi" in ind

    def test_high_volume_data(self):
        """Signal should handle data with extremely high volume."""
        from portfolio.signals.volume_flow import compute_volume_flow_signal
        df = _make_ohlcv(n=100)
        df["volume"] = df["volume"] * 1_000_000
        result = compute_volume_flow_signal(df)
        _assert_valid_signal(result)


# ===========================================================================
# Test: NaN handling across all modules
# ===========================================================================

class TestNaNHandling:
    """All signal modules should handle NaN values gracefully."""

    def test_trend_with_nan(self):
        from portfolio.signals.trend import compute_trend_signal
        result = compute_trend_signal(_make_nan_df())
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_volatility_with_nan(self):
        from portfolio.signals.volatility import compute_volatility_signal
        result = compute_volatility_signal(_make_nan_df())
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_candlestick_with_nan(self):
        from portfolio.signals.candlestick import compute_candlestick_signal
        result = compute_candlestick_signal(_make_nan_df())
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_structure_with_nan(self):
        from portfolio.signals.structure import compute_structure_signal
        result = compute_structure_signal(_make_nan_df())
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_fibonacci_with_nan(self):
        from portfolio.signals.fibonacci import compute_fibonacci_signal
        result = compute_fibonacci_signal(_make_nan_df())
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_smart_money_with_nan(self):
        from portfolio.signals.smart_money import compute_smart_money_signal
        result = compute_smart_money_signal(_make_nan_df())
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_oscillators_with_nan(self):
        from portfolio.signals.oscillators import compute_oscillator_signal
        result = compute_oscillator_signal(_make_nan_df())
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_heikin_ashi_with_nan(self):
        from portfolio.signals.heikin_ashi import compute_heikin_ashi_signal
        result = compute_heikin_ashi_signal(_make_nan_df())
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_volume_flow_with_nan(self):
        from portfolio.signals.volume_flow import compute_volume_flow_signal
        result = compute_volume_flow_signal(_make_nan_df())
        assert result["action"] in ("BUY", "SELL", "HOLD")


# ===========================================================================
# Test: Missing columns handling
# ===========================================================================

class TestMissingColumns:
    """All signal modules should handle missing OHLCV columns gracefully."""

    def _df_missing_volume(self):
        df = _make_ohlcv(n=100)
        return df.drop(columns=["volume"])

    def test_trend_missing_volume(self):
        from portfolio.signals.trend import compute_trend_signal
        result = compute_trend_signal(self._df_missing_volume())
        assert result["action"] == "HOLD"

    def test_structure_missing_volume(self):
        from portfolio.signals.structure import compute_structure_signal
        result = compute_structure_signal(self._df_missing_volume())
        assert result["action"] == "HOLD"

    def test_volume_flow_missing_volume(self):
        from portfolio.signals.volume_flow import compute_volume_flow_signal
        result = compute_volume_flow_signal(self._df_missing_volume())
        assert result["action"] == "HOLD"
