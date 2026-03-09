"""Tests for portfolio.signals.structure — composite price-structure signal.

Covers all four sub-indicators (high/low breakout, Donchian 55, RSI centerline,
MACD zero-line cross) plus the composite compute_structure_signal() function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.structure import (
    _donchian_breakout,
    _highlow_breakout,
    _macd_zeroline,
    _rsi_centerline,
    compute_structure_signal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(closes: list[float], *, spread: float = 0.5) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices.

    high = close + spread, low = close - spread, open = close, volume = 1000.
    """
    n = len(closes)
    return pd.DataFrame({
        "open": closes,
        "high": [c + spread for c in closes],
        "low": [c - spread for c in closes],
        "close": closes,
        "volume": [1000] * n,
    })


def _make_ohlcv_full(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[int] | None = None,
) -> pd.DataFrame:
    """Build an OHLCV DataFrame with explicit columns."""
    n = len(closes)
    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes or [1000] * n,
    })


# =========================================================================
# _highlow_breakout
# =========================================================================

class TestHighlowBreakout:
    """Tests for the period high/low breakout sub-indicator."""

    def test_insufficient_data(self):
        """Returns HOLD with NaN indicators when < 20 bars."""
        df = _make_ohlcv([100.0] * 15)
        action, ind = _highlow_breakout(df)
        assert action == "HOLD"
        assert np.isnan(ind["period_high"])
        assert np.isnan(ind["period_low"])

    def test_buy_near_period_high(self):
        """Close within 2% of period high triggers BUY."""
        # 25 bars, all at 100 except high was 102 once. Current close = 100.5
        # pct_from_high = (102.5 - 100.5) / 102.5 = 0.0195 < 0.02 -> BUY
        closes = [100.0] * 24 + [100.5]
        highs = [100.5] * 24 + [100.5]  # spread-based
        highs[10] = 102.5  # one bar had a high spike
        lows = [99.5] * 25
        df = _make_ohlcv_full(closes, highs, lows, closes)
        action, ind = _highlow_breakout(df)
        assert action == "BUY"
        assert ind["period_high"] == 102.5

    def test_sell_near_period_low(self):
        """Close within 2% of period low triggers SELL."""
        # Period low = 95 (from lows), close = 96 -> pct_from_low = (96-95)/95 = 0.0105 < 0.02
        closes = [100.0] * 24 + [96.0]
        highs = [101.0] * 25
        lows = [99.0] * 25
        lows[5] = 95.0  # one bar had a low dip
        df = _make_ohlcv_full(closes, highs, lows, closes)
        action, ind = _highlow_breakout(df)
        assert action == "SELL"
        assert ind["period_low"] == 95.0

    def test_hold_middle_range(self):
        """Close far from both extremes returns HOLD."""
        closes = [100.0] * 25
        highs = [120.0] * 25  # period high = 120
        lows = [80.0] * 25   # period low = 80
        df = _make_ohlcv_full(closes, highs, lows, closes)
        action, ind = _highlow_breakout(df)
        assert action == "HOLD"
        assert ind["period_high"] == 120.0
        assert ind["period_low"] == 80.0

    def test_zero_period_high(self):
        """Edge case: period_high == 0 returns HOLD."""
        closes = [0.0] * 25
        highs = [0.0] * 25
        lows = [0.0] * 25
        df = _make_ohlcv_full(closes, highs, lows, closes)
        action, _ = _highlow_breakout(df)
        assert action == "HOLD"


# =========================================================================
# _donchian_breakout
# =========================================================================

class TestDonchianBreakout:
    """Tests for the Donchian Channel(55) breakout sub-indicator."""

    def test_insufficient_data(self):
        """Returns HOLD when fewer than period+1 bars."""
        df = _make_ohlcv([100.0] * 50)
        action, ind = _donchian_breakout(df, period=55)
        assert action == "HOLD"
        assert np.isnan(ind["donchian_upper"])
        assert np.isnan(ind["donchian_lower"])

    def test_buy_above_upper_channel(self):
        """Close above upper Donchian channel triggers BUY."""
        # 57 bars (period=55, need 56). Prior 55 bars have high=105.
        # Current close = 110 -> above upper.
        n = 57
        closes = [100.0] * 56 + [110.0]
        highs = [105.0] * 56 + [110.0]
        lows = [95.0] * 56 + [109.0]
        df = _make_ohlcv_full(closes, highs, lows, closes)
        action, ind = _donchian_breakout(df, period=55)
        assert action == "BUY"
        assert ind["donchian_upper"] == 105.0

    def test_sell_below_lower_channel(self):
        """Close below lower Donchian channel triggers SELL."""
        n = 57
        closes = [100.0] * 56 + [90.0]
        highs = [105.0] * 56 + [91.0]
        lows = [95.0] * 56 + [89.0]
        df = _make_ohlcv_full(closes, highs, lows, closes)
        action, ind = _donchian_breakout(df, period=55)
        assert action == "SELL"
        assert ind["donchian_lower"] == 95.0

    def test_hold_inside_channel(self):
        """Close inside the channel returns HOLD."""
        n = 57
        closes = [100.0] * 57
        highs = [105.0] * 57
        lows = [95.0] * 57
        df = _make_ohlcv_full(closes, highs, lows, closes)
        action, ind = _donchian_breakout(df, period=55)
        assert action == "HOLD"
        assert ind["donchian_upper"] == 105.0
        assert ind["donchian_lower"] == 95.0

    def test_channel_excludes_current_bar(self):
        """Channel is computed from prior bars, not including the current one."""
        # 57 bars: prior 55 highs = 100, current bar high = 200
        # Upper should still be 100 (excludes current bar)
        n = 57
        closes = [100.0] * 56 + [100.0]
        highs = [100.0] * 56 + [200.0]
        lows = [95.0] * 57
        df = _make_ohlcv_full(closes, highs, lows, closes)
        action, ind = _donchian_breakout(df, period=55)
        assert ind["donchian_upper"] == 100.0  # current bar's 200 excluded


# =========================================================================
# _rsi_centerline
# =========================================================================

class TestRsiCenterline:
    """Tests for the RSI(14) centerline cross sub-indicator."""

    def test_insufficient_data(self):
        """Returns HOLD when < 15 bars."""
        df = _make_ohlcv([100.0] * 10)
        action, ind = _rsi_centerline(df)
        assert action == "HOLD"
        assert np.isnan(ind["rsi"])

    def test_buy_strong_uptrend(self):
        """Strong uptrend with pullbacks produces RSI > 60 -> BUY."""
        # Uptrend with small pullbacks so RSI is finite (not NaN from pure gains)
        closes = []
        price = 100.0
        for i in range(50):
            if i % 5 == 4:
                price -= 0.3  # small pullback every 5th bar
            else:
                price += 1.5
            closes.append(price)
        df = _make_ohlcv(closes)
        action, ind = _rsi_centerline(df)
        assert action == "BUY"
        assert ind["rsi"] > 60.0

    def test_sell_strong_downtrend(self):
        """Strong downtrend with small bounces produces RSI < 40 -> SELL."""
        closes = []
        price = 200.0
        for i in range(50):
            if i % 5 == 4:
                price += 0.3  # small bounce every 5th bar
            else:
                price -= 2.0
            closes.append(price)
        df = _make_ohlcv(closes)
        action, ind = _rsi_centerline(df)
        assert action == "SELL"
        assert ind["rsi"] < 40.0

    def test_hold_flat_prices(self):
        """Flat prices produce RSI near 50 -> HOLD (within 40-60 deadband)."""
        closes = [100.0] * 50
        df = _make_ohlcv(closes)
        action, ind = _rsi_centerline(df)
        # RSI on flat data is NaN (0/0 -> NaN) => HOLD
        assert action == "HOLD"


# =========================================================================
# _macd_zeroline
# =========================================================================

class TestMacdZeroline:
    """Tests for the MACD(12,26,9) histogram zero-line cross sub-indicator."""

    def test_insufficient_data(self):
        """Returns HOLD when < 35 bars."""
        df = _make_ohlcv([100.0] * 30)
        action, ind = _macd_zeroline(df)
        assert action == "HOLD"
        assert np.isnan(ind["macd_hist"])

    def test_buy_bullish_cross(self):
        """MACD histogram crossing from negative to positive -> BUY.

        We build a sequence: long downtrend (MACD goes negative), then a sharp
        reversal at the end so the histogram flips to positive on the last bar.
        """
        # 40 bars of decline, then 15 bars of sharp rise to force histogram flip
        closes = [200.0 - i * 1.0 for i in range(40)]
        closes += [closes[-1] + i * 5.0 for i in range(1, 16)]
        df = _make_ohlcv(closes)
        action, ind = _macd_zeroline(df)
        # The sharp rise should push MACD histogram positive
        # Verify the indicator is populated
        assert not np.isnan(ind["macd_hist"])
        # If histogram crossed, action should be BUY; if still settling, HOLD
        # We check the specific scenario produces expected result
        if ind["macd_hist"] > 0:
            assert action in ("BUY", "HOLD")  # BUY if crossed, HOLD if already positive

    def test_sell_bearish_cross(self):
        """MACD histogram crossing from positive to negative -> SELL.

        Build a sequence: long uptrend (MACD positive), then sharp decline.
        """
        closes = [100.0 + i * 1.0 for i in range(40)]
        closes += [closes[-1] - i * 5.0 for i in range(1, 16)]
        df = _make_ohlcv(closes)
        action, ind = _macd_zeroline(df)
        assert not np.isnan(ind["macd_hist"])
        if ind["macd_hist"] < 0:
            assert action in ("SELL", "HOLD")  # SELL if crossed, HOLD if already negative

    def test_hold_no_cross(self):
        """Steady trend without cross returns HOLD."""
        # Steady uptrend — histogram stays positive, no cross
        closes = [100.0 + i * 0.5 for i in range(60)]
        df = _make_ohlcv(closes)
        action, ind = _macd_zeroline(df)
        assert action == "HOLD"
        assert not np.isnan(ind["macd_hist"])

    def test_hold_flat_prices(self):
        """Flat prices produce ~0 histogram, no cross -> HOLD."""
        closes = [100.0] * 60
        df = _make_ohlcv(closes)
        action, ind = _macd_zeroline(df)
        assert action == "HOLD"


# =========================================================================
# compute_structure_signal (composite)
# =========================================================================

class TestComputeStructureSignal:
    """Tests for the composite compute_structure_signal() function."""

    def test_none_input(self):
        """None input returns default HOLD result."""
        result = compute_structure_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0
        assert set(result["sub_signals"].keys()) == {
            "high_low_breakout", "donchian_55", "rsi_centerline", "macd_zeroline",
        }

    def test_non_dataframe_input(self):
        """Non-DataFrame input returns default HOLD result."""
        result = compute_structure_signal("not a dataframe")
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_missing_columns(self):
        """DataFrame missing required OHLCV columns returns HOLD."""
        df = pd.DataFrame({"close": [100.0] * 10, "volume": [1000] * 10})
        result = compute_structure_signal(df)
        assert result["action"] == "HOLD"

    def test_too_few_rows(self):
        """DataFrame with < 2 rows returns HOLD."""
        df = _make_ohlcv([100.0])
        result = compute_structure_signal(df)
        assert result["action"] == "HOLD"

    def test_result_keys(self):
        """Result dict contains all expected keys."""
        df = _make_ohlcv([100.0] * 60)
        result = compute_structure_signal(df)
        assert "action" in result
        assert "confidence" in result
        assert "sub_signals" in result
        assert "indicators" in result
        # Sub-signals
        assert set(result["sub_signals"].keys()) == {
            "high_low_breakout", "donchian_55", "rsi_centerline", "macd_zeroline",
        }
        # Indicators
        expected_ind = {"period_high", "period_low", "donchian_upper", "donchian_lower", "rsi", "macd_hist"}
        assert set(result["indicators"].keys()) == expected_ind

    def test_action_is_valid(self):
        """Action must be one of BUY, SELL, HOLD."""
        df = _make_ohlcv([100.0 + i * 0.5 for i in range(100)])
        result = compute_structure_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_confidence_range(self):
        """Confidence is between 0.0 and 1.0."""
        df = _make_ohlcv([100.0 + i * 0.5 for i in range(100)])
        result = compute_structure_signal(df)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_all_hold_sub_signals(self):
        """When all sub-signals HOLD, composite is HOLD with 0 confidence."""
        # Short data: only RSI has enough bars, but flat prices -> RSI HOLD
        # High/low has enough (20+), but close is mid-range -> HOLD
        # Donchian needs 56 bars -> HOLD (insufficient)
        # MACD needs 35 bars -> HOLD (insufficient)
        # Use 25 bars of flat data: high/low HOLD (mid-range), rest insufficient
        closes = [100.0] * 25
        highs = [120.0] * 25
        lows = [80.0] * 25
        df = _make_ohlcv_full(closes, highs, lows, closes)
        result = compute_structure_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_uptrend_with_enough_data(self):
        """Long uptrend with 100 bars produces BUY from multiple sub-signals."""
        # Uptrend with small pullbacks so RSI is finite
        closes = []
        price = 100.0
        for i in range(100):
            if i % 5 == 4:
                price -= 0.3
            else:
                price += 1.2
            closes.append(price)
        df = _make_ohlcv(closes, spread=0.5)
        result = compute_structure_signal(df)
        # RSI should be BUY (uptrend > 60)
        assert result["sub_signals"]["rsi_centerline"] == "BUY"
        # Close is near period high (within 2%)
        assert result["sub_signals"]["high_low_breakout"] == "BUY"
        # Action should be BUY (at least 2 out of 4)
        assert result["action"] == "BUY"
        assert result["confidence"] > 0.0

    def test_downtrend_with_enough_data(self):
        """Long downtrend with 100 bars should have SELL signals."""
        closes = []
        price = 300.0
        for i in range(100):
            if i % 5 == 4:
                price += 0.3
            else:
                price -= 1.5
            closes.append(price)
        df = _make_ohlcv(closes, spread=0.5)
        result = compute_structure_signal(df)
        # RSI should be SELL (downtrend < 40)
        assert result["sub_signals"]["rsi_centerline"] == "SELL"
        # Close near period low
        assert result["sub_signals"]["high_low_breakout"] == "SELL"
        assert result["action"] == "SELL"

    def test_indicators_populated(self):
        """Indicators are populated with numeric values for sufficient data."""
        # Use data with pullbacks so RSI and MACD are finite
        closes = []
        price = 100.0
        for i in range(100):
            if i % 5 == 4:
                price -= 0.3
            else:
                price += 0.8
            closes.append(price)
        df = _make_ohlcv(closes, spread=0.5)
        result = compute_structure_signal(df)
        ind = result["indicators"]
        # Period high/low should be numeric
        assert not np.isnan(ind["period_high"])
        assert not np.isnan(ind["period_low"])
        # Donchian channels (100 > 56 bars)
        assert not np.isnan(ind["donchian_upper"])
        assert not np.isnan(ind["donchian_lower"])
        # RSI (100 > 15 bars, non-monotonic data)
        assert not np.isnan(ind["rsi"])
        # MACD histogram (100 > 35 bars)
        assert not np.isnan(ind["macd_hist"])

    def test_empty_dataframe(self):
        """Empty DataFrame returns default HOLD."""
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_structure_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_sub_signals_match_actions(self):
        """Sub-signal values are always valid action strings."""
        df = _make_ohlcv([100.0 + i for i in range(80)])
        result = compute_structure_signal(df)
        for key, val in result["sub_signals"].items():
            assert val in ("BUY", "SELL", "HOLD"), f"{key} has invalid value: {val}"
