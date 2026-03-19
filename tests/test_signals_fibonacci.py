"""Tests for portfolio.signals.fibonacci — composite Fibonacci and pivot signal.

Covers all 5 sub-indicators, the main compute function, edge cases, and
result structure validation.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from portfolio.signals.fibonacci import (
    MIN_ROWS,
    _compute_fib_extensions,
    _compute_fib_levels,
    _detect_trend,
    _fib_extension_signal,
    _fib_retracement_signal,
    _find_swing_high_low,
    _golden_pocket_signal,
    _near_level,
    _pivot_camarilla_signal,
    _pivot_standard_signal,
    compute_fibonacci_signal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(
    closes: list[float],
    spread_pct: float = 0.01,
    base_volume: float = 1000.0,
) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame from a list of close prices.

    High = close * (1 + spread_pct), Low = close * (1 - spread_pct),
    Open = midpoint of previous close and current close.
    """
    n = len(closes)
    highs = [c * (1 + spread_pct) for c in closes]
    lows = [c * (1 - spread_pct) for c in closes]
    opens = [closes[0]] + [(closes[i - 1] + closes[i]) / 2 for i in range(1, n)]
    volumes = [base_volume] * n
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}
    )


def _make_uptrend_df(n: int = 100, start: float = 100.0, step: float = 0.5) -> pd.DataFrame:
    """Create a steadily rising OHLCV DataFrame."""
    closes = [start + i * step for i in range(n)]
    return _make_ohlcv(closes)


def _make_downtrend_df(n: int = 100, start: float = 200.0, step: float = 0.5) -> pd.DataFrame:
    """Create a steadily falling OHLCV DataFrame."""
    closes = [start - i * step for i in range(n)]
    return _make_ohlcv(closes)


def _make_flat_df(n: int = 100, price: float = 100.0) -> pd.DataFrame:
    """Create a flat / constant-price OHLCV DataFrame."""
    closes = [price] * n
    return _make_ohlcv(closes, spread_pct=0.0)


# ---------------------------------------------------------------------------
# _detect_trend
# ---------------------------------------------------------------------------


class TestDetectTrend:
    def test_uptrend(self):
        closes = pd.Series([100 + i * 0.5 for i in range(60)])
        assert _detect_trend(closes, period=20) == "up"

    def test_downtrend(self):
        closes = pd.Series([200 - i * 0.5 for i in range(60)])
        assert _detect_trend(closes, period=20) == "down"

    def test_flat(self):
        closes = pd.Series([100.0] * 60)
        assert _detect_trend(closes, period=20) == "flat"

    def test_insufficient_data(self):
        closes = pd.Series([100.0] * 10)
        assert _detect_trend(closes, period=20) == "flat"


# ---------------------------------------------------------------------------
# _near_level
# ---------------------------------------------------------------------------


class TestNearLevel:
    def test_exact_match(self):
        assert _near_level(100.0, 100.0) is True

    def test_within_tolerance(self):
        # 0.5% away from 100 = within default 1% tolerance
        assert _near_level(100.5, 100.0) is True

    def test_outside_tolerance(self):
        # 2% away — outside 1% default tolerance
        assert _near_level(102.0, 100.0) is False

    def test_level_zero(self):
        assert _near_level(0.0, 0.0) is False


# ---------------------------------------------------------------------------
# _compute_fib_levels
# ---------------------------------------------------------------------------


class TestComputeFibLevels:
    def test_known_levels(self):
        levels = _compute_fib_levels(200.0, 100.0)
        diff = 100.0
        assert levels[0.236] == pytest.approx(200.0 - 0.236 * diff)
        assert levels[0.382] == pytest.approx(200.0 - 0.382 * diff)
        assert levels[0.500] == pytest.approx(150.0)
        assert levels[0.618] == pytest.approx(200.0 - 0.618 * diff)
        assert levels[0.786] == pytest.approx(200.0 - 0.786 * diff)

    def test_all_ratios_present(self):
        levels = _compute_fib_levels(200.0, 100.0)
        assert set(levels.keys()) == {0.236, 0.382, 0.500, 0.618, 0.786}


# ---------------------------------------------------------------------------
# _compute_fib_extensions
# ---------------------------------------------------------------------------


class TestComputeFibExtensions:
    def test_extension_values(self):
        ext = _compute_fib_extensions(200.0, 100.0)
        diff = 100.0
        assert ext["ext_up_127"] == pytest.approx(200.0 + 0.272 * diff)
        assert ext["ext_up_161"] == pytest.approx(200.0 + 0.618 * diff)
        assert ext["ext_down_127"] == pytest.approx(100.0 - 0.272 * diff)
        assert ext["ext_down_161"] == pytest.approx(100.0 - 0.618 * diff)

    def test_all_keys_present(self):
        ext = _compute_fib_extensions(200.0, 100.0)
        assert set(ext.keys()) == {"ext_up_127", "ext_up_161", "ext_down_127", "ext_down_161"}


# ---------------------------------------------------------------------------
# _fib_retracement_signal
# ---------------------------------------------------------------------------


class TestFibRetracementSignal:
    def setup_method(self):
        self.swing_high = 200.0
        self.swing_low = 100.0
        self.fib_levels = _compute_fib_levels(self.swing_high, self.swing_low)

    def test_buy_near_50_uptrend(self):
        # 50% level = 150.0. Price at 150.0 in uptrend => BUY
        action, info = _fib_retracement_signal(
            150.0, self.swing_high, self.swing_low, self.fib_levels, "up"
        )
        assert action == "BUY"
        assert info["near_50"] is True

    def test_sell_near_618_downtrend(self):
        # 61.8% level = 200 - 61.8 = 138.2. Price near 138.2 in downtrend => SELL
        level_618 = self.fib_levels[0.618]
        action, info = _fib_retracement_signal(
            level_618, self.swing_high, self.swing_low, self.fib_levels, "down"
        )
        assert action == "SELL"
        assert info["near_618"] is True

    def test_hold_when_flat(self):
        action, info = _fib_retracement_signal(
            150.0, self.swing_high, self.swing_low, self.fib_levels, "flat"
        )
        assert action == "HOLD"

    def test_hold_when_price_away_from_levels(self):
        # Price at 170 is not near 50% (150) or 61.8% (138.2)
        action, _ = _fib_retracement_signal(
            170.0, self.swing_high, self.swing_low, self.fib_levels, "up"
        )
        assert action == "HOLD"


# ---------------------------------------------------------------------------
# _golden_pocket_signal
# ---------------------------------------------------------------------------


class TestGoldenPocketSignal:
    def setup_method(self):
        self.swing_high = 200.0
        self.swing_low = 100.0
        self.fib_levels = _compute_fib_levels(self.swing_high, self.swing_low)

    def test_buy_in_pocket_uptrend(self):
        # Golden pocket: 61.8% = 138.2, 65% = 135.0. Price 137.0 is inside.
        action, info = _golden_pocket_signal(
            137.0, self.swing_high, self.swing_low, self.fib_levels, "up"
        )
        assert action == "BUY"
        assert info["in_pocket"] is True

    def test_sell_in_pocket_downtrend(self):
        action, info = _golden_pocket_signal(
            137.0, self.swing_high, self.swing_low, self.fib_levels, "down"
        )
        assert action == "SELL"
        assert info["in_pocket"] is True

    def test_hold_outside_pocket(self):
        # 170 is well above the golden pocket zone
        action, info = _golden_pocket_signal(
            170.0, self.swing_high, self.swing_low, self.fib_levels, "up"
        )
        assert action == "HOLD"
        assert info["in_pocket"] is False

    def test_hold_flat_trend_in_pocket(self):
        action, info = _golden_pocket_signal(
            137.0, self.swing_high, self.swing_low, self.fib_levels, "flat"
        )
        assert action == "HOLD"
        assert info["in_pocket"] is True  # in pocket but flat trend => HOLD

    def test_pocket_boundaries(self):
        """Verify the golden pocket bounds are computed correctly."""
        diff = self.swing_high - self.swing_low  # 100
        _, info = _golden_pocket_signal(
            137.0, self.swing_high, self.swing_low, self.fib_levels, "up"
        )
        assert info["gp_upper"] == pytest.approx(self.swing_high - 0.618 * diff)
        assert info["gp_lower"] == pytest.approx(self.swing_high - 0.650 * diff)


# ---------------------------------------------------------------------------
# _fib_extension_signal
# ---------------------------------------------------------------------------


class TestFibExtensionSignal:
    def setup_method(self):
        self.swing_high = 200.0
        self.swing_low = 100.0
        self.extensions = _compute_fib_extensions(self.swing_high, self.swing_low)

    def test_sell_at_161_upside(self):
        # ext_up_161 = 200 + 0.618 * 100 = 261.8. Price at/above => SELL
        action, info = _fib_extension_signal(
            262.0, self.swing_high, self.swing_low, self.extensions, "up"
        )
        assert action == "SELL"
        assert info["near_ext_up_161"] is True

    def test_buy_at_161_downside(self):
        # ext_down_161 = 100 - 0.618 * 100 = 38.2. Price at/below => BUY
        action, info = _fib_extension_signal(
            38.0, self.swing_high, self.swing_low, self.extensions, "down"
        )
        assert action == "BUY"
        assert info["near_ext_down_161"] is True

    def test_hold_price_between_swings(self):
        action, info = _fib_extension_signal(
            150.0, self.swing_high, self.swing_low, self.extensions, "up"
        )
        assert action == "HOLD"
        assert info["above_swing_high"] is False
        assert info["below_swing_low"] is False

    def test_hold_above_swing_but_below_161(self):
        # Price at 210 is above swing_high (200) but below ext_up_161 (261.8)
        action, info = _fib_extension_signal(
            210.0, self.swing_high, self.swing_low, self.extensions, "up"
        )
        assert action == "HOLD"
        assert info["above_swing_high"] is True


# ---------------------------------------------------------------------------
# _pivot_standard_signal
# ---------------------------------------------------------------------------


class TestPivotStandardSignal:
    def test_buy_above_r1(self):
        # H=110, L=90, C=100 => PP=100, R1=110, S1=90
        action, info = _pivot_standard_signal(110.0, 90.0, 100.0, 115.0)
        assert action == "BUY"
        assert info["r1"] == pytest.approx(110.0)

    def test_sell_below_s1(self):
        action, info = _pivot_standard_signal(110.0, 90.0, 100.0, 85.0)
        assert action == "SELL"
        assert info["s1"] == pytest.approx(90.0)

    def test_hold_between_s1_r1(self):
        action, info = _pivot_standard_signal(110.0, 90.0, 100.0, 100.0)
        assert action == "HOLD"

    def test_pivot_calculation(self):
        """Verify PP, R1, R2, S1, S2 formula."""
        h, l, c = 120.0, 80.0, 100.0
        _, info = _pivot_standard_signal(h, l, c, 100.0)
        pp = (h + l + c) / 3.0
        assert info["pivot"] == pytest.approx(pp)
        assert info["r1"] == pytest.approx(2 * pp - l)
        assert info["s1"] == pytest.approx(2 * pp - h)
        assert info["r2"] == pytest.approx(pp + (h - l))
        assert info["s2"] == pytest.approx(pp - (h - l))


# ---------------------------------------------------------------------------
# _pivot_camarilla_signal
# ---------------------------------------------------------------------------


class TestPivotCamarillaSignal:
    def test_buy_above_r3(self):
        h, l, c = 110.0, 90.0, 100.0
        hl = h - l  # 20
        r3 = c + hl * 1.1 / 4.0  # 100 + 5.5 = 105.5
        action, info = _pivot_camarilla_signal(h, l, c, 106.0)
        assert action == "BUY"
        assert info["cam_r3"] == pytest.approx(r3)

    def test_sell_below_s3(self):
        h, l, c = 110.0, 90.0, 100.0
        s3 = c - (h - l) * 1.1 / 4.0  # 100 - 5.5 = 94.5
        action, info = _pivot_camarilla_signal(h, l, c, 94.0)
        assert action == "SELL"
        assert info["cam_s3"] == pytest.approx(s3)

    def test_hold_between_levels(self):
        action, info = _pivot_camarilla_signal(110.0, 90.0, 100.0, 100.0)
        assert action == "HOLD"

    def test_r4_s4_in_info(self):
        h, l, c = 110.0, 90.0, 100.0
        _, info = _pivot_camarilla_signal(h, l, c, 100.0)
        hl = h - l
        assert info["cam_r4"] == pytest.approx(c + hl * 1.1 / 2.0)
        assert info["cam_s4"] == pytest.approx(c - hl * 1.1 / 2.0)


# ---------------------------------------------------------------------------
# compute_fibonacci_signal — integration
# ---------------------------------------------------------------------------


class TestComputeFibonacciSignal:
    def test_result_keys(self):
        df = _make_uptrend_df(100)
        result = compute_fibonacci_signal(df)
        assert "action" in result
        assert "confidence" in result
        assert "sub_signals" in result
        assert "indicators" in result

    def test_sub_signal_keys(self):
        df = _make_uptrend_df(100)
        result = compute_fibonacci_signal(df)
        expected_subs = {
            "fib_retracement",
            "golden_pocket",
            "fib_extension",
            "pivot_standard",
            "pivot_camarilla",
        }
        assert set(result["sub_signals"].keys()) == expected_subs

    def test_action_valid_values(self):
        df = _make_uptrend_df(100)
        result = compute_fibonacci_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_confidence_range(self):
        df = _make_uptrend_df(100)
        result = compute_fibonacci_signal(df)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_indicators_has_swing_levels(self):
        df = _make_uptrend_df(100)
        result = compute_fibonacci_signal(df)
        ind = result["indicators"]
        assert "swing_high" in ind
        assert "swing_low" in ind
        assert not math.isnan(ind["swing_high"])
        assert not math.isnan(ind["swing_low"])

    def test_insufficient_rows_returns_hold(self):
        df = _make_ohlcv([100.0] * (MIN_ROWS - 1))
        result = compute_fibonacci_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_none_input(self):
        result = compute_fibonacci_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_missing_columns(self):
        df = pd.DataFrame({"close": [100.0] * 60, "volume": [1000] * 60})
        result = compute_fibonacci_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_flat_prices_returns_hold(self):
        """Flat prices => swing_high == swing_low => sanity check fails => HOLD."""
        df = _make_flat_df(100, price=50.0)
        result = compute_fibonacci_signal(df)
        assert result["action"] == "HOLD"

    def test_downtrend_signal(self):
        """In a clear downtrend the signal should not produce a BUY."""
        df = _make_downtrend_df(100, start=200.0, step=0.5)
        result = compute_fibonacci_signal(df)
        assert result["action"] in ("SELL", "HOLD")

    def test_uptrend_signal(self):
        """In a clear uptrend the signal should not produce a SELL."""
        df = _make_uptrend_df(100, start=100.0, step=0.5)
        result = compute_fibonacci_signal(df)
        assert result["action"] in ("BUY", "HOLD")


# ---------------------------------------------------------------------------
# _find_swing_high_low
# ---------------------------------------------------------------------------


class TestFindSwingHighLow:
    def test_basic_swing_detection(self):
        """A V-shaped pattern should detect the peak and trough."""
        # Rise from 100 to 150, then fall to 100
        up = [100 + i * 2 for i in range(30)]
        down = [150 - i * 2 for i in range(1, 31)]
        flat_tail = [90.0] * 40
        closes = up + down + flat_tail
        df = _make_ohlcv(closes)
        sh, sl, sh_idx, sl_idx = _find_swing_high_low(df)
        # Swing high should be near the peak (~150), swing low near the trough
        assert sh > sl
        assert sh >= 140.0  # peak area

    def test_fallback_to_absolute(self):
        """With too few bars for swing detection, should fall back to absolute high/low."""
        # Exactly MIN_ROWS bars, monotonic rise — no structural peaks
        closes = [100 + i for i in range(MIN_ROWS)]
        df = _make_ohlcv(closes)
        sh, sl, _, _ = _find_swing_high_low(df)
        assert sh > sl


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_extreme_prices(self):
        """Very large price values should not cause overflow."""
        closes = [1e8 + i * 1000 for i in range(100)]
        df = _make_ohlcv(closes)
        result = compute_fibonacci_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert not math.isnan(result["confidence"])

    def test_very_small_prices(self):
        """Very small (penny-stock) prices should work."""
        closes = [0.01 + i * 0.0001 for i in range(100)]
        df = _make_ohlcv(closes, spread_pct=0.005)
        result = compute_fibonacci_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_exactly_min_rows(self):
        """Exactly MIN_ROWS bars should be accepted."""
        df = _make_uptrend_df(MIN_ROWS)
        result = compute_fibonacci_signal(df)
        # Should not fall back to the hold_result with NaN indicators
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_string_not_dataframe(self):
        result = compute_fibonacci_signal("not a dataframe")
        assert result["action"] == "HOLD"

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_fibonacci_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0
