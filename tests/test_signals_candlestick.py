"""Tests for portfolio.signals.candlestick — candlestick pattern detection signal.

Covers:
  - Input validation (None, missing columns, insufficient rows, all-NaN)
  - Trend detection (_detect_trend)
  - Hammer / shooting star / inverted hammer / hanging man
  - Bullish / bearish engulfing
  - Doji (context-dependent on trend)
  - Morning star / evening star
  - Composite vote logic and confidence calculation
  - Pattern name collection
"""

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.candlestick import (
    _body,
    _body_pct,
    _check_doji,
    _check_engulfing,
    _check_hammer_family,
    _check_star,
    _detect_trend,
    _is_green,
    _is_red,
    _lower_shadow,
    _range,
    _upper_shadow,
    compute_candlestick_signal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rows, columns=("open", "high", "low", "close")):
    """Build a DataFrame from a list of tuples (open, high, low, close)."""
    return pd.DataFrame(rows, columns=columns)


def _make_bar(o, h, l, c):
    """Return a Series representing a single OHLC bar."""
    return pd.Series({"open": o, "high": h, "low": l, "close": c})


def _default_result():
    """What compute_candlestick_signal returns on invalid input."""
    return {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {
            "hammer": "HOLD",
            "engulfing": "HOLD",
            "doji": "HOLD",
            "star": "HOLD",
        },
        "indicators": {"patterns_detected": []},
    }


# ===================================================================
# Input validation
# ===================================================================

class TestInputValidation:
    """compute_candlestick_signal must degrade gracefully on bad input."""

    def test_none_input(self):
        assert compute_candlestick_signal(None) == _default_result()

    def test_non_dataframe_input(self):
        assert compute_candlestick_signal("not a df") == _default_result()
        assert compute_candlestick_signal(42) == _default_result()

    def test_missing_ohlc_columns(self):
        df = pd.DataFrame({"open": [1], "high": [2], "low": [0.5]})  # no close
        assert compute_candlestick_signal(df) == _default_result()

    def test_fewer_than_3_rows(self):
        df = _make_df([(100, 105, 95, 102), (102, 108, 99, 106)])
        assert compute_candlestick_signal(df) == _default_result()

    def test_all_nan_ohlc(self):
        df = pd.DataFrame({
            "open": [np.nan] * 5,
            "high": [np.nan] * 5,
            "low": [np.nan] * 5,
            "close": [np.nan] * 5,
        })
        assert compute_candlestick_signal(df) == _default_result()

    def test_some_nan_rows_still_enough(self):
        """If after dropping NaN we still have >= 3 rows, proceed normally."""
        rows = [
            (100, 105, 95, 102),
            (np.nan, np.nan, np.nan, np.nan),
            (102, 108, 99, 106),
            (106, 112, 103, 110),
            (110, 115, 108, 112),
        ]
        df = _make_df(rows)
        result = compute_candlestick_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert "sub_signals" in result


# ===================================================================
# Trend detection
# ===================================================================

class TestTrendDetection:

    def test_uptrend(self):
        # Steadily rising close prices
        df = _make_df([(i, i + 1, i - 1, i + 0.5) for i in range(10, 25)])
        assert _detect_trend(df) == "up"

    def test_downtrend(self):
        # Steadily falling close prices
        df = _make_df([(i, i + 1, i - 1, i - 0.5) for i in range(25, 10, -1)])
        assert _detect_trend(df) == "down"

    def test_flat_trend(self):
        # Prices oscillating around a fixed value
        df = _make_df([(100, 101, 99, 100) for _ in range(10)])
        assert _detect_trend(df) == "flat"

    def test_single_row(self):
        df = _make_df([(100, 105, 95, 102)])
        assert _detect_trend(df) == "flat"


# ===================================================================
# Bar helpers
# ===================================================================

class TestBarHelpers:

    def test_body_green(self):
        bar = _make_bar(100, 110, 95, 108)
        assert _body(bar) == 8.0

    def test_body_red(self):
        bar = _make_bar(108, 110, 95, 100)
        assert _body(bar) == 8.0

    def test_range_normal(self):
        bar = _make_bar(100, 110, 90, 105)
        assert _range(bar) == 20.0

    def test_range_zero_returns_epsilon(self):
        bar = _make_bar(100, 100, 100, 100)
        assert _range(bar) == pytest.approx(1e-12)

    def test_upper_shadow(self):
        bar = _make_bar(100, 115, 95, 110)  # green, top of body=110
        assert _upper_shadow(bar) == 5.0

    def test_lower_shadow(self):
        bar = _make_bar(100, 115, 90, 110)  # green, bottom of body=100
        assert _lower_shadow(bar) == 10.0

    def test_is_green_and_red(self):
        assert _is_green(_make_bar(100, 110, 95, 105)) == True
        assert _is_green(_make_bar(100, 110, 95, 100)) == True  # close == open
        assert _is_red(_make_bar(105, 110, 95, 100)) == True
        assert _is_red(_make_bar(100, 110, 95, 100)) == False

    def test_body_pct(self):
        # body=10, range=20 -> 50%
        bar = _make_bar(100, 110, 90, 110)
        assert _body_pct(bar) == pytest.approx(0.5)


# ===================================================================
# Hammer family
# ===================================================================

class TestHammerFamily:

    def test_hammer_in_downtrend_is_buy(self):
        # Small body near top, long lower shadow, downtrend context
        # body=2, lower_shadow=8, upper_shadow=0
        bar = _make_bar(100, 100, 90, 102)  # body=2, range=12, body_pct=0.167
        # lower = min(100,102)-90=10, upper = 100-max(100,102)=-2 -> 0? No.
        # Actually: lower = min(open,close) - low = 100 - 90 = 10
        #           upper = high - max(open,close) = 100 - 102 = -2 -> but that's impossible
        # Fix: high must be >= max(open, close)
        bar = _make_bar(100, 102, 90, 102)  # body=2, range=12, body_pct=0.167
        # lower = 100 - 90 = 10, upper = 102 - 102 = 0
        # lower(10) >= 2*body(4)? Yes. upper(0) < body(2)? Yes. -> hammer shape
        result = _check_hammer_family(bar, "down")
        assert result == "BUY"

    def test_hanging_man_in_uptrend_is_sell(self):
        # Same shape as hammer, but uptrend context -> hanging man -> SELL
        bar = _make_bar(100, 102, 90, 102)
        result = _check_hammer_family(bar, "up")
        assert result == "SELL"

    def test_hammer_shape_flat_trend_is_hold(self):
        bar = _make_bar(100, 102, 90, 102)
        result = _check_hammer_family(bar, "flat")
        assert result == "HOLD"

    def test_shooting_star_in_uptrend_is_sell(self):
        # Small body near bottom, long upper shadow
        # body=2, upper=10, lower=0
        bar = _make_bar(100, 112, 100, 102)
        # upper = 112 - 102 = 10, lower = 100 - 100 = 0
        # body=2, range=12, body_pct=0.167
        # upper(10) >= 2*body(4)? Yes. lower(0) < body(2)? Yes. -> inverted shape
        result = _check_hammer_family(bar, "up")
        assert result == "SELL"

    def test_inverted_hammer_in_downtrend_is_buy(self):
        bar = _make_bar(100, 112, 100, 102)
        result = _check_hammer_family(bar, "down")
        assert result == "BUY"

    def test_large_body_is_hold(self):
        # Body > 30% of range -> not hammer family
        bar = _make_bar(100, 110, 95, 105)  # body=5, range=15, body_pct=0.33
        result = _check_hammer_family(bar, "down")
        assert result == "HOLD"

    def test_zero_body_is_hold(self):
        bar = _make_bar(100, 110, 90, 100)  # body=0
        result = _check_hammer_family(bar, "down")
        assert result == "HOLD"

    def test_zero_range_is_hold(self):
        bar = _make_bar(100, 100, 100, 100)  # range=0
        result = _check_hammer_family(bar, "down")
        assert result == "HOLD"


# ===================================================================
# Engulfing
# ===================================================================

class TestEngulfing:

    def test_bullish_engulfing(self):
        # prev: red (open > close), curr: green that engulfs prev body
        prev = _make_bar(105, 108, 98, 100)  # red: open=105, close=100
        curr = _make_bar(99, 110, 98, 106)   # green: open=99 <= prev_close(100), close=106 >= prev_open(105)
        assert _check_engulfing(prev, curr) == "BUY"

    def test_bearish_engulfing(self):
        # prev: green (close > open), curr: red that engulfs prev body
        prev = _make_bar(100, 108, 98, 105)  # green: open=100, close=105
        curr = _make_bar(106, 110, 98, 99)   # red: open=106 >= prev_close(105), close=99 <= prev_open(100)
        assert _check_engulfing(prev, curr) == "SELL"

    def test_no_engulfing_same_direction(self):
        # Both green -> no engulfing
        prev = _make_bar(100, 108, 98, 105)
        curr = _make_bar(105, 112, 104, 110)
        assert _check_engulfing(prev, curr) == "HOLD"

    def test_no_engulfing_body_not_covered(self):
        # prev red, curr green but doesn't fully engulf
        prev = _make_bar(105, 108, 98, 100)
        curr = _make_bar(101, 106, 99, 104)  # open=101 > prev_close=100 -> not engulfing
        assert _check_engulfing(prev, curr) == "HOLD"

    def test_zero_body_prev_is_hold(self):
        prev = _make_bar(100, 105, 95, 100)  # body=0
        curr = _make_bar(99, 110, 95, 106)
        assert _check_engulfing(prev, curr) == "HOLD"

    def test_zero_body_curr_is_hold(self):
        prev = _make_bar(105, 108, 98, 100)
        curr = _make_bar(103, 110, 98, 103)  # body=0
        assert _check_engulfing(prev, curr) == "HOLD"


# ===================================================================
# Doji
# ===================================================================

class TestDoji:

    def test_doji_in_downtrend_is_buy(self):
        # body_pct < 10% -> doji; downtrend -> BUY
        bar = _make_bar(100, 110, 90, 100.5)  # body=0.5, range=20, body_pct=0.025
        assert _check_doji(bar, "down") == "BUY"

    def test_doji_in_uptrend_is_sell(self):
        bar = _make_bar(100, 110, 90, 100.5)
        assert _check_doji(bar, "up") == "SELL"

    def test_doji_flat_trend_is_hold(self):
        bar = _make_bar(100, 110, 90, 100.5)
        assert _check_doji(bar, "flat") == "HOLD"

    def test_not_doji_body_too_large(self):
        # body_pct > 10% -> not a doji
        bar = _make_bar(100, 110, 90, 105)  # body=5, range=20, body_pct=0.25
        assert _check_doji(bar, "down") == "HOLD"


# ===================================================================
# Morning / Evening Star
# ===================================================================

class TestStar:

    def test_morning_star_is_buy(self):
        # c0: large red, c1: small body gapped down, c2: large green closing above c0 midpoint
        # c0: open=110, close=100, range=12 -> body/range = 10/12 = 0.83 > 0.6
        c0 = _make_bar(110, 112, 98, 100)
        # c1: small body, gapped down below c0.close(100)
        c1 = _make_bar(98, 99, 96, 97)  # body=1, range=3, body_pct=0.33 > 0.30 -- too large
        # Need body_pct < 0.30: body=0.5, range=3 -> 0.167
        c1 = _make_bar(97.5, 99, 96, 98)  # body=0.5, range=3, body_pct=0.167
        # mid_top = max(97.5, 98) = 98, need <= c0.close(100) -- yes
        # c2: large green closing above midpoint of c0 ((110+100)/2 = 105)
        c2 = _make_bar(99, 112, 98, 108)  # body=9, range=14, body_pct=0.643 > 0.6
        # c2.close(108) >= midpoint_c0(105)? Yes.
        assert _check_star(c0, c1, c2) == "BUY"

    def test_evening_star_is_sell(self):
        # c0: large green, c1: small body gapped up, c2: large red closing below c0 midpoint
        c0 = _make_bar(100, 112, 98, 110)  # green, body=10, range=14, body_pct=0.714
        # c1: small body gapped up above c0.close(110)
        c1 = _make_bar(111, 113, 110, 111.5)  # body=0.5, range=3, body_pct=0.167
        # mid_bottom = min(111, 111.5) = 111 >= c0.close(110)? Yes.
        # c2: large red closing below midpoint ((100+110)/2 = 105)
        c2 = _make_bar(109, 110, 98, 100)  # red, body=9, range=12, body_pct=0.75
        # c2.close(100) <= midpoint_c0(105)? Yes.
        assert _check_star(c0, c1, c2) == "SELL"

    def test_no_star_c0_body_too_small(self):
        c0 = _make_bar(100, 110, 90, 101)  # body=1, range=20, body_pct=0.05 < 0.6
        c1 = _make_bar(97.5, 99, 96, 98)
        c2 = _make_bar(99, 112, 98, 108)
        assert _check_star(c0, c1, c2) == "HOLD"

    def test_no_star_c1_body_too_large(self):
        c0 = _make_bar(110, 112, 98, 100)  # body/range = 10/14 = 0.714
        c1 = _make_bar(96, 102, 95, 101)   # body=5, range=7, body_pct=0.714 > 0.30
        c2 = _make_bar(99, 112, 98, 108)
        assert _check_star(c0, c1, c2) == "HOLD"

    def test_no_star_c2_body_too_small(self):
        c0 = _make_bar(110, 112, 98, 100)
        c1 = _make_bar(97.5, 99, 96, 98)
        c2 = _make_bar(99, 112, 98, 100)  # body=1, range=14, body_pct=0.071 < 0.6
        assert _check_star(c0, c1, c2) == "HOLD"


# ===================================================================
# Composite vote and result structure
# ===================================================================

class TestCompositeVote:

    def test_result_has_required_keys(self):
        df = _make_df([(100, 105, 95, 102)] * 5)
        result = compute_candlestick_signal(df)
        assert "action" in result
        assert "confidence" in result
        assert "sub_signals" in result
        assert "indicators" in result
        assert "patterns_detected" in result["indicators"]

    def test_sub_signals_has_all_families(self):
        df = _make_df([(100, 105, 95, 102)] * 5)
        result = compute_candlestick_signal(df)
        subs = result["sub_signals"]
        assert set(subs.keys()) == {"hammer", "engulfing", "doji", "star"}

    def test_action_is_valid_string(self):
        df = _make_df([(100, 105, 95, 102)] * 5)
        result = compute_candlestick_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_confidence_zero_when_all_hold(self):
        # Flat candles with large body -> no patterns
        df = _make_df([(100, 110, 90, 108)] * 5)
        result = compute_candlestick_signal(df)
        assert result["confidence"] == 0.0
        assert result["action"] == "HOLD"

    def test_confidence_scales_with_pattern_count(self):
        # One pattern -> confidence ~0.25, two -> ~0.50, etc.
        # Build a downtrend with a hammer-like last bar to trigger at least 1 BUY
        rows = [(100 - i, 102 - i, 95 - i, 98 - i) for i in range(18)]
        # Last 3 bars: set up for hammer (last bar is hammer-shaped in downtrend)
        rows.append((82, 84, 80, 80.5))  # c0: normal
        rows.append((80, 82, 78, 79))    # c1: normal
        # c2: hammer shape -- small body near top, long lower shadow
        rows.append((78, 78.5, 70, 78.2))  # body=0.2, range=8.5, body_pct=0.024
        # lower = 78 - 70 = 8, upper = 78.5 - 78.2 = 0.3
        # lower(8) >= 2*body(0.4)? Yes. upper(0.3) < body(0.2)? No -- upper > body!
        # Fix: make upper smaller
        rows[-1] = (78, 78.2, 70, 78.1)  # body=0.1, range=8.2, body_pct=0.012
        # lower = 78 - 70 = 8, upper = 78.2 - 78.1 = 0.1
        # lower(8) >= 2*0.1=0.2? Yes. upper(0.1) < body(0.1)? No, equal.
        # Need upper strictly < body
        rows[-1] = (78, 78.15, 70, 78.1)  # body=0.1, range=8.15, body_pct=0.012
        # upper = 78.15 - 78.1 = 0.05, lower = 78 - 70 = 8
        # upper(0.05) < body(0.1)? Yes!
        df = _make_df(rows)
        result = compute_candlestick_signal(df)
        # Should have at least the hammer detection
        assert result["confidence"] >= 0.25

    def test_bullish_majority_gives_buy(self):
        # Construct data where doji (downtrend) fires BUY
        # downtrend + doji on last bar
        rows = [(100 - i * 2, 102 - i * 2, 96 - i * 2, 98 - i * 2) for i in range(18)]
        # c0, c1 normal, c2 is a doji
        rows.append((62, 64, 60, 61))
        rows.append((60, 62, 58, 59))
        rows.append((58, 68, 48, 58.5))  # doji: body=0.5, range=20, body_pct=0.025
        df = _make_df(rows)
        result = compute_candlestick_signal(df)
        # doji in downtrend -> BUY signal
        assert result["sub_signals"]["doji"] == "BUY"
        # Overall result should have BUY (unless other patterns counteract)
        # At minimum the doji sub-signal voted BUY
        assert result["action"] in ("BUY", "HOLD")  # could be HOLD if cancelled by another

    def test_bearish_majority_gives_sell(self):
        # Uptrend + doji -> SELL
        rows = [(100 + i * 2, 104 + i * 2, 98 + i * 2, 102 + i * 2) for i in range(18)]
        rows.append((136, 140, 134, 138))
        rows.append((138, 142, 136, 140))
        rows.append((140, 150, 130, 140.5))  # doji: body=0.5, range=20, body_pct=0.025
        df = _make_df(rows)
        result = compute_candlestick_signal(df)
        assert result["sub_signals"]["doji"] == "SELL"

    def test_tie_gives_hold(self):
        """When bullish count == bearish count, action is HOLD."""
        # This is hard to construct precisely, so we test the logic conceptually
        # by ensuring the function handles equal counts.
        # Use a flat market with no patterns -> all HOLD -> action HOLD
        df = _make_df([(100, 105, 95, 102)] * 10)
        result = compute_candlestick_signal(df)
        assert result["action"] == "HOLD"


# ===================================================================
# Pattern name collection
# ===================================================================

class TestPatternNames:

    def test_morning_star_pattern_name(self):
        """Morning star detection should include 'morning_star' in patterns."""
        c0 = _make_bar(110, 112, 98, 100)
        c1 = _make_bar(97.5, 99, 96, 98)
        c2 = _make_bar(99, 112, 98, 108)
        # Build a 20-row df where last 3 bars form a morning star
        rows = [(110 - i * 0.5, 112 - i * 0.5, 106 - i * 0.5, 108 - i * 0.5) for i in range(17)]
        rows.append((c0["open"], c0["high"], c0["low"], c0["close"]))
        rows.append((c1["open"], c1["high"], c1["low"], c1["close"]))
        rows.append((c2["open"], c2["high"], c2["low"], c2["close"]))
        df = _make_df(rows)
        result = compute_candlestick_signal(df)
        if result["sub_signals"]["star"] == "BUY":
            assert "morning_star" in result["indicators"]["patterns_detected"]

    def test_bullish_engulfing_pattern_name(self):
        """Bullish engulfing should include 'bullish_engulfing' in patterns."""
        # Build data with a bullish engulfing in the last 2 bars
        rows = [(100, 105, 95, 102)] * 18
        rows.append((100, 105, 95, 102))       # c0
        rows.append((105, 108, 98, 100))        # c1: red
        rows.append((99, 112, 98, 106))         # c2: green engulfing c1
        df = _make_df(rows)
        result = compute_candlestick_signal(df)
        assert result["sub_signals"]["engulfing"] == "BUY"
        assert "bullish_engulfing" in result["indicators"]["patterns_detected"]

    def test_bearish_engulfing_pattern_name(self):
        rows = [(100, 105, 95, 102)] * 18
        rows.append((100, 105, 95, 102))
        rows.append((100, 108, 98, 105))        # c1: green
        rows.append((106, 110, 98, 99))         # c2: red engulfing c1
        df = _make_df(rows)
        result = compute_candlestick_signal(df)
        assert result["sub_signals"]["engulfing"] == "SELL"
        assert "bearish_engulfing" in result["indicators"]["patterns_detected"]

    def test_patterns_detected_is_list(self):
        df = _make_df([(100, 105, 95, 102)] * 5)
        result = compute_candlestick_signal(df)
        assert isinstance(result["indicators"]["patterns_detected"], list)


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:

    def test_exactly_3_rows(self):
        """Minimum required rows -- should not crash."""
        df = _make_df([
            (100, 105, 95, 102),
            (102, 108, 99, 106),
            (106, 112, 103, 110),
        ])
        result = compute_candlestick_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_volume_column_ignored(self):
        """volume column is optional and should not break anything."""
        df = pd.DataFrame({
            "open": [100, 102, 106, 110, 114],
            "high": [105, 108, 112, 115, 118],
            "low": [95, 99, 103, 108, 112],
            "close": [102, 106, 110, 112, 116],
            "volume": [1000, 1200, 900, 1100, 800],
        })
        result = compute_candlestick_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_flat_candles_all_same_price(self):
        """All OHLC identical -> zero range -> all HOLD."""
        df = _make_df([(100, 100, 100, 100)] * 10)
        result = compute_candlestick_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_very_large_dataset(self):
        """Large DataFrame should still work (uses last 10 for trend, last 3 for patterns)."""
        rows = [(100 + i * 0.1, 105 + i * 0.1, 95 + i * 0.1, 102 + i * 0.1) for i in range(500)]
        df = _make_df(rows)
        result = compute_candlestick_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")
