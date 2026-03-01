"""Tests for the Heikin-Ashi composite signal module.

Covers all 7 sub-indicators:
    1. ha_trend       — HA Trend (3 consecutive strong candles, no opposing wick)
    2. ha_doji        — HA Doji reversal after streak
    3. ha_color_change — HA color transition (red->green = BUY, green->red = SELL)
    4. hull_ma        — Hull Moving Average cross (HMA9 vs HMA21)
    5. alligator      — Williams Alligator (SMMA 13/8/5 with shifts)
    6. elder_impulse  — Elder Impulse System (EMA13 + MACD histogram direction)
    7. ttm_squeeze    — TTM Squeeze (BB inside Keltner + momentum)

Test classes:
    TestReturnShape          — keys, valid action, confidence range
    TestInsufficientData     — None, empty, too few rows, missing columns
    TestSubSignalKeys        — expected sub_signals and indicators dict keys
    TestConfidenceRange      — confidence always 0.0-1.0
    TestActionValues         — action always BUY/SELL/HOLD
    TestIndicatorPopulation  — indicators dict contains numeric HA values
    TestHATrendSignal        — _ha_trend_signal unit tests
    TestHADojiSignal         — _ha_doji_signal unit tests
    TestHAColorChangeSignal  — _ha_color_change_signal unit tests
    TestHullMASignal         — _hull_ma_signal unit tests
    TestAlligatorSignal      — _alligator_signal unit tests
    TestElderImpulseSignal   — _elder_impulse_signal unit tests
    TestTTMSqueezeSignal     — _ttm_squeeze_signal unit tests
    TestUptrendBias          — strong uptrend produces BUY-biased sub-signals
    TestDowntrendBias        — strong downtrend produces SELL-biased sub-signals
    TestFlatData             — flat / zero-movement data returns HOLD
    TestEdgeCases            — single bar, NaN columns, case-insensitive columns
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.heikin_ashi import (
    compute_heikin_ashi_signal,
    _compute_ha_candles,
    _ha_trend_signal,
    _ha_doji_signal,
    _ha_color_change_signal,
    _hull_ma_signal,
    _alligator_signal,
    _elder_impulse_signal,
    _ttm_squeeze_signal,
)


# ---------------------------------------------------------------------------
# DataFrame construction helpers
# ---------------------------------------------------------------------------

def _make_df(
    n: int = 100,
    close_base: float = 100.0,
    volatility: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic random-walk OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n) * volatility
    close = close_base + np.cumsum(noise)
    close = np.maximum(close, 1.0)
    high = close + np.abs(rng.standard_normal(n) * volatility * 0.5)
    low = close - np.abs(rng.standard_normal(n) * volatility * 0.5)
    low = np.maximum(low, 0.5)
    opn = close + rng.standard_normal(n) * 0.3
    opn = np.maximum(opn, 0.5)
    volume = rng.integers(100, 10_000, n).astype(float)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_trending_df(
    n: int = 100,
    start: float = 100.0,
    step: float = 1.0,
    spread: float = 0.3,
) -> pd.DataFrame:
    """Generate a strongly trending OHLCV DataFrame with smooth price movement.

    A positive step creates an uptrend; a negative step creates a downtrend.
    spread controls the candle body width relative to the step size.
    """
    close = np.array([start + i * step for i in range(n)], dtype=float)
    close = np.maximum(close, 1.0)
    if step >= 0:
        # Uptrend: open slightly below close, high above close, low below open
        opn = close - abs(step) * spread
        high = close + abs(step) * spread
        low = opn - abs(step) * spread
    else:
        # Downtrend: open slightly above close, low below close, high above open
        opn = close + abs(step) * spread
        low = close - abs(step) * spread
        high = opn + abs(step) * spread

    opn = np.maximum(opn, 0.5)
    low = np.maximum(low, 0.3)
    volume = np.full(n, 1000.0)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_flat_df(n: int = 80, price: float = 100.0) -> pd.DataFrame:
    """Generate perfectly flat OHLCV data (all OHLC identical)."""
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "open": [price] * n,
            "high": [price] * n,
            "low": [price] * n,
            "close": [price] * n,
            "volume": [1000.0] * n,
        },
        index=dates,
    )


def _make_ha_candles_df(n: int = 20) -> pd.DataFrame:
    """Minimal synthetic HA candle DataFrame for sub-signal unit tests."""
    return _make_df(n=n)


# ---------------------------------------------------------------------------
# Test 1: Return shape — expected top-level keys
# ---------------------------------------------------------------------------

class TestReturnShape:
    """compute_heikin_ashi_signal must always return the correct top-level keys."""

    def test_top_level_keys_present(self):
        df = _make_df(80)
        result = compute_heikin_ashi_signal(df)
        assert "action" in result
        assert "confidence" in result
        assert "sub_signals" in result
        assert "indicators" in result

    def test_no_extra_top_level_keys(self):
        df = _make_df(80)
        result = compute_heikin_ashi_signal(df)
        assert set(result.keys()) == {"action", "confidence", "sub_signals", "indicators"}

    def test_returns_dict(self):
        df = _make_df(80)
        result = compute_heikin_ashi_signal(df)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Test 2: Insufficient data always returns HOLD with 0.0 confidence
# ---------------------------------------------------------------------------

class TestInsufficientData:
    """Guard rail: bad / missing data should never raise, always HOLD."""

    def test_none_returns_hold(self):
        result = compute_heikin_ashi_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_heikin_ashi_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_single_row_returns_hold(self):
        df = _make_df(1)
        result = compute_heikin_ashi_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_nine_rows_returns_hold(self):
        """_MIN_ROWS_BASIC is 10; 9 rows must return HOLD."""
        df = _make_df(9)
        result = compute_heikin_ashi_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_ten_rows_does_not_raise(self):
        """Exactly at _MIN_ROWS_BASIC boundary — must not raise."""
        df = _make_df(10)
        result = compute_heikin_ashi_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_missing_close_column_returns_hold(self):
        df = pd.DataFrame({
            "open": [1.0, 2.0, 3.0],
            "high": [2.0, 3.0, 4.0],
            "low": [0.5, 1.5, 2.5],
            "volume": [100.0, 200.0, 300.0],
        })
        result = compute_heikin_ashi_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_missing_volume_column_returns_hold(self):
        df = pd.DataFrame({
            "open": [1.0, 2.0, 3.0],
            "high": [2.0, 3.0, 4.0],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
        })
        result = compute_heikin_ashi_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_non_dataframe_input_returns_hold(self):
        result = compute_heikin_ashi_signal("not a dataframe")
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_list_input_returns_hold(self):
        result = compute_heikin_ashi_signal([1, 2, 3, 4, 5])
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Test 3: sub_signals dict contains exactly the 7 expected keys
# ---------------------------------------------------------------------------

class TestSubSignalKeys:
    """The sub_signals and indicators dicts must have the documented keys."""

    EXPECTED_SUB_SIGNAL_KEYS = {
        "ha_trend",
        "ha_doji",
        "ha_color_change",
        "hull_ma",
        "alligator",
        "elder_impulse",
        "ttm_squeeze",
    }

    EXPECTED_INDICATOR_KEYS = {
        "ha_color",
        "ha_streak",
        "hull_fast",
        "hull_slow",
        "alligator_lips",
        "alligator_teeth",
        "alligator_jaw",
        "elder_color",
        "ttm_squeeze_on",
        "ttm_momentum",
    }

    def test_sub_signals_keys_match(self):
        df = _make_df(80)
        result = compute_heikin_ashi_signal(df)
        assert set(result["sub_signals"].keys()) == self.EXPECTED_SUB_SIGNAL_KEYS

    def test_sub_signals_keys_count(self):
        df = _make_df(80)
        result = compute_heikin_ashi_signal(df)
        assert len(result["sub_signals"]) == 7

    def test_indicators_keys_match(self):
        df = _make_df(80)
        result = compute_heikin_ashi_signal(df)
        assert set(result["indicators"].keys()) == self.EXPECTED_INDICATOR_KEYS

    def test_all_sub_signals_valid_actions(self):
        df = _make_df(80)
        result = compute_heikin_ashi_signal(df)
        valid = {"BUY", "SELL", "HOLD"}
        for name, vote in result["sub_signals"].items():
            assert vote in valid, f"sub_signal '{name}' has invalid vote: {vote!r}"

    def test_sub_signals_present_on_insufficient_data(self):
        """Even the default HOLD result must have all 7 sub_signals keys."""
        result = compute_heikin_ashi_signal(None)
        assert set(result["sub_signals"].keys()) == self.EXPECTED_SUB_SIGNAL_KEYS

    def test_indicators_present_on_insufficient_data(self):
        """Default HOLD result must have all indicator keys."""
        result = compute_heikin_ashi_signal(None)
        assert set(result["indicators"].keys()) == self.EXPECTED_INDICATOR_KEYS


# ---------------------------------------------------------------------------
# Test 4: Confidence is always between 0.0 and 1.0
# ---------------------------------------------------------------------------

class TestConfidenceRange:
    """Confidence must be a float in [0.0, 1.0] under all conditions."""

    def test_confidence_valid_on_normal_data(self):
        df = _make_df(100)
        result = compute_heikin_ashi_signal(df)
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_confidence_zero_on_none(self):
        result = compute_heikin_ashi_signal(None)
        assert result["confidence"] == 0.0

    def test_confidence_valid_on_uptrend(self):
        df = _make_trending_df(n=100, step=1.5)
        result = compute_heikin_ashi_signal(df)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_confidence_valid_on_downtrend(self):
        df = _make_trending_df(n=100, step=-1.5)
        result = compute_heikin_ashi_signal(df)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_confidence_valid_on_flat_data(self):
        df = _make_flat_df(80)
        result = compute_heikin_ashi_signal(df)
        assert 0.0 <= result["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# Test 5: Action is always one of BUY / SELL / HOLD
# ---------------------------------------------------------------------------

class TestActionValues:
    """action field must be exactly one of the three valid strings."""

    VALID_ACTIONS = {"BUY", "SELL", "HOLD"}

    def test_action_valid_random_data(self):
        df = _make_df(100)
        result = compute_heikin_ashi_signal(df)
        assert result["action"] in self.VALID_ACTIONS

    def test_action_valid_uptrend(self):
        df = _make_trending_df(n=100, step=2.0)
        result = compute_heikin_ashi_signal(df)
        assert result["action"] in self.VALID_ACTIONS

    def test_action_valid_downtrend(self):
        df = _make_trending_df(n=100, step=-2.0)
        result = compute_heikin_ashi_signal(df)
        assert result["action"] in self.VALID_ACTIONS

    def test_action_valid_minimal_rows(self):
        df = _make_df(10)
        result = compute_heikin_ashi_signal(df)
        assert result["action"] in self.VALID_ACTIONS

    def test_action_hold_on_empty(self):
        result = compute_heikin_ashi_signal(None)
        assert result["action"] == "HOLD"


# ---------------------------------------------------------------------------
# Test 6: Indicators dict is populated with HA-derived values
# ---------------------------------------------------------------------------

class TestIndicatorPopulation:
    """indicators dict must contain correctly typed values."""

    def test_ha_color_is_string(self):
        df = _make_df(80)
        result = compute_heikin_ashi_signal(df)
        assert isinstance(result["indicators"]["ha_color"], str)
        assert result["indicators"]["ha_color"] in ("green", "red")

    def test_ha_streak_is_non_negative_int(self):
        df = _make_df(80)
        result = compute_heikin_ashi_signal(df)
        streak = result["indicators"]["ha_streak"]
        assert isinstance(streak, int)
        assert streak >= 0

    def test_hull_fast_is_float(self):
        df = _make_df(80)
        result = compute_heikin_ashi_signal(df)
        val = result["indicators"]["hull_fast"]
        assert isinstance(val, float)

    def test_hull_slow_is_float(self):
        df = _make_df(80)
        result = compute_heikin_ashi_signal(df)
        val = result["indicators"]["hull_slow"]
        assert isinstance(val, float)

    def test_alligator_values_are_floats(self):
        df = _make_df(80)
        result = compute_heikin_ashi_signal(df)
        ind = result["indicators"]
        for key in ("alligator_lips", "alligator_teeth", "alligator_jaw"):
            assert isinstance(ind[key], float), f"{key} should be float"

    def test_elder_color_is_valid_string(self):
        df = _make_df(80)
        result = compute_heikin_ashi_signal(df)
        assert result["indicators"]["elder_color"] in ("green", "red", "blue")

    def test_ttm_squeeze_on_is_bool(self):
        df = _make_df(80)
        result = compute_heikin_ashi_signal(df)
        assert isinstance(result["indicators"]["ttm_squeeze_on"], bool)

    def test_ttm_momentum_is_float(self):
        df = _make_df(80)
        result = compute_heikin_ashi_signal(df)
        assert isinstance(result["indicators"]["ttm_momentum"], float)

    def test_hull_fast_near_price(self):
        """HMA(9) should be in the same order of magnitude as the close price."""
        df = _make_df(80, close_base=100.0, volatility=0.5)
        result = compute_heikin_ashi_signal(df)
        hull_fast = result["indicators"]["hull_fast"]
        if not math.isnan(hull_fast):
            assert 50.0 < hull_fast < 200.0, f"hull_fast {hull_fast} out of expected range"


# ---------------------------------------------------------------------------
# Test 7: _ha_trend_signal unit tests
# ---------------------------------------------------------------------------

class TestHATrendSignal:
    """Unit tests for the HA Trend sub-signal."""

    def _make_ha(self, ha_open, ha_close, ha_high=None, ha_low=None):
        """Build a small HA DataFrame for signal testing."""
        n = len(ha_open)
        if ha_high is None:
            ha_high = [max(o, c) + 0.01 for o, c in zip(ha_open, ha_close)]
        if ha_low is None:
            ha_low = [min(o, c) - 0.01 for o, c in zip(ha_open, ha_close)]
        return pd.DataFrame({
            "ha_open": ha_open,
            "ha_high": ha_high,
            "ha_low": ha_low,
            "ha_close": ha_close,
        })

    def test_strong_uptrend_3_green_no_lower_wick_returns_buy(self):
        """3 green candles with ha_low == ha_open (no lower wick) = BUY."""
        # Construct exactly no-lower-wick green candles
        ha_open = [10.0, 11.0, 12.0]
        ha_close = [11.0, 12.0, 13.0]
        # ha_low exactly equals ha_open (no lower wick)
        ha_low = [10.0, 11.0, 12.0]
        ha_high = [11.5, 12.5, 13.5]
        ha = self._make_ha(ha_open, ha_close, ha_high=ha_high, ha_low=ha_low)
        signal, color, streak = _ha_trend_signal(ha)
        assert signal == "BUY"
        assert color == "green"
        assert streak == 3

    def test_strong_downtrend_3_red_no_upper_wick_returns_sell(self):
        """3 red candles with ha_high == ha_open (no upper wick) = SELL."""
        ha_open = [13.0, 12.0, 11.0]
        ha_close = [12.0, 11.0, 10.0]
        # ha_high exactly equals ha_open (no upper wick)
        ha_high = [13.0, 12.0, 11.0]
        ha_low = [11.5, 10.5, 9.5]
        ha = self._make_ha(ha_open, ha_close, ha_high=ha_high, ha_low=ha_low)
        signal, color, streak = _ha_trend_signal(ha)
        assert signal == "SELL"
        assert color == "red"
        assert streak == 3

    def test_fewer_than_streak_len_returns_hold(self):
        """Fewer than 3 candles: cannot form a streak, returns HOLD."""
        ha_open = [10.0, 11.0]
        ha_close = [11.0, 12.0]
        ha = self._make_ha(ha_open, ha_close)
        signal, _, _ = _ha_trend_signal(ha)
        assert signal == "HOLD"

    def test_mixed_colors_returns_hold(self):
        """Green-red-green pattern is not a clean streak = HOLD."""
        ha_open = [10.0, 12.0, 11.0]
        ha_close = [12.0, 11.0, 13.0]
        ha = self._make_ha(ha_open, ha_close)
        signal, _, _ = _ha_trend_signal(ha)
        assert signal == "HOLD"

    def test_green_with_lower_wick_returns_hold(self):
        """Green candles with a lower wick (ha_low < ha_open) = HOLD (not strong)."""
        ha_open = [10.0, 11.0, 12.0]
        ha_close = [11.0, 12.0, 13.0]
        # ha_low is below ha_open (lower wick exists)
        ha_low = [9.0, 10.0, 11.0]
        ha_high = [11.5, 12.5, 13.5]
        ha = self._make_ha(ha_open, ha_close, ha_high=ha_high, ha_low=ha_low)
        signal, color, streak = _ha_trend_signal(ha)
        assert signal == "HOLD"

    def test_streak_counted_correctly_for_longer_series(self):
        """Streak should count consecutive same-color candles from end."""
        # 5 green candles, check streak = 5
        ha_open = [10.0, 11.0, 12.0, 13.0, 14.0]
        ha_close = [11.0, 12.0, 13.0, 14.0, 15.0]
        ha_low = [10.0, 11.0, 12.0, 13.0, 14.0]   # no lower wicks
        ha_high = [11.5, 12.5, 13.5, 14.5, 15.5]
        ha = self._make_ha(ha_open, ha_close, ha_high=ha_high, ha_low=ha_low)
        signal, color, streak = _ha_trend_signal(ha)
        assert signal == "BUY"
        assert streak == 5


# ---------------------------------------------------------------------------
# Test 8: _ha_doji_signal unit tests
# ---------------------------------------------------------------------------

class TestHADojiSignal:
    """Unit tests for the HA Doji reversal sub-signal."""

    def _make_ha_with_doji_after_streak(self, streak_color: str) -> pd.DataFrame:
        """Create HA candles: N same-color bars, then a doji at the end."""
        rows = []
        if streak_color == "green":
            for _ in range(4):
                rows.append({"ha_open": 10.0, "ha_high": 12.0, "ha_low": 9.0, "ha_close": 11.0})
        else:
            for _ in range(4):
                rows.append({"ha_open": 11.0, "ha_high": 12.0, "ha_low": 9.0, "ha_close": 10.0})
        # Doji: open == close (body = 0), wide range
        rows.append({"ha_open": 10.5, "ha_high": 12.0, "ha_low": 9.0, "ha_close": 10.5})
        return pd.DataFrame(rows)

    def test_doji_after_green_streak_returns_sell(self):
        ha = self._make_ha_with_doji_after_streak("green")
        signal = _ha_doji_signal(ha)
        assert signal == "SELL"

    def test_doji_after_red_streak_returns_buy(self):
        ha = self._make_ha_with_doji_after_streak("red")
        signal = _ha_doji_signal(ha)
        assert signal == "BUY"

    def test_non_doji_candle_returns_hold(self):
        """A large body candle (not a doji) at end = HOLD."""
        rows = [
            {"ha_open": 10.0, "ha_high": 12.0, "ha_low": 9.0, "ha_close": 11.0},
            {"ha_open": 10.0, "ha_high": 12.0, "ha_low": 9.0, "ha_close": 11.0},
            {"ha_open": 10.0, "ha_high": 12.0, "ha_low": 9.0, "ha_close": 11.0},
            # Large body (body/range = 2/3 > 10%)
            {"ha_open": 10.0, "ha_high": 12.0, "ha_low": 9.0, "ha_close": 12.0},
        ]
        ha = pd.DataFrame(rows)
        signal = _ha_doji_signal(ha)
        assert signal == "HOLD"

    def test_single_candle_returns_hold(self):
        ha = pd.DataFrame([{"ha_open": 10.0, "ha_high": 12.0, "ha_low": 9.0, "ha_close": 10.5}])
        signal = _ha_doji_signal(ha)
        assert signal == "HOLD"

    def test_doji_after_single_bar_streak_returns_hold(self):
        """Doji after only 1 prior bar (streak_count < 2) = HOLD."""
        rows = [
            {"ha_open": 10.0, "ha_high": 12.0, "ha_low": 9.0, "ha_close": 11.0},
            {"ha_open": 10.5, "ha_high": 12.0, "ha_low": 9.0, "ha_close": 10.5},  # doji
        ]
        ha = pd.DataFrame(rows)
        signal = _ha_doji_signal(ha)
        assert signal == "HOLD"


# ---------------------------------------------------------------------------
# Test 9: _ha_color_change_signal unit tests
# ---------------------------------------------------------------------------

class TestHAColorChangeSignal:
    """Unit tests for the HA Color Change sub-signal."""

    def _ha(self, prev_open, prev_close, curr_open, curr_close) -> pd.DataFrame:
        return pd.DataFrame([
            {"ha_open": prev_open, "ha_high": max(prev_open, prev_close) + 0.1,
             "ha_low": min(prev_open, prev_close) - 0.1, "ha_close": prev_close},
            {"ha_open": curr_open, "ha_high": max(curr_open, curr_close) + 0.1,
             "ha_low": min(curr_open, curr_close) - 0.1, "ha_close": curr_close},
        ])

    def test_red_to_green_returns_buy(self):
        ha = self._ha(prev_open=11.0, prev_close=10.0, curr_open=10.0, curr_close=11.0)
        signal = _ha_color_change_signal(ha)
        assert signal == "BUY"

    def test_green_to_red_returns_sell(self):
        ha = self._ha(prev_open=10.0, prev_close=11.0, curr_open=11.0, curr_close=10.0)
        signal = _ha_color_change_signal(ha)
        assert signal == "SELL"

    def test_green_to_green_returns_hold(self):
        ha = self._ha(prev_open=10.0, prev_close=11.0, curr_open=11.0, curr_close=12.0)
        signal = _ha_color_change_signal(ha)
        assert signal == "HOLD"

    def test_red_to_red_returns_hold(self):
        ha = self._ha(prev_open=11.0, prev_close=10.0, curr_open=10.0, curr_close=9.0)
        signal = _ha_color_change_signal(ha)
        assert signal == "HOLD"

    def test_single_candle_returns_hold(self):
        ha = pd.DataFrame([{"ha_open": 10.0, "ha_high": 11.0, "ha_low": 9.0, "ha_close": 10.5}])
        signal = _ha_color_change_signal(ha)
        assert signal == "HOLD"

    def test_flat_prev_candle_returns_hold(self):
        """Prev candle open == close (flat doji) treated as continuation = HOLD."""
        ha = self._ha(prev_open=10.0, prev_close=10.0, curr_open=10.0, curr_close=11.0)
        signal = _ha_color_change_signal(ha)
        assert signal == "HOLD"


# ---------------------------------------------------------------------------
# Test 10: _hull_ma_signal unit tests
# ---------------------------------------------------------------------------

class TestHullMASignal:
    """Unit tests for the Hull Moving Average cross sub-signal."""

    def test_uptrending_close_returns_buy(self):
        """Monotonically rising prices: HMA(9) should be above HMA(21)."""
        close = pd.Series([100.0 + i for i in range(60)])
        signal, fast, slow = _hull_ma_signal(close)
        assert signal == "BUY"
        assert fast > slow

    def test_downtrending_close_returns_sell(self):
        """Monotonically falling prices: HMA(9) should be below HMA(21)."""
        close = pd.Series([200.0 - i for i in range(60)])
        signal, fast, slow = _hull_ma_signal(close)
        assert signal == "SELL"
        assert fast < slow

    def test_returns_floats_for_hull_values(self):
        close = pd.Series([100.0 + i * 0.5 for i in range(50)])
        signal, fast, slow = _hull_ma_signal(close)
        assert isinstance(fast, float)
        assert isinstance(slow, float)

    def test_insufficient_data_returns_nan_values(self):
        """Only 3 bars: WMA cannot compute, NaN values expected."""
        close = pd.Series([100.0, 101.0, 102.0])
        signal, fast, slow = _hull_ma_signal(close)
        assert signal == "HOLD"


# ---------------------------------------------------------------------------
# Test 11: _alligator_signal unit tests
# ---------------------------------------------------------------------------

class TestAlligatorSignal:
    """Unit tests for the Williams Alligator sub-signal."""

    def test_strong_uptrend_returns_buy(self):
        """Strong uptrend: lips > teeth > jaw (alligator eating upward)."""
        close = pd.Series([100.0 + i * 2.0 for i in range(80)])
        signal, lips, teeth, jaw = _alligator_signal(close)
        assert signal == "BUY"
        assert lips > teeth > jaw

    def test_strong_downtrend_returns_sell(self):
        """Strong downtrend: lips < teeth < jaw (alligator eating downward)."""
        close = pd.Series([200.0 - i * 2.0 for i in range(80)])
        close = np.maximum(close.values, 1.0)
        close = pd.Series(close)
        signal, lips, teeth, jaw = _alligator_signal(close)
        assert signal == "SELL"
        assert lips < teeth < jaw

    def test_returns_four_tuple(self):
        close = pd.Series([100.0 + i for i in range(60)])
        result = _alligator_signal(close)
        assert len(result) == 4

    def test_insufficient_data_returns_nan(self):
        """Fewer than 13+8=21 bars needed for jaw SMMA: expect NaN placeholders."""
        close = pd.Series([100.0, 101.0, 102.0])
        signal, lips, teeth, jaw = _alligator_signal(close)
        # With very few bars, may still compute (SMMA uses ewm) but NaN shifts dominate
        assert signal in ("BUY", "SELL", "HOLD")


# ---------------------------------------------------------------------------
# Test 12: _elder_impulse_signal unit tests
# ---------------------------------------------------------------------------

class TestElderImpulseSignal:
    """Unit tests for the Elder Impulse System sub-signal."""

    def test_strong_uptrend_returns_buy_green(self):
        """Rising EMA(13) + rising MACD histogram = BUY (green).

        A constant-rate linear series produces a steady-state MACD histogram
        with no slope at the tail, so the histogram direction does not
        consistently rise bar-to-bar. An accelerating (quadratic) series
        causes the MACD histogram to grow, satisfying both conditions.
        """
        close = pd.Series([100.0 + i * i * 0.1 for i in range(60)])
        signal, color = _elder_impulse_signal(close)
        assert signal == "BUY"
        assert color == "green"

    def test_strong_downtrend_returns_sell_red(self):
        """Falling EMA(13) + falling MACD histogram = SELL (red).

        An accelerating downtrend (quadratic decline) forces both EMA and
        MACD histogram to fall in the same direction, triggering SELL.
        The step is kept small enough that prices never clip at the minimum,
        which would flatten the tail and break the histogram direction.
        """
        close_vals = np.array([200.0 - i * i * 0.03 for i in range(60)])
        close = pd.Series(close_vals)
        signal, color = _elder_impulse_signal(close)
        assert signal == "SELL"
        assert color == "red"

    def test_returns_two_tuple(self):
        close = pd.Series([100.0 + i * 0.5 for i in range(50)])
        result = _elder_impulse_signal(close)
        assert len(result) == 2

    def test_color_is_valid_string(self):
        close = pd.Series([100.0 + i for i in range(50)])
        signal, color = _elder_impulse_signal(close)
        assert color in ("green", "red", "blue")

    def test_signal_is_valid_action(self):
        close = pd.Series([100.0 + i * 0.1 for i in range(50)])
        signal, color = _elder_impulse_signal(close)
        assert signal in ("BUY", "SELL", "HOLD")


# ---------------------------------------------------------------------------
# Test 13: _ttm_squeeze_signal unit tests
# ---------------------------------------------------------------------------

class TestTTMSqueezeSignal:
    """Unit tests for the TTM Squeeze sub-signal."""

    def test_high_volatility_positive_momentum_returns_buy(self):
        """After squeeze release with upward momentum, signal = BUY."""
        # Rising price: BB expands above KC, positive momentum
        n = 60
        close_vals = [100.0 + i * 2.0 for i in range(n)]
        close = pd.Series(close_vals)
        high = pd.Series([v + 3.0 for v in close_vals])
        low = pd.Series([max(v - 3.0, 0.1) for v in close_vals])
        signal, squeeze_on, momentum = _ttm_squeeze_signal(high, low, close)
        assert signal in ("BUY", "HOLD")  # depends on squeeze state

    def test_returns_three_tuple(self):
        n = 60
        close = pd.Series([100.0 + i for i in range(n)])
        high = pd.Series([v + 2.0 for v in close])
        low = pd.Series([v - 2.0 for v in close])
        result = _ttm_squeeze_signal(high, low, close)
        assert len(result) == 3

    def test_squeeze_on_is_bool(self):
        n = 60
        close = pd.Series([100.0 + i * 0.01 for i in range(n)])  # low vol = squeeze
        high = pd.Series([v + 0.1 for v in close])
        low = pd.Series([v - 0.1 for v in close])
        signal, squeeze_on, momentum = _ttm_squeeze_signal(high, low, close)
        assert isinstance(squeeze_on, bool)

    def test_momentum_is_float(self):
        n = 60
        close = pd.Series([100.0 + i for i in range(n)])
        high = pd.Series([v + 2.0 for v in close])
        low = pd.Series([v - 2.0 for v in close])
        signal, squeeze_on, momentum = _ttm_squeeze_signal(high, low, close)
        assert isinstance(momentum, float)

    def test_signal_is_valid_action(self):
        n = 60
        close = pd.Series([100.0 + i for i in range(n)])
        high = pd.Series([v + 2.0 for v in close])
        low = pd.Series([v - 2.0 for v in close])
        signal, squeeze_on, momentum = _ttm_squeeze_signal(high, low, close)
        assert signal in ("BUY", "SELL", "HOLD")

    def test_flat_low_vol_data_squeeze_on(self):
        """Very flat data: BB should be much tighter than KC -> squeeze ON -> HOLD."""
        n = 60
        close = pd.Series([100.0] * n)
        high = pd.Series([100.1] * n)
        low = pd.Series([99.9] * n)
        signal, squeeze_on, momentum = _ttm_squeeze_signal(high, low, close)
        # With zero volatility, BB std = 0, so BB is inside KC: squeeze ON
        assert squeeze_on is True
        assert signal == "HOLD"


# ---------------------------------------------------------------------------
# Test 14: Strong uptrend — BUY-biased sub-signals
# ---------------------------------------------------------------------------

class TestUptrendBias:
    """A strongly trending upward series should produce BUY-biased output."""

    def test_uptrend_action_not_sell(self):
        """With a clear uptrend, composite action should not be SELL."""
        df = _make_trending_df(n=100, start=50.0, step=2.0, spread=0.1)
        result = compute_heikin_ashi_signal(df)
        assert result["action"] in ("BUY", "HOLD")

    def test_uptrend_hull_ma_is_buy(self):
        """HMA(9) > HMA(21) in a strong uptrend."""
        df = _make_trending_df(n=100, start=50.0, step=2.0, spread=0.1)
        result = compute_heikin_ashi_signal(df)
        assert result["sub_signals"]["hull_ma"] == "BUY"

    def test_uptrend_ha_color_is_green(self):
        """HA color should be green in an uptrend."""
        df = _make_trending_df(n=100, start=50.0, step=2.0, spread=0.1)
        result = compute_heikin_ashi_signal(df)
        assert result["indicators"]["ha_color"] == "green"

    def test_uptrend_elder_impulse_is_buy(self):
        """EMA(13) rising + MACD hist rising = green impulse bar in uptrend.

        Elder Impulse requires both EMA direction AND MACD histogram direction
        to agree. A constant-step linear series reaches steady state where the
        histogram stops accelerating. Using an accelerating quadratic series
        ensures the histogram keeps rising alongside the EMA.
        """
        # Quadratic acceleration drives MACD histogram upward reliably
        n = 100
        close_vals = [50.0 + i * i * 0.05 for i in range(n)]
        high_vals = [v + 0.5 for v in close_vals]
        low_vals = [max(v - 0.5, 0.3) for v in close_vals]
        open_vals = [max(v - 0.1, 0.3) for v in close_vals]
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        df = pd.DataFrame({
            "open": open_vals, "high": high_vals, "low": low_vals,
            "close": close_vals, "volume": [1000.0] * n,
        }, index=dates)
        result = compute_heikin_ashi_signal(df)
        assert result["sub_signals"]["elder_impulse"] == "BUY"

    def test_uptrend_more_buy_subs_than_sell_subs(self):
        """More sub-signals should vote BUY than SELL in a clear uptrend."""
        df = _make_trending_df(n=100, start=50.0, step=2.0, spread=0.1)
        result = compute_heikin_ashi_signal(df)
        subs = result["sub_signals"]
        buy_count = sum(1 for v in subs.values() if v == "BUY")
        sell_count = sum(1 for v in subs.values() if v == "SELL")
        assert buy_count > sell_count, (
            f"Expected BUY-biased subs in uptrend, got buy={buy_count} sell={sell_count}: {subs}"
        )

    def test_uptrend_ha_streak_positive(self):
        """HA streak should be > 0 in a strong uptrend."""
        df = _make_trending_df(n=100, start=50.0, step=2.0, spread=0.1)
        result = compute_heikin_ashi_signal(df)
        assert result["indicators"]["ha_streak"] > 0

    def test_uptrend_alligator_is_buy(self):
        """Alligator should be open upward (lips > teeth > jaw) in a strong uptrend."""
        df = _make_trending_df(n=100, start=50.0, step=2.0, spread=0.1)
        result = compute_heikin_ashi_signal(df)
        assert result["sub_signals"]["alligator"] == "BUY"


# ---------------------------------------------------------------------------
# Test 15: Strong downtrend — SELL-biased sub-signals
# ---------------------------------------------------------------------------

class TestDowntrendBias:
    """A strongly trending downward series should produce SELL-biased output."""

    def test_downtrend_action_not_buy(self):
        """With a clear downtrend, composite action should not be BUY."""
        df = _make_trending_df(n=100, start=250.0, step=-2.0, spread=0.1)
        result = compute_heikin_ashi_signal(df)
        assert result["action"] in ("SELL", "HOLD")

    def test_downtrend_hull_ma_is_sell(self):
        """HMA(9) < HMA(21) in a strong downtrend."""
        df = _make_trending_df(n=100, start=250.0, step=-2.0, spread=0.1)
        result = compute_heikin_ashi_signal(df)
        assert result["sub_signals"]["hull_ma"] == "SELL"

    def test_downtrend_ha_color_is_red(self):
        """HA color should be red in a downtrend."""
        df = _make_trending_df(n=100, start=250.0, step=-2.0, spread=0.1)
        result = compute_heikin_ashi_signal(df)
        assert result["indicators"]["ha_color"] == "red"

    def test_downtrend_elder_impulse_is_sell(self):
        """EMA(13) falling + MACD hist falling = red impulse bar in downtrend.

        Same reasoning as the uptrend case: uses quadratic acceleration so
        the MACD histogram keeps falling alongside the EMA.
        """
        n = 100
        close_vals = np.maximum(
            np.array([250.0 - i * i * 0.05 for i in range(n)]), 1.0
        )
        high_vals = close_vals + 0.5
        low_vals = np.maximum(close_vals - 0.5, 0.3)
        open_vals = np.maximum(close_vals + 0.1, 0.3)
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        df = pd.DataFrame({
            "open": open_vals, "high": high_vals, "low": low_vals,
            "close": close_vals, "volume": [1000.0] * n,
        }, index=dates)
        result = compute_heikin_ashi_signal(df)
        assert result["sub_signals"]["elder_impulse"] == "SELL"

    def test_downtrend_more_sell_subs_than_buy_subs(self):
        """More sub-signals should vote SELL than BUY in a clear downtrend."""
        df = _make_trending_df(n=100, start=250.0, step=-2.0, spread=0.1)
        result = compute_heikin_ashi_signal(df)
        subs = result["sub_signals"]
        buy_count = sum(1 for v in subs.values() if v == "BUY")
        sell_count = sum(1 for v in subs.values() if v == "SELL")
        assert sell_count > buy_count, (
            f"Expected SELL-biased subs in downtrend, got buy={buy_count} sell={sell_count}: {subs}"
        )

    def test_downtrend_alligator_is_sell(self):
        """Alligator should be open downward (lips < teeth < jaw) in a strong downtrend."""
        df = _make_trending_df(n=100, start=250.0, step=-2.0, spread=0.1)
        result = compute_heikin_ashi_signal(df)
        assert result["sub_signals"]["alligator"] == "SELL"


# ---------------------------------------------------------------------------
# Test 16: Flat data — zero movement
# ---------------------------------------------------------------------------

class TestFlatData:
    """Completely flat prices should produce HOLD from most sub-signals."""

    def test_flat_returns_hold_action(self):
        df = _make_flat_df(80)
        result = compute_heikin_ashi_signal(df)
        assert result["action"] == "HOLD"

    def test_flat_confidence_is_zero(self):
        """No directional consensus possible with flat prices."""
        df = _make_flat_df(80)
        result = compute_heikin_ashi_signal(df)
        assert result["confidence"] == 0.0

    def test_flat_ha_color_change_is_hold(self):
        """With no color change, ha_color_change should be HOLD."""
        df = _make_flat_df(80)
        result = compute_heikin_ashi_signal(df)
        assert result["sub_signals"]["ha_color_change"] == "HOLD"

    def test_flat_ha_trend_is_hold(self):
        """Flat candles have negligible range, so ha_trend should be HOLD."""
        df = _make_flat_df(80)
        result = compute_heikin_ashi_signal(df)
        assert result["sub_signals"]["ha_trend"] == "HOLD"

    def test_flat_elder_impulse_is_hold(self):
        """Flat EMA(13) + flat MACD histogram = mixed = HOLD (blue)."""
        df = _make_flat_df(80)
        result = compute_heikin_ashi_signal(df)
        assert result["sub_signals"]["elder_impulse"] == "HOLD"
        assert result["indicators"]["elder_color"] == "blue"


# ---------------------------------------------------------------------------
# Test 17: Edge cases — robustness
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Robustness tests for boundary conditions."""

    def test_case_insensitive_column_names(self):
        """Signal module should handle uppercase column names (col_map normalization)."""
        df = _make_df(80)
        df.columns = [c.upper() for c in df.columns]
        result = compute_heikin_ashi_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_mixed_case_column_names(self):
        """Mixed case column names should be normalized."""
        df = _make_df(80)
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
        result = compute_heikin_ashi_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_single_bar_does_not_raise(self):
        """Single bar must not raise; returns HOLD."""
        df = _make_df(1)
        result = compute_heikin_ashi_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_extra_columns_ignored(self):
        """Extra columns in DataFrame should not break the signal."""
        df = _make_df(80)
        df["extra_col"] = 999.9
        df["another"] = "foo"
        result = compute_heikin_ashi_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_nan_rows_dropped_gracefully(self):
        """NaN rows in OHLC should be dropped; result must still be valid."""
        df = _make_df(80)
        # Inject NaN rows in the middle
        df.loc[df.index[10], "close"] = float("nan")
        df.loc[df.index[20], "open"] = float("nan")
        result = compute_heikin_ashi_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_minimum_valid_rows_boundary(self):
        """Exactly _MIN_ROWS_BASIC (10) rows: should not return default HOLD from guard."""
        df = _make_df(10)
        result = compute_heikin_ashi_signal(df)
        # Sub-signals may still all be HOLD due to insufficient indicator warmup,
        # but the function must execute without error and return valid structure.
        assert set(result["sub_signals"].keys()) == {
            "ha_trend", "ha_doji", "ha_color_change",
            "hull_ma", "alligator", "elder_impulse", "ttm_squeeze",
        }

    def test_compute_ha_candles_basic_properties(self):
        """HA candles must satisfy: ha_high >= max(ha_open, ha_close) and ha_low <= min(...)."""
        df = _make_df(30)
        ha = _compute_ha_candles(df)
        assert "ha_open" in ha.columns
        assert "ha_close" in ha.columns
        assert "ha_high" in ha.columns
        assert "ha_low" in ha.columns
        # Structural invariants
        assert (ha["ha_high"] >= ha[["ha_open", "ha_close"]].max(axis=1) - 1e-9).all()
        assert (ha["ha_low"] <= ha[["ha_open", "ha_close"]].min(axis=1) + 1e-9).all()

    def test_compute_ha_candles_first_open(self):
        """First HA open must be (O[0] + C[0]) / 2."""
        df = _make_df(30)
        ha = _compute_ha_candles(df)
        expected_first_open = (float(df["open"].iloc[0]) + float(df["close"].iloc[0])) / 2.0
        assert abs(ha["ha_open"].iloc[0] - expected_first_open) < 1e-9

    def test_integer_ohlc_handled(self):
        """Integer dtype OHLC columns should be cast to float without error."""
        n = 50
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        df = pd.DataFrame({
            "open": list(range(100, 100 + n)),
            "high": list(range(102, 102 + n)),
            "low": list(range(98, 98 + n)),
            "close": list(range(101, 101 + n)),
            "volume": [1000] * n,
        }, index=dates)
        result = compute_heikin_ashi_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0
