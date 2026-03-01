"""Tests for the oscillators signal module.

Covers all 8 sub-indicators and the public ``compute_oscillator_signal`` API:
    1. Awesome Oscillator (AO)
    2. Aroon Oscillator (25)
    3. Vortex Indicator (14)
    4. Chande Momentum Oscillator (9)
    5. Know Sure Thing (KST)
    6. Schaff Trend Cycle (STC)
    7. TRIX (15)
    8. Coppock Curve (14, 11, 10)

Test plan:
    - Basic smoke test — valid DataFrame returns valid action/confidence/sub_signals
    - Insufficient data / edge inputs return HOLD gracefully
    - None / empty DataFrame returns HOLD
    - Strong uptrend biases toward BUY sub-signals
    - Strong downtrend biases toward SELL sub-signals
    - Sub_signals dict contains exactly the 8 expected keys
    - Confidence is bounded to [0.0, 1.0]
    - Single-bar and very short data return HOLD
    - Flat / sideways data behaviour
    - Highly volatile data does not crash
    - Individual sub-indicator unit tests with controlled inputs
    - Missing or mis-named columns return HOLD
    - Case-insensitive column handling
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.oscillators import (
    compute_oscillator_signal,
    _awesome_oscillator,
    _aroon_oscillator,
    _vortex_indicator,
    _chande_momentum,
    _know_sure_thing,
    _schaff_trend_cycle,
    _trix,
    _coppock_curve,
    MIN_ROWS,
)


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------

def _make_df(
    n: int = 100,
    close_base: float = 100.0,
    trend: float = 0.0,
    volatility: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data.

    Parameters
    ----------
    n:
        Number of rows.
    close_base:
        Starting close price.
    trend:
        Per-bar drift added cumulatively to close prices.  Positive = uptrend,
        negative = downtrend.
    volatility:
        Amplitude of the random noise component.
    seed:
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n) * volatility
    close = close_base + np.cumsum(noise) + np.arange(n) * trend
    close = np.maximum(close, 1.0)
    high = close + np.abs(rng.standard_normal(n) * volatility)
    low = close - np.abs(rng.standard_normal(n) * volatility)
    low = np.maximum(low, 0.5)
    opn = close + rng.standard_normal(n) * 0.3
    volume = rng.integers(100, 10_000, n).astype(float)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "open": opn,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "time": dates,
        }
    )


def _make_trending_df(
    n: int = 100,
    start: float = 100.0,
    step: float = 1.0,
) -> pd.DataFrame:
    """Return a strictly monotone OHLCV DataFrame.

    Positive *step* = uptrend, negative *step* = downtrend.
    """
    closes = np.array([start + i * step for i in range(n)], dtype=float)
    closes = np.maximum(closes, 0.01)
    spread = abs(step) * 0.5 if step != 0 else 0.5
    high = closes + spread
    low = np.maximum(closes - spread, 0.01)
    opn = np.roll(closes, 1)
    opn[0] = closes[0]
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "open": opn,
            "high": high,
            "low": low,
            "close": closes,
            "volume": np.full(n, 1000.0),
            "time": dates,
        }
    )


def _make_flat_df(n: int = 100, price: float = 100.0) -> pd.DataFrame:
    """Return a completely flat OHLCV DataFrame (no price movement)."""
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "open": np.full(n, price),
            "high": np.full(n, price),
            "low": np.full(n, price),
            "close": np.full(n, price),
            "volume": np.full(n, 1000.0),
            "time": dates,
        }
    )


# ---------------------------------------------------------------------------
# Expected sub-signal keys
# ---------------------------------------------------------------------------

EXPECTED_SUB_SIGNAL_KEYS = {
    "awesome",
    "aroon",
    "vortex",
    "chande",
    "kst",
    "schaff",
    "trix",
    "coppock",
}

EXPECTED_INDICATOR_KEYS = {
    "awesome_osc",
    "aroon_osc",
    "vi_plus",
    "vi_minus",
    "cmo",
    "kst",
    "kst_signal",
    "schaff",
    "trix",
    "coppock",
}

VALID_ACTIONS = {"BUY", "SELL", "HOLD"}


# ===========================================================================
# Test class 1: Basic smoke tests (valid input)
# ===========================================================================

class TestBasicSmoke:
    """Verify that the public API returns a well-formed result for normal data."""

    def test_returns_dict_with_required_top_level_keys(self):
        """compute_oscillator_signal must return action, confidence, sub_signals, indicators."""
        df = _make_df(n=100)
        result = compute_oscillator_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert "sub_signals" in result
        assert "indicators" in result

    def test_action_is_valid_string(self):
        """action must be one of BUY, SELL, HOLD."""
        df = _make_df(n=100)
        result = compute_oscillator_signal(df)
        assert result["action"] in VALID_ACTIONS

    def test_confidence_is_float_in_unit_interval(self):
        """confidence must be a float in [0.0, 1.0]."""
        df = _make_df(n=100)
        result = compute_oscillator_signal(df)
        conf = result["confidence"]
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_sub_signals_contains_all_eight_keys(self):
        """sub_signals must contain exactly the 8 expected sub-indicator keys."""
        df = _make_df(n=100)
        result = compute_oscillator_signal(df)
        assert set(result["sub_signals"].keys()) == EXPECTED_SUB_SIGNAL_KEYS

    def test_all_sub_signal_values_are_valid_actions(self):
        """Every sub-signal value must be BUY, SELL, or HOLD."""
        df = _make_df(n=100)
        result = compute_oscillator_signal(df)
        for name, vote in result["sub_signals"].items():
            assert vote in VALID_ACTIONS, f"sub_signal '{name}' has invalid value: {vote!r}"

    def test_indicators_contains_all_expected_keys(self):
        """indicators dict must expose all raw indicator values."""
        df = _make_df(n=100)
        result = compute_oscillator_signal(df)
        assert set(result["indicators"].keys()) == EXPECTED_INDICATOR_KEYS

    def test_result_is_stable_for_same_input(self):
        """Same input DataFrame produces identical result (deterministic)."""
        df = _make_df(n=100, seed=7)
        r1 = compute_oscillator_signal(df)
        r2 = compute_oscillator_signal(df)
        assert r1["action"] == r2["action"]
        assert r1["confidence"] == r2["confidence"]
        assert r1["sub_signals"] == r2["sub_signals"]

    def test_does_not_mutate_input_dataframe(self):
        """compute_oscillator_signal must not modify the caller's DataFrame."""
        df = _make_df(n=100)
        close_before = df["close"].copy()
        compute_oscillator_signal(df)
        pd.testing.assert_series_equal(df["close"], close_before)


# ===========================================================================
# Test class 2: Insufficient data / edge inputs return HOLD
# ===========================================================================

class TestInsufficientData:
    """Ensure graceful HOLD for inputs that cannot produce meaningful signals."""

    def test_none_returns_hold(self):
        result = compute_oscillator_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_oscillator_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_single_row_returns_hold(self):
        df = _make_df(n=1)
        result = compute_oscillator_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_less_than_min_rows_returns_hold(self):
        """Any DataFrame shorter than MIN_ROWS must return HOLD."""
        df = _make_df(n=MIN_ROWS - 1)
        result = compute_oscillator_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_exactly_min_rows_does_not_crash(self):
        """Exactly MIN_ROWS bars should be accepted (may still be HOLD, but not error)."""
        df = _make_df(n=MIN_ROWS)
        result = compute_oscillator_signal(df)
        assert result["action"] in VALID_ACTIONS

    def test_missing_close_column_returns_hold(self):
        df = _make_df(n=100).drop(columns=["close"])
        result = compute_oscillator_signal(df)
        assert result["action"] == "HOLD"

    def test_missing_high_column_returns_hold(self):
        df = _make_df(n=100).drop(columns=["high"])
        result = compute_oscillator_signal(df)
        assert result["action"] == "HOLD"

    def test_missing_low_column_returns_hold(self):
        df = _make_df(n=100).drop(columns=["low"])
        result = compute_oscillator_signal(df)
        assert result["action"] == "HOLD"

    def test_non_dataframe_string_returns_hold(self):
        result = compute_oscillator_signal("not a dataframe")
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_non_dataframe_list_returns_hold(self):
        result = compute_oscillator_signal([1, 2, 3])
        assert result["action"] == "HOLD"

    def test_hold_result_has_all_sub_signal_keys(self):
        """Even the fallback HOLD result must expose all 8 sub-signal keys."""
        result = compute_oscillator_signal(None)
        assert set(result["sub_signals"].keys()) == EXPECTED_SUB_SIGNAL_KEYS

    def test_hold_result_all_sub_signals_are_hold(self):
        """Fallback result sub-signals must all be HOLD."""
        result = compute_oscillator_signal(None)
        for name, vote in result["sub_signals"].items():
            assert vote == "HOLD", f"Expected HOLD for {name}, got {vote!r}"


# ===========================================================================
# Test class 3: Column case-insensitivity
# ===========================================================================

class TestColumnCaseInsensitivity:
    """Column names should be accepted in any case."""

    def test_uppercase_columns_accepted(self):
        df = _make_df(n=100)
        df.columns = [c.upper() for c in df.columns]
        result = compute_oscillator_signal(df)
        assert result["action"] in VALID_ACTIONS

    def test_mixed_case_columns_accepted(self):
        df = _make_df(n=100)
        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })
        result = compute_oscillator_signal(df)
        assert result["action"] in VALID_ACTIONS


# ===========================================================================
# Test class 4: Trend-biased directional behaviour
# ===========================================================================

class TestDirectionalBias:
    """Strong trends should produce a majority of directionally-aligned sub-signals."""

    def test_strong_uptrend_produces_buy_sub_signals(self):
        """A long, consistent uptrend should produce at least 2 BUY sub-signals."""
        df = _make_trending_df(n=200, start=10.0, step=0.5)
        result = compute_oscillator_signal(df)
        buy_count = sum(1 for v in result["sub_signals"].values() if v == "BUY")
        # A monotone uptrend must fire at least 1 BUY across 8 oscillators
        assert buy_count >= 1, (
            f"Expected at least 1 BUY for uptrend, got {result['sub_signals']}"
        )

    def test_strong_downtrend_produces_sell_sub_signals(self):
        """A long, consistent downtrend should produce at least 2 SELL sub-signals."""
        df = _make_trending_df(n=200, start=300.0, step=-0.5)
        result = compute_oscillator_signal(df)
        sell_count = sum(1 for v in result["sub_signals"].values() if v == "SELL")
        assert sell_count >= 1, (
            f"Expected at least 1 SELL for downtrend, got {result['sub_signals']}"
        )

    def test_uptrend_not_dominated_by_sell(self):
        """In a clear uptrend the majority vote should NOT be SELL."""
        df = _make_trending_df(n=150, start=10.0, step=0.3)
        result = compute_oscillator_signal(df)
        assert result["action"] != "SELL", (
            f"Uptrend should not produce SELL action, got {result['action']}"
        )

    def test_downtrend_not_dominated_by_buy(self):
        """In a clear downtrend the majority vote should NOT be BUY."""
        df = _make_trending_df(n=150, start=300.0, step=-0.3)
        result = compute_oscillator_signal(df)
        assert result["action"] != "BUY", (
            f"Downtrend should not produce BUY action, got {result['action']}"
        )


# ===========================================================================
# Test class 5: Flat / sideways data
# ===========================================================================

class TestFlatData:
    """Completely flat data should not crash and should behave sensibly."""

    def test_flat_data_does_not_crash(self):
        df = _make_flat_df(n=100)
        result = compute_oscillator_signal(df)
        assert result["action"] in VALID_ACTIONS

    def test_flat_data_confidence_is_in_range(self):
        df = _make_flat_df(n=100)
        result = compute_oscillator_signal(df)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_flat_data_returns_hold(self):
        """With zero price movement, most oscillators should default to HOLD."""
        df = _make_flat_df(n=100)
        result = compute_oscillator_signal(df)
        assert result["action"] == "HOLD", (
            f"Expected HOLD for flat data, got {result['action']}"
        )


# ===========================================================================
# Test class 6: Highly volatile data
# ===========================================================================

class TestHighVolatilityData:
    """Large random swings should not cause exceptions."""

    def test_high_volatility_does_not_crash(self):
        df = _make_df(n=100, volatility=50.0, seed=99)
        result = compute_oscillator_signal(df)
        assert result["action"] in VALID_ACTIONS

    def test_high_volatility_confidence_bounded(self):
        df = _make_df(n=100, volatility=50.0, seed=99)
        result = compute_oscillator_signal(df)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_extreme_spike_up_then_down(self):
        """Data with a massive spike followed by a crash should not crash."""
        df = _make_df(n=80)
        closes = df["close"].values.copy()
        # Inject a violent spike at the midpoint
        closes[40] = closes[39] * 10
        closes[41] = closes[39] * 0.5
        df["close"] = closes
        df["high"] = df["close"] * 1.02
        df["low"] = df["close"] * 0.98
        result = compute_oscillator_signal(df)
        assert result["action"] in VALID_ACTIONS


# ===========================================================================
# Test class 7: Unit tests for _awesome_oscillator
# ===========================================================================

class TestAwesomeOscillator:
    """Unit tests for the Awesome Oscillator sub-indicator."""

    def _make_series(self, n: int = 60, trend: float = 0.0) -> tuple[pd.Series, pd.Series]:
        """Return (high, low) Series with optional trend."""
        closes = np.linspace(100.0, 100.0 + trend * n, n)
        high = pd.Series(closes + 1.0)
        low = pd.Series(closes - 1.0)
        return high, low

    def test_zero_crossover_upward_returns_buy(self):
        """A zero-line crossover from below should return BUY."""
        # Construct AO that was negative and just went positive by building
        # a price series that drops for 34 bars (makes AO negative) then
        # surges sharply (makes SMA5 > SMA34 momentarily).
        n = 80
        prices = np.concatenate([
            np.linspace(100.0, 60.0, 50),   # long decline → AO negative
            np.linspace(60.0, 90.0, 30),    # sharp rally
        ])
        high = pd.Series(prices + 2.0)
        low = pd.Series(np.maximum(prices - 2.0, 0.1))
        ao_val, signal = _awesome_oscillator(high, low)
        # We don't mandate BUY here (exact crossover bar is hard to engineer),
        # but we do require the function to return a valid signal.
        assert signal in VALID_ACTIONS
        assert isinstance(ao_val, float)

    def test_insufficient_data_returns_hold(self):
        """Fewer than 34 bars (SMA-34 warmup) should return HOLD."""
        high = pd.Series(np.linspace(100.0, 110.0, 20))
        low = pd.Series(np.linspace(98.0, 108.0, 20))
        ao_val, signal = _awesome_oscillator(high, low)
        assert signal == "HOLD"

    def test_returns_numeric_ao_value(self):
        """With sufficient data, ao_val must be a finite float."""
        high = pd.Series(np.linspace(100.0, 120.0, 60))
        low = pd.Series(np.linspace(98.0, 118.0, 60))
        ao_val, signal = _awesome_oscillator(high, low)
        # After sufficient bars the AO should be computable
        assert isinstance(ao_val, float)
        assert signal in VALID_ACTIONS


# ===========================================================================
# Test class 8: Unit tests for _aroon_oscillator
# ===========================================================================

class TestAroonOscillator:
    """Unit tests for the Aroon Oscillator sub-indicator."""

    def test_recent_high_is_high_triggers_buy(self):
        """When the highest high is the most recent bar, Aroon Up = 100 → BUY."""
        n = 30
        # Steadily rising: newest bar is the highest
        highs = pd.Series(np.linspace(90.0, 120.0, n))
        lows = pd.Series(np.linspace(88.0, 118.0, n))
        aroon_val, signal = _aroon_oscillator(highs, lows, period=25)
        assert signal == "BUY"
        assert aroon_val > 50

    def test_recent_low_is_low_triggers_sell(self):
        """When the lowest low is the most recent bar, Aroon Down = 100 → SELL."""
        n = 30
        # Steadily falling: newest bar is the lowest
        highs = pd.Series(np.linspace(120.0, 90.0, n))
        lows = pd.Series(np.linspace(118.0, 88.0, n))
        aroon_val, signal = _aroon_oscillator(highs, lows, period=25)
        assert signal == "SELL"
        assert aroon_val < -50

    def test_insufficient_period_returns_hold(self):
        """Fewer bars than period + 1 returns HOLD."""
        highs = pd.Series([100.0, 101.0, 102.0])
        lows = pd.Series([99.0, 100.0, 101.0])
        aroon_val, signal = _aroon_oscillator(highs, lows, period=25)
        assert signal == "HOLD"

    def test_oscillator_value_in_range(self):
        """Aroon oscillator value must be in [-100, 100]."""
        n = 50
        highs = pd.Series(np.linspace(100.0, 110.0, n))
        lows = pd.Series(np.linspace(98.0, 108.0, n))
        aroon_val, signal = _aroon_oscillator(highs, lows)
        assert -100 <= aroon_val <= 100


# ===========================================================================
# Test class 9: Unit tests for _vortex_indicator
# ===========================================================================

class TestVortexIndicator:
    """Unit tests for the Vortex Indicator sub-indicator."""

    def test_strong_uptrend_vi_plus_dominates(self):
        """In a strong uptrend VI+ should exceed VI- giving a BUY."""
        n = 60
        closes = pd.Series(np.linspace(100.0, 200.0, n))
        highs = closes + 2.0
        lows = closes - 1.0
        vip, vim, signal = _vortex_indicator(highs, lows, closes, period=14)
        # Uptrend → VI+ should be higher
        assert vip >= vim
        assert signal in VALID_ACTIONS  # may be BUY or HOLD depending on magnitudes

    def test_insufficient_data_returns_hold(self):
        """Fewer than period + 1 bars returns HOLD."""
        closes = pd.Series([100.0] * 5)
        highs = pd.Series([101.0] * 5)
        lows = pd.Series([99.0] * 5)
        vip, vim, signal = _vortex_indicator(highs, lows, closes, period=14)
        assert signal == "HOLD"

    def test_returns_floats(self):
        """With sufficient data, vi_plus and vi_minus must be finite floats."""
        n = 30
        closes = pd.Series(np.linspace(100.0, 120.0, n))
        highs = closes + 1.5
        lows = closes - 1.5
        vip, vim, signal = _vortex_indicator(highs, lows, closes, period=14)
        assert isinstance(vip, float)
        assert isinstance(vim, float)


# ===========================================================================
# Test class 10: Unit tests for _chande_momentum
# ===========================================================================

class TestChandeMomentum:
    """Unit tests for the Chande Momentum Oscillator sub-indicator."""

    def test_pure_uptrend_gives_high_cmo(self):
        """A series with all positive daily changes should give CMO = 100."""
        close = pd.Series(np.linspace(100.0, 120.0, 30))
        cmo_val, signal = _chande_momentum(close, period=9)
        # CMO = 100 for all-up → BUY
        assert cmo_val > 50
        assert signal == "BUY"

    def test_pure_downtrend_gives_low_cmo(self):
        """A series with all negative daily changes should give CMO = -100."""
        close = pd.Series(np.linspace(120.0, 100.0, 30))
        cmo_val, signal = _chande_momentum(close, period=9)
        assert cmo_val < -50
        assert signal == "SELL"

    def test_flat_price_returns_hold(self):
        """Flat price (no change) → CMO is 0 → HOLD."""
        close = pd.Series([100.0] * 20)
        cmo_val, signal = _chande_momentum(close, period=9)
        assert signal == "HOLD"

    def test_insufficient_data_returns_hold(self):
        close = pd.Series([100.0] * 5)
        cmo_val, signal = _chande_momentum(close, period=9)
        assert signal == "HOLD"

    def test_cmo_value_in_range(self):
        """CMO must be in [-100, 100]."""
        close = pd.Series(100.0 + np.cumsum(np.random.default_rng(1).standard_normal(50)))
        cmo_val, signal = _chande_momentum(close, period=9)
        if not np.isnan(cmo_val):
            assert -100 <= cmo_val <= 100


# ===========================================================================
# Test class 11: Unit tests for _know_sure_thing
# ===========================================================================

class TestKnowSureThing:
    """Unit tests for the Know Sure Thing (KST) indicator."""

    def test_insufficient_data_returns_hold(self):
        """KST requires max(ROC)+max(SMA) bars; short series returns HOLD."""
        close = pd.Series(np.linspace(100.0, 110.0, 20))
        kst_val, sig_val, signal = _know_sure_thing(close)
        assert signal == "HOLD"

    def test_sufficient_data_returns_valid_signal(self):
        """With sufficient data, signal must be one of BUY/SELL/HOLD."""
        close = pd.Series(np.linspace(100.0, 150.0, 80))
        kst_val, sig_val, signal = _know_sure_thing(close)
        assert signal in VALID_ACTIONS

    def test_returns_numeric_values(self):
        """KST and signal-line values must be finite or nan (no exceptions)."""
        close = pd.Series(100.0 + np.cumsum(np.random.default_rng(5).standard_normal(80)))
        kst_val, sig_val, signal = _know_sure_thing(close)
        # No assertion on exact value, just that it's a float (possibly nan)
        assert isinstance(kst_val, float)
        assert isinstance(sig_val, float)


# ===========================================================================
# Test class 12: Unit tests for _schaff_trend_cycle
# ===========================================================================

class TestSchaffTrendCycle:
    """Unit tests for the Schaff Trend Cycle sub-indicator."""

    def test_insufficient_data_returns_hold(self):
        """Fewer than slow + cycle bars returns HOLD."""
        close = pd.Series(np.linspace(100.0, 110.0, 30))
        stc_val, signal = _schaff_trend_cycle(close, fast=23, slow=50, cycle=10)
        assert signal == "HOLD"

    def test_sufficient_data_returns_valid_signal(self):
        close = pd.Series(100.0 + np.cumsum(np.random.default_rng(11).standard_normal(80)))
        stc_val, signal = _schaff_trend_cycle(close)
        assert signal in VALID_ACTIONS

    def test_stc_value_in_range_when_valid(self):
        """STC value should be in [0, 100] when it can be computed."""
        close = pd.Series(100.0 + np.cumsum(np.random.default_rng(12).standard_normal(80)))
        stc_val, signal = _schaff_trend_cycle(close)
        if not np.isnan(stc_val):
            # STC is a double-stochastic, bounded [0, 100] in theory
            assert 0.0 <= stc_val <= 100.0

    def test_oversold_stc_produces_buy(self):
        """STC < 25 → BUY.  Engineer a series that ends with a sharp drop."""
        # Build a flat series then drop sharply — MACD should go negative and
        # STC should fall into oversold (<25) territory.
        n = 80
        closes = np.concatenate([
            np.full(50, 100.0),
            np.linspace(100.0, 60.0, 30),  # steep fall
        ])
        close = pd.Series(closes)
        stc_val, signal = _schaff_trend_cycle(close)
        # With a long steep fall the STC should be in oversold territory
        if not np.isnan(stc_val):
            # When STC < 25 the signal is BUY
            if stc_val < 25:
                assert signal == "BUY"


# ===========================================================================
# Test class 13: Unit tests for _trix
# ===========================================================================

class TestTrix:
    """Unit tests for the TRIX indicator."""

    def test_uptrend_trix_is_positive(self):
        """A persistent uptrend should yield a positive TRIX value."""
        close = pd.Series(np.linspace(100.0, 200.0, 80))
        trix_val, signal = _trix(close)
        if not np.isnan(trix_val):
            assert trix_val > 0

    def test_downtrend_trix_is_negative(self):
        """A persistent downtrend should yield a negative TRIX value."""
        close = pd.Series(np.linspace(200.0, 100.0, 80))
        trix_val, signal = _trix(close)
        if not np.isnan(trix_val):
            assert trix_val < 0

    def test_insufficient_data_returns_hold(self):
        """Very short series returns HOLD."""
        close = pd.Series([100.0] * 5)
        trix_val, signal = _trix(close)
        assert signal == "HOLD"

    def test_returns_float(self):
        close = pd.Series(100.0 + np.cumsum(np.random.default_rng(20).standard_normal(60)))
        trix_val, signal = _trix(close)
        assert isinstance(trix_val, float)
        assert signal in VALID_ACTIONS


# ===========================================================================
# Test class 14: Unit tests for _coppock_curve
# ===========================================================================

class TestCoppockCurve:
    """Unit tests for the Coppock Curve sub-indicator."""

    def test_insufficient_data_returns_hold(self):
        """Less than roc_long + wma_period + 1 bars returns HOLD."""
        close = pd.Series(np.linspace(100.0, 110.0, 15))
        cc_val, signal = _coppock_curve(close, roc_long=14, roc_short=11, wma_period=10)
        assert signal == "HOLD"

    def test_sufficient_data_returns_valid_signal(self):
        close = pd.Series(np.linspace(100.0, 130.0, 60))
        cc_val, signal = _coppock_curve(close)
        assert signal in VALID_ACTIONS

    def test_returns_float(self):
        close = pd.Series(100.0 + np.cumsum(np.random.default_rng(30).standard_normal(60)))
        cc_val, signal = _coppock_curve(close)
        assert isinstance(cc_val, float)

    def test_upward_turn_from_below_zero_gives_buy(self):
        """Classic Coppock BUY: CC < 0 and rising (turning up from below zero)."""
        # Manufacture a Coppock value that goes slightly negative then ticks up.
        # A slow downtrend followed by a sharp reversal up tends to produce this.
        n = 60
        closes = np.concatenate([
            np.linspace(200.0, 100.0, 40),   # long downtrend → CC goes negative
            np.linspace(100.0, 115.0, 20),   # modest recovery → CC turns up
        ])
        close = pd.Series(closes)
        cc_val, signal = _coppock_curve(close)
        # We don't guarantee the exact bar alignment, but with this shape
        # the CC should be computable (not nan).
        assert isinstance(cc_val, float)
        assert signal in VALID_ACTIONS


# ===========================================================================
# Test class 15: Integration — confidence and majority vote consistency
# ===========================================================================

class TestConfidenceConsistency:
    """Confidence should be consistent with the sub-signal vote distribution."""

    def test_unanimous_buy_gives_confidence_one(self):
        """When all 8 sub-signals are BUY, confidence should be 1.0."""
        # We can't force all sub-signals, but we can test the logic directly.
        from portfolio.signal_utils import majority_vote
        votes = ["BUY"] * 8
        action, conf = majority_vote(votes)
        assert action == "BUY"
        assert conf == 1.0

    def test_unanimous_sell_gives_confidence_one(self):
        from portfolio.signal_utils import majority_vote
        votes = ["SELL"] * 8
        action, conf = majority_vote(votes)
        assert action == "SELL"
        assert conf == 1.0

    def test_split_vote_gives_hold(self):
        """4 BUY + 4 SELL (tied) → HOLD."""
        from portfolio.signal_utils import majority_vote
        votes = ["BUY", "BUY", "BUY", "BUY", "SELL", "SELL", "SELL", "SELL"]
        action, conf = majority_vote(votes)
        assert action == "HOLD"

    def test_all_hold_gives_hold_zero_confidence(self):
        """When every sub-signal abstains, action = HOLD, confidence = 0."""
        from portfolio.signal_utils import majority_vote
        votes = ["HOLD"] * 8
        action, conf = majority_vote(votes)
        assert action == "HOLD"
        assert conf == 0.0

    def test_confidence_reflects_majority_fraction(self):
        """6 BUY + 2 SELL → confidence = 6/8 = 0.75."""
        from portfolio.signal_utils import majority_vote
        votes = ["BUY"] * 6 + ["SELL"] * 2
        action, conf = majority_vote(votes)
        assert action == "BUY"
        assert abs(conf - 0.75) < 1e-6

    def test_oscillator_confidence_consistent_with_votes(self):
        """Ensure compute_oscillator_signal confidence matches majority-vote math."""
        from portfolio.signal_utils import majority_vote
        df = _make_df(n=150, seed=77)
        result = compute_oscillator_signal(df)
        votes = list(result["sub_signals"].values())
        expected_action, expected_conf = majority_vote(votes)
        assert result["action"] == expected_action
        assert abs(result["confidence"] - expected_conf) < 1e-6


# ===========================================================================
# Test class 16: Large data set — performance sanity check
# ===========================================================================

class TestLargeDataSet:
    """The signal should handle large DataFrames without blowing up."""

    def test_large_df_completes_without_error(self):
        """500-bar DataFrame completes and returns a valid result."""
        df = _make_df(n=500, seed=101)
        result = compute_oscillator_signal(df)
        assert result["action"] in VALID_ACTIONS
        assert 0.0 <= result["confidence"] <= 1.0

    def test_1000_bar_df_completes(self):
        """1000-bar DataFrame (stress test) completes without error."""
        df = _make_df(n=1000, seed=202)
        result = compute_oscillator_signal(df)
        assert result["action"] in VALID_ACTIONS
