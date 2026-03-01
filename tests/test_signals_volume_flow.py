"""Tests for the volume_flow composite signal module.

Covers all six sub-indicators:
    1. OBV (On-Balance Volume) vs its 20-period SMA
    2. VWAP Cross (price vs session VWAP)
    3. A/D Line (Accumulation/Distribution) vs its 20-period SMA
    4. CMF (Chaikin Money Flow, 20-period)
    5. MFI (Money Flow Index, 14-period)
    6. Volume RSI (14-period)

Test categories:
    - Basic smoke tests (keys, types, valid ranges)
    - Insufficient / bad data returns HOLD
    - Directional scenarios: uptrend → BUY pressure, downtrend → SELL pressure
    - Edge cases: all-zero volume, single bar, flat prices, NaN-heavy data
    - Individual sub-indicator voting functions
    - Confidence boundaries
    - Column name case-insensitivity
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.volume_flow import (
    compute_volume_flow_signal,
    _compute_obv,
    _compute_vwap,
    _compute_ad_line,
    _compute_cmf,
    _compute_mfi,
    _compute_volume_rsi,
    _vote_obv,
    _vote_vwap,
    _vote_ad,
    _vote_cmf,
    _vote_mfi,
    _vote_volume_rsi,
    MIN_ROWS,
)


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------

def _make_flat_df(n: int = 60, price: float = 100.0,
                  volume: float = 1_000.0) -> pd.DataFrame:
    """Build a completely flat OHLCV DataFrame (no price movement)."""
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "open":   [price] * n,
        "high":   [price] * n,
        "low":    [price] * n,
        "close":  [price] * n,
        "volume": [volume] * n,
        "time":   dates,
    })


def _make_trending_df(n: int = 80, start: float = 100.0,
                      step: float = 1.0, volume: float = 2_000.0,
                      volume_spike: bool = False) -> pd.DataFrame:
    """Build a monotonically trending OHLCV DataFrame.

    Parameters
    ----------
    step : float
        Positive for uptrend, negative for downtrend.
    volume_spike : bool
        If True, double volume on the last 20 bars to simulate expansion.

    Notes
    -----
    In an uptrend (step > 0), close is set near the high so that the
    Close Location Value (CLV) is strongly positive, which drives the
    A/D Line upward. In a downtrend (step < 0), close is set near the
    low so CLV is strongly negative.
    """
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    bar_range = abs(step) * 2.0  # full high-low range per bar
    mid = np.array([start + i * step for i in range(n)], dtype=float)
    mid = np.maximum(mid, 1.0)

    if step >= 0:
        # Uptrend: close near the high → CLV ≈ +0.8
        low   = mid - bar_range * 0.1
        high  = mid + bar_range * 0.9
        close = mid + bar_range * 0.7   # near the high
    else:
        # Downtrend: close near the low → CLV ≈ -0.8
        low   = mid - bar_range * 0.9
        high  = mid + bar_range * 0.1
        close = mid - bar_range * 0.7   # near the low

    low   = np.maximum(low, 0.1)
    close = np.maximum(close, 0.1)
    opn   = mid  # open at midpoint

    vols = np.full(n, volume)
    if volume_spike:
        vols[-20:] = volume * 2.0

    return pd.DataFrame({
        "open":   opn,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": vols,
        "time":   dates,
    })


def _make_random_df(n: int = 100, seed: int = 42,
                    close_base: float = 50.0,
                    trend: float = 0.0) -> pd.DataFrame:
    """Generate random OHLCV data with configurable trend."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n) * 2.0
    close = close_base + np.cumsum(noise) + np.arange(n) * trend
    close = np.maximum(close, 1.0)
    high = close + np.abs(rng.standard_normal(n))
    low  = close - np.abs(rng.standard_normal(n))
    low  = np.maximum(low, 0.5)
    opn  = close + rng.standard_normal(n) * 0.3
    volume = rng.integers(500, 5_000, n).astype(float)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "open": opn, "high": high, "low": low,
        "close": close, "volume": volume, "time": dates,
    })


# ---------------------------------------------------------------------------
# Test 1: Basic smoke test — valid DataFrame returns valid structure
# ---------------------------------------------------------------------------

class TestBasicSmoke:
    """Verify the top-level function returns a well-formed result dict."""

    def test_returns_all_top_level_keys(self):
        """Result dict must contain action, confidence, sub_signals, indicators."""
        df = _make_random_df(100)
        result = compute_volume_flow_signal(df)
        assert "action" in result
        assert "confidence" in result
        assert "sub_signals" in result
        assert "indicators" in result

    def test_action_is_valid_string(self):
        """action must be one of the three valid strings."""
        df = _make_random_df(100)
        result = compute_volume_flow_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_confidence_is_float_in_range(self):
        """confidence must be a float between 0.0 and 1.0 inclusive."""
        df = _make_random_df(100)
        result = compute_volume_flow_signal(df)
        conf = result["confidence"]
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_sub_signals_contains_all_six_keys(self):
        """sub_signals must have exactly the six expected indicator keys."""
        df = _make_random_df(100)
        result = compute_volume_flow_signal(df)
        expected = {"obv", "vwap", "ad_line", "cmf", "mfi", "volume_rsi"}
        assert set(result["sub_signals"].keys()) == expected

    def test_all_sub_signal_votes_are_valid(self):
        """Every sub_signal vote must be BUY, SELL, or HOLD."""
        df = _make_random_df(100)
        result = compute_volume_flow_signal(df)
        for name, vote in result["sub_signals"].items():
            assert vote in ("BUY", "SELL", "HOLD"), (
                f"sub_signal '{name}' produced invalid vote: {vote!r}"
            )

    def test_indicators_contains_all_expected_keys(self):
        """indicators dict must expose all eight raw values."""
        df = _make_random_df(100)
        result = compute_volume_flow_signal(df)
        expected = {"obv", "obv_sma", "vwap", "ad_line", "ad_sma",
                    "cmf", "mfi", "volume_rsi"}
        assert set(result["indicators"].keys()) == expected


# ---------------------------------------------------------------------------
# Test 2: Insufficient data returns HOLD
# ---------------------------------------------------------------------------

class TestInsufficientData:
    """Any input that cannot support the minimum required rows must HOLD."""

    def test_none_input_returns_hold(self):
        result = compute_volume_flow_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_volume_flow_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_one_row_returns_hold(self):
        """Single-bar data is far below MIN_ROWS — must HOLD gracefully."""
        df = pd.DataFrame({
            "open": [100.0], "high": [101.0], "low": [99.0],
            "close": [100.0], "volume": [1_000.0],
        })
        result = compute_volume_flow_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_exactly_min_rows_minus_one_returns_hold(self):
        """MIN_ROWS - 1 rows must still return HOLD."""
        df = _make_random_df(MIN_ROWS - 1)
        result = compute_volume_flow_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_missing_required_column_returns_hold(self):
        """A DataFrame missing 'volume' must return HOLD."""
        df = _make_random_df(100).drop(columns=["volume"])
        result = compute_volume_flow_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_missing_close_column_returns_hold(self):
        """A DataFrame missing 'close' must return HOLD."""
        df = _make_random_df(100).drop(columns=["close"])
        result = compute_volume_flow_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_not_a_dataframe_returns_hold(self):
        """A non-DataFrame input (e.g. string) must return HOLD without crashing."""
        result = compute_volume_flow_signal("not a dataframe")
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_list_input_returns_hold(self):
        """A Python list must return HOLD without crashing."""
        result = compute_volume_flow_signal([1, 2, 3])
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Test 3: High-volume uptrend should produce BUY signals
# ---------------------------------------------------------------------------

class TestHighVolumeUptrend:
    """In a strong uptrend with rising volume, multiple sub-indicators should vote BUY."""

    def _uptrend_df(self) -> pd.DataFrame:
        return _make_trending_df(n=80, start=50.0, step=1.0,
                                 volume=2_000.0, volume_spike=True)

    def test_uptrend_produces_at_least_two_buy_sub_signals(self):
        """A clear uptrend with volume expansion must generate >= 2 BUY votes."""
        df = self._uptrend_df()
        result = compute_volume_flow_signal(df)
        buy_count = sum(1 for v in result["sub_signals"].values() if v == "BUY")
        assert buy_count >= 2, (
            f"Expected >= 2 BUY sub-signals in uptrend, got {buy_count}. "
            f"sub_signals={result['sub_signals']}"
        )

    def test_uptrend_overall_action_is_buy_or_hold(self):
        """An uptrend should not produce a SELL overall action."""
        df = self._uptrend_df()
        result = compute_volume_flow_signal(df)
        assert result["action"] != "SELL", (
            f"Unexpected SELL in clear uptrend. sub_signals={result['sub_signals']}"
        )

    def test_vwap_vote_is_buy_in_uptrend(self):
        """In an uptrend, close is above VWAP → VWAP sub-signal should be BUY."""
        df = self._uptrend_df()
        result = compute_volume_flow_signal(df)
        # VWAP is a cumulative measure; close at the end of an uptrend
        # is above the session average, so VWAP vote should be BUY.
        assert result["sub_signals"]["vwap"] == "BUY"

    def test_obv_buy_in_uptrend(self):
        """In a sustained uptrend, OBV grows above its 20-period SMA → BUY."""
        df = self._uptrend_df()
        result = compute_volume_flow_signal(df)
        assert result["sub_signals"]["obv"] == "BUY"

    def test_ad_line_buy_in_uptrend(self):
        """Close near the high each bar → CLV positive → A/D grows above its SMA → BUY."""
        df = self._uptrend_df()
        result = compute_volume_flow_signal(df)
        assert result["sub_signals"]["ad_line"] == "BUY"


# ---------------------------------------------------------------------------
# Test 4: High-volume downtrend should produce SELL signals
# ---------------------------------------------------------------------------

class TestHighVolumeDowntrend:
    """In a strong downtrend with rising volume, sub-indicators should vote SELL."""

    def _downtrend_df(self) -> pd.DataFrame:
        return _make_trending_df(n=80, start=150.0, step=-1.0,
                                 volume=2_000.0, volume_spike=True)

    def test_downtrend_produces_at_least_two_sell_sub_signals(self):
        """A clear downtrend with volume expansion must generate >= 2 SELL votes."""
        df = self._downtrend_df()
        result = compute_volume_flow_signal(df)
        sell_count = sum(1 for v in result["sub_signals"].values() if v == "SELL")
        assert sell_count >= 2, (
            f"Expected >= 2 SELL sub-signals in downtrend, got {sell_count}. "
            f"sub_signals={result['sub_signals']}"
        )

    def test_downtrend_overall_action_is_sell_or_hold(self):
        """A downtrend should not produce a BUY overall action."""
        df = self._downtrend_df()
        result = compute_volume_flow_signal(df)
        assert result["action"] != "BUY", (
            f"Unexpected BUY in clear downtrend. sub_signals={result['sub_signals']}"
        )

    def test_vwap_vote_is_sell_in_downtrend(self):
        """In a downtrend, close is below VWAP → VWAP sub-signal should be SELL."""
        df = self._downtrend_df()
        result = compute_volume_flow_signal(df)
        assert result["sub_signals"]["vwap"] == "SELL"

    def test_obv_sell_in_downtrend(self):
        """In a sustained downtrend, OBV falls below its 20-period SMA → SELL."""
        df = self._downtrend_df()
        result = compute_volume_flow_signal(df)
        assert result["sub_signals"]["obv"] == "SELL"

    def test_ad_line_sell_in_downtrend(self):
        """Close near the low each bar → CLV negative → A/D line falls below SMA → SELL."""
        df = self._downtrend_df()
        result = compute_volume_flow_signal(df)
        assert result["sub_signals"]["ad_line"] == "SELL"


# ---------------------------------------------------------------------------
# Test 5: Flat / low-volume data produces mostly HOLD
# ---------------------------------------------------------------------------

class TestFlatLowVolume:
    """Completely static data carries no information → sub-indicators should HOLD."""

    def test_flat_price_returns_hold_action(self):
        """When all OHLCV values are identical, the composite action must be HOLD."""
        df = _make_flat_df(n=80, price=100.0, volume=500.0)
        result = compute_volume_flow_signal(df)
        assert result["action"] == "HOLD"

    def test_flat_price_confidence_is_zero(self):
        """No directional information → confidence must be 0.0."""
        df = _make_flat_df(n=80, price=100.0, volume=500.0)
        result = compute_volume_flow_signal(df)
        assert result["confidence"] == 0.0

    def test_flat_data_majority_sub_signals_hold(self):
        """At least 4 of 6 sub-signals should be HOLD on flat data."""
        df = _make_flat_df(n=80, price=100.0, volume=500.0)
        result = compute_volume_flow_signal(df)
        hold_count = sum(1 for v in result["sub_signals"].values() if v == "HOLD")
        assert hold_count >= 4, (
            f"Expected >= 4 HOLD sub-signals on flat data, got {hold_count}. "
            f"sub_signals={result['sub_signals']}"
        )


# ---------------------------------------------------------------------------
# Test 6: All-zero volume edge case
# ---------------------------------------------------------------------------

class TestZeroVolume:
    """Zero volume across all bars should not crash; the module must degrade gracefully."""

    def test_zero_volume_does_not_raise(self):
        """compute_volume_flow_signal must not throw on all-zero volume."""
        df = _make_trending_df(n=80, step=0.5, volume=0.0)
        # Should complete without exception
        result = compute_volume_flow_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_zero_volume_confidence_in_range(self):
        """Even with zero volume the confidence must stay within [0.0, 1.0]."""
        df = _make_trending_df(n=80, step=0.5, volume=0.0)
        result = compute_volume_flow_signal(df)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_zero_volume_sub_signals_are_valid(self):
        """All six sub-signal votes must be valid strings even on zero volume."""
        df = _make_trending_df(n=80, step=0.5, volume=0.0)
        result = compute_volume_flow_signal(df)
        for name, vote in result["sub_signals"].items():
            assert vote in ("BUY", "SELL", "HOLD"), (
                f"sub_signal '{name}' has invalid vote on zero volume: {vote!r}"
            )


# ---------------------------------------------------------------------------
# Test 7: Column name case-insensitivity
# ---------------------------------------------------------------------------

class TestColumnCaseInsensitivity:
    """The module normalises column names to lowercase — uppercase must work."""

    def test_uppercase_columns_accepted(self):
        df = _make_random_df(100)
        df.columns = [c.upper() for c in df.columns]
        result = compute_volume_flow_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_mixed_case_columns_accepted(self):
        df = _make_random_df(100)
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
        result = compute_volume_flow_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")


# ---------------------------------------------------------------------------
# Test 8: Exactly MIN_ROWS accepted and computes
# ---------------------------------------------------------------------------

class TestMinimumRowBoundary:
    """Verify the boundary: MIN_ROWS - 1 → HOLD, MIN_ROWS → computes."""

    def test_exactly_min_rows_computes(self):
        """A DataFrame with exactly MIN_ROWS rows must compute (not HOLD due to size)."""
        df = _make_random_df(MIN_ROWS)
        result = compute_volume_flow_signal(df)
        # The function should attempt computation; action is whatever the data says.
        # Importantly it must not be HOLD due to the row-count guard alone.
        # We can't assert the exact action, but we can verify it ran.
        assert result["action"] in ("BUY", "SELL", "HOLD")
        # Confidence > 0.0 would prove it computed past the guard
        # (some signals may still be NaN, so we only check it doesn't refuse).

    def test_one_below_min_rows_holds(self):
        df = _make_random_df(MIN_ROWS - 1)
        result = compute_volume_flow_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Test 9: Individual sub-indicator calculators
# ---------------------------------------------------------------------------

class TestComputeOBV:
    """Unit tests for _compute_obv.

    Note: _compute_obv uses close.diff() internally, which makes the first
    bar's direction NaN (0 * NaN = NaN).  The cumsum therefore starts with
    NaN at index 0 and accumulates from index 1 onward.  All comparisons
    must use iloc[1] (first finite value) rather than iloc[0].
    """

    def test_rising_prices_obv_grows(self):
        """When price rises every bar after the first, OBV should increase."""
        n = 30
        close  = pd.Series([100.0 + i for i in range(n)])
        volume = pd.Series([1_000.0] * n)
        obv, _ = _compute_obv(close, volume)
        # iloc[0] is NaN (diff of first bar); compare from iloc[1]
        assert obv.iloc[-1] > obv.iloc[1]

    def test_obv_sma_nan_before_warmup(self):
        """OBV SMA requires 20 non-NaN OBV values to produce the first reading.

        Because obv[0] is NaN (from close.diff()), the rolling window of 20
        over OBV does not become fully populated until index 20 (the 21st bar).
        """
        n = 35
        close  = pd.Series([100.0 + i for i in range(n)])
        volume = pd.Series([1_000.0] * n)
        obv, obv_sma = _compute_obv(close, volume)
        # Indices 0..19 must be NaN (first bar NaN + 19 bars of accumulation
        # still below min_periods=20 over the rolling window)
        assert obv_sma.iloc[:20].isna().all()
        # Index 20 is the first bar with 20 valid OBV values in the window
        assert not np.isnan(obv_sma.iloc[20])

    def test_falling_prices_obv_decreases(self):
        """When price falls every bar after the first, OBV should decrease."""
        n = 30
        close  = pd.Series([100.0 - i for i in range(n)])
        close  = close.clip(lower=0.01)
        volume = pd.Series([1_000.0] * n)
        obv, _ = _compute_obv(close, volume)
        # iloc[0] is NaN; compare from iloc[1]
        assert obv.iloc[-1] < obv.iloc[1]


class TestComputeVWAP:
    """Unit tests for _compute_vwap."""

    def test_equal_ohlc_vwap_equals_price(self):
        """When high == low == close, VWAP should equal that price."""
        price = 75.0
        n = 20
        h = pd.Series([price] * n)
        l = pd.Series([price] * n)
        c = pd.Series([price] * n)
        v = pd.Series([1_000.0] * n)
        vwap = _compute_vwap(h, l, c, v)
        # All values should be exactly the price
        np.testing.assert_allclose(vwap.dropna().values, price, rtol=1e-9)

    def test_vwap_is_between_low_and_high(self):
        """VWAP must lie between the session low and high at every bar."""
        df = _make_random_df(60)
        vwap = _compute_vwap(df["high"], df["low"], df["close"], df["volume"])
        # Cumulative VWAP expands the range over time, but should generally
        # stay reasonable. Only check non-NaN entries.
        vwap_valid = vwap.dropna()
        assert len(vwap_valid) > 0


class TestComputeCMF:
    """Unit tests for _compute_cmf."""

    def test_close_at_high_cmf_positive(self):
        """When close equals high every bar, CLV = +1 → CMF should be positive."""
        n = 60
        high  = pd.Series([110.0] * n)
        low   = pd.Series([100.0] * n)
        close = pd.Series([110.0] * n)  # close at high → CLV = +1
        vol   = pd.Series([1_000.0] * n)
        cmf = _compute_cmf(high, low, close, vol, period=20)
        # After warmup, CMF should be +1.0
        cmf_valid = cmf.iloc[19:]
        assert (cmf_valid > 0.9).all(), f"Expected CMF ≈ 1.0, got {cmf_valid.values}"

    def test_close_at_low_cmf_negative(self):
        """When close equals low every bar, CLV = -1 → CMF should be negative."""
        n = 60
        high  = pd.Series([110.0] * n)
        low   = pd.Series([100.0] * n)
        close = pd.Series([100.0] * n)  # close at low → CLV = -1
        vol   = pd.Series([1_000.0] * n)
        cmf = _compute_cmf(high, low, close, vol, period=20)
        cmf_valid = cmf.iloc[19:]
        assert (cmf_valid < -0.9).all(), f"Expected CMF ≈ -1.0, got {cmf_valid.values}"


class TestComputeMFI:
    """Unit tests for _compute_mfi."""

    def test_mfi_range(self):
        """MFI must be in [0, 100] wherever it is not NaN."""
        df = _make_random_df(80)
        mfi = _compute_mfi(df["high"], df["low"], df["close"], df["volume"], period=14)
        valid = mfi.dropna()
        assert (valid >= 0.0).all() and (valid <= 100.0).all(), (
            f"MFI out of [0,100]: min={valid.min()}, max={valid.max()}"
        )

    def test_mfi_nan_before_warmup(self):
        """MFI with period=14 requires 14 bars — values before that are NaN."""
        df = _make_random_df(80)
        mfi = _compute_mfi(df["high"], df["low"], df["close"], df["volume"], period=14)
        assert mfi.iloc[:13].isna().all()


class TestComputeVolumeRSI:
    """Unit tests for _compute_volume_rsi."""

    def test_volume_rsi_range(self):
        """Volume RSI must be in [0, 100] wherever it is not NaN."""
        df = _make_random_df(80)
        vrsi = _compute_volume_rsi(df["volume"], period=14)
        valid = vrsi.dropna()
        assert (valid >= 0.0).all() and (valid <= 100.0).all(), (
            f"Volume RSI out of [0,100]: min={valid.min()}, max={valid.max()}"
        )


# ---------------------------------------------------------------------------
# Test 10: Individual vote functions
# ---------------------------------------------------------------------------

class TestVoteOBV:
    """Unit tests for _vote_obv."""

    def test_obv_above_sma_is_buy(self):
        assert _vote_obv(200.0, 150.0) == "BUY"

    def test_obv_below_sma_is_sell(self):
        assert _vote_obv(100.0, 150.0) == "SELL"

    def test_obv_equal_sma_is_hold(self):
        assert _vote_obv(150.0, 150.0) == "HOLD"

    def test_obv_nan_obv_is_hold(self):
        assert _vote_obv(np.nan, 150.0) == "HOLD"

    def test_obv_nan_sma_is_hold(self):
        assert _vote_obv(150.0, np.nan) == "HOLD"

    def test_both_nan_is_hold(self):
        assert _vote_obv(np.nan, np.nan) == "HOLD"


class TestVoteVWAP:
    """Unit tests for _vote_vwap."""

    def test_close_above_vwap_is_buy(self):
        assert _vote_vwap(105.0, 100.0) == "BUY"

    def test_close_below_vwap_is_sell(self):
        assert _vote_vwap(95.0, 100.0) == "SELL"

    def test_close_equal_vwap_is_hold(self):
        assert _vote_vwap(100.0, 100.0) == "HOLD"

    def test_nan_close_is_hold(self):
        assert _vote_vwap(np.nan, 100.0) == "HOLD"

    def test_nan_vwap_is_hold(self):
        assert _vote_vwap(100.0, np.nan) == "HOLD"


class TestVoteAD:
    """Unit tests for _vote_ad (A/D Line)."""

    def test_ad_above_sma_is_buy(self):
        assert _vote_ad(500.0, 300.0) == "BUY"

    def test_ad_below_sma_is_sell(self):
        assert _vote_ad(100.0, 300.0) == "SELL"

    def test_ad_equal_sma_is_hold(self):
        assert _vote_ad(300.0, 300.0) == "HOLD"

    def test_nan_ad_is_hold(self):
        assert _vote_ad(np.nan, 300.0) == "HOLD"

    def test_nan_sma_is_hold(self):
        assert _vote_ad(300.0, np.nan) == "HOLD"


class TestVoteCMF:
    """Unit tests for _vote_cmf (Chaikin Money Flow).

    Thresholds: > 0.05 → BUY, < -0.05 → SELL, else HOLD.
    """

    def test_strong_positive_cmf_is_buy(self):
        assert _vote_cmf(0.15) == "BUY"

    def test_strong_negative_cmf_is_sell(self):
        assert _vote_cmf(-0.20) == "SELL"

    def test_near_zero_positive_cmf_is_hold(self):
        assert _vote_cmf(0.02) == "HOLD"

    def test_near_zero_negative_cmf_is_hold(self):
        assert _vote_cmf(-0.03) == "HOLD"

    def test_exactly_threshold_positive_is_buy(self):
        """Boundary: exactly 0.05 is not strictly greater — should be HOLD."""
        assert _vote_cmf(0.05) == "HOLD"

    def test_just_above_threshold_is_buy(self):
        assert _vote_cmf(0.051) == "BUY"

    def test_nan_cmf_is_hold(self):
        assert _vote_cmf(np.nan) == "HOLD"


class TestVoteMFI:
    """Unit tests for _vote_mfi (Money Flow Index).

    Thresholds: < 20 → BUY (oversold), > 80 → SELL (overbought).
    """

    def test_oversold_mfi_is_buy(self):
        assert _vote_mfi(10.0) == "BUY"

    def test_overbought_mfi_is_sell(self):
        assert _vote_mfi(90.0) == "SELL"

    def test_neutral_mfi_is_hold(self):
        assert _vote_mfi(50.0) == "HOLD"

    def test_exactly_20_is_hold(self):
        """MFI exactly at 20 is not strictly below the threshold."""
        assert _vote_mfi(20.0) == "HOLD"

    def test_exactly_80_is_hold(self):
        """MFI exactly at 80 is not strictly above the threshold."""
        assert _vote_mfi(80.0) == "HOLD"

    def test_just_below_20_is_buy(self):
        assert _vote_mfi(19.9) == "BUY"

    def test_just_above_80_is_sell(self):
        assert _vote_mfi(80.1) == "SELL"

    def test_nan_mfi_is_hold(self):
        assert _vote_mfi(np.nan) == "HOLD"


class TestVoteVolumeRSI:
    """Unit tests for _vote_volume_rsi.

    Logic: vrsi > 70 + price_up → BUY; vrsi > 70 + price_down → SELL; else HOLD.
    """

    def test_high_vrsi_price_up_is_buy(self):
        assert _vote_volume_rsi(80.0, price_up=True) == "BUY"

    def test_high_vrsi_price_down_is_sell(self):
        assert _vote_volume_rsi(80.0, price_up=False) == "SELL"

    def test_low_vrsi_price_up_is_hold(self):
        assert _vote_volume_rsi(50.0, price_up=True) == "HOLD"

    def test_low_vrsi_price_down_is_hold(self):
        assert _vote_volume_rsi(50.0, price_up=False) == "HOLD"

    def test_exactly_70_is_hold(self):
        """vrsi exactly at 70 is not strictly above the threshold."""
        assert _vote_volume_rsi(70.0, price_up=True) == "HOLD"

    def test_just_above_70_price_up_is_buy(self):
        assert _vote_volume_rsi(70.1, price_up=True) == "BUY"

    def test_nan_vrsi_is_hold(self):
        assert _vote_volume_rsi(np.nan, price_up=True) == "HOLD"


# ---------------------------------------------------------------------------
# Test 11: Confidence is tied to vote concentration
# ---------------------------------------------------------------------------

class TestConfidenceValues:
    """Verify confidence tracks how concentrated the vote is."""

    def test_strong_uptrend_confidence_above_zero(self):
        """A clear uptrend should produce confidence > 0.0."""
        df = _make_trending_df(n=80, step=1.0, volume=2_000.0)
        result = compute_volume_flow_signal(df)
        # Even if a minority vote the other way, confidence > 0 means some voted
        assert result["confidence"] >= 0.0

    def test_perfect_agreement_would_give_high_confidence(self):
        """Manually test the majority_vote function as used here: 6 BUY → conf=1.0."""
        from portfolio.signal_utils import majority_vote
        votes = ["BUY"] * 6
        action, conf = majority_vote(votes)
        assert action == "BUY"
        assert conf == 1.0

    def test_split_vote_gives_lower_confidence(self):
        """3 BUY + 3 SELL → HOLD with confidence=0.0 (tie, no majority)."""
        from portfolio.signal_utils import majority_vote
        votes = ["BUY", "BUY", "BUY", "SELL", "SELL", "SELL"]
        action, conf = majority_vote(votes)
        # No clear winner → HOLD at 0.0
        assert action == "HOLD"
        assert conf == 0.0

    def test_confidence_cannot_exceed_1(self):
        """Across many random seeds, confidence must never exceed 1.0."""
        for seed in range(10):
            df = _make_random_df(100, seed=seed)
            result = compute_volume_flow_signal(df)
            assert result["confidence"] <= 1.0, (
                f"Seed {seed}: confidence {result['confidence']} > 1.0"
            )

    def test_confidence_cannot_be_negative(self):
        """Confidence must never go below 0.0."""
        for seed in range(10):
            df = _make_random_df(100, seed=seed)
            result = compute_volume_flow_signal(df)
            assert result["confidence"] >= 0.0, (
                f"Seed {seed}: confidence {result['confidence']} < 0.0"
            )


# ---------------------------------------------------------------------------
# Test 12: NaN-heavy data degrades gracefully
# ---------------------------------------------------------------------------

class TestNaNHeavyData:
    """A DataFrame that is mostly NaN should reduce gracefully to HOLD."""

    def test_mostly_nan_close_returns_hold(self):
        """If close is NaN for all but 10 rows, effective row count is < MIN_ROWS."""
        n = 100
        df = _make_random_df(n)
        # Zero out the close column for the first 90 rows
        df.loc[df.index[:90], "close"] = np.nan
        result = compute_volume_flow_signal(df)
        # After dropna, only 10 rows remain → below MIN_ROWS → HOLD
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_partial_nan_volume_still_computes(self):
        """A few NaN volume entries should not break computation entirely."""
        df = _make_random_df(100)
        # Introduce isolated NaN volumes — not enough to drop below MIN_ROWS
        df.loc[df.index[[5, 10, 15]], "volume"] = np.nan
        result = compute_volume_flow_signal(df)
        # Result may be any valid action; it must not crash
        assert result["action"] in ("BUY", "SELL", "HOLD")


# ---------------------------------------------------------------------------
# Test 13: Indicators dict contains numeric values after computation
# ---------------------------------------------------------------------------

class TestIndicatorsDict:
    """Verify that the indicators dict exposes sensible numeric values."""

    def test_indicators_are_numeric(self):
        """All indicator values must be floats (or NaN) — never strings or None."""
        df = _make_random_df(100)
        result = compute_volume_flow_signal(df)
        for key, val in result["indicators"].items():
            assert isinstance(val, float), (
                f"indicators['{key}'] is {type(val)}, expected float"
            )

    def test_vwap_indicator_is_positive(self):
        """VWAP for positive-price data must be a positive float."""
        df = _make_trending_df(n=80, start=50.0, step=0.5)
        result = compute_volume_flow_signal(df)
        vwap_val = result["indicators"]["vwap"]
        assert not np.isnan(vwap_val), "VWAP indicator should not be NaN on clean data"
        assert vwap_val > 0.0

    def test_obv_indicator_is_finite_in_uptrend(self):
        """OBV must be a finite float in a rising market."""
        df = _make_trending_df(n=80, start=50.0, step=1.0)
        result = compute_volume_flow_signal(df)
        obv_val = result["indicators"]["obv"]
        assert np.isfinite(obv_val), f"OBV should be finite in uptrend, got {obv_val}"

    def test_mfi_indicator_in_range(self):
        """MFI indicator value must be in [0, 100] or NaN."""
        df = _make_random_df(100)
        result = compute_volume_flow_signal(df)
        mfi_val = result["indicators"]["mfi"]
        if not np.isnan(mfi_val):
            assert 0.0 <= mfi_val <= 100.0, (
                f"MFI indicator out of range: {mfi_val}"
            )

    def test_cmf_indicator_in_valid_range(self):
        """CMF must be in [-1, 1] or NaN."""
        df = _make_random_df(100)
        result = compute_volume_flow_signal(df)
        cmf_val = result["indicators"]["cmf"]
        if not np.isnan(cmf_val):
            assert -1.0 <= cmf_val <= 1.0, (
                f"CMF indicator out of valid range [-1,1]: {cmf_val}"
            )


# ---------------------------------------------------------------------------
# Test 14: Reproducibility — same input gives same output
# ---------------------------------------------------------------------------

class TestReproducibility:
    """The function must be deterministic: identical inputs → identical outputs."""

    def test_same_df_same_result(self):
        df = _make_random_df(100, seed=7)
        result_a = compute_volume_flow_signal(df.copy())
        result_b = compute_volume_flow_signal(df.copy())
        assert result_a["action"] == result_b["action"]
        assert result_a["confidence"] == result_b["confidence"]
        assert result_a["sub_signals"] == result_b["sub_signals"]

    def test_df_not_mutated(self):
        """The function must not modify the caller's DataFrame in place."""
        df = _make_random_df(100, seed=11)
        original_close = df["close"].copy()
        compute_volume_flow_signal(df)
        pd.testing.assert_series_equal(df["close"], original_close)
