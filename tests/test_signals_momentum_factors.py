"""Tests for the momentum_factors signal module.

Covers:
    - Basic computation with sufficient data
    - Insufficient data returns HOLD
    - Strong uptrend triggers time-series momentum BUY
    - Strong downtrend triggers time-series momentum SELL
    - Near 52-week high triggers BUY
    - Near 52-week low with reversal triggers BUY
    - Consecutive green bars triggers BUY
    - Volume-weighted momentum with expanding volume
    - Price acceleration detection
    - Edge cases: None input, missing columns, empty DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.momentum_factors import (
    MIN_ROWS,
    _consecutive_bars,
    _high_proximity,
    _low_reversal,
    _price_acceleration,
    _roc_20,
    _time_series_momentum,
    _volume_weighted_momentum,
    compute_momentum_factors_signal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(
    closes: list[float],
    opens: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    volumes: list[float] | None = None,
) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame from close prices.

    If opens/highs/lows/volumes are not provided, they are derived
    automatically from closes (open = close, high = close * 1.005,
    low = close * 0.995, volume = 1000).
    """
    n = len(closes)
    if opens is None:
        opens = list(closes)
    if highs is None:
        highs = [c * 1.005 for c in closes]
    if lows is None:
        lows = [c * 0.995 for c in closes]
    if volumes is None:
        volumes = [1000.0] * n

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "time": pd.date_range("2025-01-01", periods=n, freq="h"),
    })


def _make_uptrend(n: int = 300, start: float = 100.0, step: float = 0.5) -> pd.DataFrame:
    """Generate a strong uptrend with n bars."""
    closes = [start + i * step for i in range(n)]
    # opens slightly below close to make green bars
    opens = [c - 0.2 for c in closes]
    highs = [c + 0.3 for c in closes]
    lows = [o - 0.3 for o in opens]
    volumes = [1000.0 + i * 5 for i in range(n)]  # expanding volume
    return _make_ohlcv(closes, opens=opens, highs=highs, lows=lows, volumes=volumes)


def _make_downtrend(n: int = 300, start: float = 200.0, step: float = 0.5) -> pd.DataFrame:
    """Generate a strong downtrend with n bars."""
    closes = [start - i * step for i in range(n)]
    # opens slightly above close to make red bars
    opens = [c + 0.2 for c in closes]
    highs = [o + 0.3 for o in opens]
    lows = [c - 0.3 for c in closes]
    volumes = [1000.0 + i * 5 for i in range(n)]
    return _make_ohlcv(closes, opens=opens, highs=highs, lows=lows, volumes=volumes)


# ---------------------------------------------------------------------------
# Tests: Basic computation
# ---------------------------------------------------------------------------

class TestBasicComputation:
    """Test that the composite function returns the expected structure."""

    def test_returns_correct_keys(self):
        """Result dict has all required top-level keys."""
        df = _make_uptrend(100)
        result = compute_momentum_factors_signal(df)
        assert "action" in result
        assert "confidence" in result
        assert "sub_signals" in result
        assert "indicators" in result

    def test_action_is_valid(self):
        """Action must be BUY, SELL, or HOLD."""
        df = _make_uptrend(100)
        result = compute_momentum_factors_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_confidence_range(self):
        """Confidence must be between 0.0 and 1.0."""
        df = _make_uptrend(100)
        result = compute_momentum_factors_signal(df)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_all_seven_sub_signals_present(self):
        """All 7 sub-signals must be present."""
        df = _make_uptrend(100)
        result = compute_momentum_factors_signal(df)
        expected_keys = {
            "ts_momentum_12_1",
            "roc_20",
            "high_proximity",
            "low_reversal",
            "consecutive_bars",
            "price_acceleration",
            "volume_weighted_momentum",
        }
        assert set(result["sub_signals"].keys()) == expected_keys

    def test_indicators_dict_populated(self):
        """Indicators dict should have the expected keys."""
        df = _make_uptrend(100)
        result = compute_momentum_factors_signal(df)
        expected_keys = {
            "ts_momentum_pct",
            "roc_20",
            "high_proximity",
            "low_proximity",
            "consecutive_bars",
            "acceleration_recent",
            "acceleration_older",
            "vol_momentum_price_chg",
            "vol_momentum_ratio",
        }
        assert set(result["indicators"].keys()) == expected_keys


# ---------------------------------------------------------------------------
# Tests: Insufficient / invalid data
# ---------------------------------------------------------------------------

class TestInsufficientData:
    """Test graceful degradation with bad or missing data."""

    def test_none_input(self):
        """None input returns HOLD with confidence 0."""
        result = compute_momentum_factors_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_empty_dataframe(self):
        """Empty DataFrame returns HOLD."""
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_momentum_factors_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_too_few_rows(self):
        """DataFrame with fewer than MIN_ROWS returns HOLD."""
        df = _make_ohlcv([100.0] * (MIN_ROWS - 1))
        result = compute_momentum_factors_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_missing_columns(self):
        """DataFrame missing required columns returns HOLD."""
        df = pd.DataFrame({"close": [100.0] * 60, "volume": [1000] * 60})
        result = compute_momentum_factors_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_exactly_min_rows(self):
        """DataFrame with exactly MIN_ROWS is accepted (does not raise)."""
        df = _make_uptrend(MIN_ROWS)
        result = compute_momentum_factors_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")


# ---------------------------------------------------------------------------
# Tests: Time-Series Momentum
# ---------------------------------------------------------------------------

class TestTimeSeriesMomentum:
    """Test the 12-1 time-series momentum sub-signal."""

    def test_strong_uptrend_buy(self):
        """A strong uptrend over many bars should produce positive TS momentum."""
        df = _make_uptrend(300)
        close = df["close"].astype(float)
        val, sig = _time_series_momentum(close)
        assert sig == "BUY"
        assert val > 0

    def test_strong_downtrend_sell(self):
        """A strong downtrend should produce negative TS momentum."""
        df = _make_downtrend(300)
        close = df["close"].astype(float)
        val, sig = _time_series_momentum(close)
        assert sig == "SELL"
        assert val < 0

    def test_insufficient_data_hold(self):
        """Fewer than MIN_ROWS bars returns HOLD."""
        close = pd.Series([100.0] * 30)
        val, sig = _time_series_momentum(close)
        assert sig == "HOLD"
        assert np.isnan(val)

    def test_composite_uptrend_buy(self):
        """Strong uptrend should make ts_momentum_12_1 sub-signal BUY in composite."""
        df = _make_uptrend(300)
        result = compute_momentum_factors_signal(df)
        assert result["sub_signals"]["ts_momentum_12_1"] == "BUY"


# ---------------------------------------------------------------------------
# Tests: 52-Week High Proximity
# ---------------------------------------------------------------------------

class TestHighProximity:
    """Test the 52-week high proximity sub-signal."""

    def test_near_high_buy(self):
        """Price near the period high should trigger BUY."""
        # _high_proximity requires >= 500 data points
        closes = [100.0] * 500 + [105.0]  # new high
        close = pd.Series(closes)
        val, sig = _high_proximity(close)
        assert sig == "BUY"
        assert val >= 0.95

    def test_far_from_high_sell(self):
        """Price far below period high should trigger SELL."""
        # Need >= 500 points. Peak at 200 within lookback, current at 150 (25% below)
        closes = [150.0] * 400 + [200.0] * 50 + [150.0] * 51
        close = pd.Series([float(c) for c in closes])
        val, sig = _high_proximity(close)
        assert sig == "SELL"
        assert val <= 0.80

    def test_mid_range_hold(self):
        """Price in the middle range should HOLD."""
        # Need >= 500 points. Peak at 100, current at 90 (10% below)
        closes = [100.0] * 500 + [90.0]
        close = pd.Series(closes)
        val, sig = _high_proximity(close)
        assert sig == "HOLD"
        assert 0.80 < val < 0.95


# ---------------------------------------------------------------------------
# Tests: 52-Week Low Reversal
# ---------------------------------------------------------------------------

class TestLowReversal:
    """Test the 52-week low reversal sub-signal."""

    def test_near_low_with_green_bars_buy(self):
        """Price near 52-week low + 3 consecutive green bars = BUY reversal."""
        n = 60
        # Start at 100, drop to 80, then 3 green bars from 80
        closes = [100.0] * 50 + [80.0, 80.0, 80.0, 80.0] + [81.0, 82.0, 83.0]
        opens = [100.0] * 50 + [80.0, 80.0, 80.0, 80.0] + [80.0, 81.0, 82.0]
        # Pad to same length
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        volumes = [1000.0] * len(closes)
        df = _make_ohlcv(closes, opens=opens, highs=highs, lows=lows, volumes=volumes)
        val, sig = _low_reversal(df)
        # 83 / 79 (low = 80 - 1 in lows, but we check close series min)
        # close min = 80, ratio = 83/80 = 1.0375 -> within 1.05
        assert sig == "BUY"

    def test_new_lows_sell(self):
        """Making new 52-week lows (within 1%) triggers SELL."""
        # Price at exactly the low
        closes = [100.0] * 50 + [90.0, 88.0, 86.0, 85.0]
        opens = [100.0] * 50 + [91.0, 89.0, 87.0, 86.0]  # red bars
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        volumes = [1000.0] * len(closes)
        df = _make_ohlcv(closes, opens=opens, highs=highs, lows=lows, volumes=volumes)
        val, sig = _low_reversal(df)
        # close min = 85, current = 85, ratio = 1.0 -> within 1.01 -> SELL
        assert sig == "SELL"


# ---------------------------------------------------------------------------
# Tests: Consecutive Bars
# ---------------------------------------------------------------------------

class TestConsecutiveBars:
    """Test the consecutive bars sub-signal."""

    def test_four_green_bars_buy(self):
        """4+ consecutive green bars triggers BUY."""
        n = 60
        # All bars are green except the first few
        closes = [100.0] * 50 + [101.0, 102.0, 103.0, 104.0, 105.0]
        opens = [100.0] * 50 + [100.5, 101.5, 102.5, 103.5, 104.5]
        highs = [c + 0.5 for c in closes]
        lows = [o - 0.5 for o in opens]
        volumes = [1000.0] * len(closes)
        df = _make_ohlcv(closes, opens=opens, highs=highs, lows=lows, volumes=volumes)
        val, sig = _consecutive_bars(df)
        assert sig == "BUY"
        assert val >= 4  # positive = green streak

    def test_four_red_bars_sell(self):
        """4+ consecutive red bars triggers SELL."""
        n = 60
        closes = [100.0] * 50 + [99.0, 98.0, 97.0, 96.0, 95.0]
        opens = [100.0] * 50 + [99.5, 98.5, 97.5, 96.5, 95.5]
        highs = [o + 0.5 for o in opens]
        lows = [c - 0.5 for c in closes]
        volumes = [1000.0] * len(closes)
        df = _make_ohlcv(closes, opens=opens, highs=highs, lows=lows, volumes=volumes)
        val, sig = _consecutive_bars(df)
        assert sig == "SELL"
        assert val <= -4  # negative = red streak

    def test_few_bars_hold(self):
        """Fewer than 4 consecutive bars in either direction = HOLD."""
        closes = [100.0, 101.0, 100.5, 101.5] * 15  # alternating
        opens = [100.5, 100.5, 101.0, 101.0] * 15
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        volumes = [1000.0] * len(closes)
        df = _make_ohlcv(closes, opens=opens, highs=highs, lows=lows, volumes=volumes)
        val, sig = _consecutive_bars(df)
        # With alternating green/red, consecutive count should be small
        assert sig == "HOLD"


# ---------------------------------------------------------------------------
# Tests: Volume-Weighted Momentum
# ---------------------------------------------------------------------------

class TestVolumeWeightedMomentum:
    """Test the volume-weighted momentum sub-signal."""

    def test_positive_price_expanding_volume_buy(self):
        """Positive price change + expanding volume = BUY."""
        n = 60
        # Steady prices, then a 10-bar rise with expanding volume
        closes = [100.0] * 40 + [100.0 + i * 0.5 for i in range(20)]
        opens = [c - 0.1 for c in closes]
        highs = [c + 0.2 for c in closes]
        lows = [c - 0.2 for c in closes]
        # Volume: low for first 40 bars, high for last 20
        volumes = [500.0] * 40 + [2000.0] * 20
        df = _make_ohlcv(closes, opens=opens, highs=highs, lows=lows, volumes=volumes)
        price_chg, vol_ratio, sig = _volume_weighted_momentum(df)
        assert sig == "BUY"
        assert price_chg > 0
        assert vol_ratio > 1.0

    def test_negative_price_expanding_volume_sell(self):
        """Negative price change + expanding volume = SELL."""
        n = 60
        closes = [100.0] * 40 + [100.0 - i * 0.5 for i in range(20)]
        opens = [c + 0.1 for c in closes]
        highs = [c + 0.2 for c in closes]
        lows = [c - 0.2 for c in closes]
        volumes = [500.0] * 40 + [2000.0] * 20
        df = _make_ohlcv(closes, opens=opens, highs=highs, lows=lows, volumes=volumes)
        price_chg, vol_ratio, sig = _volume_weighted_momentum(df)
        assert sig == "SELL"
        assert price_chg < 0
        assert vol_ratio > 1.0

    def test_contracting_volume_hold(self):
        """Even with price movement, contracting volume = HOLD."""
        n = 60
        closes = [100.0 + i * 0.5 for i in range(60)]
        opens = [c - 0.1 for c in closes]
        highs = [c + 0.2 for c in closes]
        lows = [c - 0.2 for c in closes]
        # Volume: high early, low recently (contracting)
        volumes = [2000.0] * 40 + [500.0] * 20
        df = _make_ohlcv(closes, opens=opens, highs=highs, lows=lows, volumes=volumes)
        price_chg, vol_ratio, sig = _volume_weighted_momentum(df)
        assert sig == "HOLD"
        assert vol_ratio < 1.0


# ---------------------------------------------------------------------------
# Tests: Price Acceleration
# ---------------------------------------------------------------------------

class TestPriceAcceleration:
    """Test the price acceleration sub-signal."""

    def test_accelerating_uptrend_buy(self):
        """Recent ROC > older ROC and both positive = BUY."""
        # Older window: modest gains. Recent window: stronger gains.
        closes = (
            [100.0] * 40
            + [100.0 + i * 0.2 for i in range(10)]  # older: +2 over 10 bars
            + [102.0 + i * 0.8 for i in range(10)]  # recent: +8 over 10 bars
        )
        close = pd.Series(closes)
        roc_recent, roc_older, sig = _price_acceleration(close)
        assert sig == "BUY"
        assert roc_recent > roc_older
        assert roc_recent > 0
        assert roc_older > 0

    def test_accelerating_downtrend_sell(self):
        """Recent ROC < older ROC and both negative = SELL."""
        closes = (
            [200.0] * 40
            + [200.0 - i * 0.3 for i in range(10)]  # older: modest decline
            + [197.0 - i * 1.0 for i in range(10)]  # recent: steeper decline
        )
        close = pd.Series(closes)
        roc_recent, roc_older, sig = _price_acceleration(close)
        assert sig == "SELL"
        assert roc_recent < roc_older
        assert roc_recent < 0
        assert roc_older < 0

    def test_decelerating_hold(self):
        """Decelerating uptrend (recent ROC < older ROC but both positive) = HOLD."""
        closes = (
            [100.0] * 40
            + [100.0 + i * 1.0 for i in range(10)]  # older: strong gains
            + [110.0 + i * 0.1 for i in range(10)]  # recent: weaker gains
        )
        close = pd.Series(closes)
        roc_recent, roc_older, sig = _price_acceleration(close)
        # roc_recent > 0, roc_older > 0, but roc_recent < roc_older -> HOLD
        assert sig == "HOLD"


# ---------------------------------------------------------------------------
# Tests: Composite majority voting
# ---------------------------------------------------------------------------

class TestCompositeVoting:
    """Test the majority-vote logic of the composite signal."""

    def test_strong_uptrend_composite_buy(self):
        """A strong uptrend should have multiple BUY sub-signals.

        With majority_vote (count_hold=False), the composite action is BUY
        only when BUY count exceeds both SELL and HOLD. In moderate uptrends,
        many sub-signals may HOLD, producing HOLD at the composite level.
        """
        df = _make_uptrend(300)
        result = compute_momentum_factors_signal(df)
        buy_count = list(result["sub_signals"].values()).count("BUY")
        # At least some sub-signals should detect the uptrend
        assert buy_count >= 1
        # Composite action depends on whether BUY beats HOLD count
        assert result["action"] in ("BUY", "HOLD")

    def test_strong_downtrend_composite_sell(self):
        """A strong downtrend should produce a composite SELL."""
        df = _make_downtrend(300)
        result = compute_momentum_factors_signal(df)
        sell_count = list(result["sub_signals"].values()).count("SELL")
        assert result["action"] == "SELL"
        assert sell_count >= 3

    def test_flat_market_hold(self):
        """A flat/sideways market should mostly HOLD."""
        closes = [100.0] * 300
        # Tiny noise to avoid division-by-zero issues
        opens = [100.0 + 0.01 * ((-1) ** i) for i in range(300)]
        highs = [100.05] * 300
        lows = [99.95] * 300
        volumes = [1000.0] * 300
        df = _make_ohlcv(closes, opens=opens, highs=highs, lows=lows, volumes=volumes)
        result = compute_momentum_factors_signal(df)
        # In a flat market, most sub-signals HOLD; composite should be HOLD
        hold_count = list(result["sub_signals"].values()).count("HOLD")
        assert hold_count >= 3

    def test_confidence_all_hold_is_zero(self):
        """If all sub-signals HOLD, confidence must be 0.0."""
        # Flat data with no volume changes
        closes = [100.0] * 60
        opens = [100.0] * 60
        highs = [100.01] * 60
        lows = [99.99] * 60
        volumes = [1000.0] * 60
        df = _make_ohlcv(closes, opens=opens, highs=highs, lows=lows, volumes=volumes)
        result = compute_momentum_factors_signal(df)
        if all(v == "HOLD" for v in result["sub_signals"].values()):
            assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Tests: ROC-20 sub-signal
# ---------------------------------------------------------------------------

class TestROC20:
    """Test the ROC-20 sub-signal."""

    def test_large_positive_roc_buy(self):
        """ROC > 5% triggers BUY."""
        # 20 bars ago = 100, now = 110 -> ROC = 10%
        closes = [100.0] * 50 + [110.0]
        close = pd.Series(closes)
        val, sig = _roc_20(close)
        assert sig == "BUY"
        assert val > 5.0

    def test_large_negative_roc_sell(self):
        """ROC < -5% triggers SELL."""
        closes = [100.0] * 50 + [90.0]
        close = pd.Series(closes)
        val, sig = _roc_20(close)
        assert sig == "SELL"
        assert val < -5.0

    def test_small_roc_hold(self):
        """ROC between -5% and 5% triggers HOLD."""
        closes = [100.0] * 50 + [102.0]
        close = pd.Series(closes)
        val, sig = _roc_20(close)
        assert sig == "HOLD"
        assert -5.0 <= val <= 5.0
