"""Tests for sentiment_extremity_gate signal module."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from portfolio.signals.sentiment_extremity_gate import (
    compute_sentiment_extremity_gate_signal,
    _intensity_zone,
    _price_in_range,
    _range_compression,
)


def _make_df(n=100, base_price=100.0, volatility=0.5):
    np.random.seed(42)
    close = base_price + np.cumsum(np.random.randn(n) * volatility)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_ranging_df(n=100, base=100.0, amplitude=2.0):
    """Create a range-bound DataFrame oscillating around base."""
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, n)
    close = base + amplitude * np.sin(t) + np.random.randn(n) * 0.1
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.05,
        "high": close + abs(np.random.randn(n) * 0.2),
        "low": close - abs(np.random.randn(n) * 0.2),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


class TestIntensityZone:
    def test_moderate_center(self):
        intensity, vote = _intensity_zone(50)
        assert vote == "PASS"
        assert intensity == 0.0

    def test_moderate_edge(self):
        intensity, vote = _intensity_zone(31)
        assert vote == "PASS"
        assert intensity == 19.0

    def test_transition_zone(self):
        intensity, vote = _intensity_zone(25)
        assert vote == "HOLD"
        assert intensity == 25.0

    def test_extreme_fear(self):
        intensity, vote = _intensity_zone(10)
        assert vote == "HOLD"
        assert intensity == 40.0

    def test_extreme_greed(self):
        intensity, vote = _intensity_zone(90)
        assert vote == "HOLD"
        assert intensity == 40.0

    def test_exact_boundaries(self):
        # |29-50| = 21 >= 20 → HOLD (transition zone)
        _, vote_29 = _intensity_zone(29)
        # |30-50| = 20 >= 20 → HOLD
        _, vote_30 = _intensity_zone(30)
        # |70-50| = 20 >= 20 → HOLD
        _, vote_70 = _intensity_zone(70)
        # |71-50| = 21 >= 20 → HOLD
        _, vote_71 = _intensity_zone(71)
        # |31-50| = 19 < 20 → PASS (moderate)
        _, vote_31 = _intensity_zone(31)
        _, vote_69 = _intensity_zone(69)
        assert vote_29 == "HOLD"
        assert vote_30 == "HOLD"
        assert vote_70 == "HOLD"
        assert vote_71 == "HOLD"
        assert vote_31 == "PASS"
        assert vote_69 == "PASS"


class TestPriceInRange:
    def test_at_range_low(self):
        close = pd.Series([100.0] * 19 + [90.0])
        pct, vote = _price_in_range(close)
        assert vote == "BUY"
        assert pct < 0.2

    def test_at_range_high(self):
        close = pd.Series([100.0] * 19 + [110.0])
        pct, vote = _price_in_range(close)
        assert vote == "SELL"
        assert pct > 0.8

    def test_mid_range(self):
        # Create series that ends in the middle: oscillate around 100
        close = pd.Series([98, 102, 99, 101, 100, 98, 102, 99, 101, 100,
                           98, 102, 99, 101, 100, 98, 102, 99, 101, 100.0])
        pct, vote = _price_in_range(close)
        assert vote == "HOLD"
        assert 0.2 <= pct <= 0.8

    def test_insufficient_data(self):
        close = pd.Series([100.0] * 5)
        pct, vote = _price_in_range(close)
        assert vote == "HOLD"

    def test_flat_range(self):
        close = pd.Series([100.0] * 20)
        pct, vote = _price_in_range(close)
        assert vote == "HOLD"


class TestRangeCompression:
    def test_compressed_range(self):
        # First 40 bars: wide range (H-L=4). Last 20 bars: narrow (H-L=0.4).
        # short_atr (14 bars from end) → all narrow ≈ 0.4
        # long_atr (28 bars from end) → 8 wide + 20 narrow ≈ mixed > 0.4
        # ratio = short/long < 1
        n = 60
        close = pd.Series([100.0] * n)
        high = pd.Series([102.0] * 40 + [100.2] * 20)
        low = pd.Series([98.0] * 40 + [99.8] * 20)
        ratio, vote = _range_compression(high, low, close, lookback=14)
        assert ratio < 0.5

    def test_expanding_range(self):
        # First 40 bars: narrow (H-L=0.4). Last 20 bars: wide (H-L=6).
        # short_atr → all wide ≈ 6
        # long_atr → 8 narrow + 20 wide ≈ mixed < 6
        # ratio = short/long > 1
        n = 60
        close = pd.Series([100.0] * n)
        high = pd.Series([100.2] * 40 + [103.0] * 20)
        low = pd.Series([99.8] * 40 + [97.0] * 20)
        ratio, vote = _range_compression(high, low, close, lookback=14)
        assert ratio > 1.3

    def test_insufficient_data(self):
        high = pd.Series([101.0] * 10)
        low = pd.Series([99.0] * 10)
        close = pd.Series([100.0] * 10)
        ratio, vote = _range_compression(high, low, close, lookback=14)
        assert vote == "HOLD"


class TestSignalInterface:
    @patch("portfolio.signals.sentiment_extremity_gate._get_fg_value", return_value=50)
    def test_returns_dict_with_required_keys(self, mock_fg):
        df = _make_df()
        result = compute_sentiment_extremity_gate_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    @patch("portfolio.signals.sentiment_extremity_gate._get_fg_value", return_value=50)
    def test_has_sub_signals(self, mock_fg):
        df = _make_df()
        result = compute_sentiment_extremity_gate_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    @patch("portfolio.signals.sentiment_extremity_gate._get_fg_value", return_value=50)
    def test_has_indicators(self, mock_fg):
        df = _make_df()
        result = compute_sentiment_extremity_gate_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)
        assert "fg_value" in result["indicators"]
        assert "fg_intensity" in result["indicators"]

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_sentiment_extremity_gate_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=3)
        result = compute_sentiment_extremity_gate_signal(df)
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.sentiment_extremity_gate._get_fg_value", return_value=50)
    def test_nan_handling(self, mock_fg):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        result = compute_sentiment_extremity_gate_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    @patch("portfolio.signals.sentiment_extremity_gate._get_fg_value", return_value=50)
    def test_with_context(self, mock_fg):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "ranging"}
        result = compute_sentiment_extremity_gate_signal(df, context=ctx)
        assert isinstance(result, dict)

    @patch("portfolio.signals.sentiment_extremity_gate._get_fg_value", return_value=None)
    def test_no_fg_data_returns_hold(self, mock_fg):
        df = _make_df()
        result = compute_sentiment_extremity_gate_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0


class TestExtremeRegimeGating:
    """Core thesis: extreme sentiment should force HOLD."""

    @patch("portfolio.signals.sentiment_extremity_gate._get_fg_value", return_value=10)
    def test_extreme_fear_forces_hold(self, mock_fg):
        df = _make_ranging_df()
        result = compute_sentiment_extremity_gate_signal(df)
        assert result["action"] == "HOLD"
        assert result["indicators"]["fg_intensity"] == 40.0

    @patch("portfolio.signals.sentiment_extremity_gate._get_fg_value", return_value=90)
    def test_extreme_greed_forces_hold(self, mock_fg):
        df = _make_ranging_df()
        result = compute_sentiment_extremity_gate_signal(df)
        assert result["action"] == "HOLD"
        assert result["indicators"]["fg_intensity"] == 40.0

    @patch("portfolio.signals.sentiment_extremity_gate._get_fg_value", return_value=5)
    def test_very_extreme_fear_forces_hold(self, mock_fg):
        df = _make_ranging_df()
        result = compute_sentiment_extremity_gate_signal(df)
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.sentiment_extremity_gate._get_fg_value", return_value=95)
    def test_very_extreme_greed_forces_hold(self, mock_fg):
        df = _make_ranging_df()
        result = compute_sentiment_extremity_gate_signal(df)
        assert result["action"] == "HOLD"


class TestModerateZoneTrading:
    """Core thesis: moderate sentiment enables directional signals."""

    @patch("portfolio.signals.sentiment_extremity_gate._get_fg_value", return_value=50)
    def test_neutral_fg_allows_trading(self, mock_fg):
        df = _make_ranging_df(n=100, base=100, amplitude=5)
        result = compute_sentiment_extremity_gate_signal(df)
        assert result["indicators"]["fg_intensity"] == 0.0
        assert result["sub_signals"]["intensity_zone"] == "PASS"

    @patch("portfolio.signals.sentiment_extremity_gate._get_fg_value", return_value=50)
    def test_price_at_range_low_with_neutral_fg_can_buy(self, mock_fg):
        # Need 28+ rows for range_compression (lookback=14, needs 2x)
        # Stable range first, then price drops to range low
        prices = [105.0] * 25 + [100.0] * 4 + [96.0]
        df = pd.DataFrame({
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [5000.0] * len(prices),
        })
        result = compute_sentiment_extremity_gate_signal(df)
        assert result["action"] == "BUY"
        assert result["confidence"] > 0.0

    @patch("portfolio.signals.sentiment_extremity_gate._get_fg_value", return_value=50)
    def test_price_at_range_high_with_neutral_fg_can_sell(self, mock_fg):
        prices = [95.0] * 25 + [100.0] * 4 + [104.0]
        df = pd.DataFrame({
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [5000.0] * len(prices),
        })
        result = compute_sentiment_extremity_gate_signal(df)
        assert result["action"] == "SELL"
        assert result["confidence"] > 0.0

    @patch("portfolio.signals.sentiment_extremity_gate._get_fg_value", return_value=50)
    def test_confidence_scales_with_intensity(self, mock_fg_50):
        prices = [105.0] * 25 + [100.0] * 4 + [96.0]
        df = pd.DataFrame({
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [5000.0] * len(prices),
        })
        result_neutral = compute_sentiment_extremity_gate_signal(df)

        with patch("portfolio.signals.sentiment_extremity_gate._get_fg_value", return_value=35):
            result_mild_fear = compute_sentiment_extremity_gate_signal(df)

        assert result_neutral["confidence"] >= result_mild_fear["confidence"]


class TestConfidenceCap:
    """Confidence must never exceed 0.7 (max_confidence for external data signals)."""

    @patch("portfolio.signals.sentiment_extremity_gate._get_fg_value", return_value=50)
    def test_confidence_capped_at_0_7(self, mock_fg):
        df = _make_df()
        result = compute_sentiment_extremity_gate_signal(df)
        assert result["confidence"] <= 0.7
