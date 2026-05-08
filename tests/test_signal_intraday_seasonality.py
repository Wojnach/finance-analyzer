"""Tests for intraday_seasonality signal module."""
import datetime

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.intraday_seasonality import (
    _CRYPTO_HOUR_MULT,
    _METALS_HOUR_MULT,
    _STOCKS_HOUR_MULT,
    _classify_asset,
    _get_utc_hour_and_dow,
    _hour_alpha_vote,
    compute_intraday_seasonality_signal,
)


def _make_df(n=100, trend="up", with_datetime_index=False, hour=22):
    """Create test DataFrame with realistic OHLCV data."""
    np.random.seed(42)
    if trend == "up":
        close = 100 + np.cumsum(np.abs(np.random.randn(n) * 0.3))
    elif trend == "down":
        close = 200 - np.cumsum(np.abs(np.random.randn(n) * 0.3))
    else:
        close = 100 + np.random.randn(n) * 0.5

    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })

    if with_datetime_index:
        base = datetime.datetime(2026, 5, 5, hour, 0, 0)
        idx = pd.DatetimeIndex([base + datetime.timedelta(hours=i) for i in range(n)])
        df.index = idx

    return df


class TestSignalInterface:
    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        result = compute_intraday_seasonality_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        result = compute_intraday_seasonality_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)
        assert "hour_alpha" in result["sub_signals"]
        assert "day_of_week" in result["sub_signals"]
        assert "trend_context" in result["sub_signals"]

    def test_has_indicators(self):
        df = _make_df()
        result = compute_intraday_seasonality_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)
        assert "utc_hour" in result["indicators"]
        assert "day_of_week" in result["indicators"]
        assert "hour_multiplier" in result["indicators"]
        assert "combined_multiplier" in result["indicators"]

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_intraday_seasonality_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=3)
        result = compute_intraday_seasonality_signal(df)
        assert result["action"] == "HOLD"

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        result = compute_intraday_seasonality_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_crypto_context(self):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto"}
        result = compute_intraday_seasonality_signal(df, context=ctx)
        assert result["indicators"]["asset_class"] == "crypto"

    def test_with_metals_context(self):
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "asset_class": "metals"}
        result = compute_intraday_seasonality_signal(df, context=ctx)
        assert result["indicators"]["asset_class"] == "metals"

    def test_with_stocks_context(self):
        df = _make_df()
        ctx = {"ticker": "MSTR", "asset_class": "stocks"}
        result = compute_intraday_seasonality_signal(df, context=ctx)
        assert result["indicators"]["asset_class"] == "stocks"

    def test_confidence_capped_at_0_7(self):
        df = _make_df(trend="up")
        result = compute_intraday_seasonality_signal(df)
        assert result["confidence"] <= 0.7


class TestHourMultipliers:
    def test_all_hours_covered_crypto(self):
        for h in range(24):
            assert h in _CRYPTO_HOUR_MULT

    def test_all_hours_covered_metals(self):
        for h in range(24):
            assert h in _METALS_HOUR_MULT

    def test_all_hours_covered_stocks(self):
        for h in range(24):
            assert h in _STOCKS_HOUR_MULT

    def test_crypto_peak_hours_highest(self):
        peak_hours = [21, 22, 23]
        non_peak = [14, 15, 16, 17]
        for p in peak_hours:
            for np_h in non_peak:
                assert _CRYPTO_HOUR_MULT[p] > _CRYPTO_HOUR_MULT[np_h]

    def test_metals_london_ny_overlap_highest(self):
        overlap_hours = [13, 14, 15]
        asian_hours = [2, 3, 4, 5]
        for o in overlap_hours:
            for a in asian_hours:
                assert _METALS_HOUR_MULT[o] > _METALS_HOUR_MULT[a]


class TestAssetClassification:
    def test_default_is_crypto(self):
        assert _classify_asset(None) == "crypto"
        assert _classify_asset({}) == "crypto"

    def test_explicit_asset_class(self):
        assert _classify_asset({"asset_class": "metals"}) == "metals"
        assert _classify_asset({"asset_class": "stocks"}) == "stocks"

    def test_ticker_inference(self):
        assert _classify_asset({"ticker": "XAU-USD"}) == "metals"
        assert _classify_asset({"ticker": "XAG-USD"}) == "metals"
        assert _classify_asset({"ticker": "MSTR"}) == "stocks"
        assert _classify_asset({"ticker": "BTC-USD"}) == "crypto"


class TestDatetimeIndex:
    def test_extracts_hour_from_index(self):
        df = _make_df(n=50, with_datetime_index=True, hour=22)
        hour, dow = _get_utc_hour_and_dow(df)
        assert hour == (22 + 49) % 24

    def test_high_alpha_hour_crypto(self):
        df = _make_df(n=50, trend="up", with_datetime_index=True, hour=22)
        ctx = {"asset_class": "crypto"}
        result = compute_intraday_seasonality_signal(df, context=ctx)
        assert result["indicators"]["hour_multiplier"] >= 1.0

    def test_suppress_hour_metals(self):
        df = _make_df(n=50, with_datetime_index=True, hour=3)
        ctx = {"asset_class": "metals"}
        result = compute_intraday_seasonality_signal(df, context=ctx)
        assert result["indicators"]["hour_multiplier"] <= 0.7


class TestHourAlphaVote:
    def test_high_alpha_returns_buy(self):
        vote, mult = _hour_alpha_vote(22, "crypto")
        assert vote == "BUY"
        assert mult >= 1.2

    def test_suppress_returns_hold(self):
        vote, mult = _hour_alpha_vote(3, "metals")
        assert vote == "HOLD"
        assert mult <= 0.5

    def test_neutral_returns_hold(self):
        vote, mult = _hour_alpha_vote(10, "crypto")
        assert vote == "HOLD"


class TestDirectionality:
    def test_uptrend_produces_buy(self):
        df = _make_df(n=50, trend="up")
        result = compute_intraday_seasonality_signal(df)
        if result["action"] != "HOLD":
            assert result["action"] == "BUY"

    def test_downtrend_produces_sell(self):
        df = _make_df(n=50, trend="down")
        result = compute_intraday_seasonality_signal(df)
        if result["action"] != "HOLD":
            assert result["action"] == "SELL"
