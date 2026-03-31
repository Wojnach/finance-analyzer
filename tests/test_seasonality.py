"""Tests for seasonality detrending module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.seasonality import (
    compute_hourly_profile,
    detrend_return,
    normalize_volatility,
)


def _make_hourly_data(n_days=10, base_price=30.0):
    """Generate n_days of hourly OHLCV with a known seasonal pattern."""
    n_hours = n_days * 24
    idx = pd.date_range("2026-01-01", periods=n_hours, freq="1h")
    np.random.seed(42)
    # Inject a known hour-of-day pattern: hour 14 (US open) has higher returns
    returns = np.random.randn(n_hours) * 0.001
    for i in range(n_hours):
        if idx[i].hour == 14:
            returns[i] += 0.003  # persistent positive bias at hour 14
        if idx[i].hour == 3:
            returns[i] -= 0.002  # persistent negative bias at hour 3
    prices = base_price * np.cumprod(1 + returns)
    return pd.DataFrame({"close": prices}, index=idx)


class TestComputeHourlyProfile:
    def test_returns_profile_with_24_hours(self):
        df = _make_hourly_data(n_days=10)
        profile = compute_hourly_profile(df)
        assert profile is not None
        assert len(profile) == 24
        for h in range(24):
            assert str(h) in profile
            assert "mean_return" in profile[str(h)]
            assert "mean_volatility" in profile[str(h)]
            assert "count" in profile[str(h)]

    def test_detects_hour14_positive_bias(self):
        df = _make_hourly_data(n_days=20)
        profile = compute_hourly_profile(df)
        # Hour 14 should have higher mean return than most hours
        h14_return = profile["14"]["mean_return"]
        other_returns = [profile[str(h)]["mean_return"]
                        for h in range(24) if h not in (14, 3)]
        avg_other = np.mean(other_returns)
        assert h14_return > avg_other

    def test_detects_hour3_negative_bias(self):
        df = _make_hourly_data(n_days=20)
        profile = compute_hourly_profile(df)
        h3_return = profile["3"]["mean_return"]
        assert h3_return < 0

    def test_insufficient_data_returns_none(self):
        df = pd.DataFrame({"close": [30.0] * 10},
                         index=pd.date_range("2026-01-01", periods=10, freq="1h"))
        profile = compute_hourly_profile(df)
        assert profile is None

    def test_non_datetime_index_returns_none(self):
        df = pd.DataFrame({"close": range(200)})
        profile = compute_hourly_profile(df)
        assert profile is None


class TestDetrendReturn:
    def test_removes_seasonal_bias(self):
        profile = {"14": {"mean_return": 0.003, "mean_volatility": 0.005, "count": 10}}
        raw = 0.005
        detrended = detrend_return(raw, hour=14, profile=profile)
        assert detrended == pytest.approx(0.002)  # 0.005 - 0.003

    def test_none_profile_returns_raw(self):
        assert detrend_return(0.005, 14, None) == 0.005

    def test_missing_hour_returns_raw(self):
        profile = {"0": {"mean_return": 0.001, "mean_volatility": 0.003, "count": 5}}
        assert detrend_return(0.005, 14, profile) == 0.005


class TestNormalizeVolatility:
    def test_normalizes_to_seasonal_average(self):
        profile = {"14": {"mean_return": 0.003, "mean_volatility": 0.005, "count": 10}}
        raw_vol = 0.010
        normalized = normalize_volatility(raw_vol, hour=14, profile=profile)
        assert normalized == pytest.approx(2.0)  # 0.010 / 0.005

    def test_none_profile_returns_raw(self):
        assert normalize_volatility(0.005, 14, None) == 0.005

    def test_zero_mean_vol_returns_raw(self):
        profile = {"14": {"mean_return": 0.0, "mean_volatility": 0.0, "count": 0}}
        assert normalize_volatility(0.005, 14, profile) == 0.005
