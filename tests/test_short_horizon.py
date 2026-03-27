"""Tests for portfolio/short_horizon.py — 3h horizon configuration."""

import pytest
from portfolio.short_horizon import (
    CONFIDENCE_CAP_3H,
    SLOW_SIGNALS_3H,
    time_of_day_scale_3h,
    is_slow_signal_3h,
)


class TestSlowSignals3H:
    def test_trend_is_slow(self):
        assert is_slow_signal_3h("trend")

    def test_fibonacci_is_slow(self):
        assert is_slow_signal_3h("fibonacci")

    def test_macro_regime_is_slow(self):
        assert is_slow_signal_3h("macro_regime")

    def test_rsi_is_not_slow(self):
        assert not is_slow_signal_3h("rsi")

    def test_news_event_is_not_slow(self):
        assert not is_slow_signal_3h("news_event")

    def test_qwen3_is_not_slow(self):
        assert not is_slow_signal_3h("qwen3")


class TestTimeOfDayScale3H:
    def test_peak_noise_hours_get_dampened(self):
        for hour in [10, 12, 14, 16, 17]:
            factor = time_of_day_scale_3h(hour)
            assert factor < 1.0, f"Hour {hour} should be dampened, got {factor}"

    def test_quiet_hours_get_boosted(self):
        for hour in [20, 21, 22, 23, 0]:
            factor = time_of_day_scale_3h(hour)
            assert factor > 1.0, f"Hour {hour} should be boosted, got {factor}"

    def test_neutral_hours_near_one(self):
        for hour in [7, 8, 9, 18, 19]:
            factor = time_of_day_scale_3h(hour)
            assert 0.95 <= factor <= 1.05, f"Hour {hour} should be neutral, got {factor}"

    def test_returns_float(self):
        assert isinstance(time_of_day_scale_3h(12), float)

    def test_all_hours_valid(self):
        for hour in range(24):
            factor = time_of_day_scale_3h(hour)
            assert 0.5 <= factor <= 1.5, f"Hour {hour} factor {factor} out of range"


class TestConfidenceCap3H:
    def test_cap_is_below_one(self):
        assert CONFIDENCE_CAP_3H < 1.0

    def test_cap_is_reasonable(self):
        assert 0.6 <= CONFIDENCE_CAP_3H <= 0.85
