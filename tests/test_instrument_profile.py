"""Tests for portfolio.instrument_profile — per-metal signal trust and behavior."""

import pytest

from portfolio.instrument_profile import (
    PROFILES,
    format_profile_briefing,
    get_cross_asset_drivers,
    get_ignored_signals,
    get_profile,
    get_regime_behavior,
    get_trusted_signals,
)


class TestGetProfile:
    def test_silver_profile_exists(self):
        p = get_profile("XAG-USD")
        assert p is not None
        assert p["name"] == "Silver"

    def test_gold_profile_exists(self):
        p = get_profile("XAU-USD")
        assert p is not None
        assert p["name"] == "Gold"

    def test_unknown_ticker_returns_none(self):
        assert get_profile("UNKNOWN") is None
        assert get_profile("BTC-USD") is None

    def test_profile_has_required_keys(self):
        for ticker in ("XAG-USD", "XAU-USD"):
            p = get_profile(ticker)
            assert "trusted_signals" in p
            assert "ignored_signals" in p
            assert "cross_asset_drivers" in p
            assert "regime_behaviors" in p
            assert "binance_symbol" in p


class TestTrustedSignals:
    def test_silver_has_trusted_signals(self):
        trusted = get_trusted_signals("XAG-USD")
        assert len(trusted) >= 5
        assert "econ_calendar" in trusted
        assert "claude_fundamental" in trusted

    def test_silver_ignored_includes_noise(self):
        ignored = get_ignored_signals("XAG-USD")
        assert "sentiment" in ignored
        assert "ministral" in ignored

    def test_trusted_and_ignored_dont_overlap(self):
        for ticker in ("XAG-USD", "XAU-USD"):
            trusted = set(get_trusted_signals(ticker))
            ignored = set(get_ignored_signals(ticker))
            assert trusted.isdisjoint(ignored), f"Overlap in {ticker}: {trusted & ignored}"

    def test_unknown_ticker_returns_empty(self):
        assert get_trusted_signals("UNKNOWN") == []
        assert get_ignored_signals("UNKNOWN") == []


class TestCrossAssetDrivers:
    def test_silver_has_dxy_and_gold(self):
        drivers = get_cross_asset_drivers("XAG-USD")
        assert "DXY" in drivers
        assert "gold" in drivers
        assert "copper" in drivers

    def test_gold_has_dxy_and_vix(self):
        drivers = get_cross_asset_drivers("XAU-USD")
        assert "DXY" in drivers
        assert "VIX" in drivers

    def test_drivers_have_correlation(self):
        for ticker in ("XAG-USD", "XAU-USD"):
            drivers = get_cross_asset_drivers(ticker)
            for name, driver in drivers.items():
                if "correlation" in driver:
                    assert -1.0 <= driver["correlation"] <= 1.0, f"{ticker}/{name}"

    def test_unknown_ticker_returns_empty(self):
        assert get_cross_asset_drivers("UNKNOWN") == {}


class TestRegimeBehavior:
    def test_silver_trending_up_prefers_long(self):
        behavior = get_regime_behavior("XAG-USD", "trending-up")
        assert behavior["preferred_direction"] == "LONG"

    def test_silver_trending_down_prefers_short(self):
        behavior = get_regime_behavior("XAG-USD", "trending-down")
        assert behavior["preferred_direction"] == "SHORT"

    def test_gold_high_vol_prefers_long(self):
        # Gold is safe haven — high vol = buy gold
        behavior = get_regime_behavior("XAU-USD", "high-vol")
        assert behavior["preferred_direction"] == "LONG"

    def test_unknown_regime_returns_default(self):
        behavior = get_regime_behavior("XAG-USD", "unknown-regime")
        assert behavior["preferred_direction"] == "BOTH"
        assert behavior["tp_multiplier"] == 1.0

    def test_unknown_ticker_returns_default(self):
        behavior = get_regime_behavior("UNKNOWN", "trending-up")
        assert behavior["preferred_direction"] == "BOTH"


class TestFormatBriefing:
    def test_silver_briefing_has_content(self):
        text = format_profile_briefing("XAG-USD")
        assert "Silver" in text
        assert "daily range" in text.lower()
        assert "trusted" in text.lower()

    def test_unknown_ticker_shows_message(self):
        text = format_profile_briefing("UNKNOWN")
        assert "No profile" in text

    def test_with_signal_data(self):
        signal_data = {
            "signal_reliability": {
                "XAG-USD": {
                    "econ_calendar": {"accuracy": 0.95, "total": 400},
                    "sentiment": {"accuracy": 0.05, "total": 90},
                }
            }
        }
        text = format_profile_briefing("XAG-USD", signal_data)
        assert "econ_calendar" in text
