"""Tests for advisory metals execution target scoring."""

import datetime as dt
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import metals_execution_engine as mex


class TestHoursToClose:
    def test_hours_to_metals_close_before_close(self):
        if mex.ZoneInfo is None:
            pytest.skip("zoneinfo unavailable")
        now = dt.datetime(2026, 3, 11, 20, 25, tzinfo=mex.ZoneInfo("Europe/Stockholm"))
        assert mex.hours_to_metals_close(now) == pytest.approx(1.5, abs=0.01)

    def test_hours_to_metals_close_after_close(self):
        if mex.ZoneInfo is None:
            pytest.skip("zoneinfo unavailable")
        now = dt.datetime(2026, 3, 11, 22, 5, tzinfo=mex.ZoneInfo("Europe/Stockholm"))
        assert mex.hours_to_metals_close(now) == 0.0


def _base_signal(action="BUY", buy_count=5, sell_count=2):
    return {
        "action": action,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "weighted_confidence": 0.72,
        "atr_pct": 4.4,
        "bb_mid": 85.4,
        "bb_upper": 86.3,
        "bb_lower": 84.5,
    }


class TestExecutionRecommendations:
    def test_sell_recommendation_for_active_long_position(self):
        positions = {
            "silver301": {
                "active": True,
                "name": "MINI L SILVER AVA 301",
                "units": 100,
                "_leverage": 4.3,
            }
        }
        prices = {
            "silver301": {
                "bid": 11.35,
                "ask": 11.41,
                "underlying": 85.39,
                "leverage": 4.3,
            }
        }
        signal_data = {"XAG-USD": _base_signal()}

        result = mex.build_execution_recommendations(
            positions,
            prices,
            signal_data=signal_data,
            hours_remaining=2.0,
        )

        assert "silver301" in result["sell"]
        sell = result["sell"]["silver301"]
        assert sell["recommended"]["target_price"] >= sell["current_price"]
        assert sell["recommended"]["fill_prob"] > 0.0

    def test_sell_recommendation_uses_chronos_drift_and_extra_levels(self, monkeypatch):
        positions = {
            "silver301": {
                "active": True,
                "name": "MINI L SILVER AVA 301",
                "units": 100,
                "_leverage": 4.3,
            }
        }
        prices = {
            "silver301": {
                "bid": 11.35,
                "ask": 11.41,
                "underlying": 85.39,
                "leverage": 4.3,
            }
        }
        base_signal = {
            "XAG-USD": {
                **_base_signal(),
                "regime": "trending-up",
                "extra": {
                    "fibonacci_indicators": {
                        "fib_levels": {
                            "0.236": 85.46,
                            "0.382": 85.61,
                        }
                    }
                },
            }
        }
        seen = {}

        def fake_compute_targets(**kwargs):
            seen.update(kwargs)
            return {
                "targets": [
                    {"price": 85.46, "label": "fib_236"},
                    {"price": 85.61, "label": "fib_382"},
                ],
                "squeeze_warning": False,
            }

        monkeypatch.setattr(mex, "compute_targets", fake_compute_targets)

        enriched_signal = {
            **base_signal,
            "forecast_signals": {
                "XAG-USD": {
                    "chronos_24h_pct": 18.0,
                    "chronos_24h_conf": 0.85,
                }
            },
        }

        baseline = mex.build_execution_recommendations(
            positions,
            prices,
            signal_data=base_signal,
            hours_remaining=8.0,
        )
        enriched = mex.build_execution_recommendations(
            positions,
            prices,
            signal_data=enriched_signal,
            hours_remaining=8.0,
        )

        base_sell = baseline["sell"]["silver301"]
        sell = enriched["sell"]["silver301"]
        assert sell["expected_close_underlying"] > base_sell["expected_close_underlying"]
        assert sell["candidates"][0]["label"].startswith("fib_")
        assert seen["extra"] == base_signal["XAG-USD"]["extra"]
        assert seen["regime"] == "trending-up"
        assert seen["chronos_drift"] is not None
        assert sell["plan_features"]["chronos_drift_annual"] is not None
        assert sell["plan_features"]["extra_level_count"] == 1

    def test_buy_recommendation_for_bullish_long_warrant(self):
        signal_data = {"XAG-USD": _base_signal(action="BUY", buy_count=6, sell_count=1)}
        warrant_catalog = {
            "MINI_L_SILVER_SG": {
                "name": "MINI L SILVER SG",
                "underlying": "XAG-USD",
                "direction": "LONG",
                "ask": 52.0,
                "bid": 51.8,
                "spread_pct": 0.39,
                "barrier_distance_pct": 55.0,
                "underlying_price": 85.4,
                "current_leverage": 1.56,
            }
        }

        result = mex.build_execution_recommendations(
            {},
            {},
            signal_data=signal_data,
            warrant_catalog=warrant_catalog,
            account={"buying_power": 10_000},
            hours_remaining=3.0,
        )

        assert "MINI_L_SILVER_SG" in result["buy"]
        buy = result["buy"]["MINI_L_SILVER_SG"]
        assert buy["planned_units"] > 0
        assert buy["recommended"]["target_price"] <= buy["current_price"]

    def test_buy_recommendation_propagates_squeeze_warning(self):
        signal_data = {
            "XAG-USD": {
                **_base_signal(action="BUY", buy_count=6, sell_count=1),
                "regime": "ranging",
                "extra": {
                    "volatility_sig_indicators": {
                        "bb_squeeze_on": True,
                    }
                },
            }
        }
        warrant_catalog = {
            "MINI_L_SILVER_SG": {
                "name": "MINI L SILVER SG",
                "underlying": "XAG-USD",
                "direction": "LONG",
                "ask": 52.0,
                "bid": 51.8,
                "spread_pct": 0.39,
                "barrier_distance_pct": 55.0,
                "underlying_price": 85.4,
                "current_leverage": 1.56,
            }
        }

        result = mex.build_execution_recommendations(
            {},
            {},
            signal_data=signal_data,
            warrant_catalog=warrant_catalog,
            account={"buying_power": 10_000},
            hours_remaining=3.0,
        )

        buy = result["buy"]["MINI_L_SILVER_SG"]
        assert buy["plan_features"]["regime"] == "ranging"
        assert buy["plan_features"]["squeeze_warning"] is True
        assert buy["plan_features"]["extra_level_count"] == 1

    def test_bearish_signal_prefers_short_instrument(self):
        signal_data = {"XAG-USD": _base_signal(action="SELL", buy_count=1, sell_count=6)}
        warrant_catalog = {
            "LONG": {
                "name": "MINI L SILVER SG",
                "underlying": "XAG-USD",
                "direction": "LONG",
                "ask": 52.0,
                "bid": 51.8,
                "spread_pct": 0.39,
                "barrier_distance_pct": 55.0,
                "underlying_price": 85.4,
                "current_leverage": 1.56,
            },
            "SHORT": {
                "name": "BEAR SILVER X5 AVA 12",
                "underlying": "XAG-USD",
                "direction": "SHORT",
                "ask": 15.0,
                "bid": 14.9,
                "spread_pct": 0.67,
                "barrier_distance_pct": 99.0,
                "underlying_price": 85.4,
                "current_leverage": 5.0,
            },
        }

        result = mex.build_execution_recommendations(
            {},
            {},
            signal_data=signal_data,
            warrant_catalog=warrant_catalog,
            account={"buying_power": 10_000},
            hours_remaining=3.0,
        )

        assert "SHORT" in result["buy"]
        assert "recommended" in result["buy"]["SHORT"]
        assert "LONG" not in result["buy"] or "recommended" not in result["buy"]["LONG"]

    def test_buy_filters_wide_spread(self):
        signal_data = {"XAG-USD": _base_signal(action="BUY", buy_count=6, sell_count=1)}
        warrant_catalog = {
            "WIDE": {
                "name": "MINI L SILVER SG",
                "underlying": "XAG-USD",
                "direction": "LONG",
                "ask": 52.0,
                "bid": 50.0,
                "spread_pct": 4.0,
                "barrier_distance_pct": 55.0,
                "underlying_price": 85.4,
                "current_leverage": 1.56,
            }
        }

        result = mex.build_execution_recommendations(
            {},
            {},
            signal_data=signal_data,
            warrant_catalog=warrant_catalog,
            account={"buying_power": 10_000},
            hours_remaining=3.0,
        )

        assert result["buy"]["WIDE"]["filtered_out"]
        assert "spread" in result["buy"]["WIDE"]["filtered_out"][0]

    def test_zero_hours_returns_empty_sections(self):
        result = mex.build_execution_recommendations({}, {}, hours_remaining=0.0)
        assert result["sell"] == {}
        assert result["buy"] == {}
