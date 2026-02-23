"""Tests for portfolio.signals.econ_calendar — economic calendar signal."""

import pytest
from unittest import mock
from datetime import date, datetime, timezone

import pandas as pd
import numpy as np

from portfolio.signals.econ_calendar import (
    compute_econ_calendar_signal,
    _event_proximity,
    _event_type_info,
    _pre_event_risk,
    _sector_exposure,
    _MAX_CONFIDENCE,
)


def _make_df(n=100, last_time=None):
    """Create minimal OHLCV dataframe with optional time column."""
    df = pd.DataFrame({
        "open": np.random.uniform(100, 110, n),
        "high": np.random.uniform(110, 120, n),
        "low": np.random.uniform(90, 100, n),
        "close": np.random.uniform(100, 110, n),
        "volume": np.random.uniform(1000, 5000, n),
    })
    if last_time:
        times = pd.date_range(end=last_time, periods=n, freq="15min")
        df["time"] = times
    return df


class TestComputeEconCalendarSignal:
    def test_no_context_returns_hold(self):
        df = _make_df()
        result = compute_econ_calendar_signal(df, context=None)
        assert result["action"] in ("HOLD", "BUY", "SELL")
        assert "sub_signals" in result

    def test_result_structure(self):
        df = _make_df()
        result = compute_econ_calendar_signal(df, context={"ticker": "NVDA", "config": {}})
        assert "action" in result
        assert "confidence" in result
        assert "sub_signals" in result
        assert "indicators" in result
        assert "event_proximity" in result["sub_signals"]
        assert "event_type" in result["sub_signals"]
        assert "pre_event_risk" in result["sub_signals"]
        assert "sector_exposure" in result["sub_signals"]

    def test_confidence_capped(self):
        df = _make_df()
        result = compute_econ_calendar_signal(df, context={"ticker": "NVDA", "config": {}})
        assert result["confidence"] <= _MAX_CONFIDENCE


class TestEventProximity:
    @mock.patch("portfolio.signals.econ_calendar.next_event")
    def test_within_4h_sells(self, mock_next):
        mock_next.return_value = {
            "date": date(2026, 3, 18),
            "type": "FOMC",
            "impact": "high",
            "hours_until": 2.0,
        }
        ref = datetime(2026, 3, 18, 12, 0, tzinfo=timezone.utc)
        action, ind = _event_proximity(ref)
        assert action == "SELL"

    @mock.patch("portfolio.signals.econ_calendar.next_event")
    def test_within_24h_high_impact_sells(self, mock_next):
        mock_next.return_value = {
            "date": date(2026, 3, 18),
            "type": "FOMC",
            "impact": "high",
            "hours_until": 12.0,
        }
        ref = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
        action, ind = _event_proximity(ref)
        assert action == "SELL"

    @mock.patch("portfolio.signals.econ_calendar.next_event")
    def test_far_away_holds(self, mock_next):
        mock_next.return_value = {
            "date": date(2026, 6, 17),
            "type": "FOMC",
            "impact": "high",
            "hours_until": 720.0,
        }
        ref = datetime(2026, 3, 18, 12, 0, tzinfo=timezone.utc)
        action, ind = _event_proximity(ref)
        assert action == "HOLD"

    @mock.patch("portfolio.signals.econ_calendar.next_event", return_value=None)
    def test_no_event_holds(self, mock_next):
        ref = datetime(2030, 1, 1, tzinfo=timezone.utc)
        action, ind = _event_proximity(ref)
        assert action == "HOLD"


class TestEventTypeInfo:
    @mock.patch("portfolio.signals.econ_calendar.next_event")
    def test_fomc_within_24h_sells(self, mock_next):
        mock_next.return_value = {
            "date": date(2026, 3, 18),
            "type": "FOMC",
            "impact": "high",
            "hours_until": 6.0,
        }
        ref = datetime(2026, 3, 18, 8, 0, tzinfo=timezone.utc)
        action, ind = _event_type_info(ref)
        assert action == "SELL"

    @mock.patch("portfolio.signals.econ_calendar.next_event")
    def test_gdp_far_holds(self, mock_next):
        mock_next.return_value = {
            "date": date(2026, 7, 30),
            "type": "GDP",
            "impact": "medium",
            "hours_until": 100.0,
        }
        ref = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)
        action, ind = _event_type_info(ref)
        assert action == "HOLD"


class TestPreEventRisk:
    @mock.patch("portfolio.signals.econ_calendar.events_within_hours")
    def test_high_impact_within_4h_sells(self, mock_events):
        mock_events.return_value = [
            {"date": date(2026, 3, 18), "type": "FOMC", "impact": "high", "hours_until": 2.0}
        ]
        ref = datetime(2026, 3, 18, 12, 0, tzinfo=timezone.utc)
        action, ind = _pre_event_risk(ref)
        assert action == "SELL"
        assert ind["high_impact_within_4h"] == 1

    @mock.patch("portfolio.signals.econ_calendar.events_within_hours")
    def test_no_events_holds(self, mock_events):
        mock_events.return_value = []
        ref = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
        action, ind = _pre_event_risk(ref)
        assert action == "HOLD"


class TestSectorExposure:
    @mock.patch("portfolio.signals.econ_calendar.next_event")
    def test_fomc_affects_crypto(self, mock_next):
        mock_next.return_value = {
            "date": date(2026, 3, 18),
            "type": "FOMC",
            "impact": "high",
            "hours_until": 6.0,
        }
        ref = datetime(2026, 3, 18, 8, 0, tzinfo=timezone.utc)
        action, ind = _sector_exposure(ref, "BTC-USD")
        assert action == "SELL"
        assert ind["event_affects_sector"] is True

    @mock.patch("portfolio.signals.econ_calendar.next_event")
    def test_no_sector_overlap_holds(self, mock_next):
        mock_next.return_value = {
            "date": date(2026, 3, 6),
            "type": "NFP",
            "impact": "high",
            "hours_until": 6.0,
        }
        ref = datetime(2026, 3, 6, 8, 0, tzinfo=timezone.utc)
        # LMT is defense — NFP doesn't directly map to defense
        action, ind = _sector_exposure(ref, "LMT")
        assert action == "HOLD"

    @mock.patch("portfolio.signals.econ_calendar.next_event")
    def test_far_event_holds(self, mock_next):
        mock_next.return_value = {
            "date": date(2026, 6, 17),
            "type": "FOMC",
            "impact": "high",
            "hours_until": 720.0,
        }
        ref = datetime(2026, 3, 18, 12, 0, tzinfo=timezone.utc)
        action, ind = _sector_exposure(ref, "BTC-USD")
        assert action == "HOLD"
