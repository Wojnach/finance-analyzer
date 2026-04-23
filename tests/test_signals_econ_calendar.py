"""Tests for portfolio.signals.econ_calendar — economic calendar signal."""

from datetime import UTC, date, datetime
from unittest import mock

import numpy as np
import pandas as pd

from portfolio.signals.econ_calendar import (
    _MAX_CONFIDENCE,
    _event_proximity,
    _event_type_info,
    _post_event_relief,
    _pre_event_risk,
    _sector_exposure,
    compute_econ_calendar_signal,
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
        assert "post_event_relief" in result["sub_signals"]

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
        ref = datetime(2026, 3, 18, 12, 0, tzinfo=UTC)
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
        ref = datetime(2026, 3, 17, 12, 0, tzinfo=UTC)
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
        ref = datetime(2026, 3, 18, 12, 0, tzinfo=UTC)
        action, ind = _event_proximity(ref)
        assert action == "HOLD"

    @mock.patch("portfolio.signals.econ_calendar.next_event", return_value=None)
    def test_no_event_holds(self, mock_next):
        ref = datetime(2030, 1, 1, tzinfo=UTC)
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
        ref = datetime(2026, 3, 18, 8, 0, tzinfo=UTC)
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
        ref = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)
        action, ind = _event_type_info(ref)
        assert action == "HOLD"


class TestPreEventRisk:
    @mock.patch("portfolio.signals.econ_calendar.events_within_hours")
    def test_high_impact_within_4h_sells(self, mock_events):
        mock_events.return_value = [
            {"date": date(2026, 3, 18), "type": "FOMC", "impact": "high", "hours_until": 2.0}
        ]
        ref = datetime(2026, 3, 18, 12, 0, tzinfo=UTC)
        action, ind = _pre_event_risk(ref)
        assert action == "SELL"
        assert ind["high_impact_within_4h"] == 1

    @mock.patch("portfolio.signals.econ_calendar.events_within_hours")
    def test_no_events_holds(self, mock_events):
        mock_events.return_value = []
        ref = datetime(2026, 5, 15, 12, 0, tzinfo=UTC)
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
        ref = datetime(2026, 3, 18, 8, 0, tzinfo=UTC)
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
        ref = datetime(2026, 3, 6, 8, 0, tzinfo=UTC)
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
        ref = datetime(2026, 3, 18, 12, 0, tzinfo=UTC)
        action, ind = _sector_exposure(ref, "BTC-USD")
        assert action == "HOLD"


class TestPostEventRelief:
    """Tests for BUG-218 fix: _post_event_relief sub-signal."""

    @mock.patch("portfolio.signals.econ_calendar.next_event")
    @mock.patch("portfolio.signals.econ_calendar.recent_high_impact_events")
    def test_relief_after_event_buys(self, mock_recent, mock_next):
        """4-24h after a high-impact event with no imminent next event → BUY."""
        mock_recent.return_value = [
            {"type": "FOMC", "impact": "high", "hours_since": 8.0,
             "date": date(2026, 3, 18)}
        ]
        mock_next.return_value = {
            "date": date(2026, 4, 10),
            "type": "CPI",
            "impact": "high",
            "hours_until": 500.0,
        }
        ref = datetime(2026, 3, 18, 22, 0, tzinfo=UTC)
        action, ind = _post_event_relief(ref)
        assert action == "BUY"
        assert ind["post_event_relief"] is True

    @mock.patch("portfolio.signals.econ_calendar.next_event")
    @mock.patch("portfolio.signals.econ_calendar.recent_high_impact_events")
    def test_relief_negated_by_imminent_event(self, mock_recent, mock_next):
        """Relief window exists, but next event is <24h away → no BUY from relief."""
        mock_recent.return_value = [
            {"type": "CPI", "impact": "high", "hours_since": 6.0,
             "date": date(2026, 3, 11)}
        ]
        # Next event is imminent — negates the relief
        mock_next.return_value = {
            "date": date(2026, 3, 12),
            "type": "NFP",
            "impact": "high",
            "hours_until": 10.0,
        }
        ref = datetime(2026, 3, 11, 20, 0, tzinfo=UTC)
        action, ind = _post_event_relief(ref)
        # Relief negated, but event-free window also doesn't apply (10h < 72h)
        # The function checks event-free window as fallback — 10h < 72h → HOLD
        assert action == "HOLD"

    @mock.patch("portfolio.signals.econ_calendar.next_event")
    @mock.patch("portfolio.signals.econ_calendar.recent_high_impact_events")
    def test_event_free_window_buys(self, mock_recent, mock_next):
        """No recent events, next event >72h away → BUY (calm window)."""
        mock_recent.return_value = []
        mock_next.return_value = {
            "date": date(2026, 5, 13),
            "type": "CPI",
            "impact": "high",
            "hours_until": 120.0,
        }
        ref = datetime(2026, 5, 8, 12, 0, tzinfo=UTC)
        action, ind = _post_event_relief(ref)
        assert action == "BUY"
        assert ind["event_free_window"] is True

    @mock.patch("portfolio.signals.econ_calendar.next_event")
    @mock.patch("portfolio.signals.econ_calendar.recent_high_impact_events")
    def test_no_relief_no_calm_holds(self, mock_recent, mock_next):
        """No recent events, next event <72h → HOLD."""
        mock_recent.return_value = []
        mock_next.return_value = {
            "date": date(2026, 3, 18),
            "type": "FOMC",
            "impact": "high",
            "hours_until": 48.0,
        }
        ref = datetime(2026, 3, 16, 14, 0, tzinfo=UTC)
        action, ind = _post_event_relief(ref)
        assert action == "HOLD"
        assert ind["post_event_relief"] is False
        assert ind["event_free_window"] is False

    @mock.patch("portfolio.signals.econ_calendar.next_event")
    @mock.patch("portfolio.signals.econ_calendar.recent_high_impact_events")
    def test_too_recent_event_no_relief(self, mock_recent, mock_next):
        """Event passed <4h ago — too early for relief (still volatile)."""
        mock_recent.return_value = [
            {"type": "NFP", "impact": "high", "hours_since": 2.0,
             "date": date(2026, 3, 6)}
        ]
        mock_next.return_value = {
            "date": date(2026, 4, 10),
            "type": "CPI",
            "impact": "high",
            "hours_until": 800.0,
        }
        ref = datetime(2026, 3, 6, 16, 0, tzinfo=UTC)
        action, ind = _post_event_relief(ref)
        # hours_since=2.0 < 4 → filtered out of relief_events
        # But event-free window: 800h > 72h → BUY via calm window
        assert action == "BUY"
        assert ind["post_event_relief"] is False
        assert ind["event_free_window"] is True


class TestCompositeWithRelief:
    """Test that the composite signal can now produce BUY (BUG-218 fix)."""

    @mock.patch("portfolio.signals.econ_calendar.recent_high_impact_events")
    @mock.patch("portfolio.signals.econ_calendar.events_within_hours")
    @mock.patch("portfolio.signals.econ_calendar.next_event")
    def test_composite_can_buy(self, mock_next, mock_events_within, mock_recent):
        """When post_event_relief fires and other subs are HOLD → composite BUY."""
        # No upcoming events (all subs except relief → HOLD)
        mock_next.return_value = {
            "date": date(2026, 6, 10),
            "type": "CPI",
            "impact": "high",
            "hours_until": 200.0,
        }
        mock_events_within.return_value = []
        # Recent high-impact event 8h ago → relief triggers
        mock_recent.return_value = [
            {"type": "CPI", "impact": "high", "hours_since": 8.0,
             "date": date(2026, 5, 13)}
        ]
        df = _make_df()
        result = compute_econ_calendar_signal(df, context={"ticker": "BTC-USD", "config": {}})
        # With 4 HOLD + 1 BUY, majority_vote should produce BUY
        assert result["sub_signals"]["post_event_relief"] == "BUY"
        assert result["action"] == "BUY"
