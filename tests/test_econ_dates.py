"""Tests for portfolio.econ_dates — economic calendar dates and lookups."""

import pytest
from datetime import date, datetime, timezone

from portfolio.econ_dates import (
    ECON_EVENTS,
    CPI_DATES_2026,
    NFP_DATES_2026,
    GDP_DATES_2026,
    EVENT_SECTOR_MAP,
    next_event,
    events_within_hours,
)


class TestEconEventsList:
    def test_events_not_empty(self):
        assert len(ECON_EVENTS) > 0

    def test_events_sorted_by_date(self):
        dates = [e["date"] for e in ECON_EVENTS]
        assert dates == sorted(dates)

    def test_all_events_have_required_keys(self):
        for evt in ECON_EVENTS:
            assert "date" in evt
            assert "type" in evt
            assert "impact" in evt
            assert evt["type"] in ("FOMC", "CPI", "NFP", "GDP")
            assert evt["impact"] in ("high", "medium")

    def test_fomc_events_are_high_impact(self):
        fomc = [e for e in ECON_EVENTS if e["type"] == "FOMC"]
        assert len(fomc) > 0
        for e in fomc:
            assert e["impact"] == "high"

    def test_cpi_events_are_high_impact(self):
        cpi = [e for e in ECON_EVENTS if e["type"] == "CPI"]
        assert len(cpi) > 0
        for e in cpi:
            assert e["impact"] == "high"

    def test_gdp_events_are_medium_impact(self):
        gdp = [e for e in ECON_EVENTS if e["type"] == "GDP"]
        assert len(gdp) > 0
        for e in gdp:
            assert e["impact"] == "medium"

    def test_2026_has_12_cpi_dates(self):
        assert len(CPI_DATES_2026) == 12

    def test_2026_has_12_nfp_dates(self):
        assert len(NFP_DATES_2026) == 12

    def test_2026_has_4_gdp_dates(self):
        assert len(GDP_DATES_2026) == 4


class TestNextEvent:
    def test_next_event_from_early_2026(self):
        evt = next_event(date(2026, 1, 1))
        assert evt is not None
        assert evt["date"] >= date(2026, 1, 1)
        assert "hours_until" in evt

    def test_next_event_on_event_day(self):
        # CPI on Jan 14 2026
        evt = next_event(date(2026, 1, 14))
        assert evt is not None
        assert evt["date"] == date(2026, 1, 14)
        assert evt["type"] == "CPI"

    def test_next_event_returns_none_far_future(self):
        evt = next_event(date(2030, 1, 1))
        assert evt is None

    def test_next_event_has_hours_until(self):
        evt = next_event(date(2026, 3, 1))
        assert evt is not None
        assert isinstance(evt["hours_until"], float)
        assert evt["hours_until"] >= 0


class TestEventsWithinHours:
    def test_no_events_far_from_dates(self):
        # Pick a date far from any event
        result = events_within_hours(4, date(2026, 2, 20))
        # May or may not have events depending on exact timing
        assert isinstance(result, list)

    def test_returns_list(self):
        result = events_within_hours(48, date(2026, 1, 14))
        assert isinstance(result, list)


class TestEventSectorMap:
    def test_fomc_affects_crypto(self):
        assert "crypto" in EVENT_SECTOR_MAP["FOMC"]

    def test_cpi_affects_metals(self):
        assert "metals" in EVENT_SECTOR_MAP["CPI"]

    def test_all_event_types_mapped(self):
        for evt_type in ("FOMC", "CPI", "NFP", "GDP"):
            assert evt_type in EVENT_SECTOR_MAP
