"""Comprehensive tests for portfolio.market_timing module.

Tests DST detection, NYSE close hour calculation, agent window logic,
and market state determination with mocked datetimes.
"""

from datetime import UTC, date, datetime, timedelta
from unittest.mock import patch

import pytest

from portfolio.market_timing import (
    INTERVAL_MARKET_CLOSED,
    INTERVAL_MARKET_OPEN,
    INTERVAL_WEEKEND,
    MARKET_OPEN_HOUR,
    _easter_sunday,
    _is_agent_window,
    _is_us_dst,
    _market_close_hour_utc,
    get_market_state,
    is_swedish_market_holiday,
    is_us_market_holiday,
    swedish_market_holidays,
    us_market_holidays,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc(year, month, day, hour=12, minute=0):
    """Shorthand to build a UTC-aware datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


# ===========================================================================
# 1. _is_us_dst() — dates clearly in EDT (March-November)
# ===========================================================================

class TestIsDstEdtPeriod:
    """Dates well inside the EDT window should return True."""

    def test_mid_summer(self):
        assert _is_us_dst(utc(2026, 7, 15)) is True

    def test_april(self):
        assert _is_us_dst(utc(2026, 4, 1)) is True

    def test_may(self):
        assert _is_us_dst(utc(2026, 5, 20)) is True

    def test_june(self):
        assert _is_us_dst(utc(2026, 6, 10)) is True

    def test_august(self):
        assert _is_us_dst(utc(2026, 8, 25)) is True

    def test_september(self):
        assert _is_us_dst(utc(2026, 9, 15)) is True

    def test_october(self):
        assert _is_us_dst(utc(2026, 10, 15)) is True


# ===========================================================================
# 2. _is_us_dst() — dates clearly in EST (November-March)
# ===========================================================================

class TestIsDstEstPeriod:
    """Dates well inside the EST window should return False."""

    def test_january(self):
        assert _is_us_dst(utc(2026, 1, 15)) is False

    def test_february(self):
        assert _is_us_dst(utc(2026, 2, 15)) is False

    def test_early_march(self):
        # March 1 is always before the second Sunday
        assert _is_us_dst(utc(2026, 3, 1)) is False

    def test_december(self):
        assert _is_us_dst(utc(2026, 12, 25)) is False

    def test_late_november(self):
        assert _is_us_dst(utc(2026, 11, 20)) is False

    def test_mid_november(self):
        assert _is_us_dst(utc(2026, 11, 15)) is False


# ===========================================================================
# 3. _is_us_dst() — exact DST transition boundaries
# ===========================================================================

class TestIsDstTransitionBoundaries:
    """Test the exact moments DST starts and ends.

    2026:
      - DST starts: 2nd Sunday of March = March 8 at 07:00 UTC
        (March 1 is Sunday => first Sunday=1, second Sunday=8)
      - DST ends:   1st Sunday of November = November 1 at 06:00 UTC
        (November 1 is Sunday => first Sunday=1)
    """

    # --- Spring forward (March) ---

    def test_2026_just_before_dst_start(self):
        # 2026: March 1 is Sunday, so 2nd Sunday = March 8
        # DST starts at 07:00 UTC on March 8
        just_before = datetime(2026, 3, 8, 6, 59, tzinfo=UTC)
        assert _is_us_dst(just_before) is False

    def test_2026_exact_dst_start(self):
        exact_start = datetime(2026, 3, 8, 7, 0, tzinfo=UTC)
        assert _is_us_dst(exact_start) is True

    def test_2026_just_after_dst_start(self):
        just_after = datetime(2026, 3, 8, 7, 1, tzinfo=UTC)
        assert _is_us_dst(just_after) is True

    # --- Fall back (November) ---

    def test_2026_just_before_dst_end(self):
        # 2026: November 1 is Sunday, so 1st Sunday = November 1
        # DST ends at 06:00 UTC on November 1
        just_before = datetime(2026, 11, 1, 5, 59, tzinfo=UTC)
        assert _is_us_dst(just_before) is True

    def test_2026_exact_dst_end(self):
        exact_end = datetime(2026, 11, 1, 6, 0, tzinfo=UTC)
        assert _is_us_dst(exact_end) is False

    def test_2026_just_after_dst_end(self):
        just_after = datetime(2026, 11, 1, 6, 1, tzinfo=UTC)
        assert _is_us_dst(just_after) is False

    # --- 2025 boundaries (March 1 is Saturday) ---

    def test_2025_dst_start(self):
        # 2025: March 1 is Saturday (weekday=5)
        # First Sunday = March 2, Second Sunday = March 9
        # DST starts March 9 at 07:00 UTC
        assert _is_us_dst(datetime(2025, 3, 9, 6, 59, tzinfo=UTC)) is False
        assert _is_us_dst(datetime(2025, 3, 9, 7, 0, tzinfo=UTC)) is True

    def test_2025_dst_end(self):
        # 2025: November 1 is Saturday (weekday=5)
        # First Sunday = November 2
        # DST ends November 2 at 06:00 UTC
        assert _is_us_dst(datetime(2025, 11, 2, 5, 59, tzinfo=UTC)) is True
        assert _is_us_dst(datetime(2025, 11, 2, 6, 0, tzinfo=UTC)) is False

    # --- 2027 boundaries (March 1 is Monday) ---

    def test_2027_dst_start(self):
        # 2027: March 1 is Monday (weekday=0)
        # First Sunday = March 7, Second Sunday = March 14
        # DST starts March 14 at 07:00 UTC
        assert _is_us_dst(datetime(2027, 3, 14, 6, 59, tzinfo=UTC)) is False
        assert _is_us_dst(datetime(2027, 3, 14, 7, 0, tzinfo=UTC)) is True

    def test_2027_dst_end(self):
        # 2027: November 1 is Monday (weekday=0)
        # First Sunday = November 7
        # DST ends November 7 at 06:00 UTC
        assert _is_us_dst(datetime(2027, 11, 7, 5, 59, tzinfo=UTC)) is True
        assert _is_us_dst(datetime(2027, 11, 7, 6, 0, tzinfo=UTC)) is False

    # --- 2024 boundaries (for variety, March 1 is Friday) ---

    def test_2024_dst_start(self):
        # 2024: March 1 is Friday (weekday=4)
        # First Sunday = March 3, Second Sunday = March 10
        # DST starts March 10 at 07:00 UTC
        assert _is_us_dst(datetime(2024, 3, 10, 6, 59, tzinfo=UTC)) is False
        assert _is_us_dst(datetime(2024, 3, 10, 7, 0, tzinfo=UTC)) is True

    def test_2024_dst_end(self):
        # 2024: November 1 is Friday (weekday=4)
        # First Sunday = November 3
        # DST ends November 3 at 06:00 UTC
        assert _is_us_dst(datetime(2024, 11, 3, 5, 59, tzinfo=UTC)) is True
        assert _is_us_dst(datetime(2024, 11, 3, 6, 0, tzinfo=UTC)) is False


# ===========================================================================
# 4. _is_us_dst() — leap year handling
# ===========================================================================

class TestIsDstLeapYear:
    """Leap years should not affect DST calculation (DST is March/November)."""

    def test_leap_year_2024_summer(self):
        # 2024 is a leap year (Feb 29 exists)
        assert _is_us_dst(utc(2024, 6, 15)) is True

    def test_leap_year_2024_winter(self):
        assert _is_us_dst(utc(2024, 1, 15)) is False

    def test_leap_year_2024_feb_29(self):
        # Feb 29 in a leap year is still winter (EST)
        assert _is_us_dst(utc(2024, 2, 29)) is False

    def test_non_leap_year_2025_summer(self):
        assert _is_us_dst(utc(2025, 6, 15)) is True

    def test_non_leap_year_2025_winter(self):
        assert _is_us_dst(utc(2025, 1, 15)) is False

    def test_leap_year_2028_summer(self):
        # 2028 is a leap year
        assert _is_us_dst(utc(2028, 7, 4)) is True

    def test_leap_year_2028_winter(self):
        assert _is_us_dst(utc(2028, 12, 1)) is False

    def test_century_non_leap_2100(self):
        # 2100 is NOT a leap year (divisible by 100 but not 400)
        # DST logic should still work
        assert _is_us_dst(utc(2100, 6, 15)) is True
        assert _is_us_dst(utc(2100, 1, 15)) is False


# ===========================================================================
# 5. _market_close_hour_utc() — returns 20 during EDT, 21 during EST
# ===========================================================================

class TestMarketCloseHourUtc:
    """NYSE close = 16:00 ET => 20:00 UTC (EDT) or 21:00 UTC (EST)."""

    def test_edt_returns_20(self):
        # Mid-summer, clearly EDT
        assert _market_close_hour_utc(utc(2026, 7, 15)) == 20

    def test_est_returns_21(self):
        # Mid-winter, clearly EST
        assert _market_close_hour_utc(utc(2026, 1, 15)) == 21

    def test_dst_start_day_before_transition(self):
        # Just before DST starts => EST => 21
        before = datetime(2026, 3, 8, 6, 59, tzinfo=UTC)
        assert _market_close_hour_utc(before) == 21

    def test_dst_start_day_after_transition(self):
        # Just after DST starts => EDT => 20
        after = datetime(2026, 3, 8, 7, 0, tzinfo=UTC)
        assert _market_close_hour_utc(after) == 20

    def test_dst_end_day_before_transition(self):
        # Just before DST ends => EDT => 20
        before = datetime(2026, 11, 1, 5, 59, tzinfo=UTC)
        assert _market_close_hour_utc(before) == 20

    def test_dst_end_day_after_transition(self):
        # Just after DST ends => EST => 21
        after = datetime(2026, 11, 1, 6, 0, tzinfo=UTC)
        assert _market_close_hour_utc(after) == 21

    def test_various_months(self):
        """Quick sweep across several months."""
        expected = {
            1: 21, 2: 21, 4: 20, 5: 20, 6: 20, 7: 20,
            8: 20, 9: 20, 10: 20, 12: 21,
        }
        for month, hour in expected.items():
            dt = utc(2026, month, 15)
            assert _market_close_hour_utc(dt) == hour, (
                f"Month {month}: expected {hour}, got {_market_close_hour_utc(dt)}"
            )


# ===========================================================================
# 6. _is_agent_window() — within window (weekday, 07:00-20/21:00 UTC)
# ===========================================================================

class TestIsAgentWindowInside:
    """Times that should be within the agent window (EU open through US close)."""

    def test_monday_midday(self):
        # Monday 12:00 UTC
        now = datetime(2026, 2, 23, 12, 0, tzinfo=UTC)  # Monday
        assert _is_agent_window(now) is True

    def test_wednesday_morning_edge_winter(self):
        # H47: Wednesday 08:00 UTC in winter = EU market open (CET winter)
        now = datetime(2026, 2, 25, 8, 0, tzinfo=UTC)  # Wednesday
        assert _is_agent_window(now) is True

    def test_wednesday_07_winter_before_eu_open(self):
        # H47: 07:00 UTC in winter is BEFORE EU opens (opens at 08:00 UTC)
        now = datetime(2026, 2, 25, 7, 0, tzinfo=UTC)  # Wednesday
        assert _is_agent_window(now) is False

    def test_wednesday_07_summer_at_eu_open(self):
        # H47: 07:00 UTC in summer is AT EU open (opens at 07:00 UTC in CEST)
        now = datetime(2026, 7, 15, 7, 0, tzinfo=UTC)  # Wednesday in summer
        assert _is_agent_window(now) is True

    def test_friday_afternoon(self):
        # Friday 15:00 UTC
        now = datetime(2026, 2, 27, 15, 0, tzinfo=UTC)  # Friday
        assert _is_agent_window(now) is True

    def test_tuesday_late_evening_est(self):
        # In EST (winter), window extends to 21:00 UTC (NYSE close)
        # Tuesday 20:30 UTC in February (EST)
        now = datetime(2026, 2, 24, 20, 30, tzinfo=UTC)  # Tuesday
        assert _is_agent_window(now) is True

    def test_tuesday_late_evening_edt(self):
        # In EDT (summer), window extends to 20:00 UTC (NYSE close)
        # Tuesday 19:30 UTC in July (EDT)
        now = datetime(2026, 7, 7, 19, 30, tzinfo=UTC)  # Tuesday
        assert _is_agent_window(now) is True

    def test_thursday_at_0700_summer(self):
        # Summer: EU open = 07:00 UTC (inclusive)
        now = datetime(2026, 7, 9, 7, 0, tzinfo=UTC)  # Thursday summer
        assert _is_agent_window(now) is True

    def test_thursday_at_0800_winter(self):
        # Winter: EU open = 08:00 UTC (inclusive)
        now = datetime(2026, 2, 26, 8, 0, tzinfo=UTC)  # Thursday winter
        assert _is_agent_window(now) is True

    def test_thursday_at_0700_winter_before_open(self):
        # Winter: 07:00 UTC is BEFORE EU open (08:00)
        now = datetime(2026, 2, 26, 7, 0, tzinfo=UTC)  # Thursday winter
        assert _is_agent_window(now) is False

    def test_monday_one_minute_before_end_est(self):
        # In EST, end = 21:00 UTC, so 20:59 is still inside
        now = datetime(2026, 1, 5, 20, 59, tzinfo=UTC)  # Monday
        assert _is_agent_window(now) is True

    def test_monday_one_minute_before_end_edt(self):
        # In EDT, end = 20:00 UTC, so 19:59 is still inside
        now = datetime(2026, 6, 1, 19, 59, tzinfo=UTC)  # Monday
        assert _is_agent_window(now) is True


# ===========================================================================
# 7. _is_agent_window() — outside window (weekends, too early, too late)
# ===========================================================================

class TestIsAgentWindowOutside:
    """Times that should be outside the agent window."""

    def test_saturday(self):
        now = datetime(2026, 2, 28, 12, 0, tzinfo=UTC)  # Saturday
        assert _is_agent_window(now) is False

    def test_sunday(self):
        now = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)  # Sunday
        assert _is_agent_window(now) is False

    def test_weekday_too_early(self):
        # Monday 06:59 UTC (before 07:00)
        now = datetime(2026, 2, 23, 6, 59, tzinfo=UTC)  # Monday
        assert _is_agent_window(now) is False

    def test_weekday_midnight(self):
        now = datetime(2026, 2, 24, 0, 0, tzinfo=UTC)  # Tuesday
        assert _is_agent_window(now) is False

    def test_weekday_too_late_est(self):
        # In EST (winter), window ends at 21:00 UTC (NYSE close)
        now = datetime(2026, 1, 5, 21, 0, tzinfo=UTC)  # Monday
        assert _is_agent_window(now) is False

    def test_weekday_too_late_edt(self):
        # In EDT (summer), window ends at 20:00 UTC (NYSE close)
        now = datetime(2026, 6, 1, 20, 0, tzinfo=UTC)  # Monday
        assert _is_agent_window(now) is False

    def test_weekday_23_utc(self):
        now = datetime(2026, 2, 25, 23, 0, tzinfo=UTC)  # Wednesday
        assert _is_agent_window(now) is False

    def test_sunday_evening(self):
        now = datetime(2026, 3, 1, 20, 0, tzinfo=UTC)  # Sunday
        assert _is_agent_window(now) is False

    def test_saturday_morning(self):
        now = datetime(2026, 2, 28, 7, 0, tzinfo=UTC)  # Saturday
        assert _is_agent_window(now) is False


# ===========================================================================
# 7b. _is_agent_window() — default argument (now=None) uses real time
# ===========================================================================

class TestIsAgentWindowDefault:
    """When called without arguments, _is_agent_window uses datetime.now(UTC)."""

    @patch("portfolio.market_timing.datetime")
    def test_default_now_weekday_midday(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        # _is_agent_window with no arg should use the patched now
        result = _is_agent_window()
        mock_dt.now.assert_called_once_with(UTC)
        assert result is True


# ===========================================================================
# 8. get_market_state() — market open
# ===========================================================================

class TestGetMarketStateOpen:
    """Market open: weekday, within MARKET_OPEN_HOUR .. close_hour."""

    @patch("portfolio.market_timing.datetime")
    def test_weekday_midday_edt(self, mock_dt):
        """Wednesday 14:00 UTC in summer (EDT, close=20) => open."""
        fake_now = datetime(2026, 7, 8, 14, 0, tzinfo=UTC)  # Wednesday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, symbols, interval = get_market_state()
        assert state == "open"
        assert interval == INTERVAL_MARKET_OPEN  # 60
        assert isinstance(symbols, set)
        assert len(symbols) > 0

    @patch("portfolio.market_timing.datetime")
    def test_weekday_midday_est(self, mock_dt):
        """Tuesday 14:00 UTC in winter (EST, close=21) => open."""
        fake_now = datetime(2026, 1, 6, 14, 0, tzinfo=UTC)  # Tuesday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, symbols, interval = get_market_state()
        assert state == "open"
        assert interval == 60

    @patch("portfolio.market_timing.datetime")
    def test_at_eu_open_winter(self, mock_dt):
        """H47: At EU open (08:00 UTC in winter) => open."""
        fake_now = datetime(2026, 2, 24, 8, 0, tzinfo=UTC)  # Tuesday, winter
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, symbols, interval = get_market_state()
        assert state == "open"

    @patch("portfolio.market_timing.datetime")
    def test_before_eu_open_winter(self, mock_dt):
        """H47: At 07:00 UTC in winter => closed (EU opens at 08:00)."""
        fake_now = datetime(2026, 2, 24, 7, 0, tzinfo=UTC)  # Tuesday, winter
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, _, interval = get_market_state()
        assert state == "closed"

    @patch("portfolio.market_timing.datetime")
    def test_one_hour_before_close_edt(self, mock_dt):
        """Hour 19 UTC when close is 20 (EDT) => still open."""
        fake_now = datetime(2026, 7, 8, 19, 0, tzinfo=UTC)  # Wednesday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, _, _ = get_market_state()
        assert state == "open"

    @patch("portfolio.market_timing.datetime")
    def test_one_hour_before_close_est(self, mock_dt):
        """Hour 20 UTC when close is 21 (EST) => still open."""
        fake_now = datetime(2026, 1, 6, 20, 0, tzinfo=UTC)  # Tuesday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, _, _ = get_market_state()
        assert state == "open"


# ===========================================================================
# 9. get_market_state() — market closed
# ===========================================================================

class TestGetMarketStateClosed:
    """Market closed: weekday but outside open hours."""

    @patch("portfolio.market_timing.datetime")
    def test_weekday_before_open(self, mock_dt):
        """Tuesday 05:00 UTC => closed."""
        fake_now = datetime(2026, 2, 24, 5, 0, tzinfo=UTC)  # Tuesday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, symbols, interval = get_market_state()
        assert state == "closed"
        assert interval == INTERVAL_MARKET_CLOSED  # 300
        assert isinstance(symbols, set)

    @patch("portfolio.market_timing.datetime")
    def test_weekday_after_close_edt(self, mock_dt):
        """Wednesday 20:00 UTC in summer (EDT close=20) => closed."""
        fake_now = datetime(2026, 7, 8, 20, 0, tzinfo=UTC)  # Wednesday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, _, interval = get_market_state()
        assert state == "closed"
        assert interval == 120

    @patch("portfolio.market_timing.datetime")
    def test_weekday_after_close_est(self, mock_dt):
        """Monday 21:00 UTC in winter (EST close=21) => closed."""
        fake_now = datetime(2026, 1, 5, 21, 0, tzinfo=UTC)  # Monday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, _, interval = get_market_state()
        assert state == "closed"
        assert interval == 120

    @patch("portfolio.market_timing.datetime")
    def test_weekday_at_midnight(self, mock_dt):
        """Thursday 00:00 UTC => closed."""
        fake_now = datetime(2026, 2, 26, 0, 0, tzinfo=UTC)  # Thursday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, _, interval = get_market_state()
        assert state == "closed"

    @patch("portfolio.market_timing.datetime")
    def test_weekday_hour_6_is_before_open(self, mock_dt):
        """Hour 6 UTC is before MARKET_OPEN_HOUR (7) => closed."""
        fake_now = datetime(2026, 2, 24, 6, 0, tzinfo=UTC)  # Tuesday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, _, _ = get_market_state()
        assert state == "closed"

    @patch("portfolio.market_timing.datetime")
    def test_weekday_late_night(self, mock_dt):
        """Friday 23:00 UTC => closed."""
        fake_now = datetime(2026, 2, 27, 23, 0, tzinfo=UTC)  # Friday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, _, interval = get_market_state()
        assert state == "closed"
        assert interval == 120


# ===========================================================================
# 10. get_market_state() — weekend
# ===========================================================================

class TestGetMarketStateWeekend:
    """Weekend: Saturday or Sunday regardless of hour."""

    @patch("portfolio.market_timing.datetime")
    def test_saturday_midday(self, mock_dt):
        fake_now = datetime(2026, 2, 28, 12, 0, tzinfo=UTC)  # Saturday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, symbols, interval = get_market_state()
        assert state == "weekend"
        assert interval == INTERVAL_WEEKEND  # 600
        assert isinstance(symbols, set)

    @patch("portfolio.market_timing.datetime")
    def test_sunday_midday(self, mock_dt):
        fake_now = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)  # Sunday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, _, interval = get_market_state()
        assert state == "weekend"
        assert interval == 600

    @patch("portfolio.market_timing.datetime")
    def test_saturday_during_market_hours(self, mock_dt):
        """Even at 10:00 UTC Saturday (would be open on a weekday) => weekend."""
        fake_now = datetime(2026, 2, 28, 10, 0, tzinfo=UTC)  # Saturday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, _, _ = get_market_state()
        assert state == "weekend"

    @patch("portfolio.market_timing.datetime")
    def test_sunday_morning(self, mock_dt):
        fake_now = datetime(2026, 3, 1, 7, 0, tzinfo=UTC)  # Sunday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, _, _ = get_market_state()
        assert state == "weekend"

    @patch("portfolio.market_timing.datetime")
    def test_saturday_midnight(self, mock_dt):
        fake_now = datetime(2026, 2, 28, 0, 0, tzinfo=UTC)  # Saturday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, _, _ = get_market_state()
        assert state == "weekend"

    @patch("portfolio.market_timing.datetime")
    def test_sunday_late_night(self, mock_dt):
        fake_now = datetime(2026, 3, 1, 23, 59, tzinfo=UTC)  # Sunday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, _, _ = get_market_state()
        assert state == "weekend"


# ===========================================================================
# Additional: get_market_state() returns all SYMBOLS
# ===========================================================================

class TestGetMarketStateSymbols:
    """Off-hours should only return 24/7 tickers (crypto + metals)."""

    @patch("portfolio.market_timing.datetime")
    def test_symbols_all_during_market_hours(self, mock_dt):
        from portfolio.tickers import SYMBOLS
        fake_now = datetime(2026, 2, 24, 14, 0, tzinfo=UTC)  # Tuesday 14:00 UTC
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        _, symbols, _ = get_market_state()
        assert symbols == set(SYMBOLS.keys())

    @patch("portfolio.market_timing.datetime")
    def test_symbols_crypto_metals_only_on_weekend(self, mock_dt):
        from portfolio.tickers import CRYPTO_SYMBOLS, METALS_SYMBOLS
        fake_now = datetime(2026, 2, 28, 12, 0, tzinfo=UTC)  # Saturday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        _, symbols, _ = get_market_state()
        assert symbols == CRYPTO_SYMBOLS | METALS_SYMBOLS

    @patch("portfolio.market_timing.datetime")
    def test_symbols_crypto_metals_only_at_night(self, mock_dt):
        from portfolio.tickers import CRYPTO_SYMBOLS, METALS_SYMBOLS
        fake_now = datetime(2026, 2, 24, 23, 0, tzinfo=UTC)  # Tuesday 23:00 UTC
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, symbols, interval = get_market_state()
        assert state == "closed"
        assert symbols == CRYPTO_SYMBOLS | METALS_SYMBOLS
        assert interval == 120


# ===========================================================================
# Additional: constants sanity checks
# ===========================================================================

class TestConstants:
    def test_market_open_hour(self):
        assert MARKET_OPEN_HOUR == 7

    def test_interval_market_open(self):
        assert INTERVAL_MARKET_OPEN == 60

    def test_interval_market_closed(self):
        assert INTERVAL_MARKET_CLOSED == 120

    def test_interval_weekend(self):
        assert INTERVAL_WEEKEND == 600


# ===========================================================================
# Edge cases: _is_agent_window EDT vs EST boundary affects window end
# ===========================================================================

class TestAgentWindowDstBoundary:
    """The agent window end changes from 21:00 to 20:00 when DST starts."""

    def test_est_window_end_at_2059(self):
        # EST (Feb): window ends at 21:00 (NYSE close). 20:59 is inside.
        now = datetime(2026, 2, 24, 20, 59, tzinfo=UTC)  # Tuesday
        assert _is_agent_window(now) is True

    def test_est_window_end_at_2100(self):
        # EST (Feb): window ends at 21:00. 21:00 is outside.
        now = datetime(2026, 2, 24, 21, 0, tzinfo=UTC)  # Tuesday
        assert _is_agent_window(now) is False

    def test_edt_window_end_at_1959(self):
        # EDT (Jul): window ends at 20:00 (NYSE close). 19:59 is inside.
        now = datetime(2026, 7, 7, 19, 59, tzinfo=UTC)  # Tuesday
        assert _is_agent_window(now) is True

    def test_edt_window_end_at_2000(self):
        # EDT (Jul): window ends at 20:00. 20:00 is outside.
        now = datetime(2026, 7, 7, 20, 0, tzinfo=UTC)  # Tuesday
        assert _is_agent_window(now) is False

    def test_edt_2030_would_be_inside_est_but_outside_edt(self):
        # 20:30 in summer (EDT): outside (window ends 20:00)
        now = datetime(2026, 7, 7, 20, 30, tzinfo=UTC)  # Tuesday
        assert _is_agent_window(now) is False

    def test_est_2030_inside(self):
        # 20:30 in winter (EST): inside (window ends 21:00)
        now = datetime(2026, 1, 6, 20, 30, tzinfo=UTC)  # Tuesday
        assert _is_agent_window(now) is True


# ===========================================================================
# Edge cases: get_market_state at boundary hours
# ===========================================================================

class TestGetMarketStateBoundaries:
    """Test exact boundary between open and closed states."""

    @patch("portfolio.market_timing.datetime")
    def test_at_exactly_close_hour_edt(self, mock_dt):
        """At hour 20 in EDT => hour == close_hour => NOT open => closed."""
        fake_now = datetime(2026, 7, 8, 20, 0, tzinfo=UTC)  # Wednesday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, _, _ = get_market_state()
        assert state == "closed"

    @patch("portfolio.market_timing.datetime")
    def test_at_exactly_close_hour_est(self, mock_dt):
        """At hour 21 in EST => hour == close_hour => NOT open => closed."""
        fake_now = datetime(2026, 1, 5, 21, 0, tzinfo=UTC)  # Monday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, _, _ = get_market_state()
        assert state == "closed"

    @patch("portfolio.market_timing.datetime")
    def test_one_hour_before_close_is_open_edt(self, mock_dt):
        """Hour 19 in EDT (close=20) => MARKET_OPEN_HOUR <= 19 < 20 => open."""
        fake_now = datetime(2026, 7, 8, 19, 0, tzinfo=UTC)  # Wednesday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, _, _ = get_market_state()
        assert state == "open"

    @patch("portfolio.market_timing.datetime")
    def test_at_exactly_eu_open_summer(self, mock_dt):
        """H47: At 07:00 UTC in summer (EU open) => open."""
        fake_now = datetime(2026, 7, 7, 7, 0, tzinfo=UTC)  # Tuesday, summer
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, _, _ = get_market_state()
        assert state == "open"

    @patch("portfolio.market_timing.datetime")
    def test_at_exactly_eu_open_winter(self, mock_dt):
        """H47: At 08:00 UTC in winter (EU open) => open."""
        fake_now = datetime(2026, 2, 24, 8, 0, tzinfo=UTC)  # Tuesday, winter
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, _, _ = get_market_state()
        assert state == "open"

    @patch("portfolio.market_timing.datetime")
    def test_one_hour_before_market_open(self, mock_dt):
        """At hour 6 => closed (before EU open in both summer and winter)."""
        fake_now = datetime(2026, 2, 24, 6, 0, tzinfo=UTC)  # Tuesday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, _, _ = get_market_state()
        assert state == "closed"

    @patch("portfolio.market_timing.datetime")
    def test_friday_at_close_transitions_to_closed_not_weekend(self, mock_dt):
        """Friday at close hour is still 'closed' (weekday < 5), not 'weekend'."""
        fake_now = datetime(2026, 2, 27, 21, 0, tzinfo=UTC)  # Friday (EST)
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, _, _ = get_market_state()
        assert state == "closed"  # Friday weekday=4, not >= 5


# ===========================================================================
# Parametrized: DST across multiple years
# ===========================================================================

class TestDstMultipleYears:
    """Verify DST calculation for several years by computing expected dates."""

    @pytest.mark.parametrize(
        "year, expected_second_sun_mar, expected_first_sun_nov",
        [
            # 2024: March 1 = Fri => first Sun = 3, second = 10; Nov 1 = Fri => first Sun = 3
            (2024, 10, 3),
            # 2025: March 1 = Sat => first Sun = 2, second = 9; Nov 1 = Sat => first Sun = 2
            (2025, 9, 2),
            # 2026: March 1 = Sun => first Sun = 1, second = 8; Nov 1 = Sun => first Sun = 1
            (2026, 8, 1),
            # 2027: March 1 = Mon => first Sun = 7, second = 14; Nov 1 = Mon => first Sun = 7
            (2027, 14, 7),
            # 2028: March 1 = Wed => first Sun = 5, second = 12; Nov 1 = Wed => first Sun = 5
            (2028, 12, 5),
            # 2029: March 1 = Thu => first Sun = 4, second = 11; Nov 1 = Thu => first Sun = 4
            (2029, 11, 4),
            # 2030: March 1 = Fri => first Sun = 3, second = 10; Nov 1 = Fri => first Sun = 3
            (2030, 10, 3),
        ],
    )
    def test_dst_boundaries_across_years(
        self, year, expected_second_sun_mar, expected_first_sun_nov
    ):
        # Verify the day we expect is indeed a Sunday
        assert date(year, 3, expected_second_sun_mar).weekday() == 6  # Sunday
        assert date(year, 11, expected_first_sun_nov).weekday() == 6  # Sunday

        # Before DST start => False
        before_start = datetime(
            year, 3, expected_second_sun_mar, 6, 59, tzinfo=UTC
        )
        assert _is_us_dst(before_start) is False

        # At DST start => True
        at_start = datetime(
            year, 3, expected_second_sun_mar, 7, 0, tzinfo=UTC
        )
        assert _is_us_dst(at_start) is True

        # Before DST end => True
        before_end = datetime(
            year, 11, expected_first_sun_nov, 5, 59, tzinfo=UTC
        )
        assert _is_us_dst(before_end) is True

        # At DST end => False
        at_end = datetime(
            year, 11, expected_first_sun_nov, 6, 0, tzinfo=UTC
        )
        assert _is_us_dst(at_end) is False


# ===========================================================================
# 14. _easter_sunday() — verified against known Easter dates
# ===========================================================================

class TestEasterSunday:
    """Easter algorithm validated against known dates for 2024-2030."""

    @pytest.mark.parametrize("year, expected_month, expected_day", [
        (2024, 3, 31),
        (2025, 4, 20),
        (2026, 4, 5),
        (2027, 3, 28),
        (2028, 4, 16),
        (2029, 4, 1),
        (2030, 4, 21),
    ])
    def test_known_easters(self, year, expected_month, expected_day):
        result = _easter_sunday(year)
        assert result == date(year, expected_month, expected_day)


# ===========================================================================
# 15. US market holidays — known 2026 dates
# ===========================================================================

class TestUSMarketHolidays:
    """Verify NYSE holiday calendar for 2026 against known dates."""

    def test_2026_holidays(self):
        holidays = us_market_holidays(2026)
        expected = {
            date(2026, 1, 1),    # New Year's Day (Thursday)
            date(2026, 1, 19),   # MLK Day (3rd Monday Jan)
            date(2026, 2, 16),   # Presidents' Day (3rd Monday Feb)
            date(2026, 4, 3),    # Good Friday (Easter Apr 5)
            date(2026, 5, 25),   # Memorial Day (last Monday May)
            date(2026, 6, 19),   # Juneteenth (Friday)
            date(2026, 7, 3),    # Independence Day observed (Jul 4 = Saturday → Friday)
            date(2026, 9, 7),    # Labor Day (1st Monday Sep)
            date(2026, 11, 26),  # Thanksgiving (4th Thursday Nov)
            date(2026, 12, 25),  # Christmas (Friday)
        }
        assert holidays == expected

    def test_2027_independence_day_observed(self):
        """Jul 4 2027 is a Sunday → observed Monday Jul 5."""
        holidays = us_market_holidays(2027)
        assert date(2027, 7, 5) in holidays
        assert date(2027, 7, 4) not in holidays

    def test_holiday_count(self):
        """NYSE has exactly 10 holidays per year."""
        for year in range(2024, 2031):
            holidays = us_market_holidays(year)
            assert len(holidays) == 10, f"Year {year}: expected 10, got {len(holidays)}"

    def test_no_weekend_holidays(self):
        """All returned dates should be weekdays (observed-date shifts handle this)."""
        for year in range(2024, 2031):
            for d in us_market_holidays(year):
                assert d.weekday() < 5, f"{d} is a weekend day in {year}"


class TestIsUSMarketHoliday:
    """Test the is_us_market_holiday() convenience function."""

    def test_good_friday_2026(self):
        dt = datetime(2026, 4, 3, 14, 0, tzinfo=UTC)
        assert is_us_market_holiday(dt) is True

    def test_easter_monday_not_us_holiday(self):
        """Easter Monday is NOT a US market holiday (it IS a Swedish one)."""
        dt = datetime(2026, 4, 6, 14, 0, tzinfo=UTC)
        assert is_us_market_holiday(dt) is False

    def test_regular_monday(self):
        dt = datetime(2026, 3, 2, 14, 0, tzinfo=UTC)
        assert is_us_market_holiday(dt) is False

    def test_christmas_2026(self):
        dt = datetime(2026, 12, 25, 14, 0, tzinfo=UTC)
        assert is_us_market_holiday(dt) is True

    def test_accepts_date_object(self):
        assert is_us_market_holiday(date(2026, 4, 3)) is True

    def test_default_now(self):
        """Calling with no args should not raise."""
        result = is_us_market_holiday()
        assert isinstance(result, bool)


# ===========================================================================
# 16. Swedish market holidays — known 2026 dates
# ===========================================================================

class TestSwedishMarketHolidays:
    """Verify Nasdaq Stockholm / Avanza holiday calendar for 2026."""

    def test_2026_holidays(self):
        holidays = swedish_market_holidays(2026)
        expected = {
            date(2026, 1, 1),    # New Year's Day
            date(2026, 1, 6),    # Epiphany
            date(2026, 4, 3),    # Good Friday
            date(2026, 4, 6),    # Easter Monday
            date(2026, 5, 1),    # May Day
            date(2026, 5, 14),   # Ascension Day (Easter + 39)
            date(2026, 5, 23),   # Whitsun Eve / Pingstafton (Easter + 49)
            date(2026, 6, 6),    # National Day
            date(2026, 6, 19),   # Midsummer Eve
            date(2026, 12, 24),  # Christmas Eve
            date(2026, 12, 25),  # Christmas Day
            date(2026, 12, 26),  # Boxing Day
            date(2026, 12, 31),  # New Year's Eve
        }
        assert holidays == expected

    def test_easter_monday_is_swedish_holiday(self):
        """Easter Monday 2026 (Apr 6) — the day that triggered this fix."""
        assert date(2026, 4, 6) in swedish_market_holidays(2026)

    def test_midsummer_eve_2026(self):
        """Midsummer Eve 2026 = Fri Jun 19 (Midsummer Day = Sat Jun 20)."""
        assert date(2026, 6, 19) in swedish_market_holidays(2026)

    def test_midsummer_eve_2027(self):
        """Midsummer Day 2027 = Sat Jun 26, Eve = Fri Jun 25."""
        assert date(2027, 6, 25) in swedish_market_holidays(2027)

    def test_holiday_count(self):
        """Swedish market has 13 holidays per year."""
        for year in range(2024, 2031):
            holidays = swedish_market_holidays(year)
            assert len(holidays) == 13, f"Year {year}: expected 13, got {len(holidays)}"

    def test_whitsun_eve_2026(self):
        """Whitsun Eve (Pingstafton) 2026 = Sat May 23 (Easter + 49)."""
        assert date(2026, 5, 23) in swedish_market_holidays(2026)

    def test_ascension_day_always_thursday(self):
        """Ascension Day is always a Thursday (Easter + 39 days)."""
        for year in range(2024, 2031):
            easter = _easter_sunday(year)
            ascension = easter + timedelta(days=39)
            assert ascension.weekday() == 3, f"Year {year}: Ascension not Thursday"
            assert ascension in swedish_market_holidays(year)


class TestIsSwedishMarketHoliday:
    """Test the is_swedish_market_holiday() convenience function."""

    def test_easter_monday_2026(self):
        dt = datetime(2026, 4, 6, 10, 0, tzinfo=UTC)
        assert is_swedish_market_holiday(dt) is True

    def test_good_friday_2026(self):
        dt = datetime(2026, 4, 3, 10, 0, tzinfo=UTC)
        assert is_swedish_market_holiday(dt) is True

    def test_regular_tuesday(self):
        dt = datetime(2026, 3, 3, 10, 0, tzinfo=UTC)
        assert is_swedish_market_holiday(dt) is False

    def test_accepts_date_object(self):
        assert is_swedish_market_holiday(date(2026, 4, 6)) is True


# ===========================================================================
# 17. Holiday integration — agent window and market state
# ===========================================================================

class TestHolidayAgentWindow:
    """Agent window should be False on US market holidays."""

    def test_good_friday_midday(self):
        """Good Friday 2026 (Apr 3) at 14:00 UTC — would be agent window on normal Friday."""
        now = datetime(2026, 4, 3, 14, 0, tzinfo=UTC)
        assert _is_agent_window(now) is False

    def test_mlk_day_midday(self):
        """MLK Day 2026 (Jan 19) at 14:00 UTC — Monday holiday."""
        now = datetime(2026, 1, 19, 14, 0, tzinfo=UTC)
        assert _is_agent_window(now) is False

    def test_day_after_good_friday(self):
        """Saturday after Good Friday — still False (weekend)."""
        now = datetime(2026, 4, 4, 14, 0, tzinfo=UTC)
        assert _is_agent_window(now) is False

    def test_monday_after_easter(self):
        """Easter Monday is NOT a US holiday — agent window should be True."""
        now = datetime(2026, 4, 6, 14, 0, tzinfo=UTC)
        assert _is_agent_window(now) is True


class TestHolidayMarketState:
    """get_market_state() should return 'holiday' on US holidays."""

    @patch("portfolio.market_timing.datetime")
    def test_good_friday_returns_holiday(self, mock_dt):
        fake_now = datetime(2026, 4, 3, 14, 0, tzinfo=UTC)
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, symbols, interval = get_market_state()
        assert state == "holiday"
        assert interval == INTERVAL_MARKET_CLOSED

    @patch("portfolio.market_timing.datetime")
    def test_christmas_returns_holiday(self, mock_dt):
        fake_now = datetime(2026, 12, 25, 14, 0, tzinfo=UTC)
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        state, _, _ = get_market_state()
        assert state == "holiday"

    @patch("portfolio.market_timing.datetime")
    def test_holiday_only_has_crypto_and_metals(self, mock_dt):
        """On holidays, only crypto + metals should be active (no US stocks)."""
        from portfolio.tickers import CRYPTO_SYMBOLS, METALS_SYMBOLS, STOCK_SYMBOLS

        fake_now = datetime(2026, 4, 3, 14, 0, tzinfo=UTC)
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        _, symbols, _ = get_market_state()
        # Should contain crypto + metals
        assert CRYPTO_SYMBOLS.issubset(symbols)
        assert METALS_SYMBOLS.issubset(symbols)
        # Should NOT contain any stock symbols
        assert not (symbols & STOCK_SYMBOLS)
