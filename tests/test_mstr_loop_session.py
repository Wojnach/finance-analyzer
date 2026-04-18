"""Session-window + kill-switch tests for portfolio.mstr_loop.session."""

from __future__ import annotations

import datetime
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portfolio.mstr_loop import session


def _mk(year, month, day, hour, minute):
    """Build a datetime with fake CET tz for testing session math."""
    tz = datetime.timezone(datetime.timedelta(hours=2))  # pretend CEST year-round
    return datetime.datetime(year, month, day, hour, minute, tzinfo=tz)


# ---------------------------------------------------------------------------
# in_session_window
# ---------------------------------------------------------------------------


def test_window_before_open_is_false():
    # 2026-04-17 is a Friday — test the pre-open time on a weekday.
    assert session.in_session_window(_mk(2026, 4, 17, 15, 0)) is False


def test_window_at_open_is_true():
    # 2026-04-17 is a Friday — weekday where the window is open.
    assert session.in_session_window(_mk(2026, 4, 17, 15, 30)) is True


def test_window_mid_session_is_true():
    assert session.in_session_window(_mk(2026, 4, 17, 18, 0)) is True


def test_window_at_close_exact_is_false():
    """22:00 is the close itself — not in the window."""
    assert session.in_session_window(_mk(2026, 4, 17, 22, 0)) is False


def test_window_after_close_is_false():
    assert session.in_session_window(_mk(2026, 4, 17, 23, 0)) is False


def test_window_weekend_saturday_is_false():
    # 2026-04-18 is a Saturday
    assert session.in_session_window(_mk(2026, 4, 18, 18, 0)) is False


def test_window_weekend_sunday_is_false():
    # 2026-04-19 is a Sunday
    assert session.in_session_window(_mk(2026, 4, 19, 18, 0)) is False


def test_window_weekday_friday_session_ok():
    # 2026-04-17 is a Friday
    assert session.in_session_window(_mk(2026, 4, 17, 18, 0)) is True


# ---------------------------------------------------------------------------
# in_eod_flatten_window
# ---------------------------------------------------------------------------


def test_eod_before_21_45_is_false():
    assert session.in_eod_flatten_window(_mk(2026, 4, 17, 21, 30)) is False


def test_eod_at_21_45_is_true():
    assert session.in_eod_flatten_window(_mk(2026, 4, 17, 21, 45)) is True


def test_eod_mid_flatten_is_true():
    assert session.in_eod_flatten_window(_mk(2026, 4, 17, 21, 55)) is True


def test_eod_at_close_is_true():
    assert session.in_eod_flatten_window(_mk(2026, 4, 17, 22, 0)) is True


def test_eod_after_close_is_false():
    assert session.in_eod_flatten_window(_mk(2026, 4, 17, 22, 15)) is False


def test_eod_weekend_is_false():
    assert session.in_eod_flatten_window(_mk(2026, 4, 18, 21, 50)) is False


# ---------------------------------------------------------------------------
# kill_switch_active
# ---------------------------------------------------------------------------


def test_kill_switch_inactive_by_default(tmp_path):
    assert session.kill_switch_active(str(tmp_path / "no.disabled")) is False


def test_kill_switch_active_when_file_exists(tmp_path):
    f = tmp_path / "enabled.disabled"
    f.write_text("")
    assert session.kill_switch_active(str(f)) is True


# ---------------------------------------------------------------------------
# seconds_until_next_session
# ---------------------------------------------------------------------------


def test_seconds_until_within_session_is_zero():
    # Mid-window call — caller shouldn't use this, we just return 0
    assert session.seconds_until_next_session(_mk(2026, 4, 17, 18, 0)) == 0


def test_seconds_until_before_open_same_day():
    # 10:30 AM on Friday → 5h to 15:30 open = 18000s
    sec = session.seconds_until_next_session(_mk(2026, 4, 17, 10, 30))
    assert 17900 <= sec <= 18100


def test_seconds_until_after_close_rolls_next_weekday():
    # 23:00 Friday → next weekday is Monday 15:30
    # 2.5d * 86400 + 2.5h * 3600 = 216,000 + ~9000 ≈ 225,000
    sec = session.seconds_until_next_session(_mk(2026, 4, 17, 23, 0))
    # Friday 23:00 → Monday 15:30 = 2 days 16.5h = 232200 seconds (approx)
    assert 228000 < sec < 240000


def test_seconds_until_saturday_rolls_to_monday():
    # Saturday 10:00 → Monday 15:30 = ~2d 5.5h
    sec = session.seconds_until_next_session(_mk(2026, 4, 18, 10, 0))
    # 2 days 5.5h = 192600 seconds
    assert 190000 < sec < 195000


def test_last_sunday_march():
    # Last Sunday in March 2026 = 2026-03-29
    assert session._last_sunday(2026, 3) == datetime.date(2026, 3, 29)


def test_last_sunday_october():
    # Last Sunday in October 2026 = 2026-10-25
    assert session._last_sunday(2026, 10) == datetime.date(2026, 10, 25)


def test_last_sunday_december():
    # Last Sunday in December 2026 = 2026-12-27 — exercises the year-rollover path
    assert session._last_sunday(2026, 12) == datetime.date(2026, 12, 27)
