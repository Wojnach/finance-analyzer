"""Session window + kill-switch checks for MSTR Loop.

The MSTR underlying trades NASDAQ regular hours 15:30–22:00 CET (DST
dependent). The Nordic MTF cert tracks the underlying. We don't enter
new positions outside this window and we EOD-flatten at 21:45 CET.
"""

from __future__ import annotations

import datetime
import os

from portfolio.mstr_loop import config


def _cet_now() -> datetime.datetime:
    """Return current time in CET (UTC+1/+2 depending on DST).

    Uses zoneinfo if available (Python 3.9+ stdlib). Falls back to UTC+1
    during standard time, UTC+2 during DST (rough heuristic — acceptable
    for session-window gating since the 15-min EOD buffer absorbs slop).
    """
    try:
        from zoneinfo import ZoneInfo
        return datetime.datetime.now(ZoneInfo("Europe/Stockholm"))
    except Exception:
        # Fallback: approximate via fixed offset. DST in Sweden is
        # last-Sunday-March to last-Sunday-October.
        now_utc = datetime.datetime.now(datetime.UTC)
        year = now_utc.year
        dst_start = _last_sunday(year, 3)
        dst_end = _last_sunday(year, 10)
        offset_hours = 2 if dst_start <= now_utc.date() < dst_end else 1
        return now_utc + datetime.timedelta(hours=offset_hours)


def _last_sunday(year: int, month: int) -> datetime.date:
    """Return the date of the last Sunday in a given month."""
    if month == 12:
        next_month = datetime.date(year + 1, 1, 1)
    else:
        next_month = datetime.date(year, month + 1, 1)
    last_day = next_month - datetime.timedelta(days=1)
    # weekday(): Mon=0..Sun=6
    offset = (last_day.weekday() - 6) % 7
    return last_day - datetime.timedelta(days=offset)


def _cet_minutes(cet_dt: datetime.datetime) -> int:
    """Minutes since midnight CET."""
    return cet_dt.hour * 60 + cet_dt.minute


def in_session_window(now: datetime.datetime | None = None) -> bool:
    """True if we're inside the 15:30–22:00 CET trading window.

    Weekends and US holidays are not special-cased here — the underlying
    simply won't move, the cert won't either, and the momentum rules will
    not fire an entry on flat data. An explicit holiday calendar is
    post-v1 work.
    """
    dt = now if now is not None else _cet_now()
    if dt.weekday() >= 5:  # Saturday / Sunday
        return False
    minutes = _cet_minutes(dt)
    open_m = config.SESSION_OPEN_CET_HOUR * 60 + config.SESSION_OPEN_CET_MINUTE
    close_m = config.SESSION_CLOSE_CET_HOUR * 60 + config.SESSION_CLOSE_CET_MINUTE
    return open_m <= minutes < close_m


def in_eod_flatten_window(now: datetime.datetime | None = None) -> bool:
    """True once we're inside the EOD flatten window (21:45 CET onward)."""
    dt = now if now is not None else _cet_now()
    if dt.weekday() >= 5:
        return False
    minutes = _cet_minutes(dt)
    eod_m = config.EOD_FLATTEN_CET_HOUR * 60 + config.EOD_FLATTEN_CET_MINUTE
    close_m = config.SESSION_CLOSE_CET_HOUR * 60 + config.SESSION_CLOSE_CET_MINUTE
    return eod_m <= minutes <= close_m


def kill_switch_active(path: str = config.KILL_SWITCH_FILE) -> bool:
    """True if the sentinel file exists — halt everything immediately.

    Used as an ops emergency brake. Operator: `touch data/mstr_loop.disabled`
    to halt; `rm` to resume.
    """
    return os.path.exists(path)


def seconds_until_next_session(now: datetime.datetime | None = None) -> int:
    """Seconds until the next session open (for the runner's idle sleep)."""
    dt = now if now is not None else _cet_now()
    open_m = config.SESSION_OPEN_CET_HOUR * 60 + config.SESSION_OPEN_CET_MINUTE
    close_m = config.SESSION_CLOSE_CET_HOUR * 60 + config.SESSION_CLOSE_CET_MINUTE
    minutes_now = _cet_minutes(dt)

    # If we're mid-weekend or past close on a weekday, advance to next
    # weekday's open.
    target_date = dt.date()
    if dt.weekday() >= 5 or minutes_now >= close_m:
        # Advance until we hit a weekday
        while True:
            target_date = target_date + datetime.timedelta(days=1)
            if target_date.weekday() < 5:
                break
    elif minutes_now < open_m:
        # Before open today
        pass
    else:
        # Currently inside the window — caller shouldn't call this
        return 0

    target_dt = datetime.datetime(
        target_date.year, target_date.month, target_date.day,
        config.SESSION_OPEN_CET_HOUR, config.SESSION_OPEN_CET_MINUTE,
        tzinfo=dt.tzinfo,
    )
    return max(0, int((target_dt - dt).total_seconds()))
