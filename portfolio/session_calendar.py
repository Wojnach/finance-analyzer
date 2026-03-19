"""Session calendar — instrument-specific trading hours and session state.

Provides remaining-session time, session boundaries, and session mismatch
detection for the exit optimizer.

Usage:
    from portfolio.session_calendar import get_session_info
    info = get_session_info("warrant", underlying="XAG-USD")
    # info.remaining_minutes, info.session_end, info.is_extended, ...
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta

from portfolio.market_timing import _is_us_dst


@dataclass(frozen=True)
class SessionInfo:
    """Trading session state for an instrument.

    Attributes:
        session_end: Absolute datetime (UTC) of normal session close.
        extended_end: Absolute datetime (UTC) of extended session close, if applicable.
        remaining_minutes: Minutes until effective close (extended if available).
        is_open: Whether the instrument is currently tradeable.
        is_extended: Whether we're in the extended (evening) session.
        underlying_open: Whether the underlying's primary market is open (for warrants).
        phase: Human-readable phase: "open", "extended", "pre_open", "closed".
    """
    session_end: datetime
    extended_end: datetime | None
    remaining_minutes: float
    is_open: bool
    is_extended: bool
    underlying_open: bool
    phase: str


# ---------------------------------------------------------------------------
# Session definitions (times in UTC)
# ---------------------------------------------------------------------------

# Avanza commodity warrants: 08:15-21:55 CET = 07:15-20:55 UTC (winter)
# CET = UTC+1 (winter), CEST = UTC+2 (summer)
# We handle DST for EU sessions too.

def _eu_dst(dt: datetime) -> bool:
    """Check if datetime falls in EU Central European Summer Time (CEST).

    EU DST: last Sunday of March 01:00 UTC → last Sunday of October 01:00 UTC.
    """
    year = dt.year

    # Last Sunday of March
    mar31 = datetime(year, 3, 31, tzinfo=UTC)
    last_sun_mar = 31 - (mar31.weekday() + 1) % 7
    dst_start = datetime(year, 3, last_sun_mar, 1, 0, tzinfo=UTC)

    # Last Sunday of October
    oct31 = datetime(year, 10, 31, tzinfo=UTC)
    last_sun_oct = 31 - (oct31.weekday() + 1) % 7
    dst_end = datetime(year, 10, last_sun_oct, 1, 0, tzinfo=UTC)

    return dst_start <= dt < dst_end


def _cet_offset(dt: datetime) -> int:
    """Return CET/CEST offset from UTC in hours (1 or 2)."""
    return 2 if _eu_dst(dt) else 1


def _cet_to_utc(hour: int, minute: int, dt: datetime) -> time:
    """Convert CET time to UTC time object, adjusted for DST on given date."""
    offset = _cet_offset(dt)
    utc_hour = (hour - offset) % 24
    return time(utc_hour, minute)


def _make_session_end(now: datetime, cet_hour: int, cet_minute: int) -> datetime:
    """Create a UTC datetime for today's session end from CET time."""
    offset = _cet_offset(now)
    utc_hour = cet_hour - offset
    end = now.replace(hour=utc_hour, minute=cet_minute, second=0, microsecond=0)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    return end


# Session specs: (open_cet, close_cet) as (hour, minute) tuples
SESSIONS = {
    "warrant": {
        "open_cet": (8, 15),
        "close_cet": (21, 55),
        "has_extended": False,  # Already includes evening trading
        "description": "Avanza commodity warrants",
    },
    "stock_se": {
        "open_cet": (9, 0),
        "close_cet": (17, 25),
        "has_extended": False,
        "description": "Nasdaq Stockholm equities",
    },
    "crypto": {
        "open_cet": (0, 0),
        "close_cet": (23, 59),
        "has_extended": False,
        "description": "Crypto 24/7",
    },
}


def get_session_info(instrument_type: str,
                     underlying: str | None = None,
                     now: datetime | None = None) -> SessionInfo:
    """Get current session state for an instrument.

    Args:
        instrument_type: "warrant", "stock_se", "stock_us", "crypto".
        underlying: Underlying ticker for warrants (e.g., "XAG-USD").
        now: Current UTC time. Defaults to now.

    Returns:
        SessionInfo with remaining time, phase, and session boundaries.
    """
    if now is None:
        now = datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    # Crypto: always open (24/7)
    if instrument_type == "crypto":
        # Use midnight as "session end" — effectively infinite session
        end = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return SessionInfo(
            session_end=end,
            extended_end=None,
            remaining_minutes=(end - now).total_seconds() / 60,
            is_open=True,
            is_extended=False,
            underlying_open=True,
            phase="open",
        )

    # US stocks: NYSE hours with DST
    if instrument_type == "stock_us":
        us_dst = _is_us_dst(now)
        open_utc = 13 if us_dst else 14   # 09:30 ET
        close_utc = 20 if us_dst else 21  # 16:00 ET

        session_end = now.replace(hour=close_utc, minute=0, second=0, microsecond=0)
        is_open = (now.weekday() < 5 and
                   now.replace(hour=open_utc, minute=30, second=0) <= now < session_end)

        remaining = max(0, (session_end - now).total_seconds() / 60) if is_open else 0
        phase = "open" if is_open else "closed"

        # Check if underlying is open (for warrants referencing US stocks)
        underlying_open = is_open

        return SessionInfo(
            session_end=session_end,
            extended_end=None,
            remaining_minutes=remaining,
            is_open=is_open,
            is_extended=False,
            underlying_open=underlying_open,
            phase=phase,
        )

    # EU-based instruments (warrants, Stockholm stocks)
    spec = SESSIONS.get(instrument_type, SESSIONS["warrant"])
    oh, om = spec["open_cet"]
    ch, cm = spec["close_cet"]

    session_end = _make_session_end(now, ch, cm)
    session_open = _make_session_end(now, oh, om)

    is_weekday = now.weekday() < 5
    is_open = is_weekday and session_open <= now < session_end

    remaining = max(0, (session_end - now).total_seconds() / 60) if is_open else 0
    phase = "open" if is_open else "closed"

    # For warrants, check if underlying's primary market is open
    underlying_open = True  # Metals trade ~24h
    if underlying and not underlying.endswith("-USD"):
        # US stock underlying — check US market hours
        us_info = get_session_info("stock_us", now=now)
        underlying_open = us_info.is_open

    return SessionInfo(
        session_end=session_end,
        extended_end=None,
        remaining_minutes=remaining,
        is_open=is_open,
        is_extended=False,
        underlying_open=underlying_open,
        phase=phase,
    )


def remaining_session_minutes(instrument_type: str = "warrant",
                              now: datetime | None = None) -> float:
    """Shortcut: get remaining minutes for an instrument's session."""
    info = get_session_info(instrument_type, now=now)
    return info.remaining_minutes
