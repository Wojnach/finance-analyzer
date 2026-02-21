"""Market timing utilities — DST-aware NYSE hours, market state detection."""

from datetime import date, datetime, timezone
from portfolio.tickers import SYMBOLS

# Market hours (UTC)
MARKET_OPEN_HOUR = 7  # ~Frankfurt/London open

# Loop intervals by market state
INTERVAL_MARKET_OPEN = 60     # 1 min — full speed
INTERVAL_MARKET_CLOSED = 300  # 5 min — crypto only weekday nights
INTERVAL_WEEKEND = 600        # 10 min — crypto only weekends


def _is_us_dst(dt):
    """Check if a UTC datetime falls within US Eastern Daylight Time (EDT).

    US DST rule (since 2007):
      Starts: second Sunday of March at 02:00 local (07:00 UTC)
      Ends:   first Sunday of November at 02:00 local (06:00 UTC)

    Returns True during EDT (Mar-Nov), False during EST (Nov-Mar).
    """
    year = dt.year

    # Second Sunday of March
    mar1_wd = date(year, 3, 1).weekday()  # 0=Mon..6=Sun
    first_sun_mar = 1 + (6 - mar1_wd) % 7
    second_sun_mar = first_sun_mar + 7
    dst_start = datetime(year, 3, second_sun_mar, 7, 0, tzinfo=timezone.utc)

    # First Sunday of November
    nov1_wd = date(year, 11, 1).weekday()
    first_sun_nov = 1 + (6 - nov1_wd) % 7
    dst_end = datetime(year, 11, first_sun_nov, 6, 0, tzinfo=timezone.utc)

    return dst_start <= dt < dst_end


def _market_close_hour_utc(dt):
    """Return the NYSE close hour in UTC, adjusted for DST.

    NYSE closes at 16:00 ET.
    EDT (Mar-Nov): 16:00 ET = 20:00 UTC
    EST (Nov-Mar): 16:00 ET = 21:00 UTC
    """
    if _is_us_dst(dt):
        return 20
    return 21


def _is_agent_window(now=None):
    """Check if current time is within the Layer 2 invocation window.

    Window: 1h before EU market open through 1h after US market close.
    Weekends: no agent invocation.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    current_minutes = now.hour * 60 + now.minute
    start = 6 * 60  # 06:00 UTC (1h before Frankfurt 07:00)
    if _is_us_dst(now):
        end = 21 * 60  # 21:00 UTC (1h after NYSE 20:00 EDT)
    else:
        end = 22 * 60  # 22:00 UTC (1h after NYSE 21:00 EST)
    return start <= current_minutes < end


def get_market_state():
    now = datetime.now(timezone.utc)
    weekday = now.weekday()  # 0=Mon, 6=Sun
    hour = now.hour
    all_symbols = set(SYMBOLS.keys())
    if weekday >= 5:
        return "weekend", all_symbols, INTERVAL_WEEKEND
    close_hour = _market_close_hour_utc(now)
    if MARKET_OPEN_HOUR <= hour < close_hour:
        return "open", all_symbols, INTERVAL_MARKET_OPEN
    return "closed", all_symbols, INTERVAL_MARKET_CLOSED
