"""Market timing utilities — DST-aware NYSE and EU hours, market state detection.

Includes US (NYSE) and Swedish (Nasdaq Stockholm / Avanza) holiday calendars
so the system skips stock/warrant processing on public holidays, not just weekends.
"""

from datetime import UTC, date, datetime, timedelta

from portfolio.tickers import CRYPTO_SYMBOLS, METALS_SYMBOLS, STOCK_SYMBOLS, SYMBOLS

# Backward compat: MARKET_OPEN_HOUR kept at 7 (summer value).
# Callers that need DST-aware EU open should use _eu_market_open_hour_utc().
MARKET_OPEN_HOUR = 7

# Loop intervals by market state
INTERVAL_MARKET_OPEN = 60     # 1 min — full speed
INTERVAL_MARKET_CLOSED = 120  # 2 min — crypto only weekday nights
INTERVAL_WEEKEND = 600        # 10 min — crypto only weekends


def _is_eu_dst(dt):
    """Check if a UTC datetime falls within EU Summer Time (CEST).

    EU DST rule:
      Starts: last Sunday of March at 01:00 UTC
      Ends:   last Sunday of October at 01:00 UTC

    Returns True during CEST (summer), False during CET (winter).
    """
    year = dt.year

    # Last Sunday of March
    mar31 = date(year, 3, 31)
    last_sun_mar = 31 - (mar31.weekday() + 1) % 7
    eu_dst_start = datetime(year, 3, last_sun_mar, 1, 0, tzinfo=UTC)

    # Last Sunday of October
    oct31 = date(year, 10, 31)
    last_sun_oct = 31 - (oct31.weekday() + 1) % 7
    eu_dst_end = datetime(year, 10, last_sun_oct, 1, 0, tzinfo=UTC)

    return eu_dst_start <= dt < eu_dst_end


def _eu_market_open_hour_utc(dt):
    """Return the EU market open hour in UTC, adjusted for EU DST.

    H47: London/Frankfurt open at 08:00 local time.
    CEST (summer, BST=UTC+1): 08:00 local = 07:00 UTC
    CET (winter, GMT=UTC+0): 08:00 local = 08:00 UTC

    Previously hardcoded to 7 UTC year-round, which missed the winter hour.
    """
    if _is_eu_dst(dt):
        return 7
    return 8


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
    dst_start = datetime(year, 3, second_sun_mar, 7, 0, tzinfo=UTC)

    # First Sunday of November
    nov1_wd = date(year, 11, 1).weekday()
    first_sun_nov = 1 + (6 - nov1_wd) % 7
    dst_end = datetime(year, 11, first_sun_nov, 6, 0, tzinfo=UTC)

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


# ---------------------------------------------------------------------------
# Holiday calendars
# ---------------------------------------------------------------------------


def _easter_sunday(year):
    """Compute Easter Sunday for a given year using the Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _observed(d):
    """Return the NYSE-observed date for a fixed holiday.

    If the holiday falls on Saturday, NYSE observes it Friday.
    If Sunday, NYSE observes it Monday.
    """
    if d.weekday() == 5:  # Saturday
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday
        return d + timedelta(days=1)
    return d


def _nth_weekday(year, month, weekday, n):
    """Return the nth occurrence of a weekday in a given month.

    weekday: 0=Mon, 6=Sun.  n: 1-based (1=first, 2=second, etc.)
    """
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year, month, weekday):
    """Return the last occurrence of a weekday in a given month."""
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    last_day = next_month - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def us_market_holidays(year):
    """Return the set of NYSE holiday dates for a given year.

    Covers all 10 NYSE holidays including observed-date shifts.
    """
    easter = _easter_sunday(year)
    holidays = {
        _observed(date(year, 1, 1)),                 # New Year's Day
        _nth_weekday(year, 1, 0, 3),                 # MLK Day (3rd Mon Jan)
        _nth_weekday(year, 2, 0, 3),                 # Presidents' Day (3rd Mon Feb)
        easter - timedelta(days=2),                   # Good Friday
        _last_weekday(year, 5, 0),                    # Memorial Day (last Mon May)
        _observed(date(year, 6, 19)),                 # Juneteenth
        _observed(date(year, 7, 4)),                  # Independence Day
        _nth_weekday(year, 9, 0, 1),                  # Labor Day (1st Mon Sep)
        _nth_weekday(year, 11, 3, 4),                 # Thanksgiving (4th Thu Nov)
        _observed(date(year, 12, 25)),                # Christmas
    }
    return holidays


def is_us_market_holiday(dt=None):
    """Return True if the given UTC datetime falls on a NYSE holiday."""
    if dt is None:
        dt = datetime.now(UTC)
    d = dt.date() if hasattr(dt, "date") else dt
    return d in us_market_holidays(d.year)


def swedish_market_holidays(year):
    """Return the set of Nasdaq Stockholm / Avanza holiday dates for a given year.

    Covers full days when Avanza warrant trading is closed.
    """
    easter = _easter_sunday(year)

    # Midsummer Eve: Friday before Midsummer Day (Saturday between Jun 20-26)
    # Midsummer Day = first Saturday on or after Jun 20
    jun20 = date(year, 6, 20)
    days_to_sat = (5 - jun20.weekday()) % 7
    midsummer_day = jun20 + timedelta(days=days_to_sat)
    midsummer_eve = midsummer_day - timedelta(days=1)

    holidays = {
        date(year, 1, 1),                            # New Year's Day
        date(year, 1, 6),                             # Epiphany
        easter - timedelta(days=2),                   # Good Friday
        easter + timedelta(days=1),                   # Easter Monday
        date(year, 5, 1),                             # May Day
        easter + timedelta(days=39),                  # Ascension Day
        date(year, 6, 6),                             # National Day
        midsummer_eve,                                # Midsummer Eve
        date(year, 12, 24),                           # Christmas Eve
        date(year, 12, 25),                           # Christmas Day
        date(year, 12, 26),                           # Boxing Day
        date(year, 12, 31),                           # New Year's Eve
    }
    return holidays


def is_swedish_market_holiday(dt=None):
    """Return True if the given UTC datetime falls on a Swedish market holiday."""
    if dt is None:
        dt = datetime.now(UTC)
    d = dt.date() if hasattr(dt, "date") else dt
    return d in swedish_market_holidays(d.year)


def _is_agent_window(now=None):
    """Check if current time is within the Layer 2 invocation window.

    Window: EU market open through US market close.
    Summer: 07:00–20:00 UTC
    Winter: 08:00–21:00 UTC
    Weekends and US market holidays: no agent invocation.
    """
    if now is None:
        now = datetime.now(UTC)
    if now.weekday() >= 5:
        return False
    if is_us_market_holiday(now):
        return False
    eu_open = _eu_market_open_hour_utc(now)
    close_hour = _market_close_hour_utc(now)
    return eu_open <= now.hour < close_hour


def _market_open_hour_utc(dt):
    """Return the NYSE open hour in UTC, adjusted for DST.

    NYSE opens at 09:30 ET.
    EDT (Mar-Nov): 09:30 ET = 13:30 UTC -> hour 13
    EST (Nov-Mar): 09:30 ET = 14:30 UTC -> hour 14
    """
    if _is_us_dst(dt):
        return 13
    return 14


def is_us_stock_market_open(now=None, pre_market_buffer_min=0, post_market_buffer_min=0):
    """Check if US stock market (NYSE) is currently open.

    Args:
        now: UTC datetime (default: current time)
        pre_market_buffer_min: minutes before open to consider "open"
        post_market_buffer_min: minutes after close to consider "open"

    Returns:
        True if within [open - pre_buffer, close + post_buffer] on weekdays.
    """
    if now is None:
        now = datetime.now(UTC)
    if now.weekday() >= 5:
        return False
    if is_us_market_holiday(now):
        return False

    open_hour = _market_open_hour_utc(now)
    close_hour = _market_close_hour_utc(now)

    # Convert to minutes-since-midnight for easy buffer math
    now_min = now.hour * 60 + now.minute
    open_min = open_hour * 60 + 30 - pre_market_buffer_min   # NYSE opens at :30
    close_min = close_hour * 60 + post_market_buffer_min      # NYSE closes at :00

    return open_min <= now_min < close_min


def should_skip_gpu(ticker, config=None, now=None):
    """Determine if GPU-intensive signals should be skipped for this ticker.

    Returns True for US stocks when the US market is closed.
    Returns False for crypto and metals (always run GPU signals).
    """
    if ticker not in STOCK_SYMBOLS:
        return False

    gpu_cfg = (config or {}).get("gpu_signals", {})
    if not gpu_cfg.get("skip_stocks_offhours", True):
        return False

    pre_buffer = gpu_cfg.get("pre_market_buffer_min", 30)
    post_buffer = gpu_cfg.get("post_market_buffer_min", 15)

    return not is_us_stock_market_open(
        now=now,
        pre_market_buffer_min=pre_buffer,
        post_market_buffer_min=post_buffer,
    )


def get_market_state():
    now = datetime.now(UTC)
    weekday = now.weekday()  # 0=Mon, 6=Sun
    hour = now.hour
    all_symbols = set(SYMBOLS.keys())
    always_on = CRYPTO_SYMBOLS | METALS_SYMBOLS
    if weekday >= 5:
        return "weekend", always_on, INTERVAL_WEEKEND
    # US holiday: treat like off-hours (crypto + metals only, 2-min interval)
    if is_us_market_holiday(now):
        return "holiday", always_on, INTERVAL_MARKET_CLOSED
    eu_open = _eu_market_open_hour_utc(now)
    close_hour = _market_close_hour_utc(now)
    if eu_open <= hour < close_hour:
        return "open", all_symbols, INTERVAL_MARKET_OPEN
    return "closed", always_on, INTERVAL_MARKET_CLOSED
