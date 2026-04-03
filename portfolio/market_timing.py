"""Market timing utilities — DST-aware NYSE hours, market state detection."""

from datetime import UTC, date, datetime

from portfolio.tickers import CRYPTO_SYMBOLS, METALS_SYMBOLS, STOCK_SYMBOLS, SYMBOLS

# Market hours (UTC)
MARKET_OPEN_HOUR = 7  # ~Frankfurt/London open

# Loop intervals by market state
INTERVAL_MARKET_OPEN = 60     # 1 min — full speed
INTERVAL_MARKET_CLOSED = 120  # 2 min — crypto only weekday nights
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


def _is_agent_window(now=None):
    """Check if current time is within the Layer 2 invocation window.

    Window: EU market open (07:00 UTC) through US market close.
    EDT (Mar-Nov): 07:00–20:00 UTC
    EST (Nov-Mar): 07:00–21:00 UTC
    Weekends: no agent invocation.
    """
    if now is None:
        now = datetime.now(UTC)
    if now.weekday() >= 5:
        return False
    close_hour = _market_close_hour_utc(now)
    return MARKET_OPEN_HOUR <= now.hour < close_hour


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
    close_hour = _market_close_hour_utc(now)
    if MARKET_OPEN_HOUR <= hour < close_hour:
        return "open", all_symbols, INTERVAL_MARKET_OPEN
    return "closed", always_on, INTERVAL_MARKET_CLOSED
