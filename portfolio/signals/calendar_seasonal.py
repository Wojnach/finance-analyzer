"""Calendar-based and seasonal trading signals.

Combines eight sub-indicators into a majority-vote composite:
  1. Day-of-Week Effect       (Monday=SELL, Friday=BUY)
  2. Turnaround Tuesday       (Tuesday reversal after red Monday)
  3. Month-End Effect          (Last 3 calendar days bullish)
  4. Sell in May / Halloween   (May-Oct=SELL, Nov-Apr=BUY)
  5. January Effect            (Jan=BUY, Dec=SELL)
  6. Pre-Holiday Effect        (Day before multi-day gap=BUY)
  7. FOMC Drift                (2 days before FOMC=BUY, day of/after=HOLD)
  8. Santa Claus Rally         (Last 5 trading days Dec + first 2 Jan=BUY)

Each sub-indicator votes BUY / SELL / HOLD.  The composite action is the
majority vote; confidence is the fraction of non-HOLD votes that agree with
the majority direction.  Maximum confidence is capped at 0.6 because
calendar signals are inherently weak.
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# FOMC meeting dates — imported from shared constant
# ---------------------------------------------------------------------------
from portfolio.fomc_dates import (
    FOMC_DATES_2026 as _FOMC_DATES_2026,
    FOMC_ANNOUNCEMENT_DATES as _FOMC_ANNOUNCEMENT_DATES,
)

# Maximum confidence for any calendar signal
_MAX_CONFIDENCE = 0.6

# Minimum rows needed for Turnaround Tuesday check
_MIN_BARS = 2


# ---------------------------------------------------------------------------
# Sub-signal functions
# ---------------------------------------------------------------------------

def _day_of_week_effect(last_date: date) -> tuple[str, dict]:
    """Monday historically bearish for equities (SELL), Friday bullish (BUY).

    Parameters
    ----------
    last_date : date
        Date of the last bar.

    Returns
    -------
    tuple[str, dict]
        Vote and indicators.
    """
    dow = last_date.weekday()  # 0=Mon, 4=Fri
    indicators = {"day_of_week": dow, "day_name": last_date.strftime("%A")}

    if dow == 0:  # Monday
        return "SELL", indicators
    if dow == 4:  # Friday
        return "BUY", indicators
    return "HOLD", indicators


def _turnaround_tuesday(df: pd.DataFrame, last_date: date) -> tuple[str, dict]:
    """Tuesday reversal after a red Monday.

    If the last bar is Tuesday AND the prior bar closed below its open
    (red candle on Monday), signal BUY.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with at least 2 rows.
    last_date : date
        Date of the last bar.

    Returns
    -------
    tuple[str, dict]
        Vote and indicators.
    """
    indicators = {"is_tuesday": False, "prior_bar_red": False}

    if last_date.weekday() != 1:  # Not Tuesday
        return "HOLD", indicators

    indicators["is_tuesday"] = True

    if len(df) < 2:
        return "HOLD", indicators

    prior_close = float(df["close"].iloc[-2])
    prior_open = float(df["open"].iloc[-2])
    prior_red = prior_close < prior_open
    indicators["prior_bar_red"] = prior_red

    if prior_red:
        return "BUY", indicators
    return "HOLD", indicators


def _month_end_effect(last_date: date) -> tuple[str, dict]:
    """Last 3 calendar days of the month tend to be bullish.

    Parameters
    ----------
    last_date : date
        Date of the last bar.

    Returns
    -------
    tuple[str, dict]
        Vote and indicators.
    """
    days_in_month = calendar.monthrange(last_date.year, last_date.month)[1]
    days_remaining = days_in_month - last_date.day
    is_month_end = days_remaining < 3  # last 3 calendar days (day 29, 30, 31 of a 31-day month)
    indicators = {
        "is_month_end": is_month_end,
        "days_remaining_in_month": days_remaining,
    }

    if is_month_end:
        return "BUY", indicators
    return "HOLD", indicators


def _sell_in_may(last_date: date) -> tuple[str, dict]:
    """Sell in May and go away / Halloween indicator.

    May through October = historically weaker (SELL bias).
    Only historically *strong* months vote BUY: Nov, Dec, Jan, Apr.
    Transitional months (Feb, Mar) = HOLD — they are not consistently
    strong enough to justify a standing BUY vote.

    Parameters
    ----------
    last_date : date
        Date of the last bar.

    Returns
    -------
    tuple[str, dict]
        Vote and indicators.
    """
    month = last_date.month
    is_weak_period = 5 <= month <= 10
    is_strong_month = month in (1, 4, 11, 12)  # Jan, Apr, Nov, Dec
    indicators = {"month": month, "is_weak_period": is_weak_period}

    if is_weak_period:
        return "SELL", indicators
    if is_strong_month:
        return "BUY", indicators
    return "HOLD", indicators


def _january_effect(last_date: date) -> tuple[str, dict]:
    """January historically bullish for small caps; December tax-loss selling.

    Parameters
    ----------
    last_date : date
        Date of the last bar.

    Returns
    -------
    tuple[str, dict]
        Vote and indicators.
    """
    month = last_date.month
    indicators = {"month": month}

    if month == 1:
        return "BUY", indicators
    if month == 12:
        return "SELL", indicators
    return "HOLD", indicators


def _pre_holiday_effect(last_date: date) -> tuple[str, dict]:
    """Trading day before a market holiday tends to be bullish.

    Checks for days preceding US market holidays (approximate).
    Regular Fridays are NOT counted here — that is already handled by
    ``_day_of_week_effect``.  Only true pre-holiday sessions (the
    trading day before a market closure beyond the normal weekend) vote.

    Parameters
    ----------
    last_date : date
        Date of the last bar.

    Returns
    -------
    tuple[str, dict]
        Vote and indicators.
    """
    # Major US market holidays (month, day) — approximate, does not handle
    # observed-date shifts (e.g., July 4 on Saturday → Friday off).
    _US_HOLIDAYS = [
        (1, 1),    # New Year's Day
        (1, 20),   # MLK Day (approx — 3rd Monday)
        (2, 17),   # Presidents' Day (approx — 3rd Monday)
        (5, 26),   # Memorial Day (approx — last Monday)
        (6, 19),   # Juneteenth
        (7, 4),    # Independence Day
        (9, 1),    # Labor Day (approx — 1st Monday)
        (11, 27),  # Thanksgiving (approx — 4th Thursday)
        (12, 25),  # Christmas
    ]

    next_day = last_date + timedelta(days=1)
    is_pre_holiday = (next_day.month, next_day.day) in _US_HOLIDAYS
    indicators = {"is_pre_holiday": is_pre_holiday, "day_of_week": last_date.weekday()}

    if is_pre_holiday:
        return "BUY", indicators
    return "HOLD", indicators


def _fomc_drift(last_date: date) -> tuple[str, dict]:
    """Pre-FOMC announcement drift.

    The 24 hours before a scheduled FOMC announcement tend to drift upward.
    BUY if within 2 days before an announcement date.
    HOLD on the day of the announcement or the day after (volatility).

    Parameters
    ----------
    last_date : date
        Date of the last bar.

    Returns
    -------
    tuple[str, dict]
        Vote and indicators.
    """
    indicators = {
        "is_pre_fomc": False,
        "is_fomc_day": False,
        "is_post_fomc": False,
        "days_to_fomc": None,
    }

    for fomc_date in _FOMC_ANNOUNCEMENT_DATES:
        delta = (fomc_date - last_date).days

        if delta == 0:
            # Day of FOMC announcement
            indicators["is_fomc_day"] = True
            indicators["days_to_fomc"] = 0
            return "HOLD", indicators

        if delta == -1:
            # Day after FOMC announcement
            indicators["is_post_fomc"] = True
            indicators["days_to_fomc"] = -1
            return "HOLD", indicators

        if 1 <= delta <= 2:
            # 1-2 days before FOMC announcement
            indicators["is_pre_fomc"] = True
            indicators["days_to_fomc"] = delta
            return "BUY", indicators

    return "HOLD", indicators


def _santa_claus_rally(last_date: date) -> tuple[str, dict]:
    """Santa Claus Rally: last 5 trading days of Dec + first 2 of Jan.

    Approximate by checking calendar dates: Dec 25-31 and Jan 1-3.
    The actual trading days vary by year, but this is a reasonable proxy.

    Parameters
    ----------
    last_date : date
        Date of the last bar.

    Returns
    -------
    tuple[str, dict]
        Vote and indicators.
    """
    month = last_date.month
    day = last_date.day

    # Last 5 trading days of Dec: approx Dec 24-31 (some are holidays/weekends)
    is_late_dec = month == 12 and day >= 24
    # First 2 trading days of Jan: approx Jan 1-3
    is_early_jan = month == 1 and day <= 3

    is_santa_rally = is_late_dec or is_early_jan
    indicators = {"is_santa_rally": is_santa_rally, "month": month, "day": day}

    if is_santa_rally:
        return "BUY", indicators
    return "HOLD", indicators


# ---------------------------------------------------------------------------
# Composite signal
# ---------------------------------------------------------------------------

def compute_calendar_signal(df: pd.DataFrame) -> dict:
    """Compute the composite calendar/seasonal signal.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV candle data with columns ``open``, ``high``, ``low``,
        ``close``, ``volume``, ``time``.  The ``time`` column is used
        for date extraction.  At least 2 rows required for Turnaround
        Tuesday check.

    Returns
    -------
    dict
        ``action`` (BUY / SELL / HOLD), ``confidence`` (0.0-0.6),
        ``sub_signals`` dict, and ``indicators`` dict.
    """
    result: dict = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {
            "day_of_week": "HOLD",
            "turnaround_tuesday": "HOLD",
            "month_end": "HOLD",
            "sell_in_may": "HOLD",
            "january_effect": "HOLD",
            "pre_holiday": "HOLD",
            "fomc_drift": "HOLD",
            "santa_claus_rally": "HOLD",
        },
        "indicators": {
            "day_of_week": None,
            "day_name": None,
            "is_tuesday": False,
            "prior_bar_red": False,
            "is_month_end": False,
            "days_remaining_in_month": None,
            "month": None,
            "is_weak_period": None,
            "is_pre_holiday": False,
            "is_pre_fomc": False,
            "is_fomc_day": False,
            "is_post_fomc": False,
            "days_to_fomc": None,
            "is_santa_rally": False,
        },
    }

    # ---- Validate input ----
    if df is None or not isinstance(df, pd.DataFrame):
        return result

    required_cols = {"open", "high", "low", "close", "volume", "time"}
    if not required_cols.issubset(set(df.columns)):
        return result

    if len(df) < _MIN_BARS:
        return result

    # ---- Extract date from last bar ----
    try:
        time_col = df["time"]
        if not pd.api.types.is_datetime64_any_dtype(time_col):
            time_col = pd.to_datetime(time_col)
        last_time = time_col.iloc[-1]
        if isinstance(last_time, pd.Timestamp):
            last_date = last_time.date()
        elif isinstance(last_time, datetime):
            last_date = last_time.date()
        elif isinstance(last_time, date):
            last_date = last_time
        else:
            last_date = pd.Timestamp(last_time).date()
    except Exception:
        return result

    # ---- Compute each sub-signal ----
    try:
        dow_action, dow_ind = _day_of_week_effect(last_date)
    except Exception:
        dow_action, dow_ind = "HOLD", {}

    try:
        tt_action, tt_ind = _turnaround_tuesday(df, last_date)
    except Exception:
        tt_action, tt_ind = "HOLD", {}

    try:
        me_action, me_ind = _month_end_effect(last_date)
    except Exception:
        me_action, me_ind = "HOLD", {}

    try:
        sim_action, sim_ind = _sell_in_may(last_date)
    except Exception:
        sim_action, sim_ind = "HOLD", {}

    try:
        jan_action, jan_ind = _january_effect(last_date)
    except Exception:
        jan_action, jan_ind = "HOLD", {}

    try:
        ph_action, ph_ind = _pre_holiday_effect(last_date)
    except Exception:
        ph_action, ph_ind = "HOLD", {}

    try:
        fomc_action, fomc_ind = _fomc_drift(last_date)
    except Exception:
        fomc_action, fomc_ind = "HOLD", {}

    try:
        santa_action, santa_ind = _santa_claus_rally(last_date)
    except Exception:
        santa_action, santa_ind = "HOLD", {}

    # ---- Populate sub-signals and indicators ----
    result["sub_signals"]["day_of_week"] = dow_action
    result["sub_signals"]["turnaround_tuesday"] = tt_action
    result["sub_signals"]["month_end"] = me_action
    result["sub_signals"]["sell_in_may"] = sim_action
    result["sub_signals"]["january_effect"] = jan_action
    result["sub_signals"]["pre_holiday"] = ph_action
    result["sub_signals"]["fomc_drift"] = fomc_action
    result["sub_signals"]["santa_claus_rally"] = santa_action

    result["indicators"].update(dow_ind)
    result["indicators"].update(tt_ind)
    result["indicators"].update(me_ind)
    result["indicators"].update(sim_ind)
    result["indicators"].update(jan_ind)
    result["indicators"].update(ph_ind)
    result["indicators"].update(fomc_ind)
    result["indicators"].update(santa_ind)

    # ---- Majority vote ----
    votes = [
        dow_action, tt_action, me_action, sim_action,
        jan_action, ph_action, fomc_action, santa_action,
    ]
    buy_count = votes.count("BUY")
    sell_count = votes.count("SELL")
    active_votes = buy_count + sell_count  # non-HOLD votes

    if active_votes == 0:
        # All sub-signals abstain
        result["action"] = "HOLD"
        result["confidence"] = 0.0
    elif buy_count > sell_count:
        result["action"] = "BUY"
        result["confidence"] = min(round(buy_count / active_votes, 2), _MAX_CONFIDENCE)
    elif sell_count > buy_count:
        result["action"] = "SELL"
        result["confidence"] = min(round(sell_count / active_votes, 2), _MAX_CONFIDENCE)
    else:
        # Tied between BUY and SELL -- no clear direction
        result["action"] = "HOLD"
        result["confidence"] = 0.0

    return result
