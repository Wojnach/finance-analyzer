"""Economic calendar signal — event proximity and sector risk-off.

Combines four sub-indicators into a majority-vote composite:
  1. event_proximity  — hours until next event; <4h risk-off, <24h cautious
  2. event_type       — classify event for informational purposes
  3. pre_event_risk   — binary risk-off within 4h of high-impact event
  4. sector_exposure  — is this ticker's sector affected by the upcoming event?

The ``context`` parameter is a dict with keys: ticker, config, macro.
Uses economic calendar dates from econ_dates.py and FOMC dates from fomc_dates.py.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from portfolio.econ_dates import next_event, events_within_hours, EVENT_SECTOR_MAP
from portfolio.news_keywords import TICKER_SECTORS
from portfolio.signal_utils import majority_vote

logger = logging.getLogger("portfolio.signals.econ_calendar")

# Max confidence cap
_MAX_CONFIDENCE = 0.7


def _get_current_date(df: pd.DataFrame) -> datetime:
    """Extract current timestamp from df or use now()."""
    if df is not None and "time" in df.columns and len(df) > 0:
        last_time = df["time"].iloc[-1]
        if isinstance(last_time, pd.Timestamp):
            return last_time.to_pydatetime().replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _event_proximity(ref_date) -> tuple[str, dict]:
    """Hours until next event → risk-off signal.

    <4h = SELL (risk-off before high-vol event)
    4-24h = cautious SELL (only for high-impact events)
    >24h = HOLD
    """
    evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
    indicators = {"next_event": None, "hours_until": None}

    if evt is None:
        return "HOLD", indicators

    indicators["next_event"] = f"{evt['type']} {evt['date'].isoformat()}"
    indicators["hours_until"] = evt["hours_until"]

    if evt["hours_until"] <= 4:
        return "SELL", indicators
    if evt["hours_until"] <= 24 and evt["impact"] == "high":
        return "SELL", indicators
    return "HOLD", indicators


def _event_type_info(ref_date) -> tuple[str, dict]:
    """Classify upcoming event type (informational, doesn't vote strongly).

    FOMC and CPI = high uncertainty → lean SELL.
    NFP = moderate → HOLD.
    GDP and others = low → HOLD.
    """
    evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
    indicators = {"event_type": None, "event_impact": None}

    if evt is None:
        return "HOLD", indicators

    indicators["event_type"] = evt["type"]
    indicators["event_impact"] = evt["impact"]

    # Only vote if event is within 48h
    if evt["hours_until"] > 48:
        return "HOLD", indicators

    if evt["type"] in ("FOMC", "CPI") and evt["hours_until"] <= 24:
        return "SELL", indicators
    return "HOLD", indicators


def _pre_event_risk(ref_date) -> tuple[str, dict]:
    """Binary risk-off within 4h of any high-impact event.

    This is the strongest sub-signal: if ANY high-impact event is within
    4 hours, vote SELL unconditionally.
    """
    nearby = events_within_hours(4, ref_date.date() if isinstance(ref_date, datetime) else ref_date)
    high_impact = [e for e in nearby if e["impact"] == "high"]

    indicators = {"events_within_4h": len(nearby),
                  "high_impact_within_4h": len(high_impact)}

    if high_impact:
        indicators["nearest_event"] = f"{high_impact[0]['type']} in {high_impact[0]['hours_until']}h"
        return "SELL", indicators
    return "HOLD", indicators


def _sector_exposure(ref_date, ticker: str) -> tuple[str, dict]:
    """Check if this ticker's sector is affected by upcoming events.

    Maps event types to affected sectors via EVENT_SECTOR_MAP,
    then checks if the ticker belongs to any affected sector.
    """
    evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
    indicators = {"ticker_sectors": list(TICKER_SECTORS.get(ticker, set())),
                  "event_affects_sector": False}

    if evt is None or evt["hours_until"] > 48:
        return "HOLD", indicators

    affected_sectors = EVENT_SECTOR_MAP.get(evt["type"], set())
    ticker_secs = TICKER_SECTORS.get(ticker, set())
    overlap = affected_sectors & ticker_secs

    indicators["affected_sectors"] = list(affected_sectors)
    indicators["overlap_sectors"] = list(overlap)
    indicators["event_affects_sector"] = bool(overlap)

    if overlap and evt["hours_until"] <= 24:
        return "SELL", indicators
    return "HOLD", indicators


def compute_econ_calendar_signal(df: pd.DataFrame, context: dict = None) -> dict:
    """Compute the composite economic calendar signal.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data (used to extract current timestamp).
    context : dict, optional
        Dict with keys: ticker, config, macro.

    Returns
    -------
    dict
        action, confidence, sub_signals, indicators
    """
    result = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {
            "event_proximity": "HOLD",
            "event_type": "HOLD",
            "pre_event_risk": "HOLD",
            "sector_exposure": "HOLD",
        },
        "indicators": {},
    }

    ticker = ""
    if context:
        ticker = context.get("ticker", "")

    ref_date = _get_current_date(df)

    # Compute each sub-signal
    try:
        prox_action, prox_ind = _event_proximity(ref_date)
    except Exception:
        prox_action, prox_ind = "HOLD", {}

    try:
        type_action, type_ind = _event_type_info(ref_date)
    except Exception:
        type_action, type_ind = "HOLD", {}

    try:
        risk_action, risk_ind = _pre_event_risk(ref_date)
    except Exception:
        risk_action, risk_ind = "HOLD", {}

    try:
        sec_action, sec_ind = _sector_exposure(ref_date, ticker)
    except Exception:
        sec_action, sec_ind = "HOLD", {}

    # Populate result
    result["sub_signals"]["event_proximity"] = prox_action
    result["sub_signals"]["event_type"] = type_action
    result["sub_signals"]["pre_event_risk"] = risk_action
    result["sub_signals"]["sector_exposure"] = sec_action

    result["indicators"].update({f"proximity_{k}": v for k, v in prox_ind.items()})
    result["indicators"].update({f"type_{k}": v for k, v in type_ind.items()})
    result["indicators"].update({f"risk_{k}": v for k, v in risk_ind.items()})
    result["indicators"].update({f"exposure_{k}": v for k, v in sec_ind.items()})

    # Majority vote
    votes = [prox_action, type_action, risk_action, sec_action]
    result["action"], result["confidence"] = majority_vote(votes)

    # Cap confidence
    result["confidence"] = min(result["confidence"], _MAX_CONFIDENCE)

    return result
