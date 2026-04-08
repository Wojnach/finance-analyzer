"""Economic calendar dates for 2026-2027.

Follows the same pattern as fomc_dates.py — hard-coded dates as a single
source of truth.  Used by the econ_calendar signal to gauge event proximity.

Sources:
- CPI: Bureau of Labor Statistics release schedule
- NFP: Bureau of Labor Statistics (first Friday of each month)
- GDP: Bureau of Economic Analysis advance estimate schedule
- FOMC: imported from fomc_dates.py (not duplicated here)
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from portfolio.fomc_dates import FOMC_ANNOUNCEMENT_DATES

# ---------------------------------------------------------------------------
# CPI release dates 2026 (8:30 AM ET, typically 2nd or 3rd week)
# ---------------------------------------------------------------------------

CPI_DATES_2026 = [
    date(2026, 1, 14),   # Dec 2025 CPI
    date(2026, 2, 12),   # Jan 2026 CPI
    date(2026, 3, 11),   # Feb 2026 CPI
    date(2026, 4, 10),   # Mar 2026 CPI
    date(2026, 5, 13),   # Apr 2026 CPI
    date(2026, 6, 10),   # May 2026 CPI
    date(2026, 7, 14),   # Jun 2026 CPI
    date(2026, 8, 12),   # Jul 2026 CPI
    date(2026, 9, 11),   # Aug 2026 CPI
    date(2026, 10, 13),  # Sep 2026 CPI
    date(2026, 11, 12),  # Oct 2026 CPI
    date(2026, 12, 10),  # Nov 2026 CPI
]

CPI_DATES_2027 = [
    date(2027, 1, 13),
    date(2027, 2, 10),
    date(2027, 3, 10),
    date(2027, 4, 13),
    date(2027, 5, 12),
    date(2027, 6, 10),
    date(2027, 7, 14),
    date(2027, 8, 11),
    date(2027, 9, 15),
    date(2027, 10, 13),
    date(2027, 11, 10),
    date(2027, 12, 10),
]

# ---------------------------------------------------------------------------
# Non-Farm Payrolls (first Friday of each month, 8:30 AM ET)
# ---------------------------------------------------------------------------

NFP_DATES_2026 = [
    date(2026, 1, 2),
    date(2026, 2, 6),
    date(2026, 3, 6),
    date(2026, 4, 2),   # BLS released April 2026 NFP on Apr 2 (Apr 3 = Good Friday, market closed)
    date(2026, 5, 1),
    date(2026, 6, 5),
    date(2026, 7, 2),
    date(2026, 8, 7),
    date(2026, 9, 4),
    date(2026, 10, 2),
    date(2026, 11, 6),
    date(2026, 12, 4),
]

NFP_DATES_2027 = [
    date(2027, 1, 8),
    date(2027, 2, 5),
    date(2027, 3, 5),
    date(2027, 4, 2),
    date(2027, 5, 7),
    date(2027, 6, 4),
    date(2027, 7, 2),
    date(2027, 8, 6),
    date(2027, 9, 3),
    date(2027, 10, 1),
    date(2027, 11, 5),
    date(2027, 12, 3),
]

# ---------------------------------------------------------------------------
# GDP advance estimate dates (quarterly, ~4th week of month after quarter end)
# ---------------------------------------------------------------------------

GDP_DATES_2026 = [
    date(2026, 1, 29),   # Q4 2025 advance
    date(2026, 4, 29),   # Q1 2026 advance
    date(2026, 7, 30),   # Q2 2026 advance
    date(2026, 10, 29),  # Q3 2026 advance
]

GDP_DATES_2027 = [
    date(2027, 1, 28),   # Q4 2026 advance
    date(2027, 4, 29),   # Q1 2027 advance
    date(2027, 7, 29),   # Q2 2027 advance
    date(2027, 10, 28),  # Q3 2027 advance
]

# ---------------------------------------------------------------------------
# Unified event list
# ---------------------------------------------------------------------------

# Impact levels: "high" (FOMC, CPI, NFP) or "medium" (GDP)
# Affected sectors: which sectors are most impacted

EVENT_SECTOR_MAP = {
    "FOMC": {"crypto", "metals", "big_tech", "etf"},
    "CPI": {"crypto", "metals", "big_tech", "etf"},
    "NFP": {"etf", "big_tech"},
    "GDP": {"etf", "big_tech"},
}


def _build_events() -> list[dict]:
    """Build sorted list of all economic events."""
    events = []

    for d in FOMC_ANNOUNCEMENT_DATES:
        events.append({"date": d, "type": "FOMC", "impact": "high"})

    for d in CPI_DATES_2026 + CPI_DATES_2027:
        events.append({"date": d, "type": "CPI", "impact": "high"})

    for d in NFP_DATES_2026 + NFP_DATES_2027:
        events.append({"date": d, "type": "NFP", "impact": "high"})

    for d in GDP_DATES_2026 + GDP_DATES_2027:
        events.append({"date": d, "type": "GDP", "impact": "medium"})

    events.sort(key=lambda e: e["date"])
    return events


ECON_EVENTS = _build_events()


def next_event(ref_date: date | None = None) -> dict | None:
    """Return the next economic event on or after ref_date.

    Returns dict with keys: date, type, impact, hours_until.
    Returns None if no future events in the calendar.
    """
    if ref_date is None:
        ref_date = datetime.now(UTC).date()

    for evt in ECON_EVENTS:
        if evt["date"] >= ref_date:
            # Calculate hours until (approximate: assume 14:00 UTC release)
            evt_dt = datetime.combine(evt["date"], datetime.min.time().replace(hour=14),
                                      tzinfo=UTC)
            # M5: Use datetime.now(UTC) so past-today events show hours_until=0.
            now = datetime.now(UTC)
            delta = evt_dt - now
            hours = max(0.0, delta.total_seconds() / 3600)
            return {
                "date": evt["date"],
                "type": evt["type"],
                "impact": evt["impact"],
                "hours_until": round(hours, 1),
            }
    return None


def events_within_hours(hours: float, ref_date: date | None = None) -> list[dict]:
    """Return all events within the given hours from ref_date."""
    if ref_date is None:
        ref_date = datetime.now(UTC).date()

    results = []
    now = datetime.now(UTC)
    for evt in ECON_EVENTS:
        if evt["date"] < ref_date:
            continue
        evt_dt = datetime.combine(evt["date"], datetime.min.time().replace(hour=14),
                                  tzinfo=UTC)
        delta = evt_dt - now
        hrs = delta.total_seconds() / 3600
        if 0 <= hrs <= hours:
            results.append({
                "date": evt["date"],
                "type": evt["type"],
                "impact": evt["impact"],
                "hours_until": round(hrs, 1),
            })
    return results
