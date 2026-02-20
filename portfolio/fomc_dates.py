"""Shared FOMC meeting date constants.

Single source of truth for all modules that need FOMC dates.
Each two-day meeting is listed as (start_date, announcement_date).
The announcement (rate decision) is always on day 2.

Sources: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
"""

from datetime import date

# 2026 FOMC meeting dates (two-day meetings)
FOMC_DATES_2026 = [
    date(2026, 1, 28), date(2026, 1, 29),
    date(2026, 3, 17), date(2026, 3, 18),
    date(2026, 4, 28), date(2026, 4, 29),
    date(2026, 6, 16), date(2026, 6, 17),
    date(2026, 7, 28), date(2026, 7, 29),
    date(2026, 9, 15), date(2026, 9, 16),
    date(2026, 10, 27), date(2026, 10, 28),
    date(2026, 12, 8), date(2026, 12, 9),
]

# 2027 FOMC meeting dates (two-day meetings)
FOMC_DATES_2027 = [
    date(2027, 1, 26), date(2027, 1, 27),
    date(2027, 3, 16), date(2027, 3, 17),
    date(2027, 4, 27), date(2027, 4, 28),
    date(2027, 6, 8), date(2027, 6, 9),
    date(2027, 7, 27), date(2027, 7, 28),
    date(2027, 9, 14), date(2027, 9, 15),
    date(2027, 10, 26), date(2027, 10, 27),
    date(2027, 12, 7), date(2027, 12, 8),
]

# Combined list as ISO strings (for macro_context.py which uses string comparison)
FOMC_DATES_ISO = [d.isoformat() for d in FOMC_DATES_2026 + FOMC_DATES_2027]

# Announcement dates only (day 2 of each meeting — the rate decision day)
FOMC_ANNOUNCEMENT_DATES = [
    date(2026, 1, 29),
    date(2026, 3, 18),
    date(2026, 4, 29),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 9),
    date(2027, 1, 27),
    date(2027, 3, 17),
    date(2027, 4, 28),
    date(2027, 6, 9),
    date(2027, 7, 28),
    date(2027, 9, 15),
    date(2027, 10, 27),
    date(2027, 12, 8),
]
