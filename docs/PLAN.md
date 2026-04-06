# Plan: Market Holiday Awareness

## Date: 2026-04-06

## Problem

`market_timing.py` only checks `weekday >= 5` for weekends. On public holidays
(Easter Monday, Good Friday, Christmas, etc.), the system treats them as normal
trading days:

- Main loop (`portfolio/main.py`) runs all 16 US stock tickers during "market hours"
- Metals loop (`data/metals_loop.py`) treats Avanza as open, tries to place orders
- Layer 2 agent can be invoked for stock triggers (wasting Claude CLI quota)
- GPU signals run for stocks that can't trade
- API quota wasted (Alpaca, Alpha Vantage, NewsAPI)

## Affected Functions

| Function | File | Issue |
|----------|------|-------|
| `is_us_stock_market_open()` | `portfolio/market_timing.py` | No holiday check |
| `_is_agent_window()` | `portfolio/market_timing.py` | No holiday check |
| `get_market_state()` | `portfolio/market_timing.py` | No holiday check |
| `should_skip_gpu()` | `portfolio/market_timing.py` | Delegates to above |
| `is_market_hours()` | `data/metals_loop.py` | No holiday check (Avanza) |
| `is_avanza_open()` | `data/metals_loop.py` | Delegates to above |

## Design

### 1. US Market Holidays (`is_us_market_holiday`)

NYSE has 10 observed holidays. Some are fixed-date, some are floating (nth weekday
of month). Easter (Good Friday) requires an Easter algorithm.

Approach: compute holidays for a given year dynamically. No hardcoded date lists
that go stale. Use the anonymous Gregorian Easter algorithm for Good Friday.

Holidays:
- New Year's Day (Jan 1, observed)
- MLK Day (3rd Monday Jan)
- Presidents' Day (3rd Monday Feb)
- Good Friday (Friday before Easter)
- Memorial Day (last Monday May)
- Juneteenth (Jun 19, observed)
- Independence Day (Jul 4, observed)
- Labor Day (1st Monday Sep)
- Thanksgiving (4th Thursday Nov)
- Christmas (Dec 25, observed)

"Observed" rule: if holiday falls on Saturday, observed Friday. If Sunday, observed Monday.

### 2. Swedish Market Holidays (`is_swedish_market_holiday`)

Avanza/Nasdaq Stockholm holidays (subset that also closes Avanza warrants):
- New Year's Day (Jan 1)
- Epiphany (Jan 6)
- Good Friday
- Easter Monday
- May Day (May 1)
- Ascension Day (39 days after Easter)
- National Day (Jun 6)
- Midsummer Eve (Friday before Midsummer Day, which is Sat between Jun 20-26)
- Christmas Eve (Dec 24)
- Christmas Day (Dec 25)
- Boxing Day (Dec 26)
- New Year's Eve (Dec 31)

### 3. Integration

- Add `is_us_market_holiday(dt=None)` and `is_swedish_market_holiday(dt=None)` to
  `portfolio/market_timing.py`
- Wire into `is_us_stock_market_open()`: return False if holiday
- Wire into `_is_agent_window()`: return False if US holiday
- Wire into `get_market_state()`: treat US holidays like weekends for stocks
- Add `is_swedish_market_holiday` import to `data/metals_loop.py`, wire into `is_market_hours()`

### 4. What Could Break

- False positive holiday detection -> stocks skipped on normal trading day. Mitigated
  by thorough testing with known 2026 calendar dates.
- Easter algorithm bug -> wrong Good Friday/Easter Monday. Mitigated by testing
  multiple years (2024-2030).
- Metals/crypto should NOT be affected by holidays (Binance trades 24/7). Only
  Avanza warrant trading and US stock processing should be gated.

## Execution Order

1. Write tests first (TDD) -- test known 2026 holidays
2. Implement `_easter_sunday()`, `us_market_holidays()`, `swedish_market_holidays()`
3. Wire into existing functions
4. Update `metals_loop.py` `is_market_hours()`
5. Run full test suite
6. Merge and push

## Files Modified

- `portfolio/market_timing.py` -- add holiday functions, wire into existing
- `data/metals_loop.py` -- wire Swedish holidays into `is_market_hours()`
- `tests/test_market_timing.py` -- add holiday test cases
