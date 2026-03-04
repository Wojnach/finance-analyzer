from datetime import datetime, timezone

from portfolio.focus_analysis import (
    estimate_near_close_range,
    hours_to_us_close,
    normalize_ticker,
)


def test_normalize_ticker_aliases():
    assert normalize_ticker("btc") == "BTC-USD"
    assert normalize_ticker("eth") == "ETH-USD"
    assert normalize_ticker("gold") == "XAU-USD"
    assert normalize_ticker("silver") == "XAG-USD"
    assert normalize_ticker("microstrategy") == "MSTR"


def test_hours_to_us_close_weekend_is_zero():
    # Saturday
    now = datetime(2026, 3, 7, 12, 0, tzinfo=timezone.utc)
    assert hours_to_us_close(now) == 0.0


def test_hours_to_us_close_open_window_positive():
    # Wednesday before US close during EST (close ~= 21:00 UTC in winter)
    now = datetime(2026, 1, 7, 18, 30, tzinfo=timezone.utc)
    h = hours_to_us_close(now)
    assert h > 0
    assert h < 4


def test_estimate_near_close_range_has_valid_bounds():
    low, high, move = estimate_near_close_range(
        price=100.0,
        atr_pct=2.0,
        prob_3h={"probability": 0.6},
        forecast_1h_pct=0.4,
        hours_left=2.5,
    )
    assert low is not None and high is not None
    assert low < high
    assert move > 0


def test_estimate_near_close_range_invalid_price():
    low, high, move = estimate_near_close_range(
        price=0,
        atr_pct=2.0,
        prob_3h={"probability": 0.6},
        forecast_1h_pct=0.4,
        hours_left=2.5,
    )
    assert low is None and high is None
    assert move == 0.0
