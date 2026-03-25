"""Tests for portfolio.metals_ladder."""

from portfolio.metals_ladder import (
    build_intraday_ladder,
    flash_crash_drop_pct,
    map_underlying_name,
    translate_underlying_target,
)


def _signal_entry():
    return {
        "price_usd": 86.9,
        "atr_pct": 0.35,
        "regime": "ranging",
        "extra": {
            "volatility_sig_indicators": {"bb_squeeze_on": False},
        },
    }


def test_map_underlying_name():
    assert map_underlying_name("Silver") == "XAG-USD"
    assert map_underlying_name("Gold") == "XAU-USD"
    assert map_underlying_name("Guld") == "XAU-USD"
    assert map_underlying_name("Oil") is None


def test_translate_underlying_target_for_long_product():
    price = translate_underlying_target(12.7, 86.9, 86.5, 6.3)
    assert price < 12.7


def test_flash_crash_only_active_near_us_open():
    analysis = {
        "us_market_open": {
            "phase": "pre_open",
            "historical_stats": {
                "post_open_mean_pct": -0.692,
                "post_open_avg_range_pct": 3.537,
            },
        }
    }
    assert flash_crash_drop_pct(analysis) > 1.0
    assert flash_crash_drop_pct({"us_market_open": {"phase": "not_near_open"}}) == 0.0


def test_build_intraday_ladder_includes_flash_reserve_when_near_open():
    analysis = {
        "us_market_open": {
            "phase": "pre_open",
            "historical_stats": {
                "post_open_mean_pct": -0.692,
                "post_open_avg_range_pct": 3.537,
            },
        }
    }
    ladder = build_intraday_ladder(
        _signal_entry(),
        {"3h": {"probability": 0.55}},
        ticker="XAG-USD",
        current_instrument_price=12.7,
        current_underlying_price=86.9,
        leverage=6.3,
        hours_remaining=6.0,
        analysis=analysis,
    )
    assert ladder["working_price"] > 0
    assert ladder["flash_price"] > 0
    assert ladder["flash_price"] < ladder["working_price"]
    assert ladder["exit_price"] >= ladder["current_instrument"]


def test_build_intraday_ladder_disables_flash_reserve_away_from_open():
    ladder = build_intraday_ladder(
        _signal_entry(),
        {"3h": {"probability": 0.55}},
        ticker="XAG-USD",
        current_instrument_price=12.7,
        current_underlying_price=86.9,
        leverage=6.3,
        hours_remaining=6.0,
        analysis={"us_market_open": {"phase": "not_near_open"}},
    )
    assert ladder["flash_price"] == 0.0
