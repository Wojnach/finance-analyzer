"""Tests for exit_optimizer, cost_model, and session_calendar modules.

Covers Monte Carlo path simulation, P&L computation, risk flags,
risk overrides, exit plan generation, cost models, and session calendar.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from portfolio.cost_model import (
    CRYPTO_COSTS,
    ELONGIR_COSTS,
    STOCK_COSTS,
    WARRANT_COSTS,
    CostModel,
    get_cost_model,
)
from portfolio.exit_optimizer import (
    CandidateExit,
    ExitPlan,
    MarketSnapshot,
    Position,
    _apply_risk_overrides,
    _compute_pnl_sek,
    _compute_risk_flags,
    _first_hit_times,
    _path_statistics,
    compute_exit_plan,
    compute_exit_plan_from_summary,
    simulate_intraday_paths,
)
from portfolio.session_calendar import (
    _cet_offset,
    _eu_dst,
    get_session_info,
    remaining_session_minutes,
)

# ---------------------------------------------------------------------------
# Shared fixtures and constants
# ---------------------------------------------------------------------------

BASE_TIME = datetime(2026, 3, 12, 14, 0, tzinfo=UTC)  # Thu 14:00 UTC
SEED = 42
N_PATHS = 1000


def _make_position(**overrides) -> Position:
    defaults = dict(
        symbol="XAG-USD",
        qty=220,
        entry_price_sek=85.0,
        entry_underlying_usd=32.0,
        entry_ts=BASE_TIME - timedelta(hours=2),
        instrument_type="warrant",
        leverage=5.0,
        financing_level=25.0,
    )
    defaults.update(overrides)
    return Position(**defaults)


def _make_market(**overrides) -> MarketSnapshot:
    defaults = dict(
        asof_ts=BASE_TIME,
        price=33.5,
        bid=33.48,
        ask=33.52,
        volatility_annual=0.25,
        atr_pct=2.5,
        usdsek=10.85,
        drift=0.0,
    )
    defaults.update(overrides)
    return MarketSnapshot(**defaults)


# =========================================================================
# COST MODEL TESTS
# =========================================================================


class TestCostModelTotalCostSek:
    """CostModel.total_cost_sek with various trade values."""

    def test_positive_trade_value(self):
        cm = CostModel(courtage_bps=10, min_fee_sek=1.0, spread_bps=5, slippage_bps=3)
        cost = cm.total_cost_sek(100_000)
        # courtage = max(100000 * 10/10000, 1.0) = max(100, 1) = 100
        # spread = 100000 * 5/10000 = 50
        # slippage = 100000 * 3/10000 = 30
        assert cost == pytest.approx(180.0)

    def test_min_fee_applies(self):
        cm = CostModel(courtage_bps=1, min_fee_sek=50.0, spread_bps=0, slippage_bps=0)
        cost = cm.total_cost_sek(1000)
        # courtage = max(1000 * 1/10000, 50) = max(0.1, 50) = 50
        assert cost == pytest.approx(50.0)

    def test_zero_trade_value(self):
        cm = CostModel(courtage_bps=10, min_fee_sek=5.0, spread_bps=5, slippage_bps=3)
        assert cm.total_cost_sek(0) == 0.0

    def test_negative_trade_value(self):
        cm = CostModel(courtage_bps=10, min_fee_sek=5.0, spread_bps=5, slippage_bps=3)
        assert cm.total_cost_sek(-500) == 0.0

    def test_small_trade_min_fee_floor(self):
        cm = CostModel(courtage_bps=6.9, min_fee_sek=1.0, spread_bps=0, slippage_bps=0)
        cost = cm.total_cost_sek(100)
        # courtage = max(100 * 6.9/10000, 1.0) = max(0.069, 1) = 1.0
        assert cost == pytest.approx(1.0)


class TestCostModelPercentages:
    """CostModel.total_cost_pct and round_trip_pct."""

    def test_total_cost_pct(self):
        cm = CostModel(courtage_bps=10, spread_bps=5, slippage_bps=3)
        # (10 + 5 + 3) / 100 = 0.18%
        assert cm.total_cost_pct() == pytest.approx(0.18)

    def test_round_trip_pct(self):
        cm = CostModel(courtage_bps=10, spread_bps=5, slippage_bps=3)
        assert cm.round_trip_pct() == pytest.approx(0.36)

    def test_zero_cost_pct(self):
        cm = CostModel()
        assert cm.total_cost_pct() == 0.0
        assert cm.round_trip_pct() == 0.0


class TestGetCostModel:
    """get_cost_model for known and unknown instrument types."""

    def test_warrant(self):
        assert get_cost_model("warrant") is WARRANT_COSTS

    def test_stock(self):
        assert get_cost_model("stock") is STOCK_COSTS

    def test_crypto(self):
        assert get_cost_model("crypto") is CRYPTO_COSTS

    def test_elongir(self):
        assert get_cost_model("elongir") is ELONGIR_COSTS

    def test_unknown_falls_back_to_stock(self):
        assert get_cost_model("magic_beans") is STOCK_COSTS

    def test_empty_string_falls_back(self):
        assert get_cost_model("") is STOCK_COSTS


class TestPresetCostModels:
    """Verify preset cost model field values."""

    def test_warrant_costs(self):
        assert WARRANT_COSTS.courtage_bps == 0.0
        assert WARRANT_COSTS.min_fee_sek == 0.0
        assert WARRANT_COSTS.spread_bps == 40.0
        assert WARRANT_COSTS.slippage_bps == 10.0
        assert WARRANT_COSTS.label == "avanza_warrant"

    def test_stock_costs(self):
        assert STOCK_COSTS.courtage_bps == 6.9
        assert STOCK_COSTS.min_fee_sek == 1.0
        assert STOCK_COSTS.spread_bps == 5.0
        assert STOCK_COSTS.slippage_bps == 2.0
        assert STOCK_COSTS.label == "avanza_stock"

    def test_crypto_costs(self):
        assert CRYPTO_COSTS.courtage_bps == 5.0
        assert CRYPTO_COSTS.min_fee_sek == 0.0
        assert CRYPTO_COSTS.spread_bps == 5.0
        assert CRYPTO_COSTS.slippage_bps == 5.0
        assert CRYPTO_COSTS.label == "crypto"

    def test_elongir_costs(self):
        assert ELONGIR_COSTS.courtage_bps == 25.0
        assert ELONGIR_COSTS.min_fee_sek == 0.0
        assert ELONGIR_COSTS.spread_bps == 40.0
        assert ELONGIR_COSTS.slippage_bps == 10.0
        assert ELONGIR_COSTS.label == "elongir_silver"


# =========================================================================
# SESSION CALENDAR TESTS
# =========================================================================


class TestEuDst:
    """EU DST detection for summer and winter."""

    def test_winter_january(self):
        dt = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
        assert _eu_dst(dt) is False

    def test_summer_july(self):
        dt = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
        assert _eu_dst(dt) is True

    def test_march_12_is_winter(self):
        # 2026-03-12 is before last Sunday of March (Mar 29)
        assert _eu_dst(BASE_TIME) is False

    def test_transition_day_march_29_2026(self):
        # Last Sunday of March 2026 is the 29th. DST starts at 01:00 UTC.
        before = datetime(2026, 3, 29, 0, 59, tzinfo=UTC)
        after = datetime(2026, 3, 29, 1, 0, tzinfo=UTC)
        assert _eu_dst(before) is False
        assert _eu_dst(after) is True

    def test_october_end_transition(self):
        # Last Sunday of Oct 2026 is the 25th. DST ends at 01:00 UTC.
        before = datetime(2026, 10, 25, 0, 59, tzinfo=UTC)
        after = datetime(2026, 10, 25, 1, 0, tzinfo=UTC)
        assert _eu_dst(before) is True
        assert _eu_dst(after) is False


class TestCetOffset:
    """CET offset correctness."""

    def test_winter_offset_is_1(self):
        dt = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
        assert _cet_offset(dt) == 1

    def test_summer_offset_is_2(self):
        dt = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
        assert _cet_offset(dt) == 2


class TestGetSessionInfoWarrant:
    """get_session_info for warrant instrument type."""

    def test_warrant_open_during_session(self):
        # Thu 14:00 UTC = 15:00 CET (winter). Warrant open 07:15-20:55 UTC.
        info = get_session_info("warrant", now=BASE_TIME)
        assert info.is_open is True
        assert info.phase == "open"

    def test_warrant_remaining_minutes(self):
        # Session end: 21:55 CET = 20:55 UTC. Now = 14:00 UTC.
        # Remaining = 6h55m = 415 min.
        info = get_session_info("warrant", now=BASE_TIME)
        assert info.remaining_minutes == pytest.approx(415.0, abs=1)

    def test_warrant_closed_at_night(self):
        night = datetime(2026, 3, 12, 23, 0, tzinfo=UTC)
        info = get_session_info("warrant", now=night)
        assert info.is_open is False
        assert info.remaining_minutes == 0
        assert info.phase == "closed"

    def test_warrant_underlying_usd_metals_always_open(self):
        info = get_session_info("warrant", underlying="XAG-USD", now=BASE_TIME)
        assert info.underlying_open is True

    def test_warrant_underlying_us_stock_before_open(self):
        # Use January (no US DST). US open 14:30 UTC (EST).
        winter = datetime(2026, 1, 15, 14, 0, tzinfo=UTC)
        info = get_session_info("warrant", underlying="TSM", now=winter)
        assert info.underlying_open is False

    def test_warrant_underlying_us_stock_during_us_hours(self):
        us_open = datetime(2026, 3, 12, 16, 0, tzinfo=UTC)
        info = get_session_info("warrant", underlying="TSM", now=us_open)
        assert info.underlying_open is True


class TestGetSessionInfoStockSe:
    """get_session_info for stock_se (Stockholm)."""

    def test_open_during_session(self):
        # Thu 14:00 UTC = 15:00 CET. Stockholm open 08:00-16:25 UTC (winter).
        info = get_session_info("stock_se", now=BASE_TIME)
        assert info.is_open is True

    def test_closed_after_hours(self):
        evening = datetime(2026, 3, 12, 18, 0, tzinfo=UTC)
        info = get_session_info("stock_se", now=evening)
        assert info.is_open is False


class TestGetSessionInfoStockUs:
    """get_session_info for stock_us."""

    def test_us_open_during_session(self):
        us_mid = datetime(2026, 3, 12, 17, 0, tzinfo=UTC)
        info = get_session_info("stock_us", now=us_mid)
        assert info.is_open is True
        assert info.phase == "open"

    def test_us_closed_before_open(self):
        # January: no US DST. US opens 14:30 UTC (EST). 14:00 is before open.
        winter = datetime(2026, 1, 15, 14, 0, tzinfo=UTC)
        info = get_session_info("stock_us", now=winter)
        assert info.is_open is False

    def test_us_remaining_minutes(self):
        us_mid = datetime(2026, 3, 12, 17, 0, tzinfo=UTC)
        info = get_session_info("stock_us", now=us_mid)
        # Mar 12 is US DST (EDT). Close at 20:00 UTC. 20:00 - 17:00 = 180 min.
        assert info.remaining_minutes == pytest.approx(180.0, abs=1)


class TestGetSessionInfoCrypto:
    """Crypto always-open behavior."""

    def test_crypto_always_open(self):
        info = get_session_info("crypto", now=BASE_TIME)
        assert info.is_open is True
        assert info.phase == "open"
        assert info.underlying_open is True

    def test_crypto_open_on_weekend(self):
        saturday = datetime(2026, 3, 14, 12, 0, tzinfo=UTC)
        info = get_session_info("crypto", now=saturday)
        assert info.is_open is True

    def test_crypto_remaining_minutes_positive(self):
        info = get_session_info("crypto", now=BASE_TIME)
        assert info.remaining_minutes > 0

    def test_crypto_session_end_is_next_midnight(self):
        info = get_session_info("crypto", now=BASE_TIME)
        assert info.session_end.hour == 0
        assert info.session_end.minute == 0


class TestSessionWeekend:
    """Weekend closed behavior for stocks."""

    def test_stock_se_closed_on_saturday(self):
        saturday = datetime(2026, 3, 14, 10, 0, tzinfo=UTC)
        info = get_session_info("stock_se", now=saturday)
        assert info.is_open is False

    def test_stock_us_closed_on_sunday(self):
        sunday = datetime(2026, 3, 15, 17, 0, tzinfo=UTC)
        info = get_session_info("stock_us", now=sunday)
        assert info.is_open is False

    def test_warrant_closed_on_weekend(self):
        saturday = datetime(2026, 3, 14, 12, 0, tzinfo=UTC)
        info = get_session_info("warrant", now=saturday)
        assert info.is_open is False


class TestRemainingSessionMinutes:
    """remaining_session_minutes shortcut."""

    def test_shortcut_matches_get_session_info(self):
        info = get_session_info("warrant", now=BASE_TIME)
        shortcut = remaining_session_minutes("warrant", now=BASE_TIME)
        assert shortcut == info.remaining_minutes


# =========================================================================
# EXIT OPTIMIZER TESTS
# =========================================================================


class TestSimulateIntradayPaths:
    """simulate_intraday_paths shape, antithetic, reproducibility."""

    def test_shape_correct(self):
        paths = simulate_intraday_paths(100.0, 0.25, 0.0, 120, n_paths=N_PATHS, seed=SEED)
        assert paths.shape == (N_PATHS, 121)

    def test_first_column_is_price(self):
        price = 33.5
        paths = simulate_intraday_paths(price, 0.25, 0.0, 60, n_paths=N_PATHS, seed=SEED)
        np.testing.assert_allclose(paths[:, 0], price)

    def test_antithetic_variates_mirror_log_returns(self):
        n = 200  # Must be even
        paths = simulate_intraday_paths(100.0, 0.25, 0.0, 60, n_paths=n, seed=SEED)
        n_half = n // 2
        log_returns_first = np.log(paths[:n_half, 1:] / paths[:n_half, :-1])
        log_returns_second = np.log(paths[n_half:, 1:] / paths[n_half:, :-1])
        # Antithetic: each pair sums to 2 * drift_per_step (a constant).
        # All rows of sums should be identical.
        sums = log_returns_first + log_returns_second
        # Check that all rows are equal to the first row (constant per step)
        for i in range(1, n_half):
            np.testing.assert_allclose(sums[i], sums[0], atol=1e-10)

    def test_reproducibility_with_seed(self):
        p1 = simulate_intraday_paths(100.0, 0.25, 0.0, 60, n_paths=N_PATHS, seed=SEED)
        p2 = simulate_intraday_paths(100.0, 0.25, 0.0, 60, n_paths=N_PATHS, seed=SEED)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self):
        p1 = simulate_intraday_paths(100.0, 0.25, 0.0, 60, n_paths=N_PATHS, seed=1)
        p2 = simulate_intraday_paths(100.0, 0.25, 0.0, 60, n_paths=N_PATHS, seed=2)
        assert not np.array_equal(p1, p2)

    def test_odd_n_paths(self):
        paths = simulate_intraday_paths(100.0, 0.25, 0.0, 30, n_paths=101, seed=SEED)
        assert paths.shape[0] == 101

    def test_volatility_floor_applied(self):
        # Very low vol should be floored to _MIN_VOLATILITY (0.05)
        paths = simulate_intraday_paths(100.0, 0.001, 0.0, 60, n_paths=N_PATHS, seed=SEED)
        # Should still have some variation (floor kicks in)
        assert paths[:, -1].std() > 0

    def test_remaining_minutes_zero_clamps(self):
        paths = simulate_intraday_paths(100.0, 0.25, 0.0, 0, n_paths=10, seed=SEED)
        # n_steps = max(1, 0) = 1, so shape = (10, 2)
        assert paths.shape == (10, 2)


class TestPathStatistics:
    """_path_statistics return keys and quantile ordering."""

    def test_session_max_gte_price(self):
        price = 100.0
        paths = simulate_intraday_paths(price, 0.25, 0.0, 120, n_paths=N_PATHS, seed=SEED)
        stats = _path_statistics(paths)
        # session_max should be >= initial price for most paths
        # (could be slightly below due to immediate drop, but median should be above)
        assert np.median(stats["session_max"]) >= price

    def test_quantile_ordering_max(self):
        paths = simulate_intraday_paths(100.0, 0.25, 0.0, 120, n_paths=N_PATHS, seed=SEED)
        stats = _path_statistics(paths)
        q = stats["max_quantiles"]
        assert q["p5"] <= q["p50"] <= q["p95"]

    def test_quantile_ordering_min(self):
        paths = simulate_intraday_paths(100.0, 0.25, 0.0, 120, n_paths=N_PATHS, seed=SEED)
        stats = _path_statistics(paths)
        q = stats["min_quantiles"]
        assert q["p5"] <= q["p50"] <= q["p95"]

    def test_return_keys_present(self):
        paths = simulate_intraday_paths(100.0, 0.25, 0.0, 60, n_paths=100, seed=SEED)
        stats = _path_statistics(paths)
        assert "session_max" in stats
        assert "session_min" in stats
        assert "terminal" in stats
        assert "max_quantiles" in stats
        assert "min_quantiles" in stats


class TestFirstHitTimes:
    """_first_hit_times direction and never-hit behavior."""

    def test_never_hit_returns_minus_one(self):
        # Paths that never reach an absurdly high target
        paths = simulate_intraday_paths(100.0, 0.05, 0.0, 30, n_paths=N_PATHS, seed=SEED)
        hit_times = _first_hit_times(paths, 999.0, direction="above")
        # Should be mostly -1 (some extreme paths might hit, but 999 is impossible)
        assert (hit_times == -1).sum() == N_PATHS

    def test_certain_hit_returns_positive(self):
        # Trivially hittable: target below starting price with direction "below"
        paths = simulate_intraday_paths(100.0, 0.25, 0.0, 120, n_paths=N_PATHS, seed=SEED)
        # Target at 100 (the starting price itself) — many paths will dip below
        hit_times = _first_hit_times(paths, 100.0, direction="below")
        hitting = hit_times[hit_times > 0]
        assert len(hitting) > 0
        assert all(hitting > 0)

    def test_direction_above(self):
        # Target slightly above start — most paths should hit eventually
        paths = simulate_intraday_paths(100.0, 0.25, 0.0, 120, n_paths=N_PATHS, seed=SEED)
        hit_times = _first_hit_times(paths, 100.01, direction="above")
        # Most should hit
        assert (hit_times > 0).sum() > N_PATHS * 0.5

    def test_direction_below(self):
        paths = simulate_intraday_paths(100.0, 0.25, 0.0, 120, n_paths=N_PATHS, seed=SEED)
        hit_times = _first_hit_times(paths, 99.99, direction="below")
        assert (hit_times > 0).sum() > N_PATHS * 0.5

    def test_hit_time_within_path_length(self):
        n_steps = 60
        paths = simulate_intraday_paths(100.0, 0.25, 0.0, n_steps, n_paths=N_PATHS, seed=SEED)
        hit_times = _first_hit_times(paths, 100.01, direction="above")
        hitting = hit_times[hit_times > 0]
        assert all(hitting <= n_steps)


class TestComputePnlSek:
    """_compute_pnl_sek for warrant and direct positions."""

    def test_warrant_with_financing_level_profit(self):
        pos = _make_position(financing_level=25.0, entry_price_sek=85.0, qty=100)
        mkt = _make_market(price=35.0, usdsek=10.85)
        costs = CostModel(courtage_bps=0, spread_bps=0, slippage_bps=0)
        # exit_warrant_sek = (35 - 25) * 10.85 = 108.5
        # entry_value = 85 * 100 = 8500
        # exit_value = 108.5 * 100 = 10850
        # pnl = 10850 - 8500 - 0 = 2350
        pnl = _compute_pnl_sek(pos, 35.0, mkt, costs)
        assert pnl == pytest.approx(2350.0)

    def test_warrant_with_financing_level_loss(self):
        pos = _make_position(financing_level=25.0, entry_price_sek=85.0, qty=100)
        mkt = _make_market(price=30.0, usdsek=10.85)
        costs = CostModel()
        # exit_warrant_sek = (30 - 25) * 10.85 = 54.25
        # exit_value = 5425, entry_value = 8500
        pnl = _compute_pnl_sek(pos, 30.0, mkt, costs)
        assert pnl == pytest.approx(5425.0 - 8500.0)

    def test_warrant_knockout_floors_at_zero(self):
        pos = _make_position(financing_level=25.0, entry_price_sek=85.0, qty=100)
        mkt = _make_market(price=24.0, usdsek=10.85)
        costs = CostModel()
        # exit_warrant_sek = max((24 - 25) * 10.85, 0) = 0
        pnl = _compute_pnl_sek(pos, 24.0, mkt, costs)
        assert pnl == pytest.approx(-8500.0)  # Total loss

    def test_warrant_without_financing_level_leverage(self):
        pos = _make_position(
            financing_level=None, leverage=5.0,
            entry_price_sek=85.0, entry_underlying_usd=32.0, qty=100
        )
        mkt = _make_market(price=33.6, usdsek=10.85)
        costs = CostModel()
        # pct_move = (33.6 - 32) / 32 = 0.05
        # warrant_move = 0.05 * 5 = 0.25
        # exit_warrant_sek = 85 * 1.25 = 106.25
        # pnl = 106.25*100 - 85*100 = 2125
        pnl = _compute_pnl_sek(pos, 33.6, mkt, costs)
        assert pnl == pytest.approx(2125.0)

    def test_direct_stock_position(self):
        pos = _make_position(
            instrument_type="stock", financing_level=None, leverage=1.0,
            entry_price_sek=50.0, entry_underlying_usd=50.0, qty=10
        )
        mkt = _make_market(price=55.0, usdsek=10.85)
        costs = CostModel()
        # exit_value = 10 * 55 * 10.85 = 5967.5
        # entry_value = 10 * 50 * 10.85 = 5425
        # pnl = 5967.5 - 5425 = 542.5
        pnl = _compute_pnl_sek(pos, 55.0, mkt, costs)
        assert pnl == pytest.approx(542.5)

    def test_costs_deducted(self):
        pos = _make_position(financing_level=25.0, entry_price_sek=85.0, qty=100)
        mkt = _make_market(price=35.0, usdsek=10.85)
        costs = CostModel(courtage_bps=10, spread_bps=5, slippage_bps=5)
        # exit_value = (35-25)*10.85*100 = 10850
        # cost = 10850 * (10+5+5)/10000 = 10850 * 0.002 = 21.7
        pnl = _compute_pnl_sek(pos, 35.0, mkt, costs)
        expected = 10850 - 8500 - 21.7
        assert pnl == pytest.approx(expected, abs=0.01)


class TestComputeRiskFlags:
    """_compute_risk_flags for session end, knock-out, and hold time."""

    def test_session_end_imminent(self):
        flags = _compute_risk_flags(
            target_price=34.0, position=_make_position(),
            market=_make_market(), remaining_minutes=20,
        )
        assert "SESSION_END_IMMINENT" in flags

    def test_session_end_near(self):
        flags = _compute_risk_flags(
            target_price=34.0, position=_make_position(),
            market=_make_market(), remaining_minutes=45,
        )
        assert "SESSION_END_NEAR" in flags

    def test_no_session_flag_with_plenty_of_time(self):
        flags = _compute_risk_flags(
            target_price=34.0, position=_make_position(),
            market=_make_market(), remaining_minutes=200,
        )
        assert "SESSION_END_IMMINENT" not in flags
        assert "SESSION_END_NEAR" not in flags

    def test_knockout_danger(self):
        # Price 26, financing 25.5 → distance = (26-25.5)/26*100 = 1.92%
        pos = _make_position(financing_level=25.5)
        mkt = _make_market(price=26.0)
        flags = _compute_risk_flags(None, pos, mkt, 200)
        assert "KNOCKOUT_DANGER" in flags

    def test_knockout_warning(self):
        # Price 30, financing 28 → distance = (30-28)/30*100 = 6.67%
        pos = _make_position(financing_level=28.0)
        mkt = _make_market(price=30.0)
        flags = _compute_risk_flags(None, pos, mkt, 200)
        assert "KNOCKOUT_WARNING" in flags

    def test_no_knockout_flag_when_safe(self):
        # Price 33.5, financing 25 → distance = (33.5-25)/33.5*100 = 25.4%
        pos = _make_position(financing_level=25.0)
        mkt = _make_market(price=33.5)
        flags = _compute_risk_flags(None, pos, mkt, 200)
        assert "KNOCKOUT_DANGER" not in flags
        assert "KNOCKOUT_WARNING" not in flags

    def test_hold_time_extended(self):
        pos = _make_position(entry_ts=BASE_TIME - timedelta(hours=6))
        mkt = _make_market(asof_ts=BASE_TIME)
        flags = _compute_risk_flags(None, pos, mkt, 200)
        assert "HOLD_TIME_EXTENDED" in flags

    def test_no_hold_time_flag_when_fresh(self):
        pos = _make_position(entry_ts=BASE_TIME - timedelta(hours=1))
        mkt = _make_market(asof_ts=BASE_TIME)
        flags = _compute_risk_flags(None, pos, mkt, 200)
        assert "HOLD_TIME_EXTENDED" not in flags


class TestApplyRiskOverrides:
    """_apply_risk_overrides override logic."""

    def _make_candidates(self):
        limit = CandidateExit(
            price_usd=35.0, action="limit", fill_prob=0.6,
            expected_fill_time_min=30, pnl_sek=2000, ev_sek=1200,
            pnl_pct=5.0,
        )
        market = CandidateExit(
            price_usd=33.5, action="market", fill_prob=1.0,
            expected_fill_time_min=0, pnl_sek=500, ev_sek=500,
            pnl_pct=1.5,
        )
        return [limit, market]

    def test_no_override_returns_highest_ev(self):
        candidates = self._make_candidates()
        pos = _make_position(financing_level=25.0)
        mkt = _make_market(price=33.5)
        rec = _apply_risk_overrides(candidates, pos, mkt, 200)
        # Should return first candidate (highest EV) since no override triggers
        assert rec.action == "limit"

    def test_knockout_danger_forces_market(self):
        candidates = self._make_candidates()
        pos = _make_position(financing_level=33.0)
        mkt = _make_market(price=33.5)
        # distance = (33.5-33)/33.5*100 = 1.49% < 3
        rec = _apply_risk_overrides(candidates, pos, mkt, 200)
        assert rec.action == "market"

    def test_session_ending_forces_market(self):
        candidates = self._make_candidates()
        pos = _make_position(financing_level=25.0)
        mkt = _make_market(price=33.5)
        rec = _apply_risk_overrides(candidates, pos, mkt, remaining_minutes=3)
        assert rec.action == "market"

    def test_high_knockout_probability_forces_market(self):
        candidates = self._make_candidates()
        pos = _make_position(financing_level=25.0)
        mkt = _make_market(price=33.5)
        # session_min all below the stop buffer (25 * 1.03 = 25.75)
        fake_min = np.array([20.0] * 100)  # All below buffer
        rec = _apply_risk_overrides(candidates, pos, mkt, 200, session_min=fake_min)
        assert rec.action == "market"

    def test_low_knockout_probability_no_override(self):
        candidates = self._make_candidates()
        pos = _make_position(financing_level=25.0)
        mkt = _make_market(price=33.5)
        # session_min all safely above buffer
        fake_min = np.array([30.0] * 100)
        rec = _apply_risk_overrides(candidates, pos, mkt, 200, session_min=fake_min)
        assert rec.action == "limit"


class TestComputeExitPlan:
    """compute_exit_plan full integration tests."""

    def test_returns_exit_plan(self):
        pos = _make_position()
        mkt = _make_market()
        session_end = BASE_TIME + timedelta(hours=6)
        plan = compute_exit_plan(pos, mkt, session_end, n_paths=N_PATHS, seed=SEED)
        assert isinstance(plan, ExitPlan)

    def test_candidates_sorted_by_ev_descending(self):
        pos = _make_position()
        mkt = _make_market()
        session_end = BASE_TIME + timedelta(hours=6)
        plan = compute_exit_plan(pos, mkt, session_end, n_paths=N_PATHS, seed=SEED)
        evs = [c.ev_sek for c in plan.candidates]
        assert evs == sorted(evs, reverse=True)

    def test_market_exit_always_present(self):
        pos = _make_position()
        mkt = _make_market()
        session_end = BASE_TIME + timedelta(hours=6)
        plan = compute_exit_plan(pos, mkt, session_end, n_paths=N_PATHS, seed=SEED)
        assert plan.market_exit is not None
        assert plan.market_exit.action == "market"
        assert plan.market_exit.fill_prob == 1.0

    def test_hold_to_close_candidate_present(self):
        pos = _make_position()
        mkt = _make_market()
        session_end = BASE_TIME + timedelta(hours=6)
        plan = compute_exit_plan(pos, mkt, session_end, n_paths=N_PATHS, seed=SEED)
        hold_candidates = [c for c in plan.candidates if c.action == "hold_to_close"]
        assert len(hold_candidates) == 1

    def test_session_ended_edge_case(self):
        pos = _make_position()
        mkt = _make_market()
        # Session end in the past
        session_end = BASE_TIME - timedelta(minutes=5)
        plan = compute_exit_plan(pos, mkt, session_end, n_paths=N_PATHS, seed=SEED)
        assert plan.remaining_minutes == 0
        assert len(plan.candidates) == 1
        assert plan.candidates[0].risk_flags == ("SESSION_ENDED",)
        assert plan.provenance.get("reason") == "session_ended"

    def test_higher_quantile_targets_lower_fill_prob(self):
        pos = _make_position()
        mkt = _make_market()
        session_end = BASE_TIME + timedelta(hours=6)
        plan = compute_exit_plan(pos, mkt, session_end, n_paths=N_PATHS, seed=SEED)
        limit_candidates = [c for c in plan.candidates if c.action == "limit"]
        if len(limit_candidates) >= 2:
            # Sorted by EV, but check that higher quantile = lower fill prob
            by_q = sorted(limit_candidates, key=lambda c: c.quantile or 0)
            for i in range(len(by_q) - 1):
                assert by_q[i].fill_prob >= by_q[i + 1].fill_prob

    def test_summary_produces_string(self):
        pos = _make_position()
        mkt = _make_market()
        session_end = BASE_TIME + timedelta(hours=6)
        plan = compute_exit_plan(pos, mkt, session_end, n_paths=N_PATHS, seed=SEED)
        s = plan.summary()
        assert isinstance(s, str)
        assert "Exit plan" in s

    def test_to_dict_valid_output(self):
        pos = _make_position()
        mkt = _make_market()
        session_end = BASE_TIME + timedelta(hours=6)
        plan = compute_exit_plan(pos, mkt, session_end, n_paths=N_PATHS, seed=SEED)
        d = plan.to_dict()
        assert d["symbol"] == "XAG-USD"
        assert "recommended" in d
        assert "market_exit_sek" in d
        assert "session_max" in d
        assert "session_min" in d
        assert "n_candidates" in d
        assert d["n_candidates"] == len(plan.candidates)

    def test_provenance_has_expected_keys(self):
        pos = _make_position()
        mkt = _make_market()
        session_end = BASE_TIME + timedelta(hours=6)
        plan = compute_exit_plan(pos, mkt, session_end, n_paths=N_PATHS, seed=SEED)
        prov = plan.provenance
        assert prov["model"] == "GBM_antithetic"
        assert prov["n_paths"] == N_PATHS
        assert "volatility" in prov
        assert "drift" in prov
        assert "remaining_min" in prov
        assert "instrument_type" in prov
        assert "cost_model" in prov

    def test_auto_selects_cost_model(self):
        pos = _make_position(instrument_type="warrant")
        mkt = _make_market()
        session_end = BASE_TIME + timedelta(hours=6)
        plan = compute_exit_plan(pos, mkt, session_end, n_paths=N_PATHS, seed=SEED)
        assert plan.provenance["cost_model"] == "avanza_warrant"

    def test_explicit_cost_model_used(self):
        pos = _make_position()
        mkt = _make_market()
        session_end = BASE_TIME + timedelta(hours=6)
        plan = compute_exit_plan(
            pos, mkt, session_end, costs=CRYPTO_COSTS, n_paths=N_PATHS, seed=SEED
        )
        assert plan.provenance["cost_model"] == "crypto"


class TestComputeExitPlanFromSummary:
    """compute_exit_plan_from_summary integration."""

    def _make_summary(self, ticker="XAG-USD", price=33.5, atr_pct=2.5):
        return {
            "signals": {
                ticker: {
                    "price_usd": price,
                    "extra": {"atr_pct": atr_pct},
                }
            },
            "fx_rate": 10.85,
        }

    def _make_pos_state(self):
        return {
            "shares": 220,
            "entry_price_sek": 85.0,
            "entry_underlying_usd": 32.0,
            "entry_ts": (BASE_TIME - timedelta(hours=2)).isoformat(),
        }

    def test_returns_none_for_missing_ticker(self):
        summary = self._make_summary()
        pos_state = self._make_pos_state()
        session_end = BASE_TIME + timedelta(hours=6)
        result = compute_exit_plan_from_summary(
            "FAKE-TICKER", summary, pos_state, session_end, n_paths=100
        )
        assert result is None

    def test_returns_none_for_zero_price(self):
        summary = self._make_summary(price=0)
        pos_state = self._make_pos_state()
        session_end = BASE_TIME + timedelta(hours=6)
        result = compute_exit_plan_from_summary(
            "XAG-USD", summary, pos_state, session_end, n_paths=100
        )
        assert result is None

    def test_valid_summary_returns_exit_plan(self):
        summary = self._make_summary()
        pos_state = self._make_pos_state()
        session_end = BASE_TIME + timedelta(hours=6)
        plan = compute_exit_plan_from_summary(
            "XAG-USD", summary, pos_state, session_end,
            instrument_type="warrant", financing_level=25.0,
            leverage=5.0, n_paths=N_PATHS,
        )
        assert isinstance(plan, ExitPlan)
        assert plan.symbol == "XAG-USD"
