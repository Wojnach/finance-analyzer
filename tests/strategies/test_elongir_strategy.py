"""Tests for Elongir strategy adapter."""
import pytest
from datetime import UTC, datetime
from unittest.mock import patch

from portfolio.strategies.base import SharedData


@pytest.fixture
def shared_data():
    return SharedData(
        underlying_prices={"XAG-USD": 33.5, "XAU-USD": 2345.0},
        fx_rate=10.5,
        cert_prices={},
        is_market_hours=True,
        timestamp=datetime.now(UTC),
    )


@pytest.fixture
def elongir_config():
    return {
        "elongir": {
            "poll_seconds": 30,
            "equity_sek": 100000.0,
            "financing_level": 75.03,
        },
    }


def test_elongir_strategy_creation(elongir_config):
    from portfolio.strategies.elongir_strategy import ElongirStrategy
    s = ElongirStrategy(elongir_config)
    assert s.name() == "elongir"
    assert s.poll_interval_seconds() == 30.0
    assert s.is_active() is True


def test_elongir_strategy_builds_snapshot(elongir_config, shared_data):
    from portfolio.strategies.elongir_strategy import ElongirStrategy
    s = ElongirStrategy(elongir_config)
    snap = s._build_snapshot(shared_data, klines_1m=None, klines_5m=None, klines_15m=None)
    assert snap.silver_usd == 33.5
    assert snap.fx_rate == 10.5
    assert snap.klines_1m is None


def test_elongir_strategy_builds_snapshot_default_fx(elongir_config):
    """Falls back to 10.5 when FX is 0."""
    from portfolio.strategies.elongir_strategy import ElongirStrategy
    shared = SharedData(
        underlying_prices={"XAG-USD": 33.5},
        fx_rate=0.0,
    )
    s = ElongirStrategy(elongir_config)
    snap = s._build_snapshot(shared, klines_1m=None, klines_5m=None, klines_15m=None)
    assert snap.fx_rate == 10.5


def test_elongir_strategy_tick_no_silver(elongir_config):
    """If silver price is 0, tick returns None without fetching klines."""
    from portfolio.strategies.elongir_strategy import ElongirStrategy
    shared = SharedData(
        underlying_prices={"XAG-USD": 0.0},
        fx_rate=10.5,
    )
    s = ElongirStrategy(elongir_config)
    # No need to mock fetch_klines — we return early before reaching it
    result = s.tick(shared)
    assert result is None


def test_elongir_strategy_tick_no_crash(elongir_config, shared_data):
    """Tick should not crash even without klines (incomplete snapshot)."""
    from portfolio.strategies.elongir_strategy import ElongirStrategy

    s = ElongirStrategy(elongir_config)
    with patch("portfolio.strategies.elongir_strategy.fetch_klines", return_value=None):
        result = s.tick(shared_data)
    # Incomplete snapshot (no klines) — bot may skip or return None
    assert result is None or isinstance(result, dict)


def test_elongir_strategy_status(elongir_config):
    from portfolio.strategies.elongir_strategy import ElongirStrategy
    s = ElongirStrategy(elongir_config)
    status = s.status_summary()
    assert "elongir" in status
    assert "SCANNING" in status
