"""End-to-end integration tests: orchestrator + strategy loading + both adapters."""
import time

import pytest

from portfolio.strategies.base import SharedData
from portfolio.strategies.orchestrator import StrategyOrchestrator, load_strategies


@pytest.fixture
def full_config():
    return {
        "golddigger": {
            "trade_enabled": True,
            "poll_seconds": 5,
            "bull_orderbook_id": "12345",
            "equity_sek": 100000.0,
            "use_augmented_signals": False,
            "use_signal_consensus": False,
            "use_macro_context": False,
            "use_volume_confirm": False,
            "use_chronos_forecast": False,
            "use_intraday_dxy_gate": False,
            "use_event_risk_gate": False,
        },
        "elongir": {
            "poll_seconds": 30,
            "equity_sek": 100000.0,
        },
        "avanza": {"account_id": "1625505"},
        "strategies": {
            "golddigger_enabled": True,
            "elongir_enabled": True,
        },
    }


def test_load_strategies_both(full_config):
    strategies = load_strategies(full_config)
    names = [s.name() for s in strategies]
    assert "golddigger" in names
    assert "elongir" in names
    assert len(strategies) == 2


def test_load_strategies_golddigger_only(full_config):
    full_config["strategies"]["elongir_enabled"] = False
    strategies = load_strategies(full_config)
    names = [s.name() for s in strategies]
    assert "golddigger" in names
    assert "elongir" not in names


def test_load_strategies_elongir_only(full_config):
    full_config["strategies"]["golddigger_enabled"] = False
    strategies = load_strategies(full_config)
    names = [s.name() for s in strategies]
    assert "golddigger" not in names
    assert "elongir" in names


def test_load_strategies_none_enabled(full_config):
    full_config["strategies"]["golddigger_enabled"] = False
    full_config["strategies"]["elongir_enabled"] = False
    strategies = load_strategies(full_config)
    assert len(strategies) == 0


def test_load_strategies_auto_enable():
    """Strategies auto-enable if their config section exists and no strategies section."""
    config = {
        "golddigger": {
            "poll_seconds": 5,
            "bull_orderbook_id": "12345",
            "use_augmented_signals": False,
            "use_signal_consensus": False,
            "use_macro_context": False,
            "use_volume_confirm": False,
            "use_chronos_forecast": False,
            "use_intraday_dxy_gate": False,
            "use_event_risk_gate": False,
        },
        "avanza": {"account_id": "1625505"},
    }
    strategies = load_strategies(config)
    names = [s.name() for s in strategies]
    assert "golddigger" in names
    # elongir not in config, so not loaded
    assert "elongir" not in names


def test_full_orchestrator_startup_and_shutdown(full_config):
    """Both strategies load and orchestrator starts/stops cleanly."""
    strategies = load_strategies(full_config)
    shared = SharedData(
        underlying_prices={"XAU-USD": 2345.0, "XAG-USD": 33.5},
        fx_rate=10.5,
        cert_prices={"12345": {"bid": 55.0, "ask": 56.0}},
        is_market_hours=True,
    )
    orch = StrategyOrchestrator(strategies=strategies, shared_data=shared)
    orch.start()
    time.sleep(0.5)
    orch.stop()
    summary = orch.summary()
    assert "golddigger" in summary
    assert "elongir" in summary


def test_orchestrator_summary_format(full_config):
    strategies = load_strategies(full_config)
    shared = SharedData(underlying_prices={}, fx_rate=10.5)
    orch = StrategyOrchestrator(strategies=strategies, shared_data=shared)
    summary = orch.summary()
    assert "2 strategies" in summary
    assert "5.0s" in summary   # golddigger poll
    assert "30.0s" in summary  # elongir poll
