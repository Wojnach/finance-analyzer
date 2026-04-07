"""Tests for GoldDigger strategy adapter."""
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from portfolio.strategies.base import SharedData


@pytest.fixture
def shared_data():
    return SharedData(
        underlying_prices={"XAU-USD": 2345.0, "XAG-USD": 33.5},
        fx_rate=10.5,
        cert_prices={"12345": {"bid": 55.0, "ask": 56.0, "last": 55.5}},
        is_market_hours=True,
        timestamp=datetime.now(UTC),
    )


@pytest.fixture
def gd_config():
    """Minimal config dict for GoldDigger."""
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
        "avanza": {"account_id": "1625505"},
    }


def test_golddigger_strategy_creation(gd_config):
    from portfolio.strategies.golddigger_strategy import GoldDiggerStrategy
    s = GoldDiggerStrategy(gd_config)
    assert s.name() == "golddigger"
    assert s.poll_interval_seconds() == 5.0
    assert s.is_active() is True


def test_golddigger_strategy_builds_snapshot(gd_config, shared_data):
    from unittest.mock import patch

    from portfolio.strategies.golddigger_strategy import GoldDiggerStrategy

    s = GoldDiggerStrategy(gd_config)
    with patch("portfolio.strategies.golddigger_strategy.fetch_usdsek", return_value=10.5):
        snap = s._build_snapshot(shared_data, gold_price=2345.0)
    assert snap.gold == 2345.0
    assert snap.usdsek == 10.5
    assert snap.cert_bid == 55.0
    assert snap.cert_ask == 56.0
    assert snap.data_quality == "ok"


def test_golddigger_strategy_builds_snapshot_no_cert(gd_config):
    """Snapshot works when no cert price is in shared data."""
    from portfolio.strategies.golddigger_strategy import GoldDiggerStrategy
    shared = SharedData(
        underlying_prices={"XAU-USD": 2345.0},
        fx_rate=10.5,
        cert_prices={},
        is_market_hours=True,
    )
    s = GoldDiggerStrategy(gd_config)
    snap = s._build_snapshot(shared, gold_price=2345.0)
    assert snap.gold == 2345.0
    assert snap.cert_bid is None
    assert snap.cert_ask is None


def test_golddigger_strategy_status(gd_config):
    from portfolio.strategies.golddigger_strategy import GoldDiggerStrategy
    s = GoldDiggerStrategy(gd_config)
    status = s.status_summary()
    assert "golddigger" in status
    assert "flat" in status


def test_golddigger_enqueue_trade(gd_config, tmp_path):
    from portfolio.strategies.golddigger_strategy import GoldDiggerStrategy
    queue_file = str(tmp_path / "trade_queue.json")
    s = GoldDiggerStrategy(gd_config, trade_queue_file=queue_file)
    s._enqueue_trade({
        "action": "BUY",
        "quantity": 100,
        "price": 55.5,
        "reason": "test signal",
    })
    data = json.loads(Path(queue_file).read_text(encoding="utf-8"))
    assert len(data["orders"]) == 1
    order = data["orders"][0]
    assert order["action"] == "BUY"
    assert order["status"] == "pending"
    assert order["source"] == "golddigger"
    assert order["quantity"] == 100
    assert order["ob_id"] == "12345"


def test_golddigger_enqueue_appends(gd_config, tmp_path):
    """Second enqueue should append, not overwrite."""
    from portfolio.strategies.golddigger_strategy import GoldDiggerStrategy
    queue_file = str(tmp_path / "trade_queue.json")
    s = GoldDiggerStrategy(gd_config, trade_queue_file=queue_file)
    s._enqueue_trade({"action": "BUY", "quantity": 50, "price": 55.0, "reason": "first"})
    s._enqueue_trade({"action": "SELL", "quantity": 50, "price": 56.0, "reason": "second"})
    data = json.loads(Path(queue_file).read_text(encoding="utf-8"))
    assert len(data["orders"]) == 2
    assert data["orders"][0]["action"] == "BUY"
    assert data["orders"][1]["action"] == "SELL"


def test_golddigger_tick_no_gold_returns_none(gd_config):
    """If gold price is 0 everywhere, tick returns None."""
    from unittest.mock import patch

    from portfolio.strategies.golddigger_strategy import GoldDiggerStrategy

    shared = SharedData(
        underlying_prices={"XAU-USD": 0.0},
        fx_rate=10.5,
    )
    s = GoldDiggerStrategy(gd_config)
    with patch("portfolio.strategies.golddigger_strategy.fetch_gold_price", return_value=None):
        result = s.tick(shared)
    assert result is None
