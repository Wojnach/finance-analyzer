"""Tests for strategy base protocol and SharedData."""
import pytest
from datetime import UTC, datetime

from portfolio.strategies.base import SharedData, StrategyBase


def test_shared_data_creation():
    sd = SharedData(
        underlying_prices={"XAU-USD": 2345.0, "XAG-USD": 33.5},
        fx_rate=10.5,
        cert_prices={"12345": {"bid": 55.0, "ask": 56.0}},
        is_market_hours=True,
        timestamp=datetime.now(UTC),
    )
    assert sd.underlying_prices["XAU-USD"] == 2345.0
    assert sd.fx_rate == 10.5
    assert sd.is_market_hours is True


def test_shared_data_get_price():
    sd = SharedData(
        underlying_prices={"XAU-USD": 2345.0},
        fx_rate=10.5,
    )
    assert sd.get_price("XAU-USD") == 2345.0
    assert sd.get_price("MISSING") == 0.0


def test_shared_data_get_cert():
    sd = SharedData(
        underlying_prices={},
        fx_rate=10.5,
        cert_prices={"12345": {"bid": 55.0, "ask": 56.0}},
        is_market_hours=True,
    )
    assert sd.get_cert("12345") == {"bid": 55.0, "ask": 56.0}
    assert sd.get_cert("99999") is None


def test_shared_data_defaults():
    sd = SharedData()
    assert sd.underlying_prices == {}
    assert sd.fx_rate == 0.0
    assert sd.cert_prices == {}
    assert sd.is_market_hours is False
    assert sd.get_price("anything") == 0.0


def test_strategy_base_abstract():
    """Incomplete strategy must raise TypeError."""
    class IncompleteStrategy(StrategyBase):
        pass

    with pytest.raises(TypeError):
        IncompleteStrategy()


def test_concrete_strategy():
    """A complete implementation works."""
    class DummyStrategy(StrategyBase):
        def name(self) -> str:
            return "dummy"
        def poll_interval_seconds(self) -> float:
            return 10.0
        def tick(self, shared: SharedData) -> dict | None:
            return None
        def is_active(self) -> bool:
            return True
        def status_summary(self) -> str:
            return "dummy: idle"

    s = DummyStrategy()
    assert s.name() == "dummy"
    assert s.poll_interval_seconds() == 10.0
    assert s.is_active() is True
    assert s.tick(SharedData()) is None
    assert "dummy" in s.status_summary()
