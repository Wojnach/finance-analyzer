"""Behavior tests for data/oil_swing_trader.py.

xdist-safe: every test redirects state files to a tmp_path-derived location
via monkeypatch on the cfg module. Mirrors test_crypto_swing_trader.py.
"""
from __future__ import annotations

import datetime

import pytest

from data import oil_swing_config as cfg
from data import oil_swing_trader as ost


@pytest.fixture
def isolate_state(tmp_path, monkeypatch):
    """Redirect all oil_swing state files into tmp_path for xdist safety."""
    monkeypatch.setattr(cfg, "STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr(cfg, "DECISIONS_LOG", str(tmp_path / "decisions.jsonl"))
    monkeypatch.setattr(cfg, "TRADES_LOG", str(tmp_path / "trades.jsonl"))
    monkeypatch.setattr(cfg, "VALUE_HISTORY_LOG", str(tmp_path / "value.jsonl"))
    monkeypatch.setattr(cfg, "WARRANT_CATALOG_FILE",
                        str(tmp_path / "catalog.json"))
    monkeypatch.setattr(cfg, "MOMENTUM_STATE_FILE",
                        str(tmp_path / "momentum.json"))
    return tmp_path


def _fresh_signal(action="BUY", conf=0.75, buy=4, sell=1,
                   rsi=55, macd=0.05, regime="trending"):
    return {
        "recommendation": action,
        "calibrated_confidence": conf,
        "buy_voters": buy,
        "sell_voters": sell,
        "indicators": {"rsi": rsi, "macd": macd},
        "regime": regime,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# Extractor helpers — same surface as crypto_swing_trader
# ---------------------------------------------------------------------------
class TestExtractors:
    def test_extract_action_recognizes_keys(self):
        assert ost._extract_action({"recommendation": "BUY"}) == "BUY"
        assert ost._extract_action({"action": "sell"}) == "SELL"
        assert ost._extract_action({"signal": "HOLD"}) == "HOLD"

    def test_extract_action_default_hold(self):
        assert ost._extract_action({}) == "HOLD"

    def test_extract_confidence_handles_missing(self):
        assert ost._extract_confidence({}) == 0.0
        assert ost._extract_confidence({"calibrated_confidence": 0.7}) == 0.7
        assert ost._extract_confidence({"confidence": 0.5}) == 0.5

    def test_extract_voters_int_cast(self):
        assert ost._extract_voters({"buy_voters": "4", "sell_voters": "1"}) == (4, 1)
        assert ost._extract_voters({}) == (0, 0)

    def test_extract_indicator_falls_back_to_top_level(self):
        sig = {"rsi": 60, "macd": 0.02}
        assert ost._extract_indicator(sig, "rsi") == 60
        assert ost._extract_indicator(sig, "macd") == 0.02
        assert ost._extract_indicator(sig, "missing") is None

    def test_signal_age_returns_none_for_missing_ts(self):
        assert ost._signal_age_seconds({}) is None

    def test_signal_age_seconds_for_recent(self):
        sig = {"timestamp": datetime.datetime.now(datetime.UTC).isoformat()}
        age = ost._signal_age_seconds(sig)
        assert age is not None
        assert age < 5


# ---------------------------------------------------------------------------
# Trader lifecycle
# ---------------------------------------------------------------------------
class TestTraderLifecycle:
    def test_trader_instantiates_in_dry_run(self, isolate_state):
        trader = ost.OilSwingTrader(page=None, executor=None)
        assert trader.state is not None
        assert "positions" in trader.state
        assert isinstance(trader.warrant_catalog, dict)

    def test_trader_loads_fallback_catalog_when_no_page(self, isolate_state):
        # No page provided + empty cache => fallback should populate
        trader = ost.OilSwingTrader(page=None, executor=None)
        # Fallback contains 5 OLJA warrants from data/avanza_instruments_live.json
        assert len(trader.warrant_catalog) >= 1
        # Every warrant should be OIL-USD underlying
        for w in trader.warrant_catalog.values():
            assert w.get("underlying") == "OIL-USD"

    def test_evaluate_with_hold_signal_takes_no_action(self, isolate_state):
        trader = ost.OilSwingTrader(page=None, executor=None)
        prices = {"OIL-USD": 78.40}
        signal_data = {"per_ticker": {"OIL-USD": _fresh_signal(action="HOLD",
                                                                conf=0.75)}}
        summary = trader.evaluate_and_execute(prices, signal_data)
        assert summary is not None
        # No new positions on HOLD
        assert len(trader.state.get("positions", {})) == 0


# ---------------------------------------------------------------------------
# Entry gates — first-cycle rejection due to insufficient persistence
# ---------------------------------------------------------------------------
class TestEntryGates:
    def test_first_cycle_rejects_due_to_persistence(self, isolate_state):
        """A single high-confidence signal cannot trigger a BUY; the gate
        requires SIGNAL_PERSISTENCE_CHECKS=2 cycles of consistent confidence."""
        trader = ost.OilSwingTrader(page=None, executor=None)
        prices = {"OIL-USD": 78.40}
        sig = _fresh_signal(action="BUY", conf=0.75, buy=4, sell=1)
        summary = trader.evaluate_and_execute(prices, {"per_ticker": {"OIL-USD": sig}})
        assert summary is not None
        # First cycle: history empty, persistence gate must reject
        assert len(trader.state.get("positions", {})) == 0


# ---------------------------------------------------------------------------
# DRY_RUN safety — no executor calls
# ---------------------------------------------------------------------------
class TestDryRunSafety:
    def test_no_executor_calls_in_dry_run(self, isolate_state, monkeypatch):
        """Even on a fully-greenlit BUY, DRY_RUN means no executor call."""
        monkeypatch.setattr(cfg, "DRY_RUN", True)
        calls = []

        def _executor(*args, **kwargs):
            calls.append((args, kwargs))

        trader = ost.OilSwingTrader(page=None, executor=_executor)
        prices = {"OIL-USD": 78.40}

        # Run multiple cycles to satisfy persistence gate
        for _ in range(5):
            sig = _fresh_signal(action="BUY", conf=0.85, buy=5, sell=0,
                                rsi=55, macd=0.10, regime="trending")
            trader.evaluate_and_execute(prices, {"per_ticker": {"OIL-USD": sig}})

        # In DRY_RUN, executor should never be called
        assert calls == []
