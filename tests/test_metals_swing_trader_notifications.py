"""Notification behavior tests for metals_swing_trader."""

import datetime
import os
import sys

# data/ contains script-style modules, not a package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import metals_swing_trader as mst


def _make_trader():
    trader = mst.SwingTrader.__new__(mst.SwingTrader)
    trader.page = object()
    trader.state = mst._default_state()
    trader.check_count = 0
    # __init__ also seeds these — bypassing __init__ via __new__ requires
    # re-creating them so methods that touch them don't AttributeError.
    trader.regime_history = {}
    trader.warrant_catalog = {}

    # Avoid side effects and external dependencies during unit tests.
    trader._check_exits = lambda prices, signal_data: None
    trader._check_entries = lambda prices, signal_data: None
    trader._update_macd_history = lambda signal_data: None
    trader._update_regime_history = lambda signal_data: None
    return trader


def test_periodic_summary_disabled_by_default(monkeypatch):
    trader = _make_trader()
    trader.check_count = mst.TELEGRAM_SUMMARY_INTERVAL - 1

    called = []
    trader._send_summary = lambda signal_data: called.append(signal_data)

    monkeypatch.setattr(mst, "SEND_PERIODIC_SUMMARY", False)
    trader.evaluate_and_execute({}, {"XAG-USD": {"action": "HOLD"}})

    assert trader.check_count == mst.TELEGRAM_SUMMARY_INTERVAL
    assert called == []


def test_periodic_summary_can_be_enabled(monkeypatch):
    trader = _make_trader()
    trader.check_count = mst.TELEGRAM_SUMMARY_INTERVAL - 1

    called = []
    trader._send_summary = lambda signal_data: called.append(signal_data)

    monkeypatch.setattr(mst, "SEND_PERIODIC_SUMMARY", True)
    trader.evaluate_and_execute({}, {"XAG-USD": {"action": "HOLD"}})

    assert len(called) == 1


def test_send_summary_skips_no_positions(monkeypatch):
    trader = _make_trader()
    trader.check_count = 320
    trader.state["positions"] = {}

    sent = []
    monkeypatch.setattr(mst, "_send_telegram", lambda msg: sent.append(msg))
    trader._send_summary({"XAG-USD": {"action": "HOLD", "buy_count": 0, "sell_count": 0, "rsi": 50}})

    assert sent == []


def test_send_summary_still_sends_with_open_positions(monkeypatch):
    trader = _make_trader()
    trader.check_count = 320
    entry_ts = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=2)).isoformat()
    trader.state["positions"] = {
        "p1": {
            "warrant_name": "SILVER LONG",
            "ob_id": "123",
            "api_type": "warrant",
            "entry_price": 95.0,
            "entry_ts": entry_ts,
            "trailing_active": False,
        }
    }

    sent = []
    monkeypatch.setattr(mst, "fetch_price", lambda page, ob_id, api_type: {"bid": 100.0})
    monkeypatch.setattr(mst, "_send_telegram", lambda msg: sent.append(msg))

    trader._send_summary({})

    assert len(sent) == 1
    assert "*SWING #320*" in sent[0]
    assert "1 position(s)" in sent[0]
