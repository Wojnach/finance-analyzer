"""Verify 2026-05-11 TP/SL re-anchored to warrant pct change for swing traders.

TP fires when warrant_pct_change >= TAKE_PROFIT_WARRANT_PCT.
SL fires when warrant_pct_change <= -STOP_LOSS_WARRANT_PCT.
Broker stop price = entry × (1 - STOP_LOSS_WARRANT_PCT/100).
"""
from __future__ import annotations

import datetime

import pytest

from data import crypto_swing_config as crypto_cfg
from data import crypto_swing_trader as cst
from data import oil_swing_config as oil_cfg
from data import oil_swing_trader as ost


@pytest.fixture
def crypto_isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(crypto_cfg, "STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr(crypto_cfg, "DECISIONS_LOG", str(tmp_path / "decisions.jsonl"))
    monkeypatch.setattr(crypto_cfg, "TRADES_LOG", str(tmp_path / "trades.jsonl"))
    monkeypatch.setattr(crypto_cfg, "VALUE_HISTORY_LOG", str(tmp_path / "value.jsonl"))
    monkeypatch.setattr(crypto_cfg, "WARRANT_CATALOG_FILE", str(tmp_path / "catalog.json"))
    monkeypatch.setattr(crypto_cfg, "MOMENTUM_STATE_FILE", str(tmp_path / "momentum.json"))
    return tmp_path


@pytest.fixture
def oil_isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(oil_cfg, "STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr(oil_cfg, "DECISIONS_LOG", str(tmp_path / "decisions.jsonl"))
    monkeypatch.setattr(oil_cfg, "TRADES_LOG", str(tmp_path / "trades.jsonl"))
    monkeypatch.setattr(oil_cfg, "VALUE_HISTORY_LOG", str(tmp_path / "value.jsonl"))
    monkeypatch.setattr(oil_cfg, "WARRANT_CATALOG_FILE", str(tmp_path / "catalog.json"))
    monkeypatch.setattr(oil_cfg, "MOMENTUM_STATE_FILE", str(tmp_path / "momentum.json"))
    return tmp_path


def _crypto_pos(entry_warrant=10.0, entry_underlying=70000.0, direction="LONG"):
    return {
        "pos_id": "BTC-USD_1",
        "ticker": "BTC-USD",
        "warrant_key": "TEST",
        "ob_id": "9999",
        "direction": direction,
        "leverage": 1.0,
        "units": 100,
        "entry_warrant_bid": entry_warrant,
        "entry_underlying_price": entry_underlying,
        "peak_underlying_price": entry_underlying,
        "peak_warrant_bid": entry_warrant,
        "entry_ts": datetime.datetime.now(datetime.UTC).isoformat(),
    }


def _oil_pos(entry_warrant=20.0, entry_underlying=70.0, direction="LONG"):
    return {
        "pos_id": "OIL-USD_1",
        "ticker": "OIL-USD",
        "warrant_key": "MINI_L_OLJA",
        "ob_id": "2370189",
        "direction": direction,
        "leverage": 1.52,
        "units": 50,
        "entry_warrant_bid": entry_warrant,
        "entry_underlying_price": entry_underlying,
        "peak_underlying_price": entry_underlying,
        "peak_warrant_bid": entry_warrant,
        "entry_ts": datetime.datetime.now(datetime.UTC).isoformat(),
    }


# --- Crypto TP/SL on warrant -----------------------------------------------

def test_crypto_tp_fires_on_warrant_pct(crypto_isolate):
    trader = cst.CryptoSwingTrader(page=None, executor=None)
    pos = _crypto_pos(entry_warrant=10.0)
    # +5% warrant: 10.0 -> 10.5 (TAKE_PROFIT_WARRANT_PCT = 5.0)
    should_sell, reason = trader._evaluate_exit(
        pos, current_underlying=70000.0,  # unchanged underlying
        current_warrant_bid=10.5,
        sig={},
    )
    assert should_sell, f"expected sell, got reason={reason}"
    assert "TAKE_PROFIT" in reason
    assert "warrant" in reason.lower()


def test_crypto_sl_fires_on_warrant_pct(crypto_isolate):
    trader = cst.CryptoSwingTrader(page=None, executor=None)
    pos = _crypto_pos(entry_warrant=10.0)
    # -30% warrant: 10.0 -> 7.0 (STOP_LOSS_WARRANT_PCT = 30.0)
    should_sell, reason = trader._evaluate_exit(
        pos, current_underlying=70000.0,
        current_warrant_bid=7.0,
        sig={},
    )
    assert should_sell, f"expected sell, got reason={reason}"
    assert "HARD_STOP" in reason
    assert "warrant" in reason.lower()


def test_crypto_no_exit_inside_warrant_band(crypto_isolate):
    """At +2% warrant (below TP=5) and -10% (above SL=-30), no exit fires."""
    trader = cst.CryptoSwingTrader(page=None, executor=None)
    pos = _crypto_pos(entry_warrant=10.0)
    should_sell, reason = trader._evaluate_exit(
        pos, current_underlying=70000.0,
        current_warrant_bid=10.2,  # +2%
        sig={},
    )
    assert not should_sell, f"unexpected sell at +2% warrant: {reason}"

    pos2 = _crypto_pos(entry_warrant=10.0)
    should_sell2, reason2 = trader._evaluate_exit(
        pos2, current_underlying=70000.0,
        current_warrant_bid=9.0,  # -10%
        sig={},
    )
    assert not should_sell2, f"unexpected sell at -10% warrant: {reason2}"


# --- Oil TP/SL on warrant --------------------------------------------------

def test_oil_tp_fires_on_warrant_pct(oil_isolate):
    trader = ost.OilSwingTrader(page=None, executor=None)
    pos = _oil_pos(entry_warrant=20.0)
    # +5% warrant: 20.0 -> 21.0
    should_sell, reason = trader._evaluate_exit(
        pos, current_underlying=70.0,
        current_warrant_bid=21.0,
        sig={},
    )
    assert should_sell, f"expected sell, got reason={reason}"
    assert "TAKE_PROFIT" in reason


def test_oil_sl_fires_on_warrant_pct(oil_isolate):
    trader = ost.OilSwingTrader(page=None, executor=None)
    pos = _oil_pos(entry_warrant=20.0)
    # -30% warrant: 20.0 -> 14.0
    should_sell, reason = trader._evaluate_exit(
        pos, current_underlying=70.0,
        current_warrant_bid=14.0,
        sig={},
    )
    assert should_sell, f"expected sell, got reason={reason}"
    assert "HARD_STOP" in reason


# --- Metals broker-stop price computation ---------------------------------

def test_metals_broker_stop_price_uses_warrant_pct():
    """Broker stop price = entry × (1 - STOP_LOSS_WARRANT_PCT/100).

    Verify the conversion formula directly without invoking the live
    Avanza path. The trader writes the trigger_price computed exactly
    this way at metals_swing_trader.py inside _set_stop_loss.
    """
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
    import metals_swing_config as cfg

    entry = 100.0
    expected_trigger = round(entry * (1 - cfg.STOP_LOSS_WARRANT_PCT / 100), 2)
    # STOP_LOSS_WARRANT_PCT = 30.0 → 100 * 0.70 = 70.0
    assert expected_trigger == 70.0, (
        f"expected entry*0.70 = 70.0 with STOP_LOSS_WARRANT_PCT=30, "
        f"got {expected_trigger} (constant = {cfg.STOP_LOSS_WARRANT_PCT})"
    )


def test_metals_tp_constant_value():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
    import metals_swing_config as cfg

    assert cfg.TAKE_PROFIT_WARRANT_PCT == 5.0
    assert cfg.STOP_LOSS_WARRANT_PCT == 30.0


def test_crypto_oil_warrant_constants():
    assert crypto_cfg.TAKE_PROFIT_WARRANT_PCT == 5.0
    assert crypto_cfg.STOP_LOSS_WARRANT_PCT == 30.0
    assert oil_cfg.TAKE_PROFIT_WARRANT_PCT == 5.0
    assert oil_cfg.STOP_LOSS_WARRANT_PCT == 30.0
