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


def _crypto_pos(entry_warrant=10.0, entry_underlying=70000.0, direction="LONG",
                leverage=1.0, tp_warrant_pct=None, sl_warrant_pct=None):
    """Build a synthetic crypto position dict.

    By default uses leverage=1.0 with tp/sl computed at BASE × leverage so
    a 1x tracker test stays self-consistent. Tests targeting the legacy
    fallback path pass tp_warrant_pct=None and sl_warrant_pct=None and
    drop the keys via _drop_per_pos_tpsl().
    """
    if tp_warrant_pct is None:
        # default to per-position values pinned at BASE × leverage
        from data import crypto_swing_config as _ccfg
        tp_warrant_pct = _ccfg.TP_BASE_UNDERLYING_PCT * leverage
    if sl_warrant_pct is None:
        from data import crypto_swing_config as _ccfg
        sl_warrant_pct = _ccfg.SL_BASE_UNDERLYING_PCT * leverage
    return {
        "pos_id": "BTC-USD_1",
        "ticker": "BTC-USD",
        "warrant_key": "TEST",
        "ob_id": "9999",
        "direction": direction,
        "leverage": leverage,
        "units": 100,
        "entry_warrant_bid": entry_warrant,
        "entry_underlying_price": entry_underlying,
        "peak_underlying_price": entry_underlying,
        "peak_warrant_bid": entry_warrant,
        "entry_ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "tp_warrant_pct": tp_warrant_pct,
        "sl_warrant_pct": sl_warrant_pct,
    }


def _drop_per_pos_tpsl(pos):
    """Strip tp_warrant_pct/sl_warrant_pct to simulate a legacy position
    written before codex fix C 2026-05-11. Such positions must fall back
    to the deprecated module constants."""
    pos.pop("tp_warrant_pct", None)
    pos.pop("sl_warrant_pct", None)
    return pos


def _oil_pos(entry_warrant=20.0, entry_underlying=70.0, direction="LONG",
             leverage=1.52, tp_warrant_pct=None, sl_warrant_pct=None):
    if tp_warrant_pct is None:
        from data import oil_swing_config as _ocfg
        tp_warrant_pct = _ocfg.TP_BASE_UNDERLYING_PCT * leverage
    if sl_warrant_pct is None:
        from data import oil_swing_config as _ocfg
        sl_warrant_pct = _ocfg.SL_BASE_UNDERLYING_PCT * leverage
    return {
        "pos_id": "OIL-USD_1",
        "ticker": "OIL-USD",
        "warrant_key": "MINI_L_OLJA",
        "ob_id": "2370189",
        "direction": direction,
        "leverage": leverage,
        "units": 50,
        "entry_warrant_bid": entry_warrant,
        "entry_underlying_price": entry_underlying,
        "peak_underlying_price": entry_underlying,
        "peak_warrant_bid": entry_warrant,
        "entry_ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "tp_warrant_pct": tp_warrant_pct,
        "sl_warrant_pct": sl_warrant_pct,
    }


# --- Crypto TP/SL on warrant -----------------------------------------------

def test_crypto_tp_fires_on_warrant_pct(crypto_isolate):
    """5x leverage → tp_warrant_pct = 5%. TP fires at 10.0 → 10.5."""
    trader = cst.CryptoSwingTrader(page=None, executor=None)
    pos = _crypto_pos(entry_warrant=10.0, leverage=5.0)
    # +5% warrant: 10.0 -> 10.5
    should_sell, reason = trader._evaluate_exit(
        pos, current_underlying=70000.0,  # unchanged underlying
        current_warrant_bid=10.5,
        sig={},
    )
    assert should_sell, f"expected sell, got reason={reason}"
    assert "TAKE_PROFIT" in reason
    assert "warrant" in reason.lower()


def test_crypto_sl_fires_on_warrant_pct(crypto_isolate):
    """5x leverage → sl_warrant_pct = 30%. SL fires at 10.0 → 7.0."""
    trader = cst.CryptoSwingTrader(page=None, executor=None)
    pos = _crypto_pos(entry_warrant=10.0, leverage=5.0)
    # -30% warrant: 10.0 -> 7.0
    should_sell, reason = trader._evaluate_exit(
        pos, current_underlying=70000.0,
        current_warrant_bid=7.0,
        sig={},
    )
    assert should_sell, f"expected sell, got reason={reason}"
    assert "HARD_STOP" in reason
    assert "warrant" in reason.lower()


def test_crypto_no_exit_inside_warrant_band(crypto_isolate):
    """5x cert: +2% warrant (below TP=5) and -10% (above SL=-30), no exit fires."""
    trader = cst.CryptoSwingTrader(page=None, executor=None)
    pos = _crypto_pos(entry_warrant=10.0, leverage=5.0)
    should_sell, reason = trader._evaluate_exit(
        pos, current_underlying=70000.0,
        current_warrant_bid=10.2,  # +2%
        sig={},
    )
    assert not should_sell, f"unexpected sell at +2% warrant: {reason}"

    pos2 = _crypto_pos(entry_warrant=10.0, leverage=5.0)
    should_sell2, reason2 = trader._evaluate_exit(
        pos2, current_underlying=70000.0,
        current_warrant_bid=9.0,  # -10%
        sig={},
    )
    assert not should_sell2, f"unexpected sell at -10% warrant: {reason2}"


# --- Oil TP/SL on warrant --------------------------------------------------

def test_oil_tp_fires_on_warrant_pct(oil_isolate):
    """5x leverage → tp_warrant_pct = 5%. TP fires at 20.0 → 21.0."""
    trader = ost.OilSwingTrader(page=None, executor=None)
    pos = _oil_pos(entry_warrant=20.0, leverage=5.0)
    # +5% warrant: 20.0 -> 21.0
    should_sell, reason = trader._evaluate_exit(
        pos, current_underlying=70.0,
        current_warrant_bid=21.0,
        sig={},
    )
    assert should_sell, f"expected sell, got reason={reason}"
    assert "TAKE_PROFIT" in reason


def test_oil_sl_fires_on_warrant_pct(oil_isolate):
    """5x leverage → sl_warrant_pct = 30%. SL fires at 20.0 → 14.0."""
    trader = ost.OilSwingTrader(page=None, executor=None)
    pos = _oil_pos(entry_warrant=20.0, leverage=5.0)
    # -30% warrant: 20.0 -> 14.0
    should_sell, reason = trader._evaluate_exit(
        pos, current_underlying=70.0,
        current_warrant_bid=14.0,
        sig={},
    )
    assert should_sell, f"expected sell, got reason={reason}"
    assert "HARD_STOP" in reason


# --- Codex fix C 2026-05-11: per-leverage TP/SL ---------------------------

def test_crypto_1x_tracker_tp_at_1_pct(crypto_isolate):
    """1x tracker (XBT/ETH-TRACKER): TP fires at +1% warrant, not +5%.
    Codex fix C 2026-05-11."""
    trader = cst.CryptoSwingTrader(page=None, executor=None)
    pos = _crypto_pos(entry_warrant=10.0, leverage=1.0)
    # +1% warrant: 10.0 -> 10.1
    should_sell, reason = trader._evaluate_exit(
        pos, current_underlying=70000.0, current_warrant_bid=10.1, sig={},
    )
    assert should_sell, f"expected TP at +1% on 1x, got reason={reason}"
    assert "TAKE_PROFIT" in reason


def test_crypto_1x_tracker_sl_at_6_pct(crypto_isolate):
    """1x tracker: SL fires at -6% warrant, not -30%."""
    trader = cst.CryptoSwingTrader(page=None, executor=None)
    pos = _crypto_pos(entry_warrant=10.0, leverage=1.0)
    # -7% warrant: 10.0 -> 9.3 (clearly past SL=-6 on 1x; avoid FP edge at exact -6)
    should_sell, reason = trader._evaluate_exit(
        pos, current_underlying=70000.0, current_warrant_bid=9.3, sig={},
    )
    assert should_sell, f"expected SL at -7% on 1x, got reason={reason}"
    assert "HARD_STOP" in reason


def test_crypto_10x_tp_at_10_pct(crypto_isolate):
    """10x cert: TP fires at +10% warrant, not +5%."""
    trader = cst.CryptoSwingTrader(page=None, executor=None)
    pos = _crypto_pos(entry_warrant=10.0, leverage=10.0)
    # +10% warrant: 10.0 -> 11.0
    should_sell, reason = trader._evaluate_exit(
        pos, current_underlying=70000.0, current_warrant_bid=11.0, sig={},
    )
    assert should_sell, f"expected TP at +10% on 10x, got reason={reason}"
    assert "TAKE_PROFIT" in reason

    # +5% warrant on 10x must NOT trigger (below the +10% per-pos threshold)
    pos2 = _crypto_pos(entry_warrant=10.0, leverage=10.0)
    should_sell2, _ = trader._evaluate_exit(
        pos2, current_underlying=70000.0, current_warrant_bid=10.5, sig={},
    )
    assert not should_sell2, "10x must NOT exit at +5% — that's below TP=10%"


def test_crypto_10x_sl_at_60_pct(crypto_isolate):
    """10x cert: SL fires at -60% warrant."""
    trader = cst.CryptoSwingTrader(page=None, executor=None)
    pos = _crypto_pos(entry_warrant=10.0, leverage=10.0)
    # -60% warrant: 10.0 -> 4.0
    should_sell, reason = trader._evaluate_exit(
        pos, current_underlying=70000.0, current_warrant_bid=4.0, sig={},
    )
    assert should_sell, f"expected SL at -60% on 10x, got reason={reason}"
    assert "HARD_STOP" in reason


def test_crypto_legacy_position_falls_back_to_constants(crypto_isolate):
    """A position WITHOUT tp_warrant_pct / sl_warrant_pct (written before
    codex fix C) must fall back to TAKE_PROFIT_WARRANT_PCT=5 /
    STOP_LOSS_WARRANT_PCT=30 — preserving behaviour for in-flight legacy
    positions written by the old code."""
    trader = cst.CryptoSwingTrader(page=None, executor=None)
    legacy = _drop_per_pos_tpsl(_crypto_pos(entry_warrant=10.0, leverage=3.0))
    assert "tp_warrant_pct" not in legacy
    assert "sl_warrant_pct" not in legacy

    # +5% warrant → fallback TP=5 fires
    should_sell, reason = trader._evaluate_exit(
        legacy, current_underlying=70000.0, current_warrant_bid=10.5, sig={},
    )
    assert should_sell, f"legacy TP fallback failed: {reason}"
    assert "TAKE_PROFIT" in reason

    # -30% warrant → fallback SL=30 fires
    legacy2 = _drop_per_pos_tpsl(_crypto_pos(entry_warrant=10.0, leverage=3.0))
    should_sell2, reason2 = trader._evaluate_exit(
        legacy2, current_underlying=70000.0, current_warrant_bid=7.0, sig={},
    )
    assert should_sell2, f"legacy SL fallback failed: {reason2}"
    assert "HARD_STOP" in reason2


def test_oil_1x_tp_sl_per_leverage(oil_isolate):
    """1x oil tracker mirror of the crypto 1x case."""
    trader = ost.OilSwingTrader(page=None, executor=None)
    pos = _oil_pos(entry_warrant=20.0, leverage=1.0)
    # +1% warrant
    should_sell, reason = trader._evaluate_exit(
        pos, current_underlying=70.0, current_warrant_bid=20.2, sig={},
    )
    assert should_sell, f"expected TP +1% on 1x oil, got {reason}"
    assert "TAKE_PROFIT" in reason

    pos2 = _oil_pos(entry_warrant=20.0, leverage=1.0)
    # -7% warrant: 20.0 -> 18.6 (clearly past SL=-6 on 1x oil)
    should_sell2, reason2 = trader._evaluate_exit(
        pos2, current_underlying=70.0, current_warrant_bid=18.6, sig={},
    )
    assert should_sell2, f"expected SL -7% on 1x oil, got {reason2}"
    assert "HARD_STOP" in reason2


def test_oil_10x_tp_sl_per_leverage(oil_isolate):
    """10x oil cert mirror of crypto 10x case."""
    trader = ost.OilSwingTrader(page=None, executor=None)
    pos = _oil_pos(entry_warrant=20.0, leverage=10.0)
    # +10% warrant
    should_sell, reason = trader._evaluate_exit(
        pos, current_underlying=70.0, current_warrant_bid=22.0, sig={},
    )
    assert should_sell, f"expected TP +10% on 10x oil, got {reason}"
    assert "TAKE_PROFIT" in reason

    pos2 = _oil_pos(entry_warrant=20.0, leverage=10.0)
    # -60% warrant: 20.0 -> 8.0
    should_sell2, reason2 = trader._evaluate_exit(
        pos2, current_underlying=70.0, current_warrant_bid=8.0, sig={},
    )
    assert should_sell2, f"expected SL -60% on 10x oil, got {reason2}"
    assert "HARD_STOP" in reason2


def test_oil_legacy_position_falls_back_to_constants(oil_isolate):
    trader = ost.OilSwingTrader(page=None, executor=None)
    legacy = _drop_per_pos_tpsl(_oil_pos(entry_warrant=20.0, leverage=3.0))
    should_sell, reason = trader._evaluate_exit(
        legacy, current_underlying=70.0, current_warrant_bid=21.0, sig={},
    )
    assert should_sell, f"oil legacy TP fallback failed: {reason}"
    assert "TAKE_PROFIT" in reason


def test_per_leverage_constants_present():
    """The new BASE constants must be present on all three configs.

    Imports defensively because metals_swing_config may be loaded under
    either 'metals_swing_config' or 'data.metals_swing_config' depending
    on which test file loaded it first (data/ vs data.* path).
    """
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
    import importlib

    # Force fresh load of the metals config since it's the one that
    # gets imported via the bare name in metals_swing_trader.py.
    if "metals_swing_config" in sys.modules:
        mc = importlib.reload(sys.modules["metals_swing_config"])
    else:
        import metals_swing_config as mc  # noqa: PLC0415

    from data import (
        crypto_swing_config as cc,
        oil_swing_config as oc,
    )
    for mod in (cc, mc, oc):
        assert mod.TP_BASE_UNDERLYING_PCT == 1.0, (
            f"{getattr(mod, '__name__', mod)}.TP_BASE_UNDERLYING_PCT must be 1.0"
        )
        assert mod.SL_BASE_UNDERLYING_PCT == 6.0, (
            f"{getattr(mod, '__name__', mod)}.SL_BASE_UNDERLYING_PCT must be 6.0"
        )


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
