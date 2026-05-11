"""Verify the 2026-05-11 low-cash mode for crypto + oil swing traders.

When cash < LOW_CASH_THRESHOLD_SEK, MIN_TRADE_SEK acts as the position size
itself (not just a floor) — capped at 95% of cash for the courtage buffer.
Otherwise: max(raw_alloc, MIN_TRADE_SEK) capped at 95% cash.

Note: metals_swing_trader uses a Kelly-driven sizing path; its low-cash
behaviour is covered by tests/test_metals_swing_sizing.py and exercised
through that path. This file focuses on crypto/oil where the sizing path
is the single _place_buy block.
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


def _warrant(ask=10.0, direction="LONG"):
    return {"name": "TEST", "ob_id": "9999", "underlying": "BTC-USD",
            "direction": direction, "leverage": 1.0, "barrier": None,
            "parity": 1, "ask": ask, "last": ask}


@pytest.mark.parametrize("cash,expected_units_min", [
    # Low-cash (5000 < 10_000 threshold): alloc = min(MIN_TRADE_SEK=1000, 5000*0.95=4750) = 1000
    # units = int(1000/10) = 100
    (5000.0, 100),
])
def test_crypto_low_cash_uses_min_trade_sek_as_position_size(
        crypto_isolate, cash, expected_units_min):
    trader = cst.CryptoSwingTrader(page=None, executor=None)
    trader.state["cash_sek"] = cash
    res = trader._place_buy("BTC-USD", _warrant(ask=10.0), {}, 70000.0)
    assert res["executed"], f"expected executed, got {res}"
    pos_id = res["pos_id"]
    pos = trader.state["positions"][pos_id]
    # MIN_TRADE_SEK=1000, warrant_ask=10 -> 100 units; with cash=5000 the
    # 95% cap is 4750 so MIN_TRADE_SEK clears it.
    assert pos["units"] == expected_units_min


def test_crypto_normal_cash_uses_pct_sizing(crypto_isolate):
    """cash=20000, raw_alloc = 20000 * 25% = 5000 → budget=5000."""
    trader = cst.CryptoSwingTrader(page=None, executor=None)
    trader.state["cash_sek"] = 20000.0
    res = trader._place_buy("BTC-USD", _warrant(ask=10.0), {}, 70000.0)
    assert res["executed"]
    pos = trader.state["positions"][res["pos_id"]]
    # 20000 * 25% = 5000 → units = int(5000/10) = 500
    assert pos["units"] == 500


def test_crypto_normal_cash_small_raw_alloc_floored_to_min(crypto_isolate):
    """raw_alloc < MIN_TRADE_SEK gets floored to MIN_TRADE_SEK in normal-cash mode.

    Use POSITION_SIZE_PCT=25%, set cash such that 25% of cash < 1000 but
    cash >= LOW_CASH_THRESHOLD (10_000). 11_000 * 25% = 2750 which is
    above MIN_TRADE_SEK, so we need a custom POSITION_SIZE_PCT override.
    Instead, simulate: cash=12000 with overridden POSITION_SIZE_PCT.
    """
    # Manually set POSITION_SIZE_PCT lower for this test to force raw_alloc < MIN_TRADE_SEK.
    trader = cst.CryptoSwingTrader(page=None, executor=None)
    trader.state["cash_sek"] = 12000.0
    import unittest.mock as _mock
    with _mock.patch.object(crypto_cfg, "POSITION_SIZE_PCT", 5):
        # raw_alloc = 12000 * 5% = 600 → max(600, 1000) = 1000 (floor)
        res = trader._place_buy("BTC-USD", _warrant(ask=10.0), {}, 70000.0)
    assert res["executed"]
    pos = trader.state["positions"][res["pos_id"]]
    # budget = max(600, 1000) = 1000 → units = int(1000/10) = 100
    assert pos["units"] == 100


# --- Oil mirror tests ----------------------------------------------------

def _oil_warrant(ask=20.0):
    return {"name": "MINI L OLJA", "ob_id": "2370189",
            "underlying": "OIL-USD", "direction": "LONG", "leverage": 1.52,
            "barrier": None, "parity": 1, "ask": ask, "last": ask}


def test_oil_low_cash_uses_min_trade_sek(oil_isolate):
    trader = ost.OilSwingTrader(page=None, executor=None)
    trader.state["cash_sek"] = 5000.0
    res = trader._place_buy("OIL-USD", _oil_warrant(ask=20.0), {}, 70.0)
    assert res["executed"], f"expected executed, got {res}"
    pos = trader.state["positions"][res["pos_id"]]
    # MIN_TRADE_SEK=1000, warrant_ask=20 → 50 units
    assert pos["units"] == 50


def test_oil_normal_cash_uses_pct_sizing(oil_isolate):
    trader = ost.OilSwingTrader(page=None, executor=None)
    trader.state["cash_sek"] = 20000.0
    res = trader._place_buy("OIL-USD", _oil_warrant(ask=20.0), {}, 70.0)
    assert res["executed"]
    pos = trader.state["positions"][res["pos_id"]]
    # 20000 * 25% = 5000 → units = int(5000/20) = 250
    assert pos["units"] == 250


# --- Codex fix B 2026-05-11: warrant-side loss increments cooldown -------
#
# Exit triggers fire on warrant P&L (HARD_STOP at warrant_pct_change <=
# -SL_PCT). The cooldown counter must use the SAME basis — otherwise a
# warrant-side stop-out where the underlying is up RESETS consecutive_losses
# to 0 instead of escalating cooldown, which is exactly what we don't want
# after the warrant blew through its stop.

def test_crypto_warrant_stop_increments_consecutive_losses(crypto_isolate):
    """A warrant-side stop-loss (warrant -30% while underlying flat or +1%)
    must increment consecutive_losses by exactly 1.

    Codex fix B 2026-05-11: before the fix, the counter used underlying
    P&L — a warrant SL with the underlying up would RESET the counter."""
    trader = cst.CryptoSwingTrader(page=None, executor=None)
    trader.state["cash_sek"] = 50000.0
    trader.state["consecutive_losses"] = 0

    # Place a 5x position. underlying=70000, warrant=10.0.
    warrant_5x = {**_warrant(ask=10.0), "leverage": 5.0}
    res = trader._place_buy("BTC-USD", warrant_5x, {}, 70000.0)
    assert res["executed"]
    pos_id = res["pos_id"]
    pos = trader.state["positions"][pos_id]

    # Simulate a warrant-side stop-out: warrant 10.0 → 7.0 (-30%) while
    # underlying is roughly unchanged (slight up, even).
    sell_res = trader._place_sell(
        pos_id, pos,
        current_underlying=70700.0,    # +1% on underlying (NOT a loss)
        current_warrant_bid=7.0,        # -30% on warrant (HARD STOP)
        reason="HARD_STOP warrant -30.00%",
    )
    assert sell_res["executed"]
    # consecutive_losses MUST advance to 1 — basis is warrant P&L,
    # which is -30% (loss).
    assert trader.state["consecutive_losses"] == 1, (
        f"warrant SL must increment consecutive_losses by 1; "
        f"got {trader.state['consecutive_losses']} "
        f"(underlying +1% but warrant -30% — counter must use warrant basis)"
    )


def test_oil_warrant_stop_increments_consecutive_losses(oil_isolate):
    """Mirror of crypto test for oil swing trader."""
    trader = ost.OilSwingTrader(page=None, executor=None)
    trader.state["cash_sek"] = 50000.0
    trader.state["consecutive_losses"] = 0

    warrant_5x = {**_oil_warrant(ask=20.0), "leverage": 5.0}
    res = trader._place_buy("OIL-USD", warrant_5x, {}, 70.0)
    assert res["executed"]
    pos_id = res["pos_id"]
    pos = trader.state["positions"][pos_id]

    sell_res = trader._place_sell(
        pos_id, pos,
        current_underlying=70.7,    # +1% underlying
        current_warrant_bid=14.0,    # -30% warrant
        reason="HARD_STOP warrant -30.00%",
    )
    assert sell_res["executed"]
    assert trader.state["consecutive_losses"] == 1, (
        f"oil warrant SL must increment consecutive_losses by 1; "
        f"got {trader.state['consecutive_losses']}"
    )


def test_crypto_warrant_profit_resets_consecutive_losses(crypto_isolate):
    """Inverse: a warrant-side profit (warrant +5%) resets the counter to 0
    even if underlying happened to be flat or down."""
    trader = cst.CryptoSwingTrader(page=None, executor=None)
    trader.state["cash_sek"] = 50000.0
    trader.state["consecutive_losses"] = 2  # prior loss streak

    warrant_5x = {**_warrant(ask=10.0), "leverage": 5.0}
    res = trader._place_buy("BTC-USD", warrant_5x, {}, 70000.0)
    assert res["executed"]
    pos_id = res["pos_id"]
    pos = trader.state["positions"][pos_id]

    sell_res = trader._place_sell(
        pos_id, pos,
        current_underlying=69930.0,   # -0.1% underlying
        current_warrant_bid=10.5,      # +5% warrant (TP)
        reason="TAKE_PROFIT warrant +5.00%",
    )
    assert sell_res["executed"]
    assert trader.state["consecutive_losses"] == 0, (
        f"warrant profit must reset cooldown counter; got "
        f"{trader.state['consecutive_losses']}"
    )
