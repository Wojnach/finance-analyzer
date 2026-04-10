"""Unit tests for the Kelly-based entry sizing path in metals_swing_trader.

Locks in the 2026-04-10 fix that replaces fixed POSITION_SIZE_PCT sizing
with Kelly-optimal sizing (floored at MIN_TRADE_SEK, capped at 95% cash,
falling back to fixed 30% on Kelly ImportError/Exception).

Tests bypass __init__ via __new__ and stub _evaluate_entry / _select_warrant
so only the sizing block is under test.
"""

from __future__ import annotations

import os
import sys
from typing import Any

# data/ contains script-style modules, not a package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import metals_swing_trader as mst


def _make_trader(cash: float = 2418.91, losses: int = 0):
    """Minimal SwingTrader instance with only the attributes _check_entries needs."""
    trader = mst.SwingTrader.__new__(mst.SwingTrader)
    trader.page = object()
    trader.state = mst._default_state()
    trader.state["cash_sek"] = cash
    trader.state["consecutive_losses"] = losses
    trader.cash_sync_ok = True
    trader.recon_failure_streak = 0
    trader.check_count = 0
    trader.regime_history = {}
    trader.warrant_catalog = {
        "MINI_L_SILVER_AVA_TEST": {
            "ob_id": "9999999",
            "name": "MINI L SILVER AVA TEST",
            "underlying": "XAG-USD",
            "direction": "LONG",
            "leverage": 5.0,
            "live_ask": 14.00,
        }
    }
    # Force entry gates open — we only want to test the sizing block.
    trader._evaluate_entry = lambda sig, ticker: (True, "")
    trader._select_warrant = lambda ticker, direction: {
        "key": "MINI_L_SILVER_AVA_TEST",
        "ob_id": "9999999",
        "name": "MINI L SILVER AVA TEST",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "leverage": 5.0,
        "live_ask": 14.00,
    }
    trader._has_position = lambda ticker: False
    trader._cooldown_cleared = lambda: True
    return trader


def _signal_buy_xag():
    return {
        "XAG-USD": {
            "action": "BUY",
            "buy_count": 7,
            "sell_count": 4,
            "confidence": 0.77,
            "weighted_confidence": 0.77,
            "rsi": 46.4,
            "macd": -0.05,
            "regime": "ranging",
        }
    }


def _capture_buy(trader):
    """Replace _execute_buy with a capturing stub."""
    captured: list[dict[str, Any]] = []

    def _stub(warrant, units, ask_price, underlying_ticker, sig, total_cost, direction):
        captured.append({
            "warrant": warrant,
            "units": units,
            "ask_price": ask_price,
            "underlying_ticker": underlying_ticker,
            "total_cost": total_cost,
            "direction": direction,
        })

    trader._execute_buy = _stub
    return captured


def test_kelly_alloc_used_when_cash_is_low(monkeypatch):
    """With cash=2418 and Kelly returning 1500 SEK, alloc should be 1500.

    Old behavior: alloc = 2418 * 0.30 = 725, below MIN_TRADE_SEK 1000 → SKIP.
    New behavior: Kelly recommends 1500 (floored at MIN_TRADE_SEK anyway),
    so the trade goes through.
    """
    trader = _make_trader(cash=2418.91)
    captured = _capture_buy(trader)

    def _fake_kelly(**kwargs):
        return {
            "position_sek": 1500.0,
            "half_kelly_pct": 0.12,
            "win_rate": 0.58,
            "units": 107,
        }

    # Patch where the import happens — inside the function body.
    monkeypatch.setattr(
        "portfolio.kelly_metals.recommended_metals_size",
        _fake_kelly,
    )

    trader._check_entries(prices={}, signal_data=_signal_buy_xag())

    assert len(captured) == 1, f"Expected one BUY, got {len(captured)}"
    # alloc should be max(1500, 1000) = 1500, capped at 2418*0.95=2297.96.
    # units = int(1500 / 14) = 107
    assert captured[0]["units"] == 107
    assert captured[0]["underlying_ticker"] == "XAG-USD"


def test_kelly_result_floored_at_min_trade_sek(monkeypatch):
    """If Kelly returns a recommendation below MIN_TRADE_SEK (1000), floor bumps it up."""
    trader = _make_trader(cash=2418.91)
    captured = _capture_buy(trader)

    def _fake_kelly(**kwargs):
        return {
            "position_sek": 700.0,  # Below swing's MIN_TRADE_SEK=1000
            "half_kelly_pct": 0.06,
            "win_rate": 0.52,
            "units": 50,
        }

    monkeypatch.setattr(
        "portfolio.kelly_metals.recommended_metals_size",
        _fake_kelly,
    )

    trader._check_entries(prices={}, signal_data=_signal_buy_xag())

    assert len(captured) == 1
    # alloc should be max(700, 1000) = 1000 → units = int(1000/14) = 71
    assert captured[0]["units"] == 71


def test_kelly_alloc_capped_at_95_percent_of_cash(monkeypatch):
    """If Kelly recommends more than 95% of cash, cap at the 95% ceiling."""
    trader = _make_trader(cash=2418.91)
    captured = _capture_buy(trader)

    def _fake_kelly(**kwargs):
        return {
            "position_sek": 2400.0,  # 99% of cash — should be capped at 95%
            "half_kelly_pct": 0.50,
            "win_rate": 0.75,
            "units": 171,
        }

    monkeypatch.setattr(
        "portfolio.kelly_metals.recommended_metals_size",
        _fake_kelly,
    )

    trader._check_entries(prices={}, signal_data=_signal_buy_xag())

    assert len(captured) == 1
    # alloc = min(max(2400, 1000), 2418.91 * 0.95 = 2297.96) = 2297.96
    # units = int(2297.96 / 14) = 164
    assert captured[0]["units"] == 164


def test_kelly_no_edge_skips_entry(monkeypatch):
    """If Kelly returns position_sek=0 (no edge), the entry is skipped."""
    trader = _make_trader(cash=5000)  # lots of cash — ensures the skip is Kelly-driven
    captured = _capture_buy(trader)

    def _fake_kelly(**kwargs):
        return {
            "position_sek": 0.0,  # Kelly: no edge
            "half_kelly_pct": 0.01,
            "win_rate": 0.48,
            "units": 0,
        }

    monkeypatch.setattr(
        "portfolio.kelly_metals.recommended_metals_size",
        _fake_kelly,
    )

    trader._check_entries(prices={}, signal_data=_signal_buy_xag())

    assert captured == [], "Should not have placed a BUY when Kelly returns no edge"


def test_kelly_import_error_falls_back_to_fixed_percent(monkeypatch):
    """If Kelly module raises, fall back to fixed POSITION_SIZE_PCT logic."""
    trader = _make_trader(cash=5000)  # 5000 * 30% = 1500, above MIN_TRADE_SEK 1000
    captured = _capture_buy(trader)

    def _raising_kelly(**kwargs):
        raise ImportError("mock: kelly_metals not available")

    monkeypatch.setattr(
        "portfolio.kelly_metals.recommended_metals_size",
        _raising_kelly,
    )

    trader._check_entries(prices={}, signal_data=_signal_buy_xag())

    assert len(captured) == 1
    # Fallback: alloc = 5000 * 30/100 = 1500, units = int(1500/14) = 107
    assert captured[0]["units"] == 107


def test_insufficient_cash_when_even_kelly_below_floor(monkeypatch):
    """With cash so low that alloc<MIN_TRADE_SEK even after flooring, skip with log."""
    trader = _make_trader(cash=800)  # Below MIN_TRADE_SEK floor entirely
    captured = _capture_buy(trader)

    def _fake_kelly(**kwargs):
        # Kelly recommends a small position; floor bumps it to 1000, cap at 760
        # (800 * 0.95). Result: 760, below MIN_TRADE_SEK → skip.
        return {
            "position_sek": 400.0,
            "half_kelly_pct": 0.05,
            "win_rate": 0.52,
            "units": 28,
        }

    monkeypatch.setattr(
        "portfolio.kelly_metals.recommended_metals_size",
        _fake_kelly,
    )

    trader._check_entries(prices={}, signal_data=_signal_buy_xag())

    assert captured == [], "Should skip when alloc < MIN_TRADE_SEK after cap"


def test_insufficient_cash_log_shows_alloc_not_cash(monkeypatch, caplog):
    """The log message must include both cash and alloc values, not just cash."""
    import logging
    trader = _make_trader(cash=800)
    _capture_buy(trader)

    def _fake_kelly(**kwargs):
        return {
            "position_sek": 400.0,  # floor→1000, capped at 760 → fails min
            "half_kelly_pct": 0.05,
            "win_rate": 0.52,
            "units": 28,
        }

    monkeypatch.setattr(
        "portfolio.kelly_metals.recommended_metals_size",
        _fake_kelly,
    )

    with caplog.at_level(logging.INFO, logger="metals_loop.swing_trader"):
        trader._check_entries(prices={}, signal_data=_signal_buy_xag())

    msgs = [rec.getMessage() for rec in caplog.records]
    insufficient = [m for m in msgs if "Insufficient cash" in m]
    assert len(insufficient) >= 1, f"No 'Insufficient cash' log, got: {msgs}"
    msg = insufficient[0]
    assert "cash=800" in msg
    assert "alloc=" in msg
    assert "min=1000" in msg
