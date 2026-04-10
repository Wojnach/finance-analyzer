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

    def _stub(warrant, units, ask_price, underlying_ticker, sig, total_cost, direction, kelly_rec=None):
        captured.append({
            "warrant": warrant,
            "units": units,
            "ask_price": ask_price,
            "underlying_ticker": underlying_ticker,
            "total_cost": total_cost,
            "direction": direction,
            "kelly_rec": kelly_rec,
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


# ---------------------------------------------------------------------------
# 2026-04-10 adversarial review follow-ups
# ---------------------------------------------------------------------------


def test_cash_zero_early_return(monkeypatch, caplog):
    """L1: cash <= 0 should short-circuit before Kelly is even called."""
    import logging
    trader = _make_trader(cash=0)
    captured = _capture_buy(trader)

    kelly_called = []

    def _tracking_kelly(**kwargs):
        kelly_called.append(kwargs)
        return {"position_sek": 1500.0, "half_kelly_pct": 0.1, "win_rate": 0.6, "units": 100}

    monkeypatch.setattr("portfolio.kelly_metals.recommended_metals_size", _tracking_kelly)

    with caplog.at_level(logging.INFO, logger="metals_loop.swing_trader"):
        trader._check_entries(prices={}, signal_data=_signal_buy_xag())

    assert captured == [], "Should not buy with zero cash"
    assert kelly_called == [], "Kelly should not even be called with zero cash"
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("non-positive cash" in m for m in msgs), f"missing zero-cash log: {msgs}"


def test_cash_negative_early_return(monkeypatch):
    """L1: cash < 0 (overdraft) should also short-circuit."""
    trader = _make_trader(cash=-100)
    captured = _capture_buy(trader)
    monkeypatch.setattr(
        "portfolio.kelly_metals.recommended_metals_size",
        lambda **kwargs: {"position_sek": 500.0, "half_kelly_pct": 0.05, "win_rate": 0.52, "units": 30},
    )
    trader._check_entries(prices={}, signal_data=_signal_buy_xag())
    assert captured == [], "Should not buy with negative cash"


def test_kelly_malformed_dict_missing_keys_falls_back(monkeypatch):
    """S2: Kelly returning a dict missing required keys should trigger fallback."""
    trader = _make_trader(cash=5000)
    captured = _capture_buy(trader)

    def _bad_kelly(**kwargs):
        # Missing 'half_kelly_pct' and 'win_rate' — should be treated as malformed.
        return {"position_sek": 1200.0}

    monkeypatch.setattr("portfolio.kelly_metals.recommended_metals_size", _bad_kelly)
    trader._check_entries(prices={}, signal_data=_signal_buy_xag())
    # Falls back to fixed 30%: 5000 * 0.30 = 1500 SEK, units = int(1500/14) = 107.
    assert len(captured) == 1
    assert captured[0]["units"] == 107


def test_kelly_nan_position_sek_falls_back(monkeypatch):
    """S2: Kelly returning NaN for position_sek should trigger fallback, not propagate."""
    import math
    trader = _make_trader(cash=5000)
    captured = _capture_buy(trader)

    def _nan_kelly(**kwargs):
        return {
            "position_sek": float("nan"),
            "half_kelly_pct": 0.1,
            "win_rate": 0.55,
            "units": 50,
        }

    monkeypatch.setattr("portfolio.kelly_metals.recommended_metals_size", _nan_kelly)
    trader._check_entries(prices={}, signal_data=_signal_buy_xag())
    # Falls back to fixed 30%: 5000 * 0.30 = 1500 SEK → 107 units.
    assert len(captured) == 1, "Should have bought via fallback, not propagated NaN"
    assert not math.isnan(captured[0]["total_cost"])


def test_kelly_inf_position_sek_falls_back(monkeypatch):
    """S2: Kelly returning +inf for position_sek should trigger fallback."""
    trader = _make_trader(cash=5000)
    captured = _capture_buy(trader)

    def _inf_kelly(**kwargs):
        return {
            "position_sek": float("inf"),
            "half_kelly_pct": 0.1,
            "win_rate": 0.55,
            "units": 50,
        }

    monkeypatch.setattr("portfolio.kelly_metals.recommended_metals_size", _inf_kelly)
    trader._check_entries(prices={}, signal_data=_signal_buy_xag())
    # Should fall back to 30% sizing, not propagate inf.
    assert len(captured) == 1
    assert captured[0]["total_cost"] < 1e9, "inf leaked through to total_cost"


def test_kelly_position_sek_string_falls_back(monkeypatch):
    """S2: Kelly returning a string where a number is expected → fallback."""
    trader = _make_trader(cash=5000)
    captured = _capture_buy(trader)

    def _string_kelly(**kwargs):
        return {
            "position_sek": "not a number",
            "half_kelly_pct": 0.1,
            "win_rate": 0.55,
            "units": 50,
        }

    monkeypatch.setattr("portfolio.kelly_metals.recommended_metals_size", _string_kelly)
    trader._check_entries(prices={}, signal_data=_signal_buy_xag())
    assert len(captured) == 1, "Should have fallback-bought despite string position_sek"


def test_kelly_runtime_error_falls_back_and_logs_error(monkeypatch, caplog):
    """S1: non-ImportError runtime failure should fall back AND log at ERROR."""
    import logging
    trader = _make_trader(cash=5000)
    captured = _capture_buy(trader)

    def _boom_kelly(**kwargs):
        raise ValueError("mock: simulated kelly bug")

    monkeypatch.setattr("portfolio.kelly_metals.recommended_metals_size", _boom_kelly)

    with caplog.at_level(logging.WARNING, logger="metals_loop.swing_trader"):
        trader._check_entries(prices={}, signal_data=_signal_buy_xag())

    # Fallback path should still place the order
    assert len(captured) == 1
    # Must be logged at ERROR level (not just WARNING) for runtime regressions
    error_records = [r for r in caplog.records if r.levelname == "ERROR"]
    assert any("Kelly runtime error" in r.getMessage() for r in error_records), (
        f"Expected ERROR log with 'Kelly runtime error', got: "
        f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )


def test_ask_price_zero_logs_and_skips(monkeypatch, caplog):
    """S4: invalid ask_price must log, not silently continue."""
    import logging
    trader = _make_trader(cash=5000)
    captured = _capture_buy(trader)

    # Force _select_warrant to return a warrant with ask=0
    def _bad_warrant(ticker, direction):
        return {
            "key": "MINI_L_SILVER_AVA_TEST",
            "ob_id": "9999999",
            "name": "MINI L SILVER AVA TEST",
            "underlying": "XAG-USD",
            "direction": "LONG",
            "leverage": 5.0,
            "live_leverage": 5.0,
            "live_ask": 0,  # invalid
        }

    trader._select_warrant = _bad_warrant

    with caplog.at_level(logging.INFO, logger="metals_loop.swing_trader"):
        trader._check_entries(prices={}, signal_data=_signal_buy_xag())

    assert captured == [], "Should not buy with ask=0"
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("invalid ask_price" in m for m in msgs), f"missing ask_price skip log: {msgs}"


def test_agent_summary_wrapped_correctly(monkeypatch):
    """B2: agent_summary must be wrapped as {'signals': signal_data} not raw."""
    trader = _make_trader(cash=5000)
    _capture_buy(trader)

    received_kwargs = {}

    def _capture_kwargs(**kwargs):
        received_kwargs.update(kwargs)
        return {"position_sek": 1500.0, "half_kelly_pct": 0.1, "win_rate": 0.55, "units": 107}

    monkeypatch.setattr("portfolio.kelly_metals.recommended_metals_size", _capture_kwargs)
    trader._check_entries(prices={}, signal_data=_signal_buy_xag())

    # Must be wrapped so kelly can do agent_summary["signals"][ticker].
    agent_summary = received_kwargs.get("agent_summary")
    assert isinstance(agent_summary, dict), "agent_summary must be a dict"
    assert "signals" in agent_summary, (
        "agent_summary must have 'signals' key so kelly_metals can find tickers"
    )
    assert "XAG-USD" in agent_summary["signals"], (
        "signals dict must contain XAG-USD"
    )


def test_live_leverage_preferred_over_catalog(monkeypatch):
    """B3: Kelly must receive live_leverage, not the stale catalog leverage."""
    trader = _make_trader(cash=5000)
    _capture_buy(trader)

    # Warrant with DIFFERENT live_leverage vs catalog leverage
    def _warrant_with_drift(ticker, direction):
        return {
            "key": "MINI_L_SILVER_AVA_TEST",
            "ob_id": "9999999",
            "name": "MINI L SILVER AVA TEST",
            "underlying": "XAG-USD",
            "direction": "LONG",
            "leverage": 5.0,       # catalog
            "live_leverage": 7.5,  # live (drifted)
            "live_ask": 14.00,
        }

    trader._select_warrant = _warrant_with_drift

    received_kwargs = {}

    def _capture_kwargs(**kwargs):
        received_kwargs.update(kwargs)
        return {"position_sek": 1500.0, "half_kelly_pct": 0.1, "win_rate": 0.55, "units": 107}

    monkeypatch.setattr("portfolio.kelly_metals.recommended_metals_size", _capture_kwargs)
    trader._check_entries(prices={}, signal_data=_signal_buy_xag())

    assert received_kwargs.get("leverage") == 7.5, (
        f"Kelly should receive live_leverage=7.5, got {received_kwargs.get('leverage')}"
    )


def test_trade_record_includes_kelly_metadata(monkeypatch):
    """LG2: _execute_buy must attach Kelly metadata to the trade record."""
    trader = _make_trader(cash=5000)
    captured = _capture_buy(trader)

    def _fake_kelly(**kwargs):
        return {
            "position_sek": 1500.0,
            "half_kelly_pct": 0.12,
            "win_rate": 0.58,
            "units": 107,
            "source": "test-source",
            "avg_win_pct": 3.0,
            "avg_loss_pct": 2.5,
            "consecutive_losses": 0,
        }

    monkeypatch.setattr("portfolio.kelly_metals.recommended_metals_size", _fake_kelly)
    trader._check_entries(prices={}, signal_data=_signal_buy_xag())

    assert len(captured) == 1
    kelly_rec = captured[0]["kelly_rec"]
    assert kelly_rec is not None
    assert kelly_rec["half_kelly_pct"] == 0.12
    assert kelly_rec["win_rate"] == 0.58
    assert kelly_rec["source"] == "test-source"
