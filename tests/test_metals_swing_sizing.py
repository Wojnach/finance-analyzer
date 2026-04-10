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
    # 2026-04-10 adversarial review round 2 attrs — tests bypass __init__
    # via __new__, so we must set these explicitly or _check_entries
    # AttributeErrors out before reaching the tested branch.
    trader._jit_sync_tick = -1
    trader.kelly_no_edge_count = {}
    # Stub _sync_cash so the JIT sync branch doesn't hit real Avanza.
    # Tests that need to observe _sync_cash calls can override this stub.
    trader._sync_cash = lambda: None
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


# ---------------------------------------------------------------------------
# 2026-04-10 adversarial review round 2: L3 / S3 / T1 / T4 / T5 / T6
# ---------------------------------------------------------------------------


def test_max_hold_hours_is_safety_net():
    """T5: MAX_HOLD_HOURS should be a safety net, not a primary exit.

    User explicitly removed the 5h time limit in favor of EOD-only exit
    (commit 2a65d21). If a future change tightens this back below 12h,
    this test breaks and forces a deliberate review.
    """
    from metals_swing_config import MAX_HOLD_HOURS
    assert MAX_HOLD_HOURS >= 12, (
        f"MAX_HOLD_HOURS={MAX_HOLD_HOURS} too low. User wants EOD-only "
        f"forced exit; this constant should be a 12+ hour safety net, "
        f"not a primary time-based exit rule. See commits 2a65d21 and "
        f"3844ace for context."
    )


def test_jit_cash_sync_fires_before_kelly(monkeypatch):
    """L3/T4: _check_entries must call _sync_cash before Kelly so sizing
    uses the freshest buying_power, not 30-min-stale state."""
    trader = _make_trader(cash=5000)
    _capture_buy(trader)

    # Track order of calls
    call_order: list[str] = []

    original_sync = trader._sync_cash

    def _tracking_sync():
        call_order.append("sync")
        original_sync()

    trader._sync_cash = _tracking_sync

    def _tracking_kelly(**kwargs):
        call_order.append("kelly")
        return {"position_sek": 1500.0, "half_kelly_pct": 0.1, "win_rate": 0.55, "units": 107}

    monkeypatch.setattr("portfolio.kelly_metals.recommended_metals_size", _tracking_kelly)

    # First tick: sync should fire before Kelly
    trader._check_entries(prices={}, signal_data=_signal_buy_xag())
    assert "sync" in call_order, "_sync_cash was not called"
    assert "kelly" in call_order, "Kelly was not called"
    assert call_order.index("sync") < call_order.index("kelly"), (
        f"Sync must fire BEFORE Kelly, got order={call_order}"
    )


def test_jit_cash_sync_only_once_per_tick(monkeypatch):
    """L3: multiple tickers passing _evaluate_entry in the same tick must
    only trigger ONE sync — not one per ticker."""
    trader = _make_trader(cash=5000)
    _capture_buy(trader)
    # Force the trader to have a stale tick so JIT sync fires
    trader._jit_sync_tick = -1

    sync_count = [0]
    original_sync = trader._sync_cash

    def _counting_sync():
        sync_count[0] += 1
        original_sync()

    trader._sync_cash = _counting_sync

    monkeypatch.setattr(
        "portfolio.kelly_metals.recommended_metals_size",
        lambda **kwargs: {"position_sek": 1500.0, "half_kelly_pct": 0.1, "win_rate": 0.55, "units": 107},
    )

    # Simulate both XAG and XAU passing _evaluate_entry
    signal = {
        "XAG-USD": _signal_buy_xag()["XAG-USD"],
        "XAU-USD": _signal_buy_xag()["XAG-USD"],  # same shape, different ticker
    }
    trader.check_count = 5  # Make sure _jit_sync_tick comparison fires
    trader._check_entries(prices={}, signal_data=signal)
    assert sync_count[0] == 1, f"Expected 1 sync, got {sync_count[0]}"


def test_kelly_no_edge_counter_increments(monkeypatch):
    """S3: kelly_no_edge_count tracks consecutive Kelly-rejected tickers."""
    trader = _make_trader(cash=5000)
    _capture_buy(trader)

    def _no_edge_kelly(**kwargs):
        return {"position_sek": 0.0, "half_kelly_pct": 0.01, "win_rate": 0.48}

    monkeypatch.setattr("portfolio.kelly_metals.recommended_metals_size", _no_edge_kelly)

    assert trader.kelly_no_edge_count == {}
    trader._check_entries(prices={}, signal_data=_signal_buy_xag())
    # XAG got no-edge rejected
    assert trader.kelly_no_edge_count.get("XAG-USD") == 1

    # Next tick: counter increments
    trader.check_count += 1
    trader._check_entries(prices={}, signal_data=_signal_buy_xag())
    assert trader.kelly_no_edge_count.get("XAG-USD") == 2


def test_kelly_no_edge_counter_resets_on_successful_buy(monkeypatch):
    """S3: a successful BUY should clear the no-edge counter for that ticker."""
    trader = _make_trader(cash=5000)
    _capture_buy(trader)
    trader.kelly_no_edge_count["XAG-USD"] = 5  # simulate prior streak

    def _good_kelly(**kwargs):
        return {"position_sek": 1500.0, "half_kelly_pct": 0.12, "win_rate": 0.58, "units": 107}

    monkeypatch.setattr("portfolio.kelly_metals.recommended_metals_size", _good_kelly)

    trader._check_entries(prices={}, signal_data=_signal_buy_xag())
    assert "XAG-USD" not in trader.kelly_no_edge_count, (
        "Successful BUY should clear the no-edge counter"
    )


def test_kelly_no_edge_hourly_telegram_fires(monkeypatch):
    """S3: when check_count % TELEGRAM_NO_EDGE_INTERVAL == 0 and the
    no-edge counter is non-zero, a Telegram summary should be sent."""
    import metals_swing_trader as mst

    trader = _make_trader(cash=5000)
    _capture_buy(trader)
    trader.check_count = mst.TELEGRAM_NO_EDGE_INTERVAL  # triggers the summary
    trader.kelly_no_edge_count = {"XAG-USD": 42, "XAU-USD": 38}

    telegrams_sent: list[str] = []
    monkeypatch.setattr(mst, "_send_telegram", lambda msg: telegrams_sent.append(msg))

    # Kelly returns no-edge so nothing executes but the summary still runs
    monkeypatch.setattr(
        "portfolio.kelly_metals.recommended_metals_size",
        lambda **kwargs: {"position_sek": 0.0, "half_kelly_pct": 0.01, "win_rate": 0.48},
    )

    trader._check_entries(prices={}, signal_data=_signal_buy_xag())

    assert any("Kelly rejected" in m for m in telegrams_sent), (
        f"Expected 'Kelly rejected' telegram, got: {telegrams_sent}"
    )
    # Counter should be cleared after summary fires (but may be incremented by the current tick)
    # The summary happens AFTER the per-ticker loop, so the current tick's
    # increments get cleared too.
    assert trader.kelly_no_edge_count == {}


def test_kelly_no_edge_telegram_silent_when_counter_empty(monkeypatch):
    """S3: summary must NOT fire when no tickers have a no-edge streak."""
    import metals_swing_trader as mst

    trader = _make_trader(cash=5000)
    _capture_buy(trader)
    trader.check_count = mst.TELEGRAM_NO_EDGE_INTERVAL
    # Empty counter — no-edge summary should be silent
    trader.kelly_no_edge_count = {}

    telegrams_sent: list[str] = []
    monkeypatch.setattr(mst, "_send_telegram", lambda msg: telegrams_sent.append(msg))

    # Kelly succeeds
    monkeypatch.setattr(
        "portfolio.kelly_metals.recommended_metals_size",
        lambda **kwargs: {"position_sek": 1500.0, "half_kelly_pct": 0.12, "win_rate": 0.58, "units": 107},
    )

    trader._check_entries(prices={}, signal_data=_signal_buy_xag())

    no_edge_telegrams = [m for m in telegrams_sent if "Kelly rejected" in m]
    assert no_edge_telegrams == [], (
        f"No-edge telegram should not fire on empty counter, got: {no_edge_telegrams}"
    )


def test_agent_summary_wrapped_reaches_real_kelly_weighted_confidence_path(monkeypatch):
    """T1: end-to-end integration test for B2.

    Forces Kelly's first two win-rate sources (accuracy cache, signal_log.db)
    to return nothing, then verifies that the REAL recommended_metals_size
    (not a stub) successfully reaches and uses the weighted_confidence from
    our wrapped signal_data.

    If B2 regresses (someone unwraps agent_summary), Kelly will fall
    through to _DEFAULT_WIN_RATE=0.52 and this test catches it.
    """
    import portfolio.kelly_metals as km

    trader = _make_trader(cash=5000)
    _capture_buy(trader)

    # Disable the first two win-rate sources so Kelly MUST use agent_summary
    monkeypatch.setattr(km, "_get_ticker_accuracy", lambda t: None)
    monkeypatch.setattr(km, "_get_outcome_stats", lambda t, horizon: None)

    # Track what win_rate Kelly ended up using
    real_kelly = km.recommended_metals_size
    used_win_rates: list[float] = []

    def _tracking_kelly(**kwargs):
        result = real_kelly(**kwargs)
        used_win_rates.append(float(result.get("win_rate", 0)))
        return result

    monkeypatch.setattr(
        "portfolio.kelly_metals.recommended_metals_size",
        _tracking_kelly,
    )

    # Signal with a specific weighted_confidence that Kelly should pick up.
    signal = {
        "XAG-USD": {
            "action": "BUY",
            "buy_count": 7,
            "sell_count": 4,
            "confidence": 0.77,
            "weighted_confidence": 0.65,  # Kelly should use this
            "rsi": 46.4,
            "macd": -0.05,
            "regime": "ranging",
        }
    }

    trader._check_entries(prices={}, signal_data=signal)

    assert len(used_win_rates) >= 1, "Kelly should have been called at least once"
    actual_wr = used_win_rates[0]
    # If B2 regresses, agent_summary is unwrapped and Kelly can't find the
    # ticker in signal_data.get("signals", {}) → falls through to default 0.52.
    # With the fix, Kelly finds XAG-USD via agent_summary["signals"]["XAG-USD"]
    # and uses weighted_confidence=0.65.
    assert actual_wr == 0.65, (
        f"Kelly should have used weighted_confidence=0.65 from wrapped "
        f"signal_data, got {actual_wr}. If 0.52, agent_summary wrapping "
        f"is broken again (B2 regression)."
    )


def test_eod_exit_fires_at_configured_buffer(monkeypatch):
    """T6: _check_exits must force-sell positions when minutes_to_close
    drops below EOD_EXIT_MINUTES_BEFORE.

    This pins the wall-clock behavior of the EOD exit rule. With buffer=25
    and hardcoded close_cet=21:55, EOD should fire at any _cet_hour >= 21.5.
    """
    import metals_swing_trader as mst
    from metals_swing_config import EOD_EXIT_MINUTES_BEFORE

    trader = _make_trader(cash=5000)

    # Seed a position that would otherwise sail through all exit rules
    trader.state["positions"]["pos_eod_test"] = {
        "warrant_key": "MINI_L_SILVER_AVA_TEST",
        "warrant_name": "MINI L SILVER AVA TEST",
        "ob_id": "9999999",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "units": 100,
        "entry_price": 10.0,
        "entry_underlying": 75.0,
        "entry_ts": mst._now_utc().isoformat(),
        "peak_underlying": 75.0,
        "trough_underlying": 75.0,
        "trailing_active": False,
        "stop_order_id": None,
        "leverage": 5.0,
        "fill_verified": True,
    }

    # Capture the sell call instead of hitting Avanza.
    # Real signature: _execute_sell(pos_id, pos, current_bid, underlying_price, reason)
    sell_calls: list[tuple] = []

    def _mock_sell(pos_id, pos, current_bid, underlying_price, reason):
        sell_calls.append((pos_id, reason, current_bid))
        trader.state["positions"].pop(pos_id, None)

    trader._execute_sell = _mock_sell

    # Stub fetch_price so _check_exits can get a current price
    monkeypatch.setattr(
        mst, "fetch_price",
        lambda page, ob_id, api_type: {"bid": 10.5, "ask": 10.6, "underlying": 75.0},
    )

    # Inside-buffer hour: 21:55 - 20min = 21:35 CET → minutes_to_close = 20,
    # which is < EOD_EXIT_MINUTES_BEFORE (25) → EOD should fire.
    close_cet = 21.0 + 55 / 60
    inside_buffer_hour = close_cet - (EOD_EXIT_MINUTES_BEFORE - 5) / 60.0

    monkeypatch.setattr(mst, "_cet_hour", lambda: inside_buffer_hour)

    # Use a HOLD signal so SIGNAL_REVERSAL doesn't fire before EOD
    signal = {"XAG-USD": {"action": "HOLD", "buy_count": 0, "sell_count": 0,
                          "confidence": 0.3, "rsi": 50, "regime": "ranging"}}

    trader._check_exits({}, signal)

    # Should have fired EOD exit
    assert len(sell_calls) >= 1, (
        f"Expected EOD_EXIT to fire at hour={inside_buffer_hour:.3f} "
        f"(minutes_to_close={((close_cet - inside_buffer_hour) * 60):.0f}, "
        f"buffer={EOD_EXIT_MINUTES_BEFORE}), got no sells"
    )
    reasons = [reason for (_, reason, _) in sell_calls]
    assert any("EOD" in str(r) for r in reasons), (
        f"Expected EOD in sell reason, got: {reasons}"
    )


def test_eod_exit_does_not_fire_outside_buffer(monkeypatch):
    """T6: EOD must NOT fire when minutes_to_close > buffer (normal trading)."""
    import metals_swing_trader as mst

    trader = _make_trader(cash=5000)
    trader.state["positions"]["pos_eod_test"] = {
        "warrant_key": "MINI_L_SILVER_AVA_TEST",
        "warrant_name": "MINI L SILVER AVA TEST",
        "ob_id": "9999999",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "units": 100,
        "entry_price": 10.0,
        "entry_underlying": 75.0,
        "entry_ts": mst._now_utc().isoformat(),
        "peak_underlying": 75.0,
        "trough_underlying": 75.0,
        "trailing_active": False,
        "stop_order_id": None,
        "leverage": 5.0,
        "fill_verified": True,
    }

    sell_calls: list[tuple] = []

    def _mock_sell(pos_id, pos, reason, current_price, sig=None):
        sell_calls.append((pos_id, reason, current_price))
        trader.state["positions"].pop(pos_id, None)

    trader._execute_sell = _mock_sell
    monkeypatch.setattr(
        mst, "fetch_price",
        lambda page, ob_id, api_type: {"bid": 10.5, "ask": 10.6, "underlying": 75.0},
    )

    # Mid-session hour: 15:00 CET → minutes_to_close ≈ 415 min >> buffer
    monkeypatch.setattr(mst, "_cet_hour", lambda: 15.0)

    signal = {"XAG-USD": {"action": "HOLD", "buy_count": 0, "sell_count": 0,
                          "confidence": 0.3, "rsi": 50, "regime": "ranging"}}

    trader._check_exits({}, signal)

    eod_sells = [r for (_, r, _) in sell_calls if "EOD" in str(r)]
    assert eod_sells == [], (
        f"EOD should NOT fire at 15:00 CET (minutes_to_close={((21.9167 - 15.0) * 60):.0f}), "
        f"got: {eod_sells}"
    )
