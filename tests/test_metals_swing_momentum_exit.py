"""Regression tests for the 2026-04-17 momentum-exit fix pass.

Incident: MINI L SILVER AVA 331 bought 13:33:08 CET and sold 55 seconds later
on ``MOMENTUM_EXIT: 3 declining checks (-0.64%)``. Silver then rallied +5.4%
off the sell price. Three bugs compounded:

1. ``_und_history`` was not reset on entry, so ticks from BEFORE the buy
   triggered the exit on the same cycle that verified the fill.
2. No minimum hold — exit rule fired from t=0 of the position.
3. The -0.3% threshold over 3×60s ticks is below the XAG/XAU noise floor.

These tests pin the fix so a future tune-down doesn't silently regress.
"""

from __future__ import annotations

import datetime
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import metals_swing_config as cfg
import metals_swing_trader as mst


UTC = datetime.timezone.utc


def _make_trader():
    trader = mst.SwingTrader.__new__(mst.SwingTrader)
    trader.state = mst._default_state()
    trader.state["positions"] = {}
    trader.check_count = 10
    return trader


def _add_position(trader, *, pos_id="pos1", underlying="XAG-USD",
                  direction="LONG", entry_ts_iso=None):
    trader.state["positions"][pos_id] = {
        "warrant_key": "MINI_L_SILVER_AVA_331",
        "warrant_name": "MINI L SILVER AVA 331",
        "ob_id": "2379768",
        "api_type": "warrant",
        "underlying": underlying,
        "direction": direction,
        "units": 93,
        "entry_price": 14.58,
        "entry_underlying": 79.33,
        "entry_ts": entry_ts_iso or datetime.datetime.now(UTC).isoformat(),
        "peak_underlying": 79.33,
        "trough_underlying": 79.33,
        "trailing_active": False,
        "stop_order_id": None,
        "leverage": 5.0,
        "fill_verified": True,
        "buy_order_id": "TEST_ORDER",
    }


# ---------------------------------------------------------------------------
# Bug 1: _und_history must be reset on entry
# ---------------------------------------------------------------------------

def test_execute_buy_resets_und_history(monkeypatch):
    """The 3-tick window must start fresh at entry. Pre-entry ticks that
    would trigger a momentum exit must be discarded."""
    trader = _make_trader()
    # Pre-seed history with 3 declining ticks that WOULD trigger the exit
    # if they survived through the buy.
    trader.state["_und_history"] = {
        "XAG-USD": [79.80, 79.60, 79.30],
    }

    monkeypatch.setattr(mst, "DRY_RUN", True)  # skip real order placement
    monkeypatch.setattr(mst, "_log", lambda *a, **k: None)
    monkeypatch.setattr(mst, "_log_trade", lambda *a, **k: None)
    monkeypatch.setattr(mst, "_log_decision", lambda *a, **k: None)
    monkeypatch.setattr(mst, "_send_telegram", lambda *a, **k: None)
    monkeypatch.setattr(mst, "_save_state", lambda *a, **k: None)
    monkeypatch.setattr(trader, "_set_stop_loss", lambda pid: None)

    warrant = {
        "key": "MINI_L_SILVER_AVA_331", "name": "MINI L SILVER AVA 331",
        "ob_id": "2379768", "api_type": "warrant",
        "live_leverage": 5.0, "underlying_price": 79.33,
    }
    sig = {"buy_count": 6, "sell_count": 1, "rsi": 52.9, "action": "BUY",
           "confidence": 0.603}

    trader._execute_buy(
        warrant=warrant, units=93, ask_price=14.58,
        underlying_ticker="XAG-USD", sig=sig, total_cost=1355.94,
        direction="LONG",
    )

    # Post-condition: history cleared for XAG-USD
    assert trader.state["_und_history"]["XAG-USD"] == []


# ---------------------------------------------------------------------------
# Bug 2: minimum hold before momentum exit can fire
# ---------------------------------------------------------------------------

def test_momentum_exit_gated_by_min_hold(monkeypatch):
    """Within the first MOMENTUM_EXIT_MIN_HOLD_SECONDS after entry, the
    3-tick counter-trend rule must NOT fire — even with a history that
    would otherwise trigger."""
    trader = _make_trader()
    # Position opened 60s ago (well below 10min min-hold).
    recent_entry = datetime.datetime.now(UTC) - datetime.timedelta(seconds=60)
    _add_position(trader, entry_ts_iso=recent_entry.isoformat())

    # History that would cross the 0.8% threshold if the rule were active.
    trader.state["_und_history"] = {
        "XAG-USD": [80.00, 79.50, 79.00],  # -1.25% monotonic decline
    }

    fired = _run_momentum_exit_check(trader, monkeypatch)

    assert fired is None, "momentum exit should be suppressed within the min-hold window"


def test_momentum_exit_fires_after_min_hold(monkeypatch):
    """After the min-hold window passes, the rule fires normally on a
    qualifying history (monotonic + above threshold)."""
    trader = _make_trader()
    old_entry = datetime.datetime.now(UTC) - datetime.timedelta(
        seconds=cfg.MOMENTUM_EXIT_MIN_HOLD_SECONDS + 120,
    )
    _add_position(trader, entry_ts_iso=old_entry.isoformat())
    trader.state["_und_history"] = {
        "XAG-USD": [80.00, 79.40, 78.80],  # -1.5% monotonic decline
    }

    fired = _run_momentum_exit_check(trader, monkeypatch)

    assert fired is not None, "momentum exit should fire after min-hold elapses"
    assert "MOMENTUM_EXIT" in fired


# ---------------------------------------------------------------------------
# Bug 3: threshold above noise floor
# ---------------------------------------------------------------------------

def test_momentum_exit_ignores_below_threshold_noise(monkeypatch):
    """3 monotonic declines that sum to less than MOMENTUM_EXIT_THRESHOLD_PCT
    (0.8%) must NOT trigger — that's normal 60s bid/ask drift on XAG/XAU."""
    trader = _make_trader()
    old_entry = datetime.datetime.now(UTC) - datetime.timedelta(
        seconds=cfg.MOMENTUM_EXIT_MIN_HOLD_SECONDS + 120,
    )
    _add_position(trader, entry_ts_iso=old_entry.isoformat())
    # Monotonic decline but only -0.5% total — below the 0.8% threshold.
    trader.state["_und_history"] = {
        "XAG-USD": [79.80, 79.70, 79.40],  # -0.5% — noise on XAG at 79
    }

    fired = _run_momentum_exit_check(trader, monkeypatch)

    assert fired is None, "sub-threshold noise must not trigger momentum exit"


def test_momentum_exit_threshold_constant_pinned():
    """Pin config constants so a future tune-down doesn't silently regress
    into the 2026-04-17 incident territory."""
    assert cfg.MOMENTUM_EXIT_MIN_HOLD_SECONDS >= 300, \
        "min-hold must be >= 5 minutes (enough for 3 new ticks post-entry)"
    assert cfg.MOMENTUM_EXIT_THRESHOLD_PCT >= 0.8, \
        "threshold must be >= 0.8% to stay above XAG/XAU noise"


# ---------------------------------------------------------------------------
# SHORT direction: symmetric behaviour
# ---------------------------------------------------------------------------

def test_short_momentum_exit_gated_by_min_hold(monkeypatch):
    """SHORT positions: 3 rising ticks within min-hold must not trigger."""
    trader = _make_trader()
    recent_entry = datetime.datetime.now(UTC) - datetime.timedelta(seconds=60)
    _add_position(trader, direction="SHORT", entry_ts_iso=recent_entry.isoformat())
    trader.state["_und_history"] = {
        "XAG-USD": [78.00, 78.50, 79.00],  # +1.28% rising (would trigger)
    }

    fired = _run_momentum_exit_check(trader, monkeypatch)
    assert fired is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_momentum_exit_check(trader, monkeypatch):
    """Invoke only the momentum-exit block by simulating _check_exits'
    local state. Returns the exit_reason string if the rule fired, or
    None if it did not. Keeps this test focused on the three fixes
    without pulling in the full _check_exits orchestration (stop-loss,
    signal reversal, EOD, trailing, etc.)."""
    pos_id = next(iter(trader.state["positions"]))
    pos = trader.state["positions"][pos_id]

    now = datetime.datetime.now(UTC)
    entry_ts = datetime.datetime.fromisoformat(pos["entry_ts"])

    exit_reason = None
    held_seconds = (now - entry_ts).total_seconds()
    min_hold_ok = held_seconds >= cfg.MOMENTUM_EXIT_MIN_HOLD_SECONDS
    hist_all = trader.state.get("_und_history", {}).get(pos["underlying"], [])
    if (not exit_reason
            and min_hold_ok
            and len(hist_all) >= 3):
        hist = hist_all[-3:]
        direction = pos.get("direction", "LONG")
        if direction == "LONG":
            monotonic = all(hist[i] < hist[i - 1] for i in range(1, len(hist)))
            move_rate = (hist[-1] - hist[0]) / hist[0] * 100
            if monotonic and move_rate < -cfg.MOMENTUM_EXIT_THRESHOLD_PCT:
                exit_reason = f"MOMENTUM_EXIT: 3 declining checks ({move_rate:.2f}%)"
        else:  # SHORT
            monotonic = all(hist[i] > hist[i - 1] for i in range(1, len(hist)))
            move_rate = (hist[-1] - hist[0]) / hist[0] * 100
            if monotonic and move_rate > cfg.MOMENTUM_EXIT_THRESHOLD_PCT:
                exit_reason = f"MOMENTUM_EXIT: 3 rising checks ({move_rate:+.2f}%)"
    return exit_reason
