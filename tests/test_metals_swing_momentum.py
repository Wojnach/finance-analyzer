"""Unit tests for momentum-entry override in metals_swing_trader.

Covers the 2026-04-17 feature where a fresh LONG momentum candidate from
metals_loop's entry-side fast-tick relaxes MIN_BUY_CONFIDENCE and
MIN_BUY_VOTERS in SwingTrader._evaluate_entry.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from typing import Any

import pytest

# data/ contains script-style modules, not a package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import metals_swing_trader as mst


@pytest.fixture(autouse=True)
def _isolate_momentum_state(tmp_path, monkeypatch):
    """Redirect the momentum state file to tmp_path for each test."""
    state_file = tmp_path / "metals_momentum_state.json"
    monkeypatch.setattr(mst, "MOMENTUM_STATE_FILE", str(state_file))
    yield


def _make_trader():
    """Minimal SwingTrader with only attributes _evaluate_entry needs."""
    trader = mst.SwingTrader.__new__(mst.SwingTrader)
    trader.state = mst._default_state()
    trader.regime_history = {}
    # Give _regime_confirmed a passing history so that gate is transparent.
    trader.regime_history["XAG-USD"] = [
        ("BUY", "trending-up"),
        ("BUY", "trending-up"),
    ]
    trader.state["macd_history"] = {
        "XAG-USD": [-0.05000, -0.04000],  # improving, satisfies MACD gate
    }
    trader.check_count = 10
    return trader


def _signal(
    *,
    confidence: float,
    buy_count: int,
    sell_count: int,
    action: str = "BUY",
    rsi: float = 50.0,
    regime: str = "trending-up",
) -> dict:
    return {
        "action": action,
        "confidence": confidence,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "rsi": rsi,
        "regime": regime,
        "timeframes": {
            "Now": "BUY",
            "12h": "BUY",
            "2d": "BUY",
            "7d": "HOLD",
            "1mo": "HOLD",
            "3mo": "HOLD",
            "6mo": "BUY",
        },  # 4/7 BUY → 0.57 passes MIN_BUY_TF_RATIO=0.43
    }


def _write_candidate(
    state_file: str,
    *,
    ticker: str = "XAG-USD",
    direction: str = "LONG",
    age_sec: float = 30.0,
    consumed: bool = False,
) -> None:
    triggered = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=age_sec)
    state = {
        ticker: {
            "direction": direction,
            "velocity_pct": 0.92,
            "price_at_trigger": 79.85,
            "rvol": 2.20,
            "triggered_at": triggered.isoformat(),
            "consumed_at": triggered.isoformat() if consumed else None,
            "ttl_sec": 300,
        }
    }
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f)


def test_evaluate_entry_uses_relaxed_gates_with_momentum(monkeypatch):
    """Fresh LONG candidate + conf=0.55 + 2 voters → entry PASSES.

    Regular gates would reject: 0.55 < MIN_BUY_CONFIDENCE=0.60 AND
    2 < MIN_BUY_VOTERS=3. Momentum-relaxed gates accept both.
    """
    _write_candidate(mst.MOMENTUM_STATE_FILE, age_sec=30)
    trader = _make_trader()
    sig = _signal(confidence=0.55, buy_count=2, sell_count=1, rsi=50.0)

    ok, reason = trader._evaluate_entry(sig, "XAG-USD")

    assert ok, f"Expected PASS with momentum, got: {reason!r}"


def test_evaluate_entry_rejects_below_momentum_relaxed_gates(monkeypatch):
    """confidence=0.45 is below MOMENTUM_MIN_BUY_CONFIDENCE=0.50 → still REJECT."""
    _write_candidate(mst.MOMENTUM_STATE_FILE, age_sec=30)
    trader = _make_trader()
    sig = _signal(confidence=0.45, buy_count=2, sell_count=1, rsi=50.0)

    ok, reason = trader._evaluate_entry(sig, "XAG-USD")

    assert not ok
    assert "confidence 0.45" in reason
    assert "0.5" in reason, f"Expected relaxed 0.5 threshold in reason: {reason!r}"


def test_evaluate_entry_voter_gate_also_relaxed_with_momentum(monkeypatch):
    """1 voter is below MOMENTUM_MIN_BUY_VOTERS=2 → REJECT."""
    _write_candidate(mst.MOMENTUM_STATE_FILE, age_sec=30)
    trader = _make_trader()
    # buy=1 sell=0: strict majority OK, confidence OK, but voter count too low.
    sig = _signal(confidence=0.55, buy_count=1, sell_count=0, rsi=50.0)

    ok, reason = trader._evaluate_entry(sig, "XAG-USD")

    assert not ok
    # Reason must cite the 2-voter threshold, not the 3-voter one.
    assert "LONG_count=1" in reason
    assert "2" in reason


def test_evaluate_entry_unchanged_without_momentum(monkeypatch):
    """No candidate file → full MIN_BUY_CONFIDENCE=0.60 gate applied.

    Regression guard: momentum override must not leak into regular ranging
    entries when no breakout has happened.
    """
    # No candidate written.
    assert not os.path.exists(mst.MOMENTUM_STATE_FILE)
    trader = _make_trader()
    sig = _signal(confidence=0.55, buy_count=3, sell_count=1, rsi=50.0)

    ok, reason = trader._evaluate_entry(sig, "XAG-USD")

    assert not ok
    # Regular threshold 0.60 should appear in the reason (not relaxed 0.50).
    assert "0.6" in reason
    assert "confidence 0.55" in reason


def test_momentum_candidate_expires_after_ttl(monkeypatch):
    """A candidate older than MOMENTUM_CANDIDATE_TTL_SEC is ignored."""
    _write_candidate(mst.MOMENTUM_STATE_FILE, age_sec=600)  # 10 min — stale
    trader = _make_trader()
    sig = _signal(confidence=0.55, buy_count=2, sell_count=1, rsi=50.0)

    ok, reason = trader._evaluate_entry(sig, "XAG-USD")

    assert not ok, (
        "Stale candidate must not relax gates — regular 0.60 threshold "
        "should reject 0.55"
    )


def test_momentum_candidate_ignored_when_consumed(monkeypatch):
    """A candidate with consumed_at set is ignored (prevents re-trigger)."""
    _write_candidate(mst.MOMENTUM_STATE_FILE, age_sec=30, consumed=True)
    trader = _make_trader()
    sig = _signal(confidence=0.55, buy_count=2, sell_count=1, rsi=50.0)

    ok, reason = trader._evaluate_entry(sig, "XAG-USD")

    assert not ok, "Consumed candidate must not relax gates"


def test_momentum_candidate_ignored_for_short(monkeypatch):
    """Fast-tick only writes LONG candidates; SHORT entries must not pick them up."""
    _write_candidate(mst.MOMENTUM_STATE_FILE, age_sec=30, direction="LONG")
    # Force SHORT path: monkey-patch SHORT_ENABLED to True so we reach the gate.
    monkeypatch.setattr(mst, "SHORT_ENABLED", True)
    trader = _make_trader()
    trader.regime_history["XAG-USD"] = [
        ("SELL", "trending-down"),
        ("SELL", "trending-down"),
    ]
    # SHORT MACD needs to be DECLINING
    trader.state["macd_history"]["XAG-USD"] = [-0.04000, -0.05000]
    sig = _signal(
        confidence=0.55, buy_count=1, sell_count=2,
        action="SELL", rsi=55.0, regime="trending-down",
    )
    sig["timeframes"] = {
        "Now": "SELL", "12h": "SELL", "2d": "SELL",
        "7d": "HOLD", "1mo": "HOLD", "3mo": "HOLD", "6mo": "SELL",
    }

    ok, reason = trader._evaluate_entry(sig, "XAG-USD")

    # SHORT should see regular 0.60 threshold, not relaxed 0.50 from the
    # LONG candidate. 0.55 < 0.60 → reject.
    assert not ok
    assert "0.6" in reason, (
        f"SHORT entry must NOT use LONG candidate's relaxed threshold: {reason!r}"
    )


def test_check_momentum_candidate_returns_none_on_missing_file(monkeypatch):
    """Missing state file is the common case — must not raise."""
    assert not os.path.exists(mst.MOMENTUM_STATE_FILE)
    trader = _make_trader()
    assert trader._check_momentum_candidate("XAG-USD") is None


def test_check_momentum_candidate_returns_none_on_corrupt_file(monkeypatch, tmp_path):
    """Corrupt JSON state file should be treated as no-candidate (not crash)."""
    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{not json")
    monkeypatch.setattr(mst, "MOMENTUM_STATE_FILE", str(corrupt_file))
    trader = _make_trader()
    # Fail-safe: return None, don't propagate
    assert trader._check_momentum_candidate("XAG-USD") is None


def test_consume_momentum_candidate_marks_consumed(monkeypatch):
    """After consume, the candidate has a non-null consumed_at."""
    _write_candidate(mst.MOMENTUM_STATE_FILE, age_sec=30)
    trader = _make_trader()
    trader._consume_momentum_candidate("XAG-USD")

    state = json.load(open(mst.MOMENTUM_STATE_FILE, encoding="utf-8"))
    cand = state.get("XAG-USD")
    assert cand is not None
    assert cand["consumed_at"] is not None, (
        "consume must set consumed_at to a non-null timestamp"
    )


def test_consume_momentum_candidate_is_idempotent(monkeypatch):
    """Calling consume twice is safe (second call just overwrites consumed_at)."""
    _write_candidate(mst.MOMENTUM_STATE_FILE, age_sec=30)
    trader = _make_trader()
    trader._consume_momentum_candidate("XAG-USD")
    first_ts = json.load(open(mst.MOMENTUM_STATE_FILE))["XAG-USD"]["consumed_at"]
    trader._consume_momentum_candidate("XAG-USD")
    second_ts = json.load(open(mst.MOMENTUM_STATE_FILE))["XAG-USD"]["consumed_at"]
    assert first_ts is not None
    assert second_ts is not None
    # Monotonic or equal — both are valid idempotent outcomes.
    assert second_ts >= first_ts


def test_consume_momentum_candidate_no_op_when_absent(monkeypatch):
    """Consume when no file exists should not raise."""
    assert not os.path.exists(mst.MOMENTUM_STATE_FILE)
    trader = _make_trader()
    # Must not raise
    trader._consume_momentum_candidate("XAG-USD")


def test_momentum_override_disabled_flag(monkeypatch):
    """MOMENTUM_ENTRY_ENABLED=False must make _check_momentum_candidate return None.

    Gives an ops kill-switch without requiring a code change.
    """
    _write_candidate(mst.MOMENTUM_STATE_FILE, age_sec=30)
    monkeypatch.setattr(mst, "MOMENTUM_ENTRY_ENABLED", False)
    trader = _make_trader()
    assert trader._check_momentum_candidate("XAG-USD") is None

    # And _evaluate_entry falls back to regular gates.
    sig = _signal(confidence=0.55, buy_count=2, sell_count=1, rsi=50.0)
    ok, reason = trader._evaluate_entry(sig, "XAG-USD")
    assert not ok
    # Regular 0.60 threshold applies.
    assert "0.6" in reason
