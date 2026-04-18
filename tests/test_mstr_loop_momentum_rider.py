"""Unit tests for portfolio.mstr_loop.strategies.momentum_rider."""

from __future__ import annotations

import datetime
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portfolio.mstr_loop import config
from portfolio.mstr_loop.data_provider import MstrBundle
from portfolio.mstr_loop.state import BotState, Position
from portfolio.mstr_loop.strategies.momentum_rider import MomentumRider


def _mk_bundle(**overrides):
    """Build an MstrBundle with sensible defaults; override specific fields in tests."""
    defaults = dict(
        ts="2026-04-18T16:00:00+00:00",
        source_stale_seconds=30.0,
        price_usd=165.0,
        raw_action="BUY",
        raw_weighted_confidence=0.6,
        rsi=55.0,
        macd_hist=1.2,
        bb_position="inside",
        regime="trending-up",
        atr_pct=1.5,
        buy_count=5,
        sell_count=3,
        total_voters=8,
        votes={"a": "BUY", "b": "BUY"},
        p_up_1d=0.65,
        exp_return_1d_pct=0.3,
        exp_return_3d_pct=0.8,
        heatmap=[],
        stale=False,
        weighted_score_long=0.70,
        weighted_score_short=0.20,
    )
    defaults.update(overrides)
    return MstrBundle(**defaults)


def _mk_state(**overrides):
    state = BotState(**overrides)
    return state


# ---------------------------------------------------------------------------
# Entry path
# ---------------------------------------------------------------------------


def test_entry_fires_when_score_over_threshold():
    """weighted_long ≥ 0.55, RSI in window, no position, cooldown clear → BUY."""
    bundle = _mk_bundle(weighted_score_long=0.60, rsi=60)
    state = _mk_state()
    d = MomentumRider().step(bundle, state)
    assert d is not None
    assert d.action == "BUY"
    assert d.direction == "LONG"
    assert d.cert_ob_id == config.BULL_MSTR_OB_ID
    assert d.strategy_key == "momentum_rider"


def test_entry_skipped_when_score_below_threshold():
    bundle = _mk_bundle(weighted_score_long=0.40)
    d = MomentumRider().step(bundle, _mk_state())
    assert d is None


def test_entry_skipped_when_rsi_too_high():
    """RSI 85 > MOMENTUM_RIDER_RSI_MAX → no entry (don't chase blow-offs)."""
    bundle = _mk_bundle(weighted_score_long=0.90, rsi=85)
    d = MomentumRider().step(bundle, _mk_state())
    assert d is None


def test_entry_skipped_when_rsi_too_low():
    bundle = _mk_bundle(weighted_score_long=0.90, rsi=30)
    d = MomentumRider().step(bundle, _mk_state())
    assert d is None


def test_entry_skipped_when_stale():
    bundle = _mk_bundle(weighted_score_long=0.90, stale=True)
    # Note: stale makes is_usable() False → step() returns None before
    # ever checking the specific entry gates.
    d = MomentumRider().step(bundle, _mk_state())
    assert d is None


def test_entry_skipped_during_cooldown():
    bundle = _mk_bundle(weighted_score_long=0.80)
    state = _mk_state()
    # Just exited 10 min ago, cooldown is 30 min
    recent = datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=10)
    state.last_exit_ts["momentum_rider"] = recent.isoformat()
    d = MomentumRider().step(bundle, state)
    assert d is None


def test_entry_allowed_after_cooldown():
    bundle = _mk_bundle(weighted_score_long=0.80)
    state = _mk_state()
    old = datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=60)
    state.last_exit_ts["momentum_rider"] = old.isoformat()
    d = MomentumRider().step(bundle, state)
    assert d is not None
    assert d.action == "BUY"


def test_entry_skipped_when_position_already_open():
    """Momentum rider opens one position at a time — extra BUY signals ignored."""
    bundle = _mk_bundle(weighted_score_long=0.90)
    state = _mk_state()
    state.add_position(Position(
        strategy_key="momentum_rider", direction="LONG",
        cert_ob_id=config.BULL_MSTR_OB_ID,
        entry_underlying_price=160.0, entry_cert_price=100.0,
        units=5, entry_ts="2026-04-18T16:00:00+00:00",
    ))
    d = MomentumRider().step(bundle, state)
    # Should evaluate as exit, not re-buy
    assert d is None or d.action == "SELL"


# ---------------------------------------------------------------------------
# Exit path
# ---------------------------------------------------------------------------


def _open_position(entry_price=160.0):
    return Position(
        strategy_key="momentum_rider", direction="LONG",
        cert_ob_id=config.BULL_MSTR_OB_ID,
        entry_underlying_price=entry_price, entry_cert_price=100.0,
        units=10, entry_ts="2026-04-18T16:00:00+00:00",
    )


def test_exit_on_hard_stop():
    """Underlying -2% or worse → SELL with exit_reason=stop."""
    bundle = _mk_bundle(price_usd=156.0)  # -2.5% from 160
    state = _mk_state()
    state.add_position(_open_position())
    d = MomentumRider().step(bundle, state)
    assert d is not None
    assert d.action == "SELL"
    assert d.exit_reason == "stop"


def test_exit_on_signal_flip():
    """Weighted SHORT score above threshold → SELL with exit_reason=signal_flip."""
    bundle = _mk_bundle(
        price_usd=161.0,                  # mild profit, within trail zone
        weighted_score_long=0.20,
        weighted_score_short=0.65,
    )
    state = _mk_state()
    state.add_position(_open_position())
    d = MomentumRider().step(bundle, state)
    assert d is not None
    assert d.action == "SELL"
    assert d.exit_reason == "signal_flip"


def test_exit_on_trail_after_activation():
    """Trail is active (peak captured), price pulls back > trail_distance → SELL."""
    pos = _open_position(entry_price=160.0)
    pos.trail_active = True
    pos.peak_underlying_price = 165.0  # +3.1%
    # Current 161.7 = -2% from peak 165 → trips trail
    bundle = _mk_bundle(
        price_usd=161.7,
        weighted_score_long=0.40,  # no signal flip
        weighted_score_short=0.30,
    )
    state = _mk_state()
    state.add_position(pos)
    d = MomentumRider().step(bundle, state)
    assert d is not None
    assert d.action == "SELL"
    assert d.exit_reason == "trail"


def test_no_exit_when_trail_not_active_yet():
    """Price up only +0.5% — not enough to activate trail → hold."""
    pos = _open_position(entry_price=160.0)
    bundle = _mk_bundle(
        price_usd=160.8,
        weighted_score_short=0.20,  # no signal flip
    )
    state = _mk_state()
    state.add_position(pos)
    d = MomentumRider().step(bundle, state)
    assert d is None


def test_exit_on_eod_flatten(monkeypatch):
    """21:45 CET or later → SELL with exit_reason=eod."""
    pos = _open_position()
    bundle = _mk_bundle(
        price_usd=161.0,
        weighted_score_short=0.20,
    )
    state = _mk_state()
    state.add_position(pos)
    # Patch session.in_eod_flatten_window to True
    from portfolio.mstr_loop import session
    monkeypatch.setattr(session, "in_eod_flatten_window", lambda: True)
    d = MomentumRider().step(bundle, state)
    assert d is not None
    assert d.action == "SELL"
    assert d.exit_reason == "eod"


def test_unusable_bundle_short_circuits():
    """No usable bundle → step returns None without touching state."""
    bundle = _mk_bundle(stale=True)  # is_usable() -> False
    state = _mk_state()
    state.add_position(_open_position())
    assert MomentumRider().step(bundle, state) is None
