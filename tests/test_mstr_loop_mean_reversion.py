"""Tests for portfolio.mstr_loop.strategies.mean_reversion (SHORT strategy)."""

from __future__ import annotations

import datetime
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portfolio.mstr_loop import config
from portfolio.mstr_loop.data_provider import MstrBundle
from portfolio.mstr_loop.state import BotState, Position
from portfolio.mstr_loop.strategies.mean_reversion import MeanReversion


@pytest.fixture(autouse=True)
def _populate_bear_ob(monkeypatch):
    """Set a test BEAR cert id so the strategy is operable."""
    monkeypatch.setattr(config, "BEAR_MSTR_OB_ID", "9999999")
    yield


def _mk_bundle(**overrides):
    defaults = dict(
        ts="2026-04-18T16:00:00+00:00", source_stale_seconds=30.0,
        price_usd=165.0, raw_action="SELL", raw_weighted_confidence=0.6,
        rsi=85.0, macd_hist=-1.2, bb_position="above_upper", regime="trending-up",
        atr_pct=1.5, buy_count=2, sell_count=6, total_voters=8,
        votes={"ministral": "SELL"}, p_up_1d=0.3, exp_return_1d_pct=-0.2,
        exp_return_3d_pct=-0.6, heatmap=[], stale=False,
        weighted_score_long=0.20, weighted_score_short=0.70,
        # BTC ranging — gate stays open
        btc_regime="ranging", btc_price=75000.0, btc_rsi=55.0,
    )
    defaults.update(overrides)
    return MstrBundle(**defaults)


def test_short_entry_fires_when_overbought_and_weighted_short_high():
    b = _mk_bundle(rsi=85, weighted_score_short=0.70)
    d = MeanReversion().step(b, BotState())
    assert d is not None
    assert d.action == "BUY"  # opening SHORT = buying BEAR cert
    assert d.direction == "SHORT"
    assert d.cert_ob_id == "9999999"


def test_short_skipped_when_rsi_below_min():
    b = _mk_bundle(rsi=65, weighted_score_short=0.80)
    d = MeanReversion().step(b, BotState())
    assert d is None


def test_short_skipped_when_weighted_short_low():
    b = _mk_bundle(rsi=90, weighted_score_short=0.30)
    d = MeanReversion().step(b, BotState())
    assert d is None


def test_short_refused_when_bear_ob_id_missing(monkeypatch):
    monkeypatch.setattr(config, "BEAR_MSTR_OB_ID", None)
    b = _mk_bundle(rsi=90, weighted_score_short=0.80)
    d = MeanReversion().step(b, BotState())
    assert d is None


def test_short_refused_by_btc_regime_gate():
    b = _mk_bundle(rsi=90, weighted_score_short=0.80, btc_regime="trending-up")
    d = MeanReversion().step(b, BotState())
    assert d is None  # BTC up blocks SHORT


def test_short_exit_on_hard_stop():
    pos = Position(
        strategy_key="mean_reversion", direction="SHORT",
        cert_ob_id="9999999", entry_underlying_price=170.0,
        entry_cert_price=100.0, units=10, entry_units=10,
        entry_ts="2026-04-18T16:00:00+00:00",
    )
    # SHORT loses when price rises. 170 → 173.5 = +2.06% underlying = -2.06% for SHORT
    b = _mk_bundle(price_usd=173.5, weighted_score_short=0.80)
    state = BotState()
    state.add_position(pos)
    d = MeanReversion().step(b, state)
    assert d is not None
    assert d.action == "SELL"
    assert d.exit_reason == "stop"


def test_short_exit_on_signal_flip():
    pos = Position(
        strategy_key="mean_reversion", direction="SHORT",
        cert_ob_id="9999999", entry_underlying_price=170.0,
        entry_cert_price=100.0, units=10, entry_units=10,
        entry_ts="2026-04-18T16:00:00+00:00",
    )
    # Small profit + weighted_long flip → exit
    b = _mk_bundle(price_usd=168.0, weighted_score_long=0.70, weighted_score_short=0.20)
    state = BotState()
    state.add_position(pos)
    d = MeanReversion().step(b, state)
    assert d is not None
    assert d.action == "SELL"
    assert d.exit_reason == "signal_flip"


def test_short_exit_on_trail_bounce():
    """Peak = lowest price since entry. Current = +2% above peak → trail trips."""
    pos = Position(
        strategy_key="mean_reversion", direction="SHORT",
        cert_ob_id="9999999", entry_underlying_price=170.0,
        entry_cert_price=100.0, units=10, entry_units=10,
        entry_ts="2026-04-18T16:00:00+00:00",
        trail_active=True, peak_underlying_price=162.0,  # lowest seen
    )
    # Current 165.3 → +2.04% above peak 162 → trail fires
    b = _mk_bundle(price_usd=165.3,
                   weighted_score_long=0.30, weighted_score_short=0.40)
    state = BotState()
    state.add_position(pos)
    d = MeanReversion().step(b, state)
    assert d is not None
    assert d.action == "SELL"
    assert d.exit_reason == "trail"
