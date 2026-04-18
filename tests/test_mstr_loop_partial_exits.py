"""Tests for the partial-exit ladder in execution.update_trail_state."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portfolio.mstr_loop import config, execution
from portfolio.mstr_loop.data_provider import MstrBundle
from portfolio.mstr_loop.state import BotState, Position


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr(config, "TRADES_LOG", str(tmp_path / "trades.jsonl"))
    monkeypatch.setattr(config, "SHADOW_LOG", str(tmp_path / "shadow.jsonl"))
    monkeypatch.setattr(config, "POLL_LOG", str(tmp_path / "poll.jsonl"))
    # ATR-adaptive disabled here so the tests are deterministic.
    monkeypatch.setattr(config, "ATR_ADAPTIVE_TRAIL_ENABLED", False)
    yield


def _mk_bundle(price):
    return MstrBundle(
        ts="2026-04-18T16:00:00+00:00", source_stale_seconds=30.0,
        price_usd=price, raw_action="BUY", raw_weighted_confidence=0.6, rsi=55,
        macd_hist=1.2, bb_position="inside", regime="trending-up",
        atr_pct=1.5, buy_count=5, sell_count=3, total_voters=8,
        votes={"x": "BUY"}, p_up_1d=0.65, exp_return_1d_pct=0.3, exp_return_3d_pct=0.8,
        heatmap=[], stale=False,
        weighted_score_long=0.70, weighted_score_short=0.20,
        btc_regime="ranging", btc_price=75000.0, btc_rsi=55.0,
    )


def _mk_position(entry_price=160.0, units=9, entry_units=9):
    return Position(
        strategy_key="momentum_rider", direction="LONG",
        cert_ob_id=config.BULL_MSTR_OB_ID,
        entry_underlying_price=entry_price, entry_cert_price=100.0,
        units=units, entry_units=entry_units,
        entry_ts="2026-04-18T16:00:00+00:00",
    )


def test_partial_exit_ladder_disabled(monkeypatch):
    monkeypatch.setattr(config, "PHASE", "shadow")
    monkeypatch.setattr(config, "PARTIAL_EXIT_LADDER_ENABLED", False)
    state = BotState()
    pos = _mk_position()
    state.add_position(pos)

    # Price up +5% — would trip both tranches if enabled
    execution.update_trail_state(state, _mk_bundle(168.0))
    assert pos.units == 9  # unchanged
    assert pos.units_sold == 0
    assert pos.tranches_hit == []


def test_partial_exit_first_tranche_at_plus_2pct(monkeypatch):
    monkeypatch.setattr(config, "PHASE", "shadow")
    monkeypatch.setattr(config, "PARTIAL_EXIT_LADDER_ENABLED", True)
    monkeypatch.setattr(config, "PARTIAL_EXIT_TRANCHES", [(2.0, 1 / 3), (4.0, 1 / 3)])
    state = BotState()
    pos = _mk_position(units=9, entry_units=9)
    state.add_position(pos)

    # 160 → 163.2 = +2.0% → +2% tranche fires, 4% does NOT
    execution.update_trail_state(state, _mk_bundle(163.2))
    assert pos.units == 6        # sold 3 of 9
    assert pos.units_sold == 3
    assert 2.0 in pos.tranches_hit
    assert 4.0 not in pos.tranches_hit


def test_partial_exit_both_tranches_at_plus_5pct(monkeypatch):
    monkeypatch.setattr(config, "PHASE", "shadow")
    monkeypatch.setattr(config, "PARTIAL_EXIT_LADDER_ENABLED", True)
    monkeypatch.setattr(config, "PARTIAL_EXIT_TRANCHES", [(2.0, 1 / 3), (4.0, 1 / 3)])
    state = BotState()
    pos = _mk_position(units=9, entry_units=9)
    state.add_position(pos)

    # 160 → 168 = +5% → both tranches fire
    execution.update_trail_state(state, _mk_bundle(168.0))
    assert pos.units == 3        # sold 3 + 3 of 9 = 6 remaining
    assert pos.units_sold == 6
    assert 2.0 in pos.tranches_hit
    assert 4.0 in pos.tranches_hit


def test_partial_exit_tranche_does_not_refire(monkeypatch):
    """Second cycle at same price shouldn't re-fire already-hit tranches."""
    monkeypatch.setattr(config, "PHASE", "shadow")
    monkeypatch.setattr(config, "PARTIAL_EXIT_LADDER_ENABLED", True)
    state = BotState()
    pos = _mk_position(units=9, entry_units=9)
    state.add_position(pos)

    execution.update_trail_state(state, _mk_bundle(163.2))
    units_after_first = pos.units
    # Re-call with same price — tranche already hit, should be no-op
    execution.update_trail_state(state, _mk_bundle(163.2))
    assert pos.units == units_after_first
