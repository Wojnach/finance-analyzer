"""Execution-layer tests — phase-gated buy/sell, shadow journal, paper cash math."""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portfolio.mstr_loop import config, execution
from portfolio.mstr_loop.data_provider import MstrBundle
from portfolio.mstr_loop.state import BotState, Position
from portfolio.mstr_loop.strategies.base import Decision


@pytest.fixture(autouse=True)
def _isolate_paths(tmp_path, monkeypatch):
    """Redirect all config paths to tmp_path so tests don't touch live files."""
    monkeypatch.setattr(config, "STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr(config, "TRADES_LOG", str(tmp_path / "trades.jsonl"))
    monkeypatch.setattr(config, "SHADOW_LOG", str(tmp_path / "shadow.jsonl"))
    monkeypatch.setattr(config, "POLL_LOG", str(tmp_path / "poll.jsonl"))
    yield


def _mk_bundle(price=165.0):
    return MstrBundle(
        ts="2026-04-18T16:00:00+00:00",
        source_stale_seconds=30.0, price_usd=price,
        raw_action="BUY", raw_weighted_confidence=0.6, rsi=55,
        macd_hist=1.2, bb_position="inside", regime="trending-up",
        atr_pct=1.5, buy_count=5, sell_count=3, total_voters=8,
        votes={"ministral": "BUY"},
        p_up_1d=0.65, exp_return_1d_pct=0.3, exp_return_3d_pct=0.8,
        heatmap=[], stale=False,
        weighted_score_long=0.70, weighted_score_short=0.20,
    )


def _mk_buy_decision():
    return Decision(
        strategy_key="momentum_rider", action="BUY", direction="LONG",
        cert_ob_id=config.BULL_MSTR_OB_ID, rationale="test entry",
        stop_pct_underlying=2.0, confidence=0.55,
    )


def _mk_sell_decision():
    return Decision(
        strategy_key="momentum_rider", action="SELL", direction="LONG",
        cert_ob_id=config.BULL_MSTR_OB_ID, rationale="test exit",
        exit_reason="signal_flip",
    )


# ---------------------------------------------------------------------------
# Phase B (shadow) — no cash movement, logs to SHADOW_LOG
# ---------------------------------------------------------------------------


def test_shadow_buy_appends_to_shadow_log(monkeypatch):
    monkeypatch.setattr(config, "PHASE", "shadow")
    state = BotState(cash_sek=0.0)  # no cash in shadow
    bundle = _mk_bundle()

    ok = execution.execute(_mk_buy_decision(), bundle, state)
    assert ok is True
    # Position created in memory
    assert state.has_position("momentum_rider")
    # Shadow log written
    entries = [json.loads(line) for line in open(config.SHADOW_LOG)]
    assert len(entries) == 1
    assert entries[0]["event"] == "SHADOW_BUY"
    assert entries[0]["strategy_key"] == "momentum_rider"


def test_shadow_sell_records_pnl(monkeypatch):
    monkeypatch.setattr(config, "PHASE", "shadow")
    state = BotState()
    bundle = _mk_bundle(price=165)
    execution.execute(_mk_buy_decision(), bundle, state)
    # Price rises → exit captures P&L
    bundle2 = _mk_bundle(price=170)
    ok = execution.execute(_mk_sell_decision(), bundle2, state)
    assert ok is True
    assert not state.has_position("momentum_rider")
    # Both events logged
    entries = [json.loads(line) for line in open(config.SHADOW_LOG)]
    assert [e["event"] for e in entries] == ["SHADOW_BUY", "SHADOW_SELL"]
    sell = entries[1]
    # Entry cert price was 100 (synthetic), 5x leverage, +3% underlying
    # → cert +15% → exit cert ~115 → pnl positive
    assert sell["pnl_sek"] > 0


def test_shadow_does_not_touch_trades_log(monkeypatch):
    monkeypatch.setattr(config, "PHASE", "shadow")
    state = BotState()
    execution.execute(_mk_buy_decision(), _mk_bundle(), state)
    assert not os.path.exists(config.TRADES_LOG)


# ---------------------------------------------------------------------------
# Phase C (paper) — cash deducted/credited, state.total_trades counts
# ---------------------------------------------------------------------------


def test_paper_buy_deducts_cash(monkeypatch):
    monkeypatch.setattr(config, "PHASE", "paper")
    state = BotState(cash_sek=100_000)
    bundle = _mk_bundle()

    ok = execution.execute(_mk_buy_decision(), bundle, state)
    assert ok is True
    # 30% of cash = 30000, cert_price=100 → 300 units × 100 = 30000 SEK spent
    assert state.cash_sek == pytest.approx(70_000)
    assert state.has_position("momentum_rider")


def test_paper_buy_skipped_when_cash_insufficient(monkeypatch, caplog):
    monkeypatch.setattr(config, "PHASE", "paper")
    state = BotState(cash_sek=500)  # below MIN_TRADE_SEK=1000
    ok = execution.execute(_mk_buy_decision(), _mk_bundle(), state)
    assert ok is False
    assert state.cash_sek == 500  # unchanged
    assert not state.has_position("momentum_rider")


def test_paper_buy_respects_95pct_cap(monkeypatch):
    """With cash=1500 and POSITION_SIZE_PCT=30 the floor bumps alloc to 1000;
    cap at 95% of 1500 = 1425 → final 1000."""
    monkeypatch.setattr(config, "PHASE", "paper")
    state = BotState(cash_sek=1500)
    execution.execute(_mk_buy_decision(), _mk_bundle(), state)
    # 1000 SEK spent → 500 remaining
    assert state.cash_sek == 500


def test_paper_sell_credits_cash_and_updates_stats(monkeypatch):
    monkeypatch.setattr(config, "PHASE", "paper")
    state = BotState(cash_sek=100_000)
    execution.execute(_mk_buy_decision(), _mk_bundle(price=160), state)
    # Price rose +3% underlying → cert +15% → proceeds ≈ 34,500 SEK
    ok = execution.execute(_mk_sell_decision(), _mk_bundle(price=164.8), state)
    assert ok is True
    assert state.cash_sek > 70_000  # proceeds credited
    assert state.total_trades == 1
    assert state.wins == 1
    assert state.last_exit_ts.get("momentum_rider") is not None


def test_paper_trades_log_written(monkeypatch):
    monkeypatch.setattr(config, "PHASE", "paper")
    state = BotState(cash_sek=100_000)
    execution.execute(_mk_buy_decision(), _mk_bundle(), state)
    execution.execute(_mk_sell_decision(), _mk_bundle(price=166), state)
    entries = [json.loads(line) for line in open(config.TRADES_LOG)]
    assert [e["action"] for e in entries] == ["BUY", "SELL"]


# ---------------------------------------------------------------------------
# Trail state updates — called by the loop per cycle
# ---------------------------------------------------------------------------


def test_update_trail_activates_above_threshold(monkeypatch):
    monkeypatch.setattr(config, "PHASE", "shadow")
    state = BotState()
    execution.execute(_mk_buy_decision(), _mk_bundle(price=160), state)
    pos = state.get_position("momentum_rider")
    assert pos.trail_active is False

    execution.update_trail_state(state, _mk_bundle(price=163))  # +1.875% > 1.5%
    assert pos.trail_active is True


def test_update_trail_tracks_peak_long(monkeypatch):
    monkeypatch.setattr(config, "PHASE", "shadow")
    state = BotState()
    execution.execute(_mk_buy_decision(), _mk_bundle(price=160), state)
    pos = state.get_position("momentum_rider")
    execution.update_trail_state(state, _mk_bundle(price=163))
    execution.update_trail_state(state, _mk_bundle(price=165))
    execution.update_trail_state(state, _mk_bundle(price=162))  # pullback
    # Peak should stay at 165 (highest seen so far)
    assert pos.peak_underlying_price == 165


# ---------------------------------------------------------------------------
# Notional sizing helper
# ---------------------------------------------------------------------------


def test_notional_shadow_uses_hypothetical(monkeypatch):
    monkeypatch.setattr(config, "PHASE", "shadow")
    monkeypatch.setattr(config, "SHADOW_NOTIONAL_SEK", 30_000)
    assert execution._notional_for_entry(BotState(cash_sek=0)) == 30_000


def test_notional_paper_respects_floor(monkeypatch):
    monkeypatch.setattr(config, "PHASE", "paper")
    monkeypatch.setattr(config, "POSITION_SIZE_PCT", 30)
    monkeypatch.setattr(config, "MIN_TRADE_SEK", 1000)
    # 2822 × 0.30 = 847 → floor to 1000 → cap at 95%*2822 = 2681
    assert execution._notional_for_entry(BotState(cash_sek=2822)) == 1000


def test_notional_paper_insufficient_cash_returns_zero(monkeypatch):
    monkeypatch.setattr(config, "PHASE", "paper")
    monkeypatch.setattr(config, "MIN_TRADE_SEK", 1000)
    # cash*0.95 = 855 < 1000 floor → infeasible
    assert execution._notional_for_entry(BotState(cash_sek=900)) == 0
