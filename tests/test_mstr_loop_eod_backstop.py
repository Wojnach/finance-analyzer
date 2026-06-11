"""Loop-level EOD-flatten backstop (audit B8 fix 3).

The strategy-level EOD exit is unreachable when the bundle is stale/missing
(strategies return None on unusable data). The loop now flattens open
positions in the EOD window regardless of bundle.is_usable(), using a
degraded bundle synthesized from each position's last-known entry data.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portfolio.mstr_loop import config, execution, loop, session, state
from portfolio.mstr_loop.data_provider import MstrBundle
from portfolio.mstr_loop.state import BotState, Position


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr(config, "TRADES_LOG", str(tmp_path / "trades.jsonl"))
    monkeypatch.setattr(config, "SHADOW_LOG", str(tmp_path / "shadow.jsonl"))
    monkeypatch.setattr(config, "POLL_LOG", str(tmp_path / "poll.jsonl"))
    monkeypatch.setattr(config, "PHASE", "shadow")
    monkeypatch.setattr(config, "DRAWDOWN_CHECK_ENABLED", False)
    monkeypatch.setattr(config, "SCORECARD_UPDATE_ENABLED", False)
    yield


def _seed_position(st: BotState, key="momentum_rider"):
    st.add_position(Position(
        strategy_key=key, direction="LONG", cert_ob_id="2257847",
        entry_underlying_price=160.0, entry_cert_price=100.0,
        units=10, entry_units=10, entry_ts="2026-04-18T16:00:00+00:00",
    ))


class _Strat:
    key = "momentum_rider"

    def step(self, bundle, st):  # should never be reached in degraded path
        raise AssertionError("strategy.step ran during degraded EOD backstop")


def _unusable_bundle():
    return MstrBundle(
        ts="2026-04-18T16:00:00+00:00",
        source_stale_seconds=9999.0,  # > 300 => is_usable() False
        price_usd=160.0, raw_action="HOLD", raw_weighted_confidence=0.0,
        rsi=50, macd_hist=0, bb_position="", regime="unknown", atr_pct=0,
        buy_count=0, sell_count=0, total_voters=0, votes={},
        p_up_1d=0.0, exp_return_1d_pct=0.0, exp_return_3d_pct=0.0,
        heatmap=[], stale=False, weighted_score_long=0.0,
        weighted_score_short=0.0,
    )


def test_backstop_fires_with_none_bundle(monkeypatch):
    """build_bundle() returns None inside EOD window => still flatten."""
    monkeypatch.setattr(session, "kill_switch_active", lambda: False)
    monkeypatch.setattr(session, "in_session_window", lambda now=None: False)
    monkeypatch.setattr(session, "in_eod_flatten_window", lambda now=None: True)
    monkeypatch.setattr(loop, "build_bundle", lambda: None)

    st = BotState(cash_sek=0)
    _seed_position(st)
    loop.run_cycle(st, [_Strat()], cycle_count=1)
    assert st.get_position("momentum_rider") is None  # flattened
    assert st.total_trades == 1


def test_backstop_fires_with_unusable_bundle(monkeypatch):
    """build_bundle() returns a stale (is_usable=False) bundle => flatten."""
    monkeypatch.setattr(session, "kill_switch_active", lambda: False)
    monkeypatch.setattr(session, "in_session_window", lambda now=None: False)
    monkeypatch.setattr(session, "in_eod_flatten_window", lambda now=None: True)
    monkeypatch.setattr(loop, "build_bundle", _unusable_bundle)

    st = BotState(cash_sek=0)
    _seed_position(st)
    # _Strat.step would raise if reached; degraded path must not call it.
    loop.run_cycle(st, [_Strat()], cycle_count=1)
    assert st.get_position("momentum_rider") is None


def test_no_backstop_outside_eod(monkeypatch):
    """Outside the EOD window with a None bundle, positions are NOT flattened."""
    monkeypatch.setattr(session, "kill_switch_active", lambda: False)
    monkeypatch.setattr(session, "in_session_window", lambda now=None: True)
    monkeypatch.setattr(session, "in_eod_flatten_window", lambda now=None: False)
    monkeypatch.setattr(loop, "build_bundle", lambda: None)

    st = BotState(cash_sek=0)
    _seed_position(st)
    loop.run_cycle(st, [_Strat()], cycle_count=1)
    assert st.get_position("momentum_rider") is not None  # left alone


def test_degraded_bundle_uses_entry_price():
    pos = Position(
        strategy_key="m", direction="LONG", cert_ob_id="X",
        entry_underlying_price=171.5, entry_cert_price=100.0,
        units=5, entry_units=5, entry_ts="2026-04-18T16:00:00+00:00",
    )
    b = loop._degraded_bundle_for(pos)
    assert b.price_usd == 171.5
    assert b.stale is True
    assert b.is_usable() is False
