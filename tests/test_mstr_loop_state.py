"""Serialization + recovery tests for portfolio.mstr_loop.state."""

from __future__ import annotations

import datetime
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portfolio.mstr_loop import config
from portfolio.mstr_loop.state import (
    BotState, Position, default_state, load_state, save_state,
)


def test_default_state_has_no_positions():
    s = default_state()
    assert s.positions == {}
    assert s.total_trades == 0


def test_default_state_phase_cash(monkeypatch):
    monkeypatch.setattr(config, "PHASE", "paper")
    monkeypatch.setattr(config, "INITIAL_PAPER_CASH_SEK", 100_000)
    s = default_state()
    assert s.cash_sek == 100_000
    monkeypatch.setattr(config, "PHASE", "shadow")
    s2 = default_state()
    assert s2.cash_sek == 0


def test_save_and_load_round_trip(tmp_path):
    path = str(tmp_path / "state.json")
    s = BotState(cash_sek=50_000, total_trades=3, total_pnl_sek=1200, wins=2, losses=1)
    s.add_position(Position(
        strategy_key="momentum_rider", direction="LONG",
        cert_ob_id="2257847",
        entry_underlying_price=160.0, entry_cert_price=100.0,
        units=10, entry_ts="2026-04-18T16:00:00+00:00",
        trail_active=True, peak_underlying_price=165.0, rationale="test entry",
    ))
    s.last_exit_ts["foo_strategy"] = "2026-04-17T21:00:00+00:00"
    save_state(s, path)

    s2 = load_state(path)
    assert s2.cash_sek == 50_000
    assert s2.total_trades == 3
    assert s2.wins == 2
    assert s2.losses == 1
    pos = s2.get_position("momentum_rider")
    assert pos is not None
    assert pos.units == 10
    assert pos.trail_active is True
    assert pos.peak_underlying_price == 165.0
    assert s2.last_exit_ts["foo_strategy"] == "2026-04-17T21:00:00+00:00"


def test_load_missing_file_returns_defaults(tmp_path):
    s = load_state(str(tmp_path / "nonexistent.json"))
    assert s.positions == {}


def test_load_corrupt_file_returns_defaults(tmp_path, caplog):
    path = tmp_path / "corrupt.json"
    path.write_text("{not json")
    s = load_state(str(path))
    assert s.positions == {}
    # Should log — but not raise


def test_load_non_dict_returns_defaults(tmp_path):
    path = tmp_path / "list.json"
    path.write_text(json.dumps([1, 2, 3]))
    s = load_state(str(path))
    assert s.positions == {}


def test_load_drops_malformed_position(tmp_path):
    """One malformed position entry shouldn't tank the whole state."""
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({
        "cash_sek": 50_000,
        "positions": {
            "good": {
                "strategy_key": "good", "direction": "LONG",
                "cert_ob_id": "X", "entry_underlying_price": 100,
                "entry_cert_price": 10, "units": 5,
                "entry_ts": "2026-04-18T16:00:00+00:00",
            },
            "bad": {"not a position dict": True},
        },
    }))
    s = load_state(str(path))
    assert s.cash_sek == 50_000
    assert "good" in s.positions
    assert "bad" not in s.positions


def test_cooldown_elapsed_returns_true_when_no_prior_exit():
    s = BotState()
    assert s.cooldown_elapsed("momentum_rider", 30) is True


def test_cooldown_elapsed_recent_exit_blocks():
    s = BotState()
    s.last_exit_ts["momentum_rider"] = (
        datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=5)
    ).isoformat()
    assert s.cooldown_elapsed("momentum_rider", 30) is False


def test_cooldown_elapsed_old_exit_clears():
    s = BotState()
    s.last_exit_ts["momentum_rider"] = (
        datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=60)
    ).isoformat()
    assert s.cooldown_elapsed("momentum_rider", 30) is True


def test_cooldown_elapsed_bad_ts_defaults_to_true():
    """A corrupt last_exit_ts shouldn't permanently block entries."""
    s = BotState()
    s.last_exit_ts["momentum_rider"] = "not a timestamp"
    assert s.cooldown_elapsed("momentum_rider", 30) is True


def test_position_unrealized_pct_long():
    pos = Position(
        strategy_key="momentum_rider", direction="LONG",
        cert_ob_id="X", entry_underlying_price=100, entry_cert_price=10,
        units=5, entry_ts="2026-04-18T16:00:00+00:00",
    )
    assert pos.unrealized_underlying_pct(105) == pytest.approx(5.0)
    assert pos.unrealized_underlying_pct(95) == pytest.approx(-5.0)


def test_position_unrealized_pct_short():
    pos = Position(
        strategy_key="mean_reversion", direction="SHORT",
        cert_ob_id="Y", entry_underlying_price=100, entry_cert_price=10,
        units=5, entry_ts="2026-04-18T16:00:00+00:00",
    )
    # SHORT profits when price falls
    assert pos.unrealized_underlying_pct(95) == pytest.approx(5.0)
    assert pos.unrealized_underlying_pct(105) == pytest.approx(-5.0)


def test_position_unrealized_pct_zero_entry_defensive():
    pos = Position(
        strategy_key="x", direction="LONG",
        cert_ob_id="X", entry_underlying_price=0, entry_cert_price=10,
        units=5, entry_ts="2026-04-18T16:00:00+00:00",
    )
    # Div-by-zero defence — returns 0 instead of crashing
    assert pos.unrealized_underlying_pct(100) == 0.0
