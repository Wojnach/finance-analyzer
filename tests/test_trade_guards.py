"""Tests for trade guards (overtrading prevention)."""

import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from portfolio.trade_guards import (
    check_overtrading_guards,
    record_trade,
    get_all_guard_warnings,
    _load_state,
    _save_state,
    _get_cooldown_multiplier,
    STATE_FILE,
)


@pytest.fixture(autouse=True)
def clean_state(tmp_path):
    """Use a temp state file for each test."""
    temp_state = tmp_path / "trade_guard_state.json"
    with patch("portfolio.trade_guards.STATE_FILE", temp_state):
        yield temp_state


# --- Cooldown multiplier ---

class TestCooldownMultiplier:
    def test_zero_losses(self):
        assert _get_cooldown_multiplier(0) == 1

    def test_one_loss(self):
        assert _get_cooldown_multiplier(1) == 1

    def test_two_losses(self):
        assert _get_cooldown_multiplier(2) == 2

    def test_three_losses(self):
        assert _get_cooldown_multiplier(3) == 4

    def test_four_plus_losses(self):
        assert _get_cooldown_multiplier(4) == 8
        assert _get_cooldown_multiplier(10) == 8


# --- Per-ticker cooldown ---

class TestTickerCooldown:
    def test_no_warning_when_no_prior_trade(self, clean_state):
        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            warnings = check_overtrading_guards("BTC-USD", "BUY", "bold", {})
            cooldown_warns = [w for w in warnings if w["guard"] == "ticker_cooldown"]
            assert len(cooldown_warns) == 0

    def test_warning_within_cooldown(self, clean_state):
        now = datetime.now(timezone.utc)
        state = {
            "ticker_trades": {"bold:BTC-USD": (now - timedelta(minutes=10)).isoformat()},
            "consecutive_losses": {"patient": 0, "bold": 0},
            "new_position_timestamps": {"patient": [], "bold": []},
        }
        clean_state.write_text(json.dumps(state), encoding="utf-8")

        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            warnings = check_overtrading_guards("BTC-USD", "BUY", "bold", {})
            cooldown_warns = [w for w in warnings if w["guard"] == "ticker_cooldown"]
            assert len(cooldown_warns) == 1
            assert cooldown_warns[0]["details"]["remaining_min"] > 0

    def test_no_warning_after_cooldown_expires(self, clean_state):
        now = datetime.now(timezone.utc)
        state = {
            "ticker_trades": {"bold:BTC-USD": (now - timedelta(minutes=60)).isoformat()},
            "consecutive_losses": {"patient": 0, "bold": 0},
            "new_position_timestamps": {"patient": [], "bold": []},
        }
        clean_state.write_text(json.dumps(state), encoding="utf-8")

        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            warnings = check_overtrading_guards("BTC-USD", "BUY", "bold", {})
            cooldown_warns = [w for w in warnings if w["guard"] == "ticker_cooldown"]
            assert len(cooldown_warns) == 0

    def test_loss_escalation_increases_cooldown(self, clean_state):
        now = datetime.now(timezone.utc)
        # 3 consecutive losses → 4x cooldown (30 * 4 = 120 min)
        state = {
            "ticker_trades": {"bold:BTC-USD": (now - timedelta(minutes=50)).isoformat()},
            "consecutive_losses": {"patient": 0, "bold": 3},
            "new_position_timestamps": {"patient": [], "bold": []},
        }
        clean_state.write_text(json.dumps(state), encoding="utf-8")

        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            warnings = check_overtrading_guards("BTC-USD", "BUY", "bold", {})
            cooldown_warns = [w for w in warnings if w["guard"] == "ticker_cooldown"]
            assert len(cooldown_warns) == 1
            assert cooldown_warns[0]["details"]["cooldown_min"] == 120
            assert cooldown_warns[0]["details"]["multiplier"] == 4

    def test_different_tickers_independent(self, clean_state):
        now = datetime.now(timezone.utc)
        state = {
            "ticker_trades": {"bold:BTC-USD": (now - timedelta(minutes=10)).isoformat()},
            "consecutive_losses": {"patient": 0, "bold": 0},
            "new_position_timestamps": {"patient": [], "bold": []},
        }
        clean_state.write_text(json.dumps(state), encoding="utf-8")

        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            warnings = check_overtrading_guards("ETH-USD", "BUY", "bold", {})
            cooldown_warns = [w for w in warnings if w["guard"] == "ticker_cooldown"]
            assert len(cooldown_warns) == 0

    def test_different_strategies_independent(self, clean_state):
        now = datetime.now(timezone.utc)
        state = {
            "ticker_trades": {"bold:BTC-USD": (now - timedelta(minutes=10)).isoformat()},
            "consecutive_losses": {"patient": 0, "bold": 0},
            "new_position_timestamps": {"patient": [], "bold": []},
        }
        clean_state.write_text(json.dumps(state), encoding="utf-8")

        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            warnings = check_overtrading_guards("BTC-USD", "BUY", "patient", {})
            cooldown_warns = [w for w in warnings if w["guard"] == "ticker_cooldown"]
            assert len(cooldown_warns) == 0


# --- Consecutive loss escalation ---

class TestConsecutiveLosses:
    def test_no_warning_with_zero_losses(self, clean_state):
        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            warnings = check_overtrading_guards("BTC-USD", "BUY", "bold", {})
            loss_warns = [w for w in warnings if w["guard"] == "consecutive_losses"]
            assert len(loss_warns) == 0

    def test_warning_with_two_losses(self, clean_state):
        state = {
            "ticker_trades": {},
            "consecutive_losses": {"patient": 0, "bold": 2},
            "new_position_timestamps": {"patient": [], "bold": []},
        }
        clean_state.write_text(json.dumps(state), encoding="utf-8")

        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            warnings = check_overtrading_guards("BTC-USD", "BUY", "bold", {})
            loss_warns = [w for w in warnings if w["guard"] == "consecutive_losses"]
            assert len(loss_warns) == 1
            assert loss_warns[0]["details"]["multiplier"] == 2


# --- Position rate limit ---

class TestPositionRateLimit:
    def test_no_warning_below_limit(self, clean_state):
        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            warnings = check_overtrading_guards("BTC-USD", "BUY", "bold", {})
            rate_warns = [w for w in warnings if w["guard"] == "position_rate_limit"]
            assert len(rate_warns) == 0

    def test_warning_at_limit(self, clean_state):
        now = datetime.now(timezone.utc)
        state = {
            "ticker_trades": {},
            "consecutive_losses": {"patient": 0, "bold": 0},
            "new_position_timestamps": {
                "patient": [],
                "bold": [(now - timedelta(hours=1)).isoformat()],
            },
        }
        clean_state.write_text(json.dumps(state), encoding="utf-8")

        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            warnings = check_overtrading_guards("ETH-USD", "BUY", "bold", {})
            rate_warns = [w for w in warnings if w["guard"] == "position_rate_limit"]
            assert len(rate_warns) == 1

    def test_no_warning_for_sell(self, clean_state):
        now = datetime.now(timezone.utc)
        state = {
            "ticker_trades": {},
            "consecutive_losses": {"patient": 0, "bold": 0},
            "new_position_timestamps": {
                "patient": [],
                "bold": [(now - timedelta(hours=1)).isoformat()],
            },
        }
        clean_state.write_text(json.dumps(state), encoding="utf-8")

        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            warnings = check_overtrading_guards("BTC-USD", "SELL", "bold", {})
            rate_warns = [w for w in warnings if w["guard"] == "position_rate_limit"]
            assert len(rate_warns) == 0

    def test_old_positions_dont_count(self, clean_state):
        now = datetime.now(timezone.utc)
        state = {
            "ticker_trades": {},
            "consecutive_losses": {"patient": 0, "bold": 0},
            "new_position_timestamps": {
                "patient": [],
                "bold": [(now - timedelta(hours=10)).isoformat()],
            },
        }
        clean_state.write_text(json.dumps(state), encoding="utf-8")

        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            warnings = check_overtrading_guards("ETH-USD", "BUY", "bold", {})
            rate_warns = [w for w in warnings if w["guard"] == "position_rate_limit"]
            assert len(rate_warns) == 0

    def test_patient_has_longer_window(self, clean_state):
        now = datetime.now(timezone.utc)
        # 5 hours ago — inside patient's 8h window, outside bold's 4h window
        state = {
            "ticker_trades": {},
            "consecutive_losses": {"patient": 0, "bold": 0},
            "new_position_timestamps": {
                "patient": [(now - timedelta(hours=5)).isoformat()],
                "bold": [(now - timedelta(hours=5)).isoformat()],
            },
        }
        clean_state.write_text(json.dumps(state), encoding="utf-8")

        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            bold_warns = check_overtrading_guards("ETH-USD", "BUY", "bold", {})
            patient_warns = check_overtrading_guards("ETH-USD", "BUY", "patient", {})

            bold_rate = [w for w in bold_warns if w["guard"] == "position_rate_limit"]
            patient_rate = [w for w in patient_warns if w["guard"] == "position_rate_limit"]

            assert len(bold_rate) == 0  # Outside 4h window
            assert len(patient_rate) == 1  # Inside 8h window


# --- Record trade ---

class TestRecordTrade:
    def test_records_ticker_timestamp(self, clean_state):
        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            record_trade("BTC-USD", "BUY", "bold")
            state = json.loads(clean_state.read_text(encoding="utf-8"))
            assert "bold:BTC-USD" in state["ticker_trades"]

    def test_records_buy_timestamp_for_rate_limit(self, clean_state):
        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            record_trade("BTC-USD", "BUY", "bold")
            state = json.loads(clean_state.read_text(encoding="utf-8"))
            assert len(state["new_position_timestamps"]["bold"]) == 1

    def test_sell_loss_increments_consecutive(self, clean_state):
        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            record_trade("BTC-USD", "SELL", "bold", pnl_pct=-5.0)
            state = json.loads(clean_state.read_text(encoding="utf-8"))
            assert state["consecutive_losses"]["bold"] == 1

    def test_sell_win_resets_consecutive(self, clean_state):
        state = {
            "ticker_trades": {},
            "consecutive_losses": {"patient": 0, "bold": 3},
            "new_position_timestamps": {"patient": [], "bold": []},
        }
        clean_state.write_text(json.dumps(state), encoding="utf-8")

        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            record_trade("BTC-USD", "SELL", "bold", pnl_pct=2.5)
            state = json.loads(clean_state.read_text(encoding="utf-8"))
            assert state["consecutive_losses"]["bold"] == 0

    def test_sell_no_pnl_doesnt_change_streak(self, clean_state):
        state = {
            "ticker_trades": {},
            "consecutive_losses": {"patient": 0, "bold": 2},
            "new_position_timestamps": {"patient": [], "bold": []},
        }
        clean_state.write_text(json.dumps(state), encoding="utf-8")

        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            record_trade("BTC-USD", "SELL", "bold", pnl_pct=None)
            state = json.loads(clean_state.read_text(encoding="utf-8"))
            assert state["consecutive_losses"]["bold"] == 2


# --- Config disable ---

class TestGuardConfig:
    def test_disabled_returns_empty(self, clean_state):
        config = {"trade_guards": {"enabled": False}}
        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            warnings = check_overtrading_guards("BTC-USD", "BUY", "bold", {}, config)
            assert warnings == []

    def test_custom_cooldown(self, clean_state):
        now = datetime.now(timezone.utc)
        state = {
            "ticker_trades": {"bold:BTC-USD": (now - timedelta(minutes=20)).isoformat()},
            "consecutive_losses": {"patient": 0, "bold": 0},
            "new_position_timestamps": {"patient": [], "bold": []},
        }
        clean_state.write_text(json.dumps(state), encoding="utf-8")

        # 15 min cooldown — 20 min elapsed, should be clear
        config = {"trade_guards": {"ticker_cooldown_minutes": 15}}
        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            warnings = check_overtrading_guards("BTC-USD", "BUY", "bold", {}, config)
            cooldown_warns = [w for w in warnings if w["guard"] == "ticker_cooldown"]
            assert len(cooldown_warns) == 0


# --- get_all_guard_warnings ---

class TestGetAllGuardWarnings:
    def test_hold_signals_skip_checks(self, clean_state):
        signals = {"BTC-USD": {"action": "HOLD"}}
        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            result = get_all_guard_warnings(signals, {}, {})
            assert result["warnings"] == []

    def test_buy_signal_triggers_checks(self, clean_state):
        now = datetime.now(timezone.utc)
        state = {
            "ticker_trades": {"bold:BTC-USD": (now - timedelta(minutes=10)).isoformat()},
            "consecutive_losses": {"patient": 0, "bold": 0},
            "new_position_timestamps": {"patient": [], "bold": []},
        }
        clean_state.write_text(json.dumps(state), encoding="utf-8")

        signals = {"BTC-USD": {"action": "BUY"}}
        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            result = get_all_guard_warnings(signals, {}, {})
            assert len(result["warnings"]) >= 1

    def test_disabled_returns_empty(self, clean_state):
        config = {"trade_guards": {"enabled": False}}
        signals = {"BTC-USD": {"action": "BUY"}}
        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            result = get_all_guard_warnings(signals, {}, {}, config)
            assert result["warnings"] == []
            assert "disabled" in result["summary"].lower()
