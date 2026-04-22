"""Tests for trade guards (overtrading prevention)."""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from portfolio.trade_guards import (
    _get_cooldown_multiplier,
    check_overtrading_guards,
    get_all_guard_warnings,
    record_trade,
    should_block_trade,
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
        now = datetime.now(UTC)
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
        now = datetime.now(UTC)
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
        now = datetime.now(UTC)
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
        now = datetime.now(UTC)
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
        now = datetime.now(UTC)
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
        now = datetime.now(UTC)
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
        now = datetime.now(UTC)
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
        now = datetime.now(UTC)
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
        now = datetime.now(UTC)
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
        now = datetime.now(UTC)
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
        now = datetime.now(UTC)
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


# --- ARCH-29: should_block_trade ---

class TestShouldBlockTrade:
    """ARCH-29: Convenience function for go/no-go decisions."""

    def test_no_warnings_returns_false(self):
        result = {"warnings": [], "summary": "All clear"}
        assert should_block_trade(result) is False

    def test_warning_severity_returns_false(self):
        result = {"warnings": [{"severity": "warning", "guard": "cooldown"}]}
        assert should_block_trade(result) is False

    def test_block_severity_returns_true(self):
        result = {"warnings": [{"severity": "block", "guard": "cooldown"}]}
        assert should_block_trade(result) is True

    def test_mixed_severities_returns_true(self):
        result = {"warnings": [
            {"severity": "warning", "guard": "rate_limit"},
            {"severity": "block", "guard": "cooldown"},
        ]}
        assert should_block_trade(result) is True

    def test_empty_dict_returns_false(self):
        assert should_block_trade({}) is False

    def test_missing_warnings_key_returns_false(self):
        assert should_block_trade({"summary": "All clear"}) is False


# --- C4: Empty-state warning scoped to "wiring broken" case ---

class TestC4Warning:
    """Regression: 2026-04-22 — C4 fired every cycle on empty state even when
    no trades had happened yet (portfolios untouched). Now only warns when
    portfolios DO have transactions but guard state is still empty."""

    def test_no_warning_when_no_transactions_and_no_state(
        self, clean_state, caplog, tmp_path, monkeypatch,
    ):
        import logging
        monkeypatch.setattr("portfolio.trade_guards.DATA_DIR", tmp_path)
        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            with caplog.at_level(logging.WARNING, logger="portfolio.trade_guards"):
                get_all_guard_warnings({}, {}, {}, config={})
        assert "NON-FUNCTIONAL" not in caplog.text

    def test_warning_when_transactions_exist_but_state_empty(
        self, clean_state, caplog, tmp_path, monkeypatch,
    ):
        import logging
        monkeypatch.setattr("portfolio.trade_guards.DATA_DIR", tmp_path)
        (tmp_path / "portfolio_state.json").write_text(
            json.dumps({"transactions": [{"ticker": "BTC-USD", "action": "BUY"}]}),
            encoding="utf-8",
        )
        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            with caplog.at_level(logging.WARNING, logger="portfolio.trade_guards"):
                get_all_guard_warnings({}, {}, {}, config={})
        assert "NON-FUNCTIONAL" in caplog.text
        assert "wiring appears broken" in caplog.text

    def test_warning_fires_when_only_warrants_has_transactions(
        self, clean_state, caplog, tmp_path, monkeypatch,
    ):
        """Regression: 2026-04-22 follow-up — review flagged that warrants
        portfolio was excluded, leaving C4 silent if warrants was the only
        trading strategy."""
        import logging
        monkeypatch.setattr("portfolio.trade_guards.DATA_DIR", tmp_path)
        (tmp_path / "portfolio_state_warrants.json").write_text(
            json.dumps({"transactions": [{"ticker": "XBT-TRACKER", "action": "BUY"}]}),
            encoding="utf-8",
        )
        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            with caplog.at_level(logging.WARNING, logger="portfolio.trade_guards"):
                get_all_guard_warnings({}, {}, {}, config={})
        assert "NON-FUNCTIONAL" in caplog.text


# --- C4: Positive-proof wiring heartbeat ---

class TestRecordTradeWiringHeartbeat:
    """Regression: 2026-04-22 follow-up — on first record_trade() call in a
    process, log INFO confirming the wiring is alive. Gives operators
    positive proof instead of absence-of-warning inference."""

    def test_first_call_logs_wiring_confirmed(self, clean_state, caplog):
        import logging
        import portfolio.trade_guards as tg
        tg._wiring_confirmed = False  # reset module flag for test
        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            with caplog.at_level(logging.INFO, logger="portfolio.trade_guards"):
                record_trade("BTC-USD", "BUY", "patient")
        assert "wiring confirmed" in caplog.text

    def test_subsequent_calls_dont_relog(self, clean_state, caplog):
        import logging
        import portfolio.trade_guards as tg
        tg._wiring_confirmed = False
        with patch("portfolio.trade_guards.STATE_FILE", clean_state):
            record_trade("BTC-USD", "BUY", "patient")  # triggers the one-shot log
            caplog.clear()
            with caplog.at_level(logging.INFO, logger="portfolio.trade_guards"):
                record_trade("ETH-USD", "BUY", "bold")
        assert "wiring confirmed" not in caplog.text
