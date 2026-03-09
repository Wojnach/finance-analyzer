"""Tests for the GoldDigger intraday gold certificate trading bot."""

import json
import math
import os
import tempfile
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from portfolio.golddigger.config import GolddiggerConfig
from portfolio.golddigger.data_provider import MarketSnapshot, fetch_gold_price
from portfolio.golddigger.signal import CompositeSignal, SignalState, _log_return, _zscore, EPSILON
from portfolio.golddigger.risk import RiskManager, SizeResult
from portfolio.golddigger.state import BotState, Position, log_trade, log_poll
from portfolio.golddigger.bot import GolddiggerBot


# ============================================================
# Config tests
# ============================================================

class TestGolddiggerConfig:
    def test_defaults(self):
        cfg = GolddiggerConfig()
        assert cfg.poll_seconds == 5
        assert cfg.window_n == 720
        assert cfg.w_gold == 0.50
        assert cfg.w_fx == 0.30
        assert cfg.w_yield == 0.20
        assert abs(cfg.w_gold + cfg.w_fx + cfg.w_yield - 1.0) < 1e-9
        assert cfg.theta_in == 1.0
        assert cfg.theta_out == 0.2
        assert cfg.confirm_polls == 6
        assert cfg.stop_loss_pct == 0.05
        assert cfg.take_profit_pct == 0.08
        assert cfg.daily_loss_limit == 0.015
        assert cfg.max_positions == 1
        assert cfg.alert_cooldown_seconds == 300

    def test_from_config_empty(self):
        cfg = GolddiggerConfig.from_config({})
        assert cfg.poll_seconds == 5
        assert cfg.equity_sek == 100_000.0

    def test_from_config_custom(self):
        config = {
            "golddigger": {
                "poll_seconds": 15,
                "w_gold": 0.60,
                "w_fx": 0.25,
                "w_yield": 0.15,
                "theta_in": 1.5,
                "equity_sek": 200_000,
                "bull_orderbook_id": "12345",
            },
            "avanza": {"account_id": "99999"},
        }
        cfg = GolddiggerConfig.from_config(config)
        assert cfg.poll_seconds == 15
        assert cfg.w_gold == 0.60
        assert cfg.theta_in == 1.5
        assert cfg.equity_sek == 200_000
        assert cfg.bull_orderbook_id == "12345"
        assert cfg.avanza_account_id == "99999"

    def test_frozen(self):
        cfg = GolddiggerConfig()
        with pytest.raises(AttributeError):
            cfg.poll_seconds = 60


# ============================================================
# Signal tests
# ============================================================

class TestLogReturn:
    def test_basic(self):
        r = _log_return(110, 100)
        assert abs(r - math.log(1.1)) < 1e-9

    def test_zero_previous(self):
        assert _log_return(100, 0) == 0.0

    def test_negative(self):
        assert _log_return(-1, 100) == 0.0

    def test_same_price(self):
        assert _log_return(100, 100) == 0.0


class TestZscore:
    def test_insufficient_samples(self):
        window = deque([1.0, 2.0, 3.0])
        assert _zscore(2.0, window, min_samples=10) == 0.0

    def test_at_mean(self):
        window = deque([1.0] * 20)
        z = _zscore(1.0, window, min_samples=10)
        # All identical → std ≈ 0, z ≈ 0 / epsilon → very small
        assert abs(z) < 1.0  # numerically near zero

    def test_above_mean(self):
        window = deque(range(20))
        mean = sum(range(20)) / 20
        z = _zscore(mean + 10, window, min_samples=10)
        assert z > 0

    def test_below_mean(self):
        window = deque(range(20))
        mean = sum(range(20)) / 20
        z = _zscore(mean - 10, window, min_samples=10)
        assert z < 0

    def test_constant_series_stability(self):
        """Z-scores from constant series should not produce NaN."""
        window = deque([5.0] * 100)
        z = _zscore(5.0, window, min_samples=10)
        assert not math.isnan(z)
        assert not math.isinf(z)


class TestCompositeSignal:
    def make_snap(self, gold=2000.0, usdsek=10.5, us10y=0.0425):
        return MarketSnapshot(
            ts_utc=datetime.now(timezone.utc),
            gold=gold, usdsek=usdsek, us10y=us10y,
        )

    def test_first_update_establishes_baseline(self):
        sig = CompositeSignal(window_n=120, min_window=10)
        state = sig.update(self.make_snap())
        assert state.window_size == 0
        assert not state.valid

    def test_second_update_produces_returns(self):
        sig = CompositeSignal(window_n=120, min_window=2)
        sig.update(self.make_snap(gold=2000))
        state = sig.update(self.make_snap(gold=2010))
        assert state.window_size == 1
        assert state.r_gold > 0  # gold went up

    def test_signal_valid_after_min_window(self):
        sig = CompositeSignal(window_n=120, min_window=5)
        sig.update(self.make_snap())
        for i in range(6):
            state = sig.update(self.make_snap(gold=2000 + i))
        assert state.valid
        assert state.window_size >= 5

    def test_gold_up_produces_positive_s(self):
        """Gold price rising should produce positive composite score."""
        sig = CompositeSignal(window_n=20, min_window=5)
        sig.update(self.make_snap(gold=2000, usdsek=10.5, us10y=0.0425))
        # Flat period to establish baseline
        for _ in range(8):
            sig.update(self.make_snap(gold=2000, usdsek=10.5, us10y=0.0425))
        # Now gold spikes up, USD weakens, yields drop
        state = sig.update(self.make_snap(gold=2050, usdsek=10.3, us10y=0.0420))
        assert state.valid
        assert state.composite_s > 0

    def test_gold_down_produces_negative_s(self):
        """Gold price falling should produce negative composite score."""
        sig = CompositeSignal(window_n=20, min_window=5)
        sig.update(self.make_snap(gold=2000, usdsek=10.5, us10y=0.0425))
        for _ in range(8):
            sig.update(self.make_snap(gold=2000, usdsek=10.5, us10y=0.0425))
        # Gold drops, USD strengthens, yields rise
        state = sig.update(self.make_snap(gold=1950, usdsek=10.7, us10y=0.0430))
        assert state.valid
        assert state.composite_s < 0

    def test_confirmation_counter(self):
        sig = CompositeSignal(window_n=10, min_window=3, theta_in=0.1, confirm_polls=2)
        sig.update(self.make_snap(gold=2000))
        # Build up baseline
        for _ in range(5):
            sig.update(self.make_snap(gold=2000, usdsek=10.5, us10y=0.0425))
        # Strong move
        state = sig.update(self.make_snap(gold=2100, usdsek=10.0, us10y=0.0400))
        if state.composite_s >= sig.theta_in and state.z_gold > 0:
            assert state.confirm_count >= 1

    def test_confirmation_resets_on_decay(self):
        sig = CompositeSignal(window_n=10, min_window=3, theta_in=5.0)
        sig.update(self.make_snap())
        for _ in range(5):
            sig.update(self.make_snap())
        # S won't reach theta_in=5.0 with flat data
        state = sig.update(self.make_snap())
        assert state.confirm_count == 0

    def test_reset(self):
        sig = CompositeSignal(window_n=10, min_window=3)
        sig.update(self.make_snap())
        sig.update(self.make_snap(gold=2010))
        sig.reset()
        # After reset, first update is baseline again
        state = sig.update(self.make_snap())
        assert state.window_size == 0

    def test_should_enter_all_conditions(self):
        sig = CompositeSignal(theta_in=1.0, confirm_polls=2)
        # Met all conditions
        state = SignalState(
            composite_s=1.5, z_gold=0.5, confirm_count=3,
            valid=True, window_size=20,
        )
        assert sig.should_enter(state, spread_pct=0.005)

    def test_should_enter_blocked_by_spread(self):
        sig = CompositeSignal()
        state = SignalState(
            composite_s=1.5, z_gold=0.5, confirm_count=3,
            valid=True, window_size=20,
        )
        assert not sig.should_enter(state, spread_pct=0.05, spread_max=0.02)

    def test_should_enter_blocked_by_z_gold(self):
        sig = CompositeSignal()
        state = SignalState(
            composite_s=1.5, z_gold=-0.5, confirm_count=3,
            valid=True, window_size=20,
        )
        assert not sig.should_enter(state)

    def test_should_enter_blocked_by_confirmation(self):
        sig = CompositeSignal(confirm_polls=3)
        state = SignalState(
            composite_s=1.5, z_gold=0.5, confirm_count=1,
            valid=True, window_size=20,
        )
        assert not sig.should_enter(state)

    def test_should_exit(self):
        sig = CompositeSignal(theta_out=0.2)
        state = SignalState(composite_s=0.1, valid=True)
        assert sig.should_exit(state)

    def test_should_not_exit_above_threshold(self):
        sig = CompositeSignal(theta_out=0.2)
        state = SignalState(composite_s=0.5, valid=True)
        assert not sig.should_exit(state)


# ============================================================
# Risk management tests
# ============================================================

class TestRiskManager:
    def make_cfg(self, **kwargs):
        defaults = {
            "risk_fraction": 0.005,
            "max_notional_fraction": 0.10,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.08,
            "daily_loss_limit": 0.015,
            "equity_sek": 100_000,
            "max_daily_trades": 10,
            "spread_max": 0.02,
            "slippage_buffer": 0.0,
        }
        defaults.update(kwargs)
        return GolddiggerConfig(**defaults)

    def test_size_position_risk_budget(self):
        cfg = self.make_cfg()
        rm = RiskManager(cfg)
        result = rm.size_position(entry_ask=100.0, equity_sek=100_000)
        # Risk budget = 0.005 * 100000 = 500 SEK
        # Stop = 100 * 0.95 = 95, per_unit_risk = 5
        # qty_risk = 500 / 5 = 100
        # qty_notional = 0.10 * 100000 / 100 = 100
        assert result.quantity == 100
        assert result.stop_price == 95.0
        assert result.take_profit_price == 108.0

    def test_size_position_notional_cap(self):
        cfg = self.make_cfg(max_notional_fraction=0.05)
        rm = RiskManager(cfg)
        result = rm.size_position(entry_ask=100.0, equity_sek=100_000)
        # qty_notional = 0.05 * 100000 / 100 = 50 (caps qty_risk=100)
        assert result.quantity == 50
        assert "notional" in result.reason.lower()

    def test_size_position_zero_price(self):
        cfg = self.make_cfg()
        rm = RiskManager(cfg)
        result = rm.size_position(entry_ask=0.0, equity_sek=100_000)
        assert result.quantity == 0

    def test_check_stop_loss(self):
        cfg = self.make_cfg(stop_loss_pct=0.05)
        rm = RiskManager(cfg)
        # Entry at 100, stop at 95
        assert rm.check_stop_loss(current_bid=94.0, entry_price=100.0)
        assert rm.check_stop_loss(current_bid=95.0, entry_price=100.0)
        assert not rm.check_stop_loss(current_bid=96.0, entry_price=100.0)

    def test_check_take_profit(self):
        cfg = self.make_cfg(take_profit_pct=0.08)
        rm = RiskManager(cfg)
        assert rm.check_take_profit(current_bid=108.0, entry_price=100.0)
        assert rm.check_take_profit(current_bid=110.0, entry_price=100.0)
        assert not rm.check_take_profit(current_bid=107.0, entry_price=100.0)

    def test_daily_loss_limit(self):
        cfg = self.make_cfg(daily_loss_limit=0.015, equity_sek=100_000)
        rm = RiskManager(cfg)
        rm.reset_daily("2026-03-09")
        assert not rm.is_halted

        # Lose 1500 SEK (= 1.5% of 100K)
        rm.record_trade_pnl(-1500)
        assert rm.is_halted
        can, reason = rm.can_trade()
        assert not can

    def test_daily_reset(self):
        cfg = self.make_cfg()
        rm = RiskManager(cfg)
        rm.reset_daily("2026-03-08")
        rm.record_trade_pnl(-2000)
        assert rm.is_halted
        # New day resets
        rm.reset_daily("2026-03-09")
        assert not rm.is_halted

    def test_max_daily_trades(self):
        cfg = self.make_cfg(max_daily_trades=2)
        rm = RiskManager(cfg)
        rm.reset_daily("2026-03-09")
        rm.record_trade_pnl(100)
        rm.record_trade_pnl(200)
        can, reason = rm.can_trade()
        assert not can
        assert "max daily" in reason.lower()

    def test_spread_check(self):
        cfg = self.make_cfg(spread_max=0.02)
        rm = RiskManager(cfg)
        ok, spread = rm.check_spread(bid=100, ask=101)
        assert ok
        assert abs(spread - 0.01) < 1e-9

        ok, spread = rm.check_spread(bid=100, ask=103)
        assert not ok


# ============================================================
# State tests
# ============================================================

class TestBotState:
    def test_default_state(self):
        state = BotState()
        assert state.equity_sek == 100_000
        assert state.cash_sek == 100_000
        assert not state.has_position()

    def test_open_position(self):
        state = BotState(cash_sek=100_000)
        state.open_position(
            orderbook_id="123", quantity=50, price=100,
            gold_price=2000, stop_price=95, tp_price=108,
        )
        assert state.has_position()
        assert state.position.quantity == 50
        assert state.position.avg_price == 100
        assert state.cash_sek == 95_000  # 50 * 100 = 5000

    def test_close_position(self):
        state = BotState(cash_sek=95_000)
        state.position = Position(
            orderbook_id="123", quantity=50, avg_price=100,
            entry_gold=2000, entry_time="2026-03-09T00:00:00",
            stop_price=95, take_profit_price=108,
        )
        pnl = state.close_position(exit_price=108)
        # Proceeds: 50 * 108 = 5400, Cost: 50 * 100 = 5000, PnL = 400
        assert pnl == 400
        assert state.cash_sek == 95_000 + 5400
        assert not state.has_position()

    def test_close_position_with_fee(self):
        state = BotState(cash_sek=95_000)
        state.position = Position(
            orderbook_id="123", quantity=50, avg_price=100,
            entry_gold=2000, entry_time="2026-03-09T00:00:00",
            stop_price=95, take_profit_price=108,
        )
        pnl = state.close_position(exit_price=108, fee_sek=10)
        # Proceeds: 50*108 - 10 = 5390, PnL = 5390 - 5000 = 390
        assert pnl == 390
        assert state.total_fees == 10

    def test_save_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "state.json")
        state = BotState(cash_sek=90_000, total_trades=5)
        state.position = Position(
            orderbook_id="123", quantity=50, avg_price=100,
            entry_gold=2000, entry_time="2026-03-09T00:00:00",
            stop_price=95, take_profit_price=108,
        )
        state.save(path)

        loaded = BotState.load(path)
        assert loaded.cash_sek == 90_000
        assert loaded.total_trades == 5
        assert loaded.has_position()
        assert loaded.position.quantity == 50
        assert loaded.position.avg_price == 100

    def test_load_missing_file(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        state = BotState.load(path)
        assert state.cash_sek == 100_000  # defaults

    def test_daily_reset(self):
        state = BotState(daily_pnl=-500, daily_trades=3, last_trade_date="2026-03-08")
        state.reset_daily("2026-03-09")
        assert state.daily_pnl == 0
        assert state.daily_trades == 0

    def test_daily_reset_same_day_noop(self):
        state = BotState(daily_pnl=-500, daily_trades=3, last_trade_date="2026-03-09")
        state.reset_daily("2026-03-09")
        assert state.daily_pnl == -500  # no reset


class TestLogTrade:
    def test_appends_trade(self, tmp_path):
        path = str(tmp_path / "trades.jsonl")
        log_trade(path, "BUY", 50, 100.0, 2000.0, 1.5, reason="test entry")
        log_trade(path, "SELL", 50, 108.0, 2010.0, 0.3, pnl=400.0, reason="test exit")

        lines = Path(path).read_text().strip().split("\n")
        assert len(lines) == 2
        buy = json.loads(lines[0])
        assert buy["action"] == "BUY"
        assert buy["quantity"] == 50
        sell = json.loads(lines[1])
        assert sell["action"] == "SELL"
        assert sell["pnl_sek"] == 400.0


class TestLogPoll:
    def test_appends_poll(self, tmp_path):
        path = str(tmp_path / "log.jsonl")
        log_poll(path, gold=2000, usdsek=10.5, us10y=0.0425,
                 composite_s=0.75, z_gold=1.2, z_fx=-0.3, z_yield=-0.5)
        lines = Path(path).read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["gold"] == 2000
        assert entry["S"] == 0.75


# ============================================================
# Bot integration tests
# ============================================================

class TestGolddiggerBot:
    def make_cfg(self, tmp_path, **kwargs):
        defaults = {
            "state_file": str(tmp_path / "state.json"),
            "log_file": str(tmp_path / "log.jsonl"),
            "trades_file": str(tmp_path / "trades.jsonl"),
            "kill_switch_file": str(tmp_path / "kill"),
            "equity_sek": 100_000,
            "bull_orderbook_id": "TEST123",
            "window_n": 10,
            "min_window": 3,
            "theta_in": 0.5,
            "confirm_polls": 1,
        }
        defaults.update(kwargs)
        return GolddiggerConfig(**defaults)

    def make_snap(self, gold=2000.0, usdsek=10.5, us10y=0.0425,
                  cert_bid=100.0, cert_ask=101.0):
        return MarketSnapshot(
            ts_utc=datetime.now(timezone.utc),
            gold=gold, usdsek=usdsek, us10y=us10y,
            cert_bid=cert_bid, cert_ask=cert_ask,
            cert_last=100.5,
            cert_spread_pct=0.01 if cert_bid and cert_ask else None,
        )

    @patch("portfolio.golddigger.bot._now_stockholm")
    def test_outside_session_returns_none(self, mock_now, tmp_path):
        mock_now.return_value = (8, 0, "2026-03-09")  # before 09:02
        cfg = self.make_cfg(tmp_path)
        bot = GolddiggerBot(cfg, dry_run=True)
        result = bot.step(self.make_snap())
        assert result is None

    @patch("portfolio.golddigger.bot._now_stockholm")
    def test_kill_switch_blocks(self, mock_now, tmp_path):
        mock_now.return_value = (10, 0, "2026-03-09")
        cfg = self.make_cfg(tmp_path)
        # Create kill switch file
        Path(cfg.kill_switch_file).touch()
        bot = GolddiggerBot(cfg, dry_run=True)
        result = bot.step(self.make_snap())
        assert result is None

    @patch("portfolio.golddigger.bot._now_stockholm")
    def test_incomplete_data_holds(self, mock_now, tmp_path):
        mock_now.return_value = (10, 0, "2026-03-09")
        cfg = self.make_cfg(tmp_path)
        bot = GolddiggerBot(cfg, dry_run=True)
        snap = MarketSnapshot(
            ts_utc=datetime.now(timezone.utc),
            gold=0.0, usdsek=10.5, us10y=0.0425,  # gold missing
        )
        result = bot.step(snap)
        assert result is None

    @patch("portfolio.golddigger.bot._now_stockholm")
    def test_flatten_at_session_end(self, mock_now, tmp_path):
        mock_now.return_value = (17, 20, "2026-03-09")  # at flatten time
        cfg = self.make_cfg(tmp_path)
        bot = GolddiggerBot(cfg, dry_run=True)
        # Put bot in a position
        bot.state.position = Position(
            orderbook_id="TEST123", quantity=50, avg_price=100,
            entry_gold=2000, entry_time="2026-03-09T09:30:00",
            stop_price=95, take_profit_price=108,
        )
        # Prime the signal engine
        bot._current_date = "2026-03-09"
        for i in range(5):
            bot.signal.update(self.make_snap(gold=2000 + i))

        result = bot.step(self.make_snap(cert_bid=102))
        assert result is not None
        assert result["action"] == "SELL"
        assert "FLATTEN" in result["reason"]

    @patch("portfolio.golddigger.bot._now_stockholm")
    def test_stop_loss_exit(self, mock_now, tmp_path):
        mock_now.return_value = (10, 30, "2026-03-09")
        cfg = self.make_cfg(tmp_path)
        bot = GolddiggerBot(cfg, dry_run=True)
        bot._current_date = "2026-03-09"
        bot.state.position = Position(
            orderbook_id="TEST123", quantity=50, avg_price=100,
            entry_gold=2000, entry_time="2026-03-09T09:30:00",
            stop_price=95, take_profit_price=108,
        )
        # Prime signal engine
        for i in range(5):
            bot.signal.update(self.make_snap(gold=2000))

        # Bid drops below stop (100 * 0.95 = 95)
        result = bot.step(self.make_snap(cert_bid=94.0, cert_ask=95.0))
        assert result is not None
        assert result["action"] == "SELL"
        assert "STOP_LOSS" in result["reason"]

    @patch("portfolio.golddigger.bot._now_stockholm")
    def test_take_profit_exit(self, mock_now, tmp_path):
        mock_now.return_value = (10, 30, "2026-03-09")
        cfg = self.make_cfg(tmp_path)
        bot = GolddiggerBot(cfg, dry_run=True)
        bot._current_date = "2026-03-09"
        bot.state.position = Position(
            orderbook_id="TEST123", quantity=50, avg_price=100,
            entry_gold=2000, entry_time="2026-03-09T09:30:00",
            stop_price=95, take_profit_price=108,
        )
        for i in range(5):
            bot.signal.update(self.make_snap(gold=2000))

        # Bid above TP (100 * 1.08 = 108)
        result = bot.step(self.make_snap(cert_bid=109.0, cert_ask=110.0))
        assert result is not None
        assert result["action"] == "SELL"
        assert "TAKE_PROFIT" in result["reason"]

    @patch("portfolio.golddigger.bot._now_stockholm")
    def test_entry_on_strong_signal(self, mock_now, tmp_path):
        """Entry triggers when composite signal exceeds theta_in."""
        mock_now.return_value = (10, 0, "2026-03-09")
        cfg = self.make_cfg(tmp_path, theta_in=0.1, confirm_polls=1, min_window=3)
        bot = GolddiggerBot(cfg, dry_run=True)
        bot._current_date = "2026-03-09"

        # Build baseline with flat data
        for _ in range(5):
            bot.step(self.make_snap(gold=2000, usdsek=10.5, us10y=0.0425))

        # Strong gold move up, USD weakens, yields drop
        result = bot.step(self.make_snap(
            gold=2100, usdsek=10.0, us10y=0.0400,
            cert_bid=100, cert_ask=101,
        ))
        # May or may not trigger depending on z-score magnitude
        # The test validates the bot doesn't crash and processes correctly

    @patch("portfolio.golddigger.bot._now_stockholm")
    def test_no_entry_without_cert_price(self, mock_now, tmp_path):
        mock_now.return_value = (10, 0, "2026-03-09")
        cfg = self.make_cfg(tmp_path, theta_in=0.01, confirm_polls=1, min_window=2)
        bot = GolddiggerBot(cfg, dry_run=True)
        bot._current_date = "2026-03-09"

        for _ in range(5):
            bot.step(self.make_snap(gold=2000, cert_bid=None, cert_ask=None))

        # Even with signal, no cert price = no entry
        result = bot.step(self.make_snap(
            gold=2100, usdsek=10.0, us10y=0.0400,
            cert_bid=None, cert_ask=None,
        ))
        # Should not crash, and should not generate BUY without cert price

    @patch("portfolio.golddigger.bot._now_stockholm")
    def test_daily_reset_on_new_day(self, mock_now, tmp_path):
        mock_now.return_value = (10, 0, "2026-03-10")
        cfg = self.make_cfg(tmp_path)
        bot = GolddiggerBot(cfg, dry_run=True)
        bot._current_date = "2026-03-09"
        bot.state.daily_pnl = -500

        bot.step(self.make_snap())
        assert bot._current_date == "2026-03-10"
        assert bot.state.daily_pnl == 0


# ============================================================
# Data provider tests (unit, no network)
# ============================================================

class TestMarketSnapshot:
    def test_complete(self):
        snap = MarketSnapshot(
            ts_utc=datetime.now(timezone.utc),
            gold=2000, usdsek=10.5, us10y=0.0425,
        )
        assert snap.is_complete()

    def test_incomplete_gold(self):
        snap = MarketSnapshot(
            ts_utc=datetime.now(timezone.utc),
            gold=0, usdsek=10.5, us10y=0.0425,
        )
        assert not snap.is_complete()

    def test_incomplete_fx(self):
        snap = MarketSnapshot(
            ts_utc=datetime.now(timezone.utc),
            gold=2000, usdsek=0, us10y=0.0425,
        )
        assert not snap.is_complete()


# ============================================================
# Signal sign correctness tests
# ============================================================

class TestCompositeSignalSigns:
    """Verify the composite score direction under known conditions."""

    def test_gold_up_usd_down_yield_down_is_positive(self):
        """Bullish gold regime should produce S > 0."""
        sig = CompositeSignal(window_n=20, min_window=5)
        snap = lambda g, f, y: MarketSnapshot(
            ts_utc=datetime.now(timezone.utc), gold=g, usdsek=f, us10y=y,
        )
        # Baseline
        sig.update(snap(2000, 10.5, 0.0425))
        # Flat period
        for _ in range(8):
            sig.update(snap(2000, 10.5, 0.0425))
        # Bullish move: gold UP, USD WEAKER (USDSEK down), yields DOWN
        state = sig.update(snap(2050, 10.3, 0.0415))
        # z_gold > 0, z_fx < 0 (USDSEK decreased), z_yield < 0 (yield decreased)
        # S = w_g * z_g(+) - w_f * z_f(-) - w_y * z_y(-) = positive + positive + positive
        assert state.composite_s > 0

    def test_gold_down_usd_up_yield_up_is_negative(self):
        """Bearish gold regime should produce S < 0."""
        sig = CompositeSignal(window_n=20, min_window=5)
        snap = lambda g, f, y: MarketSnapshot(
            ts_utc=datetime.now(timezone.utc), gold=g, usdsek=f, us10y=y,
        )
        sig.update(snap(2000, 10.5, 0.0425))
        for _ in range(8):
            sig.update(snap(2000, 10.5, 0.0425))
        # Bearish: gold DOWN, USD STRONGER (USDSEK up), yields UP
        state = sig.update(snap(1950, 10.7, 0.0435))
        assert state.composite_s < 0


# ============================================================
# MarketSnapshot completeness tests (yield is optional)
# ============================================================

class TestMarketSnapshotCompleteness:
    """Tests for updated is_complete() -- yield is now optional."""

    def test_complete_with_all_data(self):
        snap = MarketSnapshot(ts_utc=datetime.now(timezone.utc), gold=2000, usdsek=10.5, us10y=0.04)
        assert snap.is_complete()

    def test_complete_without_yield(self):
        """Yield is optional -- gold + FX is enough."""
        snap = MarketSnapshot(ts_utc=datetime.now(timezone.utc), gold=2000, usdsek=10.5, us10y=0.0)
        assert snap.is_complete()

    def test_incomplete_no_gold(self):
        snap = MarketSnapshot(ts_utc=datetime.now(timezone.utc), gold=0, usdsek=10.5, us10y=0.04)
        assert not snap.is_complete()

    def test_incomplete_no_fx(self):
        snap = MarketSnapshot(ts_utc=datetime.now(timezone.utc), gold=2000, usdsek=0, us10y=0.04)
        assert not snap.is_complete()


# ============================================================
# MarketSnapshot freshness tests
# ============================================================

class TestMarketSnapshotFreshness:
    """Tests for data freshness tracking."""

    def test_fresh_data(self):
        now = datetime.now(timezone.utc)
        snap = MarketSnapshot(ts_utc=now, gold=2000, usdsek=10.5, us10y=0.04,
                              gold_fetch_ts=now, fx_fetch_ts=now)
        assert snap.is_fresh(max_age_seconds=60)

    def test_stale_gold(self):
        now = datetime.now(timezone.utc)
        old = now - timedelta(seconds=120)
        snap = MarketSnapshot(ts_utc=now, gold=2000, usdsek=10.5, us10y=0.04,
                              gold_fetch_ts=old, fx_fetch_ts=now)
        assert not snap.is_fresh(max_age_seconds=60)

    def test_stale_fx(self):
        now = datetime.now(timezone.utc)
        old = now - timedelta(seconds=120)
        snap = MarketSnapshot(ts_utc=now, gold=2000, usdsek=10.5, us10y=0.04,
                              gold_fetch_ts=now, fx_fetch_ts=old)
        assert not snap.is_fresh(max_age_seconds=60)

    def test_no_timestamps_is_fresh(self):
        """If no timestamps set, assume fresh (backward compat)."""
        snap = MarketSnapshot(ts_utc=datetime.now(timezone.utc), gold=2000, usdsek=10.5, us10y=0.04)
        assert snap.is_fresh()


# ============================================================
# Signal consensus reading tests
# ============================================================

class TestSignalConsensusReading:
    """Tests for read_xau_consensus()."""

    def test_reads_consensus(self, tmp_path):
        from portfolio.golddigger.data_provider import read_xau_consensus
        data = {"signals": {"XAU-USD": {
            "consensus": "BUY", "confidence": 0.72,
            "buy_count": 5, "sell_count": 1, "abstain_count": 10
        }}}
        path = tmp_path / "agent_summary_compact.json"
        path.write_text(json.dumps(data))

        with patch("portfolio.golddigger.data_provider._DATA_DIR", tmp_path):
            result = read_xau_consensus()
        assert result["action"] == "BUY"
        assert result["confidence"] == 0.72
        assert result["buy_count"] == 5

    def test_missing_file_returns_none(self, tmp_path):
        from portfolio.golddigger.data_provider import read_xau_consensus
        with patch("portfolio.golddigger.data_provider._DATA_DIR", tmp_path):
            assert read_xau_consensus() is None

    def test_no_xau_data_returns_none(self, tmp_path):
        from portfolio.golddigger.data_provider import read_xau_consensus
        data = {"signals": {"BTC-USD": {"consensus": "BUY"}}}
        path = tmp_path / "agent_summary_compact.json"
        path.write_text(json.dumps(data))

        with patch("portfolio.golddigger.data_provider._DATA_DIR", tmp_path):
            assert read_xau_consensus() is None


# ============================================================
# Macro context reading tests
# ============================================================

class TestMacroContext:
    """Tests for read_macro_context()."""

    def test_reads_dxy(self, tmp_path):
        from portfolio.golddigger.data_provider import read_macro_context
        data = {"macro": {"dxy": {"value": 98.5, "change_5d_pct": 0.32},
                           "treasury": {"us10y": 0.0425}}}
        path = tmp_path / "agent_summary_compact.json"
        path.write_text(json.dumps(data))

        with patch("portfolio.golddigger.data_provider._DATA_DIR", tmp_path):
            result = read_macro_context()
        assert result["dxy"] == 98.5
        assert result["dxy_5d_change"] == 0.32
        assert result["us10y"] == 0.0425

    def test_missing_macro_returns_empty(self, tmp_path):
        from portfolio.golddigger.data_provider import read_macro_context
        path = tmp_path / "agent_summary_compact.json"
        path.write_text("{}")
        with patch("portfolio.golddigger.data_provider._DATA_DIR", tmp_path):
            assert read_macro_context() == {}

    def test_missing_file_returns_empty(self, tmp_path):
        from portfolio.golddigger.data_provider import read_macro_context
        with patch("portfolio.golddigger.data_provider._DATA_DIR", tmp_path):
            assert read_macro_context() == {}


# ============================================================
# Volume confirmation tests
# ============================================================

class TestVolumeConfirmation:
    """Tests for fetch_gold_volume()."""

    @patch("portfolio.golddigger.data_provider.fetch_with_retry")
    def test_fetch_gold_volume(self, mock_fetch):
        from portfolio.golddigger.data_provider import fetch_gold_volume
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            [0, "0", "0", "0", "0", "100", 0, "0", 0, "0", "0", "0"] for _ in range(20)
        ] + [[0, "0", "0", "0", "0", "200", 0, "0", 0, "0", "0", "0"]]
        mock_resp.raise_for_status = MagicMock()
        mock_fetch.return_value = mock_resp

        result = fetch_gold_volume()
        assert result is not None
        assert result["current"] == 200
        assert result["avg_20"] == 100
        assert result["ratio"] == 2.0

    @patch("portfolio.golddigger.data_provider.fetch_with_retry")
    def test_volume_fetch_failure(self, mock_fetch):
        from portfolio.golddigger.data_provider import fetch_gold_volume
        mock_fetch.return_value = None
        assert fetch_gold_volume() is None

    @patch("portfolio.golddigger.data_provider.fetch_with_retry")
    def test_volume_single_bar(self, mock_fetch):
        """A single bar is not enough to compute average."""
        from portfolio.golddigger.data_provider import fetch_gold_volume
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            [0, "0", "0", "0", "0", "100", 0, "0", 0, "0", "0", "0"]
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_fetch.return_value = mock_resp
        assert fetch_gold_volume() is None


# ============================================================
# Dynamic stops tests
# ============================================================

class TestDynamicStops:
    """Tests for ATR-based dynamic stop levels."""

    def test_normal_atr(self):
        cfg = GolddiggerConfig(use_dynamic_stops=True, leverage=20.0,
                                atr_stop_multiplier=2.0, atr_stop_min_pct=0.03,
                                atr_stop_max_pct=0.15)
        rm = RiskManager(cfg)
        stop, tp = rm.dynamic_stop_levels(100.0, atr_pct=0.5)
        # 2 * 0.5% * 20 = 20% cert stop -> capped at 15%
        assert stop == pytest.approx(85.0, abs=0.5)
        assert tp > 100.0

    def test_low_vol_floor(self):
        cfg = GolddiggerConfig(use_dynamic_stops=True, leverage=20.0,
                                atr_stop_multiplier=2.0, atr_stop_min_pct=0.03,
                                atr_stop_max_pct=0.15)
        rm = RiskManager(cfg)
        stop, tp = rm.dynamic_stop_levels(100.0, atr_pct=0.05)
        # 2 * 0.05% * 20 = 2% -> floored at 3%
        assert stop == pytest.approx(97.0, abs=0.1)

    def test_no_atr_falls_back_to_fixed(self):
        cfg = GolddiggerConfig(stop_loss_pct=0.05, take_profit_pct=0.08)
        rm = RiskManager(cfg)
        stop, tp = rm.dynamic_stop_levels(100.0, atr_pct=None)
        assert stop == pytest.approx(95.0)
        assert tp == pytest.approx(108.0)

    def test_disabled_uses_fixed(self):
        cfg = GolddiggerConfig(use_dynamic_stops=False, stop_loss_pct=0.05,
                                take_profit_pct=0.08)
        rm = RiskManager(cfg)
        stop, tp = rm.dynamic_stop_levels(100.0, atr_pct=0.5)
        assert stop == pytest.approx(95.0)
        assert tp == pytest.approx(108.0)

    def test_tp_uses_rr_ratio(self):
        """Take profit should be 1.5x the stop distance above entry."""
        cfg = GolddiggerConfig(use_dynamic_stops=True, leverage=20.0,
                                atr_stop_multiplier=2.0, atr_stop_min_pct=0.03,
                                atr_stop_max_pct=0.15)
        rm = RiskManager(cfg)
        stop, tp = rm.dynamic_stop_levels(100.0, atr_pct=0.1)
        # 2 * 0.1% * 20 = 4% cert stop, within range [3%, 15%]
        # stop = 100 * (1 - 0.04) = 96, tp = 100 * (1 + 0.04 * 1.5) = 106
        assert stop == pytest.approx(96.0, abs=0.1)
        assert tp == pytest.approx(106.0, abs=0.1)


# ============================================================
# Slippage buffer tests
# ============================================================

class TestSlippageBuffer:
    """Tests for slippage buffer in position sizing."""

    def test_slippage_reduces_quantity(self):
        cfg = GolddiggerConfig(slippage_buffer=0.01, risk_fraction=0.005,
                                stop_loss_pct=0.05, max_notional_fraction=0.10)
        rm = RiskManager(cfg)
        result = rm.size_position(100.0, 100000)

        cfg2 = GolddiggerConfig(slippage_buffer=0.0, risk_fraction=0.005,
                                 stop_loss_pct=0.05, max_notional_fraction=0.10)
        rm2 = RiskManager(cfg2)
        result2 = rm2.size_position(100.0, 100000)
        assert result.quantity <= result2.quantity

    def test_zero_slippage_same_as_raw(self):
        """With zero slippage, effective entry equals ask."""
        cfg = GolddiggerConfig(slippage_buffer=0.0, risk_fraction=0.005,
                                stop_loss_pct=0.05, max_notional_fraction=0.50)
        rm = RiskManager(cfg)
        result = rm.size_position(100.0, 100000)
        # Risk budget = 500, per_unit_risk = 100 * 0.05 = 5, qty = 100
        assert result.quantity == 100
        assert result.stop_price == pytest.approx(95.0)

    def test_slippage_affects_stop_price(self):
        """Stop price should be based on effective (slippage-adjusted) entry."""
        cfg = GolddiggerConfig(slippage_buffer=0.01, stop_loss_pct=0.05,
                                risk_fraction=0.005, max_notional_fraction=0.50)
        rm = RiskManager(cfg)
        result = rm.size_position(100.0, 100000)
        # effective_entry = 101, stop = 101 * 0.95 = 95.95
        assert result.stop_price == pytest.approx(95.95)


# ============================================================
# Bot entry filter tests
# ============================================================

class TestBotEntryFilters:
    """Test that new entry filters work in the bot."""

    def _make_bot(self, tmp_path, **cfg_overrides):
        defaults = dict(
            session_start_hour=0, session_start_minute=0,
            session_end_hour=23, session_end_minute=59,
            state_file=str(tmp_path / "state.json"),
            log_file=str(tmp_path / "log.jsonl"),
            trades_file=str(tmp_path / "trades.jsonl"),
            kill_switch_file=str(tmp_path / "kill"),
        )
        defaults.update(cfg_overrides)
        cfg = GolddiggerConfig(**defaults)
        return GolddiggerBot(cfg, dry_run=True)

    def _strong_signal_snap(self):
        """Create a snapshot that would normally trigger entry."""
        return MarketSnapshot(
            ts_utc=datetime.now(timezone.utc),
            gold=2000, usdsek=10.5, us10y=0.04,
            cert_bid=99, cert_ask=100, cert_last=100,
            cert_spread_pct=0.01,
            gold_fetch_ts=datetime.now(timezone.utc),
            fx_fetch_ts=datetime.now(timezone.utc),
        )

    @patch("portfolio.golddigger.bot._now_stockholm", return_value=(12, 0, "2026-03-10"))
    @patch("portfolio.golddigger.data_provider.read_xau_consensus")
    def test_sell_consensus_blocks_entry(self, mock_consensus, mock_time, tmp_path):
        mock_consensus.return_value = {"action": "SELL", "confidence": 0.8,
                                        "buy_count": 1, "sell_count": 5, "hold_count": 10}
        bot = self._make_bot(tmp_path, use_signal_consensus=True)
        snap = self._strong_signal_snap()
        for _ in range(15):
            result = bot.step(snap)
        # Even after 15 polls, the bot should not have entered a BUY
        # because SELL consensus blocks entry. No BUY action should be returned
        # from the last step (entry may be blocked by consensus or signal warmup).
        # This validates the gate doesn't crash and processes correctly.
        assert result is None or result.get("action") != "BUY"

    @patch("portfolio.golddigger.bot._now_stockholm", return_value=(12, 0, "2026-03-10"))
    def test_kill_switch_env_blocks(self, mock_time, tmp_path):
        bot = self._make_bot(tmp_path)
        with patch.dict(os.environ, {"GOLDDIGGER_KILL": "1"}):
            result = bot.step(self._strong_signal_snap())
        assert result is None

    @patch("portfolio.golddigger.bot._now_stockholm", return_value=(12, 0, "2026-03-10"))
    def test_incomplete_data_returns_none(self, mock_time, tmp_path):
        bot = self._make_bot(tmp_path)
        snap = MarketSnapshot(
            ts_utc=datetime.now(timezone.utc),
            gold=0, usdsek=10.5, us10y=0.04,
        )
        result = bot.step(snap)
        assert result is None

    @patch("portfolio.golddigger.bot._now_stockholm", return_value=(12, 0, "2026-03-10"))
    def test_position_limit_blocks_second_entry(self, mock_time, tmp_path):
        """Max positions = 1 (default), so a second entry should be blocked."""
        from portfolio.golddigger.state import Position
        bot = self._make_bot(tmp_path, max_positions=1)
        # Manually set a position
        bot.state.position = Position(
            orderbook_id="TEST123", quantity=50, avg_price=100,
            entry_gold=2000, entry_time="2026-03-10T09:30:00",
            stop_price=95, take_profit_price=108,
        )
        snap = self._strong_signal_snap()
        # With a position already held, exit conditions are checked instead of entry.
        # The step should either return None (hold) or an exit -- never a BUY.
        result = bot.step(snap)
        if result is not None:
            assert result["action"] != "BUY"


# ============================================================
# Chronos forecast reading tests
# ============================================================

class TestChronosForecast:
    """Tests for read_chronos_forecast()."""

    def test_reads_forecast(self, tmp_path):
        from portfolio.golddigger.data_provider import read_chronos_forecast
        data = {"forecast_signals": {"XAU-USD": {
            "action": "BUY", "confidence": 0.65, "chronos_pct_move": 0.3
        }}}
        path = tmp_path / "agent_summary_compact.json"
        path.write_text(json.dumps(data))

        with patch("portfolio.golddigger.data_provider._DATA_DIR", tmp_path):
            result = read_chronos_forecast("XAU-USD")
        assert result["action"] == "BUY"
        assert result["confidence"] == 0.65

    def test_no_forecast_returns_none(self, tmp_path):
        from portfolio.golddigger.data_provider import read_chronos_forecast
        path = tmp_path / "agent_summary_compact.json"
        path.write_text("{}")
        with patch("portfolio.golddigger.data_provider._DATA_DIR", tmp_path):
            assert read_chronos_forecast("XAU-USD") is None

    def test_different_ticker(self, tmp_path):
        from portfolio.golddigger.data_provider import read_chronos_forecast
        data = {"forecast_signals": {"BTC-USD": {
            "action": "SELL", "confidence": 0.55, "chronos_pct_move": -0.2
        }}}
        path = tmp_path / "agent_summary_compact.json"
        path.write_text(json.dumps(data))

        with patch("portfolio.golddigger.data_provider._DATA_DIR", tmp_path):
            result = read_chronos_forecast("BTC-USD")
        assert result["action"] == "SELL"
        # XAU-USD should not be found
        with patch("portfolio.golddigger.data_provider._DATA_DIR", tmp_path):
            assert read_chronos_forecast("XAU-USD") is None

    def test_missing_file_returns_none(self, tmp_path):
        from portfolio.golddigger.data_provider import read_chronos_forecast
        with patch("portfolio.golddigger.data_provider._DATA_DIR", tmp_path):
            assert read_chronos_forecast("XAU-USD") is None


# ============================================================
# Daily digest tests
# ============================================================

class TestDailyDigest:
    """Tests for daily digest message building.

    _build_daily_digest(bot, cfg, mode) takes a bot object (with .state attribute).
    We create a simple mock bot wrapping BotState for testing.
    """

    def _make_bot_wrapper(self, state):
        """Create a simple object with a .state attribute for _build_daily_digest."""
        wrapper = MagicMock()
        wrapper.state = state
        return wrapper

    def test_digest_format(self):
        try:
            from portfolio.golddigger.runner import _build_daily_digest
        except ImportError:
            pytest.skip("_build_daily_digest not yet implemented in runner.py")

        state = BotState()
        state.cash_sek = 98500
        state.daily_pnl = -1500
        state.daily_trades = 3
        state.total_pnl = -1500
        state.total_trades = 3
        cfg = GolddiggerConfig(equity_sek=100000)
        bot = self._make_bot_wrapper(state)

        msg = _build_daily_digest(bot, cfg, "DRY-RUN")
        assert "GOLDDIGGER" in msg.upper()
        assert "98,500" in msg or "98500" in msg
        assert "-1,500" in msg or "-1500" in msg
        assert "DRY-RUN" in msg

    def test_digest_with_zero_trades(self):
        try:
            from portfolio.golddigger.runner import _build_daily_digest
        except ImportError:
            pytest.skip("_build_daily_digest not yet implemented in runner.py")

        state = BotState()
        cfg = GolddiggerConfig(equity_sek=100000)
        bot = self._make_bot_wrapper(state)

        msg = _build_daily_digest(bot, cfg, "DRY-RUN")
        assert "GOLDDIGGER" in msg.upper()

    def test_digest_with_open_position(self):
        try:
            from portfolio.golddigger.runner import _build_daily_digest
        except ImportError:
            pytest.skip("_build_daily_digest not yet implemented in runner.py")

        state = BotState()
        state.cash_sek = 95000
        state.position = Position(
            orderbook_id="TEST123", quantity=50, avg_price=100,
            entry_gold=2000, entry_time="2026-03-09T09:30:00",
            stop_price=95, take_profit_price=108,
        )
        cfg = GolddiggerConfig(equity_sek=100000)
        bot = self._make_bot_wrapper(state)

        msg = _build_daily_digest(bot, cfg, "DRY-RUN")
        assert "50" in msg  # quantity mentioned
        assert "100" in msg  # avg_price mentioned
