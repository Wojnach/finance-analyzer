"""Tests for the Elongir silver dip-trading bot."""

import json
import math
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from portfolio.elongir.config import ElongirConfig
from portfolio.elongir.state import (
    BotState,
    Position,
    warrant_price_sek,
    effective_leverage,
    buy_price,
    sell_price,
    log_trade,
    log_poll,
)
from portfolio.elongir.indicators import (
    compute_rsi,
    compute_macd,
    compute_bb,
    compute_ema,
    compute_volume_ratio,
    compute_atr,
    IndicatorSet,
    TimeframeIndicators,
    compute_all,
    _extract_ohlcv,
)
from portfolio.elongir.signal import DipDetector, ReversalDetector
from portfolio.elongir.risk import (
    compute_position_size,
    compute_stop,
    compute_tp,
    check_daily_limits,
)
from portfolio.elongir.bot import ElongirBot
from portfolio.elongir.data_provider import MarketSnapshot


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestElongirConfig:
    def test_default_values(self):
        cfg = ElongirConfig()
        assert cfg.poll_seconds == 30
        assert cfg.rsi_oversold == 30.0
        assert cfg.rsi_overbought == 70.0
        assert cfg.financing_level == 75.03
        assert cfg.spread_pct == 0.008
        assert cfg.commission_pct == 0.0025
        assert cfg.equity_sek == 100_000.0
        assert cfg.position_size_pct == 0.30
        assert cfg.max_positions == 1
        assert cfg.stop_loss_pct == 2.0
        assert cfg.take_profit_pct == 2.0
        assert cfg.trailing_start_pct == 1.5
        assert cfg.trailing_distance_pct == 0.7
        assert cfg.max_hold_hours == 5.0
        assert cfg.daily_loss_limit_pct == 3.0
        assert cfg.max_daily_trades == 6
        assert cfg.telegram_report_interval == 3600

    def test_from_config(self):
        raw = {
            "elongir": {
                "poll_seconds": 15,
                "financing_level": 80.0,
                "equity_sek": 200000,
                "stop_loss_pct": 3.0,
            }
        }
        cfg = ElongirConfig.from_config(raw)
        assert cfg.poll_seconds == 15
        assert cfg.financing_level == 80.0
        assert cfg.equity_sek == 200000
        assert cfg.stop_loss_pct == 3.0
        # Defaults for unspecified
        assert cfg.rsi_oversold == 30.0

    def test_from_config_empty(self):
        cfg = ElongirConfig.from_config({})
        assert cfg.poll_seconds == 30
        assert cfg.equity_sek == 100_000.0

    def test_frozen(self):
        cfg = ElongirConfig()
        with pytest.raises(AttributeError):
            cfg.poll_seconds = 99


# ---------------------------------------------------------------------------
# Warrant pricing tests
# ---------------------------------------------------------------------------

class TestWarrantPricing:
    def test_warrant_price_basic(self):
        # silver=90, financing=75.03, fx=10.5
        # warrant = (90 - 75.03) * 10.5 = 14.97 * 10.5 = 157.185
        price = warrant_price_sek(90.0, 10.5, 75.03)
        assert abs(price - 157.185) < 0.01

    def test_warrant_price_at_financing(self):
        price = warrant_price_sek(75.03, 10.5, 75.03)
        assert price == 0.0

    def test_warrant_price_below_financing(self):
        price = warrant_price_sek(70.0, 10.5, 75.03)
        assert price == 0.0

    def test_effective_leverage(self):
        # silver=90, financing=75.03
        # lev = 90 / (90 - 75.03) = 90 / 14.97 = 6.01
        lev = effective_leverage(90.0, 75.03)
        assert abs(lev - 6.01) < 0.1

    def test_effective_leverage_at_financing(self):
        lev = effective_leverage(75.03, 75.03)
        assert lev == float("inf")

    def test_buy_price(self):
        # mid=100, spread=0.8%
        # ask = 100 * (1 + 0.004) = 100.4
        ask = buy_price(100.0, 0.008)
        assert abs(ask - 100.4) < 0.01

    def test_sell_price(self):
        # mid=100, spread=0.8%
        # bid = 100 * (1 - 0.004) = 99.6
        bid = sell_price(100.0, 0.008)
        assert abs(bid - 99.6) < 0.01


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------

class TestBotState:
    def test_default_state(self):
        state = BotState()
        assert state.cash_sek == 100_000.0
        assert state.position is None
        assert state.daily_pnl == 0.0
        assert state.total_trades == 0
        assert state.wins == 0
        assert state.losses == 0
        assert state.signal_state == "SCANNING"
        assert not state.has_position()

    def test_has_position(self):
        state = BotState()
        assert not state.has_position()
        state.position = Position(
            entry_silver_usd=90.0,
            entry_warrant_sek=100.0,
            entry_time="2026-01-01T00:00:00Z",
            quantity=10,
            cost_sek=1000.0,
            stop_price_usd=88.0,
            trailing_peak_usd=90.0,
        )
        assert state.has_position()

    def test_equity_no_position(self):
        state = BotState(cash_sek=50000)
        assert state.equity() == 50000

    def test_equity_with_position(self):
        state = BotState(cash_sek=70000)
        state.position = Position(
            entry_silver_usd=90.0,
            entry_warrant_sek=100.0,
            entry_time="2026-01-01T00:00:00Z",
            quantity=100,
            cost_sek=10000.0,
            stop_price_usd=88.0,
            trailing_peak_usd=90.0,
        )
        # warrant_mid at silver=90, fx=10.5: (90-75.03)*10.5 = 157.185
        # bid = 157.185 * (1 - 0.004) = 156.556
        # equity = 70000 + 100 * 156.556 = 85655.6
        eq = state.equity(silver_usd=90.0, fx_rate=10.5)
        assert eq > 70000

    def test_save_load(self, tmp_path):
        path = str(tmp_path / "state.json")
        state = BotState(cash_sek=80000, total_trades=5, wins=3, losses=2)
        state.save(path)

        loaded = BotState.load(path)
        assert loaded.cash_sek == 80000
        assert loaded.total_trades == 5
        assert loaded.wins == 3

    def test_save_load_with_position(self, tmp_path):
        path = str(tmp_path / "state.json")
        state = BotState(cash_sek=70000)
        state.position = Position(
            entry_silver_usd=90.0,
            entry_warrant_sek=100.0,
            entry_time="2026-01-01T00:00:00Z",
            quantity=50,
            cost_sek=5000.0,
            stop_price_usd=88.0,
            trailing_peak_usd=91.0,
            trailing_active=True,
        )
        state.save(path)

        loaded = BotState.load(path)
        assert loaded.has_position()
        assert loaded.position.quantity == 50
        assert loaded.position.trailing_active is True

    def test_load_missing_file(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        state = BotState.load(path)
        assert state.cash_sek == 100_000.0

    def test_reset_daily(self):
        state = BotState(daily_pnl=-500, daily_trades=3, halted=True,
                         halted_reason="limit", last_trade_date="2026-01-01")
        state.reset_daily("2026-01-02")
        assert state.daily_pnl == 0.0
        assert state.daily_trades == 0
        assert state.halted is False

    def test_reset_daily_same_date(self):
        state = BotState(daily_pnl=-500, daily_trades=3, last_trade_date="2026-01-01")
        state.reset_daily("2026-01-01")
        # Same date -- no reset
        assert state.daily_pnl == -500

    def test_update_drawdown(self):
        state = BotState(equity_peak=100000, max_drawdown_pct=0.0)
        state.update_drawdown(95000)
        assert state.max_drawdown_pct == 5.0
        state.update_drawdown(97000)
        assert state.max_drawdown_pct == 5.0  # doesn't decrease
        state.update_drawdown(101000)
        assert state.equity_peak == 101000


# ---------------------------------------------------------------------------
# Indicator tests
# ---------------------------------------------------------------------------

class TestIndicators:
    def test_compute_rsi_oversold(self):
        # Build a declining series
        closes = [100 - i * 0.5 for i in range(20)]
        rsi = compute_rsi(closes)
        assert rsi is not None
        assert rsi < 30  # should be oversold

    def test_compute_rsi_overbought(self):
        # Build a rising series
        closes = [80 + i * 0.5 for i in range(20)]
        rsi = compute_rsi(closes)
        assert rsi is not None
        assert rsi > 70  # should be overbought

    def test_compute_rsi_insufficient_data(self):
        assert compute_rsi([1, 2, 3]) is None

    def test_compute_rsi_all_gains(self):
        closes = [float(i) for i in range(20)]
        rsi = compute_rsi(closes)
        assert rsi == 100.0

    def test_compute_macd(self):
        # Need enough data for slow=26 + signal=9
        closes = [90 + i * 0.1 for i in range(50)]
        ml, sl, hist = compute_macd(closes)
        assert ml is not None
        assert sl is not None
        assert hist is not None

    def test_compute_macd_insufficient(self):
        ml, sl, hist = compute_macd([1, 2, 3])
        assert ml is None

    def test_compute_bb(self):
        closes = [90.0 + (i % 5) * 0.1 for i in range(25)]
        lower, mid, upper = compute_bb(closes)
        assert lower is not None
        assert mid is not None
        assert upper is not None
        assert lower < mid < upper

    def test_compute_bb_insufficient(self):
        lower, mid, upper = compute_bb([1, 2])
        assert lower is None

    def test_compute_ema(self):
        values = [float(i) for i in range(20)]
        ema = compute_ema(values, 10)
        assert ema is not None
        assert 10 < ema < 20  # weighted toward recent

    def test_compute_ema_insufficient(self):
        assert compute_ema([1, 2], 10) is None

    def test_compute_volume_ratio(self):
        # 25 volumes: first 25 are 100, then override last 5 to 200
        volumes = [100.0] * 25
        volumes[-5:] = [200.0] * 5
        ratio = compute_volume_ratio(volumes, recent=5, avg_period=20)
        assert ratio is not None
        # avg of last 20 = (15*100 + 5*200)/20 = 125, recent 5 avg = 200
        # ratio = 200 / 125 = 1.6
        assert abs(ratio - 1.6) < 0.1

    def test_compute_volume_ratio_insufficient(self):
        assert compute_volume_ratio([100], recent=5, avg_period=20) is None

    def test_compute_atr(self):
        highs = [91.0 + i * 0.1 for i in range(20)]
        lows = [89.0 + i * 0.1 for i in range(20)]
        closes = [90.0 + i * 0.1 for i in range(20)]
        atr = compute_atr(highs, lows, closes, period=14)
        assert atr is not None
        assert atr > 0

    def test_compute_atr_insufficient(self):
        assert compute_atr([1], [0], [0.5], period=14) is None

    def test_extract_ohlcv(self):
        klines = [
            [0, "90.0", "91.0", "89.0", "90.5", "1000", 0, "0", 0, "0", "0", "0"],
            [0, "90.5", "92.0", "90.0", "91.0", "1200", 0, "0", 0, "0", "0", "0"],
        ]
        opens, highs, lows, closes, volumes = _extract_ohlcv(klines)
        assert closes == [90.5, 91.0]
        assert highs == [91.0, 92.0]
        assert volumes == [1000.0, 1200.0]


class TestComputeAll:
    def _make_klines(self, n, base_price=90.0):
        """Generate fake klines."""
        klines = []
        for i in range(n):
            o = base_price + i * 0.01
            h = o + 0.5
            l = o - 0.3
            c = o + 0.1
            v = 1000.0
            klines.append([0, str(o), str(h), str(l), str(c), str(v),
                           0, "0", 0, "0", "0", "0"])
        return klines

    def test_compute_all_complete(self):
        snap = MarketSnapshot(
            silver_usd=90.0,
            fx_rate=10.5,
            klines_1m=self._make_klines(100),
            klines_5m=self._make_klines(60),
            klines_15m=self._make_klines(40),
        )
        iset = compute_all(snap)
        assert iset.silver_usd == 90.0
        assert iset.tf_1m.rsi is not None
        assert iset.tf_5m.rsi is not None
        assert iset.tf_15m.rsi is not None

    def test_compute_all_no_klines(self):
        snap = MarketSnapshot(silver_usd=90.0, fx_rate=10.5)
        iset = compute_all(snap)
        assert iset.tf_1m.rsi is None
        assert iset.tf_5m.rsi is None


# ---------------------------------------------------------------------------
# Signal tests
# ---------------------------------------------------------------------------

class TestDipDetector:
    def _make_config(self, **kwargs):
        return ElongirConfig(**kwargs)

    def test_initial_state(self):
        dd = DipDetector(self._make_config())
        assert dd.state == "SCANNING"

    def test_no_dip_when_rsi_normal(self):
        dd = DipDetector(self._make_config())
        indicators = IndicatorSet()
        indicators.tf_5m = TimeframeIndicators(
            rsi=50.0, bb_lower=88.0, high_1h=91.0
        )
        result = dd.update(indicators, 90.0)
        assert result is None
        assert dd.state == "SCANNING"

    def test_dip_detected(self):
        dd = DipDetector(self._make_config(min_dip_pct=1.0))
        indicators = IndicatorSet()
        indicators.tf_5m = TimeframeIndicators(
            rsi=25.0, bb_lower=89.5, high_1h=92.0,
            macd_histogram=-0.1,
        )
        indicators.tf_15m = TimeframeIndicators(bb_lower=89.0)
        # Price 89.0, below BB lower 89.5, RSI=25 < 30, drop from 92 = 3.3%
        result = dd.update(indicators, 89.0)
        assert result is None
        assert dd.state == "DIP_DETECTED"

    def test_full_buy_cycle(self):
        cfg = self._make_config(
            min_dip_pct=1.0,
            rsi_oversold=30.0,
            rsi_recovery=35.0,
            macd_improving_checks=2,
        )
        dd = DipDetector(cfg)

        # Poll 1: dip detected
        ind = IndicatorSet()
        ind.tf_5m = TimeframeIndicators(
            rsi=25.0, bb_lower=89.5, high_1h=92.0, macd_histogram=-0.5
        )
        ind.tf_15m = TimeframeIndicators(bb_lower=89.0)
        dd.update(ind, 89.0)
        assert dd.state == "DIP_DETECTED"

        # Poll 2: MACD improving (1st)
        ind.tf_5m = TimeframeIndicators(
            rsi=26.0, bb_lower=89.5, high_1h=92.0, macd_histogram=-0.3
        )
        dd.update(ind, 89.2)
        assert dd.state == "DIP_DETECTED"  # need 2 consecutive improvements

        # Poll 3: MACD improving (2nd) + RSI turning up
        ind.tf_5m = TimeframeIndicators(
            rsi=28.0, bb_lower=89.5, high_1h=92.0, macd_histogram=-0.1
        )
        dd.update(ind, 89.4)
        assert dd.state == "CONFIRMING_BUY"

        # Poll 4: RSI crosses above recovery threshold
        ind.tf_5m = TimeframeIndicators(
            rsi=36.0, bb_lower=89.5, high_1h=92.0, macd_histogram=0.05
        )
        result = dd.update(ind, 89.8)
        assert result == "BUY"
        assert dd.state == "SCANNING"  # reset after trigger


class TestReversalDetector:
    def _make_config(self, **kwargs):
        return ElongirConfig(**kwargs)

    def test_hard_stop(self):
        rd = ReversalDetector(self._make_config(stop_loss_pct=2.0))
        ind = IndicatorSet()
        ind.tf_5m = TimeframeIndicators(rsi=40.0, macd_histogram=-0.1)
        result = rd.update(ind, 88.0, 90.0, "2026-01-01T00:00:00Z", 90.0, False)
        # Drop = (88-90)/90 = -2.22%, exceeds 2% stop
        assert result == "STOP"

    def test_take_profit(self):
        rd = ReversalDetector(self._make_config(take_profit_pct=2.0))
        ind = IndicatorSet()
        ind.tf_5m = TimeframeIndicators(rsi=60.0, macd_histogram=0.1)
        result = rd.update(ind, 91.9, 90.0, "2026-01-01T00:00:00Z", 91.9, False)
        # Gain = (91.9-90)/90 = 2.11%, exceeds 2% TP
        assert result == "TAKE_PROFIT"

    def test_time_stop(self):
        rd = ReversalDetector(self._make_config(max_hold_hours=5.0))
        ind = IndicatorSet()
        ind.tf_5m = TimeframeIndicators(rsi=50.0, macd_histogram=0.0)
        entry_time = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
        result = rd.update(ind, 90.0, 90.0, entry_time, 90.0, False)
        assert result == "TIME_STOP"

    def test_trailing_stop(self):
        rd = ReversalDetector(self._make_config(trailing_distance_pct=0.7))
        ind = IndicatorSet()
        ind.tf_5m = TimeframeIndicators(rsi=55.0, macd_histogram=0.1)
        # Use recent entry time so TIME_STOP doesn't fire first
        recent_time = datetime.now(timezone.utc).isoformat()
        # Peak=92.0, current=91.3 -> drop = (92-91.3)/92 = 0.76% > 0.7%
        result = rd.update(ind, 91.3, 90.0, recent_time, 92.0, True)
        assert result == "TRAILING_STOP"

    def test_sell_signal(self):
        cfg = self._make_config(rsi_overbought=70.0)
        rd = ReversalDetector(cfg)
        ind = IndicatorSet()
        recent_time = datetime.now(timezone.utc).isoformat()
        # First poll to set prev_macd_hist
        ind.tf_5m = TimeframeIndicators(rsi=65.0, macd_histogram=0.5)
        rd.update(ind, 91.0, 90.0, recent_time, 91.0, False)

        # Second poll: RSI overbought + MACD declining
        ind.tf_5m = TimeframeIndicators(rsi=72.0, macd_histogram=0.3)
        result = rd.update(ind, 91.5, 90.0, recent_time, 91.5, False)
        assert result == "SELL_SIGNAL"

    def test_no_exit(self):
        rd = ReversalDetector(self._make_config())
        ind = IndicatorSet()
        ind.tf_5m = TimeframeIndicators(rsi=50.0, macd_histogram=0.1)
        result = rd.update(ind, 90.5, 90.0, datetime.now(timezone.utc).isoformat(), 90.5, False)
        assert result is None

    def test_should_activate_trailing(self):
        rd = ReversalDetector(self._make_config(trailing_start_pct=1.5))
        # 1.5% gain above entry of 90.0 = 91.35
        assert rd.should_activate_trailing(91.4, 90.0)
        assert not rd.should_activate_trailing(91.0, 90.0)


# ---------------------------------------------------------------------------
# Risk tests
# ---------------------------------------------------------------------------

class TestRisk:
    def _cfg(self, **kw):
        return ElongirConfig(**kw)

    def test_compute_position_size(self):
        cfg = self._cfg(position_size_pct=0.30, commission_pct=0.0025)
        result = compute_position_size(100000, 100.0, cfg)
        # allocation = 100000 * 0.30 = 30000
        # quantity = floor(30000 / 100) = 300
        # cost = 300 * 100 = 30000
        # fee = 30000 * 0.0025 = 75
        assert result.quantity == 300
        assert result.cost_sek == 30000.0
        assert result.fee_sek == 75.0
        assert result.total_cost_sek == 30075.0

    def test_position_size_zero_ask(self):
        cfg = self._cfg()
        result = compute_position_size(100000, 0.0, cfg)
        assert result.quantity == 0

    def test_position_size_insufficient_cash(self):
        cfg = self._cfg(position_size_pct=0.30)
        result = compute_position_size(10, 100.0, cfg)
        assert result.quantity == 0

    def test_compute_stop(self):
        cfg = self._cfg(stop_loss_pct=2.0)
        stop = compute_stop(90.0, cfg)
        assert abs(stop - 88.2) < 0.01

    def test_compute_tp(self):
        cfg = self._cfg(take_profit_pct=2.0)
        tp = compute_tp(90.0, cfg)
        assert abs(tp - 91.8) < 0.01

    def test_check_daily_limits_ok(self):
        cfg = self._cfg(max_daily_trades=6, daily_loss_limit_pct=3.0)
        ok, reason = check_daily_limits(2, -100, 100000, cfg)
        assert ok

    def test_check_daily_limits_max_trades(self):
        cfg = self._cfg(max_daily_trades=6)
        ok, reason = check_daily_limits(6, 0, 100000, cfg)
        assert not ok
        assert "Max daily trades" in reason

    def test_check_daily_limits_loss(self):
        cfg = self._cfg(daily_loss_limit_pct=3.0)
        ok, reason = check_daily_limits(0, -3000, 100000, cfg)
        assert not ok
        assert "Daily loss limit" in reason


# ---------------------------------------------------------------------------
# Bot tests
# ---------------------------------------------------------------------------

class TestElongirBot:
    def _make_bot(self, tmp_path, **cfg_kw):
        cfg = ElongirConfig(
            state_file=str(tmp_path / "state.json"),
            log_file=str(tmp_path / "log.jsonl"),
            trades_file=str(tmp_path / "trades.jsonl"),
            **cfg_kw,
        )
        state = BotState()
        return ElongirBot(cfg, state)

    def _make_snapshot(self, silver=90.0, fx=10.5, klines=True):
        snap = MarketSnapshot(
            silver_usd=silver,
            fx_rate=fx,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        if klines:
            snap.klines_1m = self._make_klines(100, silver)
            snap.klines_5m = self._make_klines(60, silver)
            snap.klines_15m = self._make_klines(40, silver)
        return snap

    def _make_klines(self, n, base=90.0):
        klines = []
        for i in range(n):
            o = base + i * 0.01
            h = o + 0.3
            l = o - 0.2
            c = o + 0.05
            v = 500.0
            klines.append([0, str(o), str(h), str(l), str(c), str(v),
                           0, "0", 0, "0", "0", "0"])
        return klines

    @patch("portfolio.elongir.risk.check_session", return_value=True)
    @patch("portfolio.elongir.risk.get_stockholm_time", return_value=(10, 30, "2026-03-12"))
    def test_step_no_action(self, mock_time, mock_session, tmp_path):
        bot = self._make_bot(tmp_path)
        snap = self._make_snapshot()
        action = bot.step(snap)
        # Normal market conditions, no dip -- should return None
        assert action is None

    def test_step_incomplete_snapshot(self, tmp_path):
        bot = self._make_bot(tmp_path)
        snap = MarketSnapshot(silver_usd=0, fx_rate=0)
        action = bot.step(snap)
        assert action is None

    @patch("portfolio.elongir.risk.check_session", return_value=False)
    @patch("portfolio.elongir.risk.get_stockholm_time", return_value=(3, 0, "2026-03-12"))
    def test_step_outside_session(self, mock_time, mock_session, tmp_path):
        bot = self._make_bot(tmp_path)
        snap = self._make_snapshot()
        action = bot.step(snap)
        assert action is None

    @patch("portfolio.elongir.risk.check_session", return_value=True)
    @patch("portfolio.elongir.risk.get_stockholm_time", return_value=(10, 30, "2026-03-12"))
    def test_step_halted(self, mock_time, mock_session, tmp_path):
        bot = self._make_bot(tmp_path)
        bot.state.halted = True
        bot.state.halted_reason = "test"
        snap = self._make_snapshot()
        action = bot.step(snap)
        assert action is None

    @patch("portfolio.elongir.risk.check_session", return_value=True)
    @patch("portfolio.elongir.risk.get_stockholm_time", return_value=(10, 30, "2026-03-12"))
    def test_execute_buy(self, mock_time, mock_session, tmp_path):
        bot = self._make_bot(tmp_path)
        snap = self._make_snapshot(silver=90.0, fx=10.5)

        # Manually trigger a buy through the bot's internal method
        from portfolio.elongir.indicators import compute_all
        indicators = compute_all(snap)
        w_mid = warrant_price_sek(90.0, 10.5, 75.03)
        action = bot._execute_buy(snap, indicators, w_mid)

        assert action is not None
        assert action["type"] == "BUY"
        assert action["quantity"] > 0
        assert bot.state.has_position()
        assert bot.state.total_trades == 1

    @patch("portfolio.elongir.risk.check_session", return_value=True)
    @patch("portfolio.elongir.risk.get_stockholm_time", return_value=(10, 30, "2026-03-12"))
    def test_execute_sell(self, mock_time, mock_session, tmp_path):
        bot = self._make_bot(tmp_path)
        # Set up a position first
        bot.state.position = Position(
            entry_silver_usd=90.0,
            entry_warrant_sek=100.0,
            entry_time=datetime.now(timezone.utc).isoformat(),
            quantity=100,
            cost_sek=10000.0,
            stop_price_usd=88.0,
            trailing_peak_usd=91.0,
        )
        bot.state.cash_sek = 90000

        snap = self._make_snapshot(silver=91.5, fx=10.5)
        w_mid = warrant_price_sek(91.5, 10.5, 75.03)
        action = bot._execute_sell(snap, "TAKE_PROFIT", w_mid)

        assert action is not None
        assert action["type"] == "SELL"
        assert not bot.state.has_position()
        assert bot.state.cash_sek > 90000  # got proceeds back

    @patch("portfolio.elongir.risk.check_session", return_value=True)
    @patch("portfolio.elongir.risk.get_stockholm_time", return_value=(10, 30, "2026-03-12"))
    def test_daily_reset(self, mock_time, mock_session, tmp_path):
        bot = self._make_bot(tmp_path)
        bot.state.daily_pnl = -500
        bot.state.daily_trades = 3
        bot.state.last_trade_date = "2026-03-11"  # yesterday

        snap = self._make_snapshot()
        bot.step(snap)

        assert bot.state.daily_pnl == 0.0
        assert bot.state.daily_trades == 0


# ---------------------------------------------------------------------------
# Log tests
# ---------------------------------------------------------------------------

class TestLogging:
    def test_log_trade(self, tmp_path):
        trades_file = str(tmp_path / "trades.jsonl")
        log_trade(
            trades_file,
            action="BUY",
            quantity=100,
            warrant_price_sek_val=150.0,
            silver_usd=90.0,
            fx_rate=10.5,
            fee_sek=37.5,
            reason="test buy",
        )
        with open(trades_file) as f:
            line = json.loads(f.readline())
        assert line["action"] == "BUY"
        assert line["quantity"] == 100
        assert line["silver_usd"] == 90.0

    def test_log_poll(self, tmp_path):
        log_file = str(tmp_path / "log.jsonl")
        log_poll(
            log_file,
            silver_usd=90.0,
            fx_rate=10.5,
            warrant_mid=157.0,
            signal_state="SCANNING",
            rsi_5m=45.0,
        )
        with open(log_file) as f:
            line = json.loads(f.readline())
        assert line["silver_usd"] == 90.0
        assert line["state"] == "SCANNING"


# ---------------------------------------------------------------------------
# Data provider tests
# ---------------------------------------------------------------------------

class TestMarketSnapshot:
    def test_is_complete(self):
        snap = MarketSnapshot(silver_usd=90.0, fx_rate=10.5)
        assert snap.is_complete()

    def test_is_not_complete(self):
        snap = MarketSnapshot(silver_usd=0, fx_rate=10.5)
        assert not snap.is_complete()

    def test_is_not_complete_no_fx(self):
        snap = MarketSnapshot(silver_usd=90.0, fx_rate=0)
        assert not snap.is_complete()
