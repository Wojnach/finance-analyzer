"""Tests for ISKBETS — intraday quick-gamble mode."""

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root on path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio.iskbets import (
    _evaluate_entry,
    _load_config,
    _load_state,
    _parse_gate_response,
    _save_config,
    _save_state,
    check_exits,
    compute_atr_15m,
    format_entry_alert,
    format_exit_alert,
    format_position_status,
    handle_command,
    invoke_layer2_gate,
)
from portfolio.telegram_poller import TelegramPoller


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """Redirect ISKBETS data files to a temp dir."""
    import portfolio.iskbets as mod

    monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mod, "CONFIG_FILE", tmp_path / "iskbets_config.json")
    monkeypatch.setattr(mod, "STATE_FILE", tmp_path / "iskbets_state.json")
    return tmp_path


@pytest.fixture
def iskbets_cfg():
    """Standard ISKBETS config."""
    return {
        "enabled": True,
        "min_bigbet_conditions": 2,
        "min_buy_votes": 3,
        "entry_cutoff_et": "14:30",
        "hard_stop_atr_mult": 2.0,
        "stage1_atr_mult": 1.5,
        "trailing_atr_mult": 1.0,
        "time_exit_et": "15:50",
    }


@pytest.fixture
def sample_signals():
    """Sample signals dict with buy votes."""
    return {
        "BTC-USD": {
            "action": "BUY",
            "confidence": 0.8,
            "indicators": {
                "rsi": 25,
                "macd_hist": 0.5,
                "macd_hist_prev": -0.3,
                "ema9": 67000,
                "ema21": 66500,
                "bb_upper": 69000,
                "bb_lower": 65000,
                "bb_mid": 67000,
                "price_vs_bb": "below_lower",
                "atr": 1500,
                "atr_pct": 2.2,
            },
            "extra": {
                "_buy_count": 4,
                "_sell_count": 1,
                "_total_applicable": 11,
                "fear_greed": 8,
                "fear_greed_class": "Extreme Fear",
                "volume_ratio": 2.5,
                "volume_action": "SELL",
            },
        },
        "MSTR": {
            "action": "BUY",
            "confidence": 0.7,
            "indicators": {
                "rsi": 28,
                "macd_hist": 0.1,
                "macd_hist_prev": -0.2,
                "ema9": 130,
                "ema21": 128,
                "bb_upper": 140,
                "bb_lower": 120,
                "bb_mid": 130,
                "price_vs_bb": "inside",
                "atr": 3.5,
                "atr_pct": 2.7,
            },
            "extra": {
                "_buy_count": 2,
                "_sell_count": 0,
                "_total_applicable": 7,
                "fear_greed": 8,
                "fear_greed_class": "Extreme Fear",
                "volume_ratio": 1.0,
                "volume_action": "HOLD",
            },
        },
    }


@pytest.fixture
def sample_tf_data():
    """Sample timeframe data."""
    return {
        "BTC-USD": [
            ("Now", {"indicators": {"rsi": 25, "price_vs_bb": "below_lower"}}),
            ("12h", {"indicators": {"rsi": 30, "price_vs_bb": "below_lower"}}),
            ("2d", {"indicators": {"rsi": 35, "price_vs_bb": "inside"}}),
        ],
        "MSTR": [
            ("Now", {"indicators": {"rsi": 28, "price_vs_bb": "inside"}}),
            ("12h", {"indicators": {"rsi": 35, "price_vs_bb": "inside"}}),
        ],
    }


@pytest.fixture
def sample_prices():
    return {"BTC-USD": 66000, "ETH-USD": 1950, "MSTR": 129.50, "PLTR": 132, "NVDA": 185}


@pytest.fixture
def app_config():
    return {
        "telegram": {"token": "fake", "chat_id": "12345"},
        "iskbets": {
            "enabled": True,
            "min_bigbet_conditions": 2,
            "min_buy_votes": 3,
            "entry_cutoff_et": "14:30",
            "hard_stop_atr_mult": 2.0,
            "stage1_atr_mult": 1.5,
            "trailing_atr_mult": 1.0,
            "time_exit_et": "15:50",
        },
        "alpaca": {"key": "fake", "secret": "fake"},
    }


def _make_position(
    ticker="BTC-USD",
    entry_price=66000,
    amount_sek=100000,
    atr=1500,
    stop_at_breakeven=False,
    highest_price=None,
    sell_signal_streak=0,
):
    """Helper to create a test position dict."""
    return {
        "ticker": ticker,
        "entry_price_usd": entry_price,
        "amount_sek": amount_sek,
        "shares": amount_sek / (entry_price * 10.5),
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "atr_15m": atr,
        "stop_loss": entry_price - 2 * atr,
        "stage1_target": entry_price + 1.5 * atr,
        "stop_at_breakeven": stop_at_breakeven,
        "highest_price": highest_price or entry_price,
        "sell_signal_streak": sell_signal_streak,
        "fx_rate": 10.5,
    }


# ── Tests: Entry Evaluation ──────────────────────────────────────────────


class TestEntryEvaluation:
    @patch("portfolio.iskbets._before_cutoff", return_value=True)
    def test_entry_passes_both_gates(self, mock_cutoff, sample_signals, sample_prices, sample_tf_data, iskbets_cfg, app_config):
        """BTC-USD has 4 buy votes and ≥2 bigbet conditions → should enter."""
        should_enter, conditions = _evaluate_entry(
            "BTC-USD", sample_signals, sample_prices, sample_tf_data, iskbets_cfg, app_config
        )
        assert should_enter
        assert len(conditions) >= 2
        assert any("BUY votes" in c for c in conditions)

    @patch("portfolio.iskbets._before_cutoff", return_value=True)
    def test_entry_fails_insufficient_buy_votes(self, mock_cutoff, sample_signals, sample_prices, sample_tf_data, iskbets_cfg, app_config):
        """MSTR has only 2 buy votes (< min_buy_votes=3) → should not enter."""
        should_enter, conditions = _evaluate_entry(
            "MSTR", sample_signals, sample_prices, sample_tf_data, iskbets_cfg, app_config
        )
        assert not should_enter

    @patch("portfolio.iskbets._before_cutoff", return_value=False)
    def test_entry_fails_past_cutoff(self, mock_cutoff, sample_signals, sample_prices, sample_tf_data, iskbets_cfg, app_config):
        """Past 14:30 ET → no entry regardless of signals."""
        should_enter, conditions = _evaluate_entry(
            "BTC-USD", sample_signals, sample_prices, sample_tf_data, iskbets_cfg, app_config
        )
        assert not should_enter

    @patch("portfolio.iskbets._before_cutoff", return_value=True)
    def test_entry_fails_missing_ticker(self, mock_cutoff, sample_signals, sample_prices, sample_tf_data, iskbets_cfg, app_config):
        """Unknown ticker → no entry."""
        should_enter, conditions = _evaluate_entry(
            "FAKE-TICKER", sample_signals, sample_prices, sample_tf_data, iskbets_cfg, app_config
        )
        assert not should_enter


# ── Tests: Exit Priority ─────────────────────────────────────────────────


class TestExitChecks:
    def test_hard_stop(self, iskbets_cfg, sample_signals):
        """Price below hard stop → exit."""
        pos = _make_position(entry_price=66000, atr=1500)
        state = {"active_position": pos}
        # Hard stop = 66000 - 2*1500 = 63000
        prices = {"BTC-USD": 62500}

        result = check_exits(state, prices, sample_signals, {}, iskbets_cfg)
        assert result is not None
        assert result[0] == "hard_stop"

    def test_hard_stop_not_hit(self, iskbets_cfg, sample_signals):
        """Price above hard stop → no exit."""
        pos = _make_position(entry_price=66000, atr=1500)
        state = {"active_position": pos}
        prices = {"BTC-USD": 65000}

        result = check_exits(state, prices, sample_signals, {}, iskbets_cfg)
        assert result is None

    @patch("portfolio.iskbets._past_time_exit", return_value=True)
    def test_time_exit(self, mock_time, iskbets_cfg, sample_signals):
        """Past 15:50 ET → time exit."""
        pos = _make_position(entry_price=66000, atr=1500)
        state = {"active_position": pos}
        prices = {"BTC-USD": 67000}

        result = check_exits(state, prices, sample_signals, {}, iskbets_cfg)
        assert result is not None
        assert result[0] == "time_exit"

    @patch("portfolio.iskbets._past_time_exit", return_value=False)
    def test_stage1_hit(self, mock_time, iskbets_cfg, sample_signals):
        """Price above stage 1 target → move stop to breakeven."""
        pos = _make_position(entry_price=66000, atr=1500)
        state = {"active_position": pos}
        # Stage 1 = 66000 + 1.5*1500 = 68250
        prices = {"BTC-USD": 68500}

        result = check_exits(state, prices, sample_signals, {}, iskbets_cfg)
        assert result is not None
        assert result[0] == "stage1_hit"
        assert state["active_position"]["stop_at_breakeven"] is True
        assert state["active_position"]["stop_loss"] == 66000  # Breakeven

    @patch("portfolio.iskbets._past_time_exit", return_value=False)
    def test_trailing_stop(self, mock_time, iskbets_cfg, sample_signals):
        """After stage1, trailing stop triggers when price drops."""
        pos = _make_position(
            entry_price=66000, atr=1500,
            stop_at_breakeven=True, highest_price=69000,
        )
        state = {"active_position": pos}
        # Trailing stop = max(66000, 69000 - 1.0*1500) = 67500
        prices = {"BTC-USD": 67400}

        result = check_exits(state, prices, sample_signals, {}, iskbets_cfg)
        assert result is not None
        assert result[0] == "trailing_stop"

    @patch("portfolio.iskbets._past_time_exit", return_value=False)
    def test_trailing_stop_not_hit(self, mock_time, iskbets_cfg, sample_signals):
        """After stage1, price still above trailing stop → no exit."""
        pos = _make_position(
            entry_price=66000, atr=1500,
            stop_at_breakeven=True, highest_price=69000,
        )
        state = {"active_position": pos}
        # Trailing stop = max(66000, 69000 - 1500) = 67500
        prices = {"BTC-USD": 68000}

        result = check_exits(state, prices, sample_signals, {}, iskbets_cfg)
        assert result is None

    @patch("portfolio.iskbets._past_time_exit", return_value=False)
    def test_signal_reversal_needs_two_cycles(self, mock_time, iskbets_cfg):
        """Signal reversal requires 2 consecutive cycles of ≥3 sell votes."""
        pos = _make_position(entry_price=66000, atr=1500, sell_signal_streak=0)
        state = {"active_position": pos}
        prices = {"BTC-USD": 65500}
        signals = {
            "BTC-USD": {
                "extra": {"_sell_count": 4, "_buy_count": 0, "_total_applicable": 11},
                "indicators": {},
            }
        }

        # First cycle: streak goes to 1, no exit yet
        result = check_exits(state, prices, signals, {}, iskbets_cfg)
        assert result is None
        assert state["active_position"]["sell_signal_streak"] == 1

        # Second cycle: streak goes to 2, exit fires
        result = check_exits(state, prices, signals, {}, iskbets_cfg)
        assert result is not None
        assert result[0] == "signal_reversal"

    @patch("portfolio.iskbets._past_time_exit", return_value=False)
    def test_signal_reversal_resets_on_no_sells(self, mock_time, iskbets_cfg):
        """Sell streak resets when sell votes drop below 3."""
        pos = _make_position(entry_price=66000, atr=1500, sell_signal_streak=1)
        state = {"active_position": pos}
        prices = {"BTC-USD": 65500}
        signals = {
            "BTC-USD": {
                "extra": {"_sell_count": 1, "_buy_count": 2, "_total_applicable": 11},
                "indicators": {},
            }
        }

        result = check_exits(state, prices, signals, {}, iskbets_cfg)
        assert result is None
        assert state["active_position"]["sell_signal_streak"] == 0

    def test_hard_stop_beats_time_exit(self, iskbets_cfg, sample_signals):
        """Hard stop takes priority over time exit."""
        pos = _make_position(entry_price=66000, atr=1500)
        state = {"active_position": pos}
        prices = {"BTC-USD": 62000}  # Below hard stop

        with patch("portfolio.iskbets._past_time_exit", return_value=True):
            result = check_exits(state, prices, sample_signals, {}, iskbets_cfg)
            assert result[0] == "hard_stop"

    def test_no_position_returns_none(self, iskbets_cfg, sample_signals):
        """No active position → None."""
        state = {"active_position": None}
        result = check_exits(state, {}, sample_signals, {}, iskbets_cfg)
        assert result is None


# ── Tests: ATR Computation ───────────────────────────────────────────────


class TestATR:
    @patch("portfolio.iskbets.requests.get")
    def test_compute_atr_binance(self, mock_get, app_config):
        """ATR from Binance 15m candles."""
        # Simulate 20 candles with known values
        candles = []
        for i in range(20):
            base = 66000 + i * 100
            candles.append([
                1000000 + i, str(base), str(base + 500), str(base - 300),
                str(base + 200), "1000", 1000000 + i + 1, "0", "0", "0", "0", "0",
            ])

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = candles
        mock_get.return_value = mock_resp

        atr = compute_atr_15m("BTC-USD", app_config)
        assert atr > 0
        assert isinstance(atr, float)

    def test_unknown_ticker_raises(self, app_config):
        """Unknown ticker raises ValueError."""
        with pytest.raises(ValueError, match="Unknown ticker"):
            compute_atr_15m("FAKE-COIN", app_config)


# ── Tests: Stage 1 → Breakeven ───────────────────────────────────────────


class TestStage1Breakeven:
    @patch("portfolio.iskbets._past_time_exit", return_value=False)
    def test_breakeven_stop_after_stage1(self, mock_time, iskbets_cfg, sample_signals):
        """After stage 1 hit, stop is at entry price (breakeven)."""
        pos = _make_position(entry_price=66000, atr=1500)
        state = {"active_position": pos}
        prices = {"BTC-USD": 68300}  # Above stage1 = 68250

        result = check_exits(state, prices, sample_signals, {}, iskbets_cfg)
        assert result[0] == "stage1_hit"
        assert state["active_position"]["stop_loss"] == 66000

    @patch("portfolio.iskbets._past_time_exit", return_value=False)
    def test_trailing_uses_breakeven_floor(self, mock_time, iskbets_cfg, sample_signals):
        """Trailing stop never goes below breakeven (entry price)."""
        pos = _make_position(
            entry_price=66000, atr=1500,
            stop_at_breakeven=True, highest_price=67000,
        )
        state = {"active_position": pos}
        # trailing = max(66000, 67000 - 1500) = max(66000, 65500) = 66000
        prices = {"BTC-USD": 66500}

        result = check_exits(state, prices, sample_signals, {}, iskbets_cfg)
        # Price 66500 > trailing 66000 → no exit
        assert result is None


# ── Tests: Trailing Stop Tracking ─────────────────────────────────────────


class TestTrailingStop:
    @patch("portfolio.iskbets._past_time_exit", return_value=False)
    def test_highest_price_updated(self, mock_time, iskbets_cfg, sample_signals):
        """highest_price is updated as price increases."""
        pos = _make_position(
            entry_price=66000, atr=1500,
            stop_at_breakeven=True, highest_price=68000,
        )
        state = {"active_position": pos}
        prices = {"BTC-USD": 69500}

        check_exits(state, prices, sample_signals, {}, iskbets_cfg)
        assert state["active_position"]["highest_price"] == 69500

    @patch("portfolio.iskbets._past_time_exit", return_value=False)
    def test_highest_price_not_lowered(self, mock_time, iskbets_cfg, sample_signals):
        """highest_price is never lowered."""
        pos = _make_position(
            entry_price=66000, atr=1500,
            stop_at_breakeven=True, highest_price=70000,
        )
        state = {"active_position": pos}
        prices = {"BTC-USD": 69000}

        check_exits(state, prices, sample_signals, {}, iskbets_cfg)
        assert state["active_position"]["highest_price"] == 70000


# ── Tests: Command Parsing ────────────────────────────────────────────────


class TestCommandParsing:
    @patch("portfolio.iskbets.compute_atr_15m", return_value=3.5)
    @patch("portfolio.main.fetch_usd_sek", return_value=10.5)
    def test_bought_command(self, mock_fx, mock_atr, tmp_data_dir, app_config):
        """'bought MSTR 129.50 100000' sets active position."""
        result = handle_command("bought", "MSTR 129.50 100000", app_config)
        assert "\u2705" in result
        assert "Position tracked" in result

        state = _load_state()
        pos = state["active_position"]
        assert pos["ticker"] == "MSTR"
        assert pos["entry_price_usd"] == 129.50
        assert pos["amount_sek"] == 100000

    def test_sold_command(self, tmp_data_dir, app_config):
        """'sold' closes position and logs P&L."""
        state = {
            "active_position": _make_position(
                ticker="MSTR", entry_price=129.50,
                amount_sek=100000, atr=3.5,
            ),
            "trade_history": [],
        }
        _save_state(state)

        result = handle_command("sold", "", app_config)
        assert "ISKBETS closed" in result
        assert "MSTR" in result

        state = _load_state()
        assert state["active_position"] is None
        assert len(state["trade_history"]) == 1

    def test_cancel_command(self, tmp_data_dir, app_config):
        """'cancel' disables ISKBETS."""
        _save_config({"enabled": True, "tickers": ["MSTR"]})
        result = handle_command("cancel", "", app_config)
        assert "disabled" in result

    def test_status_no_position(self, tmp_data_dir, app_config):
        """'status' with no position shows scanning state."""
        _save_config({"enabled": True, "tickers": ["MSTR", "PLTR"]})
        _save_state({"active_position": None, "trade_history": []})
        result = handle_command("status", "", app_config)
        assert "scanning" in result.lower() or "not active" in result.lower()

    def test_bought_invalid_args(self, tmp_data_dir, app_config):
        """'bought' with bad args returns usage message."""
        result = handle_command("bought", "MSTR abc", app_config)
        assert "Usage" in result

    def test_bought_unknown_ticker(self, tmp_data_dir, app_config):
        """'bought FAKE 100 100000' returns error."""
        result = handle_command("bought", "FAKE 100 100000", app_config)
        assert "Unknown" in result


# ── Tests: P&L Calculation ────────────────────────────────────────────────


class TestPnL:
    def test_pnl_with_exit_price(self, tmp_data_dir, app_config):
        """P&L calculated correctly when sold at specific price."""
        pos = _make_position(
            ticker="MSTR", entry_price=130.0, amount_sek=100000, atr=3.5,
        )
        pos["fx_rate"] = 10.0
        state = {"active_position": pos, "trade_history": []}
        _save_state(state)

        result = handle_command("sold", "135", app_config)
        state = _load_state()
        trade = state["trade_history"][0]
        assert trade["exit_price_usd"] == 135.0
        # Entry at 130, exit at 135, +5/130 = +3.85%
        assert abs(trade["pnl_pct"] - 3.85) < 0.1

    def test_exit_alert_pnl(self):
        """format_exit_alert computes correct P&L display."""
        msg = format_exit_alert(
            ticker="MSTR",
            price=135.0,
            exit_type="trailing_stop",
            entry_price=130.0,
            amount_sek=100000,
            entry_time=datetime.now(timezone.utc).isoformat(),
            fx_rate=10.0,
        )
        assert "SELL MSTR" in msg
        assert "+3.8%" in msg or "+3.9%" in msg  # ~3.85%


# ── Tests: Config Loading & Expiry ────────────────────────────────────────


class TestConfig:
    def test_load_config_missing(self, tmp_data_dir):
        """No config file → returns None."""
        assert _load_config() is None

    def test_load_config_disabled(self, tmp_data_dir):
        """Disabled config → returns None."""
        _save_config({"enabled": False})
        assert _load_config() is None

    def test_load_config_expired(self, tmp_data_dir):
        """Expired config → auto-disables and returns None."""
        _save_config({
            "enabled": True,
            "tickers": ["BTC-USD"],
            "expiry": "2020-01-01T00:00:00Z",
        })
        assert _load_config() is None

    def test_load_config_valid(self, tmp_data_dir):
        """Valid config → returns dict."""
        _save_config({
            "enabled": True,
            "tickers": ["BTC-USD"],
            "expiry": "2099-01-01T00:00:00Z",
        })
        cfg = _load_config()
        assert cfg is not None
        assert cfg["tickers"] == ["BTC-USD"]


# ── Tests: Telegram Poller ────────────────────────────────────────────────


class TestTelegramPoller:
    def test_parse_command_bought(self):
        poller = TelegramPoller(
            {"telegram": {"token": "t", "chat_id": "123"}},
            on_command=lambda c, a, cfg: None,
        )
        cmd, args = poller._parse_command("bought MSTR 129.50 100000")
        assert cmd == "bought"
        assert "MSTR" in args

    def test_parse_command_sold(self):
        poller = TelegramPoller(
            {"telegram": {"token": "t", "chat_id": "123"}},
            on_command=lambda c, a, cfg: None,
        )
        cmd, args = poller._parse_command("sold")
        assert cmd == "sold"

    def test_parse_command_cancel(self):
        poller = TelegramPoller(
            {"telegram": {"token": "t", "chat_id": "123"}},
            on_command=lambda c, a, cfg: None,
        )
        cmd, args = poller._parse_command("cancel")
        assert cmd == "cancel"

    def test_parse_command_status(self):
        poller = TelegramPoller(
            {"telegram": {"token": "t", "chat_id": "123"}},
            on_command=lambda c, a, cfg: None,
        )
        cmd, args = poller._parse_command("status")
        assert cmd == "status"

    def test_parse_command_unknown(self):
        poller = TelegramPoller(
            {"telegram": {"token": "t", "chat_id": "123"}},
            on_command=lambda c, a, cfg: None,
        )
        cmd, args = poller._parse_command("hello world")
        assert cmd is None

    def test_ignores_wrong_chat_id(self):
        calls = []

        def handler(cmd, args, cfg):
            calls.append(cmd)
            return "ok"

        poller = TelegramPoller(
            {"telegram": {"token": "t", "chat_id": "123"}},
            on_command=handler,
        )
        poller._startup_time = time.time() - 120

        update = {
            "update_id": 1,
            "message": {
                "chat": {"id": 999},
                "date": int(time.time()),
                "text": "bought MSTR 130 100000",
            },
        }
        poller._handle_update(update)
        assert len(calls) == 0

    def test_processes_valid_message(self):
        calls = []

        def handler(cmd, args, cfg):
            calls.append((cmd, args))
            return "ok"

        poller = TelegramPoller(
            {"telegram": {"token": "t", "chat_id": "123"}},
            on_command=handler,
        )
        poller._startup_time = time.time() - 120

        update = {
            "update_id": 1,
            "message": {
                "chat": {"id": 123},
                "date": int(time.time()),
                "text": "bought MSTR 130 100000",
            },
        }
        with patch.object(poller, "_send_reply"):
            poller._handle_update(update)
        assert len(calls) == 1
        assert calls[0][0] == "bought"


# ── Tests: Alert Formatting ───────────────────────────────────────────────


class TestFormatting:
    def test_entry_alert_format(self):
        msg = format_entry_alert(
            ticker="BTC-USD",
            price=66000,
            conditions=["RSI 25 (oversold)", "F&G: 8 (Extreme Fear)", "4 BUY votes"],
            atr=1500,
            iskbets_cfg={
                "hard_stop_atr_mult": 2.0,
                "stage1_atr_mult": 1.5,
                "entry_cutoff_et": "14:30",
            },
        )
        assert "ISKBETS" in msg
        assert "BTC" in msg
        assert "66,000" in msg
        assert "Stop" in msg or "stop" in msg.lower()
        assert "Target #1" in msg

    def test_exit_alert_format(self):
        msg = format_exit_alert(
            ticker="BTC-USD",
            price=63000,
            exit_type="hard_stop",
            entry_price=66000,
            amount_sek=100000,
            entry_time=datetime.now(timezone.utc).isoformat(),
            fx_rate=10.5,
        )
        assert "SELL BTC" in msg
        assert "Hard stop" in msg

    def test_status_format(self):
        pos = _make_position(ticker="BTC-USD", entry_price=66000, atr=1500)
        msg = format_position_status(pos, 67000, 10.5)
        assert "ISKBETS Status" in msg
        assert "BTC-USD" in msg
        assert "Entry" in msg
        assert "Stop" in msg

    def test_entry_alert_with_l2_reasoning(self):
        """format_entry_alert includes Claude reasoning when provided."""
        msg = format_entry_alert(
            ticker="BTC-USD",
            price=66000,
            conditions=["RSI 25 (oversold)"],
            atr=1500,
            iskbets_cfg={"hard_stop_atr_mult": 2.0, "stage1_atr_mult": 1.5},
            l2_reasoning="Clean breakout with volume confirmation",
        )
        assert "Claude: Clean breakout" in msg

    def test_entry_alert_no_l2_reasoning(self):
        """format_entry_alert omits Claude line when reasoning is empty."""
        msg = format_entry_alert(
            ticker="BTC-USD",
            price=66000,
            conditions=["RSI 25 (oversold)"],
            atr=1500,
            iskbets_cfg={"hard_stop_atr_mult": 2.0, "stage1_atr_mult": 1.5},
            l2_reasoning="",
        )
        assert "Claude:" not in msg


# ── Tests: Layer 2 Gate ──────────────────────────────────────────────────


class TestLayer2Gate:
    @patch("portfolio.iskbets.subprocess.run")
    def test_gate_approved(self, mock_run, tmp_data_dir):
        """Claude approves → returns (True, reasoning)."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="DECISION: APPROVE\nREASONING: Clean setup with volume expansion.",
        )
        approved, reasoning = invoke_layer2_gate(
            "BTC-USD", 66000, ["RSI oversold"], {}, {}, 1500,
            {"layer2_gate": True}, {},
        )
        assert approved is True
        assert "Clean setup" in reasoning
        mock_run.assert_called_once()

    @patch("portfolio.iskbets.subprocess.run")
    def test_gate_skipped(self, mock_run, tmp_data_dir):
        """Claude skips → returns (False, reasoning)."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="DECISION: SKIP\nREASONING: All long TFs opposing, chasing.",
        )
        approved, reasoning = invoke_layer2_gate(
            "BTC-USD", 66000, ["RSI oversold"], {}, {}, 1500,
            {"layer2_gate": True}, {},
        )
        assert approved is False
        assert "opposing" in reasoning

    @patch("portfolio.iskbets.subprocess.run")
    def test_gate_timeout_fallback(self, mock_run, tmp_data_dir):
        """Timeout → fallback to approve."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=30)
        approved, reasoning = invoke_layer2_gate(
            "BTC-USD", 66000, ["RSI oversold"], {}, {}, 1500,
            {"layer2_gate": True}, {},
        )
        assert approved is True
        assert reasoning == ""

    def test_gate_disabled(self, tmp_data_dir):
        """layer2_gate=False → approve without subprocess call."""
        approved, reasoning = invoke_layer2_gate(
            "BTC-USD", 66000, ["RSI oversold"], {}, {}, 1500,
            {"layer2_gate": False}, {},
        )
        assert approved is True
        assert reasoning == ""

    @patch("portfolio.iskbets.subprocess.run")
    def test_gate_parse_failure(self, mock_run, tmp_data_dir):
        """Garbage output → fallback to approve."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="I'm not sure what to do here, this is random text.",
        )
        approved, reasoning = invoke_layer2_gate(
            "BTC-USD", 66000, ["RSI oversold"], {}, {}, 1500,
            {"layer2_gate": True}, {},
        )
        assert approved is True
        assert reasoning == ""


class TestParseGateResponse:
    def test_approve(self):
        approved, reasoning = _parse_gate_response(
            "DECISION: APPROVE\nREASONING: Looks good."
        )
        assert approved is True
        assert reasoning == "Looks good."

    def test_skip(self):
        approved, reasoning = _parse_gate_response(
            "DECISION: SKIP\nREASONING: All TFs bearish."
        )
        assert approved is False
        assert "bearish" in reasoning

    def test_empty(self):
        approved, reasoning = _parse_gate_response("")
        assert approved is True
        assert reasoning == ""
