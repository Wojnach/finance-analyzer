"""Tests for Big Bet Layer 2 evaluation, cooldown, and stale bet logic."""

import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

from portfolio.bigbet import (
    MAX_ACTIVE_BET_SECONDS,
    TOTAL_CONDITIONS,
    _build_eval_prompt,
    _evaluate_conditions,
    _format_alert,
    _format_window_closed,
    _parse_eval_response,
    _resolve_cooldown_minutes,
    check_bigbet,
    invoke_layer2_eval,
)


# --- Fixtures for original tests ---

def _make_signals_orig(ticker="BTC-USD"):
    return {
        ticker: {
            "indicators": {
                "rsi": 22.0,
                "macd_hist": -5.0,
                "price_vs_bb": "below_lower",
                "atr_pct": 3.2,
            },
            "extra": {
                "_buy_count": 4,
                "_sell_count": 1,
                "_total_applicable": 11,
                "fear_greed": 8,
                "volume_ratio": 2.5,
            },
        }
    }


def _make_tf_data_orig(ticker="BTC-USD"):
    labels = ["Now", "12h", "2d", "7d", "1mo", "3mo", "6mo"]
    actions = ["SELL", "HOLD", "SELL", "HOLD", "SELL", "SELL", "HOLD"]
    return {ticker: [(l, {"action": a}) for l, a in zip(labels, actions)]}


PRICES = {"BTC-USD": 65000.0}
CONFIG = {"telegram": {"token": "fake", "chat_id": "123"}}
CONDITIONS = ["RSI 22 (oversold) on 15m", "Below lower BB on Now, 12h", "F&G: 8 (Extreme Fear)"]


# --- _parse_eval_response tests ---

def test_eval_returns_probability():
    output = "PROBABILITY: 7/10\nREASONING: Good setup with volume confirmation and extreme fear."
    prob, reason = _parse_eval_response(output)
    assert prob == 7
    assert "Good setup" in reason


def test_eval_parse_failure():
    output = "I think this is a great trade opportunity!"
    prob, reason = _parse_eval_response(output)
    assert prob is None
    assert reason == ""


def test_eval_parse_clamps_range():
    output = "PROBABILITY: 15/10\nREASONING: Off the charts!"
    prob, reason = _parse_eval_response(output)
    assert prob == 10  # clamped to max

    output2 = "PROBABILITY: 0/10\nREASONING: Terrible."
    prob2, reason2 = _parse_eval_response(output2)
    assert prob2 == 1  # clamped to min


# --- invoke_layer2_eval tests ---

@patch("portfolio.bigbet.subprocess.run")
@patch.dict("os.environ", {}, clear=False)
def test_invoke_eval_success(mock_run, tmp_path, monkeypatch):
    monkeypatch.setattr("portfolio.bigbet.EVAL_LOG_FILE", tmp_path / "log.jsonl")
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="PROBABILITY: 7/10\nREASONING: Strong capitulation setup.",
    )
    prob, reason = invoke_layer2_eval(
        "BTC-USD", "BULL", CONDITIONS, _make_signals_orig(), _make_tf_data_orig(), PRICES, CONFIG
    )
    assert prob == 7
    assert "capitulation" in reason
    mock_run.assert_called_once()
    # Verify log was written
    assert (tmp_path / "log.jsonl").exists()


@patch("portfolio.bigbet.subprocess.run")
@patch.dict("os.environ", {}, clear=False)
def test_invoke_eval_timeout_fallback(mock_run, tmp_path, monkeypatch):
    monkeypatch.setattr("portfolio.bigbet.EVAL_LOG_FILE", tmp_path / "log.jsonl")
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=30)
    prob, reason = invoke_layer2_eval(
        "BTC-USD", "BULL", CONDITIONS, _make_signals_orig(), _make_tf_data_orig(), PRICES, CONFIG
    )
    assert prob is None
    assert reason == ""


# --- _format_alert tests ---

def test_format_alert_with_evaluation():
    msg = _format_alert(
        "BTC-USD", "BULL", CONDITIONS, PRICES, 10.5, {"fg": 8},
        probability=7, l2_reasoning="Strong capitulation setup.",
    )
    assert "Claude: 7/10" in msg
    assert "Strong capitulation setup" in msg


def test_format_alert_without_evaluation():
    msg = _format_alert(
        "BTC-USD", "BULL", CONDITIONS, PRICES, 10.5, {"fg": 8},
    )
    assert "Claude:" not in msg
    # Still has all the normal content
    assert "BIG BET: BULL BTC-USD" in msg
    assert "3/6 conditions met" in msg


# --- _build_eval_prompt tests ---

def test_build_eval_prompt_content():
    prompt = _build_eval_prompt(
        "BTC-USD", "BULL", CONDITIONS, _make_signals_orig(), _make_tf_data_orig(), PRICES
    )
    assert "BTC-USD" in prompt
    assert "BULL" in prompt
    assert "4B/1S/6H" in prompt
    assert "PROBABILITY:" in prompt
    assert "RSI 22" in prompt


# ---------------------------------------------------------------------------
# Helpers for new cooldown / stale-bet tests
# ---------------------------------------------------------------------------

def _make_signals(ticker, rsi=50, macd_hist=0, macd_hist_prev=0,
                  price_vs_bb="inside", volume_ratio=1.0,
                  volume_action="HOLD", fear_greed=50,
                  fear_greed_class="Neutral"):
    """Build a minimal signals dict for one ticker."""
    return {
        ticker: {
            "indicators": {
                "rsi": rsi,
                "macd_hist": macd_hist,
                "macd_hist_prev": macd_hist_prev,
                "price_vs_bb": price_vs_bb,
            },
            "extra": {
                "volume_ratio": volume_ratio,
                "volume_action": volume_action,
                "fear_greed": fear_greed,
                "fear_greed_class": fear_greed_class,
                "_buy_count": 0,
                "_sell_count": 0,
                "_total_applicable": 21,
            },
        }
    }


def _make_tf_data(ticker, bb_positions=None):
    """Build minimal tf_data with controllable BB positions per TF.

    bb_positions: list of (label, price_vs_bb) tuples, e.g.
        [("Now", "below_lower"), ("12h", "below_lower"), ("2d", "inside")]
    """
    if bb_positions is None:
        bb_positions = []
    result = []
    for label, pos in bb_positions:
        result.append((label, {"indicators": {"price_vs_bb": pos, "rsi": 50}}))
    return {ticker: result}


def _bull_setup(ticker="BTC-USD"):
    """Create signals + tf_data that produce >= 3 bull conditions.

    Conditions triggered:
      1. RSI 20 oversold
      2. Below lower BB on Now, 12h (2 TFs)
      3. F&G 10
    """
    signals = _make_signals(
        ticker, rsi=20, fear_greed=10, fear_greed_class="Extreme Fear",
        macd_hist=-5, macd_hist_prev=-10,  # MACD turning up while oversold
    )
    tf_data = _make_tf_data(ticker, [
        ("Now", "below_lower"), ("12h", "below_lower"), ("2d", "inside"),
    ])
    prices_usd = {ticker: 65000}
    return signals, tf_data, prices_usd


def _neutral_setup(ticker="BTC-USD"):
    """Create signals + tf_data with 0 conditions (all neutral)."""
    signals = _make_signals(ticker, rsi=50, fear_greed=50)
    tf_data = _make_tf_data(ticker, [("Now", "inside"), ("12h", "inside")])
    prices_usd = {ticker: 66000}
    return signals, tf_data, prices_usd


# ---------------------------------------------------------------------------
# _resolve_cooldown_minutes
# ---------------------------------------------------------------------------

class TestResolveCooldownMinutes:
    def test_default_is_10_minutes(self):
        assert _resolve_cooldown_minutes({}) == 10

    def test_cooldown_minutes_key(self):
        assert _resolve_cooldown_minutes({"cooldown_minutes": 5}) == 5

    def test_cooldown_hours_backward_compat(self):
        """Legacy cooldown_hours is converted to minutes."""
        assert _resolve_cooldown_minutes({"cooldown_hours": 2}) == 120

    def test_cooldown_minutes_takes_precedence(self):
        """When both keys present, cooldown_minutes wins."""
        cfg = {"cooldown_minutes": 15, "cooldown_hours": 4}
        assert _resolve_cooldown_minutes(cfg) == 15

    def test_cooldown_hours_zero(self):
        assert _resolve_cooldown_minutes({"cooldown_hours": 0}) == 0


# ---------------------------------------------------------------------------
# _format_window_closed
# ---------------------------------------------------------------------------

class TestFormatWindowClosed:
    def test_basic_format(self):
        msg = _format_window_closed("BTC-USD", "BULL", 65000, 66200, 30)
        assert "BIG BET CLOSED" in msg
        assert "BULL" in msg
        assert "BTC-USD" in msg
        assert "30m" in msg
        assert "+1.8%" in msg
        assert "$65,000.00" in msg
        assert "$66,200.00" in msg

    def test_negative_price_change(self):
        msg = _format_window_closed("ETH-USD", "BEAR", 2000, 1900, 45)
        assert "-5.0%" in msg

    def test_zero_trigger_price(self):
        """No crash when trigger price is 0."""
        msg = _format_window_closed("BTC-USD", "BULL", 0, 66000, 10)
        assert "Current: $66,000.00" in msg
        # Should not show entry price comparison
        assert "Entry price" not in msg


# ---------------------------------------------------------------------------
# check_bigbet -- cooldown behavior
# ---------------------------------------------------------------------------

class TestCheckBigbetCooldown:
    """Verify the 10-minute default cooldown and backwards compat."""

    @patch("portfolio.bigbet.invoke_layer2_eval", return_value=(7, "Looks good"))
    @patch("portfolio.bigbet._send_telegram")
    @patch("portfolio.bigbet._save_state")
    @patch("portfolio.bigbet._load_state", return_value={"cooldowns": {}, "price_history": {}, "active_bets": {}})
    def test_alert_fires_when_cooldown_expired(self, mock_load, mock_save, mock_tg, mock_eval):
        """Alert fires when no prior cooldown exists."""
        signals, tf_data, prices_usd = _bull_setup()
        config = {}

        check_bigbet(signals, prices_usd, 10.5, tf_data, config)

        mock_tg.assert_called()
        # Verify the message contains BIG BET
        call_args = mock_tg.call_args[0][0]
        assert "BIG BET" in call_args

    @patch("portfolio.bigbet.invoke_layer2_eval", return_value=(None, ""))
    @patch("portfolio.bigbet._send_telegram")
    @patch("portfolio.bigbet._save_state")
    @patch("portfolio.bigbet._load_state")
    def test_alert_blocked_during_cooldown(self, mock_load, mock_save, mock_tg, mock_eval):
        """Alert is suppressed when inside cooldown window (default 10min)."""
        now = time.time()
        # Last alert was 5 minutes ago -- within 10-min cooldown
        mock_load.return_value = {
            "cooldowns": {"BTC-USD_BULL": now - 300},
            "price_history": {},
            "active_bets": {},
        }
        signals, tf_data, prices_usd = _bull_setup()
        config = {}

        check_bigbet(signals, prices_usd, 10.5, tf_data, config)

        # invoke_layer2_eval should NOT be called (blocked by cooldown)
        mock_eval.assert_not_called()

    @patch("portfolio.bigbet.invoke_layer2_eval", return_value=(None, ""))
    @patch("portfolio.bigbet._send_telegram")
    @patch("portfolio.bigbet._save_state")
    @patch("portfolio.bigbet._load_state")
    def test_alert_fires_after_cooldown_expires(self, mock_load, mock_save, mock_tg, mock_eval):
        """Alert fires when cooldown has elapsed (>10min ago)."""
        now = time.time()
        # Last alert was 11 minutes ago -- outside 10-min cooldown
        mock_load.return_value = {
            "cooldowns": {"BTC-USD_BULL": now - 660},
            "price_history": {},
            "active_bets": {},
        }
        signals, tf_data, prices_usd = _bull_setup()
        config = {}

        check_bigbet(signals, prices_usd, 10.5, tf_data, config)

        mock_eval.assert_called_once()
        # Telegram called at least once (for the alert)
        assert mock_tg.call_count >= 1

    @patch("portfolio.bigbet.invoke_layer2_eval", return_value=(None, ""))
    @patch("portfolio.bigbet._send_telegram")
    @patch("portfolio.bigbet._save_state")
    @patch("portfolio.bigbet._load_state")
    def test_legacy_cooldown_hours_config(self, mock_load, mock_save, mock_tg, mock_eval):
        """Legacy cooldown_hours=1 means 60-min cooldown."""
        now = time.time()
        # Last alert 30 minutes ago -- within 1-hour cooldown
        mock_load.return_value = {
            "cooldowns": {"BTC-USD_BULL": now - 1800},
            "price_history": {},
            "active_bets": {},
        }
        signals, tf_data, prices_usd = _bull_setup()
        config = {"bigbet": {"cooldown_hours": 1}}

        check_bigbet(signals, prices_usd, 10.5, tf_data, config)

        # Should be blocked -- 30 min < 60 min cooldown
        mock_eval.assert_not_called()


# ---------------------------------------------------------------------------
# check_bigbet -- active bet tracking
# ---------------------------------------------------------------------------

class TestActiveBetTracking:
    """Verify active bets are created, tracked, and expired."""

    @patch("portfolio.bigbet.invoke_layer2_eval", return_value=(7, "Looks good"))
    @patch("portfolio.bigbet._send_telegram")
    @patch("portfolio.bigbet._save_state")
    @patch("portfolio.bigbet._load_state", return_value={"cooldowns": {}, "price_history": {}, "active_bets": {}})
    def test_alert_creates_active_bet(self, mock_load, mock_save, mock_tg, mock_eval):
        """When alert fires, an active bet entry is created in state."""
        signals, tf_data, prices_usd = _bull_setup()
        config = {}

        check_bigbet(signals, prices_usd, 10.5, tf_data, config)

        # Check saved state contains active_bets
        saved_state = mock_save.call_args[0][0]
        assert "active_bets" in saved_state
        assert "BTC-USD_BULL" in saved_state["active_bets"]
        bet = saved_state["active_bets"]["BTC-USD_BULL"]
        assert bet["price_at_trigger"] == 65000
        assert len(bet["conditions"]) >= 3
        assert "triggered_at" in bet

    @patch("portfolio.bigbet.invoke_layer2_eval", return_value=(None, ""))
    @patch("portfolio.bigbet._send_telegram")
    @patch("portfolio.bigbet._save_state")
    @patch("portfolio.bigbet._load_state")
    def test_stale_bet_sends_window_closed(self, mock_load, mock_save, mock_tg, mock_eval):
        """When conditions fade, a 'window closed' notification is sent."""
        now = time.time()
        mock_load.return_value = {
            "cooldowns": {"BTC-USD_BULL": now - 900},  # alert 15 min ago
            "price_history": {},
            "active_bets": {
                "BTC-USD_BULL": {
                    "triggered_at": now - 900,
                    "conditions": ["RSI 20 oversold", "Below lower BB", "F&G: 10"],
                    "price_at_trigger": 65000,
                }
            },
        }
        # Neutral signals -- conditions no longer met
        signals, tf_data, prices_usd = _neutral_setup()
        config = {}

        check_bigbet(signals, prices_usd, 10.5, tf_data, config)

        # Should have sent a "window closed" message
        assert mock_tg.call_count >= 1
        close_msg = mock_tg.call_args_list[0][0][0]
        assert "BIG BET CLOSED" in close_msg
        assert "BULL" in close_msg
        assert "BTC-USD" in close_msg

        # Active bet should be removed from state
        saved_state = mock_save.call_args[0][0]
        assert "BTC-USD_BULL" not in saved_state.get("active_bets", {})

    @patch("portfolio.bigbet.invoke_layer2_eval", return_value=(None, ""))
    @patch("portfolio.bigbet._send_telegram")
    @patch("portfolio.bigbet._save_state")
    @patch("portfolio.bigbet._load_state")
    def test_stale_bet_auto_expires_after_6h(self, mock_load, mock_save, mock_tg, mock_eval):
        """Active bets are force-expired after 6 hours."""
        now = time.time()
        triggered_at = now - (MAX_ACTIVE_BET_SECONDS + 60)  # 6h + 1min ago
        mock_load.return_value = {
            "cooldowns": {"ETH-USD_BEAR": triggered_at},
            "price_history": {},
            "active_bets": {
                "ETH-USD_BEAR": {
                    "triggered_at": triggered_at,
                    "conditions": ["RSI 85 overbought"],
                    "price_at_trigger": 2000,
                }
            },
        }
        # Even if conditions are still met, 6h expiry should fire
        signals = _make_signals("ETH-USD", rsi=85, fear_greed=90, fear_greed_class="Greed")
        tf_data = _make_tf_data("ETH-USD", [
            ("Now", "above_upper"), ("12h", "above_upper"),
        ])
        prices_usd = {"ETH-USD": 2100}
        config = {}

        check_bigbet(signals, prices_usd, 10.5, tf_data, config)

        # Should send a window-closed message for the 6h expiry
        close_calls = [
            c for c in mock_tg.call_args_list
            if "BIG BET CLOSED" in c[0][0]
        ]
        assert len(close_calls) >= 1
        assert "ETH-USD" in close_calls[0][0][0]

    @patch("portfolio.bigbet.invoke_layer2_eval", return_value=(None, ""))
    @patch("portfolio.bigbet._send_telegram")
    @patch("portfolio.bigbet._save_state")
    @patch("portfolio.bigbet._load_state")
    def test_active_bet_not_expired_when_conditions_still_met(self, mock_load, mock_save, mock_tg, mock_eval):
        """Active bet stays if conditions are still met and < 6h old."""
        now = time.time()
        mock_load.return_value = {
            "cooldowns": {"BTC-USD_BULL": now - 300},  # 5 min ago
            "price_history": {},
            "active_bets": {
                "BTC-USD_BULL": {
                    "triggered_at": now - 300,
                    "conditions": ["RSI 20 oversold", "Below lower BB", "F&G: 10"],
                    "price_at_trigger": 65000,
                }
            },
        }
        # Conditions still met
        signals, tf_data, prices_usd = _bull_setup()
        config = {}

        check_bigbet(signals, prices_usd, 10.5, tf_data, config)

        # No "window closed" message should be sent
        close_calls = [
            c for c in mock_tg.call_args_list
            if "BIG BET CLOSED" in c[0][0]
        ]
        assert len(close_calls) == 0

        # Active bet should still be in saved state
        saved_state = mock_save.call_args[0][0]
        assert "BTC-USD_BULL" in saved_state.get("active_bets", {})

    @patch("portfolio.bigbet.invoke_layer2_eval", return_value=(None, ""))
    @patch("portfolio.bigbet._send_telegram")
    @patch("portfolio.bigbet._save_state")
    @patch("portfolio.bigbet._load_state")
    def test_window_closed_price_change_calculation(self, mock_load, mock_save, mock_tg, mock_eval):
        """Window closed message shows correct price change percentage."""
        now = time.time()
        mock_load.return_value = {
            "cooldowns": {"BTC-USD_BULL": now - 1200},
            "price_history": {},
            "active_bets": {
                "BTC-USD_BULL": {
                    "triggered_at": now - 1200,
                    "conditions": ["RSI oversold"],
                    "price_at_trigger": 65000,
                }
            },
        }
        # Neutral signals -- conditions no longer met. Price moved up.
        signals, tf_data, prices_usd = _neutral_setup()
        prices_usd["BTC-USD"] = 66300  # +2%
        config = {}

        check_bigbet(signals, prices_usd, 10.5, tf_data, config)

        close_msg = mock_tg.call_args_list[0][0][0]
        assert "+2.0%" in close_msg
        assert "$65,000.00" in close_msg
        assert "$66,300.00" in close_msg


# ---------------------------------------------------------------------------
# Edge cases -- empty / missing state
# ---------------------------------------------------------------------------

class TestEdgeCases:
    @patch("portfolio.bigbet.invoke_layer2_eval", return_value=(None, ""))
    @patch("portfolio.bigbet._send_telegram")
    @patch("portfolio.bigbet._save_state")
    @patch("portfolio.bigbet._load_state", return_value={"cooldowns": {}, "price_history": {}})
    def test_no_crash_on_missing_active_bets_key(self, mock_load, mock_save, mock_tg, mock_eval):
        """State without active_bets key doesn't crash."""
        signals, tf_data, prices_usd = _neutral_setup()
        config = {}

        # Should not raise
        check_bigbet(signals, prices_usd, 10.5, tf_data, config)

    @patch("portfolio.bigbet.invoke_layer2_eval", return_value=(None, ""))
    @patch("portfolio.bigbet._send_telegram")
    @patch("portfolio.bigbet._save_state")
    @patch("portfolio.bigbet._load_state", return_value={"cooldowns": {}, "price_history": {}, "active_bets": {}})
    def test_no_crash_on_empty_signals(self, mock_load, mock_save, mock_tg, mock_eval):
        """Empty signals dict doesn't crash."""
        config = {}
        check_bigbet({}, {}, 10.5, {}, config)

    @patch("portfolio.bigbet.invoke_layer2_eval", return_value=(None, ""))
    @patch("portfolio.bigbet._send_telegram")
    @patch("portfolio.bigbet._save_state")
    @patch("portfolio.bigbet._load_state", return_value={"cooldowns": {}, "price_history": {}, "active_bets": {}})
    def test_no_crash_on_zero_price(self, mock_load, mock_save, mock_tg, mock_eval):
        """Zero price ticker is skipped."""
        signals = _make_signals("BTC-USD", rsi=20, fear_greed=10)
        tf_data = _make_tf_data("BTC-USD", [("Now", "below_lower"), ("12h", "below_lower")])
        prices_usd = {"BTC-USD": 0}
        config = {}

        check_bigbet(signals, prices_usd, 10.5, tf_data, config)

        # No alert should fire (price is 0)
        mock_tg.assert_not_called()

    @patch("portfolio.bigbet.invoke_layer2_eval", return_value=(None, ""))
    @patch("portfolio.bigbet._send_telegram")
    @patch("portfolio.bigbet._save_state")
    @patch("portfolio.bigbet._load_state")
    def test_active_bet_ticker_gone_from_signals(self, mock_load, mock_save, mock_tg, mock_eval):
        """Active bet for ticker no longer in signals gets expired."""
        now = time.time()
        mock_load.return_value = {
            "cooldowns": {},
            "price_history": {},
            "active_bets": {
                "GONE-USD_BULL": {
                    "triggered_at": now - 600,
                    "conditions": ["RSI oversold"],
                    "price_at_trigger": 100,
                }
            },
        }
        # GONE-USD is not in signals at all
        signals, tf_data, prices_usd = _neutral_setup("BTC-USD")
        config = {}

        check_bigbet(signals, prices_usd, 10.5, tf_data, config)

        # Should send a window closed for GONE-USD
        assert mock_tg.call_count >= 1
        close_msg = mock_tg.call_args_list[0][0][0]
        assert "GONE-USD" in close_msg
        assert "BIG BET CLOSED" in close_msg
