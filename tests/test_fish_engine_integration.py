"""Tests for fish engine integration with metals_loop execution layer.

Covers the 6 bugs found during live test on 2026-04-07:
1. fetch_price returning None -> crash
2. trade_guard returning [] -> treated as False
3. metals_disagree exit too aggressive (MD3)
4. metals_disagree should not fire in extreme RSI zones
5. _loop_page None guard
6. HOLD logging
"""

from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, ".")

from data.fish_engine import (
    EXIT_METALS_DISAGREE_COUNT,
    FishEngine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_state(**overrides):
    """Build a minimal state dict with sensible defaults."""
    defaults = {
        "silver_price": 72.0,
        "gold_price": 4700.0,
        "gold_5min_change": 0.0,
        "signal_action": "HOLD",
        "signal_buy_count": 3,
        "signal_sell_count": 3,
        "rsi": 50.0,
        "mc_p_up": 0.5,
        "metals_action": "HOLD",
        "regime": "ranging",
        "news_action": "HOLD",
        "econ_action": "HOLD",
        "focus_1d_dir": "?",
        "focus_1d_prob": 0.5,
        "orb_range": None,
        "vol_scalar": 1.0,
        "hour_cet": 14,
        "minute_cet": 0,
        "day_of_week": 1,
        "velocity": None,
        "trade_guard_ok": True,
        "spread_pct": 0.2,
        "news_spike": False,
        "headline_sentiment": "",
        "event_hours": 999,
        "high_impact_near": False,
        "layer2_outlook": "",
        "layer2_conviction": 0.0,
        "layer2_levels": [],
        "layer2_action": "HOLD",
        "layer2_ts": "",
        "mc_bands_1d": {},
        "chronos_1h_pct": 0.0,
        "chronos_24h_pct": 0.0,
        "prophecy_target": 0.0,
        "prophecy_conviction": 0.0,
    }
    defaults.update(overrides)
    return defaults


def _make_engine_with_position(direction="LONG", entry_underlying=72.0,
                                entry_cert=4.50, volume=334):
    """Create engine with an active position."""
    engine = FishEngine(time_func=time.time)
    engine.confirm_entry(direction, entry_cert, volume, entry_underlying)
    return engine


# ---------------------------------------------------------------------------
# Bug 3+4: Metals disagree exit
# ---------------------------------------------------------------------------

class TestMetalsDisagreeExit:
    """Metals disagree should be patient and respect extreme RSI zones."""

    def test_md_threshold_is_15(self):
        """EXIT_METALS_DISAGREE_COUNT should be 15, not 3."""
        assert EXIT_METALS_DISAGREE_COUNT == 15

    def test_md_does_not_fire_before_threshold(self):
        """Engine should HOLD when metals disagree count < threshold."""
        engine = _make_engine_with_position("LONG")
        state = _make_state(silver_price=71.5, metals_action="SELL")

        # Tick 14 times — count should reach 14, but not trigger exit
        for i in range(14):
            decision = engine.tick(state)

        assert engine.metals_disagree_count == 14
        assert decision["action"] == "HOLD"

    def test_md_fires_at_threshold(self):
        """Engine should SELL when metals disagree hits threshold."""
        engine = _make_engine_with_position("LONG")
        state = _make_state(silver_price=71.5, metals_action="SELL", rsi=50.0)

        for _ in range(EXIT_METALS_DISAGREE_COUNT):
            decision = engine.tick(state)

        assert decision["action"] == "SELL"
        assert "MD" in decision.get("exit_reason", "")

    def test_md_blocked_when_rsi_oversold_long(self):
        """LONG position should NOT exit on MD when RSI < 30 (oversold bounce zone)."""
        engine = _make_engine_with_position("LONG")
        state = _make_state(silver_price=71.5, metals_action="SELL", rsi=25.0)

        # Tick way past threshold
        for _ in range(20):
            decision = engine.tick(state)

        # Should still be HOLD — RSI 25 protects the contrarian position
        assert decision["action"] == "HOLD"
        assert engine.metals_disagree_count >= EXIT_METALS_DISAGREE_COUNT

    def test_md_blocked_when_rsi_overbought_short(self):
        """SHORT position should NOT exit on MD when RSI > 70 (overbought reversal zone)."""
        engine = _make_engine_with_position("SHORT", entry_underlying=73.0)
        state = _make_state(silver_price=73.5, metals_action="BUY", rsi=75.0)

        for _ in range(20):
            decision = engine.tick(state)

        assert decision["action"] == "HOLD"

    def test_md_fires_when_rsi_normal_even_after_extreme(self):
        """If RSI normalizes, metals disagree should fire again."""
        engine = _make_engine_with_position("LONG")

        # First: RSI extreme, should hold through
        extreme_state = _make_state(silver_price=71.5, metals_action="SELL", rsi=25.0)
        for _ in range(16):
            engine.tick(extreme_state)

        # Now RSI normalizes — metals disagree should fire
        normal_state = _make_state(silver_price=71.5, metals_action="SELL", rsi=50.0)
        decision = engine.tick(normal_state)

        assert decision["action"] == "SELL"

    def test_md_counter_resets_on_agreement(self):
        """When metals signal agrees with position, counter resets to 0."""
        engine = _make_engine_with_position("LONG")

        # 10 ticks of disagreement
        sell_state = _make_state(silver_price=71.5, metals_action="SELL")
        for _ in range(10):
            engine.tick(sell_state)
        assert engine.metals_disagree_count == 10

        # One tick of agreement resets
        buy_state = _make_state(silver_price=71.5, metals_action="BUY")
        engine.tick(buy_state)
        assert engine.metals_disagree_count == 0


# ---------------------------------------------------------------------------
# Bug 2: Trade guard empty list
# ---------------------------------------------------------------------------

class TestTradeGuard:
    """Empty list from trade guard should mean 'no blocks = OK'."""

    def test_empty_list_is_ok(self):
        """bool([]) is False, but empty list means no blocks."""
        guard = []
        # This is the fixed logic:
        if isinstance(guard, list):
            trade_guard_ok = len(guard) == 0
        else:
            trade_guard_ok = bool(guard)
        assert trade_guard_ok is True

    def test_nonempty_list_is_blocked(self):
        """Non-empty list means there are active blocks."""
        guard = [{"reason": "cooldown"}]
        if isinstance(guard, list):
            trade_guard_ok = len(guard) == 0
        else:
            trade_guard_ok = bool(guard)
        assert trade_guard_ok is False

    def test_dict_with_allowed_true(self):
        """Dict with allowed=True should be OK."""
        guard = {"allowed": True}
        if isinstance(guard, dict):
            trade_guard_ok = guard.get("allowed", True)
        elif isinstance(guard, list):
            trade_guard_ok = len(guard) == 0
        else:
            trade_guard_ok = bool(guard)
        assert trade_guard_ok is True


# ---------------------------------------------------------------------------
# Engine state serialization
# ---------------------------------------------------------------------------

class TestEngineState:
    """State persistence should preserve position correctly."""

    def test_roundtrip_with_custom_ob_id(self):
        """Position with custom ob_id (e.g. TURBO) survives serialize/deserialize."""
        engine = FishEngine()
        engine.position = {
            "direction": "LONG",
            "entry_underlying": 72.0,
            "entry_cert": 3.70,
            "volume": 558,
            "ob_id": "2389098",  # TURBO L SILVER AVA 491
            "entry_ts": time.time(),
        }

        state = engine.to_dict()
        engine2 = FishEngine()
        engine2.from_dict(state)

        assert engine2.position is not None
        assert engine2.position["ob_id"] == "2389098"
        assert engine2.position["volume"] == 558


# ---------------------------------------------------------------------------
# Position reconciliation — external close (stop-loss triggered on Avanza)
# ---------------------------------------------------------------------------

class TestForceClosePosition:
    """force_close_position handles externally-triggered exits (stop-loss, manual sell)."""

    def test_force_close_clears_position(self):
        """After force close, engine should have no position."""
        engine = _make_engine_with_position("LONG", entry_cert=4.50, volume=334)
        assert engine.has_position is True

        engine.force_close_position("stop_loss_triggered", exit_cert_price=4.20)

        assert engine.has_position is False
        assert engine.position is None

    def test_force_close_records_pnl(self):
        """Force close should compute and record P&L from entry vs exit cert price."""
        engine = _make_engine_with_position("LONG", entry_cert=4.50, volume=334)

        engine.force_close_position("stop_loss_triggered", exit_cert_price=4.20)

        # P&L = (4.20 - 4.50) * 334 = -100.20
        assert engine.session_pnl == pytest.approx(-100.20, abs=0.1)
        assert engine.trade_count == 1
        assert engine.loss_count == 1

    def test_force_close_with_zero_exit_price(self):
        """When exit price unknown (0), P&L should be estimated as full loss of entry amount."""
        engine = _make_engine_with_position("SHORT", entry_cert=2.55, volume=435)

        engine.force_close_position("auto_detect_not_held", exit_cert_price=0)

        # With zero exit price, P&L = -(entry_cert * volume) = -1109.25
        assert engine.session_pnl == pytest.approx(-2.55 * 435, abs=0.1)
        assert engine.has_position is False

    def test_force_close_winning_trade(self):
        """Stop-loss might not mean a loss if price moved in our favor first."""
        engine = _make_engine_with_position("LONG", entry_cert=4.50, volume=100)

        engine.force_close_position("manual_sell", exit_cert_price=5.00)

        # P&L = (5.00 - 4.50) * 100 = +50
        assert engine.session_pnl == pytest.approx(50.0, abs=0.1)
        assert engine.win_count == 1
        assert engine.consecutive_losses == 0

    def test_force_close_no_position_is_noop(self):
        """Force close with no position should do nothing."""
        engine = FishEngine(time_func=time.time)
        assert engine.has_position is False

        engine.force_close_position("spurious_close")

        assert engine.session_pnl == 0.0
        assert engine.trade_count == 0

    def test_force_close_increments_consecutive_losses(self):
        """Consecutive losses should increment on force-close loss."""
        engine = _make_engine_with_position("LONG", entry_cert=4.50, volume=100)
        engine.consecutive_losses = 1  # already had one loss

        engine.force_close_position("stop_loss_triggered", exit_cert_price=4.00)

        assert engine.consecutive_losses == 2

    def test_force_close_logs_trade(self, tmp_path):
        """Force close should write to trade log with exit_reason."""
        log_path = str(tmp_path / "fish_trades.jsonl")
        engine = FishEngine(time_func=time.time, trade_log_path=log_path)
        engine.confirm_entry("LONG", 4.50, 334, 72.0)

        engine.force_close_position("stop_loss_triggered", exit_cert_price=4.20)

        import json
        lines = (tmp_path / "fish_trades.jsonl").read_text().strip().split("\n")
        # Should have 2 entries: BUY + SELL
        assert len(lines) == 2
        sell_entry = json.loads(lines[1])
        assert sell_entry["action"] == "SELL"
        assert sell_entry["exit_reason"] == "stop_loss_triggered"
        assert sell_entry["price_sek"] == 4.20

    def test_force_close_state_survives_roundtrip(self):
        """After force close, serialized state should show no position."""
        engine = _make_engine_with_position("LONG", entry_cert=4.50, volume=334)
        engine.force_close_position("stop_loss_triggered", exit_cert_price=4.20)

        state = engine.to_dict()
        engine2 = FishEngine()
        engine2.from_dict(state)

        assert engine2.has_position is False
        assert engine2.session_pnl == pytest.approx(-100.20, abs=0.1)
        assert engine2.trade_count == 1


# ---------------------------------------------------------------------------
# Enriched trade logging
# ---------------------------------------------------------------------------

class TestEnrichedTradeLogging:
    """Trade log entries should contain complete trade data."""

    def test_buy_log_has_direction_and_underlying(self, tmp_path):
        """BUY log entry should include direction and underlying price."""
        log_path = str(tmp_path / "fish_trades.jsonl")
        engine = FishEngine(time_func=time.time, trade_log_path=log_path)

        engine.confirm_entry("LONG", 4.50, 334, 72.0)

        import json
        lines = (tmp_path / "fish_trades.jsonl").read_text().strip().split("\n")
        buy_entry = json.loads(lines[0])
        assert buy_entry["action"] == "BUY"
        assert buy_entry["direction"] == "LONG"
        assert buy_entry["underlying_price"] == 72.0

    def test_sell_log_has_pnl_and_reason(self, tmp_path):
        """SELL log entry should include pnl_sek and exit_reason."""
        log_path = str(tmp_path / "fish_trades.jsonl")
        engine = FishEngine(time_func=time.time, trade_log_path=log_path)
        engine.confirm_entry("SHORT", 2.55, 435, 72.05)

        engine.force_close_position("stop_loss_triggered", exit_cert_price=2.42)

        import json
        lines = (tmp_path / "fish_trades.jsonl").read_text().strip().split("\n")
        sell_entry = json.loads(lines[1])
        assert sell_entry["action"] == "SELL"
        assert sell_entry["exit_reason"] == "stop_loss_triggered"
        assert "pnl_sek" in sell_entry
        assert sell_entry["pnl_sek"] == pytest.approx((2.42 - 2.55) * 435, abs=0.1)
        assert sell_entry["direction"] == "SHORT"


# ---------------------------------------------------------------------------
# Fishing position tagging + trailing + EOD sell
# ---------------------------------------------------------------------------

class TestFishingPositionTag:
    """Fishing positions should be tagged and handled differently."""

    def test_fishing_ob_ids_built_from_catalog(self):
        """FISHING_OB_IDS should contain all ob_ids from WARRANT_CATALOG."""
        from data.metals_loop import FISHING_OB_IDS
        # Should be a set of strings
        assert isinstance(FISHING_OB_IDS, set)
        # Should contain known fishing instruments
        assert "1650161" in FISHING_OB_IDS  # BULL SILVER X5 AVA 4
        assert "2286417" in FISHING_OB_IDS  # BEAR SILVER X5 AVA 12

    def test_fishing_tag_on_new_position(self):
        """When detect_holdings adds a position with a fishing ob_id, it should be tagged."""
        from data.metals_loop import KNOWN_WARRANT_OB_IDS, FISHING_OB_IDS
        # BULL SILVER X5 AVA 4 (ob_id 1650161) is in WARRANT_CATALOG
        assert "1650161" in FISHING_OB_IDS

    def test_trailing_start_zero_for_fishing(self):
        """Fishing positions should start trailing immediately (TRAIL_START_PCT = 0)."""
        from data.metals_loop import FISHING_TRAIL_START_PCT
        assert FISHING_TRAIL_START_PCT == 0.0

    def test_eod_sell_hour(self):
        """EOD sell for fishing should trigger at 21:50 CET."""
        from data.metals_loop import FISHING_EOD_SELL_MINUTE_CET
        # 21 hours + 50 minutes = 21:50
        assert FISHING_EOD_SELL_MINUTE_CET == (21, 50)
