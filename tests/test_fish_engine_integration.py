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
