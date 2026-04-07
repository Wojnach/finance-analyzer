"""Comprehensive tests for data.fish_engine.FishEngine.

Tests cover: voting system, Layer 2 weight, staleness, MC-informed TP/SL,
auto-disable at 21:55, session-end forced sell, cooldown enforcement,
mode detection (auto-switch after 3 losses), position tracking, state persistence.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta

sys.path.insert(0, ".")
from data.fish_engine import (
    COOLDOWN_HIGH_CONV,
    COOLDOWN_NORMAL,
    LONG_OB,
    SHORT_OB,
    FishEngine,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(**overrides):
    """Build a minimal state dict with sensible defaults."""
    defaults = {
        "silver_price": 75.0,
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


def make_engine(t=1000000.0, **kwargs):
    """Create an engine with a fixed time function (default t=1000000)."""
    return FishEngine(time_func=lambda: t, trade_log_path="/dev/null", **kwargs)


def _put_position(engine, direction="LONG", entry_underlying=75.0,
                   entry_cert=7.0, volume=150, entry_ts=None):
    """Manually set a position on the engine (bypass confirm_entry)."""
    if entry_ts is None:
        entry_ts = engine._time() - 600  # 10 min ago
    ob_id = engine.long_ob if direction == "LONG" else engine.short_ob
    engine.position = {
        "direction": direction,
        "entry_underlying": entry_underlying,
        "entry_cert": entry_cert,
        "volume": volume,
        "ob_id": ob_id,
        "entry_ts": entry_ts,
    }


# ===================================================================
# 1. Voting system -- 2+ tactics must agree for entry
# ===================================================================


class TestVotingSystem:
    """Rule 3: MIN_VOTES (2) required. Single tactic = no trade."""

    def test_no_tactic_votes_gives_hold(self):
        engine = make_engine()
        decision = engine.tick(make_state())
        assert decision["action"] == "HOLD"
        assert "no tactic votes" in decision["reason"]

    def test_single_tactic_not_enough(self):
        """One tactic alone should not trigger a BUY."""
        engine = make_engine()
        # Gold-lead only: big gold move but nothing else agrees
        state = make_state(gold_5min_change=0.8, hour_cet=15)
        decision = engine.tick(state)
        assert decision["action"] == "HOLD"
        assert "only 1 tactic" in decision["reason"]

    def test_two_tactics_agree_long_triggers_buy(self):
        """Two tactics agreeing LONG should produce a BUY LONG."""
        engine = make_engine()
        # ORB breakout LONG + gold-lead LONG
        engine.set_orb_range(high=74.0, low=72.0)
        state = make_state(
            silver_price=74.5,  # above ORB high
            gold_5min_change=0.8,  # gold leading
            hour_cet=15,  # not dead zone
        )
        decision = engine.tick(state)
        assert decision["action"] == "BUY"
        assert decision["direction"] == "LONG"
        assert len(decision["tactics_agreed"]) >= 2
        assert "orb" in decision["tactics_agreed"]
        assert "gold_lead" in decision["tactics_agreed"]

    def test_two_tactics_agree_short_triggers_buy(self):
        """Two tactics agreeing SHORT should produce a BUY SHORT."""
        engine = make_engine()
        engine.set_orb_range(high=76.0, low=74.0)
        state = make_state(
            silver_price=73.5,  # below ORB low
            gold_5min_change=-0.8,  # gold leading down
            hour_cet=15,
        )
        decision = engine.tick(state)
        assert decision["action"] == "BUY"
        assert decision["direction"] == "SHORT"
        assert len(decision["tactics_agreed"]) >= 2

    def test_conflicting_votes_gives_hold(self):
        """One LONG + one SHORT = conflict, HOLD."""
        engine = make_engine()
        engine.set_orb_range(high=74.0, low=72.0)
        state = make_state(
            silver_price=74.5,  # ORB says LONG
            gold_5min_change=-0.8,  # gold-lead says SHORT
            hour_cet=15,
        )
        decision = engine.tick(state)
        assert decision["action"] == "HOLD"
        assert "conflict" in decision["reason"]

    def test_long_wins_over_short_when_more_votes(self):
        """More LONG votes than SHORT should trigger LONG."""
        engine = make_engine()
        engine.set_orb_range(high=74.0, low=72.0)
        # Gold-lead LONG, ORB LONG, sentiment LONG
        state = make_state(
            silver_price=74.5,
            gold_5min_change=0.8,
            hour_cet=15,
            news_spike=True,
            headline_sentiment="positive",
        )
        decision = engine.tick(state)
        assert decision["action"] == "BUY"
        assert decision["direction"] == "LONG"

    def test_confidence_scales_with_vote_count(self):
        """Confidence = min(1.0, num_votes / 4.0)."""
        engine = make_engine()
        engine.set_orb_range(high=74.0, low=72.0)
        state = make_state(
            silver_price=74.5,
            gold_5min_change=0.8,
            hour_cet=15,
        )
        decision = engine.tick(state)
        assert decision["action"] == "BUY"
        # 2 votes => confidence = 2/4 = 0.5
        assert decision["confidence"] == 0.5


# ===================================================================
# 2. Layer 2 vote weight -- counts as 2
# ===================================================================


class TestLayer2VoteWeight:
    """Layer 2 vote contributes both 'layer2' and 'layer2_w' entries."""

    def test_layer2_bullish_adds_two_votes(self):
        """A bullish Layer 2 should count as 2 LONG votes (enough alone)."""
        engine = make_engine()
        ts = datetime.now(UTC).isoformat()
        state = make_state(
            layer2_outlook="bullish",
            layer2_conviction=0.7,
            layer2_ts=ts,
        )
        decision = engine.tick(state)
        # layer2 + layer2_w = 2 votes >= MIN_VOTES
        assert decision["action"] == "BUY"
        assert decision["direction"] == "LONG"
        assert "layer2" in decision["tactics_agreed"]
        assert "layer2_w" in decision["tactics_agreed"]

    def test_layer2_bearish_adds_two_votes(self):
        engine = make_engine()
        ts = datetime.now(UTC).isoformat()
        state = make_state(
            layer2_outlook="bearish",
            layer2_conviction=0.6,
            layer2_ts=ts,
        )
        decision = engine.tick(state)
        assert decision["action"] == "BUY"
        assert decision["direction"] == "SHORT"
        assert "layer2" in decision["tactics_agreed"]
        assert "layer2_w" in decision["tactics_agreed"]

    def test_layer2_low_conviction_ignored(self):
        """Conviction < 0.4 should discard Layer 2 vote."""
        engine = make_engine()
        ts = datetime.now(UTC).isoformat()
        state = make_state(
            layer2_outlook="bullish",
            layer2_conviction=0.3,  # below 0.4 threshold
            layer2_ts=ts,
        )
        decision = engine.tick(state)
        assert decision["action"] == "HOLD"

    def test_layer2_empty_outlook_ignored(self):
        engine = make_engine()
        state = make_state(
            layer2_outlook="",
            layer2_conviction=0.9,
            layer2_ts=datetime.now(UTC).isoformat(),
        )
        decision = engine.tick(state)
        assert decision["action"] == "HOLD"


# ===================================================================
# 3. Layer 2 stale data -- >4h old should be ignored
# ===================================================================


class TestLayer2Staleness:
    """Votes older than 4 hours should be discarded."""

    def test_fresh_layer2_accepted(self):
        """Timestamp within 4h is accepted."""
        engine = make_engine()
        ts = datetime.now(UTC).isoformat()
        state = make_state(
            layer2_outlook="bullish",
            layer2_conviction=0.8,
            layer2_ts=ts,
        )
        decision = engine.tick(state)
        assert decision["action"] == "BUY"

    def test_stale_layer2_ignored(self):
        """Timestamp older than 4h should be ignored."""
        engine = make_engine()
        old_ts = (datetime.now(UTC) - timedelta(hours=5)).isoformat()
        state = make_state(
            layer2_outlook="bullish",
            layer2_conviction=0.9,
            layer2_ts=old_ts,
        )
        decision = engine.tick(state)
        assert decision["action"] == "HOLD"

    def test_exactly_4h_old_accepted(self):
        """Timestamp exactly 4h old should still be accepted (>4 check)."""
        engine = make_engine()
        ts = (datetime.now(UTC) - timedelta(hours=4)).isoformat()
        state = make_state(
            layer2_outlook="bullish",
            layer2_conviction=0.8,
            layer2_ts=ts,
        )
        decision = engine.tick(state)
        # Exactly 4h is not > 4h, so should be accepted
        assert decision["action"] == "BUY"

    def test_layer2_bad_timestamp_still_works(self):
        """Unparseable timestamp should not crash (exception caught)."""
        engine = make_engine()
        state = make_state(
            layer2_outlook="bullish",
            layer2_conviction=0.8,
            layer2_ts="not-a-timestamp",
        )
        decision = engine.tick(state)
        # Bad timestamp => exception caught => staleness check passes
        assert decision["action"] == "BUY"


# ===================================================================
# 4. MC-informed TP/SL -- dynamic levels from price bands
# ===================================================================


class TestMCInformedTPSL:
    """Dynamic TP/SL from Monte Carlo bands instead of fixed +2%/-3%."""

    def test_long_tp_uses_mc_75_percentile(self):
        """LONG TP from mc_bands_1d['75']."""
        t = 1000000.0
        engine = make_engine(t=t)
        entry_price = 75.0
        _put_position(engine, direction="LONG", entry_underlying=entry_price, entry_ts=t - 600)
        # MC says 75th percentile is 76.2 => TP at (76.2-75)/75*100 = 1.6%
        # But min TP is 1.0%, so 1.6% is valid.
        # Current price at 76.3 > entry * (1+1.6%) => triggers TP
        state = make_state(
            silver_price=76.3,
            mc_bands_1d={"5": 73.0, "25": 74.0, "75": 76.2, "95": 77.5},
        )
        decision = engine.tick(state)
        assert decision["action"] == "SELL"
        assert "TP" in decision["exit_reason"]
        assert "(MC)" in decision["exit_reason"]

    def test_long_sl_uses_mc_5_percentile(self):
        """LONG SL from mc_bands_1d['5']."""
        t = 1000000.0
        engine = make_engine(t=t)
        entry_price = 75.0
        _put_position(engine, direction="LONG", entry_underlying=entry_price, entry_ts=t - 600)
        # MC says 5th percentile is 73.5 => SL at (73.5-75)/75*100 = -2.0%
        # Price drops below: 73.4
        state = make_state(
            silver_price=73.4,
            mc_bands_1d={"5": 73.5, "25": 74.0, "75": 76.5, "95": 78.0},
        )
        decision = engine.tick(state)
        assert decision["action"] == "SELL"
        assert "SL" in decision["exit_reason"]

    def test_short_tp_uses_mc_25_percentile(self):
        """SHORT TP from mc_bands_1d['25']."""
        t = 1000000.0
        engine = make_engine(t=t)
        entry_price = 75.0
        _put_position(engine, direction="SHORT", entry_underlying=entry_price, entry_ts=t - 600)
        # MC says 25th percentile is 73.5 => TP at (75-73.5)/75*100 = 2.0%
        # Price at 73.4 => move_pct = (75-73.4)/75*100 = 2.13% > 2.0%
        state = make_state(
            silver_price=73.4,
            mc_bands_1d={"5": 72.0, "25": 73.5, "75": 76.5, "95": 78.0},
        )
        decision = engine.tick(state)
        assert decision["action"] == "SELL"
        assert "TP" in decision["exit_reason"]

    def test_short_sl_uses_mc_95_percentile(self):
        """SHORT SL from mc_bands_1d['95']."""
        t = 1000000.0
        engine = make_engine(t=t)
        entry_price = 75.0
        _put_position(engine, direction="SHORT", entry_underlying=entry_price, entry_ts=t - 600)
        # MC says 95th percentile is 76.5 => SL at (75-76.5)/75*100 = -2.0%
        # Price at 76.6 => move_pct = (75-76.6)/75*100 = -2.13% < -2.0%
        state = make_state(
            silver_price=76.6,
            mc_bands_1d={"5": 72.0, "25": 73.5, "75": 76.5, "95": 76.5},
        )
        decision = engine.tick(state)
        assert decision["action"] == "SELL"
        assert "SL" in decision["exit_reason"]

    def test_fallback_to_fixed_when_no_mc_bands(self):
        """Without mc_bands_1d, use fixed EXIT_TP_PCT / EXIT_SL_PCT."""
        t = 1000000.0
        engine = make_engine(t=t)
        entry_price = 75.0
        _put_position(engine, direction="LONG", entry_underlying=entry_price, entry_ts=t - 600)
        # Fixed TP is +2% => 76.5 needed. Price at 76.6 triggers it.
        state = make_state(silver_price=76.6, mc_bands_1d={})
        decision = engine.tick(state)
        assert decision["action"] == "SELL"
        assert "TP" in decision["exit_reason"]
        # No MC tag
        assert "(MC)" not in decision["exit_reason"]

    def test_mc_tp_enforces_minimum_1pct(self):
        """MC TP threshold has a min of 1.0% to cover friction."""
        t = 1000000.0
        engine = make_engine(t=t)
        entry_price = 75.0
        _put_position(engine, direction="LONG", entry_underlying=entry_price, entry_ts=t - 600)
        # MC 75th is 75.3 => raw TP = 0.4% which is below 1.0% min
        # So effective TP = 1.0% => need price >= 75.75
        state = make_state(
            silver_price=75.5,  # +0.67%, below 1.0% threshold
            mc_bands_1d={"5": 73.0, "25": 74.0, "75": 75.3, "95": 77.0},
        )
        decision = engine.tick(state)
        assert decision["action"] == "HOLD"  # Not enough for TP

    def test_mc_sl_enforces_max_neg1pct(self):
        """MC SL threshold has a max of -1.0% (min losses)."""
        t = 1000000.0
        engine = make_engine(t=t)
        entry_price = 75.0
        _put_position(engine, direction="LONG", entry_underlying=entry_price, entry_ts=t - 600)
        # MC 5th is 74.8 => raw SL = -0.27% which is above -1.0% min
        # So effective SL = -1.0% => triggers at 74.25
        state = make_state(
            silver_price=74.5,  # -0.67%, above -1.0% threshold
            mc_bands_1d={"5": 74.8, "25": 74.9, "75": 76.0, "95": 77.0},
        )
        decision = engine.tick(state)
        assert decision["action"] == "HOLD"  # Not deep enough for SL


# ===================================================================
# 5. Auto-disable at 21:55 CET
# ===================================================================


class TestAutoDisable:
    """Engine returns HOLD when hour_cet >= 22 or (hour==21, min>=55)."""

    def test_hold_after_2200(self):
        engine = make_engine()
        state = make_state(hour_cet=22, minute_cet=0)
        decision = engine.tick(state)
        assert decision["action"] == "HOLD"
        assert "market closed" in decision["reason"]

    def test_hold_at_2155(self):
        engine = make_engine()
        state = make_state(hour_cet=21, minute_cet=55)
        decision = engine.tick(state)
        assert decision["action"] == "HOLD"
        assert "market closed" in decision["reason"]

    def test_hold_at_2159(self):
        engine = make_engine()
        state = make_state(hour_cet=21, minute_cet=59)
        decision = engine.tick(state)
        assert decision["action"] == "HOLD"

    def test_active_at_2154(self):
        """21:54 is still within trading hours -- should not auto-disable."""
        engine = make_engine()
        state = make_state(hour_cet=21, minute_cet=54)
        decision = engine.tick(state)
        # Should not say market closed
        assert "market closed" not in decision.get("reason", "")

    def test_active_at_2100(self):
        engine = make_engine()
        state = make_state(hour_cet=21, minute_cet=0)
        decision = engine.tick(state)
        assert "market closed" not in decision.get("reason", "")

    def test_hold_at_2300(self):
        engine = make_engine()
        state = make_state(hour_cet=23, minute_cet=0)
        decision = engine.tick(state)
        assert decision["action"] == "HOLD"


# ===================================================================
# 6. Session end forces SELL if holding
# ===================================================================


class TestSessionEndForcesSell:
    """At 21:55, if the engine has an open position, it must force-sell."""

    def test_force_sell_long_at_session_end(self):
        t = 1000000.0
        engine = make_engine(t=t)
        _put_position(engine, direction="LONG", entry_ts=t - 3600)
        state = make_state(hour_cet=21, minute_cet=55)
        decision = engine.tick(state)
        assert decision["action"] == "SELL"
        assert decision["exit_reason"] == "SESSION_END"
        assert decision["direction"] == "LONG"
        assert decision["volume"] == 150  # from _put_position default

    def test_force_sell_short_at_session_end(self):
        t = 1000000.0
        engine = make_engine(t=t)
        _put_position(engine, direction="SHORT", entry_ts=t - 1800)
        state = make_state(hour_cet=22, minute_cet=5)
        decision = engine.tick(state)
        assert decision["action"] == "SELL"
        assert decision["exit_reason"] == "SESSION_END"
        assert decision["direction"] == "SHORT"

    def test_force_sell_calculates_hold_minutes(self):
        t = 1000000.0
        engine = make_engine(t=t)
        _put_position(engine, direction="LONG", entry_ts=t - 1200)  # 20 min ago
        state = make_state(hour_cet=21, minute_cet=57)
        decision = engine.tick(state)
        assert decision["action"] == "SELL"
        assert decision["hold_minutes"] == 20.0

    def test_session_end_returns_instrument_ob(self):
        t = 1000000.0
        engine = make_engine(t=t)
        _put_position(engine, direction="LONG", entry_ts=t - 600)
        state = make_state(hour_cet=21, minute_cet=55)
        decision = engine.tick(state)
        assert decision["instrument_ob"] == LONG_OB


# ===================================================================
# 7. Cooldown enforcement
# ===================================================================


class TestCooldownEnforcement:
    """5 min between trades normally, 2 min for high-conviction exits."""

    def test_normal_cooldown_blocks_entry(self):
        """After a normal exit, a 300s cooldown blocks new entries."""
        t = 1000000.0
        engine = make_engine(t=t)
        # Set up a previous trade cooldown
        engine._last_trade_ts = t - 100  # 100 seconds ago
        engine._cooldown_seconds = COOLDOWN_NORMAL  # 300s

        engine.set_orb_range(high=74.0, low=72.0)
        state = make_state(
            silver_price=74.5,
            gold_5min_change=0.8,
            hour_cet=15,
        )
        decision = engine.tick(state)
        assert decision["action"] == "HOLD"
        assert "cooldown" in decision["reason"]

    def test_normal_cooldown_expires_allows_entry(self):
        """After cooldown expires, entry is allowed."""
        t = 1000000.0
        engine = make_engine(t=t)
        engine._last_trade_ts = t - 400  # 400 seconds ago, > 300
        engine._cooldown_seconds = COOLDOWN_NORMAL

        engine.set_orb_range(high=74.0, low=72.0)
        state = make_state(
            silver_price=74.5,
            gold_5min_change=0.8,
            hour_cet=15,
        )
        decision = engine.tick(state)
        assert decision["action"] == "BUY"

    def test_high_conviction_exit_sets_shorter_cooldown(self):
        """A high-conviction exit (e.g., RSI overbought) should set 120s cooldown."""
        t = 1000000.0
        engine = make_engine(t=t)
        _put_position(engine, direction="LONG", entry_underlying=75.0, entry_ts=t - 600)
        # RSI > 70 triggers high-conviction exit
        state = make_state(silver_price=75.5, rsi=72)
        decision = engine.tick(state)
        assert decision["action"] == "SELL"
        assert engine._cooldown_seconds == COOLDOWN_HIGH_CONV

    def test_normal_exit_sets_longer_cooldown(self):
        """A non-conviction exit (e.g., TP) should set 300s cooldown."""
        t = 1000000.0
        engine = make_engine(t=t)
        _put_position(engine, direction="LONG", entry_underlying=75.0, entry_ts=t - 600)
        # TP at +2% underlying (no MC bands)
        state = make_state(silver_price=76.6, mc_bands_1d={})
        decision = engine.tick(state)
        assert decision["action"] == "SELL"
        assert engine._cooldown_seconds == COOLDOWN_NORMAL

    def test_cooldown_remaining_in_reason(self):
        """HOLD reason should include remaining cooldown seconds."""
        t = 1000000.0
        engine = make_engine(t=t)
        engine._last_trade_ts = t - 200
        engine._cooldown_seconds = COOLDOWN_NORMAL  # 300s => 100s remaining
        state = make_state()
        decision = engine.tick(state)
        assert decision["action"] == "HOLD"
        assert "100" in decision["reason"]


# ===================================================================
# 8. Mode detection -- auto-switch after 3 losses
# ===================================================================


class TestModeDetection:
    """Auto-switch from momentum to straddle after 3 consecutive losses."""

    def test_starts_in_momentum(self):
        engine = make_engine()
        assert engine.mode == "momentum"

    def test_three_consecutive_losses_switches_to_straddle(self):
        engine = make_engine()
        # Simulate 3 consecutive losses via confirm_exit
        for _ in range(3):
            _put_position(engine, direction="LONG")
            engine.confirm_exit(pnl=-50.0)
        assert engine.mode == "straddle"
        assert engine.consecutive_losses == 3

    def test_two_losses_stay_momentum(self):
        engine = make_engine()
        for _ in range(2):
            _put_position(engine, direction="LONG")
            engine.confirm_exit(pnl=-50.0)
        assert engine.mode == "momentum"
        assert engine.consecutive_losses == 2

    def test_win_resets_consecutive_losses(self):
        engine = make_engine()
        _put_position(engine, direction="LONG")
        engine.confirm_exit(pnl=-50.0)
        _put_position(engine, direction="LONG")
        engine.confirm_exit(pnl=-50.0)
        assert engine.consecutive_losses == 2

        _put_position(engine, direction="LONG")
        engine.confirm_exit(pnl=100.0)  # win
        assert engine.consecutive_losses == 0
        assert engine.mode == "momentum"  # still momentum

    def test_loss_after_win_restarts_count(self):
        engine = make_engine()
        _put_position(engine, direction="LONG")
        engine.confirm_exit(pnl=100.0)  # win
        _put_position(engine, direction="LONG")
        engine.confirm_exit(pnl=-50.0)  # loss
        assert engine.consecutive_losses == 1

    def test_already_straddle_stays_straddle_on_losses(self):
        engine = make_engine()
        engine.set_mode("straddle", floor=73.0, ceil=77.0)
        for _ in range(4):
            _put_position(engine, direction="LONG")
            engine.confirm_exit(pnl=-50.0)
        assert engine.mode == "straddle"  # no double-switch

    def test_set_mode_explicitly(self):
        engine = make_engine()
        engine.set_mode("straddle", floor=73.0, ceil=77.0)
        assert engine.mode == "straddle"
        assert engine.straddle_floor == 73.0
        assert engine.straddle_ceil == 77.0
        assert engine.straddle_bull_filled is False
        assert engine.straddle_bear_filled is False


# ===================================================================
# 9. Position tracking -- confirm_entry / confirm_exit
# ===================================================================


class TestPositionTracking:
    """confirm_entry and confirm_exit update state correctly."""

    def test_confirm_entry_long(self):
        engine = make_engine()
        assert engine.position is None
        engine.confirm_entry("LONG", 7.5, 150, 75.0)
        assert engine.position is not None
        assert engine.position["direction"] == "LONG"
        assert engine.position["entry_cert"] == 7.5
        assert engine.position["volume"] == 150
        assert engine.position["entry_underlying"] == 75.0
        assert engine.position["ob_id"] == LONG_OB

    def test_confirm_entry_short(self):
        engine = make_engine()
        engine.confirm_entry("SHORT", 2.5, 400, 75.0)
        assert engine.position["direction"] == "SHORT"
        assert engine.position["ob_id"] == SHORT_OB

    def test_confirm_entry_sets_last_trade_ts(self):
        t = 1000000.0
        engine = make_engine(t=t)
        engine.confirm_entry("LONG", 7.0, 150, 75.0)
        assert engine._last_trade_ts == t

    def test_confirm_entry_resets_metals_disagree(self):
        engine = make_engine()
        engine.metals_disagree_count = 5
        engine.confirm_entry("LONG", 7.0, 150, 75.0)
        assert engine.metals_disagree_count == 0

    def test_confirm_exit_winning_trade(self):
        engine = make_engine()
        engine.confirm_entry("LONG", 7.0, 150, 75.0)
        engine.confirm_exit(pnl=200.0)
        assert engine.position is None
        assert engine.session_pnl == 200.0
        assert engine.trade_count == 1
        assert engine.win_count == 1
        assert engine.loss_count == 0
        assert engine.consecutive_losses == 0

    def test_confirm_exit_losing_trade(self):
        engine = make_engine()
        engine.confirm_entry("LONG", 7.0, 150, 75.0)
        engine.confirm_exit(pnl=-100.0)
        assert engine.position is None
        assert engine.session_pnl == -100.0
        assert engine.trade_count == 1
        assert engine.win_count == 0
        assert engine.loss_count == 1
        assert engine.consecutive_losses == 1

    def test_confirm_exit_no_position_is_noop(self):
        engine = make_engine()
        engine.confirm_exit(pnl=100.0)  # no position set
        assert engine.trade_count == 0
        assert engine.session_pnl == 0.0

    def test_multiple_trades_accumulate(self):
        engine = make_engine()
        engine.confirm_entry("LONG", 7.0, 150, 75.0)
        engine.confirm_exit(pnl=100.0)
        engine.confirm_entry("SHORT", 2.5, 400, 75.0)
        engine.confirm_exit(pnl=-50.0)
        engine.confirm_entry("LONG", 7.2, 140, 75.5)
        engine.confirm_exit(pnl=75.0)
        assert engine.trade_count == 3
        assert engine.win_count == 2
        assert engine.loss_count == 1
        assert engine.session_pnl == 125.0

    def test_has_position_property(self):
        engine = make_engine()
        assert engine.has_position is False
        engine.confirm_entry("LONG", 7.0, 150, 75.0)
        assert engine.has_position is True
        engine.confirm_exit(pnl=0.0)
        assert engine.has_position is False

    def test_confirm_entry_in_straddle_mode_sets_filled_flags(self):
        engine = make_engine()
        engine.set_mode("straddle", floor=73.0, ceil=77.0)
        engine.confirm_entry("LONG", 7.0, 150, 75.0)
        assert engine.straddle_bull_filled is True
        assert engine.straddle_bear_filled is False

        engine.confirm_exit(pnl=50.0)
        engine.confirm_entry("SHORT", 2.5, 400, 75.0)
        assert engine.straddle_bear_filled is True


# ===================================================================
# 10. State persistence -- to_dict / from_dict round-trip
# ===================================================================


class TestStatePersistence:
    """to_dict and from_dict should round-trip all state correctly."""

    def test_roundtrip_default_state(self):
        engine = make_engine()
        d = engine.to_dict()
        engine2 = make_engine()
        engine2.from_dict(d)
        assert engine2.to_dict() == d

    def test_roundtrip_with_position(self):
        t = 1000000.0
        engine = make_engine(t=t)
        engine.confirm_entry("LONG", 7.5, 150, 75.0)
        d = engine.to_dict()

        engine2 = make_engine(t=t)
        engine2.from_dict(d)
        assert engine2.position is not None
        assert engine2.position["direction"] == "LONG"
        assert engine2.position["entry_cert"] == 7.5

    def test_roundtrip_after_trades(self):
        engine = make_engine()
        engine.confirm_entry("LONG", 7.0, 150, 75.0)
        engine.confirm_exit(pnl=100.0)
        engine.confirm_entry("SHORT", 2.5, 400, 75.0)
        engine.confirm_exit(pnl=-50.0)
        engine.confirm_entry("LONG", 7.0, 150, 75.0)
        engine.confirm_exit(pnl=-30.0)

        d = engine.to_dict()
        engine2 = make_engine()
        engine2.from_dict(d)

        assert engine2.session_pnl == 20.0
        assert engine2.trade_count == 3
        assert engine2.win_count == 1
        assert engine2.loss_count == 2
        assert engine2.consecutive_losses == 2
        assert engine2.position is None

    def test_roundtrip_straddle_mode(self):
        engine = make_engine()
        engine.set_mode("straddle", floor=73.0, ceil=77.0)
        engine.straddle_bull_filled = True
        d = engine.to_dict()

        engine2 = make_engine()
        engine2.from_dict(d)
        assert engine2.mode == "straddle"
        assert engine2.straddle_floor == 73.0
        assert engine2.straddle_ceil == 77.0
        assert engine2.straddle_bull_filled is True
        assert engine2.straddle_bear_filled is False

    def test_roundtrip_orb_range(self):
        engine = make_engine()
        engine.set_orb_range(high=76.0, low=73.5)
        d = engine.to_dict()

        engine2 = make_engine()
        engine2.from_dict(d)
        assert engine2.orb_range_formed is True
        assert engine2.orb_range_high == 76.0
        assert engine2.orb_range_low == 73.5

    def test_roundtrip_mc_history(self):
        engine = make_engine()
        engine._mc_history = [0.6, 0.7, 0.8]
        d = engine.to_dict()

        engine2 = make_engine()
        engine2.from_dict(d)
        assert engine2._mc_history == [0.6, 0.7, 0.8]

    def test_roundtrip_cooldown(self):
        t = 1000000.0
        engine = make_engine(t=t)
        engine._last_trade_ts = t - 100
        engine._cooldown_seconds = COOLDOWN_NORMAL
        d = engine.to_dict()

        engine2 = make_engine(t=t)
        engine2.from_dict(d)
        assert engine2._last_trade_ts == t - 100
        assert engine2._cooldown_seconds == COOLDOWN_NORMAL

    def test_from_dict_handles_missing_keys(self):
        """from_dict should use defaults for missing keys."""
        engine = make_engine()
        engine.from_dict({})  # empty dict
        assert engine.mode == "momentum"
        assert engine.session_pnl == 0
        assert engine.position is None
        assert engine._mc_history == []

    def test_to_dict_keys(self):
        """Verify all expected keys are in the serialized dict."""
        engine = make_engine()
        d = engine.to_dict()
        expected_keys = {
            "position", "metals_disagree_count", "session_pnl",
            "trade_count", "win_count", "loss_count", "mode",
            "straddle_floor", "straddle_ceil", "straddle_bull_filled",
            "straddle_bear_filled", "consecutive_losses", "mc_history",
            "last_trade_ts", "cooldown_seconds", "orb_range_high",
            "orb_range_low", "orb_range_formed",
        }
        assert set(d.keys()) == expected_keys


# ===================================================================
# Additional edge cases and integration scenarios
# ===================================================================


class TestExitRules:
    """Exercise specific exit conditions."""

    def test_rsi_overbought_exit_long(self):
        t = 1000000.0
        engine = make_engine(t=t)
        _put_position(engine, direction="LONG", entry_underlying=75.0, entry_ts=t - 600)
        state = make_state(silver_price=75.5, rsi=72)
        decision = engine.tick(state)
        assert decision["action"] == "SELL"
        assert "RSI" in decision["exit_reason"]

    def test_rsi_oversold_exit_short(self):
        t = 1000000.0
        engine = make_engine(t=t)
        _put_position(engine, direction="SHORT", entry_underlying=75.0, entry_ts=t - 600)
        state = make_state(silver_price=74.5, rsi=28)
        decision = engine.tick(state)
        assert decision["action"] == "SELL"
        assert "RSI" in decision["exit_reason"]

    def test_combo_exit_rsi_plus_mc(self):
        """RSI > 62 and MC < 0.35 triggers combo exit for LONG."""
        t = 1000000.0
        engine = make_engine(t=t)
        _put_position(engine, direction="LONG", entry_underlying=75.0, entry_ts=t - 600)
        state = make_state(silver_price=75.3, rsi=65, mc_p_up=0.30)
        decision = engine.tick(state)
        assert decision["action"] == "SELL"
        assert "COMB" in decision["exit_reason"]

    def test_signal_flip_exit_long(self):
        """Sell count exceeding buy count by > 4 triggers exit."""
        t = 1000000.0
        engine = make_engine(t=t)
        _put_position(engine, direction="LONG", entry_underlying=75.0, entry_ts=t - 600)
        state = make_state(silver_price=75.1, signal_buy_count=2, signal_sell_count=7)
        decision = engine.tick(state)
        assert decision["action"] == "SELL"
        assert "flip" in decision["exit_reason"]

    def test_signal_flip_exit_short(self):
        t = 1000000.0
        engine = make_engine(t=t)
        _put_position(engine, direction="SHORT", entry_underlying=75.0, entry_ts=t - 600)
        state = make_state(silver_price=74.9, signal_buy_count=7, signal_sell_count=2)
        decision = engine.tick(state)
        assert decision["action"] == "SELL"
        assert "flip" in decision["exit_reason"]

    def test_metals_disagree_exit(self):
        """N consecutive metals disagrees triggers exit (15 after 2026-04-07 live test)."""
        from data.fish_engine import EXIT_METALS_DISAGREE_COUNT
        t = 1000000.0
        engine = make_engine(t=t)
        _put_position(engine, direction="LONG", entry_underlying=75.0, entry_ts=t - 600)
        for _ in range(EXIT_METALS_DISAGREE_COUNT):
            state = make_state(silver_price=75.1, metals_action="SELL")
            decision = engine.tick(state)
        assert decision["action"] == "SELL"
        assert "MD" in decision["exit_reason"]

    def test_max_hold_time_exit(self):
        """Position held > MAX_HOLD_NORMAL (120 min) triggers exit."""
        t = 1000000.0
        engine = make_engine(t=t)
        _put_position(engine, direction="LONG", entry_underlying=75.0,
                       entry_ts=t - 7300)  # ~121.7 min ago
        state = make_state(silver_price=75.1)
        decision = engine.tick(state)
        assert decision["action"] == "SELL"
        assert "hold" in decision["exit_reason"].lower()

    def test_max_hold_shorter_near_event(self):
        """Event proximity reduces max hold to 60 min."""
        t = 1000000.0
        engine = make_engine(t=t)
        _put_position(engine, direction="LONG", entry_underlying=75.0,
                       entry_ts=t - 3700)  # ~61.7 min ago
        state = make_state(silver_price=75.1, event_hours=12)
        decision = engine.tick(state)
        assert decision["action"] == "SELL"

    def test_holding_within_limits_gives_hold(self):
        """Position within normal bounds should HOLD."""
        t = 1000000.0
        engine = make_engine(t=t)
        _put_position(engine, direction="LONG", entry_underlying=75.0, entry_ts=t - 600)
        state = make_state(silver_price=75.5, rsi=55, mc_p_up=0.55)
        decision = engine.tick(state)
        assert decision["action"] == "HOLD"
        assert decision["direction"] == "LONG"
        assert "holding" in decision["reason"]


class TestSpreadGating:
    """Spread > MAX_SPREAD_PCT should block entry."""

    def test_wide_spread_blocks_entry(self):
        engine = make_engine()
        engine.set_orb_range(high=74.0, low=72.0)
        state = make_state(
            silver_price=74.5,
            gold_5min_change=0.8,
            hour_cet=15,
            spread_pct=1.5,  # > 1.0%
        )
        decision = engine.tick(state)
        assert decision["action"] == "HOLD"
        assert "spread" in decision["reason"]

    def test_narrow_spread_allows_entry(self):
        engine = make_engine()
        engine.set_orb_range(high=74.0, low=72.0)
        state = make_state(
            silver_price=74.5,
            gold_5min_change=0.8,
            hour_cet=15,
            spread_pct=0.5,
        )
        decision = engine.tick(state)
        assert decision["action"] == "BUY"


class TestTradeGuard:
    """trade_guard_ok=False should block entry."""

    def test_trade_guard_blocks_entry(self):
        engine = make_engine()
        engine.set_orb_range(high=74.0, low=72.0)
        state = make_state(
            silver_price=74.5,
            gold_5min_change=0.8,
            hour_cet=15,
            trade_guard_ok=False,
        )
        decision = engine.tick(state)
        assert decision["action"] == "HOLD"
        assert "trade guard" in decision["reason"]


class TestTemporalPatterns:
    """Tactic 5: Temporal pattern voting."""

    def test_temporal_pattern_votes(self):
        patterns = [
            {"day": 1, "hour_cet": 15, "direction": "BULL", "probability": 75},
        ]
        engine = make_engine(temporal_patterns=patterns)
        engine.set_orb_range(high=74.0, low=72.0)
        state = make_state(
            silver_price=74.5,
            day_of_week=1,
            hour_cet=15,
        )
        decision = engine.tick(state)
        # temporal + orb = 2 votes
        assert decision["action"] == "BUY"
        assert "temporal" in decision["tactics_agreed"]

    def test_low_probability_pattern_ignored(self):
        patterns = [
            {"day": 1, "hour_cet": 15, "direction": "BULL", "probability": 60},
        ]
        engine = make_engine(temporal_patterns=patterns)
        state = make_state(day_of_week=1, hour_cet=15)
        decision = engine.tick(state)
        # probability < 68, so temporal doesn't vote
        assert decision["action"] == "HOLD"


class TestSentimentVoting:
    """Tactic 6: News spike + sentiment direction."""

    def test_positive_sentiment_votes_long(self):
        engine = make_engine()
        engine.set_orb_range(high=74.0, low=72.0)
        state = make_state(
            silver_price=74.5,
            hour_cet=15,
            news_spike=True,
            headline_sentiment="positive",
        )
        decision = engine.tick(state)
        assert decision["action"] == "BUY"
        assert "sentiment" in decision["tactics_agreed"]

    def test_no_news_spike_no_vote(self):
        engine = make_engine()
        state = make_state(
            news_spike=False,
            headline_sentiment="positive",
        )
        decision = engine.tick(state)
        # sentiment tactic requires news_spike=True
        assert "sentiment" not in decision.get("tactics_agreed", [])


class TestMomentumTactic:
    """Tactic 1: Momentum requires signal+metals agree + MC stable 2 checks."""

    def test_momentum_long(self):
        engine = make_engine()
        # Need 2 ticks to build MC history
        state1 = make_state(
            signal_action="BUY",
            metals_action="BUY",
            mc_p_up=0.75,
            gold_5min_change=0.8,
            hour_cet=15,
        )
        engine.tick(state1)  # first MC reading

        state2 = make_state(
            signal_action="BUY",
            metals_action="BUY",
            mc_p_up=0.75,
            gold_5min_change=0.8,
            hour_cet=15,
        )
        decision = engine.tick(state2)
        assert decision["action"] == "BUY"
        assert "momentum" in decision["tactics_agreed"]

    def test_momentum_needs_two_mc_readings(self):
        """Single MC reading should not trigger momentum."""
        engine = make_engine()
        state = make_state(
            signal_action="BUY",
            metals_action="BUY",
            mc_p_up=0.75,
            gold_5min_change=0.8,
            hour_cet=15,
        )
        decision = engine.tick(state)
        # Only 1 MC reading, momentum won't vote
        assert "momentum" not in decision.get("tactics_agreed", [])


class TestStraddleMode:
    """Tactic 2: Straddle mode entry at floor/ceiling."""

    def test_straddle_long_at_floor(self):
        """Straddle floor hit + gold-lead LONG = 2 votes => BUY LONG."""
        engine = make_engine()
        engine.set_mode("straddle", floor=73.0, ceil=77.0)
        state = make_state(
            silver_price=72.5,  # below floor => straddle votes LONG
            gold_5min_change=0.8,  # gold-lead also votes LONG
            hour_cet=15,
        )
        decision = engine.tick(state)
        assert decision["action"] == "BUY"
        assert decision["direction"] == "LONG"
        assert "straddle" in decision["tactics_agreed"]

    def test_straddle_short_at_ceiling(self):
        """Straddle ceiling hit + gold-lead SHORT = 2 votes => BUY SHORT."""
        engine = make_engine()
        engine.set_mode("straddle", floor=73.0, ceil=77.0)
        state = make_state(
            silver_price=77.5,  # above ceiling => straddle votes SHORT
            gold_5min_change=-0.8,  # gold-lead also votes SHORT
            hour_cet=15,
        )
        decision = engine.tick(state)
        assert decision["action"] == "BUY"
        assert decision["direction"] == "SHORT"
        assert "straddle" in decision["tactics_agreed"]

    def test_straddle_no_vote_after_cancel_time(self):
        engine = make_engine()
        engine.set_mode("straddle", floor=73.0, ceil=77.0)
        state = make_state(
            silver_price=72.5,
            hour_cet=18,
            minute_cet=55,
        )
        decision = engine.tick(state)
        # straddle should not vote after cancel time
        assert "straddle" not in decision.get("tactics_agreed", [])

    def test_straddle_bull_already_filled_no_long(self):
        engine = make_engine()
        engine.set_mode("straddle", floor=73.0, ceil=77.0)
        engine.straddle_bull_filled = True
        state = make_state(silver_price=72.5, hour_cet=15)
        decision = engine.tick(state)
        assert "straddle" not in decision.get("tactics_agreed", [])


class TestGoldLeadTactic:
    """Tactic 3: Gold-leads-silver."""

    def test_gold_lead_long(self):
        engine = make_engine()
        engine.set_orb_range(high=74.0, low=72.0)
        state = make_state(
            silver_price=74.5,
            gold_5min_change=0.8,
            hour_cet=15,
        )
        decision = engine.tick(state)
        assert "gold_lead" in decision["tactics_agreed"]

    def test_gold_lead_short(self):
        engine = make_engine()
        engine.set_orb_range(high=76.0, low=74.0)
        state = make_state(
            silver_price=73.5,
            gold_5min_change=-0.8,
            hour_cet=15,
        )
        decision = engine.tick(state)
        assert "gold_lead" in decision["tactics_agreed"]

    def test_gold_lead_dead_zone_low_confidence_blocks(self):
        """In dead zone, gold lead requires confidence >= 0.7."""
        engine = make_engine()
        state = make_state(
            gold_5min_change=0.55,  # confidence = 0.55 < 0.7
            hour_cet=12,  # dead zone (10-14)
        )
        decision = engine.tick(state)
        assert "gold_lead" not in decision.get("tactics_agreed", [])

    def test_gold_lead_dead_zone_high_confidence_passes(self):
        """In dead zone with confidence >= 0.7, gold lead still votes."""
        engine = make_engine()
        engine.set_orb_range(high=74.0, low=72.0)
        state = make_state(
            silver_price=74.5,
            gold_5min_change=0.9,  # confidence = 0.9 >= 0.7
            hour_cet=12,  # dead zone
        )
        decision = engine.tick(state)
        assert "gold_lead" in decision["tactics_agreed"]


class TestORBTactic:
    """Tactic 4: Opening Range Breakout."""

    def test_orb_long_above_high(self):
        engine = make_engine()
        engine.set_orb_range(high=74.0, low=72.0)
        state = make_state(silver_price=74.5, gold_5min_change=0.8, hour_cet=15)
        decision = engine.tick(state)
        assert "orb" in decision["tactics_agreed"]

    def test_orb_short_below_low(self):
        engine = make_engine()
        engine.set_orb_range(high=76.0, low=74.0)
        state = make_state(silver_price=73.5, gold_5min_change=-0.8, hour_cet=15)
        decision = engine.tick(state)
        assert "orb" in decision["tactics_agreed"]

    def test_orb_no_vote_within_range(self):
        engine = make_engine()
        engine.set_orb_range(high=76.0, low=72.0)
        state = make_state(silver_price=74.0)
        decision = engine.tick(state)
        assert "orb" not in decision.get("tactics_agreed", [])

    def test_orb_not_formed_no_vote(self):
        engine = make_engine()
        # don't call set_orb_range
        state = make_state(silver_price=80.0)
        decision = engine.tick(state)
        assert "orb" not in decision.get("tactics_agreed", [])

    def test_set_orb_range_validation(self):
        engine = make_engine()
        engine.set_orb_range(high=72.0, low=74.0)  # high < low
        assert engine.orb_range_formed is False
        engine.set_orb_range(high=0, low=0)  # zeros
        assert engine.orb_range_formed is False


class TestSizeScaling:
    """Tactic 7/8: Vol-targeting and time gating affect size_scalar."""

    def test_us_session_boost(self):
        engine = make_engine()
        engine.set_orb_range(high=74.0, low=72.0)
        state = make_state(
            silver_price=74.5,
            gold_5min_change=0.8,
            hour_cet=15,  # US session (14-17)
            vol_scalar=1.0,
        )
        decision = engine.tick(state)
        assert decision["action"] == "BUY"
        assert decision["size_scalar"] == 1.2  # 1.0 * 1.2

    def test_dead_zone_penalty(self):
        engine = make_engine()
        engine.set_orb_range(high=74.0, low=72.0)
        state = make_state(
            silver_price=74.5,
            gold_5min_change=0.8,
            hour_cet=12,  # dead zone (10-14)
            vol_scalar=1.0,
        )
        decision = engine.tick(state)
        assert decision["action"] == "BUY"
        assert decision["size_scalar"] == 0.7  # 1.0 * 0.7

    def test_size_scalar_clamped_min(self):
        engine = make_engine()
        engine.set_orb_range(high=74.0, low=72.0)
        state = make_state(
            silver_price=74.5,
            gold_5min_change=0.8,
            hour_cet=12,
            vol_scalar=0.1,  # 0.1 * 0.7 = 0.07 -> clamped to 0.25
        )
        decision = engine.tick(state)
        assert decision["action"] == "BUY"
        assert decision["size_scalar"] == 0.25

    def test_size_scalar_clamped_max(self):
        engine = make_engine()
        engine.set_orb_range(high=74.0, low=72.0)
        state = make_state(
            silver_price=74.5,
            gold_5min_change=0.8,
            hour_cet=15,
            vol_scalar=2.5,  # 2.5 * 1.2 = 3.0 -> clamped to 2.0
        )
        decision = engine.tick(state)
        assert decision["action"] == "BUY"
        assert decision["size_scalar"] == 2.0


class TestChronosConfidenceModifier:
    """Chronos forecast reduces confidence when disagreeing."""

    def test_chronos_reduces_long_confidence_when_bearish(self):
        engine = make_engine()
        engine.set_orb_range(high=74.0, low=72.0)
        state = make_state(
            silver_price=74.5,
            gold_5min_change=0.8,
            hour_cet=15,
            chronos_24h_pct=-0.5,  # bearish, disagrees with LONG
        )
        decision = engine.tick(state)
        assert decision["action"] == "BUY"
        assert decision["direction"] == "LONG"
        # 2 votes / 4 * 0.7 = 0.35
        assert decision["confidence"] == 0.35

    def test_chronos_reduces_short_confidence_when_bullish(self):
        engine = make_engine()
        engine.set_orb_range(high=76.0, low=74.0)
        state = make_state(
            silver_price=73.5,
            gold_5min_change=-0.8,
            hour_cet=15,
            chronos_24h_pct=0.5,  # bullish, disagrees with SHORT
        )
        decision = engine.tick(state)
        assert decision["action"] == "BUY"
        assert decision["direction"] == "SHORT"
        assert decision["confidence"] == 0.35

    def test_chronos_no_effect_when_aligned(self):
        engine = make_engine()
        engine.set_orb_range(high=74.0, low=72.0)
        state = make_state(
            silver_price=74.5,
            gold_5min_change=0.8,
            hour_cet=15,
            chronos_24h_pct=0.5,  # bullish, agrees with LONG
        )
        decision = engine.tick(state)
        assert decision["confidence"] == 0.5  # unmodified


class TestSessionStats:
    """get_session_stats() reflects accurate state."""

    def test_initial_stats(self):
        engine = make_engine()
        stats = engine.get_session_stats()
        assert stats["session_pnl"] == 0
        assert stats["trade_count"] == 0
        assert stats["win_rate"] == 0
        assert stats["mode"] == "momentum"
        assert stats["has_position"] is False

    def test_stats_after_trades(self):
        engine = make_engine()
        engine.confirm_entry("LONG", 7.0, 150, 75.0)
        engine.confirm_exit(pnl=100.0)
        engine.confirm_entry("SHORT", 2.5, 400, 75.0)
        engine.confirm_exit(pnl=-50.0)
        stats = engine.get_session_stats()
        assert stats["session_pnl"] == 50.0
        assert stats["trade_count"] == 2
        assert stats["win_count"] == 1
        assert stats["loss_count"] == 1
        assert stats["win_rate"] == 0.5
        assert stats["has_position"] is False

    def test_stats_with_open_position(self):
        engine = make_engine()
        engine.confirm_entry("LONG", 7.0, 150, 75.0)
        stats = engine.get_session_stats()
        assert stats["has_position"] is True


class TestNoSilverPrice:
    """Zero/missing silver price should return HOLD."""

    def test_zero_silver_price(self):
        engine = make_engine()
        state = make_state(silver_price=0)
        decision = engine.tick(state)
        assert decision["action"] == "HOLD"
        assert "no silver price" in decision["reason"]

    def test_negative_silver_price(self):
        engine = make_engine()
        state = make_state(silver_price=-1)
        decision = engine.tick(state)
        assert decision["action"] == "HOLD"


class TestORBFromState:
    """ORB range can be loaded from state dict."""

    def test_orb_from_state_dict(self):
        engine = make_engine()
        state = make_state(
            silver_price=74.5,
            orb_range={"high": 74.0, "low": 72.0, "formed": True},
            gold_5min_change=0.8,
            hour_cet=15,
        )
        decision = engine.tick(state)
        assert decision["action"] == "BUY"
        assert "orb" in decision["tactics_agreed"]

    def test_orb_from_state_not_formed(self):
        engine = make_engine()
        state = make_state(
            silver_price=74.5,
            orb_range={"high": 74.0, "low": 72.0, "formed": False},
        )
        decision = engine.tick(state)
        assert "orb" not in decision.get("tactics_agreed", [])


class TestLoadTemporalPatterns:
    """load_temporal_patterns() method."""

    def test_load_patterns(self):
        engine = make_engine()
        patterns = [
            {"day": 0, "hour_cet": 10, "direction": "BEAR", "probability": 80},
            {"day": 2, "hour_cet": 16, "direction": "BULL", "probability": 70},
        ]
        engine.load_temporal_patterns(patterns)
        assert len(engine._temporal_patterns) == 2
        assert (0, 10) in engine._temporal_patterns
        assert (2, 16) in engine._temporal_patterns
