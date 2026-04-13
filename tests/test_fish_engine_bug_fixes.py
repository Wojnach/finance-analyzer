"""Tests for the 2026-04-13 fish engine 3-bug fix bundle.

Covers:
- Bug 1: signal-age guard rejects stale buy decisions
- Bug 3: peak-drawdown veto suppresses ORB chase

Bug 2a (stop-cancel retry) is a metals_loop integration patch tested
separately in tests/test_stops_cancel_resilience.py.
"""
from __future__ import annotations

import os
import time
from unittest.mock import patch

import pytest


# --- Bug 1: signal-age guard -----------------------------------------------


def _make_decision(signal_ts):
    """Build a fish engine LONG-direction BUY decision dict for tests."""
    return {
        "action": "BUY",
        "direction": "LONG",
        "reason": "test 3 tactics agree LONG",
        "tactics_agreed": ["orb", "layer2", "layer2_w"],
        "size_scalar": 1.0,
        "exit_reason": None,
        "confidence": 0.75,
        "instrument_ob": "1650161",
        "hold_minutes": 0,
        "signal_ts": signal_ts,
    }


class TestSignalAgeGuard:
    """Bug 1: fish engine BUY rejected if decision is stale."""

    def test_signal_ts_stamped_by_evaluate_entry(self):
        """fish_engine._evaluate_entry stamps a signal_ts on every directional decision."""
        from data.fish_engine import FishEngine

        # Mock time so the stamp is deterministic
        eng = FishEngine(time_func=lambda: 1_000_000.0)
        eng.set_orb_range(high=10.0, low=9.0)
        # Build a state that produces 2+ LONG votes (enough for MIN_VOTES)
        state = {
            "silver_price": 11.0,  # above orb_range_high -> ORB votes LONG
            "spread_pct": 0.1,
            "trade_guard_ok": True,
            "hour_cet": 14, "minute_cet": 0, "day_of_week": 1,
            "vol_scalar": 1.0,
            "mc_p_up": 0.7,
            "gold_5min_change": 1.0,  # above GOLD_LEAD_THRESHOLD -> votes LONG
        }
        decision = eng.tick(state)
        # If the engine voted BUY at all, signal_ts must be set
        if decision.get("action") == "BUY":
            assert "signal_ts" in decision
            assert decision["signal_ts"] == 1_000_000.0

    def test_execute_buy_rejects_stale_decision(self):
        """metals_loop._fish_engine_execute_buy SKIPS a decision older than the threshold."""
        from data import metals_loop as ml

        # Force the env override to a known value
        with patch.dict(os.environ, {"FISH_MAX_SIGNAL_AGE_SEC": "60"}):
            stale_decision = _make_decision(signal_ts=time.time() - 90.0)
            logged = []
            with patch.object(ml, "_loop_page", object()), \
                 patch.object(ml, "log", lambda msg: logged.append(msg)), \
                 patch.object(ml, "fetch_price_with_fallback") as mock_fetch:
                ml._fish_engine_execute_buy(stale_decision, price=8.0)
                # Must have logged the skip reason; must NOT have fetched price
                assert any("signal stale" in m for m in logged), \
                    f"expected 'signal stale' SKIP log; got: {logged}"
                mock_fetch.assert_not_called()

    def test_execute_buy_proceeds_for_fresh_decision(self):
        """A fresh decision proceeds past the age guard (other guards may still skip)."""
        from data import metals_loop as ml

        with patch.dict(os.environ, {"FISH_MAX_SIGNAL_AGE_SEC": "60"}):
            fresh_decision = _make_decision(signal_ts=time.time() - 5.0)
            logged = []
            with patch.object(ml, "_loop_page", object()), \
                 patch.object(ml, "log", lambda msg: logged.append(msg)), \
                 patch.object(ml, "fetch_price_with_fallback", return_value=None):
                ml._fish_engine_execute_buy(fresh_decision, price=8.0)
                # Should NOT log the stale-skip line; downstream price-fetch
                # returned None so it'll skip elsewhere — that's fine.
                assert not any("signal stale" in m for m in logged), \
                    f"unexpected stale skip; got: {logged}"

    def test_missing_signal_ts_treated_as_fresh(self):
        """Backward-compat: decisions without signal_ts (manual force-close paths)
        are NOT rejected by the age guard."""
        from data import metals_loop as ml

        with patch.dict(os.environ, {"FISH_MAX_SIGNAL_AGE_SEC": "60"}):
            decision_no_ts = _make_decision(signal_ts=None)
            decision_no_ts.pop("signal_ts")
            logged = []
            with patch.object(ml, "_loop_page", object()), \
                 patch.object(ml, "log", lambda msg: logged.append(msg)), \
                 patch.object(ml, "fetch_price_with_fallback", return_value=None):
                ml._fish_engine_execute_buy(decision_no_ts, price=8.0)
                assert not any("signal stale" in m for m in logged)


# --- Bug 3: peak-drawdown ORB veto ----------------------------------------


class TestPeakDrawdownVeto:
    """Bug 3: ORB tactic suppressed during active pullback from recent peak."""

    @pytest.fixture
    def engine_with_orb(self):
        from data.fish_engine import FishEngine
        # Time control
        clock = {"t": 1_000_000.0}
        eng = FishEngine(time_func=lambda: clock["t"])
        eng.set_orb_range(high=10.0, low=9.0)
        return eng, clock

    def test_fresh_breakout_no_pullback_votes_long(self, engine_with_orb):
        """No prior peak, breakout above orb high -> ORB votes LONG."""
        eng, clock = engine_with_orb
        votes = {}
        # Set peak/trough manually to simulate "first tick at this price"
        eng.underlying_peak_price = 11.0
        eng.underlying_peak_ts = clock["t"]  # peak is right now
        eng._vote_orb(silver_price=11.0, votes=votes, now=clock["t"])
        assert votes.get("orb") == "LONG"

    def test_long_blocked_when_peak_recent_and_drawdown_exceeded(self, engine_with_orb):
        """Peak 60s ago at 11.0, current 10.95 (-0.45%) -> suppress LONG."""
        eng, clock = engine_with_orb
        eng.underlying_peak_price = 11.0
        eng.underlying_peak_ts = clock["t"] - 60  # 60s ago
        votes = {}
        # Current price 10.95 = 0.45% below 11.0 peak, above orb_high 10.0
        eng._vote_orb(silver_price=10.95, votes=votes, now=clock["t"])
        assert "orb" not in votes, "ORB should be suppressed during active pullback"

    def test_long_allowed_when_peak_old(self, engine_with_orb):
        """Peak 10 minutes ago, current price still above orb -> LONG allowed
        (peak too old to imply a pullback signal)."""
        eng, clock = engine_with_orb
        eng.underlying_peak_price = 11.0
        eng.underlying_peak_ts = clock["t"] - 600  # 10 min ago
        votes = {}
        eng._vote_orb(silver_price=10.95, votes=votes, now=clock["t"])
        assert votes.get("orb") == "LONG"

    def test_long_allowed_when_drawdown_below_threshold(self, engine_with_orb):
        """Peak 60s ago at 11.0, current 10.99 (-0.09%, below 0.3% threshold) -> LONG OK."""
        eng, clock = engine_with_orb
        eng.underlying_peak_price = 11.0
        eng.underlying_peak_ts = clock["t"] - 60
        votes = {}
        eng._vote_orb(silver_price=10.99, votes=votes, now=clock["t"])
        assert votes.get("orb") == "LONG"

    def test_short_breakdown_blocked_during_active_bounce(self, engine_with_orb):
        """Mirror: trough 60s ago at 8.0, now 8.04 (+0.5% bounce) -> suppress SHORT."""
        eng, clock = engine_with_orb
        # Need trough below orb_range_low (9.0)
        eng.underlying_trough_price = 8.0
        eng.underlying_trough_ts = clock["t"] - 60
        votes = {}
        # Price 8.04 < orb_range_low 9.0, but bounced 0.5% off trough
        eng._vote_orb(silver_price=8.04, votes=votes, now=clock["t"])
        assert "orb" not in votes

    def test_veto_disabled_via_env(self, engine_with_orb):
        """FISH_PEAK_DRAWDOWN_VETO_ENABLED=0 disables the suppression."""
        eng, clock = engine_with_orb
        eng.underlying_peak_price = 11.0
        eng.underlying_peak_ts = clock["t"] - 60
        votes = {}
        with patch.dict(os.environ, {"FISH_PEAK_DRAWDOWN_VETO_ENABLED": "0"}):
            eng._vote_orb(silver_price=10.95, votes=votes, now=clock["t"])
        assert votes.get("orb") == "LONG", "veto must be bypassable for backtests"

    def test_no_now_arg_falls_back_to_no_veto(self, engine_with_orb):
        """Backward-compat: calling _vote_orb without `now` skips the veto."""
        eng, _clock = engine_with_orb
        eng.underlying_peak_price = 11.0
        eng.underlying_peak_ts = 1_000_000.0
        votes = {}
        eng._vote_orb(silver_price=10.95, votes=votes)  # default now=0.0
        assert votes.get("orb") == "LONG"

    def test_tick_updates_peak_and_trough(self, engine_with_orb):
        """tick() must update peak/trough on each call before voting."""
        eng, clock = engine_with_orb
        # Initial state — no peak/trough set yet
        assert eng.underlying_peak_price == 0.0
        assert eng.underlying_trough_price == float("inf")
        # First tick at price 10.5
        clock["t"] = 1_000_010.0
        state = {
            "silver_price": 10.5, "spread_pct": 0.1, "trade_guard_ok": True,
            "hour_cet": 14, "minute_cet": 0, "day_of_week": 1,
            "vol_scalar": 1.0, "mc_p_up": 0.5,
        }
        eng.tick(state)
        assert eng.underlying_peak_price == 10.5
        assert eng.underlying_trough_price == 10.5
        # Tick at higher price -> peak moves up, trough stays
        clock["t"] = 1_000_020.0
        state["silver_price"] = 11.0
        eng.tick(state)
        assert eng.underlying_peak_price == 11.0
        assert eng.underlying_trough_price == 10.5
        # Tick at lower price -> trough moves down, peak stays
        clock["t"] = 1_000_030.0
        state["silver_price"] = 10.0
        eng.tick(state)
        assert eng.underlying_peak_price == 11.0
        assert eng.underlying_trough_price == 10.0
