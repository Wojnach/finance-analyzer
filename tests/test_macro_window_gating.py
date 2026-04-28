"""Tests for macro-event regime gating (auto-adapt signal weights during
event-heavy windows).

Caught 2026-04-28: 19 of 21 flagged per-ticker accuracy degradations
were statistically real, coincident with the densest macro-event week
of 2026 (FOMC, CPI, NFP, four central banks, Mag 7 earnings). Technical
and sentiment signals trained on price-pattern continuity systematically
misvote when news drives prices.

This module pins the contract for:
1. ``portfolio.econ_dates.is_macro_window`` — detection logic.
2. ``portfolio.signal_engine`` — force-HOLD + downweight overlay applied
   to MACRO_WINDOW_FORCE_HOLD_SIGNALS / MACRO_WINDOW_DOWNWEIGHT_SIGNALS
   when ``_is_macro_window_cached()`` returns True.
"""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from unittest.mock import patch

import pytest

from portfolio import econ_dates
from portfolio.econ_dates import is_macro_window
from portfolio.signal_engine import (
    MACRO_WINDOW_DOWNWEIGHT_MULTIPLIER,
    MACRO_WINDOW_DOWNWEIGHT_SIGNALS,
    MACRO_WINDOW_FORCE_HOLD_SIGNALS,
    _MACRO_WINDOW_CACHE_TTL_S,
    _weighted_consensus,
)


def _evt(d: date, evt_type: str = "FOMC", impact: str = "high") -> dict:
    return {"date": d, "type": evt_type, "impact": impact}


@pytest.fixture
def empty_calendar(monkeypatch):
    """Replace ECON_EVENTS with an empty list for deterministic tests."""
    monkeypatch.setattr(econ_dates, "ECON_EVENTS", [])
    yield


@pytest.fixture
def reset_macro_cache():
    """Wipe the macro-window cache so each test starts fresh.

    The cache is module-level state that would otherwise let an earlier
    test's reading leak into a later test. We also reset the transition
    logger so log spam doesn't clutter test output.
    """
    from portfolio import signal_engine
    signal_engine._macro_window_cache["value"] = False
    signal_engine._macro_window_cache["ts"] = 0.0
    signal_engine._macro_window_last_state["active"] = None
    yield
    signal_engine._macro_window_cache["value"] = False
    signal_engine._macro_window_cache["ts"] = 0.0
    signal_engine._macro_window_last_state["active"] = None


class TestIsMacroWindow:
    """Detection edges: forward window, backward window, impact filter."""

    def test_empty_calendar_returns_false(self, empty_calendar):
        now = datetime(2026, 4, 28, 14, 0, tzinfo=UTC)
        assert is_macro_window(now=now) is False

    def test_high_impact_12h_future_in_window(self, monkeypatch):
        now = datetime(2026, 4, 28, 8, 0, tzinfo=UTC)
        # Event lands at 14:00 UTC same day → +6h
        monkeypatch.setattr(
            econ_dates, "ECON_EVENTS", [_evt(date(2026, 4, 28))]
        )
        assert is_macro_window(now=now) is True

    def test_high_impact_60h_future_in_window(self, monkeypatch):
        now = datetime(2026, 4, 28, 0, 0, tzinfo=UTC)
        # Event 2.5 days out at 14:00 UTC → +62h, within default 72h
        monkeypatch.setattr(
            econ_dates, "ECON_EVENTS", [_evt(date(2026, 4, 30))]
        )
        assert is_macro_window(now=now) is True

    def test_high_impact_96h_future_outside_window(self, monkeypatch):
        now = datetime(2026, 4, 28, 0, 0, tzinfo=UTC)
        # Event 4 days out at 14:00 UTC → +110h, outside default 72h
        monkeypatch.setattr(
            econ_dates, "ECON_EVENTS", [_evt(date(2026, 5, 2))]
        )
        assert is_macro_window(now=now) is False

    def test_high_impact_12h_past_in_lookback(self, monkeypatch):
        # now is 12h after event release at 14:00
        now = datetime(2026, 4, 28, 2, 0, tzinfo=UTC)
        monkeypatch.setattr(
            econ_dates, "ECON_EVENTS", [_evt(date(2026, 4, 27))]
        )
        # Event at 2026-04-27T14:00 → 12h ago at our `now`
        assert is_macro_window(now=now) is True

    def test_high_impact_36h_past_outside_default_lookback(self, monkeypatch):
        now = datetime(2026, 4, 29, 2, 0, tzinfo=UTC)
        monkeypatch.setattr(
            econ_dates, "ECON_EVENTS", [_evt(date(2026, 4, 27))]
        )
        # Event at 2026-04-27T14:00 → 36h ago, outside default 24h lookback
        assert is_macro_window(now=now) is False

    def test_medium_impact_event_filtered_out_by_default(self, monkeypatch):
        now = datetime(2026, 4, 28, 8, 0, tzinfo=UTC)
        monkeypatch.setattr(
            econ_dates, "ECON_EVENTS",
            [_evt(date(2026, 4, 28), evt_type="GDP", impact="medium")],
        )
        # Default impact_filter=("high",) — medium ignored
        assert is_macro_window(now=now) is False

    def test_lookback_zero_disables_backward_window(self, monkeypatch):
        # Event 12h in the past — would normally trigger
        now = datetime(2026, 4, 28, 2, 0, tzinfo=UTC)
        monkeypatch.setattr(
            econ_dates, "ECON_EVENTS", [_evt(date(2026, 4, 27))]
        )
        assert is_macro_window(now=now, lookback_hours=0) is False
        # And confirm forward still works
        future_now = datetime(2026, 4, 27, 8, 0, tzinfo=UTC)
        assert is_macro_window(now=future_now, lookback_hours=0) is True

    def test_lookahead_zero_disables_forward_window(self, monkeypatch):
        now = datetime(2026, 4, 28, 8, 0, tzinfo=UTC)
        monkeypatch.setattr(
            econ_dates, "ECON_EVENTS", [_evt(date(2026, 4, 28))]
        )
        assert is_macro_window(now=now, lookahead_hours=0) is False

    def test_event_exactly_at_lookahead_boundary(self, monkeypatch):
        """Boundary inclusivity: lookahead_hours boundary should COUNT as in window."""
        now = datetime(2026, 4, 28, 14, 0, tzinfo=UTC)
        monkeypatch.setattr(
            econ_dates, "ECON_EVENTS", [_evt(date(2026, 5, 1))]
        )
        # Event at 2026-05-01T14:00 → exactly +72h
        assert is_macro_window(now=now, lookahead_hours=72) is True

    def test_explicit_impact_filter_includes_medium(self, monkeypatch):
        now = datetime(2026, 4, 28, 8, 0, tzinfo=UTC)
        monkeypatch.setattr(
            econ_dates, "ECON_EVENTS",
            [_evt(date(2026, 4, 28), evt_type="GDP", impact="medium")],
        )
        assert is_macro_window(now=now, impact_filter=("high", "medium")) is True


class TestSignalEngineMacroWindowOverlay:
    """The overlay applied inside ``_weighted_consensus``: force-HOLD a
    set of signals, downweight another set, while leaving everything
    else untouched. Composes with regime/horizon multipliers."""

    @pytest.fixture
    def stub_macro_active(self, monkeypatch):
        """Make ``_is_macro_window_cached`` return True for this test."""
        monkeypatch.setattr(
            "portfolio.signal_engine._is_macro_window_cached",
            lambda *_a, **_kw: True,
        )

    @pytest.fixture
    def stub_macro_inactive(self, monkeypatch):
        """Make ``_is_macro_window_cached`` return False for this test."""
        monkeypatch.setattr(
            "portfolio.signal_engine._is_macro_window_cached",
            lambda *_a, **_kw: False,
        )

    def test_macro_inactive_force_hold_signal_votes_normally(
        self, stub_macro_inactive,
    ):
        """When macro window is inactive, claude_fundamental's BUY vote
        survives the gate and contributes to consensus."""
        votes = {"claude_fundamental": "BUY", "rsi": "BUY"}
        accuracy = {
            "claude_fundamental": {"accuracy": 0.6, "total": 200},
            "rsi": {"accuracy": 0.55, "total": 200},
        }
        action, conf = _weighted_consensus(
            votes, accuracy, "ranging", activation_rates={},
        )
        assert action == "BUY"

    def test_macro_active_force_holds_claude_fundamental(self, stub_macro_active):
        """During a macro window, claude_fundamental is force-HOLD'd
        regardless of its vote — its 30-120min LLM cascade lag and
        >75% BUY bias make it dominantly wrong."""
        # claude_fundamental BUY would normally outvote a single SELL,
        # but with it forced to HOLD and rsi the only voter, RSI's SELL wins.
        votes = {"claude_fundamental": "BUY", "rsi": "SELL"}
        accuracy = {
            "claude_fundamental": {"accuracy": 0.6, "total": 200},
            "rsi": {"accuracy": 0.55, "total": 200},
        }
        action, _conf = _weighted_consensus(
            votes, accuracy, "ranging", activation_rates={},
        )
        assert action == "SELL"

    def test_macro_active_downweights_sentiment(self, stub_macro_active):
        """sentiment vote still counts but at half weight. With a
        balanced opposing vote of equal accuracy, sentiment loses the
        consensus tug-of-war it would have won at full weight.

        Uses ``trending-up`` because sentiment is REGIME_GATED to HOLD in
        ranging/trending-down — those regimes already suppress it
        regardless of the macro overlay, which would mask the test signal.
        """
        # sentiment BUY at 0.7 acc should beat rsi SELL at 0.65 normally.
        # Macro × 0.5 → sentiment effective ~0.35 < 0.65 → SELL.
        votes = {"sentiment": "BUY", "rsi": "SELL"}
        accuracy = {
            "sentiment": {"accuracy": 0.70, "total": 500},
            "rsi": {"accuracy": 0.65, "total": 500},
        }
        action, _conf = _weighted_consensus(
            votes, accuracy, "trending-up", activation_rates={},
        )
        assert action == "SELL"

    def test_macro_inactive_sentiment_vote_full_weight(self, stub_macro_inactive):
        """Inverse of the above: with macro inactive, sentiment's higher
        accuracy beats rsi outright (in a regime that doesn't gate it).
        """
        votes = {"sentiment": "BUY", "rsi": "SELL"}
        accuracy = {
            "sentiment": {"accuracy": 0.70, "total": 500},
            "rsi": {"accuracy": 0.65, "total": 500},
        }
        action, _conf = _weighted_consensus(
            votes, accuracy, "trending-up", activation_rates={},
        )
        assert action == "BUY"

    def test_macro_active_does_not_affect_unrelated_signals(
        self, stub_macro_active,
    ):
        """Signals outside the FORCE_HOLD / DOWNWEIGHT sets keep full weight.
        A macro window must NOT silently penalize rsi/macd/ema."""
        votes = {"rsi": "BUY", "macd": "BUY", "ema": "BUY"}
        accuracy = {
            "rsi": {"accuracy": 0.55, "total": 500},
            "macd": {"accuracy": 0.55, "total": 500},
            "ema": {"accuracy": 0.55, "total": 500},
        }
        action, conf = _weighted_consensus(
            votes, accuracy, "ranging", activation_rates={},
        )
        assert action == "BUY"
        # All three votes survive → high consensus
        assert conf >= 0.99

    def test_force_hold_set_does_not_overlap_downweight_set(self):
        """Belt-and-braces: a signal can't be in both sets or the
        force-HOLD pre-pass collides with the downweight branch."""
        overlap = MACRO_WINDOW_FORCE_HOLD_SIGNALS & MACRO_WINDOW_DOWNWEIGHT_SIGNALS
        assert overlap == frozenset(), f"overlap detected: {overlap}"

    def test_constants_are_frozen(self):
        """The two signal sets must be frozensets so they're hashable
        and can't be mutated at runtime by an unrelated import."""
        assert isinstance(MACRO_WINDOW_FORCE_HOLD_SIGNALS, frozenset)
        assert isinstance(MACRO_WINDOW_DOWNWEIGHT_SIGNALS, frozenset)

    def test_downweight_multiplier_in_expected_range(self):
        """Sanity: multiplier must be in (0, 1]. >1 would BOOST during
        macro windows, which would be the opposite of intent. <=0 would
        zero out the vote, which is force-HOLD's job."""
        assert 0 < MACRO_WINDOW_DOWNWEIGHT_MULTIPLIER <= 1.0


class TestMacroWindowCache:
    """The 5min TTL cache around ``is_macro_window`` to avoid per-cycle cost."""

    def test_cache_hit_within_ttl_skips_recompute(
        self, reset_macro_cache, monkeypatch,
    ):
        from portfolio import signal_engine
        calls = {"n": 0}

        def fake_is_macro_window():
            calls["n"] += 1
            return True

        monkeypatch.setattr(econ_dates, "is_macro_window", fake_is_macro_window)

        # First call: cache miss → underlying called once.
        signal_engine._is_macro_window_cached(now_ts=1000.0)
        # Second call within TTL: cache hit → no extra underlying call.
        signal_engine._is_macro_window_cached(now_ts=1000.0 + 60)
        signal_engine._is_macro_window_cached(now_ts=1000.0 + 200)

        assert calls["n"] == 1

    def test_cache_expires_after_ttl(self, reset_macro_cache, monkeypatch):
        from portfolio import signal_engine
        calls = {"n": 0}

        def fake_is_macro_window():
            calls["n"] += 1
            return True

        monkeypatch.setattr(econ_dates, "is_macro_window", fake_is_macro_window)

        signal_engine._is_macro_window_cached(now_ts=1000.0)
        # After TTL, underlying called again.
        signal_engine._is_macro_window_cached(
            now_ts=1000.0 + _MACRO_WINDOW_CACHE_TTL_S + 1,
        )

        assert calls["n"] == 2

    def test_underlying_failure_treated_as_inactive(
        self, reset_macro_cache, monkeypatch, caplog,
    ):
        """If the econ_dates lookup raises, we must default to False
        (inactive) rather than letting the exception propagate up
        into _weighted_consensus and crash the loop."""
        from portfolio import signal_engine

        def boom():
            raise RuntimeError("simulated calendar error")

        monkeypatch.setattr(econ_dates, "is_macro_window", boom)

        result = signal_engine._is_macro_window_cached(now_ts=2000.0)
        assert result is False
