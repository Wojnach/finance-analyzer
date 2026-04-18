"""Tests for portfolio.mstr_loop.risk — drawdown, BTC regime, earnings gates."""

from __future__ import annotations

import datetime
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portfolio.mstr_loop import config, risk
from portfolio.mstr_loop.data_provider import MstrBundle
from portfolio.mstr_loop.state import BotState, Position


def _mk_bundle(btc_regime="trending-up", **overrides):
    defaults = dict(
        ts="2026-04-18T16:00:00+00:00", source_stale_seconds=30.0,
        price_usd=165.0, raw_action="BUY", raw_weighted_confidence=0.6,
        rsi=55, macd_hist=1.2, bb_position="inside", regime="trending-up",
        atr_pct=1.5, buy_count=5, sell_count=3, total_voters=8,
        votes={"x": "BUY"}, p_up_1d=0.65, exp_return_1d_pct=0.3,
        exp_return_3d_pct=0.8, heatmap=[], stale=False,
        weighted_score_long=0.70, weighted_score_short=0.20,
        btc_regime=btc_regime, btc_price=75000.0, btc_rsi=55.0,
    )
    defaults.update(overrides)
    return MstrBundle(**defaults)


# ---------------------------------------------------------------------------
# Drawdown circuit breaker
# ---------------------------------------------------------------------------


def test_drawdown_no_halt_when_equity_flat():
    s = BotState(cash_sek=100_000, peak_equity_sek=100_000,
                 session_start_equity_sek=100_000,
                 week_start_equity_sek=100_000)
    halted, reason = risk.drawdown_halt_active(s)
    assert halted is False
    assert reason == ""


def test_drawdown_daily_halt_fires(monkeypatch):
    monkeypatch.setattr(config, "DRAWDOWN_DAILY_HALT_PCT", -3.0)
    s = BotState(cash_sek=96_000, session_start_equity_sek=100_000,
                 week_start_equity_sek=100_000)
    halted, reason = risk.drawdown_halt_active(s)
    assert halted is True
    assert "daily_drawdown" in reason
    # Halt gets armed with a 24h expiry
    assert s.daily_halted_until != ""


def test_drawdown_weekly_halt_fires(monkeypatch):
    monkeypatch.setattr(config, "DRAWDOWN_WEEKLY_HALT_PCT", -8.0)
    s = BotState(cash_sek=90_000, session_start_equity_sek=96_000,
                 week_start_equity_sek=100_000)
    # Daily OK (96k→90k = -6.25%), weekly NOT (100k→90k = -10%)
    # So daily fires first at -6.25% < -3%. Widen daily gate for this test.
    monkeypatch.setattr(config, "DRAWDOWN_DAILY_HALT_PCT", -50.0)
    halted, reason = risk.drawdown_halt_active(s)
    assert halted is True
    assert "weekly_drawdown" in reason


def test_drawdown_respects_active_halt():
    """Once armed, halt stays until its timestamp expires."""
    future = (datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=5)).isoformat()
    s = BotState(daily_halted_until=future)
    halted, reason = risk.drawdown_halt_active(s)
    assert halted is True
    assert "daily_drawdown_halt" in reason


def test_drawdown_clears_expired_halt():
    """Halt that's past its expiry clears and doesn't block."""
    past = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=1)).isoformat()
    s = BotState(daily_halted_until=past,
                 cash_sek=100_000, session_start_equity_sek=100_000,
                 week_start_equity_sek=100_000)
    halted, reason = risk.drawdown_halt_active(s)
    assert halted is False
    assert s.daily_halted_until == ""  # got cleared


def test_update_peaks_sets_session_start():
    """First call sets baselines; subsequent same-day calls don't reset them."""
    s = BotState(cash_sek=100_000)
    risk.update_drawdown_peaks(s, current_equity=100_000)
    first_session_start = s.session_start_equity_sek
    first_ts = s.session_start_ts
    assert first_session_start == 100_000
    # Second call same day — baseline sticks
    risk.update_drawdown_peaks(s, current_equity=105_000)
    assert s.session_start_equity_sek == first_session_start
    assert s.session_start_ts == first_ts
    assert s.peak_equity_sek == 105_000


def test_update_peaks_disabled(monkeypatch):
    """DRAWDOWN_CHECK_ENABLED=False still updates peaks (so history accumulates)
    but drawdown_halt_active returns no-halt."""
    monkeypatch.setattr(config, "DRAWDOWN_CHECK_ENABLED", False)
    s = BotState(cash_sek=90_000, session_start_equity_sek=100_000,
                 week_start_equity_sek=100_000)
    halted, reason = risk.drawdown_halt_active(s)
    assert halted is False


# ---------------------------------------------------------------------------
# BTC regime gate
# ---------------------------------------------------------------------------


def test_btc_regime_down_refuses_long():
    b = _mk_bundle(btc_regime="trending-down")
    halted, reason = risk.btc_regime_refuses_long(b)
    assert halted is True
    assert "trending-down" in reason


def test_btc_regime_up_allows_long():
    b = _mk_bundle(btc_regime="trending-up")
    halted, _ = risk.btc_regime_refuses_long(b)
    assert halted is False


def test_btc_regime_up_refuses_short():
    b = _mk_bundle(btc_regime="trending-up")
    halted, reason = risk.btc_regime_refuses_short(b)
    assert halted is True
    assert "trending-up" in reason


def test_btc_regime_ranging_allows_both_sides():
    b = _mk_bundle(btc_regime="ranging")
    halted_l, _ = risk.btc_regime_refuses_long(b)
    halted_s, _ = risk.btc_regime_refuses_short(b)
    assert halted_l is False
    assert halted_s is False


def test_btc_regime_gate_disabled(monkeypatch):
    monkeypatch.setattr(config, "BTC_REGIME_GATE_ENABLED", False)
    b = _mk_bundle(btc_regime="trending-down")
    halted, _ = risk.btc_regime_refuses_long(b)
    assert halted is False


# ---------------------------------------------------------------------------
# Earnings blackout (defensive — works without earnings_calendar module)
# ---------------------------------------------------------------------------


def test_earnings_disabled_returns_no_halt(monkeypatch):
    monkeypatch.setattr(config, "EARNINGS_BLACKOUT_ENABLED", False)
    halted, _ = risk.earnings_blackout_active()
    assert halted is False


def test_earnings_blackout_fires_within_window(monkeypatch):
    """Mock get_earnings_date to return days_until=1, within the before=2 window."""
    fake = type("M", (), {"get_earnings_date": staticmethod(
        lambda t: {"earnings_date": "2026-04-20", "days_until": 1}
    )})
    monkeypatch.setitem(sys.modules, "portfolio.earnings_calendar", fake)
    monkeypatch.setattr(config, "EARNINGS_BLACKOUT_ENABLED", True)
    monkeypatch.setattr(config, "EARNINGS_BLACKOUT_DAYS_BEFORE", 2)
    halted, reason = risk.earnings_blackout_active()
    assert halted is True
    assert "earnings_blackout" in reason


def test_earnings_blackout_fires_day_after(monkeypatch):
    """days_until=-1 means earnings happened yesterday — still in post-window."""
    fake = type("M", (), {"get_earnings_date": staticmethod(
        lambda t: {"earnings_date": "2026-04-17", "days_until": -1}
    )})
    monkeypatch.setitem(sys.modules, "portfolio.earnings_calendar", fake)
    monkeypatch.setattr(config, "EARNINGS_BLACKOUT_ENABLED", True)
    monkeypatch.setattr(config, "EARNINGS_BLACKOUT_DAYS_AFTER", 1)
    halted, reason = risk.earnings_blackout_active()
    assert halted is True


def test_earnings_blackout_outside_window(monkeypatch):
    """days_until=10 is well beyond the window — no halt."""
    fake = type("M", (), {"get_earnings_date": staticmethod(
        lambda t: {"earnings_date": "2026-04-28", "days_until": 10}
    )})
    monkeypatch.setitem(sys.modules, "portfolio.earnings_calendar", fake)
    halted, _ = risk.earnings_blackout_active()
    assert halted is False


def test_earnings_missing_module_returns_no_halt(monkeypatch):
    """If earnings_calendar import fails, gate defaults to inactive (fail-safe)."""
    # Remove cached import if present
    sys.modules.pop("portfolio.earnings_calendar", None)
    # Make import raise
    class _Blocker:
        def find_spec(self, name, path=None, target=None):
            if name == "portfolio.earnings_calendar":
                raise ImportError("mocked")
            return None
    # Simpler: mock the import call directly
    import builtins
    real = builtins.__import__
    def blocker(name, *args, **kwargs):
        if name == "portfolio.earnings_calendar":
            raise ImportError("blocked for test")
        return real(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocker)
    halted, _ = risk.earnings_blackout_active()
    assert halted is False


# ---------------------------------------------------------------------------
# Combined gate (any_entry_halt_active)
# ---------------------------------------------------------------------------


def test_combined_gate_prefers_drawdown_reason():
    """If drawdown halts, that reason is returned (not BTC/earnings)."""
    future = (datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=5)).isoformat()
    s = BotState(daily_halted_until=future)
    b = _mk_bundle(btc_regime="trending-down")
    halted, reason = risk.any_entry_halt_active(s, b, direction="LONG")
    assert halted is True
    assert "daily_drawdown_halt" in reason


def test_combined_gate_passes_through_btc_regime():
    """No drawdown, no earnings — BTC regime blocks LONG when down."""
    s = BotState(cash_sek=100_000, session_start_equity_sek=100_000,
                 week_start_equity_sek=100_000)
    b = _mk_bundle(btc_regime="trending-down")
    halted, reason = risk.any_entry_halt_active(s, b, direction="LONG")
    assert halted is True
    assert "btc_regime" in reason


def test_combined_gate_clean_pass():
    """All gates clear → returns (False, '')."""
    s = BotState(cash_sek=100_000, session_start_equity_sek=100_000,
                 week_start_equity_sek=100_000)
    b = _mk_bundle(btc_regime="trending-up")
    halted, reason = risk.any_entry_halt_active(s, b, direction="LONG")
    assert halted is False
    assert reason == ""


# ---------------------------------------------------------------------------
# ATR-adaptive trail distance
# ---------------------------------------------------------------------------


def test_atr_trail_disabled_returns_fallback(monkeypatch):
    monkeypatch.setattr(config, "ATR_ADAPTIVE_TRAIL_ENABLED", False)
    b = _mk_bundle(atr_pct=2.0)
    assert risk.effective_trail_distance_pct(b, 2.5) == 2.5


def test_atr_trail_scales_with_atr(monkeypatch):
    monkeypatch.setattr(config, "ATR_ADAPTIVE_TRAIL_ENABLED", True)
    monkeypatch.setattr(config, "ATR_ADAPTIVE_MULT", 1.5)
    monkeypatch.setattr(config, "ATR_ADAPTIVE_TRAIL_MIN_PCT", 1.0)
    monkeypatch.setattr(config, "ATR_ADAPTIVE_TRAIL_MAX_PCT", 5.0)
    b = _mk_bundle(atr_pct=2.0)
    # 1.5 × 2.0 = 3.0, within [1.0, 5.0]
    assert risk.effective_trail_distance_pct(b, 999.0) == pytest.approx(3.0)


def test_atr_trail_clamps_to_min(monkeypatch):
    monkeypatch.setattr(config, "ATR_ADAPTIVE_TRAIL_ENABLED", True)
    monkeypatch.setattr(config, "ATR_ADAPTIVE_MULT", 1.5)
    monkeypatch.setattr(config, "ATR_ADAPTIVE_TRAIL_MIN_PCT", 1.5)
    monkeypatch.setattr(config, "ATR_ADAPTIVE_TRAIL_MAX_PCT", 5.0)
    b = _mk_bundle(atr_pct=0.5)
    # 1.5 × 0.5 = 0.75 → clamped up to MIN=1.5
    assert risk.effective_trail_distance_pct(b, 999.0) == pytest.approx(1.5)


def test_atr_trail_clamps_to_max(monkeypatch):
    monkeypatch.setattr(config, "ATR_ADAPTIVE_TRAIL_ENABLED", True)
    monkeypatch.setattr(config, "ATR_ADAPTIVE_MULT", 1.5)
    monkeypatch.setattr(config, "ATR_ADAPTIVE_TRAIL_MIN_PCT", 1.0)
    monkeypatch.setattr(config, "ATR_ADAPTIVE_TRAIL_MAX_PCT", 4.0)
    b = _mk_bundle(atr_pct=5.0)
    # 1.5 × 5.0 = 7.5 → clamped down to MAX=4.0
    assert risk.effective_trail_distance_pct(b, 999.0) == pytest.approx(4.0)


def test_atr_trail_zero_atr_returns_fallback(monkeypatch):
    """atr_pct=0 means no data; fall back to the caller's fixed pct."""
    monkeypatch.setattr(config, "ATR_ADAPTIVE_TRAIL_ENABLED", True)
    b = _mk_bundle(atr_pct=0.0)
    assert risk.effective_trail_distance_pct(b, 2.0) == 2.0
