"""Cross-cutting risk gates for MSTR Loop — drawdown, earnings, BTC regime.

Keeps strategies pure (entry/exit logic only) by centralising these
macro/portfolio-level refusals here. Called by loop.py before every
strategy step; any gate returning True halts all new entries for that
cycle.

Design: each gate is a pure function that reads state + optional config
overrides and returns (halted: bool, reason: str). Composition happens
in `any_halt_active()` which returns the first matching reason so logs
name the specific halt rather than a generic "risk gate fired".
"""

from __future__ import annotations

import datetime
import logging
import os
from typing import Any

from portfolio.mstr_loop import config
from portfolio.mstr_loop.data_provider import MstrBundle
from portfolio.mstr_loop.state import BotState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drawdown circuit breaker
# ---------------------------------------------------------------------------

def _current_equity_sek(state: BotState) -> float:
    """Total equity = cash + mark-to-market of open positions.

    Mark-to-market uses the *entry cert price* as a proxy when we don't have
    a live quote; this is conservative (treats an open winner as flat for
    drawdown purposes, delaying a false halt). Loop refreshes peak equity
    at cycle start using the real live cert quote when available.
    """
    equity = state.cash_sek
    for pos in state.positions.values():
        equity += pos.entry_cert_price * pos.units
    return equity


def update_drawdown_peaks(state: BotState, current_equity: float | None = None) -> None:
    """Maintain peak_equity and rolling daily/weekly baselines.

    Call once per cycle before gate checks. Rolls `session_start_equity`
    forward when a new calendar date starts (simple proxy for "new US
    session"); rolls `week_start_equity` forward when a new ISO week begins.
    """
    eq = current_equity if current_equity is not None else _current_equity_sek(state)
    state.peak_equity_sek = max(state.peak_equity_sek, eq)

    now_utc = datetime.datetime.now(datetime.UTC)
    today_iso = now_utc.date().isoformat()
    iso_year, iso_week, _ = now_utc.isocalendar()
    week_tag = f"{iso_year}-W{iso_week:02d}"

    if state.session_start_ts != today_iso:
        state.session_start_ts = today_iso
        state.session_start_equity_sek = eq
    if state.week_start_ts != week_tag:
        state.week_start_ts = week_tag
        state.week_start_equity_sek = eq


def drawdown_halt_active(state: BotState) -> tuple[bool, str]:
    """Return (halted, reason) based on daily/weekly P&L vs rolling bases.

    A halt's `until` timestamp is stored on state; gate lifts automatically
    once now > until.
    """
    if not config.DRAWDOWN_CHECK_ENABLED:
        return False, ""
    now_utc = datetime.datetime.now(datetime.UTC)
    # Honor active-halt timestamps first
    for attr, label in [("daily_halted_until", "daily"), ("weekly_halted_until", "weekly")]:
        ts = getattr(state, attr, "")
        if ts:
            try:
                halt_until = datetime.datetime.fromisoformat(ts)
                if halt_until.tzinfo is None:
                    halt_until = halt_until.replace(tzinfo=datetime.UTC)
                if now_utc < halt_until:
                    remaining = (halt_until - now_utc).total_seconds() / 60
                    return True, f"{label}_drawdown_halt_{remaining:.0f}min_remaining"
                setattr(state, attr, "")  # expired — clear
            except (TypeError, ValueError):
                setattr(state, attr, "")  # malformed — clear

    current_equity = _current_equity_sek(state)

    if state.session_start_equity_sek > 0:
        daily_pnl_pct = (current_equity - state.session_start_equity_sek) / state.session_start_equity_sek * 100
        if daily_pnl_pct <= config.DRAWDOWN_DAILY_HALT_PCT:
            # Arm a 24h halt
            state.daily_halted_until = (now_utc + datetime.timedelta(hours=24)).isoformat()
            return True, f"daily_drawdown_{daily_pnl_pct:.2f}%_vs_{config.DRAWDOWN_DAILY_HALT_PCT}%_floor"

    if state.week_start_equity_sek > 0:
        weekly_pnl_pct = (current_equity - state.week_start_equity_sek) / state.week_start_equity_sek * 100
        if weekly_pnl_pct <= config.DRAWDOWN_WEEKLY_HALT_PCT:
            state.weekly_halted_until = (now_utc + datetime.timedelta(days=7)).isoformat()
            return True, f"weekly_drawdown_{weekly_pnl_pct:.2f}%_vs_{config.DRAWDOWN_WEEKLY_HALT_PCT}%_floor"

    return False, ""


# ---------------------------------------------------------------------------
# BTC-regime gate
# ---------------------------------------------------------------------------

def btc_regime_refuses_long(bundle: MstrBundle) -> tuple[bool, str]:
    """Refuse LONG entries when BTC is in confirmed down-trend regime."""
    if not config.BTC_REGIME_GATE_ENABLED:
        return False, ""
    if bundle.btc_regime in config.BTC_REGIME_DOWN_TAGS:
        return True, f"btc_regime={bundle.btc_regime}_refuses_LONG"
    return False, ""


def btc_regime_refuses_short(bundle: MstrBundle) -> tuple[bool, str]:
    """Refuse SHORT entries when BTC is in confirmed up-trend regime."""
    if not config.BTC_REGIME_GATE_ENABLED:
        return False, ""
    if bundle.btc_regime in config.BTC_REGIME_UP_TAGS:
        return True, f"btc_regime={bundle.btc_regime}_refuses_SHORT"
    return False, ""


# ---------------------------------------------------------------------------
# Earnings blackout
# ---------------------------------------------------------------------------

def earnings_blackout_active(
    ticker: str = "MSTR",
    days_before: int | None = None,
    days_after: int | None = None,
) -> tuple[bool, str]:
    """Return (active, reason) — True if within ±N days of the next MSTR earnings.

    Lazy-imports portfolio.earnings_calendar so the loop can start without
    the Alpha Vantage dep in tests or offline environments.
    """
    if not config.EARNINGS_BLACKOUT_ENABLED:
        return False, ""
    before = days_before if days_before is not None else config.EARNINGS_BLACKOUT_DAYS_BEFORE
    after = days_after if days_after is not None else config.EARNINGS_BLACKOUT_DAYS_AFTER
    try:
        from portfolio.earnings_calendar import get_earnings_date
    except ImportError:
        logger.debug("earnings_blackout: earnings_calendar unavailable — gate defaults to inactive")
        return False, ""
    try:
        info = get_earnings_date(ticker)
    except Exception:
        logger.warning("earnings_blackout: get_earnings_date failed — gate defaults to inactive", exc_info=True)
        return False, ""
    if not info or not info.get("earnings_date"):
        return False, ""
    days_until = info.get("days_until")
    if days_until is None:
        return False, ""
    # Window: -before through +after relative to the event. days_until < 0
    # means the event has passed.
    if -after <= days_until <= before:
        return True, f"earnings_blackout_MSTR_{info['earnings_date']}_days_until={days_until}"
    return False, ""


# ---------------------------------------------------------------------------
# Combined gate
# ---------------------------------------------------------------------------

def effective_trail_distance_pct(bundle: MstrBundle, fallback_pct: float) -> float:
    """Return the trail distance to use this cycle.

    When ATR_ADAPTIVE_TRAIL_ENABLED, scales trail distance by realized vol:
        trail = clamp(MIN, MAX, ATR_MULT × atr_pct)
    In quiet sessions atr_pct is small → tighter trail. In wild sessions
    atr_pct is large → wider trail (stops whipsaw).

    Fallback when disabled or atr_pct unavailable: caller's configured
    fixed trail distance.
    """
    if not config.ATR_ADAPTIVE_TRAIL_ENABLED:
        return fallback_pct
    if bundle.atr_pct <= 0:
        return fallback_pct
    raw = config.ATR_ADAPTIVE_MULT * bundle.atr_pct
    return max(config.ATR_ADAPTIVE_TRAIL_MIN_PCT,
               min(config.ATR_ADAPTIVE_TRAIL_MAX_PCT, raw))


def any_entry_halt_active(state: BotState, bundle: MstrBundle, direction: str = "LONG") -> tuple[bool, str]:
    """Single composed gate called before each strategy entry.

    Returns the FIRST matching halt reason so logs name the specific gate.
    Drawdown and earnings apply to both directions; BTC regime gate direction-
    aware.
    """
    halted, reason = drawdown_halt_active(state)
    if halted:
        return True, reason

    halted, reason = earnings_blackout_active()
    if halted:
        return True, reason

    if direction == "LONG":
        halted, reason = btc_regime_refuses_long(bundle)
        if halted:
            return True, reason
    elif direction == "SHORT":
        halted, reason = btc_regime_refuses_short(bundle)
        if halted:
            return True, reason

    return False, ""
