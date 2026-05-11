"""Per-bot trading-status reader for the dashboard home page.

Every Avanza-trading bot writes a state file. This module reads them
and projects each onto a small UI-shaped dict:

    {"bot": "golddigger",
     "label": "GoldDigger",
     "state": "SCANNING" | "TRADING" | "HALTED" | "COOLDOWN"
              | "OUTSIDE_HOURS" | "UNKNOWN",
     "reason": "<human-readable why>",
     "since_seconds": <int|None>,
     "position": {...} | None,
     "stats": {...},
     "error": "<optional>"}

The user's question "why aren't the loops trading on Avanza?" is
answered by the (state, reason) pair. State precedence:

    1. UNKNOWN  — state file missing or unreadable
    2. HALTED   — bot wrote halted=True (reads halted_reason)
    3. COOLDOWN — fishing engine: last_trade + cooldown_s > now
    4. TRADING  — open position present
    5. OUTSIDE_HOURS — outside the Avanza trading window
                      (08:30–21:30 Europe/Stockholm)
    6. SCANNING — running normally, no signal strong enough yet

GoldDigger and Elongir maintain ``halted_reason`` themselves, so we
surface it verbatim. Metals + fishing don't yet persist a "why no
trade" reason, so we fall back to inference from their state. See
``docs/PLAN.md`` and the plan file at
``/root/.claude/plans/merry-tinkering-cake.md``.
"""

from __future__ import annotations

from datetime import datetime, time as dtime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from portfolio.file_utils import load_json

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Avanza trading session in Europe/Stockholm — DST handled by zoneinfo.
# 2026-05-11: unified to 08:30–21:30 across all four bots after a user
# report that the dashboard was rendering OUTSIDE_HOURS at 14:23 CEST
# even though Elongir's actual config session is 08:30–21:30. The old
# 15:30–21:55 window matched GoldDigger's US-focused config and was
# misapplied here to metals/elongir/fishing. EU open is ~09:00 CET, US
# close is ~22:00 CET; 08:30–21:30 brackets the warrant-tradeable window
# the user actually trades on and matches the per-bot configs after the
# parallel patch to portfolio/golddigger/config.py.
SESSION_TZ = ZoneInfo("Europe/Stockholm")
SESSION_OPEN = dtime(8, 30)
SESSION_CLOSE = dtime(21, 30)

UTC = timezone.utc


def compute(
    data_dir: Path | None = None,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Build the trading-status payload for all four Avanza bots."""
    dd = Path(data_dir) if data_dir else DATA_DIR
    now = now_utc or datetime.now(UTC)
    return {
        "ts": now.isoformat(),
        "session_open": _in_session(now),
        "bots": [
            _golddigger(dd, now),
            _elongir(dd, now),
            _metals(dd, now),
            _fishing(dd, now),
        ],
    }


# ---------------------------------------------------------------------------
# Per-bot readers
# ---------------------------------------------------------------------------


def _golddigger(dd: Path, now: datetime) -> dict[str, Any]:
    state = load_json(dd / "golddigger_state.json", default=None)
    if state is None:
        return _unknown("golddigger", "GoldDigger")
    halted = bool(state.get("halted"))
    reason = (state.get("halted_reason") or "").strip()
    position = state.get("position")
    if halted:
        return _emit(
            "golddigger", "GoldDigger", "HALTED",
            reason or "halted (no reason recorded)",
            position=position, state=state,
        )
    if position:
        return _emit(
            "golddigger", "GoldDigger", "TRADING",
            "position open",
            position=position, state=state,
        )
    if not _in_session(now):
        return _emit(
            "golddigger", "GoldDigger", "OUTSIDE_HOURS",
            _next_open_hint(now),
            position=None, state=state,
        )
    return _emit(
        "golddigger", "GoldDigger", "SCANNING",
        "in session, no entry signal yet",
        position=None, state=state,
    )


def _elongir(dd: Path, now: datetime) -> dict[str, Any]:
    state = load_json(dd / "elongir_state.json", default=None)
    if state is None:
        return _unknown("elongir", "Elongir")
    halted = bool(state.get("halted"))
    reason = (state.get("halted_reason") or "").strip()
    position = state.get("position")
    if halted:
        return _emit(
            "elongir", "Elongir", "HALTED",
            reason or "halted (no reason recorded)",
            position=position, state=state,
        )
    if position:
        return _emit(
            "elongir", "Elongir", "TRADING",
            "position open",
            position=position, state=state,
        )
    if not _in_session(now):
        return _emit(
            "elongir", "Elongir", "OUTSIDE_HOURS",
            _next_open_hint(now),
            position=None, state=state,
        )
    return _emit(
        "elongir", "Elongir", "SCANNING",
        "in session, no dip detected",
        position=None, state=state,
    )


def _metals(dd: Path, now: datetime) -> dict[str, Any]:
    state = load_json(dd / "metals_swing_state.json", default=None)
    guard = load_json(dd / "metals_guard_state.json", default={}) or {}
    if state is None:
        return _unknown("metals", "Metals swing")
    positions = state.get("positions") or []
    has_position = bool(positions) if isinstance(positions, list) else bool(positions)
    if has_position:
        return _emit(
            "metals", "Metals swing", "TRADING",
            f"holding {len(positions)} position(s)" if isinstance(positions, list) else "position open",
            position=positions, state=state,
        )
    if not _in_session(now):
        return _emit(
            "metals", "Metals swing", "OUTSIDE_HOURS",
            _next_open_hint(now),
            position=None, state=state,
        )
    consecutive_losses = guard.get("consecutive_losses") or state.get("consecutive_losses") or 0
    last_buy_ts = state.get("last_buy_ts")
    reason = "in session, no signal"
    if consecutive_losses and consecutive_losses >= 3:
        reason = f"in session, {consecutive_losses} consecutive losses (caution)"
    elif last_buy_ts:
        reason = "in session, between trades"
    return _emit(
        "metals", "Metals swing", "SCANNING",
        reason,
        position=None, state=state,
    )


def _fishing(dd: Path, now: datetime) -> dict[str, Any]:
    state = load_json(dd / "fish_engine_state.json", default=None)
    if state is None:
        return _unknown("fishing", "Fishing engine")
    position = state.get("position")
    if position:
        return _emit(
            "fishing", "Fishing engine", "TRADING",
            "position open",
            position=position, state=state,
        )
    last_trade_ts = state.get("last_trade_ts")
    cooldown_s = state.get("cooldown_seconds") or 0
    if last_trade_ts and cooldown_s:
        try:
            last = datetime.fromtimestamp(float(last_trade_ts), tz=UTC)
            cool_until = last.timestamp() + float(cooldown_s)
            remaining = cool_until - now.timestamp()
            if remaining > 0:
                losses = state.get("consecutive_losses") or 0
                detail = f"{int(remaining)}s remaining"
                if losses:
                    detail = f"{losses} losses, {detail}"
                return _emit(
                    "fishing", "Fishing engine", "COOLDOWN",
                    detail,
                    position=None, state=state,
                )
        except Exception:
            pass
    if not _in_session(now):
        return _emit(
            "fishing", "Fishing engine", "OUTSIDE_HOURS",
            _next_open_hint(now),
            position=None, state=state,
        )
    return _emit(
        "fishing", "Fishing engine", "SCANNING",
        f"mode={state.get('mode') or 'idle'}",
        position=None, state=state,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _emit(
    bot: str,
    label: str,
    state_name: str,
    reason: str,
    *,
    position: Any,
    state: dict,
) -> dict[str, Any]:
    return {
        "bot": bot,
        "label": label,
        "state": state_name,
        "reason": reason,
        "position": position,
        "stats": _extract_stats(state),
    }


def _unknown(bot: str, label: str) -> dict[str, Any]:
    return {
        "bot": bot,
        "label": label,
        "state": "UNKNOWN",
        "reason": "state file missing or unreadable",
        "position": None,
        "stats": {},
        "error": "state file not found",
    }


def _extract_stats(state: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in (
        "daily_pnl",
        "daily_trades",
        "total_trades",
        "total_pnl",
        "consecutive_losses",
        "cash_sek",
        "equity_sek",
    ):
        if k in state:
            out[k] = state[k]
    return out


def _in_session(now_utc: datetime) -> bool:
    """True iff Europe/Stockholm wall-clock time is within the warrant
    session (Mon–Fri, 15:30–21:55 inclusive of open, exclusive of close).

    Codex P1 finding 2026-05-04: weekday check matters — Saturday and
    Sunday at 16:00 local time would otherwise read as session_open and
    bots would render as SCANNING when they're correctly idle.
    """
    local = now_utc.astimezone(SESSION_TZ)
    if local.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    t = local.timetz().replace(tzinfo=None)
    return SESSION_OPEN <= t < SESSION_CLOSE


def _next_open_hint(now_utc: datetime) -> str:
    """Human-readable 'next 15:30 CEST in 2h 14m'.

    Rolls forward to the next weekday open and uses the *target* date's
    tzname() so the suffix flips between CET and CEST automatically
    across the DST boundary.
    """
    from datetime import timedelta

    local_now = now_utc.astimezone(SESSION_TZ)
    target = local_now.replace(
        hour=SESSION_OPEN.hour, minute=SESSION_OPEN.minute,
        second=0, microsecond=0,
    )
    if local_now >= target:
        target = target + timedelta(days=1)
    while target.weekday() >= 5:  # skip weekend(s) into next Monday
        target = target + timedelta(days=1)
    delta = target - local_now
    hours = int(delta.total_seconds() // 3600)
    mins = int((delta.total_seconds() % 3600) // 60)
    zone = target.tzname() or "Stockholm"
    return f"next 15:30 {zone} in {hours}h {mins:02d}m"
