"""Shared state for MSTR Loop — cash + per-strategy positions.

Each strategy gets its own position slot (keyed by strategy_key) so multiple
strategies can coexist without clobbering. Same pattern as
metals_swing_trader's positions dict, generalized to any strategy.

Atomic I/O only — never raw json.dump. Corrupt state file falls back to a
fresh default rather than crashing the loop.
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
import os
from typing import Any

from portfolio.file_utils import atomic_write_json, load_json

from portfolio.mstr_loop import config

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Position:
    """A single open position held by one strategy."""
    strategy_key: str
    direction: str              # "LONG" or "SHORT"
    cert_ob_id: str             # which Avanza instrument
    entry_underlying_price: float   # MSTR spot at entry
    entry_cert_price: float     # cert ask price at fill
    units: int                  # cert units held (CURRENT — decrements on partial exits)
    entry_ts: str               # ISO-8601 UTC
    trail_active: bool = False  # switched on at +trail_activation_pct
    peak_underlying_price: float = 0.0  # highest underlying since entry
    stop_price_cert: float | None = None  # broker-side stop price (Phase D)
    rationale: str = ""         # why we entered (audit trail)
    # Partial-exit ladder (v2 Tier 2)
    entry_units: int = 0                  # original units at entry (for tranche math)
    units_sold: int = 0                   # cumulative units exited via tranches
    tranches_hit: list[float] = dataclasses.field(default_factory=list)  # profit_pct values already fired

    def unrealized_underlying_pct(self, current_underlying_price: float) -> float:
        """Underlying price change % since entry (direction-aware)."""
        if self.entry_underlying_price <= 0:
            return 0.0
        raw = (current_underlying_price - self.entry_underlying_price) / self.entry_underlying_price * 100
        return raw if self.direction == "LONG" else -raw


@dataclasses.dataclass
class BotState:
    """Shared state across all strategies in the loop."""
    cash_sek: float = 0.0
    positions: dict[str, Position] = dataclasses.field(default_factory=dict)
    # Per-strategy metadata
    last_exit_ts: dict[str, str] = dataclasses.field(default_factory=dict)
    # Running counters (for scorecard + telegram reports)
    total_trades: int = 0
    total_pnl_sek: float = 0.0
    wins: int = 0
    losses: int = 0
    # Bot-level audit
    session_started_ts: str = ""
    last_cycle_ts: str = ""
    # Drawdown circuit breaker (v2 Tier 1)
    peak_equity_sek: float = 0.0          # highest equity observed across history
    session_start_equity_sek: float = 0.0  # equity at the start of the current US session (for daily %)
    week_start_equity_sek: float = 0.0     # equity at start of rolling 7d window
    session_start_ts: str = ""             # resets daily at session-open
    week_start_ts: str = ""                # resets weekly
    daily_halted_until: str = ""           # ISO timestamp when daily halt lifts
    weekly_halted_until: str = ""          # ISO timestamp when weekly halt lifts

    def has_position(self, strategy_key: str) -> bool:
        return strategy_key in self.positions

    def get_position(self, strategy_key: str) -> Position | None:
        return self.positions.get(strategy_key)

    def add_position(self, pos: Position) -> None:
        self.positions[pos.strategy_key] = pos

    def remove_position(self, strategy_key: str) -> Position | None:
        return self.positions.pop(strategy_key, None)

    def cooldown_elapsed(self, strategy_key: str, minutes: int) -> bool:
        """True if last exit (if any) is older than `minutes` — entry allowed."""
        last = self.last_exit_ts.get(strategy_key)
        if not last:
            return True
        try:
            last_dt = datetime.datetime.fromisoformat(last)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=datetime.UTC)
        except (TypeError, ValueError):
            return True
        elapsed = (datetime.datetime.now(datetime.UTC) - last_dt).total_seconds() / 60
        return elapsed >= minutes


def default_state() -> BotState:
    """Return a fresh state with the configured starting cash per phase."""
    cash = 0.0
    if config.PHASE == "paper":
        cash = float(config.INITIAL_PAPER_CASH_SEK)
    elif config.PHASE == "live":
        # Live cash is synced from Avanza at loop startup; default to 0 here
        # so a pre-sync entry attempt is blocked by the Kelly cash-check.
        cash = 0.0
    return BotState(
        cash_sek=cash,
        session_started_ts=datetime.datetime.now(datetime.UTC).isoformat(),
    )


def _position_to_dict(pos: Position) -> dict[str, Any]:
    return dataclasses.asdict(pos)


def _position_from_dict(d: dict[str, Any]) -> Position:
    allowed = {f.name for f in dataclasses.fields(Position)}
    return Position(**{k: v for k, v in d.items() if k in allowed})


def _state_to_dict(state: BotState) -> dict[str, Any]:
    return {
        "cash_sek": state.cash_sek,
        "positions": {k: _position_to_dict(v) for k, v in state.positions.items()},
        "last_exit_ts": dict(state.last_exit_ts),
        "total_trades": state.total_trades,
        "total_pnl_sek": state.total_pnl_sek,
        "wins": state.wins,
        "losses": state.losses,
        "session_started_ts": state.session_started_ts,
        "last_cycle_ts": state.last_cycle_ts,
        "peak_equity_sek": state.peak_equity_sek,
        "session_start_equity_sek": state.session_start_equity_sek,
        "week_start_equity_sek": state.week_start_equity_sek,
        "session_start_ts": state.session_start_ts,
        "week_start_ts": state.week_start_ts,
        "daily_halted_until": state.daily_halted_until,
        "weekly_halted_until": state.weekly_halted_until,
    }


def _state_from_dict(d: dict[str, Any]) -> BotState:
    raw_positions = d.get("positions") or {}
    positions: dict[str, Position] = {}
    for key, pos_dict in raw_positions.items():
        try:
            positions[key] = _position_from_dict(pos_dict)
        except (TypeError, ValueError):
            logger.warning("state: dropping malformed position %r", key, exc_info=True)
    return BotState(
        cash_sek=float(d.get("cash_sek") or 0.0),
        positions=positions,
        last_exit_ts=dict(d.get("last_exit_ts") or {}),
        total_trades=int(d.get("total_trades") or 0),
        total_pnl_sek=float(d.get("total_pnl_sek") or 0.0),
        wins=int(d.get("wins") or 0),
        losses=int(d.get("losses") or 0),
        session_started_ts=str(d.get("session_started_ts") or ""),
        last_cycle_ts=str(d.get("last_cycle_ts") or ""),
        peak_equity_sek=float(d.get("peak_equity_sek") or 0.0),
        session_start_equity_sek=float(d.get("session_start_equity_sek") or 0.0),
        week_start_equity_sek=float(d.get("week_start_equity_sek") or 0.0),
        session_start_ts=str(d.get("session_start_ts") or ""),
        week_start_ts=str(d.get("week_start_ts") or ""),
        daily_halted_until=str(d.get("daily_halted_until") or ""),
        weekly_halted_until=str(d.get("weekly_halted_until") or ""),
    )


def load_state(path: str = config.STATE_FILE) -> BotState:
    """Load state from disk, fall back to `default_state()` on any error.

    Non-fatal on corrupt JSON — we log and use defaults rather than crash
    the loop (bootstrap problem: a broken state file during Phase C would
    otherwise prevent the bot from ever running again without manual fix).
    """
    if not os.path.exists(path):
        return default_state()
    try:
        raw = load_json(path)
        if not isinstance(raw, dict):
            logger.warning("state: %s is not a dict, using defaults", path)
            return default_state()
        return _state_from_dict(raw)
    except Exception:
        logger.exception("state: load failed for %s, using defaults", path)
        return default_state()


def save_state(state: BotState, path: str = config.STATE_FILE) -> None:
    """Persist state atomically. Non-fatal on write failure (logs only)."""
    try:
        atomic_write_json(path, _state_to_dict(state), ensure_ascii=False)
    except Exception:
        logger.exception("state: save failed for %s", path)
