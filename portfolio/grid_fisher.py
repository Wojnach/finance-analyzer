"""Grid market-maker orchestrator for leveraged warrants.

Holds the runtime state of a multi-tier limit ladder per (instrument)
plus the lifecycle around it:

  STARTUP   load state -> reconcile against live Avanza -> resume or reset
  TICK      check signal -> arm/cancel ladder -> place tiers -> detect fills
            -> rotate filled buys into sell+stop -> log -> sweep at EOD
  SHUTDOWN  persist state

This file contains the state-machine + persistence layer. Order placement
and Avanza side effects land in a subsequent batch — the public surface
here is split so the state code is unit-testable in isolation.

State is held in ``data/grid_fisher_state.json`` (atomic JSON via
``file_utils``) and every transition is appended to
``data/grid_fisher_decisions.jsonl``. Both are read by the dashboard for
ops visibility.
"""

from __future__ import annotations

import datetime as _dt
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from portfolio.file_utils import (
    atomic_append_jsonl,
    atomic_write_json,
    load_json,
)
from portfolio.grid_fisher_config import (
    GRID_ACTIVE_INSTRUMENTS,
    GRID_DECISIONS_LOG,
    GRID_DIRECTION_FLIP_COOLDOWN_MIN,
    GRID_PER_INSTRUMENT_MAX_SEK,
    GRID_PER_SESSION_LOSS_LIMIT_SEK,
    GRID_STATE_FILE,
    GRID_STATE_SCHEMA_VERSION,
)

logger = logging.getLogger("portfolio.grid_fisher")


# ---------------------------------------------------------------------------
# Tier / order status enums (string for JSON friendliness)
# ---------------------------------------------------------------------------
ORDER_ARMED = "ARMED"
ORDER_FILLED = "FILLED"
ORDER_CANCELLED = "CANCELLED"
ORDER_REJECTED = "REJECTED"


# ---------------------------------------------------------------------------
# Schema dataclasses (used for in-memory representation; persisted as dicts)
# ---------------------------------------------------------------------------


@dataclass
class TierOrder:
    """One leg in the buy or sell ladder."""

    tier: int
    order_id: Optional[str]
    price: float
    qty: int
    placed_ts: str
    status: str = ORDER_ARMED
    fill_ts: Optional[str] = None
    fill_price: Optional[float] = None
    linked_buy_tier: Optional[int] = None
    p_fill_session: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "order_id": self.order_id,
            "price": self.price,
            "qty": self.qty,
            "placed_ts": self.placed_ts,
            "status": self.status,
            "fill_ts": self.fill_ts,
            "fill_price": self.fill_price,
            "linked_buy_tier": self.linked_buy_tier,
            "p_fill_session": self.p_fill_session,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TierOrder":
        return cls(
            tier=int(d["tier"]),
            order_id=d.get("order_id"),
            price=float(d["price"]),
            qty=int(d["qty"]),
            placed_ts=str(d["placed_ts"]),
            status=str(d.get("status", ORDER_ARMED)),
            fill_ts=d.get("fill_ts"),
            fill_price=d.get("fill_price"),
            linked_buy_tier=d.get("linked_buy_tier"),
            p_fill_session=d.get("p_fill_session"),
        )


@dataclass
class InstrumentState:
    """Per-instrument grid state."""

    ob_id: str
    ticker: str
    cert_name: str
    active_direction: Optional[str] = None  # "LONG"/"SHORT"/None
    buy_ladder: list[TierOrder] = field(default_factory=list)
    sell_ladder: list[TierOrder] = field(default_factory=list)
    stop_loss_id: Optional[str] = None
    stop_loss_price: Optional[float] = None
    inventory_units: int = 0
    avg_entry_price: float = 0.0
    session_pnl_sek: float = 0.0
    fills_this_session: int = 0
    consecutive_losses: int = 0
    cooldown_until: Optional[str] = None
    last_direction_flip_ts: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ob_id": self.ob_id,
            "ticker": self.ticker,
            "cert_name": self.cert_name,
            "active_direction": self.active_direction,
            "buy_ladder": [o.to_dict() for o in self.buy_ladder],
            "sell_ladder": [o.to_dict() for o in self.sell_ladder],
            "stop_loss_id": self.stop_loss_id,
            "stop_loss_price": self.stop_loss_price,
            "inventory_units": self.inventory_units,
            "avg_entry_price": self.avg_entry_price,
            "session_pnl_sek": self.session_pnl_sek,
            "fills_this_session": self.fills_this_session,
            "consecutive_losses": self.consecutive_losses,
            "cooldown_until": self.cooldown_until,
            "last_direction_flip_ts": self.last_direction_flip_ts,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InstrumentState":
        return cls(
            ob_id=str(d["ob_id"]),
            ticker=str(d["ticker"]),
            cert_name=str(d.get("cert_name", "")),
            active_direction=d.get("active_direction"),
            buy_ladder=[TierOrder.from_dict(o) for o in d.get("buy_ladder", [])],
            sell_ladder=[TierOrder.from_dict(o) for o in d.get("sell_ladder", [])],
            stop_loss_id=d.get("stop_loss_id"),
            stop_loss_price=d.get("stop_loss_price"),
            inventory_units=int(d.get("inventory_units", 0) or 0),
            avg_entry_price=float(d.get("avg_entry_price", 0.0) or 0.0),
            session_pnl_sek=float(d.get("session_pnl_sek", 0.0) or 0.0),
            fills_this_session=int(d.get("fills_this_session", 0) or 0),
            consecutive_losses=int(d.get("consecutive_losses", 0) or 0),
            cooldown_until=d.get("cooldown_until"),
            last_direction_flip_ts=d.get("last_direction_flip_ts"),
        )

    # ---- queries -----------------------------------------------------------

    def armed_buy_tiers(self) -> list[TierOrder]:
        return [o for o in self.buy_ladder if o.status == ORDER_ARMED]

    def armed_sell_tiers(self) -> list[TierOrder]:
        return [o for o in self.sell_ladder if o.status == ORDER_ARMED]

    def planned_notional_sek(self) -> float:
        """SEK at risk = armed buys (notional) + inventory @ avg entry."""
        armed = sum(o.price * o.qty for o in self.armed_buy_tiers())
        held = self.inventory_units * self.avg_entry_price
        return armed + held

    def hit_per_instrument_cap(self) -> bool:
        return self.planned_notional_sek() >= GRID_PER_INSTRUMENT_MAX_SEK

    def session_loss_breached(self) -> bool:
        return self.session_pnl_sek <= -abs(GRID_PER_SESSION_LOSS_LIMIT_SEK)

    def in_cooldown(self, now_iso: Optional[str] = None) -> bool:
        if not self.cooldown_until:
            return False
        now_iso = now_iso or _utcnow_iso()
        return now_iso < self.cooldown_until


@dataclass
class GridFisherState:
    """Top-level state container — what lives on disk."""

    version: int = GRID_STATE_SCHEMA_VERSION
    session_id: str = ""
    halted: bool = False
    halt_reason: Optional[str] = None
    global_session_pnl_sek: float = 0.0
    global_max_dd_sek: float = 0.0
    by_instrument: dict[str, InstrumentState] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "session_id": self.session_id,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "global_session_pnl_sek": self.global_session_pnl_sek,
            "global_max_dd_sek": self.global_max_dd_sek,
            "by_instrument": {
                ob: inst.to_dict() for ob, inst in self.by_instrument.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GridFisherState":
        return cls(
            version=int(d.get("version", GRID_STATE_SCHEMA_VERSION)),
            session_id=str(d.get("session_id", "")),
            halted=bool(d.get("halted", False)),
            halt_reason=d.get("halt_reason"),
            global_session_pnl_sek=float(
                d.get("global_session_pnl_sek", 0.0) or 0.0
            ),
            global_max_dd_sek=float(d.get("global_max_dd_sek", 0.0) or 0.0),
            by_instrument={
                ob: InstrumentState.from_dict(inst)
                for ob, inst in d.get("by_instrument", {}).items()
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    """ISO8601 UTC timestamp with 'Z' suffix — matches the rest of the repo."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_session_id() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")


def _seed_state_for_active_instruments(
    state: GridFisherState,
    catalog: dict[str, dict[str, Any]],
) -> None:
    """Ensure every instrument in GRID_ACTIVE_INSTRUMENTS has a record.

    *catalog* maps cert_name -> full metadata (from fin_fish_config.FULL_CATALOG).
    The seeded record is empty (no direction, no orders) — it just gets a
    slot in ``state.by_instrument`` so subsequent tick code can find it.
    """
    cert_by_ob: dict[str, str] = {}
    for cert_name, meta in catalog.items():
        ob = str(meta.get("ob_id") or "")
        if ob:
            cert_by_ob[ob] = cert_name

    for ticker, by_dir in GRID_ACTIVE_INSTRUMENTS.items():
        for direction, ob_id in by_dir.items():
            ob = str(ob_id)
            if ob in state.by_instrument:
                continue
            cert_name = cert_by_ob.get(ob, f"unknown_ob_{ob}")
            state.by_instrument[ob] = InstrumentState(
                ob_id=ob,
                ticker=ticker,
                cert_name=cert_name,
            )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


_state_lock = threading.Lock()  # serialises load/save across threads


def load_state(
    state_path: str | Path = GRID_STATE_FILE,
) -> GridFisherState:
    """Load state from *state_path*. Returns a fresh state on missing/corrupt/old."""
    raw = load_json(state_path, default=None)
    if not isinstance(raw, dict):
        logger.info("grid_fisher: no prior state found at %s", state_path)
        return GridFisherState(session_id=_today_session_id())

    file_version = raw.get("version")
    if file_version != GRID_STATE_SCHEMA_VERSION:
        # Don't silently coerce — start fresh and log so the operator notices.
        logger.warning(
            "grid_fisher: state file %s has version=%r, expected %d — resetting",
            state_path, file_version, GRID_STATE_SCHEMA_VERSION,
        )
        return GridFisherState(session_id=_today_session_id())

    try:
        return GridFisherState.from_dict(raw)
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning(
            "grid_fisher: state file %s is malformed (%s) — resetting",
            state_path, exc,
        )
        return GridFisherState(session_id=_today_session_id())


def save_state(
    state: GridFisherState,
    state_path: str | Path = GRID_STATE_FILE,
) -> None:
    """Atomically persist state to disk."""
    with _state_lock:
        atomic_write_json(state_path, state.to_dict())


def log_decision(
    category: str,
    *,
    ob_id: Optional[str] = None,
    ticker: Optional[str] = None,
    decisions_path: str | Path = GRID_DECISIONS_LOG,
    **fields: Any,
) -> None:
    """Append a decision entry to the grid-fisher journal.

    ``category`` is a short tag (e.g. ``"placement"``, ``"fill"``,
    ``"rotate"``, ``"cancel"``, ``"halt"``, ``"flip"``).
    """
    entry: dict[str, Any] = {
        "ts": _utcnow_iso(),
        "category": category,
    }
    if ob_id is not None:
        entry["ob_id"] = ob_id
    if ticker is not None:
        entry["ticker"] = ticker
    entry.update(fields)
    atomic_append_jsonl(decisions_path, entry)


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def roll_session_if_new_day(state: GridFisherState) -> bool:
    """If the session date has advanced, reset per-session counters.

    Returns ``True`` if the session was rolled (so the caller can persist
    state). Per-instrument inventory / orders are NOT cleared — those
    represent positions that may still be live; the EOD sweep handles
    end-of-day closing separately.
    """
    today = _today_session_id()
    if state.session_id == today:
        return False
    logger.info(
        "grid_fisher: rolling session %s -> %s",
        state.session_id or "<new>",
        today,
    )
    state.session_id = today
    state.global_session_pnl_sek = 0.0
    state.global_max_dd_sek = 0.0
    for inst in state.by_instrument.values():
        inst.session_pnl_sek = 0.0
        inst.fills_this_session = 0
    return True


def flip_direction(
    inst: InstrumentState,
    new_direction: str,
    *,
    cooldown_min: int = GRID_DIRECTION_FLIP_COOLDOWN_MIN,
) -> None:
    """Transition the instrument to a new direction with cooldown.

    The existing inventory + sell ladder are NOT touched — they exit on
    their own. Only the buy ladder gets cleared; the caller is responsible
    for cancelling those orders against Avanza.
    """
    if new_direction not in ("LONG", "SHORT"):
        raise ValueError(f"direction must be LONG or SHORT, got {new_direction!r}")
    if inst.active_direction == new_direction:
        return
    inst.active_direction = new_direction
    inst.last_direction_flip_ts = _utcnow_iso()
    inst.buy_ladder = []  # caller cancels live orders before calling this
    cooldown_dt = _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(minutes=cooldown_min)
    inst.cooldown_until = cooldown_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def record_fill(
    inst: InstrumentState,
    tier_idx: int,
    fill_price: float,
    *,
    side: str = "buy",
    fill_ts: Optional[str] = None,
) -> Optional[TierOrder]:
    """Mark a buy or sell tier as FILLED and update inventory + P&L.

    Returns the updated ``TierOrder`` or ``None`` if no matching armed
    tier was found.
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    ladder = inst.buy_ladder if side == "buy" else inst.sell_ladder
    target: Optional[TierOrder] = None
    for o in ladder:
        if o.tier == tier_idx and o.status == ORDER_ARMED:
            target = o
            break
    if target is None:
        return None

    target.status = ORDER_FILLED
    target.fill_ts = fill_ts or _utcnow_iso()
    target.fill_price = fill_price
    inst.fills_this_session += 1

    if side == "buy":
        # Weighted-average entry
        prev_units = inst.inventory_units
        prev_avg = inst.avg_entry_price
        new_units = prev_units + target.qty
        if new_units > 0:
            inst.avg_entry_price = (
                (prev_avg * prev_units) + (fill_price * target.qty)
            ) / new_units
        inst.inventory_units = new_units
    else:  # sell
        # Realise P&L on the qty exiting
        pnl_per_unit = fill_price - inst.avg_entry_price
        realised = pnl_per_unit * target.qty
        inst.session_pnl_sek += realised
        inst.inventory_units = max(0, inst.inventory_units - target.qty)
        if inst.inventory_units == 0:
            inst.avg_entry_price = 0.0
        if realised < 0:
            inst.consecutive_losses += 1
        else:
            inst.consecutive_losses = 0

    return target


def cancel_buy_tier(
    inst: InstrumentState,
    tier_idx: int,
) -> Optional[TierOrder]:
    """Mark a buy tier as CANCELLED."""
    for o in inst.buy_ladder:
        if o.tier == tier_idx and o.status == ORDER_ARMED:
            o.status = ORDER_CANCELLED
            return o
    return None


def prune_terminal_orders(inst: InstrumentState) -> None:
    """Drop FILLED/CANCELLED/REJECTED tiers from in-memory ladders.

    Persisted decision log retains them. Pruning keeps the in-memory
    ladder bounded so each tick only sees ARMED orders.
    """
    inst.buy_ladder = [o for o in inst.buy_ladder if o.status == ORDER_ARMED]
    inst.sell_ladder = [o for o in inst.sell_ladder if o.status == ORDER_ARMED]


# ---------------------------------------------------------------------------
# Health / safety queries
# ---------------------------------------------------------------------------


def should_halt_global(state: GridFisherState) -> Optional[str]:
    """Return a halt reason string if the global drawdown breaks a limit,
    else None. Caller flips ``state.halted = True`` and logs.

    Currently uses session-wide P&L; richer drawdown tracking lives in
    ``risk_management.py`` and is wired in via a later batch.
    """
    # Per-session global loss limit = sum of per-instrument limits.
    threshold = -abs(GRID_PER_SESSION_LOSS_LIMIT_SEK) * max(
        len(state.by_instrument), 1
    )
    if state.global_session_pnl_sek <= threshold:
        return f"global_session_pnl<{threshold:.0f}sek"
    return None


def summarise(state: GridFisherState) -> dict[str, Any]:
    """Compact dict for dashboard / Telegram. Excludes raw ladders."""
    return {
        "session_id": state.session_id,
        "halted": state.halted,
        "halt_reason": state.halt_reason,
        "global_session_pnl_sek": round(state.global_session_pnl_sek, 2),
        "by_instrument": {
            ob: {
                "ticker": inst.ticker,
                "cert_name": inst.cert_name,
                "active_direction": inst.active_direction,
                "armed_buys": len(inst.armed_buy_tiers()),
                "armed_sells": len(inst.armed_sell_tiers()),
                "inventory_units": inst.inventory_units,
                "avg_entry_price": round(inst.avg_entry_price, 4),
                "session_pnl_sek": round(inst.session_pnl_sek, 2),
                "fills_this_session": inst.fills_this_session,
                "cooldown_until": inst.cooldown_until,
            }
            for ob, inst in state.by_instrument.items()
        },
    }
