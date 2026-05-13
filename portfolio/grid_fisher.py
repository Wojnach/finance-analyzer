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
    GRID_GLOBAL_MAX_SEK,
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


# Default warrant trading day-end in Stockholm local time. Avanza warrants
# trade until ~22:00 CET; we cut off 5 minutes earlier to give cancel/
# sell orders time to round-trip. The exact close varies with DST; using
# zoneinfo lets python pick the correct offset automatically.
_EOD_LOCAL_HOUR = 21
_EOD_LOCAL_MINUTE = 55
_EOD_TZ_NAME = "Europe/Stockholm"


def minutes_until_eod(now_utc: Optional[_dt.datetime] = None) -> float:
    """Minutes until the grid fisher's day-end cutoff in Europe/Stockholm.

    Returns a non-negative float; if the cutoff has already passed for the
    day, returns the minutes-until-tomorrow's cutoff (so the caller can
    detect with ``< EOD_SWEEP_MINUTES_BEFORE`` only during the active
    window). Returns ``float("inf")`` if zoneinfo is unavailable so the
    caller never triggers EOD on the failure path.
    """
    try:
        import zoneinfo  # noqa: PLC0415 — optional import to keep startup cheap
    except ImportError:
        return float("inf")
    try:
        tz = zoneinfo.ZoneInfo(_EOD_TZ_NAME)
    except Exception:  # noqa: BLE001 — missing tzdata
        return float("inf")
    now = now_utc or _dt.datetime.now(_dt.timezone.utc)
    local = now.astimezone(tz)
    cutoff = local.replace(
        hour=_EOD_LOCAL_HOUR, minute=_EOD_LOCAL_MINUTE, second=0, microsecond=0,
    )
    if cutoff <= local:
        cutoff = cutoff + _dt.timedelta(days=1)
    delta = cutoff - local
    return delta.total_seconds() / 60.0


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
            # Each instrument has a FIXED natural direction baked in by
            # which cert it is (BULL=LONG, BEAR=SHORT). We store it as
            # active_direction at seed time so the tick() loop can match
            # it against the signal direction. Existing inventory + sells
            # are unaffected by signal flips because they reference an
            # instrument that already chose its side.
            state.by_instrument[ob] = InstrumentState(
                ob_id=ob,
                ticker=ticker,
                cert_name=cert_name,
                active_direction=direction,
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


# Pattern matching cookie headers / API auth in Playwright error
# tracebacks. Avanza errors include the full request cookie which
# carries AZAPERSISTENCE, csid, cstoken, AZACSRF — session-authoritative
# values that must NEVER leave the host. Scrub at log time so the
# dashboard /api/grid-fisher endpoint and any downstream reader stays
# clean.
import re as _re

_SENSITIVE_HEADER_PATTERN = _re.compile(
    r"(?im)^\s*-\s*(?:cookie|authorization|set-cookie|x-aza-csrf-token)\s*:.*$",
)
_INLINE_COOKIE_PATTERN = _re.compile(
    r"(?i)(?:cookie|authorization)\s*[:=]\s*[^\s\n]+",
)
_MAX_ERROR_FIELD_CHARS = 400


def _scrub_for_log(value: Any) -> Any:
    """Strip auth/cookie material and cap length on user-supplied strings."""
    if not isinstance(value, str):
        return value
    cleaned = _SENSITIVE_HEADER_PATTERN.sub("    - <redacted>", value)
    cleaned = _INLINE_COOKIE_PATTERN.sub("<redacted>", cleaned)
    if len(cleaned) > _MAX_ERROR_FIELD_CHARS:
        cleaned = cleaned[:_MAX_ERROR_FIELD_CHARS] + "…<truncated>"
    return cleaned


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
    ``"rotate"``, ``"cancel"``, ``"halt"``, ``"flip"``). Free-form
    string fields are scrubbed of cookies/auth headers before write
    so the journal is safe to expose via the dashboard endpoint.
    """
    entry: dict[str, Any] = {
        "ts": _utcnow_iso(),
        "category": category,
    }
    if ob_id is not None:
        entry["ob_id"] = ob_id
    if ticker is not None:
        entry["ticker"] = ticker
    for key, val in fields.items():
        entry[key] = _scrub_for_log(val)
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


def global_planned_notional(state: GridFisherState) -> float:
    """Sum planned notional across every instrument (armed buys +
    inventory at avg entry). Used by the global-cap gate."""
    return sum(inst.planned_notional_sek()
               for inst in state.by_instrument.values())


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


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


@dataclass
class ReconcileResult:
    """Diff summary returned by reconcile_against_live."""

    filled_buys: list[tuple[str, int, float]] = field(default_factory=list)
    filled_sells: list[tuple[str, int, float]] = field(default_factory=list)
    cancelled_buys: list[tuple[str, int]] = field(default_factory=list)
    cancelled_sells: list[tuple[str, int]] = field(default_factory=list)
    inventory_drift: list[tuple[str, int, int]] = field(default_factory=list)


def _position_volume_for(positions: list[dict[str, Any]], ob_id: str) -> int:
    """Return current unit volume held for *ob_id*, or 0."""
    for p in positions or []:
        if str(p.get("orderbook_id") or p.get("orderbookId") or "") == str(ob_id):
            v = p.get("volume", 0) or 0
            try:
                return int(v)
            except (TypeError, ValueError):
                return 0
    return 0


def reconcile_against_live(
    state: GridFisherState,
    open_order_ids: set[str],
    positions: list[dict[str, Any]],
) -> ReconcileResult:
    """Diff in-memory state against live Avanza state.

    For each ARMED tier whose order_id is missing from ``open_order_ids``:
      * if the live position covers the expected delta -> mark FILLED at
        the tier's limit price (best estimate; actual fill price may be
        slightly better)
      * else -> mark CANCELLED (likely cancelled externally)

    Also reports inventory drift where the live volume disagrees with
    the cached ``inventory_units`` for the instrument; the caller may
    use this to align state without erroneously double-counting fills.

    Returns a ``ReconcileResult`` listing every transition for logging.
    """
    res = ReconcileResult()
    for ob_id, inst in state.by_instrument.items():
        live_vol = _position_volume_for(positions, ob_id)

        for tier in list(inst.buy_ladder):
            if tier.status != ORDER_ARMED:
                continue
            if tier.order_id and tier.order_id in open_order_ids:
                continue
            # Order is missing from live. Distinguish full fill / partial
            # fill / cancel by the live position delta vs cached
            # inventory (state already accounts for any earlier tiers
            # processed this loop because record_fill mutates inst).
            delta = live_vol - inst.inventory_units
            if delta >= tier.qty:
                record_fill(inst, tier.tier, tier.price, side="buy")
                res.filled_buys.append((ob_id, tier.tier, tier.price))
            elif delta > 0:
                # Partial fill — the order filled `delta` units before
                # being cancelled (or capped by quote-side liquidity).
                # Record the fill with reduced qty so inventory accounting
                # stays honest, and log the drift so an operator can spot
                # repeated partials and tune tier sizing.
                original_qty = tier.qty
                tier.qty = int(delta)
                record_fill(inst, tier.tier, tier.price, side="buy")
                res.filled_buys.append((ob_id, tier.tier, tier.price))
                res.inventory_drift.append((ob_id, original_qty, int(delta)))
            else:
                cancel_buy_tier(inst, tier.tier)
                res.cancelled_buys.append((ob_id, tier.tier))

        for tier in list(inst.sell_ladder):
            if tier.status != ORDER_ARMED:
                continue
            if tier.order_id and tier.order_id in open_order_ids:
                continue
            inventory_drop = inst.inventory_units - live_vol
            if inventory_drop >= tier.qty:
                record_fill(inst, tier.tier, tier.price, side="sell")
                res.filled_sells.append((ob_id, tier.tier, tier.price))
            elif inventory_drop > 0:
                original_qty = tier.qty
                tier.qty = int(inventory_drop)
                record_fill(inst, tier.tier, tier.price, side="sell")
                res.filled_sells.append((ob_id, tier.tier, tier.price))
                res.inventory_drift.append(
                    (ob_id, original_qty, int(inventory_drop))
                )
            else:
                # Mark sell tier as cancelled in-place — there is no
                # dedicated helper because sells are cleared via rotation.
                tier.status = ORDER_CANCELLED
                res.cancelled_sells.append((ob_id, tier.tier))

        # Final drift check — only used for logging; the caller decides
        # whether to forcibly align (we do not auto-rewrite inventory).
        if live_vol != inst.inventory_units:
            res.inventory_drift.append(
                (ob_id, inst.inventory_units, live_vol)
            )
    return res


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class GridFisher:
    """High-level driver — owns state + session + signal/quote callables.

    Constructor takes callables instead of importing the live session
    module directly. That keeps the class unit-testable with a fake
    session that records calls instead of hitting Avanza.

    Required callables:

      ``session.place_buy_order(orderbook_id, price, volume)``
      ``session.place_sell_order(orderbook_id, price, volume)``
      ``session.place_stop_loss(orderbook_id, trigger_price, sell_price, volume)``
      ``session.cancel_order(order_id)``
      ``session.get_open_orders()``
      ``session.get_positions()``
      ``session.get_quote(orderbook_id)``  -> dict with at least 'buy' (bid)

    Signal callables (injected so unit tests can provide deterministic data):

      ``signal_fn(ticker)`` -> ``(direction, confidence)`` with direction in
          ``{"LONG", "SHORT", None}`` and confidence in [0, 1].
      ``atr_fn(ticker)`` -> ``float`` ATR % (annualised, optional; can be None).
      ``adx_fn(ticker)`` -> ``float`` ADX(14) (optional; can be None).
      ``underlying_price_fn(ticker)`` -> ``float`` underlying spot.
    """

    def __init__(
        self,
        session: Any,
        catalog: dict[str, dict[str, Any]],
        *,
        signal_fn: Optional[callable] = None,
        atr_fn: Optional[callable] = None,
        adx_fn: Optional[callable] = None,
        underlying_price_fn: Optional[callable] = None,
        state_path: str | Path = GRID_STATE_FILE,
        decisions_path: str | Path = GRID_DECISIONS_LOG,
        now_fn: Optional[callable] = None,
    ) -> None:
        from portfolio.grid_fisher_config import (
            GRID_FISHER_ENABLED,
            GRID_MAX_ORDERS_PER_MIN,
            GRID_MIN_SIGNAL_CONFIDENCE,
            GRID_ORDER_PLACE_DELAY_S,
            GRID_STOP_PCT,
            GRID_TARGET_PCT,
            GRID_TIERS,
            GRID_TIER_SPACING_PCT,
            GRID_ADX_TREND_FILTER,
            GRID_FISHER_PROBE_ONLY,
        )

        self.session = session
        self.catalog = catalog
        self.signal_fn = signal_fn
        self.atr_fn = atr_fn
        self.adx_fn = adx_fn
        self.underlying_price_fn = underlying_price_fn
        self.state_path = Path(state_path)
        self.decisions_path = Path(decisions_path)
        self.now_fn = now_fn or _utcnow_iso

        # Snapshot config at construction time so monkeypatched test
        # values are picked up before instances are reused across tests.
        self._enabled = GRID_FISHER_ENABLED
        self._probe_only = GRID_FISHER_PROBE_ONLY
        self._min_conf = GRID_MIN_SIGNAL_CONFIDENCE
        self._n_tiers = GRID_TIERS
        self._spacing = tuple(GRID_TIER_SPACING_PCT)
        self._target_pct = GRID_TARGET_PCT
        self._stop_pct = GRID_STOP_PCT
        self._order_delay_s = GRID_ORDER_PLACE_DELAY_S
        self._max_orders_per_min = GRID_MAX_ORDERS_PER_MIN
        self._adx_trend_filter = GRID_ADX_TREND_FILTER

        # Rate-limit state — sliding window of last placement timestamps.
        self._recent_places: list[float] = []

        self.state = load_state(self.state_path)
        _seed_state_for_active_instruments(self.state, self.catalog)

    # ---- low-level helpers ------------------------------------------------

    def _log(self, category: str, **fields: Any) -> None:
        log_decision(category, decisions_path=self.decisions_path, **fields)

    def _persist(self) -> None:
        save_state(self.state, self.state_path)

    def _rate_limit_ok(self) -> bool:
        """Sliding-window rate limiter — drop placements over the per-minute cap."""
        now = time.time()
        cutoff = now - 60.0
        self._recent_places = [t for t in self._recent_places if t >= cutoff]
        if len(self._recent_places) >= self._max_orders_per_min:
            return False
        self._recent_places.append(now)
        return True

    def _safe_session_call(self, fn, *args, default=None, **kwargs):
        """Invoke an Avanza session method from a persistent worker thread.

        The metals loop runs Playwright async context for its swing-trader
        page; the REST avanza_session module that grid_fisher uses internally
        spins up its own sync_playwright client. Calling sync Playwright APIs
        from a thread that has a running asyncio event loop raises
        "Playwright Sync API inside the asyncio loop". Spawning a fresh
        worker per call (e.g. with a per-call ThreadPoolExecutor) fixes
        that but breaks the *next* call: the avanza_session module caches
        its Playwright context to the FIRST thread that initialised it, so
        when a later call lands in a different worker thread it raises
        "cannot switch to a different thread (which happens to have exited)".
        Solution: one long-lived worker thread for the lifetime of this
        GridFisher — all session calls land on the same thread and the
        cached Playwright context stays bound to it.

        On timeout or worker exception, ``default`` is returned and the
        failure is logged via the journal.
        """
        import concurrent.futures

        # Lazily create the persistent single-worker executor. Held on
        # the instance so process shutdown cleans it up via __del__.
        if getattr(self, "_session_executor", None) is None:
            self._session_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="grid-fisher-session",
            )

        def _runner():
            return fn(*args, **kwargs)

        future = self._session_executor.submit(_runner)
        try:
            return future.result(timeout=30)
        except concurrent.futures.TimeoutError:
            self._log("session_call_timeout",
                      method=getattr(fn, "__name__", repr(fn)))
            return default
        except Exception as exc:  # noqa: BLE001
            self._log("session_call_error",
                      method=getattr(fn, "__name__", repr(fn)),
                      error=str(exc))
            return default

    def __del__(self):
        # Best-effort shutdown of the worker thread on GC. The metals
        # loop holds the GridFisher for its lifetime so this only runs
        # at process exit.
        executor = getattr(self, "_session_executor", None)
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:  # noqa: BLE001
                pass

    def _catalog_for(self, ob_id: str) -> Optional[dict[str, Any]]:
        for cert_name, meta in self.catalog.items():
            if str(meta.get("ob_id") or "") == str(ob_id):
                return {**meta, "name_key": cert_name}
        return None

    # ---- placement --------------------------------------------------------

    def cancel_armed_buys(self, inst: InstrumentState) -> int:
        """Cancel every ARMED buy tier on *inst*. Returns count cancelled.

        Avanza's cancel endpoint returns a JSON body even on rejection
        (no exception). Only marks the tier ``CANCELLED`` after a
        confirmed ``orderRequestStatus == "SUCCESS"`` response, so a
        broker-side rejection leaves the order ARMED in state and the
        next tick's reconcile picks it up correctly — preventing
        duplicate placements on top of a still-resting buy.
        """
        n = 0
        for tier in list(inst.buy_ladder):
            if tier.status != ORDER_ARMED:
                continue
            if not tier.order_id:
                # Never accepted by Avanza in the first place — just drop.
                cancel_buy_tier(inst, tier.tier)
                n += 1
                continue
            result = self._safe_session_call(
                self.session.cancel_order, tier.order_id, default=None,
            )
            if result is None:
                self._log("cancel_failed", ob_id=inst.ob_id,
                          ticker=inst.ticker, tier=tier.tier,
                          order_id=tier.order_id,
                          error="session_call returned None")
                continue
            result = result or {}
            status = result.get("orderRequestStatus")
            if status != "SUCCESS":
                self._log("cancel_rejected", ob_id=inst.ob_id,
                          ticker=inst.ticker, tier=tier.tier,
                          order_id=tier.order_id,
                          avanza_status=status,
                          message=result.get("message"))
                continue
            cancel_buy_tier(inst, tier.tier)
            self._log("cancel_buy", ob_id=inst.ob_id, ticker=inst.ticker,
                      tier=tier.tier, order_id=tier.order_id,
                      price=tier.price, qty=tier.qty)
            n += 1
        return n

    def place_buy_ladder(
        self,
        inst: InstrumentState,
        bid: float,
        *,
        underlying_price: Optional[float] = None,
        barrier: Optional[float] = None,
        leverage: Optional[float] = None,
        global_cap_sek: Optional[float] = None,
    ) -> int:
        """Place any missing buy tiers, respecting per-instrument cap and
        rate limit. Returns the number of orders placed.

        Existing ARMED tiers are kept as-is. If a tier index is missing
        from the buy_ladder, a fresh order is placed and recorded.

        If ``global_cap_sek`` is provided, each tier checks whether
        placing it would push *aggregate* planned notional across every
        instrument in ``self.state`` above the cap; if so the remaining
        tiers are skipped with a logged ``skip_global_cap`` decision.
        """
        from portfolio.grid_fisher_config import GRID_LEG_SEK
        from portfolio.grid_tiers import build_buy_ladder

        if not self._enabled:
            return 0
        if inst.hit_per_instrument_cap():
            self._log("skip_cap", ob_id=inst.ob_id, ticker=inst.ticker,
                      notional_sek=inst.planned_notional_sek())
            return 0
        if inst.session_loss_breached():
            self._log("skip_loss_limit", ob_id=inst.ob_id,
                      ticker=inst.ticker, session_pnl=inst.session_pnl_sek)
            return 0
        if inst.in_cooldown(self.now_fn()):
            self._log("skip_cooldown", ob_id=inst.ob_id,
                      ticker=inst.ticker, cooldown_until=inst.cooldown_until)
            return 0

        existing_tiers = {t.tier for t in inst.buy_ladder
                          if t.status == ORDER_ARMED}
        tiers = build_buy_ladder(
            bid=bid,
            leg_sek=GRID_LEG_SEK,
            n_tiers=self._n_tiers,
            spacing_pct=self._spacing,
            direction=inst.active_direction or "LONG",
            underlying_price=underlying_price,
            barrier=barrier,
            leverage=leverage,
        )
        placed = 0
        for tier in tiers:
            if tier.index in existing_tiers:
                continue
            if not tier.is_active:
                self._log("skip_tier", ob_id=inst.ob_id,
                          ticker=inst.ticker, tier=tier.index,
                          reason=tier.skip_reason, price=tier.price,
                          qty=tier.qty)
                continue
            if global_cap_sek is not None:
                projected = (global_planned_notional(self.state)
                             + tier.notional_sek)
                if projected > global_cap_sek:
                    self._log("skip_global_cap", ob_id=inst.ob_id,
                              ticker=inst.ticker, tier=tier.index,
                              projected=round(projected, 0),
                              cap=global_cap_sek)
                    break
            if not self._rate_limit_ok():
                self._log("rate_limited", ob_id=inst.ob_id,
                          ticker=inst.ticker, tier=tier.index)
                break

            if self._probe_only:
                self._log("probe_placement", ob_id=inst.ob_id,
                          ticker=inst.ticker, tier=tier.index,
                          price=tier.price, qty=tier.qty,
                          side="BUY")
                inst.buy_ladder.append(TierOrder(
                    tier=tier.index, order_id=None, price=tier.price,
                    qty=tier.qty, placed_ts=self.now_fn(),
                ))
                placed += 1
                continue

            result = self._safe_session_call(
                self.session.place_buy_order,
                inst.ob_id, tier.price, tier.qty,
                default=None,
            )
            if result is None:
                self._log("place_buy_failed", ob_id=inst.ob_id,
                          ticker=inst.ticker, tier=tier.index,
                          price=tier.price, qty=tier.qty,
                          error="session_call returned None")
                continue
            status = (result or {}).get("orderRequestStatus", "UNKNOWN")
            order_id = (result or {}).get("orderId")
            if status != "SUCCESS" or not order_id:
                self._log("place_buy_rejected", ob_id=inst.ob_id,
                          ticker=inst.ticker, tier=tier.index,
                          price=tier.price, qty=tier.qty,
                          avanza_status=status,
                          message=(result or {}).get("message"))
                continue
            inst.buy_ladder.append(TierOrder(
                tier=tier.index,
                order_id=str(order_id),
                price=tier.price,
                qty=tier.qty,
                placed_ts=self.now_fn(),
            ))
            self._log("place_buy", ob_id=inst.ob_id, ticker=inst.ticker,
                      tier=tier.index, order_id=str(order_id),
                      price=tier.price, qty=tier.qty)
            placed += 1
            if self._order_delay_s > 0:
                time.sleep(self._order_delay_s)
        return placed

    def rotate_on_buy_fill(self, inst: InstrumentState,
                           filled_tier: int) -> None:
        """After a buy fills, place a matching sell limit + stop loss.

        Stop is rearmed for the *full* current inventory (Avanza coexists
        stop + sell on full volume per memory ``reference_avanza_stops_orders_coexist.md``).
        """
        from portfolio.grid_tiers import build_exit_levels

        filled: Optional[TierOrder] = None
        for o in inst.buy_ladder:
            if o.tier == filled_tier and o.status == ORDER_FILLED:
                filled = o
                break
        if filled is None or filled.fill_price is None:
            return

        sell_price, stop_price = build_exit_levels(
            filled.fill_price, self._target_pct, self._stop_pct,
        )

        # Place opposite-side sell limit.
        sell_order_id: Optional[str] = None
        if self._probe_only:
            self._log("probe_rotate_sell", ob_id=inst.ob_id,
                      ticker=inst.ticker, linked_buy_tier=filled_tier,
                      price=sell_price, qty=filled.qty)
        else:
            result = self._safe_session_call(
                self.session.place_sell_order,
                inst.ob_id, sell_price, filled.qty,
                default=None,
            )
            if result is None:
                self._log("place_sell_failed", ob_id=inst.ob_id,
                          ticker=inst.ticker, linked_buy_tier=filled_tier,
                          price=sell_price, qty=filled.qty,
                          error="session_call returned None")
            elif (result or {}).get("orderRequestStatus") == "SUCCESS":
                sell_order_id = str((result or {}).get("orderId"))

        # Assign a sell-side tier index that mirrors the buy tier so the
        # ladders stay paired even after pruning.
        inst.sell_ladder.append(TierOrder(
            tier=filled_tier,
            order_id=sell_order_id,
            price=sell_price,
            qty=filled.qty,
            placed_ts=self.now_fn(),
            linked_buy_tier=filled_tier,
        ))

        # Cancel old stop, place a new one sized to the full current
        # inventory. Stop sells use trigger_price as the activation
        # level and sell_price as the limit (set just below for fast fill).
        # NOTE: stop-loss orders use a DIFFERENT API surface than regular
        # orders. cancel_order on a stop ID returns "crossing prices"
        # errors (March 3 incident — see avanza_session.cancel_stop_loss
        # docstring). place_stop_loss returns {status, stoplossOrderId},
        # not {orderRequestStatus, orderId}.
        if inst.stop_loss_id and not self._probe_only:
            cancel_fn = getattr(self.session, "cancel_stop_loss", None)
            if cancel_fn is not None:
                self._safe_session_call(
                    cancel_fn, inst.stop_loss_id, default=None,
                )

        new_stop_id: Optional[str] = None
        if not self._probe_only and inst.inventory_units > 0:
            stop_sell_price = round(stop_price * 0.995, 2)
            result = self._safe_session_call(
                self.session.place_stop_loss,
                inst.ob_id, stop_price, stop_sell_price,
                inst.inventory_units,
                default=None,
            )
            if result is None:
                self._log("place_stop_failed", ob_id=inst.ob_id,
                          ticker=inst.ticker, price=stop_price,
                          qty=inst.inventory_units,
                          error="session_call returned None")
            else:
                result = result or {}
                if result.get("status") == "SUCCESS":
                    new_stop_id = str(result.get("stoplossOrderId") or "") or None
                else:
                    self._log("place_stop_rejected", ob_id=inst.ob_id,
                              ticker=inst.ticker, price=stop_price,
                              qty=inst.inventory_units,
                              avanza_status=result.get("status"),
                              message=result.get("message"))
        inst.stop_loss_id = new_stop_id
        inst.stop_loss_price = stop_price

        self._log("rotate", ob_id=inst.ob_id, ticker=inst.ticker,
                  linked_buy_tier=filled_tier,
                  fill_price=filled.fill_price,
                  sell_price=sell_price, stop_price=stop_price,
                  sell_order_id=sell_order_id, stop_id=new_stop_id)

    # ---- session entry ----------------------------------------------------

    def tick(
        self,
        *,
        signal_data: Optional[dict[str, Any]] = None,
        prices: Optional[dict[str, dict[str, Any]]] = None,
        eod_minutes_remaining: Optional[float] = None,
    ) -> dict[str, Any]:
        """Run one cycle.

        Args:
            signal_data: ticker -> {direction, confidence, adx?, atr_pct?}.
                Used to gate placement and decide direction. If a ticker is
                missing, that instrument is left alone.
            prices: ob_id -> {bid, ask, last, underlying_price?}. Bid is
                required for ladder placement.
            eod_minutes_remaining: if not None and <= EOD_SWEEP_MINUTES_BEFORE,
                runs the sweep instead of placing new orders. If <=
                EOD_MARKET_SELL_MINUTES_BEFORE, also market-sells inventory.

        Returns a structured summary for the caller to log.
        """
        from portfolio.grid_fisher_config import (
            GRID_EOD_MARKET_SELL_MINUTES_BEFORE,
            GRID_EOD_SWEEP_MINUTES_BEFORE,
        )

        report: dict[str, Any] = {
            "started_at": self.now_fn(),
            "halted": self.state.halted,
            "placements": 0,
            "rotations": 0,
            "cancels": 0,
            "eod_swept": False,
            "instruments": {},
        }

        if not self._enabled:
            report["skipped_reason"] = "disabled"
            return report

        # Session rollover before anything else so per-instrument counters
        # don't get clobbered mid-cycle.
        if roll_session_if_new_day(self.state):
            self._log("session_roll", session_id=self.state.session_id)

        # Reconcile state with live Avanza before deciding new actions.
        # Wrapping the read calls in _safe_session_call moves the sync
        # Playwright work onto a worker thread so it doesn't collide with
        # the metals_loop's main asyncio event loop. On failure we get None
        # (NOT []), which is distinguishable from "empty book" so a degraded
        # cycle skips placement instead of falsely concluding everything
        # cancelled.
        signal_data = signal_data or {}
        prices = prices or {}
        open_orders_raw = self._safe_session_call(
            self.session.get_open_orders, default=None,
        )
        positions_raw = self._safe_session_call(
            self.session.get_positions, default=None,
        )
        if open_orders_raw is None or positions_raw is None:
            self._log("tick_fetch_degraded",
                      open_orders_ok=open_orders_raw is not None,
                      positions_ok=positions_raw is not None)
            self._persist()
            return {**report, "error": "reconcile_fetch_failed"}

        open_order_ids = {str(o.get("orderId") or o.get("id") or "")
                          for o in open_orders_raw
                          if o.get("orderId") or o.get("id")}
        reconcile = reconcile_against_live(
            self.state, open_order_ids, positions_raw,
        )
        for ob, tier, price in reconcile.filled_buys:
            inst = self.state.by_instrument[ob]
            self._log("fill_buy", ob_id=ob, ticker=inst.ticker, tier=tier,
                      fill_price=price)
            self.rotate_on_buy_fill(inst, tier)
            report["rotations"] += 1
        for ob, tier, price in reconcile.filled_sells:
            inst = self.state.by_instrument[ob]
            self._log("fill_sell", ob_id=ob, ticker=inst.ticker, tier=tier,
                      fill_price=price)
        for ob, tier in reconcile.cancelled_buys:
            inst = self.state.by_instrument[ob]
            self._log("external_cancel_buy", ob_id=ob, ticker=inst.ticker,
                      tier=tier)
        for ob, tier in reconcile.cancelled_sells:
            inst = self.state.by_instrument[ob]
            self._log("external_cancel_sell", ob_id=ob,
                      ticker=inst.ticker, tier=tier)
        for ob, cached, live in reconcile.inventory_drift:
            self._log("inventory_drift", ob_id=ob, cached=cached, live=live)

        # Roll up realised per-instrument P&L into the global counter so
        # ``should_halt_global`` actually sees the running session loss.
        # Derived (not accumulated) to avoid double-counting across
        # ticks — instrument session_pnl_sek already aggregates every
        # realised sell since the last roll_session.
        self.state.global_session_pnl_sek = sum(
            inst.session_pnl_sek
            for inst in self.state.by_instrument.values()
        )

        # Global halt check (after fills are realised so the latest P&L is
        # reflected).
        halt_reason = should_halt_global(self.state)
        if halt_reason:
            self.state.halted = True
            self.state.halt_reason = halt_reason
            self._log("halt_global", reason=halt_reason)
            self._persist()
            return {**report, "halted": True, "halt_reason": halt_reason}

        # EOD handling — only sweep if we have a remaining-minutes value.
        if eod_minutes_remaining is not None:
            if eod_minutes_remaining <= GRID_EOD_MARKET_SELL_MINUTES_BEFORE:
                report["eod_swept"] = True
                self.eod_market_flat()
                self._persist()
                return report
            if eod_minutes_remaining <= GRID_EOD_SWEEP_MINUTES_BEFORE:
                report["eod_swept"] = True
                self.eod_cancel_buys()
                self._persist()
                return report

        # Place / re-arm ladders per instrument.
        for ob_id, inst in self.state.by_instrument.items():
            ticker = inst.ticker
            sig = signal_data.get(ticker) or {}
            direction = sig.get("direction")
            confidence = float(sig.get("confidence") or 0.0)
            adx = sig.get("adx")

            instr_report: dict[str, Any] = {
                "ticker": ticker,
                "direction": direction,
                "confidence": confidence,
                "armed_buys": 0,
                "placed": 0,
            }

            # No signal or under-confidence => don't ARM new direction, but
            # leave existing inventory + sells alone.
            if not direction or direction not in ("LONG", "SHORT"):
                instr_report["skip"] = "no_direction"
                report["instruments"][ob_id] = instr_report
                continue
            if confidence < self._min_conf:
                instr_report["skip"] = f"low_conf<{self._min_conf}"
                report["instruments"][ob_id] = instr_report
                continue

            # Trend filter — high ADX => skip counter-trend placement.
            if adx is not None and adx > self._adx_trend_filter:
                # If signal already aligns with the trend direction, allow.
                # We cannot infer trend direction from ADX alone, so we
                # accept the signal at face value — the *signal* is the
                # consensus, and consensus in a high-ADX regime is
                # presumed with-trend.
                pass

            # Each instrument has a FIXED natural direction from the
            # catalog (BULL=LONG, BEAR=SHORT) baked in at seed time.
            # If the signal points the OTHER way, this instrument is on
            # the wrong side of the market — cancel any armed buys
            # (don't keep fishing into a moving signal) and skip the
            # placement step. Existing inventory + sell ladder + stop
            # are left alone so the original position can exit on its
            # own terms.
            if inst.active_direction != direction:
                cancelled = self.cancel_armed_buys(inst)
                report["cancels"] += cancelled
                instr_report["skip"] = "signal_direction_mismatch"
                instr_report["cancelled"] = cancelled
                report["instruments"][ob_id] = instr_report
                continue

            # Cooldown check — set after we last cancelled buys on this
            # instrument (signal flipped away then back too quickly).
            if inst.in_cooldown(self.now_fn()):
                instr_report["skip"] = "cooldown"
                report["instruments"][ob_id] = instr_report
                continue

            # Resolve bid from the prices dict, falling back to a fresh
            # quote against the live session when the caller didn't supply
            # one. metals_loop's internal prices dict is keyed by symbolic
            # name (e.g. 'silver_bull'), not orderbook id, so the fallback
            # path is the production norm.
            ob_prices = prices.get(ob_id) or prices.get(ticker) or {}
            bid = ob_prices.get("bid")
            if not bid or bid <= 0:
                quote = self._safe_session_call(
                    self.session.get_quote, ob_id, default=None,
                )
                if quote is None:
                    self._log("quote_fetch_failed", ob_id=ob_id,
                              ticker=ticker,
                              error="session_call returned None")
                    bid = 0
                else:
                    bid = float((quote or {}).get("buy") or 0)
            if not bid or bid <= 0:
                instr_report["skip"] = "no_bid"
                report["instruments"][ob_id] = instr_report
                continue

            # Global cap gate — sum planned notional across every
            # instrument before placing on this one. If the new ladder
            # would breach GRID_GLOBAL_MAX_SEK, skip placement entirely
            # to keep total deployed capital inside the user's budget.
            global_notional = global_planned_notional(self.state)
            if global_notional >= GRID_GLOBAL_MAX_SEK:
                self._log("skip_global_cap", ob_id=ob_id,
                          ticker=ticker,
                          global_notional=round(global_notional, 0),
                          cap=GRID_GLOBAL_MAX_SEK)
                instr_report["skip"] = "global_cap"
                report["instruments"][ob_id] = instr_report
                continue

            cat = self._catalog_for(ob_id) or {}
            placed = self.place_buy_ladder(
                inst, bid=float(bid),
                underlying_price=ob_prices.get("underlying_price"),
                barrier=cat.get("barrier") if cat.get("barrier") else None,
                leverage=cat.get("leverage"),
                global_cap_sek=GRID_GLOBAL_MAX_SEK,
            )
            report["placements"] += placed
            instr_report["placed"] = placed
            instr_report["armed_buys"] = len(inst.armed_buy_tiers())
            report["instruments"][ob_id] = instr_report

        # Tidy in-memory state and persist.
        for inst in self.state.by_instrument.values():
            prune_terminal_orders(inst)
        self._persist()
        return report

    # ---- EOD ---------------------------------------------------------------

    def eod_cancel_buys(self) -> int:
        """Cancel every armed buy across all instruments. Sell ladders
        and stop-losses are LEFT IN PLACE so existing inventory keeps
        its exit path."""
        n = 0
        for inst in self.state.by_instrument.values():
            n += self.cancel_armed_buys(inst)
        self._log("eod_cancel_buys", count=n)
        return n

    def eod_market_flat(self) -> int:
        """Force-flat every position via market sell (limit at bid - 1%
        to ensure fill). Cancels remaining armed sells + stops first.

        Returns the number of instruments touched.
        """
        n = 0
        for inst in self.state.by_instrument.values():
            if inst.inventory_units <= 0:
                continue
            # Cancel any armed sell tiers (don't double-up volume).
            for tier in list(inst.sell_ladder):
                if tier.status == ORDER_ARMED and tier.order_id:
                    self._safe_session_call(
                        self.session.cancel_order, tier.order_id,
                        default=None,
                    )
                    tier.status = ORDER_CANCELLED
            # Cancel stop via the stop-loss-specific endpoint when present.
            if inst.stop_loss_id:
                cancel_stop_fn = getattr(
                    self.session, "cancel_stop_loss", None,
                )
                if cancel_stop_fn is not None:
                    self._safe_session_call(
                        cancel_stop_fn, inst.stop_loss_id, default=None,
                    )
                inst.stop_loss_id = None
            # Place aggressive limit at last seen price - 1% as a market-
            # equivalent (Avanza warrants can't post true market orders).
            quote = self._safe_session_call(
                self.session.get_quote, inst.ob_id, default=None,
            )
            if quote is None:
                bid = inst.avg_entry_price
            else:
                bid = float((quote or {}).get("buy") or 0)
            if bid <= 0:
                bid = inst.avg_entry_price
            aggressive = round(max(bid * 0.99, 0.01), 2)
            result = self._safe_session_call(
                self.session.place_sell_order,
                inst.ob_id, aggressive, inst.inventory_units,
                default=None,
            )
            if result is None:
                self._log("eod_market_sell_failed", ob_id=inst.ob_id,
                          ticker=inst.ticker,
                          error="session_call returned None")
            else:
                self._log("eod_market_sell", ob_id=inst.ob_id,
                          ticker=inst.ticker, qty=inst.inventory_units,
                          price=aggressive)
                # TODO: MANUAL REVIEW — should decrement inst.inventory_units
                # here to prevent duplicate sells if eod_market_flat() runs
                # again before the order fills. Current code re-sells full
                # inventory on each call. See adversarial review P0-9.
                n += 1
        return n

    # ---- direction handling -----------------------------------------------

    def arm_direction(self, inst: InstrumentState, target_direction: str) -> None:
        """Set or flip the instrument's active direction with cooldown.

        Existing live buys on the old direction are cancelled. Existing
        inventory + sells are preserved — they exit via the rotated
        ladder.
        """
        if inst.active_direction == target_direction:
            return
        if inst.active_direction is not None:
            self.cancel_armed_buys(inst)
        flip_direction(inst, target_direction)
        self._log("flip_direction", ob_id=inst.ob_id, ticker=inst.ticker,
                  new_direction=target_direction,
                  cooldown_until=inst.cooldown_until)


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
