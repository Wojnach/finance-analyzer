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
    GRID_CASH_SAFETY_BUFFER_SEK,
    GRID_DECISIONS_LOG,
    GRID_DIRECTION_FLIP_COOLDOWN_MIN,
    GRID_GLOBAL_MAX_SEK,
    GRID_GLOBAL_SESSION_LOSS_LIMIT_SEK,
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

# Sentinel sell-tier index for the EOD market-flat liquidation order. Negative
# so it can never collide with a real tier index (>=0). Tracking the EOD sell
# as a sell-ladder tier (2026-05-28) lets reconcile_against_live record its fill
# and decrement inventory_units — without it the EOD sell was invisible to
# reconcile, leaving phantom inventory that re-fired a full-size sell next session.
EOD_SELL_TIER = -1


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
    # 2026-05-14 (P0-9 grid-fisher EOD duplicate-sell fix): order id of the
    # EOD market-sell once placed this session. ``eod_market_flat`` checks
    # this before placing a sell so a 60s tick window that fires inside
    # GRID_EOD_MARKET_SELL_MINUTES_BEFORE doesn't queue a second full-size
    # sell on top of the still-resting first one. Cleared by
    # ``roll_session_if_new_day``.
    eod_sell_order_id: Optional[str] = None
    # 2026-05-18 (Gate B — silent-rejection back-off): consecutive count of
    # ``external_cancel_buy`` reconciliations that fired within
    # GRID_RAPID_CANCEL_THRESHOLD_S of a tier's ``placed_ts``. Reaches the
    # threshold → instrument cooldown for GRID_RAPID_CANCEL_COOLDOWN_S.
    # Reset by a successful fill or by ``roll_session_if_new_day``.
    rapid_cancel_count: int = 0
    stop_needs_rearm: bool = False
    last_rapid_cancel_ts: Optional[str] = None

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
            "eod_sell_order_id": self.eod_sell_order_id,
            "rapid_cancel_count": self.rapid_cancel_count,
            "last_rapid_cancel_ts": self.last_rapid_cancel_ts,
            "stop_needs_rearm": self.stop_needs_rearm,
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
            eod_sell_order_id=d.get("eod_sell_order_id"),
            rapid_cancel_count=int(d.get("rapid_cancel_count", 0) or 0),
            last_rapid_cancel_ts=d.get("last_rapid_cancel_ts"),
            stop_needs_rearm=bool(d.get("stop_needs_rearm", False)),
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

# 2026-06-12 (audit B4 fix 5): dynamic EOD cutoff from the Avanza
# marketPlace.todayClosingTime field. IMPORTANT nuance (memory
# reference_avanza_trading_hours): for AVA market-maker-quoted certificates
# todayClosingTime reports the EXCHANGE close (First North ~17:30) while
# the MM keeps quoting until ~22:00 — so on a NORMAL day the fetched value
# must NOT replace the 21:55 default. We use it only to detect EARLY-close
# days (Swedish half-days, ~13:00 exchange close): a fetched close at or
# below _EOD_EARLY_CLOSE_THRESHOLD_MIN signals a shortened session and the
# cutoff becomes (close - margin). Fetch failure → fall back to 21:55
# minus a safety margin and emit an `eod_close_time_fallback` decision-log
# entry (binding premortem hook).
_EOD_CLOSE_MARGIN_MIN = 5
_EOD_FALLBACK_HOUR = 21
_EOD_FALLBACK_MINUTE = 50  # 21:55 default minus the 5-min safety margin
_EOD_EARLY_CLOSE_THRESHOLD_MIN = 17 * 60  # < 17:00 → treat as half-day
# 2026-06-12 (review fix 18d9d0cc #3): plausibility floor for the early-
# close detector. Swedish half-days close at 13:00; a garbage
# todayClosingTime like 09:00 would otherwise set the cutoff to 08:55 and
# kill the grid for the whole day. Only [12:00, 17:00) is accepted as a
# genuine early close — anything below the floor is treated as a fetch
# failure (21:50 fallback + eod_close_time_fallback log).
_EOD_EARLY_CLOSE_MIN_PLAUSIBLE_MIN = 12 * 60
_EOD_CLOSE_FETCH_RETRY_S = 1800.0  # re-try a failed fetch after 30 min

_close_time_lock = threading.Lock()
_close_time_cache: dict[str, tuple[int, int]] = {}  # local date -> cutoff
_close_fetch_failed_mono: dict[str, float] = {}  # local date -> last fail ts
_close_fallback_logged: set[str] = set()  # dates where fallback was logged

# 2026-06-12 (review fix 18d9d0cc #1): while halted, re-attempt cancelling
# any still-ARMED buy tier every N ticks (~N minutes on the 60s loop). A
# cancel that timed out during the halt transition would otherwise rest at
# the broker until the EOD window.
_HALT_CANCEL_RETRY_TICKS = 5


def _parse_hhmm(raw: Any) -> Optional[tuple[int, int]]:
    """Parse 'HH:MM' / 'HH:MM:SS' into (hour, minute), else None."""
    if not isinstance(raw, str):
        return None
    parts = raw.strip().split(":")
    if len(parts) < 2:
        return None
    try:
        h, m = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return None
    return h, m


def _extract_today_closing_time(payload: Any) -> Optional[tuple[int, int]]:
    """Pull todayClosingTime out of a market-guide payload, defensively.

    Avanza has renamed/nested fields before (see get_buying_power's
    multi-shape fallback) so we probe every plausible location.
    """
    if not isinstance(payload, dict):
        return None
    candidates = [
        (payload.get("marketPlace") or {}) if isinstance(
            payload.get("marketPlace"), dict) else {},
        (payload.get("listing") or {}) if isinstance(
            payload.get("listing"), dict) else {},
        (payload.get("orderbook") or {}) if isinstance(
            payload.get("orderbook"), dict) else {},
        payload,
    ]
    for obj in candidates:
        for key in ("todayClosingTime", "closingTime"):
            parsed = _parse_hhmm(obj.get(key))
            if parsed is not None:
                return parsed
        nested = obj.get("marketPlace")
        if isinstance(nested, dict):
            for key in ("todayClosingTime", "closingTime"):
                parsed = _parse_hhmm(nested.get(key))
                if parsed is not None:
                    return parsed
    return None


def _default_close_fetch() -> Any:
    """Fetch a market-guide payload for the first active grid instrument."""
    from portfolio.avanza_session import api_get  # noqa: PLC0415 — lazy

    for by_dir in GRID_ACTIVE_INSTRUMENTS.values():
        for ob_id in by_dir.values():
            return api_get(f"/_api/market-guide/certificate/{ob_id}")
    raise RuntimeError("no active grid instruments configured")


def resolve_eod_cutoff_hm(
    fetch_payload_fn: Optional[callable] = None,
    *,
    log_path: str | Path = GRID_DECISIONS_LOG,
) -> tuple[int, int]:
    """Return today's (hour, minute) EOD cutoff in Europe/Stockholm.

    Cached per local date. Successful resolutions cache for the whole day;
    failures fall back to 21:50 (21:55 minus safety margin) with an
    ``eod_close_time_fallback`` decision-log entry (once per date) and a
    retry after ``_EOD_CLOSE_FETCH_RETRY_S``. Never raises.
    """
    try:
        import zoneinfo  # noqa: PLC0415
        local_date = _dt.datetime.now(
            zoneinfo.ZoneInfo(_EOD_TZ_NAME)
        ).strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001 — tzdata missing
        local_date = _today_session_id()

    with _close_time_lock:
        cached = _close_time_cache.get(local_date)
        if cached is not None:
            return cached
        last_fail = _close_fetch_failed_mono.get(local_date)
        if last_fail is not None and (
            time.monotonic() - last_fail < _EOD_CLOSE_FETCH_RETRY_S
        ):
            return _EOD_FALLBACK_HOUR, _EOD_FALLBACK_MINUTE

    fetch = fetch_payload_fn or _default_close_fetch
    close_hm: Optional[tuple[int, int]] = None
    fetch_error: Optional[str] = None
    try:
        close_hm = _extract_today_closing_time(fetch())
        if close_hm is None:
            fetch_error = "todayClosingTime missing/unparseable in payload"
        elif (close_hm[0] * 60 + close_hm[1]) < _EOD_EARLY_CLOSE_MIN_PLAUSIBLE_MIN:
            # 2026-06-12 (review fix 18d9d0cc #3): implausibly early close
            # (Swedish half-days close 13:00) — treat as a fetch failure
            # rather than letting a garbage value gate the grid all day.
            fetch_error = (
                f"implausible todayClosingTime "
                f"{close_hm[0]:02d}:{close_hm[1]:02d} (< 12:00)"
            )
            close_hm = None
    except Exception as exc:  # noqa: BLE001 — any fetch failure → fallback
        fetch_error = f"{type(exc).__name__}: {exc}"

    if close_hm is None:
        with _close_time_lock:
            _close_fetch_failed_mono[local_date] = time.monotonic()
            should_log = local_date not in _close_fallback_logged
            if should_log:
                _close_fallback_logged.add(local_date)
        if should_log:
            log_decision(
                "eod_close_time_fallback",
                decisions_path=log_path,
                error=fetch_error,
                fallback=f"{_EOD_FALLBACK_HOUR:02d}:{_EOD_FALLBACK_MINUTE:02d}",
            )
        logger.warning(
            "grid_fisher: todayClosingTime fetch failed (%s) — EOD cutoff "
            "falls back to %02d:%02d",
            fetch_error, _EOD_FALLBACK_HOUR, _EOD_FALLBACK_MINUTE,
        )
        return _EOD_FALLBACK_HOUR, _EOD_FALLBACK_MINUTE

    close_min = close_hm[0] * 60 + close_hm[1]
    if close_min < _EOD_EARLY_CLOSE_THRESHOLD_MIN:
        # Half-day: exchange (and the MM session keyed to it) closes early.
        cutoff_min = max(0, close_min - _EOD_CLOSE_MARGIN_MIN)
        cutoff = (cutoff_min // 60, cutoff_min % 60)
        logger.info(
            "grid_fisher: early close detected (todayClosingTime=%02d:%02d) "
            "— EOD cutoff %02d:%02d", close_hm[0], close_hm[1], *cutoff,
        )
    else:
        # Normal day: the MM quotes well past the exchange close (memory
        # reference_avanza_trading_hours) — keep the 21:55 default.
        cutoff = (_EOD_LOCAL_HOUR, _EOD_LOCAL_MINUTE)
    with _close_time_lock:
        _close_time_cache[local_date] = cutoff
    return cutoff


def minutes_until_eod(
    now_utc: Optional[_dt.datetime] = None,
    *,
    cutoff_hm: Optional[tuple[int, int]] = None,
) -> float:
    """Minutes until the grid fisher's day-end cutoff in Europe/Stockholm.

    ``cutoff_hm`` overrides the default 21:55 cutoff — production callers
    pass ``resolve_eod_cutoff_hm()`` so half-days flatten before the real
    close (audit B4 fix 5). Returns a non-negative float; if the cutoff has
    already passed for the day, returns the minutes-until-tomorrow's cutoff
    (so the caller can detect with ``< EOD_SWEEP_MINUTES_BEFORE`` only
    during the active window). Returns ``float("inf")`` if zoneinfo is
    unavailable so the caller never triggers EOD on the failure path.
    """
    try:
        import zoneinfo  # noqa: PLC0415 — optional import to keep startup cheap
    except ImportError:
        return float("inf")
    try:
        tz = zoneinfo.ZoneInfo(_EOD_TZ_NAME)
    except Exception:  # noqa: BLE001 — missing tzdata
        return float("inf")
    eod_hour, eod_minute = cutoff_hm or (_EOD_LOCAL_HOUR, _EOD_LOCAL_MINUTE)
    now = now_utc or _dt.datetime.now(_dt.timezone.utc)
    local = now.astimezone(tz)
    cutoff = local.replace(
        hour=eod_hour, minute=eod_minute, second=0, microsecond=0,
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
    """Load state from *state_path*. Returns a fresh state on missing/corrupt.

    Version handling:
      * File version equals current → normal load.
      * File version is missing or older than current → load through
        ``from_dict`` (all new fields default via ``.get``). The
        in-memory version field is updated to the current schema on the
        next ``save_state``.
      * File version is **newer** than current → bail and start fresh.
        This prevents a forward-incompatible state file from being
        silently truncated when an older binary tries to read it
        (premortem N7, 2026-05-18).
    """
    raw = load_json(state_path, default=None)
    if not isinstance(raw, dict):
        logger.info("grid_fisher: no prior state found at %s", state_path)
        return GridFisherState(session_id=_today_session_id())

    file_version = raw.get("version")
    try:
        file_version_int = int(file_version) if file_version is not None else 0
    except (TypeError, ValueError):
        file_version_int = 0

    if file_version_int > GRID_STATE_SCHEMA_VERSION:
        logger.critical(
            "grid_fisher: state file %s has version=%r > supported %d. "
            "Refusing to read forward-incompatible state — starting fresh. "
            "Operator: inspect %s and resolve before next session.",
            state_path, file_version, GRID_STATE_SCHEMA_VERSION, state_path,
        )
        return GridFisherState(session_id=_today_session_id())

    if file_version_int < GRID_STATE_SCHEMA_VERSION:
        logger.info(
            "grid_fisher: state file %s has version=%r, migrating to %d via defaults",
            state_path, file_version, GRID_STATE_SCHEMA_VERSION,
        )

    try:
        state = GridFisherState.from_dict(raw)
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning(
            "grid_fisher: state file %s is malformed (%s) — resetting",
            state_path, exc,
        )
        return GridFisherState(session_id=_today_session_id())

    # Stamp the loaded state with the current schema version so the next
    # save reflects the actual on-disk layout.
    state.version = GRID_STATE_SCHEMA_VERSION
    return state


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
    # 2026-06-12 (audit B4 fix 3): the halt flag is per-session — clear it
    # on rollover. Previously it was never reset, so it stayed True in
    # state/dashboard forever while trading silently resumed anyway (tick
    # used to re-derive the halt from P&L instead of checking the flag).
    # Now tick() honors state.halted explicitly, so without this reset a
    # single halt would freeze the grid permanently.
    if state.halted:
        logger.info(
            "grid_fisher: clearing session halt on rollover (was: %s)",
            state.halt_reason,
        )
    state.halted = False
    state.halt_reason = None
    for inst in state.by_instrument.values():
        inst.session_pnl_sek = 0.0
        inst.fills_this_session = 0
        # Clear the EOD-sell flag so the new session can re-arm sweeps on
        # any remaining inventory (e.g. partial fills carried over). The
        # corresponding order is from yesterday's session; either it filled
        # (inventory already 0) or it expired/got cancelled overnight.
        inst.eod_sell_order_id = None
        # P0 (FGL 2026-06-06): if inventory carried over (yesterday's EOD sell
        # never filled) and the stop was already nulled at EOD, clearing the
        # sell flag alone leaves this inventory with NO stop AND NO sell until
        # the next EOD window. Flag re-arm so the tick-time honor path restores
        # a protective stop. Guard on `not stop_loss_id` — never double-stop a
        # lot that still has a live stop (the honor path does not cancel first).
        if inst.inventory_units > 0 and not inst.stop_loss_id:
            inst.stop_needs_rearm = True
        # Gate B counters reset at session boundary — yesterday's
        # silent-rejection state shouldn't follow us into a new
        # trading day where the issue may have resolved.
        inst.rapid_cancel_count = 0
        inst.last_rapid_cancel_ts = None
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
    # Any fill clears the rapid-cancel streak — the orderbook is alive.
    inst.rapid_cancel_count = 0
    inst.last_rapid_cancel_ts = None

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

    2026-06-12 (audit B4 fix 9): fixed global limit from config instead of
    per-instrument limit × seeded instrument count — the derived form
    scaled with catalog size (6 seeded pairs → -3000 SEK, ~46% of budget)
    and was unreachable before every instrument froze individually.
    """
    threshold = -abs(GRID_GLOBAL_SESSION_LOSS_LIMIT_SEK)
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
    # (ob_id, extra_units) where live volume EXCEEDS recorded inventory — a buy
    # order that partially filled while still resting (so reconcile's tier loop
    # skipped it). Those units carry no stop until the order fully fills/cancels;
    # the caller flags a stop rearm so the tick protects them. (2026-05-28)
    under_protected: list[tuple[str, int]] = field(default_factory=list)
    # (ob_id, missing_buy_count, missing_sell_count) where a buy AND a sell
    # both disappeared in the same tick — volume-delta inference is
    # confounded (offsetting fills net to delta≈0), so the instrument was
    # deferred this tick. (2026-06-12, audit B4 fix 8)
    ambiguous: list[tuple[str, int, int]] = field(default_factory=list)


# 2026-06-12 (audit B4 fix 8): consecutive same-tick buy+sell-disappearance
# counter per ob_id. In-memory only (a restart just re-defers one tick).
# First ambiguous tick → defer (re-poll next tick, record nothing). If the
# SAME ambiguity persists on the next tick (orders genuinely gone, not a
# transient read glitch), fall back to the volume heuristic rather than
# stalling forever — an indefinitely-deferred fill would leave new
# inventory without rotation or stop, which is worse than a possibly
# mis-attributed P&L booking.
_ambiguous_streak: dict[str, int] = {}


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

        # 2026-06-12 (audit B4 fix 8): if a buy tier AND a sell tier both
        # vanished from the open list this tick, the volume deltas can
        # offset (buy 74 fills + sell 74 fills → delta 0) and BOTH legs
        # would be misclassified as CANCELLED: realised P&L never booked,
        # avg entry wrong, no rotation/stop on the new lot. Conservative
        # path: defer the instrument one tick (treat as unknown, re-poll);
        # only if the same ambiguity persists next tick do we fall through
        # to the volume heuristic (see _ambiguous_streak above).
        missing_buys = [
            t for t in inst.buy_ladder
            if t.status == ORDER_ARMED and t.order_id
            and t.order_id not in open_order_ids
        ]
        missing_sells = [
            t for t in inst.sell_ladder
            if t.status == ORDER_ARMED and t.order_id
            and t.order_id not in open_order_ids
        ]
        if missing_buys and missing_sells:
            streak = _ambiguous_streak.get(ob_id, 0) + 1
            if streak <= 1:
                _ambiguous_streak[ob_id] = streak
                res.ambiguous.append(
                    (ob_id, len(missing_buys), len(missing_sells))
                )
                continue
        _ambiguous_streak.pop(ob_id, None)

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

        # Final drift check — logged for every mismatch.
        if live_vol != inst.inventory_units:
            res.inventory_drift.append(
                (ob_id, inst.inventory_units, live_vol)
            )
            # 2026-05-28: when we hold MORE than recorded, a buy order
            # partially filled while still resting (Avanza keeps a partial in
            # the open list with reduced volume, so the tier-loop `continue`
            # above skipped it). Those filled units never went through
            # record_fill / rotate_on_buy_fill, so they have NO stop-loss. Align
            # inventory up to the live volume and flag a stop rearm so the tick
            # places a protective stop on the real held size. Estimate the
            # entry of the extra units from the cheapest armed buy tier (the
            # resting order that is filling); keep prior avg if no armed tier.
            # We forgo the take-profit rotation on these units (they exit via
            # stop / EOD) — a missed-profit, never a naked position.
            if live_vol > inst.inventory_units:
                extra = live_vol - inst.inventory_units
                armed = inst.armed_buy_tiers()
                est_price = min((o.price for o in armed), default=0.0)
                if est_price <= 0:
                    est_price = inst.avg_entry_price
                if est_price > 0:
                    prev_units = inst.inventory_units
                    inst.avg_entry_price = (
                        (inst.avg_entry_price * prev_units) + (est_price * extra)
                    ) / live_vol
                inst.inventory_units = live_vol
                inst.stop_needs_rearm = True
                res.under_protected.append((ob_id, int(extra)))
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
        account_id: Optional[str] = None,
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

        # 2026-06-12 (review fix 18d9d0cc #1): countdown for re-attempting
        # cancellation of buy tiers still ARMED while the session is halted
        # (a cancel that timed out during the halt transition). 0 → retry
        # on the next halted tick, then every _HALT_CANCEL_RETRY_TICKS.
        self._halt_cancel_retry_countdown = 0

        # Live-cash gate. When ``account_id`` is provided the tick consults
        # ``session.get_buying_power(account_id)`` and clamps the effective
        # global cap to (buying_power - GRID_CASH_SAFETY_BUFFER_SEK). When
        # ``None`` (e.g. unit tests with a mock session) the gate is bypassed
        # and the original hardcoded GRID_GLOBAL_MAX_SEK applies — preserves
        # previous behaviour for callers that haven't opted in.
        self.account_id = account_id
        self._buying_power_cache: Optional[tuple[float, float]] = None  # (mono_ts, value)

        # Quote cache for Gate A (quote-staleness pre-placement check).
        # Keyed by ob_id. Each entry is (cache_set_monotonic_ts, raw_quote_dict).
        # Lock-guarded because the metals_loop tick can race the worker
        # thread that fetches the underlying quote via _safe_session_call.
        self._quote_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._quote_cache_lock = threading.Lock()
        # Track which ob_ids we've already warned about a missing
        # `timeOfLast` field so we don't spam the decision log when an
        # endpoint returns an unexpected payload shape.
        self._quote_shape_warned: set[str] = set()

        self.state = load_state(self.state_path)
        _seed_state_for_active_instruments(self.state, self.catalog)

    # ---- low-level helpers ------------------------------------------------

    def _log(self, category: str, **fields: Any) -> None:
        log_decision(category, decisions_path=self.decisions_path, **fields)

    def _persist(self) -> None:
        save_state(self.state, self.state_path)

    def _fetch_buying_power_sek(self) -> Optional[float]:
        """Pull live buying power from Avanza, with a short in-memory cache.

        Returns the cached value when fresh (< GRID_BUYING_POWER_CACHE_SECS),
        a stale cached value within GRID_BUYING_POWER_STALE_GRACE_SECS if the
        fresh fetch fails, or None if no usable reading exists.

        ``self.session`` must expose ``get_buying_power(account_id) -> dict |
        None`` with key ``buying_power``. Matches the contract from
        ``portfolio.avanza_session.get_buying_power`` / the unified
        ``portfolio.avanza.account.get_buying_power``.
        """
        if self.account_id is None:
            return None
        from portfolio.grid_fisher_config import (
            GRID_BUYING_POWER_CACHE_SECS,
            GRID_BUYING_POWER_STALE_GRACE_SECS,
        )
        now_mono = time.monotonic()
        if self._buying_power_cache is not None:
            cached_ts, cached_val = self._buying_power_cache
            if now_mono - cached_ts < GRID_BUYING_POWER_CACHE_SECS:
                return cached_val

        getter = getattr(self.session, "get_buying_power", None)
        if not callable(getter):
            self._log("buying_power_unavailable",
                      reason="session has no get_buying_power")
            return None

        result = self._safe_session_call(getter, self.account_id, default=None)
        bp: Optional[float] = None
        if isinstance(result, dict):
            raw = result.get("buying_power")
            if raw is not None:
                try:
                    bp = float(raw)
                except (TypeError, ValueError):
                    bp = None
        elif isinstance(result, (int, float)):
            bp = float(result)

        if bp is not None and bp >= 0:
            self._buying_power_cache = (now_mono, bp)
            return bp

        # Fresh fetch failed — fall back to the cached reading if it's
        # still inside the stale-grace window. Past the grace, return
        # None so the caller fails-closed instead of trading against
        # stale balance state (this is what surfaced the 2026-05-13
        # OLJAB-on-empty-cash incident).
        if self._buying_power_cache is not None:
            cached_ts, cached_val = self._buying_power_cache
            age = now_mono - cached_ts
            if age < GRID_BUYING_POWER_STALE_GRACE_SECS:
                self._log("buying_power_stale_reuse",
                          age_s=round(age, 1),
                          cached_value=round(cached_val, 0))
                return cached_val
            self._log("buying_power_stale_expired",
                      age_s=round(age, 1))
        else:
            self._log("buying_power_fetch_failed",
                      reason="no cached fallback available")
        return None

    def _effective_global_cap(self) -> tuple[float, dict[str, Any]]:
        """Compute the per-tick cap on aggregate planned notional.

        Returns ``(cap_sek, debug_fields)`` where ``cap_sek`` is the lesser
        of ``GRID_GLOBAL_MAX_SEK`` and ``(buying_power - safety buffer)``.
        ``debug_fields`` carries the inputs for logging/observability.

        Behaviour:
        * No ``account_id`` configured → bypass (returns config cap). This
          preserves the previous behaviour for unit-test callers that pass
          a mock session.
        * Fresh / cached buying power available → clamp to live cash.
        * Buying-power fetch failed AND no fresh-enough cache → return 0
          (fail-closed). The tick logs ``skip_global_cap`` for every
          instrument, no orders go out.
        """
        # Use module-level constants so monkeypatches on this module
        # (e.g. tests setting ``gf.GRID_GLOBAL_MAX_SEK = 100``) take
        # effect. Late-importing from grid_fisher_config would bypass
        # the patch and surprise callers.
        cfg_cap = GRID_GLOBAL_MAX_SEK
        buffer_sek = GRID_CASH_SAFETY_BUFFER_SEK
        if self.account_id is None:
            return float(cfg_cap), {
                "source": "config_only",
                "config_cap": cfg_cap,
            }
        bp = self._fetch_buying_power_sek()
        if bp is None:
            return 0.0, {
                "source": "fail_closed",
                "config_cap": cfg_cap,
                "buffer": buffer_sek,
                "reason": "buying_power_unavailable",
            }
        clamped = max(0.0, bp - float(buffer_sek))
        cap = min(float(cfg_cap), clamped)
        return cap, {
            "source": "live_buying_power",
            "config_cap": cfg_cap,
            "buffer": buffer_sek,
            "buying_power": round(bp, 0),
            "clamped": round(clamped, 0),
        }

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
        """Invoke an Avanza session method, returning ``default`` on failure.

        2026-06-12 (audit B4 fix 1): the per-GridFisher 'grid-fisher-session'
        worker thread is gone. It was one of three competing Playwright
        context initializers — whichever thread initialised avanza_session's
        singleton first owned it, and the others got greenlet "cannot switch
        to a different thread" on every call (16,719 such session_call_error
        entries 2026-05-11→2026-06-09; the grid was completely blind on 9
        full trading days). avanza_session now pins ALL Playwright traffic
        to its own module-level single-worker executor with per-call
        timeouts and queue-wait/consecutive-failure escalation, so we call
        straight through from any thread (asyncio loops included). The
        error envelope is preserved: timeout/exception → journal entry +
        ``default``.
        """
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            is_timeout = False
            try:
                from portfolio.avanza_session import AvanzaSessionTimeout
                is_timeout = isinstance(exc, AvanzaSessionTimeout)
            except Exception:  # noqa: BLE001 — classification only
                pass
            if is_timeout:
                self._log("session_call_timeout",
                          method=getattr(fn, "__name__", repr(fn)),
                          error=str(exc))
            else:
                self._log("session_call_error",
                          method=getattr(fn, "__name__", repr(fn)),
                          error=str(exc))
            return default

    # ---- Gate A: quote-staleness pre-placement check --------------------

    def _fetch_quote_cached(self, ob_id: str) -> Optional[dict[str, Any]]:
        """Return the most recent ``get_quote`` payload for *ob_id*.

        Uses a per-instrument cache with TTL ``GRID_QUOTE_CACHE_SECS`` so
        the staleness check doesn't double the per-tick get_quote count
        (``place_buy_ladder`` already issues one as a bid fallback). The
        cache is thread-safe (premortem N3) — the metals_loop main
        thread and the single worker thread inside ``_safe_session_call``
        share this dict.

        Returns ``None`` if no quote could be fetched. The caller treats
        ``None`` as "stale" and fail-safe-skips placement.
        """
        from portfolio.grid_fisher_config import GRID_QUOTE_CACHE_SECS

        now_mono = time.monotonic()
        with self._quote_cache_lock:
            entry = self._quote_cache.get(ob_id)
            if entry is not None:
                cached_ts, cached_q = entry
                if now_mono - cached_ts < GRID_QUOTE_CACHE_SECS:
                    return cached_q

        quote = self._safe_session_call(
            self.session.get_quote, ob_id, default=None,
        )
        if quote is None:
            return None

        with self._quote_cache_lock:
            # Race-resistant: if another thread populated a fresher
            # quote between our miss and now, keep that one. Strictly
            # ``<`` (not ``<=``) so a same-tick collision can't overwrite
            # a fresher value with our older one.
            existing = self._quote_cache.get(ob_id)
            if existing is None or existing[0] < now_mono:
                self._quote_cache[ob_id] = (now_mono, quote)
        return quote

    def _quote_time_of_last_age_s(
        self, ob_id: str, quote: Optional[dict[str, Any]],
    ) -> Optional[float]:
        """Return age (seconds) of *quote*'s last trade, or None if absent.

        Avanza publishes ``timeOfLast`` as a UTC millisecond epoch on the
        ``/_api/market-guide/stock/{ob_id}/quote`` payload (verified
        empirically 2026-05-18). A missing or non-numeric value is logged
        once per ob_id (so an Avanza schema change is loud) and treated
        as "stale → skip" by the caller.
        """
        if not isinstance(quote, dict):
            return None
        raw = quote.get("timeOfLast")
        if raw is None:
            if ob_id not in self._quote_shape_warned:
                self._quote_shape_warned.add(ob_id)
                self._log("quote_field_missing", ob_id=ob_id,
                          field="timeOfLast",
                          quote_keys=sorted(list(quote.keys()))[:12])
            return None
        try:
            t_ms = float(raw)
        except (TypeError, ValueError):
            if ob_id not in self._quote_shape_warned:
                self._quote_shape_warned.add(ob_id)
                self._log("quote_field_invalid", ob_id=ob_id,
                          field="timeOfLast", value=str(raw)[:80])
            return None
        age_s = (time.time() * 1000.0 - t_ms) / 1000.0
        if age_s < 0:
            # Future-dated timestamp from Avanza (data corruption /
            # clock drift). Don't silently clamp to 0 — that would let
            # a corrupted response sneak past Gate A even when the
            # orderbook is actually closed. One-shot warning + treat as
            # stale (caller short-circuits placement).
            if ob_id not in self._quote_shape_warned:
                self._quote_shape_warned.add(ob_id)
                self._log("quote_field_future_dated", ob_id=ob_id,
                          field="timeOfLast", age_s=round(age_s, 1),
                          value=str(raw)[:80])
            return None
        return age_s

    # ---- Gate B: rapid-cancel back-off ----------------------------------

    def _maybe_arm_rapid_cancel_cooldown(
        self, inst: InstrumentState, *, tier_idx: int,
    ) -> None:
        """Increment Gate B counter and arm a cooldown if the threshold is hit.

        Triggered for every ``external_cancel_buy`` discovered by
        ``reconcile_against_live``. The just-cancelled tier still lives
        in ``inst.buy_ladder`` (status=CANCELLED, pruned later in the
        tick) so we can read its ``placed_ts``. If the cancel age is
        below ``GRID_RAPID_CANCEL_THRESHOLD_S``, the cancel is treated
        as a silent broker reject and counted toward the consecutive
        streak. Two in a row arms a 6 h cooldown (default). Anything
        slower than the threshold resets the streak.

        Persists state immediately so a process restart between this
        write and the end-of-tick ``_persist`` doesn't lose the counter
        (premortem N6).
        """
        from portfolio.grid_fisher_config import (
            GRID_RAPID_CANCEL_COOLDOWN_S,
            GRID_RAPID_CANCEL_MAX_CONSECUTIVE,
            GRID_RAPID_CANCEL_THRESHOLD_S,
        )

        # Find the CANCELLED tier so we can compute the cancel age.
        target: Optional[TierOrder] = None
        for o in inst.buy_ladder:
            if o.tier == tier_idx and o.status == ORDER_CANCELLED:
                target = o
                break
        if target is None or not target.placed_ts:
            return

        try:
            placed_dt = _dt.datetime.fromisoformat(
                target.placed_ts.replace("Z", "+00:00")
            )
        except ValueError:
            return
        if placed_dt.tzinfo is None:
            placed_dt = placed_dt.replace(tzinfo=_dt.timezone.utc)
        now_utc = _dt.datetime.now(_dt.timezone.utc)
        age_s = (now_utc - placed_dt).total_seconds()

        if age_s >= GRID_RAPID_CANCEL_THRESHOLD_S:
            # Slow cancel — orderbook is alive; clear the streak.
            if inst.rapid_cancel_count:
                inst.rapid_cancel_count = 0
                inst.last_rapid_cancel_ts = None
                self._persist()
            return

        inst.rapid_cancel_count += 1
        inst.last_rapid_cancel_ts = _utcnow_iso()

        if inst.rapid_cancel_count >= GRID_RAPID_CANCEL_MAX_CONSECUTIVE:
            cooldown_dt = now_utc + _dt.timedelta(
                seconds=GRID_RAPID_CANCEL_COOLDOWN_S
            )
            inst.cooldown_until = cooldown_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            self._log(
                "rapid_cancel_backoff",
                ob_id=inst.ob_id,
                ticker=inst.ticker,
                tier=tier_idx,
                age_s=round(age_s, 1),
                count=inst.rapid_cancel_count,
                cooldown_until=inst.cooldown_until,
                threshold_s=GRID_RAPID_CANCEL_THRESHOLD_S,
            )

        # Persist immediately so a restart between this write and the
        # end-of-tick ``_persist`` preserves the streak (premortem N6).
        self._persist()

    def _is_quote_stale(self, ob_id: str) -> tuple[bool, Optional[float]]:
        """Gate A check. Returns ``(is_stale, age_s)``.

        ``is_stale=True`` means: skip placement. Conditions:
          * fetch failed (treat as stale, fail-safe)
          * timeOfLast missing/invalid (treat as stale, fail-safe)
          * age > GRID_QUOTE_STALENESS_THRESHOLD_S
        """
        from portfolio.grid_fisher_config import GRID_QUOTE_STALENESS_THRESHOLD_S

        quote = self._fetch_quote_cached(ob_id)
        if quote is None:
            return True, None
        age = self._quote_time_of_last_age_s(ob_id, quote)
        if age is None:
            return True, None
        return age > GRID_QUOTE_STALENESS_THRESHOLD_S, age

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

        # Gate A — abort placement if the orderbook hasn't traded
        # recently. Added 2026-05-18 after FNSE OLJAB X5 spun a place /
        # silent-cancel loop after hours: Avanza accepted submissions but
        # the orderbook was dead so every order auto-cancelled. ``quote``
        # data is cached per instrument with GRID_QUOTE_CACHE_SECS TTL so
        # we don't add an extra get_quote per tick when the bid fallback
        # path also fetches one.
        is_stale, age_s = self._is_quote_stale(inst.ob_id)
        if is_stale:
            from portfolio.grid_fisher_config import (
                GRID_QUOTE_STALENESS_THRESHOLD_S,
            )
            self._log("skip_quote_stale", ob_id=inst.ob_id,
                      ticker=inst.ticker,
                      time_of_last_age_s=(
                          round(age_s, 1) if age_s is not None else None
                      ),
                      threshold_s=GRID_QUOTE_STALENESS_THRESHOLD_S)
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
            # 2026-05-28: re-check the PER-INSTRUMENT cap per tier, not only at
            # function entry. hit_per_instrument_cap() at the top sees only the
            # state before this ladder; when the instrument already carries
            # rotated inventory below the cap (e.g. ~2400 SEK < 3000), the entry
            # gate passes and the loop would then place both fresh legs
            # (2x1200) — pushing single-instrument planned notional to ~4800 SEK,
            # ~60% over GRID_PER_INSTRUMENT_MAX_SEK. planned_notional_sek()
            # accumulates as tiers are appended below, so this projection tracks
            # the running total. Mirrors the global-cap per-tier guard.
            inst_projected = inst.planned_notional_sek() + tier.notional_sek
            if inst_projected > GRID_PER_INSTRUMENT_MAX_SEK:
                self._log("skip_cap", ob_id=inst.ob_id, ticker=inst.ticker,
                          tier=tier.index, projected=round(inst_projected, 0),
                          cap=GRID_PER_INSTRUMENT_MAX_SEK)
                break
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
        if new_stop_id is not None:
            inst.stop_loss_id = new_stop_id
            inst.stop_loss_price = stop_price
            inst.stop_needs_rearm = False
        else:
            inst.stop_needs_rearm = True
            logger.critical(
                "Stop-rearm FAILED for %s (%s) — keeping old stop_loss_id=%s "
                "to avoid naked position (FGL P0-3)",
                inst.ticker, inst.ob_id, inst.stop_loss_id,
            )
            try:
                from portfolio.file_utils import atomic_append_jsonl
                atomic_append_jsonl(
                    str(Path(__file__).resolve().parent.parent / "data" / "critical_errors.jsonl"),
                    {
                        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                        "level": "critical",
                        "category": "grid_fisher_naked_position",
                        "caller": "grid_fisher.rotate_on_buy_fill",
                        "message": f"Stop rearm failed for {inst.ticker} — position has no broker-side stop",
                        "context": {"ob_id": inst.ob_id, "old_stop_id": inst.stop_loss_id},
                    },
                )
            except Exception:
                logger.debug("grid_fisher: state cleanup failed", exc_info=True)

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
            # 2026-06-12 (review fix 18d9d0cc #4): tiers carried over ARMED
            # into a new session (e.g. a halted session that crashed before
            # the EOD sweep) reference yesterday's day orders at the broker.
            # Don't leave them for the heuristic reconcile to guess at —
            # log the carryover and cancel carried BUY tiers via the
            # verified-cancel path (only flips to CANCELLED on confirmed
            # broker SUCCESS). Sell tiers are logged but not state-cancelled
            # here: their day orders expired overnight and reconcile settles
            # them against live volume; blindly dropping them would lose the
            # exit-tracking for carried inventory.
            for inst in self.state.by_instrument.values():
                carried_buys = inst.armed_buy_tiers()
                carried_sells = inst.armed_sell_tiers()
                if not carried_buys and not carried_sells:
                    continue
                self._log("stale_armed_carryover", ob_id=inst.ob_id,
                          ticker=inst.ticker,
                          armed_buys=len(carried_buys),
                          armed_sells=len(carried_sells))
                if carried_buys:
                    report["cancels"] += self.cancel_armed_buys(inst)

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
            # Gate B — rapid-cancel back-off.
            self._maybe_arm_rapid_cancel_cooldown(inst, tier_idx=tier)
        for ob, tier in reconcile.cancelled_sells:
            inst = self.state.by_instrument[ob]
            self._log("external_cancel_sell", ob_id=ob,
                      ticker=inst.ticker, tier=tier)
        for ob, cached, live in reconcile.inventory_drift:
            self._log("inventory_drift", ob_id=ob, cached=cached, live=live)
        for ob, n_buys, n_sells in reconcile.ambiguous:
            inst = self.state.by_instrument[ob]
            self._log("reconcile_ambiguous", ob_id=ob, ticker=inst.ticker,
                      missing_buys=n_buys, missing_sells=n_sells,
                      action="deferred_one_tick")

        # 2026-05-28: units from a partial fill on a still-resting buy were just
        # aligned into inventory by reconcile (under_protected) and flagged for a
        # stop rearm. If the instrument has no stop level yet (these units never
        # rotated), derive one from the (estimated) avg entry + configured stop
        # pct so the rearm block below can actually place protection.
        if reconcile.under_protected:
            from portfolio.grid_tiers import build_exit_levels
            for ob, extra in reconcile.under_protected:
                inst = self.state.by_instrument[ob]
                self._log("under_protected_inventory", ob_id=ob,
                          ticker=inst.ticker, extra_units=extra,
                          inventory_units=inst.inventory_units)
                if inst.stop_loss_price is None and inst.avg_entry_price > 0:
                    _, stop_price = build_exit_levels(
                        inst.avg_entry_price, self._target_pct, self._stop_pct,
                    )
                    inst.stop_loss_price = stop_price

        for inst in self.state.by_instrument.values():
            if inst.stop_needs_rearm and inst.inventory_units > 0 and inst.stop_loss_price:
                if not self._probe_only:
                    sp = inst.stop_loss_price
                    result = self._safe_session_call(
                        self.session.place_stop_loss,
                        inst.ob_id, sp, round(sp * 0.995, 2),
                        inst.inventory_units, default=None,
                    )
                    sid = None
                    if isinstance(result, dict) and result.get("status") == "SUCCESS":
                        sid = str(result.get("stoplossOrderId") or "") or None
                    if sid:
                        inst.stop_loss_id = sid
                        inst.stop_needs_rearm = False
                        self._log("stop_rearm_retry_ok", ob_id=inst.ob_id,
                                  ticker=inst.ticker, stop_id=sid)
                    else:
                        self._log("stop_rearm_retry_fail", ob_id=inst.ob_id,
                                  ticker=inst.ticker)

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
        # reflected). 2026-06-12 (audit B4 fix 3): the halted flag is now
        # authoritative — tick checks state.halted explicitly instead of
        # re-deriving from P&L every cycle (roll_session_if_new_day clears
        # it). On the halt TRANSITION we cancel every armed buy immediately:
        # previously they kept resting at the broker and could fill while
        # the system considered itself halted.
        if not self.state.halted:
            halt_reason = should_halt_global(self.state)
            if halt_reason:
                self.state.halted = True
                self.state.halt_reason = halt_reason
                halt_cancelled = 0
                for inst in self.state.by_instrument.values():
                    halt_cancelled += self.cancel_armed_buys(inst)
                report["cancels"] += halt_cancelled
                self._log("halt_global", reason=halt_reason,
                          cancelled_buys=halt_cancelled)
        report["halted"] = self.state.halted
        if self.state.halted:
            report["halt_reason"] = self.state.halt_reason
            # 2026-06-12 (review fix 18d9d0cc #1): a cancel that timed out
            # during the halt transition leaves its tier ARMED at the broker
            # — and the `if not state.halted` guard above never re-enters.
            # Re-attempt cancellation of any still-ARMED buy tier while
            # halted. Idempotent; throttled to every
            # _HALT_CANCEL_RETRY_TICKS ticks so the per-tier cancel_failed
            # logging inside cancel_armed_buys doesn't fire every 60s for
            # the rest of the session.
            if any(inst.armed_buy_tiers()
                   for inst in self.state.by_instrument.values()):
                self._halt_cancel_retry_countdown -= 1
                if self._halt_cancel_retry_countdown <= 0:
                    self._halt_cancel_retry_countdown = _HALT_CANCEL_RETRY_TICKS
                    retry_cancelled = 0
                    for inst in self.state.by_instrument.values():
                        if inst.armed_buy_tiers():
                            retry_cancelled += self.cancel_armed_buys(inst)
                    report["cancels"] += retry_cancelled
                    self._log("halt_cancel_retry",
                              cancelled_buys=retry_cancelled)

        # EOD handling — only sweep if we have a remaining-minutes value.
        # 2026-06-12 (audit B4 fix 3): runs BEFORE the halt early-return.
        # Previously a halted session skipped the sweep entirely, carrying
        # leveraged warrant inventory overnight precisely on the worst-loss
        # day — the one day the flat matters most.
        if eod_minutes_remaining is not None:
            if eod_minutes_remaining <= GRID_EOD_MARKET_SELL_MINUTES_BEFORE:
                report["eod_swept"] = True
                self.eod_cancel_buys()
                self.eod_market_flat()
                self._persist()
                return report
            if eod_minutes_remaining <= GRID_EOD_SWEEP_MINUTES_BEFORE:
                report["eod_swept"] = True
                self.eod_cancel_buys()
                self._persist()
                return report

        # Halted: no new placements for the rest of the session (the EOD
        # paths above still run each tick so inventory exits at day end).
        if self.state.halted:
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
            # instrument before placing on this one. The effective cap is
            # the lesser of the config max and (live Avanza buying power -
            # safety buffer); when no account_id was wired in, falls back
            # to the config max for back-compat. 2026-05-13: previously
            # this was the raw config constant, which is why grid-fisher
            # kept attempting OLJAB placements with ~3 097 SEK cash on
            # account 1625505 — the cap was right for the budget but
            # never compared against the live balance.
            global_notional = global_planned_notional(self.state)
            effective_cap, cap_debug = self._effective_global_cap()
            if global_notional >= effective_cap:
                self._log("skip_global_cap", ob_id=ob_id,
                          ticker=ticker,
                          global_notional=round(global_notional, 0),
                          cap=round(effective_cap, 0),
                          **cap_debug)
                instr_report["skip"] = "global_cap"
                report["instruments"][ob_id] = instr_report
                continue

            cat = self._catalog_for(ob_id) or {}
            placed = self.place_buy_ladder(
                inst, bid=float(bid),
                underlying_price=ob_prices.get("underlying_price"),
                barrier=cat.get("barrier") if cat.get("barrier") else None,
                leverage=cat.get("leverage"),
                global_cap_sek=effective_cap,
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
            # 2026-05-14 (P0-9 fix): if an EOD sell was already placed this
            # session, do NOT queue another one. The tick window is 60s but
            # warrant fills can lag for several minutes during illiquid
            # close auctions; without this guard, every subsequent tick
            # inside GRID_EOD_MARKET_SELL_MINUTES_BEFORE would stack a
            # fresh full-inventory sell on top of the still-resting one,
            # eventually short-selling the position once they fill.
            if inst.eod_sell_order_id is not None:
                self._log(
                    "eod_market_sell_skip_in_flight",
                    ob_id=inst.ob_id, ticker=inst.ticker,
                    eod_sell_order_id=inst.eod_sell_order_id,
                    inventory_units=inst.inventory_units,
                )
                continue
            # Cancel any armed sell tiers (don't double-up volume).
            # 2026-06-12 (audit B4 fix 4): only mark CANCELLED on a
            # confirmed orderRequestStatus=="SUCCESS" — mirrors
            # cancel_armed_buys. Previously the tier was flipped to
            # CANCELLED unconditionally; on a failed cancel the old sell
            # kept resting at the broker while the full-inventory EOD sell
            # was placed on top → combined sell volume > position
            # (short.sell.not.allowed at best, double fill at worst). On
            # any unconfirmed cancel: leave the tier ARMED, skip this
            # instrument's EOD sell this tick (stop left intact too), and
            # retry on the next tick inside the EOD window.
            sell_cancel_failed = False
            for tier in list(inst.sell_ladder):
                if tier.status != ORDER_ARMED:
                    continue
                if not tier.order_id:
                    # Never accepted by Avanza — nothing resting to cancel.
                    tier.status = ORDER_CANCELLED
                    continue
                result = self._safe_session_call(
                    self.session.cancel_order, tier.order_id,
                    default=None,
                )
                status = (result or {}).get("orderRequestStatus") \
                    if isinstance(result, dict) else None
                if status != "SUCCESS":
                    sell_cancel_failed = True
                    self._log("eod_cancel_sell_failed", ob_id=inst.ob_id,
                              ticker=inst.ticker, tier=tier.tier,
                              order_id=tier.order_id,
                              avanza_status=status,
                              message=(result or {}).get("message")
                              if isinstance(result, dict) else
                              "session_call returned None")
                    continue
                tier.status = ORDER_CANCELLED
            if sell_cancel_failed:
                # Combined sell volume must never exceed the position —
                # retry the whole flat sequence next tick.
                continue
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
                # P0 (FGL 2026-06-06): the stop was already nulled above
                # (line ~1956), so a failed sell here would leave the lot with
                # NO stop AND NO sell = naked overnight leveraged exposure — the
                # exact invariant this sweep exists to forbid. Flag re-arm so the
                # tick-time honor path restores a protective stop next tick.
                inst.stop_needs_rearm = True
                # Leave eod_sell_order_id unset so a future tick can retry
                # once the session call recovers. Without this, a single
                # transient Avanza error would skip the sweep entirely.
                continue
            status = (result or {}).get("orderRequestStatus", "UNKNOWN")
            order_id = (result or {}).get("orderId")
            if status != "SUCCESS" or not order_id:
                self._log("eod_market_sell_rejected", ob_id=inst.ob_id,
                          ticker=inst.ticker,
                          status=status,
                          qty=inst.inventory_units,
                          price=aggressive)
                # P0 (FGL 2026-06-06): stop already nulled above — re-arm so the
                # rejected lot is not left naked overnight (see None-branch).
                inst.stop_needs_rearm = True
                # Avanza rejected (e.g. trading halt, instrument suspended).
                # Same retry-next-tick rationale as the None branch.
                continue
            inst.eod_sell_order_id = str(order_id)
            # 2026-05-28: also track the EOD sell as a sell-ladder tier so
            # reconcile_against_live sees its fill and decrements inventory_units.
            # Previously only eod_sell_order_id was stored, so when the EOD order
            # filled at the broker inventory_units was never reduced — leaving
            # phantom inventory that triggered another full-size SELL the next
            # session (roll_session clears eod_sell_order_id but left inventory
            # stale). The EOD_SELL_TIER index can't collide with real tiers.
            inst.sell_ladder.append(TierOrder(
                tier=EOD_SELL_TIER,
                order_id=str(order_id),
                price=aggressive,
                qty=inst.inventory_units,
                placed_ts=self.now_fn(),
            ))
            self._log("eod_market_sell", ob_id=inst.ob_id,
                      ticker=inst.ticker, qty=inst.inventory_units,
                      price=aggressive, order_id=str(order_id))
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
                # Gate B visibility — surface the rapid-cancel counters
                # on the dashboard payload so operators can see WHY an
                # instrument is in cooldown and spot repeated silent-
                # reject patterns before the cooldown expires.
                "rapid_cancel_count": inst.rapid_cancel_count,
                "last_rapid_cancel_ts": inst.last_rapid_cancel_ts,
            }
            for ob, inst in state.by_instrument.items()
        },
    }
