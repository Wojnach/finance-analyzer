"""Fin Snipe Manager: stateful Avanza ladder/order manager.

The manager reconciles three pieces of state each cycle:
1. Live open orders
2. Live held positions
3. The current Fin Snipe ladder for each supported instrument

It then:
- maintains one or two resting BUY limits while flat
- switches to an automated SELL target when a fill creates a position
- cancels stale/mismatched orders so only the intended ladder remains

Dry-run is the default. Use ``--live`` explicitly to execute actions.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import logging
import os
import time
from pathlib import Path
from typing import Any

from portfolio.avanza_control import delete_order_live, delete_stop_loss, place_order, place_stop_loss
from portfolio.avanza_session import _get_playwright_context, close_playwright, verify_session
from portfolio.exit_optimizer import MarketSnapshot, Position, compute_exit_plan
from portfolio.file_utils import (
    atomic_append_jsonl,
    atomic_write_json,
    load_json,
    prune_jsonl,
)
from portfolio.fin_snipe import build_snapshots
from portfolio.metals_ladder import translate_underlying_target
from portfolio.process_lock import acquire_lock_file, release_lock_file
from portfolio.session_calendar import get_session_info

BASE_DIR = Path(__file__).resolve().parent.parent
STATE_FILE = BASE_DIR / "data" / "fin_snipe_state.json"
MANAGER_LOG_FILE = BASE_DIR / "data" / "fin_snipe_manager_log.jsonl"
PREDICTION_LOG_FILE = BASE_DIR / "data" / "fin_snipe_predictions.jsonl"
LOCK_FILE = BASE_DIR / "data" / "fin_snipe_manager.singleton.lock"

FLASH_ENTRY_VOLUME_PCT = 0.30
DEFAULT_HOURS = 6.0
DEFAULT_INTERVAL_SECONDS = 60
LOG_PRUNE_BYTES = 25_000_000
LOG_MAX_ENTRIES = 20_000
EXIT_OPTIMIZER_N_PATHS = 2000
EXIT_OPTIMIZER_SEED = 42
HARD_STOP_CERT_PCT = 0.05
HARD_STOP_SELL_BUFFER_PCT = 0.01
HARD_STOP_VALID_DAYS = 8
MIN_STOP_DISTANCE_PCT = 1.0
FAST_RECHECK_SECONDS = 5
MAX_FAST_RECHECK_CYCLES = 6  # After 6 consecutive fast rechecks, fall back to normal interval
MAX_CANCEL_RETRIES = 3  # After N consecutive failed cancels, mark order as dead
DEAD_ORDER_EXPIRY_HOURS = 4  # Remove dead order reservations after this many hours
CRITICAL_ALERT_COOLDOWN_SECONDS = 1800  # 30 min between same-category alerts

logger = logging.getLogger("portfolio.fin_snipe_manager")

# Throttle state for critical Telegram alerts (category -> last_sent ISO timestamp)
_critical_alert_last: dict[str, str] = {}


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _notify_critical(category: str, message: str) -> None:
    """Send a throttled Telegram alert for critical fin_snipe_manager events.

    Categories: 'session_expired', 'naked_position', 'execution_failure',
    'phantom_orders'. Throttled to one per category per CRITICAL_ALERT_COOLDOWN_SECONDS.
    """
    now = dt.datetime.now(dt.timezone.utc)
    last_raw = _critical_alert_last.get(category)
    if last_raw:
        try:
            last = dt.datetime.fromisoformat(last_raw)
            if (now - last).total_seconds() < CRITICAL_ALERT_COOLDOWN_SECONDS:
                logger.debug("Critical alert throttled: %s", category)
                return
        except (ValueError, TypeError):
            pass

    _critical_alert_last[category] = now.isoformat()
    try:
        import json as _json
        from portfolio.message_store import send_or_store
        config = _json.load(open(BASE_DIR / "config.json"))
        send_or_store(message, config, category="error")
    except Exception:
        logger.warning("Failed to send critical alert: %s", message, exc_info=True)


def _extract_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _new_session_id() -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"fin-snipe-{stamp}-pid{os.getpid()}"


def _host_name() -> str:
    return (
        os.environ.get("COMPUTERNAME")
        or os.environ.get("HOSTNAME")
        or "unknown-host"
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _append_log(path: Path, event: str, payload: dict[str, Any]) -> None:
    entry = {
        "ts": _now_utc(),
        "source": "fin_snipe_manager",
        "event": event,
        **_json_safe(payload),
    }
    atomic_append_jsonl(path, entry)


def _maybe_prune_log(path: Path) -> None:
    try:
        if path.exists() and path.stat().st_size >= LOG_PRUNE_BYTES:
            prune_jsonl(path, max_entries=LOG_MAX_ENTRIES)
    except Exception:
        logger.debug("Fin Snipe log prune failed for %s", path, exc_info=True)


def _round_order_price(price: float) -> float:
    if price <= 0:
        return 0.0
    if price < 1:
        return round(price, 3)
    return round(price, 2)


def _price_abs_tolerance(price: float) -> float:
    if price < 1:
        return 0.002
    if price < 20:
        return 0.02
    return 0.25


def _price_matches(left: float, right: float) -> bool:
    ref = max(abs(left), abs(right), 1.0)
    tol = max(_price_abs_tolerance(ref), ref * 0.0025)
    return abs(left - right) <= tol


def _active_orders(snapshot: dict, side: str) -> list[dict]:
    wanted = side.upper()
    return [
        order for order in (snapshot.get("open_orders") or [])
        if str(order.get("side") or "").upper() == wanted
        and str(order.get("state") or "").upper() == "ACTIVE"
    ]


def _stop_order_id(order: dict) -> str:
    return str(order.get("id") or order.get("orderId") or "")


def _stop_order_status(order: dict) -> str:
    return str(order.get("status") or order.get("state") or "").upper()


def _stop_order_trigger(order: dict) -> float:
    trigger = order.get("trigger") or {}
    return float(_extract_value(trigger.get("value")) or 0.0)


def _stop_order_sell_price(order: dict) -> float:
    stop_order = order.get("order") or {}
    return float(_extract_value(stop_order.get("price")) or 0.0)


def _stop_order_volume(order: dict) -> int:
    stop_order = order.get("order") or {}
    return int(_extract_value(stop_order.get("volume")) or 0)


def _active_stop_orders(snapshot: dict) -> list[dict]:
    return [
        order for order in (snapshot.get("stop_orders") or [])
        if _stop_order_status(order) == "ACTIVE"
    ]


def _relevant_stop_orders(snapshot: dict) -> list[dict]:
    return [
        order for order in (snapshot.get("stop_orders") or [])
        if _stop_order_status(order) in {"ACTIVE", "ERROR"}
        and str(((order.get("order") or {}).get("type") or "")).upper() == "SELL"
    ]


def _candidate_entry_orders(snapshot: dict, instrument_state: dict, desired_buys: list[dict]) -> list[dict]:
    active_buys = _active_orders(snapshot, "BUY")
    managed_ids = {str(order_id) for order_id in (instrument_state.get("managed_order_ids") or []) if order_id}
    if managed_ids:
        return [order for order in active_buys if str(order.get("orderId") or "") in managed_ids]

    # Only adopt unknown orders on first-ever cycle (no prior state).
    # Once the instrument has state, strict managed-only to protect manual orders.
    if instrument_state:
        return []

    adopted: list[dict] = []
    for order in active_buys:
        price = float(order.get("price") or 0.0)
        volume = int(order.get("volume") or 0)
        if any(
            volume == int(target.get("volume") or 0)
            and _price_matches(price, float(target.get("price") or 0.0))
            for target in desired_buys
        ):
            adopted.append(order)
    return adopted


def _candidate_exit_orders(snapshot: dict, instrument_state: dict) -> list[dict]:
    dead_ids = {str(oid) for oid in (instrument_state.get("dead_order_ids") or []) if oid}
    active_sells = [o for o in _active_orders(snapshot, "SELL") if str(o.get("orderId") or "") not in dead_ids]
    managed_ids = {str(order_id) for order_id in (instrument_state.get("managed_order_ids") or []) if order_id}
    matched = [order for order in active_sells if str(order.get("orderId") or "") in managed_ids]
    return matched


def _candidate_stop_orders(snapshot: dict, instrument_state: dict) -> list[dict]:
    relevant = _relevant_stop_orders(snapshot)
    managed_ids = {str(order_id) for order_id in (instrument_state.get("managed_stop_ids") or []) if order_id}
    matched = [order for order in relevant if _stop_order_id(order) in managed_ids]
    return matched


def _stage_replacements(
    cancels: list[dict],
    placements: list[dict],
    *,
    event: str,
    events: list[str],
    emergency: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Two-phase order replacement: cancel this cycle, place next cycle.

    When ``emergency=True``, both cancel and place happen in the same cycle.
    This is used when the position has NO existing sell or stop protection —
    waiting a full cycle would leave the position completely naked.
    """
    if cancels and placements:
        if emergency:
            events.append(f"{event}_emergency")
            return cancels, placements  # Both in one cycle — position is naked
        events.append(event)
        return cancels, []
    return cancels, placements


def _estimate_entry_underlying(snapshot: dict, instrument_state: dict) -> float:
    saved = float(instrument_state.get("entry_underlying") or 0.0)
    if saved > 0:
        return saved

    current_underlying = float(snapshot.get("current_underlying") or 0.0)
    current_price = float(snapshot.get("current_instrument_price") or 0.0)
    entry_price = float(snapshot.get("position_average_price") or 0.0)
    leverage = float(snapshot.get("leverage") or 0.0)
    if current_underlying <= 0 or current_price <= 0 or entry_price <= 0 or leverage <= 0:
        return current_underlying

    instrument_return = (current_price / entry_price) - 1.0
    underlying_return = instrument_return / leverage
    base = 1.0 + underlying_return
    if base <= 0:
        return current_underlying
    return current_underlying / base


def _compute_exit_target(snapshot: dict, instrument_state: dict) -> dict[str, Any]:
    ladder = snapshot["ladder"]
    fallback_price = _round_order_price(ladder.get("exit_price") or 0.0)
    fallback_underlying = float(ladder.get("exit_underlying") or snapshot.get("current_underlying") or 0.0)
    current_bid = float(snapshot.get("current_bid") or 0.0)

    result = {
        "price": max(fallback_price, _round_order_price(current_bid)) if current_bid > 0 else fallback_price,
        "underlying_price": fallback_underlying,
        "source": "ladder",
        "fill_prob": None,
        "expected_fill_time_min": None,
        "stop_hit_prob": None,
        "risk_flags": [],
        "action": "limit",
        "optimizer_price": None,
        "optimizer_underlying_price": None,
    }

    position_volume = int(snapshot.get("position_volume") or 0)
    position_avg = float(snapshot.get("position_average_price") or 0.0)
    current_underlying = float(snapshot.get("current_underlying") or 0.0)
    leverage = float(snapshot.get("leverage") or 0.0)
    signal_entry = snapshot.get("signal_entry") or {}
    atr_pct = float(
        (signal_entry.get("extra") or {}).get("atr_pct")
        or signal_entry.get("atr_pct")
        or 0.0
    )
    session = get_session_info("warrant", underlying=snapshot.get("ticker"))
    if (
        position_volume <= 0
        or position_avg <= 0
        or current_underlying <= 0
        or leverage <= 0
        or not session.is_open
        or session.remaining_minutes < 2
    ):
        return result

    try:
        market_summary = _summarize_market(snapshot)
        underlying_summary = market_summary.get("underlying") or {}
        plan = compute_exit_plan(
            Position(
                symbol=snapshot["ticker"],
                qty=position_volume,
                entry_price_sek=position_avg,
                entry_underlying_usd=_estimate_entry_underlying(snapshot, instrument_state),
                entry_ts=dt.datetime.now(dt.timezone.utc),
                instrument_type="warrant",
                leverage=leverage,
                financing_level=None,
            ),
            MarketSnapshot(
                asof_ts=dt.datetime.now(dt.timezone.utc),
                price=current_underlying,
                bid=float(underlying_summary.get("bid") or current_underlying),
                ask=float(underlying_summary.get("ask") or current_underlying),
                atr_pct=atr_pct if atr_pct > 0 else None,
                usdsek=1.0,
                drift=0.0,
            ),
            session.session_end,
            n_paths=EXIT_OPTIMIZER_N_PATHS,
            seed=EXIT_OPTIMIZER_SEED,
        )
        target_underlying = float(plan.recommended.price_usd or fallback_underlying)
        translated = translate_underlying_target(
            float(snapshot.get("current_instrument_price") or current_bid or position_avg),
            current_underlying,
            target_underlying,
            leverage,
        )
        exit_price = _round_order_price(translated or fallback_price)
        minimum_profit_price = _round_order_price(max(position_avg, 0.0))
        if plan.recommended.action == "market" and current_bid > 0:
            exit_price = _round_order_price(current_bid)
            source = "quant_exit_optimizer_market"
        elif current_bid > 0:
            exit_price = max(exit_price, _round_order_price(current_bid))
            source = "quant_exit_optimizer"
            if exit_price < minimum_profit_price:
                exit_price = max(fallback_price, minimum_profit_price)
                source = "ladder_profit_guard"
        else:
            source = "quant_exit_optimizer"

        result.update({
            "price": exit_price,
            "underlying_price": target_underlying,
            "source": source,
            "fill_prob": float(plan.recommended.fill_prob),
            "expected_fill_time_min": float(plan.recommended.expected_fill_time_min),
            "stop_hit_prob": float(plan.stop_hit_prob),
            "risk_flags": list(plan.recommended.risk_flags),
            "action": plan.recommended.action,
            "optimizer_price": _round_order_price(translated or 0.0),
            "optimizer_underlying_price": target_underlying,
        })
    except Exception:
        logger.warning("Exit optimizer failed for %s", snapshot.get("orderbook_id"), exc_info=True)
    return result


def _compute_stop_plan(snapshot: dict, *, has_existing_stop: bool = False) -> dict[str, Any] | None:
    position_volume = int(snapshot.get("position_volume") or 0)
    position_avg = float(snapshot.get("position_average_price") or 0.0)
    current_bid = float(snapshot.get("current_bid") or 0.0)
    if position_volume <= 0 or position_avg <= 0:
        return None

    trigger_price = _round_order_price(position_avg * (1.0 - HARD_STOP_CERT_PCT))
    sell_price = _round_order_price(trigger_price * (1.0 - HARD_STOP_SELL_BUFFER_PCT))
    if trigger_price <= 0 or sell_price <= 0:
        return None

    distance_pct = ((current_bid - trigger_price) / current_bid * 100.0) if current_bid > 0 else None

    # Hysteresis: if we already have a managed stop, keep it regardless of distance.
    # Only skip placement of NEW stops when too close.
    if not has_existing_stop and distance_pct is not None and distance_pct < MIN_STOP_DISTANCE_PCT:
        return {
            "skip": True,
            "reason": "stop_too_close",
            "distance_pct": round(distance_pct, 2),
            "trigger_price": trigger_price,
            "sell_price": sell_price,
            "volume": position_volume,
        }

    return {
        "skip": False,
        "reason": "keep_existing" if has_existing_stop and distance_pct is not None and distance_pct < MIN_STOP_DISTANCE_PCT else "entry_minus_5pct",
        "distance_pct": round(distance_pct, 2) if distance_pct is not None else None,
        "trigger_price": trigger_price,
        "sell_price": sell_price,
        "volume": position_volume,
        "valid_days": HARD_STOP_VALID_DAYS,
    }


def _summarize_order(order: dict) -> dict[str, Any]:
    orderbook = order.get("orderbook") or {}
    account = order.get("account") or {}
    return {
        "order_id": str(order.get("orderId") or ""),
        "side": str(order.get("side") or "").upper(),
        "state": str(order.get("state") or "").upper(),
        "price": float(order.get("price") or 0.0),
        "volume": int(order.get("volume") or 0),
        "orderbook_id": str(order.get("orderbookId") or orderbook.get("id") or ""),
        "name": orderbook.get("name") or "",
        "account_id": str(account.get("accountId") or account.get("id") or ""),
    }


def _summarize_market(snapshot: dict) -> dict[str, Any]:
    market = snapshot.get("market") or {}
    quote = snapshot.get("quote") or {}
    underlying = (market.get("underlying") or {})
    underlying_quote = underlying.get("quote") or {}
    indicators = market.get("keyIndicators") or {}
    return {
        "quote": {
            "bid": float(snapshot.get("current_bid") or 0.0),
            "ask": float(snapshot.get("current_ask") or 0.0),
            "last": float(snapshot.get("current_last") or 0.0),
        },
        "instrument": {
            "price": float(snapshot.get("current_instrument_price") or 0.0),
            "type": snapshot.get("instrument_type") or "",
            "leverage": float(snapshot.get("leverage") or 0.0),
        },
        "underlying": {
            "name": str(underlying.get("name") or "").strip(),
            "price": float(snapshot.get("current_underlying") or 0.0),
            "bid": float(_extract_value(underlying_quote.get("buy")) or 0.0),
            "ask": float(_extract_value(underlying_quote.get("sell")) or 0.0),
            "last": float(_extract_value(underlying_quote.get("last")) or 0.0),
        },
        "risk": {
            "barrier_level": float(_extract_value(indicators.get("barrierLevel")) or 0.0),
            "financing_level": float(_extract_value(indicators.get("financingLevel")) or 0.0),
            "leverage": float(_extract_value(indicators.get("leverage")) or snapshot.get("leverage") or 0.0),
            "parity": float(_extract_value(indicators.get("parity")) or 0.0),
        },
        "raw_quote": {
            "buy": _extract_value(quote.get("buy")),
            "sell": _extract_value(quote.get("sell")),
            "last": _extract_value(quote.get("last")),
        },
    }


def _summarize_snapshot(snapshot: dict) -> dict[str, Any]:
    return {
        "orderbook_id": snapshot["orderbook_id"],
        "name": snapshot["name"],
        "ticker": snapshot["ticker"],
        "market": _summarize_market(snapshot),
        "position": {
            "volume": int(snapshot.get("position_volume") or 0),
            "average_price": float(snapshot.get("position_average_price") or 0.0),
            "value_sek": float(snapshot.get("position_value_sek") or 0.0),
        },
        "open_orders": [_summarize_order(order) for order in (snapshot.get("open_orders") or [])],
        "stop_orders": [
            {
                "stop_id": _stop_order_id(order),
                "status": _stop_order_status(order),
                "trigger_price": _stop_order_trigger(order),
                "sell_price": _stop_order_sell_price(order),
                "volume": _stop_order_volume(order),
                "message": str(order.get("message") or ""),
            }
            for order in (snapshot.get("stop_orders") or [])
        ],
        "ladder": _json_safe(snapshot.get("ladder") or {}),
    }


def _summarize_action(action: dict) -> dict[str, Any]:
    summary = {
        "action": action.get("action"),
        "order_type": action.get("order_type") or "limit_order",
        "side": action.get("side"),
        "role": action.get("role"),
        "orderbook_id": str(action.get("orderbook_id") or ""),
        "account_id": str(action.get("account_id") or ""),
        "name": action.get("name") or "",
        "price": float(action.get("price") or 0.0),
        "volume": int(action.get("volume") or 0),
    }
    if action.get("order_id"):
        summary["order_id"] = str(action.get("order_id") or "")
    if action.get("trigger_price") is not None:
        summary["trigger_price"] = float(action.get("trigger_price") or 0.0)
    if action.get("valid_days") is not None:
        summary["valid_days"] = int(action.get("valid_days") or 0)
    return summary


def _summarize_result(result: dict) -> dict[str, Any]:
    summary = _summarize_action(result)
    summary["ok"] = bool(result.get("ok") or result.get("dry_run"))
    if "dry_run" in result:
        summary["dry_run"] = bool(result.get("dry_run"))
    if result.get("result") is not None:
        summary["result"] = _json_safe(result.get("result"))
    return summary


def _cycle_metadata(
    *,
    session_id: str,
    cycle_index: int,
    live: bool,
    hours_remaining: float,
    orderbook_filter: set[str] | None,
    budget_sek: float | None,
    simulate_flash_window: bool,
    state_path: Path,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "cycle_index": cycle_index,
        "cycle_id": f"{session_id}:{cycle_index}",
        "live": live,
        "hours_remaining": round(float(hours_remaining), 4),
        "orderbook_filter": sorted(str(orderbook_id) for orderbook_id in (orderbook_filter or set())),
        "budget_sek": None if budget_sek is None else round(float(budget_sek), 2),
        "simulate_flash_window": bool(simulate_flash_window),
        "pid": os.getpid(),
        "host": _host_name(),
        "state_path": str(state_path),
    }


def log_cycle_plan(
    *,
    session_id: str,
    cycle_index: int,
    live: bool,
    hours_remaining: float,
    orderbook_filter: set[str] | None,
    budget_sek: float | None,
    simulate_flash_window: bool,
    state_path: Path,
    snapshots: list[dict],
    state_before: dict,
    plans: list[dict],
    actions: list[dict],
    manager_log_path: Path = MANAGER_LOG_FILE,
    prediction_log_path: Path = PREDICTION_LOG_FILE,
) -> None:
    meta = _cycle_metadata(
        session_id=session_id,
        cycle_index=cycle_index,
        live=live,
        hours_remaining=hours_remaining,
        orderbook_filter=orderbook_filter,
        budget_sek=budget_sek,
        simulate_flash_window=simulate_flash_window,
        state_path=state_path,
    )
    plan_map = {str(plan["orderbook_id"]): plan for plan in plans}
    instrument_entries: list[dict[str, Any]] = []

    for snapshot in snapshots:
        orderbook_id = str(snapshot["orderbook_id"])
        plan = plan_map.get(orderbook_id, {})
        previous_state = ((state_before.get("instruments") or {}).get(orderbook_id) or {})
        prediction_entry = {
            "ts": _now_utc(),
            "source": "fin_snipe_manager",
            "event": "prediction_snapshot",
            **meta,
            "orderbook_id": orderbook_id,
            "name": snapshot["name"],
            "ticker": snapshot["ticker"],
            "mode": plan.get("mode") or "unknown",
            "events": list(plan.get("events") or []),
            "entry_volume": int(plan.get("entry_volume") or 0),
            "position_volume": int(plan.get("position_volume") or 0),
            "state_before": _json_safe(previous_state),
            "snapshot": _summarize_snapshot(snapshot),
            "exit_target": _json_safe(plan.get("exit_target") or {}),
            "stop_plan": _json_safe(plan.get("stop_plan") or {}),
            "planned_actions": [_summarize_action(action) for action in (plan.get("actions") or [])],
        }
        atomic_append_jsonl(prediction_log_path, prediction_entry)
        instrument_entries.append({
            "orderbook_id": orderbook_id,
            "name": snapshot["name"],
            "ticker": snapshot["ticker"],
            "mode": plan.get("mode") or "unknown",
            "events": list(plan.get("events") or []),
            "entry_volume": int(plan.get("entry_volume") or 0),
            "position_volume": int(plan.get("position_volume") or 0),
            "state_before": _json_safe(previous_state),
            "snapshot": _summarize_snapshot(snapshot),
            "exit_target": _json_safe(plan.get("exit_target") or {}),
            "stop_plan": _json_safe(plan.get("stop_plan") or {}),
            "planned_actions": [_summarize_action(action) for action in (plan.get("actions") or [])],
        })

    _append_log(
        manager_log_path,
        "cycle_plan",
        {
            **meta,
            "instrument_count": len(instrument_entries),
            "action_count": len(actions),
            "instruments": instrument_entries,
        },
    )


def log_action_result(
    result: dict,
    *,
    session_id: str,
    cycle_index: int,
    live: bool,
    manager_log_path: Path = MANAGER_LOG_FILE,
) -> None:
    _append_log(
        manager_log_path,
        "action_result",
        {
            "session_id": session_id,
            "cycle_index": cycle_index,
            "cycle_id": f"{session_id}:{cycle_index}",
            "live": live,
            "result": _summarize_result(result),
        },
    )


def log_cycle_results(
    *,
    session_id: str,
    cycle_index: int,
    live: bool,
    state_after: dict,
    results: list[dict],
    manager_log_path: Path = MANAGER_LOG_FILE,
    prediction_log_path: Path = PREDICTION_LOG_FILE,
) -> None:
    _append_log(
        manager_log_path,
        "cycle_complete",
        {
            "session_id": session_id,
            "cycle_index": cycle_index,
            "cycle_id": f"{session_id}:{cycle_index}",
            "live": live,
            "ok_count": sum(1 for result in results if result.get("ok") or result.get("dry_run")),
            "fail_count": sum(1 for result in results if not (result.get("ok") or result.get("dry_run"))),
            "result_count": len(results),
            "results": [_summarize_result(result) for result in results],
            "state_after": _json_safe(state_after),
        },
    )
    _maybe_prune_log(manager_log_path)
    _maybe_prune_log(prediction_log_path)


def log_cycle_failure(
    exc: Exception,
    *,
    session_id: str,
    cycle_index: int,
    live: bool,
    hours_remaining: float,
    orderbook_filter: set[str] | None,
    budget_sek: float | None,
    simulate_flash_window: bool,
    state_path: Path,
    manager_log_path: Path = MANAGER_LOG_FILE,
) -> None:
    meta = _cycle_metadata(
        session_id=session_id,
        cycle_index=cycle_index,
        live=live,
        hours_remaining=hours_remaining,
        orderbook_filter=orderbook_filter,
        budget_sek=budget_sek,
        simulate_flash_window=simulate_flash_window,
        state_path=state_path,
    )
    _append_log(
        manager_log_path,
        "cycle_failure",
        {
            **meta,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        },
    )
    _maybe_prune_log(manager_log_path)


def _managed_orders(snapshot: dict, instrument_state: dict, side: str, managed_only: bool) -> list[dict]:
    """Return active orders relevant to the current Fin Snipe strategy."""
    all_orders = _active_orders(snapshot, side)
    if not managed_only:
        return all_orders

    managed_ids = {str(order_id) for order_id in (instrument_state.get("managed_order_ids") or []) if order_id}
    if not managed_ids:
        return []
    return [order for order in all_orders if str(order.get("orderId") or "") in managed_ids]


def _default_state() -> dict:
    return {
        "version": 1,
        "updated_at": None,
        "instruments": {},
    }


def load_state(path: Path = STATE_FILE) -> dict:
    state = load_json(path, default=_default_state())
    if not isinstance(state, dict):
        return _default_state()
    state.setdefault("version", 1)
    state.setdefault("updated_at", None)
    state.setdefault("instruments", {})
    return state


def save_state(state: dict, path: Path = STATE_FILE) -> None:
    atomic_write_json(path, state, ensure_ascii=False)


def _seed_entry_volume(snapshot: dict, instrument_state: dict) -> int:
    saved_volume = int(instrument_state.get("entry_volume") or 0)
    if saved_volume > 0:
        return saved_volume

    open_buy_volume = sum(int(order.get("volume") or 0) for order in _active_orders(snapshot, "BUY"))
    position_volume = int(snapshot.get("position_volume") or 0)
    observed_volume = open_buy_volume + position_volume
    return max(observed_volume, 0)


def _entry_volume_from_budget(snapshot: dict, budget_sek: float) -> int:
    ladder = snapshot["ladder"]
    working_price = _round_order_price(ladder.get("working_price") or ladder.get("mean_price") or 0.0)
    if working_price <= 0:
        return 0
    return max(int(budget_sek // working_price), 0)


def _budgeted_entry_volume(snapshot: dict, instrument_state: dict, budget_sek: float) -> int:
    max_volume = _entry_volume_from_budget(snapshot, budget_sek)
    if max_volume <= 0:
        return 0

    saved_volume = int(instrument_state.get("entry_volume") or 0)
    if saved_volume > 0:
        return min(saved_volume, max_volume)
    return max_volume


def _desired_buy_orders(snapshot: dict, entry_volume: int) -> list[dict]:
    if entry_volume <= 0:
        return []

    ladder = snapshot["ladder"]
    current_bid = float(snapshot.get("current_bid") or 0.0)
    working_price = _round_order_price(ladder.get("working_price") or ladder.get("mean_price") or 0.0)
    if current_bid > 0:
        working_price = min(working_price, _round_order_price(current_bid))

    flash_price = _round_order_price(ladder.get("flash_price") or 0.0)
    if flash_price > 0 and current_bid > 0:
        flash_price = min(flash_price, _round_order_price(current_bid))

    if flash_price <= 0 or entry_volume < 2 or _price_matches(working_price, flash_price):
        return [{"side": "BUY", "price": working_price, "volume": entry_volume, "role": "working", "order_type": "limit_order"}]

    flash_volume = max(1, int(round(entry_volume * FLASH_ENTRY_VOLUME_PCT)))
    flash_volume = min(flash_volume, entry_volume - 1)
    working_volume = entry_volume - flash_volume
    return [
        {"side": "BUY", "price": working_price, "volume": working_volume, "role": "working", "order_type": "limit_order"},
        {"side": "BUY", "price": flash_price, "volume": flash_volume, "role": "flash", "order_type": "limit_order"},
    ]


def _dead_sell_volume(snapshot: dict, instrument_state: dict) -> int:
    """Volume reserved by phantom/dead sell orders still showing in Avanza's API."""
    dead_ids = {str(oid) for oid in (instrument_state.get("dead_order_ids") or []) if oid}
    if not dead_ids:
        return 0
    return sum(
        int(o.get("volume") or 0)
        for o in _active_orders(snapshot, "SELL")
        if str(o.get("orderId") or "") in dead_ids
    )


def _desired_sell_orders(snapshot: dict, instrument_state: dict) -> tuple[list[dict], dict[str, Any]]:
    position_volume = int(snapshot.get("position_volume") or 0)
    if position_volume <= 0:
        return [], {}

    # Subtract volume reserved by phantom sell orders that we can't cancel
    reserved = _dead_sell_volume(snapshot, instrument_state)
    sellable_volume = position_volume - reserved
    if sellable_volume <= 0:
        return [], {}

    exit_target = _compute_exit_target(snapshot, instrument_state)
    exit_price = _round_order_price(exit_target.get("price") or 0.0)
    if exit_price <= 0:
        return [], exit_target
    return ([{
        "side": "SELL",
        "price": exit_price,
        "volume": sellable_volume,
        "role": "exit",
        "order_type": "limit_order",
    }], exit_target)


def _desired_stop_orders(snapshot: dict, *, has_existing_stop: bool = False) -> tuple[list[dict], dict[str, Any] | None]:
    stop_plan = _compute_stop_plan(snapshot, has_existing_stop=has_existing_stop)
    if not stop_plan or stop_plan.get("skip"):
        return [], stop_plan
    return ([{
        "side": "SELL",
        "order_type": "stop_loss",
        "trigger_price": float(stop_plan["trigger_price"]),
        "price": float(stop_plan["sell_price"]),
        "volume": int(stop_plan["volume"]),
        "role": "protective_stop",
        "valid_days": int(stop_plan.get("valid_days") or HARD_STOP_VALID_DAYS),
    }], stop_plan)


def _reconcile_orders(existing: list[dict], desired: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (cancels, placements) for one side on one instrument."""
    cancels: list[dict] = []
    placements: list[dict] = []
    unmatched = list(existing)

    for target in desired:
        matched_index = None
        for idx, order in enumerate(unmatched):
            price = float(order.get("price") or 0.0)
            volume = int(order.get("volume") or 0)
            if volume == int(target["volume"]) and _price_matches(price, float(target["price"])):
                matched_index = idx
                break
        if matched_index is None:
            placements.append({"action": "place", **target})
            continue
        unmatched.pop(matched_index)

    for order in unmatched:
        cancels.append({
            "action": "cancel",
            "order_id": str(order.get("orderId") or ""),
            "side": str(order.get("side") or "").upper(),
            "price": float(order.get("price") or 0.0),
            "volume": int(order.get("volume") or 0),
        })
    return cancels, placements


def _stop_matches(existing: dict, target: dict) -> bool:
    return (
        _stop_order_volume(existing) == int(target.get("volume") or 0)
        and _price_matches(_stop_order_trigger(existing), float(target.get("trigger_price") or 0.0))
        and _price_matches(_stop_order_sell_price(existing), float(target.get("price") or 0.0))
    )


def _reconcile_stop_orders(existing: list[dict], desired: list[dict]) -> tuple[list[dict], list[dict]]:
    cancels: list[dict] = []
    placements: list[dict] = []
    unmatched = list(existing)

    for target in desired:
        matched_index = None
        for idx, order in enumerate(unmatched):
            if _stop_matches(order, target):
                matched_index = idx
                break
        if matched_index is None:
            placements.append({"action": "place", **target})
            continue
        unmatched.pop(matched_index)

    for order in unmatched:
        cancels.append({
            "action": "cancel",
            "order_type": "stop_loss",
            "order_id": _stop_order_id(order),
            "side": "SELL",
            "trigger_price": _stop_order_trigger(order),
            "price": _stop_order_sell_price(order),
            "volume": _stop_order_volume(order),
        })
    return cancels, placements


def plan_instrument(
    snapshot: dict,
    instrument_state: dict | None = None,
    *,
    budget_sek: float | None = None,
) -> dict:
    """Build the desired state and actions for one instrument snapshot."""
    instrument_state = copy.deepcopy(instrument_state or {})

    # Prune stale tracking data: remove entries for orders no longer on Avanza
    active_order_ids = {str(o.get("orderId") or "") for o in (snapshot.get("open_orders") or [])}
    active_stop_ids = {_stop_order_id(o) for o in (snapshot.get("stop_orders") or [])}
    all_live_ids = active_order_ids | active_stop_ids
    if instrument_state.get("dead_order_ids"):
        instrument_state["dead_order_ids"] = [
            oid for oid in instrument_state["dead_order_ids"] if str(oid) in all_live_ids
        ]
    if instrument_state.get("cancel_fail_counts"):
        instrument_state["cancel_fail_counts"] = {
            oid: count for oid, count in instrument_state["cancel_fail_counts"].items()
            if str(oid) in all_live_ids
        }
    # Time-based expiry for dead orders: if a dead order has been dead for
    # longer than DEAD_ORDER_EXPIRY_HOURS, remove it regardless of API state.
    # This prevents permanently blocked selling when phantom orders persist.
    dead_ts = dict(instrument_state.get("dead_order_timestamps") or {})
    now = dt.datetime.now(dt.timezone.utc)
    expired_dead = set()
    for oid, ts_str in list(dead_ts.items()):
        try:
            marked_at = dt.datetime.fromisoformat(ts_str)
            if (now - marked_at).total_seconds() > DEAD_ORDER_EXPIRY_HOURS * 3600:
                expired_dead.add(oid)
        except (ValueError, TypeError):
            expired_dead.add(oid)  # Can't parse timestamp — remove
    if expired_dead:
        instrument_state["dead_order_ids"] = [
            oid for oid in (instrument_state.get("dead_order_ids") or [])
            if oid not in expired_dead
        ]
        for oid in expired_dead:
            dead_ts.pop(oid, None)
        logger.info("Expired %d dead order(s) after %dh: %s", len(expired_dead), DEAD_ORDER_EXPIRY_HOURS, expired_dead)
    instrument_state["dead_order_timestamps"] = dead_ts

    entry_volume = (
        _budgeted_entry_volume(snapshot, instrument_state, budget_sek)
        if budget_sek is not None
        else _seed_entry_volume(snapshot, instrument_state)
    )
    position_volume = int(snapshot.get("position_volume") or 0)
    actions: list[dict] = []
    events: list[str] = []
    exit_target: dict[str, Any] = {}
    stop_plan: dict[str, Any] | None = None
    desired_buys: list[dict] = []
    desired_sells: list[dict] = []
    desired_stops: list[dict] = []
    open_buys: list[dict] = []
    open_sells: list[dict] = []
    open_stops: list[dict] = []

    if position_volume > 0:
        mode = "exit"
        desired_sells, exit_target = _desired_sell_orders(snapshot, instrument_state)
        has_managed_stop = bool(instrument_state.get("managed_stop_ids"))
        desired_stops, stop_plan = _desired_stop_orders(snapshot, has_existing_stop=has_managed_stop)
        open_buys = _candidate_entry_orders(
            snapshot,
            instrument_state,
            _desired_buy_orders(snapshot, entry_volume) if entry_volume > 0 else [],
        )
        open_sells = _candidate_exit_orders(snapshot, instrument_state)
        open_stops = _candidate_stop_orders(snapshot, instrument_state)
        if instrument_state.get("mode") != "exit":
            events.append("position_detected")
        if int(instrument_state.get("last_position_volume") or 0) not in (0, position_volume):
            events.append("position_volume_changed")
        # Detect naked position: no existing managed sell or stop orders.
        # In emergency mode, bypass two-phase staged replacement — cancel
        # and re-place in the same cycle to avoid leaving the position
        # completely unprotected for a full cycle interval.
        sell_naked = len(open_sells) == 0
        stop_naked = len(open_stops) == 0
        cancel_buys, _ = _reconcile_orders(open_buys, desired_buys)
        cancel_sells, place_sells = _reconcile_orders(open_sells, desired_sells)
        cancel_sells, place_sells = _stage_replacements(
            cancel_sells,
            place_sells,
            event="sell_reprice_pending",
            events=events,
            emergency=sell_naked,
        )
        if stop_plan and stop_plan.get("skip"):
            events.append(str(stop_plan.get("reason") or "stop_skipped"))
        cancel_stops, place_stops = _reconcile_stop_orders(open_stops, desired_stops)
        cancel_stops, place_stops = _stage_replacements(
            cancel_stops,
            place_stops,
            event="stop_reprice_pending",
            events=events,
            emergency=stop_naked,
        )
        actions.extend(cancel_buys)
        actions.extend(cancel_sells)
        actions.extend(place_sells)
        actions.extend(cancel_stops)
        actions.extend(place_stops)
    else:
        desired_buys = _desired_buy_orders(snapshot, entry_volume) if entry_volume > 0 else []
        open_buys = _candidate_entry_orders(snapshot, instrument_state, desired_buys)
        open_sells = _managed_orders(snapshot, instrument_state, "SELL", True)
        open_stops = _candidate_stop_orders(snapshot, instrument_state)
        if entry_volume > 0 or open_buys or open_sells or open_stops:
            mode = "entry"
            if instrument_state.get("mode") == "exit":
                events.append("position_flat_rearm")
            cancel_buys, place_buys = _reconcile_orders(open_buys, desired_buys)
            cancel_sells, _ = _reconcile_orders(open_sells, desired_sells)
            cancel_stops, _ = _reconcile_stop_orders(open_stops, desired_stops)
            actions.extend(cancel_buys)
            actions.extend(cancel_sells)
            actions.extend(cancel_stops)
            actions.extend(place_buys)
        else:
            mode = "idle"

    managed_order_ids = [
        str(order.get("orderId") or "")
        for order in (open_buys + open_sells)
        if str(order.get("orderId") or "")
    ]
    managed_stop_ids = [
        _stop_order_id(order)
        for order in open_stops
        if _stop_order_id(order)
    ]
    current_bid = float(snapshot.get("current_bid") or 0.0)
    stop_distance_pct = None
    if stop_plan and not stop_plan.get("skip") and current_bid > 0:
        stop_distance_pct = (current_bid - float(stop_plan.get("trigger_price") or 0.0)) / current_bid * 100.0

    next_state = {
        "budget_sek": budget_sek if budget_sek is not None else instrument_state.get("budget_sek"),
        "entry_volume": entry_volume,
        "entry_underlying": _estimate_entry_underlying(snapshot, instrument_state) if position_volume > 0 else instrument_state.get("entry_underlying"),
        "managed_order_ids": managed_order_ids,
        "managed_stop_ids": managed_stop_ids,
        "dead_order_ids": list(instrument_state.get("dead_order_ids") or []),
        "dead_order_timestamps": dict(instrument_state.get("dead_order_timestamps") or {}),
        "cancel_fail_counts": dict(instrument_state.get("cancel_fail_counts") or {}),
        "mode": mode,
        "last_position_volume": position_volume,
        "last_working_price": snapshot["ladder"].get("working_price"),
        "last_flash_price": snapshot["ladder"].get("flash_price"),
        "last_exit_price": snapshot["ladder"].get("exit_price"),
        "last_exit_target_price": exit_target.get("price"),
        "last_exit_underlying": exit_target.get("underlying_price"),
        "last_exit_source": exit_target.get("source"),
        "last_exit_fill_prob": exit_target.get("fill_prob"),
        "last_exit_time_min": exit_target.get("expected_fill_time_min"),
        "last_exit_stop_hit_prob": exit_target.get("stop_hit_prob"),
        "last_exit_risk_flags": list(exit_target.get("risk_flags") or []),
        "last_stop_trigger": stop_plan.get("trigger_price") if stop_plan else None,
        "last_stop_sell": stop_plan.get("sell_price") if stop_plan else None,
        "last_stop_reason": stop_plan.get("reason") if stop_plan else None,
        "last_stop_distance_pct": round(stop_distance_pct, 2) if stop_distance_pct is not None else stop_plan.get("distance_pct") if stop_plan else None,
        "updated_at": _now_utc(),
    }

    for action in actions:
        action["orderbook_id"] = snapshot["orderbook_id"]
        action["account_id"] = snapshot.get("account_id") or ""
        action["name"] = snapshot["name"]

    return {
        "orderbook_id": snapshot["orderbook_id"],
        "name": snapshot["name"],
        "ticker": snapshot["ticker"],
        "mode": mode,
        "entry_volume": entry_volume,
        "position_volume": position_volume,
        "actions": actions,
        "events": events,
        "state": next_state,
        "exit_target": exit_target,
        "stop_plan": stop_plan,
    }


def plan_cycle(
    snapshots: list[dict],
    state: dict | None = None,
    budgets: dict[str, float] | None = None,
) -> tuple[dict, list[dict], list[dict]]:
    """Build the full-cycle Fin Snipe plan for all supported instruments."""
    current_state = copy.deepcopy(state or _default_state())
    current_state.setdefault("instruments", {})
    plans: list[dict] = []
    actions: list[dict] = []

    for snapshot in snapshots:
        orderbook_id = snapshot["orderbook_id"]
        instrument_state = (current_state.get("instruments") or {}).get(orderbook_id) or {}
        try:
            plan = plan_instrument(
                snapshot,
                instrument_state,
                budget_sek=(budgets or {}).get(orderbook_id),
            )
        except Exception:
            name = snapshot.get("name") or orderbook_id
            logger.error(
                "plan_instrument failed for %s (%s) — skipping",
                name, orderbook_id, exc_info=True,
            )
            _notify_critical(
                "plan_failure",
                f"*SNIPE ALERT* plan_instrument failed for {name} — instrument skipped this cycle",
            )
            continue
        plans.append(plan)
        actions.extend(plan["actions"])
        current_state["instruments"][orderbook_id] = plan["state"]

    current_state["updated_at"] = _now_utc()
    return current_state, plans, actions


def _page_with_session():
    ctx = _get_playwright_context()
    page = ctx.new_page()
    page.goto("https://www.avanza.se/min-ekonomi/oversikt.html", wait_until="domcontentloaded", timeout=15000)
    return page


def execute_actions(
    actions: list[dict],
    *,
    dry_run: bool = True,
    on_result: Any | None = None,
) -> list[dict]:
    """Execute planned actions in order, or simulate them when dry-running."""
    if not actions:
        return []
    if dry_run:
        results = [{"dry_run": True, **action} for action in actions]
        if on_result is not None:
            for result in results:
                on_result(result)
        return results

    page = _page_with_session()
    results: list[dict] = []
    try:
        for action in actions:
            account_id = str(action.get("account_id") or "") or None
            order_type = str(action.get("order_type") or "limit_order")
            if action["action"] == "cancel" and order_type == "stop_loss":
                ok, result = delete_stop_loss(page, account_id, action["order_id"])
            elif action["action"] == "cancel":
                ok, result = delete_order_live(page, account_id, action["order_id"])
            elif order_type == "stop_loss":
                ok, stop_id = place_stop_loss(
                    page,
                    account_id,
                    action["orderbook_id"],
                    float(action["trigger_price"]),
                    float(action["price"]),
                    int(action["volume"]),
                    valid_days=int(action.get("valid_days") or HARD_STOP_VALID_DAYS),
                )
                result = {"stop_id": stop_id}
            else:
                ok, result = place_order(
                    page,
                    account_id,
                    action["orderbook_id"],
                    action["side"],
                    float(action["price"]),
                    int(action["volume"]),
                )
            results.append({
                "ok": ok,
                "result": result,
                **action,
            })
            if on_result is not None:
                on_result(results[-1])
    finally:
        try:
            page.close()
        except Exception:
            pass
        close_playwright()
    return results


def apply_execution_results_to_state(state: dict, results: list[dict]) -> dict:
    """Update managed order ids after a live manager cycle."""
    next_state = copy.deepcopy(state)
    instruments = next_state.setdefault("instruments", {})
    for result in results:
        orderbook_id = str(result.get("orderbook_id") or "")
        if not orderbook_id:
            continue
        inst_state = instruments.setdefault(orderbook_id, {})
        managed_ids = [
            str(order_id)
            for order_id in (inst_state.get("managed_order_ids") or [])
            if order_id
        ]
        managed_stop_ids = [
            str(order_id)
            for order_id in (inst_state.get("managed_stop_ids") or [])
            if order_id
        ]
        dead_order_ids = list(inst_state.get("dead_order_ids") or [])
        dead_order_timestamps = dict(inst_state.get("dead_order_timestamps") or {})
        order_type = str(result.get("order_type") or "limit_order")
        if result.get("action") == "cancel" and result.get("ok"):
            cancelled_id = str(result.get("order_id") or "")
            http_status = int((result.get("result") or {}).get("http_status") or 0)
            if order_type == "stop_loss":
                managed_stop_ids = [order_id for order_id in managed_stop_ids if order_id != cancelled_id]
            else:
                managed_ids = [order_id for order_id in managed_ids if order_id != cancelled_id]
            # Track phantom orders: DELETE returned 404 means the order doesn't exist
            # on the order book, but Avanza's list-orders API may still return it.
            # Exclude these from candidate selection on future cycles.
            if http_status == 404 and cancelled_id and cancelled_id not in dead_order_ids:
                dead_order_ids.append(cancelled_id)
                dead_order_timestamps[cancelled_id] = _now_utc()
            # Reset fail count on success
            cancel_fail_counts = dict(inst_state.get("cancel_fail_counts") or {})
            cancel_fail_counts.pop(cancelled_id, None)
            inst_state["cancel_fail_counts"] = cancel_fail_counts
        elif result.get("action") == "cancel" and not result.get("ok"):
            # Track consecutive cancel failures -- auto-dead after MAX_CANCEL_RETRIES
            failed_id = str(result.get("order_id") or "")
            if failed_id:
                cancel_fail_counts = dict(inst_state.get("cancel_fail_counts") or {})
                count = cancel_fail_counts.get(failed_id, 0) + 1
                if count >= MAX_CANCEL_RETRIES:
                    if order_type == "stop_loss":
                        managed_stop_ids = [oid for oid in managed_stop_ids if oid != failed_id]
                    else:
                        managed_ids = [oid for oid in managed_ids if oid != failed_id]
                    if failed_id not in dead_order_ids:
                        dead_order_ids.append(failed_id)
                        dead_order_timestamps[failed_id] = _now_utc()
                    cancel_fail_counts.pop(failed_id, None)
                    logger.warning(
                        "Order %s failed cancel %d times -- marked dead", failed_id, count
                    )
                else:
                    cancel_fail_counts[failed_id] = count
                inst_state["cancel_fail_counts"] = cancel_fail_counts
        elif result.get("action") == "place" and result.get("ok"):
            if order_type == "stop_loss":
                placed_id = (
                    str((result.get("result") or {}).get("stop_id") or "")
                    or str((((result.get("result") or {}).get("parsed") or {}).get("stoplossOrderId")) or "")
                )
                if placed_id and placed_id not in managed_stop_ids:
                    managed_stop_ids.append(placed_id)
                elif not placed_id:
                    logger.warning(
                        "Stop-loss placed OK but no ID extracted for %s — order will be untracked",
                        orderbook_id,
                    )
            else:
                placed_id = (
                    str((result.get("result") or {}).get("order_id") or "")
                    or str((((result.get("result") or {}).get("parsed") or {}).get("orderId")) or "")
                )
                if placed_id and placed_id not in managed_ids:
                    managed_ids.append(placed_id)
                elif not placed_id:
                    logger.warning(
                        "Order placed OK but no ID extracted for %s — order will be untracked",
                        orderbook_id,
                    )
        inst_state["managed_order_ids"] = managed_ids
        inst_state["managed_stop_ids"] = managed_stop_ids
        inst_state["dead_order_ids"] = dead_order_ids
        inst_state["dead_order_timestamps"] = dead_order_timestamps
    return next_state


def summarize_plans(plans: list[dict]) -> str:
    lines: list[str] = []
    for plan in plans:
        lines.append(
            f"{plan['name']} [{plan['mode']}] entry={plan['entry_volume']} held={plan['position_volume']} actions={len(plan['actions'])}"
        )
        for action in plan["actions"]:
            if action["action"] == "cancel":
                if action.get("order_type") == "stop_loss":
                    lines.append(
                        f"  cancel STOP {action['volume']} trig={float(action.get('trigger_price') or 0.0):.2f} sell={action['price']:.2f} (id {action['order_id']})"
                    )
                else:
                    lines.append(
                        f"  cancel {action['side']} {action['volume']} @ {action['price']:.2f} (id {action['order_id']})"
                    )
            else:
                if action.get("order_type") == "stop_loss":
                    lines.append(
                        f"  place STOP {action['volume']} trig={float(action.get('trigger_price') or 0.0):.2f} sell={float(action['price']):.2f} [{action.get('role','')}]"
                    )
                else:
                    lines.append(
                        f"  place {action['side']} {action['volume']} @ {float(action['price']):.2f} [{action.get('role','')}]"
                    )
    return "\n".join(lines) if lines else "No managed Fin Snipe actions."


def run_cycle(
    *,
    hours_remaining: float = DEFAULT_HOURS,
    orderbook_filter: set[str] | None = None,
    budget_sek: float | None = None,
    simulate_flash_window: bool = False,
    live: bool = False,
    state_path: Path = STATE_FILE,
    session_id: str | None = None,
    cycle_index: int = 1,
    manager_log_path: Path = MANAGER_LOG_FILE,
    prediction_log_path: Path = PREDICTION_LOG_FILE,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """Plan and optionally execute one manager cycle."""
    session_id = session_id or _new_session_id()
    state = _default_state()
    snapshots: list[dict] = []
    plans: list[dict] = []
    actions: list[dict] = []
    try:
        if not verify_session():
            if live:
                _notify_critical(
                    "session_expired",
                    "*SNIPE ALERT* Avanza session expired — cannot manage orders. Re-login required.",
                )
            raise RuntimeError("Avanza session invalid or expired.")

        snapshots = build_snapshots(
            hours_remaining,
            orderbook_filter,
            simulate_flash_window=simulate_flash_window,
        )
        state = load_state(state_path)
        budgets = {}
        if budget_sek is not None and orderbook_filter:
            for orderbook_id in orderbook_filter:
                budgets[str(orderbook_id)] = float(budget_sek)
        next_state, plans, actions = plan_cycle(snapshots, state, budgets)
        log_cycle_plan(
            session_id=session_id,
            cycle_index=cycle_index,
            live=live,
            hours_remaining=hours_remaining,
            orderbook_filter=orderbook_filter,
            budget_sek=budget_sek,
            simulate_flash_window=simulate_flash_window,
            state_path=state_path,
            snapshots=snapshots,
            state_before=state,
            plans=plans,
            actions=actions,
            manager_log_path=manager_log_path,
            prediction_log_path=prediction_log_path,
        )
        results = execute_actions(
            actions,
            dry_run=not live,
            on_result=lambda result: log_action_result(
                result,
                session_id=session_id,
                cycle_index=cycle_index,
                live=live,
                manager_log_path=manager_log_path,
            ),
        )
        # Alert on execution failures
        if live and results:
            failed = [r for r in results if not r.get("ok") and not r.get("dry_run")]
            if failed:
                fail_summary = ", ".join(
                    f"{r.get('action')}/{r.get('side','')}" for r in failed[:3]
                )
                _notify_critical(
                    "execution_failure",
                    f"*SNIPE ALERT* {len(failed)}/{len(results)} actions failed: {fail_summary}",
                )

        final_state = apply_execution_results_to_state(next_state, results) if live else next_state
        if live:
            save_state(final_state, state_path)

        # Alert on naked positions (held position with no managed sell or stop)
        if live:
            for plan in plans:
                if plan.get("position_volume", 0) > 0 and plan.get("mode") == "exit":
                    inst_st = (final_state.get("instruments") or {}).get(plan["orderbook_id"]) or {}
                    has_sell = bool(inst_st.get("managed_order_ids"))
                    has_stop = bool(inst_st.get("managed_stop_ids"))
                    if not has_sell and not has_stop:
                        _notify_critical(
                            "naked_position",
                            f"*SNIPE ALERT* {plan.get('name', '?')} has {plan['position_volume']} units with NO sell/stop protection",
                        )
        log_cycle_results(
            session_id=session_id,
            cycle_index=cycle_index,
            live=live,
            state_after=final_state,
            results=results,
            manager_log_path=manager_log_path,
            prediction_log_path=prediction_log_path,
        )
        return final_state, plans, actions, results
    except Exception as exc:
        log_cycle_failure(
            exc,
            session_id=session_id,
            cycle_index=cycle_index,
            live=live,
            hours_remaining=hours_remaining,
            orderbook_filter=orderbook_filter,
            budget_sek=budget_sek,
            simulate_flash_window=simulate_flash_window,
            state_path=state_path,
            manager_log_path=manager_log_path,
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Fin Snipe Manager: maintain live Avanza entry/exit ladders.")
    parser.add_argument("--hours", type=float, default=DEFAULT_HOURS, help="Planning horizon in hours.")
    parser.add_argument("--orderbook", action="append", default=[], help="Optional orderbook id filter.")
    parser.add_argument("--budget", type=float, default=None, help="Explicit SEK budget for the selected orderbook(s).")
    parser.add_argument("--simulate-flash-window", action="store_true", help="Force the flash-crash reserve path on silver.")
    parser.add_argument("--live", action="store_true", help="Execute cancel/place actions. Default is dry-run.")
    parser.add_argument("--loop", action="store_true", help="Run continuously.")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SECONDS, help="Loop interval in seconds.")
    args = parser.parse_args()
    session_id = _new_session_id()
    cycle_index = 0
    lock_handle = acquire_lock_file(
        LOCK_FILE,
        owner=session_id,
        metadata={
            "live": args.live,
            "loop": args.loop,
            "orderbooks": ",".join(sorted(args.orderbook)) if args.orderbook else "",
        },
    )
    if lock_handle is None:
        print(f"Another Fin Snipe Manager instance already holds {LOCK_FILE}")
        return 1

    def _one_cycle() -> list[dict]:
        nonlocal cycle_index
        cycle_index += 1
        _state, plans, _actions, results = run_cycle(
            hours_remaining=args.hours,
            orderbook_filter=set(args.orderbook) or None,
            budget_sek=args.budget,
            simulate_flash_window=args.simulate_flash_window,
            live=args.live,
            session_id=session_id,
            cycle_index=cycle_index,
        )
        print(summarize_plans(plans))
        if results:
            print("")
            for result in results:
                status = "OK" if result.get("ok", False) or result.get("dry_run") else "FAIL"
                if result["action"] == "cancel":
                    if result.get("order_type") == "stop_loss":
                        print(f"{status} cancel STOP {result['order_id']}")
                    else:
                        print(f"{status} cancel {result['order_id']}")
                else:
                    if result.get("order_type") == "stop_loss":
                        print(
                            f"{status} place STOP {result['volume']} trig={float(result.get('trigger_price') or 0.0):.2f} sell={float(result['price']):.2f}"
                        )
                    else:
                        print(f"{status} place {result['side']} {result['volume']} @ {float(result['price']):.2f}")
        print("")
        print(f"log: {MANAGER_LOG_FILE}")
        print(f"predictions: {PREDICTION_LOG_FILE}")
        return plans

    try:
        if not args.loop:
            _one_cycle()
            return 0

        fast_recheck_count = 0
        while True:
            plans = _one_cycle()
            needs_fast_recheck = any(
                event in {"sell_reprice_pending", "stop_reprice_pending"}
                for plan in plans
                for event in (plan.get("events") or [])
            )
            if needs_fast_recheck:
                fast_recheck_count += 1
                if fast_recheck_count > MAX_FAST_RECHECK_CYCLES:
                    logger.warning(
                        "Fast recheck limit reached (%d cycles) — falling back to normal interval",
                        fast_recheck_count,
                    )
                    time.sleep(max(args.interval, 5))
                else:
                    time.sleep(FAST_RECHECK_SECONDS)
            else:
                fast_recheck_count = 0
                time.sleep(max(args.interval, 5))
    finally:
        release_lock_file(lock_handle)


if __name__ == "__main__":
    raise SystemExit(main())
