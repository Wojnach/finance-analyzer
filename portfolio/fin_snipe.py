"""Fin Snipe: intraday metals bid/exit ladder reporter.

This is the named entry point for the Avanza metals ladder workflow so the
feature is recognizable in the codebase and can later grow into a fuller
intraday snipe/scalp manager.
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

from portfolio.avanza_session import api_get, verify_session
from portfolio.file_utils import load_json
from portfolio.metals_ladder import build_intraday_ladder, map_underlying_name

BASE_DIR = Path(__file__).resolve().parent.parent
SUMMARY_PATH = BASE_DIR / "data" / "agent_summary.json"
SILVER_ANALYSIS_PATH = BASE_DIR / "data" / "silver_analysis.json"

logger = logging.getLogger("portfolio.fin_snipe")


def _value(value):
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _load_json(path: Path) -> dict:
    """Load a JSON file safely via file_utils (atomic-read, TOCTOU-safe)."""
    return load_json(path) or {}


def fetch_open_orders() -> list[dict]:
    """Return all open Avanza orders."""
    payload = api_get("/_api/trading/rest/orders")
    if isinstance(payload, dict):
        payload = payload.get("orders", [])
    return list(payload or [])


def fetch_stop_losses() -> list[dict]:
    """Return all Avanza stop-loss orders for the current session."""
    try:
        payload = api_get("/_api/trading/stoploss")
    except Exception:
        logger.warning("Failed to fetch Avanza stop-loss orders", exc_info=True)
        return []
    if isinstance(payload, dict):
        payload = payload.get("orders", [])
    return list(payload or [])


def fetch_positions_by_orderbook() -> dict[str, dict]:
    """Return current positions keyed by Avanza orderbook id."""
    payload = api_get("/_api/position-data/positions")
    positions: dict[str, dict] = {}
    for item in (payload.get("withOrderbook") or []):
        orderbook = ((item.get("instrument") or {}).get("orderbook") or {})
        orderbook_id = str(orderbook.get("id") or "")
        if not orderbook_id:
            continue
        positions[orderbook_id] = {
            "name": (item.get("instrument") or {}).get("name") or orderbook.get("name") or "",
            "instrument_type": ((item.get("instrument") or {}).get("type") or orderbook.get("type") or ""),
            "account_id": str((item.get("account") or {}).get("id") or ""),
            "volume": int(_value(item.get("volume")) or 0),
            "average_price": float(_value(item.get("averageAcquiredPrice")) or 0.0),
            "value_sek": float(_value(item.get("value")) or 0.0),
        }
    return positions


def _fetch_market_guide(orderbook_id: str, instrument_type: str) -> dict:
    inst_type = "warrant" if instrument_type.lower() == "warrant" else "certificate"
    return api_get(f"/_api/market-guide/{inst_type}/{orderbook_id}")


def _analysis_for_ticker(ticker: str, simulate_flash_window: bool = False) -> dict:
    if ticker == "XAG-USD":
        analysis = _load_json(SILVER_ANALYSIS_PATH)
        if simulate_flash_window:
            analysis = copy.deepcopy(analysis)
            analysis.setdefault("us_market_open", {})["phase"] = "pre_open"
        return analysis
    return {}


def _format_line(label: str, underlying: float, price: float) -> str:
    return f"  {label}: underlying {underlying:.4f} -> cert {price:.2f}"


def build_snapshots(
    hours_remaining: float,
    orderbook_filter: set[str] | None = None,
    *,
    simulate_flash_window: bool = False,
) -> list[dict]:
    summary = _load_json(SUMMARY_PATH)
    signals = summary.get("signals") or {}
    focus_probabilities = summary.get("focus_probabilities") or {}
    snapshots: list[dict] = []
    buy_orders = fetch_open_orders()
    stop_losses = fetch_stop_losses()
    positions_by_orderbook = fetch_positions_by_orderbook()
    grouped_orders: dict[str, list[dict]] = {}
    grouped_stop_losses: dict[str, list[dict]] = {}
    for order in buy_orders:
        orderbook_id = str(order.get("orderbookId") or "")
        if orderbook_filter and orderbook_id not in orderbook_filter:
            continue
        grouped_orders.setdefault(orderbook_id, []).append(order)
    for stop_order in stop_losses:
        orderbook_id = str(((stop_order.get("orderbook") or {}).get("id")) or "")
        if not orderbook_id:
            continue
        if orderbook_filter and orderbook_id not in orderbook_filter:
            continue
        grouped_stop_losses.setdefault(orderbook_id, []).append(stop_order)

    managed_orderbooks = set(grouped_orders) | set(grouped_stop_losses) | set(positions_by_orderbook)
    if orderbook_filter:
        managed_orderbooks &= set(orderbook_filter)

    for orderbook_id in sorted(managed_orderbooks):
        active_orders = list(grouped_orders.get(orderbook_id) or [])
        position = positions_by_orderbook.get(orderbook_id) or {}
        exemplar = active_orders[0] if active_orders else {}
        exemplar_book = exemplar.get("orderbook") or {}
        name = (
            exemplar_book.get("name")
            or position.get("name")
            or f"orderbook {orderbook_id}"
        )
        account_id = str((exemplar.get("account") or {}).get("accountId") or position.get("account_id") or "")
        inst_type = exemplar_book.get("instrumentType") or position.get("instrument_type") or "Warrant"
        market = _fetch_market_guide(orderbook_id, inst_type)
        underlying_name = ((market.get("underlying") or {}).get("name") or "").strip()
        ticker = map_underlying_name(underlying_name)
        if not ticker or ticker not in signals:
            if ticker:
                logger.debug(
                    "Skipping orderbook %s (%s): ticker %s not in agent_summary signals",
                    orderbook_id, name, ticker,
                )
            continue

        quote = market.get("quote") or {}
        underlying = (market.get("underlying") or {}).get("quote") or {}
        indicators = market.get("keyIndicators") or {}
        current_price = float(_value(quote.get("sell")) or _value(quote.get("last")) or 0.0)
        current_underlying = float(_value(underlying.get("last")) or 0.0)
        leverage = float(_value(indicators.get("leverage")) or 1.0)
        if current_price <= 0 or current_underlying <= 0 or leverage <= 0:
            continue

        ladder = build_intraday_ladder(
            signals[ticker],
            focus_probabilities.get(ticker),
            ticker=ticker,
            current_instrument_price=current_price,
            current_underlying_price=current_underlying,
            leverage=leverage,
            hours_remaining=hours_remaining,
            analysis=_analysis_for_ticker(
                ticker,
                simulate_flash_window=simulate_flash_window,
            ),
        )

        snapshots.append({
            "orderbook_id": orderbook_id,
            "name": name,
            "instrument_type": inst_type,
            "account_id": account_id,
            "ticker": ticker,
            "signal_entry": signals.get(ticker) or {},
            "focus_probability": focus_probabilities.get(ticker) or {},
            "market": market,
            "quote": quote,
            "current_bid": float(_value(quote.get("buy")) or 0.0),
            "current_ask": float(_value(quote.get("sell")) or 0.0),
            "current_last": float(_value(quote.get("last")) or 0.0),
            "current_underlying": current_underlying,
            "current_instrument_price": current_price,
            "leverage": leverage,
            "open_orders": active_orders,
            "stop_orders": list(grouped_stop_losses.get(orderbook_id) or []),
            "position_volume": int(position.get("volume") or 0),
            "position_average_price": float(position.get("average_price") or 0.0),
            "position_value_sek": float(position.get("value_sek") or 0.0),
            "ladder": ladder,
        })

    return snapshots


def build_reports(
    hours_remaining: float,
    orderbook_filter: set[str] | None = None,
    *,
    simulate_flash_window: bool = False,
) -> list[str]:
    reports: list[str] = []
    snapshots = build_snapshots(
        hours_remaining,
        orderbook_filter,
        simulate_flash_window=simulate_flash_window,
    )
    for snapshot in snapshots:
        active_orders = list(snapshot["open_orders"])
        if not any(str(order.get("side") or "").upper() == "BUY" for order in active_orders):
            continue
        name = snapshot["name"]
        orderbook_id = snapshot["orderbook_id"]
        ticker = snapshot["ticker"]
        quote = snapshot["quote"]
        ladder = snapshot["ladder"]
        active_orders.sort(key=lambda item: float(item.get("price") or 0.0), reverse=True)

        lines = [
            f"{name} ({orderbook_id})",
            f"  live: bid {_value(quote.get('buy'))} / ask {_value(quote.get('sell'))} / last {_value(quote.get('last'))}",
            f"  working bid: {ladder['working_price']:.2f}  | mean entry: {ladder['mean_price']:.2f}",
        ]
        if ladder["flash_price"] > 0:
            lines.append(
                f"  flash reserve: {ladder['flash_price']:.2f}  | extra drop {ladder['flash_crash_drop_pct']:.2f}%"
            )
        elif ticker == "XAG-USD":
            lines.append("  flash reserve: disabled outside the US-open window")
        lines.append(f"  exit target: {ladder['exit_price']:.2f}  | stretch: {ladder['stretch_exit_price']:.2f}")
        lines.append(_format_line("working", ladder["working_underlying"], ladder["working_price"]))
        if ladder["flash_price"] > 0:
            lines.append(_format_line("flash", ladder["flash_underlying"], ladder["flash_price"]))
        lines.append(_format_line("exit", ladder["exit_underlying"], ladder["exit_price"]))
        for existing in active_orders:
            lines.append(
                f"  open BUY: id {existing.get('orderId')}  {existing.get('volume')} @ {float(existing.get('price') or 0.0):.2f}"
            )
        reports.append("\n".join(lines))

    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Fin Snipe: report current metals bid/exit ladders for Avanza BUY orders.")
    parser.add_argument("--hours", type=float, default=6.0, help="Planning horizon in hours (default: 6).")
    parser.add_argument("--orderbook", action="append", default=[], help="Optional orderbook id filter.")
    parser.add_argument(
        "--simulate-flash-window",
        action="store_true",
        help="Pretend silver is in the US-open flash window and include the deeper reserve bid.",
    )
    args = parser.parse_args()

    if not verify_session():
        raise SystemExit("Avanza session invalid or expired. Run scripts/avanza_login.py first.")

    reports = build_reports(
        args.hours,
        set(args.orderbook) or None,
        simulate_flash_window=args.simulate_flash_window,
    )
    if not reports:
        print("No supported metals BUY orders found.")
        return 0

    print("\n\n".join(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
