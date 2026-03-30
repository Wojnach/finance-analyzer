"""Trading operations — orders, stop-losses, deals.

Typed wrappers around :class:`~portfolio.avanza.client.AvanzaClient`
for placing, modifying, and cancelling orders and stop-losses.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from avanza.constants import (
    Condition,
    OrderType,
    StopLossPriceType,
    StopLossTriggerType,
)
from avanza.entities import StopLossOrderEvent, StopLossTrigger

from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.types import (
    Deal,
    Order,
    OrderResult,
    StopLoss,
    StopLossResult,
)

logger = logging.getLogger("portfolio.avanza.trading")


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


def place_order(
    side: str,
    ob_id: str,
    price: float,
    volume: int,
    condition: str = "NORMAL",
    valid_until: str | None = None,
    account_id: str | None = None,
) -> OrderResult:
    """Place a BUY or SELL order.

    Args:
        side: ``"BUY"`` or ``"SELL"``.
        ob_id: Avanza orderbook ID.
        price: Limit price.
        volume: Number of units.
        condition: Order condition (``"NORMAL"``, ``"FILL_OR_KILL"``,
            ``"FILL_AND_KILL"``).
        valid_until: ISO date string (default: today).
        account_id: Override default account.

    Returns:
        :class:`~portfolio.avanza.types.OrderResult`.

    Raises:
        ValueError: If volume < 1 or price <= 0.
    """
    if volume < 1:
        raise ValueError(f"volume must be >= 1, got {volume}")
    if price <= 0:
        raise ValueError(f"price must be > 0, got {price}")

    client = AvanzaClient.get_instance()
    acct = account_id or client.account_id
    valid = date.fromisoformat(valid_until) if valid_until else date.today()

    raw: dict[str, Any] = client.avanza.place_order(
        acct,
        ob_id,
        OrderType(side),
        price,
        valid,
        volume,
        condition=Condition(condition),
    )

    logger.info(
        "place_order side=%s ob_id=%s price=%s vol=%d -> %s",
        side,
        ob_id,
        price,
        volume,
        raw.get("orderRequestStatus"),
    )
    return OrderResult.from_api(raw)


def modify_order(
    order_id: str,
    ob_id: str,
    price: float,
    volume: int,
    condition: str = "NORMAL",
    valid_until: str | None = None,
    account_id: str | None = None,
) -> OrderResult:
    """Modify an existing order.

    Args:
        order_id: Existing order ID to modify.
        ob_id: Avanza orderbook ID (unused by API but kept for consistency).
        price: New limit price.
        volume: New volume.
        condition: Order condition (unused by edit_order API).
        valid_until: ISO date string (default: today).
        account_id: Override default account.

    Returns:
        :class:`~portfolio.avanza.types.OrderResult`.
    """
    client = AvanzaClient.get_instance()
    acct = account_id or client.account_id
    valid = date.fromisoformat(valid_until) if valid_until else date.today()

    raw: dict[str, Any] = client.avanza.edit_order(
        order_id,
        acct,
        price,
        valid,
        volume,
    )

    logger.info(
        "modify_order order_id=%s price=%s vol=%d -> %s",
        order_id,
        price,
        volume,
        raw.get("orderRequestStatus"),
    )
    return OrderResult.from_api(raw)


def cancel_order(
    order_id: str,
    account_id: str | None = None,
) -> bool:
    """Cancel an existing order.

    Returns:
        ``True`` if the cancellation was accepted.
    """
    client = AvanzaClient.get_instance()
    acct = account_id or client.account_id
    raw: dict[str, Any] = client.avanza.delete_order(acct, order_id)
    status = str(raw.get("orderRequestStatus", "")).upper()
    success = status == "SUCCESS"
    logger.info("cancel_order order_id=%s -> %s", order_id, status)
    return success


def get_orders() -> list[Order]:
    """Fetch all open/recent orders.

    Returns:
        List of :class:`~portfolio.avanza.types.Order`.
    """
    client = AvanzaClient.get_instance()
    raw: Any = client.get_orders_raw()

    orders_list: list[dict[str, Any]]
    if isinstance(raw, dict):
        orders_list = raw.get("orders", [])
    elif isinstance(raw, list):
        orders_list = raw
    else:
        orders_list = []

    return [Order.from_api(o) for o in orders_list]


def get_deals() -> list[Deal]:
    """Fetch recent deals (executions).

    Returns:
        List of :class:`~portfolio.avanza.types.Deal`.
    """
    client = AvanzaClient.get_instance()
    raw: Any = client.get_deals_raw()

    deals_list: list[dict[str, Any]]
    if isinstance(raw, dict):
        deals_list = raw.get("deals", [])
    elif isinstance(raw, list):
        deals_list = raw
    else:
        deals_list = []

    return [Deal.from_api(d) for d in deals_list]


# ---------------------------------------------------------------------------
# Stop-losses
# ---------------------------------------------------------------------------


def place_stop_loss(
    ob_id: str,
    trigger_price: float,
    sell_price: float,
    volume: int,
    valid_days: int = 8,
    trigger_type: str = "LESS_OR_EQUAL",
    value_type: str = "MONETARY",
    account_id: str | None = None,
) -> StopLossResult:
    """Place a stop-loss order.

    Args:
        ob_id: Avanza orderbook ID.
        trigger_price: Price that triggers the stop.
        sell_price: Limit price for the sell order when triggered.
        volume: Number of units to sell.
        valid_days: Days until the stop-loss expires (default 8).
        trigger_type: Trigger direction (``"LESS_OR_EQUAL"``,
            ``"FOLLOW_DOWNWARDS"``, etc.).
        value_type: Price type (``"MONETARY"`` or ``"PERCENTAGE"``).
        account_id: Override default account.

    Returns:
        :class:`~portfolio.avanza.types.StopLossResult`.
    """
    client = AvanzaClient.get_instance()
    acct = account_id or client.account_id
    valid_until = date.today() + timedelta(days=valid_days)

    trigger = StopLossTrigger(
        type=StopLossTriggerType(trigger_type),
        value=trigger_price,
        valid_until=valid_until,
        value_type=StopLossPriceType(value_type),
    )

    order_event = StopLossOrderEvent(
        type=OrderType.SELL,
        price=sell_price,
        volume=volume,
        valid_days=valid_days,
        price_type=StopLossPriceType(value_type),
        short_selling_allowed=False,
    )

    raw: dict[str, Any] = client.avanza.place_stop_loss_order(
        "0",  # parent_stop_loss_id — "0" for new stop-loss
        acct,
        ob_id,
        trigger,
        order_event,
    )

    logger.info(
        "place_stop_loss ob_id=%s trigger=%.4f sell=%.4f vol=%d -> %s",
        ob_id,
        trigger_price,
        sell_price,
        volume,
        raw.get("status", raw.get("orderRequestStatus")),
    )
    return StopLossResult.from_api(raw)


def place_trailing_stop(
    ob_id: str,
    trail_percent: float,
    volume: int,
    valid_days: int = 8,
    account_id: str | None = None,
) -> StopLossResult:
    """Place a trailing stop-loss (follows price downwards by percentage).

    Args:
        ob_id: Avanza orderbook ID.
        trail_percent: Trailing distance as percentage (e.g. ``5.0`` for 5%).
        volume: Number of units to sell.
        valid_days: Days until the stop-loss expires.
        account_id: Override default account.

    Returns:
        :class:`~portfolio.avanza.types.StopLossResult`.
    """
    return place_stop_loss(
        ob_id=ob_id,
        trigger_price=trail_percent,
        sell_price=0.0,  # Not applicable for trailing stops
        volume=volume,
        valid_days=valid_days,
        trigger_type="FOLLOW_DOWNWARDS",
        value_type="PERCENTAGE",
        account_id=account_id,
    )


def get_stop_losses() -> list[StopLoss]:
    """Fetch all active stop-losses.

    Returns:
        List of :class:`~portfolio.avanza.types.StopLoss`.
    """
    client = AvanzaClient.get_instance()
    raw: Any = client.get_all_stop_losses_raw()

    sl_list: list[dict[str, Any]]
    if isinstance(raw, dict):
        sl_list = raw.get("stopLosses", raw.get("stopLossOrders", []))
    elif isinstance(raw, list):
        sl_list = raw
    else:
        sl_list = []

    return [StopLoss.from_api(sl) for sl in sl_list]


def delete_stop_loss(
    stop_id: str,
    account_id: str | None = None,
) -> bool:
    """Delete a stop-loss order.  Idempotent — 404 is treated as success.

    Returns:
        ``True`` if the deletion succeeded (or the stop-loss was already gone).
    """
    client = AvanzaClient.get_instance()
    acct = account_id or client.account_id
    try:
        client.avanza.delete_stop_loss_order(acct, stop_id)
        logger.info("delete_stop_loss stop_id=%s -> OK", stop_id)
        return True
    except Exception as exc:
        # 404 means already deleted — treat as success
        exc_str = str(exc).lower()
        if "404" in exc_str or "not found" in exc_str:
            logger.info("delete_stop_loss stop_id=%s -> already gone (404)", stop_id)
            return True
        logger.error("delete_stop_loss stop_id=%s -> FAILED: %s", stop_id, exc)
        return False
