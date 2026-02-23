"""Avanza order confirmation flow — human-in-the-loop for real money.

Workflow:
1. Layer 2 calls request_order() → saves intent to pending orders, returns details
2. Layer 2 sends Telegram message with order details + "Reply CONFIRM to execute"
3. Main loop calls check_pending_orders() each cycle
4. On CONFIRM reply → execute order via avanza_client, notify via Telegram
5. On timeout (5 min) → expire the pending order, notify
"""

import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from portfolio.avanza_client import place_buy_order, place_sell_order
from portfolio.file_utils import atomic_write_json
from portfolio.http_retry import fetch_with_retry
from portfolio.telegram_notifications import send_telegram

logger = logging.getLogger("portfolio.avanza_orders")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PENDING_FILE = DATA_DIR / "avanza_pending_orders.json"
EXPIRY_MINUTES = 5


def _load_pending() -> list[dict]:
    """Load pending orders from disk."""
    if not PENDING_FILE.exists():
        return []
    try:
        return json.loads(PENDING_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read pending orders, returning empty")
        return []


def _save_pending(orders: list[dict]) -> None:
    """Save pending orders to disk atomically."""
    atomic_write_json(PENDING_FILE, orders)


def request_order(
    action: str,
    orderbook_id: str,
    instrument_name: str,
    config_key: str,
    volume: int,
    price: float,
) -> dict:
    """Create a pending order awaiting Telegram confirmation.

    Args:
        action: "BUY" or "SELL"
        orderbook_id: Avanza orderbook ID
        instrument_name: Human-readable name (e.g. "SAAB B")
        config_key: Config key (e.g. "SAAB-B")
        volume: Number of shares
        price: Limit price in SEK

    Returns:
        The pending order dict (includes id, total_sek, expires)
    """
    if action not in ("BUY", "SELL"):
        raise ValueError(f"action must be BUY or SELL, got {action!r}")
    if volume < 1:
        raise ValueError(f"volume must be >= 1, got {volume}")
    if price <= 0:
        raise ValueError(f"price must be > 0, got {price}")

    now = datetime.now(timezone.utc)
    order = {
        "id": str(uuid.uuid4()),
        "timestamp": now.isoformat(),
        "action": action,
        "orderbook_id": str(orderbook_id),
        "instrument_name": instrument_name,
        "config_key": config_key,
        "volume": volume,
        "price": price,
        "total_sek": round(volume * price, 2),
        "status": "pending_confirmation",
        "expires": (now + timedelta(minutes=EXPIRY_MINUTES)).isoformat(),
    }

    pending = _load_pending()
    pending.append(order)
    _save_pending(pending)
    logger.info("Order requested: %s %dx %s @ %.2f SEK (id=%s)",
                action, volume, instrument_name, price, order["id"])
    return order


def get_pending_orders() -> list[dict]:
    """Get all orders with status 'pending_confirmation'."""
    return [o for o in _load_pending() if o["status"] == "pending_confirmation"]


def check_pending_orders(config: dict) -> list[dict]:
    """Check for Telegram confirmations and expire stale orders.

    Called by the main loop each cycle. Polls Telegram getUpdates for
    CONFIRM replies. Executes confirmed orders and expires timed-out ones.

    Args:
        config: App config dict (with telegram.token, telegram.chat_id)

    Returns:
        List of orders that were acted on (confirmed or expired) this cycle
    """
    pending = _load_pending()
    if not pending:
        return []

    acted_on = []
    now = datetime.now(timezone.utc)

    # Check for CONFIRM replies in Telegram
    confirmed = _check_telegram_confirm(config)

    for order in pending:
        if order["status"] != "pending_confirmation":
            continue

        expires = datetime.fromisoformat(order["expires"])

        if confirmed:
            # Confirm the most recent pending order
            order["status"] = "confirmed"
            acted_on.append(order)
            confirmed = False  # One CONFIRM per order
            _execute_confirmed_order(order, config)

        elif now > expires:
            order["status"] = "expired"
            acted_on.append(order)
            _notify_expired(order, config)

    _save_pending(pending)
    return acted_on


def _check_telegram_confirm(config: dict) -> bool:
    """Poll Telegram for a CONFIRM reply from the configured chat.

    Uses getUpdates with a stored offset to avoid reprocessing old messages.
    """
    token = config.get("telegram", {}).get("token", "")
    chat_id = str(config.get("telegram", {}).get("chat_id", ""))
    if not token or not chat_id:
        return False

    # Load stored offset
    offset_file = DATA_DIR / "avanza_telegram_offset.txt"
    offset = 0
    if offset_file.exists():
        try:
            offset = int(offset_file.read_text().strip())
        except (ValueError, OSError):
            pass

    params = {"timeout": 1, "allowed_updates": ["message"]}
    if offset:
        params["offset"] = offset

    try:
        r = fetch_with_retry(
            f"https://api.telegram.org/bot{token}/getUpdates",
            params=params,
            timeout=5,
        )
        if r is None or not r.ok:
            return False
        data = r.json()
        if not data.get("ok"):
            return False
    except Exception as e:
        logger.warning("Telegram getUpdates failed: %s", e)
        return False

    found_confirm = False
    for update in data.get("result", []):
        update_id = update.get("update_id", 0)
        # Always advance offset
        if update_id >= offset:
            offset = update_id + 1

        msg = update.get("message", {})
        if str(msg.get("chat", {}).get("id")) != chat_id:
            continue

        text = (msg.get("text") or "").strip().upper()
        if text == "CONFIRM":
            found_confirm = True

    # Save offset
    try:
        offset_file.write_text(str(offset))
    except OSError:
        pass

    return found_confirm


def _execute_confirmed_order(order: dict, config: dict) -> None:
    """Execute a confirmed order on Avanza and notify via Telegram."""
    action = order["action"]
    try:
        if action == "BUY":
            result = place_buy_order(
                orderbook_id=order["orderbook_id"],
                price=order["price"],
                volume=order["volume"],
            )
        else:
            result = place_sell_order(
                orderbook_id=order["orderbook_id"],
                price=order["price"],
                volume=order["volume"],
            )

        status = result.get("orderRequestStatus", "UNKNOWN")
        order_id = result.get("orderId", "?")
        msg_text = result.get("message", "")

        if status == "SUCCESS":
            order["status"] = "executed"
            order["avanza_order_id"] = order_id
            msg = (
                f"AVANZA {action} EXECUTED\n"
                f"{order['instrument_name']}: {order['volume']}x @ {order['price']:.2f} SEK\n"
                f"Total: {order['total_sek']:,.0f} SEK\n"
                f"Order ID: {order_id}"
            )
            logger.info("Order executed: %s (avanza_id=%s)", order["id"], order_id)
        else:
            order["status"] = "failed"
            order["error"] = msg_text
            msg = (
                f"AVANZA {action} FAILED\n"
                f"{order['instrument_name']}: {order['volume']}x @ {order['price']:.2f} SEK\n"
                f"Error: {msg_text}"
            )
            logger.error("Order failed: %s — %s", order["id"], msg_text)

        send_telegram(msg, config)

    except Exception as e:
        order["status"] = "error"
        order["error"] = str(e)
        logger.error("Order execution error: %s — %s", order["id"], e)
        try:
            send_telegram(
                f"AVANZA ORDER ERROR\n{order['instrument_name']}: {e}",
                config,
            )
        except Exception:
            pass


def _notify_expired(order: dict, config: dict) -> None:
    """Notify via Telegram that a pending order expired."""
    msg = (
        f"AVANZA ORDER EXPIRED\n"
        f"{order['action']} {order['instrument_name']}: "
        f"{order['volume']}x @ {order['price']:.2f} SEK\n"
        f"No confirmation received within {EXPIRY_MINUTES} min."
    )
    logger.info("Order expired: %s", order["id"])
    try:
        send_telegram(msg, config)
    except Exception as e:
        logger.warning("Failed to send expiry notification: %s", e)
