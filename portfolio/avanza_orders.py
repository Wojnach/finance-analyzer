"""Avanza order confirmation flow — human-in-the-loop for real money.

Workflow:
1. Layer 2 calls request_order() → saves intent to pending orders, returns details
   (including a unique 6-hex `confirm_token`).
2. Layer 2 sends Telegram message with order details + "Reply CONFIRM <token>
   to execute".
3. Main loop calls check_pending_orders() each cycle.
4. On CONFIRM <token> reply → execute the order whose token matches, notify
   via Telegram.
5. On timeout (5 min) → expire the pending order, notify.

P1-10 (2026-05-02): per-order `confirm_token` eliminates three races the
old bare-CONFIRM design suffered from (see test class docstrings):
- stale-CONFIRM race (replayed CONFIRM confirms a NEWER order)
- wrong-order race (sort-by-time-DESC matches the wrong order)
- no-pending-yet race (CONFIRM lands before the order it was for)

Bare CONFIRM (no token) is still accepted but ONLY matches LEGACY orders
that have no `confirm_token` field — i.e. orders that were already in
flight when this code was deployed. New orders MUST be confirmed by token.
"""

import contextlib
import logging
import re
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from portfolio.avanza_control import place_buy_order, place_sell_order
from portfolio.file_utils import atomic_write_json, load_json
from portfolio.http_retry import fetch_with_retry
from portfolio.telegram_notifications import send_telegram

logger = logging.getLogger("portfolio.avanza_orders")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PENDING_FILE = DATA_DIR / "avanza_pending_orders.json"
EXPIRY_MINUTES = 5

# P1-10 (2026-05-02): per-order confirmation nonce. 6 hex chars = 24 bits
# of entropy ≈ ~16M possible tokens. Collision probability across the at-most
# ~5 in-flight pending orders is effectively zero (birthday bound:
# ~5^2/(2*16M) ≈ 7.5e-7). Long enough to survive typos, short enough that
# users will actually type it on a phone keyboard.
_CONFIRM_TOKEN_HEX_CHARS = 6
# Token validation: anything outside [0-9a-f] is silently dropped rather
# than confirmed against an unknown order. This prevents 'CONFIRM xyz' (a
# typo) from accidentally confirming any order via the legacy bare-CONFIRM
# path or matching a token-holding order.
_HEX_TOKEN_RE = re.compile(r"^[0-9a-f]+$")
# CONFIRM prefix matcher. Word boundary required because "confirmed" /
# "confirms" / "confirmation" parse to "confirm" + a hex-valid suffix
# ("ed", "s", "ation") which would silently match against legacy orders
# or non-existent tokens. Anchored at start since the user is asked to
# reply with "CONFIRM <token>" as the entire message.
_CONFIRM_PREFIX_RE = re.compile(r"^confirm(?:\s+|$)")


def _generate_confirm_token() -> str:
    """Return a fresh hex token for a new pending order. Module-level
    indirection keeps tests deterministic via patch.object if ever needed."""
    return secrets.token_hex(_CONFIRM_TOKEN_HEX_CHARS // 2)


def _load_pending() -> list[dict]:
    """Load pending orders from disk."""
    result = load_json(PENDING_FILE, default=[])
    if result is None:
        logger.warning("Failed to read pending orders, returning empty")
        return []
    return result


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
        The pending order dict (includes id, total_sek, expires, and
        ``confirm_token``). The caller MUST include ``confirm_token`` in
        the Telegram notification asking the user to reply
        ``CONFIRM <token>``. Without that, the user sees the prompt but
        has no way to confirm — bare CONFIRM only matches legacy orders
        without a token.
    """
    if action not in ("BUY", "SELL"):
        raise ValueError(f"action must be BUY or SELL, got {action!r}")
    if volume < 1:
        raise ValueError(f"volume must be >= 1, got {volume}")
    if price <= 0:
        raise ValueError(f"price must be > 0, got {price}")

    now = datetime.now(UTC)
    confirm_token = _generate_confirm_token()
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
        "confirm_token": confirm_token,
    }

    pending = _load_pending()
    pending.append(order)
    _save_pending(pending)
    # Log the token at INFO so an operator reading agent.log can read it
    # if they need to confirm out-of-band (e.g. the agent's Telegram message
    # got truncated). The token is per-order, expires in 5 min, and only
    # confirms one specific order — leak surface is minimal.
    # Suppressed false-positive: Token truncated to 4 chars + '****' before logging; line above documents trade-off.
    # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure
    logger.info(
        "Order requested: %s %dx %s @ %.2f SEK (id=%s, confirm_token=%s…)",
        action, volume, instrument_name, price, order["id"],
        confirm_token[:4] + "****" if confirm_token else "N/A",
    )
    return order


def get_pending_orders() -> list[dict]:
    """Get all orders with status 'pending_confirmation'."""
    return [o for o in _load_pending() if o["status"] == "pending_confirmation"]


def check_pending_orders(config: dict) -> list[dict]:
    """Check for Telegram confirmations and expire stale orders.

    Called by the main loop each cycle. Polls Telegram getUpdates for
    CONFIRM <token> replies. Executes confirmed orders (matched by token)
    and expires timed-out ones.

    P1-10 (2026-05-02): a CONFIRM <token> reply confirms ONLY the order
    whose ``confirm_token`` matches. Bare CONFIRM (no token) still works
    but ONLY matches LEGACY orders without a token field — so freshly
    created orders cannot be silently confirmed by a stale CONFIRM that
    was replayed by a getUpdates offset bug.

    Args:
        config: App config dict (with telegram.token, telegram.chat_id,
            and optionally telegram.allowed_user_id for sender auth).

    Returns:
        List of orders that were acted on (confirmed or expired) this cycle.
    """
    pending = _load_pending()
    if not pending:
        return []

    acted_on = []
    now = datetime.now(UTC)

    # Set of tokens that arrived this cycle. Bare CONFIRM is "" (empty
    # string) — only matches legacy orders without a token.
    confirmed_tokens = _check_telegram_confirm(config)

    for order in pending:
        if order["status"] != "pending_confirmation":
            continue

        expires = datetime.fromisoformat(order["expires"])
        order_token = order.get("confirm_token", "")

        if now > expires:
            order["status"] = "expired"
            acted_on.append(order)
            _notify_expired(order, config)
            continue

        # P1-10: matching rules.
        # 1. Order has a token AND that token is in confirmed_tokens → confirm.
        # 2. Order has NO token (legacy in-flight order) AND bare CONFIRM
        #    arrived ("" in the set) → confirm. This is the backwards-compat
        #    path for orders that existed before the deploy.
        # 3. Otherwise → no confirmation this cycle (may still expire).
        confirmed_by_token = bool(order_token) and order_token in confirmed_tokens
        confirmed_legacy = (not order_token) and ("" in confirmed_tokens)

        if confirmed_by_token or confirmed_legacy:
            order["status"] = "confirmed"
            acted_on.append(order)
            if confirmed_by_token:
                confirmed_tokens.discard(order_token)
            else:
                confirmed_tokens.discard("")
            _execute_confirmed_order(order, config)

    _save_pending(pending)
    return acted_on


def _check_telegram_confirm(config: dict) -> set[str]:
    """Poll Telegram for CONFIRM <token> replies from the configured chat.

    Returns ``set[str]`` of matched tokens (lowercase hex). Bare CONFIRM
    (with no token) is represented as ``""`` and matches only LEGACY
    pending orders without a ``confirm_token`` field. Anything that's not
    valid hex after CONFIRM (e.g. ``CONFIRM xyz`` typo) is silently
    dropped — never matched against an order — so a typo doesn't
    accidentally confirm via the legacy path.

    Uses getUpdates with a stored offset to avoid reprocessing old messages.

    AV-P1-3 (2026-05-02): Sender-authenticated when
    ``telegram.allowed_user_id`` is set. Without sender auth, the chat-only
    filter is bypassable in two ways:
      - Group chats: anyone admitted can send CONFIRM and execute the
        pending order.
      - Bot-token compromise: an attacker who has the bot token can
        deliver fake updates with the right ``chat_id`` and execute orders.
    When ``allowed_user_id`` is unset the chat-only check is preserved
    (backwards-compatible). The offset still advances on dropped messages
    so we don't re-process the rejected update every cycle.

    P1-10 (2026-05-02): return type changed from ``bool`` to ``set[str]``
    so each pending order can match its own token. Bare CONFIRM is still
    captured (as ``""``) for the legacy backwards-compat path.
    """
    token = config.get("telegram", {}).get("token", "")
    chat_id = str(config.get("telegram", {}).get("chat_id", ""))
    if not token or not chat_id:
        return set()

    # AV-P1-3 (2026-05-02): optional sender allow-list. Accept either int
    # or string in config — Telegram's `from.id` is always int, so coerce
    # both sides to str for comparison so format mistakes don't accidentally
    # admit/reject a real user.
    raw_allowed_user = config.get("telegram", {}).get("allowed_user_id")
    allowed_user = str(raw_allowed_user) if raw_allowed_user is not None else None

    # Load stored offset (BUG-128: now atomic JSON; handles legacy plain-text format)
    offset_file = DATA_DIR / "avanza_telegram_offset.txt"
    offset = 0
    offset_data = load_json(offset_file)
    if isinstance(offset_data, dict):
        offset = int(offset_data.get("offset", 0))
    elif offset_file.exists():
        with contextlib.suppress(ValueError, OSError):
            offset = int(offset_file.read_text().strip())

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
            return set()
        data = r.json()
        if not data.get("ok"):
            return set()
    except Exception as e:
        logger.warning("Telegram getUpdates failed: %s", e)
        return set()

    found_tokens: set[str] = set()
    for update in data.get("result", []):
        update_id = update.get("update_id", 0)
        # Always advance offset (AV-P1-3: applies to dropped messages too —
        # otherwise a single rejected CONFIRM would replay every cycle).
        if update_id >= offset:
            offset = update_id + 1

        msg = update.get("message", {})
        if str(msg.get("chat", {}).get("id")) != chat_id:
            continue

        # AV-P1-3 (2026-05-02): sender authentication. Fail-closed:
        # missing `from` field with auth enabled drops the message.
        if allowed_user is not None:
            sender = msg.get("from") or {}
            sender_id = sender.get("id")
            if sender_id is None or str(sender_id) != allowed_user:
                logger.warning(
                    "Dropping Telegram message from unauthorized sender id=%r "
                    "(allowed=%s, chat=%s)",
                    sender_id, allowed_user, chat_id,
                )
                continue

        # P1-10 (2026-05-02): parse "CONFIRM <token>" or bare "CONFIRM".
        # Lowercase + collapse whitespace so user-typed variants normalize.
        # Word-boundary match is critical here — without it, "confirmed"
        # parses as "confirm" + "ed" and "ed" IS valid hex (defense vs an
        # accidental "confirmed by my broker" message in the chat).
        text = (msg.get("text") or "").strip().lower()
        m = _CONFIRM_PREFIX_RE.match(text)
        if not m:
            continue
        # Anything after the matched prefix (which includes the word
        # "confirm" + whitespace OR end-of-string) is the candidate.
        rest = text[m.end():].strip()
        if not rest:
            # Bare CONFIRM — legacy backwards-compat path.
            found_tokens.add("")
            continue
        # Take the first whitespace-separated token. Anything trailing is
        # ignored (lets the user paste extra text without breaking the match).
        candidate = rest.split()[0]
        if _HEX_TOKEN_RE.match(candidate):
            found_tokens.add(candidate)
        else:
            logger.warning(
                "Dropping CONFIRM with non-hex token %r (must be lowercase "
                "[0-9a-f] from request_order's confirm_token)",
                candidate,
            )

    # Save offset atomically to prevent corruption on crash (BUG-128)
    try:
        atomic_write_json(offset_file, {"offset": offset})
    except OSError as e:
        logger.warning("Failed to save Telegram offset: %s", e)

    return found_tokens


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

        if result is None:
            order["status"] = "error"
            order["error"] = "API returned no response"
            logger.error("Order API returned None: %s", order["id"])
            send_telegram(
                f"AVANZA {action} ERROR\n"
                f"{order['instrument_name']}: API returned no response",
                config,
            )
            return

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
        except Exception as e:
            logger.warning("Order error notification failed: %s", e)


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
