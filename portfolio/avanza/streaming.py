"""CometD/Bayeux WebSocket streaming client for Avanza push data.

Connects to ``wss://www.avanza.se/_push/cometd`` and subscribes to
real-time channels for quotes, order depths, trades, orders, and deals.
Runs a background daemon thread with automatic reconnection.

Usage::

    stream = AvanzaStream(push_subscription_id="abc123")
    stream.on_quote("856394", lambda msg: print(msg))
    stream.start()
    # ... later ...
    stream.stop()
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import websocket

logger = logging.getLogger("portfolio.avanza.streaming")

WS_URL = "wss://www.avanza.se/_push/cometd"

# Reconnect backoff
_MIN_BACKOFF = 1.0
_MAX_BACKOFF = 60.0
_BACKOFF_FACTOR = 2.0

# CometD heartbeat interval (seconds)
_HEARTBEAT_INTERVAL = 30.0


class AvanzaStream:
    """CometD/Bayeux WebSocket client for Avanza push data.

    Register callbacks with :meth:`on_quote`, :meth:`on_order_depth`, etc.
    before calling :meth:`start`.  The read loop runs in a daemon thread
    and dispatches messages to registered callbacks by channel.
    """

    def __init__(self, push_subscription_id: str) -> None:
        self._push_sub_id = push_subscription_id
        self._callbacks: dict[str, list[Callable[[dict], None]]] = {}
        self._client_id: str | None = None
        self._ws: websocket.WebSocket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._backoff = _MIN_BACKOFF

    # ------------------------------------------------------------------
    # Public registration (before start)
    # ------------------------------------------------------------------

    def on_quote(self, ob_id: str, callback: Callable[[dict], None]) -> None:
        """Register a callback for quote updates on *ob_id*."""
        channel = f"/quotes/{ob_id}"
        self._callbacks.setdefault(channel, []).append(callback)

    def on_order_depth(self, ob_id: str, callback: Callable[[dict], None]) -> None:
        """Register a callback for order depth updates on *ob_id*."""
        channel = f"/orderdepths/{ob_id}"
        self._callbacks.setdefault(channel, []).append(callback)

    def on_trades(self, ob_id: str, callback: Callable[[dict], None]) -> None:
        """Register a callback for trade updates on *ob_id*."""
        channel = f"/trades/{ob_id}"
        self._callbacks.setdefault(channel, []).append(callback)

    def on_orders(self, account_ids: list[str], callback: Callable[[dict], None]) -> None:
        """Register a callback for order updates on the given accounts."""
        channel = "/orders/_" + ",".join(account_ids)
        self._callbacks.setdefault(channel, []).append(callback)

    def on_deals(self, account_ids: list[str], callback: Callable[[dict], None]) -> None:
        """Register a callback for deal updates on the given accounts."""
        channel = "/deals/_" + ",".join(account_ids)
        self._callbacks.setdefault(channel, []).append(callback)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("AvanzaStream already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="avanza-stream")
        self._thread.start()
        logger.info("AvanzaStream started (subscriptions=%d)", len(self._callbacks))

    def stop(self) -> None:
        """Close the WebSocket and join the background thread."""
        self._stop_event.set()
        if self._ws is not None:
            with contextlib.suppress(Exception):
                self._ws.close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._client_id = None
        logger.info("AvanzaStream stopped")

    # ------------------------------------------------------------------
    # Internal: run loop with reconnection
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Connect, handshake, subscribe, and read — with reconnection."""
        while not self._stop_event.is_set():
            try:
                self._connect()
                self._do_handshake()
                for channel in self._callbacks:
                    self._subscribe_channel(channel)
                self._backoff = _MIN_BACKOFF  # Reset on successful connect
                self._read_loop()
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                logger.warning(
                    "AvanzaStream connection error: %s — reconnecting in %.0fs",
                    exc,
                    self._backoff,
                )
                self._stop_event.wait(self._backoff)
                self._backoff = min(self._backoff * _BACKOFF_FACTOR, _MAX_BACKOFF)
            finally:
                if self._ws is not None:
                    with contextlib.suppress(Exception):
                        self._ws.close()
                    self._ws = None

    def _connect(self) -> None:
        """Open WebSocket connection to Avanza push endpoint."""
        self._ws = websocket.create_connection(
            WS_URL,
            timeout=_HEARTBEAT_INTERVAL + 10,
        )
        logger.debug("WebSocket connected to %s", WS_URL)

    def _do_handshake(self) -> None:
        """Perform CometD/Bayeux handshake and extract clientId."""
        handshake_msg = [{
            "channel": "/meta/handshake",
            "ext": {"subscriptionId": self._push_sub_id},
            "version": "1.0",
            "supportedConnectionTypes": ["websocket"],
        }]
        self._ws.send(json.dumps(handshake_msg))  # type: ignore[union-attr]
        response = self._ws.recv()  # type: ignore[union-attr]
        msgs = json.loads(response)

        if not isinstance(msgs, list) or len(msgs) == 0:
            raise RuntimeError(f"Invalid handshake response: {response}")

        handshake_resp = msgs[0]
        if not handshake_resp.get("successful", False):
            raise RuntimeError(f"Handshake failed: {handshake_resp}")

        self._client_id = handshake_resp["clientId"]
        logger.debug("Handshake successful, clientId=%s", self._client_id)

        # Send initial connect message
        connect_msg = [{
            "channel": "/meta/connect",
            "clientId": self._client_id,
            "connectionType": "websocket",
        }]
        self._ws.send(json.dumps(connect_msg))  # type: ignore[union-attr]

    def _subscribe_channel(self, channel: str) -> None:
        """Subscribe to a single CometD channel."""
        sub_msg = [{
            "channel": "/meta/subscribe",
            "subscription": channel,
            "clientId": self._client_id,
        }]
        self._ws.send(json.dumps(sub_msg))  # type: ignore[union-attr]
        logger.debug("Subscribed to %s", channel)

    def _read_loop(self) -> None:
        """Read messages from WebSocket, dispatch, and send heartbeats."""
        last_heartbeat = time.monotonic()

        while not self._stop_event.is_set():
            # Send heartbeat if needed
            now = time.monotonic()
            if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
                heartbeat_msg = [{
                    "channel": "/meta/connect",
                    "clientId": self._client_id,
                    "connectionType": "websocket",
                }]
                self._ws.send(json.dumps(heartbeat_msg))  # type: ignore[union-attr]
                last_heartbeat = now

            try:
                raw = self._ws.recv()  # type: ignore[union-attr]
            except websocket.WebSocketTimeoutException:
                continue  # Timeout is expected, just loop and heartbeat
            except websocket.WebSocketConnectionClosedException:
                logger.info("WebSocket connection closed")
                return  # Will reconnect in _run_loop

            if not raw:
                continue

            try:
                msgs = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Failed to parse WebSocket message: %s", raw[:200])
                continue

            if not isinstance(msgs, list):
                msgs = [msgs]

            for msg in msgs:
                self._dispatch_message(msg)

    def _dispatch_message(self, msg: dict[str, Any]) -> None:
        """Route a CometD message to registered callbacks by channel."""
        channel = msg.get("channel", "")

        # Ignore meta channels (handshake, connect, subscribe responses)
        if channel.startswith("/meta/"):
            return

        callbacks = self._callbacks.get(channel, [])
        data = msg.get("data", msg)

        for cb in callbacks:
            try:
                cb(data)
            except Exception as exc:
                logger.error(
                    "Callback error on channel %s: %s",
                    channel,
                    exc,
                    exc_info=True,
                )
