"""Tests for portfolio.avanza.streaming — CometD WebSocket client."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from portfolio.avanza.streaming import AvanzaStream

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def stream() -> AvanzaStream:
    """Create a fresh AvanzaStream instance for each test."""
    return AvanzaStream(push_subscription_id="test-push-sub-123")


def _make_mock_ws(handshake_response: dict | None = None) -> MagicMock:
    """Create a mock WebSocket with a working handshake response."""
    ws = MagicMock()
    if handshake_response is None:
        handshake_response = {
            "channel": "/meta/handshake",
            "successful": True,
            "clientId": "client-abc-456",
        }
    ws.recv.return_value = json.dumps([handshake_response])
    return ws


# ---------------------------------------------------------------------------
# Handshake
# ---------------------------------------------------------------------------

class TestHandshake:
    def test_handshake_sends_correct_message(self, stream: AvanzaStream):
        """Handshake sends the right CometD message with subscriptionId."""
        ws = _make_mock_ws()
        stream._ws = ws

        stream._do_handshake()

        # First call is the handshake message
        handshake_call = ws.send.call_args_list[0]
        sent = json.loads(handshake_call[0][0])
        assert isinstance(sent, list)
        assert len(sent) == 1
        assert sent[0]["channel"] == "/meta/handshake"
        assert sent[0]["ext"]["subscriptionId"] == "test-push-sub-123"
        assert sent[0]["version"] == "1.0"
        assert sent[0]["supportedConnectionTypes"] == ["websocket"]

    def test_handshake_extracts_client_id(self, stream: AvanzaStream):
        """Handshake correctly extracts clientId from response."""
        ws = _make_mock_ws()
        stream._ws = ws

        stream._do_handshake()

        assert stream._client_id == "client-abc-456"

    def test_handshake_sends_connect_after(self, stream: AvanzaStream):
        """After handshake, a /meta/connect message is sent."""
        ws = _make_mock_ws()
        stream._ws = ws

        stream._do_handshake()

        # Second call should be the connect message
        assert ws.send.call_count == 2
        connect_call = ws.send.call_args_list[1]
        sent = json.loads(connect_call[0][0])
        assert sent[0]["channel"] == "/meta/connect"
        assert sent[0]["clientId"] == "client-abc-456"
        assert sent[0]["connectionType"] == "websocket"

    def test_handshake_failure_raises(self, stream: AvanzaStream):
        """Handshake raises RuntimeError when server returns unsuccessful."""
        ws = _make_mock_ws({
            "channel": "/meta/handshake",
            "successful": False,
            "error": "authentication failed",
        })
        stream._ws = ws

        with pytest.raises(RuntimeError, match="Handshake failed"):
            stream._do_handshake()

    def test_handshake_invalid_response_raises(self, stream: AvanzaStream):
        """Handshake raises RuntimeError on empty/invalid response."""
        ws = MagicMock()
        ws.recv.return_value = json.dumps([])
        stream._ws = ws

        with pytest.raises(RuntimeError, match="Invalid handshake response"):
            stream._do_handshake()


# ---------------------------------------------------------------------------
# Subscribe
# ---------------------------------------------------------------------------

class TestSubscribe:
    def test_subscribe_sends_correct_channel(self, stream: AvanzaStream):
        """Subscribe sends the right CometD message with channel and clientId."""
        ws = MagicMock()
        stream._ws = ws
        stream._client_id = "client-abc-456"

        stream._subscribe_channel("/quotes/856394")

        sent = json.loads(ws.send.call_args[0][0])
        assert isinstance(sent, list)
        assert len(sent) == 1
        assert sent[0]["channel"] == "/meta/subscribe"
        assert sent[0]["subscription"] == "/quotes/856394"
        assert sent[0]["clientId"] == "client-abc-456"


# ---------------------------------------------------------------------------
# Callback dispatch
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_callback_dispatched_on_matching_channel(self, stream: AvanzaStream):
        """Callback is called when a message arrives on a subscribed channel."""
        received = []
        stream.on_quote("856394", lambda msg: received.append(msg))

        stream._dispatch_message({
            "channel": "/quotes/856394",
            "data": {"bid": 123.45, "ask": 123.50},
        })

        assert len(received) == 1
        assert received[0] == {"bid": 123.45, "ask": 123.50}

    def test_no_callback_for_unsubscribed_channel(self, stream: AvanzaStream):
        """No callback fires for a channel without registered callbacks."""
        received = []
        stream.on_quote("856394", lambda msg: received.append(msg))

        # Different orderbook ID — should not trigger callback
        stream._dispatch_message({
            "channel": "/quotes/999999",
            "data": {"bid": 10.0, "ask": 11.0},
        })

        assert len(received) == 0

    def test_meta_channels_ignored(self, stream: AvanzaStream):
        """Meta channels (/meta/*) are silently ignored."""
        received = []
        stream._callbacks["/meta/connect"] = [lambda msg: received.append(msg)]

        stream._dispatch_message({
            "channel": "/meta/connect",
            "successful": True,
        })

        # Even though we (abnormally) registered a callback on /meta/,
        # the dispatch method ignores meta channels.
        assert len(received) == 0

    def test_multiple_callbacks_on_same_channel(self, stream: AvanzaStream):
        """Multiple callbacks on the same channel all fire."""
        r1, r2 = [], []
        stream.on_quote("856394", lambda msg: r1.append(msg))
        stream.on_quote("856394", lambda msg: r2.append(msg))

        stream._dispatch_message({
            "channel": "/quotes/856394",
            "data": {"last": 42.0},
        })

        assert len(r1) == 1
        assert len(r2) == 1

    def test_callback_error_does_not_crash(self, stream: AvanzaStream):
        """If a callback raises, other callbacks still fire and no crash."""
        r1 = []

        def bad_callback(msg):
            raise ValueError("boom")

        stream.on_quote("856394", bad_callback)
        stream.on_quote("856394", lambda msg: r1.append(msg))

        # Should not raise
        stream._dispatch_message({
            "channel": "/quotes/856394",
            "data": {"last": 99.0},
        })

        assert len(r1) == 1


# ---------------------------------------------------------------------------
# Channel registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_on_quote_registers_correct_channel(self, stream: AvanzaStream):
        """on_quote registers under /quotes/{obId}."""
        stream.on_quote("856394", lambda m: None)
        assert "/quotes/856394" in stream._callbacks

    def test_on_order_depth_registers_correct_channel(self, stream: AvanzaStream):
        """on_order_depth registers under /orderdepths/{obId}."""
        stream.on_order_depth("856394", lambda m: None)
        assert "/orderdepths/856394" in stream._callbacks

    def test_on_trades_registers_correct_channel(self, stream: AvanzaStream):
        """on_trades registers under /trades/{obId}."""
        stream.on_trades("856394", lambda m: None)
        assert "/trades/856394" in stream._callbacks

    def test_on_orders_registers_correct_channel(self, stream: AvanzaStream):
        """on_orders registers under /orders/_{accountIds}."""
        stream.on_orders(["1625505", "9999999"], lambda m: None)
        assert "/orders/_1625505,9999999" in stream._callbacks

    def test_on_deals_registers_correct_channel(self, stream: AvanzaStream):
        """on_deals registers under /deals/_{accountIds}."""
        stream.on_deals(["1625505"], lambda m: None)
        assert "/deals/_1625505" in stream._callbacks


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    @patch("portfolio.avanza.streaming.websocket.create_connection")
    def test_stop_sets_event_and_closes(self, mock_create, stream: AvanzaStream):
        """stop() sets the stop event and closes the WebSocket."""
        ws = MagicMock()
        stream._ws = ws
        stream._stop_event.clear()

        stream.stop()

        assert stream._stop_event.is_set()
        ws.close.assert_called_once()

    def test_start_creates_daemon_thread(self, stream: AvanzaStream):
        """start() creates a daemon thread."""
        # We patch _run_loop to prevent actual connection
        with patch.object(stream, "_run_loop"):
            stream.start()
            assert stream._thread is not None
            assert stream._thread.daemon is True
            stream.stop()
