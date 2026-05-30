"""Tests for Avanza order confirmation flow."""

import json
import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

# Pre-mock avanza_client trading functions so avanza_orders can import them
# without needing real avanza-api installed
_mock_avanza_client = sys.modules.get("portfolio.avanza_client")
if _mock_avanza_client is None or not hasattr(_mock_avanza_client, "place_buy_order"):
    # Ensure the module exists with the trading functions as attributes
    import portfolio.avanza_client as _avc
    if not hasattr(_avc, "place_buy_order"):
        _avc.place_buy_order = MagicMock()
        _avc.place_sell_order = MagicMock()

import portfolio.avanza_orders as mod


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path, monkeypatch):
    """Redirect DATA_DIR and PENDING_FILE to a temp directory."""
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mod, "PENDING_FILE", tmp_path / "avanza_pending_orders.json")
    return tmp_path


@pytest.fixture
def config():
    return {"telegram": {"token": "fake-token", "chat_id": "123456"}}


# --- request_order ---


class TestRequestOrder:
    def test_creates_pending_order(self, tmp_data_dir):
        order = mod.request_order(
            action="BUY",
            orderbook_id="5533",
            instrument_name="SAAB B",
            config_key="SAAB-B",
            volume=50,
            price=245.0,
        )
        assert order["action"] == "BUY"
        assert order["orderbook_id"] == "5533"
        assert order["instrument_name"] == "SAAB B"
        assert order["volume"] == 50
        assert order["price"] == 245.0
        assert order["total_sek"] == 12250.0
        assert order["status"] == "pending_confirmation"
        assert "id" in order
        assert "expires" in order

    def test_persists_to_disk(self, tmp_data_dir):
        mod.request_order("BUY", "5533", "SAAB B", "SAAB-B", 50, 245.0)
        data = json.loads(mod.PENDING_FILE.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["instrument_name"] == "SAAB B"

    def test_appends_multiple_orders(self, tmp_data_dir):
        mod.request_order("BUY", "5533", "SAAB B", "SAAB-B", 50, 245.0)
        mod.request_order("SELL", "1234", "SEB C", "SEB-C", 100, 150.0)
        data = json.loads(mod.PENDING_FILE.read_text(encoding="utf-8"))
        assert len(data) == 2

    def test_sell_order(self, tmp_data_dir):
        order = mod.request_order("SELL", "1234", "SEB C", "SEB-C", 100, 150.0)
        assert order["action"] == "SELL"
        assert order["total_sek"] == 15000.0

    def test_rejects_invalid_action(self):
        with pytest.raises(ValueError, match="BUY or SELL"):
            mod.request_order("HOLD", "5533", "SAAB B", "SAAB-B", 50, 245.0)

    def test_rejects_zero_volume(self):
        with pytest.raises(ValueError, match="volume"):
            mod.request_order("BUY", "5533", "SAAB B", "SAAB-B", 0, 245.0)

    def test_rejects_negative_price(self):
        with pytest.raises(ValueError, match="price"):
            mod.request_order("BUY", "5533", "SAAB B", "SAAB-B", 50, -1.0)

    def test_expiry_is_5_minutes_ahead(self, tmp_data_dir):
        order = mod.request_order("BUY", "5533", "SAAB B", "SAAB-B", 50, 245.0)
        ts = datetime.fromisoformat(order["timestamp"])
        expires = datetime.fromisoformat(order["expires"])
        diff = (expires - ts).total_seconds()
        assert 299 <= diff <= 301  # ~5 minutes


# --- get_pending_orders ---


class TestGetPendingOrders:
    def test_returns_only_pending(self, tmp_data_dir):
        mod.request_order("BUY", "5533", "SAAB B", "SAAB-B", 50, 245.0)
        # Manually mark one as executed
        data = json.loads(mod.PENDING_FILE.read_text(encoding="utf-8"))
        data[0]["status"] = "executed"
        mod.PENDING_FILE.write_text(json.dumps(data), encoding="utf-8")
        # Add another pending
        mod.request_order("SELL", "1234", "SEB C", "SEB-C", 100, 150.0)

        pending = mod.get_pending_orders()
        assert len(pending) == 1
        assert pending[0]["action"] == "SELL"

    def test_empty_when_no_file(self, tmp_data_dir):
        assert mod.get_pending_orders() == []


# --- check_pending_orders ---


class TestCheckPendingOrders:
    def test_expires_stale_orders(self, tmp_data_dir, config):
        """Orders past their expiry are marked expired."""
        now = datetime.now(UTC)
        order = {
            "id": "test-1",
            "timestamp": (now - timedelta(minutes=10)).isoformat(),
            "action": "BUY",
            "orderbook_id": "5533",
            "instrument_name": "SAAB B",
            "config_key": "SAAB-B",
            "volume": 50,
            "price": 245.0,
            "total_sek": 12250.0,
            "status": "pending_confirmation",
            "expires": (now - timedelta(minutes=5)).isoformat(),
        }
        mod._save_pending([order])

        with patch.object(mod, "_check_telegram_confirm", return_value=set()), \
             patch.object(mod, "_notify_expired") as mock_notify:
            acted = mod.check_pending_orders(config)

        assert len(acted) == 1
        assert acted[0]["status"] == "expired"
        mock_notify.assert_called_once()

    def test_confirms_pending_order(self, tmp_data_dir, config):
        """CONFIRM reply with the right token triggers order execution."""
        order = mod.request_order("BUY", "5533", "SAAB B", "SAAB-B", 50, 245.0)

        with patch.object(mod, "_check_telegram_confirm",
                          return_value={order["confirm_token"]}), \
             patch.object(mod, "_execute_confirmed_order") as mock_exec:
            acted = mod.check_pending_orders(config)

        assert len(acted) == 1
        assert acted[0]["status"] == "confirmed"
        mock_exec.assert_called_once()

    def test_noop_when_no_pending(self, tmp_data_dir, config):
        """No action when no pending orders exist."""
        acted = mod.check_pending_orders(config)
        assert acted == []

    def test_only_targeted_order_confirmed_per_cycle(self, tmp_data_dir, config):
        """A CONFIRM with a specific token confirms ONLY that order, not
        the most recent. (P1-10: this is the headline race fix — the old
        implementation matched the most recent, which silently confirmed
        a NEWER order if the user was confirming an older one.)"""
        order_a = mod.request_order("BUY", "5533", "SAAB B", "SAAB-B", 50, 245.0)
        order_b = mod.request_order("SELL", "1234", "SEB C", "SEB-C", 100, 150.0)

        # User confirms order A's token only.
        with patch.object(mod, "_check_telegram_confirm",
                          return_value={order_a["confirm_token"]}), \
             patch.object(mod, "_execute_confirmed_order"):
            acted = mod.check_pending_orders(config)

        confirmed = [a for a in acted if a["status"] == "confirmed"]
        assert len(confirmed) == 1
        # Order A confirmed, B not.
        assert confirmed[0]["instrument_name"] == "SAAB B"


# --- _check_telegram_confirm ---


class TestCheckTelegramConfirm:
    """P1-10 (2026-05-02): `_check_telegram_confirm` returns set[str] of
    matched tokens. Bare CONFIRM (no token) maps to "" (legacy compat).
    Tests below were updated from bool to set assertions."""

    def test_returns_token_set_on_confirm(self, tmp_data_dir, config):
        """Detects CONFIRM message from Telegram. Bare CONFIRM → {""} set."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "ok": True,
            "result": [
                {
                    "update_id": 100,
                    "message": {
                        "chat": {"id": 123456},
                        "text": "CONFIRM",
                    },
                }
            ],
        }
        with patch.object(mod, "fetch_with_retry", return_value=mock_resp):
            assert mod._check_telegram_confirm(config) == {""}

    def test_returns_empty_set_on_no_updates(self, tmp_data_dir, config):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"ok": True, "result": []}
        with patch.object(mod, "fetch_with_retry", return_value=mock_resp):
            assert mod._check_telegram_confirm(config) == set()

    def test_returns_empty_set_on_wrong_chat(self, tmp_data_dir, config):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "ok": True,
            "result": [
                {
                    "update_id": 100,
                    "message": {
                        "chat": {"id": 999999},
                        "text": "CONFIRM",
                    },
                }
            ],
        }
        with patch.object(mod, "fetch_with_retry", return_value=mock_resp):
            assert mod._check_telegram_confirm(config) == set()

    def test_case_insensitive_confirm(self, tmp_data_dir, config):
        """'confirm' in lowercase still works (legacy bare CONFIRM)."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "ok": True,
            "result": [
                {
                    "update_id": 100,
                    "message": {
                        "chat": {"id": 123456},
                        "text": "confirm",
                    },
                }
            ],
        }
        with patch.object(mod, "fetch_with_retry", return_value=mock_resp):
            assert mod._check_telegram_confirm(config) == {""}

    def test_advances_offset(self, tmp_data_dir, config):
        """Offset file is updated after processing (BUG-128: now atomic JSON)."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "ok": True,
            "result": [
                {"update_id": 42, "message": {"chat": {"id": 123456}, "text": "hello"}},
            ],
        }
        with patch.object(mod, "fetch_with_retry", return_value=mock_resp):
            mod._check_telegram_confirm(config)

        offset_file = tmp_data_dir / "avanza_telegram_offset.txt"
        assert offset_file.exists()
        import json
        data = json.loads(offset_file.read_text(encoding="utf-8"))
        assert data["offset"] == 43

    def test_reads_legacy_plaintext_offset(self, tmp_data_dir, config):
        """BUG-128: Backwards compatibility with old plain-text offset format."""
        offset_file = tmp_data_dir / "avanza_telegram_offset.txt"
        offset_file.write_text("99", encoding="utf-8")

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "ok": True,
            "result": [
                {"update_id": 100, "message": {"chat": {"id": 123456}, "text": "hello"}},
            ],
        }
        with patch.object(mod, "fetch_with_retry", return_value=mock_resp) as mock_fetch:
            mod._check_telegram_confirm(config)

        # Should have used offset=99 in the API call
        call_kwargs = mock_fetch.call_args
        assert call_kwargs[1].get("params", {}).get("offset") == 99 or \
               call_kwargs.kwargs.get("params", {}).get("offset") == 99

    def test_reads_json_offset_format(self, tmp_data_dir, config):
        """BUG-128: Reads new JSON offset format correctly."""
        offset_file = tmp_data_dir / "avanza_telegram_offset.txt"
        offset_file.write_text('{"offset": 77}', encoding="utf-8")

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"ok": True, "result": []}
        with patch.object(mod, "fetch_with_retry", return_value=mock_resp) as mock_fetch:
            mod._check_telegram_confirm(config)

        call_kwargs = mock_fetch.call_args
        assert call_kwargs[1].get("params", {}).get("offset") == 77 or \
               call_kwargs.kwargs.get("params", {}).get("offset") == 77

    def test_returns_empty_set_on_api_error(self, tmp_data_dir, config):
        with patch.object(mod, "fetch_with_retry", return_value=None):
            assert mod._check_telegram_confirm(config) == set()

    def test_returns_empty_set_missing_token(self, tmp_data_dir):
        assert mod._check_telegram_confirm({"telegram": {}}) == set()


# --- AV-P1-3: sender authentication on CONFIRM ---


class TestCheckTelegramConfirmSenderAuth:
    """When `telegram.allowed_user_id` is configured, only CONFIRM messages
    whose sender matches the allowed user are honored. This protects against
    group-chat misuse and bot-token compromise where an attacker could deliver
    fake updates for the configured chat_id.

    When `allowed_user_id` is NOT configured, the existing chat-only check
    remains in place (backwards compatible)."""

    def test_allowed_user_passes(self, tmp_data_dir):
        """A CONFIRM from the allowed user is honored."""
        config = {
            "telegram": {
                "token": "fake-token",
                "chat_id": "123456",
                "allowed_user_id": 7777,
            }
        }
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "ok": True,
            "result": [
                {
                    "update_id": 100,
                    "message": {
                        "chat": {"id": 123456},
                        "from": {"id": 7777},
                        "text": "CONFIRM",
                    },
                }
            ],
        }
        with patch.object(mod, "fetch_with_retry", return_value=mock_resp):
            assert mod._check_telegram_confirm(config) == {""}

    def test_disallowed_user_dropped(self, tmp_data_dir):
        """A CONFIRM from a different user (in the same chat) is dropped.

        This is the attack: someone in the group chat (or an attacker who
        compromised the bot token and is delivering fake updates to the
        right chat_id) sends CONFIRM. With sender authentication, it must
        not execute the pending order.
        """
        config = {
            "telegram": {
                "token": "fake-token",
                "chat_id": "123456",
                "allowed_user_id": 7777,
            }
        }
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "ok": True,
            "result": [
                {
                    "update_id": 100,
                    "message": {
                        "chat": {"id": 123456},
                        "from": {"id": 9999},  # NOT the allowed user
                        "text": "CONFIRM",
                    },
                }
            ],
        }
        with patch.object(mod, "fetch_with_retry", return_value=mock_resp):
            assert mod._check_telegram_confirm(config) == set()

    def test_missing_from_field_dropped_when_auth_enabled(self, tmp_data_dir):
        """If sender info is missing from the message but allowed_user_id is
        set, fail-closed: drop the message rather than honor it."""
        config = {
            "telegram": {
                "token": "fake-token",
                "chat_id": "123456",
                "allowed_user_id": 7777,
            }
        }
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "ok": True,
            "result": [
                {
                    "update_id": 100,
                    "message": {
                        "chat": {"id": 123456},
                        # no "from" field
                        "text": "CONFIRM",
                    },
                }
            ],
        }
        with patch.object(mod, "fetch_with_retry", return_value=mock_resp):
            assert mod._check_telegram_confirm(config) == set()

    def test_allowed_user_as_string(self, tmp_data_dir):
        """allowed_user_id may be configured as either int or string —
        both formats must work."""
        config = {
            "telegram": {
                "token": "fake-token",
                "chat_id": "123456",
                "allowed_user_id": "7777",  # string!
            }
        }
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "ok": True,
            "result": [
                {
                    "update_id": 100,
                    "message": {
                        "chat": {"id": 123456},
                        "from": {"id": 7777},  # int from Telegram
                        "text": "CONFIRM",
                    },
                }
            ],
        }
        with patch.object(mod, "fetch_with_retry", return_value=mock_resp):
            assert mod._check_telegram_confirm(config) == {""}

    def test_no_allowed_user_falls_back_to_chat_only(self, tmp_data_dir):
        """Backwards compat: if allowed_user_id is not configured, the
        existing chat-only check still works (so existing deployments
        don't break)."""
        config = {
            "telegram": {
                "token": "fake-token",
                "chat_id": "123456",
                # no allowed_user_id
            }
        }
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "ok": True,
            "result": [
                {
                    "update_id": 100,
                    "message": {
                        "chat": {"id": 123456},
                        "from": {"id": 9999},  # any sender
                        "text": "CONFIRM",
                    },
                }
            ],
        }
        with patch.object(mod, "fetch_with_retry", return_value=mock_resp):
            assert mod._check_telegram_confirm(config) == {""}

    def test_offset_advances_even_when_sender_dropped(self, tmp_data_dir):
        """Even when CONFIRM is dropped due to sender mismatch, the
        getUpdates offset must still advance so we don't replay the
        rejected message every cycle."""
        config = {
            "telegram": {
                "token": "fake-token",
                "chat_id": "123456",
                "allowed_user_id": 7777,
            }
        }
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "ok": True,
            "result": [
                {
                    "update_id": 200,
                    "message": {
                        "chat": {"id": 123456},
                        "from": {"id": 9999},
                        "text": "CONFIRM",
                    },
                }
            ],
        }
        with patch.object(mod, "fetch_with_retry", return_value=mock_resp):
            mod._check_telegram_confirm(config)

        offset_file = tmp_data_dir / "avanza_telegram_offset.txt"
        assert offset_file.exists()
        data = json.loads(offset_file.read_text(encoding="utf-8"))
        assert data["offset"] == 201


# --- _execute_confirmed_order ---


class TestExecuteConfirmedOrder:
    def test_buy_success(self, config):
        order = {
            "id": "test-1",
            "action": "BUY",
            "orderbook_id": "5533",
            "instrument_name": "SAAB B",
            "volume": 50,
            "price": 245.0,
            "total_sek": 12250.0,
        }
        with patch.object(mod, "place_buy_order") as mock_buy, \
             patch.object(mod, "send_telegram") as mock_tg:
            mock_buy.return_value = {
                "orderId": "AV-123",
                "orderRequestStatus": "SUCCESS",
                "message": "",
            }
            mod._execute_confirmed_order(order, config)

        assert order["status"] == "executed"
        assert order["avanza_order_id"] == "AV-123"
        mock_tg.assert_called_once()
        msg = mock_tg.call_args[0][0]
        assert "EXECUTED" in msg
        assert "SAAB B" in msg

    def test_sell_success(self, config):
        order = {
            "id": "test-2",
            "action": "SELL",
            "orderbook_id": "1234",
            "instrument_name": "SEB C",
            "volume": 100,
            "price": 150.0,
            "total_sek": 15000.0,
        }
        with patch.object(mod, "place_sell_order") as mock_sell, \
             patch.object(mod, "send_telegram"):
            mock_sell.return_value = {
                "orderId": "AV-456",
                "orderRequestStatus": "SUCCESS",
                "message": "",
            }
            mod._execute_confirmed_order(order, config)

        assert order["status"] == "executed"
        mock_sell.assert_called_once_with(
            orderbook_id="1234", price=150.0, volume=100,
        )

    def test_api_error_marks_failed(self, config):
        order = {
            "id": "test-3",
            "action": "BUY",
            "orderbook_id": "5533",
            "instrument_name": "SAAB B",
            "volume": 50,
            "price": 245.0,
            "total_sek": 12250.0,
        }
        with patch.object(mod, "place_buy_order") as mock_buy, \
             patch.object(mod, "send_telegram"):
            mock_buy.return_value = {
                "orderId": "",
                "orderRequestStatus": "ERROR",
                "message": "Insufficient funds",
            }
            mod._execute_confirmed_order(order, config)

        assert order["status"] == "failed"
        assert "Insufficient funds" in order["error"]

    def test_exception_marks_error(self, config):
        order = {
            "id": "test-4",
            "action": "BUY",
            "orderbook_id": "5533",
            "instrument_name": "SAAB B",
            "volume": 50,
            "price": 245.0,
            "total_sek": 12250.0,
        }
        with patch.object(mod, "place_buy_order", side_effect=RuntimeError("Auth failed")), \
             patch.object(mod, "send_telegram"):
            mod._execute_confirmed_order(order, config)

        assert order["status"] == "error"
        assert "Auth failed" in order["error"]

    def test_success_without_orderid_marks_error(self, config):
        """API returns SUCCESS but no orderId — must reject, not save placeholder."""
        order = {
            "id": "test-no-id",
            "action": "BUY",
            "orderbook_id": "5533",
            "instrument_name": "SAAB B",
            "volume": 50,
            "price": 245.0,
            "total_sek": 12250.0,
        }
        with patch.object(mod, "place_buy_order") as mock_buy, \
             patch.object(mod, "send_telegram") as mock_tg:
            mock_buy.return_value = {
                "orderRequestStatus": "SUCCESS",
                "message": "",
            }
            mod._execute_confirmed_order(order, config)

        assert order["status"] == "error"
        assert "no orderId" in order["error"]
        assert mock_tg.called


# --- _notify_expired ---


class TestNotifyExpired:
    def test_sends_expiry_telegram(self, config):
        order = {
            "id": "test-5",
            "action": "BUY",
            "instrument_name": "SAAB B",
            "volume": 50,
            "price": 245.0,
        }
        with patch.object(mod, "send_telegram") as mock_tg:
            mod._notify_expired(order, config)

        mock_tg.assert_called_once()
        msg = mock_tg.call_args[0][0]
        assert "EXPIRED" in msg
        assert "SAAB B" in msg
        assert "5 min" in msg

    def test_handles_telegram_failure(self, config):
        order = {"id": "x", "action": "BUY", "instrument_name": "X", "volume": 1, "price": 1.0}
        with patch.object(mod, "send_telegram", side_effect=Exception("down")):
            mod._notify_expired(order, config)  # Should not raise


# ===========================================================================
# P1-10 (2026-05-02 last-followups): per-order CONFIRM nonce
# ===========================================================================
#
# Three races the bare-CONFIRM design has:
# 1. Stale-CONFIRM race: the offset-save catches an OSError and silently
#    fails; next cycle replays the SAME CONFIRM and confirms a NEWER order.
# 2. Wrong-order race: order A is requested, then B is requested, then user
#    sends CONFIRM intending A. Sort-by-timestamp-DESC confirms B.
# 3. No-pending-yet race: user sends CONFIRM, a NEW order C lands before
#    the next polling cycle, CONFIRM confirms C.
#
# Fix: per-order hex nonce (`confirm_token`). Telegram message becomes
# "Reply CONFIRM <token>". `_check_telegram_confirm` returns a set of
# matched tokens (with `""` for bare CONFIRM, used only for LEGACY orders
# without a token field — backwards compat for in-flight orders at deploy).
#
# `check_pending_orders` matches the pending order whose `confirm_token`
# equals one of the tokens. Bare CONFIRM only matches legacy orders (no
# token field).


class TestRequestOrderNonce:
    """`request_order` generates a unique 6-hex-char `confirm_token` on
    every order so multiple in-flight orders can be disambiguated."""

    def test_request_order_returns_confirm_token(self, tmp_data_dir):
        order = mod.request_order("BUY", "5533", "SAAB B", "SAAB-B", 50, 245.0)
        assert "confirm_token" in order
        token = order["confirm_token"]
        assert isinstance(token, str)
        assert len(token) == 6
        # Hex alphabet only — anything else makes the user-facing prompt
        # ambiguous (case, special chars, etc.).
        assert all(c in "0123456789abcdef" for c in token)

    def test_each_order_gets_unique_token(self, tmp_data_dir):
        tokens = set()
        for _ in range(100):
            order = mod.request_order("BUY", "5533", "SAAB B", "SAAB-B", 1, 1.0)
            tokens.add(order["confirm_token"])
        # 100 hex tokens of 6 chars each — collisions on a real RNG should
        # be effectively zero. If this fails, the token generator is
        # deterministic.
        assert len(tokens) == 100

    def test_token_persists_to_disk(self, tmp_data_dir):
        order = mod.request_order("BUY", "5533", "SAAB B", "SAAB-B", 50, 245.0)
        data = json.loads(mod.PENDING_FILE.read_text(encoding="utf-8"))
        assert data[0]["confirm_token"] == order["confirm_token"]


class TestCheckTelegramConfirmTokens:
    """`_check_telegram_confirm` returns a set[str] of matched tokens.
    `""` is used for bare CONFIRM (legacy backwards compat).

    Bare CONFIRM still flows through (legacy in-flight orders), but new
    orders MUST be confirmed by their specific token to eliminate the
    race."""

    def test_returns_token_for_confirm_with_nonce(self, tmp_data_dir):
        config = {"telegram": {"token": "fake-token", "chat_id": "123456"}}
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "ok": True,
            "result": [
                {
                    "update_id": 100,
                    "message": {
                        "chat": {"id": 123456},
                        "text": "CONFIRM a1b2c3",
                    },
                }
            ],
        }
        with patch.object(mod, "fetch_with_retry", return_value=mock_resp):
            result = mod._check_telegram_confirm(config)
        # Set with the matched token only.
        assert result == {"a1b2c3"}

    def test_token_is_lowercased(self, tmp_data_dir):
        """User may type CONFIRM A1B2C3 — normalize to lowercase so the
        match against order['confirm_token'] (always lowercase hex) works."""
        config = {"telegram": {"token": "fake-token", "chat_id": "123456"}}
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "ok": True,
            "result": [
                {
                    "update_id": 100,
                    "message": {
                        "chat": {"id": 123456},
                        "text": "confirm A1B2C3",
                    },
                }
            ],
        }
        with patch.object(mod, "fetch_with_retry", return_value=mock_resp):
            result = mod._check_telegram_confirm(config)
        assert result == {"a1b2c3"}

    def test_bare_confirm_returns_empty_string_token(self, tmp_data_dir):
        """Bare 'CONFIRM' (no token) → empty-string token in the result.
        Used only for legacy backwards compat (orders without a token)."""
        config = {"telegram": {"token": "fake-token", "chat_id": "123456"}}
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "ok": True,
            "result": [
                {
                    "update_id": 100,
                    "message": {
                        "chat": {"id": 123456},
                        "text": "CONFIRM",
                    },
                }
            ],
        }
        with patch.object(mod, "fetch_with_retry", return_value=mock_resp):
            result = mod._check_telegram_confirm(config)
        assert result == {""}

    def test_returns_empty_set_when_no_confirm(self, tmp_data_dir):
        config = {"telegram": {"token": "fake-token", "chat_id": "123456"}}
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"ok": True, "result": []}
        with patch.object(mod, "fetch_with_retry", return_value=mock_resp):
            result = mod._check_telegram_confirm(config)
        assert result == set()

    def test_multiple_tokens_in_one_poll(self, tmp_data_dir):
        """If multiple CONFIRM messages arrive between polls, each token
        is collected. Each pending order will then look up its own token."""
        config = {"telegram": {"token": "fake-token", "chat_id": "123456"}}
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "ok": True,
            "result": [
                {"update_id": 100, "message": {"chat": {"id": 123456}, "text": "CONFIRM aaa111"}},
                {"update_id": 101, "message": {"chat": {"id": 123456}, "text": "CONFIRM bbb222"}},
            ],
        }
        with patch.object(mod, "fetch_with_retry", return_value=mock_resp):
            result = mod._check_telegram_confirm(config)
        assert result == {"aaa111", "bbb222"}

    def test_token_with_extra_whitespace_normalized(self, tmp_data_dir):
        """User may type 'CONFIRM   a1b2c3' (extra spaces). Normalize so
        the lookup still works."""
        config = {"telegram": {"token": "fake-token", "chat_id": "123456"}}
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "ok": True,
            "result": [
                {"update_id": 100, "message": {"chat": {"id": 123456}, "text": "CONFIRM   a1b2c3  "}},
            ],
        }
        with patch.object(mod, "fetch_with_retry", return_value=mock_resp):
            result = mod._check_telegram_confirm(config)
        assert result == {"a1b2c3"}

    def test_garbage_after_confirm_rejected(self, tmp_data_dir):
        """'CONFIRM not-a-token' (anything that's not a hex string) →
        rejected. Better to drop than to silently confirm a wrong order
        because the user typed a typo."""
        config = {"telegram": {"token": "fake-token", "chat_id": "123456"}}
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "ok": True,
            "result": [
                {"update_id": 100, "message": {"chat": {"id": 123456}, "text": "CONFIRM xyz"}},
            ],
        }
        with patch.object(mod, "fetch_with_retry", return_value=mock_resp):
            result = mod._check_telegram_confirm(config)
        # 'xyz' is not valid hex — drop.
        assert result == set()

    def test_confirms_typo_does_not_match_bare(self, tmp_data_dir):
        """'CONFIRMS' (extra letter) MUST NOT be treated as bare CONFIRM.
        Otherwise an unrelated message starting with 'confirm' would match
        legacy orders. Defensive — typo plus accidental fat-finger could
        execute real money."""
        config = {"telegram": {"token": "fake-token", "chat_id": "123456"}}
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "ok": True,
            "result": [
                {"update_id": 100, "message": {"chat": {"id": 123456}, "text": "CONFIRMS"}},
                {"update_id": 101, "message": {"chat": {"id": 123456}, "text": "confirmed"}},
                {"update_id": 102, "message": {"chat": {"id": 123456}, "text": "confirmation"}},
            ],
        }
        with patch.object(mod, "fetch_with_retry", return_value=mock_resp):
            result = mod._check_telegram_confirm(config)
        # 'confirms' / 'confirmed' / 'confirmation' all parse to non-hex
        # remainders ('s' / 'ed' / 'ation'); none should match bare.
        assert result == set()


class TestCheckPendingOrdersTokenMatching:
    """`check_pending_orders` must match the SPECIFIC order whose
    `confirm_token` was confirmed — not the most recent. This eliminates
    the wrong-order race."""

    def test_token_match_confirms_specific_order(self, tmp_data_dir):
        """Two orders A (older) and B (newer). Token for A is sent → A is
        confirmed (NOT B). This is the headline race fix."""
        config = {"telegram": {"token": "fake-token", "chat_id": "123456"}}
        order_a = mod.request_order("BUY", "5533", "SAAB B", "SAAB-B", 50, 245.0)
        order_b = mod.request_order("SELL", "1234", "SEB C", "SEB-C", 100, 150.0)
        token_a = order_a["confirm_token"]

        with patch.object(mod, "_check_telegram_confirm", return_value={token_a}), \
             patch.object(mod, "_execute_confirmed_order") as mock_exec:
            acted = mod.check_pending_orders(config)

        confirmed = [o for o in acted if o["status"] == "confirmed"]
        assert len(confirmed) == 1
        # The confirmed order is A (the one whose token was sent).
        assert confirmed[0]["confirm_token"] == token_a
        assert confirmed[0]["instrument_name"] == "SAAB B"
        # Order B is left alone.
        pending_after = mod.get_pending_orders()
        assert len(pending_after) == 1
        assert pending_after[0]["instrument_name"] == "SEB C"

    def test_unknown_token_confirms_nothing(self, tmp_data_dir):
        """A token that doesn't match any pending order → no confirmation,
        no execution. Pending orders unchanged."""
        config = {"telegram": {"token": "fake-token", "chat_id": "123456"}}
        mod.request_order("BUY", "5533", "SAAB B", "SAAB-B", 50, 245.0)

        with patch.object(mod, "_check_telegram_confirm", return_value={"unknown"}), \
             patch.object(mod, "_execute_confirmed_order") as mock_exec:
            acted = mod.check_pending_orders(config)

        assert acted == []
        mock_exec.assert_not_called()
        # Pending order survives.
        assert len(mod.get_pending_orders()) == 1

    def test_legacy_order_without_token_accepts_bare_confirm(self, tmp_data_dir):
        """Backwards-compat: an in-flight order without a `confirm_token`
        (loaded from disk after a deploy) still accepts bare CONFIRM. This
        prevents stuck orders during the upgrade window."""
        config = {"telegram": {"token": "fake-token", "chat_id": "123456"}}
        now = datetime.now(UTC)
        legacy_order = {
            "id": "legacy-1",
            "timestamp": now.isoformat(),
            "action": "BUY",
            "orderbook_id": "5533",
            "instrument_name": "SAAB B",
            "config_key": "SAAB-B",
            "volume": 50,
            "price": 245.0,
            "total_sek": 12250.0,
            "status": "pending_confirmation",
            "expires": (now + timedelta(minutes=5)).isoformat(),
            # NOTE: no confirm_token field.
        }
        mod._save_pending([legacy_order])

        # Bare CONFIRM → "" token in the set.
        with patch.object(mod, "_check_telegram_confirm", return_value={""}), \
             patch.object(mod, "_execute_confirmed_order") as mock_exec:
            acted = mod.check_pending_orders(config)

        assert len(acted) == 1
        assert acted[0]["status"] == "confirmed"
        mock_exec.assert_called_once()

    def test_bare_confirm_does_NOT_match_token_holding_order(self, tmp_data_dir):
        """The race fix: a bare CONFIRM (no token) MUST NOT match a new
        order that has its own token. Otherwise a stale CONFIRM could
        still confirm a newer order — defeating the whole point."""
        config = {"telegram": {"token": "fake-token", "chat_id": "123456"}}
        # New order with a token.
        mod.request_order("BUY", "5533", "SAAB B", "SAAB-B", 50, 245.0)

        with patch.object(mod, "_check_telegram_confirm", return_value={""}), \
             patch.object(mod, "_execute_confirmed_order") as mock_exec:
            acted = mod.check_pending_orders(config)

        # Nothing happens — bare CONFIRM doesn't match a token-holding order.
        assert acted == []
        mock_exec.assert_not_called()
        assert len(mod.get_pending_orders()) == 1

    def test_two_tokens_confirm_two_orders_same_cycle(self, tmp_data_dir):
        """Two CONFIRMs in one poll → two orders confirmed (not one)."""
        config = {"telegram": {"token": "fake-token", "chat_id": "123456"}}
        order_a = mod.request_order("BUY", "5533", "SAAB B", "SAAB-B", 50, 245.0)
        order_b = mod.request_order("SELL", "1234", "SEB C", "SEB-C", 100, 150.0)

        confirmed_tokens = {order_a["confirm_token"], order_b["confirm_token"]}
        with patch.object(mod, "_check_telegram_confirm", return_value=confirmed_tokens), \
             patch.object(mod, "_execute_confirmed_order") as mock_exec:
            acted = mod.check_pending_orders(config)

        confirmed = [o for o in acted if o["status"] == "confirmed"]
        assert len(confirmed) == 2
        assert mock_exec.call_count == 2

    def test_expired_order_still_expires_without_token(self, tmp_data_dir):
        """Expiry is independent of token matching."""
        config = {"telegram": {"token": "fake-token", "chat_id": "123456"}}
        now = datetime.now(UTC)
        order = {
            "id": "stale-1",
            "timestamp": (now - timedelta(minutes=10)).isoformat(),
            "action": "BUY",
            "orderbook_id": "5533",
            "instrument_name": "SAAB B",
            "config_key": "SAAB-B",
            "volume": 50,
            "price": 245.0,
            "total_sek": 12250.0,
            "status": "pending_confirmation",
            "expires": (now - timedelta(minutes=5)).isoformat(),
            "confirm_token": "abcdef",
        }
        mod._save_pending([order])

        with patch.object(mod, "_check_telegram_confirm", return_value=set()), \
             patch.object(mod, "_notify_expired") as mock_notify:
            acted = mod.check_pending_orders(config)

        assert len(acted) == 1
        assert acted[0]["status"] == "expired"
        mock_notify.assert_called_once()

    def test_expired_order_not_confirmed_even_with_matching_token(self, tmp_data_dir):
        """B2: Expired order must expire even if CONFIRM token arrives."""
        config = {"telegram": {"token": "fake-token", "chat_id": "123456"}}
        now = datetime.now(UTC)
        order = {
            "id": "late-confirm-1",
            "timestamp": (now - timedelta(minutes=10)).isoformat(),
            "action": "BUY",
            "orderbook_id": "5533",
            "instrument_name": "SAAB B",
            "config_key": "SAAB-B",
            "volume": 50,
            "price": 245.0,
            "total_sek": 12250.0,
            "status": "pending_confirmation",
            "expires": (now - timedelta(minutes=1)).isoformat(),
            "confirm_token": "abcdef",
        }
        mod._save_pending([order])

        with patch.object(mod, "_check_telegram_confirm", return_value={"abcdef"}), \
             patch.object(mod, "_execute_confirmed_order") as mock_exec, \
             patch.object(mod, "_notify_expired") as mock_notify:
            acted = mod.check_pending_orders(config)

        assert len(acted) == 1
        assert acted[0]["status"] == "expired"
        mock_exec.assert_not_called()
        mock_notify.assert_called_once()
