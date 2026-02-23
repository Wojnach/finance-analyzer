"""Tests for Avanza order confirmation flow."""

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
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
        now = datetime.now(timezone.utc)
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

        with patch.object(mod, "_check_telegram_confirm", return_value=False), \
             patch.object(mod, "_notify_expired") as mock_notify:
            acted = mod.check_pending_orders(config)

        assert len(acted) == 1
        assert acted[0]["status"] == "expired"
        mock_notify.assert_called_once()

    def test_confirms_pending_order(self, tmp_data_dir, config):
        """CONFIRM reply triggers order execution."""
        mod.request_order("BUY", "5533", "SAAB B", "SAAB-B", 50, 245.0)

        with patch.object(mod, "_check_telegram_confirm", return_value=True), \
             patch.object(mod, "_execute_confirmed_order") as mock_exec:
            acted = mod.check_pending_orders(config)

        assert len(acted) == 1
        assert acted[0]["status"] == "confirmed"
        mock_exec.assert_called_once()

    def test_noop_when_no_pending(self, tmp_data_dir, config):
        """No action when no pending orders exist."""
        acted = mod.check_pending_orders(config)
        assert acted == []

    def test_only_one_order_confirmed_per_cycle(self, tmp_data_dir, config):
        """A single CONFIRM only confirms one order."""
        mod.request_order("BUY", "5533", "SAAB B", "SAAB-B", 50, 245.0)
        mod.request_order("SELL", "1234", "SEB C", "SEB-C", 100, 150.0)

        with patch.object(mod, "_check_telegram_confirm", return_value=True), \
             patch.object(mod, "_execute_confirmed_order"):
            acted = mod.check_pending_orders(config)

        confirmed = [a for a in acted if a["status"] == "confirmed"]
        assert len(confirmed) == 1


# --- _check_telegram_confirm ---


class TestCheckTelegramConfirm:
    def test_returns_true_on_confirm(self, tmp_data_dir, config):
        """Detects CONFIRM message from Telegram."""
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
            assert mod._check_telegram_confirm(config) is True

    def test_returns_false_on_no_updates(self, tmp_data_dir, config):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"ok": True, "result": []}
        with patch.object(mod, "fetch_with_retry", return_value=mock_resp):
            assert mod._check_telegram_confirm(config) is False

    def test_returns_false_on_wrong_chat(self, tmp_data_dir, config):
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
            assert mod._check_telegram_confirm(config) is False

    def test_case_insensitive_confirm(self, tmp_data_dir, config):
        """'confirm' in lowercase still works."""
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
            assert mod._check_telegram_confirm(config) is True

    def test_advances_offset(self, tmp_data_dir, config):
        """Offset file is updated after processing."""
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
        assert int(offset_file.read_text().strip()) == 43

    def test_returns_false_on_api_error(self, tmp_data_dir, config):
        with patch.object(mod, "fetch_with_retry", return_value=None):
            assert mod._check_telegram_confirm(config) is False

    def test_returns_false_missing_token(self, tmp_data_dir):
        assert mod._check_telegram_confirm({"telegram": {}}) is False


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
