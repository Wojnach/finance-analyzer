"""Tests for portfolio.avanza.trading — orders, stop-losses, deals."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from portfolio.avanza.auth import AvanzaAuth
from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.trading import (
    cancel_order,
    delete_stop_loss,
    get_deals,
    get_orders,
    get_stop_losses,
    modify_order,
    place_order,
    place_stop_loss,
    place_trailing_stop,
)
from portfolio.avanza.types import Deal, Order, OrderResult, StopLoss, StopLossResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset singletons before and after every test."""
    AvanzaClient.reset()
    AvanzaAuth.reset()
    yield
    AvanzaClient.reset()
    AvanzaAuth.reset()


def _make_mock_client():
    client = MagicMock()
    client._push_subscription_id = "push-123"
    client._security_token = "csrf-abc"
    client._authentication_session = "auth-xyz"
    client._customer_id = "cust-42"
    client._session = MagicMock()
    return client


def _make_config():
    return {
        "avanza": {
            "username": "testuser",
            "password": "testpass",
            "totp_secret": "TESTSECRET",
        }
    }


@pytest.fixture()
def mock_avanza():
    """Set up AvanzaClient singleton with a mocked underlying avanza lib."""
    with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
        mock_client = _make_mock_client()
        mock_create.return_value = mock_client
        AvanzaClient.get_instance(_make_config())
        yield mock_client


# ---------------------------------------------------------------------------
# place_order
# ---------------------------------------------------------------------------

class TestPlaceOrder:
    def test_success(self, mock_avanza):
        mock_avanza.place_order.return_value = {
            "orderRequestStatus": "SUCCESS",
            "orderId": "ORD-123",
            "message": "",
        }
        result = place_order("BUY", "2213050", price=5.80, volume=100)
        assert isinstance(result, OrderResult)
        assert result.success is True
        assert result.order_id == "ORD-123"

        # Verify the call
        call_args = mock_avanza.place_order.call_args
        assert call_args[0][0] == "1625505"  # default account
        assert call_args[0][1] == "2213050"  # ob_id

    def test_failure(self, mock_avanza):
        mock_avanza.place_order.return_value = {
            "orderRequestStatus": "ERROR",
            "orderId": "",
            "message": "Insufficient funds",
        }
        result = place_order("BUY", "2213050", price=100.0, volume=10)
        assert result.success is False
        assert result.message == "Insufficient funds"

    def test_custom_account_id(self, mock_avanza):
        mock_avanza.place_order.return_value = {
            "orderRequestStatus": "SUCCESS",
            "orderId": "ORD-456",
            "message": "",
        }
        place_order("SELL", "123", price=10.0, volume=50, account_id="9999999")
        call_args = mock_avanza.place_order.call_args
        assert call_args[0][0] == "9999999"

    def test_volume_validation(self, mock_avanza):
        with pytest.raises(ValueError, match="volume must be >= 1"):
            place_order("BUY", "123", price=10.0, volume=0)

    def test_price_validation(self, mock_avanza):
        with pytest.raises(ValueError, match="price must be > 0"):
            place_order("BUY", "123", price=0, volume=10)

    def test_negative_price_validation(self, mock_avanza):
        with pytest.raises(ValueError, match="price must be > 0"):
            place_order("BUY", "123", price=-5.0, volume=10)

    def test_custom_condition(self, mock_avanza):
        mock_avanza.place_order.return_value = {
            "orderRequestStatus": "SUCCESS",
            "orderId": "ORD-789",
            "message": "",
        }
        place_order("BUY", "123", price=10.0, volume=5, condition="FILL_OR_KILL")
        call_kwargs = mock_avanza.place_order.call_args
        # Compare by .value to avoid xdist mock contamination of avanza.constants
        assert call_kwargs[1]["condition"].value == "FILL_OR_KILL"


# ---------------------------------------------------------------------------
# modify_order
# ---------------------------------------------------------------------------

class TestModifyOrder:
    def test_success(self, mock_avanza):
        mock_avanza.edit_order.return_value = {
            "orderRequestStatus": "SUCCESS",
            "orderId": "ORD-123",
            "message": "",
        }
        result = modify_order("ORD-123", "2213050", price=5.90, volume=100)
        assert isinstance(result, OrderResult)
        assert result.success is True
        mock_avanza.edit_order.assert_called_once()

        call_args = mock_avanza.edit_order.call_args[0]
        assert call_args[0] == "ORD-123"  # order_id
        assert call_args[1] == "1625505"  # account_id


# ---------------------------------------------------------------------------
# cancel_order
# ---------------------------------------------------------------------------

class TestCancelOrder:
    def test_success(self, mock_avanza):
        mock_avanza.delete_order.return_value = {
            "orderRequestStatus": "SUCCESS",
            "orderId": "ORD-123",
        }
        result = cancel_order("ORD-123")
        assert result is True
        mock_avanza.delete_order.assert_called_once_with("1625505", "ORD-123")

    def test_failure(self, mock_avanza):
        mock_avanza.delete_order.return_value = {
            "orderRequestStatus": "ERROR",
            "orderId": "ORD-123",
        }
        result = cancel_order("ORD-123")
        assert result is False

    def test_custom_account(self, mock_avanza):
        mock_avanza.delete_order.return_value = {
            "orderRequestStatus": "SUCCESS",
            "orderId": "ORD-999",
        }
        cancel_order("ORD-999", account_id="7777777")
        mock_avanza.delete_order.assert_called_once_with("7777777", "ORD-999")


# ---------------------------------------------------------------------------
# get_orders
# ---------------------------------------------------------------------------

class TestGetOrders:
    def test_returns_orders_from_list(self, mock_avanza):
        mock_avanza.get_orders.return_value = [
            {"orderId": "O1", "orderBookId": "123", "orderType": "BUY", "price": 10.0, "volume": 100, "status": "OPEN", "accountId": "1625505"},
            {"orderId": "O2", "orderBookId": "456", "orderType": "SELL", "price": 20.0, "volume": 50, "status": "FILLED", "accountId": "1625505"},
        ]
        orders = get_orders()
        assert len(orders) == 2
        assert all(isinstance(o, Order) for o in orders)
        assert orders[0].order_id == "O1"
        assert orders[1].side == "SELL"

    def test_returns_orders_from_dict(self, mock_avanza):
        mock_avanza.get_orders.return_value = {
            "orders": [
                {"orderId": "O1", "orderBookId": "123", "orderType": "BUY", "price": 10.0, "volume": 100, "status": "OPEN", "accountId": "1625505"},
            ]
        }
        orders = get_orders()
        assert len(orders) == 1

    def test_empty(self, mock_avanza):
        mock_avanza.get_orders.return_value = []
        assert get_orders() == []


# ---------------------------------------------------------------------------
# get_deals
# ---------------------------------------------------------------------------

class TestGetDeals:
    def test_returns_deals(self, mock_avanza):
        mock_avanza.get_deals.return_value = [
            {"dealId": "D1", "orderBookId": "123", "orderType": "BUY", "price": 10.0, "volume": 100, "dealTime": "14:30:00", "accountId": "1625505"},
        ]
        deals = get_deals()
        assert len(deals) == 1
        assert isinstance(deals[0], Deal)
        assert deals[0].deal_id == "D1"

    def test_empty(self, mock_avanza):
        mock_avanza.get_deals.return_value = []
        assert get_deals() == []


# ---------------------------------------------------------------------------
# place_stop_loss
# ---------------------------------------------------------------------------

class TestPlaceStopLoss:
    def test_success(self, mock_avanza):
        mock_avanza.place_stop_loss_order.return_value = {
            "status": "ACTIVE",
            "stopLossId": "SL-100",
        }
        result = place_stop_loss(
            ob_id="2213050",
            trigger_price=5.50,
            sell_price=5.40,
            volume=100,
        )
        assert isinstance(result, StopLossResult)
        assert result.success is True
        assert result.stop_id == "SL-100"

        # Verify the call
        call_args = mock_avanza.place_stop_loss_order.call_args[0]
        assert call_args[0] == "0"  # parent_stop_loss_id
        assert call_args[1] == "1625505"  # account_id
        assert call_args[2] == "2213050"  # ob_id

    def test_custom_trigger_type(self, mock_avanza):
        mock_avanza.place_stop_loss_order.return_value = {
            "status": "ACTIVE",
            "stopLossId": "SL-200",
        }
        place_stop_loss(
            ob_id="123",
            trigger_price=100.0,
            sell_price=95.0,
            volume=50,
            trigger_type="MORE_OR_EQUAL",
        )
        # The trigger object passed to the API should have the right type
        trigger = mock_avanza.place_stop_loss_order.call_args[0][3]
        # Compare by .value to avoid xdist mock contamination of avanza.constants
        assert trigger.type.value == "MORE_OR_EQUAL"


# ---------------------------------------------------------------------------
# place_trailing_stop
# ---------------------------------------------------------------------------

class TestPlaceTrailingStop:
    def test_delegates_to_place_stop_loss(self, mock_avanza):
        mock_avanza.place_stop_loss_order.return_value = {
            "status": "ACTIVE",
            "stopLossId": "SL-300",
        }
        result = place_trailing_stop(
            ob_id="2213050",
            trail_percent=5.0,
            volume=100,
        )
        assert result.success is True
        assert result.stop_id == "SL-300"

        trigger = mock_avanza.place_stop_loss_order.call_args[0][3]
        # Compare by .value to avoid xdist mock contamination of avanza.constants
        assert trigger.type.value == "FOLLOW_DOWNWARDS"
        assert trigger.value == 5.0
        assert trigger.value_type.value == "PERCENTAGE"


# ---------------------------------------------------------------------------
# get_stop_losses
# ---------------------------------------------------------------------------

class TestGetStopLosses:
    def test_returns_stop_losses(self, mock_avanza):
        mock_avanza.get_all_stop_losses.return_value = [
            {
                "id": "SL-1",
                "orderBookId": "2213050",
                "trigger": {"value": 5.50, "type": "LESS_OR_EQUAL"},
                "orderEvent": {"price": 5.40, "volume": 100},
                "status": "ACTIVE",
                "accountId": "1625505",
            },
        ]
        sls = get_stop_losses()
        assert len(sls) == 1
        assert isinstance(sls[0], StopLoss)
        assert sls[0].stop_id == "SL-1"
        assert sls[0].trigger_price == 5.50

    def test_empty(self, mock_avanza):
        mock_avanza.get_all_stop_losses.return_value = []
        assert get_stop_losses() == []


# ---------------------------------------------------------------------------
# delete_stop_loss
# ---------------------------------------------------------------------------

class TestDeleteStopLoss:
    def test_success(self, mock_avanza):
        mock_avanza.delete_stop_loss_order.return_value = None
        result = delete_stop_loss("SL-100")
        assert result is True
        mock_avanza.delete_stop_loss_order.assert_called_once_with("1625505", "SL-100")

    def test_404_is_idempotent(self, mock_avanza):
        mock_avanza.delete_stop_loss_order.side_effect = Exception("404 Not Found")
        result = delete_stop_loss("SL-GONE")
        assert result is True  # 404 treated as success

    def test_other_error_returns_false(self, mock_avanza):
        mock_avanza.delete_stop_loss_order.side_effect = Exception("500 Server Error")
        result = delete_stop_loss("SL-BAD")
        assert result is False

    def test_custom_account(self, mock_avanza):
        mock_avanza.delete_stop_loss_order.return_value = None
        delete_stop_loss("SL-100", account_id="9999999")
        mock_avanza.delete_stop_loss_order.assert_called_once_with("9999999", "SL-100")
