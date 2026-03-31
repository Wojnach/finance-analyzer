"""Tests for portfolio.avanza.tick_rules — price rounding."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from portfolio.avanza.auth import AvanzaAuth
from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.tick_rules import clear_cache, get_tick_rules, round_to_tick
from portfolio.avanza.types import TickEntry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset singletons and tick cache before and after every test."""
    AvanzaClient.reset()
    AvanzaAuth.reset()
    clear_cache()
    yield
    AvanzaClient.reset()
    AvanzaAuth.reset()
    clear_cache()


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
# Shared raw tick data (typical certificate tick table)
# ---------------------------------------------------------------------------

RAW_TICK_TABLE = {
    "tickSizeList": [
        {"min": 0.0, "max": 0.999, "tick": 0.001},
        {"min": 1.0, "max": 9.999, "tick": 0.01},
        {"min": 10.0, "max": 99.99, "tick": 0.05},
        {"min": 100.0, "max": 999.99, "tick": 0.10},
    ]
}


# ---------------------------------------------------------------------------
# get_tick_rules
# ---------------------------------------------------------------------------

class TestGetTickRules:
    def test_returns_entries(self, mock_avanza):
        mock_avanza.get_order_book.return_value = RAW_TICK_TABLE
        entries = get_tick_rules("2213050")
        assert len(entries) == 4
        assert all(isinstance(e, TickEntry) for e in entries)
        assert entries[0].min_price == 0.0
        assert entries[0].tick_size == 0.001
        assert entries[-1].tick_size == 0.10

    def test_cache_second_call_no_api(self, mock_avanza):
        mock_avanza.get_order_book.return_value = RAW_TICK_TABLE
        entries1 = get_tick_rules("2213050")
        entries2 = get_tick_rules("2213050")
        assert entries1 is entries2  # Same object (cached)
        mock_avanza.get_order_book.assert_called_once()  # Only one API call

    def test_different_ob_ids_separate_cache(self, mock_avanza):
        mock_avanza.get_order_book.return_value = RAW_TICK_TABLE
        get_tick_rules("111")
        get_tick_rules("222")
        assert mock_avanza.get_order_book.call_count == 2

    def test_sorted_by_min_price(self, mock_avanza):
        mock_avanza.get_order_book.return_value = {
            "tickSizeList": [
                {"min": 10.0, "max": 99.99, "tick": 0.05},
                {"min": 0.0, "max": 9.999, "tick": 0.01},
            ]
        }
        entries = get_tick_rules("123")
        assert entries[0].min_price == 0.0
        assert entries[1].min_price == 10.0

    def test_alternative_key_name(self, mock_avanza):
        mock_avanza.get_order_book.return_value = {
            "tickSizes": [
                {"minPrice": 0.0, "maxPrice": 9.999, "tickSize": 0.01},
            ]
        }
        entries = get_tick_rules("456")
        assert len(entries) == 1
        assert entries[0].tick_size == 0.01


# ---------------------------------------------------------------------------
# round_to_tick
# ---------------------------------------------------------------------------

class TestRoundToTick:
    @pytest.fixture(autouse=True)
    def _setup_tick_table(self, mock_avanza):
        mock_avanza.get_order_book.return_value = RAW_TICK_TABLE

    def test_round_down_sub_one(self):
        # Price 0.5555 with tick 0.001 -> floor to 0.555
        result = round_to_tick(0.5555, "2213050", direction="down")
        assert result == 0.555

    def test_round_up_sub_one(self):
        # Price 0.5551 with tick 0.001 -> ceil to 0.556
        result = round_to_tick(0.5551, "2213050", direction="up")
        assert result == 0.556

    def test_round_down_medium_range(self):
        # Price 5.873 with tick 0.01 -> floor to 5.87
        result = round_to_tick(5.873, "2213050", direction="down")
        assert result == 5.87

    def test_round_up_medium_range(self):
        # Price 5.871 with tick 0.01 -> ceil to 5.88
        result = round_to_tick(5.871, "2213050", direction="up")
        assert result == 5.88

    def test_round_down_large_range(self):
        # Price 47.37 with tick 0.05 -> floor to 47.35
        result = round_to_tick(47.37, "2213050", direction="down")
        assert result == 47.35

    def test_round_up_large_range(self):
        # Price 47.31 with tick 0.05 -> ceil to 47.35
        result = round_to_tick(47.31, "2213050", direction="up")
        assert result == 47.35

    def test_exact_tick_price_unchanged_down(self):
        # Price 5.80 is already on a tick (0.01) -> stays 5.80
        result = round_to_tick(5.80, "2213050", direction="down")
        assert result == 5.80

    def test_exact_tick_price_unchanged_up(self):
        result = round_to_tick(5.80, "2213050", direction="up")
        assert result == 5.80

    def test_exact_tick_boundary_price(self):
        # Price 100.0 with tick 0.10 -> stays 100.0
        result = round_to_tick(100.0, "2213050", direction="down")
        assert result == 100.0

    def test_round_hundred_range(self):
        # Price 150.35 with tick 0.10 -> floor to 150.3
        result = round_to_tick(150.35, "2213050", direction="down")
        assert result == 150.3

    def test_round_hundred_range_up(self):
        # Price 150.35 with tick 0.10 -> ceil to 150.4
        result = round_to_tick(150.35, "2213050", direction="up")
        assert result == 150.4

    def test_default_direction_is_down(self):
        result = round_to_tick(5.879, "2213050")
        assert result == 5.87

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction must be"):
            round_to_tick(5.80, "2213050", direction="sideways")


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------

class TestClearCache:
    def test_clears_cache(self, mock_avanza):
        mock_avanza.get_order_book.return_value = RAW_TICK_TABLE
        get_tick_rules("2213050")
        clear_cache()
        get_tick_rules("2213050")
        # After clearing, the API should be called again
        assert mock_avanza.get_order_book.call_count == 2
