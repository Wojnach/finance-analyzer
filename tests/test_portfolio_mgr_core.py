"""Tests for portfolio.portfolio_mgr — load_state, save_state, portfolio_value."""

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from portfolio import portfolio_mgr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(cash=500_000, holdings=None, transactions=None, **extra):
    """Build a minimal portfolio state dict."""
    state = {
        "cash_sek": cash,
        "holdings": holdings or {},
        "transactions": transactions or [],
        "start_date": "2026-01-01T00:00:00+00:00",
        "initial_value_sek": 500_000,
    }
    state.update(extra)
    return state


def _write_state(path, state):
    """Write a JSON state file to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")


# ===========================================================================
# load_state
# ===========================================================================

class TestLoadStateDefault:
    """load_state when no file exists should return the default state."""

    def test_returns_dict(self, tmp_path):
        fake = tmp_path / "missing.json"
        with patch.object(portfolio_mgr, "STATE_FILE", fake):
            result = portfolio_mgr.load_state()
        assert isinstance(result, dict)

    def test_default_cash(self, tmp_path):
        fake = tmp_path / "missing.json"
        with patch.object(portfolio_mgr, "STATE_FILE", fake):
            result = portfolio_mgr.load_state()
        assert result["cash_sek"] == 500_000

    def test_default_holdings_empty(self, tmp_path):
        fake = tmp_path / "missing.json"
        with patch.object(portfolio_mgr, "STATE_FILE", fake):
            result = portfolio_mgr.load_state()
        assert result["holdings"] == {}

    def test_default_transactions_empty(self, tmp_path):
        fake = tmp_path / "missing.json"
        with patch.object(portfolio_mgr, "STATE_FILE", fake):
            result = portfolio_mgr.load_state()
        assert result["transactions"] == []

    def test_default_initial_value(self, tmp_path):
        fake = tmp_path / "missing.json"
        with patch.object(portfolio_mgr, "STATE_FILE", fake):
            result = portfolio_mgr.load_state()
        assert result["initial_value_sek"] == 500_000

    def test_default_has_start_date(self, tmp_path):
        fake = tmp_path / "missing.json"
        with patch.object(portfolio_mgr, "STATE_FILE", fake):
            result = portfolio_mgr.load_state()
        # start_date should be a valid ISO-8601 string
        assert "start_date" in result
        dt = datetime.fromisoformat(result["start_date"])
        assert dt.tzinfo is not None  # timezone-aware


class TestLoadStateExisting:
    """load_state when a file exists should return its contents."""

    def test_loads_custom_cash(self, tmp_path):
        fake = tmp_path / "state.json"
        state = _make_state(cash=123_456)
        _write_state(fake, state)
        with patch.object(portfolio_mgr, "STATE_FILE", fake):
            result = portfolio_mgr.load_state()
        assert result["cash_sek"] == 123_456

    def test_loads_holdings(self, tmp_path):
        fake = tmp_path / "state.json"
        holdings = {"BTC-USD": {"shares": 0.5, "avg_cost_usd": 60000}}
        state = _make_state(holdings=holdings)
        _write_state(fake, state)
        with patch.object(portfolio_mgr, "STATE_FILE", fake):
            result = portfolio_mgr.load_state()
        assert result["holdings"]["BTC-USD"]["shares"] == 0.5
        assert result["holdings"]["BTC-USD"]["avg_cost_usd"] == 60000

    def test_preserves_extra_fields(self, tmp_path):
        fake = tmp_path / "state.json"
        state = _make_state(total_fees_sek=42.5, custom_field="hello")
        _write_state(fake, state)
        with patch.object(portfolio_mgr, "STATE_FILE", fake):
            result = portfolio_mgr.load_state()
        assert result["total_fees_sek"] == 42.5
        assert result["custom_field"] == "hello"

    def test_preserves_transactions(self, tmp_path):
        fake = tmp_path / "state.json"
        txns = [{"action": "BUY", "ticker": "ETH-USD", "shares": 5.0}]
        state = _make_state(transactions=txns)
        _write_state(fake, state)
        with patch.object(portfolio_mgr, "STATE_FILE", fake):
            result = portfolio_mgr.load_state()
        assert len(result["transactions"]) == 1
        assert result["transactions"][0]["ticker"] == "ETH-USD"


# ===========================================================================
# save_state
# ===========================================================================

class TestSaveState:
    """save_state should persist state via atomic_write_json."""

    def test_creates_file(self, tmp_path):
        fake = tmp_path / "state.json"
        state = _make_state(cash=300_000)
        with patch.object(portfolio_mgr, "STATE_FILE", fake):
            portfolio_mgr.save_state(state)
        assert fake.exists()

    def test_file_content_is_valid_json(self, tmp_path):
        fake = tmp_path / "state.json"
        state = _make_state(cash=300_000)
        with patch.object(portfolio_mgr, "STATE_FILE", fake):
            portfolio_mgr.save_state(state)
        loaded = json.loads(fake.read_text(encoding="utf-8"))
        assert loaded["cash_sek"] == 300_000

    def test_atomic_write_called(self, tmp_path):
        fake = tmp_path / "state.json"
        state = _make_state()
        with patch.object(portfolio_mgr, "_atomic_write_json") as mock_aw:
            with patch.object(portfolio_mgr, "STATE_FILE", fake):
                portfolio_mgr.save_state(state)
            mock_aw.assert_called_once_with(fake, state)

    def test_round_trip(self, tmp_path):
        """save_state then load_state should return the same data."""
        fake = tmp_path / "state.json"
        holdings = {
            "NVDA": {"shares": 10.0, "avg_cost_usd": 180.0},
            "BTC-USD": {"shares": 0.25, "avg_cost_usd": 65000.0},
        }
        original = _make_state(cash=250_000, holdings=holdings, total_fees_sek=99.9)
        with patch.object(portfolio_mgr, "STATE_FILE", fake):
            portfolio_mgr.save_state(original)
            loaded = portfolio_mgr.load_state()
        assert loaded["cash_sek"] == original["cash_sek"]
        assert loaded["holdings"] == original["holdings"]
        assert loaded["total_fees_sek"] == original["total_fees_sek"]


# ===========================================================================
# portfolio_value
# ===========================================================================

class TestPortfolioValue:
    """portfolio_value should compute cash + sum(shares * price_usd * fx_rate)."""

    def test_cash_only_no_holdings(self):
        state = _make_state(cash=500_000)
        assert portfolio_mgr.portfolio_value(state, {}, 10.0) == 500_000

    def test_cash_only_empty_holdings_dict(self):
        state = _make_state(cash=100_000, holdings={})
        assert portfolio_mgr.portfolio_value(state, {"BTC-USD": 70000}, 10.0) == 100_000

    def test_single_holding(self):
        holdings = {"BTC-USD": {"shares": 1.0, "avg_cost_usd": 60000}}
        state = _make_state(cash=0, holdings=holdings)
        prices = {"BTC-USD": 70000}
        fx = 10.0
        # value = 0 + 1.0 * 70000 * 10.0 = 700_000
        assert portfolio_mgr.portfolio_value(state, prices, fx) == 700_000

    def test_multiple_holdings(self):
        holdings = {
            "BTC-USD": {"shares": 0.5, "avg_cost_usd": 60000},
            "ETH-USD": {"shares": 10.0, "avg_cost_usd": 2000},
        }
        state = _make_state(cash=100_000, holdings=holdings)
        prices = {"BTC-USD": 68000, "ETH-USD": 2500}
        fx = 10.5
        # BTC: 0.5 * 68000 * 10.5 = 357_000
        # ETH: 10.0 * 2500 * 10.5 = 262_500
        # total = 100_000 + 357_000 + 262_500 = 719_500
        expected = 100_000 + (0.5 * 68000 * 10.5) + (10.0 * 2500 * 10.5)
        assert portfolio_mgr.portfolio_value(state, prices, fx) == expected

    def test_ticker_not_in_prices_is_ignored(self):
        holdings = {
            "BTC-USD": {"shares": 1.0, "avg_cost_usd": 60000},
            "UNKNOWN": {"shares": 100.0, "avg_cost_usd": 50},
        }
        state = _make_state(cash=50_000, holdings=holdings)
        prices = {"BTC-USD": 70000}  # UNKNOWN not present
        fx = 10.0
        # Only BTC counted: 50_000 + 1.0 * 70000 * 10.0 = 750_000
        assert portfolio_mgr.portfolio_value(state, prices, fx) == 750_000

    def test_zero_shares_ignored(self):
        holdings = {"BTC-USD": {"shares": 0, "avg_cost_usd": 60000}}
        state = _make_state(cash=200_000, holdings=holdings)
        prices = {"BTC-USD": 70000}
        fx = 10.0
        # shares == 0 so holding is skipped
        assert portfolio_mgr.portfolio_value(state, prices, fx) == 200_000

    def test_fx_rate_multiplier(self):
        holdings = {"ETH-USD": {"shares": 2.0, "avg_cost_usd": 2000}}
        state = _make_state(cash=0, holdings=holdings)
        prices = {"ETH-USD": 3000}
        # fx_rate=1.0 => 2 * 3000 * 1.0 = 6000
        assert portfolio_mgr.portfolio_value(state, prices, 1.0) == 6000
        # fx_rate=11.0 => 2 * 3000 * 11.0 = 66000
        assert portfolio_mgr.portfolio_value(state, prices, 11.0) == 66_000

    def test_negative_cash_edge_case(self):
        """Negative cash (should not happen but handle gracefully)."""
        state = _make_state(cash=-10_000)
        prices = {}
        fx = 10.0
        assert portfolio_mgr.portfolio_value(state, prices, fx) == -10_000

    def test_negative_cash_with_holdings(self):
        holdings = {"BTC-USD": {"shares": 0.1, "avg_cost_usd": 60000}}
        state = _make_state(cash=-5_000, holdings=holdings)
        prices = {"BTC-USD": 100_000}
        fx = 10.0
        # -5_000 + 0.1 * 100_000 * 10.0 = -5_000 + 100_000 = 95_000
        assert portfolio_mgr.portfolio_value(state, prices, fx) == 95_000

    def test_missing_holdings_key_in_state(self):
        """State dict without 'holdings' key (uses .get default)."""
        state = {"cash_sek": 500_000}
        assert portfolio_mgr.portfolio_value(state, {"BTC-USD": 70000}, 10.0) == 500_000

    def test_fractional_shares(self):
        holdings = {"BTC-USD": {"shares": 0.00123, "avg_cost_usd": 60000}}
        state = _make_state(cash=0, holdings=holdings)
        prices = {"BTC-USD": 70000}
        fx = 10.85
        expected = 0.00123 * 70000 * 10.85
        assert portfolio_mgr.portfolio_value(state, prices, fx) == pytest.approx(expected)
