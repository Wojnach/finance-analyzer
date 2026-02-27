"""Tests for portfolio.warrant_portfolio — leverage-aware P&L tracking."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ============================================================
# warrant_pnl
# ============================================================

class TestWarrantPnl:
    """Tests for warrant_pnl()."""

    def test_basic_positive_pnl(self):
        from portfolio.warrant_portfolio import warrant_pnl
        holding = {
            "units": 100,
            "entry_price_sek": 54.50,
            "underlying": "XAG-USD",
            "leverage": 5,
            "underlying_entry_price_usd": 79.50,
        }
        # Underlying went from 79.50 to 89.50 = +12.58%
        result = warrant_pnl(holding, current_underlying_usd=89.50, fx_rate=10.50)

        assert result is not None
        assert result["underlying_change_pct"] == pytest.approx(12.58, abs=0.1)
        # Warrant P&L = 12.58% * 5 = 62.89%
        assert result["pnl_pct"] == pytest.approx(62.89, abs=1.0)
        assert result["pnl_sek"] > 0
        assert result["source"] == "implied"

    def test_plan_verification_case(self):
        """Plan says: entry at $79.50, current $89.50 → +12.6% spot → 5x = +63% implied."""
        from portfolio.warrant_portfolio import warrant_pnl
        holding = {
            "units": 100,
            "entry_price_sek": 54.50,
            "underlying": "XAG-USD",
            "leverage": 5,
            "underlying_entry_price_usd": 79.50,
        }
        result = warrant_pnl(holding, 89.50, 10.50)

        spot_change = (89.50 - 79.50) / 79.50
        assert spot_change == pytest.approx(0.1258, abs=0.001)
        assert result["pnl_pct"] == pytest.approx(spot_change * 5 * 100, abs=0.5)

    def test_negative_pnl(self):
        from portfolio.warrant_portfolio import warrant_pnl
        holding = {
            "units": 50,
            "entry_price_sek": 100.0,
            "leverage": 5,
            "underlying_entry_price_usd": 80.0,
        }
        # Underlying dropped from 80 to 76 = -5%
        result = warrant_pnl(holding, 76.0, 10.5)

        assert result["underlying_change_pct"] == pytest.approx(-5.0, abs=0.1)
        # Warrant: -5% * 5x = -25%
        assert result["pnl_pct"] == pytest.approx(-25.0, abs=0.5)
        assert result["pnl_sek"] < 0

    def test_leverage_1x(self):
        """1x leverage (XBT Tracker) should match underlying change."""
        from portfolio.warrant_portfolio import warrant_pnl
        holding = {
            "units": 10,
            "entry_price_sek": 200.0,
            "leverage": 1,
            "underlying_entry_price_usd": 67000,
        }
        # BTC: 67000 → 70000 = +4.48%
        result = warrant_pnl(holding, 70000, 10.5)

        assert result["pnl_pct"] == pytest.approx(4.48, abs=0.1)

    def test_zero_leverage(self):
        from portfolio.warrant_portfolio import warrant_pnl
        holding = {
            "units": 10, "entry_price_sek": 100.0,
            "leverage": 0, "underlying_entry_price_usd": 80.0,
        }
        result = warrant_pnl(holding, 90.0, 10.5)
        # 0 leverage means 0% P&L regardless of underlying
        assert result["pnl_pct"] == pytest.approx(0.0)

    def test_missing_holding(self):
        from portfolio.warrant_portfolio import warrant_pnl
        assert warrant_pnl(None, 89.5, 10.5) is None

    def test_missing_price(self):
        from portfolio.warrant_portfolio import warrant_pnl
        holding = {"units": 10, "entry_price_sek": 50, "leverage": 5, "underlying_entry_price_usd": 80}
        assert warrant_pnl(holding, None, 10.5) is None
        assert warrant_pnl(holding, 89.5, None) is None

    def test_missing_underlying_entry(self):
        from portfolio.warrant_portfolio import warrant_pnl
        holding = {"units": 10, "entry_price_sek": 50, "leverage": 5}
        assert warrant_pnl(holding, 89.5, 10.5) is None

    def test_zero_units(self):
        from portfolio.warrant_portfolio import warrant_pnl
        holding = {"units": 0, "entry_price_sek": 50, "leverage": 5, "underlying_entry_price_usd": 80}
        assert warrant_pnl(holding, 89.5, 10.5) is None

    def test_entry_value_and_total_value(self):
        from portfolio.warrant_portfolio import warrant_pnl
        holding = {
            "units": 100, "entry_price_sek": 50.0,
            "leverage": 5, "underlying_entry_price_usd": 80.0,
        }
        result = warrant_pnl(holding, 80.0, 10.5)  # no change

        assert result["entry_value_sek"] == pytest.approx(5000.0)
        assert result["total_value_sek"] == pytest.approx(5000.0)
        assert result["pnl_sek"] == pytest.approx(0.0, abs=0.01)


# ============================================================
# load_warrant_state / save_warrant_state
# ============================================================

class TestWarrantState:
    """Tests for load/save warrant state."""

    def test_load_default_when_missing(self, tmp_path):
        from portfolio.warrant_portfolio import load_warrant_state
        with patch("portfolio.warrant_portfolio.WARRANT_STATE_FILE", tmp_path / "missing.json"):
            state = load_warrant_state()
        assert state == {"holdings": {}, "transactions": []}

    def test_load_existing(self, tmp_path):
        from portfolio.warrant_portfolio import load_warrant_state
        state_file = tmp_path / "warrants.json"
        state_file.write_text(json.dumps({
            "holdings": {"MINI-SILVER": {"units": 100}},
            "transactions": [{"action": "BUY"}],
        }))
        with patch("portfolio.warrant_portfolio.WARRANT_STATE_FILE", state_file):
            state = load_warrant_state()
        assert state["holdings"]["MINI-SILVER"]["units"] == 100
        assert len(state["transactions"]) == 1

    def test_save_and_load(self, tmp_path):
        from portfolio.warrant_portfolio import load_warrant_state, save_warrant_state
        state_file = tmp_path / "warrants.json"
        with patch("portfolio.warrant_portfolio.WARRANT_STATE_FILE", state_file):
            save_warrant_state({
                "holdings": {"MINI-SILVER": {"units": 50}},
                "transactions": [],
            })
            loaded = load_warrant_state()
        assert loaded["holdings"]["MINI-SILVER"]["units"] == 50


# ============================================================
# get_warrant_summary
# ============================================================

class TestGetWarrantSummary:
    """Tests for get_warrant_summary()."""

    @patch("portfolio.warrant_portfolio.load_warrant_state")
    def test_empty_holdings(self, mock_load):
        from portfolio.warrant_portfolio import get_warrant_summary
        mock_load.return_value = {"holdings": {}, "transactions": []}

        result = get_warrant_summary({"XAG-USD": 89.5}, 10.5)
        assert result["positions"] == {}
        assert result["total_value_sek"] == 0
        assert result["total_pnl_sek"] == 0

    @patch("portfolio.warrant_portfolio.load_warrant_state")
    def test_single_position(self, mock_load):
        from portfolio.warrant_portfolio import get_warrant_summary
        mock_load.return_value = {
            "holdings": {
                "MINI-SILVER": {
                    "units": 100, "entry_price_sek": 54.50,
                    "underlying": "XAG-USD", "leverage": 5,
                    "underlying_entry_price_usd": 79.50,
                    "name": "MINI L SILVER AVA 140",
                }
            },
            "transactions": [],
        }

        result = get_warrant_summary({"XAG-USD": 89.50}, 10.50)
        assert "MINI-SILVER" in result["positions"]
        pos = result["positions"]["MINI-SILVER"]
        assert pos["leverage"] == 5
        assert pos["pnl"] is not None
        assert pos["pnl"]["pnl_sek"] > 0
        assert result["total_pnl_sek"] > 0

    @patch("portfolio.warrant_portfolio.load_warrant_state")
    def test_missing_underlying_price(self, mock_load):
        from portfolio.warrant_portfolio import get_warrant_summary
        mock_load.return_value = {
            "holdings": {
                "MINI-SILVER": {
                    "units": 100, "entry_price_sek": 54.50,
                    "underlying": "XAG-USD", "leverage": 5,
                    "underlying_entry_price_usd": 79.50,
                }
            },
            "transactions": [],
        }

        # No XAG-USD in prices
        result = get_warrant_summary({"BTC-USD": 67000}, 10.50)
        assert result["total_value_sek"] == 0

    @patch("portfolio.warrant_portfolio.load_warrant_state")
    def test_multiple_positions(self, mock_load):
        from portfolio.warrant_portfolio import get_warrant_summary
        mock_load.return_value = {
            "holdings": {
                "MINI-SILVER": {
                    "units": 100, "entry_price_sek": 54.50,
                    "underlying": "XAG-USD", "leverage": 5,
                    "underlying_entry_price_usd": 79.50,
                },
                "XBT-TRACKER": {
                    "units": 5, "entry_price_sek": 200.0,
                    "underlying": "BTC-USD", "leverage": 1,
                    "underlying_entry_price_usd": 65000,
                },
            },
            "transactions": [],
        }

        result = get_warrant_summary({"XAG-USD": 89.50, "BTC-USD": 67000}, 10.50)
        assert len(result["positions"]) == 2
        assert result["total_pnl_sek"] > 0


# ============================================================
# record_warrant_transaction
# ============================================================

class TestRecordWarrantTransaction:
    """Tests for record_warrant_transaction()."""

    @patch("portfolio.warrant_portfolio.save_warrant_state")
    @patch("portfolio.warrant_portfolio.load_warrant_state")
    def test_buy_new_position(self, mock_load, mock_save):
        from portfolio.warrant_portfolio import record_warrant_transaction
        mock_load.return_value = {"holdings": {}, "transactions": []}

        record_warrant_transaction(
            "MINI-SILVER", "BUY", 100, 54.50, 79.50, 5,
            name="MINI L SILVER AVA 140", underlying="XAG-USD",
        )

        mock_save.assert_called_once()
        state = mock_save.call_args[0][0]
        assert "MINI-SILVER" in state["holdings"]
        assert state["holdings"]["MINI-SILVER"]["units"] == 100
        assert state["holdings"]["MINI-SILVER"]["entry_price_sek"] == 54.50
        assert len(state["transactions"]) == 1

    @patch("portfolio.warrant_portfolio.save_warrant_state")
    @patch("portfolio.warrant_portfolio.load_warrant_state")
    def test_buy_average_in(self, mock_load, mock_save):
        from portfolio.warrant_portfolio import record_warrant_transaction
        mock_load.return_value = {
            "holdings": {
                "MINI-SILVER": {
                    "units": 100, "entry_price_sek": 50.0,
                    "underlying": "XAG-USD", "leverage": 5,
                    "underlying_entry_price_usd": 79.50,
                }
            },
            "transactions": [],
        }

        record_warrant_transaction("MINI-SILVER", "BUY", 100, 60.0, 85.0, 5)

        state = mock_save.call_args[0][0]
        assert state["holdings"]["MINI-SILVER"]["units"] == 200
        # Avg price: (100*50 + 100*60) / 200 = 55
        assert state["holdings"]["MINI-SILVER"]["entry_price_sek"] == pytest.approx(55.0)

    @patch("portfolio.warrant_portfolio.save_warrant_state")
    @patch("portfolio.warrant_portfolio.load_warrant_state")
    def test_sell_partial(self, mock_load, mock_save):
        from portfolio.warrant_portfolio import record_warrant_transaction
        mock_load.return_value = {
            "holdings": {
                "MINI-SILVER": {
                    "units": 100, "entry_price_sek": 50.0,
                    "underlying": "XAG-USD", "leverage": 5,
                    "underlying_entry_price_usd": 79.50,
                }
            },
            "transactions": [],
        }

        record_warrant_transaction("MINI-SILVER", "SELL", 50, 70.0, 89.50, 5)

        state = mock_save.call_args[0][0]
        assert state["holdings"]["MINI-SILVER"]["units"] == 50

    @patch("portfolio.warrant_portfolio.save_warrant_state")
    @patch("portfolio.warrant_portfolio.load_warrant_state")
    def test_sell_full_removes(self, mock_load, mock_save):
        from portfolio.warrant_portfolio import record_warrant_transaction
        mock_load.return_value = {
            "holdings": {
                "MINI-SILVER": {
                    "units": 100, "entry_price_sek": 50.0,
                    "underlying": "XAG-USD", "leverage": 5,
                    "underlying_entry_price_usd": 79.50,
                }
            },
            "transactions": [],
        }

        record_warrant_transaction("MINI-SILVER", "SELL", 100, 70.0, 89.50, 5)

        state = mock_save.call_args[0][0]
        assert "MINI-SILVER" not in state["holdings"]

    @patch("portfolio.warrant_portfolio.save_warrant_state")
    @patch("portfolio.warrant_portfolio.load_warrant_state")
    def test_transaction_logged(self, mock_load, mock_save):
        from portfolio.warrant_portfolio import record_warrant_transaction
        mock_load.return_value = {"holdings": {}, "transactions": []}

        record_warrant_transaction("MINI-SILVER", "BUY", 100, 54.50, 79.50, 5,
                                    name="MINI L SILVER", underlying="XAG-USD")

        state = mock_save.call_args[0][0]
        txn = state["transactions"][0]
        assert txn["config_key"] == "MINI-SILVER"
        assert txn["action"] == "BUY"
        assert txn["units"] == 100
        assert txn["price_sek"] == 54.50
        assert txn["name"] == "MINI L SILVER"
        assert txn["underlying"] == "XAG-USD"
        assert "timestamp" in txn
