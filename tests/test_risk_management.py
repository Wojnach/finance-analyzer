"""Tests for portfolio/risk_management.py.

Covers: check_drawdown, compute_stop_levels, get_position_ages,
_compute_portfolio_value, transaction_cost_analysis.
"""

import datetime
import json
import pathlib

import pytest

from portfolio.risk_management import (
    _compute_portfolio_value,
    check_drawdown,
    compute_stop_levels,
    get_position_ages,
    transaction_cost_analysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: pathlib.Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_jsonl(path: pathlib.Path, entries: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _make_portfolio(cash=500_000, holdings=None, transactions=None,
                    initial_value=500_000, total_fees=0):
    return {
        "cash_sek": cash,
        "holdings": holdings or {},
        "transactions": transactions or [],
        "initial_value_sek": initial_value,
        "total_fees_sek": total_fees,
    }


def _make_summary(signals=None, fx_rate=10.0):
    return {
        "signals": signals or {},
        "fx_rate": fx_rate,
    }


# ===================================================================
# _compute_portfolio_value
# ===================================================================

class TestComputePortfolioValue:
    def test_cash_only(self):
        pf = _make_portfolio(cash=250_000)
        summary = _make_summary()
        assert _compute_portfolio_value(pf, summary) == 250_000

    def test_with_holdings_live_price(self):
        pf = _make_portfolio(cash=100_000, holdings={
            "BTC-USD": {"shares": 2, "avg_cost_usd": 50_000},
        })
        summary = _make_summary(
            signals={"BTC-USD": {"price_usd": 60_000}},
            fx_rate=10.0,
        )
        # 100_000 + 2 * 60_000 * 10 = 100_000 + 1_200_000 = 1_300_000
        assert _compute_portfolio_value(pf, summary) == 1_300_000

    def test_fallback_to_avg_cost(self):
        """When ticker not in signals, use avg_cost_usd from holdings."""
        pf = _make_portfolio(cash=50_000, holdings={
            "XYZ": {"shares": 10, "avg_cost_usd": 100},
        })
        summary = _make_summary(fx_rate=10.0)
        # 50_000 + 10 * 100 * 10 = 50_000 + 10_000 = 60_000
        assert _compute_portfolio_value(pf, summary) == 60_000

    def test_zero_shares_ignored(self):
        pf = _make_portfolio(cash=100_000, holdings={
            "BTC-USD": {"shares": 0, "avg_cost_usd": 60_000},
        })
        summary = _make_summary(
            signals={"BTC-USD": {"price_usd": 70_000}},
            fx_rate=10.0,
        )
        assert _compute_portfolio_value(pf, summary) == 100_000

    def test_negative_shares_ignored(self):
        pf = _make_portfolio(cash=100_000, holdings={
            "BTC-USD": {"shares": -5, "avg_cost_usd": 60_000},
        })
        summary = _make_summary()
        assert _compute_portfolio_value(pf, summary) == 100_000


# ===================================================================
# check_drawdown
# ===================================================================

class TestCheckDrawdown:
    def test_no_drawdown(self, tmp_path, monkeypatch):
        """Full cash at initial value -> 0% drawdown."""
        pf_path = tmp_path / "portfolio_state.json"
        _write_json(pf_path, _make_portfolio(cash=500_000))
        summary_path = tmp_path / "agent_summary.json"
        _write_json(summary_path, _make_summary())

        # Patch DATA_DIR so it won't find history file
        monkeypatch.setattr("portfolio.risk_management.DATA_DIR", tmp_path)

        result = check_drawdown(str(pf_path), agent_summary_path=str(summary_path))
        assert result["breached"] is False
        assert result["current_drawdown_pct"] == 0.0
        assert result["peak_value"] == 500_000
        assert result["current_value"] == 500_000

    def test_drawdown_breached(self, tmp_path, monkeypatch):
        """Cash dropped 25% -> exceeds 20% threshold."""
        pf_path = tmp_path / "portfolio_state.json"
        _write_json(pf_path, _make_portfolio(cash=375_000))  # 25% drawdown
        summary_path = tmp_path / "agent_summary.json"
        _write_json(summary_path, _make_summary())

        monkeypatch.setattr("portfolio.risk_management.DATA_DIR", tmp_path)

        result = check_drawdown(str(pf_path), max_drawdown_pct=20.0,
                                agent_summary_path=str(summary_path))
        assert result["breached"] is True
        assert result["current_drawdown_pct"] == 25.0
        assert result["peak_value"] == 500_000

    def test_custom_threshold(self, tmp_path, monkeypatch):
        """Custom threshold of 10%: 15% drawdown -> breached."""
        pf_path = tmp_path / "portfolio_state.json"
        _write_json(pf_path, _make_portfolio(cash=425_000))  # 15% drawdown
        monkeypatch.setattr("portfolio.risk_management.DATA_DIR", tmp_path)

        result = check_drawdown(str(pf_path), max_drawdown_pct=10.0,
                                agent_summary_path=str(tmp_path / "nonexistent.json"))
        assert result["breached"] is True

    def test_drawdown_not_breached_within_threshold(self, tmp_path, monkeypatch):
        """5% drawdown, 20% threshold -> not breached."""
        pf_path = tmp_path / "portfolio_state.json"
        _write_json(pf_path, _make_portfolio(cash=475_000))
        monkeypatch.setattr("portfolio.risk_management.DATA_DIR", tmp_path)

        result = check_drawdown(str(pf_path), max_drawdown_pct=20.0,
                                agent_summary_path=str(tmp_path / "nonexistent.json"))
        assert result["breached"] is False
        assert result["current_drawdown_pct"] == 5.0

    def test_peak_from_history_file(self, tmp_path, monkeypatch):
        """Peak tracked from portfolio_value_history.jsonl."""
        monkeypatch.setattr("portfolio.risk_management.DATA_DIR", tmp_path)

        pf_path = tmp_path / "portfolio_state.json"
        _write_json(pf_path, _make_portfolio(cash=480_000))

        # Write history with a previous peak of 550K
        _write_jsonl(tmp_path / "portfolio_value_history.jsonl", [
            {"patient_value_sek": 510_000, "bold_value_sek": 520_000},
            {"patient_value_sek": 550_000, "bold_value_sek": 530_000},
            {"patient_value_sek": 500_000, "bold_value_sek": 510_000},
        ])

        result = check_drawdown(str(pf_path),
                                agent_summary_path=str(tmp_path / "nonexistent.json"))
        # Peak should be 550K from history (patient, since filename is 'portfolio_state')
        assert result["peak_value"] == 550_000
        # Drawdown: (550000 - 480000) / 550000 * 100 ~= 12.7273%
        assert abs(result["current_drawdown_pct"] - 12.7273) < 0.01

    def test_bold_portfolio_uses_bold_key(self, tmp_path, monkeypatch):
        """Bold portfolio reads bold_value_sek from history."""
        monkeypatch.setattr("portfolio.risk_management.DATA_DIR", tmp_path)

        pf_path = tmp_path / "portfolio_state_bold.json"
        _write_json(pf_path, _make_portfolio(cash=400_000))

        _write_jsonl(tmp_path / "portfolio_value_history.jsonl", [
            {"patient_value_sek": 510_000, "bold_value_sek": 600_000},
        ])

        result = check_drawdown(str(pf_path),
                                agent_summary_path=str(tmp_path / "nonexistent.json"))
        # Peak should be 600K from bold history
        assert result["peak_value"] == 600_000
        # Drawdown: (600000 - 400000) / 600000 * 100 = 33.33%
        assert abs(result["current_drawdown_pct"] - 33.3333) < 0.01

    def test_current_value_new_peak(self, tmp_path, monkeypatch):
        """If current value is above historical peak, peak updates."""
        monkeypatch.setattr("portfolio.risk_management.DATA_DIR", tmp_path)

        pf_path = tmp_path / "portfolio_state.json"
        _write_json(pf_path, _make_portfolio(cash=600_000))

        result = check_drawdown(str(pf_path),
                                agent_summary_path=str(tmp_path / "nonexistent.json"))
        assert result["peak_value"] == 600_000
        assert result["current_drawdown_pct"] == 0.0

    def test_empty_holdings_uses_cash(self, tmp_path, monkeypatch):
        """Portfolio with empty holdings -> value = cash only."""
        monkeypatch.setattr("portfolio.risk_management.DATA_DIR", tmp_path)

        pf_path = tmp_path / "portfolio_state.json"
        _write_json(pf_path, _make_portfolio(cash=450_000, holdings={}))

        result = check_drawdown(str(pf_path),
                                agent_summary_path=str(tmp_path / "nonexistent.json"))
        assert result["current_value"] == 450_000

    def test_missing_agent_summary_falls_back_to_cash(self, tmp_path, monkeypatch):
        """When agent_summary.json is missing, falls back to cash for value."""
        monkeypatch.setattr("portfolio.risk_management.DATA_DIR", tmp_path)

        pf_path = tmp_path / "portfolio_state.json"
        _write_json(pf_path, _make_portfolio(
            cash=300_000,
            holdings={"BTC-USD": {"shares": 1, "avg_cost_usd": 50_000}},
        ))

        result = check_drawdown(str(pf_path),
                                agent_summary_path=str(tmp_path / "nonexistent.json"))
        # Falls back to cash only
        assert result["current_value"] == 300_000


# ===================================================================
# compute_stop_levels
# ===================================================================

class TestComputeStopLevels:
    def test_basic_stop_level(self):
        """2x ATR stop below entry price."""
        holdings = {
            "BTC-USD": {"shares": 1, "avg_cost_usd": 60_000},
        }
        summary = _make_summary(signals={
            "BTC-USD": {"price_usd": 58_000, "atr_pct": 3.0},
        })

        result = compute_stop_levels(holdings, summary)
        assert "BTC-USD" in result
        r = result["BTC-USD"]
        # stop = 60000 * (1 - 2 * 3.0 / 100) = 60000 * 0.94 = 56400
        assert r["stop_price_usd"] == 56_400.0
        assert r["triggered"] is False  # 58000 > 56400
        assert r["entry_price_usd"] == 60_000.0
        assert r["current_price_usd"] == 58_000.0
        assert r["atr_pct"] == 3.0

    def test_stop_triggered(self):
        """Price dropped below 2x ATR stop -> triggered."""
        holdings = {
            "BTC-USD": {"shares": 1, "avg_cost_usd": 60_000},
        }
        summary = _make_summary(signals={
            "BTC-USD": {"price_usd": 55_000, "atr_pct": 3.0},
        })

        result = compute_stop_levels(holdings, summary)
        r = result["BTC-USD"]
        # stop = 56400, price = 55000 < 56400
        assert r["triggered"] is True

    def test_pnl_calculation(self):
        """PnL percentage computed correctly."""
        holdings = {
            "ETH-USD": {"shares": 5, "avg_cost_usd": 2000},
        }
        summary = _make_summary(signals={
            "ETH-USD": {"price_usd": 2200, "atr_pct": 4.0},
        })

        result = compute_stop_levels(holdings, summary)
        r = result["ETH-USD"]
        # PnL: (2200 - 2000) / 2000 * 100 = 10.0%
        assert r["pnl_pct"] == 10.0

    def test_ticker_not_in_summary(self):
        """Ticker not in agent_summary -> note and None values."""
        holdings = {
            "MSTR": {"shares": 10, "avg_cost_usd": 130},
        }
        summary = _make_summary(signals={})

        result = compute_stop_levels(holdings, summary)
        r = result["MSTR"]
        assert r["current_price_usd"] is None
        assert r["stop_price_usd"] is None
        assert r["triggered"] is False
        assert "note" in r

    def test_zero_shares_skipped(self):
        """Holdings with 0 shares should be excluded."""
        holdings = {
            "BTC-USD": {"shares": 0, "avg_cost_usd": 60_000},
        }
        summary = _make_summary()
        result = compute_stop_levels(holdings, summary)
        assert result == {}

    def test_zero_entry_price_skipped(self):
        """Holdings with 0 avg_cost_usd should be excluded."""
        holdings = {
            "BTC-USD": {"shares": 1, "avg_cost_usd": 0},
        }
        summary = _make_summary()
        result = compute_stop_levels(holdings, summary)
        assert result == {}

    def test_empty_holdings(self):
        """Empty holdings -> empty result."""
        result = compute_stop_levels({}, _make_summary())
        assert result == {}

    def test_distance_to_stop(self):
        """Distance to stop expressed as percentage."""
        holdings = {
            "ETH-USD": {"shares": 1, "avg_cost_usd": 2000},
        }
        summary = _make_summary(signals={
            "ETH-USD": {"price_usd": 2000, "atr_pct": 5.0},
        })

        result = compute_stop_levels(holdings, summary)
        r = result["ETH-USD"]
        # stop = 2000 * (1 - 2*5/100) = 2000 * 0.90 = 1800
        assert r["stop_price_usd"] == 1800.0
        # distance = (2000 - 1800) / 1800 * 100 = 11.1111%
        assert abs(r["distance_to_stop_pct"] - 11.1111) < 0.01

    def test_multiple_holdings(self):
        """Multiple holdings each get their own stop level."""
        holdings = {
            "BTC-USD": {"shares": 1, "avg_cost_usd": 60_000},
            "ETH-USD": {"shares": 10, "avg_cost_usd": 2000},
        }
        summary = _make_summary(signals={
            "BTC-USD": {"price_usd": 62_000, "atr_pct": 3.0},
            "ETH-USD": {"price_usd": 1900, "atr_pct": 5.0},
        })

        result = compute_stop_levels(holdings, summary)
        assert "BTC-USD" in result
        assert "ETH-USD" in result
        assert result["BTC-USD"]["triggered"] is False
        # ETH stop = 2000 * (1 - 0.10) = 1800. price=1900 > 1800 -> NOT triggered
        assert result["ETH-USD"]["triggered"] is False


# ===================================================================
# get_position_ages
# ===================================================================

class TestGetPositionAges:
    def test_basic_age(self):
        """Calculate age from first BUY timestamp."""
        buy_ts = (datetime.datetime.now(datetime.timezone.utc) -
                  datetime.timedelta(hours=48)).isoformat()
        portfolio = _make_portfolio(
            holdings={"BTC-USD": {"shares": 1}},
            transactions=[
                {"ticker": "BTC-USD", "action": "BUY", "timestamp": buy_ts},
            ],
        )

        result = get_position_ages(portfolio)
        assert "BTC-USD" in result
        assert abs(result["BTC-USD"]["age_hours"] - 48) < 1  # within 1 hour tolerance
        assert abs(result["BTC-USD"]["age_days"] - 2) < 0.1
        assert result["BTC-USD"]["num_buys"] == 1
        assert result["BTC-USD"]["num_sells"] == 0

    def test_multiple_buys_uses_earliest(self):
        """Multiple BUY transactions -> age from the earliest one."""
        ts_old = (datetime.datetime.now(datetime.timezone.utc) -
                  datetime.timedelta(hours=100)).isoformat()
        ts_new = (datetime.datetime.now(datetime.timezone.utc) -
                  datetime.timedelta(hours=10)).isoformat()

        portfolio = _make_portfolio(
            holdings={"ETH-USD": {"shares": 5}},
            transactions=[
                {"ticker": "ETH-USD", "action": "BUY", "timestamp": ts_new},
                {"ticker": "ETH-USD", "action": "BUY", "timestamp": ts_old},
            ],
        )

        result = get_position_ages(portfolio)
        # Should use the older timestamp
        assert abs(result["ETH-USD"]["age_hours"] - 100) < 1
        assert result["ETH-USD"]["num_buys"] == 2

    def test_counts_sells(self):
        """SELL transactions are counted."""
        buy_ts = (datetime.datetime.now(datetime.timezone.utc) -
                  datetime.timedelta(hours=72)).isoformat()
        sell_ts = (datetime.datetime.now(datetime.timezone.utc) -
                   datetime.timedelta(hours=24)).isoformat()

        portfolio = _make_portfolio(
            holdings={"BTC-USD": {"shares": 0.5}},
            transactions=[
                {"ticker": "BTC-USD", "action": "BUY", "timestamp": buy_ts},
                {"ticker": "BTC-USD", "action": "SELL", "timestamp": sell_ts},
            ],
        )

        result = get_position_ages(portfolio)
        assert result["BTC-USD"]["num_buys"] == 1
        assert result["BTC-USD"]["num_sells"] == 1

    def test_zero_shares_excluded(self):
        """Holdings with 0 shares should not appear in result."""
        portfolio = _make_portfolio(
            holdings={"BTC-USD": {"shares": 0}},
            transactions=[
                {"ticker": "BTC-USD", "action": "BUY",
                 "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()},
            ],
        )

        result = get_position_ages(portfolio)
        assert "BTC-USD" not in result

    def test_empty_holdings(self):
        """Empty holdings -> empty result."""
        portfolio = _make_portfolio(holdings={})
        result = get_position_ages(portfolio)
        assert result == {}

    def test_empty_transactions(self):
        """Holdings exist but no transactions -> not included (no first_buy_ts)."""
        portfolio = _make_portfolio(
            holdings={"BTC-USD": {"shares": 1}},
            transactions=[],
        )
        result = get_position_ages(portfolio)
        assert result == {}

    def test_no_buy_for_ticker(self):
        """Transactions exist but none are BUY for the held ticker."""
        portfolio = _make_portfolio(
            holdings={"BTC-USD": {"shares": 1}},
            transactions=[
                {"ticker": "ETH-USD", "action": "BUY",
                 "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()},
            ],
        )
        result = get_position_ages(portfolio)
        assert "BTC-USD" not in result

    def test_naive_timestamp_treated_as_utc(self):
        """Naive (no timezone) timestamps should be treated as UTC."""
        # Use a naive timestamp (no +00:00)
        naive_ts = (datetime.datetime.now(datetime.timezone.utc) -
                    datetime.timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")
        portfolio = _make_portfolio(
            holdings={"BTC-USD": {"shares": 1}},
            transactions=[
                {"ticker": "BTC-USD", "action": "BUY", "timestamp": naive_ts},
            ],
        )

        result = get_position_ages(portfolio)
        assert "BTC-USD" in result
        assert abs(result["BTC-USD"]["age_hours"] - 24) < 1


# ===================================================================
# transaction_cost_analysis
# ===================================================================

class TestTransactionCostAnalysis:
    def test_no_transactions(self):
        """Portfolio with no transactions -> zero costs."""
        pf = _make_portfolio()
        result = transaction_cost_analysis(pf)
        assert result["total_fees_sek"] == 0
        assert result["total_trades"] == 0
        assert result["buy_count"] == 0
        assert result["sell_count"] == 0
        assert result["fees_as_pct_of_pnl"] is None

    def test_basic_fees(self):
        """Accumulated fees from portfolio state and transactions."""
        pf = _make_portfolio(
            cash=490_000,
            total_fees=100,
            transactions=[
                {"action": "BUY", "fee_sek": 50, "total_sek": 75_000, "ticker": "BTC-USD"},
                {"action": "SELL", "fee_sek": 50, "total_sek": 70_000, "ticker": "BTC-USD"},
            ],
        )

        result = transaction_cost_analysis(pf)
        assert result["total_fees_sek"] == 100  # max(100, 100)
        assert result["total_trades"] == 2
        assert result["buy_count"] == 1
        assert result["sell_count"] == 1
        assert result["avg_fee_per_trade"] == 50
        assert result["total_buy_volume_sek"] == 75_000
        assert result["total_sell_volume_sek"] == 70_000

    def test_fees_as_pct_of_initial(self):
        """Fee percentage relative to initial portfolio value."""
        pf = _make_portfolio(
            cash=495_000,
            initial_value=500_000,
            total_fees=500,
            transactions=[
                {"action": "BUY", "fee_sek": 500, "total_sek": 5000, "ticker": "X"},
            ],
        )
        result = transaction_cost_analysis(pf)
        # 500 / 500000 * 100 = 0.1%
        assert result["fees_as_pct_of_initial"] == 0.1

    def test_fees_as_pct_of_pnl(self):
        """Fee as percentage of absolute P&L."""
        pf = _make_portfolio(
            cash=490_000,
            initial_value=500_000,
            total_fees=200,
            transactions=[
                {"action": "BUY", "fee_sek": 100, "total_sek": 10_000, "ticker": "X"},
                {"action": "SELL", "fee_sek": 100, "total_sek": 9_000, "ticker": "X"},
            ],
        )
        result = transaction_cost_analysis(pf)
        # PnL = 490000 - 500000 = -10000
        # fees_pct_of_pnl = 200 / 10000 * 100 = 2.0%
        assert result["fees_as_pct_of_pnl"] == 2.0

    def test_null_total_fees_treated_as_zero(self):
        """None total_fees_sek in portfolio state -> treated as 0."""
        pf = _make_portfolio(cash=500_000)
        pf["total_fees_sek"] = None
        result = transaction_cost_analysis(pf)
        assert result["total_fees_sek"] == 0

    def test_open_positions_pnl_note(self):
        """With open positions, pnl_note indicates approximation."""
        pf = _make_portfolio(
            cash=400_000,
            holdings={"BTC-USD": {"shares": 1}},
            transactions=[
                {"action": "BUY", "fee_sek": 50, "total_sek": 100_000, "ticker": "BTC-USD"},
            ],
        )
        result = transaction_cost_analysis(pf)
        assert "approximate" in result["pnl_note"]

    def test_closed_positions_pnl_note(self):
        """All positions closed -> exact PnL note."""
        pf = _make_portfolio(
            cash=495_000,
            holdings={"BTC-USD": {"shares": 0}},
            transactions=[
                {"action": "BUY", "fee_sek": 50, "total_sek": 100_000, "ticker": "BTC-USD"},
                {"action": "SELL", "fee_sek": 50, "total_sek": 95_000, "ticker": "BTC-USD"},
            ],
        )
        result = transaction_cost_analysis(pf)
        assert "exact" in result["pnl_note"]
