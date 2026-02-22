"""Tests for portfolio/kelly_sizing.py.

Covers: kelly_fraction, _compute_trade_stats, _get_signal_accuracy,
_get_ticker_signal_accuracy, recommended_size.
"""

import json
import pathlib

import pytest

from portfolio.kelly_sizing import (
    kelly_fraction,
    _compute_trade_stats,
    _get_signal_accuracy,
    _get_ticker_signal_accuracy,
    recommended_size,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: pathlib.Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_portfolio(cash=500_000, holdings=None, transactions=None,
                    initial_value=500_000):
    return {
        "cash_sek": cash,
        "holdings": holdings or {},
        "transactions": transactions or [],
        "initial_value_sek": initial_value,
    }


# ===================================================================
# kelly_fraction
# ===================================================================

class TestKellyFraction:
    def test_positive_edge(self):
        """60% win rate, avg win = avg loss -> Kelly = 0.2."""
        # f* = (p*b - q) / b where b=1, p=0.6, q=0.4
        # f* = (0.6 - 0.4) / 1 = 0.2
        result = kelly_fraction(0.6, 2.0, 2.0)
        assert abs(result - 0.2) < 1e-10

    def test_no_edge(self):
        """50% win rate, equal win/loss -> Kelly = 0."""
        # f* = (0.5*1 - 0.5) / 1 = 0.0
        result = kelly_fraction(0.5, 1.0, 1.0)
        assert result == 0.0

    def test_negative_edge(self):
        """30% win rate, equal win/loss -> negative edge -> clamped to 0."""
        result = kelly_fraction(0.3, 1.0, 1.0)
        assert result == 0.0

    def test_high_win_ratio(self):
        """Win/loss ratio amplifies edge."""
        # p=0.5, b=avg_win/avg_loss = 3/1 = 3
        # f* = (0.5*3 - 0.5) / 3 = (1.5 - 0.5) / 3 = 1/3
        result = kelly_fraction(0.5, 3.0, 1.0)
        assert abs(result - 1.0 / 3.0) < 1e-10

    def test_zero_win_prob(self):
        """0% win probability -> returns 0."""
        assert kelly_fraction(0.0, 2.0, 1.0) == 0.0

    def test_100_pct_win_prob(self):
        """100% win probability -> returns 0 (boundary excluded)."""
        assert kelly_fraction(1.0, 2.0, 1.0) == 0.0

    def test_zero_avg_win(self):
        """Zero average win -> returns 0."""
        assert kelly_fraction(0.6, 0.0, 1.0) == 0.0

    def test_zero_avg_loss(self):
        """Zero average loss -> returns 0."""
        assert kelly_fraction(0.6, 2.0, 0.0) == 0.0

    def test_negative_avg_win(self):
        """Negative average win -> returns 0."""
        assert kelly_fraction(0.6, -1.0, 1.0) == 0.0

    def test_negative_avg_loss(self):
        """Negative average loss -> returns 0."""
        assert kelly_fraction(0.6, 1.0, -1.0) == 0.0

    def test_clamped_to_max_1(self):
        """Very high edge doesn't exceed 1.0."""
        # p=0.99, b=100/0.01=10000
        # f* = (0.99*10000 - 0.01) / 10000 ~ 0.99 -> should be <=1
        result = kelly_fraction(0.99, 100.0, 0.01)
        assert result <= 1.0

    def test_near_boundary_win_prob(self):
        """Win probability very close to 0 or 1."""
        assert kelly_fraction(0.001, 2.0, 1.0) == 0.0  # Tiny win prob, negative edge
        result = kelly_fraction(0.999, 2.0, 1.0)
        assert 0 < result <= 1.0

    def test_typical_crypto_scenario(self):
        """Typical crypto: 55% win rate, avg win 4%, avg loss 3%."""
        # b = 4/3, p=0.55, q=0.45
        # f* = (0.55 * 4/3 - 0.45) / (4/3)
        #    = (0.7333 - 0.45) / 1.3333
        #    = 0.2833 / 1.3333 = 0.2125
        result = kelly_fraction(0.55, 4.0, 3.0)
        assert abs(result - 0.2125) < 0.001


# ===================================================================
# _compute_trade_stats
# ===================================================================

class TestComputeTradeStats:
    def test_insufficient_data(self):
        """Fewer than 2 round-trips -> returns None."""
        txs = [
            {"ticker": "BTC-USD", "action": "BUY", "shares": 1, "total_sek": 100_000},
            {"ticker": "BTC-USD", "action": "SELL", "shares": 1, "total_sek": 110_000},
        ]
        result = _compute_trade_stats(txs)
        assert result is None  # Only 1 round-trip

    def test_two_round_trips(self):
        """Two round-trips give valid stats."""
        txs = [
            {"ticker": "BTC-USD", "action": "BUY", "shares": 1, "total_sek": 100_000},
            {"ticker": "BTC-USD", "action": "SELL", "shares": 1, "total_sek": 110_000},
            {"ticker": "BTC-USD", "action": "SELL", "shares": 1, "total_sek": 90_000},
        ]
        result = _compute_trade_stats(txs)
        assert result is not None
        assert result["total_trades"] == 2
        assert result["wins"] + result["losses"] == 2

    def test_all_wins(self):
        """All winning trades."""
        txs = [
            {"ticker": "X", "action": "BUY", "shares": 10, "total_sek": 10_000},
            {"ticker": "X", "action": "SELL", "shares": 5, "total_sek": 6_000},
            {"ticker": "X", "action": "SELL", "shares": 5, "total_sek": 7_000},
        ]
        result = _compute_trade_stats(txs)
        assert result is not None
        assert result["win_rate"] == 1.0
        assert result["losses"] == 0

    def test_all_losses(self):
        """All losing trades."""
        txs = [
            {"ticker": "X", "action": "BUY", "shares": 10, "total_sek": 10_000},
            {"ticker": "X", "action": "SELL", "shares": 5, "total_sek": 4_000},
            {"ticker": "X", "action": "SELL", "shares": 5, "total_sek": 3_000},
        ]
        result = _compute_trade_stats(txs)
        assert result is not None
        assert result["win_rate"] == 0.0
        assert result["wins"] == 0

    def test_ticker_filter(self):
        """Filter to specific ticker."""
        txs = [
            {"ticker": "BTC-USD", "action": "BUY", "shares": 1, "total_sek": 100_000},
            {"ticker": "BTC-USD", "action": "SELL", "shares": 1, "total_sek": 110_000},
            {"ticker": "BTC-USD", "action": "SELL", "shares": 1, "total_sek": 120_000},
            {"ticker": "ETH-USD", "action": "BUY", "shares": 10, "total_sek": 50_000},
            {"ticker": "ETH-USD", "action": "SELL", "shares": 10, "total_sek": 30_000},
            {"ticker": "ETH-USD", "action": "SELL", "shares": 10, "total_sek": 25_000},
        ]
        btc_result = _compute_trade_stats(txs, ticker="BTC-USD")
        eth_result = _compute_trade_stats(txs, ticker="ETH-USD")
        assert btc_result is not None
        assert btc_result["win_rate"] == 1.0
        assert eth_result is not None
        assert eth_result["win_rate"] == 0.0

    def test_no_sells(self):
        """Only BUY transactions -> returns None."""
        txs = [
            {"ticker": "X", "action": "BUY", "shares": 1, "total_sek": 100},
        ]
        assert _compute_trade_stats(txs) is None

    def test_empty_transactions(self):
        """Empty list -> returns None."""
        assert _compute_trade_stats([]) is None


# ===================================================================
# _get_signal_accuracy
# ===================================================================

class TestGetSignalAccuracy:
    def test_consensus_accuracy(self):
        """Primary source: consensus accuracy from signal_accuracy_1d."""
        summary = {
            "signal_accuracy_1d": {
                "consensus": {"accuracy": 0.65},
            },
        }
        assert _get_signal_accuracy(summary) == 0.65

    def test_fallback_to_weighted_confidence(self):
        """Fallback to ticker's weighted_confidence."""
        summary = {
            "signal_accuracy_1d": {},
            "signals": {
                "BTC-USD": {"weighted_confidence": 0.72},
            },
        }
        assert _get_signal_accuracy(summary, ticker="BTC-USD") == 0.72

    def test_last_resort_50_50(self):
        """No accuracy data at all -> returns 0.5."""
        assert _get_signal_accuracy({}) == 0.5
        assert _get_signal_accuracy({"signal_accuracy_1d": {}}) == 0.5

    def test_zero_consensus_accuracy_uses_fallback(self):
        """Consensus accuracy of 0 triggers fallback."""
        summary = {
            "signal_accuracy_1d": {"consensus": {"accuracy": 0}},
            "signals": {"BTC-USD": {"weighted_confidence": 0.6}},
        }
        assert _get_signal_accuracy(summary, ticker="BTC-USD") == 0.6


# ===================================================================
# _get_ticker_signal_accuracy
# ===================================================================

class TestGetTickerSignalAccuracy:
    def test_weighted_accuracy(self):
        """Compute weighted average of signal accuracies for active voters."""
        summary = {
            "signal_accuracy_1d": {
                "signals": {
                    "rsi": {"accuracy": 0.60, "samples": 100},
                    "macd": {"accuracy": 0.70, "samples": 50},
                    "ema": {"accuracy": 0.50, "samples": 200},
                },
            },
            "signal_weights": {
                "rsi": {"normalized_weight": 1.0},
                "macd": {"normalized_weight": 2.0},
                "ema": {"normalized_weight": 1.0},
            },
            "signals": {
                "BTC-USD": {
                    "extra": {
                        "_votes": {
                            "rsi": "BUY",
                            "macd": "BUY",
                            "ema": "HOLD",  # HOLD -> excluded
                        },
                    },
                },
            },
        }
        result = _get_ticker_signal_accuracy(summary, "BTC-USD")
        # Only rsi (BUY) and macd (BUY) are active
        # weighted = (0.60 * 1.0 + 0.70 * 2.0) / (1.0 + 2.0) = 2.0 / 3.0 = 0.6667
        assert result is not None
        assert abs(result - 2.0 / 3.0) < 0.001

    def test_no_votes(self):
        """No votes -> returns None."""
        summary = {"signal_accuracy_1d": {}, "signals": {}}
        assert _get_ticker_signal_accuracy(summary, "BTC-USD") is None

    def test_low_sample_signals_excluded(self):
        """Signals with <5 samples are excluded."""
        summary = {
            "signal_accuracy_1d": {
                "signals": {
                    "rsi": {"accuracy": 0.90, "samples": 3},  # too few
                },
            },
            "signal_weights": {"rsi": {"normalized_weight": 1.0}},
            "signals": {
                "BTC-USD": {"extra": {"_votes": {"rsi": "BUY"}}},
            },
        }
        assert _get_ticker_signal_accuracy(summary, "BTC-USD") is None


# ===================================================================
# recommended_size
# ===================================================================

class TestRecommendedSize:
    def test_basic_sizing(self, tmp_path, monkeypatch):
        """Basic sizing with known inputs."""
        pf_path = tmp_path / "portfolio_state.json"
        _write_json(pf_path, _make_portfolio(cash=500_000))

        summary = {
            "signal_accuracy_1d": {"consensus": {"accuracy": 0.6}},
            "signals": {
                "BTC-USD": {"price_usd": 60_000, "atr_pct": 3.0},
            },
        }

        result = recommended_size("BTC-USD", portfolio_path=pf_path,
                                  agent_summary=summary, strategy="patient")
        assert "kelly_pct" in result
        assert "half_kelly_pct" in result
        assert "recommended_sek" in result
        assert "max_alloc_sek" in result
        assert result["max_alloc_sek"] == 75_000  # 500K * 0.15

    def test_bold_strategy_max_alloc(self, tmp_path):
        """Bold strategy uses 30% allocation fraction."""
        pf_path = tmp_path / "portfolio_state_bold.json"
        _write_json(pf_path, _make_portfolio(cash=500_000))

        summary = {
            "signal_accuracy_1d": {"consensus": {"accuracy": 0.6}},
            "signals": {"BTC-USD": {"atr_pct": 3.0}},
        }

        result = recommended_size("BTC-USD", portfolio_path=pf_path,
                                  agent_summary=summary, strategy="bold")
        assert result["max_alloc_sek"] == 150_000  # 500K * 0.30

    def test_below_minimum_trade(self, tmp_path):
        """Recommended below 500 SEK -> set to 0."""
        pf_path = tmp_path / "portfolio_state.json"
        _write_json(pf_path, _make_portfolio(cash=100))  # Very low cash

        summary = {
            "signal_accuracy_1d": {"consensus": {"accuracy": 0.55}},
            "signals": {"BTC-USD": {"atr_pct": 3.0}},
        }

        result = recommended_size("BTC-USD", portfolio_path=pf_path,
                                  agent_summary=summary, strategy="patient")
        assert result["recommended_sek"] == 0

    def test_zero_cash(self, tmp_path):
        """Zero cash -> zero recommended size."""
        pf_path = tmp_path / "portfolio_state.json"
        _write_json(pf_path, _make_portfolio(cash=0))

        summary = {
            "signal_accuracy_1d": {"consensus": {"accuracy": 0.7}},
            "signals": {"BTC-USD": {"atr_pct": 3.0}},
        }

        result = recommended_size("BTC-USD", portfolio_path=pf_path,
                                  agent_summary=summary, strategy="patient")
        assert result["recommended_sek"] == 0
        assert result["max_alloc_sek"] == 0

    def test_with_trade_history(self, tmp_path):
        """Uses trade history when available for win/loss estimates."""
        pf_path = tmp_path / "portfolio_state.json"
        _write_json(pf_path, _make_portfolio(
            cash=400_000,
            transactions=[
                # Need at least 2 round-trips with both wins and losses
                # so avg_win_pct > 0 AND avg_loss_pct > 0
                {"ticker": "BTC-USD", "action": "BUY", "shares": 1, "total_sek": 100_000},
                # Sell 1: price_per_share = 60000/0.5 = 120000 > 100000 -> win
                {"ticker": "BTC-USD", "action": "SELL", "shares": 0.5, "total_sek": 60_000},
                # Sell 2: price_per_share = 40000/0.5 = 80000 < 100000 -> loss
                {"ticker": "BTC-USD", "action": "SELL", "shares": 0.5, "total_sek": 40_000},
            ],
        ))

        summary = {
            "signal_accuracy_1d": {"consensus": {"accuracy": 0.6}},
            "signals": {"BTC-USD": {"atr_pct": 3.0}},
        }

        result = recommended_size("BTC-USD", portfolio_path=pf_path,
                                  agent_summary=summary, strategy="patient")
        assert "trade history" in result["source"]

    def test_source_describes_data_used(self, tmp_path):
        """Source field describes which data was used."""
        pf_path = tmp_path / "portfolio_state.json"
        _write_json(pf_path, _make_portfolio(cash=500_000))

        summary = {
            "signal_accuracy_1d": {"consensus": {"accuracy": 0.6}},
            "signals": {"BTC-USD": {"atr_pct": 3.0}},
        }

        result = recommended_size("BTC-USD", portfolio_path=pf_path,
                                  agent_summary=summary, strategy="patient")
        assert "ATR-based" in result["source"] or "trade history" in result["source"]

    def test_missing_portfolio_file(self, tmp_path):
        """Non-existent portfolio file -> _load_json returns empty dict, cash=0."""
        pf_path = tmp_path / "nonexistent.json"
        summary = {
            "signal_accuracy_1d": {"consensus": {"accuracy": 0.6}},
            "signals": {"BTC-USD": {"atr_pct": 3.0}},
        }

        result = recommended_size("BTC-USD", portfolio_path=pf_path,
                                  agent_summary=summary, strategy="patient")
        assert result["recommended_sek"] == 0
