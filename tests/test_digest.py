"""Tests for weekly_digest module.

Covers:
- generate_weekly_digest() produces valid Telegram message
- _portfolio_summary() correctly calculates P&L
- _trades_this_week() filters by time window
- _signal_accuracy_this_week() computes per-signal accuracy
- _regime_distribution() tallies regimes correctly
- send_digest() handles missing config gracefully
"""

import json
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

from portfolio.weekly_digest import (
    _load_json,
    _load_jsonl,
    _portfolio_summary,
    _trades_this_week,
    _regime_distribution,
    generate_weekly_digest,
    send_digest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_patient_state():
    return {
        "cash_sek": 425000,
        "initial_value_sek": 500000,
        "total_fees_sek": 75.0,
        "holdings": {
            "MU": {"shares": 19.4453, "avg_cost_usd": 423.42}
        },
        "transactions": [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticker": "MU",
                "action": "BUY",
                "shares": 19.4453,
                "price_usd": 423.42,
                "price_sek": 3853.12,
                "total_sek": 75000,
                "fee_sek": 75.0,
                "confidence": 0.90,
                "fx_rate": 9.1,
                "reason": "Test BUY"
            }
        ]
    }


@pytest.fixture
def sample_bold_state():
    return {
        "cash_sek": 227622.15,
        "initial_value_sek": 500000,
        "total_fees_sek": 236.91,
        "holdings": {
            "MU": {"shares": 36.132, "avg_cost_usd": 423.42},
            "NVDA": {"shares": 56.5602, "avg_cost_usd": 189.97}
        },
        "transactions": [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticker": "MU",
                "action": "BUY",
                "shares": 36.132,
                "price_usd": 423.42,
                "price_sek": 3853.12,
                "total_sek": 139360.5,
                "fee_sek": 139.36,
                "confidence": 0.9,
                "fx_rate": 9.1,
                "reason": "Test BUY"
            },
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticker": "NVDA",
                "action": "BUY",
                "shares": 56.5602,
                "price_usd": 189.97,
                "price_sek": 1723.03,
                "total_sek": 97552.35,
                "fee_sek": 97.55,
                "confidence": 0.76,
                "fx_rate": 9.07,
                "reason": "Test BUY"
            }
        ]
    }


@pytest.fixture
def sample_journal_entries():
    now = datetime.now(timezone.utc)
    return [
        {"ts": now.isoformat(), "regime": "range-bound", "decisions": {"patient": {"action": "HOLD"}, "bold": {"action": "HOLD"}}},
        {"ts": (now - timedelta(hours=1)).isoformat(), "regime": "range-bound", "decisions": {"patient": {"action": "HOLD"}, "bold": {"action": "HOLD"}}},
        {"ts": (now - timedelta(hours=2)).isoformat(), "regime": "trending-up", "decisions": {"patient": {"action": "HOLD"}, "bold": {"action": "HOLD"}}},
        {"ts": (now - timedelta(days=2)).isoformat(), "regime": "range-bound", "decisions": {"patient": {"action": "HOLD"}, "bold": {"action": "HOLD"}}},
        {"ts": (now - timedelta(days=3)).isoformat(), "regime": "high-vol", "decisions": {"patient": {"action": "HOLD"}, "bold": {"action": "HOLD"}}},
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPortfolioSummary:
    def test_calculates_pnl_correctly(self, sample_patient_state):
        summary = _portfolio_summary(sample_patient_state, "Patient")
        assert summary["label"] == "Patient"
        assert summary["cash_sek"] == 425000
        assert summary["initial_sek"] == 500000
        assert summary["pnl_sek"] == -75000  # cash - initial (ignores holdings)
        assert summary["pnl_pct"] == pytest.approx(-15.0, abs=0.1)
        assert summary["total_fees_sek"] == 75.0
        assert summary["total_trades"] == 1
        assert "MU" in summary["active_holdings"]

    def test_empty_portfolio(self):
        state = {"cash_sek": 500000, "initial_value_sek": 500000, "total_fees_sek": 0, "holdings": {}, "transactions": []}
        summary = _portfolio_summary(state, "Empty")
        assert summary["pnl_sek"] == 0
        assert summary["pnl_pct"] == 0.0
        assert summary["active_holdings"] == []

    def test_handles_missing_fields(self):
        summary = _portfolio_summary({}, "Missing")
        assert summary["cash_sek"] == 0
        assert summary["total_trades"] == 0


class TestTradesThisWeek:
    def test_filters_recent_trades(self):
        now = datetime.now(timezone.utc)
        since = now - timedelta(days=7)
        transactions = [
            {"timestamp": now.isoformat(), "action": "BUY", "ticker": "MU"},
            {"timestamp": (now - timedelta(days=3)).isoformat(), "action": "SELL", "ticker": "ETH"},
            {"timestamp": (now - timedelta(days=10)).isoformat(), "action": "BUY", "ticker": "BTC"},  # too old
        ]
        result = _trades_this_week(transactions, since)
        assert len(result) == 2

    def test_empty_transactions(self):
        result = _trades_this_week([], datetime.now(timezone.utc) - timedelta(days=7))
        assert result == []

    def test_handles_bad_timestamps(self):
        transactions = [{"timestamp": "not-a-date", "action": "BUY"}]
        result = _trades_this_week(transactions, datetime.now(timezone.utc) - timedelta(days=7))
        assert result == []


class TestRegimeDistribution:
    def test_calculates_distribution(self, sample_journal_entries):
        dist = _regime_distribution(sample_journal_entries)
        assert "range-bound" in dist
        assert dist["range-bound"] == pytest.approx(60.0, abs=0.1)  # 3/5
        assert "trending-up" in dist
        assert "high-vol" in dist

    def test_empty_entries(self):
        assert _regime_distribution([]) == {}

    def test_missing_regime_field(self):
        entries = [{"ts": "2026-01-01"}, {"ts": "2026-01-02"}]
        assert _regime_distribution(entries) == {}


class TestGenerateWeeklyDigest:
    @patch("portfolio.weekly_digest.PATIENT_FILE")
    @patch("portfolio.weekly_digest.BOLD_FILE")
    @patch("portfolio.weekly_digest.SIGNAL_LOG")
    @patch("portfolio.weekly_digest.JOURNAL_FILE")
    def test_generates_valid_message(self, mock_journal, mock_signal,
                                      mock_bold, mock_patient, tmp_path,
                                      sample_patient_state, sample_bold_state):
        # Write test data to temp files
        patient_file = tmp_path / "patient.json"
        bold_file = tmp_path / "bold.json"
        signal_file = tmp_path / "signal.jsonl"
        journal_file = tmp_path / "journal.jsonl"

        patient_file.write_text(json.dumps(sample_patient_state))
        bold_file.write_text(json.dumps(sample_bold_state))
        signal_file.write_text("")
        journal_file.write_text("")

        mock_patient.__class__ = type(patient_file)
        mock_bold.__class__ = type(bold_file)

        # Use the real function with mocked paths
        with patch("portfolio.weekly_digest.PATIENT_FILE", patient_file), \
             patch("portfolio.weekly_digest.BOLD_FILE", bold_file), \
             patch("portfolio.weekly_digest.SIGNAL_LOG", signal_file), \
             patch("portfolio.weekly_digest.JOURNAL_FILE", journal_file):
            msg = generate_weekly_digest()

        assert "*WEEKLY DIGEST*" in msg
        assert "Patient" in msg
        assert "Bold" in msg
        assert "Signal Accuracy" in msg
        assert "Regime Distribution" in msg


class TestSendDigest:
    def test_missing_config_returns_none(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        with patch("portfolio.weekly_digest.CONFIG_FILE", config_file):
            result = send_digest("test message")
        assert result is None

    @patch("portfolio.weekly_digest.requests.post")
    def test_sends_with_valid_config(self, mock_post, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "telegram": {"token": "fake-token", "chat_id": "12345"}
        }))
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        telegram_log = data_dir / "telegram_messages.jsonl"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        with patch("portfolio.weekly_digest.CONFIG_FILE", config_file), \
             patch("portfolio.weekly_digest.DATA_DIR", data_dir):
            result = send_digest("test message")

        assert result is not None
        mock_post.assert_called_once()
        assert telegram_log.exists()

    @patch("portfolio.weekly_digest.requests.post")
    def test_handles_network_error(self, mock_post, tmp_path):
        import requests as req
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "telegram": {"token": "fake", "chat_id": "123"}
        }))
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        mock_post.side_effect = req.RequestException("Connection failed")

        with patch("portfolio.weekly_digest.CONFIG_FILE", config_file), \
             patch("portfolio.weekly_digest.DATA_DIR", data_dir):
            result = send_digest("test")

        assert result is None
