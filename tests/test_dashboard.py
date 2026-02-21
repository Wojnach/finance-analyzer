"""Tests for the Flask dashboard API routes.

Covers:
- All GET/POST API routes
- Authentication middleware (token query param and Bearer header)
- Missing data handling (404 for missing JSON files)
- Helpers: _read_json, _read_jsonl
- Portfolio validation endpoint
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from dashboard.app import app, _read_json, _read_jsonl


@pytest.fixture
def client():
    """Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def tmp_data(tmp_path):
    """Patch DATA_DIR to a temporary directory for isolated tests."""
    with patch("dashboard.app.DATA_DIR", tmp_path):
        yield tmp_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestReadJson:
    def test_returns_parsed_json(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text('{"key": "value"}', encoding="utf-8")
        assert _read_json(f) == {"key": "value"}

    def test_returns_none_for_missing(self, tmp_path):
        assert _read_json(tmp_path / "missing.json") is None

    def test_handles_array(self, tmp_path):
        f = tmp_path / "arr.json"
        f.write_text("[1, 2, 3]", encoding="utf-8")
        assert _read_json(f) == [1, 2, 3]


class TestReadJsonl:
    def test_returns_entries(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_text('{"a":1}\n{"a":2}\n{"a":3}\n', encoding="utf-8")
        result = _read_jsonl(f)
        assert len(result) == 3
        assert result[0] == {"a": 1}

    def test_returns_empty_for_missing(self, tmp_path):
        assert _read_jsonl(tmp_path / "missing.jsonl") == []

    def test_respects_limit(self, tmp_path):
        f = tmp_path / "big.jsonl"
        lines = [json.dumps({"i": i}) for i in range(10)]
        f.write_text("\n".join(lines), encoding="utf-8")
        result = _read_jsonl(f, limit=3)
        assert len(result) == 3
        assert result[0]["i"] == 7  # last 3 entries

    def test_skips_bad_lines(self, tmp_path):
        f = tmp_path / "bad.jsonl"
        f.write_text('{"ok":1}\nnot json\n{"ok":2}\n', encoding="utf-8")
        result = _read_jsonl(f)
        assert len(result) == 2

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "blanks.jsonl"
        f.write_text('{"a":1}\n\n\n{"a":2}\n', encoding="utf-8")
        result = _read_jsonl(f)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestAuth:
    def test_no_token_configured_allows_access(self, client, tmp_data):
        """No dashboard_token in config = open access."""
        with patch("dashboard.app._get_dashboard_token", return_value=None):
            resp = client.get("/api/invocations")
            assert resp.status_code == 200

    def test_valid_query_token(self, client, tmp_data):
        with patch("dashboard.app._get_dashboard_token", return_value="secret123"):
            resp = client.get("/api/invocations?token=secret123")
            assert resp.status_code == 200

    def test_invalid_query_token(self, client, tmp_data):
        with patch("dashboard.app._get_dashboard_token", return_value="secret123"):
            resp = client.get("/api/invocations?token=wrong")
            assert resp.status_code == 401

    def test_missing_token_returns_401(self, client, tmp_data):
        with patch("dashboard.app._get_dashboard_token", return_value="secret123"):
            resp = client.get("/api/invocations")
            assert resp.status_code == 401

    def test_valid_bearer_header(self, client, tmp_data):
        with patch("dashboard.app._get_dashboard_token", return_value="secret123"):
            resp = client.get(
                "/api/invocations",
                headers={"Authorization": "Bearer secret123"},
            )
            assert resp.status_code == 200

    def test_invalid_bearer_header(self, client, tmp_data):
        with patch("dashboard.app._get_dashboard_token", return_value="secret123"):
            resp = client.get(
                "/api/invocations",
                headers={"Authorization": "Bearer wrong"},
            )
            assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Static route
# ---------------------------------------------------------------------------


class TestIndex:
    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"html" in resp.data.lower()


# ---------------------------------------------------------------------------
# API routes — no auth (patched away)
# ---------------------------------------------------------------------------


def _no_auth():
    return patch("dashboard.app._get_dashboard_token", return_value=None)


class TestApiSummary:
    def test_returns_combined_data(self, client, tmp_data):
        (tmp_data / "agent_summary.json").write_text('{"signals": {}}', encoding="utf-8")
        (tmp_data / "portfolio_state.json").write_text('{"cash_sek": 500000}', encoding="utf-8")
        (tmp_data / "portfolio_state_bold.json").write_text('{"cash_sek": 450000}', encoding="utf-8")
        (tmp_data / "telegram_messages.jsonl").write_text('{"ts":"t","text":"hi"}\n', encoding="utf-8")

        with _no_auth():
            resp = client.get("/api/summary")
        data = resp.get_json()
        assert "signals" in data
        assert "portfolio" in data
        assert "portfolio_bold" in data
        assert "telegrams" in data
        assert data["portfolio"]["cash_sek"] == 500000

    def test_returns_nulls_when_files_missing(self, client, tmp_data):
        with _no_auth():
            resp = client.get("/api/summary")
        data = resp.get_json()
        assert data["signals"] is None
        assert data["portfolio"] is None


class TestApiSignals:
    def test_returns_signals(self, client, tmp_data):
        (tmp_data / "agent_summary.json").write_text('{"signals": {"BTC-USD": {}}}', encoding="utf-8")
        with _no_auth():
            resp = client.get("/api/signals")
        assert resp.status_code == 200
        assert "BTC-USD" in resp.get_json()["signals"]

    def test_404_when_missing(self, client, tmp_data):
        with _no_auth():
            resp = client.get("/api/signals")
        assert resp.status_code == 404


class TestApiPortfolio:
    def test_returns_patient_portfolio(self, client, tmp_data):
        (tmp_data / "portfolio_state.json").write_text('{"cash_sek": 500000}', encoding="utf-8")
        with _no_auth():
            resp = client.get("/api/portfolio")
        assert resp.status_code == 200
        assert resp.get_json()["cash_sek"] == 500000

    def test_404_when_missing(self, client, tmp_data):
        with _no_auth():
            resp = client.get("/api/portfolio")
        assert resp.status_code == 404


class TestApiPortfolioBold:
    def test_returns_bold_portfolio(self, client, tmp_data):
        (tmp_data / "portfolio_state_bold.json").write_text('{"cash_sek": 450000}', encoding="utf-8")
        with _no_auth():
            resp = client.get("/api/portfolio-bold")
        assert resp.status_code == 200
        assert resp.get_json()["cash_sek"] == 450000

    def test_404_when_missing(self, client, tmp_data):
        with _no_auth():
            resp = client.get("/api/portfolio-bold")
        assert resp.status_code == 404


class TestApiInvocations:
    def test_returns_entries(self, client, tmp_data):
        (tmp_data / "invocations.jsonl").write_text(
            '{"ts":"2026-02-21T12:00:00","reasons":["signal_consensus"]}\n',
            encoding="utf-8",
        )
        with _no_auth():
            resp = client.get("/api/invocations")
        data = resp.get_json()
        assert len(data) == 1
        assert data[0]["reasons"] == ["signal_consensus"]

    def test_empty_when_missing(self, client, tmp_data):
        with _no_auth():
            resp = client.get("/api/invocations")
        assert resp.get_json() == []


class TestApiTelegrams:
    def test_returns_entries(self, client, tmp_data):
        (tmp_data / "telegram_messages.jsonl").write_text(
            '{"ts":"t","text":"hello"}\n', encoding="utf-8"
        )
        with _no_auth():
            resp = client.get("/api/telegrams")
        data = resp.get_json()
        assert len(data) == 1
        assert data[0]["text"] == "hello"


class TestApiSignalLog:
    def test_returns_entries(self, client, tmp_data):
        (tmp_data / "signal_log.jsonl").write_text(
            '{"ts":"t","signals":{}}\n', encoding="utf-8"
        )
        with _no_auth():
            resp = client.get("/api/signal-log")
        assert resp.status_code == 200
        assert len(resp.get_json()) == 1


class TestApiAccuracy:
    def test_returns_accuracy_data(self, client, tmp_data):
        with _no_auth():
            with patch("dashboard.app.Path"):
                # Mock the accuracy_stats imports
                mock_sa = {"rsi": {"correct": 10, "total": 20, "accuracy": 0.5}}
                mock_ca = {"correct": 50, "total": 100, "accuracy": 0.5}
                mock_ta = {"BTC-USD": {"correct": 5, "total": 10}}
                with patch.dict("sys.modules", {
                    "portfolio.accuracy_stats": MagicMock(
                        signal_accuracy=MagicMock(return_value=mock_sa),
                        consensus_accuracy=MagicMock(return_value=mock_ca),
                        per_ticker_accuracy=MagicMock(return_value=mock_ta),
                    )
                }):
                    resp = client.get("/api/accuracy")
        assert resp.status_code == 200


class TestApiIskbets:
    def test_returns_config_and_state(self, client, tmp_data):
        (tmp_data / "iskbets_config.json").write_text('{"enabled": true}', encoding="utf-8")
        (tmp_data / "iskbets_state.json").write_text('{"positions": []}', encoding="utf-8")
        with _no_auth():
            resp = client.get("/api/iskbets")
        data = resp.get_json()
        assert data["config"]["enabled"] is True
        assert data["state"]["positions"] == []


class TestApiLoraStatus:
    def test_returns_lora_state(self, client):
        with _no_auth():
            with patch("dashboard.app.TRAINING_DIR", Path("/nonexistent")):
                resp = client.get("/api/lora-status")
        data = resp.get_json()
        assert data["state"] is None
        assert data["training_progress"] is None


class TestApiEquityCurve:
    def test_returns_history(self, client, tmp_data):
        (tmp_data / "portfolio_value_history.jsonl").write_text(
            '{"ts":"t","value":500000}\n{"ts":"t2","value":505000}\n',
            encoding="utf-8",
        )
        with _no_auth():
            resp = client.get("/api/equity-curve")
        data = resp.get_json()
        assert len(data) == 2
        assert data[1]["value"] == 505000

    def test_empty_when_missing(self, client, tmp_data):
        with _no_auth():
            resp = client.get("/api/equity-curve")
        assert resp.get_json() == []


class TestApiTriggers:
    def test_returns_trigger_events(self, client, tmp_data):
        (tmp_data / "invocations.jsonl").write_text(
            '{"ts":"t","reasons":["cooldown"]}\n', encoding="utf-8"
        )
        with _no_auth():
            resp = client.get("/api/triggers")
        data = resp.get_json()
        assert len(data) == 1


class TestApiSignalHeatmap:
    def test_returns_heatmap_grid(self, client, tmp_data):
        summary = {
            "signals": {
                "BTC-USD": {
                    "action": "BUY",
                    "extra": {
                        "_votes": {
                            "rsi": "BUY", "macd": "SELL", "ema": "HOLD",
                            "bb": None, "fear_greed": "BUY", "sentiment": "HOLD",
                            "ministral": "HOLD", "ml": "BUY", "funding": "HOLD",
                            "volume": "HOLD", "custom_lora": "HOLD",
                            "trend": "BUY", "momentum": "SELL",
                            "volume_flow": "HOLD", "volatility_sig": "HOLD",
                            "candlestick": "HOLD", "structure": "HOLD",
                            "fibonacci": "HOLD", "smart_money": "HOLD",
                            "oscillators": "HOLD", "heikin_ashi": "HOLD",
                            "mean_reversion": "HOLD", "calendar": "HOLD",
                            "macro_regime": "HOLD", "momentum_factors": "HOLD",
                        }
                    }
                }
            }
        }
        (tmp_data / "agent_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        with _no_auth():
            resp = client.get("/api/signal-heatmap")
        data = resp.get_json()
        assert "BTC-USD" in data["tickers"]
        assert data["heatmap"]["BTC-USD"]["rsi"] == "BUY"
        assert data["heatmap"]["BTC-USD"]["macd"] == "SELL"
        assert data["heatmap"]["BTC-USD"]["bb"] == "HOLD"  # None → "HOLD"
        assert len(data["signals"]) == 24

    def test_404_when_no_summary(self, client, tmp_data):
        with _no_auth():
            resp = client.get("/api/signal-heatmap")
        assert resp.status_code == 404


class TestApiAccuracyHistory:
    def test_returns_history_entries(self, client, tmp_data):
        entry = json.dumps({
            "ts": "2026-02-20T00:00:00+00:00",
            "signals": {"rsi": {"accuracy": 0.5, "total": 100}},
        })
        (tmp_data / "accuracy_snapshots.jsonl").write_text(
            entry + "\n", encoding="utf-8"
        )
        with _no_auth():
            resp = client.get("/api/accuracy-history")
        data = resp.get_json()
        assert len(data) == 1
        assert data[0]["signals"]["rsi"]["accuracy"] == 0.5

    def test_empty_when_missing(self, client, tmp_data):
        with _no_auth():
            resp = client.get("/api/accuracy-history")
        assert resp.get_json() == []


class TestApiTrades:
    def test_returns_combined_trades(self, client, tmp_data):
        patient = {
            "cash_sek": 425000,
            "transactions": [
                {"timestamp": "2026-02-15T10:00:00Z", "ticker": "MU", "action": "BUY",
                 "total_sek": 75000, "price_usd": 420}
            ],
        }
        bold = {
            "cash_sek": 350000,
            "transactions": [
                {"timestamp": "2026-02-14T10:00:00Z", "ticker": "NVDA", "action": "BUY",
                 "total_sek": 100000, "price_usd": 180}
            ],
        }
        (tmp_data / "portfolio_state.json").write_text(
            json.dumps(patient), encoding="utf-8"
        )
        (tmp_data / "portfolio_state_bold.json").write_text(
            json.dumps(bold), encoding="utf-8"
        )
        with _no_auth():
            resp = client.get("/api/trades")
        data = resp.get_json()
        assert len(data) == 2
        # Sorted by timestamp: NVDA first (Feb 14), then MU (Feb 15)
        assert data[0]["ticker"] == "NVDA"
        assert data[0]["strategy"] == "bold"
        assert data[1]["ticker"] == "MU"
        assert data[1]["strategy"] == "patient"

    def test_empty_when_no_files(self, client, tmp_data):
        with _no_auth():
            resp = client.get("/api/trades")
        assert resp.get_json() == []

    def test_empty_when_no_transactions(self, client, tmp_data):
        (tmp_data / "portfolio_state.json").write_text(
            '{"cash_sek": 500000}', encoding="utf-8"
        )
        with _no_auth():
            resp = client.get("/api/trades")
        assert resp.get_json() == []


class TestApiHealth:
    def test_returns_health_data(self, client):
        mock_health = {
            "loop_alive": True,
            "cycles": 100,
            "agent_silent": False,
        }
        with _no_auth():
            with patch("portfolio.health.get_health_summary", return_value=mock_health):
                resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["loop_alive"] is True
        assert data["cycles"] == 100


# ---------------------------------------------------------------------------
# Portfolio validation
# ---------------------------------------------------------------------------


class TestApiValidatePortfolio:
    def test_valid_portfolio_no_trades(self, client):
        portfolio = {
            "cash_sek": 500000,
            "initial_value_sek": 500000,
            "holdings": {},
            "transactions": [],
        }
        with _no_auth():
            resp = client.post(
                "/api/validate-portfolio",
                data=json.dumps(portfolio),
                content_type="application/json",
            )
        data = resp.get_json()
        assert data["valid"] is True
        assert data["errors"] == []

    def test_valid_portfolio_with_trades(self, client):
        portfolio = {
            "cash_sek": 425000,
            "initial_value_sek": 500000,
            "holdings": {
                "BTC-USD": {"shares": 0.1, "avg_cost_usd": 65000}
            },
            "transactions": [
                {
                    "action": "BUY",
                    "ticker": "BTC-USD",
                    "shares": 0.1,
                    "total_sek": 75000,
                    "fee_sek": 37.5,
                }
            ],
        }
        with _no_auth():
            resp = client.post(
                "/api/validate-portfolio",
                data=json.dumps(portfolio),
                content_type="application/json",
            )
        data = resp.get_json()
        assert data["valid"] is True

    def test_negative_cash_detected(self, client):
        portfolio = {
            "cash_sek": -100,
            "initial_value_sek": 500000,
            "holdings": {},
            "transactions": [],
        }
        with _no_auth():
            resp = client.post(
                "/api/validate-portfolio",
                data=json.dumps(portfolio),
                content_type="application/json",
            )
        data = resp.get_json()
        assert data["valid"] is False
        assert any("negative" in e for e in data["errors"])

    def test_cash_mismatch_detected(self, client):
        portfolio = {
            "cash_sek": 500000,  # should be 425000 after a 75K buy
            "initial_value_sek": 500000,
            "holdings": {"BTC-USD": {"shares": 0.1}},
            "transactions": [
                {"action": "BUY", "ticker": "BTC-USD", "shares": 0.1, "total_sek": 75000, "fee_sek": 37.5}
            ],
        }
        with _no_auth():
            resp = client.post(
                "/api/validate-portfolio",
                data=json.dumps(portfolio),
                content_type="application/json",
            )
        data = resp.get_json()
        assert data["valid"] is False
        assert any("Cash mismatch" in e for e in data["errors"])

    def test_holdings_mismatch_detected(self, client):
        portfolio = {
            "cash_sek": 425000,
            "initial_value_sek": 500000,
            "holdings": {"BTC-USD": {"shares": 0.5}},  # should be 0.1
            "transactions": [
                {"action": "BUY", "ticker": "BTC-USD", "shares": 0.1, "total_sek": 75000, "fee_sek": 37.5}
            ],
        }
        with _no_auth():
            resp = client.post(
                "/api/validate-portfolio",
                data=json.dumps(portfolio),
                content_type="application/json",
            )
        data = resp.get_json()
        assert data["valid"] is False
        assert any("Holdings mismatch" in e for e in data["errors"])

    def test_no_json_body(self, client):
        with _no_auth():
            resp = client.post("/api/validate-portfolio")
        assert resp.status_code == 400

    def test_missing_cash_field(self, client):
        with _no_auth():
            resp = client.post(
                "/api/validate-portfolio",
                data=json.dumps({"holdings": {}, "transactions": []}),
                content_type="application/json",
            )
        data = resp.get_json()
        assert data["valid"] is False
        assert any("Missing cash_sek" in e for e in data["errors"])
