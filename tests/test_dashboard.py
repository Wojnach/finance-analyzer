"""Tests for the Flask dashboard API routes.

Covers:
- All GET/POST API routes
- Authentication middleware (token query param and Bearer header)
- Missing data handling (404 for missing JSON files)
- Helpers: _read_json, _read_jsonl
- Portfolio validation endpoint
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dashboard.app import _read_json, _read_jsonl, app


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

    def test_summary_telegrams_skip_non_dict_lines(self, client, tmp_data):
        (tmp_data / "agent_summary.json").write_text('{"signals": {}}', encoding="utf-8")
        (tmp_data / "portfolio_state.json").write_text('{"cash_sek": 500000}', encoding="utf-8")
        (tmp_data / "portfolio_state_bold.json").write_text('{"cash_sek": 450000}', encoding="utf-8")
        (tmp_data / "telegram_messages.jsonl").write_text(
            '"raw string"\n{"ts":"t1","text":"ok"}\n[1,2,3]\n',
            encoding="utf-8",
        )

        with _no_auth():
            resp = client.get("/api/summary")
        data = resp.get_json()
        assert len(data["telegrams"]) == 1
        assert data["telegrams"][0]["text"] == "ok"

    def test_summary_sanitizes_nan_values_for_browser_json(self, client, tmp_data):
        payload = {
            "signals": {
                "BTC-USD": {
                    "indicators": {"sma50": float("nan")},
                }
            }
        }
        (tmp_data / "agent_summary.json").write_text(json.dumps(payload), encoding="utf-8")

        with _no_auth():
            resp = client.get("/api/summary")

        assert resp.status_code == 200
        assert "NaN" not in resp.get_data(as_text=True)
        assert resp.get_json()["signals"]["signals"]["BTC-USD"]["indicators"]["sma50"] is None


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

    def test_skips_non_dict_json_lines(self, client, tmp_data):
        (tmp_data / "telegram_messages.jsonl").write_text(
            '"raw string"\n{"ts":"t1","text":"ok"}\n[1,2,3]\n',
            encoding="utf-8",
        )
        with _no_auth():
            resp = client.get("/api/telegrams")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == 1
        assert data[0]["text"] == "ok"

    def test_category_filter_is_case_insensitive(self, client, tmp_data):
        (tmp_data / "telegram_messages.jsonl").write_text(
            '{"ts":"t1","category":"Trade","text":"buy"}\n'
            '{"ts":"t2","category":"analysis","text":"hold"}\n',
            encoding="utf-8",
        )
        with _no_auth():
            resp = client.get("/api/telegrams?category=trade")
        data = resp.get_json()
        assert len(data) == 1
        assert data[0]["text"] == "buy"


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
        with _no_auth(), patch("dashboard.app.Path"):
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
        with _no_auth(), patch("dashboard.app.TRAINING_DIR", Path("/nonexistent")):
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
        assert len(data["signals"]) == 30  # 11 core + 19 enhanced

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


class TestApiLocalLlmTrends:
    def test_returns_latest_and_series(self, client, tmp_data):
        latest = {
            "date": "2026-03-09",
            "exported_at": "2026-03-09T18:10:00+00:00",
            "days": 30,
            "ministral": {
                "overall": {"accuracy": 0.6, "samples": 20, "correct": 12},
                "by_ticker": {"BTC-USD": {"accuracy": 0.7, "samples": 10, "correct": 7}},
            },
            "health": {
                "chronos": {"success_rate": 0.9, "total": 10},
                "kronos": {"success_rate": 0.5, "total": 8},
            },
            "forecast": {
                "raw": {
                    "1h": {"chronos_1h": {"accuracy": 0.5, "correct": 5, "total": 10}},
                    "24h": {"chronos_24h": {"accuracy": 0.6, "correct": 6, "total": 10}},
                },
                "effective": {
                    "1h": {"chronos_1h": {"accuracy": 0.7, "correct": 7, "total": 10}},
                    "24h": {"chronos_24h": {"accuracy": 0.8, "correct": 8, "total": 10}},
                },
            },
            "gating_counts": {"forecast": {"raw": 4, "held": 2, "insufficient_data": 1}},
            "recommendations": [],
        }
        history = [
            latest,
            {
                "date": "2026-03-10",
                "exported_at": "2026-03-10T18:10:00+00:00",
                "days": 30,
                "ministral": {
                    "overall": {"accuracy": 0.65, "samples": 22, "correct": 14},
                    "by_ticker": {"BTC-USD": {"accuracy": 0.75, "samples": 12, "correct": 9}},
                },
                "health": {
                    "chronos": {"success_rate": 1.0, "total": 12},
                    "kronos": {"success_rate": 0.55, "total": 9},
                },
                "forecast": {
                    "raw": {
                        "1h": {"chronos_1h": {"accuracy": 0.55, "correct": 11, "total": 20}},
                        "24h": {"chronos_24h": {"accuracy": 0.65, "correct": 13, "total": 20}},
                    },
                    "effective": {
                        "1h": {"chronos_1h": {"accuracy": 0.75, "correct": 15, "total": 20}},
                        "24h": {"chronos_24h": {"accuracy": 0.85, "correct": 17, "total": 20}},
                    },
                },
                "gating_counts": {"forecast": {"raw": 5, "held": 1}},
                "recommendations": [],
            },
        ]

        (tmp_data / "local_llm_report_latest.json").write_text(
            json.dumps(latest), encoding="utf-8"
        )
        (tmp_data / "local_llm_report_history.jsonl").write_text(
            "\n".join(json.dumps(entry) for entry in history) + "\n",
            encoding="utf-8",
        )

        with _no_auth():
            resp = client.get("/api/local-llm-trends?limit=1&ticker=BTC-USD")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ticker"] == "BTC-USD"
        assert data["latest"]["date"] == "2026-03-09"
        assert len(data["series"]) == 1
        assert data["series"][0]["date"] == "2026-03-10"
        assert data["series"][0]["ministral_ticker_accuracy"] == 0.75
        assert data["series"][0]["forecast_effective_24h_accuracy"] == 0.85
        assert data["series"][0]["forecast_gating_raw"] == 5

    def test_empty_shape_when_missing(self, client, tmp_data):
        with _no_auth():
            resp = client.get("/api/local-llm-trends")

        assert resp.status_code == 200
        assert resp.get_json() == {"ticker": None, "latest": None, "series": []}


class TestApiMetalsAccuracy:
    def test_returns_stats_when_file_exists(self, client, tmp_data):
        payload = {"stats": {"xag": {"1h": {"accuracy": 0.6, "total": 10}}}}
        (tmp_data / "metals_signal_accuracy.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        with _no_auth():
            resp = client.get("/api/metals-accuracy")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["stats"]["xag"]["1h"]["accuracy"] == 0.6

    def test_returns_empty_shape_when_missing(self, client, tmp_data):
        with _no_auth():
            resp = client.get("/api/metals-accuracy")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["error"] == "no data"
        assert data["stats"] == {}

    def test_requires_auth_when_configured(self, client, tmp_data):
        with patch("dashboard.app._get_dashboard_token", return_value="secret"):
            resp = client.get("/api/metals-accuracy")
        assert resp.status_code == 401


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
        with _no_auth(), patch("portfolio.health.get_health_summary", return_value=mock_health):
            resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["loop_alive"] is True
        assert data["cycles"] == 100


# ---------------------------------------------------------------------------
# Portfolio validation
# ---------------------------------------------------------------------------


class TestApiValidatePortfolio:
    """Tests for /api/validate-portfolio endpoint.

    Uses portfolio_validator.validate_portfolio() which performs comprehensive
    checks including cash, holdings, fees, and transaction field completeness.
    """

    def test_valid_portfolio_no_trades(self, client):
        portfolio = {
            "cash_sek": 500000,
            "initial_value_sek": 500000,
            "total_fees_sek": 0,
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
            "total_fees_sek": 37.5,
            "holdings": {
                "BTC-USD": {"shares": 0.1, "avg_cost_usd": 65000}
            },
            "transactions": [
                {
                    "timestamp": "2026-02-20T12:00:00+00:00",
                    "action": "BUY",
                    "ticker": "BTC-USD",
                    "shares": 0.1,
                    "price_usd": 65000,
                    "total_sek": 75000,
                    "fee_sek": 37.5,
                    "reason": "Test buy",
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
            "total_fees_sek": 0,
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
        assert any("negative" in e.lower() for e in data["errors"])

    def test_cash_mismatch_detected(self, client):
        portfolio = {
            "cash_sek": 500000,  # should be 425000 after a 75K buy
            "initial_value_sek": 500000,
            "total_fees_sek": 37.5,
            "holdings": {"BTC-USD": {"shares": 0.1, "avg_cost_usd": 65000}},
            "transactions": [
                {
                    "timestamp": "2026-02-20T12:00:00+00:00",
                    "action": "BUY",
                    "ticker": "BTC-USD",
                    "shares": 0.1,
                    "price_usd": 65000,
                    "total_sek": 75000,
                    "fee_sek": 37.5,
                    "reason": "Test buy",
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
        assert data["valid"] is False
        assert any("cash" in e.lower() for e in data["errors"])

    def test_holdings_mismatch_detected(self, client):
        portfolio = {
            "cash_sek": 425000,
            "initial_value_sek": 500000,
            "total_fees_sek": 37.5,
            "holdings": {"BTC-USD": {"shares": 0.5, "avg_cost_usd": 65000}},  # should be 0.1
            "transactions": [
                {
                    "timestamp": "2026-02-20T12:00:00+00:00",
                    "action": "BUY",
                    "ticker": "BTC-USD",
                    "shares": 0.1,
                    "price_usd": 65000,
                    "total_sek": 75000,
                    "fee_sek": 37.5,
                    "reason": "Test buy",
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
        assert data["valid"] is False
        assert any("mismatch" in e.lower() for e in data["errors"])

    def test_no_json_body(self, client):
        with _no_auth():
            resp = client.post("/api/validate-portfolio")
        assert resp.status_code == 400

    def test_missing_cash_field(self, client):
        with _no_auth():
            resp = client.post(
                "/api/validate-portfolio",
                data=json.dumps({"holdings": {}, "transactions": [], "total_fees_sek": 0}),
                content_type="application/json",
            )
        data = resp.get_json()
        assert data["valid"] is False
        assert any("cash_sek" in e for e in data["errors"])


# ---------------------------------------------------------------------------
# _read_jsonl streaming (deque-based)
# ---------------------------------------------------------------------------


class TestReadJsonlStreaming:
    """Verify that _read_jsonl streams line-by-line using deque."""

    def test_uses_deque_limit(self, tmp_path):
        """Deque maxlen limits entries to last N without loading all into memory."""
        f = tmp_path / "stream.jsonl"
        lines = [json.dumps({"i": i}) for i in range(200)]
        f.write_text("\n".join(lines), encoding="utf-8")
        result = _read_jsonl(f, limit=5)
        assert len(result) == 5
        # Should be the last 5 entries
        assert result[0]["i"] == 195
        assert result[4]["i"] == 199

    def test_limit_larger_than_file(self, tmp_path):
        """When file has fewer entries than limit, return all entries."""
        f = tmp_path / "small.jsonl"
        lines = [json.dumps({"i": i}) for i in range(3)]
        f.write_text("\n".join(lines), encoding="utf-8")
        result = _read_jsonl(f, limit=100)
        assert len(result) == 3

    def test_returns_list_not_deque(self, tmp_path):
        """Result must be a list, not a deque."""
        f = tmp_path / "type.jsonl"
        f.write_text('{"a":1}\n', encoding="utf-8")
        result = _read_jsonl(f)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# /api/decisions
# ---------------------------------------------------------------------------


def _sample_journal_entry(
    ts="2026-02-22T10:00:00+00:00",
    trigger="cooldown",
    regime="range-bound",
    patient_action="HOLD",
    bold_action="HOLD",
    patient_reasoning="No setup",
    bold_reasoning="No breakout",
    tickers=None,
):
    """Build a sample layer2_journal.jsonl entry."""
    return {
        "ts": ts,
        "trigger": trigger,
        "regime": regime,
        "reflection": "",
        "continues": None,
        "decisions": {
            "patient": {"action": patient_action, "reasoning": patient_reasoning},
            "bold": {"action": bold_action, "reasoning": bold_reasoning},
        },
        "tickers": tickers or {},
        "watchlist": [],
        "prices": {},
    }


class TestApiDecisions:
    def _write_journal(self, tmp_data, entries):
        lines = [json.dumps(e) for e in entries]
        (tmp_data / "layer2_journal.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def test_returns_json_array(self, client, tmp_data):
        self._write_journal(tmp_data, [_sample_journal_entry()])
        with _no_auth():
            resp = client.get("/api/decisions")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        assert len(data) == 1

    def test_empty_journal_returns_empty(self, client, tmp_data):
        with _no_auth():
            resp = client.get("/api/decisions")
        assert resp.status_code == 200
        assert resp.get_json() == []

    def test_newest_first(self, client, tmp_data):
        entries = [
            _sample_journal_entry(ts="2026-02-22T08:00:00+00:00"),
            _sample_journal_entry(ts="2026-02-22T10:00:00+00:00"),
            _sample_journal_entry(ts="2026-02-22T12:00:00+00:00"),
        ]
        self._write_journal(tmp_data, entries)
        with _no_auth():
            resp = client.get("/api/decisions")
        data = resp.get_json()
        assert len(data) == 3
        # Newest first (reversed)
        assert data[0]["ts"] == "2026-02-22T12:00:00+00:00"
        assert data[2]["ts"] == "2026-02-22T08:00:00+00:00"

    def test_filter_by_ticker(self, client, tmp_data):
        entries = [
            _sample_journal_entry(
                ts="2026-02-22T08:00:00+00:00",
                tickers={"BTC-USD": {"outlook": "bullish", "thesis": "test", "conviction": 0.5, "levels": []}},
            ),
            _sample_journal_entry(
                ts="2026-02-22T09:00:00+00:00",
                tickers={"ETH-USD": {"outlook": "bearish", "thesis": "test", "conviction": 0.3, "levels": []}},
            ),
            _sample_journal_entry(
                ts="2026-02-22T10:00:00+00:00",
                tickers={
                    "BTC-USD": {"outlook": "neutral", "thesis": "", "conviction": 0.0, "levels": []},
                    "ETH-USD": {"outlook": "neutral", "thesis": "", "conviction": 0.0, "levels": []},
                },
            ),
        ]
        self._write_journal(tmp_data, entries)
        with _no_auth():
            resp = client.get("/api/decisions?ticker=BTC-USD")
        data = resp.get_json()
        assert len(data) == 2
        # All returned entries should contain BTC-USD in tickers
        for entry in data:
            assert "BTC-USD" in entry["tickers"]

    def test_filter_by_action(self, client, tmp_data):
        entries = [
            _sample_journal_entry(patient_action="BUY", bold_action="HOLD"),
            _sample_journal_entry(patient_action="HOLD", bold_action="SELL"),
            _sample_journal_entry(patient_action="HOLD", bold_action="HOLD"),
        ]
        self._write_journal(tmp_data, entries)
        with _no_auth():
            resp = client.get("/api/decisions?action=BUY")
        data = resp.get_json()
        assert len(data) == 1
        assert data[0]["decisions"]["patient"]["action"] == "BUY"

    def test_filter_by_action_sell(self, client, tmp_data):
        entries = [
            _sample_journal_entry(patient_action="HOLD", bold_action="SELL"),
            _sample_journal_entry(patient_action="HOLD", bold_action="HOLD"),
        ]
        self._write_journal(tmp_data, entries)
        with _no_auth():
            resp = client.get("/api/decisions?action=SELL")
        data = resp.get_json()
        assert len(data) == 1
        assert data[0]["decisions"]["bold"]["action"] == "SELL"

    def test_filter_by_strategy(self, client, tmp_data):
        entries = [
            _sample_journal_entry(patient_action="BUY", bold_action="HOLD"),
            _sample_journal_entry(patient_action="HOLD", bold_action="BUY"),
        ]
        self._write_journal(tmp_data, entries)
        with _no_auth():
            resp = client.get("/api/decisions?strategy=bold&action=BUY")
        data = resp.get_json()
        # Only the entry where bold=BUY should match
        assert len(data) == 1
        assert data[0]["decisions"]["bold"]["action"] == "BUY"

    def test_filter_by_strategy_only(self, client, tmp_data):
        entries = [
            _sample_journal_entry(patient_action="BUY", bold_action="HOLD"),
            _sample_journal_entry(patient_action="HOLD", bold_action="HOLD"),
        ]
        self._write_journal(tmp_data, entries)
        with _no_auth():
            # strategy=patient with no action filter: should match all entries
            # since patient strategy exists in all entries
            resp = client.get("/api/decisions?strategy=patient")
        data = resp.get_json()
        assert len(data) == 2

    def test_limit_parameter(self, client, tmp_data):
        entries = [
            _sample_journal_entry(ts=f"2026-02-22T0{i}:00:00+00:00")
            for i in range(9)
        ]
        self._write_journal(tmp_data, entries)
        with _no_auth():
            resp = client.get("/api/decisions?limit=3")
        data = resp.get_json()
        assert len(data) == 3

    def test_limit_max_500(self, client, tmp_data):
        """Limit is capped at 500 even if a larger value is requested."""
        self._write_journal(tmp_data, [_sample_journal_entry()])
        with _no_auth():
            resp = client.get("/api/decisions?limit=9999")
        # Should not error; just caps at 500
        assert resp.status_code == 200

    def test_invalid_limit_uses_default(self, client, tmp_data):
        self._write_journal(tmp_data, [_sample_journal_entry()])
        with _no_auth():
            resp = client.get("/api/decisions?limit=abc")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == 1

    def test_combined_filters(self, client, tmp_data):
        entries = [
            _sample_journal_entry(
                ts="2026-02-22T08:00:00+00:00",
                patient_action="BUY",
                bold_action="HOLD",
                tickers={"BTC-USD": {"outlook": "bullish", "thesis": "", "conviction": 0.5, "levels": []}},
            ),
            _sample_journal_entry(
                ts="2026-02-22T09:00:00+00:00",
                patient_action="BUY",
                bold_action="HOLD",
                tickers={"ETH-USD": {"outlook": "bullish", "thesis": "", "conviction": 0.5, "levels": []}},
            ),
        ]
        self._write_journal(tmp_data, entries)
        with _no_auth():
            resp = client.get("/api/decisions?action=BUY&strategy=patient&ticker=BTC-USD")
        data = resp.get_json()
        assert len(data) == 1
        assert data[0]["ts"] == "2026-02-22T08:00:00+00:00"

    def test_skips_non_dict_json_lines(self, client, tmp_data):
        lines = [
            json.dumps("raw string line"),
            json.dumps(_sample_journal_entry(ts="2026-02-22T10:00:00+00:00")),
            json.dumps(["bad", "shape"]),
        ]
        (tmp_data / "layer2_journal.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        with _no_auth():
            resp = client.get("/api/decisions")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == 1
        assert data[0]["ts"] == "2026-02-22T10:00:00+00:00"


# ---------------------------------------------------------------------------
# Warrant endpoint
# ---------------------------------------------------------------------------

class TestApiWarrants:
    def test_returns_warrant_data(self, client, tmp_data):
        data = {
            "holdings": {
                "MINI-SILVER": {
                    "units": 100,
                    "entry_price_sek": 340.50,
                    "underlying": "XAG-USD",
                    "leverage": 5,
                    "name": "MINI L SILVER AVA 140",
                }
            },
            "transactions": [],
        }
        (tmp_data / "portfolio_state_warrants.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
        with _no_auth():
            resp = client.get("/api/warrants")
        assert resp.status_code == 200
        result = resp.get_json()
        assert "MINI-SILVER" in result["holdings"]
        assert result["holdings"]["MINI-SILVER"]["leverage"] == 5

    def test_returns_empty_when_missing(self, client, tmp_data):
        with _no_auth():
            resp = client.get("/api/warrants")
        assert resp.status_code == 200
        result = resp.get_json()
        assert result["holdings"] == {}

    def test_returns_empty_holdings(self, client, tmp_data):
        (tmp_data / "portfolio_state_warrants.json").write_text(
            json.dumps({"holdings": {}, "transactions": []}), encoding="utf-8"
        )
        with _no_auth():
            resp = client.get("/api/warrants")
        result = resp.get_json()
        assert result["holdings"] == {}

    def test_requires_auth_when_configured(self, client, tmp_data):
        with patch("dashboard.app._get_dashboard_token", return_value="secret"):
            resp = client.get("/api/warrants")
        assert resp.status_code == 401

    def test_auth_with_token(self, client, tmp_data):
        (tmp_data / "portfolio_state_warrants.json").write_text(
            json.dumps({"holdings": {"X": {"units": 1}}, "transactions": []}),
            encoding="utf-8",
        )
        with patch("dashboard.app._get_dashboard_token", return_value="secret"):
            resp = client.get("/api/warrants?token=secret")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Risk endpoint
# ---------------------------------------------------------------------------

class TestApiRisk:
    def test_returns_risk_data(self, client, tmp_data):
        data = {
            "monte_carlo": {
                "BTC-USD": {
                    "price_usd": 65000,
                    "price_bands_1d": {"p5": 63000, "p25": 64000, "p50": 65000, "p75": 66000, "p95": 67000},
                    "p_stop_hit_1d": 0.12,
                    "expected_return_1d": {"mean_pct": 0.3, "std_pct": 2.0, "skew": -0.1},
                }
            },
            "portfolio_var": {
                "patient": {"var_95_usd": -1200, "cvar_95_usd": -1500, "n_positions": 1},
                "bold": {"var_95_usd": -800, "cvar_95_usd": -1100, "n_positions": 1},
            },
        }
        (tmp_data / "agent_summary_compact.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
        with _no_auth():
            resp = client.get("/api/risk")
        assert resp.status_code == 200
        result = resp.get_json()
        assert "BTC-USD" in result["monte_carlo"]
        assert result["portfolio_var"]["patient"]["var_95_usd"] == -1200

    def test_returns_empty_when_missing(self, client, tmp_data):
        with _no_auth():
            resp = client.get("/api/risk")
        assert resp.status_code == 200
        result = resp.get_json()
        assert result["monte_carlo"] == {}
        assert result["portfolio_var"] == {}

    def test_returns_empty_when_no_mc_section(self, client, tmp_data):
        (tmp_data / "agent_summary_compact.json").write_text(
            json.dumps({"signals": {}}), encoding="utf-8"
        )
        with _no_auth():
            resp = client.get("/api/risk")
        result = resp.get_json()
        assert result["monte_carlo"] == {}

    def test_requires_auth_when_configured(self, client, tmp_data):
        with patch("dashboard.app._get_dashboard_token", return_value="secret"):
            resp = client.get("/api/risk")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Signal heatmap — verify updated signal lists
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Metals endpoint
# ---------------------------------------------------------------------------

class TestApiMetals:
    def test_returns_combined_metals_data(self, client, tmp_data):
        ctx = {
            "timestamp": "2026-03-02T13:31:41+00:00",
            "positions": {
                "gold": {"name": "BULL GULD X8 N", "bid": 979.9, "entry": 972.4, "pnl_pct": 0.77},
                "silver79": {"name": "MINI L SILVER AVA 79", "bid": 64.19, "pnl_pct": -1.44},
                "silver301": {"name": "MINI L SILVER AVA 301", "bid": 19.47, "pnl_pct": -5.94},
            },
            "totals": {"invested": 14910, "current": 14579, "pnl_pct": -2.22},
            "underlying": {"gold": {"price": 5413.1}, "silver": {"price": 94.29}},
            "risk": {"drawdown": {"level": "OK", "current_drawdown_pct": -2.22}},
        }
        dec = {"ts": "2026-03-02T10:52:58+00:00", "tier": 1, "decision": "HOLD"}
        hist = {"period": "2026-01-01 to 2026-03-02", "metals": {}}
        tech = {"technicals": {"1m": {"rsi": 46.2, "macd_line": -0.14}}}

        (tmp_data / "metals_context.json").write_text(json.dumps(ctx), encoding="utf-8")
        (tmp_data / "metals_decisions.jsonl").write_text(json.dumps(dec) + "\n", encoding="utf-8")
        (tmp_data / "metals_history.json").write_text(json.dumps(hist), encoding="utf-8")
        (tmp_data / "silver_analysis.json").write_text(json.dumps(tech), encoding="utf-8")

        with _no_auth():
            resp = client.get("/api/metals")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "context" in data
        assert "decisions" in data
        assert "history" in data
        assert "technicals" in data
        assert data["context"]["totals"]["pnl_pct"] == -2.22
        assert data["context"]["positions"]["gold"]["bid"] == 979.9

    def test_returns_nulls_when_no_files(self, client, tmp_data):
        with _no_auth():
            resp = client.get("/api/metals")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["context"] is None
        assert data["decisions"] == []
        assert data["history"] is None
        assert data["technicals"] is None

    def test_decisions_newest_first(self, client, tmp_data):
        lines = [
            json.dumps({"ts": "2026-03-02T10:00:00+00:00", "tier": 1}),
            json.dumps({"ts": "2026-03-02T11:00:00+00:00", "tier": 2}),
            json.dumps({"ts": "2026-03-02T12:00:00+00:00", "tier": 3}),
        ]
        (tmp_data / "metals_decisions.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        # Provide empty context so endpoint returns 200
        (tmp_data / "metals_context.json").write_text("{}", encoding="utf-8")

        with _no_auth():
            resp = client.get("/api/metals")
        data = resp.get_json()
        # newest first
        assert data["decisions"][0]["ts"] == "2026-03-02T12:00:00+00:00"
        assert data["decisions"][2]["ts"] == "2026-03-02T10:00:00+00:00"

    def test_requires_auth_when_configured(self, client, tmp_data):
        with patch("dashboard.app._get_dashboard_token", return_value="secret"):
            resp = client.get("/api/metals")
        assert resp.status_code == 401

    def test_auth_with_token(self, client, tmp_data):
        (tmp_data / "metals_context.json").write_text('{"positions": {}}', encoding="utf-8")
        with patch("dashboard.app._get_dashboard_token", return_value="secret"):
            resp = client.get("/api/metals?token=secret")
        assert resp.status_code == 200

    def test_partial_data_graceful(self, client, tmp_data):
        """Only context file exists, others missing — should still return 200."""
        ctx = {"positions": {"gold": {"bid": 950}}, "totals": {"pnl_pct": -1.0}}
        (tmp_data / "metals_context.json").write_text(json.dumps(ctx), encoding="utf-8")
        with _no_auth():
            resp = client.get("/api/metals")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["context"]["positions"]["gold"]["bid"] == 950
        assert data["decisions"] == []
        assert data["history"] is None
        assert data["technicals"] is None

    def test_decisions_limit_50(self, client, tmp_data):
        """Only last 50 decisions are returned."""
        lines = [json.dumps({"ts": f"2026-03-02T{i:02d}:00:00+00:00", "tier": 1}) for i in range(60)]
        (tmp_data / "metals_decisions.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (tmp_data / "metals_context.json").write_text("{}", encoding="utf-8")
        with _no_auth():
            resp = client.get("/api/metals")
        data = resp.get_json()
        assert len(data["decisions"]) == 50

    def test_builds_fallback_context_from_live_files(self, client, tmp_data):
        decision = {
            "ts": "2026-03-11T12:01:56+00:00",
            "check_count": 60,
            "tier": 3,
            "trigger": "EMERGENCY drawdown breached: -41.3%",
            "positions": {
                "silver301": {
                    "name": "MINI L SILVER AVA 301",
                    "units": 1105,
                    "entry": 13.921176,
                    "bid": 12.03,
                    "stop": 13.23,
                    "pnl_pct": -13.58,
                    "from_peak_pct": -5.57,
                    "dist_stop_pct": -9.98,
                }
            },
            "risk": {"drawdown_pct": -41.32},
            "llm": {
                "XAG-USD": {
                    "consensus_action": "BUY",
                    "consensus_conf": 1.0,
                    "chronos_1h": "up",
                    "chronos_1h_pct_move": 0.288,
                }
            },
        }
        signal = {
            "ts": "2026-03-11T12:01:56+00:00",
            "check": 60,
            "prices": {"silver301": 12.03, "silver301_und": 86.21, "XAG-USD": 86.21},
            "llm": {
                "XAG-USD": {
                    "consensus_action": "BUY",
                    "consensus_conf": 1.0,
                    "chronos_1h": "up",
                    "chronos_1h_pct_move": 0.288,
                }
            },
            "triggered": True,
            "trigger_reasons": ["EMERGENCY drawdown breached: -41.3%"],
        }
        history_point = {
            "ts": "2026-03-11T12:01:54+00:00",
            "total_value": 13293.1,
            "total_invested": 15382.9,
            "pnl_pct": -13.58,
            "positions": {"silver301": {"bid": 12.03, "value": 13293.1, "pnl_pct": -13.58}},
        }
        tech = {
            "context": {"gold_price": 5179.79},
            "price": {"current": 86.21},
            "technicals": {"1m": {"rsi": 51.6}},
        }
        positions_state = {
            "silver301": {"active": True, "units": 1105, "entry": 13.921176, "stop": 13.23}
        }

        (tmp_data / "metals_decisions.jsonl").write_text(json.dumps(decision) + "\n", encoding="utf-8")
        (tmp_data / "metals_signal_log.jsonl").write_text(json.dumps(signal) + "\n", encoding="utf-8")
        (tmp_data / "metals_value_history.jsonl").write_text(json.dumps(history_point) + "\n", encoding="utf-8")
        (tmp_data / "silver_analysis.json").write_text(json.dumps(tech), encoding="utf-8")
        (tmp_data / "metals_positions_state.json").write_text(json.dumps(positions_state), encoding="utf-8")

        with _no_auth():
            resp = client.get("/api/metals")

        assert resp.status_code == 200
        data = resp.get_json()
        context = data["context"]
        assert context["positions"]["silver301"]["bid"] == 12.03
        assert context["totals"]["current"] == 13293.0
        assert context["underlying"]["gold"]["price"] == 5179.79
        assert context["signals"]["forecast_signals"]["XAG-USD"]["action"] == "BUY"
        assert context["llm_predictions"]["predictions"]["XAG-USD"]["chronos_1h"]["pct_move"] == 0.288
        assert context["risk"]["trade_guards"]["status"] == "warnings"

    def test_merges_fallback_into_partial_context(self, client, tmp_data):
        partial_context = {
            "totals": {"pnl_pct": -1.0},
            "risk": {"drawdown": {"level": "WARNING"}},
        }
        decision = {
            "ts": "2026-03-11T12:01:56+00:00",
            "check_count": 60,
            "tier": 3,
            "trigger": "EMERGENCY drawdown breached: -41.3%",
            "positions": {
                "silver301": {
                    "name": "MINI L SILVER AVA 301",
                    "units": 1105,
                    "entry": 13.921176,
                    "bid": 12.03,
                    "stop": 13.23,
                    "pnl_pct": -13.58,
                }
            },
            "risk": {"drawdown_pct": -41.32},
        }
        signal = {
            "ts": "2026-03-11T12:01:56+00:00",
            "check": 60,
            "prices": {"silver301": 12.03, "XAG-USD": 86.21},
            "triggered": False,
            "trigger_reasons": [],
        }
        history_point = {
            "ts": "2026-03-11T12:01:54+00:00",
            "total_value": 13293.1,
            "total_invested": 15382.9,
            "pnl_pct": -13.58,
            "positions": {"silver301": {"bid": 12.03, "value": 13293.1, "pnl_pct": -13.58}},
        }
        tech = {"context": {"gold_price": 5179.79}, "price": {"current": 86.21}}
        positions_state = {
            "silver301": {"active": True, "units": 1105, "entry": 13.921176, "stop": 13.23}
        }

        (tmp_data / "metals_context.json").write_text(json.dumps(partial_context), encoding="utf-8")
        (tmp_data / "metals_decisions.jsonl").write_text(json.dumps(decision) + "\n", encoding="utf-8")
        (tmp_data / "metals_signal_log.jsonl").write_text(json.dumps(signal) + "\n", encoding="utf-8")
        (tmp_data / "metals_value_history.jsonl").write_text(json.dumps(history_point) + "\n", encoding="utf-8")
        (tmp_data / "silver_analysis.json").write_text(json.dumps(tech), encoding="utf-8")
        (tmp_data / "metals_positions_state.json").write_text(json.dumps(positions_state), encoding="utf-8")

        with _no_auth():
            resp = client.get("/api/metals")

        assert resp.status_code == 200
        data = resp.get_json()
        context = data["context"]
        assert context["totals"]["pnl_pct"] == -1.0
        assert context["totals"]["current"] == 13293.0
        assert context["positions"]["silver301"]["entry"] == 13.921176
        assert context["risk"]["drawdown"]["level"] == "WARNING"
        assert context["risk"]["drawdown"]["current_drawdown_pct"] == -41.32


class TestApiGoldDigger:
    def test_normalizes_compact_state_log_and_trades(self, client, tmp_data):
        state = {
            "equity_sek": 100000.0,
            "daily_pnl": 120.0,
            "daily_trades": 1,
            "last_poll_time": "2026-03-11T12:01:00+00:00",
            "halted": False,
            "position": {
                "quantity": 250,
                "avg_price": 9.5,
                "stop": 8.9,
                "take_profit_price": 10.4,
            },
        }
        log_lines = [
            {
                "ts": "2026-03-11T11:59:00+00:00",
                "gold": 5192.9,
                "usdsek": 9.11,
                "S": 0.4,
                "z_g": 0.4,
                "z_f": 0.1,
                "z_y": -0.1,
                "cert_bid": 9.6,
            },
            {
                "ts": "2026-03-11T12:00:00+00:00",
                "gold": 5193.1,
                "usdsek": 9.11,
                "S": 0.9,
                "z_g": 0.9,
                "z_f": 0.0,
                "z_y": 0.0,
                "cert_bid": 9.7,
            },
            {
                "ts": "2026-03-11T12:01:00+00:00",
                "gold": 5193.4,
                "usdsek": 9.11,
                "S": 1.2,
                "z_g": 1.2,
                "z_f": 0.0,
                "z_y": 0.0,
                "cert_bid": 9.8,
            },
        ]
        trade = {
            "ts": "2026-03-11T12:01:05+00:00",
            "action": "BUY",
            "quantity": 250,
            "price_sek": 9.8,
            "composite_s": 1.2,
            "reason": "entry threshold met",
        }

        (tmp_data / "golddigger_state.json").write_text(json.dumps(state), encoding="utf-8")
        (tmp_data / "golddigger_log.jsonl").write_text(
            "\n".join(json.dumps(line) for line in log_lines) + "\n",
            encoding="utf-8",
        )
        (tmp_data / "golddigger_trades.jsonl").write_text(json.dumps(trade) + "\n", encoding="utf-8")

        with _no_auth(), patch(
            "dashboard.app._get_config",
            return_value={
                "golddigger": {
                    "theta_in": 0.7,
                    "theta_out": 0.1,
                    "confirm_polls": 3,
                    "poll_seconds": 5,
                    "max_daily_trades": 10,
                    "risk_fraction": 0.005,
                    "max_notional_fraction": 0.10,
                    "leverage": 20.0,
                    "spread_max": 0.02,
                }
            },
        ):
            resp = client.get("/api/golddigger")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["state"]["composite_score"] == 1.2
        assert data["state"]["z_gold"] == 1.2
        assert data["state"]["confirm_count"] == 2
        assert data["state"]["position"]["shares"] == 250
        assert data["state"]["position"]["side"] == "BUY"
        assert data["log"][0]["composite_score"] == 1.2
        assert data["trades"][0]["shares"] == 250
        assert data["trades"][0]["total_sek"] == 2450.0
        assert data["trades"][0]["composite_score"] == 1.2

    def test_returns_empty_payload_when_files_missing(self, client, tmp_data):
        with _no_auth():
            resp = client.get("/api/golddigger")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["state"] is None
        assert data["log"] == []
        assert data["trades"] == []


class TestSignalHeatmapUpdated:
    def test_core_signals_includes_custom_lora(self, client, tmp_data):
        summary = {
            "signals": {
                "BTC-USD": {
                    "extra": {
                        "_votes": {"rsi": "BUY", "custom_lora": "HOLD", "trend": "SELL"}
                    }
                }
            }
        }
        (tmp_data / "agent_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        with _no_auth():
            resp = client.get("/api/signal-heatmap")
        result = resp.get_json()
        assert "custom_lora" in result["core_signals"]
        assert len(result["core_signals"]) == 11
        assert len(result["enhanced_signals"]) == 19


# ---------------------------------------------------------------------------
# BUG-130: TTL Cache
# ---------------------------------------------------------------------------


class TestDashboardCache:
    """Verify dashboard file reads are cached with TTL."""

    def test_cached_json_returns_same_object(self, tmp_path):
        """Repeated reads within TTL return cached result."""
        from dashboard.app import _cache, _read_json

        f = tmp_path / "cached.json"
        f.write_text('{"val": 1}', encoding="utf-8")

        _cache.clear()
        result1 = _read_json(f)
        f.write_text('{"val": 2}', encoding="utf-8")
        result2 = _read_json(f)  # Should still return cached {"val": 1}

        assert result1 == {"val": 1}
        assert result2 == {"val": 1}  # Cached, not re-read

    def test_cache_expires_after_ttl(self, tmp_path):
        """After TTL, file is re-read."""
        import time as _time

        from dashboard.app import _cache, _cached_read
        from portfolio.file_utils import load_json as _lj

        f = tmp_path / "expire.json"
        f.write_text('{"v": "old"}', encoding="utf-8")

        _cache.clear()
        r1 = _cached_read(f"test:{f}", 0.1, lambda: _lj(f))
        assert r1 == {"v": "old"}

        f.write_text('{"v": "new"}', encoding="utf-8")
        _time.sleep(0.15)

        r2 = _cached_read(f"test:{f}", 0.1, lambda: _lj(f))
        assert r2 == {"v": "new"}

    def test_cached_jsonl_with_different_limits(self, tmp_path):
        """Different limit params produce different cache entries."""
        from dashboard.app import _cache, _read_jsonl

        f = tmp_path / "multi.jsonl"
        lines = "\n".join(f'{{"i": {i}}}' for i in range(10))
        f.write_text(lines, encoding="utf-8")

        _cache.clear()
        r3 = _read_jsonl(f, limit=3)
        r5 = _read_jsonl(f, limit=5)

        assert len(r3) == 3
        assert len(r5) == 5
