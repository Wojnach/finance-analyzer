"""Tests for dashboard/silver.py — the #silver page's ticker-accuracy API.

accuracy_by_ticker_signal_cached is mocked directly (patched on
dashboard.silver, not via sys.modules — silver.py imports the function at
module load time, so the sys.modules-swap trick used in test_dashboard.py's
TestApiAccuracy wouldn't reach it). See dashboard/silver.py's own note about
the TICKER_ACCURACY_CACHE_FILE staleness footgun this avoids.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import dashboard.silver as silver
from dashboard.app import app

_TOKEN = "test-token-for-silver"
_AUTH_HEADERS = {"Authorization": f"Bearer {_TOKEN}"}


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with patch("dashboard.auth._get_dashboard_token", return_value=_TOKEN):
        with app.test_client() as c:
            yield c


def _no_auth():
    return patch("dashboard.auth._get_dashboard_token", return_value=None)


class TestAuthRequired:
    def test_requires_auth(self, client):
        resp = client.get("/api/silver/accuracy")
        assert resp.status_code == 401


class TestDefaultTicker:
    def test_no_ticker_param_defaults_to_xag(self, client):
        with patch.object(silver, "accuracy_by_ticker_signal_cached", return_value={}):
            resp = client.get("/api/silver/accuracy", headers=_AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.get_json()["ticker"] == "XAG-USD"

    def test_unknown_ticker_is_400(self, client):
        resp = client.get("/api/silver/accuracy?ticker=NOPE-USD", headers=_AUTH_HEADERS)
        assert resp.status_code == 400


class TestHorizons:
    def test_all_horizons_present_with_signals(self, client):
        mock_data = {
            "3h": {"rsi": {"correct": 10, "total": 20, "pct": 50.0}},
            "1d": {
                "rsi": {"correct": 12, "total": 20, "pct": 60.0},
                "bb": {"correct": 5, "total": 10, "pct": 50.0},
            },
            "3d": {},
            "5d": {},
        }

        def _fake(horizon, min_samples=0):
            return {"XAG-USD": mock_data.get(horizon, {})}

        with patch.object(
            silver, "accuracy_by_ticker_signal_cached", side_effect=_fake
        ):
            resp = client.get(
                "/api/silver/accuracy?ticker=XAG-USD", headers=_AUTH_HEADERS
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert set(data["horizons"].keys()) == {"3h", "1d", "3d", "5d"}
        assert data["horizons"]["3h"]["n_signals"] == 1
        assert data["horizons"]["1d"]["n_signals"] == 2
        assert data["horizons"]["1d"]["signals"]["bb"]["pct"] == 50.0

    def test_horizon_with_no_data_keeps_key_with_empty_signals(self, client):
        """10d-style gap: a horizon with zero rows for this ticker must still
        show up as an (empty) key, not vanish — mirrors /api/accuracy's rule
        for horizons with zero outcome samples."""
        with patch.object(silver, "accuracy_by_ticker_signal_cached", return_value={}):
            resp = client.get(
                "/api/silver/accuracy?ticker=XAG-USD", headers=_AUTH_HEADERS
            )
        data = resp.get_json()
        for h in silver.HORIZONS:
            assert h in data["horizons"]
            assert data["horizons"][h]["signals"] == {}
            assert data["horizons"][h]["n_signals"] == 0

    def test_ticker_not_in_result_is_treated_as_empty(self, client):
        """accuracy_by_ticker_signal_cached only includes tickers with at
        least one signal above min_samples — a ticker missing from the dict
        (not just present-with-empty-dict) must not KeyError."""
        with patch.object(
            silver,
            "accuracy_by_ticker_signal_cached",
            return_value={"BTC-USD": {"rsi": {}}},
        ):
            resp = client.get(
                "/api/silver/accuracy?ticker=XAG-USD", headers=_AUTH_HEADERS
            )
        assert resp.status_code == 200
        data = resp.get_json()
        for h in silver.HORIZONS:
            assert data["horizons"][h]["signals"] == {}


class TestUpdatedTs:
    def test_reads_shared_time_from_cache_file(self, client, tmp_path):
        cache_file = tmp_path / "ticker_signal_accuracy_cache.json"
        cache_file.write_text('{"time": 1784380000.0, "1d": {}}', encoding="utf-8")
        with (
            patch.object(
                silver.accuracy_stats, "TICKER_ACCURACY_CACHE_FILE", cache_file
            ),
            patch.object(silver, "accuracy_by_ticker_signal_cached", return_value={}),
        ):
            resp = client.get(
                "/api/silver/accuracy?ticker=XAG-USD", headers=_AUTH_HEADERS
            )
        assert resp.get_json()["updated_ts"] == 1784380000.0

    def test_missing_cache_file_is_null_not_error(self, client, tmp_path):
        with (
            patch.object(
                silver.accuracy_stats,
                "TICKER_ACCURACY_CACHE_FILE",
                tmp_path / "missing.json",
            ),
            patch.object(silver, "accuracy_by_ticker_signal_cached", return_value={}),
        ):
            resp = client.get(
                "/api/silver/accuracy?ticker=XAG-USD", headers=_AUTH_HEADERS
            )
        assert resp.status_code == 200
        assert resp.get_json()["updated_ts"] is None
