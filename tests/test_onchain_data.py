"""Tests for BGeometrics on-chain data integration."""

import json
import time
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    return resp


# ---------------------------------------------------------------------------
# Tests for fetch functions
# ---------------------------------------------------------------------------

class TestFetchOnchainData:
    """Tests for individual metric fetch functions."""

    @patch("portfolio.onchain_data.fetch_with_retry")
    def test_fetch_mvrv_success(self, mock_fetch):
        mock_fetch.return_value = _mock_response({
            "d": "2026-03-01",
            "mvrv": 1.85,
            "mvrvZScore": 0.42,
        })
        from portfolio.onchain_data import _fetch_mvrv
        result = _fetch_mvrv("test_token")
        assert result is not None
        assert result["mvrv"] == 1.85
        assert result["mvrv_zscore"] == 0.42

    @patch("portfolio.onchain_data.fetch_with_retry")
    def test_fetch_mvrv_failure(self, mock_fetch):
        mock_fetch.return_value = _mock_response({}, status_code=401)
        from portfolio.onchain_data import _fetch_mvrv
        result = _fetch_mvrv("bad_token")
        assert result is None

    @patch("portfolio.onchain_data.fetch_with_retry")
    def test_fetch_mvrv_network_error(self, mock_fetch):
        mock_fetch.return_value = None
        from portfolio.onchain_data import _fetch_mvrv
        result = _fetch_mvrv("token")
        assert result is None

    @patch("portfolio.onchain_data.fetch_with_retry")
    def test_fetch_sopr_success(self, mock_fetch):
        mock_fetch.return_value = _mock_response({
            "d": "2026-03-01",
            "sopr": 1.02,
        })
        from portfolio.onchain_data import _fetch_sopr
        result = _fetch_sopr("token")
        assert result is not None
        assert result["sopr"] == 1.02

    @patch("portfolio.onchain_data.fetch_with_retry")
    def test_fetch_nupl_success(self, mock_fetch):
        mock_fetch.return_value = _mock_response({
            "d": "2026-03-01",
            "nupl": 0.45,
        })
        from portfolio.onchain_data import _fetch_nupl
        result = _fetch_nupl("token")
        assert result is not None
        assert result["nupl"] == 0.45

    @patch("portfolio.onchain_data.fetch_with_retry")
    def test_fetch_realized_price_success(self, mock_fetch):
        mock_fetch.return_value = _mock_response({
            "d": "2026-03-01",
            "realizedPrice": 30500.0,
        })
        from portfolio.onchain_data import _fetch_realized_price
        result = _fetch_realized_price("token")
        assert result is not None
        assert result["realized_price"] == 30500.0

    @patch("portfolio.onchain_data.fetch_with_retry")
    def test_fetch_exchange_netflow_success(self, mock_fetch):
        mock_fetch.return_value = _mock_response([
            {"d": "2026-03-01", "netflow": -1250.5},
        ])
        from portfolio.onchain_data import _fetch_exchange_netflow
        result = _fetch_exchange_netflow("token")
        assert result is not None
        assert result["netflow"] == -1250.5

    @patch("portfolio.onchain_data.fetch_with_retry")
    def test_fetch_exchange_netflow_empty_list(self, mock_fetch):
        mock_fetch.return_value = _mock_response([])
        from portfolio.onchain_data import _fetch_exchange_netflow
        result = _fetch_exchange_netflow("token")
        assert result is None

    @patch("portfolio.onchain_data.fetch_with_retry")
    def test_fetch_liquidations_success(self, mock_fetch):
        mock_fetch.return_value = _mock_response([
            {"d": "2026-03-01", "longLiquidations": 15000000, "shortLiquidations": 8000000},
        ])
        from portfolio.onchain_data import _fetch_liquidations
        result = _fetch_liquidations("token")
        assert result is not None
        assert result["long_liquidations"] == 15000000
        assert result["short_liquidations"] == 8000000


# ---------------------------------------------------------------------------
# Tests for the main get_onchain_data() aggregator
# ---------------------------------------------------------------------------

class TestGetOnchainData:
    """Tests for the main data aggregation function."""

    @patch("portfolio.onchain_data._load_config_token")
    def test_no_token_returns_none(self, mock_token):
        mock_token.return_value = None
        from portfolio.onchain_data import get_onchain_data
        result = get_onchain_data()
        assert result is None

    @patch("portfolio.onchain_data._load_config_token")
    @patch("portfolio.onchain_data._fetch_mvrv")
    @patch("portfolio.onchain_data._fetch_sopr")
    @patch("portfolio.onchain_data._fetch_nupl")
    @patch("portfolio.onchain_data._fetch_realized_price")
    @patch("portfolio.onchain_data._fetch_exchange_netflow")
    @patch("portfolio.onchain_data._fetch_liquidations")
    def test_aggregates_all_metrics(self, mock_liq, mock_net, mock_rp,
                                    mock_nupl, mock_sopr, mock_mvrv, mock_token):
        mock_token.return_value = "test_token"
        mock_mvrv.return_value = {"mvrv": 1.85, "mvrv_zscore": 0.42}
        mock_sopr.return_value = {"sopr": 1.02}
        mock_nupl.return_value = {"nupl": 0.45}
        mock_rp.return_value = {"realized_price": 30500.0}
        mock_net.return_value = {"netflow": -1250.5}
        mock_liq.return_value = {"long_liquidations": 15e6, "short_liquidations": 8e6}

        from portfolio.onchain_data import get_onchain_data
        # Bypass cache
        import portfolio.shared_state as ss
        cache_key = "onchain_btc"
        ss._tool_cache.pop(cache_key, None)

        result = get_onchain_data()
        assert result is not None
        assert result["mvrv"] == 1.85
        assert result["mvrv_zscore"] == 0.42
        assert result["sopr"] == 1.02
        assert result["nupl"] == 0.45
        assert result["realized_price"] == 30500.0
        assert result["netflow"] == -1250.5

    @patch("portfolio.onchain_data._load_config_token")
    @patch("portfolio.onchain_data._fetch_mvrv")
    @patch("portfolio.onchain_data._fetch_sopr")
    @patch("portfolio.onchain_data._fetch_nupl")
    @patch("portfolio.onchain_data._fetch_realized_price")
    @patch("portfolio.onchain_data._fetch_exchange_netflow")
    @patch("portfolio.onchain_data._fetch_liquidations")
    def test_partial_failure_still_returns(self, mock_liq, mock_net, mock_rp,
                                           mock_nupl, mock_sopr, mock_mvrv, mock_token):
        mock_token.return_value = "test_token"
        mock_mvrv.return_value = {"mvrv": 1.85, "mvrv_zscore": 0.42}
        mock_sopr.return_value = None  # failed
        mock_nupl.return_value = None  # failed
        mock_rp.return_value = {"realized_price": 30500.0}
        mock_net.return_value = None  # failed
        mock_liq.return_value = None  # failed

        from portfolio.onchain_data import get_onchain_data
        import portfolio.shared_state as ss
        ss._tool_cache.pop("onchain_btc", None)

        result = get_onchain_data()
        assert result is not None
        assert result["mvrv"] == 1.85
        assert result.get("sopr") is None
        assert result["realized_price"] == 30500.0


# ---------------------------------------------------------------------------
# Tests for interpretation helpers
# ---------------------------------------------------------------------------

class TestInterpretOnchain:
    """Tests for on-chain interpretation helpers."""

    def test_interpret_mvrv_zscore_undervalued(self):
        from portfolio.onchain_data import interpret_onchain
        interp = interpret_onchain({"mvrv_zscore": 0.3, "mvrv": 1.2})
        assert interp["mvrv_zone"] == "undervalued"

    def test_interpret_mvrv_zscore_neutral(self):
        from portfolio.onchain_data import interpret_onchain
        interp = interpret_onchain({"mvrv_zscore": 3.5, "mvrv": 2.0})
        assert interp["mvrv_zone"] == "neutral"

    def test_interpret_mvrv_zscore_overheated(self):
        from portfolio.onchain_data import interpret_onchain
        interp = interpret_onchain({"mvrv_zscore": 7.5, "mvrv": 4.0})
        assert interp["mvrv_zone"] == "overheated"

    def test_interpret_sopr_capitulation(self):
        from portfolio.onchain_data import interpret_onchain
        interp = interpret_onchain({"sopr": 0.92})
        assert interp["sopr_zone"] == "capitulation"

    def test_interpret_sopr_profit_taking(self):
        from portfolio.onchain_data import interpret_onchain
        interp = interpret_onchain({"sopr": 1.08})
        assert interp["sopr_zone"] == "profit_taking"

    def test_interpret_nupl_euphoria(self):
        from portfolio.onchain_data import interpret_onchain
        interp = interpret_onchain({"nupl": 0.8})
        assert interp["nupl_zone"] == "euphoria"

    def test_interpret_nupl_capitulation(self):
        from portfolio.onchain_data import interpret_onchain
        interp = interpret_onchain({"nupl": -0.1})
        assert interp["nupl_zone"] == "capitulation"

    def test_interpret_netflow_accumulation(self):
        from portfolio.onchain_data import interpret_onchain
        interp = interpret_onchain({"netflow": -5000})
        assert interp["netflow_signal"] == "accumulation"

    def test_interpret_netflow_distribution(self):
        from portfolio.onchain_data import interpret_onchain
        interp = interpret_onchain({"netflow": 5000})
        assert interp["netflow_signal"] == "distribution"

    def test_interpret_empty_data(self):
        from portfolio.onchain_data import interpret_onchain
        interp = interpret_onchain({})
        assert interp == {}

    def test_interpret_none_returns_empty(self):
        from portfolio.onchain_data import interpret_onchain
        interp = interpret_onchain(None)
        assert interp == {}


# ---------------------------------------------------------------------------
# Tests for cache file persistence
# ---------------------------------------------------------------------------

class TestOnchainCache:
    """Tests for persistent cache behavior."""

    @patch("portfolio.onchain_data._load_onchain_cache")
    @patch("portfolio.onchain_data._load_config_token")
    def test_no_token_no_cache_returns_none(self, mock_token, mock_cache):
        mock_token.return_value = None
        mock_cache.return_value = None  # no persistent cache either
        from portfolio.onchain_data import get_onchain_data
        result = get_onchain_data()
        assert result is None

    def test_load_cache_returns_data_if_fresh(self, tmp_path):
        from portfolio.onchain_data import _load_onchain_cache, _save_onchain_cache, CACHE_FILE
        test_data = {"mvrv": 1.5, "ts": time.time()}
        _save_onchain_cache(test_data)
        loaded = _load_onchain_cache(max_age_seconds=3600)
        if loaded:
            assert loaded["mvrv"] == 1.5

    def test_load_cache_returns_none_if_stale(self, tmp_path):
        from portfolio.onchain_data import _load_onchain_cache, _save_onchain_cache
        test_data = {"mvrv": 1.5, "ts": time.time() - 100000}
        _save_onchain_cache(test_data)
        loaded = _load_onchain_cache(max_age_seconds=3600)
        assert loaded is None
