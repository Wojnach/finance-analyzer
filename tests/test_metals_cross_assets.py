"""Tests for metals cross-asset data fetcher."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from portfolio.shared_state import _tool_cache


@pytest.fixture(autouse=True)
def _clear_cross_cache():
    """Clear cross-asset cache entries between tests."""
    keys = [k for k in _tool_cache if k.startswith("cross_")]
    for k in keys:
        del _tool_cache[k]
    yield
    keys = [k for k in _tool_cache if k.startswith("cross_")]
    for k in keys:
        del _tool_cache[k]


class TestGetCopperData:
    @patch("portfolio.metals_cross_assets._yf_download")
    def test_returns_change_and_price(self, mock_dl):
        from portfolio.metals_cross_assets import get_copper_data
        idx = pd.date_range("2026-03-01", periods=30, freq="B")
        df = pd.DataFrame({"Close": range(400, 430)}, index=idx)
        mock_dl.return_value = df

        result = get_copper_data.__wrapped__()
        assert result is not None
        assert "price" in result
        assert "change_1d_pct" in result
        assert "change_5d_pct" in result
        assert "sma20" in result

    @patch("portfolio.metals_cross_assets._yf_download")
    def test_returns_none_on_failure(self, mock_dl):
        from portfolio.metals_cross_assets import get_copper_data
        mock_dl.return_value = pd.DataFrame()
        result = get_copper_data.__wrapped__()
        assert result is None


class TestGetGVZ:
    @patch("portfolio.metals_cross_assets._yf_download")
    def test_returns_gvz_level(self, mock_dl):
        from portfolio.metals_cross_assets import get_gvz
        idx = pd.date_range("2026-03-01", periods=30, freq="B")
        df = pd.DataFrame({"Close": [18.0] * 29 + [22.0]}, index=idx)
        mock_dl.return_value = df

        result = get_gvz.__wrapped__()
        assert result is not None
        assert result["level"] == pytest.approx(22.0)
        assert "zscore" in result
        assert "change_1d_pct" in result


class TestGoldSilverRatio:
    @patch("portfolio.metals_cross_assets._yf_download")
    def test_computes_ratio(self, mock_dl):
        from portfolio.metals_cross_assets import get_gold_silver_ratio
        idx = pd.date_range("2026-03-01", periods=30, freq="B")
        gold = pd.DataFrame({"Close": [3000.0] * 30}, index=idx)
        silver = pd.DataFrame({"Close": [30.0] * 30}, index=idx)
        mock_dl.side_effect = [gold, silver]

        result = get_gold_silver_ratio.__wrapped__()
        assert result is not None
        assert result["ratio"] == pytest.approx(100.0)
        assert "sma20" in result
        assert "zscore" in result


class TestGetOilData:
    @patch("portfolio.metals_cross_assets._yf_download")
    def test_returns_price_and_change(self, mock_dl):
        from portfolio.metals_cross_assets import get_oil_data
        idx = pd.date_range("2026-03-01", periods=30, freq="B")
        df = pd.DataFrame({"Close": list(range(60, 90))}, index=idx)
        mock_dl.return_value = df

        result = get_oil_data.__wrapped__()
        assert result is not None
        assert "price" in result
        assert "change_1d_pct" in result
        assert "change_5d_pct" in result

    @patch("portfolio.metals_cross_assets._yf_download")
    def test_returns_none_on_failure(self, mock_dl):
        from portfolio.metals_cross_assets import get_oil_data
        mock_dl.return_value = pd.DataFrame()
        result = get_oil_data.__wrapped__()
        assert result is None


class TestGetAllCrossAssetData:
    @patch("portfolio.metals_cross_assets._yf_download")
    def test_includes_oil_key(self, mock_dl):
        from portfolio.metals_cross_assets import get_all_cross_asset_data
        idx = pd.date_range("2026-03-01", periods=30, freq="B")
        df = pd.DataFrame({"Close": [100.0] * 30}, index=idx)
        mock_dl.return_value = df

        result = get_all_cross_asset_data()
        assert "oil" in result


class TestGetSPYReturn:
    @patch("portfolio.metals_cross_assets._yf_download")
    def test_returns_spy_change(self, mock_dl):
        from portfolio.metals_cross_assets import get_spy_return
        idx = pd.date_range("2026-03-01", periods=30, freq="B")
        closes = list(range(500, 530))
        df = pd.DataFrame({"Close": closes}, index=idx)
        mock_dl.return_value = df

        result = get_spy_return.__wrapped__()
        assert result is not None
        assert "change_1d_pct" in result
        assert "change_5d_pct" in result
