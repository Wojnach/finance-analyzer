"""Tests for VIX fetch via yfinance with regime classification."""

from unittest.mock import patch, MagicMock
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers to build mock DataFrames returned by yf.Ticker("^VIX").history()
# ---------------------------------------------------------------------------

def _make_hist(closes, index=None):
    """Build a DataFrame that looks like yfinance history output."""
    if index is None:
        index = pd.date_range("2026-03-06", periods=len(closes), freq="B")
    df = pd.DataFrame({"Close": closes, "Open": closes, "High": closes, "Low": closes}, index=index)
    return df


def _make_multiindex_hist(closes):
    """Build a MultiIndex-column DataFrame (yfinance sometimes returns this)."""
    index = pd.date_range("2026-03-06", periods=len(closes), freq="B")
    arrays = [["Close", "Open", "High", "Low"], ["^VIX", "^VIX", "^VIX", "^VIX"]]
    cols = pd.MultiIndex.from_arrays(arrays)
    data = [[c, c, c, c] for c in closes]
    return pd.DataFrame(data, index=index, columns=cols)


def _patch_yf_ticker(hist_df):
    """Create a mock for yfinance.Ticker that returns the given history DataFrame."""
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = hist_df
    return patch("yfinance.Ticker", return_value=mock_ticker)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFetchVixReturnStructure:
    """Test that fetch_vix returns a dict with the expected keys."""

    def test_returns_dict_with_expected_keys(self):
        with _patch_yf_ticker(_make_hist([18.0, 19.5, 20.0, 21.0, 22.5])):
            from portfolio.data_collector import fetch_vix
            result = fetch_vix()

        assert result is not None
        assert "value" in result
        assert "prev_close" in result
        assert "change_pct" in result
        assert "regime_hint" in result

    def test_value_matches_last_close(self):
        with _patch_yf_ticker(_make_hist([15.0, 16.0, 17.0, 18.0, 19.5])):
            from portfolio.data_collector import fetch_vix
            result = fetch_vix()

        assert result["value"] == 19.5
        assert result["prev_close"] == 18.0


class TestFetchVixFailure:
    """Test graceful handling of yfinance errors."""

    def test_yfinance_exception_returns_none(self):
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = Exception("network error")
        with patch("yfinance.Ticker", return_value=mock_ticker):
            from portfolio.data_collector import fetch_vix
            result = fetch_vix()

        assert result is None

    def test_empty_dataframe_returns_none(self):
        with _patch_yf_ticker(pd.DataFrame()):
            from portfolio.data_collector import fetch_vix
            result = fetch_vix()

        assert result is None

    def test_none_history_returns_none(self):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = None
        with patch("yfinance.Ticker", return_value=mock_ticker):
            from portfolio.data_collector import fetch_vix
            result = fetch_vix()

        assert result is None


class TestVixRegimeClassification:
    """Test VIX regime hints at boundary values."""

    def test_high_vol_at_30(self):
        with _patch_yf_ticker(_make_hist([28.0, 30.0])):
            from portfolio.data_collector import fetch_vix
            result = fetch_vix()
        assert result["regime_hint"] == "high-vol"

    def test_high_vol_above_30(self):
        with _patch_yf_ticker(_make_hist([40.0, 45.0])):
            from portfolio.data_collector import fetch_vix
            result = fetch_vix()
        assert result["regime_hint"] == "high-vol"

    def test_elevated_at_20(self):
        with _patch_yf_ticker(_make_hist([19.0, 20.0])):
            from portfolio.data_collector import fetch_vix
            result = fetch_vix()
        assert result["regime_hint"] == "elevated"

    def test_elevated_at_29(self):
        with _patch_yf_ticker(_make_hist([28.0, 29.0])):
            from portfolio.data_collector import fetch_vix
            result = fetch_vix()
        assert result["regime_hint"] == "elevated"

    def test_normal_at_15(self):
        with _patch_yf_ticker(_make_hist([14.0, 15.0])):
            from portfolio.data_collector import fetch_vix
            result = fetch_vix()
        assert result["regime_hint"] == "normal"

    def test_normal_at_19(self):
        with _patch_yf_ticker(_make_hist([18.0, 19.0])):
            from portfolio.data_collector import fetch_vix
            result = fetch_vix()
        assert result["regime_hint"] == "normal"

    def test_complacent_below_15(self):
        with _patch_yf_ticker(_make_hist([11.0, 12.0])):
            from portfolio.data_collector import fetch_vix
            result = fetch_vix()
        assert result["regime_hint"] == "complacent"

    def test_complacent_at_14_99(self):
        with _patch_yf_ticker(_make_hist([13.0, 14.99])):
            from portfolio.data_collector import fetch_vix
            result = fetch_vix()
        assert result["regime_hint"] == "complacent"


class TestVixChangePct:
    """Test change_pct calculation."""

    def test_positive_change(self):
        with _patch_yf_ticker(_make_hist([20.0, 22.0])):
            from portfolio.data_collector import fetch_vix
            result = fetch_vix()
        assert result["change_pct"] == 10.0  # (22-20)/20 * 100

    def test_negative_change(self):
        with _patch_yf_ticker(_make_hist([25.0, 20.0])):
            from portfolio.data_collector import fetch_vix
            result = fetch_vix()
        assert result["change_pct"] == -20.0  # (20-25)/25 * 100

    def test_zero_change(self):
        with _patch_yf_ticker(_make_hist([18.0, 18.0])):
            from portfolio.data_collector import fetch_vix
            result = fetch_vix()
        assert result["change_pct"] == 0.0


class TestVixSingleRow:
    """Test single-row DataFrame edge case (no prev_close comparison crash)."""

    def test_single_row_uses_same_as_prev(self):
        with _patch_yf_ticker(_make_hist([22.5])):
            from portfolio.data_collector import fetch_vix
            result = fetch_vix()

        assert result is not None
        assert result["value"] == 22.5
        assert result["prev_close"] == 22.5
        assert result["change_pct"] == 0.0


class TestVixMultiIndexColumns:
    """Test handling of MultiIndex columns from yfinance."""

    def test_multiindex_columns_flattened(self):
        with _patch_yf_ticker(_make_multiindex_hist([18.0, 20.0, 22.0])):
            from portfolio.data_collector import fetch_vix
            result = fetch_vix()

        assert result is not None
        assert result["value"] == 22.0
        assert result["prev_close"] == 20.0


class TestVixZeroPrevClose:
    """Edge case: prev_close is zero (avoid ZeroDivisionError)."""

    def test_zero_prev_close_no_crash(self):
        with _patch_yf_ticker(_make_hist([0.0, 15.0])):
            from portfolio.data_collector import fetch_vix
            result = fetch_vix()

        assert result is not None
        assert result["change_pct"] == 0


class TestVixRounding:
    """Test that values are properly rounded."""

    def test_values_rounded_to_2_decimals(self):
        with _patch_yf_ticker(_make_hist([18.123456, 19.876543])):
            from portfolio.data_collector import fetch_vix
            result = fetch_vix()

        assert result["value"] == 19.88
        assert result["prev_close"] == 18.12
        # change_pct = (19.876543 - 18.123456) / 18.123456 * 100 = 9.6729...
        assert isinstance(result["change_pct"], float)
        assert result["change_pct"] == round(
            (19.876543 - 18.123456) / 18.123456 * 100, 2
        )


class TestMacroHeadlineVix:
    """Test that VIX appears in the macro headline string."""

    def test_vix_in_macro_headline(self):
        from portfolio.reporting import _macro_headline
        summary = {
            "macro": {
                "vix": {"value": 22.5, "change_pct": 3.1, "regime_hint": "elevated"},
            },
            "fear_greed": {},
        }
        headline = _macro_headline(summary)
        assert "VIX 22.5" in headline

    def test_vix_rising_arrow(self):
        from portfolio.reporting import _macro_headline
        summary = {
            "macro": {
                "vix": {"value": 25.0, "change_pct": 5.0, "regime_hint": "elevated"},
            },
            "fear_greed": {},
        }
        headline = _macro_headline(summary)
        # Up arrow for positive change
        assert "VIX 25.0\u2191" in headline

    def test_vix_falling_arrow(self):
        from portfolio.reporting import _macro_headline
        summary = {
            "macro": {
                "vix": {"value": 18.0, "change_pct": -2.0, "regime_hint": "normal"},
            },
            "fear_greed": {},
        }
        headline = _macro_headline(summary)
        assert "VIX 18.0\u2193" in headline

    def test_no_vix_in_macro(self):
        from portfolio.reporting import _macro_headline
        summary = {"macro": {}, "fear_greed": {}}
        headline = _macro_headline(summary)
        assert "VIX" not in headline
