"""Tests for seasonality profile updater."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest


class TestUpdateSeasonalityProfiles:
    @patch("portfolio.seasonality_updater._fetch_hourly_klines")
    @patch("portfolio.seasonality_updater.save_profiles")
    def test_computes_and_saves_profiles(self, mock_save, mock_fetch):
        from portfolio.seasonality_updater import update_seasonality_profiles
        import numpy as np

        # Generate 10 days of hourly data
        idx = pd.date_range("2026-01-01", periods=240, freq="1h", tz="UTC")
        np.random.seed(42)
        prices = 30.0 + np.cumsum(np.random.randn(240) * 0.01)
        df = pd.DataFrame({"close": prices}, index=idx)
        mock_fetch.return_value = df

        result = update_seasonality_profiles(["XAG-USD"])
        assert "XAG-USD" in result
        assert len(result["XAG-USD"]) == 24
        mock_save.assert_called_once()

    @patch("portfolio.seasonality_updater._fetch_hourly_klines")
    @patch("portfolio.seasonality_updater.save_profiles")
    def test_handles_fetch_failure(self, mock_save, mock_fetch):
        from portfolio.seasonality_updater import update_seasonality_profiles
        mock_fetch.return_value = None
        result = update_seasonality_profiles(["XAG-USD"])
        assert "XAG-USD" not in result
