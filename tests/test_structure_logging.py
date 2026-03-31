"""Tests verifying that structure.py logs exceptions instead of swallowing them silently.

BUG-115: Previously, all 4 sub-indicator try/except blocks in structure.py
had no logging. This test ensures exceptions are logged via logger.exception().
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pandas as pd

from portfolio.signals.structure import compute_structure_signal


def _make_ohlcv(closes: list[float], *, spread: float = 0.5) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame({
        "open": closes,
        "high": [c + spread for c in closes],
        "low": [c - spread for c in closes],
        "close": closes,
        "volume": [1000] * n,
    })


class TestStructureLogging:
    """Verify that sub-signal failures are logged, not silently swallowed."""

    def test_highlow_exception_logged(self, caplog):
        """When _highlow_breakout raises, the exception should be logged."""
        df = _make_ohlcv([100.0] * 60)
        with patch("portfolio.signals.structure._highlow_breakout",
                    side_effect=ValueError("test error")):
            with caplog.at_level(logging.ERROR, logger="portfolio.signals.structure"):
                result = compute_structure_signal(df)
        assert result["sub_signals"]["high_low_breakout"] == "HOLD"
        assert "high_low_breakout sub-signal failed" in caplog.text

    def test_donchian_exception_logged(self, caplog):
        """When _donchian_breakout raises, the exception should be logged."""
        df = _make_ohlcv([100.0] * 60)
        with patch("portfolio.signals.structure._donchian_breakout",
                    side_effect=RuntimeError("donchian error")):
            with caplog.at_level(logging.ERROR, logger="portfolio.signals.structure"):
                result = compute_structure_signal(df)
        assert result["sub_signals"]["donchian_55"] == "HOLD"
        assert "donchian_55 sub-signal failed" in caplog.text

    def test_rsi_exception_logged(self, caplog):
        """When _rsi_centerline raises, the exception should be logged."""
        df = _make_ohlcv([100.0] * 60)
        with patch("portfolio.signals.structure._rsi_centerline",
                    side_effect=TypeError("rsi error")):
            with caplog.at_level(logging.ERROR, logger="portfolio.signals.structure"):
                result = compute_structure_signal(df)
        assert result["sub_signals"]["rsi_centerline"] == "HOLD"
        assert "rsi_centerline sub-signal failed" in caplog.text

    def test_macd_exception_logged(self, caplog):
        """When _macd_zeroline raises, the exception should be logged."""
        df = _make_ohlcv([100.0] * 60)
        with patch("portfolio.signals.structure._macd_zeroline",
                    side_effect=ZeroDivisionError("macd error")):
            with caplog.at_level(logging.ERROR, logger="portfolio.signals.structure"):
                result = compute_structure_signal(df)
        assert result["sub_signals"]["macd_zeroline"] == "HOLD"
        assert "macd_zeroline sub-signal failed" in caplog.text

    def test_normal_execution_no_error_logs(self, caplog):
        """Normal execution should not produce error logs."""
        df = _make_ohlcv([100.0 + i * 0.1 for i in range(60)])
        with caplog.at_level(logging.ERROR, logger="portfolio.signals.structure"):
            result = compute_structure_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert "sub-signal failed" not in caplog.text
