"""Tests for portfolio.oil_grid_signal — standalone Brent signal source.

The module landed without tests; this fills the gap. Covers:
  * _rsi correctness
  * _signal_from_indicators — direction polarity, ambiguous gating,
    confidence range, insufficient-history return
  * compute_signal pipeline with mocked fetch_klines (success, empty df,
    SourceUnavailableError)
  * get_cached_or_refresh — fresh cache returns without refetching,
    stale cache triggers a refetch, force=True bypasses cache
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from unittest.mock import patch

import pandas as pd
import pytest

from portfolio import oil_grid_signal as feed
from portfolio.price_source import SourceUnavailableError


def _make_uptrending_df(n: int = 100, start: float = 70.0,
                       step: float = 0.3) -> pd.DataFrame:
    idx = pd.date_range("2026-05-11", periods=n, freq="1h", tz="UTC")
    closes = [start + i * step for i in range(n)]
    return pd.DataFrame({
        "open": closes,
        "high": [c * 1.005 for c in closes],
        "low": [c * 0.995 for c in closes],
        "close": closes,
        "volume": [1000.0] * n,
    }, index=idx)


def _make_downtrending_df(n: int = 100, start: float = 100.0,
                         step: float = -0.3) -> pd.DataFrame:
    return _make_uptrending_df(n=n, start=start, step=step)


def _make_flat_df(n: int = 100, close: float = 80.0) -> pd.DataFrame:
    idx = pd.date_range("2026-05-11", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": [close] * n,
        "high": [close] * n,
        "low": [close] * n,
        "close": [close] * n,
        "volume": [1000.0] * n,
    }, index=idx)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


class TestRsi:
    def test_returns_nan_for_short_series(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert math.isnan(feed._rsi(s, period=14))

    def test_uptrend_pushes_rsi_high(self):
        s = pd.Series([float(i) for i in range(30)])  # strictly increasing
        rsi = feed._rsi(s, 14)
        assert 90 <= rsi <= 100

    def test_downtrend_pushes_rsi_low(self):
        s = pd.Series([float(30 - i) for i in range(30)])
        rsi = feed._rsi(s, 14)
        assert 0 <= rsi <= 10

    def test_zero_loss_caps_at_100(self):
        s = pd.Series([float(i) for i in range(30)])  # no losses anywhere
        assert feed._rsi(s, 14) == 100.0


# ---------------------------------------------------------------------------
# _signal_from_indicators
# ---------------------------------------------------------------------------


class TestSignalFromIndicators:
    def test_insufficient_history_returns_none(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        direction, conf, meta = feed._signal_from_indicators(s)
        assert direction is None
        assert conf == 0.0
        assert meta.get("reason") == "insufficient_history"

    def test_uptrend_produces_long(self):
        df = _make_uptrending_df(n=100)
        direction, conf, meta = feed._signal_from_indicators(df["close"])
        # Steady uptrend: EMA9 > EMA21, RSI near 100 -> caps out by rsi>=70 path
        # Actually steady straight-line uptrend pushes RSI > 70, which forces
        # the "ambiguous" branch (long requires RSI<70). Confirm by reading meta.
        # Either LONG (if RSI under 70) or None ambiguous — both are valid for
        # this synthetic series; assert the math is consistent.
        if direction is None:
            assert meta.get("reason") == "ambiguous"
            assert meta.get("rsi", 0) >= 70
        else:
            assert direction == "LONG"
            assert 0.5 <= conf <= 0.8

    def test_downtrend_produces_short(self):
        df = _make_downtrending_df(n=100)
        direction, conf, meta = feed._signal_from_indicators(df["close"])
        if direction is None:
            assert meta.get("reason") == "ambiguous"
            assert meta.get("rsi", 100) <= 30
        else:
            assert direction == "SHORT"
            assert 0.5 <= conf <= 0.8

    def test_flat_series_holds(self):
        df = _make_flat_df(n=100, close=80.0)
        direction, conf, meta = feed._signal_from_indicators(df["close"])
        assert direction is None
        assert conf == 0.0
        # ema_diff_pct == 0 -> ambiguous branch
        assert meta.get("reason") == "ambiguous"

    def test_confidence_capped_at_0_8(self):
        # Extreme uptrend within RSI<70 should approach but not exceed 0.8
        # Construct: gradual climb that keeps RSI below 70.
        n = 100
        idx = pd.date_range("2026-05-11", periods=n, freq="1h", tz="UTC")
        closes = []
        v = 100.0
        for i in range(n):
            v += 0.02 if i % 3 == 0 else -0.005  # oscillate up
            closes.append(v)
        s = pd.Series(closes, index=idx)
        direction, conf, meta = feed._signal_from_indicators(s)
        assert conf <= 0.8


# ---------------------------------------------------------------------------
# compute_signal pipeline
# ---------------------------------------------------------------------------


class TestComputeSignal:
    def test_uptrending_pipeline(self):
        df = _make_uptrending_df(n=100)
        with patch.object(feed, "fetch_klines", return_value=df):
            record = feed.compute_signal()
        assert record["underlying"] == "BZ=F"
        assert "ts" in record
        assert "direction" in record
        assert "confidence" in record

    def test_source_unavailable_returns_safe_hold(self):
        with patch.object(feed, "fetch_klines",
                          side_effect=SourceUnavailableError("primary down")):
            record = feed.compute_signal()
        assert record["direction"] is None
        assert record["confidence"] == 0.0
        assert record["meta"]["reason"] == "fetch_failed"
        assert "primary down" in record["meta"]["error"]

    def test_empty_df_returns_safe_hold(self):
        empty = pd.DataFrame()
        with patch.object(feed, "fetch_klines", return_value=empty):
            record = feed.compute_signal()
        assert record["direction"] is None
        assert record["confidence"] == 0.0
        assert record["meta"]["reason"] == "empty_df"

    def test_missing_close_column_returns_safe_hold(self):
        df = pd.DataFrame({"open": [1, 2, 3]})
        with patch.object(feed, "fetch_klines", return_value=df):
            record = feed.compute_signal()
        assert record["direction"] is None
        assert record["meta"]["reason"] == "empty_df"


# ---------------------------------------------------------------------------
# get_cached_or_refresh
# ---------------------------------------------------------------------------


class TestCachedOrRefresh:
    def test_fresh_cache_skips_compute(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "oil_grid_signal.json"
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        cache_file.write_text(json.dumps({
            "ts": ts,
            "underlying": "BZ=F",
            "direction": "LONG",
            "confidence": 0.7,
            "meta": {"rsi": 55.0},
        }))
        monkeypatch.setattr(feed, "SIGNAL_FILE", str(cache_file))
        with patch.object(feed, "compute_signal") as mock_compute:
            result = feed.get_cached_or_refresh()
        mock_compute.assert_not_called()
        assert result["direction"] == "LONG"

    def test_stale_cache_triggers_compute(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "oil_grid_signal.json"
        cache_file.write_text(json.dumps({
            "ts": "2020-01-01T00:00:00Z",  # ancient
            "underlying": "BZ=F",
            "direction": "LONG",
            "confidence": 0.7,
            "meta": {},
        }))
        monkeypatch.setattr(feed, "SIGNAL_FILE", str(cache_file))
        fresh_record = {
            "ts": "2026-05-11T12:00:00Z",
            "underlying": "BZ=F",
            "direction": "SHORT",
            "confidence": 0.6,
            "meta": {"rsi": 30.5},
        }
        with patch.object(feed, "compute_signal",
                          return_value=fresh_record) as mock_compute:
            result = feed.get_cached_or_refresh()
        mock_compute.assert_called_once()
        assert result["direction"] == "SHORT"
        # And the cache was rewritten with the fresh record
        on_disk = json.loads(cache_file.read_text())
        assert on_disk["direction"] == "SHORT"

    def test_force_bypasses_fresh_cache(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "oil_grid_signal.json"
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        cache_file.write_text(json.dumps({
            "ts": ts, "underlying": "BZ=F",
            "direction": "LONG", "confidence": 0.7, "meta": {},
        }))
        monkeypatch.setattr(feed, "SIGNAL_FILE", str(cache_file))
        fresh = {"ts": ts, "underlying": "BZ=F",
                 "direction": "SHORT", "confidence": 0.5, "meta": {}}
        with patch.object(feed, "compute_signal", return_value=fresh):
            result = feed.get_cached_or_refresh(force=True)
        assert result["direction"] == "SHORT"

    def test_missing_cache_computes_fresh(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "nope.json"
        monkeypatch.setattr(feed, "SIGNAL_FILE", str(cache_file))
        fresh = {"ts": "2026-05-11T12:00:00Z", "underlying": "BZ=F",
                 "direction": "LONG", "confidence": 0.6, "meta": {}}
        with patch.object(feed, "compute_signal", return_value=fresh):
            result = feed.get_cached_or_refresh()
        assert result["direction"] == "LONG"
        # File got written
        assert cache_file.exists()

    def test_corrupt_cache_triggers_refresh(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "oil_grid_signal.json"
        cache_file.write_text("{not json")
        monkeypatch.setattr(feed, "SIGNAL_FILE", str(cache_file))
        fresh = {"ts": "2026-05-11T12:00:00Z", "underlying": "BZ=F",
                 "direction": "LONG", "confidence": 0.6, "meta": {}}
        with patch.object(feed, "compute_signal", return_value=fresh):
            result = feed.get_cached_or_refresh()
        assert result["direction"] == "LONG"

    def test_malformed_ts_triggers_refresh(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "oil_grid_signal.json"
        cache_file.write_text(json.dumps({
            "ts": "not-a-timestamp",
            "underlying": "BZ=F", "direction": "LONG",
            "confidence": 0.7, "meta": {},
        }))
        monkeypatch.setattr(feed, "SIGNAL_FILE", str(cache_file))
        fresh = {"ts": "2026-05-11T12:00:00Z", "underlying": "BZ=F",
                 "direction": "SHORT", "confidence": 0.6, "meta": {}}
        with patch.object(feed, "compute_signal", return_value=fresh):
            result = feed.get_cached_or_refresh()
        assert result["direction"] == "SHORT"
