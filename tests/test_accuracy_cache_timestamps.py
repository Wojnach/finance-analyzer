"""Tests for per-horizon accuracy cache timestamps (BUG-133).

Verifies that writing cache for one horizon doesn't make stale data for
other horizons appear fresh.
"""

import json
import time
from unittest import mock

import pytest

from portfolio.accuracy_stats import (
    ACCURACY_CACHE_TTL,
    load_cached_accuracy,
    write_accuracy_cache,
)


@pytest.fixture()
def cache_file(tmp_path):
    """Redirect accuracy cache to temp file."""
    cache_path = tmp_path / "accuracy_cache.json"
    with mock.patch("portfolio.accuracy_stats.ACCURACY_CACHE_FILE", cache_path):
        yield cache_path


class TestPerHorizonTimestamps:
    """BUG-133: Accuracy cache uses per-horizon timestamps."""

    def test_write_creates_per_horizon_timestamp(self, cache_file):
        """write_accuracy_cache stores time_{horizon} key."""
        data = {"rsi": {"accuracy": 0.65, "total": 100, "correct": 65, "pct": 65.0}}
        write_accuracy_cache("1d", data)

        raw = json.loads(cache_file.read_text())
        assert "time_1d" in raw
        assert "1d" in raw
        assert raw["1d"] == data

    def test_write_preserves_legacy_time_key(self, cache_file):
        """write_accuracy_cache also writes legacy 'time' for backwards compat."""
        data = {"rsi": {"accuracy": 0.65, "total": 100, "correct": 65, "pct": 65.0}}
        write_accuracy_cache("1d", data)

        raw = json.loads(cache_file.read_text())
        assert "time" in raw
        assert abs(raw["time"] - raw["time_1d"]) < 1.0  # near-identical

    def test_different_horizons_get_independent_timestamps(self, cache_file):
        """Each horizon gets its own timestamp."""
        data_1d = {"rsi": {"accuracy": 0.65, "total": 100, "correct": 65, "pct": 65.0}}
        data_3h = {"rsi": {"accuracy": 0.55, "total": 50, "correct": 28, "pct": 55.0}}

        write_accuracy_cache("1d", data_1d)
        # Simulate time passing
        raw = json.loads(cache_file.read_text())
        raw["time_1d"] -= 7200  # 2 hours ago
        raw["time"] -= 7200
        cache_file.write_text(json.dumps(raw))

        # Write 3h — this should NOT refresh 1d's timestamp
        write_accuracy_cache("3h", data_3h)

        raw2 = json.loads(cache_file.read_text())
        # 3h should be fresh
        assert time.time() - raw2["time_3h"] < 5
        # 1d should still be old (2 hours ago)
        assert time.time() - raw2["time_1d"] > 7000

    def test_stale_horizon_not_served_when_other_is_fresh(self, cache_file):
        """Loading a stale horizon returns None even if another is fresh."""
        data_1d = {"rsi": {"accuracy": 0.65, "total": 100, "correct": 65, "pct": 65.0}}
        data_3h = {"rsi": {"accuracy": 0.55, "total": 50, "correct": 28, "pct": 55.0}}

        write_accuracy_cache("1d", data_1d)
        write_accuracy_cache("3h", data_3h)

        # Age 1d past TTL
        raw = json.loads(cache_file.read_text())
        raw["time_1d"] -= ACCURACY_CACHE_TTL + 100
        cache_file.write_text(json.dumps(raw))

        # 3h should still be fresh
        assert load_cached_accuracy("3h") is not None
        # 1d should be stale
        assert load_cached_accuracy("1d") is None

    def test_legacy_cache_format_backwards_compat(self, cache_file):
        """Old cache files with only 'time' key still work."""
        legacy = {
            "1d": {"rsi": {"accuracy": 0.65, "total": 100, "correct": 65, "pct": 65.0}},
            "time": time.time(),  # Only legacy key, no time_1d
        }
        cache_file.write_text(json.dumps(legacy))

        result = load_cached_accuracy("1d")
        assert result is not None
        assert result["rsi"]["accuracy"] == 0.65

    def test_missing_cache_file_returns_none(self, cache_file):
        """No cache file returns None gracefully."""
        assert load_cached_accuracy("1d") is None


class TestBlendAccuracyData:
    """ARCH-23: Shared blend_accuracy_data function."""

    def test_blend_both_sources(self):
        from portfolio.accuracy_stats import blend_accuracy_data

        alltime = {"rsi": {"accuracy": 0.60, "total": 200, "correct": 120, "pct": 60.0}}
        recent = {"rsi": {"accuracy": 0.70, "total": 100, "correct": 70, "pct": 70.0}}

        result = blend_accuracy_data(alltime, recent)
        # Default: 70% recent + 30% alltime = 0.7*0.70 + 0.3*0.60 = 0.67
        assert abs(result["rsi"]["accuracy"] - 0.67) < 0.01

    def test_blend_fast_on_divergence(self):
        from portfolio.accuracy_stats import blend_accuracy_data

        alltime = {"rsi": {"accuracy": 0.50, "total": 200, "correct": 100, "pct": 50.0}}
        recent = {"rsi": {"accuracy": 0.80, "total": 100, "correct": 80, "pct": 80.0}}

        # Divergence = 0.30 > 0.15 threshold → fast blend: 90% recent + 10% alltime
        result = blend_accuracy_data(alltime, recent)
        expected = 0.9 * 0.80 + 0.1 * 0.50  # = 0.77
        assert abs(result["rsi"]["accuracy"] - expected) < 0.01

    def test_blend_uses_alltime_when_recent_insufficient(self):
        from portfolio.accuracy_stats import blend_accuracy_data

        alltime = {"rsi": {"accuracy": 0.60, "total": 200, "correct": 120, "pct": 60.0}}
        recent = {"rsi": {"accuracy": 0.80, "total": 10, "correct": 8, "pct": 80.0}}

        result = blend_accuracy_data(alltime, recent)
        # Only 10 recent samples < 50 threshold → use alltime
        assert result["rsi"]["accuracy"] == 0.60

    def test_blend_alltime_only(self):
        from portfolio.accuracy_stats import blend_accuracy_data

        alltime = {"rsi": {"accuracy": 0.60, "total": 200, "correct": 120, "pct": 60.0}}
        result = blend_accuracy_data(alltime, None)
        assert result == alltime

    def test_blend_recent_only(self):
        from portfolio.accuracy_stats import blend_accuracy_data

        recent = {"rsi": {"accuracy": 0.70, "total": 100, "correct": 70, "pct": 70.0}}
        result = blend_accuracy_data(None, recent)
        assert result == recent

    def test_blend_empty_returns_empty(self):
        from portfolio.accuracy_stats import blend_accuracy_data

        assert blend_accuracy_data(None, None) == {}
        assert blend_accuracy_data({}, {}) == {}

    def test_blend_custom_params(self):
        from portfolio.accuracy_stats import blend_accuracy_data

        alltime = {"rsi": {"accuracy": 0.50, "total": 200, "correct": 100, "pct": 50.0}}
        recent = {"rsi": {"accuracy": 0.80, "total": 100, "correct": 80, "pct": 80.0}}

        # Custom: normal_weight=0.5, fast_weight=0.5 (always 50/50)
        result = blend_accuracy_data(
            alltime, recent,
            normal_weight=0.5, fast_weight=0.5,
        )
        expected = 0.5 * 0.80 + 0.5 * 0.50  # = 0.65
        assert abs(result["rsi"]["accuracy"] - expected) < 0.01


class TestLoadJsonOSError:
    """BUG-139: load_json handles OSError gracefully."""

    def test_permission_error_returns_default(self, tmp_path):
        from portfolio.file_utils import load_json

        f = tmp_path / "locked.json"
        f.write_text('{"key": "value"}')

        with mock.patch("pathlib.Path.read_text", side_effect=PermissionError("locked")):
            result = load_json(f)
            assert result is None

    def test_permission_error_with_custom_default(self, tmp_path):
        from portfolio.file_utils import load_json

        f = tmp_path / "locked.json"
        f.write_text('{"key": "value"}')

        with mock.patch("pathlib.Path.read_text", side_effect=PermissionError("locked")):
            result = load_json(f, default={"fallback": True})
            assert result == {"fallback": True}

    def test_oserror_returns_default(self, tmp_path):
        from portfolio.file_utils import load_json

        f = tmp_path / "bad.json"
        f.write_text('{"key": "value"}')

        with mock.patch("pathlib.Path.read_text", side_effect=OSError("disk error")):
            result = load_json(f)
            assert result is None

    def test_normal_read_still_works(self, tmp_path):
        from portfolio.file_utils import load_json

        f = tmp_path / "good.json"
        f.write_text('{"key": "value"}')

        result = load_json(f)
        assert result == {"key": "value"}


class TestEntriesParameter:
    """ARCH-24: Functions accept optional entries= parameter."""

    def test_signal_accuracy_uses_provided_entries(self):
        """signal_accuracy uses provided entries instead of loading from disk."""
        from portfolio.accuracy_stats import signal_accuracy

        entries = [{
            "ts": "2026-03-01T00:00:00",
            "tickers": {
                "BTC-USD": {
                    "signals": {"rsi": "BUY", "macd": "SELL"},
                    "consensus": "BUY",
                },
            },
            "outcomes": {
                "BTC-USD": {
                    "1d": {"change_pct": 2.0},
                },
            },
        }]

        with mock.patch("portfolio.accuracy_stats.load_entries") as mock_load:
            result = signal_accuracy("1d", entries=entries)
            mock_load.assert_not_called()

        assert result["rsi"]["correct"] == 1
        assert result["rsi"]["total"] == 1
        assert result["macd"]["correct"] == 0
        assert result["macd"]["total"] == 1

    def test_consensus_accuracy_uses_provided_entries(self):
        """consensus_accuracy uses provided entries."""
        from portfolio.accuracy_stats import consensus_accuracy

        entries = [{
            "ts": "2026-03-01T00:00:00",
            "tickers": {
                "BTC-USD": {
                    "signals": {"rsi": "BUY"},
                    "consensus": "BUY",
                },
            },
            "outcomes": {
                "BTC-USD": {
                    "1d": {"change_pct": 2.0},
                },
            },
        }]

        with mock.patch("portfolio.accuracy_stats.load_entries") as mock_load:
            result = consensus_accuracy("1d", entries=entries)
            mock_load.assert_not_called()

        assert result["correct"] == 1

    def test_signal_utility_uses_provided_entries(self):
        """signal_utility uses provided entries."""
        from portfolio.accuracy_stats import signal_utility

        entries = [{
            "ts": "2026-03-01T00:00:00",
            "tickers": {
                "BTC-USD": {
                    "signals": {"rsi": "BUY"},
                },
            },
            "outcomes": {
                "BTC-USD": {
                    "1d": {"change_pct": 5.0},
                },
            },
        }]

        with mock.patch("portfolio.accuracy_stats.load_entries") as mock_load:
            result = signal_utility("1d", entries=entries)
            mock_load.assert_not_called()

        assert result["rsi"]["samples"] == 1
        assert result["rsi"]["avg_return"] == 5.0
