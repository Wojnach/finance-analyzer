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
        # (2026-04-16: restored to 0.70 after the 0.75 tuning amplified noise
        # during the W12-W13 crash -> W14-W16 recovery transition.)
        assert abs(result["rsi"]["accuracy"] - 0.67) < 0.01

    def test_blend_fast_on_divergence(self):
        from portfolio.accuracy_stats import blend_accuracy_data

        alltime = {"rsi": {"accuracy": 0.50, "total": 200, "correct": 100, "pct": 50.0}}
        recent = {"rsi": {"accuracy": 0.80, "total": 100, "correct": 80, "pct": 80.0}}

        # Divergence = 0.30 > 0.15 threshold -> fast blend: 90% recent + 10% alltime
        # (2026-04-16: restored from 0.95 back to 0.90 to keep an all-time anchor
        # that damps single-week noise during regime transitions.)
        result = blend_accuracy_data(alltime, recent)
        expected = 0.90 * 0.80 + 0.10 * 0.50  # = 0.77
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
        # 2026-05-10: blend_accuracy_data now annotates with `enabled` flag
        # (DISABLED_SIGNALS lookup). Assert the original fields survive
        # rather than full equality so future enrichments don't break this.
        for k, v in alltime["rsi"].items():
            assert result["rsi"][k] == v

    def test_blend_recent_only(self):
        from portfolio.accuracy_stats import blend_accuracy_data

        recent = {"rsi": {"accuracy": 0.70, "total": 100, "correct": 70, "pct": 70.0}}
        result = blend_accuracy_data(None, recent)
        # See test_blend_alltime_only — same enrichment-tolerant assertion.
        for k, v in recent["rsi"].items():
            assert result["rsi"][k] == v

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


    def test_blend_directional_counts_max_not_sum(self):
        # 2026-06-11 (B6 audit): directional totals now report max(alltime,
        # recent), NOT the sum. `recent` is a strict subset of `alltime`
        # (same log, `since=` cutoff), so summing double-counted the recent
        # window and tripped the 30-sample directional gate early. This test
        # was previously named "..._sum_not_max" and asserted the bug.
        from portfolio.accuracy_stats import blend_accuracy_data

        alltime = {
            "rsi": {
                "accuracy": 0.60, "total": 500, "correct": 300, "pct": 60.0,
                "buy_accuracy": 0.62, "sell_accuracy": 0.58,
                "total_buy": 300, "total_sell": 200,
            }
        }
        recent = {
            "rsi": {
                "accuracy": 0.70, "total": 100, "correct": 70, "pct": 70.0,
                "buy_accuracy": 0.75, "sell_accuracy": 0.65,
                "total_buy": 60, "total_sell": 40,
            }
        }
        result = blend_accuracy_data(alltime, recent)
        assert result["rsi"]["total_buy"] == 300, "max(300, 60), not sum"
        assert result["rsi"]["total_sell"] == 200, "max(200, 40), not sum"


class TestAccuracyCacheMeta:
    """get_accuracy_cache_meta() — 2026-07-18 truth-layer /api/accuracy meta."""

    def test_no_cache_file_returns_null(self, cache_file):
        from portfolio.accuracy_stats import get_accuracy_cache_meta

        assert get_accuracy_cache_meta("1d") == {"updated_ts": None, "age_sec": None}

    def test_reads_time_and_time_consensus_keys(self, cache_file):
        from portfolio.accuracy_stats import get_accuracy_cache_meta

        now = time.time()
        cache_file.write_text(
            json.dumps({"time_1d": now - 100, "time_consensus_1d": now - 50})
        )
        meta = get_accuracy_cache_meta("1d")
        assert meta["updated_ts"] == now - 100
        assert 95 <= meta["age_sec"] <= 105

    def test_picks_older_of_the_two_timestamps(self, cache_file):
        """A stale consensus write shouldn't be masked by a fresher
        per-signal write, or vice versa — meta should reflect the older."""
        from portfolio.accuracy_stats import get_accuracy_cache_meta

        now = time.time()
        cache_file.write_text(
            json.dumps({"time_1d": now - 10, "time_consensus_1d": now - 9000})
        )
        meta = get_accuracy_cache_meta("1d")
        assert meta["updated_ts"] == now - 9000

    def test_missing_horizon_key_returns_null(self, cache_file):
        from portfolio.accuracy_stats import get_accuracy_cache_meta

        cache_file.write_text(json.dumps({"time_3d": time.time()}))
        assert get_accuracy_cache_meta("1d") == {"updated_ts": None, "age_sec": None}


class TestOldestSignalLogTs:
    """get_oldest_signal_log_ts() — cheap first-line read for the
    /api/accuracy unavailable_reason message."""

    @pytest.fixture()
    def signal_log(self, tmp_path):
        log_path = tmp_path / "signal_log.jsonl"
        with mock.patch("portfolio.accuracy_stats.SIGNAL_LOG", log_path):
            yield log_path

    def test_reads_first_line_ts(self, signal_log):
        from portfolio.accuracy_stats import get_oldest_signal_log_ts

        signal_log.write_text(
            '{"ts": "2026-07-11T22:14:02.700652+00:00", "tickers": {}}\n'
            '{"ts": "2026-07-18T00:00:00+00:00", "tickers": {}}\n'
        )
        ts = get_oldest_signal_log_ts()
        assert ts is not None
        # Should be the FIRST line's ts, not the last.
        from datetime import datetime

        assert abs(ts - datetime.fromisoformat("2026-07-11T22:14:02.700652+00:00").timestamp()) < 1

    def test_missing_file_returns_none(self, signal_log):
        from portfolio.accuracy_stats import get_oldest_signal_log_ts

        assert get_oldest_signal_log_ts() is None

    def test_malformed_first_line_returns_none(self, signal_log):
        from portfolio.accuracy_stats import get_oldest_signal_log_ts

        signal_log.write_text("not json\n")
        assert get_oldest_signal_log_ts() is None

    def test_missing_ts_field_returns_none(self, signal_log):
        from portfolio.accuracy_stats import get_oldest_signal_log_ts

        signal_log.write_text('{"tickers": {}}\n')
        assert get_oldest_signal_log_ts() is None


class TestPurgeStaleCacheKey:
    """2026-07-18: per_ticker_consensus_{h} for a horizon whose backing
    outcome rows dropped to zero must not linger in the cache forever."""

    def test_get_or_compute_per_ticker_purges_stale_key_on_zero_result(
        self, cache_file, monkeypatch
    ):
        import portfolio.accuracy_stats as acc_mod

        # Seed a stale per_ticker_consensus_10d left over from when 10d had
        # outcome rows (mirrors the real 2026-07-12 leftover in prod).
        cache_file.write_text(
            json.dumps(
                {
                    "per_ticker_consensus_10d": {"BTC-USD": {"correct": 1, "total": 2}},
                    "time_per_ticker_consensus_10d": time.time() - 999999,
                }
            )
        )
        monkeypatch.setattr(acc_mod, "per_ticker_accuracy", lambda horizon: {})

        result = acc_mod.get_or_compute_per_ticker_accuracy("10d")

        assert result == {}
        raw = json.loads(cache_file.read_text())
        assert "per_ticker_consensus_10d" not in raw
        assert "time_per_ticker_consensus_10d" not in raw

    def test_leaves_other_keys_untouched(self, cache_file, monkeypatch):
        import portfolio.accuracy_stats as acc_mod

        cache_file.write_text(
            json.dumps(
                {
                    "per_ticker_consensus_10d": {"BTC-USD": {"correct": 1, "total": 2}},
                    "time_per_ticker_consensus_10d": time.time(),
                    "per_ticker_consensus_1d": {"BTC-USD": {"correct": 5, "total": 5}},
                    "time_per_ticker_consensus_1d": time.time(),
                }
            )
        )
        monkeypatch.setattr(acc_mod, "per_ticker_accuracy", lambda horizon: {})

        acc_mod.get_or_compute_per_ticker_accuracy("10d")

        raw = json.loads(cache_file.read_text())
        assert raw["per_ticker_consensus_1d"] == {"BTC-USD": {"correct": 5, "total": 5}}

    def test_nonzero_result_still_writes_normally(self, cache_file, monkeypatch):
        import portfolio.accuracy_stats as acc_mod

        result_data = {"BTC-USD": {"correct": 3, "total": 5}}
        monkeypatch.setattr(acc_mod, "per_ticker_accuracy", lambda horizon: result_data)

        result = acc_mod.get_or_compute_per_ticker_accuracy("1d")

        assert result == result_data
        raw = json.loads(cache_file.read_text())
        assert raw["per_ticker_consensus_1d"] == result_data


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
        # 2026-06-10 (audit batch 2): macd is in DISABLED_SIGNALS, so its
        # untagged-row vote lands in the shadow bucket, not the headline.
        assert result["macd"]["total"] == 0
        assert result["macd"]["shadow_total"] == 1
        assert result["macd"]["shadow_correct"] == 0

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
