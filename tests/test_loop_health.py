"""Tests for portfolio/loop_health.py."""
from __future__ import annotations

import datetime
import json

import pytest

from portfolio import loop_health


def _ts(offset_seconds: float) -> str:
    """ISO timestamp `offset_seconds` ago (negative = past)."""
    base = datetime.datetime(2026, 5, 2, 12, 0, 0, tzinfo=datetime.UTC)
    return (base + datetime.timedelta(seconds=offset_seconds)).isoformat()


@pytest.fixture
def fixed_now():
    return datetime.datetime(2026, 5, 2, 12, 0, 0, tzinfo=datetime.UTC)


# ---------------------------------------------------------------------------
# read_loop_status
# ---------------------------------------------------------------------------
class TestReadLoopStatus:
    def test_missing_file_returns_missing_state(self, tmp_path, fixed_now):
        result = loop_health.read_loop_status("crypto",
                                                tmp_path / "no.heartbeat",
                                                now=fixed_now)
        assert result["state"] == "missing"
        assert result["age_seconds"] is None
        assert result["payload"] is None
        assert result["error"] is None

    def test_fresh_heartbeat_returns_fresh(self, tmp_path, fixed_now):
        path = tmp_path / "hb.json"
        path.write_text(json.dumps({"ts": _ts(-30), "status": "ok",
                                      "cycle": 5}))
        result = loop_health.read_loop_status("crypto", path, now=fixed_now)
        assert result["state"] == "fresh"
        assert result["age_seconds"] == 30.0
        assert result["payload"]["cycle"] == 5

    def test_stale_heartbeat_returns_stale(self, tmp_path, fixed_now):
        path = tmp_path / "hb.json"
        path.write_text(json.dumps({"ts": _ts(-3600), "status": "ok"}))  # 1h old
        result = loop_health.read_loop_status("crypto", path, now=fixed_now)
        assert result["state"] == "stale"
        assert result["age_seconds"] == 3600.0

    def test_custom_threshold_overrides_default(self, tmp_path, fixed_now):
        path = tmp_path / "hb.json"
        path.write_text(json.dumps({"ts": _ts(-100), "status": "ok"}))
        # 100s old, default threshold 300s = fresh
        assert loop_health.read_loop_status(
            "crypto", path, now=fixed_now)["state"] == "fresh"
        # 100s old, threshold 60s = stale
        assert loop_health.read_loop_status(
            "crypto", path, now=fixed_now,
            stale_threshold_seconds=60)["state"] == "stale"

    def test_malformed_json_returns_unparseable(self, tmp_path, fixed_now):
        path = tmp_path / "hb.json"
        path.write_text("{not valid json")
        result = loop_health.read_loop_status("crypto", path, now=fixed_now)
        assert result["state"] == "unparseable"
        assert "json decode" in result["error"]

    def test_missing_ts_field_returns_unparseable(self, tmp_path, fixed_now):
        path = tmp_path / "hb.json"
        path.write_text(json.dumps({"status": "ok", "no_ts_field": True}))
        result = loop_health.read_loop_status("crypto", path, now=fixed_now)
        assert result["state"] == "unparseable"
        assert result["error"] == "no ts field"

    def test_unparseable_ts_returns_unparseable(self, tmp_path, fixed_now):
        path = tmp_path / "hb.json"
        path.write_text(json.dumps({"ts": "not-a-timestamp"}))
        result = loop_health.read_loop_status("crypto", path, now=fixed_now)
        assert result["state"] == "unparseable"
        assert "ts parse" in result["error"]

    def test_z_suffix_timestamp_handled(self, tmp_path, fixed_now):
        """Heartbeats written with Z suffix instead of +00:00 still parse."""
        path = tmp_path / "hb.json"
        ts_z = _ts(-60).replace("+00:00", "Z")
        path.write_text(json.dumps({"ts": ts_z, "status": "ok"}))
        result = loop_health.read_loop_status("crypto", path, now=fixed_now)
        assert result["state"] == "fresh"

    def test_naive_timestamp_treated_as_utc(self, tmp_path, fixed_now):
        """A timezone-naive ts is assumed UTC, not local time."""
        path = tmp_path / "hb.json"
        naive_ts = "2026-05-02T11:59:00"  # 60s before fixed_now (12:00 UTC)
        path.write_text(json.dumps({"ts": naive_ts, "status": "ok"}))
        result = loop_health.read_loop_status("crypto", path, now=fixed_now)
        assert result["state"] == "fresh"
        assert result["age_seconds"] == 60.0


# ---------------------------------------------------------------------------
# read_loop_health (rollup)
# ---------------------------------------------------------------------------
class TestReadLoopHealthRollup:
    def test_all_fresh(self, tmp_path, fixed_now):
        crypto = tmp_path / "crypto.heartbeat"
        oil = tmp_path / "oil.heartbeat"
        crypto.write_text(json.dumps({"ts": _ts(-30), "status": "ok"}))
        oil.write_text(json.dumps({"ts": _ts(-45), "status": "ok"}))

        rollup = loop_health.read_loop_health(
            repo_root=tmp_path,
            files={"crypto": "crypto.heartbeat", "oil": "oil.heartbeat"},
            now=fixed_now,
        )
        assert rollup["any_unhealthy"] is False
        assert rollup["unhealthy"] == []
        assert rollup["loops"]["crypto"]["state"] == "fresh"
        assert rollup["loops"]["oil"]["state"] == "fresh"

    def test_one_stale_flagged(self, tmp_path, fixed_now):
        crypto = tmp_path / "crypto.heartbeat"
        oil = tmp_path / "oil.heartbeat"
        crypto.write_text(json.dumps({"ts": _ts(-30), "status": "ok"}))
        oil.write_text(json.dumps({"ts": _ts(-3600), "status": "ok"}))

        rollup = loop_health.read_loop_health(
            repo_root=tmp_path,
            files={"crypto": "crypto.heartbeat", "oil": "oil.heartbeat"},
            now=fixed_now,
        )
        assert rollup["any_unhealthy"] is True
        assert rollup["unhealthy"] == ["oil"]
        assert rollup["loops"]["oil"]["state"] == "stale"

    def test_missing_files_flagged(self, tmp_path, fixed_now):
        # Neither file exists
        rollup = loop_health.read_loop_health(
            repo_root=tmp_path,
            files={"crypto": "no.heartbeat", "oil": "neither.heartbeat"},
            now=fixed_now,
        )
        assert rollup["any_unhealthy"] is True
        assert set(rollup["unhealthy"]) == {"crypto", "oil"}
        assert rollup["loops"]["crypto"]["state"] == "missing"
        assert rollup["loops"]["oil"]["state"] == "missing"

    def test_default_files_resolved_relative_to_repo_root(self):
        """Sanity: the default DEFAULT_HEARTBEAT_FILES paths exist as keys
        with the expected names. The actual files may or may not exist
        depending on whether loops have run."""
        assert "crypto" in loop_health.DEFAULT_HEARTBEAT_FILES
        assert "oil" in loop_health.DEFAULT_HEARTBEAT_FILES
        assert loop_health.DEFAULT_HEARTBEAT_FILES["crypto"].endswith(
            "crypto_loop.heartbeat")

    def test_rollup_includes_checked_at_and_threshold(self, tmp_path, fixed_now):
        rollup = loop_health.read_loop_health(
            repo_root=tmp_path, files={}, now=fixed_now,
        )
        assert rollup["checked_at"] == fixed_now.isoformat()
        assert rollup["stale_threshold_seconds"] == loop_health.STALE_THRESHOLD_SECONDS
