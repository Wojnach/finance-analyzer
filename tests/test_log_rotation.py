"""Tests for portfolio.log_rotation — age-based JSONL and size-based text rotation."""

import datetime
import gzip
import json
import os
from pathlib import Path
from unittest import mock

import pytest

import portfolio.log_rotation as lr


@pytest.fixture(autouse=True)
def _isolate_dirs(tmp_path, monkeypatch):
    """Redirect DATA_DIR and ARCHIVE_DIR to tmp_path for every test."""
    data = tmp_path / "data"
    data.mkdir()
    archive = data / "archive"
    monkeypatch.setattr(lr, "DATA_DIR", data)
    monkeypatch.setattr(lr, "ARCHIVE_DIR", archive)
    return data


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, entries: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
    return lines


def _ts(days_ago: int) -> str:
    dt = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days_ago)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# _parse_ts
# ---------------------------------------------------------------------------

class TestParseTs:
    def test_timezone_aware(self):
        result = lr._parse_ts("2026-05-01T12:00:00+00:00")
        assert result is not None
        assert result.tzinfo is not None

    def test_naive_gets_utc(self):
        result = lr._parse_ts("2026-05-01T12:00:00")
        assert result is not None
        assert result.tzinfo is not None

    def test_none_returns_none(self):
        assert lr._parse_ts(None) is None

    def test_garbage_returns_none(self):
        assert lr._parse_ts("not-a-date") is None


# ---------------------------------------------------------------------------
# _file_size_mb
# ---------------------------------------------------------------------------

class TestFileSizeMb:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"x" * 1024)
        assert abs(lr._file_size_mb(f) - 0.000976) < 0.001

    def test_missing_file(self, tmp_path):
        assert lr._file_size_mb(tmp_path / "nope.txt") == 0.0


# ---------------------------------------------------------------------------
# rotate_jsonl
# ---------------------------------------------------------------------------

class TestRotateJsonl:
    def test_missing_file_returns_not_found(self, _isolate_dirs):
        result = lr.rotate_jsonl("nonexistent.jsonl", {"max_age_days": 7})
        assert result["status"] == "not_found"

    def test_all_recent_entries_nothing_to_archive(self, _isolate_dirs):
        path = _isolate_dirs / "test.jsonl"
        _write_jsonl(path, [{"ts": _ts(1), "val": "a"}, {"ts": _ts(2), "val": "b"}])
        result = lr.rotate_jsonl("test.jsonl", {"max_age_days": 7, "ts_field": "ts"})
        assert result["status"] == "nothing_to_archive"
        assert result["kept"] == 2
        assert result["archived"] == 0

    def test_old_entries_archived(self, _isolate_dirs):
        path = _isolate_dirs / "test.jsonl"
        entries = [
            {"ts": _ts(60), "val": "old"},
            {"ts": _ts(1), "val": "recent"},
        ]
        _write_jsonl(path, entries)
        result = lr.rotate_jsonl("test.jsonl", {
            "max_age_days": 30, "ts_field": "ts", "compress": True,
        })
        assert result["status"] == "rotated"
        assert result["archived"] == 1
        assert result["kept"] == 1
        remaining = _read_jsonl(path)
        assert len(remaining) == 1
        assert remaining[0]["val"] == "recent"
        assert (_isolate_dirs / "archive").exists()

    def test_dry_run_does_not_modify(self, _isolate_dirs):
        path = _isolate_dirs / "test.jsonl"
        entries = [{"ts": _ts(60), "val": "old"}, {"ts": _ts(1), "val": "new"}]
        _write_jsonl(path, entries)
        result = lr.rotate_jsonl("test.jsonl", {
            "max_age_days": 30, "ts_field": "ts",
        }, dry_run=True)
        assert result["status"] == "dry_run"
        assert result["archived"] == 1
        remaining = _read_jsonl(path)
        assert len(remaining) == 2

    def test_malformed_json_lines_kept(self, _isolate_dirs):
        path = _isolate_dirs / "test.jsonl"
        with open(path, "w") as f:
            f.write('{"ts": "' + _ts(1) + '", "val": "ok"}\n')
            f.write("NOT VALID JSON\n")
        result = lr.rotate_jsonl("test.jsonl", {"max_age_days": 30, "ts_field": "ts"})
        assert result["parse_failures"] == 1
        assert result["kept"] == 2

    def test_missing_ts_field_entries_kept(self, _isolate_dirs):
        path = _isolate_dirs / "test.jsonl"
        _write_jsonl(path, [{"val": "no_ts"}, {"ts": _ts(1), "val": "has_ts"}])
        result = lr.rotate_jsonl("test.jsonl", {"max_age_days": 30, "ts_field": "ts"})
        assert result["parse_failures"] == 1
        assert result["kept"] == 2

    def test_archive_appends_to_existing_gz(self, _isolate_dirs):
        archive_dir = _isolate_dirs / "archive"
        archive_dir.mkdir()
        month = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=60)).strftime("%Y-%m")
        gz_path = archive_dir / f"test.{month}.jsonl.gz"
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            f.write(json.dumps({"ts": _ts(65), "val": "prev"}) + "\n")

        path = _isolate_dirs / "test.jsonl"
        _write_jsonl(path, [
            {"ts": _ts(60), "val": "old"},
            {"ts": _ts(1), "val": "recent"},
        ])
        lr.rotate_jsonl("test.jsonl", {
            "max_age_days": 30, "ts_field": "ts", "compress": True,
        })
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2

    def test_empty_lines_skipped(self, _isolate_dirs):
        path = _isolate_dirs / "test.jsonl"
        with open(path, "w") as f:
            f.write('{"ts": "' + _ts(1) + '"}\n')
            f.write("\n")
            f.write("   \n")
        result = lr.rotate_jsonl("test.jsonl", {"max_age_days": 30, "ts_field": "ts"})
        assert result["total_lines"] == 1

    def test_size_cap_prunes_oldest_kept_entries(self, _isolate_dirs):
        """After age-based archival, if file exceeds max_size_mb, drop oldest kept."""
        path = _isolate_dirs / "test.jsonl"
        # Production order: oldest at top, newest at bottom (append-only)
        entries = [{"ts": _ts(20 - i), "val": f"entry_{20 - i}", "pad": "x" * 200}
                   for i in range(20)]
        _write_jsonl(path, entries)
        original_size = path.stat().st_size
        max_size_bytes = original_size // 2
        max_size_mb = max_size_bytes / (1024 * 1024)
        result = lr.rotate_jsonl("test.jsonl", {
            "max_age_days": 30, "ts_field": "ts",
            "max_size_mb": max_size_mb, "compress": True,
        })
        assert path.stat().st_size <= max_size_bytes * 1.1
        remaining = _read_jsonl(path)
        assert len(remaining) < 20
        assert remaining[-1]["val"] == "entry_1"

    def test_size_cap_no_prune_when_under(self, _isolate_dirs):
        """Size cap does not prune when file is already under max_size_mb."""
        path = _isolate_dirs / "test.jsonl"
        _write_jsonl(path, [{"ts": _ts(1), "val": "a"}, {"ts": _ts(2), "val": "b"}])
        result = lr.rotate_jsonl("test.jsonl", {
            "max_age_days": 30, "ts_field": "ts", "max_size_mb": 100,
        })
        remaining = _read_jsonl(path)
        assert len(remaining) == 2

    def test_size_cap_preserves_newest_entries(self, _isolate_dirs):
        """Size pruning keeps newest entries (at end of file = most recent)."""
        path = _isolate_dirs / "test.jsonl"
        # Production order: oldest first (day_10), newest last (day_1)
        entries = [{"ts": _ts(10 - i), "val": f"day_{10 - i}", "pad": "x" * 500}
                   for i in range(10)]
        _write_jsonl(path, entries)
        entry_size = path.stat().st_size / 10
        max_size_mb = (entry_size * 3) / (1024 * 1024)
        result = lr.rotate_jsonl("test.jsonl", {
            "max_age_days": 30, "ts_field": "ts",
            "max_size_mb": max_size_mb, "compress": True,
        })
        remaining = _read_jsonl(path)
        assert len(remaining) <= 4
        assert remaining[-1]["val"] == "day_1"

    def test_size_cap_archives_pruned_entries(self, _isolate_dirs):
        """Entries pruned by size cap are archived, not lost."""
        path = _isolate_dirs / "test.jsonl"
        # Production order: oldest first, newest last
        entries = [{"ts": _ts(20 - i), "val": f"entry_{20 - i}", "pad": "x" * 200}
                   for i in range(20)]
        _write_jsonl(path, entries)
        original_size = path.stat().st_size
        max_size_mb = (original_size // 3) / (1024 * 1024)
        result = lr.rotate_jsonl("test.jsonl", {
            "max_age_days": 30, "ts_field": "ts",
            "max_size_mb": max_size_mb, "compress": True,
        })
        assert result.get("size_pruned", 0) > 0
        archive_dir = _isolate_dirs / "archive"
        assert archive_dir.exists()
        archived_files = list(archive_dir.glob("test.*.jsonl.gz"))
        assert len(archived_files) > 0


# ---------------------------------------------------------------------------
# rotate_text
# ---------------------------------------------------------------------------

class TestRotateText:
    def test_missing_file(self, _isolate_dirs):
        result = lr.rotate_text("nope.log", {"max_size_mb": 10})
        assert result["status"] == "not_found"

    def test_under_threshold(self, _isolate_dirs):
        path = _isolate_dirs / "small.log"
        path.write_text("small file")
        result = lr.rotate_text("small.log", {"max_size_mb": 10})
        assert result["status"] == "under_threshold"

    def test_over_threshold_rotates(self, _isolate_dirs):
        path = _isolate_dirs / "big.log"
        path.write_bytes(b"x" * (11 * 1024 * 1024))
        result = lr.rotate_text("big.log", {
            "max_size_mb": 10, "keep_rotations": 3, "compress": True,
        })
        assert result["status"] == "rotated"
        assert path.read_text() == ""
        gz = _isolate_dirs / "archive" / "big.log.1.gz"
        assert gz.exists()

    def test_dry_run(self, _isolate_dirs):
        path = _isolate_dirs / "big.log"
        path.write_bytes(b"x" * (11 * 1024 * 1024))
        result = lr.rotate_text("big.log", {"max_size_mb": 10}, dry_run=True)
        assert result["status"] == "dry_run_would_rotate"
        assert path.stat().st_size > 10 * 1024 * 1024

    def test_rotation_chain_shifts(self, _isolate_dirs):
        archive = _isolate_dirs / "archive"
        archive.mkdir()
        (archive / "big.log.1.gz").write_bytes(b"rotation1")

        path = _isolate_dirs / "big.log"
        path.write_bytes(b"y" * (11 * 1024 * 1024))
        lr.rotate_text("big.log", {
            "max_size_mb": 10, "keep_rotations": 3, "compress": True,
        })
        assert (archive / "big.log.2.gz").exists()
        assert (archive / "big.log.2.gz").read_bytes() == b"rotation1"
        assert (archive / "big.log.1.gz").exists()

    def test_oldest_rotation_deleted(self, _isolate_dirs):
        archive = _isolate_dirs / "archive"
        archive.mkdir()
        (archive / "big.log.1.gz").write_bytes(b"r1")
        (archive / "big.log.2.gz").write_bytes(b"r2")
        (archive / "big.log.3.gz").write_bytes(b"oldest")

        path = _isolate_dirs / "big.log"
        path.write_bytes(b"y" * (11 * 1024 * 1024))
        lr.rotate_text("big.log", {
            "max_size_mb": 10, "keep_rotations": 3, "compress": True,
        })
        assert (archive / "big.log.3.gz").read_bytes() == b"r2"
        assert (archive / "big.log.2.gz").read_bytes() == b"r1"

    def test_no_compress_copies_plain(self, _isolate_dirs):
        path = _isolate_dirs / "plain.log"
        path.write_text("content here")
        result = lr.rotate_text("plain.log", {
            "max_size_mb": 0, "keep_rotations": 2, "compress": False,
        })
        assert result["status"] == "rotated"
        rotation = _isolate_dirs / "archive" / "plain.log.1"
        assert rotation.exists()
        assert rotation.read_text() == "content here"


# ---------------------------------------------------------------------------
# rotate_file dispatcher
# ---------------------------------------------------------------------------

class TestRotateFile:
    def test_routes_jsonl(self, _isolate_dirs):
        path = _isolate_dirs / "test.jsonl"
        _write_jsonl(path, [{"ts": _ts(1)}])
        result = lr.rotate_file("test.jsonl", {"type": "jsonl", "max_age_days": 30, "ts_field": "ts"})
        assert result["status"] == "nothing_to_archive"

    def test_routes_text(self, _isolate_dirs):
        path = _isolate_dirs / "test.log"
        path.write_text("small")
        result = lr.rotate_file("test.log", {"type": "text", "max_size_mb": 10})
        assert result["status"] == "under_threshold"


# ---------------------------------------------------------------------------
# rotate_all
# ---------------------------------------------------------------------------

class TestRotateAll:
    def test_handles_errors_gracefully(self, _isolate_dirs, monkeypatch):
        monkeypatch.setattr(lr, "ROTATION_POLICIES", {
            "good.jsonl": {"type": "jsonl", "max_age_days": 30, "ts_field": "ts"},
            "bad.jsonl": {"type": "jsonl", "max_age_days": 30, "ts_field": "ts"},
        })
        path = _isolate_dirs / "good.jsonl"
        _write_jsonl(path, [{"ts": _ts(1)}])

        with mock.patch.object(lr, "rotate_file", side_effect=[
            {"file": "good.jsonl", "status": "nothing_to_archive"},
            RuntimeError("boom"),
        ]):
            results = lr.rotate_all()
        assert len(results) == 2
        assert results[0]["status"] == "nothing_to_archive"
        assert results[1]["status"] == "error"
        assert "boom" in results[1]["error"]


# ---------------------------------------------------------------------------
# get_file_stats / get_data_dir_size
# ---------------------------------------------------------------------------

class TestStats:
    def test_data_dir_size(self, _isolate_dirs):
        (_isolate_dirs / "a.txt").write_bytes(b"x" * 1024)
        (_isolate_dirs / "b.txt").write_bytes(b"y" * 2048)
        size = lr.get_data_dir_size()
        assert size > 0

    def test_file_stats_with_jsonl(self, _isolate_dirs, monkeypatch):
        monkeypatch.setattr(lr, "ROTATION_POLICIES", {
            "test.jsonl": {"type": "jsonl", "max_age_days": 30, "ts_field": "ts"},
        })
        path = _isolate_dirs / "test.jsonl"
        _write_jsonl(path, [{"ts": _ts(1)}, {"ts": _ts(2)}])
        stats = lr.get_file_stats()
        assert len(stats) == 1
        assert stats[0]["lines"] == 2

    def test_file_stats_missing_file(self, _isolate_dirs, monkeypatch):
        monkeypatch.setattr(lr, "ROTATION_POLICIES", {
            "nope.jsonl": {"type": "jsonl", "max_age_days": 30, "ts_field": "ts"},
        })
        stats = lr.get_file_stats()
        assert stats[0]["size_mb"] == 0
        assert stats[0]["lines"] is None


# ---------------------------------------------------------------------------
# _gzip_file
# ---------------------------------------------------------------------------

class TestGzipFile:
    def test_roundtrip(self, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("hello world")
        dst = tmp_path / "dst.gz"
        lr._gzip_file(src, dst)
        assert dst.exists()
        with gzip.open(dst, "rt") as f:
            assert f.read() == "hello world"
