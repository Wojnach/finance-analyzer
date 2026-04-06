"""Tests for portfolio.file_utils — atomic JSON write and shared I/O helpers."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from portfolio.file_utils import atomic_append_jsonl, atomic_write_json, load_json, load_jsonl, require_json


class TestAtomicWriteJson:
    """Tests for atomic_write_json."""

    def test_writes_valid_json(self, tmp_path):
        path = tmp_path / "test.json"
        data = {"key": "value", "num": 42}
        atomic_write_json(path, data)
        assert path.exists()
        assert json.loads(path.read_text(encoding="utf-8")) == data

    def test_overwrites_existing_file(self, tmp_path):
        path = tmp_path / "test.json"
        path.write_text('{"old": true}', encoding="utf-8")
        atomic_write_json(path, {"new": True})
        assert json.loads(path.read_text(encoding="utf-8")) == {"new": True}

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "test.json"
        atomic_write_json(path, {"nested": True})
        assert path.exists()
        assert json.loads(path.read_text(encoding="utf-8")) == {"nested": True}

    def test_custom_indent(self, tmp_path):
        path = tmp_path / "test.json"
        atomic_write_json(path, {"a": 1}, indent=4)
        content = path.read_text(encoding="utf-8")
        assert "    " in content  # 4-space indent

    def test_no_indent(self, tmp_path):
        path = tmp_path / "test.json"
        atomic_write_json(path, {"a": 1}, indent=None)
        content = path.read_text(encoding="utf-8")
        assert "\n" not in content.strip()

    def test_handles_string_path(self, tmp_path):
        path = str(tmp_path / "test.json")
        atomic_write_json(path, {"str_path": True})
        assert json.loads(Path(path).read_text(encoding="utf-8")) == {"str_path": True}

    def test_default_str_serializer(self, tmp_path):
        """default=str handles non-serializable types like Path objects."""
        path = tmp_path / "test.json"
        data = {"path": Path("/some/path")}
        atomic_write_json(path, data)
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["path"] == str(Path("/some/path"))

    def test_no_partial_write_on_error(self, tmp_path):
        path = tmp_path / "test.json"
        path.write_text('{"original": true}', encoding="utf-8")

        with patch("portfolio.file_utils.json.dump", side_effect=OSError("disk full")), pytest.raises(OSError):
            atomic_write_json(path, {"new": True})
        # Original file should be unchanged
        assert json.loads(path.read_text(encoding="utf-8")) == {"original": True}

    def test_no_temp_file_left_on_error(self, tmp_path):
        path = tmp_path / "test.json"

        with patch("portfolio.file_utils.json.dump", side_effect=OSError("disk full")), pytest.raises(OSError):
            atomic_write_json(path, {"new": True})
        # No .tmp files should remain
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_empty_dict(self, tmp_path):
        path = tmp_path / "test.json"
        atomic_write_json(path, {})
        assert json.loads(path.read_text(encoding="utf-8")) == {}

    def test_list_data(self, tmp_path):
        path = tmp_path / "test.json"
        data = [1, 2, 3, "four"]
        atomic_write_json(path, data)
        assert json.loads(path.read_text(encoding="utf-8")) == data

    def test_unicode_content(self, tmp_path):
        path = tmp_path / "test.json"
        data = {"emoji": "hello", "swedish": "åäö"}
        atomic_write_json(path, data)
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result == data

    def test_large_data(self, tmp_path):
        path = tmp_path / "test.json"
        data = {f"key_{i}": f"value_{i}" for i in range(1000)}
        atomic_write_json(path, data)
        result = json.loads(path.read_text(encoding="utf-8"))
        assert len(result) == 1000


class TestLoadJson:
    """Tests for load_json."""

    def test_loads_valid_json(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text('{"key": "value"}', encoding="utf-8")
        assert load_json(path) == {"key": "value"}

    def test_returns_default_when_missing(self, tmp_path):
        path = tmp_path / "missing.json"
        assert load_json(path) is None

    def test_returns_custom_default(self, tmp_path):
        path = tmp_path / "missing.json"
        assert load_json(path, default={}) == {}

    def test_returns_default_on_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        assert load_json(path) is None

    def test_returns_default_on_empty_file(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("", encoding="utf-8")
        assert load_json(path) is None

    def test_accepts_string_path(self, tmp_path):
        path = tmp_path / "test.json"
        path.write_text('{"a": 1}', encoding="utf-8")
        assert load_json(str(path)) == {"a": 1}

    def test_unicode_content(self, tmp_path):
        path = tmp_path / "test.json"
        data = {"swedish": "åäö"}
        path.write_text(json.dumps(data), encoding="utf-8")
        assert load_json(path) == data

    def test_loads_list(self, tmp_path):
        path = tmp_path / "test.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        assert load_json(path) == [1, 2, 3]


class TestLoadJsonl:
    """Tests for load_jsonl."""

    def test_loads_entries(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
        result = load_jsonl(path)
        assert result == [{"a": 1}, {"a": 2}]

    def test_returns_empty_list_when_missing(self, tmp_path):
        path = tmp_path / "missing.jsonl"
        assert load_jsonl(path) == []

    def test_skips_empty_lines(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"a": 1}\n\n\n{"a": 2}\n', encoding="utf-8")
        assert load_jsonl(path) == [{"a": 1}, {"a": 2}]

    def test_skips_malformed_lines(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"a": 1}\nnot json\n{"a": 2}\n', encoding="utf-8")
        result = load_jsonl(path)
        assert result == [{"a": 1}, {"a": 2}]

    def test_limit_keeps_last_n(self, tmp_path):
        path = tmp_path / "data.jsonl"
        lines = [json.dumps({"i": i}) for i in range(10)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = load_jsonl(path, limit=3)
        assert len(result) == 3
        assert result[0] == {"i": 7}
        assert result[2] == {"i": 9}

    def test_limit_none_returns_all(self, tmp_path):
        path = tmp_path / "data.jsonl"
        lines = [json.dumps({"i": i}) for i in range(5)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = load_jsonl(path)
        assert len(result) == 5

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        assert load_jsonl(path) == []

    def test_accepts_string_path(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"a": 1}\n', encoding="utf-8")
        assert load_jsonl(str(path)) == [{"a": 1}]


class TestAtomicAppendJsonl:
    """Tests for atomic_append_jsonl."""

    def test_creates_file_and_appends(self, tmp_path):
        path = tmp_path / "log.jsonl"
        atomic_append_jsonl(path, {"event": "start"})
        assert path.exists()
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"event": "start"}

    def test_appends_to_existing(self, tmp_path):
        path = tmp_path / "log.jsonl"
        path.write_text('{"event": "first"}\n', encoding="utf-8")
        atomic_append_jsonl(path, {"event": "second"})
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"event": "first"}
        assert json.loads(lines[1]) == {"event": "second"}

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "log.jsonl"
        atomic_append_jsonl(path, {"nested": True})
        assert path.exists()

    def test_unicode_content(self, tmp_path):
        path = tmp_path / "log.jsonl"
        atomic_append_jsonl(path, {"text": "åäö"})
        line = path.read_text(encoding="utf-8").strip()
        assert json.loads(line) == {"text": "åäö"}

    def test_multiple_appends(self, tmp_path):
        path = tmp_path / "log.jsonl"
        for i in range(5):
            atomic_append_jsonl(path, {"i": i})
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5
        assert [json.loads(line)["i"] for line in lines] == [0, 1, 2, 3, 4]


class TestAtomicWriteJsonFsync:
    """H34: Verify fsync is called before os.replace."""

    def test_fsync_called_before_replace(self, tmp_path):
        """atomic_write_json must flush+fsync before rename for crash safety."""
        path = tmp_path / "test.json"
        call_order = []

        orig_fsync = __import__("os").fsync
        orig_replace = __import__("os").replace

        def track_fsync(fd):
            call_order.append("fsync")
            return orig_fsync(fd)

        def track_replace(src, dst):
            call_order.append("replace")
            return orig_replace(src, dst)

        with patch("portfolio.file_utils.os.fsync", side_effect=track_fsync), \
             patch("portfolio.file_utils.os.replace", side_effect=track_replace):
            atomic_write_json(path, {"durable": True})

        assert "fsync" in call_order
        assert "replace" in call_order
        assert call_order.index("fsync") < call_order.index("replace")


class TestLoadJsonCorruptionLogging:
    """H35: load_json should log WARNING on corrupt JSON."""

    def test_logs_warning_on_corrupt_json(self, tmp_path, caplog):
        path = tmp_path / "corrupt.json"
        path.write_text("{invalid json", encoding="utf-8")
        import logging
        with caplog.at_level(logging.WARNING, logger="portfolio.file_utils"):
            result = load_json(path)
        assert result is None
        assert "corrupt JSON" in caplog.text

    def test_no_warning_on_missing_file(self, tmp_path, caplog):
        path = tmp_path / "missing.json"
        import logging
        with caplog.at_level(logging.WARNING, logger="portfolio.file_utils"):
            result = load_json(path)
        assert result is None
        assert "corrupt" not in caplog.text


class TestRequireJson:
    """H35: require_json raises on corruption instead of silently returning default."""

    def test_loads_valid_json(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text('{"key": "value"}', encoding="utf-8")
        assert require_json(path) == {"key": "value"}

    def test_raises_on_missing_file(self, tmp_path):
        path = tmp_path / "missing.json"
        with pytest.raises(FileNotFoundError):
            require_json(path)

    def test_raises_on_corrupt_json(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{invalid json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            require_json(path)

    def test_raises_on_empty_file(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            require_json(path)
