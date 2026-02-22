"""Tests for portfolio.file_utils — atomic JSON write utility."""
import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch

from portfolio.file_utils import atomic_write_json


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

        with patch("portfolio.file_utils.json.dump", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                atomic_write_json(path, {"new": True})
        # Original file should be unchanged
        assert json.loads(path.read_text(encoding="utf-8")) == {"original": True}

    def test_no_temp_file_left_on_error(self, tmp_path):
        path = tmp_path / "test.json"

        with patch("portfolio.file_utils.json.dump", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
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
