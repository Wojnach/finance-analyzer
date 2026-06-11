"""Tests for prune_jsonl in file_utils (BUG-59)."""
import json

from portfolio.file_utils import prune_jsonl


def _write_jsonl(path, entries):
    """Helper: write a list of dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _read_jsonl(path):
    """Helper: read JSONL file into list of dicts."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


class TestPruneJsonl:
    """Test the prune_jsonl utility function."""

    def test_prune_removes_oldest(self, tmp_path):
        path = tmp_path / "test.jsonl"
        entries = [{"i": i} for i in range(100)]
        _write_jsonl(path, entries)

        removed = prune_jsonl(path, max_entries=30)

        assert removed == 70
        result = _read_jsonl(path)
        assert len(result) == 30
        # Should keep the most recent (last) entries
        assert result[0] == {"i": 70}
        assert result[-1] == {"i": 99}

    def test_no_pruning_when_under_limit(self, tmp_path):
        path = tmp_path / "test.jsonl"
        entries = [{"i": i} for i in range(10)]
        _write_jsonl(path, entries)

        removed = prune_jsonl(path, max_entries=100)

        assert removed == 0
        result = _read_jsonl(path)
        assert len(result) == 10

    def test_no_pruning_when_at_limit(self, tmp_path):
        path = tmp_path / "test.jsonl"
        entries = [{"i": i} for i in range(50)]
        _write_jsonl(path, entries)

        removed = prune_jsonl(path, max_entries=50)

        assert removed == 0

    def test_missing_file_returns_zero(self, tmp_path):
        path = tmp_path / "nonexistent.jsonl"
        removed = prune_jsonl(path, max_entries=10)
        assert removed == 0

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        removed = prune_jsonl(path, max_entries=10)
        assert removed == 0

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "blanks.jsonl"
        with open(path, "w") as f:
            f.write('{"a":1}\n\n{"a":2}\n\n\n{"a":3}\n')

        removed = prune_jsonl(path, max_entries=2)

        assert removed == 1
        result = _read_jsonl(path)
        assert len(result) == 2
        assert result[0] == {"a": 2}
        assert result[1] == {"a": 3}

    def test_large_file_pruning(self, tmp_path):
        path = tmp_path / "large.jsonl"
        entries = [{"ts": f"2026-01-{i:05d}", "data": "x" * 50} for i in range(10000)]
        _write_jsonl(path, entries)

        removed = prune_jsonl(path, max_entries=5000)

        assert removed == 5000
        result = _read_jsonl(path)
        assert len(result) == 5000
        assert result[0]["ts"] == "2026-01-05000"
        assert result[-1]["ts"] == "2026-01-09999"

    def test_atomic_write_preserves_on_success(self, tmp_path):
        """Verify the file is valid JSON after pruning."""
        path = tmp_path / "atomic.jsonl"
        entries = [{"i": i, "msg": "hello world"} for i in range(200)]
        _write_jsonl(path, entries)

        prune_jsonl(path, max_entries=50)

        result = _read_jsonl(path)
        assert len(result) == 50
        for entry in result:
            assert "i" in entry
            assert "msg" in entry

    def test_default_max_entries(self, tmp_path):
        """Default is 5000 entries."""
        path = tmp_path / "default.jsonl"
        entries = [{"i": i} for i in range(5010)]
        _write_jsonl(path, entries)

        removed = prune_jsonl(path)  # uses default max_entries=5000

        assert removed == 10
        result = _read_jsonl(path)
        assert len(result) == 5000

    def test_malformed_lines_dropped(self, tmp_path):
        """Malformed (partial-write) lines should be dropped during prune."""
        path = tmp_path / "corrupt.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for i in range(10):
                f.write(json.dumps({"i": i}) + "\n")
            # Simulate a partial write (truncated JSON)
            f.write('{"i": 10, "data": "trunc\n')
            for i in range(11, 20):
                f.write(json.dumps({"i": i}) + "\n")

        # 19 valid lines + 1 corrupt = should keep 19 valid, prune to 5
        removed = prune_jsonl(path, max_entries=5)
        result = _read_jsonl(path)
        assert len(result) == 5
        # Should be the last 5 valid entries (i=15..19)
        assert result[0] == {"i": 15}
        assert result[-1] == {"i": 19}

    def test_all_malformed_lines_result_in_empty(self, tmp_path):
        """File with only malformed lines should result in no pruning needed."""
        path = tmp_path / "all_corrupt.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write("not json\n")
            f.write("{broken\n")
            f.write("also bad\n")

        removed = prune_jsonl(path, max_entries=10)
        assert removed == 0  # 0 valid lines <= max_entries


# ---------------------------------------------------------------------------
# 2026-06-11 audit batch 9: prune must reuse the recovery decoder so it keeps
# legacy concatenated-object lines (}{ with no newline) instead of dropping
# them — they carry real data the recovery decoder was added to preserve.
# ---------------------------------------------------------------------------

class TestPruneKeepsRecoverableLines:
    def test_concatenated_object_line_is_kept_and_healed(self, tmp_path):
        path = tmp_path / "concat.jsonl"
        # Two normal lines + one legacy line with two objects concatenated.
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"i": 0}) + "\n")
            f.write(json.dumps({"i": 1}) + json.dumps({"i": 2}) + "\n")
            f.write(json.dumps({"i": 3}) + "\n")

        # 4 logical objects total; prune to 3 -> drop the oldest 1.
        removed = prune_jsonl(path, max_entries=3)
        assert removed == 1

        kept = _read_jsonl(path)
        # The concatenated objects survived AND were split onto their own
        # physical lines (file healed). Oldest ({"i":0}) dropped.
        assert {"i": 1} in kept
        assert {"i": 2} in kept
        assert {"i": 3} in kept
        assert {"i": 0} not in kept
        # Every kept line is now individually parseable (healed).
        assert len(kept) == 3

    def test_concatenated_lines_not_dropped_when_under_limit(self, tmp_path):
        path = tmp_path / "concat2.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"a": 1}) + json.dumps({"a": 2}) + "\n")

        # 2 objects, under limit -> no-op, file untouched, no data loss.
        removed = prune_jsonl(path, max_entries=5000)
        assert removed == 0
        # Original line still recoverable via load_jsonl.
        from portfolio.file_utils import load_jsonl
        objs = load_jsonl(path)
        assert {"a": 1} in objs and {"a": 2} in objs
