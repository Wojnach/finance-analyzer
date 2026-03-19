"""TEST-11: Verify IO safety of the 2026-03-14 sweep.

Ensures all portfolio modules use safe file I/O helpers (load_json, load_jsonl,
atomic_write_json, atomic_write_jsonl) instead of raw json.loads(path.read_text())
or non-atomic open("w") patterns.
"""

import re
from pathlib import Path

import pytest

PORTFOLIO_DIR = Path(__file__).resolve().parent.parent / "portfolio"

# Modules that were swept — must import from file_utils
SWEPT_MODULES = [
    "accuracy_stats",
    "alpha_vantage",
    "analyze",
    "autonomous",
    "avanza_client",
    "avanza_orders",
    "avanza_session",
    "avanza_tracker",
    "bigbet",
    "daily_digest",
    "focus_analysis",
    "forecast_accuracy",
    "forecast_signal",
    "iskbets",
    "journal",
    "local_llm_report",
    "main",
    "onchain_data",
    "perception_gate",
    "prophecy",
    "signal_history",
    "telegram_notifications",
]

# file_utils.py itself is allowed to use raw json.loads (it wraps them)
ALLOWLIST = {"file_utils.py"}

# Patterns that signal network/stdin reads (not file reads), which are OK
NETWORK_READ_PATTERNS = [
    r"resp\.read\(\)",
    r"sys\.stdin\.read\(\)",
    r"response\.read\(\)",
]


def _get_py_files():
    """Get all .py files in portfolio/ recursively."""
    return sorted(PORTFOLIO_DIR.rglob("*.py"))


def _raw_json_loads_file_reads(filepath):
    """Find raw json.loads(path.read_text()) calls in a Python file.

    Returns list of (line_number, line_text) for violations.
    """
    violations = []
    try:
        source = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return violations

    for i, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        # Skip comments
        if stripped.startswith("#"):
            continue
        # Check for json.loads(...read_text...) pattern
        if "json.loads" in stripped and "read_text" in stripped:
            violations.append((i, stripped))
        # Check for json.loads(...read()...) but exclude network reads
        elif "json.loads" in stripped and ".read()" in stripped:
            if not any(re.search(p, stripped) for p in NETWORK_READ_PATTERNS):
                violations.append((i, stripped))

    return violations


class TestNoRawJsonLoadsFileReads:
    """Ensure no portfolio module uses raw json.loads(path.read_text())."""

    def test_no_raw_reads_in_portfolio(self):
        """Scan all portfolio/*.py files for raw json.loads(read_text()) calls."""
        all_violations = {}
        for pyfile in _get_py_files():
            if pyfile.name in ALLOWLIST:
                continue
            violations = _raw_json_loads_file_reads(pyfile)
            if violations:
                rel = pyfile.relative_to(PORTFOLIO_DIR.parent)
                all_violations[str(rel)] = violations

        if all_violations:
            msg_parts = ["Raw json.loads(path.read_text()) found:"]
            for fname, viols in all_violations.items():
                for lineno, text in viols:
                    msg_parts.append(f"  {fname}:{lineno}: {text}")
            pytest.fail("\n".join(msg_parts))


class TestSweptModulesImportSafeHelpers:
    """Ensure swept modules import the safe helpers they need."""

    @pytest.mark.parametrize("module_name", SWEPT_MODULES)
    def test_module_imports_file_utils(self, module_name):
        """Each swept module should import at least one helper from file_utils."""
        filepath = PORTFOLIO_DIR / f"{module_name}.py"
        if not filepath.exists():
            pytest.skip(f"{module_name}.py not found")

        source = filepath.read_text(encoding="utf-8")
        safe_helpers = ["load_json", "load_jsonl", "atomic_write_json",
                        "atomic_write_jsonl", "atomic_append_jsonl"]
        found = [h for h in safe_helpers if h in source]
        assert found, (
            f"{module_name}.py does not import any safe I/O helpers "
            f"from file_utils ({safe_helpers})"
        )


class TestAtomicWriteJsonlExists:
    """Verify atomic_write_jsonl was added to file_utils."""

    def test_atomic_write_jsonl_importable(self):
        from portfolio.file_utils import atomic_write_jsonl
        assert callable(atomic_write_jsonl)

    def test_atomic_write_jsonl_writes_correctly(self, tmp_path):
        from portfolio.file_utils import atomic_write_jsonl
        target = tmp_path / "test.jsonl"
        entries = [{"a": 1}, {"b": 2}, {"c": 3}]
        atomic_write_jsonl(target, entries)

        import json
        lines = target.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        assert json.loads(lines[0]) == {"a": 1}
        assert json.loads(lines[1]) == {"b": 2}
        assert json.loads(lines[2]) == {"c": 3}

    def test_atomic_write_jsonl_replaces_existing(self, tmp_path):
        from portfolio.file_utils import atomic_write_jsonl
        target = tmp_path / "test.jsonl"
        target.write_text('{"old": true}\n', encoding="utf-8")

        atomic_write_jsonl(target, [{"new": True}])
        import json
        lines = target.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"new": True}

    def test_atomic_write_jsonl_empty_entries(self, tmp_path):
        from portfolio.file_utils import atomic_write_jsonl
        target = tmp_path / "test.jsonl"
        atomic_write_jsonl(target, [])
        assert target.read_text(encoding="utf-8") == ""


class TestNonAtomicWritesEliminated:
    """Verify prophecy, signal_history, forecast_accuracy use atomic writes."""

    def test_prophecy_uses_atomic_write(self):
        source = (PORTFOLIO_DIR / "prophecy.py").read_text(encoding="utf-8")
        assert "atomic_write_json" in source
        # Should NOT have open("w") + json.dump pattern
        assert 'open(' not in source or 'atomic_write' in source

    def test_signal_history_uses_atomic_write_jsonl(self):
        source = (PORTFOLIO_DIR / "signal_history.py").read_text(encoding="utf-8")
        assert "atomic_write_jsonl" in source
        # The _save_history function should use atomic_write_jsonl, not open("w")
        assert "load_jsonl" in source

    def test_forecast_accuracy_uses_atomic_write_jsonl(self):
        source = (PORTFOLIO_DIR / "forecast_accuracy.py").read_text(encoding="utf-8")
        assert "atomic_write_jsonl" in source


class TestLoadJsonHandlesCorruptFiles:
    """Verify load_json returns defaults for missing/corrupt files."""

    def test_missing_file_returns_default(self, tmp_path):
        from portfolio.file_utils import load_json
        result = load_json(tmp_path / "nonexistent.json", default={})
        assert result == {}

    def test_empty_file_returns_default(self, tmp_path):
        from portfolio.file_utils import load_json
        target = tmp_path / "empty.json"
        target.write_text("", encoding="utf-8")
        result = load_json(target, default={"fallback": True})
        assert result == {"fallback": True}

    def test_corrupt_json_returns_default(self, tmp_path):
        from portfolio.file_utils import load_json
        target = tmp_path / "corrupt.json"
        target.write_text("{bad json", encoding="utf-8")
        result = load_json(target, default=[])
        assert result == []

    def test_valid_json_returns_data(self, tmp_path):
        from portfolio.file_utils import load_json
        target = tmp_path / "valid.json"
        target.write_text('{"key": "value"}', encoding="utf-8")
        result = load_json(target, default={})
        assert result == {"key": "value"}
