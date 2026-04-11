"""Tests for portfolio.portfolio_validator — validate_portfolio_file()
and the underlying integrity checks.

Focused on the A-PR-3 fix: file loading must go through file_utils.load_json
so that concurrent atomic writes from portfolio_mgr cannot trigger spurious
"invalid JSON" errors.
"""

import json
from pathlib import Path

import pytest

from portfolio import portfolio_validator
from portfolio.portfolio_validator import (
    validate_portfolio,
    validate_portfolio_file,
)


@pytest.fixture
def valid_portfolio_dict():
    """A minimal portfolio_state dict that passes validate_portfolio()."""
    return {
        "cash_sek": 500000.0,
        "initial_value_sek": 500000.0,
        "total_fees_sek": 0.0,
        "holdings": {},
        "transactions": [],
    }


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


class TestValidatePortfolioFile:
    """A-PR-3: validate_portfolio_file must use file_utils.load_json,
    not raw open()+json.load()."""

    def test_valid_file_returns_no_errors(self, tmp_path, valid_portfolio_dict):
        path = tmp_path / "portfolio.json"
        _write_json(path, valid_portfolio_dict)
        errs = validate_portfolio_file(str(path))
        assert errs == []

    def test_missing_file_returns_descriptive_error(self, tmp_path):
        path = tmp_path / "does_not_exist.json"
        errs = validate_portfolio_file(str(path))
        assert len(errs) == 1
        assert "not found" in errs[0].lower()

    def test_malformed_json_returns_descriptive_error(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not valid json", encoding="utf-8")
        errs = validate_portfolio_file(str(path))
        assert len(errs) == 1
        assert "invalid" in errs[0].lower() or "unreadable" in errs[0].lower()

    def test_uses_file_utils_load_json(self, tmp_path, valid_portfolio_dict, monkeypatch):
        """A-PR-3 invariant: the function delegates to file_utils.load_json,
        not to raw json.load(). This guards against silent regression to
        the unsafe pattern."""
        path = tmp_path / "portfolio.json"
        _write_json(path, valid_portfolio_dict)

        load_json_calls = []
        original_load_json = portfolio_validator.load_json

        def spy(p, *args, **kwargs):
            load_json_calls.append(str(p))
            return original_load_json(p, *args, **kwargs)

        monkeypatch.setattr(portfolio_validator, "load_json", spy)
        validate_portfolio_file(str(path))
        assert len(load_json_calls) == 1, "validate_portfolio_file did not call file_utils.load_json"
        assert load_json_calls[0].endswith("portfolio.json")

    def test_no_raw_json_load_in_module_source(self):
        """A-PR-3 belt-and-suspenders: scan the source file to ensure no
        raw json.load / open file pattern slipped back in. If a future
        commit reintroduces it, this test fails immediately. Comments
        and docstrings are stripped before scanning so they can mention
        the disallowed pattern textually."""
        src = Path(portfolio_validator.__file__).read_text(encoding="utf-8")
        # Strip comments and docstrings — only check executable code.
        import ast
        tree = ast.parse(src)
        # Walk the AST and forbid any Call to json.load or builtin open
        # whose result is iterated as JSON. We check both name resolution
        # patterns: json.load(...) and bare open(...).
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "load":
                    if isinstance(func.value, ast.Name) and func.value.id == "json":
                        raise AssertionError(
                            f"Raw json.load() call at line {node.lineno} — "
                            "use file_utils.load_json() instead (A-PR-3)"
                        )
                if isinstance(func, ast.Name) and func.id == "open":
                    raise AssertionError(
                        f"Raw open() call at line {node.lineno} — "
                        "use file_utils.load_json() instead (A-PR-3)"
                    )


class TestValidatePortfolioInvariants:
    """Smoke-test the underlying validate_portfolio() so the file-loading
    fix doesn't accidentally regress the actual validation logic."""

    def test_negative_cash_flagged(self, valid_portfolio_dict):
        valid_portfolio_dict["cash_sek"] = -1.0
        errs = validate_portfolio(valid_portfolio_dict)
        assert any("cash" in e.lower() for e in errs)

    def test_minimal_valid_portfolio_passes(self, valid_portfolio_dict):
        errs = validate_portfolio(valid_portfolio_dict)
        assert errs == [], f"Expected no errors, got: {errs}"
