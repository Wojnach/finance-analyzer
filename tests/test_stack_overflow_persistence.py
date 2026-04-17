"""Tests for stack overflow counter persistence in agent_invocation.py."""

import json
from pathlib import Path
from unittest.mock import patch

from portfolio.agent_invocation import (
    _load_stack_overflow_counter,
    _save_stack_overflow_counter,
)


class TestStackOverflowPersistence:
    """Verify load/save of _consecutive_stack_overflows counter."""

    def test_load_returns_zero_when_file_missing(self, tmp_path):
        missing = tmp_path / "stack_overflow_counter.json"
        with patch("portfolio.agent_invocation._STACK_OVERFLOW_FILE", missing):
            assert _load_stack_overflow_counter() == 0

    def test_load_returns_zero_on_corrupt_json(self, tmp_path):
        corrupt = tmp_path / "stack_overflow_counter.json"
        corrupt.write_text("NOT VALID JSON {{{")
        with patch("portfolio.agent_invocation._STACK_OVERFLOW_FILE", corrupt):
            assert _load_stack_overflow_counter() == 0

    def test_save_writes_correct_json(self, tmp_path):
        f = tmp_path / "stack_overflow_counter.json"
        with patch("portfolio.agent_invocation._STACK_OVERFLOW_FILE", f):
            _save_stack_overflow_counter(3)

        data = json.loads(f.read_text())
        assert data["count"] == 3
        assert "updated" in data

    def test_round_trip(self, tmp_path):
        f = tmp_path / "stack_overflow_counter.json"
        with patch("portfolio.agent_invocation._STACK_OVERFLOW_FILE", f):
            _save_stack_overflow_counter(7)
            assert _load_stack_overflow_counter() == 7

    def test_round_trip_zero(self, tmp_path):
        f = tmp_path / "stack_overflow_counter.json"
        with patch("portfolio.agent_invocation._STACK_OVERFLOW_FILE", f):
            _save_stack_overflow_counter(0)
            assert _load_stack_overflow_counter() == 0
