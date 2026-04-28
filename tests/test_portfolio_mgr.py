"""Tests for portfolio_mgr.py — safety-critical financial state management.

Covers: load/save roundtrip, backup rotation, corruption recovery,
atomic read-modify-write, and validated state defaults."""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from portfolio.portfolio_mgr import (
    INITIAL_CASH_SEK,
    _rotate_backups,
    _validated_state,
    _load_state_from,
    _save_state_to,
    portfolio_value,
    update_state,
)


@pytest.fixture
def state_dir(tmp_path):
    """Create a temporary directory for portfolio state files."""
    return tmp_path


@pytest.fixture
def state_file(state_dir):
    """Path to a temporary state file."""
    return state_dir / "portfolio_state.json"


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


class TestValidatedState:
    """_validated_state fills missing keys and corrects types."""

    def test_none_returns_defaults(self):
        result = _validated_state(None)
        assert result["cash_sek"] == INITIAL_CASH_SEK
        assert result["holdings"] == {}
        assert result["transactions"] == []
        assert "start_date" in result

    def test_empty_dict_returns_defaults(self):
        result = _validated_state({})
        assert result["cash_sek"] == INITIAL_CASH_SEK
        assert result["holdings"] == {}

    def test_non_dict_returns_defaults(self):
        result = _validated_state("corrupted")
        assert result["cash_sek"] == INITIAL_CASH_SEK
        assert "start_date" in result

    def test_merges_with_existing(self):
        loaded = {"cash_sek": 123456, "holdings": {"BTC": {"shares": 0.5}}}
        result = _validated_state(loaded)
        assert result["cash_sek"] == 123456
        assert result["holdings"]["BTC"]["shares"] == 0.5
        assert result["transactions"] == []  # filled from defaults

    def test_corrects_bad_holdings_type(self):
        result = _validated_state({"holdings": "not a dict"})
        assert result["holdings"] == {}

    def test_corrects_bad_transactions_type(self):
        result = _validated_state({"transactions": "not a list"})
        assert result["transactions"] == []


class TestRotateBackups:
    """_rotate_backups creates rolling .bak copies."""

    def test_creates_first_backup(self, state_file):
        _write_json(state_file, {"cash_sek": 100})
        _rotate_backups(state_file)
        bak = state_file.with_suffix(".json.bak")
        assert bak.exists()
        assert json.loads(bak.read_text())["cash_sek"] == 100

    def test_rotates_existing_backups(self, state_file):
        _write_json(state_file, {"cash_sek": 300})
        # Create pre-existing .bak
        bak1 = state_file.with_suffix(".json.bak")
        _write_json(bak1, {"cash_sek": 200})

        _rotate_backups(state_file)

        # .bak should now be current (300), .bak2 should be previous (200)
        assert json.loads(bak1.read_text())["cash_sek"] == 300
        bak2 = state_file.with_suffix(".json.bak2")
        assert bak2.exists()
        assert json.loads(bak2.read_text())["cash_sek"] == 200

    def test_skips_when_file_missing(self, state_file):
        # Should not raise
        _rotate_backups(state_file)

    def test_skips_when_file_empty(self, state_file):
        state_file.write_text("")
        _rotate_backups(state_file)
        bak = state_file.with_suffix(".json.bak")
        assert not bak.exists()


class TestLoadState:
    """_load_state_from handles missing, valid, and corrupt files."""

    def test_load_missing_returns_defaults(self, state_file):
        result = _load_state_from(state_file)
        assert result["cash_sek"] == INITIAL_CASH_SEK
        assert result["holdings"] == {}

    def test_load_valid_roundtrip(self, state_file):
        data = {"cash_sek": 42000, "holdings": {}, "transactions": []}
        _write_json(state_file, data)
        result = _load_state_from(state_file)
        assert result["cash_sek"] == 42000

    def test_load_corrupt_falls_back_to_backup(self, state_file):
        # Write corrupt main file
        state_file.write_text("{invalid json", encoding="utf-8")
        # Write valid backup
        bak = state_file.with_suffix(".json.bak")
        _write_json(bak, {"cash_sek": 99999, "holdings": {}, "transactions": []})

        result = _load_state_from(state_file)
        assert result["cash_sek"] == 99999

    def test_load_all_corrupt_returns_defaults(self, state_file):
        # Write corrupt main file and corrupt backups
        state_file.write_text("garbage", encoding="utf-8")
        for suffix in [".json.bak", ".json.bak2", ".json.bak3"]:
            state_file.with_suffix(suffix).write_text("garbage", encoding="utf-8")

        result = _load_state_from(state_file)
        assert result["cash_sek"] == INITIAL_CASH_SEK  # defaults


class TestSaveState:
    """_save_state_to writes atomically with backup rotation."""

    def test_save_creates_file(self, state_file):
        data = {"cash_sek": 55555, "holdings": {}, "transactions": []}
        _save_state_to(state_file, data)
        assert state_file.exists()
        loaded = json.loads(state_file.read_text())
        assert loaded["cash_sek"] == 55555

    def test_save_creates_backup_of_previous(self, state_file):
        # First save
        _write_json(state_file, {"cash_sek": 11111})
        # Second save (should create backup of 11111)
        _save_state_to(state_file, {"cash_sek": 22222})

        bak = state_file.with_suffix(".json.bak")
        assert bak.exists()
        assert json.loads(bak.read_text())["cash_sek"] == 11111
        assert json.loads(state_file.read_text())["cash_sek"] == 22222


class TestUpdateState:
    """update_state performs atomic read-modify-write."""

    def test_update_modifies_state(self, state_file):
        _write_json(state_file, {
            "cash_sek": 100000, "holdings": {}, "transactions": [],
        })
        with patch("portfolio.portfolio_mgr.STATE_FILE", state_file):
            result = update_state(lambda s: s.update({"cash_sek": 90000}) or s)
        assert result["cash_sek"] == 90000
        # Verify it's persisted
        loaded = json.loads(state_file.read_text())
        assert loaded["cash_sek"] == 90000

    def test_update_creates_backup(self, state_file):
        _write_json(state_file, {
            "cash_sek": 100000, "holdings": {}, "transactions": [],
        })
        with patch("portfolio.portfolio_mgr.STATE_FILE", state_file):
            update_state(lambda s: s.update({"cash_sek": 50000}) or s)
        bak = state_file.with_suffix(".json.bak")
        assert bak.exists()
        assert json.loads(bak.read_text())["cash_sek"] == 100000

    def test_update_on_missing_file_uses_defaults(self, state_file):
        with patch("portfolio.portfolio_mgr.STATE_FILE", state_file):
            result = update_state(lambda s: s.update({"cash_sek": 42}) or s)
        assert result["cash_sek"] == 42

    def test_update_mutate_in_place(self, state_file):
        """mutate_fn that modifies dict in-place and returns None."""
        _write_json(state_file, {
            "cash_sek": 100000, "holdings": {}, "transactions": [],
        })
        with patch("portfolio.portfolio_mgr.STATE_FILE", state_file):
            result = update_state(lambda s: s.__setitem__("cash_sek", 77777))
        assert result["cash_sek"] == 77777


class TestPortfolioValue:
    """portfolio_value handles edge cases in fx_rate and prices."""

    def test_nan_fx_rate_returns_cash_only(self):
        """BUG-232: NaN fx_rate should not propagate into portfolio value."""
        state = {"cash_sek": 100000, "holdings": {"BTC-USD": {"shares": 1}}}
        result = portfolio_value(state, {"BTC-USD": 50000}, float("nan"))
        assert result == 100000

    def test_inf_fx_rate_returns_cash_only(self):
        state = {"cash_sek": 100000, "holdings": {"BTC-USD": {"shares": 1}}}
        result = portfolio_value(state, {"BTC-USD": 50000}, float("inf"))
        assert result == 100000

    def test_negative_inf_fx_rate_returns_cash_only(self):
        state = {"cash_sek": 100000, "holdings": {"BTC-USD": {"shares": 1}}}
        result = portfolio_value(state, {"BTC-USD": 50000}, float("-inf"))
        assert result == 100000

    def test_zero_fx_rate_returns_cash_only(self):
        state = {"cash_sek": 100000, "holdings": {"BTC-USD": {"shares": 1}}}
        result = portfolio_value(state, {"BTC-USD": 50000}, 0)
        assert result == 100000

    def test_valid_fx_rate_computes_value(self):
        state = {"cash_sek": 100000, "holdings": {"BTC-USD": {"shares": 0.1}}}
        result = portfolio_value(state, {"BTC-USD": 50000}, 10.0)
        assert result == 100000 + 0.1 * 50000 * 10.0
