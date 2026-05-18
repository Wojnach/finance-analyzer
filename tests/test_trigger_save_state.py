"""Tests for trigger._save_state — P1.2 empty-ticker-set guard."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from portfolio.trigger import _save_state


@pytest.fixture()
def state_file(tmp_path):
    sf = tmp_path / "trigger_state.json"
    sf.write_text("{}", encoding="utf-8")
    with patch("portfolio.trigger.STATE_FILE", sf):
        yield sf


def _read(sf):
    return json.loads(sf.read_text(encoding="utf-8"))


class TestSaveStatePruning:
    """P1.2: _save_state must not wipe baselines when ticker set is empty."""

    def test_normal_prune_removes_stale_tickers(self, state_file):
        state = {
            "triggered_consensus": {"BTC-USD": 0.8, "XAG-USD": 0.6, "REMOVED": 0.5},
            "_current_tickers": {"BTC-USD", "XAG-USD"},
        }
        _save_state(state)
        saved = _read(state_file)
        assert "BTC-USD" in saved["triggered_consensus"]
        assert "XAG-USD" in saved["triggered_consensus"]
        assert "REMOVED" not in saved["triggered_consensus"]

    def test_empty_ticker_set_preserves_all_baselines(self, state_file):
        state = {
            "triggered_consensus": {"BTC-USD": 0.8, "ETH-USD": 0.7, "XAG-USD": 0.6},
            "_current_tickers": set(),
        }
        _save_state(state)
        saved = _read(state_file)
        assert len(saved["triggered_consensus"]) == 3
        assert "BTC-USD" in saved["triggered_consensus"]

    def test_none_current_tickers_preserves_all_baselines(self, state_file):
        state = {
            "triggered_consensus": {"BTC-USD": 0.8, "ETH-USD": 0.7},
            "_current_tickers": None,
        }
        _save_state(state)
        saved = _read(state_file)
        assert len(saved["triggered_consensus"]) == 2

    def test_no_current_tickers_key_preserves_all_baselines(self, state_file):
        state = {
            "triggered_consensus": {"BTC-USD": 0.8},
        }
        _save_state(state)
        saved = _read(state_file)
        assert "BTC-USD" in saved["triggered_consensus"]

    def test_current_tickers_not_persisted(self, state_file):
        state = {
            "triggered_consensus": {"BTC-USD": 0.8},
            "_current_tickers": {"BTC-USD"},
        }
        _save_state(state)
        saved = _read(state_file)
        assert "_current_tickers" not in saved
