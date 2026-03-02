"""Tests for trigger.py _save_state() pruning logic.

Covers the state pruning that removes triggered_consensus entries for tickers
no longer in the active tracking set (_current_tickers), and ensures the
internal _current_tickers field is stripped before persisting to disk.

BUG-38 regression: empty set must still trigger pruning (use `is not None`,
not truthiness).
"""

import pytest

import portfolio.trigger as trigger_mod
from portfolio.trigger import _save_state


@pytest.fixture(autouse=True)
def _patch_atomic_write(monkeypatch):
    """Capture what _save_state writes instead of touching the filesystem.

    Stores the written data on the module as ``trigger_mod._last_written``
    so each test can inspect it.
    """
    def fake_write(path, data):
        trigger_mod._last_written = data
        trigger_mod._last_written_path = path

    monkeypatch.setattr(trigger_mod, "atomic_write_json", fake_write)
    monkeypatch.setattr(trigger_mod, "STATE_FILE", "fake_trigger_state.json")
    yield
    # cleanup
    for attr in ("_last_written", "_last_written_path"):
        if hasattr(trigger_mod, attr):
            delattr(trigger_mod, attr)


# ── BUG-38 regression ────────────────────────────────────────────────


def test_bug38_empty_set_prunes_all():
    """BUG-38: _current_tickers is an empty set (not None).

    An empty set is falsy in Python, so ``if current_tickers:`` skips pruning.
    The fix must use ``if current_tickers is not None:`` so that an empty set
    still triggers the prune path, resulting in an empty triggered_consensus.
    """
    state = {
        "triggered_consensus": {"BTC-USD": "BUY", "ETH-USD": "SELL"},
        "_current_tickers": set(),
    }
    _save_state(state)
    written = trigger_mod._last_written
    # With the bug present, the old entries survive.  After fix, they are pruned.
    assert written["triggered_consensus"] == {}


def test_bug38_empty_set_does_not_crash():
    """Empty _current_tickers + empty triggered_consensus must not raise."""
    state = {
        "triggered_consensus": {},
        "_current_tickers": set(),
    }
    _save_state(state)
    assert trigger_mod._last_written["triggered_consensus"] == {}


# ── Normal pruning ────────────────────────────────────────────────────


def test_prune_removes_stale_tickers():
    """Only tickers present in _current_tickers survive pruning."""
    state = {
        "triggered_consensus": {
            "BTC-USD": "BUY",
            "ETH-USD": "SELL",
            "MSTR": "BUY",
        },
        "_current_tickers": {"BTC-USD", "ETH-USD"},
    }
    _save_state(state)
    written = trigger_mod._last_written
    assert "BTC-USD" in written["triggered_consensus"]
    assert "ETH-USD" in written["triggered_consensus"]
    assert "MSTR" not in written["triggered_consensus"]


def test_prune_keeps_all_when_all_current():
    """When every ticker is in _current_tickers, nothing is removed."""
    state = {
        "triggered_consensus": {
            "BTC-USD": "BUY",
            "ETH-USD": "SELL",
        },
        "_current_tickers": {"BTC-USD", "ETH-USD"},
    }
    _save_state(state)
    assert len(trigger_mod._last_written["triggered_consensus"]) == 2


def test_prune_with_superset_current_tickers():
    """_current_tickers may contain tickers not in triggered_consensus."""
    state = {
        "triggered_consensus": {"BTC-USD": "BUY"},
        "_current_tickers": {"BTC-USD", "ETH-USD", "NVDA"},
    }
    _save_state(state)
    tc = trigger_mod._last_written["triggered_consensus"]
    assert tc == {"BTC-USD": "BUY"}


# ── _current_tickers is None / missing ────────────────────────────────


def test_no_current_tickers_key_skips_pruning():
    """When _current_tickers is absent, no pruning occurs (backwards compat)."""
    state = {
        "triggered_consensus": {
            "BTC-USD": "BUY",
            "OLD-TICKER": "SELL",
        },
    }
    _save_state(state)
    tc = trigger_mod._last_written["triggered_consensus"]
    assert tc == {"BTC-USD": "BUY", "OLD-TICKER": "SELL"}


def test_current_tickers_explicit_none_skips_pruning():
    """Explicitly setting _current_tickers to None also skips pruning."""
    state = {
        "triggered_consensus": {"BTC-USD": "BUY"},
        "_current_tickers": None,
    }
    _save_state(state)
    # state.get("_current_tickers", set()) returns None here, and the
    # truthiness / is-not-None check should skip pruning.
    assert trigger_mod._last_written["triggered_consensus"] == {"BTC-USD": "BUY"}


# ── _current_tickers stripped from persisted state ────────────────────


def test_current_tickers_removed_from_persisted_state():
    """_current_tickers is an internal field and must not be written to disk."""
    state = {
        "triggered_consensus": {"BTC-USD": "BUY"},
        "_current_tickers": {"BTC-USD"},
        "some_other_field": 42,
    }
    _save_state(state)
    written = trigger_mod._last_written
    assert "_current_tickers" not in written


def test_current_tickers_removed_even_when_none():
    """_current_tickers=None is also stripped before persisting."""
    state = {
        "_current_tickers": None,
    }
    _save_state(state)
    assert "_current_tickers" not in trigger_mod._last_written


# ── Edge cases ────────────────────────────────────────────────────────


def test_empty_state():
    """Completely empty state dict must not crash."""
    _save_state({})
    assert trigger_mod._last_written == {}


def test_other_state_fields_preserved():
    """Fields besides triggered_consensus and _current_tickers survive."""
    state = {
        "last_trigger_time": 1234567890,
        "prices": {"BTC-USD": 67000},
        "triggered_consensus": {"BTC-USD": "BUY", "MSTR": "SELL"},
        "_current_tickers": {"BTC-USD"},
    }
    _save_state(state)
    written = trigger_mod._last_written
    assert written["last_trigger_time"] == 1234567890
    assert written["prices"] == {"BTC-USD": 67000}
    assert "MSTR" not in written["triggered_consensus"]
    assert "_current_tickers" not in written
