"""Tests for trigger startup grace period.

Verifies that the first check_triggers call after a restart updates
the baseline without triggering Layer 2.
"""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import portfolio.trigger as trigger_mod
from portfolio.trigger import check_triggers, _GRACE_PERIOD_KEY

# Use project-local temp dir to avoid Windows permission issues with %TEMP%
_TEST_TMP_ROOT = Path(__file__).resolve().parent.parent / "data" / "_test_tmp"


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    """Isolate trigger state to a temp directory."""
    _TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(tempfile.mkdtemp(dir=_TEST_TMP_ROOT))
    state_file = tmp_path / "trigger_state.json"
    pf_file = tmp_path / "portfolio_state.json"
    pf_bold_file = tmp_path / "portfolio_state_bold.json"
    monkeypatch.setattr(trigger_mod, "STATE_FILE", state_file)
    monkeypatch.setattr(trigger_mod, "PORTFOLIO_FILE", pf_file)
    monkeypatch.setattr(trigger_mod, "PORTFOLIO_BOLD_FILE", pf_bold_file)
    # Reset the module-level grace flag before each test
    trigger_mod._startup_grace_active = True
    yield state_file
    shutil.rmtree(tmp_path, ignore_errors=True)


def _make_signals(*tickers_actions):
    """Helper: build signals dict."""
    sigs = {}
    for ticker, action in tickers_actions:
        sigs[ticker] = {"action": action, "confidence": 0.5}
    return sigs


class TestStartupGracePeriod:
    def test_first_call_returns_no_trigger(self, isolate):
        """First call after restart should NOT trigger."""
        signals = _make_signals(("BTC-USD", "BUY"), ("ETH-USD", "SELL"))
        prices = {"BTC-USD": 70000, "ETH-USD": 2000}

        triggered, reasons = check_triggers(signals, prices, {}, {})
        assert triggered is False
        assert reasons == []

    def test_first_call_saves_pid(self, isolate):
        """First call should save current PID in state."""
        signals = _make_signals(("BTC-USD", "HOLD"))
        check_triggers(signals, {"BTC-USD": 70000}, {}, {})

        state = json.loads(isolate.read_text(encoding="utf-8"))
        assert state[_GRACE_PERIOD_KEY] == os.getpid()

    def test_first_call_updates_baseline(self, isolate):
        """First call should update price baseline to current values."""
        signals = _make_signals(("BTC-USD", "BUY"))
        check_triggers(signals, {"BTC-USD": 70000}, {}, {})

        state = json.loads(isolate.read_text(encoding="utf-8"))
        assert state["last"]["prices"]["BTC-USD"] == 70000

    def test_first_call_updates_consensus_baseline(self, isolate):
        """First call should update triggered_consensus to current state."""
        signals = _make_signals(("BTC-USD", "BUY"), ("ETH-USD", "SELL"))
        check_triggers(signals, {"BTC-USD": 70000, "ETH-USD": 2000}, {}, {})

        state = json.loads(isolate.read_text(encoding="utf-8"))
        assert state["triggered_consensus"]["BTC-USD"] == "BUY"
        assert state["triggered_consensus"]["ETH-USD"] == "SELL"

    def test_second_call_triggers_normally(self, isolate):
        """Second call (same PID) should use normal trigger logic."""
        signals = _make_signals(("BTC-USD", "HOLD"))
        # First call: grace period
        check_triggers(signals, {"BTC-USD": 70000}, {}, {})

        # Second call: BTC now BUY from HOLD — should trigger
        signals2 = _make_signals(("BTC-USD", "BUY"))
        triggered, reasons = check_triggers(signals2, {"BTC-USD": 70000}, {}, {})
        assert triggered is True
        assert any("BTC-USD" in r and "BUY" in r for r in reasons)

    def test_same_pid_no_grace(self, isolate):
        """If state already has current PID, no grace period (normal restart)."""
        # Pre-seed state with current PID
        state = {_GRACE_PERIOD_KEY: os.getpid(), "last": {"prices": {}, "signals": {}}}
        isolate.write_text(json.dumps(state), encoding="utf-8")

        signals = _make_signals(("BTC-USD", "BUY"))
        triggered, reasons = check_triggers(signals, {"BTC-USD": 70000}, {}, {})
        # Should trigger normally since PID matches (not a restart)
        assert triggered is True

    def test_different_pid_triggers_grace(self, isolate):
        """If state has a DIFFERENT PID, grace period activates."""
        state = {
            _GRACE_PERIOD_KEY: 99999,  # old PID
            "last": {
                "prices": {"BTC-USD": 50000},
                "signals": {"BTC-USD": {"action": "HOLD", "confidence": 0}},
            },
            "triggered_consensus": {"BTC-USD": "HOLD"},
        }
        isolate.write_text(json.dumps(state), encoding="utf-8")

        # BTC moved from 50K to 70K (40% move) — would normally trigger
        signals = _make_signals(("BTC-USD", "BUY"))
        triggered, reasons = check_triggers(signals, {"BTC-USD": 70000}, {}, {})

        # Grace period should suppress it
        assert triggered is False
        assert reasons == []

    def test_grace_only_fires_once(self, isolate):
        """Grace period should only activate on the FIRST call."""
        signals = _make_signals(("BTC-USD", "HOLD"))
        # First: grace
        check_triggers(signals, {"BTC-USD": 70000}, {}, {})

        # Reset grace flag manually to simulate what would happen on re-import
        # (it shouldn't re-trigger because PID now matches)
        trigger_mod._startup_grace_active = True

        # Second call with same PID — should NOT grace again
        signals2 = _make_signals(("BTC-USD", "BUY"))
        triggered, reasons = check_triggers(signals2, {"BTC-USD": 70000}, {}, {})
        assert triggered is True  # normal trigger, not suppressed


class TestGracePeriodPreservesState:
    def test_preserves_last_full_review_time(self, isolate):
        """Grace period should NOT reset last_full_review_time."""
        review_time = time.time() - 3600  # 1 hour ago
        state = {
            _GRACE_PERIOD_KEY: 99999,
            "last_full_review_time": review_time,
            "last": {"prices": {}, "signals": {}},
        }
        isolate.write_text(json.dumps(state), encoding="utf-8")

        signals = _make_signals(("BTC-USD", "HOLD"))
        check_triggers(signals, {"BTC-USD": 70000}, {}, {})

        new_state = json.loads(isolate.read_text(encoding="utf-8"))
        assert new_state["last_full_review_time"] == review_time

    def test_preserves_sustained_counts(self, isolate):
        """Grace period should NOT reset sustained signal counts."""
        state = {
            _GRACE_PERIOD_KEY: 99999,
            "sustained_counts": {"BTC-USD": {"action": "BUY", "count": 5}},
            "last": {"prices": {}, "signals": {}},
        }
        isolate.write_text(json.dumps(state), encoding="utf-8")

        signals = _make_signals(("BTC-USD", "BUY"))
        check_triggers(signals, {"BTC-USD": 70000}, {}, {})

        new_state = json.loads(isolate.read_text(encoding="utf-8"))
        assert new_state["sustained_counts"]["BTC-USD"]["count"] == 5
