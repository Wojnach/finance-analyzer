"""Tests for ranging regime trigger dampening (2026-04-22).

When a ticker's regime is "ranging", low-confidence consensus crossings
(HOLD → BUY/SELL with confidence < RANGING_CONSENSUS_MIN_CONFIDENCE) are
suppressed to prevent the 20+ wasted Layer 2 invocations pattern.

Covers:
1. Low-confidence consensus in ranging regime is suppressed
2. High-confidence consensus in ranging regime still triggers
3. Low-confidence consensus in non-ranging regime still triggers
4. Suppressed trigger updates baseline (no re-fire next cycle)
5. Signal without extra/regime dict still triggers (backward compat)
6. RANGING_CONSENSUS_MIN_CONFIDENCE=0 disables dampening
"""

import json
import time

import pytest

import portfolio.trigger as trigger_mod
from portfolio.trigger import (
    RANGING_CONSENSUS_MIN_CONFIDENCE,
    check_triggers,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolate_state_files(tmp_path, monkeypatch):
    """Redirect state files to tmp_path."""
    state_file = tmp_path / "trigger_state.json"
    portfolio_file = tmp_path / "portfolio_state.json"
    portfolio_bold_file = tmp_path / "portfolio_state_bold.json"

    monkeypatch.setattr(trigger_mod, "STATE_FILE", state_file)
    monkeypatch.setattr(trigger_mod, "PORTFOLIO_FILE", portfolio_file)
    monkeypatch.setattr(trigger_mod, "PORTFOLIO_BOLD_FILE", portfolio_bold_file)

    return {"state_file": state_file}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sig(action="HOLD", confidence=0.5, regime=None):
    """Create a signal dict for one ticker, optionally with regime in extra."""
    sig = {"action": action, "confidence": confidence}
    if regime is not None:
        sig["extra"] = {"_regime": regime}
    return sig


def _suppress_cooldown(state_file):
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
    else:
        state = {}
    state["last_trigger_time"] = time.time() + 99999
    state_file.write_text(json.dumps(state), encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRangingDampening:
    def test_low_confidence_ranging_suppressed(self, isolate_state_files):
        """HOLD → BUY at 20% confidence in ranging regime should NOT trigger."""
        prices = {"BTC-USD": 78000}

        # Seed with HOLD
        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})
        _suppress_cooldown(isolate_state_files["state_file"])

        # Low-confidence BUY in ranging regime
        triggered, reasons = check_triggers(
            {"BTC-USD": _sig("BUY", 0.20, regime="ranging")},
            prices, {}, {},
        )

        consensus_reasons = [r for r in reasons if "consensus" in r]
        assert len(consensus_reasons) == 0, (
            f"Expected suppression but got: {consensus_reasons}"
        )

    def test_high_confidence_ranging_triggers(self, isolate_state_files):
        """HOLD → BUY at 50% confidence in ranging regime should still trigger."""
        prices = {"BTC-USD": 78000}

        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})
        _suppress_cooldown(isolate_state_files["state_file"])

        triggered, reasons = check_triggers(
            {"BTC-USD": _sig("BUY", 0.50, regime="ranging")},
            prices, {}, {},
        )

        consensus_reasons = [r for r in reasons if "consensus" in r]
        assert len(consensus_reasons) == 1
        assert "BTC-USD" in consensus_reasons[0]

    def test_low_confidence_trending_triggers(self, isolate_state_files):
        """HOLD → BUY at 20% confidence in trending-up regime should trigger."""
        prices = {"BTC-USD": 78000}

        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})
        _suppress_cooldown(isolate_state_files["state_file"])

        triggered, reasons = check_triggers(
            {"BTC-USD": _sig("BUY", 0.20, regime="trending-up")},
            prices, {}, {},
        )

        consensus_reasons = [r for r in reasons if "consensus" in r]
        assert len(consensus_reasons) == 1

    def test_suppressed_updates_baseline(self, isolate_state_files):
        """Suppressed trigger should update baseline so it doesn't re-fire."""
        prices = {"BTC-USD": 78000}

        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})
        _suppress_cooldown(isolate_state_files["state_file"])

        # First ranging low-confidence BUY — suppressed
        check_triggers(
            {"BTC-USD": _sig("BUY", 0.20, regime="ranging")},
            prices, {}, {},
        )
        _suppress_cooldown(isolate_state_files["state_file"])

        # Second ranging low-confidence BUY — should NOT trigger either
        # (baseline updated to BUY, so this is BUY→BUY, not HOLD→BUY)
        triggered, reasons = check_triggers(
            {"BTC-USD": _sig("BUY", 0.20, regime="ranging")},
            prices, {}, {},
        )
        consensus_reasons = [r for r in reasons if "consensus" in r]
        assert len(consensus_reasons) == 0

    def test_no_extra_dict_triggers_normally(self, isolate_state_files):
        """Signal without extra dict should trigger normally (backward compat)."""
        prices = {"BTC-USD": 78000}

        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})
        _suppress_cooldown(isolate_state_files["state_file"])

        # No regime info — default to "unknown", dampening should not apply
        triggered, reasons = check_triggers(
            {"BTC-USD": _sig("BUY", 0.20)},  # no regime
            prices, {}, {},
        )

        consensus_reasons = [r for r in reasons if "consensus" in r]
        assert len(consensus_reasons) == 1

    def test_dampening_disabled_when_zero(self, isolate_state_files, monkeypatch):
        """Setting RANGING_CONSENSUS_MIN_CONFIDENCE=0 disables dampening."""
        monkeypatch.setattr(trigger_mod, "RANGING_CONSENSUS_MIN_CONFIDENCE", 0.0)
        prices = {"BTC-USD": 78000}

        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})
        _suppress_cooldown(isolate_state_files["state_file"])

        triggered, reasons = check_triggers(
            {"BTC-USD": _sig("BUY", 0.10, regime="ranging")},
            prices, {}, {},
        )

        consensus_reasons = [r for r in reasons if "consensus" in r]
        assert len(consensus_reasons) == 1

    def test_sell_also_dampened_in_ranging(self, isolate_state_files):
        """HOLD → SELL at low confidence in ranging should also be suppressed."""
        prices = {"XAG-USD": 77.80}

        check_triggers({"XAG-USD": _sig("HOLD")}, prices, {}, {})
        _suppress_cooldown(isolate_state_files["state_file"])

        triggered, reasons = check_triggers(
            {"XAG-USD": _sig("SELL", 0.25, regime="ranging")},
            prices, {}, {},
        )

        consensus_reasons = [r for r in reasons if "consensus" in r]
        assert len(consensus_reasons) == 0

    def test_exactly_at_threshold_triggers(self, isolate_state_files):
        """Confidence exactly at RANGING_CONSENSUS_MIN_CONFIDENCE should trigger."""
        prices = {"BTC-USD": 78000}

        check_triggers({"BTC-USD": _sig("HOLD")}, prices, {}, {})
        _suppress_cooldown(isolate_state_files["state_file"])

        triggered, reasons = check_triggers(
            {"BTC-USD": _sig("BUY", RANGING_CONSENSUS_MIN_CONFIDENCE, regime="ranging")},
            prices, {}, {},
        )

        consensus_reasons = [r for r in reasons if "consensus" in r]
        assert len(consensus_reasons) == 1
