"""Tests for the perception gate pre-invocation filter."""

import json
import pytest
from unittest.mock import patch, MagicMock

from portfolio.perception_gate import should_invoke, _BYPASS_KEYWORDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(enabled=True, min_strength=0.3, skip_tiers=None):
    return {
        "perception_gate": {
            "enabled": enabled,
            "min_signal_strength": min_strength,
            "skip_tiers": skip_tiers if skip_tiers is not None else [1],
        }
    }


def _summary(signals=None):
    """Return a mock compact summary."""
    return {"signals": signals or {}}


def _sig(action="BUY", confidence=0.6):
    return {"action": action, "confidence": confidence}


# ---------------------------------------------------------------------------
# Gate disabled
# ---------------------------------------------------------------------------

class TestGateDisabled:
    def test_disabled_always_passes(self):
        ok, reason = should_invoke(["cooldown"], tier=1, config=_cfg(enabled=False))
        assert ok is True
        assert "disabled" in reason

    def test_missing_config_section_passes(self):
        ok, reason = should_invoke(["cooldown"], tier=1, config={})
        assert ok is True


# ---------------------------------------------------------------------------
# Tier filtering
# ---------------------------------------------------------------------------

class TestTierFiltering:
    def test_t1_in_skip_tiers_evaluated(self):
        """T1 is in skip_tiers — gate logic runs."""
        with patch("portfolio.perception_gate._load_compact_summary", return_value=_summary()):
            ok, reason = should_invoke(["cooldown"], tier=1, config=_cfg())
            assert ok is False  # no non-HOLD signals

    def test_t2_not_in_skip_tiers_passes(self):
        ok, reason = should_invoke(["cooldown"], tier=2, config=_cfg(skip_tiers=[1]))
        assert ok is True
        assert "not in skip_tiers" in reason

    def test_t3_not_in_skip_tiers_passes(self):
        ok, reason = should_invoke(["cooldown"], tier=3, config=_cfg(skip_tiers=[1]))
        assert ok is True

    def test_custom_skip_tiers(self):
        """If skip_tiers=[1,2], T2 is also evaluated."""
        with patch("portfolio.perception_gate._load_compact_summary",
                    return_value=_summary()):
            ok, _ = should_invoke(["cooldown"], tier=2, config=_cfg(skip_tiers=[1, 2]))
            assert ok is False  # empty signals → no non-HOLD


# ---------------------------------------------------------------------------
# Bypass triggers
# ---------------------------------------------------------------------------

class TestBypassTriggers:
    def test_consensus_trigger_bypasses(self):
        ok, reason = should_invoke(
            ["BTC-USD consensus BUY (80%)"], tier=1, config=_cfg()
        )
        assert ok is True
        assert "consensus" in reason

    def test_fg_crossed_bypasses(self):
        ok, reason = should_invoke(
            ["F&G crossed 20"], tier=1, config=_cfg()
        )
        assert ok is True
        assert "F&G crossed" in reason

    def test_post_trade_bypasses(self):
        ok, reason = should_invoke(
            ["post-trade reassessment"], tier=1, config=_cfg()
        )
        assert ok is True
        assert "post-trade" in reason

    def test_cooldown_does_not_bypass(self):
        with patch("portfolio.perception_gate._load_compact_summary",
                    return_value=_summary()):
            ok, _ = should_invoke(["cooldown"], tier=1, config=_cfg())
            assert ok is False


# ---------------------------------------------------------------------------
# Signal strength checks
# ---------------------------------------------------------------------------

class TestSignalStrength:
    def test_no_non_hold_signals_skips(self):
        signals = {"BTC-USD": _sig("HOLD", 0.0), "ETH-USD": _sig("HOLD", 0.0)}
        with patch("portfolio.perception_gate._load_compact_summary",
                    return_value=_summary(signals)):
            ok, reason = should_invoke(["cooldown"], tier=1, config=_cfg())
            assert ok is False
            assert "no non-HOLD" in reason

    def test_strong_signal_passes(self):
        signals = {"BTC-USD": _sig("BUY", 0.7)}
        with patch("portfolio.perception_gate._load_compact_summary",
                    return_value=_summary(signals)):
            ok, reason = should_invoke(["cooldown"], tier=1, config=_cfg(min_strength=0.3))
            assert ok is True
            assert "active signals" in reason

    def test_weak_signal_below_threshold_skips(self):
        signals = {"BTC-USD": _sig("BUY", 0.2)}
        with patch("portfolio.perception_gate._load_compact_summary",
                    return_value=_summary(signals)):
            ok, reason = should_invoke(["cooldown"], tier=1, config=_cfg(min_strength=0.3))
            assert ok is False
            assert "max confidence" in reason

    def test_custom_min_strength(self):
        signals = {"BTC-USD": _sig("BUY", 0.5)}
        with patch("portfolio.perception_gate._load_compact_summary",
                    return_value=_summary(signals)):
            ok, _ = should_invoke(["cooldown"], tier=1, config=_cfg(min_strength=0.6))
            assert ok is False

    def test_no_summary_passes_through(self):
        with patch("portfolio.perception_gate._load_compact_summary", return_value=None):
            ok, reason = should_invoke(["cooldown"], tier=1, config=_cfg())
            assert ok is True
            assert "no summary" in reason


# ---------------------------------------------------------------------------
# Integration with agent_invocation
# ---------------------------------------------------------------------------

class TestAgentInvocationIntegration:
    @patch("portfolio.perception_gate.load_config", return_value=_cfg())
    @patch("portfolio.perception_gate._load_compact_summary", return_value=_summary())
    @patch("portfolio.agent_invocation._log_trigger")
    @patch("portfolio.journal.write_context", return_value=0)
    def test_gate_skips_invocation(self, mock_ctx, mock_log, mock_summary, mock_config):
        from portfolio.agent_invocation import invoke_agent
        result = invoke_agent(["cooldown"], tier=1)
        assert result is False
        mock_log.assert_called_once()
        args = mock_log.call_args
        assert args[0][1] == "skipped_gate"
