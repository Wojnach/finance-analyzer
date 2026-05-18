"""Tests for claude_gate.check_claude_gates — P1.3 gate extraction."""

from unittest.mock import patch

from portfolio.claude_gate import check_claude_gates


class TestCheckClaudeGates:
    def test_blocked_when_disabled(self):
        with patch("portfolio.claude_gate.CLAUDE_ENABLED", False):
            allowed, reason = check_claude_gates("test_caller")
        assert not allowed
        assert "CLAUDE_ENABLED" in reason

    def test_blocked_when_config_disabled(self):
        with (
            patch("portfolio.claude_gate.CLAUDE_ENABLED", True),
            patch("portfolio.claude_gate._load_config_layer2_enabled", return_value=False),
        ):
            allowed, reason = check_claude_gates("test_caller")
        assert not allowed
        assert "layer2.enabled" in reason

    def test_allowed_when_all_gates_pass(self):
        with (
            patch("portfolio.claude_gate.CLAUDE_ENABLED", True),
            patch("portfolio.claude_gate._load_config_layer2_enabled", return_value=True),
            patch("portfolio.claude_gate._count_today_invocations", return_value=0),
        ):
            allowed, reason = check_claude_gates("test_caller")
        assert allowed
        assert reason == "ok"

    def test_allowed_but_warns_above_threshold(self):
        with (
            patch("portfolio.claude_gate.CLAUDE_ENABLED", True),
            patch("portfolio.claude_gate._load_config_layer2_enabled", return_value=True),
            patch("portfolio.claude_gate._count_today_invocations", return_value=100),
        ):
            allowed, reason = check_claude_gates("test_caller")
        assert allowed
