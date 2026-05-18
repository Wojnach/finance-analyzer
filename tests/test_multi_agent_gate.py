"""Tests for multi_agent_layer2 specialist launch gate integration — P1.3."""

from unittest.mock import patch

from portfolio.multi_agent_layer2 import launch_specialists


class TestSpecialistGating:
    def test_launch_blocked_when_claude_disabled(self):
        with patch("portfolio.multi_agent_layer2.check_claude_gates", return_value=(False, "CLAUDE_ENABLED=False")):
            procs = launch_specialists("BTC-USD", ["test"])
        assert procs == []

    def test_launch_proceeds_when_gates_pass(self):
        with (
            patch("portfolio.multi_agent_layer2.check_claude_gates", return_value=(True, "ok")),
            patch("portfolio.multi_agent_layer2.shutil.which", return_value=None),
        ):
            procs = launch_specialists("BTC-USD", ["test"])
        assert procs == []
