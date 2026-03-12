"""Integration tests: check_agent_completion() is called from the main loop.

Verifies BUG-39 fix: check_agent_completion() must be called at the start
of each run() cycle so completed agents are detected promptly.
"""

from contextlib import ExitStack
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate_from_disk(tmp_path, monkeypatch):
    """Prevent tests from touching real state files."""
    monkeypatch.setattr("portfolio.main.DATA_DIR", tmp_path)
    monkeypatch.setattr("portfolio.main.CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr("portfolio.main.STATE_FILE", tmp_path / "portfolio_state.json")


def _apply_stubs(stack):
    """Apply patches for run()'s core dependencies (module-level imports only)."""
    config = {
        "telegram": {"token": "x", "chat_id": "1"},
        "layer2": {"enabled": False},
        "bigbet": {"enabled": False},
    }
    targets = {
        "portfolio.main._load_config": MagicMock(return_value=config),
        "portfolio.main.load_state": MagicMock(return_value={
            "cash_sek": 500000, "holdings": {}, "initial_value_sek": 500000,
            "transactions": [], "total_fees_sek": 0,
        }),
        "portfolio.main.fetch_usd_sek": MagicMock(return_value=10.5),
        "portfolio.main.get_market_state": MagicMock(
            return_value=("closed", set(), 300)
        ),
        "portfolio.main.collect_timeframes": MagicMock(return_value=[]),
        "portfolio.main.write_agent_summary": MagicMock(return_value={}),
        "portfolio.main.portfolio_value": MagicMock(return_value=500000),
        "portfolio.main.save_state": MagicMock(),
        # check_triggers is imported locally in run(), patch at source
        "portfolio.trigger.check_triggers": MagicMock(return_value=(False, [])),
    }
    mocks = {}
    for target, mock_obj in targets.items():
        mocks[target] = stack.enter_context(patch(target, mock_obj))
    return mocks


class TestCheckAgentCompletionCalledFromRun:
    """Verify check_agent_completion() is invoked at the start of run()."""

    def test_run_calls_check_agent_completion(self):
        """run() must call check_agent_completion() each cycle."""
        from portfolio.main import run
        with ExitStack() as stack:
            _apply_stubs(stack)
            mock_check = stack.enter_context(
                patch("portfolio.main.check_agent_completion", return_value=None)
            )
            run()
            mock_check.assert_called_once()

    def test_run_calls_check_agent_completion_before_signals(self):
        """check_agent_completion() must be called BEFORE write_agent_summary."""
        from portfolio.main import run
        call_order = []

        with ExitStack() as stack:
            mocks = _apply_stubs(stack)
            mocks["portfolio.main.write_agent_summary"].side_effect = (
                lambda *a, **kw: call_order.append("write_summary")
            )
            stack.enter_context(
                patch(
                    "portfolio.main.check_agent_completion",
                    side_effect=lambda: call_order.append("check_completion"),
                )
            )
            run()

        assert "check_completion" in call_order
        if "write_summary" in call_order:
            assert call_order.index("check_completion") < call_order.index(
                "write_summary"
            )

    def test_run_continues_if_check_agent_completion_raises(self):
        """run() must not crash if check_agent_completion() raises."""
        from portfolio.main import run
        with ExitStack() as stack:
            _apply_stubs(stack)
            stack.enter_context(
                patch(
                    "portfolio.main.check_agent_completion",
                    side_effect=RuntimeError("boom"),
                )
            )
            # Should not raise — the exception is caught
            run()

    def test_check_agent_completion_result_logged(self):
        """When check_agent_completion() returns a result, it should be logged."""
        from portfolio.main import run
        completion_result = {
            "status": "success",
            "exit_code": 0,
            "duration_s": 42.5,
            "tier": 2,
        }
        with ExitStack() as stack:
            _apply_stubs(stack)
            stack.enter_context(
                patch(
                    "portfolio.main.check_agent_completion",
                    return_value=completion_result,
                )
            )
            mock_logger = stack.enter_context(
                patch("portfolio.main.logger")
            )
            run()
            log_calls = [str(c) for c in mock_logger.info.call_args_list]
            assert any(
                "Agent completed" in s for s in log_calls
            ), f"Expected 'Agent completed' in log, got: {log_calls}"
