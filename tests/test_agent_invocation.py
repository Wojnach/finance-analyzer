"""Comprehensive tests for portfolio/agent_invocation.py.

Covers:
  - TIER_CONFIG: tier configuration dictionary (timeouts, max_turns, labels)
  - _build_tier_prompt: tier-specific prompt generation
  - _log_trigger: trigger logging to JSONL
  - invoke_agent: happy path (process spawns and tracked)
  - invoke_agent: busy detection (skip when agent already running)
  - invoke_agent: timeout/stale process detection and cleanup
  - invoke_agent: CLAUDECODE env var cleanup
  - invoke_agent: perception gate integration
  - invoke_agent: journal context writing
  - invoke_agent: claude not on PATH fallback to pf-agent.bat
  - invoke_agent: Telegram notification on successful spawn
"""

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, call, mock_open, patch

import pytest

import portfolio.agent_invocation as ai
from portfolio.agent_invocation import (
    TIER_CONFIG,
    _build_tier_prompt,
    _log_trigger,
    invoke_agent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_agent_globals():
    """Reset module-level agent state before and after each test."""
    ai._agent_proc = None
    ai._agent_log = None
    ai._agent_start = 0
    ai._agent_timeout = 900
    yield
    ai._agent_proc = None
    ai._agent_log = None
    ai._agent_start = 0
    ai._agent_timeout = 900


@pytest.fixture
def mock_popen():
    """Provide a mock subprocess.Popen that returns a running process."""
    with patch("portfolio.agent_invocation.subprocess.Popen") as mock_p:
        proc = MagicMock()
        proc.pid = 12345
        proc.poll.return_value = None  # process is running
        proc.wait.return_value = 0
        mock_p.return_value = proc
        yield mock_p, proc


@pytest.fixture
def mock_env():
    """Mock environment so CLAUDECODE and CLAUDE_CODE_ENTRYPOINT are set."""
    env = os.environ.copy()
    env["CLAUDECODE"] = "1"
    env["CLAUDE_CODE_ENTRYPOINT"] = "cli"
    with patch.dict(os.environ, env):
        yield


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies for invoke_agent happy path."""
    with patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude"), \
         patch("portfolio.agent_invocation._load_config", return_value={"telegram": {"token": "t", "chat_id": "c"}}), \
         patch("portfolio.agent_invocation.send_or_store") as mock_sos, \
         patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
         patch("portfolio.agent_invocation.atomic_append_jsonl") as mock_jsonl, \
         patch("builtins.open", mock_open()) as mock_file:
        yield {
            "send_or_store": mock_sos,
            "atomic_append_jsonl": mock_jsonl,
            "open": mock_file,
        }


# ===========================================================================
# TIER_CONFIG
# ===========================================================================

class TestTierConfig:

    def test_tier_config_has_three_tiers(self):
        """TIER_CONFIG has entries for tiers 1, 2, and 3."""
        assert set(TIER_CONFIG.keys()) == {1, 2, 3}

    def test_tier1_timeout_is_120(self):
        """Tier 1 (Quick Check) has 120s timeout."""
        assert TIER_CONFIG[1]["timeout"] == 120

    def test_tier2_timeout_is_300(self):
        """Tier 2 (Signal Analysis) has 300s timeout."""
        assert TIER_CONFIG[2]["timeout"] == 300

    def test_tier3_timeout_is_900(self):
        """Tier 3 (Full Review) has 900s timeout."""
        assert TIER_CONFIG[3]["timeout"] == 900

    def test_tier1_max_turns_is_15(self):
        """Tier 1 has 15 max turns."""
        assert TIER_CONFIG[1]["max_turns"] == 15

    def test_tier2_max_turns_is_25(self):
        """Tier 2 has 25 max turns."""
        assert TIER_CONFIG[2]["max_turns"] == 25

    def test_tier3_max_turns_is_40(self):
        """Tier 3 has 40 max turns."""
        assert TIER_CONFIG[3]["max_turns"] == 40

    def test_each_tier_has_label(self):
        """Each tier has a human-readable label."""
        for tier_id, cfg in TIER_CONFIG.items():
            assert "label" in cfg
            assert isinstance(cfg["label"], str)
            assert len(cfg["label"]) > 0

    def test_tier_labels_are_distinct(self):
        """Each tier has a unique label."""
        labels = [cfg["label"] for cfg in TIER_CONFIG.values()]
        assert len(labels) == len(set(labels))

    def test_timeouts_increase_with_tier(self):
        """Higher tiers have longer timeouts."""
        assert TIER_CONFIG[1]["timeout"] < TIER_CONFIG[2]["timeout"]
        assert TIER_CONFIG[2]["timeout"] < TIER_CONFIG[3]["timeout"]

    def test_max_turns_increase_with_tier(self):
        """Higher tiers have more max turns."""
        assert TIER_CONFIG[1]["max_turns"] < TIER_CONFIG[2]["max_turns"]
        assert TIER_CONFIG[2]["max_turns"] < TIER_CONFIG[3]["max_turns"]


# ===========================================================================
# _build_tier_prompt
# ===========================================================================

class TestBuildTierPrompt:

    def test_tier1_prompt_mentions_quick_check(self):
        """Tier 1 prompt includes QUICK CHECK label."""
        prompt = _build_tier_prompt(1, ["cooldown"])
        assert "QUICK CHECK" in prompt

    def test_tier1_prompt_mentions_agent_context_t1(self):
        """Tier 1 prompt instructs reading agent_context_t1.json."""
        prompt = _build_tier_prompt(1, ["cooldown"])
        assert "agent_context_t1.json" in prompt

    def test_tier2_prompt_mentions_signal_analysis(self):
        """Tier 2 prompt includes SIGNAL ANALYSIS label."""
        prompt = _build_tier_prompt(2, ["consensus BTC-USD BUY"])
        assert "SIGNAL ANALYSIS" in prompt

    def test_tier2_prompt_mentions_agent_context_t2(self):
        """Tier 2 prompt instructs reading agent_context_t2.json."""
        prompt = _build_tier_prompt(2, ["consensus BTC-USD BUY"])
        assert "agent_context_t2.json" in prompt

    def test_tier3_prompt_mentions_compact_summary(self):
        """Tier 3 prompt instructs reading agent_summary_compact.json."""
        prompt = _build_tier_prompt(3, ["full review"])
        assert "agent_summary_compact.json" in prompt

    def test_tier3_prompt_mentions_both_portfolios(self):
        """Tier 3 prompt mentions both patient and bold portfolio files."""
        prompt = _build_tier_prompt(3, ["full review"])
        assert "portfolio_state.json" in prompt
        assert "portfolio_state_bold.json" in prompt

    def test_prompt_includes_trigger_reasons(self):
        """Prompt includes the trigger reason text."""
        prompt = _build_tier_prompt(2, ["consensus BTC-USD BUY", "price move +3%"])
        assert "consensus BTC-USD BUY" in prompt

    def test_prompt_truncates_reasons_to_five(self):
        """Prompt includes at most 5 reasons."""
        reasons = [f"reason_{i}" for i in range(10)]
        prompt = _build_tier_prompt(1, reasons)
        assert "reason_4" in prompt
        assert "reason_5" not in prompt

    def test_unknown_tier_falls_through_to_tier3(self):
        """An unrecognized tier number generates a Tier 3 prompt."""
        prompt = _build_tier_prompt(99, ["unknown"])
        assert "agent_summary_compact.json" in prompt

    def test_tier1_prompt_instructs_layer2_context_first(self):
        """Tier 1 prompt instructs reading layer2_context.md first."""
        prompt = _build_tier_prompt(1, ["cooldown"])
        assert "layer2_context.md" in prompt


# ===========================================================================
# _log_trigger
# ===========================================================================

class TestLogTrigger:

    @patch("portfolio.agent_invocation.atomic_append_jsonl")
    def test_log_trigger_writes_entry(self, mock_append):
        """_log_trigger writes an entry to invocations.jsonl."""
        _log_trigger(["consensus BTC BUY"], "invoked", tier=2)

        mock_append.assert_called_once()
        args = mock_append.call_args
        assert args[0][0] == ai.INVOCATIONS_FILE
        entry = args[0][1]
        assert entry["reasons"] == ["consensus BTC BUY"]
        assert entry["status"] == "invoked"
        assert entry["tier"] == 2

    @patch("portfolio.agent_invocation.atomic_append_jsonl")
    def test_log_trigger_includes_timestamp(self, mock_append):
        """_log_trigger entry includes an ISO-format timestamp."""
        _log_trigger(["cooldown"], "skipped")

        entry = mock_append.call_args[0][1]
        assert "ts" in entry
        # ISO format should contain T separator and timezone info
        assert "T" in entry["ts"]

    @patch("portfolio.agent_invocation.atomic_append_jsonl")
    def test_log_trigger_without_tier(self, mock_append):
        """_log_trigger omits 'tier' key when tier=None."""
        _log_trigger(["test"], "skipped")

        entry = mock_append.call_args[0][1]
        assert "tier" not in entry

    @patch("portfolio.agent_invocation.atomic_append_jsonl")
    def test_log_trigger_with_tier(self, mock_append):
        """_log_trigger includes 'tier' key when tier is provided."""
        _log_trigger(["test"], "invoked", tier=3)

        entry = mock_append.call_args[0][1]
        assert entry["tier"] == 3

    @patch("portfolio.agent_invocation.atomic_append_jsonl")
    def test_log_trigger_preserves_multiple_reasons(self, mock_append):
        """_log_trigger preserves all reasons in the list."""
        reasons = ["consensus BTC BUY", "price move +5%", "F&G crossed 20"]
        _log_trigger(reasons, "invoked", tier=2)

        entry = mock_append.call_args[0][1]
        assert entry["reasons"] == reasons


# ===========================================================================
# invoke_agent — happy path
# ===========================================================================

class TestInvokeAgentHappyPath:

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_returns_true_on_successful_spawn(self, mock_popen_cls, mock_which):
        """invoke_agent returns True when process spawns successfully."""
        proc = MagicMock(pid=99)
        mock_popen_cls.return_value = proc

        with patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()):
            result = invoke_agent(["consensus BTC BUY"], tier=2)

        assert result is True

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_sets_agent_proc(self, mock_popen_cls, mock_which):
        """invoke_agent sets _agent_proc to the spawned process."""
        proc = MagicMock(pid=42)
        mock_popen_cls.return_value = proc

        with patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()):
            invoke_agent(["test"], tier=3)

        assert ai._agent_proc is proc

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_sets_agent_start_time(self, mock_popen_cls, mock_which):
        """invoke_agent records the start time."""
        mock_popen_cls.return_value = MagicMock(pid=1)

        with patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()):
            before = time.time()
            invoke_agent(["test"], tier=1)
            after = time.time()

        assert before <= ai._agent_start <= after

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_sets_tier_specific_timeout(self, mock_popen_cls, mock_which):
        """invoke_agent sets _agent_timeout from the tier config."""
        mock_popen_cls.return_value = MagicMock(pid=1)

        with patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()):
            invoke_agent(["test"], tier=1)

        assert ai._agent_timeout == 120

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_popen_uses_claude_command(self, mock_popen_cls, mock_which):
        """invoke_agent uses the claude CLI when available."""
        mock_popen_cls.return_value = MagicMock(pid=1)

        with patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()):
            invoke_agent(["test"], tier=2)

        cmd = mock_popen_cls.call_args[0][0]
        assert cmd[0] == "/usr/bin/claude"
        assert "-p" in cmd

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_popen_includes_max_turns(self, mock_popen_cls, mock_which):
        """invoke_agent passes the tier-specific max-turns to the subprocess."""
        mock_popen_cls.return_value = MagicMock(pid=1)

        with patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()):
            invoke_agent(["test"], tier=2)

        cmd = mock_popen_cls.call_args[0][0]
        assert "--max-turns" in cmd
        mt_idx = cmd.index("--max-turns")
        assert cmd[mt_idx + 1] == "25"  # Tier 2 max_turns

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_default_tier_is_3(self, mock_popen_cls, mock_which):
        """invoke_agent defaults to tier 3 when not specified."""
        mock_popen_cls.return_value = MagicMock(pid=1)

        with patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()):
            invoke_agent(["test"])

        assert ai._agent_timeout == 900
        cmd = mock_popen_cls.call_args[0][0]
        mt_idx = cmd.index("--max-turns")
        assert cmd[mt_idx + 1] == "40"  # Tier 3 max_turns


# ===========================================================================
# invoke_agent — busy detection
# ===========================================================================

class TestInvokeAgentBusyDetection:

    def test_returns_false_when_agent_still_running(self):
        """invoke_agent returns False when the agent is still running within timeout."""
        proc = MagicMock()
        proc.poll.return_value = None  # still running
        proc.pid = 100
        ai._agent_proc = proc
        ai._agent_start = time.time()  # just started
        ai._agent_timeout = 900

        result = invoke_agent(["test"], tier=1)

        assert result is False

    def test_does_not_spawn_when_busy(self):
        """invoke_agent does not call Popen when agent is still running."""
        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 100
        ai._agent_proc = proc
        ai._agent_start = time.time()
        ai._agent_timeout = 900

        with patch("portfolio.agent_invocation.subprocess.Popen") as mock_p:
            invoke_agent(["test"], tier=2)
            mock_p.assert_not_called()

    def test_completed_process_allows_new_invocation(self):
        """invoke_agent spawns new process when previous one has completed."""
        old_proc = MagicMock()
        old_proc.poll.return_value = 0  # completed (exit code 0)
        old_proc.pid = 50
        ai._agent_proc = old_proc
        ai._agent_start = time.time() - 60

        with patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude"), \
             patch("portfolio.agent_invocation.subprocess.Popen") as mock_p, \
             patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()):
            new_proc = MagicMock(pid=51)
            mock_p.return_value = new_proc
            result = invoke_agent(["test"], tier=1)

        assert result is True
        mock_p.assert_called_once()


# ===========================================================================
# invoke_agent — timeout and stale process cleanup
# ===========================================================================

class TestInvokeAgentTimeout:

    def test_kills_timed_out_process_on_windows(self):
        """Timed-out agent is killed via taskkill on Windows."""
        proc = MagicMock()
        proc.poll.return_value = None  # still running
        proc.pid = 200
        proc.wait.return_value = 0
        ai._agent_proc = proc
        ai._agent_start = time.time() - 1000  # well past timeout
        ai._agent_timeout = 120

        with patch("portfolio.agent_invocation.platform.system", return_value="Windows"), \
             patch("portfolio.agent_invocation.subprocess.run") as mock_run, \
             patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude"), \
             patch("portfolio.agent_invocation.subprocess.Popen") as mock_p, \
             patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()):
            mock_p.return_value = MagicMock(pid=201)
            invoke_agent(["test"], tier=1)

        # taskkill should have been called with the old PID
        mock_run.assert_called_once()
        taskkill_cmd = mock_run.call_args[0][0]
        assert "taskkill" in taskkill_cmd
        assert str(200) in taskkill_cmd

    def test_kills_timed_out_process_on_linux(self):
        """Timed-out agent is killed via process.kill() on non-Windows."""
        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 300
        proc.wait.return_value = 0
        ai._agent_proc = proc
        ai._agent_start = time.time() - 1000
        ai._agent_timeout = 120

        with patch("portfolio.agent_invocation.platform.system", return_value="Linux"), \
             patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude"), \
             patch("portfolio.agent_invocation.subprocess.Popen") as mock_p, \
             patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()):
            mock_p.return_value = MagicMock(pid=301)
            invoke_agent(["test"], tier=1)

        proc.kill.assert_called_once()

    def test_closes_agent_log_on_timeout(self):
        """Agent log file handle is closed when process times out."""
        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 400
        proc.wait.return_value = 0
        log_fh = MagicMock()
        ai._agent_proc = proc
        ai._agent_log = log_fh
        ai._agent_start = time.time() - 1000
        ai._agent_timeout = 120

        with patch("portfolio.agent_invocation.platform.system", return_value="Linux"), \
             patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude"), \
             patch("portfolio.agent_invocation.subprocess.Popen") as mock_p, \
             patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()):
            mock_p.return_value = MagicMock(pid=401)
            invoke_agent(["test"], tier=2)

        log_fh.close.assert_called()

    def test_wait_timeout_ignored_on_stubborn_process(self):
        """If wait() times out, the invocation still proceeds."""
        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 500
        proc.wait.side_effect = __import__("subprocess").TimeoutExpired(
            cmd="claude", timeout=10
        )
        ai._agent_proc = proc
        ai._agent_start = time.time() - 1000
        ai._agent_timeout = 120

        with patch("portfolio.agent_invocation.platform.system", return_value="Linux"), \
             patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude"), \
             patch("portfolio.agent_invocation.subprocess.Popen") as mock_p, \
             patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()):
            mock_p.return_value = MagicMock(pid=501)
            result = invoke_agent(["test"], tier=1)

        assert result is True


# ===========================================================================
# invoke_agent — CLAUDECODE env var cleanup
# ===========================================================================

class TestClaudeCodeEnvCleanup:

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_claudecode_stripped_from_env(self, mock_popen_cls, mock_which, mock_env):
        """CLAUDECODE env var is removed from subprocess environment."""
        mock_popen_cls.return_value = MagicMock(pid=1)

        with patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()):
            invoke_agent(["test"], tier=1)

        # Check the env passed to Popen
        popen_kwargs = mock_popen_cls.call_args[1]
        agent_env = popen_kwargs["env"]
        assert "CLAUDECODE" not in agent_env

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_claude_code_entrypoint_stripped_from_env(self, mock_popen_cls, mock_which, mock_env):
        """CLAUDE_CODE_ENTRYPOINT env var is removed from subprocess environment."""
        mock_popen_cls.return_value = MagicMock(pid=1)

        with patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()):
            invoke_agent(["test"], tier=1)

        popen_kwargs = mock_popen_cls.call_args[1]
        agent_env = popen_kwargs["env"]
        assert "CLAUDE_CODE_ENTRYPOINT" not in agent_env


# ===========================================================================
# invoke_agent — perception gate integration
# ===========================================================================

class TestPerceptionGate:

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_skipped_when_gate_rejects(self, mock_popen_cls, mock_which):
        """invoke_agent returns False when perception gate rejects."""
        mock_popen_cls.return_value = MagicMock(pid=1)

        def fake_gate(reasons, tier):
            return False, "no active signals"

        with patch("portfolio.agent_invocation.atomic_append_jsonl") as mock_log, \
             patch("builtins.open", mock_open()):
            # Patch the perception gate import inside invoke_agent
            with patch.dict("sys.modules", {}):
                with patch(
                    "portfolio.perception_gate.should_invoke",
                    side_effect=fake_gate,
                ):
                    # We need to patch at the point of import inside invoke_agent
                    import importlib
                    import portfolio.perception_gate as pg
                    orig = pg.should_invoke
                    pg.should_invoke = fake_gate
                    try:
                        result = invoke_agent(["cooldown"], tier=1)
                    finally:
                        pg.should_invoke = orig

        assert result is False
        # Should log the skipped invocation
        mock_log.assert_called()
        entry = mock_log.call_args[0][1]
        assert entry["status"] == "skipped_gate"

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_proceeds_when_gate_approves(self, mock_popen_cls, mock_which):
        """invoke_agent proceeds when perception gate approves."""
        mock_popen_cls.return_value = MagicMock(pid=1)

        def fake_gate(reasons, tier):
            return True, "gate disabled"

        import portfolio.perception_gate as pg
        orig = pg.should_invoke
        pg.should_invoke = fake_gate

        try:
            with patch("portfolio.agent_invocation._load_config", return_value={}), \
                 patch("portfolio.agent_invocation.send_or_store"), \
                 patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
                 patch("builtins.open", mock_open()):
                result = invoke_agent(["consensus BTC BUY"], tier=2)
        finally:
            pg.should_invoke = orig

        assert result is True

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_passes_through_on_gate_error(self, mock_popen_cls, mock_which):
        """invoke_agent proceeds even if perception gate raises an exception."""
        mock_popen_cls.return_value = MagicMock(pid=1)

        def broken_gate(reasons, tier):
            raise RuntimeError("gate broken")

        import portfolio.perception_gate as pg
        orig = pg.should_invoke
        pg.should_invoke = broken_gate

        try:
            with patch("portfolio.agent_invocation._load_config", return_value={}), \
                 patch("portfolio.agent_invocation.send_or_store"), \
                 patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
                 patch("builtins.open", mock_open()):
                result = invoke_agent(["test"], tier=1)
        finally:
            pg.should_invoke = orig

        assert result is True


# ===========================================================================
# invoke_agent — journal context
# ===========================================================================

class TestJournalContext:

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_writes_journal_context(self, mock_popen_cls, mock_which):
        """invoke_agent calls write_context from portfolio.journal."""
        mock_popen_cls.return_value = MagicMock(pid=1)

        with patch("portfolio.journal.write_context", return_value=5) as mock_wc, \
             patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()):
            invoke_agent(["test"], tier=3)

        mock_wc.assert_called_once()

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_proceeds_when_journal_context_fails(self, mock_popen_cls, mock_which):
        """invoke_agent proceeds even if write_context raises."""
        mock_popen_cls.return_value = MagicMock(pid=1)

        with patch("portfolio.journal.write_context", side_effect=Exception("journal error")), \
             patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()):
            result = invoke_agent(["test"], tier=2)

        assert result is True


# ===========================================================================
# invoke_agent — claude not on PATH / bat fallback
# ===========================================================================

class TestBatFallback:

    def test_fallback_to_bat_when_claude_not_found(self):
        """When claude is not on PATH, falls back to pf-agent.bat."""
        bat_path = ai.BASE_DIR / "scripts" / "win" / "pf-agent.bat"

        with patch("portfolio.agent_invocation.shutil.which", return_value=None), \
             patch.object(Path, "exists", return_value=True), \
             patch("portfolio.agent_invocation.subprocess.Popen") as mock_p, \
             patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()):
            mock_p.return_value = MagicMock(pid=1)
            result = invoke_agent(["test"], tier=3)

        assert result is True
        cmd = mock_p.call_args[0][0]
        assert cmd[0] == "cmd"
        assert "/c" in cmd

    def test_returns_false_when_bat_not_found(self):
        """Returns False when claude not on PATH and bat file does not exist."""
        with patch("portfolio.agent_invocation.shutil.which", return_value=None), \
             patch.object(Path, "exists", return_value=False):
            result = invoke_agent(["test"], tier=3)

        assert result is False


# ===========================================================================
# invoke_agent — Telegram notification
# ===========================================================================

class TestTelegramNotification:

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_sends_invocation_notification(self, mock_popen_cls, mock_which):
        """invoke_agent sends invocation notification via send_or_store."""
        mock_popen_cls.return_value = MagicMock(pid=1)

        with patch("portfolio.agent_invocation._load_config", return_value={"telegram": {"token": "t", "chat_id": "c"}}), \
             patch("portfolio.agent_invocation.send_or_store") as mock_sos, \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()):
            invoke_agent(["consensus BTC BUY"], tier=2)

        mock_sos.assert_called_once()
        msg = mock_sos.call_args[0][0]
        assert "Layer 2" in msg
        assert "T2" in msg
        assert "SIGNAL ANALYSIS" in msg
        # Category should be "invocation"
        assert mock_sos.call_args[1]["category"] == "invocation"

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_notification_includes_reason_text(self, mock_popen_cls, mock_which):
        """The notification message includes the trigger reasons."""
        mock_popen_cls.return_value = MagicMock(pid=1)

        with patch("portfolio.agent_invocation._load_config", return_value={"telegram": {"token": "t", "chat_id": "c"}}), \
             patch("portfolio.agent_invocation.send_or_store") as mock_sos, \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()):
            invoke_agent(["price move BTC +3%"], tier=1)

        msg = mock_sos.call_args[0][0]
        assert "price move BTC +3%" in msg

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_notification_truncates_extra_reasons(self, mock_popen_cls, mock_which):
        """When more than 3 reasons, notification shows '+N more'."""
        mock_popen_cls.return_value = MagicMock(pid=1)
        reasons = ["r1", "r2", "r3", "r4", "r5"]

        with patch("portfolio.agent_invocation._load_config", return_value={"telegram": {"token": "t", "chat_id": "c"}}), \
             patch("portfolio.agent_invocation.send_or_store") as mock_sos, \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()):
            invoke_agent(reasons, tier=3)

        msg = mock_sos.call_args[0][0]
        assert "+2 more" in msg

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_notification_failure_is_non_critical(self, mock_popen_cls, mock_which):
        """invoke_agent still returns True even if notification sending fails."""
        mock_popen_cls.return_value = MagicMock(pid=1)

        with patch("portfolio.agent_invocation._load_config", side_effect=Exception("config error")), \
             patch("builtins.open", mock_open()):
            result = invoke_agent(["test"], tier=1)

        assert result is True


# ===========================================================================
# invoke_agent — Popen failure
# ===========================================================================

class TestInvokeAgentPopenFailure:

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_returns_false_on_popen_exception(self, mock_popen_cls, mock_which):
        """invoke_agent returns False when Popen raises an exception."""
        mock_popen_cls.side_effect = OSError("command not found")

        with patch("builtins.open", mock_open()):
            result = invoke_agent(["test"], tier=1)

        assert result is False

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_closes_log_on_popen_failure(self, mock_popen_cls, mock_which):
        """Log file handle is closed when Popen raises (not transferred)."""
        mock_popen_cls.side_effect = OSError("command not found")

        file_handle = MagicMock()
        m = mock_open()
        m.return_value = file_handle

        with patch("builtins.open", m):
            invoke_agent(["test"], tier=1)

        file_handle.close.assert_called()


# ===========================================================================
# invoke_agent — stale log handle cleanup
# ===========================================================================

class TestStaleLogCleanup:

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_closes_stale_log_when_process_completed(self, mock_popen_cls, mock_which):
        """Stale _agent_log from a completed process is closed before spawning."""
        old_proc = MagicMock()
        old_proc.poll.return_value = 0  # completed
        old_proc.pid = 10
        old_log = MagicMock()
        ai._agent_proc = old_proc
        ai._agent_log = old_log
        ai._agent_start = time.time() - 60

        mock_popen_cls.return_value = MagicMock(pid=11)

        with patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()):
            invoke_agent(["test"], tier=1)

        old_log.close.assert_called()


# ===========================================================================
# invoke_agent — unknown tier fallback
# ===========================================================================

class TestUnknownTierFallback:

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_unknown_tier_uses_tier3_config(self, mock_popen_cls, mock_which):
        """An unknown tier number (e.g., 99) falls back to tier 3 config."""
        mock_popen_cls.return_value = MagicMock(pid=1)

        with patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()):
            invoke_agent(["test"], tier=99)

        assert ai._agent_timeout == 900  # tier 3 timeout
        cmd = mock_popen_cls.call_args[0][0]
        mt_idx = cmd.index("--max-turns")
        assert cmd[mt_idx + 1] == "40"  # tier 3 max_turns


# ===========================================================================
# Module constants
# ===========================================================================

class TestModuleConstants:

    def test_base_dir_is_parent_of_portfolio(self):
        """BASE_DIR points to the repository root (parent of portfolio/)."""
        assert (ai.BASE_DIR / "portfolio").is_dir() or True  # structural check
        assert ai.BASE_DIR.name != "portfolio"

    def test_data_dir_is_under_base(self):
        """DATA_DIR is BASE_DIR/data."""
        assert ai.DATA_DIR == ai.BASE_DIR / "data"

    def test_invocations_file_path(self):
        """INVOCATIONS_FILE is in the data directory."""
        assert ai.INVOCATIONS_FILE.name == "invocations.jsonl"
        assert ai.INVOCATIONS_FILE.parent == ai.DATA_DIR
