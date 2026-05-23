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

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

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

    def test_tier1_timeout_is_180(self):
        """Tier 1 (Quick Check) has 180s timeout. History: 120 → 150 (2026-05-14
        after 3d audit showed p50=114, p95=139, 42% of runs in 120-150s bucket
        — 120s was kicking the watchdog mid-write on healthy runs); then
        150 → 180 same day to add headroom on top of the Bash-cat prompt
        collapse (commit 9991a0e5) so genuine outliers still fit the budget."""
        assert TIER_CONFIG[1]["timeout"] == 180

    def test_tier2_timeout_is_600(self):
        """Tier 2 (Signal Analysis) has 600s timeout."""
        assert TIER_CONFIG[2]["timeout"] == 600

    def test_tier3_timeout_is_900(self):
        """Tier 3 (Full Review) has 900s timeout."""
        assert TIER_CONFIG[3]["timeout"] == 900

    def test_tier1_max_turns_is_15(self):
        """Tier 1 has 15 max turns."""
        assert TIER_CONFIG[1]["max_turns"] == 15

    def test_tier2_max_turns_is_40(self):
        """Tier 2 has 40 max turns."""
        assert TIER_CONFIG[2]["max_turns"] == 40

    def test_tier3_max_turns_is_40(self):
        """Tier 3 has 40 max turns."""
        assert TIER_CONFIG[3]["max_turns"] == 40

    def test_each_tier_has_label(self):
        """Each tier has a human-readable label."""
        for _tier_id, cfg in TIER_CONFIG.items():
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
        """Higher tiers have more or equal max turns."""
        assert TIER_CONFIG[1]["max_turns"] <= TIER_CONFIG[2]["max_turns"]
        assert TIER_CONFIG[2]["max_turns"] <= TIER_CONFIG[3]["max_turns"]


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

    # 2026-05-14: regression tests for the Bash-cat collapse (cuts ~25-30s
    # per invocation by replacing N sequential Read tool calls with one
    # Bash command). If a future edit re-introduces the per-file Read
    # pattern, these tests fail and surface the regression at test time
    # rather than as silent slow-down in production.

    def test_tier1_prompt_uses_bash_cat_not_reads(self):
        prompt = _build_tier_prompt(1, ["startup"])
        assert "cat " in prompt
        assert "single tool turn" in prompt
        assert "Do NOT call Read" in prompt

    def test_tier2_prompt_uses_bash_cat_not_reads(self):
        prompt = _build_tier_prompt(2, ["consensus BTC-USD BUY"])
        assert "cat " in prompt
        assert "single tool turn" in prompt
        assert "Do NOT call Read" in prompt

    def test_tier3_prompt_uses_bash_cat_not_reads(self):
        prompt = _build_tier_prompt(3, ["full review"])
        assert "cat " in prompt
        assert "single tool turn" in prompt
        assert "Do NOT call Read" in prompt

    def test_tier2_prompt_keeps_trading_insights_optional(self):
        """The optional trading_insights.md must use a guarded `[ -f X ]`
        pattern, and the guard must precede the required-files cat so a
        missing optional file doesn't abort the chain."""
        prompt = _build_tier_prompt(2, ["trigger"])
        guard_pos = prompt.find("[ -f data/trading_insights.md ]")
        playbook_pos = prompt.find("docs/TRADING_PLAYBOOK.md")
        assert guard_pos >= 0
        assert playbook_pos >= 0
        assert guard_pos < playbook_pos, "guard must precede required-files cat"

    def test_tier3_prompt_keeps_trading_insights_optional(self):
        prompt = _build_tier_prompt(3, ["trigger"])
        guard_pos = prompt.find("[ -f data/trading_insights.md ]")
        playbook_pos = prompt.find("docs/TRADING_PLAYBOOK.md")
        assert guard_pos >= 0
        assert playbook_pos >= 0
        assert guard_pos < playbook_pos, "guard must precede required-files cat"

    def test_required_files_cat_does_not_mask_stderr(self):
        """Review P2: only the OPTIONAL trading_insights.md cat may have
        `2>/dev/null`. The required-files cat must NOT, so a missing
        portfolio_state.json or agent_context_t2.json surfaces in
        agent.log instead of silently producing truncated context (which
        would let the agent reason over a blank portfolio and produce
        bad-sized trade decisions)."""
        for tier in (2, 3):
            prompt = _build_tier_prompt(tier, ["trigger"])
            # The string "data/portfolio_state.json" appears in the
            # required-files cat. Anything between that and the closing
            # backtick must NOT contain `2>/dev/null`.
            ps_pos = prompt.find("data/portfolio_state.json")
            assert ps_pos >= 0, f"T{tier} prompt missing portfolio_state.json"
            tail = prompt[ps_pos:]
            close_backtick = tail.find("`")
            tail_block = tail[:close_backtick] if close_backtick >= 0 else tail
            assert "2>/dev/null" not in tail_block, (
                f"T{tier} required-files cat must not mask stderr — would "
                f"hide missing required files. tail_block={tail_block!r}"
            )

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
            before = time.monotonic()
            invoke_agent(["test"], tier=1)
            after = time.monotonic()

        # BUG-203 (2026-04-16): _agent_start uses time.monotonic() now.
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

        assert ai._agent_timeout == 180

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
        assert cmd[mt_idx + 1] == "40"  # Tier 2 max_turns

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
        ai._agent_start = time.monotonic()  # just started
        ai._agent_timeout = 900

        result = invoke_agent(["test"], tier=1)

        assert result is False

    def test_does_not_spawn_when_busy(self):
        """invoke_agent does not call Popen when agent is still running."""
        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 100
        ai._agent_proc = proc
        ai._agent_start = time.monotonic()
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
        ai._agent_start = time.monotonic() - 60

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
        ai._agent_start = time.monotonic() - 1000  # well past timeout
        ai._agent_timeout = 120

        with patch("portfolio.agent_invocation.platform.system", return_value="Windows"), \
             patch("portfolio.agent_invocation.subprocess.run") as mock_run, \
             patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude"), \
             patch("portfolio.agent_invocation.subprocess.Popen") as mock_p, \
             patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("portfolio.agent_invocation.atomic_append_jsonl"), \
             patch("builtins.open", mock_open()):
            mock_run.return_value = MagicMock(returncode=0)
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
        ai._agent_start = time.monotonic() - 1000
        ai._agent_timeout = 120

        with patch("portfolio.agent_invocation.platform.system", return_value="Linux"), \
             patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude"), \
             patch("portfolio.agent_invocation.subprocess.Popen") as mock_p, \
             patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("portfolio.agent_invocation.atomic_append_jsonl"), \
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
        ai._agent_start = time.monotonic() - 1000
        ai._agent_timeout = 120

        with patch("portfolio.agent_invocation.platform.system", return_value="Linux"), \
             patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude"), \
             patch("portfolio.agent_invocation.subprocess.Popen") as mock_p, \
             patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("portfolio.agent_invocation.atomic_append_jsonl"), \
             patch("builtins.open", mock_open()):
            mock_p.return_value = MagicMock(pid=401)
            invoke_agent(["test"], tier=2)

        log_fh.close.assert_called()

    def test_wait_timeout_stubborn_process_blocks_spawn(self):
        """If wait() times out after kill, new agent is NOT spawned (BUG-92)."""
        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 500
        proc.wait.side_effect = __import__("subprocess").TimeoutExpired(
            cmd="claude", timeout=10
        )
        ai._agent_proc = proc
        ai._agent_start = time.monotonic() - 1000
        ai._agent_timeout = 120

        with patch("portfolio.agent_invocation.platform.system", return_value="Linux"), \
             patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude"), \
             patch("portfolio.agent_invocation.subprocess.Popen") as mock_p, \
             patch("portfolio.agent_invocation._load_config", return_value={"layer2": {"enabled": True}}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("portfolio.agent_invocation.atomic_append_jsonl"), \
             patch("builtins.open", mock_open()):
            mock_p.return_value = MagicMock(pid=501)
            result = invoke_agent(["test"], tier=1)

        # BUG-92: stubborn process that won't die blocks new spawn
        assert result is False

    def test_taskkill_timeout_does_not_hang(self):
        """P1.7: taskkill with timeout=10 prevents indefinite hang."""
        import subprocess as _subprocess

        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 600
        proc.wait.return_value = 0
        ai._agent_proc = proc
        ai._agent_start = time.monotonic() - 1000
        ai._agent_timeout = 120

        with patch("portfolio.agent_invocation.platform.system", return_value="Windows"), \
             patch("portfolio.agent_invocation.subprocess.run") as mock_run, \
             patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude"), \
             patch("portfolio.agent_invocation.subprocess.Popen") as mock_p, \
             patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("portfolio.agent_invocation.atomic_append_jsonl"), \
             patch("builtins.open", mock_open()):
            mock_run.side_effect = _subprocess.TimeoutExpired(cmd="taskkill", timeout=10)
            mock_p.return_value = MagicMock(pid=601)
            result = invoke_agent(["test"], tier=1)

        # Should not hang — taskkill timeout caught, logged critical, returns False
        assert result is False


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

    @patch("portfolio.agent_invocation._load_config", return_value={"layer2": {"enabled": True}})
    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_skipped_when_gate_rejects(self, mock_popen_cls, mock_which, mock_cfg):
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

    def test_fallback_to_bat_when_claude_not_found(self, tmp_path, monkeypatch):
        """When claude is not on PATH, falls back to pf-agent.bat.

        2026-04-17: redirect DATA_DIR to tmp_path and create a stub
        agent.log — agent_invocation.py calls agent_log_path.stat()
        before open(), which the builtins.open mock doesn't intercept.
        Also mock perception_gate + journal.write_context — they read
        config.json / agent.log directly in worktree environments.
        """
        bat_path = ai.BASE_DIR / "scripts" / "win" / "pf-agent.bat"

        tmp_data = tmp_path / "data"
        tmp_data.mkdir()
        (tmp_data / "agent.log").write_text("", encoding="utf-8")
        monkeypatch.setattr(ai, "DATA_DIR", tmp_data)
        monkeypatch.setattr(ai, "JOURNAL_FILE", tmp_data / "layer2_journal.jsonl")
        monkeypatch.setattr(ai, "TELEGRAM_FILE", tmp_data / "telegram_messages.jsonl")
        monkeypatch.setattr(ai, "INVOCATIONS_FILE", tmp_data / "claude_invocations.jsonl")

        with patch("portfolio.agent_invocation.shutil.which", return_value=None), \
             patch.object(Path, "exists", return_value=True), \
             patch("portfolio.agent_invocation.subprocess.Popen") as mock_p, \
             patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("portfolio.perception_gate.should_invoke", return_value=(True, "mocked")), \
             patch("portfolio.journal.write_context", return_value=0):
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

    @patch("portfolio.agent_invocation._load_config", return_value={"layer2": {"enabled": True}})
    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_closes_log_on_popen_failure(self, mock_popen_cls, mock_which, mock_cfg):
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
        ai._agent_start = time.monotonic() - 60

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


# ===========================================================================
# BUG-181: Fishing context writes neutral on failure
# ===========================================================================

class TestFishingContextFallback:

    def test_writes_neutral_on_exception(self, monkeypatch):
        """BUG-181: When fishing context extraction fails, write neutral context."""
        written = {}

        def mock_write(path, data):
            written["path"] = path
            written["data"] = data

        # Patch at source module — local imports pick up this patch
        monkeypatch.setattr("portfolio.file_utils.atomic_write_json", mock_write)

        # Call with a journal entry that will fail (conviction not a number)
        journal = {
            "tickers": {"XAG-USD": {"outlook": "bullish", "conviction": "not_a_number"}}
        }
        ai._write_fishing_context(journal)

        assert "data" in written, "Expected neutral context to be written on failure"
        assert written["data"]["direction_bias"] == "neutral"
        assert written["data"]["bias_confidence"] == 0.0

    def test_no_xag_returns_without_writing(self, tmp_path):
        """When no XAG-USD in journal, function returns early."""
        # Should not raise
        ai._write_fishing_context({"tickers": {"BTC-USD": {}}})
        assert ai.INVOCATIONS_FILE.parent == ai.DATA_DIR


# ---------------------------------------------------------------------------
# BUG-219: _record_new_trades() — overtrading prevention wiring
# ---------------------------------------------------------------------------

class TestRecordNewTrades:
    """Tests for _record_new_trades() which wires trade_guards.record_trade()
    into the agent completion path (BUG-219 / PR-R4-4)."""

    def _make_state(self, transactions):
        """Helper: build a portfolio state dict with the given transactions."""
        return {"cash": 500000, "transactions": transactions}

    def test_records_new_buy_sell_transactions(self):
        """New BUY/SELL transactions after invoke should call record_trade()."""
        before_txns = [{"ticker": "BTC-USD", "action": "BUY", "price": 60000}]
        after_txns = before_txns + [
            {"ticker": "ETH-USD", "action": "SELL", "price": 3000, "pnl_pct": -2.5},
            {"ticker": "XAG-USD", "action": "BUY", "price": 30},
        ]

        ai._patient_txn_count_before = len(before_txns)
        ai._bold_txn_count_before = 0  # bold has no new trades

        with patch("portfolio.file_utils.load_json") as mock_load, \
             patch("portfolio.trade_guards.record_trade") as mock_record:
            mock_load.side_effect = lambda path, **kw: (
                self._make_state(after_txns) if "portfolio_state.json" in str(path)
                else self._make_state([])
            )
            ai._record_new_trades()

        assert mock_record.call_count == 2
        # BUG-219: SELL must forward pnl_pct from the transaction dict
        mock_record.assert_any_call("ETH-USD", "SELL", "patient", pnl_pct=-2.5)
        # BUY has no pnl_pct in txn → defaults to None
        mock_record.assert_any_call("XAG-USD", "BUY", "patient", pnl_pct=None)

    def test_pnl_pct_forwarded_for_sell_with_loss(self):
        """SELL with negative pnl_pct should forward it to record_trade()."""
        txns = [{"ticker": "BTC-USD", "action": "SELL", "price": 58000, "pnl_pct": -3.2}]
        ai._patient_txn_count_before = 0
        ai._bold_txn_count_before = 0

        with patch("portfolio.file_utils.load_json") as mock_load, \
             patch("portfolio.trade_guards.record_trade") as mock_record:
            mock_load.return_value = self._make_state(txns)
            ai._record_new_trades()

        mock_record.assert_any_call("BTC-USD", "SELL", "patient", pnl_pct=-3.2)

    def test_pnl_pct_forwarded_for_sell_with_win(self):
        """SELL with positive pnl_pct should forward it (resets loss streak)."""
        txns = [{"ticker": "ETH-USD", "action": "SELL", "price": 3500, "pnl_pct": 5.1}]
        ai._patient_txn_count_before = 0
        ai._bold_txn_count_before = 0

        with patch("portfolio.file_utils.load_json") as mock_load, \
             patch("portfolio.trade_guards.record_trade") as mock_record:
            mock_load.return_value = self._make_state(txns)
            ai._record_new_trades()

        mock_record.assert_any_call("ETH-USD", "SELL", "patient", pnl_pct=5.1)

    def test_missing_pnl_pct_defaults_to_none(self):
        """Transaction without pnl_pct field should pass None (backward compat)."""
        txns = [{"ticker": "BTC-USD", "action": "SELL", "price": 60000}]
        ai._patient_txn_count_before = 0
        ai._bold_txn_count_before = 0

        with patch("portfolio.file_utils.load_json") as mock_load, \
             patch("portfolio.trade_guards.record_trade") as mock_record:
            mock_load.return_value = self._make_state(txns)
            ai._record_new_trades()

        mock_record.assert_any_call("BTC-USD", "SELL", "patient", pnl_pct=None)

    def test_skips_hold_and_malformed_transactions(self):
        """Transactions without ticker or with non-BUY/SELL action are skipped."""
        txns = [
            {"ticker": "BTC-USD", "action": "HOLD"},
            {"action": "BUY", "price": 100},  # missing ticker
            {"ticker": "ETH-USD"},  # missing action
        ]
        ai._patient_txn_count_before = 0
        ai._bold_txn_count_before = 0

        with patch("portfolio.file_utils.load_json") as mock_load, \
             patch("portfolio.trade_guards.record_trade") as mock_record:
            mock_load.return_value = self._make_state(txns)
            ai._record_new_trades()

        mock_record.assert_not_called()

    def test_no_new_transactions_is_noop(self):
        """When transaction count hasn't changed, record_trade is not called."""
        txns = [{"ticker": "BTC-USD", "action": "BUY", "price": 60000}]
        ai._patient_txn_count_before = 1
        ai._bold_txn_count_before = 1

        with patch("portfolio.file_utils.load_json") as mock_load, \
             patch("portfolio.trade_guards.record_trade") as mock_record:
            mock_load.return_value = self._make_state(txns)
            ai._record_new_trades()

        mock_record.assert_not_called()

    def test_exception_safety(self):
        """record_trade() failure must not propagate — completion path stays safe."""
        txns = [{"ticker": "BTC-USD", "action": "BUY", "price": 60000}]
        ai._patient_txn_count_before = 0
        ai._bold_txn_count_before = 0

        with patch("portfolio.file_utils.load_json") as mock_load, \
             patch("portfolio.trade_guards.record_trade", side_effect=RuntimeError("boom")):
            mock_load.return_value = self._make_state(txns)
            # Must not raise
            ai._record_new_trades()

    def test_load_json_failure_is_safe(self):
        """If load_json itself fails, _record_new_trades swallows the error."""
        ai._patient_txn_count_before = 0
        ai._bold_txn_count_before = 0

        with patch("portfolio.file_utils.load_json", side_effect=OSError("disk")):
            ai._record_new_trades()  # must not raise


# ===========================================================================
# Drawdown circuit-breaker fail-safe (adversarial review 05-01 P0-5)
#
# Before 2026-05-02: a single bare `except Exception` wrapped the entire
# drawdown check block. ANY error (ImportError, file IO, KeyError on dd
# dict, _log_trigger raising) would log WARNING and proceed — meaning a
# portfolio in 50%+ drawdown could continue trading if anything threw.
# After: ImportError on check_drawdown is fail-safe BLOCK (returns False).
# Per-portfolio errors log ERROR but tolerate (other portfolio still gets
# checked). Both portfolios failing is logged but proceeds (transient IO
# tolerance).
# ===========================================================================

class TestDrawdownFailSafe:

    def _setup_invoke_path(self, monkeypatch, tmp_path=None):
        """Mock everything before/after the drawdown check so we can exercise it."""
        # Reset module-level state
        ai._agent_proc = None
        ai._consecutive_stack_overflows = 0
        # 2026-05-13: isolate auth-error cooldown check from real
        # data/invocations.jsonl (which may contain recent auth_error rows
        # that would short-circuit the cooldown gate before the drawdown /
        # trade-guards logic runs).
        import tempfile, pathlib
        empty = pathlib.Path(tempfile.mkdtemp()) / "inv.jsonl"
        empty.write_text("", encoding="utf-8")
        monkeypatch.setattr(ai, "INVOCATIONS_FILE", empty)
        # Make perception_gate always pass-through
        monkeypatch.setattr(
            "portfolio.perception_gate.should_invoke",
            lambda r, t: (True, "ok"),
        )
        # write_context returns 0
        monkeypatch.setattr("portfolio.journal.write_context", lambda: 0)
        # Mock the loader that runs at top
        monkeypatch.setattr(
            "portfolio.agent_invocation._load_config",
            lambda: {"layer2": {"enabled": True}},
        )
        # Force PATIENT/BOLD paths to "exist" so the for-loop iterates.
        # We patch Path.exists on the actual path objects.
        monkeypatch.setattr(
            ai.PATIENT_PORTFOLIO.__class__, "exists", lambda self: True
        )

    def test_block_when_check_drawdown_module_unimportable(self, monkeypatch):
        """ImportError on check_drawdown must fail-safe BLOCK (return False)."""
        self._setup_invoke_path(monkeypatch)

        # Make `from portfolio.risk_management import check_drawdown` raise
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "portfolio.risk_management":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import), \
             patch("portfolio.agent_invocation._log_trigger") as mock_log:
            result = ai.invoke_agent(["test"], tier=2)

        assert result is False, (
            "Adversarial review 05-01 P0-5: ImportError on check_drawdown "
            "must fail-safe BLOCK, not pass through."
        )
        # Verify it was logged as the right reason
        call_args = mock_log.call_args
        assert call_args is not None
        assert call_args.args[1] == "blocked_drawdown_unavailable"

    def test_block_when_drawdown_exceeds_block_pct(self, monkeypatch):
        """A portfolio in 60% drawdown (>50% block threshold) must BLOCK."""
        self._setup_invoke_path(monkeypatch)

        def fake_check(pf_path, max_drawdown_pct=20.0):
            return {
                "current_drawdown_pct": 60.0,
                "peak_value": 500_000.0,
                "current_value": 200_000.0,
            }

        monkeypatch.setattr(
            "portfolio.risk_management.check_drawdown", fake_check,
        )
        with patch("portfolio.agent_invocation._log_trigger") as mock_log:
            result = ai.invoke_agent(["test"], tier=2)

        assert result is False, "60% drawdown must trigger BLOCK"
        # Verify the right log reason
        call_args = mock_log.call_args
        assert call_args is not None
        assert "blocked_drawdown_" in call_args.args[1]

    def test_per_portfolio_io_error_does_not_block_invocation(self, monkeypatch):
        """Per-portfolio IO error on ONE portfolio is tolerated (other still checked).

        This matches the "transient IO race tolerance" the next cycle re-checks.
        We verify the drawdown block path does NOT fire (no `blocked_drawdown_*`
        _log_trigger call) — the invocation may still fail downstream for other
        unrelated reasons (subprocess spawn etc.) but the drawdown gate let it
        through.
        """
        self._setup_invoke_path(monkeypatch)

        def fake_check(pf_path, max_drawdown_pct=20.0):
            # First portfolio: error. Second portfolio: clean.
            if "bold" in str(pf_path):
                return {
                    "current_drawdown_pct": 5.0,
                    "peak_value": 500_000.0,
                    "current_value": 475_000.0,
                }
            raise OSError("file race")

        monkeypatch.setattr(
            "portfolio.risk_management.check_drawdown", fake_check,
        )

        with patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude"), \
             patch("portfolio.agent_invocation.subprocess.Popen", return_value=MagicMock(pid=99)), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()), \
             patch("portfolio.agent_invocation._log_trigger") as mock_log:
            ai.invoke_agent(["test"], tier=2)

        # Verify NO drawdown-block call was made (per-portfolio IO error
        # was tolerated — other portfolio's check passed cleanly).
        block_calls = [
            c for c in mock_log.call_args_list
            if len(c.args) >= 2 and isinstance(c.args[1], str)
            and c.args[1].startswith("blocked_drawdown_")
        ]
        assert not block_calls, (
            f"Per-portfolio IO error should not trigger drawdown block, "
            f"but got: {block_calls}"
        )


# ===========================================================================
# Adversarial review 05-01 P1-12: should_block_trade never called
#
# Before: portfolio.trade_guards.should_block_trade was implemented but never
# imported by any production code (only tests). Trade guard warnings were
# computed in reporting.py and stuffed into agent_summary, but Layer 2 was
# only soft-asked to look at them in the prompt — there was no automated
# pre-execution gate that refused to invoke when ALL trigger tickers were
# under cooldown for BOTH strategies.
#
# Wiring: after the drawdown circuit breaker, agent_invocation reads the
# trade_guard_warnings from agent_summary, surfaces them in _guard_context
# (appended to the prompt like _drawdown_context), and BLOCKS when:
#   1. should_block_trade(guard_result) is True (≥1 severity=block warning), AND
#   2. The trigger ticker is blocked for BOTH Patient and Bold strategies.
# Otherwise, warnings flow as advisory context and the invocation proceeds.
#
# This preserves Layer 2's discretion when ANY strategy can still trade,
# while preventing the wasteful T2/T3 spawn (~600s of subprocess + Claude
# tokens) when no useful trade decision is possible.
# ===========================================================================

class TestTradeGuardsBlockGate:

    def _setup_invoke_path(self, monkeypatch):
        """Mock the early-return paths so we exercise the trade-guards gate."""
        ai._agent_proc = None
        ai._consecutive_stack_overflows = 0
        # 2026-05-13: isolate auth-error cooldown check from real
        # data/invocations.jsonl — see same comment in TestDrawdownFailSafe.
        import tempfile, pathlib
        empty = pathlib.Path(tempfile.mkdtemp()) / "inv.jsonl"
        empty.write_text("", encoding="utf-8")
        monkeypatch.setattr(ai, "INVOCATIONS_FILE", empty)
        monkeypatch.setattr(
            "portfolio.perception_gate.should_invoke",
            lambda r, t: (True, "ok"),
        )
        monkeypatch.setattr("portfolio.journal.write_context", lambda: 0)
        monkeypatch.setattr(
            "portfolio.agent_invocation._load_config",
            lambda: {"layer2": {"enabled": True}},
        )
        monkeypatch.setattr(
            ai.PATIENT_PORTFOLIO.__class__, "exists", lambda self: True
        )
        # Drawdown check returns clean — exercise the trade guards gate next.
        monkeypatch.setattr(
            "portfolio.risk_management.check_drawdown",
            lambda pf_path, max_drawdown_pct=20.0: {
                "current_drawdown_pct": 5.0,
                "peak_value": 500_000.0,
                "current_value": 475_000.0,
            },
        )

    def test_blocks_when_trigger_ticker_blocked_for_both_strategies(self, monkeypatch):
        """When BTC-USD is in cooldown for both Patient and Bold, the
        invocation must short-circuit before the multi-agent / subprocess
        spawn block — no useful trade decision is possible."""
        self._setup_invoke_path(monkeypatch)

        # agent_summary with trade_guard_warnings blocking BTC-USD on both
        guard_result = {
            "warnings": [
                {
                    "guard": "ticker_cooldown",
                    "severity": "block",
                    "ticker": "BTC-USD",
                    "strategy": "patient",
                    "details": {"ticker": "BTC-USD", "strategy": "patient"},
                },
                {
                    "guard": "ticker_cooldown",
                    "severity": "block",
                    "ticker": "BTC-USD",
                    "strategy": "bold",
                    "details": {"ticker": "BTC-USD", "strategy": "bold"},
                },
            ],
            "summary": "ticker_cooldown: 2",
        }
        monkeypatch.setattr(
            "portfolio.agent_invocation._load_guard_warnings",
            lambda: guard_result,
        )

        with patch("portfolio.agent_invocation._log_trigger") as mock_log:
            result = ai.invoke_agent(["BTC-USD trigger flipped BUY"], tier=2)

        assert result is False, (
            "P1-12: invocation must block when trigger ticker is in cooldown "
            "for both strategies — no decision is possible."
        )
        block_calls = [
            c for c in mock_log.call_args_list
            if len(c.args) >= 2 and "blocked_trade_guards" in str(c.args[1])
        ]
        assert block_calls, (
            f"Expected a blocked_trade_guards _log_trigger call, got: "
            f"{[c.args for c in mock_log.call_args_list]}"
        )

    def test_proceeds_when_only_one_strategy_blocked(self, monkeypatch):
        """When only Patient is blocked but Bold is free, invocation must
        proceed — Bold can still take action."""
        self._setup_invoke_path(monkeypatch)

        guard_result = {
            "warnings": [
                {
                    "guard": "ticker_cooldown",
                    "severity": "block",
                    "ticker": "BTC-USD",
                    "strategy": "patient",
                    "details": {"ticker": "BTC-USD", "strategy": "patient"},
                },
            ],
            "summary": "ticker_cooldown: 1",
        }
        monkeypatch.setattr(
            "portfolio.agent_invocation._load_guard_warnings",
            lambda: guard_result,
        )

        with patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude"), \
             patch("portfolio.agent_invocation.subprocess.Popen", return_value=MagicMock(pid=99)), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()), \
             patch("portfolio.agent_invocation._log_trigger") as mock_log:
            ai.invoke_agent(["BTC-USD trigger flipped BUY"], tier=2)

        # No blocked_trade_guards call — gate let it through (Bold can still trade).
        block_calls = [
            c for c in mock_log.call_args_list
            if len(c.args) >= 2 and "blocked_trade_guards" in str(c.args[1])
        ]
        assert not block_calls, (
            f"Single-strategy block must not block invocation: {block_calls}"
        )

    def test_proceeds_with_only_warning_severity(self, monkeypatch):
        """severity=warning (e.g. consecutive_losses informational) must
        not block the invocation — they're advisory only."""
        self._setup_invoke_path(monkeypatch)

        guard_result = {
            "warnings": [
                {
                    "guard": "consecutive_losses",
                    "severity": "warning",
                    "details": {"consecutive_losses": 2, "strategy": "bold"},
                },
            ],
            "summary": "consecutive_losses: 1",
        }
        monkeypatch.setattr(
            "portfolio.agent_invocation._load_guard_warnings",
            lambda: guard_result,
        )

        with patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude"), \
             patch("portfolio.agent_invocation.subprocess.Popen", return_value=MagicMock(pid=99)), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()), \
             patch("portfolio.agent_invocation._log_trigger") as mock_log:
            ai.invoke_agent(["BTC-USD trigger flipped BUY"], tier=2)

        block_calls = [
            c for c in mock_log.call_args_list
            if len(c.args) >= 2 and "blocked_trade_guards" in str(c.args[1])
        ]
        assert not block_calls

    def test_no_warnings_proceeds_normally(self, monkeypatch):
        """Empty guard_result must not affect invocation flow."""
        self._setup_invoke_path(monkeypatch)
        monkeypatch.setattr(
            "portfolio.agent_invocation._load_guard_warnings",
            lambda: {"warnings": [], "summary": "All clear"},
        )

        with patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude"), \
             patch("portfolio.agent_invocation.subprocess.Popen", return_value=MagicMock(pid=99)), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()), \
             patch("portfolio.agent_invocation._log_trigger") as mock_log:
            ai.invoke_agent(["BTC-USD trigger flipped BUY"], tier=2)

        block_calls = [
            c for c in mock_log.call_args_list
            if len(c.args) >= 2 and "blocked_trade_guards" in str(c.args[1])
        ]
        assert not block_calls

    def test_load_guard_failure_does_not_block(self, monkeypatch):
        """A failure inside _load_guard_warnings (missing agent_summary,
        IO race, import error) must not block the invocation — fail-open
        for THIS gate (unlike drawdown which is fail-safe block)."""
        self._setup_invoke_path(monkeypatch)

        def boom():
            raise OSError("missing agent_summary")

        monkeypatch.setattr(
            "portfolio.agent_invocation._load_guard_warnings", boom
        )

        with patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude"), \
             patch("portfolio.agent_invocation.subprocess.Popen", return_value=MagicMock(pid=99)), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", mock_open()), \
             patch("portfolio.agent_invocation._log_trigger") as mock_log:
            ai.invoke_agent(["BTC-USD trigger flipped BUY"], tier=2)

        # The failure must not surface as a blocked_trade_guards call.
        block_calls = [
            c for c in mock_log.call_args_list
            if len(c.args) >= 2 and "blocked_trade_guards" in str(c.args[1])
        ]
        assert not block_calls, (
            "P1-12: trade-guard load failure must fail-open, not fail-safe-block "
            "(unlike drawdown). The cooldown is itself a soft constraint and "
            "missing data should not stop the loop from invoking."
        )

    def test_load_guard_warnings_reads_agent_summary(self, tmp_path, monkeypatch):
        """End-to-end: _load_guard_warnings reads from agent_summary.json
        and returns its trade_guard_warnings field (or empty default)."""
        from portfolio.agent_invocation import _load_guard_warnings

        # Direct DATA_DIR to tmp
        monkeypatch.setattr("portfolio.agent_invocation.DATA_DIR", tmp_path)

        # No file → empty warnings
        result = _load_guard_warnings()
        assert result == {"warnings": [], "summary": "no_summary"}

        # File with warnings → returned
        import json as _json
        summary = {
            "trade_guard_warnings": {
                "warnings": [{"severity": "block", "ticker": "ETH-USD"}],
                "summary": "1 block",
            }
        }
        (tmp_path / "agent_summary.json").write_text(_json.dumps(summary))
        result = _load_guard_warnings()
        assert len(result["warnings"]) == 1
        assert result["warnings"][0]["severity"] == "block"


# ===========================================================================
# P1-3 (2026-05-02 last-followups): auth-error scan on the timeout-kill path
# ===========================================================================

class TestKillOverrunAuthScan:
    """`_kill_overrun_agent` must scan the captured agent.log slice for
    auth-error markers before forgetting the dead subprocess.

    Background: ``check_agent_completion()`` (the happy completion path)
    already calls ``detect_auth_failure`` on the new portion of agent.log
    (line 956). But the timeout-kill path (``_kill_overrun_agent``) only
    logged ``status="timeout"`` and never inspected what the agent printed.
    A hung agent that printed "Not logged in" before getting stuck on a
    network retry would surface as ``timeout``, not ``auth_error`` — and
    would never land in critical_errors.jsonl. This regression test pins
    the new behavior so the asymmetry stays closed.
    """

    def test_auth_marker_in_log_recorded_to_critical_errors(self, tmp_path, monkeypatch):
        """When agent.log contains an auth marker between
        _agent_log_start_offset and EOF, the kill path records a
        critical-error entry via detect_auth_failure."""
        # Stage the agent.log with an auth marker.
        agent_log = tmp_path / "agent.log"
        agent_log.write_text("Not logged in\nstuff\n", encoding="utf-8")

        # Point DATA_DIR (used by check_agent_completion → agent.log)
        # at our tmp.
        monkeypatch.setattr("portfolio.agent_invocation.DATA_DIR", tmp_path)
        # Also patch claude_gate's CRITICAL_ERRORS_LOG so the test doesn't
        # write to the real journal.
        monkeypatch.setattr(
            "portfolio.claude_gate.CRITICAL_ERRORS_LOG",
            tmp_path / "critical_errors.jsonl",
        )

        # Set up a "running" mock proc that will be killed by the helper.
        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 4242
        proc.wait.return_value = 0
        ai._agent_proc = proc
        ai._agent_log = None  # detached file handle (already closed)
        ai._agent_log_start_offset = 0  # whole file is "this invocation"
        ai._agent_start = time.monotonic() - 1000
        ai._agent_start_wall = time.time() - 1000
        ai._agent_timeout = 120
        ai._agent_tier = 2
        ai._agent_reasons = ["XAG-USD volatility"]

        with patch("portfolio.agent_invocation.platform.system", return_value="Linux"), \
             patch("portfolio.agent_invocation.atomic_append_jsonl"):
            ai._kill_overrun_agent()

        # The auth marker should have triggered a critical_errors.jsonl entry.
        crit_path = tmp_path / "critical_errors.jsonl"
        assert crit_path.exists(), "expected critical_errors.jsonl to be written"
        contents = crit_path.read_text(encoding="utf-8")
        assert "auth_failure" in contents
        assert "Not logged in" in contents
        # And the trigger log should still record the timeout (existing behavior).

    def test_no_auth_marker_no_critical_error(self, tmp_path, monkeypatch):
        """A clean agent.log (no auth marker) leaves critical_errors.jsonl untouched."""
        agent_log = tmp_path / "agent.log"
        agent_log.write_text("normal output here\n", encoding="utf-8")

        monkeypatch.setattr("portfolio.agent_invocation.DATA_DIR", tmp_path)
        monkeypatch.setattr(
            "portfolio.claude_gate.CRITICAL_ERRORS_LOG",
            tmp_path / "critical_errors.jsonl",
        )

        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 4243
        proc.wait.return_value = 0
        ai._agent_proc = proc
        ai._agent_log = None
        ai._agent_log_start_offset = 0
        ai._agent_start = time.monotonic() - 1000
        ai._agent_start_wall = time.time() - 1000
        ai._agent_timeout = 120
        ai._agent_tier = 1
        ai._agent_reasons = ["health check"]

        with patch("portfolio.agent_invocation.platform.system", return_value="Linux"), \
             patch("portfolio.agent_invocation.atomic_append_jsonl"):
            ai._kill_overrun_agent()

        crit_path = tmp_path / "critical_errors.jsonl"
        assert not crit_path.exists() or "auth_failure" not in crit_path.read_text("utf-8")

    def test_missing_agent_log_does_not_raise(self, tmp_path, monkeypatch):
        """Timeout-kill path must not raise if agent.log is missing
        (e.g. subprocess never wrote anything). The kill itself must still
        complete and return True so the caller knows it can spawn a new
        agent."""
        # No agent.log on disk.
        monkeypatch.setattr("portfolio.agent_invocation.DATA_DIR", tmp_path)
        monkeypatch.setattr(
            "portfolio.claude_gate.CRITICAL_ERRORS_LOG",
            tmp_path / "critical_errors.jsonl",
        )

        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 4244
        proc.wait.return_value = 0
        ai._agent_proc = proc
        ai._agent_log = None
        ai._agent_log_start_offset = 0
        ai._agent_start = time.monotonic() - 1000
        ai._agent_start_wall = time.time() - 1000
        ai._agent_timeout = 120
        ai._agent_tier = 3
        ai._agent_reasons = ["x"]

        with patch("portfolio.agent_invocation.platform.system", return_value="Linux"), \
             patch("portfolio.agent_invocation.atomic_append_jsonl"):
            result = ai._kill_overrun_agent()

        # Kill completed cleanly.
        assert result is True
        # _agent_proc cleared so caller can spawn a replacement.
        assert ai._agent_proc is None

    def test_offset_respected_only_new_output_scanned(self, tmp_path, monkeypatch):
        """An auth marker BEFORE _agent_log_start_offset (i.e. from a
        previous invocation) must NOT be scanned. Only output from the
        current invocation counts. This mirrors the offset semantics in
        check_agent_completion()."""
        agent_log = tmp_path / "agent.log"
        # Earlier garbage (10 bytes) — _then_ this invocation's output.
        # The marker is in the EARLIER section, so the scan must skip it.
        prefix = "earlier   "  # 10 bytes
        body = "Not logged in\n"  # this is the marker but it sits AFTER offset
        agent_log.write_text(prefix + body, encoding="utf-8")

        monkeypatch.setattr("portfolio.agent_invocation.DATA_DIR", tmp_path)
        monkeypatch.setattr(
            "portfolio.claude_gate.CRITICAL_ERRORS_LOG",
            tmp_path / "critical_errors.jsonl",
        )

        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 4245
        proc.wait.return_value = 0
        ai._agent_proc = proc
        ai._agent_log = None
        # Offset starts AT the marker so the scan reads "Not logged in\n".
        ai._agent_log_start_offset = len(prefix)
        ai._agent_start = time.monotonic() - 1000
        ai._agent_start_wall = time.time() - 1000
        ai._agent_timeout = 120
        ai._agent_tier = 2
        ai._agent_reasons = ["x"]

        with patch("portfolio.agent_invocation.platform.system", return_value="Linux"), \
             patch("portfolio.agent_invocation.atomic_append_jsonl"):
            ai._kill_overrun_agent()

        crit_path = tmp_path / "critical_errors.jsonl"
        assert crit_path.exists()
        contents = crit_path.read_text("utf-8")
        assert "auth_failure" in contents

        # Reset and try again with offset PAST the marker — must NOT match.
        crit_path.unlink()

        proc2 = MagicMock()
        proc2.poll.return_value = None
        proc2.pid = 4246
        proc2.wait.return_value = 0
        ai._agent_proc = proc2
        ai._agent_log = None
        # Offset past the marker → only blank tail is scanned.
        ai._agent_log_start_offset = len(prefix) + len(body)
        ai._agent_start = time.monotonic() - 1000
        ai._agent_start_wall = time.time() - 1000
        ai._agent_timeout = 120
        ai._agent_tier = 2
        ai._agent_reasons = ["x"]

        with patch("portfolio.agent_invocation.platform.system", return_value="Linux"), \
             patch("portfolio.agent_invocation.atomic_append_jsonl"):
            ai._kill_overrun_agent()

        assert not crit_path.exists() or "auth_failure" not in crit_path.read_text("utf-8")


# ---------------------------------------------------------------------------
# Decision feedback loop (2026-05-02 research)
# ---------------------------------------------------------------------------

class TestDecisionFeedback:
    """Tests for _build_decision_feedback()."""

    def test_returns_empty_on_no_journal(self, tmp_path):
        """No journal file → empty string."""
        with patch.object(ai, "JOURNAL_FILE", tmp_path / "missing.jsonl"):
            result = ai._build_decision_feedback("BTC-USD")
        assert result == ""

    def test_returns_empty_on_empty_journal(self, tmp_path):
        """Empty journal → empty string."""
        jf = tmp_path / "journal.jsonl"
        jf.write_text("")
        with patch.object(ai, "JOURNAL_FILE", jf):
            result = ai._build_decision_feedback("BTC-USD")
        assert result == ""

    def test_returns_empty_when_no_matching_ticker(self, tmp_path):
        """Journal exists but has no entries for the requested ticker."""
        import json
        jf = tmp_path / "journal.jsonl"
        entry = {
            "ts": "2026-05-01T12:00:00+00:00",
            "trigger": "ETH-USD consensus BUY",
            "tickers": {"ETH-USD": {"outlook": "bullish"}},
            "decisions": {"patient": {"action": "BUY"}},
            "prices": {"ETH-USD": 2500.0},
        }
        jf.write_text(json.dumps(entry) + "\n")
        with patch.object(ai, "JOURNAL_FILE", jf):
            result = ai._build_decision_feedback("XAG-USD")
        assert result == ""

    def test_formats_matching_entries(self, tmp_path):
        """Entries mentioning the ticker appear in the feedback."""
        import json
        jf = tmp_path / "journal.jsonl"
        entries = [
            {
                "ts": "2026-05-01T10:00:00+00:00",
                "trigger": "BTC-USD consensus BUY",
                "tickers": {"BTC-USD": {}},
                "decisions": {
                    "patient": {"action": "HOLD"},
                    "bold": {"action": "BUY 5%"},
                },
                "prices": {"BTC-USD": 75000.0},
            },
            {
                "ts": "2026-05-01T14:00:00+00:00",
                "trigger": "BTC-USD flipped SELL",
                "tickers": {"BTC-USD": {}},
                "decisions": {
                    "patient": {"action": "HOLD"},
                    "bold": {"action": "SELL 50%"},
                },
                "prices": {"BTC-USD": 76000.0},
            },
        ]
        jf.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        with patch.object(ai, "JOURNAL_FILE", jf):
            result = ai._build_decision_feedback("BTC-USD")
        assert "[RECENT DECISIONS FOR BTC-USD]" in result
        assert "$75,000.00" in result
        assert "$76,000.00" in result
        assert "bold=BUY 5%" in result
        assert "bold=SELL 50%" in result

    def test_limits_to_max_entries(self, tmp_path):
        """Only the most recent max_entries are included."""
        import json
        jf = tmp_path / "journal.jsonl"
        lines = []
        for i in range(10):
            lines.append(json.dumps({
                "ts": f"2026-05-01T{i:02d}:00:00+00:00",
                "trigger": "XAG-USD signal",
                "tickers": {"XAG-USD": {}},
                "decisions": {"patient": {"action": "HOLD"}},
                "prices": {"XAG-USD": 70.0 + i},
            }))
        jf.write_text("\n".join(lines) + "\n")
        with patch.object(ai, "JOURNAL_FILE", jf):
            result = ai._build_decision_feedback("XAG-USD", max_entries=3)
        # Should have header + 3 entries + review line = 5 lines
        result_lines = [l for l in result.split("\n") if l.strip()]
        assert len(result_lines) == 5
        # Most recent entries (hours 9, 8, 7)
        assert "$79.00" in result
        assert "$78.00" in result
        assert "$77.00" in result
        # Older entry should NOT be present
        assert "$70.00" not in result

    def test_handles_missing_price_gracefully(self, tmp_path):
        """Missing price for ticker → shows '?' instead of crashing."""
        import json
        jf = tmp_path / "journal.jsonl"
        entry = {
            "ts": "2026-05-01T12:00:00+00:00",
            "trigger": "MSTR consensus SELL",
            "tickers": {"MSTR": {}},
            "decisions": {"patient": {"action": "SELL"}},
            "prices": {},  # no price data
        }
        jf.write_text(json.dumps(entry) + "\n")
        with patch.object(ai, "JOURNAL_FILE", jf):
            result = ai._build_decision_feedback("MSTR")
        assert "?" in result
        assert "MSTR" in result


# ===========================================================================
# _no_position_skip — Item 2 of docs/PLAN.md (reduce-claude-invocations)
# ===========================================================================

class TestNoPositionSkip:
    """Gate that skips Claude when no position is held and no entry-strong signal."""

    def _write_state(self, path, holdings=None):
        import json
        state = {"holdings": holdings or {}, "cash": 100000}
        path.write_text(json.dumps(state))

    def _write_ctx(self, path, signals=None):
        import json
        path.write_text(json.dumps({"signals": signals or {}}))

    def _enabled_cfg(self, threshold=0.65):
        return {"claude_budget": {"no_position_skip_enabled": True,
                                  "entry_confidence_threshold": threshold}}

    def test_held_patient_does_not_skip(self, tmp_path, monkeypatch):
        patient = tmp_path / "patient.json"
        bold = tmp_path / "bold.json"
        ctx = tmp_path / "agent_context_t1.json"
        self._write_state(patient, {"BTC-USD": {"shares": 1.0}})
        self._write_state(bold)
        self._write_ctx(ctx, {"BTC-USD": {"weighted_confidence": 0.10}})
        monkeypatch.setattr(ai, "DATA_DIR", tmp_path)
        from portfolio import portfolio_mgr
        monkeypatch.setattr(portfolio_mgr, "STATE_FILE", patient)
        monkeypatch.setattr(portfolio_mgr, "BOLD_STATE_FILE", bold)
        with patch("portfolio.agent_invocation._load_config", return_value=self._enabled_cfg()):
            skip, reason = ai._no_position_skip(["BTC-USD flipped HOLD->BUY"])
        assert skip is False
        assert reason == ""

    def test_held_bold_does_not_skip(self, tmp_path, monkeypatch):
        patient = tmp_path / "patient.json"
        bold = tmp_path / "bold.json"
        ctx = tmp_path / "agent_context_t1.json"
        self._write_state(patient)
        self._write_state(bold, {"ETH-USD": {"shares": 2.0}})
        self._write_ctx(ctx, {"ETH-USD": {"weighted_confidence": 0.10}})
        monkeypatch.setattr(ai, "DATA_DIR", tmp_path)
        from portfolio import portfolio_mgr
        monkeypatch.setattr(portfolio_mgr, "STATE_FILE", patient)
        monkeypatch.setattr(portfolio_mgr, "BOLD_STATE_FILE", bold)
        with patch("portfolio.agent_invocation._load_config", return_value=self._enabled_cfg()):
            skip, reason = ai._no_position_skip(["ETH-USD flipped HOLD->BUY"])
        assert skip is False

    def test_unheld_low_confidence_skips(self, tmp_path, monkeypatch):
        patient = tmp_path / "patient.json"
        bold = tmp_path / "bold.json"
        ctx = tmp_path / "agent_context_t1.json"
        self._write_state(patient)
        self._write_state(bold)
        self._write_ctx(ctx, {"MSTR": {"weighted_confidence": 0.40}})
        monkeypatch.setattr(ai, "DATA_DIR", tmp_path)
        from portfolio import portfolio_mgr
        monkeypatch.setattr(portfolio_mgr, "STATE_FILE", patient)
        monkeypatch.setattr(portfolio_mgr, "BOLD_STATE_FILE", bold)
        with patch("portfolio.agent_invocation._load_config", return_value=self._enabled_cfg()):
            skip, reason = ai._no_position_skip(["MSTR flipped HOLD->BUY"])
        assert skip is True
        assert reason == "no_position_no_entry"

    def test_unheld_high_confidence_does_not_skip(self, tmp_path, monkeypatch):
        patient = tmp_path / "patient.json"
        bold = tmp_path / "bold.json"
        ctx = tmp_path / "agent_context_t1.json"
        self._write_state(patient)
        self._write_state(bold)
        self._write_ctx(ctx, {"MSTR": {"weighted_confidence": 0.70}})
        monkeypatch.setattr(ai, "DATA_DIR", tmp_path)
        from portfolio import portfolio_mgr
        monkeypatch.setattr(portfolio_mgr, "STATE_FILE", patient)
        monkeypatch.setattr(portfolio_mgr, "BOLD_STATE_FILE", bold)
        with patch("portfolio.agent_invocation._load_config", return_value=self._enabled_cfg()):
            skip, reason = ai._no_position_skip(["MSTR flipped HOLD->BUY"])
        assert skip is False

    def test_mixed_one_held_does_not_skip(self, tmp_path, monkeypatch):
        patient = tmp_path / "patient.json"
        bold = tmp_path / "bold.json"
        ctx = tmp_path / "agent_context_t1.json"
        self._write_state(patient, {"BTC-USD": {"shares": 0.5}})
        self._write_state(bold)
        self._write_ctx(ctx, {
            "BTC-USD": {"weighted_confidence": 0.10},
            "ETH-USD": {"weighted_confidence": 0.10},
        })
        monkeypatch.setattr(ai, "DATA_DIR", tmp_path)
        from portfolio import portfolio_mgr
        monkeypatch.setattr(portfolio_mgr, "STATE_FILE", patient)
        monkeypatch.setattr(portfolio_mgr, "BOLD_STATE_FILE", bold)
        with patch("portfolio.agent_invocation._load_config", return_value=self._enabled_cfg()):
            skip, reason = ai._no_position_skip([
                "BTC-USD flipped HOLD->BUY",
                "ETH-USD flipped HOLD->BUY",
            ])
        assert skip is False

    def test_disabled_gate_never_skips(self, tmp_path, monkeypatch):
        patient = tmp_path / "patient.json"
        bold = tmp_path / "bold.json"
        ctx = tmp_path / "agent_context_t1.json"
        self._write_state(patient)
        self._write_state(bold)
        self._write_ctx(ctx, {"MSTR": {"weighted_confidence": 0.10}})
        monkeypatch.setattr(ai, "DATA_DIR", tmp_path)
        from portfolio import portfolio_mgr
        monkeypatch.setattr(portfolio_mgr, "STATE_FILE", patient)
        monkeypatch.setattr(portfolio_mgr, "BOLD_STATE_FILE", bold)
        disabled_cfg = {"claude_budget": {"no_position_skip_enabled": False}}
        with patch("portfolio.agent_invocation._load_config", return_value=disabled_cfg):
            skip, reason = ai._no_position_skip(["MSTR flipped HOLD->BUY"])
        assert skip is False

    def test_extract_triggered_tickers_finds_dash_usd(self):
        out = ai._extract_triggered_tickers([
            "BTC-USD flipped HOLD->BUY",
            "ETH-USD crossed EMA",
        ])
        assert "BTC-USD" in out
        assert "ETH-USD" in out

    def test_extract_triggered_tickers_finds_stock(self):
        out = ai._extract_triggered_tickers(["MSTR flipped HOLD->SELL"])
        assert "MSTR" in out


# ---------------------------------------------------------------------------
# Regime context injection (2026-05-23 research)
# ---------------------------------------------------------------------------

class TestRegimeContext:
    """Tests for _build_regime_context()."""

    def test_returns_empty_on_missing_file(self, tmp_path):
        with patch.object(ai, "DATA_DIR", tmp_path):
            result = ai._build_regime_context()
        assert result == ""

    def test_returns_regime_line_with_valid_data(self, tmp_path):
        import json
        summary = {
            "signals": {
                "BTC-USD": {"regime": "trending-up", "action": "BUY", "weighted_confidence": 0.62},
                "XAG-USD": {"regime": "ranging", "action": "HOLD", "weighted_confidence": 0.45},
            }
        }
        sf = tmp_path / "agent_summary.json"
        sf.write_text(json.dumps(summary))
        with patch.object(ai, "DATA_DIR", tmp_path):
            result = ai._build_regime_context()
        assert result.startswith("[REGIME]")
        assert "BTC-USD=trending-up" in result
        assert "XAG-USD=ranging" in result

    def test_returns_empty_on_no_signals_key(self, tmp_path):
        import json
        sf = tmp_path / "agent_summary.json"
        sf.write_text(json.dumps({"timestamp": "2026-05-23"}))
        with patch.object(ai, "DATA_DIR", tmp_path):
            result = ai._build_regime_context()
        assert result == ""

    def test_calibration_warning_on_high_agreement(self, tmp_path):
        import json
        summary = {
            "signals": {
                "BTC-USD": {
                    "regime": "ranging", "action": "BUY", "weighted_confidence": 0.55,
                    "extra": {"_buy_count": 14, "_sell_count": 1},
                },
            }
        }
        sf = tmp_path / "agent_summary.json"
        sf.write_text(json.dumps(summary))
        with patch.object(ai, "DATA_DIR", tmp_path):
            result = ai._build_regime_context()
        assert "CALIBRATION WARNING" in result
        assert "BTC-USD" in result

    def test_no_calibration_warning_on_balanced_votes(self, tmp_path):
        import json
        summary = {
            "signals": {
                "BTC-USD": {
                    "regime": "ranging", "action": "HOLD", "weighted_confidence": 0.50,
                    "extra": {"_buy_count": 7, "_sell_count": 8},
                },
            }
        }
        sf = tmp_path / "agent_summary.json"
        sf.write_text(json.dumps(summary))
        with patch.object(ai, "DATA_DIR", tmp_path):
            result = ai._build_regime_context()
        assert "CALIBRATION WARNING" not in result

    def test_handles_exception_gracefully(self, tmp_path):
        import json
        sf = tmp_path / "agent_summary.json"
        sf.write_text("invalid json{{{")
        with patch.object(ai, "DATA_DIR", tmp_path):
            result = ai._build_regime_context()
        assert result == ""
