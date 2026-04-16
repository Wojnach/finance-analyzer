"""Tests for the BUG-ECHO fix to claude_gate.detect_auth_failure.

Background: the original detector simply substring-matched "Not logged in"
anywhere in the agent's combined stdout. Layer 2 agents that surface the
critical_errors.jsonl protocol from CLAUDE.md include the literal string
"Not logged in" inside their conversational output (quoting earlier
errors). That triggered a NEW false-positive auth_failure entry, which
the next agent then surfaced again, in an infinite loop.

The fix narrows the match: marker must be at the start of a non-quoted
line within the first ~16 lines of CLI output. Agent chat content
appears later AND is always wrapped in quotes/backticks/parens when it
mentions the marker.
"""

from __future__ import annotations

import portfolio.claude_gate as cg


class TestRealAuthFailureStillDetected:
    """Lock in current behavior: real CLI errors must keep tripping."""

    def test_bare_marker_at_top_detected(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cg, "CRITICAL_ERRORS_LOG", tmp_path / "ce.jsonl")
        output = "Not logged in\nPlease run /login\n"
        assert cg.detect_auth_failure(output, caller="test") is True

    def test_marker_with_dash_continuation_detected(self, monkeypatch, tmp_path):
        """Real CLI output: 'Not logged in - Please run /login' on one line."""
        monkeypatch.setattr(cg, "CRITICAL_ERRORS_LOG", tmp_path / "ce.jsonl")
        output = "Not logged in - Please run /login\n"
        assert cg.detect_auth_failure(output, caller="test") is True

    def test_invalid_api_key_at_top_detected(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cg, "CRITICAL_ERRORS_LOG", tmp_path / "ce.jsonl")
        output = "Invalid API key\n"
        assert cg.detect_auth_failure(output, caller="test") is True

    def test_marker_after_a_few_blank_lines_still_detected(self, monkeypatch, tmp_path):
        """CLI may print an empty line or two of buffer flush before the error."""
        monkeypatch.setattr(cg, "CRITICAL_ERRORS_LOG", tmp_path / "ce.jsonl")
        output = "\n\nNot logged in\n"
        assert cg.detect_auth_failure(output, caller="test") is True


class TestEchoedMarkerRejected:
    """The bug fix: agent chat content quoting the marker must NOT trip."""

    def test_marker_in_single_quotes_rejected(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cg, "CRITICAL_ERRORS_LOG", tmp_path / "ce.jsonl")
        output = (
            "I'm looking at the critical errors journal:\n"
            "claude CLI subprocess printed 'Not logged in' - OAuth session\n"
        )
        assert cg.detect_auth_failure(output, caller="test") is False

    def test_marker_in_double_quotes_rejected(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cg, "CRITICAL_ERRORS_LOG", tmp_path / "ce.jsonl")
        output = (
            'The CLI prints "Not logged in" to stdout when OAuth fails.\n'
        )
        assert cg.detect_auth_failure(output, caller="test") is False

    def test_marker_in_backticks_rejected(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cg, "CRITICAL_ERRORS_LOG", tmp_path / "ce.jsonl")
        output = "The marker is `Not logged in` per docstring.\n"
        assert cg.detect_auth_failure(output, caller="test") is False

    def test_marker_in_blockquote_rejected(self, monkeypatch, tmp_path):
        """Markdown blockquote (>) should be treated as quoted content."""
        monkeypatch.setattr(cg, "CRITICAL_ERRORS_LOG", tmp_path / "ce.jsonl")
        output = "> Not logged in - Please run /login\n"
        assert cg.detect_auth_failure(output, caller="test") is False

    def test_marker_in_parenthetical_rejected(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cg, "CRITICAL_ERRORS_LOG", tmp_path / "ce.jsonl")
        output = "the error (Not logged in) appeared earlier today\n"
        assert cg.detect_auth_failure(output, caller="test") is False

    def test_marker_indented_rejected(self, monkeypatch, tmp_path):
        """Leading whitespace = quoted code-block content, not CLI output."""
        monkeypatch.setattr(cg, "CRITICAL_ERRORS_LOG", tmp_path / "ce.jsonl")
        output = "    Not logged in - Please run /login\n"
        assert cg.detect_auth_failure(output, caller="test") is False

    def test_marker_past_first_16_lines_rejected(self, monkeypatch, tmp_path):
        """CLI auth errors print at the very top, never deep in agent chat."""
        monkeypatch.setattr(cg, "CRITICAL_ERRORS_LOG", tmp_path / "ce.jsonl")
        prelude = "\n".join([f"some agent line {i}" for i in range(20)])
        output = prelude + "\nNot logged in\n"
        assert cg.detect_auth_failure(output, caller="test") is False

    def test_real_session_echo_pattern_rejected(self, monkeypatch, tmp_path):
        """The exact pattern from data/agent.log lines 105519-105547 today."""
        monkeypatch.setattr(cg, "CRITICAL_ERRORS_LOG", tmp_path / "ce.jsonl")
        output = (
            "I'll check the unresolved critical errors first.\n"
            "\n"
            "**Startup check found 1 unresolved critical error:**\n"
            "\n"
            "```\n"
            "[2026-04-16T13:45:45+00:00] auth_failure caller=layer2_t1\n"
            "claude CLI subprocess printed 'Not logged in' - OAuth session not being read.\n"
            "```\n"
            "\n"
            "How would you like to proceed?\n"
        )
        assert cg.detect_auth_failure(output, caller="test") is False

    def test_marker_inside_fenced_code_block_rejected(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cg, "CRITICAL_ERRORS_LOG", tmp_path / "ce.jsonl")
        output = (
            "Quoting the journal entry verbatim:\n"
            "```\n"
            "Not logged in\n"
            "```\n"
            "End of quote.\n"
        )
        assert cg.detect_auth_failure(output, caller="test") is False


class TestStdoutStderrSeparation:
    """Codex P2 follow-up: detect_auth_failure() now sees each stream
    independently from invoke_claude / invoke_claude_text, so a marker
    on stderr can't be hidden by stdout content."""

    def test_marker_on_stderr_after_busy_stdout_detected(self, monkeypatch, tmp_path):
        """Repro of Codex's exact scenario: 20 lines of stdout, then
        stderr has the marker. With the old concat, the marker would be
        pushed past the 16-line limit. Scanned independently, stderr
        starts at line 1 and the marker is detected."""
        monkeypatch.setattr(cg, "CRITICAL_ERRORS_LOG", tmp_path / "ce.jsonl")
        stderr_only = "Not logged in\n"
        # Simulating what invoke_claude now does (separate stream call)
        assert cg.detect_auth_failure(stderr_only, caller="t") is True

    def test_marker_at_top_of_stdout_detected(self, monkeypatch, tmp_path):
        """The other Codex case: marker on stdout line 1, stderr empty."""
        monkeypatch.setattr(cg, "CRITICAL_ERRORS_LOG", tmp_path / "ce.jsonl")
        stdout = "Not logged in\n" + "\n".join(f"line {i}" for i in range(20))
        assert cg.detect_auth_failure(stdout, caller="t") is True

    def test_marker_concatenated_into_last_stdout_line_does_not_count(
        self, monkeypatch, tmp_path
    ):
        """If a caller still concatenates stdout+stderr without a newline,
        the marker glued onto a stdout line MUST be ignored (start-of-line
        check). This locks in safety for any third caller that hasn't
        been updated."""
        monkeypatch.setattr(cg, "CRITICAL_ERRORS_LOG", tmp_path / "ce.jsonl")
        merged_no_newline = "some stdoutNot logged in"
        assert cg.detect_auth_failure(merged_no_newline, caller="t") is False


class TestEdgeCases:

    def test_empty_output_returns_false(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cg, "CRITICAL_ERRORS_LOG", tmp_path / "ce.jsonl")
        assert cg.detect_auth_failure("", caller="test") is False

    def test_real_failure_writes_to_critical_errors(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cg, "CRITICAL_ERRORS_LOG", tmp_path / "ce.jsonl")
        cg.detect_auth_failure("Not logged in\n", caller="layer2_t1")
        # File should now contain the entry
        assert (tmp_path / "ce.jsonl").exists()
        content = (tmp_path / "ce.jsonl").read_text()
        assert "auth_failure" in content
        assert "layer2_t1" in content

    def test_echoed_marker_does_NOT_write_to_critical_errors(self, monkeypatch, tmp_path):
        """The whole point: echoes must not journal."""
        monkeypatch.setattr(cg, "CRITICAL_ERRORS_LOG", tmp_path / "ce.jsonl")
        cg.detect_auth_failure(
            "the agent quoted 'Not logged in' from yesterday's journal\n",
            caller="layer2_t1",
        )
        assert not (tmp_path / "ce.jsonl").exists()
