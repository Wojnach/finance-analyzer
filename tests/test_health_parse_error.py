"""Tests for check_agent_silence() handling corrupt timestamps in health.py."""

from unittest.mock import patch

from portfolio.health import check_agent_silence


class TestCheckAgentSilenceCorruptTimestamp:
    """check_agent_silence must return silent=True on corrupt timestamps,
    not raise ValueError/TypeError."""

    @patch("portfolio.health.load_health")
    @patch("portfolio.health.last_jsonl_entry", return_value=None)
    def test_non_iso_string_returns_silent(self, mock_last, mock_health):
        mock_health.return_value = {"last_invocation_ts": "not-a-timestamp"}
        result = check_agent_silence()
        assert result["silent"] is True
        assert result["age_seconds"] == float("inf")

    @patch("portfolio.health.load_health")
    @patch("portfolio.health.last_jsonl_entry", return_value=None)
    def test_none_timestamp_returns_silent(self, mock_last, mock_health):
        mock_health.return_value = {"last_invocation_ts": None}
        result = check_agent_silence()
        assert result["silent"] is True

    @patch("portfolio.health.load_health")
    @patch("portfolio.health.last_jsonl_entry", return_value=None)
    def test_integer_timestamp_returns_silent(self, mock_last, mock_health):
        """Integer (wrong type) should be caught, not crash."""
        mock_health.return_value = {"last_invocation_ts": 123}
        result = check_agent_silence()
        assert result["silent"] is True
        assert result["age_seconds"] == float("inf")

    @patch("portfolio.health.load_health")
    @patch("portfolio.health.last_jsonl_entry", return_value=None)
    def test_missing_key_returns_silent(self, mock_last, mock_health):
        """No last_invocation_ts at all should fall through to silent."""
        mock_health.return_value = {}
        result = check_agent_silence()
        assert result["silent"] is True
