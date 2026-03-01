"""Tests for Kronos stdout contamination fix.

The root cause of Kronos' 98.2% failure rate is HuggingFace's from_pretrained()
printing to stdout during model loading, which contaminates the JSON output.
The fix has two parts:
  1. kronos_infer.py redirects stdout to stderr during model work
  2. forecast.py extracts JSON from potentially contaminated stdout (defense-in-depth)

These tests verify the caller-side defense-in-depth: _extract_json_from_stdout()
and the updated _run_kronos() that uses it.
"""

import json
import time
from unittest.mock import patch, MagicMock

import pytest

from portfolio.signals.forecast import (
    _run_kronos,
    _extract_json_from_stdout,
    _kronos_circuit_open,
    reset_circuit_breakers,
)


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset circuit breakers and enable Kronos for each test."""
    import portfolio.signals.forecast as mod
    orig_kronos = mod._KRONOS_ENABLED
    mod._KRONOS_ENABLED = True
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()
    mod._KRONOS_ENABLED = orig_kronos


# --- _extract_json_from_stdout tests ---

class TestExtractJsonFromStdout:
    def test_extract_clean_json(self):
        """Clean JSON parses directly."""
        stdout = '{"method":"kronos","results":{"1h":{"direction":"up"}}}'
        result = _extract_json_from_stdout(stdout)
        assert result is not None
        assert result["method"] == "kronos"

    def test_extract_json_with_prefix(self):
        """HuggingFace print statements before JSON should be skipped."""
        stdout = 'Loading weights from local directory\n{"method":"kronos","results":{"1h":{"direction":"up"}}}'
        result = _extract_json_from_stdout(stdout)
        assert result is not None
        assert result["method"] == "kronos"

    def test_extract_no_json(self):
        """Stdout with no JSON at all returns None."""
        result = _extract_json_from_stdout("just some text output")
        assert result is None

    def test_extract_empty_string(self):
        """Empty stdout returns None."""
        result = _extract_json_from_stdout("")
        assert result is None

    def test_extract_none_input(self):
        """None input returns None."""
        result = _extract_json_from_stdout(None)
        assert result is None

    def test_extract_multi_line_warnings_then_json(self):
        """Multiple warning lines before JSON."""
        stdout = (
            "WARNING: some cuda thing\n"
            "Loading checkpoint shards: 100%\n"
            "Model loaded successfully\n"
            '{"method":"kronos","results":{"1h":{"direction":"down","pct_move":-0.5}}}'
        )
        result = _extract_json_from_stdout(stdout)
        assert result is not None
        assert result["method"] == "kronos"
        assert result["results"]["1h"]["direction"] == "down"

    def test_extract_json_with_trailing_newline(self):
        """JSON followed by trailing whitespace/newline."""
        stdout = '{"method":"kronos","results":{}}\n\n'
        result = _extract_json_from_stdout(stdout)
        assert result is not None
        assert result["method"] == "kronos"

    def test_extract_json_on_last_line(self):
        """JSON on the last line of multi-line output."""
        stdout = "line1\nline2\nline3\n" + json.dumps({"method": "kronos", "results": {"1h": {}}})
        result = _extract_json_from_stdout(stdout)
        assert result is not None
        assert result["method"] == "kronos"

    def test_extract_preserves_nested_structure(self):
        """Nested JSON structure is fully preserved."""
        nested = {
            "method": "kronos",
            "results": {
                "1h": {"direction": "up", "pct_move": 1.23, "confidence": 0.65},
                "24h": {"direction": "down", "pct_move": -0.45, "confidence": 0.3},
            },
        }
        stdout = "Some prefix\n" + json.dumps(nested)
        result = _extract_json_from_stdout(stdout)
        assert result == nested


# --- _run_kronos with contaminated stdout tests ---

class TestRunKronosContaminatedStdout:
    """Test that _run_kronos handles contaminated stdout via _extract_json_from_stdout."""

    @patch("portfolio.signals.forecast.subprocess.run")
    def test_stdout_with_prefix_text_then_json(self, mock_run):
        """Simulates the actual bug: stdout has print statements before JSON."""
        valid_json = json.dumps({
            "method": "kronos",
            "results": {"1h": {"direction": "up", "pct_move": 0.5, "confidence": 0.6}},
        })
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=f"Loading weights from local directory\n{valid_json}",
            stderr="",
        )
        result = _run_kronos([{"close": 100}] * 50)
        assert result is not None
        assert result["method"] == "kronos"
        assert result["results"]["1h"]["direction"] == "up"

    @patch("portfolio.signals.forecast.subprocess.run")
    def test_stdout_clean_json_still_works(self, mock_run):
        """Clean stdout continues to work after the fix."""
        valid_json = json.dumps({
            "method": "kronos",
            "results": {"1h": {"direction": "down", "pct_move": -0.3, "confidence": 0.4}},
        })
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=valid_json,
            stderr="",
        )
        result = _run_kronos([{"close": 100}] * 50)
        assert result is not None
        assert result["method"] == "kronos"

    @patch("portfolio.signals.forecast.subprocess.run")
    def test_stdout_garbage_no_json(self, mock_run):
        """Stdout with no JSON at all returns None and trips breaker."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="CUDA error: out of memory\nSegfault",
            stderr="",
        )
        result = _run_kronos([{"close": 100}] * 50)
        assert result is None
        assert _kronos_circuit_open()

    @patch("portfolio.signals.forecast.subprocess.run")
    def test_stdout_multi_line_warnings_then_json(self, mock_run):
        """Multiple HuggingFace/CUDA warning lines before JSON."""
        valid_json = json.dumps({
            "method": "kronos",
            "results": {"24h": {"direction": "up", "pct_move": 1.2, "confidence": 0.7}},
        })
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "Some weights of the model checkpoint were not used\n"
                "- classifier.weight\n"
                "- classifier.bias\n"
                f"{valid_json}"
            ),
            stderr="",
        )
        result = _run_kronos([{"close": 100}] * 50)
        assert result is not None
        assert result["results"]["24h"]["pct_move"] == 1.2

    @patch("portfolio.signals.forecast.subprocess.run")
    def test_error_diagnostic_logged_on_failure(self, mock_run, caplog):
        """When extraction fails, actual stdout content is logged."""
        import logging
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="just garbage output with no json whatsoever",
            stderr="",
        )
        with caplog.at_level(logging.WARNING, logger="portfolio.signals.forecast"):
            result = _run_kronos([{"close": 100}] * 50, _ticker="BTC-USD")
        assert result is None
        # Should log the actual stdout for diagnostics
        assert any("stdout" in r.message.lower() or "extract" in r.message.lower()
                    for r in caplog.records)

    @patch("portfolio.signals.forecast.subprocess.run")
    def test_circuit_breaker_trips_on_json_failure(self, mock_run):
        """Breaker trips when no JSON is extractable from stdout."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Loading model...\nDone.\n",
            stderr="",
        )
        result = _run_kronos([{"close": 100}] * 50)
        assert result is None
        assert _kronos_circuit_open()

    @patch("portfolio.signals.forecast.subprocess.run")
    def test_empty_results_still_trips_breaker(self, mock_run):
        """Valid JSON but empty results still trips breaker (existing behavior)."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"method": "none", "results": {}}',
            stderr="",
        )
        result = _run_kronos([{"close": 100}] * 50)
        assert result is None
        assert _kronos_circuit_open()
