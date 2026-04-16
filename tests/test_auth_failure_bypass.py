"""Tests for BUG-200/BUG-201 (2026-04-16).

bigbet.py, iskbets.py, and analyze.py invoke `claude -p` directly with
subprocess.run() -- bypassing claude_gate's detect_auth_failure wrapper.
When the OAuth session expires, claude exits 0 while printing "Not logged
in" to stdout. Without the detector, those modules:

- bigbet: parsed output as valid (probability=None) but never escalated
  the auth failure to critical_errors.jsonl, so the startup-check chain
  never surfaced the issue.
- iskbets: parsed output and DEFAULTED to approved=True -- a real safety
  gap, not just a detection gap.

These tests lock in the fix: auth failure must be detected, recorded to
critical_errors.jsonl, and downgraded to the safe default.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from portfolio import claude_gate


_FAKE_SIGNALS = {
    "BTC-USD": {
        "indicators": {"rsi": 18.0, "macd_hist": -5.0, "price_vs_bb": "below_lower", "atr_pct": 3.2},
        "extra": {"_buy_count": 4, "_sell_count": 1, "_total_applicable": 11, "fear_greed": 8, "volume_ratio": 3.0},
    }
}
_FAKE_TF_DATA = {"BTC-USD": [(lbl, {"action": "SELL"}) for lbl in ("Now", "12h", "2d")]}
_FAKE_PRICES = {"BTC-USD": 65000.0}
_FAKE_CFG = {"telegram": {"token": "t", "chat_id": "c"}}
_FAKE_CONDITIONS = ["RSI 18 (oversold)", "F&G: 8 (Extreme Fear)"]


def _auth_failure_run(output: str = "Not logged in -- Please run /login", stderr: str = ""):
    """Return a MagicMock mimicking an auth-failed subprocess.run result.

    Real auth failure pattern: exit 0 (looks successful) + 'Not logged in' on
    stdout. This is the exact shape we need to detect.
    """
    return MagicMock(returncode=0, stdout=output, stderr=stderr)


# ---------------------------------------------------------------------------
# bigbet.invoke_layer2_eval
# ---------------------------------------------------------------------------


@patch("portfolio.bigbet.subprocess.run")
def test_bigbet_auth_failure_records_critical_and_returns_none(
    mock_run, tmp_path, monkeypatch
):
    """Auth failure in bigbet path -> safe default (None) + critical_errors entry."""
    from portfolio import bigbet

    mock_run.return_value = _auth_failure_run()
    monkeypatch.setattr(bigbet, "EVAL_LOG_FILE", tmp_path / "bigbet_log.jsonl")
    crit_path = tmp_path / "crit.jsonl"
    monkeypatch.setattr(claude_gate, "CRITICAL_ERRORS_LOG", crit_path)

    prob, reason = bigbet.invoke_layer2_eval(
        "BTC-USD", "BULL", _FAKE_CONDITIONS, _FAKE_SIGNALS, _FAKE_TF_DATA, _FAKE_PRICES, _FAKE_CFG,
    )

    assert prob is None
    assert reason == ""

    assert crit_path.exists(), "critical_errors.jsonl must be written on auth failure"
    entries = [json.loads(line) for line in crit_path.read_text().splitlines() if line.strip()]
    assert len(entries) == 1
    assert entries[0]["category"] == "auth_failure"
    assert entries[0]["caller"] == "bigbet_layer2"
    assert entries[0]["context"]["ticker"] == "BTC-USD"
    assert entries[0]["context"]["direction"] == "BULL"


@patch("portfolio.bigbet.subprocess.run")
def test_bigbet_normal_output_does_not_trigger_auth_detector(
    mock_run, tmp_path, monkeypatch
):
    """Healthy claude output must not trip the auth detector."""
    from portfolio import bigbet

    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="PROBABILITY: 7/10\nREASONING: Clean capitulation setup.",
        stderr="",
    )
    monkeypatch.setattr(bigbet, "EVAL_LOG_FILE", tmp_path / "bigbet_log.jsonl")
    crit_path = tmp_path / "crit.jsonl"
    monkeypatch.setattr(claude_gate, "CRITICAL_ERRORS_LOG", crit_path)

    prob, reason = bigbet.invoke_layer2_eval(
        "BTC-USD", "BULL", _FAKE_CONDITIONS, _FAKE_SIGNALS, _FAKE_TF_DATA, _FAKE_PRICES, _FAKE_CFG,
    )

    assert prob == 7
    assert "capitulation" in reason
    assert not crit_path.exists()


# ---------------------------------------------------------------------------
# iskbets.invoke_layer2_gate
# ---------------------------------------------------------------------------


@patch("portfolio.iskbets.subprocess.run")
def test_iskbets_auth_failure_overrides_default_approve(mock_run, tmp_path, monkeypatch):
    """Auth failure in iskbets gate -> MUST override default-approve to False.

    This is a real safety gap: _parse_gate_response defaults to True when
    it can't find a DECISION line, so a 'Not logged in' output would have
    been interpreted as APPROVED for a warrant trade without this fix.
    """
    from portfolio import iskbets

    mock_run.return_value = _auth_failure_run()
    monkeypatch.setattr(iskbets, "GATE_LOG_FILE", tmp_path / "gate_log.jsonl")
    crit_path = tmp_path / "crit.jsonl"
    monkeypatch.setattr(claude_gate, "CRITICAL_ERRORS_LOG", crit_path)

    approved, reasoning = iskbets.invoke_layer2_gate(
        "BTC-USD", 66000.0, ["RSI oversold"], {}, {}, 1500.0, {}, {},
    )

    assert approved is False, "auth failure must override default-approve"
    assert "auth" in reasoning.lower()

    entries = [json.loads(line) for line in crit_path.read_text().splitlines() if line.strip()]
    assert len(entries) == 1
    assert entries[0]["category"] == "auth_failure"
    assert entries[0]["caller"] == "iskbets_l2_gate"
    assert entries[0]["context"]["ticker"] == "BTC-USD"


@patch("portfolio.iskbets.subprocess.run")
def test_iskbets_healthy_output_still_default_approves_on_parse_fail(
    mock_run, tmp_path, monkeypatch
):
    """Garbage output (not auth failure) keeps the existing default-approve
    behavior. We're only overriding the auth-failure case, not all parse
    failures.
    """
    from portfolio import iskbets

    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="I'm not sure what to do here.",
        stderr="",
    )
    monkeypatch.setattr(iskbets, "GATE_LOG_FILE", tmp_path / "gate_log.jsonl")
    crit_path = tmp_path / "crit.jsonl"
    monkeypatch.setattr(claude_gate, "CRITICAL_ERRORS_LOG", crit_path)

    approved, reasoning = iskbets.invoke_layer2_gate(
        "BTC-USD", 66000.0, ["RSI oversold"], {}, {}, 1500.0, {}, {},
    )

    assert approved is True
    assert reasoning == ""
    assert not crit_path.exists()


# ---------------------------------------------------------------------------
# analyze.main (CLI wrapper, manual-invoke path)
# ---------------------------------------------------------------------------


@patch("portfolio.analyze.subprocess.run")
def test_analyze_auth_failure_records_and_exits_cleanly(
    mock_run, tmp_path, monkeypatch, capsys
):
    """analyze.py is a manual CLI tool -- still needs to record auth failure
    so future Claude sessions see the issue via check_critical_errors.py.
    """
    from portfolio import analyze

    mock_run.return_value = _auth_failure_run()

    crit_path = tmp_path / "crit.jsonl"
    monkeypatch.setattr(claude_gate, "CRITICAL_ERRORS_LOG", crit_path)

    summary_path = tmp_path / "agent_summary.json"
    summary = {
        "signals": {
            "BTC-USD": {
                "price_usd": 65000.0,
                "rsi": 50.0,
                "macd_hist": 0.0,
                "bb_position": "middle",
                "atr_pct": 2.0,
                "regime": "ranging",
                "weighted_confidence": 0.5,
                "confluence_score": 0.5,
                "extra": {"_buy_count": 0, "_sell_count": 0, "_total_applicable": 21, "_votes": {}},
                "action": "HOLD",
                "confidence": 0.5,
            }
        }
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(analyze, "AGENT_SUMMARY_FILE", summary_path)

    log_path = tmp_path / "analysis_log.jsonl"
    monkeypatch.setattr(analyze, "ANALYSIS_LOG_FILE", log_path)

    analyze.run_analysis("BTC-USD")

    entries = [json.loads(line) for line in crit_path.read_text().splitlines() if line.strip()]
    assert len(entries) == 1
    assert entries[0]["category"] == "auth_failure"
    assert entries[0]["caller"] == "analyze_cli"

    out = capsys.readouterr().out
    assert "auth" in out.lower() or "login" in out.lower()
