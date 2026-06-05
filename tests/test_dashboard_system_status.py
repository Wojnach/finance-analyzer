"""Tests for dashboard.system_status — the home-page rollup aggregator.

All file inputs are written into a tmp_path data dir and ``compute()``
is called with that path so the tests never read real production files.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dashboard import system_status as ss

UTC = timezone.utc


def _now() -> datetime:
    return datetime.now(UTC)


def _ts(offset_seconds: float = 0.0) -> str:
    return (_now() + timedelta(seconds=offset_seconds)).isoformat()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r) + "\n")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


class TestHeartbeat:
    def test_fresh_heartbeat_green(self, tmp_path: Path):
        _write_json(
            tmp_path / "health_state.json",
            {"last_heartbeat": _ts(-30), "cycle_count": 5, "error_count": 0},
        )
        out = ss.compute(data_dir=tmp_path)
        assert out["heartbeat"]["age_seconds"] is not None
        assert 0 < out["heartbeat"]["age_seconds"] < 60
        assert out["heartbeat"]["cycle_count"] == 5
        assert out["overall"] == "GREEN"

    def test_stale_heartbeat_yellow(self, tmp_path: Path):
        _write_json(
            tmp_path / "health_state.json",
            {"last_heartbeat": _ts(-300), "cycle_count": 5},
        )
        out = ss.compute(data_dir=tmp_path)
        assert out["overall"] == "YELLOW"
        assert any("heartbeat" in r for r in out["reasons"])

    def test_silent_loop_red(self, tmp_path: Path):
        _write_json(
            tmp_path / "health_state.json",
            {"last_heartbeat": _ts(-3600), "cycle_count": 5},
        )
        out = ss.compute(data_dir=tmp_path)
        assert out["overall"] == "RED"
        assert any("silent" in r for r in out["reasons"])

    def test_missing_heartbeat_red(self, tmp_path: Path):
        # No health_state.json at all → load_json returns {}
        out = ss.compute(data_dir=tmp_path)
        assert out["heartbeat"]["age_seconds"] is None
        assert out["overall"] == "RED"

    def test_malformed_heartbeat_records_error(self, tmp_path: Path):
        _write_json(
            tmp_path / "health_state.json",
            {"last_heartbeat": "not-a-timestamp", "cycle_count": 1},
        )
        out = ss.compute(data_dir=tmp_path)
        assert "error" in out["heartbeat"]


# ---------------------------------------------------------------------------
# Errors / contract violations
# ---------------------------------------------------------------------------


class TestErrors:
    def test_resolved_pair_does_not_count(self, tmp_path: Path):
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        ts1 = _ts(-3600)
        _write_jsonl(
            tmp_path / "critical_errors.jsonl",
            [
                {"ts": ts1, "level": "critical", "category": "contract_violation",
                 "caller": "x", "message": "bad"},
                {"ts": _ts(-1800), "level": "info", "category": "resolution",
                 "resolves_ts": ts1, "message": "fixed"},
            ],
        )
        out = ss.compute(data_dir=tmp_path)
        assert out["errors"]["unresolved"] == 0

    def test_unresolved_critical_counts(self, tmp_path: Path):
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        _write_jsonl(
            tmp_path / "critical_errors.jsonl",
            [
                {"ts": _ts(-3600), "level": "critical", "category": "x",
                 "caller": "a", "message": "first"},
                {"ts": _ts(-1800), "level": "critical", "category": "y",
                 "caller": "b", "message": "second"},
            ],
        )
        out = ss.compute(data_dir=tmp_path)
        assert out["errors"]["unresolved"] == 2
        assert len(out["errors"]["recent"]) == 2
        # newest first
        assert out["errors"]["recent"][0]["caller"] == "b"

    def test_malformed_line_isolated(self, tmp_path: Path):
        # Manually write a junk line
        path = tmp_path / "critical_errors.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"ts": _ts(-100), "level": "critical", "category": "x",
                        "caller": "a", "message": "ok"})
            + "\nNOT-JSON-LINE\n",
            encoding="utf-8",
        )
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        out = ss.compute(data_dir=tmp_path)
        # Aggregator must not crash; valid line is counted.
        assert out["errors"]["unresolved"] == 1


class TestContractViolations:
    def test_recent_critical_only(self, tmp_path: Path):
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        _write_jsonl(
            tmp_path / "contract_violations.jsonl",
            [
                {"ts": _ts(-3600), "severity": "CRITICAL",
                 "invariant": "x", "message": "recent"},
                {"ts": _ts(-200000), "severity": "CRITICAL",
                 "invariant": "x", "message": "stale"},
                {"ts": _ts(-100), "severity": "WARNING",
                 "invariant": "y", "message": "noise"},
            ],
        )
        out = ss.compute(data_dir=tmp_path)
        assert out["contract_violations"]["unresolved"] == 1
        assert out["contract_violations"]["recent"][0]["message"] == "recent"

    def test_layer2_journal_activity_resolved_by_journal_entry(
        self, tmp_path: Path
    ):
        """If a layer2_journal.jsonl entry exists with ts >= the violation's
        details.trigger_time, the violation has been implicitly resolved."""
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        trigger_ts = _ts(-3000)
        _write_jsonl(
            tmp_path / "contract_violations.jsonl",
            [
                {
                    "ts": _ts(-2700),
                    "severity": "CRITICAL",
                    "invariant": "layer2_journal_activity",
                    "message": "Layer 2 trigger fired but no journal entry",
                    "details": {"trigger_time": trigger_ts},
                },
            ],
        )
        # Journal entry written AFTER the trigger -> resolves.
        _write_jsonl(
            tmp_path / "layer2_journal.jsonl",
            [{"ts": _ts(-2400), "trigger": "x", "decisions": {}}],
        )
        out = ss.compute(data_dir=tmp_path)
        assert out["contract_violations"]["unresolved"] == 0

    def test_layer2_violation_not_resolved_when_journal_predates_trigger(
        self, tmp_path: Path
    ):
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        trigger_ts = _ts(-1000)
        _write_jsonl(
            tmp_path / "contract_violations.jsonl",
            [
                {
                    "ts": _ts(-500),
                    "severity": "CRITICAL",
                    "invariant": "layer2_journal_activity",
                    "message": "no journal",
                    "details": {"trigger_time": trigger_ts},
                },
            ],
        )
        _write_jsonl(
            tmp_path / "layer2_journal.jsonl",
            [{"ts": _ts(-2000), "trigger": "x", "decisions": {}}],
        )
        out = ss.compute(data_dir=tmp_path)
        assert out["contract_violations"]["unresolved"] == 1

    def test_accuracy_degradation_dedup_by_alert_set(self, tmp_path: Path):
        """Identity is the sorted (scope, key) set from details.alerts —
        not the rendered message text whose drifting percentages would
        otherwise generate fresh hashes each cycle."""
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        alerts = [
            {"scope": "signal", "key": "sentiment"},
            {"scope": "signal", "key": "claude_fundamental"},
        ]
        _write_jsonl(
            tmp_path / "contract_violations.jsonl",
            [
                {"ts": _ts(-3600), "severity": "CRITICAL",
                 "invariant": "accuracy_degradation",
                 "message": "2 signal(s) dropped >15pp (33.7% -> 33.2%)",
                 "details": {"alerts": alerts}},
                {"ts": _ts(-1800), "severity": "CRITICAL",
                 "invariant": "accuracy_degradation",
                 "message": "2 signal(s) dropped >15pp (33.5% -> 32.9%)",
                 "details": {"alerts": alerts}},
            ],
        )
        out = ss.compute(data_dir=tmp_path)
        assert out["contract_violations"]["unresolved"] == 1

    def test_accuracy_degradation_distinct_alert_sets_both_surface(
        self, tmp_path: Path
    ):
        """When a new signal joins the degradation set, the identity key
        changes and the row should NOT collapse against the previous one."""
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        _write_jsonl(
            tmp_path / "contract_violations.jsonl",
            [
                {"ts": _ts(-3600), "severity": "CRITICAL",
                 "invariant": "accuracy_degradation",
                 "message": "2 signal(s) dropped >15pp...",
                 "details": {"alerts": [
                     {"scope": "signal", "key": "sentiment"},
                     {"scope": "signal", "key": "momentum_factors"},
                 ]}},
                {"ts": _ts(-1800), "severity": "CRITICAL",
                 "invariant": "accuracy_degradation",
                 "message": "3 signal(s) dropped >15pp...",
                 "details": {"alerts": [
                     {"scope": "signal", "key": "sentiment"},
                     {"scope": "signal", "key": "momentum_factors"},
                     {"scope": "signal", "key": "structure"},
                 ]}},
            ],
        )
        out = ss.compute(data_dir=tmp_path)
        assert out["contract_violations"]["unresolved"] == 2

    def test_layer2_distinct_trigger_times_both_surface(
        self, tmp_path: Path
    ):
        """Two layer2_journal_activity violations on different triggers
        whose messages happen to round to the same minute count must
        stay separate. Earlier dedup-by-message[:200] would have merged
        them."""
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        _write_jsonl(
            tmp_path / "contract_violations.jsonl",
            [
                {"ts": _ts(-3600), "severity": "CRITICAL",
                 "invariant": "layer2_journal_activity",
                 "message": "Layer 2 trigger fired 5m ago (X)",
                 "details": {"trigger_time": _ts(-3900)}},
                {"ts": _ts(-1800), "severity": "CRITICAL",
                 "invariant": "layer2_journal_activity",
                 "message": "Layer 2 trigger fired 5m ago (X)",
                 "details": {"trigger_time": _ts(-2100)}},
            ],
        )
        out = ss.compute(data_dir=tmp_path)
        # Different trigger_time => distinct identity, both surface
        # (assuming neither is resolved by a later journal entry).
        assert out["contract_violations"]["unresolved"] == 2

    def test_violation_resolved_via_critical_errors_chain_production_shape(
        self, tmp_path: Path
    ):
        """Production resolution with the *real* CE row shape.

        ``record_critical_error`` writes the payload under ``context``,
        not ``details``. The cross-stream identity match must accept
        either key — without that, the resolution path silently misses
        for every accuracy_degradation incident in production.
        (Claude review of a85a646f, P1-1.)
        """
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        crit_err_ts = _ts(-3000)
        _write_jsonl(
            tmp_path / "contract_violations.jsonl",
            [
                # CV row: ``details`` (matches loop_contract._log_violations)
                {"ts": _ts(-3500), "severity": "CRITICAL",
                 "invariant": "accuracy_degradation",
                 "message": "12 signal(s) dropped >15pp...",
                 "details": {"alerts": [
                     {"scope": "signal", "key": "sentiment"},
                 ]}},
            ],
        )
        _write_jsonl(
            tmp_path / "critical_errors.jsonl",
            [
                # CE row: ``context`` (matches claude_gate.record_critical_error)
                {"ts": crit_err_ts, "level": "critical",
                 "caller": "accuracy_degradation",
                 "category": "accuracy_degradation",
                 "message": "12 signal(s) dropped >15pp...",
                 "context": {"alerts": [
                     {"scope": "signal", "key": "sentiment"},
                 ]},
                 "resolution": None},
                {"ts": _ts(-300), "level": "info", "category": "resolution",
                 "resolves_ts": crit_err_ts, "message": "fixed"},
            ],
        )
        out = ss.compute(data_dir=tmp_path)
        assert out["contract_violations"]["unresolved"] == 0

    def test_layer2_cross_stream_match_uses_context_trigger_time(
        self, tmp_path: Path
    ):
        """layer2_journal_activity CE rows carry ``trigger_time`` under
        ``context``, not ``details``. (Claude review P1-1.)"""
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        trigger_ts = _ts(-2700)
        _write_jsonl(
            tmp_path / "contract_violations.jsonl",
            [
                {"ts": _ts(-2400), "severity": "CRITICAL",
                 "invariant": "layer2_journal_activity",
                 "message": "Layer 2 trigger fired 5m ago (X)",
                 "details": {"trigger_time": trigger_ts}},
            ],
        )
        _write_jsonl(
            tmp_path / "critical_errors.jsonl",
            [
                {"ts": _ts(-2300), "level": "critical",
                 "caller": "layer2_journal_activity",
                 "category": "contract_violation",  # inline path's actual shape
                 "message": "Layer 2 trigger fired 5m ago (X)",
                 "context": {"trigger_time": trigger_ts},
                 "resolution": None},
            ],
        )
        out = ss.compute(data_dir=tmp_path)
        assert out["errors"]["unresolved"] == 1
        assert out["contract_violations"]["unresolved"] == 0

    def test_escalated_prefix_does_not_break_cross_stream_dedup(
        self, tmp_path: Path
    ):
        """ViolationTracker promotes a warning by prepending
        ``ESCALATED (Nx consecutive): `` to the message. The source
        strips this before hashing; the dashboard must too, otherwise
        escalated CV rows never match their pre-escalation CE row.
        (Claude review of a85a646f, P1-2.)"""
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        _write_jsonl(
            tmp_path / "contract_violations.jsonl",
            [
                {"ts": _ts(-1800), "severity": "CRITICAL",
                 "invariant": "accuracy_degradation",
                 "message": "ESCALATED (3x consecutive): 12 signal(s) "
                            "dropped >15pp...",
                 "details": {"alerts": [
                     {"scope": "signal", "key": "sentiment"},
                 ]}},
            ],
        )
        _write_jsonl(
            tmp_path / "critical_errors.jsonl",
            [
                {"ts": _ts(-3000), "level": "critical",
                 "caller": "accuracy_degradation",
                 "category": "accuracy_degradation",
                 "message": "12 signal(s) dropped >15pp...",  # un-prefixed
                 "context": {"alerts": [
                     {"scope": "signal", "key": "sentiment"},
                 ]},
                 "resolution": None},
            ],
        )
        out = ss.compute(data_dir=tmp_path)
        assert out["errors"]["unresolved"] == 1
        # Without the prefix strip, the CV row would surface as a duplicate.
        assert out["contract_violations"]["unresolved"] == 0

    def test_violation_dedupes_against_unresolved_critical_error(
        self, tmp_path: Path
    ):
        """Cross-stream noise reduction: when both streams represent the
        same incident, surface it only once (under ERR, since that side
        is already resolution-aware). The CV side stays quiet."""
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        _write_jsonl(
            tmp_path / "contract_violations.jsonl",
            [
                {"ts": _ts(-3500), "severity": "CRITICAL",
                 "invariant": "accuracy_degradation",
                 "message": "12 signal(s) dropped >15pp...",
                 "details": {"alerts": [
                     {"scope": "signal", "key": "sentiment"},
                 ]}},
            ],
        )
        _write_jsonl(
            tmp_path / "critical_errors.jsonl",
            [
                {"ts": _ts(-3000), "level": "critical",
                 "caller": "accuracy_degradation",
                 "category": "accuracy_degradation",
                 "message": "12 signal(s) dropped >15pp...",
                 "context": {"alerts": [
                     {"scope": "signal", "key": "sentiment"},
                 ]},
                 "resolution": None},
            ],
        )
        out = ss.compute(data_dir=tmp_path)
        # Errors panel: 1 unresolved.
        assert out["errors"]["unresolved"] == 1
        # Violations panel: 0 (already counted under errors).
        assert out["contract_violations"]["unresolved"] == 0

    def test_two_distinct_unresolved_violations_both_surface(
        self, tmp_path: Path
    ):
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        _write_jsonl(
            tmp_path / "contract_violations.jsonl",
            [
                {"ts": _ts(-3600), "severity": "CRITICAL",
                 "invariant": "min_success_rate", "message": "0% rate"},
                {"ts": _ts(-1800), "severity": "CRITICAL",
                 "invariant": "session_alive", "message": "dead session"},
            ],
        )
        out = ss.compute(data_dir=tmp_path)
        assert out["contract_violations"]["unresolved"] == 2


# ---------------------------------------------------------------------------
# LLM inference
# ---------------------------------------------------------------------------


class TestLLMInference:
    def test_chronos_kronos_from_report(self, tmp_path: Path):
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        _write_json(
            tmp_path / "local_llm_report_latest.json",
            {"health": {
                "chronos": {"ok": 99, "fail": 1, "total": 100, "success_rate": 0.99},
                "kronos": {"ok": 80, "fail": 20, "total": 100, "success_rate": 0.80},
            }},
        )
        out = ss.compute(data_dir=tmp_path)
        keys = [m["key"] for m in out["llm_inference"]["models"]]
        assert "chronos" in keys
        assert "kronos" in keys
        chronos = next(m for m in out["llm_inference"]["models"] if m["key"] == "chronos")
        assert chronos["success_pct"] == 99.0
        # Weighted average: (99*100 + 80*100) / 200 = 89.5
        assert out["llm_inference"]["overall_pct"] == 89.5

    def test_signal_health_llms(self, tmp_path: Path):
        _write_json(
            tmp_path / "health_state.json",
            {
                "last_heartbeat": _ts(-10),
                "signal_health": {
                    "claude_fundamental": {"total_calls": 1000, "total_failures": 50},
                    "forecast": {"total_calls": 500, "total_failures": 0},
                },
            },
        )
        out = ss.compute(data_dir=tmp_path)
        models = out["llm_inference"]["models"]
        cf = next(m for m in models if m["key"] == "claude_fundamental")
        assert cf["success_pct"] == 95.0
        fc = next(m for m in models if m["key"] == "forecast")
        assert fc["success_pct"] == 100.0

    def test_llm_red_below_yellow_threshold(self, tmp_path: Path):
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        _write_json(
            tmp_path / "local_llm_report_latest.json",
            {"health": {"chronos": {"ok": 50, "fail": 50}}},
        )
        out = ss.compute(data_dir=tmp_path)
        assert out["overall"] == "RED"


# ---------------------------------------------------------------------------
# Layer 2
# ---------------------------------------------------------------------------


class TestLayer2:
    def test_24h_success_rate(self, tmp_path: Path):
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        _write_jsonl(
            tmp_path / "claude_invocations.jsonl",
            [
                {"timestamp": _ts(-3600), "caller": "x", "status": "invoked",
                 "model": "sonnet", "duration_seconds": 30},
                {"timestamp": _ts(-1800), "caller": "x", "status": "timeout",
                 "model": "sonnet", "duration_seconds": 184},
                {"timestamp": _ts(-600), "caller": "x", "status": "invoked",
                 "model": "sonnet", "duration_seconds": 22},
                {"timestamp": _ts(-90000), "caller": "x", "status": "invoked",
                 "model": "sonnet", "duration_seconds": 12},  # > 24h, ignored
            ],
        )
        out = ss.compute(data_dir=tmp_path)
        l2 = out["layer2"]
        assert l2["triggers_24h"] == 3
        assert l2["success_24h"] == 2
        assert l2["success_pct"] == round(100 * 2 / 3, 1)
        # Latest within window is the -600s entry
        assert l2["latest"]["status"] == "invoked"

    def test_layer2_low_success_yellow(self, tmp_path: Path):
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        _write_jsonl(
            tmp_path / "claude_invocations.jsonl",
            [
                {"timestamp": _ts(-1000 - i * 60), "caller": "x",
                 "status": "invoked" if i < 7 else "timeout"}
                for i in range(10)
            ],
        )
        out = ss.compute(data_dir=tmp_path)
        # 7/10 = 70% — between YELLOW (60) and GREEN (85) → YELLOW
        assert out["layer2"]["success_pct"] == 70.0
        assert out["overall"] == "YELLOW"

    def test_layer2_below_min_triggers_skips_gate(self, tmp_path: Path):
        # Only 1 trigger over 24h, even at 0% it shouldn't drag the
        # overall to RED — too small a sample.
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        _write_jsonl(
            tmp_path / "claude_invocations.jsonl",
            [{"timestamp": _ts(-100), "caller": "x", "status": "timeout"}],
        )
        out = ss.compute(data_dir=tmp_path)
        assert out["layer2"]["triggers_24h"] == 1
        assert out["overall"] == "GREEN"


# ---------------------------------------------------------------------------
# Signal aggregate
# ---------------------------------------------------------------------------


class TestSignalAggregate:
    def test_per_ticker_counts(self, tmp_path: Path):
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        _write_jsonl(
            tmp_path / "signal_log.jsonl",
            [{
                "ts": _ts(-30),
                "tickers": {
                    "BTC-USD": {
                        "consensus": "HOLD",
                        "buy_count": 3,
                        "sell_count": 1,
                        "total_voters": 4,
                        "regime": "ranging",
                        "signals": {f"sig{i}": "BUY" for i in range(3)} |
                                   {"sigS": "SELL"} |
                                   {f"sigH{i}": "HOLD" for i in range(20)},
                    },
                },
            }],
        )
        out = ss.compute(data_dir=tmp_path)
        tickers = out["signal_aggregate"]["tickers"]
        assert len(tickers) == 1
        t = tickers[0]
        assert t["ticker"] == "BTC-USD"
        assert t["buy"] == 3
        assert t["sell"] == 1
        assert t["hold"] == 20
        assert t["abstain"] == 20
        assert t["total"] == 24
        assert t["confidence"] == round(4 / 24, 3)


# ---------------------------------------------------------------------------
# Color rollup
# ---------------------------------------------------------------------------


class TestColor:
    def test_all_green(self, tmp_path: Path, monkeypatch):
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-30)})
        # Force the Claude gate ACTIVE so this "nominal" assertion is
        # deterministic regardless of the repo's real layer2.enabled state
        # (a frozen gate legitimately appends a reason — see TestColorFrozenReason).
        import portfolio.claude_gate as cg

        monkeypatch.setattr(cg, "CLAUDE_ENABLED", True, raising=False)
        monkeypatch.setattr(cg, "_load_config_layer2_enabled", lambda: True, raising=False)
        _write_metals_loop(tmp_path, "True")
        out = ss.compute(data_dir=tmp_path)
        assert out["overall"] == "GREEN"
        assert out["reasons"] == ["all systems nominal"]

    def test_many_errors_red(self, tmp_path: Path):
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        _write_jsonl(
            tmp_path / "critical_errors.jsonl",
            [
                {"ts": _ts(-i * 60), "level": "critical", "category": "x",
                 "caller": "a", "message": f"err{i}"}
                for i in range(10)
            ],
        )
        out = ss.compute(data_dir=tmp_path)
        assert out["overall"] == "RED"
        assert any("unresolved errors" in r for r in out["reasons"])

    def test_payload_shape(self, tmp_path: Path):
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        out = ss.compute(data_dir=tmp_path)
        for k in (
            "ts", "overall", "reasons", "heartbeat", "errors",
            "contract_violations", "llm_inference", "layer2",
            "signal_aggregate", "pnl_footer",
        ):
            assert k in out, f"missing key: {k}"


# ---------------------------------------------------------------------------
# P&L footer (deprioritised but still present)
# ---------------------------------------------------------------------------


class TestPnLFooter:
    def test_reads_portfolio_states(self, tmp_path: Path):
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        _write_json(
            tmp_path / "portfolio_state.json",
            {"portfolio_value": 495_100.0, "starting_capital": 500_000.0},
        )
        _write_json(
            tmp_path / "portfolio_state_bold.json",
            {"portfolio_value": 488_300.0, "starting_capital": 500_000.0},
        )
        out = ss.compute(data_dir=tmp_path)
        assert out["pnl_footer"]["patient_value_sek"] == 495_100.0
        assert out["pnl_footer"]["bold_value_sek"] == 488_300.0


# ---------------------------------------------------------------------------
# Codex P1 / P2 regressions (2026-05-04)
# ---------------------------------------------------------------------------


class TestCodex20260504Regressions:
    """Lock in fixes for the codex review findings on this PR."""

    def test_unresolved_critical_far_back_still_counts(self, tmp_path: Path):
        """P1: 500 newer info/resolution rows after an older critical
        used to drop the older one out of the unresolved count. Must
        survive the full file scan now.
        """
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        rows = []
        # 4 unresolved criticals near the start of the file
        for i in range(4):
            rows.append({
                "ts": _ts(-3600 - i),
                "level": "critical",
                "category": "x",
                "caller": "old",
                "message": f"older critical {i}",
            })
        # Then 600 newer info rows that would push older entries out
        # of any 500-line tail.
        for i in range(600):
            rows.append({
                "ts": _ts(-100 - i / 100),
                "level": "info",
                "category": "noise",
                "caller": "n/a",
                "message": "fluff",
            })
        _write_jsonl(tmp_path / "critical_errors.jsonl", rows)
        out = ss.compute(data_dir=tmp_path)
        assert out["errors"]["unresolved"] == 4

    def test_layer2_more_than_2000_in_24h(self, tmp_path: Path):
        """P2: with 3000 invocations in the last 24h the previous tail
        cap of 2000 would have undercounted. Adaptive growth must bring
        all of them in.
        """
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        # Spread 3000 invocations evenly across the last 23 hours
        rows = []
        for i in range(3000):
            offset = -82_800 * (i / 2999)  # 0 to 23h ago, evenly spaced
            rows.append({
                "timestamp": _ts(offset),
                "caller": "x",
                "status": "invoked",
                "model": "sonnet",
            })
        _write_jsonl(tmp_path / "claude_invocations.jsonl", rows)
        out = ss.compute(data_dir=tmp_path)
        assert out["layer2"]["triggers_24h"] == 3000

    def test_signal_log_non_dict_does_not_500(self, tmp_path: Path):
        """P2: an unexpected ``[]`` payload in signal_log.jsonl must
        surface as an in-band error, not raise.
        """
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        path = tmp_path / "signal_log.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([]) + "\n", encoding="utf-8")
        out = ss.compute(data_dir=tmp_path)
        sa = out["signal_aggregate"]
        assert sa["tickers"] == []
        assert "error" in sa  # surfaced, not raised

    def test_llm_report_garbage_value_skipped(self, tmp_path: Path):
        """P2: a string instead of an int in the local_llm_report no
        longer raises; the bad model is skipped.
        """
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        _write_json(
            tmp_path / "local_llm_report_latest.json",
            {"health": {
                "chronos": {"ok": "oops", "fail": 0},
                "kronos": {"ok": 100, "fail": 5},
            }},
        )
        out = ss.compute(data_dir=tmp_path)
        keys = [m["key"] for m in out["llm_inference"]["models"]]
        assert "chronos" not in keys  # skipped
        assert "kronos" in keys       # still works


class TestDashboardSourceIdentityEquivalence:
    """Pin the equivalence between the dashboard's identity-key path and
    the source's ``_hash_violation_identity``. Without this, future
    per-invariant overrides added to the source can silently desync the
    dashboard the same way the missing ``context``/``ESCALATED`` strip
    bugs did. (Claude review of a85a646f, P2-1.)"""

    @staticmethod
    def _entry_from_violation(v):
        # Mirror what loop_contract._log_violations writes.
        return {
            "ts": "2026-05-05T00:00:00+00:00",
            "severity": v.severity,
            "invariant": v.invariant,
            "message": v.message,
            "details": v.details,
        }

    @pytest.mark.parametrize(
        "invariant,message,details",
        [
            ("min_success_rate", "0% rate", {}),
            ("session_alive", "dead session", {"alive": False}),
            (
                "accuracy_degradation",
                "12 signal(s) dropped >15pp 33.7%->33.2%",
                {"alerts": [
                    {"scope": "signal", "key": "sentiment"},
                    {"scope": "signal", "key": "structure"},
                ]},
            ),
            (
                "accuracy_degradation",
                "different drift text 99.9%->10%",
                {"alerts": [
                    {"scope": "signal", "key": "sentiment"},
                    {"scope": "signal", "key": "structure"},
                ]},
            ),
            (
                "layer2_journal_activity",
                "Layer 2 trigger fired 5m ago (BTC)",
                {"trigger_time": "2026-05-05T00:00:00+00:00"},
            ),
        ],
    )
    def test_identity_payload_matches_source(
        self, invariant, message, details
    ):
        from portfolio.loop_contract import (
            Violation,
            _hash_violation_identity,
            violation_identity_payload,
        )
        import hashlib

        v = Violation(
            invariant=invariant, severity="CRITICAL",
            message=message, details=details,
        )
        # Source side: tracker may have prepended the prefix; use
        # the raw constructor message here (no prefix).
        source_hash = _hash_violation_identity(v)
        # Dashboard side: build the entry shape, run through the
        # mirror, then hash by the same SHA-1 algorithm.
        entry = self._entry_from_violation(v)
        dash_payload = ss._identity_key_for_dict(entry)
        dash_hash = hashlib.sha256(dash_payload.encode("utf-8")).hexdigest()
        assert dash_hash == source_hash, (
            f"Dashboard identity drifted from source for {invariant!r}: "
            f"dashboard payload={dash_payload!r} → {dash_hash}, "
            f"source → {source_hash}. Both must call "
            f"violation_identity_payload."
        )
        # Sanity: the shared helper directly returns the same payload
        # for the source-side input.
        assert dash_payload == violation_identity_payload(
            invariant, message, details,
        )

    def test_escalated_message_strips_to_pre_promotion_payload(self):
        """Two dicts representing the same incident — one before
        ViolationTracker promotion, one after — must produce the same
        identity payload."""
        pre = {
            "ts": "2026-05-05T00:00:00+00:00",
            "severity": "WARNING",
            "invariant": "accuracy_degradation",
            "message": "2 signal(s) dropped >15pp...",
            "details": {"alerts": [
                {"scope": "signal", "key": "sentiment"},
            ]},
        }
        post = {**pre, "severity": "CRITICAL",
                "message": "ESCALATED (3x consecutive): "
                           + pre["message"]}
        assert ss._identity_key_for_dict(pre) == ss._identity_key_for_dict(post)

    def test_critical_errors_context_payload_matches_source(self):
        """A critical_errors row written by record_critical_error uses
        ``context`` instead of ``details``. Identity must still match
        the matching contract_violations row's identity."""
        from portfolio.loop_contract import (
            Violation,
            _hash_violation_identity,
        )
        import hashlib

        details = {"alerts": [
            {"scope": "signal", "key": "sentiment"},
        ]}
        message = "12 signal(s) dropped >15pp..."
        v = Violation(
            invariant="accuracy_degradation", severity="CRITICAL",
            message=message, details=details,
        )
        source_hash = _hash_violation_identity(v)
        ce_entry = {
            "ts": "2026-05-05T00:00:00+00:00",
            "level": "critical",
            "caller": "accuracy_degradation",
            "category": "accuracy_degradation",
            "message": message,
            "context": details,  # NOT details
            "resolution": None,
        }
        ce_payload = ss._identity_key_for_dict(ce_entry)
        ce_hash = hashlib.sha256(ce_payload.encode("utf-8")).hexdigest()
        assert ce_hash == source_hash



# ---------------------------------------------------------------------------
# Claude gate (Layer 2 ACTIVE vs FROZEN indicator) — added 2026-06-06
# ---------------------------------------------------------------------------


def _write_metals_loop(dd: Path, value: str | None) -> None:
    """Write a stub metals_loop.py with (or without) the CLAUDE_ENABLED const."""
    dd.mkdir(parents=True, exist_ok=True)
    body = "# stub\nCHECK_INTERVAL = 60\n"
    if value is not None:
        body = f"# stub\nCLAUDE_ENABLED = {value}   # comment\nCHECK_INTERVAL = 60\n"
    (dd / "metals_loop.py").write_text(body, encoding="utf-8")


class TestParseMetalsClaudeEnabled:
    def test_true(self, tmp_path):
        _write_metals_loop(tmp_path, "True")
        assert ss._parse_metals_claude_enabled(tmp_path / "metals_loop.py") is True

    def test_false(self, tmp_path):
        _write_metals_loop(tmp_path, "False")
        assert ss._parse_metals_claude_enabled(tmp_path / "metals_loop.py") is False

    def test_missing_file(self, tmp_path):
        assert ss._parse_metals_claude_enabled(tmp_path / "nope.py") is None

    def test_missing_constant(self, tmp_path):
        _write_metals_loop(tmp_path, None)
        assert ss._parse_metals_claude_enabled(tmp_path / "metals_loop.py") is None

    def test_ignores_indented_occurrence(self, tmp_path):
        # An indented (in-function) assignment must NOT match — only the
        # top-level constant counts.
        (tmp_path / "metals_loop.py").write_text(
            "def f():\n    CLAUDE_ENABLED = True\n    return 1\n"
            "CLAUDE_ENABLED = False\n",
            encoding="utf-8",
        )
        assert ss._parse_metals_claude_enabled(tmp_path / "metals_loop.py") is False


class TestClaudeGate:
    def _patch(self, monkeypatch, gate_enabled, config_enabled):
        import portfolio.claude_gate as cg

        monkeypatch.setattr(cg, "CLAUDE_ENABLED", gate_enabled, raising=False)
        monkeypatch.setattr(
            cg, "_load_config_layer2_enabled", lambda: config_enabled, raising=False
        )

    def test_frozen_when_all_off(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, gate_enabled=False, config_enabled=False)
        _write_metals_loop(tmp_path, "False")
        out = ss._claude_gate(tmp_path)
        assert out["enabled"] is False
        assert out["label"] == "FROZEN"
        assert out["config_layer2_enabled"] is False
        assert out["claude_gate_enabled"] is False
        assert out["metals_claude_enabled"] is False

    def test_frozen_when_any_one_off(self, tmp_path, monkeypatch):
        # config still on, but the master gate is off -> still FROZEN.
        self._patch(monkeypatch, gate_enabled=False, config_enabled=True)
        _write_metals_loop(tmp_path, "True")
        out = ss._claude_gate(tmp_path)
        assert out["enabled"] is False
        assert out["label"] == "FROZEN"

    def test_active_when_all_on(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, gate_enabled=True, config_enabled=True)
        _write_metals_loop(tmp_path, "True")
        out = ss._claude_gate(tmp_path)
        assert out["enabled"] is True
        assert out["label"] == "ACTIVE"

    def test_compute_attaches_gate(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, gate_enabled=False, config_enabled=False)
        _write_metals_loop(tmp_path, "False")
        payload = ss.compute(tmp_path)
        gate = payload["layer2"]["gate"]
        assert gate["label"] == "FROZEN"


class TestColorFrozenReason:
    def _healthy(self, gate_enabled):
        return {
            "heartbeat": {"age_seconds": 10},
            "errors": {"unresolved": 0},
            "contract_violations": {"unresolved": 0},
            "llm_inference": {},
            "layer2": {"gate": {"enabled": gate_enabled}},
        }

    def test_frozen_adds_reason_without_red(self):
        severity, reasons = ss._color(self._healthy(False))
        # Intentional freeze must NOT escalate the hero to RED/YELLOW.
        assert severity == "GREEN"
        assert any("frozen" in r.lower() for r in reasons)

    def test_active_no_frozen_reason(self):
        severity, reasons = ss._color(self._healthy(True))
        assert severity == "GREEN"
        assert not any("frozen" in r.lower() for r in reasons)
