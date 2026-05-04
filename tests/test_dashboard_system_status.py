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
    def test_all_green(self, tmp_path: Path):
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-30)})
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

    def test_heartbeat_payload_includes_expected_interval(self, tmp_path: Path):
        """Hero footer comparator: expected_heartbeat_seconds is sourced
        from portfolio.health._HEARTBEAT_KEEPALIVE_INTERVAL_S so the
        dashboard never drifts from the loop's truth.
        """
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        out = ss.compute(data_dir=tmp_path)
        exp = out["heartbeat"]["expected_heartbeat_seconds"]
        assert isinstance(exp, int)
        assert exp > 0

    def test_errors_recent_includes_message_and_category(self, tmp_path: Path):
        """Health-view rows need ts/category/message keys to render."""
        _write_json(tmp_path / "health_state.json", {"last_heartbeat": _ts(-10)})
        _write_jsonl(
            tmp_path / "critical_errors.jsonl",
            [{"ts": _ts(-300), "level": "critical", "category": "auth_outage",
              "caller": "layer2", "message": "claude -p exited 0 with Not logged in"}],
        )
        out = ss.compute(data_dir=tmp_path)
        recent = out["errors"]["recent"]
        assert len(recent) >= 1
        first = recent[0]
        for key in ("ts", "category", "message"):
            assert key in first, f"missing key: {key}"

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

