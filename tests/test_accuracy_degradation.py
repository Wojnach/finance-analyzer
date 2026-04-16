"""Unit tests for portfolio.accuracy_degradation.

All file paths are monkeypatched to tmp_path so the suite stays xdist-safe.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import portfolio.accuracy_degradation as deg
import portfolio.accuracy_stats as acc_mod


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

def _isolate_state(monkeypatch, tmp_path):
    """Point all state files at tmp_path so each test starts clean."""
    monkeypatch.setattr(deg, "ALERT_STATE_FILE", tmp_path / "alert_state.json")
    monkeypatch.setattr(deg, "SNAPSHOT_STATE_FILE", tmp_path / "snap_state.json")
    monkeypatch.setattr(acc_mod, "ACCURACY_SNAPSHOTS_FILE",
                        tmp_path / "snapshots.jsonl")


def _stub_econ_safe(monkeypatch):
    """Default econ helpers return no events so blackout never trips."""
    monkeypatch.setattr(
        "portfolio.econ_dates.events_within_hours",
        lambda hours: [],
    )
    monkeypatch.setattr(
        "portfolio.econ_dates.recent_high_impact_events",
        lambda hours, impact_filter=("high",): [],
    )


def _make_snapshot(*, ts: datetime, signals_recent=None,
                   per_ticker_recent=None, forecast_recent=None,
                   consensus_recent=None) -> dict:
    snap = {"ts": ts.isoformat(), "signals": {}}
    if signals_recent is not None:
        snap["signals_recent"] = signals_recent
    if per_ticker_recent is not None:
        snap["per_ticker_recent"] = per_ticker_recent
    if forecast_recent is not None:
        snap["forecast_recent"] = forecast_recent
    if consensus_recent is not None:
        snap["consensus_recent"] = consensus_recent
    return snap


def _stub_current(monkeypatch, *, signals=None, per_ticker=None,
                  forecast=None, consensus=None):
    if signals is not None:
        monkeypatch.setattr(
            "portfolio.accuracy_stats.signal_accuracy_recent",
            lambda h="1d", days=7: signals,
        )
    if per_ticker is not None:
        monkeypatch.setattr(deg, "_per_ticker_recent",
                            lambda horizon, days: per_ticker)
    if forecast is not None:
        monkeypatch.setattr(
            "portfolio.forecast_accuracy.cached_forecast_accuracy",
            lambda horizon="24h", days=7, use_raw_sub_signals=True: forecast,
        )
    if consensus is not None:
        def _stub_consensus(horizon="1d", entries=None, days=None):
            return consensus
        monkeypatch.setattr(
            "portfolio.accuracy_stats.consensus_accuracy",
            _stub_consensus,
        )


def _write_baseline(monkeypatch, tmp_path, snapshot):
    """Write a single baseline snapshot to the snapshots file."""
    from portfolio.file_utils import atomic_append_jsonl
    snap_path = tmp_path / "snapshots.jsonl"
    atomic_append_jsonl(snap_path, snapshot)
    # Ensure deg loader sees the file
    monkeypatch.setattr(acc_mod, "ACCURACY_SNAPSHOTS_FILE", snap_path)


# ---------------------------------------------------------------------------
# 1) Snapshot writer
# ---------------------------------------------------------------------------

class TestSaveFullAccuracySnapshot:

    def test_writes_all_four_scopes_with_lifetime_and_recent(self, monkeypatch, tmp_path):
        _isolate_state(monkeypatch, tmp_path)

        # Lifetime sources used by save_accuracy_snapshot internals
        lifetime_sigs = {
            "rsi": {"accuracy": 0.55, "total": 1240, "correct": 0,
                    "pct": 55.0, "correct_buy": 0, "total_buy": 0,
                    "buy_accuracy": 0.0, "correct_sell": 0,
                    "total_sell": 0, "sell_accuracy": 0.0},
        }
        monkeypatch.setattr(acc_mod, "signal_accuracy", lambda h="1d": lifetime_sigs)

        # Recent-window source
        monkeypatch.setattr(
            acc_mod, "signal_accuracy_recent",
            lambda h, days=7: {"rsi": {"accuracy": 0.49, "total": 280}},
        )

        # Per-ticker lifetime + recent
        monkeypatch.setattr(
            acc_mod, "accuracy_by_ticker_signal_cached",
            lambda h: {"BTC-USD": {"rsi": {"accuracy": 0.58, "total": 210}}},
        )
        monkeypatch.setattr(
            deg, "_per_ticker_recent",
            lambda horizon, days: {"BTC-USD": {"rsi": {"accuracy": 0.51,
                                                       "total": 56}}},
        )

        # Forecast (Chronos/Kronos)
        monkeypatch.setattr(
            "portfolio.forecast_accuracy.cached_forecast_accuracy",
            lambda horizon="24h", days=7, use_raw_sub_signals=True: {
                "chronos_24h": {"accuracy": 0.51, "total": 420, "correct": 214},
                "kronos_24h": {"accuracy": 0.49, "total": 380, "correct": 186},
            },
        )

        # Consensus
        def stub_consensus(horizon="1d", entries=None, days=None):
            base = {"accuracy": 0.56, "total": 8800, "correct": 4928}
            if days is not None:
                return {"accuracy": 0.52, "total": 220, "correct": 114}
            return base
        monkeypatch.setattr(acc_mod, "consensus_accuracy", stub_consensus)

        snap = deg.save_full_accuracy_snapshot()

        # Lifetime block from save_accuracy_snapshot()
        assert "rsi" in snap["signals"]
        # All four scope blocks present (recent-window where needed)
        assert "signals_recent" in snap
        assert "per_ticker" in snap
        assert "per_ticker_recent" in snap
        assert "forecast_recent" in snap
        assert "consensus" in snap
        assert "consensus_recent" in snap

        assert snap["signals_recent"]["rsi"]["accuracy"] == 0.49
        assert snap["per_ticker"]["BTC-USD"]["rsi"]["accuracy"] == 0.58
        assert snap["per_ticker_recent"]["BTC-USD"]["rsi"]["accuracy"] == 0.51
        assert snap["forecast_recent"]["chronos_24h"]["accuracy"] == 0.51
        assert snap["consensus_recent"]["accuracy"] == 0.52

    def test_back_compat_reads_single_block_old_snapshots(self, monkeypatch, tmp_path):
        _isolate_state(monkeypatch, tmp_path)
        # Old snapshot has only the legacy `signals` block.
        old = _make_snapshot(ts=datetime.now(UTC) - timedelta(days=10))
        old["signals"] = {"rsi": {"accuracy": 0.5, "total": 999}}
        _write_baseline(monkeypatch, tmp_path, old)

        snapshots = deg._load_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0]["signals"]["rsi"]["accuracy"] == 0.5
        # Absent keys are absent — not crashing on access
        assert snapshots[0].get("signals_recent") is None


# ---------------------------------------------------------------------------
# 2) Comparison source — recent-window, not lifetime (Codex P1#1)
# ---------------------------------------------------------------------------

class TestComparisonSource:

    def test_check_uses_recent_window_not_lifetime(self, monkeypatch, tmp_path):
        _isolate_state(monkeypatch, tmp_path)
        _stub_econ_safe(monkeypatch)

        baseline = _make_snapshot(
            ts=datetime.now(UTC) - timedelta(days=7, hours=1),
            signals_recent={"rsi": {"accuracy": 0.62, "total": 200}},
        )
        _write_baseline(monkeypatch, tmp_path, baseline)

        # Lifetime would barely move, so test that lifetime would NOT trigger.
        # We stub signal_accuracy (lifetime) to flat 60% and signal_accuracy_recent
        # (recent) to a clear collapse. The detector must use the latter.
        monkeypatch.setattr(
            acc_mod, "signal_accuracy",
            lambda h="1d": {"rsi": {"accuracy": 0.60, "total": 50000,
                                    "correct": 30000}},
        )
        _stub_current(monkeypatch,
                      signals={"rsi": {"accuracy": 0.42, "total": 280}})

        violations = deg.check_degradation()
        assert len(violations) == 1
        assert "rsi" in violations[0].message


# ---------------------------------------------------------------------------
# 3) Threshold gates
# ---------------------------------------------------------------------------

class TestThresholdGates:

    def _basic_baseline(self, monkeypatch, tmp_path):
        baseline = _make_snapshot(
            ts=datetime.now(UTC) - timedelta(days=7, hours=1),
            signals_recent={"rsi": {"accuracy": 0.62, "total": 200}},
        )
        _write_baseline(monkeypatch, tmp_path, baseline)

    def test_below_threshold_no_alert(self, monkeypatch, tmp_path):
        _isolate_state(monkeypatch, tmp_path)
        _stub_econ_safe(monkeypatch)
        self._basic_baseline(monkeypatch, tmp_path)
        _stub_current(
            monkeypatch,
            signals={"rsi": {"accuracy": 0.55, "total": 280}},  # 7pp drop
        )
        violations = deg.check_degradation()
        assert violations == []

    def test_above_threshold_emits_warning(self, monkeypatch, tmp_path):
        _isolate_state(monkeypatch, tmp_path)
        _stub_econ_safe(monkeypatch)
        self._basic_baseline(monkeypatch, tmp_path)
        _stub_current(
            monkeypatch,
            signals={"rsi": {"accuracy": 0.42, "total": 280}},  # 20pp drop, <50%
        )
        violations = deg.check_degradation()
        assert len(violations) == 1
        assert violations[0].severity == deg.SEVERITY_WARNING

    def test_drop_but_above_floor_no_alert(self, monkeypatch, tmp_path):
        """75% -> 55% is a 20pp drop but still above the 50% floor."""
        _isolate_state(monkeypatch, tmp_path)
        _stub_econ_safe(monkeypatch)
        baseline = _make_snapshot(
            ts=datetime.now(UTC) - timedelta(days=7, hours=1),
            signals_recent={"rsi": {"accuracy": 0.75, "total": 200}},
        )
        _write_baseline(monkeypatch, tmp_path, baseline)
        _stub_current(
            monkeypatch,
            signals={"rsi": {"accuracy": 0.55, "total": 280}},
        )
        violations = deg.check_degradation()
        assert violations == []


# ---------------------------------------------------------------------------
# 4) Severity classification
# ---------------------------------------------------------------------------

class TestSeverity:

    def test_three_signals_escalates_to_critical(self, monkeypatch, tmp_path):
        _isolate_state(monkeypatch, tmp_path)
        _stub_econ_safe(monkeypatch)
        baseline_signals = {
            f"sig_{i}": {"accuracy": 0.62, "total": 200} for i in range(3)
        }
        baseline = _make_snapshot(
            ts=datetime.now(UTC) - timedelta(days=7, hours=1),
            signals_recent=baseline_signals,
        )
        _write_baseline(monkeypatch, tmp_path, baseline)

        current_signals = {
            f"sig_{i}": {"accuracy": 0.42, "total": 280} for i in range(3)
        }
        _stub_current(monkeypatch, signals=current_signals)

        violations = deg.check_degradation()
        assert len(violations) == 1
        assert violations[0].severity == deg.SEVERITY_CRITICAL

    def test_consensus_drop_emits_critical(self, monkeypatch, tmp_path):
        _isolate_state(monkeypatch, tmp_path)
        _stub_econ_safe(monkeypatch)
        baseline = _make_snapshot(
            ts=datetime.now(UTC) - timedelta(days=7, hours=1),
            signals_recent={},
            consensus_recent={"accuracy": 0.62, "total": 5000},
        )
        _write_baseline(monkeypatch, tmp_path, baseline)
        _stub_current(
            monkeypatch,
            signals={},
            consensus={"accuracy": 0.42, "total": 5000, "correct": 2100},
        )
        violations = deg.check_degradation()
        assert len(violations) == 1
        assert violations[0].severity == deg.SEVERITY_CRITICAL


# ---------------------------------------------------------------------------
# 5) Anti-noise gates
# ---------------------------------------------------------------------------

class TestAntiNoise:

    def test_min_samples_gate_skips_low_n(self, monkeypatch, tmp_path):
        _isolate_state(monkeypatch, tmp_path)
        _stub_econ_safe(monkeypatch)
        baseline = _make_snapshot(
            ts=datetime.now(UTC) - timedelta(days=7, hours=1),
            signals_recent={"rsi": {"accuracy": 0.62, "total": 50}},  # < 100
        )
        _write_baseline(monkeypatch, tmp_path, baseline)
        _stub_current(
            monkeypatch,
            signals={"rsi": {"accuracy": 0.42, "total": 120}},
        )
        violations = deg.check_degradation()
        assert violations == []

    def test_snapshot_age_under_6d_gate_returns_empty(self, monkeypatch, tmp_path):
        _isolate_state(monkeypatch, tmp_path)
        _stub_econ_safe(monkeypatch)
        baseline = _make_snapshot(
            ts=datetime.now(UTC) - timedelta(days=4),  # too young
            signals_recent={"rsi": {"accuracy": 0.62, "total": 200}},
        )
        _write_baseline(monkeypatch, tmp_path, baseline)
        _stub_current(
            monkeypatch,
            signals={"rsi": {"accuracy": 0.42, "total": 280}},
        )
        violations = deg.check_degradation()
        assert violations == []

    def test_post_event_fomc_blackout_skips_check(self, monkeypatch, tmp_path):
        _isolate_state(monkeypatch, tmp_path)
        # Backward window non-empty -> blackout
        monkeypatch.setattr(
            "portfolio.econ_dates.events_within_hours",
            lambda hours: [],
        )
        monkeypatch.setattr(
            "portfolio.econ_dates.recent_high_impact_events",
            lambda hours, impact_filter=("high",): [
                {"date": datetime.now(UTC).date(), "type": "FOMC",
                 "impact": "high", "hours_since": 12.0},
            ],
        )

        baseline = _make_snapshot(
            ts=datetime.now(UTC) - timedelta(days=7, hours=1),
            signals_recent={"rsi": {"accuracy": 0.62, "total": 200}},
        )
        _write_baseline(monkeypatch, tmp_path, baseline)
        _stub_current(
            monkeypatch,
            signals={"rsi": {"accuracy": 0.42, "total": 280}},
        )

        violations = deg.check_degradation()
        assert violations == []  # blackout suppresses everything

    def test_pre_event_fomc_blackout_skips_check(self, monkeypatch, tmp_path):
        _isolate_state(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "portfolio.econ_dates.events_within_hours",
            lambda hours: [
                {"date": datetime.now(UTC).date(), "type": "CPI",
                 "impact": "high", "hours_until": 18.0},
            ],
        )
        monkeypatch.setattr(
            "portfolio.econ_dates.recent_high_impact_events",
            lambda hours, impact_filter=("high",): [],
        )

        baseline = _make_snapshot(
            ts=datetime.now(UTC) - timedelta(days=7, hours=1),
            signals_recent={"rsi": {"accuracy": 0.62, "total": 200}},
        )
        _write_baseline(monkeypatch, tmp_path, baseline)
        _stub_current(
            monkeypatch,
            signals={"rsi": {"accuracy": 0.42, "total": 280}},
        )
        violations = deg.check_degradation()
        assert violations == []


# ---------------------------------------------------------------------------
# 6) Cooldown filter (Telegram path)
# ---------------------------------------------------------------------------

class TestCooldown:

    def test_cooldown_blocks_repeat_emission(self, monkeypatch, tmp_path):
        _isolate_state(monkeypatch, tmp_path)
        alerts = [{"key": "rsi", "scope": "signal",
                   "old_accuracy_pct": 62.0, "new_accuracy_pct": 42.0,
                   "drop_pp": 20.0, "old_samples": 200, "new_samples": 280}]

        first = deg.filter_alerts_by_cooldown(alerts, now_ts=1_000_000.0)
        second = deg.filter_alerts_by_cooldown(alerts, now_ts=1_000_500.0)
        assert len(first) == 1
        assert second == []

    def test_cooldown_expires_at_25h_re_alerts(self, monkeypatch, tmp_path):
        _isolate_state(monkeypatch, tmp_path)
        alerts = [{"key": "rsi", "scope": "signal",
                   "old_accuracy_pct": 62.0, "new_accuracy_pct": 42.0,
                   "drop_pp": 20.0, "old_samples": 200, "new_samples": 280}]

        deg.filter_alerts_by_cooldown(alerts, now_ts=1_000_000.0)
        twenty_five_h = 1_000_000.0 + 25 * 3600
        replay = deg.filter_alerts_by_cooldown(alerts, now_ts=twenty_five_h)
        assert len(replay) == 1


# ---------------------------------------------------------------------------
# 7) Throttle replays cached violations (Codex P1#2)
# ---------------------------------------------------------------------------

class TestThrottleReplay:

    def test_throttle_replays_cached_violations_does_not_return_empty(
            self, monkeypatch, tmp_path):
        _isolate_state(monkeypatch, tmp_path)
        _stub_econ_safe(monkeypatch)

        baseline = _make_snapshot(
            ts=datetime.now(UTC) - timedelta(days=7, hours=1),
            signals_recent={"rsi": {"accuracy": 0.62, "total": 200}},
        )
        _write_baseline(monkeypatch, tmp_path, baseline)
        _stub_current(
            monkeypatch,
            signals={"rsi": {"accuracy": 0.42, "total": 280}},
        )

        # First call: full check, populates cache
        first = deg.check_degradation()
        assert len(first) == 1

        # Second call within throttle: must replay, NOT return [].
        # Codex P1#2: returning [] would clear ViolationTracker counters.
        second = deg.check_degradation()
        assert len(second) == 1
        assert second[0].invariant == deg.DEGRADATION_INVARIANT
        assert second[0].severity == first[0].severity
        assert second[0].message == first[0].message

    def test_full_check_reruns_after_throttle_window(self, monkeypatch, tmp_path):
        _isolate_state(monkeypatch, tmp_path)
        _stub_econ_safe(monkeypatch)

        baseline = _make_snapshot(
            ts=datetime.now(UTC) - timedelta(days=7, hours=1),
            signals_recent={"rsi": {"accuracy": 0.62, "total": 200}},
        )
        _write_baseline(monkeypatch, tmp_path, baseline)

        calls = {"n": 0}

        def stub_recent(h, days=7):
            calls["n"] += 1
            return {"rsi": {"accuracy": 0.42, "total": 280}}

        monkeypatch.setattr(acc_mod, "signal_accuracy_recent", stub_recent)

        deg.check_degradation()
        deg.check_degradation()
        # Full re-check requested via near-zero throttle
        deg.check_degradation(throttle_seconds=0.0)
        assert calls["n"] == 2  # first call, then forced re-check


# ---------------------------------------------------------------------------
# 8) Per-ticker / forecast scope keys
# ---------------------------------------------------------------------------

class TestScopeKeys:

    def test_per_ticker_alert_format_uses_ticker_signal_key(self, monkeypatch, tmp_path):
        _isolate_state(monkeypatch, tmp_path)
        _stub_econ_safe(monkeypatch)
        baseline = _make_snapshot(
            ts=datetime.now(UTC) - timedelta(days=7, hours=1),
            per_ticker_recent={"BTC-USD": {"rsi": {"accuracy": 0.62, "total": 200}}},
        )
        _write_baseline(monkeypatch, tmp_path, baseline)
        _stub_current(
            monkeypatch,
            per_ticker={"BTC-USD": {"rsi": {"accuracy": 0.42, "total": 280}}},
        )
        violations = deg.check_degradation()
        assert len(violations) == 1
        assert "BTC-USD::rsi" in violations[0].details["alerts"][0]["key"]

    def test_forecast_alert_format_uses_forecast_prefix_key(self, monkeypatch, tmp_path):
        _isolate_state(monkeypatch, tmp_path)
        _stub_econ_safe(monkeypatch)
        baseline = _make_snapshot(
            ts=datetime.now(UTC) - timedelta(days=7, hours=1),
            forecast_recent={"chronos_24h": {"accuracy": 0.62, "total": 200}},
        )
        _write_baseline(monkeypatch, tmp_path, baseline)
        _stub_current(
            monkeypatch,
            forecast={"chronos_24h": {"accuracy": 0.42, "total": 280, "correct": 117}},
        )
        violations = deg.check_degradation()
        assert len(violations) == 1
        keys = [a["key"] for a in violations[0].details["alerts"]]
        assert "forecast::chronos_24h" in keys


# ---------------------------------------------------------------------------
# 9) Daily summary builder
# ---------------------------------------------------------------------------

class TestDailySummary:

    def test_summary_contains_consensus_top_drops_top_gains_and_split_llm_line(self):
        latest = {
            "ts": "2026-04-16T06:00:00+00:00",
            "consensus_recent": {"accuracy": 0.56, "total": 880},
            "forecast_recent": {
                "chronos_24h": {"accuracy": 0.51, "total": 420},
                "kronos_24h": {"accuracy": 0.49, "total": 380},
            },
            "signals_recent": {
                "rsi": {"accuracy": 0.44, "total": 1240},
                "macd": {"accuracy": 0.41, "total": 800},
                "obv": {"accuracy": 0.61, "total": 980},
                "ministral": {"accuracy": 0.53, "total": 200},
                "qwen3": {"accuracy": 0.47, "total": 200},
            },
        }
        baseline = {
            "ts": (datetime.now(UTC) - timedelta(days=7)).isoformat(),
            "consensus_recent": {"accuracy": 0.58, "total": 800},
            "signals_recent": {
                "rsi": {"accuracy": 0.62, "total": 1200},
                "macd": {"accuracy": 0.60, "total": 750},
                "obv": {"accuracy": 0.48, "total": 950},
            },
        }

        body = deg.build_daily_summary(
            latest=latest, baseline=baseline,
            now=datetime(2026, 4, 16, 6, 0, tzinfo=UTC),
        )

        # Header + consensus
        assert "*ACCURACY DAILY*" in body
        assert "Consensus: 56% recent7d" in body
        assert "(Δ -2.0pp vs prev 7d)" in body
        # Forecast vs LLM split — Codex P2#4
        assert "Forecast:  chronos 51% · kronos 49%" in body
        assert "LLM:       ministral 53% · qwen3 47%" in body
        # Top drops + gains
        assert "*Degraded" in body
        assert "rsi" in body
        assert "macd" in body
        assert "*Improved" in body
        assert "obv" in body
        # Footer
        assert "Snapshot age" in body
        assert "5 signals tracked" in body
