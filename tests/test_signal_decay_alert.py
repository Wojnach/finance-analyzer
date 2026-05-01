"""Tests for portfolio.signal_decay_alert module."""

import json

from portfolio.signal_decay_alert import (
    check_signal_decay,
    format_decay_report,
    log_decay_alerts,
    run_decay_check,
)


def _make_cache(alltime_3h=None, recent_3h=None, alltime_1d=None, recent_1d=None):
    """Build a minimal accuracy_cache dict."""
    cache = {}
    if alltime_3h is not None:
        cache["3h"] = alltime_3h
    if recent_3h is not None:
        cache["3h_recent"] = recent_3h
    if alltime_1d is not None:
        cache["1d"] = alltime_1d
    if recent_1d is not None:
        cache["1d_recent"] = recent_1d
    return cache


def _write_cache(tmp_path, cache):
    path = tmp_path / "accuracy_cache.json"
    path.write_text(json.dumps(cache))
    return str(path)


class TestCheckSignalDecay:
    """Tests for check_signal_decay()."""

    def test_no_file_returns_empty(self, tmp_path):
        alerts = check_signal_decay(str(tmp_path / "nonexistent.json"))
        assert alerts == []

    def test_invalid_json_returns_empty(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json{{{")
        alerts = check_signal_decay(str(bad))
        assert alerts == []

    def test_no_decay_no_alerts(self, tmp_path):
        """Signal with same alltime and recent accuracy => no alert."""
        cache = _make_cache(
            alltime_3h={"rsi": {"accuracy": 0.65, "total": 200}},
            recent_3h={"rsi": {"accuracy": 0.63, "total": 60}},
        )
        path = _write_cache(tmp_path, cache)
        alerts = check_signal_decay(path)
        assert alerts == []

    def test_small_drop_no_alert(self, tmp_path):
        """Drop of 5pp (below 10pp threshold) => no alert."""
        cache = _make_cache(
            alltime_3h={"rsi": {"accuracy": 0.65, "total": 200}},
            recent_3h={"rsi": {"accuracy": 0.60, "total": 60}},
        )
        path = _write_cache(tmp_path, cache)
        alerts = check_signal_decay(path)
        assert alerts == []

    def test_warning_level_decay(self, tmp_path):
        """Drop of 12pp => warning alert."""
        cache = _make_cache(
            alltime_3h={"macd": {"accuracy": 0.70, "total": 500}},
            recent_3h={"macd": {"accuracy": 0.58, "total": 60}},
        )
        path = _write_cache(tmp_path, cache)
        alerts = check_signal_decay(path)
        assert len(alerts) == 1
        assert alerts[0]["signal"] == "macd"
        assert alerts[0]["horizon"] == "3h"
        assert alerts[0]["severity"] == "warning"
        assert alerts[0]["drop_pp"] == 12.0

    def test_critical_level_decay(self, tmp_path):
        """Drop of 25pp => critical alert."""
        cache = _make_cache(
            alltime_1d={"trend": {"accuracy": 0.75, "total": 300}},
            recent_1d={"trend": {"accuracy": 0.50, "total": 80}},
        )
        path = _write_cache(tmp_path, cache)
        alerts = check_signal_decay(path)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "critical"
        assert alerts[0]["drop_pp"] == 25.0

    def test_insufficient_recent_samples_skipped(self, tmp_path):
        """Recent samples below MIN_RECENT_SAMPLES => skip."""
        cache = _make_cache(
            alltime_3h={"rsi": {"accuracy": 0.70, "total": 200}},
            recent_3h={"rsi": {"accuracy": 0.40, "total": 10}},  # only 10 samples
        )
        path = _write_cache(tmp_path, cache)
        alerts = check_signal_decay(path)
        assert alerts == []

    def test_insufficient_alltime_samples_skipped(self, tmp_path):
        """Alltime samples below MIN_ALLTIME_SAMPLES => skip."""
        cache = _make_cache(
            alltime_3h={"rsi": {"accuracy": 0.70, "total": 30}},  # only 30 samples
            recent_3h={"rsi": {"accuracy": 0.40, "total": 60}},
        )
        path = _write_cache(tmp_path, cache)
        alerts = check_signal_decay(path)
        assert alerts == []

    def test_multiple_signals_sorted_by_drop(self, tmp_path):
        """Multiple decaying signals sorted worst-first."""
        cache = _make_cache(
            alltime_3h={
                "rsi": {"accuracy": 0.70, "total": 200},
                "macd": {"accuracy": 0.65, "total": 200},
            },
            recent_3h={
                "rsi": {"accuracy": 0.55, "total": 60},   # 15pp drop
                "macd": {"accuracy": 0.40, "total": 60},   # 25pp drop
            },
        )
        path = _write_cache(tmp_path, cache)
        alerts = check_signal_decay(path)
        assert len(alerts) == 2
        assert alerts[0]["signal"] == "macd"   # 25pp drop first
        assert alerts[1]["signal"] == "rsi"    # 15pp drop second

    def test_both_horizons_checked(self, tmp_path):
        """Alerts can come from both 3h and 1d horizons."""
        cache = _make_cache(
            alltime_3h={"rsi": {"accuracy": 0.70, "total": 200}},
            recent_3h={"rsi": {"accuracy": 0.55, "total": 60}},
            alltime_1d={"trend": {"accuracy": 0.75, "total": 300}},
            recent_1d={"trend": {"accuracy": 0.50, "total": 80}},
        )
        path = _write_cache(tmp_path, cache)
        alerts = check_signal_decay(path)
        assert len(alerts) == 2
        horizons = {a["horizon"] for a in alerts}
        assert horizons == {"3h", "1d"}

    def test_improvement_not_alerted(self, tmp_path):
        """Recent accuracy BETTER than alltime => no alert (negative drop)."""
        cache = _make_cache(
            alltime_3h={"rsi": {"accuracy": 0.55, "total": 200}},
            recent_3h={"rsi": {"accuracy": 0.70, "total": 60}},
        )
        path = _write_cache(tmp_path, cache)
        alerts = check_signal_decay(path)
        assert alerts == []

    def test_missing_alltime_signal_skipped(self, tmp_path):
        """Signal in recent but not in alltime => skip (no baseline)."""
        cache = _make_cache(
            alltime_3h={},
            recent_3h={"new_signal": {"accuracy": 0.30, "total": 60}},
        )
        path = _write_cache(tmp_path, cache)
        alerts = check_signal_decay(path)
        assert alerts == []

    def test_non_dict_values_skipped(self, tmp_path):
        """Non-dict entries in cache are handled gracefully."""
        cache = _make_cache(
            alltime_3h="not a dict",
            recent_3h={"rsi": {"accuracy": 0.40, "total": 60}},
        )
        path = _write_cache(tmp_path, cache)
        alerts = check_signal_decay(path)
        assert alerts == []


class TestFormatDecayReport:
    """Tests for format_decay_report()."""

    def test_empty_alerts_returns_empty_string(self):
        assert format_decay_report([]) == ""

    def test_warning_format(self):
        alerts = [{
            "signal": "rsi",
            "horizon": "3h",
            "alltime_acc": 70.0,
            "recent_acc": 55.0,
            "drop_pp": 15.0,
            "recent_samples": 60,
            "alltime_samples": 200,
            "severity": "warning",
        }]
        report = format_decay_report(alerts)
        assert "SIGNAL DECAY ALERT" in report
        assert "WARNING" in report
        assert "rsi" in report
        assert "CRITICAL" not in report

    def test_critical_format(self):
        alerts = [{
            "signal": "trend",
            "horizon": "1d",
            "alltime_acc": 75.0,
            "recent_acc": 50.0,
            "drop_pp": 25.0,
            "recent_samples": 80,
            "alltime_samples": 300,
            "severity": "critical",
        }]
        report = format_decay_report(alerts)
        assert "CRITICAL" in report
        assert "trend" in report

    def test_mixed_severities(self):
        alerts = [
            {
                "signal": "trend", "horizon": "1d",
                "alltime_acc": 75.0, "recent_acc": 50.0, "drop_pp": 25.0,
                "recent_samples": 80, "alltime_samples": 300, "severity": "critical",
            },
            {
                "signal": "rsi", "horizon": "3h",
                "alltime_acc": 70.0, "recent_acc": 55.0, "drop_pp": 15.0,
                "recent_samples": 60, "alltime_samples": 200, "severity": "warning",
            },
        ]
        report = format_decay_report(alerts)
        assert "CRITICAL" in report
        assert "WARNING" in report
        assert "2 signals degrading" in report


class TestLogDecayAlerts:
    """Tests for log_decay_alerts()."""

    def test_no_alerts_logs_info(self, caplog):
        import logging
        with caplog.at_level(logging.INFO, logger="portfolio.signal_decay_alert"):
            log_decay_alerts([])
        assert "no degradation detected" in caplog.text

    def test_alerts_logged_as_warning(self, caplog):
        import logging
        alerts = [{
            "signal": "rsi", "horizon": "3h",
            "alltime_acc": 70.0, "recent_acc": 55.0, "drop_pp": 15.0,
            "recent_samples": 60, "alltime_samples": 200, "severity": "warning",
        }]
        with caplog.at_level(logging.WARNING, logger="portfolio.signal_decay_alert"):
            log_decay_alerts(alerts)
        assert "Signal decay detected" in caplog.text


class TestRunDecayCheck:
    """Tests for run_decay_check() integration."""

    def test_run_returns_alerts(self, tmp_path, monkeypatch):
        cache = _make_cache(
            alltime_3h={"rsi": {"accuracy": 0.70, "total": 200}},
            recent_3h={"rsi": {"accuracy": 0.55, "total": 60}},
        )
        path = _write_cache(tmp_path, cache)
        monkeypatch.setattr(
            "portfolio.signal_decay_alert.check_signal_decay",
            lambda p=None: check_signal_decay(path),
        )
        alerts = run_decay_check()
        assert len(alerts) == 1
        assert alerts[0]["signal"] == "rsi"
