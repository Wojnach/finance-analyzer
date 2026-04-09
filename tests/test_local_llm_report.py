import json
from datetime import UTC, datetime

from portfolio.local_llm_report import build_local_llm_report, maybe_export_local_llm_report


def test_build_local_llm_report_recommendations(tmp_path, monkeypatch):
    pred_path = tmp_path / "forecast_predictions.jsonl"
    pred_entries = [
        {
            "ts": "2026-04-07T12:00:00+00:00",  # recent entry, well within 30-day window
            "gating_action": "raw",
            "subsignal_gating": {
                "chronos_1h": {"gating": "held"},
                "chronos_24h": {"gating": "raw"},
            },
        }
    ]
    with open(pred_path, "w", encoding="utf-8") as f:
        for entry in pred_entries:
            f.write(json.dumps(entry) + "\n")

    monkeypatch.setattr(
        "portfolio.local_llm_report.load_config",
        lambda: {
            "forecast": {"chronos_model": "amazon/chronos-t5-small"},
            "local_models": {"ministral": {"hold_threshold": 0.55}},
        },
    )
    monkeypatch.setattr(
        "portfolio.local_llm_report.accuracy_by_signal_ticker",
        lambda signal_name, horizon="1d", days=30: {
            "BTC-USD": {"accuracy": 0.52, "samples": 20, "correct": 10}
        },
    )

    def fake_compute(horizon="24h", days=30, predictions_file=None, use_raw_sub_signals=False):
        if use_raw_sub_signals:
            return {}
        return {"chronos_24h": {"accuracy": 0.6, "correct": 6, "total": 10, "by_ticker": {}}}

    monkeypatch.setattr("portfolio.local_llm_report.compute_forecast_accuracy", fake_compute)
    monkeypatch.setattr(
        "portfolio.local_llm_report.load_health_stats",
        lambda health_file=None: {
            "kronos": {"ok": 5, "fail": 5, "total": 10, "success_rate": 0.5},
            "chronos": {"ok": 10, "fail": 0, "total": 10, "success_rate": 1.0},
        },
    )

    report = build_local_llm_report(days=30, predictions_file=pred_path)

    assert report["gating_counts"]["forecast"]["raw"] == 1
    assert report["gating_counts"]["subsignals"]["chronos_1h"]["held"] == 1
    assert any("chronos-bolt-small" in rec for rec in report["recommendations"])
    assert any("--forecast-outcomes" in rec for rec in report["recommendations"])
    assert any("Kronos gated" in rec or "Kronos" in rec for rec in report["recommendations"])
    assert any("Ministral" in rec for rec in report["recommendations"])


def test_build_local_llm_report_ministral_summary(tmp_path, monkeypatch):
    pred_path = tmp_path / "forecast_predictions.jsonl"
    pred_path.write_text("", encoding="utf-8")

    monkeypatch.setattr("portfolio.local_llm_report.load_config", lambda: {})
    monkeypatch.setattr(
        "portfolio.local_llm_report.accuracy_by_signal_ticker",
        lambda signal_name, horizon="1d", days=30: {
            "BTC-USD": {"accuracy": 0.60, "samples": 10, "correct": 6},
            "ETH-USD": {"accuracy": 0.70, "samples": 10, "correct": 7},
        },
    )
    monkeypatch.setattr(
        "portfolio.local_llm_report.compute_forecast_accuracy",
        lambda horizon="24h", days=30, predictions_file=None, use_raw_sub_signals=False: {},
    )
    monkeypatch.setattr("portfolio.local_llm_report.load_health_stats", lambda health_file=None: {})

    report = build_local_llm_report(days=30, predictions_file=pred_path)

    assert report["ministral"]["overall"]["samples"] == 20
    assert report["ministral"]["overall"]["correct"] == 13
    assert report["ministral"]["overall"]["accuracy"] == 0.65
    assert report["config"]["local_llm_report"]["report_days"] == 30


def test_build_local_llm_report_falls_back_without_config_json(tmp_path, monkeypatch):
    pred_path = tmp_path / "forecast_predictions.jsonl"
    pred_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "portfolio.local_llm_report.load_config",
        lambda: (_ for _ in ()).throw(FileNotFoundError("missing config.json")),
    )
    monkeypatch.setattr(
        "portfolio.local_llm_report.CONFIG_EXAMPLE_FILE",
        tmp_path / "config.example.json",
    )
    monkeypatch.setattr(
        "portfolio.local_llm_report.accuracy_by_signal_ticker",
        lambda signal_name, horizon="1d", days=30: {},
    )
    monkeypatch.setattr(
        "portfolio.local_llm_report.compute_forecast_accuracy",
        lambda horizon="24h", days=30, predictions_file=None, use_raw_sub_signals=False: {},
    )
    monkeypatch.setattr("portfolio.local_llm_report.load_health_stats", lambda health_file=None: {})

    report = build_local_llm_report(days=30, predictions_file=pred_path)

    assert report["config"]["forecast"] == {}
    assert report["config"]["local_llm_report"]["daily_export_enabled"] is True


def test_build_local_llm_report_sanitizes_config(tmp_path, monkeypatch):
    pred_path = tmp_path / "forecast_predictions.jsonl"
    pred_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "portfolio.local_llm_report.load_config",
        lambda: {
            "forecast": {"chronos_model": "amazon/chronos-bolt-small"},
            "local_models": {"ministral": {"hold_threshold": 0.55}},
            "exchange": {"apiKey": "SECRET", "secret": "SECRET"},
            "telegram": {"token": "SECRET", "chat_id": "123"},
        },
    )
    monkeypatch.setattr(
        "portfolio.local_llm_report.accuracy_by_signal_ticker",
        lambda signal_name, horizon="1d", days=30: {},
    )
    monkeypatch.setattr(
        "portfolio.local_llm_report.compute_forecast_accuracy",
        lambda horizon="24h", days=30, predictions_file=None, use_raw_sub_signals=False: {},
    )
    monkeypatch.setattr("portfolio.local_llm_report.load_health_stats", lambda health_file=None: {})

    report = build_local_llm_report(days=30, predictions_file=pred_path)

    assert "exchange" not in report["config"]
    assert "telegram" not in report["config"]
    assert report["config"]["forecast"]["chronos_model"] == "amazon/chronos-bolt-small"


def test_maybe_export_local_llm_report_writes_daily_snapshot_once(tmp_path, monkeypatch):
    latest_file = tmp_path / "local_llm_report_latest.json"
    history_file = tmp_path / "local_llm_report_history.jsonl"
    state_file = tmp_path / "local_llm_report_export_state.json"

    monkeypatch.setattr(
        "portfolio.local_llm_report.build_local_llm_report",
        lambda days=30, config=None, predictions_file=None, health_file=None: {
            "days": days,
            "config": {"forecast": {}, "local_models": {}, "local_llm_report": {}},
            "health": {"chronos": {"success_rate": 1.0, "ok": 1, "total": 1}},
            "ministral": {"overall": {"accuracy": 0.6, "correct": 6, "samples": 10}, "by_ticker": {}},
            "forecast": {"raw": {}, "effective": {}},
            "gating_counts": {"forecast": {}, "subsignals": {}},
            "recommendations": [],
        },
    )

    config = {
        "local_llm_report": {
            "daily_export_enabled": True,
            "report_days": 45,
            "history_max_entries": 5,
        }
    }
    now = datetime(2026, 3, 9, 8, 0, tzinfo=UTC)

    first = maybe_export_local_llm_report(
        config=config,
        now=now,
        latest_file=latest_file,
        history_file=history_file,
        state_file=state_file,
    )
    second = maybe_export_local_llm_report(
        config=config,
        now=now,
        latest_file=latest_file,
        history_file=history_file,
        state_file=state_file,
    )

    history_entries = [
        json.loads(line)
        for line in history_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert first["days"] == 45
    assert second is None
    assert len(history_entries) == 1
    assert json.loads(latest_file.read_text(encoding="utf-8"))["date"] == "2026-03-09"
