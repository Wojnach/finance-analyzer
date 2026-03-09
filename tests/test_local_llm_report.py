import json

from portfolio.local_llm_report import build_local_llm_report


def test_build_local_llm_report_recommendations(tmp_path, monkeypatch):
    pred_path = tmp_path / "forecast_predictions.jsonl"
    pred_entries = [
        {
            "ts": "2026-03-09T12:00:00+00:00",
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

    assert report["config"] == {}
