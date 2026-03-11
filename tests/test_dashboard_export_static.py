import json
from pathlib import Path

import dashboard.export_static as export_mod


class _Resp:
    status_code = 200

    @staticmethod
    def get_json():
        return {"ok": True}


class _FakeClient:
    def __init__(self, calls):
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, route):
        self._calls.append(route)
        return _Resp()


class _FakeApp:
    def __init__(self, calls):
        self._calls = calls

    def test_client(self):
        return _FakeClient(self._calls)


def test_endpoints_include_frontend_required_routes():
    routes = {route for route, _ in export_mod.ENDPOINTS}
    assert "/api/accuracy" in routes
    assert "/api/golddigger" in routes
    assert "/api/metals-accuracy" in routes
    assert "/api/lora-status" in routes
    assert "/api/local-llm-trends" in routes


def test_export_all_appends_dashboard_token(monkeypatch, tmp_path):
    calls = []
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"dashboard_token": "secret123"}), encoding="utf-8")

    monkeypatch.setattr(export_mod, "CONFIG_PATH", cfg)
    monkeypatch.setattr(export_mod, "ENDPOINTS", [("/api/summary", "summary.json")])
    monkeypatch.setattr(export_mod, "app", _FakeApp(calls))

    result = export_mod.export_all(out_dir=tmp_path / "out")

    assert result["failed"] == []
    assert result["ok"] == ["summary.json"]
    assert calls == ["/api/summary?token=secret123"]


def test_export_all_sanitizes_nan_values(monkeypatch, tmp_path):
    class _NaNResp:
        status_code = 200

        @staticmethod
        def get_json():
            return {"value": float("nan")}

    class _NaNClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, route):
            return _NaNResp()

    class _NaNApp:
        @staticmethod
        def test_client():
            return _NaNClient()

    monkeypatch.setattr(export_mod, "ENDPOINTS", [("/api/summary", "summary.json")])
    monkeypatch.setattr(export_mod, "app", _NaNApp())

    result = export_mod.export_all(out_dir=tmp_path / "out")

    assert result["failed"] == []
    written = (tmp_path / "out" / "summary.json").read_text(encoding="utf-8")
    assert '"value":null' in written
    assert "NaN" not in written
