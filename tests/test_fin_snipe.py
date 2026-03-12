"""Tests for portfolio.fin_snipe module."""

import json

import pytest


def test_load_json_uses_file_utils(tmp_path, monkeypatch):
    """_load_json delegates to file_utils.load_json (TOCTOU-safe)."""
    import portfolio.fin_snipe as mod

    calls = []
    original_load = mod.load_json

    def tracking_load(path):
        calls.append(str(path))
        return {"test": True}

    monkeypatch.setattr(mod, "load_json", tracking_load)

    result = mod._load_json(tmp_path / "test.json")
    assert result == {"test": True}
    assert len(calls) == 1


def test_load_json_returns_empty_on_missing(tmp_path, monkeypatch):
    """_load_json returns {} when file_utils.load_json returns None."""
    import portfolio.fin_snipe as mod

    monkeypatch.setattr(mod, "load_json", lambda path: None)
    result = mod._load_json(tmp_path / "nonexistent.json")
    assert result == {}


def test_fetch_open_buy_orders_removed():
    """_fetch_open_buy_orders dead code has been removed."""
    import portfolio.fin_snipe as mod

    assert not hasattr(mod, "_fetch_open_buy_orders")
