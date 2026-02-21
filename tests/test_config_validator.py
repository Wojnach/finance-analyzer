"""Tests for portfolio.config_validator."""

import json
import logging

import pytest

from portfolio.config_validator import validate_config, validate_config_file


# --- validate_config (dict-based) ---


def _valid_config():
    """Minimal valid config with all required keys."""
    return {
        "telegram": {"token": "123:ABC", "chat_id": "456"},
        "alpaca": {"key": "ak_test", "secret": "sk_test"},
        "mistral_api_key": "mk_test",
        "iskbets": {"min_bigbet_conditions": 2},
    }


def test_valid_config_passes():
    errors = validate_config(_valid_config())
    assert errors == []


def test_missing_telegram_token():
    cfg = _valid_config()
    del cfg["telegram"]["token"]
    errors = validate_config(cfg)
    assert any("telegram.token" in e for e in errors)


def test_missing_telegram_chat_id():
    cfg = _valid_config()
    del cfg["telegram"]["chat_id"]
    errors = validate_config(cfg)
    assert any("telegram.chat_id" in e for e in errors)


def test_missing_telegram_section():
    cfg = _valid_config()
    del cfg["telegram"]
    errors = validate_config(cfg)
    assert any("telegram.token" in e for e in errors)
    assert any("telegram.chat_id" in e for e in errors)


def test_missing_alpaca_key():
    cfg = _valid_config()
    del cfg["alpaca"]["key"]
    errors = validate_config(cfg)
    assert any("alpaca.key" in e for e in errors)


def test_missing_alpaca_secret():
    cfg = _valid_config()
    del cfg["alpaca"]["secret"]
    errors = validate_config(cfg)
    assert any("alpaca.secret" in e for e in errors)


def test_empty_telegram_token():
    cfg = _valid_config()
    cfg["telegram"]["token"] = "  "
    errors = validate_config(cfg)
    assert any("telegram.token" in e for e in errors)


def test_empty_alpaca_key():
    cfg = _valid_config()
    cfg["alpaca"]["key"] = ""
    errors = validate_config(cfg)
    assert any("alpaca.key" in e for e in errors)


def test_multiple_missing_keys():
    errors = validate_config({})
    assert len(errors) == 4  # all 4 required keys missing


# --- validate_config_file ---


def test_validate_config_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "portfolio.config_validator.CONFIG_FILE", tmp_path / "nonexistent.json"
    )
    with pytest.raises(ValueError, match="not found"):
        validate_config_file()


def test_validate_config_file_invalid_keys(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"telegram": {}}))
    monkeypatch.setattr("portfolio.config_validator.CONFIG_FILE", cfg_path)
    with pytest.raises(ValueError, match="validation failed"):
        validate_config_file()


def test_validate_config_file_valid(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(_valid_config()))
    monkeypatch.setattr("portfolio.config_validator.CONFIG_FILE", cfg_path)
    result = validate_config_file()
    assert result["telegram"]["token"] == "123:ABC"


def test_validate_config_file_warns_optional(tmp_path, monkeypatch, caplog):
    cfg = _valid_config()
    del cfg["mistral_api_key"]
    del cfg["iskbets"]
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    monkeypatch.setattr("portfolio.config_validator.CONFIG_FILE", cfg_path)

    with caplog.at_level(logging.WARNING, logger="portfolio.config_validator"):
        result = validate_config_file()

    assert result is not None
    assert any("mistral_api_key" in r.message for r in caplog.records)
    assert any("iskbets" in r.message for r in caplog.records)


def test_validate_config_file_no_warn_when_optional_present(tmp_path, monkeypatch, caplog):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(_valid_config()))
    monkeypatch.setattr("portfolio.config_validator.CONFIG_FILE", cfg_path)

    with caplog.at_level(logging.WARNING, logger="portfolio.config_validator"):
        validate_config_file()

    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("optional" in m for m in warning_messages)
