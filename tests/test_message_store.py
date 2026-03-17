"""Tests for portfolio.message_store readability hardening."""

import json


def test_sanitize_message_text_strips_control_bytes():
    from portfolio.message_store import sanitize_message_text

    raw = (
        "*T1 CHECK* · MU \x05 -4.4% · F&G 13/37\n\n"
        "_Held 9.7sh @ \x13 | ATR 0.58% | stop ~\x00 OK_\n\n\n"
        "_P:461K MU 9.7sh(-4.4%) · B:458K_"
    )

    cleaned = sanitize_message_text(raw)

    assert "\x05" not in cleaned
    assert "\x13" not in cleaned
    assert "\x00" not in cleaned
    assert "*T1 CHECK* · MU -4.4% · F&G 13/37" in cleaned
    assert "_Held 9.7sh @ | ATR 0.58% | stop ~ OK_" in cleaned
    assert "\n\n\n" not in cleaned


def test_sanitize_message_text_repairs_common_mojibake():
    from portfolio.message_store import sanitize_message_text

    raw = "Patient: HOLD â DXY weak Â· watch XAG â close"

    cleaned = sanitize_message_text(raw)

    assert cleaned == "Patient: HOLD — DXY weak · watch XAG → close"


def test_send_or_store_logs_and_sends_cleaned_text(monkeypatch, tmp_path):
    import portfolio.message_store as message_store

    sent = []
    log_path = tmp_path / "telegram_messages.jsonl"

    monkeypatch.setattr(message_store, "MESSAGES_FILE", log_path)
    monkeypatch.setattr(
        message_store,
        "_do_send_telegram",
        lambda msg, config: sent.append(msg) or True,
    )

    raw = "*T1 CHECK* · MU \x05 -4.4% Â· F&G 13/37"
    result = message_store.send_or_store(
        raw,
        {"telegram": {"token": "x", "chat_id": "y"}},
        category="analysis",
    )

    assert result is True
    assert sent == ["*T1 CHECK* · MU -4.4% · F&G 13/37"]

    entries = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(entries) == 1
    assert entries[0]["text"] == sent[0]
    assert entries[0]["category"] == "analysis"
    assert entries[0]["sent"] is True
