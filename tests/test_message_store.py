"""Tests for portfolio.message_store readability hardening + truncation safety."""

import json
from unittest.mock import MagicMock


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


# ---------------------------------------------------------------------------
# BUG-131: Telegram truncation preserves Markdown integrity
# ---------------------------------------------------------------------------


class TestTelegramTruncation:
    """Verify that long messages are truncated at line boundaries."""

    def test_truncates_at_newline_boundary(self, monkeypatch, tmp_path):
        """BUG-131: Truncation should cut at last newline, not mid-line."""
        import portfolio.message_store as ms

        monkeypatch.setattr(ms, "MESSAGES_FILE", tmp_path / "msgs.jsonl")
        monkeypatch.setenv("NO_TELEGRAM", "")  # Ensure we enter _do_send_telegram

        # Build a message that exceeds 4096 chars with distinct lines
        lines = [f"Line {i}: " + "x" * 50 for i in range(100)]
        long_msg = "\n".join(lines)
        assert len(long_msg) > 4096

        sent_messages = []

        def fake_fetch(url, method="GET", json_body=None, timeout=30, **kw):
            sent_messages.append(json_body.get("text", ""))
            resp = MagicMock()
            resp.ok = True
            return resp

        monkeypatch.setattr(ms, "fetch_with_retry", fake_fetch)
        monkeypatch.delenv("NO_TELEGRAM", raising=False)

        ms._do_send_telegram(
            long_msg,
            {"telegram": {"token": "t", "chat_id": "c"}},
        )

        assert len(sent_messages) == 1
        sent = sent_messages[0]
        assert len(sent) <= 4096
        assert sent.endswith("...(truncated)")
        # Should end with a complete line before truncation marker
        lines_sent = sent.split("\n")
        # The last line is "...(truncated)", the one before should be complete
        assert lines_sent[-2].startswith("Line ")

    def test_short_message_not_truncated(self, monkeypatch, tmp_path):
        """Messages under 4096 chars should not be modified."""
        import portfolio.message_store as ms

        monkeypatch.setattr(ms, "MESSAGES_FILE", tmp_path / "msgs.jsonl")

        sent_messages = []

        def fake_fetch(url, method="GET", json_body=None, timeout=30, **kw):
            sent_messages.append(json_body.get("text", ""))
            resp = MagicMock()
            resp.ok = True
            return resp

        monkeypatch.setattr(ms, "fetch_with_retry", fake_fetch)
        monkeypatch.delenv("NO_TELEGRAM", raising=False)

        short_msg = "*Bold text*\nSome content\n_italic_"
        ms._do_send_telegram(
            short_msg,
            {"telegram": {"token": "t", "chat_id": "c"}},
        )

        assert sent_messages[0] == short_msg

    def test_truncation_doesnt_split_markdown_bold(self, monkeypatch, tmp_path):
        """A line with *bold* should either be fully included or excluded."""
        import portfolio.message_store as ms

        monkeypatch.setattr(ms, "MESSAGES_FILE", tmp_path / "msgs.jsonl")

        sent_messages = []

        def fake_fetch(url, method="GET", json_body=None, timeout=30, **kw):
            sent_messages.append(json_body.get("text", ""))
            resp = MagicMock()
            resp.ok = True
            return resp

        monkeypatch.setattr(ms, "fetch_with_retry", fake_fetch)
        monkeypatch.delenv("NO_TELEGRAM", raising=False)

        # Build message where cutting at char boundary would split a *bold* tag
        prefix = "A" * 4050 + "\n"
        bold_line = "*This bold text should not be split*\n"
        suffix = "More text\n" * 10
        msg = prefix + bold_line + suffix
        assert len(msg) > 4096

        ms._do_send_telegram(
            msg,
            {"telegram": {"token": "t", "chat_id": "c"}},
        )

        sent = sent_messages[0]
        # Should not contain an unclosed bold marker
        # Count asterisks in the truncated message (excluding the marker)
        content = sent.replace("...(truncated)", "")
        star_count = content.count("*")
        assert star_count % 2 == 0, f"Unclosed Markdown: {star_count} asterisks"
