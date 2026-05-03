"""Tests for loop_health.write_heartbeat() — the shared helper that
metals_loop and golddigger (and any future loop) call to emit the
watchdog-compatible heartbeat file.

Coverage:
- Schema written matches what read_loop_status / read_loop_health expect.
- n_positions / cycle / ok / extra fields round-trip correctly.
- Failure path swallows exceptions and returns False.
- Override timestamp produces deterministic output.
"""
from __future__ import annotations

import datetime
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portfolio import loop_health


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_writes_minimal_payload(tmp_path):
    hb = tmp_path / "x_loop.heartbeat"
    ok = loop_health.write_heartbeat(hb, cycle=1)
    assert ok is True
    assert hb.exists()

    payload = _read(hb)
    # Required by read_loop_status:
    assert isinstance(payload["ts"], str)
    datetime.datetime.fromisoformat(payload["ts"].replace("Z", "+00:00"))
    # Default operator fields:
    assert payload["status"] == "ok"
    assert payload["cycle"] == 1
    assert payload["ok"] is True
    assert payload["n_positions"] == 0


def test_writes_extra_fields(tmp_path):
    hb = tmp_path / "x_loop.heartbeat"
    loop_health.write_heartbeat(
        hb, cycle=42, n_positions=3, extra={"phase": "shadow", "regime": "trending_up"}
    )
    payload = _read(hb)
    assert payload["cycle"] == 42
    assert payload["n_positions"] == 3
    assert payload["phase"] == "shadow"
    assert payload["regime"] == "trending_up"


def test_extra_cannot_override_required_fields(tmp_path):
    """`extra` is a merge — caller can override defaults like ok if they
    really want to. This is intentional to keep the helper flexible, but
    we document the behavior here so a future refactor doesn't quietly
    break it."""
    hb = tmp_path / "x_loop.heartbeat"
    loop_health.write_heartbeat(
        hb, cycle=1, ok=True, extra={"ok": False, "custom_status": "degraded"}
    )
    payload = _read(hb)
    # extra wins — caller signaled they wanted to override.
    assert payload["ok"] is False
    assert payload["custom_status"] == "degraded"


def test_uses_provided_timestamp(tmp_path):
    """Caller can pin `now` for reproducible tests."""
    hb = tmp_path / "x_loop.heartbeat"
    pinned = datetime.datetime(2026, 5, 3, 12, 0, 0, tzinfo=datetime.UTC)
    loop_health.write_heartbeat(hb, cycle=1, now=pinned)
    payload = _read(hb)
    assert payload["ts"] == "2026-05-03T12:00:00+00:00"


def test_failure_returns_false_and_does_not_raise(tmp_path, monkeypatch):
    """If atomic_write_json fails, the helper must NOT propagate — live
    trading loops cannot crash on telemetry failure."""
    hb = tmp_path / "x_loop.heartbeat"

    def _boom(*_a, **_kw):
        raise OSError("disk full (simulated)")

    from portfolio import file_utils
    monkeypatch.setattr(file_utils, "atomic_write_json", _boom)

    result = loop_health.write_heartbeat(hb, cycle=1)
    assert result is False
    assert not hb.exists()


def test_round_trips_through_read_loop_status(tmp_path):
    """End-to-end: write_heartbeat output is read as 'fresh' by the
    consumer side — guarantees the schema match."""
    hb = tmp_path / "metals_loop.heartbeat"
    pinned = datetime.datetime(2026, 5, 3, 12, 0, 0, tzinfo=datetime.UTC)
    loop_health.write_heartbeat(hb, cycle=99, n_positions=2, now=pinned)

    # Read 60s later — should be 'fresh'.
    later = pinned + datetime.timedelta(seconds=60)
    status = loop_health.read_loop_status("metals", hb, now=later)
    assert status["state"] == "fresh"
    assert status["age_seconds"] == 60.0
    assert status["payload"]["cycle"] == 99
    assert status["payload"]["n_positions"] == 2


def test_round_trips_through_read_loop_health(tmp_path):
    """All five default loops can be assembled into a healthy rollup."""
    pinned = datetime.datetime(2026, 5, 3, 12, 0, 0, tzinfo=datetime.UTC)

    # Materialize one heartbeat per default loop in a fake repo root.
    (tmp_path / "data").mkdir(exist_ok=True)
    for name, rel in loop_health.DEFAULT_HEARTBEAT_FILES.items():
        full = tmp_path / rel
        loop_health.write_heartbeat(full, cycle=1, now=pinned)

    later = pinned + datetime.timedelta(seconds=30)
    rollup = loop_health.read_loop_health(repo_root=tmp_path, now=later)

    assert rollup["any_unhealthy"] is False, rollup["unhealthy"]
    for name in loop_health.DEFAULT_HEARTBEAT_FILES:
        assert rollup["loops"][name]["state"] == "fresh", name


# ---------------------------------------------------------------------------
# 2026-05-04 wrapper-migration contract tests.
# crypto_loop / oil_loop / mstr_loop now delegate to loop_health.write_heartbeat;
# verify each shim still produces a payload that read_loop_status accepts as
# fresh and that the schema (ts/status/cycle/ok/n_positions) is preserved.
# ---------------------------------------------------------------------------

class TestWrapperShims:
    def test_crypto_wrapper_produces_compatible_payload(self, tmp_path, monkeypatch):
        """data/crypto_loop.write_heartbeat unpacks `extra` correctly."""
        from data import crypto_loop
        hb = tmp_path / "crypto_loop.heartbeat"
        monkeypatch.setattr(crypto_loop, "HEARTBEAT_FILE", str(hb))

        crypto_loop.write_heartbeat({"cycle": 7, "ok": True, "n_positions": 2})

        payload = _read(hb)
        assert payload["cycle"] == 7
        assert payload["ok"] is True
        assert payload["n_positions"] == 2
        assert payload["status"] == "ok"
        # Schema is consumable by read_loop_status.
        status = loop_health.read_loop_status("crypto", hb)
        assert status["state"] == "fresh"

    def test_crypto_wrapper_passes_through_unknown_keys(self, tmp_path, monkeypatch):
        """Extra keys not in the canonical set ride along as context."""
        from data import crypto_loop
        hb = tmp_path / "crypto_loop.heartbeat"
        monkeypatch.setattr(crypto_loop, "HEARTBEAT_FILE", str(hb))

        crypto_loop.write_heartbeat({
            "cycle": 1, "ok": True, "n_positions": 0,
            "fast_tick_alerts": 3, "slow_phase_seen": False,
        })
        payload = _read(hb)
        assert payload["fast_tick_alerts"] == 3
        assert payload["slow_phase_seen"] is False

    def test_oil_wrapper_produces_compatible_payload(self, tmp_path, monkeypatch):
        from data import oil_loop
        hb = tmp_path / "oil_loop.heartbeat"
        monkeypatch.setattr(oil_loop, "HEARTBEAT_FILE", str(hb))

        oil_loop.write_heartbeat({"cycle": 42, "ok": False, "n_positions": 1})

        payload = _read(hb)
        assert payload["cycle"] == 42
        assert payload["ok"] is False
        assert payload["n_positions"] == 1
        status = loop_health.read_loop_status("oil", hb)
        assert status["state"] == "fresh"

    def test_mstr_wrapper_produces_compatible_payload(self, tmp_path, monkeypatch):
        """portfolio.mstr_loop._write_heartbeat threads phase through extra."""
        from portfolio.mstr_loop import config as mstr_config
        from portfolio.mstr_loop import loop as mstr_loop
        from portfolio.mstr_loop.state import default_state

        hb = tmp_path / "mstr_loop.heartbeat"
        monkeypatch.setattr(mstr_config, "HEARTBEAT_FILE", str(hb))

        mstr_loop._write_heartbeat(default_state(), cycle_count=99)

        payload = _read(hb)
        assert payload["cycle"] == 99
        assert payload["ok"] is True
        assert payload["n_positions"] == 0
        assert payload["phase"] == mstr_config.PHASE
        status = loop_health.read_loop_status("mstr", hb)
        assert status["state"] == "fresh"

    def test_wrapper_failure_does_not_raise(self, tmp_path, monkeypatch):
        """Defence in depth: even if the shared helper imports fail, the
        wrapper must not propagate. (Real-world this would be a circular
        import or sys.modules tampering.)"""
        from data import crypto_loop

        # Force the inner import to fail.
        import builtins
        real_import = builtins.__import__

        def _failing_import(name, *args, **kwargs):
            if name == "portfolio.loop_health":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _failing_import)
        # Must not raise.
        crypto_loop.write_heartbeat({"cycle": 1})


# ---------------------------------------------------------------------------
# 2026-05-04 codex P3-2: shim must coerce-then-call all inside the
# try/except, so a malformed `extra` (non-dict, bad cycle, etc.) cannot
# raise out to the live trading loop.
# ---------------------------------------------------------------------------

class TestWrapperCoercionSafety:
    def test_crypto_wrapper_swallows_non_int_cycle(self, tmp_path, monkeypatch):
        from data import crypto_loop
        hb = tmp_path / "crypto_loop.heartbeat"
        monkeypatch.setattr(crypto_loop, "HEARTBEAT_FILE", str(hb))

        # Must NOT raise even though cycle="N/A".
        crypto_loop.write_heartbeat({"cycle": "N/A", "ok": True, "n_positions": 0})
        # Heartbeat still written with cycle defaulted to 0.
        payload = _read(hb)
        assert payload["cycle"] == 0
        assert payload["ok"] is True

    def test_crypto_wrapper_swallows_non_int_n_positions(self, tmp_path, monkeypatch):
        from data import crypto_loop
        hb = tmp_path / "crypto_loop.heartbeat"
        monkeypatch.setattr(crypto_loop, "HEARTBEAT_FILE", str(hb))

        crypto_loop.write_heartbeat({"cycle": 1, "ok": True,
                                      "n_positions": {"weird": "shape"}})
        payload = _read(hb)
        assert payload["n_positions"] == 0

    def test_crypto_wrapper_swallows_non_dict_extra(self, tmp_path, monkeypatch):
        """If a future caller passes a list (or any non-dict), the dict()
        coercion would raise — must be caught silently."""
        from data import crypto_loop
        hb = tmp_path / "crypto_loop.heartbeat"
        monkeypatch.setattr(crypto_loop, "HEARTBEAT_FILE", str(hb))

        # Pass a list — dict(list) tries to make pairs, fails for non-pair items.
        crypto_loop.write_heartbeat([1, 2, 3])
        # No raise. May or may not write the file (failure may be at coerce
        # time), but the loop didn't crash.

    def test_oil_wrapper_swallows_non_int_cycle(self, tmp_path, monkeypatch):
        from data import oil_loop
        hb = tmp_path / "oil_loop.heartbeat"
        monkeypatch.setattr(oil_loop, "HEARTBEAT_FILE", str(hb))
        oil_loop.write_heartbeat({"cycle": None, "ok": True, "n_positions": 0})
        payload = _read(hb)
        assert payload["cycle"] == 0


# ---------------------------------------------------------------------------
# 2026-05-04 codex P3-1: load_jsonl_tail must keep the first decoded
# line when the seek lands exactly on a newline boundary.
# ---------------------------------------------------------------------------

class TestLoadJsonlTailBoundary:
    """Note: tests write files in BINARY mode so Windows' default
    text-mode CRLF translation doesn't skew our byte-offset arithmetic.
    pathlib's Path.write_text(...) on Windows still does \\n -> \\r\\n
    expansion unless newline='' is passed explicitly."""

    def test_keeps_first_line_when_seek_lands_on_newline(self, tmp_path):
        """Construct a file where the requested tail_bytes EXACTLY contains
        a complete trailing record (seek lands on '\\n' boundary)."""
        from portfolio.file_utils import load_jsonl_tail

        entries = [json.dumps({"i": i}) for i in range(10)]
        full = ("\n".join(entries) + "\n").encode("utf-8")
        f = tmp_path / "boundary.jsonl"
        f.write_bytes(full)

        # tail_bytes that lands exactly on a newline before entry index 7
        # (i.e., reads entries 7,8,9 — three intact entries).
        prefix = ("\n".join(entries[:7]) + "\n").encode("utf-8")
        target_tail_bytes = len(full) - len(prefix)

        rows = load_jsonl_tail(f, max_entries=10, tail_bytes=target_tail_bytes)
        # Must return entries 7, 8, 9 (3 entries) — NOT 8, 9 (which would be
        # the bug where the first intact line was dropped).
        assert [e["i"] for e in rows] == [7, 8, 9], (
            f"got {[e['i'] for e in rows]} — first intact line dropped on boundary"
        )

    def test_drops_first_line_when_seek_lands_mid_line(self, tmp_path):
        """When seek lands inside a record (not on \\n), the first
        decoded chunk is truncated and must be dropped."""
        from portfolio.file_utils import load_jsonl_tail

        entries = [json.dumps({"i": i, "pad": "x" * 200}) for i in range(10)]
        full = ("\n".join(entries) + "\n").encode("utf-8")
        f = tmp_path / "midline.jsonl"
        f.write_bytes(full)

        # tail_bytes that lands inside entry 5 (mid-line).
        prefix_to_entry5 = ("\n".join(entries[:5]) + "\n").encode("utf-8")
        target_tail_bytes = len(full) - len(prefix_to_entry5) - 50

        rows = load_jsonl_tail(f, max_entries=10, tail_bytes=target_tail_bytes)
        # First parsed line should be entry 6 (entry 5 was truncated).
        assert rows[0]["i"] == 6
