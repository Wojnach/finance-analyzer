"""Tests for dashboard.trading_status — per-bot trading-state reader.

Each test writes mock state files to tmp_path and pins ``now_utc`` so
the Avanza-warrant session check (15:30–21:55 Europe/Stockholm) is
deterministic regardless of when the suite runs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from dashboard import trading_status as ts

UTC = timezone.utc
STO = ZoneInfo("Europe/Stockholm")


def _w(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")


def _at(local_hh: int, local_mm: int = 0) -> datetime:
    """Return a UTC datetime that maps to the given local-clock time
    in Europe/Stockholm on a fixed weekday inside DST (2026-05-04)."""
    local = datetime(2026, 5, 4, local_hh, local_mm, tzinfo=STO)
    return local.astimezone(UTC)


# ---------------------------------------------------------------------------
# Session window
# ---------------------------------------------------------------------------


class TestSessionWindow:
    def test_open_at_1530(self):
        assert ts._in_session(_at(15, 30)) is True

    def test_open_at_2154(self):
        assert ts._in_session(_at(21, 54)) is True

    def test_closed_at_2155(self):
        assert ts._in_session(_at(21, 55)) is False

    def test_closed_morning(self):
        assert ts._in_session(_at(9, 0)) is False

    def test_weekend_closed(self):
        """Codex P1 finding 2026-05-04: Saturday/Sunday at 16:00 must
        report as session_closed even though the wall-clock time is
        inside the weekday window."""
        # 2026-05-02 is a Saturday; 2026-05-03 is a Sunday.
        sat_local = datetime(2026, 5, 2, 16, 0, tzinfo=STO).astimezone(UTC)
        sun_local = datetime(2026, 5, 3, 16, 0, tzinfo=STO).astimezone(UTC)
        assert ts._in_session(sat_local) is False
        assert ts._in_session(sun_local) is False

    def test_friday_open(self):
        """And Friday is still open."""
        fri_local = datetime(2026, 5, 1, 16, 0, tzinfo=STO).astimezone(UTC)
        assert ts._in_session(fri_local) is True

    def test_weekend_hint_rolls_to_monday(self):
        """`_next_open_hint` on Saturday afternoon must point at the
        next *weekday* open (Monday), not Sunday."""
        sat_local = datetime(2026, 5, 2, 16, 0, tzinfo=STO).astimezone(UTC)
        hint = ts._next_open_hint(sat_local)
        # Saturday 16:00 -> Monday 15:30 = ~47h 30m away.
        assert "h" in hint
        # Must NOT be near 24h (which would mean it pointed at Sunday).
        # Acceptable range: 40h to 50h.
        import re
        m = re.search(r"(\d+)h", hint)
        assert m, hint
        hours = int(m.group(1))
        assert 40 <= hours <= 50, f"hint pointed at wrong day: {hint}"


# ---------------------------------------------------------------------------
# GoldDigger
# ---------------------------------------------------------------------------


class TestGoldDigger:
    def test_halted_surfaces_reason(self, tmp_path: Path):
        _w(tmp_path / "golddigger_state.json", {
            "halted": True, "halted_reason": "daily loss limit -1.5%",
            "position": None,
        })
        out = ts.compute(data_dir=tmp_path, now_utc=_at(16, 0))
        gd = next(b for b in out["bots"] if b["bot"] == "golddigger")
        assert gd["state"] == "HALTED"
        assert "daily loss limit" in gd["reason"]

    def test_position_held(self, tmp_path: Path):
        _w(tmp_path / "golddigger_state.json", {
            "halted": False, "halted_reason": "",
            "position": {"qty": 100, "entry": 1234.5},
        })
        out = ts.compute(data_dir=tmp_path, now_utc=_at(16, 0))
        gd = next(b for b in out["bots"] if b["bot"] == "golddigger")
        assert gd["state"] == "TRADING"

    def test_outside_hours_when_idle(self, tmp_path: Path):
        _w(tmp_path / "golddigger_state.json", {
            "halted": False, "halted_reason": "", "position": None,
        })
        out = ts.compute(data_dir=tmp_path, now_utc=_at(9, 0))
        gd = next(b for b in out["bots"] if b["bot"] == "golddigger")
        assert gd["state"] == "OUTSIDE_HOURS"
        # DST-aware: CEST in summer (the test's 2026-05-04 fixture date)
        # and CET in winter. The code now uses target.tzname() so both
        # are valid suffixes.
        assert "next 15:30" in gd["reason"]
        assert ("CEST" in gd["reason"]) or ("CET" in gd["reason"])

    def test_scanning_in_session(self, tmp_path: Path):
        _w(tmp_path / "golddigger_state.json", {
            "halted": False, "halted_reason": "", "position": None,
        })
        out = ts.compute(data_dir=tmp_path, now_utc=_at(16, 0))
        gd = next(b for b in out["bots"] if b["bot"] == "golddigger")
        assert gd["state"] == "SCANNING"

    def test_unknown_when_state_missing(self, tmp_path: Path):
        out = ts.compute(data_dir=tmp_path, now_utc=_at(16, 0))
        gd = next(b for b in out["bots"] if b["bot"] == "golddigger")
        assert gd["state"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# Elongir
# ---------------------------------------------------------------------------


class TestElongir:
    def test_halted_with_reason(self, tmp_path: Path):
        _w(tmp_path / "elongir_state.json", {
            "halted": True, "halted_reason": "max daily trades reached",
            "position": None,
        })
        out = ts.compute(data_dir=tmp_path, now_utc=_at(16, 0))
        el = next(b for b in out["bots"] if b["bot"] == "elongir")
        assert el["state"] == "HALTED"
        assert "max daily trades" in el["reason"]


# ---------------------------------------------------------------------------
# Metals swing
# ---------------------------------------------------------------------------


class TestMetals:
    def test_position_held(self, tmp_path: Path):
        _w(tmp_path / "metals_swing_state.json", {
            "positions": [{"ticker": "MINI L SILVER", "qty": 50}],
        })
        out = ts.compute(data_dir=tmp_path, now_utc=_at(16, 0))
        m = next(b for b in out["bots"] if b["bot"] == "metals")
        assert m["state"] == "TRADING"

    def test_consecutive_losses_caution(self, tmp_path: Path):
        _w(tmp_path / "metals_swing_state.json", {
            "positions": [], "consecutive_losses": 4, "last_buy_ts": None,
        })
        _w(tmp_path / "metals_guard_state.json", {"consecutive_losses": 4})
        out = ts.compute(data_dir=tmp_path, now_utc=_at(16, 0))
        m = next(b for b in out["bots"] if b["bot"] == "metals")
        assert m["state"] == "SCANNING"
        assert "caution" in m["reason"]


# ---------------------------------------------------------------------------
# Fishing engine
# ---------------------------------------------------------------------------


class TestFishing:
    def test_cooldown_active(self, tmp_path: Path):
        # last_trade 100s ago, cooldown 300s → 200s remaining
        now = _at(16, 0)
        _w(tmp_path / "fish_engine_state.json", {
            "position": None,
            "last_trade_ts": now.timestamp() - 100,
            "cooldown_seconds": 300,
            "consecutive_losses": 12,
            "mode": "straddle",
        })
        out = ts.compute(data_dir=tmp_path, now_utc=now)
        f = next(b for b in out["bots"] if b["bot"] == "fishing")
        assert f["state"] == "COOLDOWN"
        assert "12 losses" in f["reason"]
        # Allow ±2s drift between timestamp arithmetic and clock read.
        assert "199" in f["reason"] or "200" in f["reason"]

    def test_cooldown_expired_scans(self, tmp_path: Path):
        now = _at(16, 0)
        _w(tmp_path / "fish_engine_state.json", {
            "position": None,
            "last_trade_ts": now.timestamp() - 1000,
            "cooldown_seconds": 300,
            "mode": "straddle",
        })
        out = ts.compute(data_dir=tmp_path, now_utc=now)
        f = next(b for b in out["bots"] if b["bot"] == "fishing")
        assert f["state"] == "SCANNING"


# ---------------------------------------------------------------------------
# Payload shape
# ---------------------------------------------------------------------------


class TestShape:
    def test_full_payload(self, tmp_path: Path):
        out = ts.compute(data_dir=tmp_path, now_utc=_at(16, 0))
        assert "ts" in out
        assert "session_open" in out
        assert out["session_open"] is True
        bot_keys = {b["bot"] for b in out["bots"]}
        assert bot_keys == {"golddigger", "elongir", "metals", "fishing"}

    def test_session_open_flag_outside_hours(self, tmp_path: Path):
        out = ts.compute(data_dir=tmp_path, now_utc=_at(9, 0))
        assert out["session_open"] is False
