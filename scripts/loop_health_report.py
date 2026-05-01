"""Two-week paper-mode health report — sends a Telegram summary.

Runs the oil + MSTR scorecards, reads the crypto/oil heartbeat files, and
builds a single sub-300-word Telegram message. If a loop has no heartbeat
AND no shadow/decision events, that's flagged as "scheduled task likely
never registered/started".

Designed for the one-shot scheduled task `PF-LoopHealthReport-20260515`
(2 weeks after the 2026-05-01 midfinance merge). Re-usable — register it
again with a new date if the loops haven't been started by 2026-05-15.

Usage:
    .venv/Scripts/python.exe scripts/loop_health_report.py
"""
from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"

HEARTBEAT_STALE_SECONDS = 300       # >5min old = stale
HEARTBEAT_MISSING = "<missing>"


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _heartbeat_status(path: Path) -> dict:
    """Return {age_seconds, is_fresh, is_missing, payload}."""
    if not path.exists():
        return {"age_seconds": None, "is_fresh": False, "is_missing": True,
                "payload": None}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"age_seconds": None, "is_fresh": False, "is_missing": True,
                "payload": None}
    ts_str = payload.get("ts")
    age_seconds = None
    is_fresh = False
    if ts_str:
        try:
            ts = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=datetime.UTC)
            age_seconds = (_now_utc() - ts).total_seconds()
            is_fresh = age_seconds < HEARTBEAT_STALE_SECONDS
        except ValueError:
            pass
    return {"age_seconds": age_seconds, "is_fresh": is_fresh,
            "is_missing": False, "payload": payload}


def _run_scorecard(script: Path) -> dict | None:
    """Run a scorecard script and parse its JSON output file."""
    if not script.exists():
        return {"error": f"missing script {script.name}"}
    try:
        result = subprocess.run(
            [str(PYTHON), str(script)],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"run failed: {exc}"}
    # Each scorecard writes a sibling json next to its decision logs
    candidates = {
        "mstr_loop_scorecard.py": DATA / "mstr_loop_scorecard.json",
        "oil_loop_scorecard.py": DATA / "oil_loop_scorecard.json",
    }
    out_file = candidates.get(script.name)
    if out_file and out_file.exists():
        try:
            return json.loads(out_file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return {"error": f"parse failed: {exc}",
                    "stdout_tail": result.stdout[-200:]}
    return {"error": "no output json", "stdout_tail": result.stdout[-200:]}


def _format_heartbeat(label: str, hb: dict) -> str:
    if hb["is_missing"]:
        return f"{label}: ❌ NO HEARTBEAT — task likely never started"
    age = hb["age_seconds"]
    fresh = "✅" if hb["is_fresh"] else "⚠️ stale"
    cycle = (hb.get("payload") or {}).get("cycle", "?")
    return f"{label}: {fresh} age={age:.0f}s cycle={cycle}"


def _format_oil_scorecard(card: dict) -> list[str]:
    lines = []
    if card is None:
        return ["oil scorecard: no data"]
    if "error" in card:
        return [f"oil scorecard: {card['error']}"]
    win = card.get("observation_window") or {}
    lines.append(f"oil window: {win.get('days_observed', 0)}/{win.get('min_days', 30)}d")
    insts = card.get("instruments") or {}
    if not insts:
        lines.append("oil: 0 closed trades — DRY_RUN paper mode")
        return lines
    for ticker, stats in insts.items():
        n = stats.get("n_trades", 0)
        wr = stats.get("win_rate_pct", 0)
        unit = stats.get("pnl_unit", "")
        ex = stats.get(f"expectancy_per_trade_{unit}", 0)
        status = stats.get("live_flip_status", "?")
        lines.append(f"  {ticker}: n={n} wr={wr}% expectancy={ex} → {status}")
        gates = stats.get("live_flip_gates", {})
        failing = [g for g, ok in gates.items() if not ok]
        if failing:
            lines.append(f"    failing gates: {failing}")
    return lines


def _format_mstr_scorecard(card: dict) -> list[str]:
    lines = []
    if card is None:
        return ["mstr scorecard: no data"]
    if "error" in card:
        return [f"mstr scorecard: {card['error']}"]
    win = card.get("phase_a_window") or {}
    lines.append(f"mstr Phase-A window: {win.get('days_observed', 0)}/"
                  f"{win.get('min_days', 90)}d")
    strategies = card.get("strategies") or {}
    if not strategies:
        lines.append("mstr: 0 closed shadow trades")
        return lines
    for key, stats in strategies.items():
        n = stats.get("n_trades", 0)
        wr = stats.get("win_rate_pct", 0)
        ex = stats.get("expectancy_sek_per_trade", 0)
        gates = stats.get("graduation_gates") or {}
        failing = [g for g, ok in gates.items() if not ok]
        gate_str = "PASS" if not failing and gates else f"failing {failing}"
        lines.append(f"  {key}: n={n} wr={wr}% ex={ex} → {gate_str}")
    return lines


def build_report() -> str:
    crypto_hb = _heartbeat_status(DATA / "crypto_loop.heartbeat")
    oil_hb = _heartbeat_status(DATA / "oil_loop.heartbeat")

    # MSTR has no heartbeat — proxy via the poll log if it exists
    mstr_poll = DATA / "mstr_loop_poll.jsonl"
    mstr_hb_line = (f"mstr poll log: ✅ exists ({mstr_poll.stat().st_size} bytes)"
                     if mstr_poll.exists()
                     else "mstr poll log: ❌ MISSING — task likely never started")

    oil_card = _run_scorecard(REPO / "scripts" / "oil_loop_scorecard.py")
    mstr_card = _run_scorecard(REPO / "scripts" / "mstr_loop_scorecard.py")

    lines = [
        "🔬 Loop Paper-Mode Health Check (T+2w)",
        "",
        _format_heartbeat("crypto", crypto_hb),
        _format_heartbeat("oil", oil_hb),
        mstr_hb_line,
        "",
        *_format_oil_scorecard(oil_card),
        "",
        *_format_mstr_scorecard(mstr_card),
    ]

    # Action hint
    any_missing = (crypto_hb["is_missing"] or oil_hb["is_missing"]
                   or not mstr_poll.exists())
    if any_missing:
        lines.append("")
        lines.append("⚠️ One or more loops show no signs of life. Check:")
        lines.append("  schtasks /query /tn PF-{Crypto,Oil,Mstr}Loop")
        lines.append("  If absent, run scripts/win/install-*-loop-task.ps1")
        lines.append("  Then Start-ScheduledTask -TaskName 'PF-...Loop'")
    return "\n".join(lines)


def send(report: str) -> None:
    """Try Telegram; fall back to stdout."""
    try:
        sys.path.insert(0, str(REPO))
        from portfolio.file_utils import load_json
        from portfolio.telegram_notifications import send_telegram
        cfg = load_json("config.json") or {}
        if cfg.get("telegram", {}).get("token"):
            send_telegram(report, cfg)
            print("Sent to Telegram.")
            return
        print("(no telegram token in config — printing instead)")
    except Exception as exc:  # noqa: BLE001
        print(f"(telegram wiring failed: {exc} — printing instead)")
    print(report)


if __name__ == "__main__":
    report = build_report()
    send(report)
