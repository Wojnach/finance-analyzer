"""Shared utilities for the metals monitoring system.

Canonical implementations of functions duplicated across metals_loop.py,
metals_swing_trader.py, and other metals modules. Import from here to avoid
copy-paste drift.

Created as part of Batch 3 (ARCH-12: deduplicate shared functions).
"""

import datetime
import json
import os

import requests


def _load_config():
    """Load config.json from the project root."""
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    if not os.path.exists(cfg_path):
        cfg_path = "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


_config = None


def _get_config():
    global _config
    if _config is None:
        try:
            _config = _load_config()
        except Exception:
            _config = {}
    return _config


def send_telegram(msg):
    """Send a Telegram message using config.json credentials."""
    try:
        cfg = _get_config()
        token = cfg.get("telegram", {}).get("token", "")
        chat_id = cfg.get("telegram", {}).get("chat_id", "")
        if not token or not chat_id:
            print("[WARN] Telegram not configured", flush=True)
            return
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        print(f"[TG ERROR] {e}", flush=True)


def get_cet_time():
    """Get current CET/CEST time from timeapi.io, fallback to zoneinfo (DST-safe).

    Returns:
        tuple: (hour_float, time_string, source)
    """
    try:
        r = requests.get(
            "http://timeapi.io/api/time/current/zone?timeZone=Europe/Stockholm",
            timeout=3,
        )
        if r.status_code == 200:
            data = r.json()
            h = data["hour"]
            m = data["minute"]
            return h + m / 60, f"{h:02d}:{m:02d} CET", "timeapi"
    except Exception as e:
        print(f"[WARN] timeapi.io failed: {e}", flush=True)
    # Fallback: zoneinfo handles DST correctly (CET/CEST)
    try:
        from zoneinfo import ZoneInfo
        now = datetime.datetime.now(ZoneInfo("Europe/Stockholm"))
        h = now.hour
        m = now.minute
        return h + m / 60, f"{h:02d}:{m:02d} CET", "zoneinfo"
    except ImportError:
        # Last resort: UTC+1 (wrong during summer DST)
        now = datetime.datetime.now(datetime.timezone.utc)
        h = (now.hour + 1) % 24
        m = now.minute
        return h + m / 60, f"{h:02d}:{m:02d} CET", "system_utc+1"


def cet_hour():
    """Get current CET hour as float (e.g. 14.5 = 14:30)."""
    h, _, _ = get_cet_time()
    return h


def cet_time_str():
    """Get current CET time as 'HH:MM CET' string."""
    _, ts, _ = get_cet_time()
    return ts


def is_market_hours():
    """Check if Avanza commodity warrant market is open (Mon-Fri 08:15-21:55 CET)."""
    now = datetime.datetime.now(datetime.timezone.utc)
    weekday = now.weekday()
    h = cet_hour()
    return weekday < 5 and 8.25 <= h <= 21.92


def log(msg):
    """Print a timestamped log message."""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def pnl_pct(current, entry):
    """Calculate P&L percentage."""
    if entry == 0:
        return 0
    return ((current - entry) / entry) * 100
