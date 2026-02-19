#!/usr/bin/env python3
"""iskbet — CLI to activate/deactivate ISKBETS intraday mode.

Usage:
    python scripts/iskbet.py btc 8h             # BTC-USD, 8h, default 100K SEK
    python scripts/iskbet.py mstr pltr 4h        # multiple tickers
    python scripts/iskbet.py btc 8h 50000        # custom amount
    python scripts/iskbet.py off                  # disable
    python scripts/iskbet.py status               # show state
"""

import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIG_FILE = BASE_DIR / "config.json"
ISKBETS_CONFIG = DATA_DIR / "iskbets_config.json"
ISKBETS_STATE = DATA_DIR / "iskbets_state.json"
SUMMARY_FILE = DATA_DIR / "agent_summary.json"

TICKER_ALIASES = {
    "btc": "BTC-USD",
    "btcusd": "BTC-USD",
    "btc-usd": "BTC-USD",
    "eth": "ETH-USD",
    "ethusd": "ETH-USD",
    "eth-usd": "ETH-USD",
    "mstr": "MSTR",
    "pltr": "PLTR",
    "nvda": "NVDA",
}

VALID_TICKERS = {"BTC-USD", "ETH-USD", "MSTR", "PLTR", "NVDA"}

DURATION_RE = re.compile(r"^(\d+)([hmd])$", re.IGNORECASE)

DEFAULT_AMOUNT = 100_000


def normalize_ticker(raw):
    return TICKER_ALIASES.get(raw.lower().replace(" ", ""), raw.upper())


def parse_duration(s):
    """Parse '8h', '2d', '30m' into timedelta."""
    m = DURATION_RE.match(s)
    if not m:
        return None
    val, unit = int(m.group(1)), m.group(2).lower()
    if unit == "h":
        return timedelta(hours=val)
    elif unit == "d":
        return timedelta(days=val)
    elif unit == "m":
        return timedelta(minutes=val)
    return None


def atomic_write_json(path, data):
    path.parent.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def send_telegram(msg):
    try:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        token = config["telegram"]["token"]
        chat_id = config["telegram"]["chat_id"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        print("Warning: Telegram not configured")
        return False
    try:
        import requests

        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
            timeout=30,
        )
        return r.ok
    except Exception as e:
        print(f"Telegram error: {e}")
        return False


def file_age_minutes(path):
    try:
        return (time.time() - path.stat().st_mtime) / 60
    except OSError:
        return 999


def cmd_status():
    if not ISKBETS_CONFIG.exists():
        print("ISKBETS: not active")
        return

    try:
        cfg = json.loads(ISKBETS_CONFIG.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError):
        print("ISKBETS: not active (config unreadable)")
        return

    if not cfg.get("enabled", False):
        print("ISKBETS: disabled")
        return

    tickers = cfg.get("tickers", [])
    amount = cfg.get("amount_sek", DEFAULT_AMOUNT)
    expiry = cfg.get("expiry", "?")

    # Parse remaining time
    remaining = ""
    try:
        exp_dt = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
        delta = exp_dt - datetime.now(timezone.utc)
        if delta.total_seconds() <= 0:
            remaining = " (EXPIRED)"
        else:
            hours = int(delta.total_seconds() // 3600)
            mins = int((delta.total_seconds() % 3600) // 60)
            remaining = f" ({hours}h {mins}m remaining)"
    except (ValueError, TypeError, AttributeError):
        pass

    print(f"ISKBETS: ACTIVE{remaining}")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Amount:  {amount:,.0f} SEK")
    print(f"  Expiry:  {expiry}")

    # Check for active position
    if ISKBETS_STATE.exists():
        try:
            state = json.loads(ISKBETS_STATE.read_text(encoding="utf-8"))
            pos = state.get("active_position")
            if pos:
                ticker = pos["ticker"]
                entry = pos["entry_price_usd"]
                stop = pos.get("stop_loss", 0)
                be = pos.get("stop_at_breakeven", False)
                print(f"\n  Position: {ticker} @ ${entry:,.2f}")
                print(f"  Stop:     ${stop:,.2f} {'(breakeven)' if be else '(hard)'}")
            history = state.get("trade_history", [])
            if history:
                print(f"  Trades:   {len(history)} completed")
        except (json.JSONDecodeError, KeyError):
            pass


def cmd_off():
    if ISKBETS_CONFIG.exists():
        try:
            cfg = json.loads(ISKBETS_CONFIG.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            cfg = {}
        cfg["enabled"] = False
        atomic_write_json(ISKBETS_CONFIG, cfg)
        print("ISKBETS: disabled")
        send_telegram("*ISKBETS disabled* via CLI")
    else:
        print("ISKBETS: already not active")


def cmd_activate(tickers, duration, amount):
    now = datetime.now(timezone.utc)
    expiry = now + duration

    # Warn if replacing active session
    if ISKBETS_CONFIG.exists():
        try:
            old = json.loads(ISKBETS_CONFIG.read_text(encoding="utf-8"))
            if old.get("enabled", False):
                old_exp = old.get("expiry", "")
                try:
                    old_dt = datetime.fromisoformat(old_exp.replace("Z", "+00:00"))
                    if old_dt > now:
                        print(f"WARNING: Replacing active session (was scanning {', '.join(old.get('tickers', []))})")
                except (ValueError, TypeError):
                    pass
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    cfg = {
        "enabled": True,
        "tickers": tickers,
        "amount_sek": amount,
        "expiry": expiry.isoformat(),
    }
    atomic_write_json(ISKBETS_CONFIG, cfg)

    # Format duration for display
    total_secs = int(duration.total_seconds())
    if total_secs >= 86400:
        dur_str = f"{total_secs // 86400}d"
    elif total_secs >= 3600:
        dur_str = f"{total_secs // 3600}h"
    else:
        dur_str = f"{total_secs // 60}m"

    ticker_str = ", ".join(tickers)
    print(f"ISKBETS: ACTIVATED")
    print(f"  Tickers:  {ticker_str}")
    print(f"  Amount:   {amount:,.0f} SEK")
    print(f"  Duration: {dur_str}")
    print(f"  Expiry:   {expiry.strftime('%Y-%m-%d %H:%M UTC')}")

    # Warn if loop may be down
    age = file_age_minutes(SUMMARY_FILE)
    if age > 5:
        print(f"\n  WARNING: agent_summary.json is {age:.0f}m old — loop may not be running!")

    # Send Telegram notification
    msg = f"*ISKBETS active*: scanning {ticker_str} for {dur_str} ({amount:,.0f} SEK)"
    if send_telegram(msg):
        print("  Telegram sent")
    else:
        print("  Telegram send failed")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__.strip())
        return

    # Handle subcommands
    if args[0].lower() == "status":
        cmd_status()
        return
    if args[0].lower() == "off":
        cmd_off()
        return

    # Parse: tickers, duration, optional amount
    tickers = []
    duration = None
    amount = DEFAULT_AMOUNT

    for arg in args:
        # Check if it's a duration
        d = parse_duration(arg)
        if d is not None:
            duration = d
            continue

        # Check if it's a number (amount)
        try:
            val = float(arg)
            amount = int(val)
            continue
        except ValueError:
            pass

        # Must be a ticker
        ticker = normalize_ticker(arg)
        if ticker not in VALID_TICKERS:
            print(f"Unknown ticker: {arg} (valid: btc, eth, mstr, pltr, nvda)")
            sys.exit(1)
        if ticker not in tickers:
            tickers.append(ticker)

    if not tickers:
        print("Error: no tickers specified")
        print("Usage: iskbet btc 8h [amount]")
        sys.exit(1)

    if duration is None:
        print("Error: no duration specified (e.g. 8h, 4h, 2d, 30m)")
        sys.exit(1)

    cmd_activate(tickers, duration, amount)


if __name__ == "__main__":
    main()
