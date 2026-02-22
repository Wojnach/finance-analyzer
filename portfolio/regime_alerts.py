"""Alert when market regime transitions.

Detects changes from one regime to another and sends Telegram notification.
Tracks regime history for pattern detection.
"""

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from portfolio.file_utils import load_json, load_jsonl

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIG_FILE = BASE_DIR / "config.json"
REGIME_HISTORY_FILE = DATA_DIR / "regime_history.jsonl"

VALID_REGIMES = {
    "trending-up",
    "trending-down",
    "range-bound",
    "ranging",       # alias used in journal
    "high-vol",
    "breakout",
    "capitulation",
}


def _load_json(path):
    return load_json(path, default={})


def _load_jsonl(path):
    return load_jsonl(path)


def _get_last_regime(ticker):
    """Get the most recent regime for a ticker from history.

    Args:
        ticker: Ticker symbol.

    Returns:
        str or None: Last known regime, or None if no history.
    """
    entries = _load_jsonl(REGIME_HISTORY_FILE)
    for entry in reversed(entries):
        if entry.get("ticker") == ticker:
            return entry.get("new_regime")
    return None


def check_regime_transition(current_regime, ticker):
    """Check if the market regime has changed for a ticker.

    Compares current_regime against the last known regime for this ticker.

    Args:
        current_regime: The current detected regime string.
        ticker: Ticker symbol.

    Returns:
        dict or None: Transition info if regime changed, None otherwise.
            Keys: ticker, old_regime, new_regime, timestamp.
    """
    old_regime = _get_last_regime(ticker)

    # Normalize: treat "ranging" and "range-bound" as equivalent
    def _normalize(r):
        if r == "ranging":
            return "range-bound"
        return r

    if old_regime is None:
        # First time seeing this ticker — log it but don't alert
        return None

    if _normalize(old_regime) == _normalize(current_regime):
        return None

    return {
        "ticker": ticker,
        "old_regime": old_regime,
        "new_regime": current_regime,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def log_regime_change(ticker, old_regime, new_regime):
    """Append a regime transition to the regime history file.

    Args:
        ticker: Ticker symbol.
        old_regime: Previous regime.
        new_regime: New regime.
    """
    DATA_DIR.mkdir(exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "old_regime": old_regime,
        "new_regime": new_regime,
    }
    with open(REGIME_HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_regime_distribution(ticker, days=7):
    """Compute regime distribution for a ticker over a time window.

    Args:
        ticker: Ticker symbol.
        days: Number of days to look back.

    Returns:
        dict: Mapping regime -> percentage of transitions in window.
              Empty dict if no data.
    """
    entries = _load_jsonl(REGIME_HISTORY_FILE)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    regime_list = []
    for entry in entries:
        if entry.get("ticker") != ticker:
            continue
        ts_str = entry.get("ts", "")
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts >= cutoff:
                regime_list.append(entry.get("new_regime", "unknown"))
        except (ValueError, TypeError):
            continue

    if not regime_list:
        return {}

    counter = Counter(regime_list)
    total = len(regime_list)
    return {regime: round(count / total * 100, 1) for regime, count in counter.most_common()}


def get_regime_history(ticker, limit=20):
    """Get recent regime transitions for a ticker.

    Args:
        ticker: Ticker symbol.
        limit: Maximum entries to return.

    Returns:
        list[dict]: Recent regime change entries, most recent last.
    """
    entries = _load_jsonl(REGIME_HISTORY_FILE)
    ticker_entries = [e for e in entries if e.get("ticker") == ticker]
    return ticker_entries[-limit:]


def send_regime_alert(ticker, old_regime, new_regime):
    """Send a Telegram alert about a regime transition.

    Also logs the message to telegram_messages.jsonl.

    Args:
        ticker: Ticker symbol.
        old_regime: Previous regime.
        new_regime: New regime.

    Returns:
        requests.Response or None.
    """
    now = datetime.now(timezone.utc)
    msg = (
        f"*REGIME SHIFT: {ticker}*\n"
        f"`{old_regime}` -> `{new_regime}`\n"
        f"_{now.strftime('%H:%M UTC %b %d')}_"
    )

    # Get distribution for context
    dist = get_regime_distribution(ticker, days=7)
    if dist:
        msg += "\n\n_7d regime breakdown:_"
        for regime, pct in dist.items():
            msg += f"\n  `{regime:<16} {pct:>5.1f}%`"

    config = _load_json(CONFIG_FILE)
    token = config.get("telegram", {}).get("token")
    chat_id = config.get("telegram", {}).get("chat_id")

    if not token or not chat_id:
        print(f"REGIME ALERT (no Telegram config): {ticker} {old_regime} -> {new_regime}")
        return None

    # Save locally
    log_file = DATA_DIR / "telegram_messages.jsonl"
    log_entry = {
        "ts": now.isoformat(),
        "text": msg,
        "type": "regime_alert",
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    # Send via shared module
    try:
        from portfolio.telegram_notifications import send_telegram
        result = send_telegram(msg, config)
        return result
    except Exception as e:
        print(f"ERROR sending regime alert: {e}")
        return None


def check_and_alert(current_regime, ticker):
    """Convenience function: check for transition, log it, and alert if changed.

    Intended to be called from the main loop or Layer 2 on each invocation.

    Args:
        current_regime: Current detected regime.
        ticker: Ticker symbol.

    Returns:
        dict or None: Transition info if regime changed, None otherwise.
    """
    transition = check_regime_transition(current_regime, ticker)

    # Always log current regime (even if no change) for first-time tickers
    old = _get_last_regime(ticker)
    if old is None:
        log_regime_change(ticker, "unknown", current_regime)
        return None

    if transition is None:
        return None

    old_regime = transition["old_regime"]
    new_regime = transition["new_regime"]

    log_regime_change(ticker, old_regime, new_regime)
    send_regime_alert(ticker, old_regime, new_regime)

    return transition
