"""Track signal voting history for flip-flop detection and persistence scoring.

Maintains a rolling window of the last N votes per signal per ticker.
Signals that maintain direction for 3+ consecutive checks are "persistent".
Signals that flip every 1-2 checks are "noisy".
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
HISTORY_FILE = DATA_DIR / "signal_history.jsonl"

MAX_ENTRIES_PER_TICKER = 50

SIGNAL_NAMES = [
    "rsi", "macd", "ema", "bb", "fear_greed", "sentiment",
    "ministral", "ml", "funding", "volume", "custom_lora",
    "trend", "momentum", "volume_flow", "volatility_sig",
    "candlestick", "structure", "fibonacci", "smart_money",
    "oscillators", "heikin_ashi", "mean_reversion",
    "calendar", "macro_regime", "momentum_factors",
]


def _load_history():
    """Load all history entries from JSONL file.

    Returns:
        list[dict]: All history entries.
    """
    if not HISTORY_FILE.exists():
        return []
    entries = []
    for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _save_history(entries):
    """Write all history entries back to JSONL file.

    Args:
        entries: List of history entry dicts.
    """
    DATA_DIR.mkdir(exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _entries_for_ticker(entries, ticker):
    """Filter entries for a specific ticker.

    Args:
        entries: All history entries.
        ticker: Ticker symbol.

    Returns:
        list[dict]: Entries for this ticker, ordered by time.
    """
    return [e for e in entries if e.get("ticker") == ticker]


def update_history(ticker, votes_dict):
    """Append current signal votes to history file for a ticker.

    Trims to keep only the last MAX_ENTRIES_PER_TICKER entries per ticker.

    Args:
        ticker: Ticker symbol (e.g. "BTC-USD").
        votes_dict: Dict mapping signal_name -> vote ("BUY"/"SELL"/"HOLD").
    """
    entries = _load_history()

    new_entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "votes": {sig: votes_dict.get(sig, "HOLD") for sig in SIGNAL_NAMES},
    }
    entries.append(new_entry)

    # Trim: keep only last MAX_ENTRIES_PER_TICKER per ticker
    by_ticker = defaultdict(list)
    for e in entries:
        by_ticker[e.get("ticker", "unknown")].append(e)

    trimmed = []
    for t, t_entries in by_ticker.items():
        trimmed.extend(t_entries[-MAX_ENTRIES_PER_TICKER:])

    # Sort by timestamp for stable ordering
    trimmed.sort(key=lambda e: e.get("ts", ""))
    _save_history(trimmed)


def get_persistence_scores(ticker):
    """Compute persistence score for each signal for a ticker.

    Persistence score measures how often a signal maintains the same direction
    across consecutive checks. Score of 1.0 means perfectly persistent (never
    flips), 0.0 means flips every single check.

    The score is computed as: 1 - (flip_count / (N - 1)) where N is the number
    of non-HOLD votes and flip_count is the number of direction changes.
    Only considers non-HOLD votes (BUY/SELL) since HOLD is absence of signal.

    Args:
        ticker: Ticker symbol.

    Returns:
        dict: Mapping signal_name -> persistence score (0.0 to 1.0).
              Signals with fewer than 3 non-HOLD votes return 0.5 (neutral).
    """
    entries = _entries_for_ticker(_load_history(), ticker)
    if not entries:
        return {sig: 0.5 for sig in SIGNAL_NAMES}

    scores = {}
    for sig in SIGNAL_NAMES:
        # Extract non-HOLD votes in order
        active_votes = []
        for e in entries:
            vote = e.get("votes", {}).get(sig, "HOLD")
            if vote in ("BUY", "SELL"):
                active_votes.append(vote)

        if len(active_votes) < 3:
            scores[sig] = 0.5  # insufficient data
            continue

        # Count direction flips
        flips = 0
        for i in range(1, len(active_votes)):
            if active_votes[i] != active_votes[i - 1]:
                flips += 1

        max_flips = len(active_votes) - 1
        scores[sig] = round(1.0 - (flips / max_flips), 4) if max_flips > 0 else 1.0

    return scores


def get_noisy_signals(ticker, threshold=0.3):
    """Get list of signals that flip too often for a ticker.

    Args:
        ticker: Ticker symbol.
        threshold: Persistence score below this is considered noisy.

    Returns:
        list[str]: Signal names with persistence below threshold.
    """
    scores = get_persistence_scores(ticker)
    return [sig for sig, score in scores.items() if score < threshold]


def get_signal_streaks(ticker):
    """Get current voting streak for each signal for a ticker.

    A streak is the number of consecutive checks a signal has maintained the
    same vote (including HOLD). Useful for detecting sustained signals.

    Args:
        ticker: Ticker symbol.

    Returns:
        dict: Mapping signal_name -> {"current_vote": str, "streak_count": int}.
    """
    entries = _entries_for_ticker(_load_history(), ticker)
    if not entries:
        return {sig: {"current_vote": "HOLD", "streak_count": 0} for sig in SIGNAL_NAMES}

    streaks = {}
    for sig in SIGNAL_NAMES:
        votes = [e.get("votes", {}).get(sig, "HOLD") for e in entries]
        if not votes:
            streaks[sig] = {"current_vote": "HOLD", "streak_count": 0}
            continue

        current = votes[-1]
        count = 0
        for v in reversed(votes):
            if v == current:
                count += 1
            else:
                break

        streaks[sig] = {"current_vote": current, "streak_count": count}

    return streaks


def get_summary(ticker):
    """Get a combined summary of persistence, noise, and streaks for a ticker.

    Convenience function that calls all three analysis functions.

    Args:
        ticker: Ticker symbol.

    Returns:
        dict with keys: persistence_scores, noisy_signals, streaks, entries_count.
    """
    entries = _entries_for_ticker(_load_history(), ticker)
    return {
        "entries_count": len(entries),
        "persistence_scores": get_persistence_scores(ticker),
        "noisy_signals": get_noisy_signals(ticker),
        "streaks": get_signal_streaks(ticker),
    }
