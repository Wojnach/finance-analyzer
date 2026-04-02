"""TinyLoRA RL training scaffolding — data collection + config, no actual GPU work.

Collects (context, reward) pairs from signal_log.jsonl for GRPO-style training.
Training is only allowed after market hours (22:00-08:00 CET weekdays, all day weekends).
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger("portfolio.tinylora_trainer")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SIGNAL_LOG = DATA_DIR / "signal_log.jsonl"

# CET = UTC+1 (simplified; DST handled separately in production)
CET = timezone(timedelta(hours=1))

# Neutral threshold — price moves within ±0.05% are treated as no signal
_NEUTRAL_THRESHOLD = 0.05


def is_training_allowed(now=None):
    """Return True if training is allowed (outside market hours).

    Market hours: 08:00-22:00 CET on weekdays.
    Training allowed: 22:00-08:00 CET weekdays, all day on weekends.
    """
    if now is None:
        now = datetime.now(CET)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=CET)
    else:
        now = now.astimezone(CET)

    # Weekends are always allowed (Saturday=5, Sunday=6)
    if now.weekday() >= 5:
        return True

    # Weekday: only allowed before 08:00 or at/after 22:00
    hour = now.hour
    return hour < 8 or hour >= 22


def collect_training_pairs(log_path=None, ticker="XAG-USD", horizon="1d"):
    """Read signal_log.jsonl and extract (context_string, reward) pairs.

    Parameters
    ----------
    log_path : str or Path, optional
        Path to the signal log JSONL. Defaults to DATA_DIR / signal_log.jsonl.
    ticker : str
        Ticker to extract pairs for.
    horizon : str
        Outcome horizon to use for reward (e.g., "1d", "3d").

    Returns
    -------
    list[tuple[str, int]]
        List of (context_string, reward) pairs.
        reward: +1 if consensus matched price move, -1 if wrong, 0 if neutral/HOLD.
    """
    if log_path is None:
        log_path = SIGNAL_LOG
    log_path = Path(log_path)

    if not log_path.exists():
        logger.warning("Signal log not found: %s", log_path)
        return []

    pairs = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            tickers = entry.get("tickers", {})
            if ticker not in tickers:
                continue

            ticker_data = tickers[ticker]
            price = ticker_data.get("price_usd", 0.0)
            consensus = ticker_data.get("consensus", "HOLD")
            signals = ticker_data.get("signals", {})

            # Build context string
            signal_parts = [f"{name}={val}" for name, val in sorted(signals.items())]
            signals_str = ", ".join(signal_parts) if signal_parts else "none"
            context = (
                f"Ticker: {ticker}, Price: ${price:.2f}, "
                f"Signals: [{signals_str}], Consensus: {consensus}"
            )

            # Determine reward from outcomes
            outcomes = entry.get("outcomes", {})
            ticker_outcomes = outcomes.get(ticker, {})
            horizon_data = ticker_outcomes.get(horizon, {})
            change_pct = horizon_data.get("change_pct")

            if change_pct is None:
                # No outcome data — skip this entry
                continue

            if consensus == "HOLD" or abs(change_pct) < _NEUTRAL_THRESHOLD:
                reward = 0
            elif (consensus == "BUY" and change_pct > 0) or \
                 (consensus == "SELL" and change_pct < 0):
                reward = 1
            else:
                reward = -1

            pairs.append((context, reward))

    logger.info("Collected %d training pairs for %s/%s", len(pairs), ticker, horizon)
    return pairs


def prepare_training_config(
    model_path,
    adapter_dir,
    rank=1,
    learning_rate=1e-4,
    epochs=3,
):
    """Return a training config dict for TinyLoRA GRPO training.

    Parameters
    ----------
    model_path : str
        Path to the base model.
    adapter_dir : str
        Directory to save the LoRA adapter.
    rank : int
        LoRA rank (default 1 for TinyLoRA).
    learning_rate : float
        Learning rate.
    epochs : int
        Number of training epochs.

    Returns
    -------
    dict
        Training configuration.
    """
    return {
        "method": "GRPO",
        "model_path": str(model_path),
        "adapter_dir": str(adapter_dir),
        "rank": rank,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "estimated_params": rank * 13,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not is_training_allowed():
        print("Training not allowed during market hours (08:00-22:00 CET weekdays).")
        raise SystemExit(0)

    pairs = collect_training_pairs()
    if not pairs:
        print("No training pairs found.")
        raise SystemExit(0)

    positive = sum(1 for _, r in pairs if r == 1)
    negative = sum(1 for _, r in pairs if r == -1)
    neutral = sum(1 for _, r in pairs if r == 0)
    print(f"Training pairs: {len(pairs)} total")
    print(f"  Positive (correct): {positive}")
    print(f"  Negative (wrong):   {negative}")
    print(f"  Neutral (hold/flat): {neutral}")
