import logging
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import atomic_write_json, load_json
from portfolio.http_retry import fetch_json

logger = logging.getLogger(__name__)

CRYPTO_TICKERS = {"BTC", "ETH", "BTC-USD", "ETH-USD"}

# Sustained fear/greed tracking — used by signal_engine to gate contrarian
# signals during prolonged extreme sentiment regimes (e.g., 46-day fear streaks).
EXTREME_FEAR_THRESHOLD = 20
EXTREME_GREED_THRESHOLD = 80
_STREAK_FILE = Path("data/fear_greed_streak.json")


def get_sustained_fear_days() -> int:
    """Return consecutive days the Fear & Greed index has been <= EXTREME_FEAR_THRESHOLD.

    Returns 0 if not in an extreme fear streak, or if tracking data is unavailable.
    """
    try:
        data = load_json(_STREAK_FILE)
        if data and data.get("streak_type") == "extreme_fear":
            return data.get("streak_days", 0)
    except Exception:
        logger.debug("Could not read fear streak file", exc_info=True)
    return 0


def update_fear_streak(fg_value: int) -> dict:
    """Update the sustained fear/greed streak tracker.

    Called after each successful F&G fetch. Persists streak state to disk
    so it survives process restarts.
    """
    data = load_json(_STREAK_FILE, default={}) or {}

    now = datetime.now(UTC).isoformat()
    prev_type = data.get("streak_type", "neutral")
    prev_days = data.get("streak_days", 0)

    if fg_value <= EXTREME_FEAR_THRESHOLD:
        if prev_type == "extreme_fear":
            data["streak_days"] = prev_days + 1
        else:
            data = {"streak_type": "extreme_fear", "streak_days": 1,
                    "streak_started": now}
    elif fg_value >= EXTREME_GREED_THRESHOLD:
        if prev_type == "extreme_greed":
            data["streak_days"] = prev_days + 1
        else:
            data = {"streak_type": "extreme_greed", "streak_days": 1,
                    "streak_started": now}
    else:
        data = {"streak_type": "neutral", "streak_days": 0,
                "streak_started": now}

    data["last_value"] = fg_value
    data["last_updated"] = now
    try:
        atomic_write_json(_STREAK_FILE, data)
    except Exception:
        logger.debug("Could not write fear streak file", exc_info=True)
    return data


def _classify(value):
    if value <= 20:
        return "Extreme Fear"
    if value <= 40:
        return "Fear"
    if value <= 60:
        return "Neutral"
    if value <= 80:
        return "Greed"
    return "Extreme Greed"


def get_crypto_fear_greed() -> dict:
    body = fetch_json("https://api.alternative.me/fng/", timeout=10,
                      label="crypto_fear_greed")
    if body is None:
        return None
    data = body["data"][0]
    return {
        "value": int(data["value"]),
        "classification": data["value_classification"],
        "timestamp": datetime.fromtimestamp(
            int(data["timestamp"]), tz=UTC
        ).isoformat(),
    }


def get_stock_fear_greed() -> dict:
    import yfinance as yf

    vix = yf.Ticker("^VIX")
    h = vix.history(period="5d")
    if h.empty:
        return None
    vix_val = float(h["Close"].iloc[-1])
    if vix_val >= 40:
        value = 5
    elif vix_val >= 30:
        value = int(5 + (40 - vix_val) * 1.5)
    elif vix_val >= 20:
        value = int(20 + (30 - vix_val) * 3)
    elif vix_val >= 15:
        value = int(50 + (20 - vix_val) * 6)
    else:
        value = int(80 + (15 - vix_val) * 4)
    value = max(0, min(100, value))
    return {
        "value": value,
        "classification": _classify(value),
        "timestamp": datetime.now(UTC).isoformat(),
        "vix": round(vix_val, 2),
    }


def get_fear_greed(ticker=None) -> dict:
    if ticker is None or ticker.upper().replace("-USD", "") in CRYPTO_TICKERS:
        return get_crypto_fear_greed()
    return get_stock_fear_greed()


if __name__ == "__main__":
    print("=== Crypto Fear & Greed ===")
    result = get_crypto_fear_greed()
    print(f"  Value: {result['value']} ({result['classification']})")
    print(f"  Timestamp: {result['timestamp']}")
    print("\n=== Stock Fear & Greed (VIX) ===")
    result = get_stock_fear_greed()
    if result:
        print(f"  Value: {result['value']} ({result['classification']})")
        print(f"  VIX: {result['vix']}")
    else:
        print("  Failed to fetch VIX data")
