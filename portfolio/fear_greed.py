import requests
from datetime import datetime, timezone

CRYPTO_TICKERS = {"BTC", "ETH", "SOL", "BTC-USD", "ETH-USD", "SOL-USD"}


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
    resp = requests.get("https://api.alternative.me/fng/", timeout=10)
    resp.raise_for_status()
    data = resp.json()["data"][0]
    return {
        "value": int(data["value"]),
        "classification": data["value_classification"],
        "timestamp": datetime.fromtimestamp(
            int(data["timestamp"]), tz=timezone.utc
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "vix": round(vix_val, 2),
    }


def get_fear_greed(ticker=None) -> dict:
    if ticker is None or ticker.upper().replace("-USD", "") in {"BTC", "ETH"}:
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
