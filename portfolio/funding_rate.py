import time
import requests

BINANCE_FAPI = "https://fapi.binance.com/fapi/v1"
SYMBOL_MAP = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
}

_cache = {}
TTL = 900


def get_funding_rate(ticker):
    if ticker not in SYMBOL_MAP:
        return None

    now = time.time()
    cached = _cache.get(ticker)
    if cached and now - cached["time"] < TTL:
        return cached["data"]

    symbol = SYMBOL_MAP[ticker]
    r = requests.get(
        f"{BINANCE_FAPI}/premiumIndex",
        params={"symbol": symbol},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()

    rate = float(data["lastFundingRate"])
    # Normal funding ~0.01% (0.0001). Thresholds:
    #   > 0.03% → overleveraged longs → contrarian SELL
    #   < -0.01% → overleveraged shorts → contrarian BUY
    if rate > 0.0003:
        action = "SELL"
    elif rate < -0.0001:
        action = "BUY"
    else:
        action = "HOLD"

    result = {
        "rate": rate,
        "rate_pct": round(rate * 100, 4),
        "action": action,
        "mark_price": float(data["markPrice"]),
    }
    _cache[ticker] = {"data": result, "time": now}
    return result


if __name__ == "__main__":
    for t in ["BTC-USD", "ETH-USD", "SOL-USD"]:
        print(f"{t}: {get_funding_rate(t)}")
