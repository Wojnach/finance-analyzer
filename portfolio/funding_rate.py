import time

from portfolio.api_utils import BINANCE_FAPI_BASE as BINANCE_FAPI
from portfolio.http_retry import fetch_json
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
    data = fetch_json(
        f"{BINANCE_FAPI}/premiumIndex",
        params={"symbol": symbol},
        timeout=10,
        label="funding_rate",
    )
    if data is None:
        return None

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
