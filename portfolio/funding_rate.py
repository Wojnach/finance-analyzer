from portfolio.api_utils import BINANCE_FAPI_BASE as BINANCE_FAPI
from portfolio.http_retry import fetch_json
from portfolio.shared_state import _cached, FUNDING_RATE_TTL

SYMBOL_MAP = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
}


def _fetch_funding_rate(ticker):
    """Fetch and interpret funding rate for a single ticker."""
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

    return {
        "rate": rate,
        "rate_pct": round(rate * 100, 4),
        "action": action,
        "mark_price": float(data["markPrice"]),
    }


def get_funding_rate(ticker):
    if ticker not in SYMBOL_MAP:
        return None
    return _cached(f"funding_rate_{ticker}", FUNDING_RATE_TTL,
                   _fetch_funding_rate, ticker)


if __name__ == "__main__":
    for t in ["BTC-USD", "ETH-USD", "SOL-USD"]:
        print(f"{t}: {get_funding_rate(t)}")
