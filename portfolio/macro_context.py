import time

_cache = {}
DXY_TTL = 3600
VOLUME_TTL = 300


def get_dxy():
    now = time.time()
    cached = _cache.get("dxy")
    if cached and now - cached["time"] < DXY_TTL:
        return cached["data"]

    import yfinance as yf

    t = yf.Ticker("DX-Y.NYB")
    h = t.history(period="30d")
    if h.empty:
        return None

    close = h["Close"]
    current = float(close.iloc[-1])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    pct_5d = (
        float((close.iloc[-1] / close.iloc[-5] - 1) * 100) if len(close) >= 5 else 0
    )

    if current > sma20:
        trend = "strong"
    else:
        trend = "weak"

    result = {
        "value": round(current, 2),
        "sma20": round(sma20, 2),
        "trend": trend,
        "change_5d_pct": round(pct_5d, 2),
    }
    _cache["dxy"] = {"data": result, "time": now}
    return result


def get_volume_context(klines_df):
    if klines_df is None or klines_df.empty:
        return None
    vol = klines_df["volume"].astype(float)
    current = float(vol.iloc[-1])
    avg20 = (
        float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else float(vol.mean())
    )
    ratio = current / avg20 if avg20 > 0 else 1.0

    return {
        "current": round(current, 2),
        "avg20": round(avg20, 2),
        "ratio": round(ratio, 2),
        "spike": ratio > 2.0,
    }


if __name__ == "__main__":
    dxy = get_dxy()
    print(f"DXY: {dxy}")
