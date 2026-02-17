import time
import joblib
import numpy as np
import pandas as pd
import requests
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "ml_classifier.joblib"
FEATURES_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "ml_feature_names.joblib"
)
BINANCE_BASE = "https://api.binance.com/api/v3"
LABEL_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}

_model_cache = {"model": None, "mtime": 0}
_pred_cache = {}
PRED_TTL = 900


def _load_model():
    mtime = MODEL_PATH.stat().st_mtime
    if _model_cache["model"] is None or mtime != _model_cache["mtime"]:
        _model_cache["model"] = joblib.load(MODEL_PATH)
        _model_cache["mtime"] = mtime
    return _model_cache["model"]


def compute_features(df, symbol_flag=0):
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    opn = df["open"].astype(float)
    volume = df["volume"].astype(float)

    feats = pd.DataFrame(index=df.index)

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    eps = np.finfo(float).eps
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    feats["rsi14"] = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, eps)))

    avg_gain7 = gain.ewm(alpha=1 / 7, min_periods=7, adjust=False).mean()
    avg_loss7 = loss.ewm(alpha=1 / 7, min_periods=7, adjust=False).mean()
    feats["rsi7"] = 100 - (100 / (1 + avg_gain7 / avg_loss7.replace(0, eps)))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    feats["macd_hist"] = macd_hist
    feats["macd_slope"] = macd_hist - macd_hist.shift(1)

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    feats["ema_ratio"] = ema9 / ema21
    feats["ema_cross"] = (ema9 > ema21).astype(int)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    feats["bb_pctb"] = (close - bb_lower) / (bb_upper - bb_lower)
    feats["bb_width"] = (bb_upper - bb_lower) / bb_mid

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    feats["atr_pct"] = atr14 / close * 100

    for n in [1, 3, 6, 12, 24]:
        feats[f"ret_{n}"] = close.pct_change(n)

    feats["vol_ratio"] = volume / volume.rolling(20).mean()

    body_top = pd.concat([opn, close], axis=1).max(axis=1)
    body_bot = pd.concat([opn, close], axis=1).min(axis=1)
    feats["upper_wick_pct"] = (high - body_top) / close * 100
    feats["lower_wick_pct"] = (body_bot - low) / close * 100
    feats["range_pct"] = (high - low) / close * 100

    dt = pd.to_datetime(df["date"])
    hour = dt.dt.hour
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    feats["symbol"] = symbol_flag

    return feats


def get_ml_signal(ticker):
    now = time.time()
    cached = _pred_cache.get(ticker)
    if cached and now - cached["time"] < PRED_TTL:
        return cached["data"]

    symbol_map = {
        "BTC-USD": ("BTCUSDT", 0),
        "ETH-USD": ("ETHUSDT", 1),
    }
    if ticker not in symbol_map:
        return None
    binance_sym, sym_flag = symbol_map[ticker]

    r = requests.get(
        f"{BINANCE_BASE}/klines",
        params={"symbol": binance_sym, "interval": "1h", "limit": 100},
        timeout=10,
    )
    r.raise_for_status()
    raw = r.json()

    df = pd.DataFrame(
        raw,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_vol",
            "trades",
            "taker_base",
            "taker_quote",
            "ignore",
        ],
    )
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    model = _load_model()
    features = compute_features(df, symbol_flag=sym_flag)
    last_row = features.iloc[[-1]].values

    pred = model.predict(last_row)[0]
    proba = model.predict_proba(last_row)[0]

    result = {
        "action": LABEL_MAP[pred],
        "confidence": round(float(proba.max()), 4),
    }
    _pred_cache[ticker] = {"data": result, "time": now}
    return result


if __name__ == "__main__":
    for t in ["BTC-USD", "ETH-USD"]:
        print(f"{t}: {get_ml_signal(t)}")
