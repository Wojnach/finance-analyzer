import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from portfolio.http_retry import fetch_with_retry
from portfolio.shared_state import _RateLimiter

BASE_DIR = Path(__file__).resolve().parent.parent

_yfinance_limiter = _RateLimiter(30, "yfinance")
DATA_DIR = BASE_DIR / "data"
SIGNAL_LOG = DATA_DIR / "signal_log.jsonl"

HORIZONS = {"1d": 86400, "3d": 259200, "5d": 432000, "10d": 864000}
from portfolio.tickers import (
    BINANCE_SPOT_MAP,
    BINANCE_FAPI_MAP,
    BINANCE_MAP,
    YF_MAP,
    SIGNAL_NAMES,
)


def _derive_signal_vote(name, indicators, extra):
    if name == "rsi":
        rsi = indicators.get("rsi")
        if rsi is None:
            return "HOLD"
        if rsi < 30:
            return "BUY"
        if rsi > 70:
            return "SELL"
        return "HOLD"

    if name == "macd":
        hist = indicators.get("macd_hist")
        hist_prev = indicators.get("macd_hist_prev")
        if hist is None or hist_prev is None:
            return "HOLD"
        if hist > 0 and hist_prev <= 0:
            return "BUY"
        if hist < 0 and hist_prev >= 0:
            return "SELL"
        return "HOLD"

    if name == "ema":
        ema9 = indicators.get("ema9")
        ema21 = indicators.get("ema21")
        if ema9 is None or ema21 is None:
            return "HOLD"
        ema_gap_pct = abs(ema9 - ema21) / ema21 * 100 if ema21 != 0 else 0
        if ema_gap_pct < 0.5:
            return "HOLD"
        return "BUY" if ema9 > ema21 else "SELL"

    if name == "bb":
        pos = indicators.get("price_vs_bb")
        if pos == "below_lower":
            return "BUY"
        if pos == "above_upper":
            return "SELL"
        return "HOLD"

    if name == "fear_greed":
        fg = extra.get("fear_greed")
        if fg is None:
            return "HOLD"
        if fg <= 20:
            return "BUY"
        if fg >= 80:
            return "SELL"
        return "HOLD"

    if name == "sentiment":
        sent = extra.get("sentiment")
        conf = extra.get("sentiment_conf", 0)
        if sent == "positive" and conf > 0.4:
            return "BUY"
        if sent == "negative" and conf > 0.4:
            return "SELL"
        return "HOLD"

    if name == "ministral":
        return extra.get("ministral_action", "HOLD")

    if name == "ml":
        return extra.get("ml_action", "HOLD")

    if name == "funding":
        return extra.get("funding_action", "HOLD")

    if name == "volume":
        return extra.get("volume_action", "HOLD")

    # custom_lora removed — signal disabled, no longer tracked

    return "HOLD"


def log_signal_snapshot(signals_dict, prices_usd, fx_rate, trigger_reasons):
    ts = datetime.now(timezone.utc).isoformat()
    tickers = {}

    for ticker, sig_data in signals_dict.items():
        indicators = sig_data.get("indicators", {})
        extra = sig_data.get("extra", {})
        price = prices_usd.get(ticker, indicators.get("close"))

        passed_votes = extra.get("_votes")
        if passed_votes:
            signals = {name: passed_votes.get(name, "HOLD") for name in SIGNAL_NAMES}
        else:
            signals = {}
            for name in SIGNAL_NAMES:
                signals[name] = _derive_signal_vote(name, indicators, extra)

        buy_count = sum(1 for v in signals.values() if v == "BUY")
        sell_count = sum(1 for v in signals.values() if v == "SELL")

        consensus = sig_data.get("action", "HOLD")
        total_voters = buy_count + sell_count

        tickers[ticker] = {
            "price_usd": price,
            "consensus": consensus,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "total_voters": total_voters,
            "signals": signals,
        }

    entry = {
        "ts": ts,
        "trigger_reasons": trigger_reasons,
        "fx_rate": fx_rate,
        "tickers": tickers,
        "outcomes": {},
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SIGNAL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    # Dual-write to SQLite
    try:
        from portfolio.signal_db import SignalDB
        db = SignalDB()
        db.insert_snapshot(entry)
        db.close()
    except Exception:
        pass  # SQLite write is best-effort; JSONL is the primary

    return entry


def _fetch_current_price(ticker):
    if ticker in BINANCE_FAPI_MAP:
        symbol = BINANCE_FAPI_MAP[ticker]
        r = fetch_with_retry(
            "https://fapi.binance.com/fapi/v1/ticker/price",
            params={"symbol": symbol},
            timeout=5,
        )
        if r is None:
            return None
        r.raise_for_status()
        return float(r.json()["price"])

    if ticker in BINANCE_SPOT_MAP:
        symbol = BINANCE_SPOT_MAP[ticker]
        r = fetch_with_retry(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": symbol},
            timeout=5,
        )
        if r is None:
            return None
        r.raise_for_status()
        return float(r.json()["price"])

    if ticker in YF_MAP:
        import yfinance as yf

        _yfinance_limiter.wait()
        t = yf.Ticker(YF_MAP[ticker])
        h = t.history(period="5d")
        if h.empty:
            return None
        return float(h["Close"].iloc[-1])

    return None


def _fetch_historical_price(ticker, target_ts):
    if ticker in BINANCE_FAPI_MAP:
        symbol = BINANCE_FAPI_MAP[ticker]
        start_ms = int(target_ts * 1000)
        r = fetch_with_retry(
            "https://fapi.binance.com/fapi/v1/klines",
            params={
                "symbol": symbol,
                "interval": "1h",
                "startTime": start_ms,
                "limit": 1,
            },
            timeout=10,
        )
        if r is None:
            return None
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        return float(data[0][4])

    if ticker in BINANCE_SPOT_MAP:
        symbol = BINANCE_SPOT_MAP[ticker]
        start_ms = int(target_ts * 1000)
        r = fetch_with_retry(
            "https://api.binance.com/api/v3/klines",
            params={
                "symbol": symbol,
                "interval": "1h",
                "startTime": start_ms,
                "limit": 1,
            },
            timeout=10,
        )
        if r is None:
            return None
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        return float(data[0][4])

    if ticker in YF_MAP:
        import yfinance as yf

        _yfinance_limiter.wait()
        target_dt = datetime.fromtimestamp(target_ts, tz=timezone.utc)
        start_date = (target_dt - timedelta(days=5)).strftime("%Y-%m-%d")
        end_date = (target_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        t = yf.Ticker(YF_MAP[ticker])
        h = t.history(start=start_date, end=end_date)
        if h.empty:
            return None
        target_date = target_dt.date()
        candidates = h[h.index.date <= target_date]
        if candidates.empty:
            return float(h["Close"].iloc[0])
        return float(candidates["Close"].iloc[-1])

    return None


def backfill_outcomes():
    if not SIGNAL_LOG.exists():
        return 0

    entries = []
    with open(SIGNAL_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    now = datetime.now(timezone.utc)
    now_ts = now.timestamp()
    price_cache = {}
    updated = 0

    for entry in entries:
        entry_ts = datetime.fromisoformat(entry["ts"]).timestamp()
        tickers = entry.get("tickers", {})
        outcomes = entry.get("outcomes", {})

        all_filled = True
        for ticker in tickers:
            if ticker not in outcomes:
                outcomes[ticker] = {"1d": None, "3d": None, "5d": None, "10d": None}
            for h_key in HORIZONS:
                if outcomes[ticker].get(h_key) is None:
                    all_filled = False

        if all_filled and all(
            all(outcomes[t].get(h) is not None for h in HORIZONS) for t in tickers
        ):
            continue

        entry_updated = False
        for ticker in tickers:
            if ticker not in outcomes:
                outcomes[ticker] = {"1d": None, "3d": None, "5d": None, "10d": None}

            base_price = tickers[ticker].get("price_usd")
            for h_key, h_seconds in HORIZONS.items():
                if outcomes[ticker].get(h_key) is not None:
                    continue
                target_ts = entry_ts + h_seconds
                if now_ts < target_ts:
                    continue

                cache_key = (ticker, int(target_ts // 3600))
                if cache_key not in price_cache:
                    try:
                        price_cache[cache_key] = _fetch_historical_price(
                            ticker, target_ts
                        )
                    except Exception:
                        price_cache[cache_key] = None

                hist_price = price_cache[cache_key]
                if hist_price is None:
                    continue

                change_pct = 0.0
                if base_price and base_price > 0:
                    change_pct = round(
                        ((hist_price - base_price) / base_price) * 100, 2
                    )

                outcome_ts_str = datetime.fromtimestamp(
                    target_ts, tz=timezone.utc
                ).isoformat()
                outcomes[ticker][h_key] = {
                    "price_usd": round(hist_price, 2),
                    "change_pct": change_pct,
                    "ts": outcome_ts_str,
                }
                entry_updated = True

                # Dual-write outcome to SQLite
                try:
                    from portfolio.signal_db import SignalDB
                    _db = SignalDB()
                    _db.update_outcome(
                        entry["ts"], ticker, h_key,
                        round(hist_price, 2), change_pct, outcome_ts_str,
                    )
                    _db.close()
                except Exception:
                    pass

        entry["outcomes"] = outcomes
        if entry_updated:
            updated += 1

    import os, tempfile

    fd, tmp = tempfile.mkstemp(dir=SIGNAL_LOG.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        os.replace(tmp, SIGNAL_LOG)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    return updated


if __name__ == "__main__":
    count = backfill_outcomes()
    print(f"Backfilled {count} entries")
