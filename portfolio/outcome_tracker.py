import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from portfolio.api_utils import BINANCE_BASE, BINANCE_FAPI_BASE
from portfolio.file_utils import atomic_append_jsonl
from portfolio.http_retry import fetch_with_retry
from portfolio.shared_state import _yfinance_limiter

logger = logging.getLogger("portfolio.outcome_tracker")

_MWU_HORIZON = "1d"  # which outcome horizon to use for MWU weight updates

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SIGNAL_LOG = DATA_DIR / "signal_log.jsonl"

HORIZONS = {"3h": 10800, "4h": 14400, "12h": 43200, "1d": 86400, "3d": 259200, "5d": 432000, "10d": 864000}
from portfolio.tickers import (
    BINANCE_FAPI_MAP,
    BINANCE_SPOT_MAP,
    SIGNAL_NAMES,
    YF_MAP,
)


def _derive_signal_vote(name, indicators, extra):
    if name == "rsi":
        rsi = indicators.get("rsi")
        if rsi is None:
            return "HOLD"
        # BUG-111: Should use adaptive thresholds from rsi_p20/rsi_p80
        rsi_lower = indicators.get("rsi_p20", 30)
        rsi_upper = indicators.get("rsi_p80", 70)
        rsi_lower = max(rsi_lower, 15)
        rsi_upper = min(rsi_upper, 85)
        if rsi < rsi_lower:
            return "BUY"
        if rsi > rsi_upper:
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

    if name == "qwen3":
        return extra.get("qwen3_action", "HOLD")

    # custom_lora removed — signal disabled, no longer tracked

    return "HOLD"


def log_signal_snapshot(signals_dict, prices_usd, fx_rate, trigger_reasons):
    ts = datetime.now(UTC).isoformat()
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

        regime = extra.get("_regime", "unknown")

        tickers[ticker] = {
            "price_usd": price,
            "consensus": consensus,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "total_voters": total_voters,
            "signals": signals,
            "regime": regime,
        }

    entry = {
        "ts": ts,
        "trigger_reasons": trigger_reasons,
        "fx_rate": fx_rate,
        "tickers": tickers,
        "outcomes": {},
    }

    atomic_append_jsonl(SIGNAL_LOG, entry)

    # Dual-write to SQLite
    try:
        from portfolio.signal_db import SignalDB
        db = SignalDB()
        db.insert_snapshot(entry)
        db.close()
    except Exception as e:
        logger.debug("SQLite snapshot write failed: %s", e)

    return entry


def _fetch_current_price(ticker):
    if ticker in BINANCE_FAPI_MAP:
        symbol = BINANCE_FAPI_MAP[ticker]
        r = fetch_with_retry(
            f"{BINANCE_FAPI_BASE}/ticker/price",
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
            f"{BINANCE_BASE}/ticker/price",
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
            f"{BINANCE_FAPI_BASE}/klines",
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
            f"{BINANCE_BASE}/klines",
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
        target_dt = datetime.fromtimestamp(target_ts, tz=UTC)
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


def backfill_outcomes(max_entries=2000):
    """Backfill price outcomes for signal log entries.

    Memory-optimized: only parses the last ``max_entries`` lines as JSON.
    Head entries are streamed as raw bytes during rewrite (BUG-112).

    Args:
        max_entries: Only process the last N entries to limit memory usage.
            Older entries are assumed to be fully backfilled already.
    """
    if not SIGNAL_LOG.exists():
        return 0

    file_size = SIGNAL_LOG.stat().st_size
    if file_size == 0:
        return 0

    # Phase 1: Count total lines (fast binary scan, no JSON parsing)
    total_lines = 0
    with open(SIGNAL_LOG, "rb") as f:
        for _ in f:
            total_lines += 1

    head_count = max(0, total_lines - max_entries) if max_entries else 0

    # Phase 2: Skip head lines, parse only the tail as JSON
    head_end_offset = 0
    entries = []
    with open(SIGNAL_LOG, "rb") as f:
        for _ in range(head_count):
            f.readline()  # skip without JSON parsing
        head_end_offset = f.tell()

        for raw_line in f:
            stripped = raw_line.strip()
            if stripped:
                try:
                    entries.append(json.loads(stripped))
                except json.JSONDecodeError:
                    continue

    now = datetime.now(UTC)
    now_ts = now.timestamp()
    price_cache = {}
    updated = 0

    # Tickers we can actually fetch prices for — skip unknown/removed tickers
    known_tickers = set(BINANCE_SPOT_MAP) | set(BINANCE_FAPI_MAP) | set(YF_MAP)

    # Open SignalDB once for all dual-writes (avoids per-outcome open/close)
    _db = None
    try:
        from portfolio.signal_db import SignalDB
        _db = SignalDB()
    except Exception as e:
        logger.debug("SignalDB open failed: %s", e)

    for entry in entries:
        entry_ts = datetime.fromisoformat(entry["ts"]).timestamp()
        tickers = entry.get("tickers", {})
        outcomes = entry.get("outcomes", {})

        all_filled = True
        for ticker in tickers:
            if ticker not in outcomes:
                outcomes[ticker] = {h: None for h in HORIZONS}
            for h_key in HORIZONS:
                if outcomes[ticker].get(h_key) is None:
                    all_filled = False

        if all_filled and all(
            all(outcomes[t].get(h) is not None for h in HORIZONS) for t in tickers
        ):
            continue

        entry_updated = False
        for ticker in tickers:
            if ticker not in known_tickers:
                continue  # skip removed/unknown tickers (e.g. AI)
            if ticker not in outcomes:
                outcomes[ticker] = {h: None for h in HORIZONS}

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
                    target_ts, tz=UTC
                ).isoformat()
                outcomes[ticker][h_key] = {
                    "price_usd": round(hist_price, 2),
                    "change_pct": change_pct,
                    "ts": outcome_ts_str,
                }
                entry_updated = True

                # Dual-write outcome to SQLite
                if _db is not None:
                    try:
                        _db.update_outcome(
                            entry["ts"], ticker, h_key,
                            round(hist_price, 2), change_pct, outcome_ts_str,
                        )
                    except Exception as e:
                        logger.debug("SQLite outcome write failed: %s", e)

        entry["outcomes"] = outcomes

        # MWU weight update: for each newly-confirmed 1d outcome, update
        # signal weights based on whether each non-HOLD signal was correct.
        if entry_updated:
            try:
                from portfolio.accuracy_stats import _vote_correct
                from portfolio.signal_weights import SignalWeightManager

                _mwu_mgr = SignalWeightManager()
                _mwu_outcomes: dict[str, bool] = {}

                for ticker, ticker_data in tickers.items():
                    outcome_1d = outcomes.get(ticker, {}).get(_MWU_HORIZON)
                    if outcome_1d is None:
                        continue
                    change_pct = outcome_1d.get("change_pct")
                    if change_pct is None:
                        continue
                    signals = ticker_data.get("signals", {})
                    for sig_name, vote in signals.items():
                        if vote == "HOLD":
                            continue
                        correct = _vote_correct(vote, change_pct)
                        if correct is None:
                            continue  # neutral outcome — price didn't move enough
                        # Accumulate: if a signal appears for multiple tickers,
                        # aggregate as "correct if majority correct" — simple
                        # approach: last value wins per (sig_name, ticker) pair.
                        # Using compound key to avoid cross-ticker averaging.
                        key = f"{sig_name}::{ticker}"
                        _mwu_outcomes[key] = correct

                if _mwu_outcomes:
                    # Map compound keys back to signal names for the manager
                    # The manager tracks per signal_name, so we aggregate across
                    # tickers: a signal is updated once per (signal, ticker) pair.
                    for compound_key, correct in _mwu_outcomes.items():
                        sig_name = compound_key.split("::")[0]
                        _mwu_mgr.update(sig_name, correct)
                    _mwu_mgr.save()
            except Exception:
                logger.debug("MWU weight update failed", exc_info=True)

        if entry_updated:
            updated += 1

    if _db is not None:
        try:
            _db.close()
        except Exception as e:
            logger.debug("SignalDB close failed: %s", e)

    import os
    import tempfile

    fd, tmp = tempfile.mkstemp(dir=SIGNAL_LOG.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f_out:
            # Stream head bytes verbatim from original file (no JSON parsing)
            if head_end_offset > 0:
                with open(SIGNAL_LOG, "rb") as f_in:
                    remaining = head_end_offset
                    while remaining > 0:
                        chunk = f_in.read(min(65536, remaining))
                        if not chunk:
                            break
                        f_out.write(chunk)
                        remaining -= len(chunk)
            # Write modified tail entries
            for entry in entries:
                f_out.write((json.dumps(entry) + "\n").encode("utf-8"))
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
