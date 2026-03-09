"""Data providers for GoldDigger — gold price, USD/SEK, US10Y yield, certificate quotes.

Uses Binance FAPI for gold (XAUUSDT), existing fx_rates for USD/SEK,
FRED for US10Y yield, and Avanza Playwright session for certificate bid/ask.
"""

import json
import logging
import math
import sys as _sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from portfolio.circuit_breaker import CircuitBreaker
from portfolio.http_retry import fetch_with_retry

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
if str(_DATA_DIR) not in _sys.path:
    _sys.path.insert(0, str(_DATA_DIR))

logger = logging.getLogger("portfolio.golddigger.data")

BINANCE_FAPI_BASE = "https://fapi.binance.com/fapi/v1"

_gold_cb = CircuitBreaker("golddigger_gold", failure_threshold=5, recovery_timeout=60)
_fred_cb = CircuitBreaker("golddigger_fred", failure_threshold=3, recovery_timeout=300)

# FRED yield cache (yields don't change intraday often)
_yield_cache: dict = {"value": None, "time": 0}
_YIELD_CACHE_TTL = 3600  # 1 hour


@dataclass
class MarketSnapshot:
    """A single point-in-time market data sample."""
    ts_utc: datetime
    gold: float           # XAUUSD price
    usdsek: float         # USD/SEK exchange rate
    us10y: float          # US 10Y yield as decimal (e.g., 0.0425)
    cert_bid: Optional[float] = None
    cert_ask: Optional[float] = None
    cert_last: Optional[float] = None
    cert_spread_pct: Optional[float] = None
    data_quality: str = "ok"  # "ok", "partial", "stale"
    gold_fetch_ts: Optional[datetime] = None
    fx_fetch_ts: Optional[datetime] = None
    gold_volume_ratio: Optional[float] = None

    def is_complete(self) -> bool:
        return self.gold > 0 and self.usdsek > 0

    def is_fresh(self, max_age_seconds: float = 90.0) -> bool:
        now = datetime.now(timezone.utc)
        for ts in [self.gold_fetch_ts, self.fx_fetch_ts]:
            if ts is not None and (now - ts).total_seconds() > max_age_seconds:
                return False
        return True


def fetch_gold_price() -> Optional[float]:
    """Fetch XAUUSDT from Binance FAPI (futures). Returns price or None."""
    if not _gold_cb.allow_request():
        logger.warning("Gold circuit breaker OPEN")
        return None
    try:
        r = fetch_with_retry(
            f"{BINANCE_FAPI_BASE}/ticker/price",
            params={"symbol": "XAUUSDT"},
            timeout=10,
        )
        if r is None:
            _gold_cb.record_failure()
            return None
        r.raise_for_status()
        price = float(r.json()["price"])
        _gold_cb.record_success()
        return price
    except Exception as e:
        logger.warning("Gold price fetch failed: %s", e)
        _gold_cb.record_failure()
        return None


def fetch_usdsek() -> Optional[float]:
    """Fetch USD/SEK via existing fx_rates module."""
    try:
        from portfolio.fx_rates import fetch_usd_sek
        return fetch_usd_sek()
    except Exception as e:
        logger.warning("USD/SEK fetch failed: %s", e)
        return None


def fetch_us10y(fred_api_key: str = "") -> Optional[float]:
    """Fetch US 10Y yield from FRED API. Returns yield as decimal (e.g., 0.0425).

    Cached for 1 hour since Treasury yields are based on daily closes.
    Falls back to cached value if API is unavailable.
    """
    now = time.time()
    if _yield_cache["value"] is not None and now - _yield_cache["time"] < _YIELD_CACHE_TTL:
        return _yield_cache["value"]

    if not fred_api_key:
        logger.debug("No FRED API key — using cached yield or None")
        return _yield_cache.get("value")

    if not _fred_cb.allow_request():
        logger.warning("FRED circuit breaker OPEN")
        return _yield_cache.get("value")

    try:
        r = fetch_with_retry(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": "DGS10",
                "api_key": fred_api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 5,
            },
            timeout=15,
        )
        if r is None:
            _fred_cb.record_failure()
            return _yield_cache.get("value")
        r.raise_for_status()
        observations = r.json().get("observations", [])
        for obs in observations:
            val = obs.get("value", ".")
            if val != ".":
                yield_pct = float(val)
                yield_decimal = yield_pct / 100.0
                _yield_cache["value"] = yield_decimal
                _yield_cache["time"] = now
                _fred_cb.record_success()
                return yield_decimal
        logger.warning("FRED returned no valid yield observations")
        _fred_cb.record_failure()
        return _yield_cache.get("value")
    except Exception as e:
        logger.warning("FRED US10Y fetch failed: %s", e)
        _fred_cb.record_failure()
        return _yield_cache.get("value")


def fetch_certificate_price(page, orderbook_id: str, api_type: str = "warrant") -> Optional[dict]:
    """Fetch certificate bid/ask/last from Avanza via Playwright page.

    Returns dict: {bid, ask, last, spread_pct} or None.
    """
    if not orderbook_id:
        return None
    try:
        from metals_avanza_helpers import fetch_price
        data = fetch_price(page, orderbook_id, api_type)
        if data is None:
            return None
        bid = data.get("bid")
        ask = data.get("ask")
        last = data.get("last")
        spread_pct = None
        if bid and ask and bid > 0:
            spread_pct = (ask - bid) / bid
        return {
            "bid": bid,
            "ask": ask,
            "last": last,
            "spread_pct": spread_pct,
        }
    except Exception as e:
        logger.warning("Certificate price fetch failed: %s", e)
        return None


def _load_json_safe(path) -> Optional[dict]:
    """Load JSON file safely, return None on failure."""
    try:
        if not Path(path).exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def fetch_gold_volume(lookback_bars: int = 20) -> Optional[dict]:
    """Fetch XAUUSDT volume from Binance FAPI. Returns {current, avg_20, ratio}."""
    try:
        r = fetch_with_retry(
            f"{BINANCE_FAPI_BASE}/klines",
            params={"symbol": "XAUUSDT", "interval": "5m", "limit": lookback_bars + 1},
            timeout=10,
        )
        if r is None:
            return None
        r.raise_for_status()
        bars = r.json()
        volumes = [float(bar[5]) for bar in bars]
        if len(volumes) < 2:
            return None
        current = volumes[-1]
        avg = sum(volumes[:-1]) / len(volumes[:-1])
        return {
            "current": current,
            "avg_20": avg,
            "ratio": current / avg if avg > 0 else 1.0,
        }
    except Exception as e:
        logger.warning("Gold volume fetch failed: %s", e)
        return None


def read_xau_consensus() -> Optional[dict]:
    """Read latest XAU-USD signal consensus from agent_summary_compact.json."""
    path = _DATA_DIR / "agent_summary_compact.json"
    data = _load_json_safe(path)
    if not data:
        return None
    signals = data.get("signals", {}).get("XAU-USD", {})
    if not signals:
        return None
    return {
        "action": signals.get("consensus", "HOLD"),
        "confidence": signals.get("confidence", 0.0),
        "buy_count": signals.get("buy_count", 0),
        "sell_count": signals.get("sell_count", 0),
        "hold_count": signals.get("abstain_count", 0),
    }


def read_macro_context() -> dict:
    """Read DXY and macro data from agent_summary_compact.json."""
    path = _DATA_DIR / "agent_summary_compact.json"
    data = _load_json_safe(path)
    if not data:
        return {}
    macro = data.get("macro", {})
    return {
        "dxy": macro.get("dxy", {}).get("value"),
        "dxy_5d_change": macro.get("dxy", {}).get("change_5d_pct"),
        "us10y": macro.get("treasury", {}).get("us10y"),
    }


def read_chronos_forecast(ticker: str = "XAU-USD") -> Optional[dict]:
    """Read Chronos forecast from agent_summary_compact.json."""
    path = _DATA_DIR / "agent_summary_compact.json"
    data = _load_json_safe(path)
    if not data:
        return None
    forecasts = data.get("forecast_signals", {}).get(ticker)
    if not forecasts:
        return None
    return {
        "action": forecasts.get("action", "HOLD"),
        "confidence": forecasts.get("confidence", 0.0),
        "pct_move": forecasts.get("chronos_pct_move"),
    }


def read_xau_atr() -> Optional[float]:
    """Read XAU-USD ATR percentage from agent_summary_compact.json."""
    path = _DATA_DIR / "agent_summary_compact.json"
    data = _load_json_safe(path)
    if not data:
        return None
    xau = data.get("signals", {}).get("XAU-USD", {})
    return xau.get("atr_pct")


def collect_snapshot(
    fred_api_key: str = "",
    page=None,
    orderbook_id: str = "",
    api_type: str = "warrant",
    fetch_volume: bool = False,
) -> MarketSnapshot:
    """Collect a complete market snapshot from all data sources.

    Fetches gold, FX, yield, and optionally certificate quotes.
    Returns a MarketSnapshot even if some fields are missing (data_quality
    will reflect the completeness).
    """
    ts = datetime.now(timezone.utc)
    gold = fetch_gold_price()
    gold_fetch_ts = datetime.now(timezone.utc) if gold is not None else None
    usdsek = fetch_usdsek()
    fx_fetch_ts = datetime.now(timezone.utc) if usdsek is not None else None
    us10y = fetch_us10y(fred_api_key)

    cert_data = None
    if page and orderbook_id:
        cert_data = fetch_certificate_price(page, orderbook_id, api_type)

    # Optional volume fetch
    gold_volume_ratio = None
    if fetch_volume:
        vol_data = fetch_gold_volume()
        if vol_data:
            gold_volume_ratio = vol_data.get("ratio")

    # Determine data quality
    missing = []
    if gold is None:
        missing.append("gold")
        gold = 0.0
    if usdsek is None:
        missing.append("usdsek")
        usdsek = 0.0
    if us10y is None:
        missing.append("us10y")
        us10y = 0.0

    quality = "ok"
    if missing:
        quality = "partial" if len(missing) < 3 else "stale"
        logger.warning("Missing data sources: %s", ", ".join(missing))

    snap = MarketSnapshot(
        ts_utc=ts,
        gold=gold,
        usdsek=usdsek,
        us10y=us10y,
        data_quality=quality,
        gold_fetch_ts=gold_fetch_ts,
        fx_fetch_ts=fx_fetch_ts,
        gold_volume_ratio=gold_volume_ratio,
    )
    if cert_data:
        snap.cert_bid = cert_data.get("bid")
        snap.cert_ask = cert_data.get("ask")
        snap.cert_last = cert_data.get("last")
        snap.cert_spread_pct = cert_data.get("spread_pct")

    return snap
