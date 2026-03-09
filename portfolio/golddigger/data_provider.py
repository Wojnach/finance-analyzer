"""Data providers for GoldDigger — gold price, USD/SEK, US10Y yield, certificate quotes.

Uses Binance FAPI for gold (XAUUSDT), existing fx_rates for USD/SEK,
FRED for US10Y yield, and Avanza Playwright session for certificate bid/ask.
"""

import json
import logging
import sys as _sys
import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from portfolio.circuit_breaker import CircuitBreaker
from portfolio.http_retry import fetch_with_retry

if TYPE_CHECKING:
    from portfolio.golddigger.config import GolddiggerConfig

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
if str(_DATA_DIR) not in _sys.path:
    _sys.path.insert(0, str(_DATA_DIR))

logger = logging.getLogger("portfolio.golddigger.data")

BINANCE_FAPI_BASE = "https://fapi.binance.com/fapi/v1"

_gold_cb = CircuitBreaker("golddigger_gold", failure_threshold=5, recovery_timeout=60)
_fred_cb = CircuitBreaker("golddigger_fred", failure_threshold=3, recovery_timeout=300)

# FRED yield cache (yields don't change intraday often)
_yield_cache: dict = {"value": None, "time": 0, "series_id": None}
_YIELD_CACHE_TTL = 900  # 15 min (daily data, but catch updates sooner)
_proxy_cache: dict[str, dict] = {}


@dataclass
class MarketSnapshot:
    """A single point-in-time market data sample."""
    ts_utc: datetime
    gold: float           # XAUUSD price
    usdsek: float         # USD/SEK exchange rate
    us10y: float          # US 10Y yield as decimal (e.g., 0.0425)
    us10y_source: Optional[str] = None
    us10y_change_pct: Optional[float] = None
    dxy: Optional[float] = None
    dxy_source: Optional[str] = None
    dxy_change_pct: Optional[float] = None
    next_event_type: Optional[str] = None
    next_event_hours: Optional[float] = None
    event_risk_active: bool = False
    event_risk_phase: Optional[str] = None
    cert_bid: Optional[float] = None
    cert_ask: Optional[float] = None
    cert_last: Optional[float] = None
    cert_spread_pct: Optional[float] = None
    data_quality: str = "ok"  # "ok", "partial", "stale"
    gold_fetch_ts: Optional[datetime] = None
    fx_fetch_ts: Optional[datetime] = None
    macro_fetch_ts: Optional[datetime] = None
    gold_volume_ratio: Optional[float] = None

    def is_complete(self) -> bool:
        return self.gold > 0 and self.usdsek > 0

    def is_fresh(self, max_age_seconds: float = 90.0) -> bool:
        now = datetime.now(timezone.utc)
        for ts in [self.gold_fetch_ts, self.fx_fetch_ts]:
            if ts is not None and (now - ts).total_seconds() > max_age_seconds:
                return False
        return True


def _coerce_utc(ts_value) -> Optional[datetime]:
    """Convert a timestamp-like value to timezone-aware UTC."""
    if ts_value is None:
        return None
    if isinstance(ts_value, datetime):
        dt = ts_value
    elif hasattr(ts_value, "to_pydatetime"):
        dt = ts_value.to_pydatetime()
    else:
        try:
            dt = datetime.fromisoformat(str(ts_value))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fetch_yfinance_proxy(
    *,
    cache_key: str,
    ticker: str,
    interval: str,
    lookback_bars: int,
    ttl_seconds: int,
    max_bar_age_minutes: float,
) -> Optional[dict]:
    """Fetch an intraday market proxy via yfinance with cache + staleness checks."""
    now = time.time()
    cached = _proxy_cache.get(cache_key)
    if cached and now - cached["time"] < ttl_seconds:
        return cached["value"]

    try:
        from portfolio.data_collector import yfinance_klines
        from portfolio.shared_state import _yfinance_limiter

        _yfinance_limiter.wait()
        df = yfinance_klines(ticker, interval=interval, limit=max(lookback_bars, 2))
        if df is None or df.empty or "close" not in df.columns or "time" not in df.columns:
            return cached["value"] if cached else None

        last_close = float(df["close"].iloc[-1])
        base_idx = max(0, len(df) - min(len(df), lookback_bars))
        base_close = float(df["close"].iloc[base_idx])
        last_time = _coerce_utc(df["time"].iloc[-1])
        if last_time is None:
            return cached["value"] if cached else None

        bar_age_minutes = max(
            0.0,
            (datetime.now(timezone.utc) - last_time).total_seconds() / 60.0,
        )
        if bar_age_minutes > max_bar_age_minutes:
            logger.warning(
                "Proxy %s stale: %.1f min old (limit %.1f)",
                ticker,
                bar_age_minutes,
                max_bar_age_minutes,
            )
            return cached["value"] if cached else None

        change_pct = 0.0
        if base_close > 0:
            change_pct = (last_close / base_close - 1.0) * 100.0

        result = {
            "ticker": ticker,
            "interval": interval,
            "value": last_close,
            "change_pct": round(change_pct, 4),
            "bar_age_minutes": round(bar_age_minutes, 2),
            "as_of": last_time.isoformat(),
        }
        _proxy_cache[cache_key] = {"value": result, "time": now}
        return result
    except Exception as e:
        logger.warning("Intraday proxy fetch failed for %s: %s", ticker, e)
        return cached["value"] if cached else None


def fetch_gold_price(symbol: str = "XAUUSDT") -> Optional[float]:
    """Fetch a gold proxy from Binance FAPI. Returns price or None."""
    if not _gold_cb.allow_request():
        logger.warning("Gold circuit breaker OPEN")
        return None
    try:
        r = fetch_with_retry(
            f"{BINANCE_FAPI_BASE}/ticker/price",
            params={"symbol": symbol},
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
        logger.warning("Gold price fetch failed for %s: %s", symbol, e)
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


def fetch_dxy_context(cfg: "GolddiggerConfig") -> Optional[dict]:
    """Fetch a DXY proxy for intraday gating, with macro-context fallback."""
    proxy = _fetch_yfinance_proxy(
        cache_key="dxy_proxy",
        ticker=cfg.dxy_proxy_ticker,
        interval=cfg.dxy_proxy_interval,
        lookback_bars=cfg.dxy_proxy_lookback_bars,
        ttl_seconds=cfg.dxy_proxy_ttl_seconds,
        max_bar_age_minutes=cfg.dxy_proxy_max_bar_age_minutes,
    )
    if proxy is not None:
        return {
            "value": proxy["value"],
            "change_pct": proxy["change_pct"],
            "source": f"yfinance:{cfg.dxy_proxy_ticker}",
            "as_of": proxy["as_of"],
        }

    macro = read_macro_context()
    if macro.get("dxy") is not None:
        return {
            "value": macro.get("dxy"),
            "change_pct": macro.get("dxy_5d_change"),
            "source": "macro_context",
            "as_of": None,
        }
    return None


def fetch_us10y_context(
    fred_api_key: str = "",
    *,
    source: str = "auto",
    yfinance_ticker: str = "^TNX",
    interval: str = "15m",
    lookback_bars: int = 5,
    ttl_seconds: int = 300,
    max_bar_age_minutes: float = 45.0,
    fred_series: str = "DGS10",
) -> Optional[dict]:
    """Fetch the active rates proxy context with safe fallbacks."""
    if source in ("auto", "yfinance"):
        proxy = _fetch_yfinance_proxy(
            cache_key=f"rates_proxy:{yfinance_ticker}:{interval}",
            ticker=yfinance_ticker,
            interval=interval,
            lookback_bars=lookback_bars,
            ttl_seconds=ttl_seconds,
            max_bar_age_minutes=max_bar_age_minutes,
        )
        if proxy is not None:
            return {
                "value": float(proxy["value"]) / 100.0,
                "change_pct": proxy["change_pct"],
                "source": f"yfinance:{yfinance_ticker}",
                "as_of": proxy["as_of"],
            }

    if source in ("auto", "fred"):
        fred_value = fetch_us10y(fred_api_key, series_id=fred_series)
        if fred_value is not None:
            return {
                "value": fred_value,
                "change_pct": None,
                "source": f"fred:{fred_series}",
                "as_of": None,
            }

    if source in ("auto", "macro"):
        macro = read_macro_context()
        if macro.get("us10y") is not None:
            return {
                "value": macro["us10y"],
                "change_pct": macro.get("us10y_change_5d_pct"),
                "source": "macro_context",
                "as_of": None,
            }
    return None


def fetch_us10y(fred_api_key: str = "", series_id: str = "DGS10") -> Optional[float]:
    """Fetch US 10Y yield from FRED API. Returns yield as decimal (e.g., 0.0425).

    Cached for 1 hour since Treasury yields are based on daily closes.
    Falls back to cached value if API is unavailable.
    """
    now = time.time()
    if (
        _yield_cache["value"] is not None
        and _yield_cache.get("series_id") == series_id
        and now - _yield_cache["time"] < _YIELD_CACHE_TTL
    ):
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
                "series_id": series_id,
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
                _yield_cache["series_id"] = series_id
                _fred_cb.record_success()
                return yield_decimal
        logger.warning("FRED returned no valid yield observations for %s", series_id)
        _fred_cb.record_failure()
        return _yield_cache.get("value")
    except Exception as e:
        logger.warning("FRED US10Y fetch failed for %s: %s", series_id, e)
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


def fetch_gold_volume(symbol: str = "XAUUSDT", lookback_bars: int = 20) -> Optional[dict]:
    """Fetch gold proxy volume from Binance FAPI. Returns {current, avg_20, ratio}."""
    try:
        r = fetch_with_retry(
            f"{BINANCE_FAPI_BASE}/klines",
            params={"symbol": symbol, "interval": "5m", "limit": lookback_bars + 1},
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
        logger.warning("Gold volume fetch failed for %s: %s", symbol, e)
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
    treasury = macro.get("treasury", {}) if isinstance(macro, dict) else {}
    us10y = treasury.get("us10y")
    us10y_change = treasury.get("us10y_change_5d_pct")
    ten_y = treasury.get("10y", {})
    if us10y is None and isinstance(ten_y, dict):
        ten_y_pct = ten_y.get("yield_pct")
        if ten_y_pct is not None:
            us10y = float(ten_y_pct) / 100.0 if float(ten_y_pct) > 1 else float(ten_y_pct)
        us10y_change = ten_y.get("change_5d", us10y_change)
    return {
        "dxy": macro.get("dxy", {}).get("value"),
        "dxy_5d_change": macro.get("dxy", {}).get("change_5d_pct", macro.get("dxy", {}).get("change_5d")),
        "us10y": us10y,
        "us10y_change_5d_pct": us10y_change,
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


def read_event_risk(
    hours_before: float = 4.0,
    hours_after: float = 1.0,
    block_types: tuple[str, ...] = ("FOMC", "CPI", "NFP"),
) -> Optional[dict]:
    """Return current event-risk context for metals around scheduled macro releases."""
    try:
        from portfolio.econ_dates import ECON_EVENTS, EVENT_SECTOR_MAP, next_event
    except Exception as e:
        logger.warning("Event-risk calendar unavailable: %s", e)
        return None

    now = datetime.now(timezone.utc)
    active_types = set(block_types)
    next_evt = next_event(now.date())

    for evt in ECON_EVENTS:
        evt_type = evt.get("type")
        if evt_type not in active_types:
            continue
        if "metals" not in EVENT_SECTOR_MAP.get(evt_type, set()):
            continue
        evt_dt = datetime.combine(evt["date"], dt_time(hour=14), tzinfo=timezone.utc)
        hours_to_event = (evt_dt - now).total_seconds() / 3600.0
        if -hours_after <= hours_to_event <= hours_before:
            return {
                "active": True,
                "event_type": evt_type,
                "impact": evt.get("impact"),
                "hours_to_event": round(hours_to_event, 2),
                "phase": "pre" if hours_to_event >= 0 else "post",
            }

    if next_evt and next_evt.get("type") in active_types and "metals" in EVENT_SECTOR_MAP.get(next_evt["type"], set()):
        return {
            "active": False,
            "event_type": next_evt["type"],
            "impact": next_evt.get("impact"),
            "hours_to_event": next_evt.get("hours_until"),
            "phase": None,
        }
    return None


def read_xau_atr() -> Optional[float]:
    """Read XAU-USD ATR percentage from agent_summary_compact.json."""
    path = _DATA_DIR / "agent_summary_compact.json"
    data = _load_json_safe(path)
    if not data:
        return None
    xau = data.get("signals", {}).get("XAU-USD", {})
    return xau.get("atr_pct")


def collect_snapshot(
    cfg: Optional["GolddiggerConfig"] = None,
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
    gold_symbol = getattr(cfg, "binance_gold_symbol", "XAUUSDT")
    fred_api_key = getattr(cfg, "fred_api_key", fred_api_key)
    fred_series = getattr(cfg, "fred_series", "DGS10")
    rates_source = getattr(cfg, "rates_source", "auto")
    rates_proxy_ticker = getattr(cfg, "rates_proxy_ticker", "^TNX")
    rates_proxy_interval = getattr(cfg, "rates_proxy_interval", "15m")
    rates_proxy_lookback_bars = getattr(cfg, "rates_proxy_lookback_bars", 5)
    rates_proxy_ttl_seconds = getattr(cfg, "rates_proxy_ttl_seconds", 300)
    rates_proxy_max_bar_age_minutes = getattr(cfg, "rates_proxy_max_bar_age_minutes", 45.0)
    fetch_volume = fetch_volume or bool(getattr(cfg, "use_volume_confirm", False))
    fetch_dxy = bool(getattr(cfg, "use_intraday_dxy_gate", False))
    orderbook_id = orderbook_id or getattr(cfg, "bull_orderbook_id", "")
    api_type = getattr(cfg, "cert_api_type", api_type)

    ts = datetime.now(timezone.utc)
    gold = fetch_gold_price(gold_symbol)
    gold_fetch_ts = datetime.now(timezone.utc) if gold is not None else None
    usdsek = fetch_usdsek()
    fx_fetch_ts = datetime.now(timezone.utc) if usdsek is not None else None
    rate_ctx = fetch_us10y_context(
        fred_api_key,
        source=rates_source,
        yfinance_ticker=rates_proxy_ticker,
        interval=rates_proxy_interval,
        lookback_bars=rates_proxy_lookback_bars,
        ttl_seconds=rates_proxy_ttl_seconds,
        max_bar_age_minutes=rates_proxy_max_bar_age_minutes,
        fred_series=fred_series,
    )
    us10y = rate_ctx["value"] if rate_ctx else None

    dxy_ctx = fetch_dxy_context(cfg) if cfg is not None and fetch_dxy else None
    event_ctx = read_event_risk(
        hours_before=getattr(cfg, "event_risk_hours_before", 4.0),
        hours_after=getattr(cfg, "event_risk_hours_after", 1.0),
        block_types=getattr(cfg, "event_risk_block_types", ("FOMC", "CPI", "NFP")),
    ) if cfg is not None and getattr(cfg, "use_event_risk_gate", False) else None
    macro_fetch_ts = datetime.now(timezone.utc) if (rate_ctx or dxy_ctx or event_ctx) else None

    cert_data = None
    if page and orderbook_id:
        cert_data = fetch_certificate_price(page, orderbook_id, api_type)

    # Optional volume fetch
    gold_volume_ratio = None
    if fetch_volume:
        vol_data = fetch_gold_volume(gold_symbol)
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
        us10y_source=rate_ctx.get("source") if rate_ctx else None,
        us10y_change_pct=rate_ctx.get("change_pct") if rate_ctx else None,
        dxy=dxy_ctx.get("value") if dxy_ctx else None,
        dxy_source=dxy_ctx.get("source") if dxy_ctx else None,
        dxy_change_pct=dxy_ctx.get("change_pct") if dxy_ctx else None,
        next_event_type=event_ctx.get("event_type") if event_ctx else None,
        next_event_hours=event_ctx.get("hours_to_event") if event_ctx else None,
        event_risk_active=bool(event_ctx and event_ctx.get("active")),
        event_risk_phase=event_ctx.get("phase") if event_ctx else None,
        data_quality=quality,
        gold_fetch_ts=gold_fetch_ts,
        fx_fetch_ts=fx_fetch_ts,
        macro_fetch_ts=macro_fetch_ts,
        gold_volume_ratio=gold_volume_ratio,
    )
    if cert_data:
        snap.cert_bid = cert_data.get("bid")
        snap.cert_ask = cert_data.get("ask")
        snap.cert_last = cert_data.get("last")
        snap.cert_spread_pct = cert_data.get("spread_pct")

    return snap
