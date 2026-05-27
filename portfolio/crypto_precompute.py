"""Crypto Deep Context Precomputer — BTC + ETH consolidated.

Mirrors `portfolio/metals_precompute.py` for the crypto subsystem.
Fetches shared crypto market data once and writes
`data/crypto_deep_context.json` (single file, contains BTC + ETH sections;
the schema mirrors how metals_precompute writes silver/gold separately
but unified into one file because BTC and ETH share heavily on data —
no second HTTP round-trip pays off).

Run manually:
    .venv/Scripts/python.exe portfolio/crypto_precompute.py

Auto-runs every 4h via the main loop's `_run_post_cycle()` (when wired in;
this module is currently inert until called).
"""
from __future__ import annotations

import datetime
import logging
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.file_utils import atomic_write_json, load_json

logger = logging.getLogger("portfolio.crypto_precompute")

_STATE_FILE = "data/crypto_precompute_state.json"
_OUTPUT_FILE = "data/crypto_deep_context.json"
_DEFAULT_INTERVAL_SEC = 4 * 3600  # 4 hours — aligned with L2 + evolution cycle

# Per-source refresh intervals (seconds)
_REFRESH_INTERVALS = {
    "fear_greed": 1 * 3600,          # alternative.me Fear & Greed: 1h
    "onchain": 6 * 3600,             # On-chain BTC: 6h
    "funding": 30 * 60,              # Binance FAPI funding: 30 min
    "btc_history": 1 * 3600,         # 1h candles: 1h
    "eth_history": 1 * 3600,
    "dxy": 4 * 3600,                 # DXY (yfinance): 4h
    "spy": 4 * 3600,                 # SPY (yfinance): 4h
    "gold": 4 * 3600,                # Gold (yfinance): 4h
}

_REQUEST_TIMEOUT = 15


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def maybe_precompute_crypto(config: Any = None) -> dict | None:
    """Run crypto precompute if enough time has elapsed since last run.

    Called from main loop. Self-checking — safe to call every cycle; only
    executes when the interval has elapsed. Mirror of
    `metals_precompute.maybe_precompute_metals`.
    """
    interval = _DEFAULT_INTERVAL_SEC
    if config:
        try:
            interval = config.get("crypto", {}).get(
                "precompute_interval_sec", _DEFAULT_INTERVAL_SEC
            )
        except AttributeError:
            interval = getattr(config, "crypto_precompute_interval_sec",
                               _DEFAULT_INTERVAL_SEC)

    state = load_json(_STATE_FILE, default={})
    last_run = state.get("last_run_epoch", 0)
    now = time.time()

    if (now - last_run) < interval:
        return None

    try:
        result = precompute(config)
        atomic_write_json(_STATE_FILE, {
            "last_run_epoch": now,
            "last_run_iso": datetime.datetime.now(datetime.UTC).isoformat(),
            "status": "ok",
        })
        logger.info("Crypto precompute completed (interval=%ds)", interval)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("Crypto precompute failed: %s", exc)
        atomic_write_json(_STATE_FILE, {
            "last_run_epoch": last_run,  # Don't reset — retry next cycle
            "last_run_iso": state.get("last_run_iso", ""),
            "status": f"error: {exc}",
            "last_error_epoch": now,
        })
        return None


def precompute(config: Any = None) -> dict[str, Any]:
    """Aggregate crypto deep context into the cached JSON file."""
    generated_at = datetime.datetime.now(datetime.UTC).isoformat()
    market = _fetch_market_data(config)
    ctx = _build_context(market, generated_at)
    atomic_write_json(_OUTPUT_FILE, ctx)
    logger.info("Crypto deep context written to %s", _OUTPUT_FILE)
    return ctx


# ---------------------------------------------------------------------------
# Market data fetch
# ---------------------------------------------------------------------------
def _fetch_market_data(config: Any = None) -> dict[str, Any]:
    """Fetch all shared crypto market data with graceful fallbacks.

    Each source is wrapped in try/except so a single network blip doesn't
    kill the whole precompute cycle. Failures are logged at WARNING and the
    affected fields land as None in the output (caller treats None as
    "no data" — same convention as metals_precompute).
    """
    out: dict[str, Any] = {
        "fear_greed": None,
        "btc_funding_rate": None,
        "eth_funding_rate": None,
        "btc_open_interest": None,
        "eth_open_interest": None,
        "onchain_btc": None,
        "btc_price_usd": None,
        "eth_price_usd": None,
        "btc_24h_pct": None,
        "eth_24h_pct": None,
        "dxy_close": None,
        "dxy_change_pct": None,
        "spy_close": None,
        "spy_change_pct": None,
        "gold_close": None,
        "gold_change_pct": None,
        "btc_dominance": None,
    }

    # Fear & Greed (existing helper in data/crypto_data.py)
    try:
        from data.crypto_data import get_fear_greed
        out["fear_greed"] = get_fear_greed()
    except Exception as exc:  # noqa: BLE001
        logger.warning("fear_greed fetch failed: %s", exc)

    # On-chain BTC summary
    try:
        from data.crypto_data import get_onchain_summary
        out["onchain_btc"] = get_onchain_summary()
    except Exception as exc:  # noqa: BLE001
        logger.warning("onchain_btc fetch failed: %s", exc)

    # Binance spot prices + 24h change
    try:
        import requests
        for sym, key_price, key_pct in (
            ("BTCUSDT", "btc_price_usd", "btc_24h_pct"),
            ("ETHUSDT", "eth_price_usd", "eth_24h_pct"),
        ):
            r = requests.get(
                "https://api.binance.com/api/v3/ticker/24hr",
                params={"symbol": sym},
                timeout=_REQUEST_TIMEOUT,
            )
            if r.status_code == 200:
                d = r.json()
                out[key_price] = float(d.get("lastPrice", 0)) or None
                out[key_pct] = float(d.get("priceChangePercent", 0)) or None
    except Exception as exc:  # noqa: BLE001
        logger.warning("binance spot fetch failed: %s", exc)

    # Binance FAPI funding rate + open interest
    try:
        import requests
        for sym, key_funding, key_oi in (
            ("BTCUSDT", "btc_funding_rate", "btc_open_interest"),
            ("ETHUSDT", "eth_funding_rate", "eth_open_interest"),
        ):
            try:
                r = requests.get(
                    "https://fapi.binance.com/fapi/v1/premiumIndex",
                    params={"symbol": sym},
                    timeout=_REQUEST_TIMEOUT,
                )
                if r.status_code == 200:
                    out[key_funding] = float(r.json().get("lastFundingRate", 0))
            except Exception:
                logger.debug("crypto_precompute: data parse error", exc_info=True)
            try:
                r = requests.get(
                    "https://fapi.binance.com/fapi/v1/openInterest",
                    params={"symbol": sym},
                    timeout=_REQUEST_TIMEOUT,
                )
                if r.status_code == 200:
                    out[key_oi] = float(r.json().get("openInterest", 0))
            except Exception:
                logger.debug("crypto_precompute: data parse error", exc_info=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("binance fapi fetch failed: %s", exc)

    # yfinance: DXY, SPY, gold (used by signals/crypto_cross_asset)
    try:
        import yfinance as yf
        for sym, key_close, key_pct in (
            ("DX-Y.NYB", "dxy_close", "dxy_change_pct"),
            ("SPY", "spy_close", "spy_change_pct"),
            ("GLD", "gold_close", "gold_change_pct"),
        ):
            try:
                hist = yf.Ticker(sym).history(period="5d", interval="1d")
                if hist is not None and not hist.empty and len(hist) >= 2:
                    last = float(hist["Close"].iloc[-1])
                    prev = float(hist["Close"].iloc[-2])
                    out[key_close] = round(last, 4)
                    out[key_pct] = round((last / prev - 1.0) * 100.0, 3) if prev else None
            except Exception:
                logger.debug("crypto_precompute: data parse error", exc_info=True)
    except ImportError:
        logger.info("yfinance not available — skipping DXY/SPY/gold")
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance batch fetch failed: %s", exc)

    # BTC dominance (CoinGecko global) — informational only
    try:
        import requests
        r = requests.get(
            "https://api.coingecko.com/api/v3/global",
            timeout=_REQUEST_TIMEOUT,
        )
        if r.status_code == 200:
            d = r.json().get("data", {})
            out["btc_dominance"] = d.get("market_cap_percentage", {}).get("btc")
    except Exception as exc:  # noqa: BLE001
        logger.warning("btc_dominance fetch failed: %s", exc)

    return out


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------
def _build_context(market: dict[str, Any], generated_at: str) -> dict[str, Any]:
    return {
        "generated_at": generated_at,
        "schema_version": 1,
        "shared": {
            "fear_greed": market.get("fear_greed"),
            "btc_dominance": market.get("btc_dominance"),
            "dxy": {
                "close": market.get("dxy_close"),
                "change_pct": market.get("dxy_change_pct"),
            },
            "spy": {
                "close": market.get("spy_close"),
                "change_pct": market.get("spy_change_pct"),
            },
            "gold": {
                "close": market.get("gold_close"),
                "change_pct": market.get("gold_change_pct"),
            },
        },
        "btc": {
            "price_usd": market.get("btc_price_usd"),
            "change_24h_pct": market.get("btc_24h_pct"),
            "funding_rate": market.get("btc_funding_rate"),
            "open_interest": market.get("btc_open_interest"),
            "onchain": market.get("onchain_btc"),
        },
        "eth": {
            "price_usd": market.get("eth_price_usd"),
            "change_24h_pct": market.get("eth_24h_pct"),
            "funding_rate": market.get("eth_funding_rate"),
            "open_interest": market.get("eth_open_interest"),
        },
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = precompute()
    print("Crypto deep context written to", _OUTPUT_FILE)
    print("Sections:", list(result.keys()))
