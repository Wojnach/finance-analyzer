"""MSTR Deep Context Precomputer — leveraged BTC proxy stock.

Mirrors `portfolio/metals_precompute.py` and `portfolio/crypto_precompute.py`.
Produces `data/mstr_deep_context.json` with NAV premium, BTC correlation,
options skew, short interest, and analyst consensus — the data layer that
`portfolio/mstr_loop/data_provider.py` and Layer 2 currently fetch ad-hoc.

Run manually:
    .venv/Scripts/python.exe portfolio/mstr_precompute.py

Auto-runs every 4h via the main loop's `_run_post_cycle()` (when wired in).
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

logger = logging.getLogger("portfolio.mstr_precompute")

_STATE_FILE = "data/mstr_precompute_state.json"
_OUTPUT_FILE = "data/mstr_deep_context.json"
_DEFAULT_INTERVAL_SEC = 4 * 3600  # 4h

# MSTR's BTC holdings as of FY2025 Q4 (refresh manually after 8-K filings).
# 2026-01: held ~424,150 BTC (approx). Treat as a config knob; precompute
# logs the value used so we can audit if NAV premium math drifts.
_DEFAULT_BTC_HOLDINGS = 471_107  # 2026-04 estimate; update after next 8-K
_DEFAULT_DEBT_USD = 8_500_000_000  # rough convertible + senior notes outstanding
_DEFAULT_SHARES_OUTSTANDING = 287_000_000

_REQUEST_TIMEOUT = 15


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def maybe_precompute_mstr(config: Any = None) -> dict | None:
    interval = _DEFAULT_INTERVAL_SEC
    if config:
        try:
            interval = config.get("mstr", {}).get(
                "precompute_interval_sec", _DEFAULT_INTERVAL_SEC
            )
        except AttributeError:
            interval = _DEFAULT_INTERVAL_SEC

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
        logger.info("MSTR precompute completed (interval=%ds)", interval)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("MSTR precompute failed: %s", exc)
        atomic_write_json(_STATE_FILE, {
            "last_run_epoch": last_run,
            "last_run_iso": state.get("last_run_iso", ""),
            "status": f"error: {exc}",
            "last_error_epoch": now,
        })
        return None


def precompute(config: Any = None) -> dict[str, Any]:
    generated_at = datetime.datetime.now(datetime.UTC).isoformat()
    market = _fetch_market_data(config)

    nav = _compute_nav_premium(
        mstr_price=market.get("mstr_price"),
        btc_price=market.get("btc_price_usd"),
        btc_holdings=market.get("btc_holdings", _DEFAULT_BTC_HOLDINGS),
        debt_usd=market.get("debt_usd", _DEFAULT_DEBT_USD),
        shares_outstanding=market.get("shares_outstanding",
                                      _DEFAULT_SHARES_OUTSTANDING),
    )

    ctx = {
        "generated_at": generated_at,
        "schema_version": 1,
        "underlying_btc": {
            "price_usd": market.get("btc_price_usd"),
            "change_24h_pct": market.get("btc_24h_pct"),
        },
        "stock": {
            "price": market.get("mstr_price"),
            "change_pct": market.get("mstr_change_pct"),
            "volume": market.get("mstr_volume"),
            "high_52w": market.get("mstr_52w_high"),
            "low_52w": market.get("mstr_52w_low"),
        },
        "nav": nav,
        "options": market.get("options"),
        "correlation_btc_30d": market.get("correlation_btc_30d"),
        "short_interest_pct": market.get("short_interest_pct"),
        "analyst_consensus": market.get("analyst_consensus"),
    }
    atomic_write_json(_OUTPUT_FILE, ctx)
    logger.info("MSTR deep context written to %s", _OUTPUT_FILE)
    return ctx


# ---------------------------------------------------------------------------
# Market data fetch
# ---------------------------------------------------------------------------
def _fetch_market_data(config: Any = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "mstr_price": None,
        "mstr_change_pct": None,
        "mstr_volume": None,
        "mstr_52w_high": None,
        "mstr_52w_low": None,
        "btc_price_usd": None,
        "btc_24h_pct": None,
        "options": None,
        "correlation_btc_30d": None,
        "short_interest_pct": None,
        "analyst_consensus": None,
        "btc_holdings": _DEFAULT_BTC_HOLDINGS,
        "debt_usd": _DEFAULT_DEBT_USD,
        "shares_outstanding": _DEFAULT_SHARES_OUTSTANDING,
    }

    # MSTR price + history via yfinance
    try:
        import yfinance as yf
        tk = yf.Ticker("MSTR")
        hist = tk.history(period="60d", interval="1d")
        if hist is not None and not hist.empty:
            last = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else last
            out["mstr_price"] = round(last, 4)
            out["mstr_change_pct"] = round((last / prev - 1.0) * 100.0, 3) if prev else None
            out["mstr_volume"] = int(hist["Volume"].iloc[-1])

        info = getattr(tk, "info", {}) or {}
        out["mstr_52w_high"] = info.get("fiftyTwoWeekHigh")
        out["mstr_52w_low"] = info.get("fiftyTwoWeekLow")
        out["short_interest_pct"] = info.get("shortPercentOfFloat")
        rec = info.get("recommendationKey") or info.get("recommendationMean")
        if rec is not None:
            out["analyst_consensus"] = rec

        # 30d correlation with BTC
        try:
            btc_hist = yf.Ticker("BTC-USD").history(period="60d", interval="1d")
            if btc_hist is not None and not btc_hist.empty and not hist.empty:
                import numpy as np
                m_ret = hist["Close"].pct_change().dropna().tail(30)
                b_ret = btc_hist["Close"].pct_change().dropna().tail(30)
                # align by date
                joined = m_ret.to_frame("m").join(b_ret.to_frame("b"), how="inner")
                if len(joined) >= 10:
                    corr = float(joined["m"].corr(joined["b"]))
                    if not np.isnan(corr):
                        out["correlation_btc_30d"] = round(corr, 3)
        except Exception:  # noqa: BLE001
            pass

        # Options surface (nearest expiry — Greeks not computed here)
        try:
            expiries = tk.options or []
            if expiries:
                expiry = expiries[0]
                chain = tk.option_chain(expiry)
                calls = chain.calls
                puts = chain.puts
                if not calls.empty and not puts.empty:
                    call_iv = float(calls["impliedVolatility"].mean())
                    put_iv = float(puts["impliedVolatility"].mean())
                    out["options"] = {
                        "expiry": expiry,
                        "call_iv_mean": round(call_iv, 4),
                        "put_iv_mean": round(put_iv, 4),
                        "iv_skew": round(put_iv - call_iv, 4),
                        "n_calls": int(len(calls)),
                        "n_puts": int(len(puts)),
                    }
        except Exception:  # noqa: BLE001
            pass
    except ImportError:
        logger.info("yfinance not available — MSTR fetch skipped")
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance MSTR fetch failed: %s", exc)

    # BTC price (Binance, fast + free)
    try:
        import requests
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/24hr",
            params={"symbol": "BTCUSDT"},
            timeout=_REQUEST_TIMEOUT,
        )
        if r.status_code == 200:
            d = r.json()
            out["btc_price_usd"] = float(d.get("lastPrice", 0)) or None
            out["btc_24h_pct"] = float(d.get("priceChangePercent", 0)) or None
    except Exception as exc:  # noqa: BLE001
        logger.warning("binance BTC fetch failed: %s", exc)

    return out


# ---------------------------------------------------------------------------
# NAV premium math
# ---------------------------------------------------------------------------
def _compute_nav_premium(
    mstr_price: float | None,
    btc_price: float | None,
    btc_holdings: float,
    debt_usd: float,
    shares_outstanding: float,
) -> dict[str, Any]:
    """NAV premium = (market cap - software business value - net BTC value) / net BTC value.

    Simplification used here: ignore the small software-segment value (typical
    estimates ~$1B vs $40-100B of BTC NAV). Approximation:

        net_btc_nav_usd = btc_holdings * btc_price - debt_usd
        market_cap_usd  = mstr_price * shares_outstanding
        premium         = market_cap / net_btc_nav - 1.0

    Returns dict with all intermediate values so the caller can render or
    sanity-check. None values pass through cleanly.
    """
    if (mstr_price is None or btc_price is None
            or shares_outstanding <= 0 or btc_holdings <= 0):
        return {
            "mstr_price": mstr_price,
            "btc_price": btc_price,
            "btc_holdings": btc_holdings,
            "debt_usd": debt_usd,
            "shares_outstanding": shares_outstanding,
            "net_btc_nav_usd": None,
            "market_cap_usd": None,
            "premium": None,
        }

    net_btc_nav_usd = btc_holdings * btc_price - debt_usd
    market_cap_usd = mstr_price * shares_outstanding
    if net_btc_nav_usd <= 0:
        premium = None  # underwater on debt — premium math undefined
    else:
        premium = round(market_cap_usd / net_btc_nav_usd - 1.0, 4)

    return {
        "mstr_price": mstr_price,
        "btc_price": btc_price,
        "btc_holdings": btc_holdings,
        "debt_usd": debt_usd,
        "shares_outstanding": shares_outstanding,
        "net_btc_nav_usd": round(net_btc_nav_usd, 2),
        "market_cap_usd": round(market_cap_usd, 2),
        "premium": premium,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = precompute()
    print("MSTR deep context written to", _OUTPUT_FILE)
    print("Sections:", list(result.keys()))
