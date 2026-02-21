"""USD/SEK exchange rate fetching with caching and staleness alerts."""

import logging
import time

from portfolio.http_retry import fetch_with_retry
from portfolio.api_utils import load_config as _load_config

logger = logging.getLogger("portfolio.fx_rates")

_fx_cache = {"rate": None, "time": 0}
_FX_STALE_THRESHOLD = 7200  # 2 hours — warn if FX rate hasn't been refreshed


def fetch_usd_sek():
    now = time.time()
    if _fx_cache["rate"] and now - _fx_cache["time"] < 3600:
        age_secs = now - _fx_cache["time"]
        if age_secs > _FX_STALE_THRESHOLD:
            logger.warning(f"FX rate is stale ({age_secs / 3600:.1f}h old)")
        return _fx_cache["rate"]
    try:
        r = fetch_with_retry(
            "https://api.frankfurter.app/latest",
            params={"from": "USD", "to": "SEK"},
            timeout=10,
        )
        if r is None:
            raise ConnectionError("FX rate request failed after retries")
        r.raise_for_status()
        rate = float(r.json()["rates"]["SEK"])
        _fx_cache["rate"] = rate
        _fx_cache["time"] = now
        return rate
    except Exception as e:
        logger.warning(f"FX rate fetch failed: {e}")
    if _fx_cache["rate"]:
        age_secs = now - _fx_cache["time"]
        if age_secs > _FX_STALE_THRESHOLD:
            logger.warning(f"Using stale FX rate ({age_secs / 3600:.1f}h old)")
            _fx_alert_telegram(age_secs)
        return _fx_cache["rate"]
    # Last resort: hardcoded fallback
    logger.warning("Using hardcoded FX fallback rate 10.50 SEK — no cached or live rate available")
    _fx_alert_telegram(None)
    return 10.50


def _fx_alert_telegram(age_secs):
    """Send a one-shot Telegram alert about FX rate issues. Fires at most once per 4h."""
    global _fx_cache
    last_alert = _fx_cache.get("_last_fx_alert", 0)
    now = time.time()
    if now - last_alert < 14400:  # 4h cooldown between alerts
        return
    try:
        config = _load_config()
        if age_secs is not None:
            msg = f"_FX WARNING: USD/SEK rate is {age_secs / 3600:.1f}h stale. API may be down._"
        else:
            msg = "_FX WARNING: Using hardcoded fallback rate 10.50 SEK. No live or cached rate available._"
        # Import send_telegram late to avoid circular imports
        from portfolio.telegram_notifications import send_telegram
        send_telegram(msg, config)
        _fx_cache["_last_fx_alert"] = now
    except Exception:
        pass  # non-critical
