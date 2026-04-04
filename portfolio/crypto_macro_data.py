"""Crypto macro data — options, exchange reserves, ETF flows, gold-BTC ratio.

Fetches data from free, no-auth-required APIs:
  - Deribit public REST API (options: max pain, OI, put/call ratio)
  - BGeometrics exchange netflow trend (already fetched, we track history)
  - Gold-BTC ratio (computed from existing price data)

Cache: 15 min for options (markets move), 1h for ratios.
All fetches use http_retry for resilience.

Usage:
    from portfolio.crypto_macro_data import get_crypto_macro_data
    data = get_crypto_macro_data("BTC-USD")
"""

import json
import logging
import time
from pathlib import Path

from portfolio.file_utils import atomic_append_jsonl
from portfolio.http_retry import fetch_json
from portfolio.shared_state import _cached

logger = logging.getLogger("portfolio.crypto_macro_data")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Cache TTLs
OPTIONS_TTL = 900       # 15 min — options data changes slowly
RATIO_TTL = 3600        # 1h — gold/btc ratio
NETFLOW_HIST_TTL = 3600 # 1h — netflow history

# Deribit public API (no auth required)
DERIBIT_BASE = "https://www.deribit.com/api/v2/public"

# Persistent netflow history file
NETFLOW_HISTORY_FILE = DATA_DIR / "exchange_netflow_history.jsonl"


# ---------------------------------------------------------------------------
# Deribit Options (free, no auth)
# ---------------------------------------------------------------------------

def _fetch_deribit_options(currency="BTC"):
    """Fetch options book summary from Deribit public API.

    Returns dict with max_pain, total_oi, put_call_ratio, nearest_expiry,
    or None on failure.
    """
    url = f"{DERIBIT_BASE}/get_book_summary_by_currency"
    params = {"currency": currency, "kind": "option"}
    data = fetch_json(url, params=params, timeout=20, retries=2,
                      label="deribit:options_summary")
    if not data:
        return None

    result_list = data.get("result")
    if not result_list or not isinstance(result_list, list):
        return None

    # Parse instrument names: BTC-28MAR26-70000-C
    # Format: {currency}-{expiry}-{strike}-{C|P}
    import datetime
    from collections import defaultdict

    expiry_data = defaultdict(lambda: {"calls": defaultdict(float),
                                        "puts": defaultdict(float),
                                        "total_call_oi": 0.0,
                                        "total_put_oi": 0.0})

    for item in result_list:
        name = item.get("instrument_name", "")
        oi = item.get("open_interest", 0) or 0
        if oi <= 0:
            continue

        parts = name.split("-")
        if len(parts) != 4:
            continue

        _, expiry_str, strike_str, option_type = parts
        try:
            strike = float(strike_str)
        except (ValueError, TypeError):
            continue

        ed = expiry_data[expiry_str]
        if option_type == "C":
            ed["calls"][strike] += oi
            ed["total_call_oi"] += oi
        elif option_type == "P":
            ed["puts"][strike] += oi
            ed["total_put_oi"] += oi

    if not expiry_data:
        return None

    # Find nearest expiry by parsing date strings
    def _parse_expiry(s):
        """Parse Deribit expiry like '28MAR26' to a date."""
        try:
            return datetime.datetime.strptime(s, "%d%b%y").date()
        except ValueError:
            return None

    now = datetime.date.today()
    nearest_expiry = None
    nearest_date = None
    for exp_str in expiry_data:
        d = _parse_expiry(exp_str)
        if d and d >= now:
            if nearest_date is None or d < nearest_date:
                nearest_date = d
                nearest_expiry = exp_str

    if not nearest_expiry:
        # Fall back to first expiry with most OI
        nearest_expiry = max(expiry_data,
                             key=lambda e: expiry_data[e]["total_call_oi"] +
                                           expiry_data[e]["total_put_oi"])

    ed = expiry_data[nearest_expiry]
    all_strikes = sorted(set(list(ed["calls"].keys()) + list(ed["puts"].keys())))

    if not all_strikes:
        return None

    # Compute max pain: strike where total loss for option buyers is maximized
    # For each candidate strike price:
    #   Call loss = sum of call_oi * max(0, strike - candidate) for all strikes
    #   Put loss = sum of put_oi * max(0, candidate - strike) for all strikes
    #   Total pain = call_loss + put_loss
    # Max pain = strike with highest total pain

    max_pain_strike = None
    max_pain_value = -1

    for candidate in all_strikes:
        total_pain = 0
        for strike in all_strikes:
            call_oi = ed["calls"].get(strike, 0)
            put_oi = ed["puts"].get(strike, 0)
            # Call holder loses if price < strike (ITM calls lose nothing)
            # Actually: option BUYER pain = how much the option expires worthless
            # Max pain for call buyers at candidate: if candidate < strike, call expires OTM
            # Call buyer loss = call_oi * max(0, strike - candidate) ... no wait
            # Standard max pain:
            # For calls at strike K: if expiry price P < K, call expires worthless,
            #   pain = 0 (buyer already lost premium, not counted)
            # For calls at strike K: if P >= K, call is ITM, pain to SELLERS
            # Actually the standard approach:
            # For each candidate expiry price P:
            #   call_pain = sum(call_oi[K] * max(0, P - K)) for all K
            #   put_pain = sum(put_oi[K] * max(0, K - P)) for all K
            #   total = call_pain + put_pain (intrinsic value = money paid out)
            # Max pain = P that MINIMIZES total payout (i.e., max pain for buyers)
            call_pain = call_oi * max(0, candidate - strike)
            put_pain = put_oi * max(0, strike - candidate)
            total_pain += call_pain + put_pain

        # Max pain for buyers = strike where payout is MINIMIZED
        # So we want the candidate with MINIMUM total_pain
        if max_pain_strike is None or total_pain < max_pain_value:
            max_pain_value = total_pain
            max_pain_strike = candidate

    total_call_oi = ed["total_call_oi"]
    total_put_oi = ed["total_put_oi"]
    put_call_ratio = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else None

    # Also compute total OI across ALL expiries
    grand_call_oi = sum(e["total_call_oi"] for e in expiry_data.values())
    grand_put_oi = sum(e["total_put_oi"] for e in expiry_data.values())
    grand_pcr = round(grand_put_oi / grand_call_oi, 3) if grand_call_oi > 0 else None

    days_to_expiry = (nearest_date - now).days if nearest_date else None

    return {
        "max_pain": max_pain_strike,
        "nearest_expiry": nearest_expiry,
        "days_to_expiry": days_to_expiry,
        "nearest_call_oi": total_call_oi,
        "nearest_put_oi": total_put_oi,
        "nearest_pcr": put_call_ratio,
        "total_call_oi": grand_call_oi,
        "total_put_oi": grand_put_oi,
        "total_pcr": grand_pcr,
    }


def get_deribit_options(currency="BTC"):
    """Get Deribit options data with caching."""
    return _cached(f"deribit_options_{currency}", OPTIONS_TTL,
                   _fetch_deribit_options, currency)


# ---------------------------------------------------------------------------
# Gold-BTC Ratio (computed from existing price data)
# ---------------------------------------------------------------------------

def compute_gold_btc_ratio():
    """Compute Gold/BTC ratio from latest agent_summary prices.

    Returns dict with current ratio, 7d/14d/30d history from price snapshots,
    and trend direction.
    """
    try:
        from portfolio.file_utils import load_json
        summary = load_json(DATA_DIR / "agent_summary_compact.json")
        if not summary:
            return None

        signals = summary.get("signals", {})
        btc_price = signals.get("BTC-USD", {}).get("price_usd")
        gold_price = signals.get("XAU-USD", {}).get("price_usd")

        if not btc_price or not gold_price or btc_price <= 0:
            return None

        current_ratio = gold_price / btc_price

        # Try to get historical ratios from price snapshots
        history = _load_ratio_history()
        _append_ratio_history(current_ratio, gold_price, btc_price)

        # Compute trend from history
        trend = "flat"
        ratio_7d = None
        ratio_14d = None

        if history and len(history) >= 2:
            # Get ratio from ~7 days ago (assuming hourly snapshots)
            now = time.time()
            for entry in reversed(history):
                age_days = (now - entry.get("ts", 0)) / 86400
                if age_days >= 7 and ratio_7d is None:
                    ratio_7d = entry.get("ratio")
                if age_days >= 14 and ratio_14d is None:
                    ratio_14d = entry.get("ratio")
                    break

            if ratio_7d:
                change_7d = (current_ratio - ratio_7d) / ratio_7d
                if change_7d > 0.02:
                    trend = "gold_outperforming"  # gold gaining vs BTC
                elif change_7d < -0.02:
                    trend = "btc_outperforming"   # BTC gaining vs gold = rotation
                else:
                    trend = "flat"

        return {
            "gold_btc_ratio": round(current_ratio, 6),
            "gold_price": gold_price,
            "btc_price": btc_price,
            "ratio_7d_ago": ratio_7d,
            "ratio_14d_ago": ratio_14d,
            "trend": trend,
        }
    except Exception:
        logger.warning("Failed to compute gold/BTC ratio", exc_info=True)
        return None


RATIO_HISTORY_FILE = DATA_DIR / "gold_btc_ratio_history.jsonl"


def _load_ratio_history(max_age_days=30):
    """Load gold/BTC ratio history from JSONL file."""
    try:
        if not RATIO_HISTORY_FILE.exists():
            return []
        cutoff = time.time() - (max_age_days * 86400)
        entries = []
        with open(RATIO_HISTORY_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("ts", 0) >= cutoff:
                        entries.append(entry)
                except (json.JSONDecodeError, KeyError):
                    continue
        return entries
    except Exception as e:
        logger.warning("Gold/BTC ratio history load failed: %s", e, exc_info=True)
        return []


def _append_ratio_history(ratio, gold_price, btc_price):
    """Append current ratio to history file (at most once per hour)."""
    try:
        # Check if we already have a recent entry (within 1 hour)
        history = _load_ratio_history(max_age_days=1)
        if history:
            latest_ts = history[-1].get("ts", 0)
            if time.time() - latest_ts < 3600:
                return  # Already have a recent entry

        entry = {
            "ts": time.time(),
            "ratio": round(ratio, 6),
            "gold": gold_price,
            "btc": btc_price,
        }
        atomic_append_jsonl(RATIO_HISTORY_FILE, entry)
    except Exception:
        logger.warning("Failed to append ratio history", exc_info=True)


# ---------------------------------------------------------------------------
# Exchange Netflow Trend (from BGeometrics data we already collect)
# ---------------------------------------------------------------------------

NETFLOW_HISTORY_MAX_DAYS = 30


def get_exchange_netflow_trend():
    """Analyze exchange netflow trend from on-chain data.

    Uses the BGeometrics netflow we already fetch in onchain_data.py.
    Tracks history in a JSONL file to detect multi-day accumulation/distribution.

    Returns dict with trend direction and strength.
    """
    try:
        from portfolio.onchain_data import get_onchain_data
        onchain = get_onchain_data()

        netflow = None
        if onchain:
            netflow = onchain.get("netflow")
            if netflow is not None:
                _append_netflow_history(netflow)

        # Load history and compute trend
        history = _load_netflow_history()
        if not history or len(history) < 3:
            return {
                "current_netflow": netflow,
                "trend": "insufficient_data",
                "consecutive_negative": 0,
                "sum_7d": None,
            }

        now = time.time()
        recent_7d = [e for e in history
                     if now - e.get("ts", 0) < 7 * 86400]
        recent_14d = [e for e in history
                      if now - e.get("ts", 0) < 14 * 86400]

        # Count consecutive negative netflows (accumulation)
        consecutive_neg = 0
        for entry in reversed(history):
            if entry.get("netflow", 0) < 0:
                consecutive_neg += 1
            else:
                break

        sum_7d = sum(e.get("netflow", 0) for e in recent_7d) if recent_7d else None
        sum_14d = sum(e.get("netflow", 0) for e in recent_14d) if recent_14d else None

        # Determine trend
        trend = "neutral"
        if sum_7d is not None:
            if sum_7d < -1000:  # > 1000 BTC net outflow in 7d
                trend = "strong_accumulation"
            elif sum_7d < -100:
                trend = "accumulation"
            elif sum_7d > 1000:
                trend = "strong_distribution"
            elif sum_7d > 100:
                trend = "distribution"

        return {
            "current_netflow": netflow,
            "trend": trend,
            "consecutive_negative": consecutive_neg,
            "sum_7d": round(sum_7d, 1) if sum_7d else None,
            "sum_14d": round(sum_14d, 1) if sum_14d else None,
            "data_points_7d": len(recent_7d),
        }
    except Exception:
        logger.warning("Failed to get netflow trend", exc_info=True)
        return None


def _load_netflow_history():
    """Load netflow history from JSONL file."""
    try:
        if not NETFLOW_HISTORY_FILE.exists():
            return []
        cutoff = time.time() - (NETFLOW_HISTORY_MAX_DAYS * 86400)
        entries = []
        with open(NETFLOW_HISTORY_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("ts", 0) >= cutoff:
                        entries.append(entry)
                except (json.JSONDecodeError, KeyError):
                    continue
        return entries
    except Exception as e:
        logger.warning("Exchange netflow history load failed: %s", e, exc_info=True)
        return []


def _append_netflow_history(netflow):
    """Append netflow data point (at most once per 6 hours)."""
    try:
        history = _load_netflow_history()
        if history:
            latest_ts = history[-1].get("ts", 0)
            if time.time() - latest_ts < 21600:  # 6h
                return

        entry = {"ts": time.time(), "netflow": netflow}
        atomic_append_jsonl(NETFLOW_HISTORY_FILE, entry)
    except Exception:
        logger.warning("Failed to append netflow history", exc_info=True)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def get_crypto_macro_data(ticker="BTC-USD"):
    """Get all crypto macro data for a ticker.

    Returns dict with options, gold_btc_ratio, netflow_trend, or None on failure.
    """
    currency = "BTC" if "BTC" in ticker else "ETH" if "ETH" in ticker else None
    if not currency:
        return None

    result = {}

    # Options data (BTC and ETH both available on Deribit)
    options = get_deribit_options(currency)
    if options:
        result["options"] = options

    # Gold-BTC ratio (only meaningful for BTC, but include for ETH too)
    ratio = _cached("gold_btc_ratio", RATIO_TTL, compute_gold_btc_ratio)
    if ratio:
        result["gold_btc_ratio"] = ratio

    # Exchange netflow trend (BTC only from BGeometrics)
    if currency == "BTC":
        netflow = _cached("exchange_netflow_trend", NETFLOW_HIST_TTL,
                          get_exchange_netflow_trend)
        if netflow:
            result["netflow_trend"] = netflow

    return result if result else None
