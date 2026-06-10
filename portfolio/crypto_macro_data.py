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

    def _chain_oi(exp_str):
        e = expiry_data[exp_str]
        return e["total_call_oi"] + e["total_put_oi"]

    future = {}
    for exp_str in expiry_data:
        d = _parse_expiry(exp_str)
        if d and d >= now:
            future[exp_str] = d

    # Raw nearest expiry (any listing, including Deribit dailies). Kept ONLY
    # for the expiry-proximity sub-signal in signals/crypto_macro.py, which
    # needs to know whether the true nearest expiry is a genuine quarterly.
    nearest_expiry = min(future, key=lambda s: future[s]) if future else None
    nearest_date = future.get(nearest_expiry)

    # 2026-06-10 (audit batch 2): max pain / PCR / OI metrics are now computed
    # on the nearest MONTHLY/quarterly chain instead of the raw nearest expiry.
    # Deribit lists daily expiries, so the raw nearest is a thin 0-1 DTE chain
    # with unrepresentative OI — max pain and nearest_pcr were day-to-day noise
    # and the "gravity weakens >7d from expiry" gate downstream could never
    # trip. Monthly/quarterly expiries are the last Friday of the month, which
    # always lands on day >= 22; among same-window candidates we take the chain
    # with the largest OI so a thin late-month daily can't shadow the real
    # monthly.
    monthly_like = {s: d for s, d in future.items() if d.day >= 22}
    metrics_expiry = None
    if monthly_like:
        first_monthly_date = min(monthly_like.values())
        same_window = [s for s, d in monthly_like.items()
                       if (d - first_monthly_date).days <= 7]
        metrics_expiry = max(same_window, key=_chain_oi)
    elif future:
        # No monthly-like listing (sparse/odd API response) — most liquid
        # future chain is the least-bad approximation.
        metrics_expiry = max(future, key=_chain_oi)

    if not metrics_expiry:
        # Fall back to the expiry with most OI (all listings already expired —
        # stale API data; matches pre-2026-06-10 fallback behavior).
        metrics_expiry = max(expiry_data, key=_chain_oi)
    if not nearest_expiry:
        nearest_expiry = metrics_expiry
        nearest_date = _parse_expiry(metrics_expiry)
        if nearest_date and nearest_date < now:
            nearest_date = None

    metrics_date = future.get(metrics_expiry)

    ed = expiry_data[metrics_expiry]
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

    # Metrics-chain DTE (monthly/quarterly chain — drives the options-gravity
    # ">7d from expiry" gate downstream).
    days_to_expiry = (metrics_date - now).days if metrics_date else None

    # Raw nearest-expiry context for the expiry-proximity sub-signal:
    # DTE, quarterly classification (last Friday of Mar/Jun/Sep/Dec always
    # lands on day >= 22), and the chain's share of grand OI so a daily
    # expiry can't masquerade as a meaningful quarterly.
    nearest_days = (nearest_date - now).days if nearest_date else None
    nearest_is_quarterly = bool(
        nearest_date
        and nearest_date.month in (3, 6, 9, 12)
        and nearest_date.day >= 22
    )
    grand_oi = grand_call_oi + grand_put_oi
    raw_ed = expiry_data[nearest_expiry]
    raw_oi = raw_ed["total_call_oi"] + raw_ed["total_put_oi"]
    nearest_oi_share = round(raw_oi / grand_oi, 4) if grand_oi > 0 else None

    return {
        # 2026-06-10 (audit batch 2): max_pain / days_to_expiry / nearest_*_oi
        # / nearest_pcr describe the monthly-or-quarterly metrics chain (see
        # metrics_expiry selection above), NOT the raw nearest daily expiry.
        # The nearest_expiry* fields below keep the raw nearest-listing info.
        "max_pain": max_pain_strike,
        "metrics_expiry": metrics_expiry,
        "days_to_expiry": days_to_expiry,
        "nearest_call_oi": total_call_oi,
        "nearest_put_oi": total_put_oi,
        "nearest_pcr": put_call_ratio,
        "total_call_oi": grand_call_oi,
        "total_put_oi": grand_put_oi,
        "total_pcr": grand_pcr,
        "nearest_expiry": nearest_expiry,
        "nearest_expiry_days": nearest_days,
        "nearest_is_quarterly": nearest_is_quarterly,
        "nearest_expiry_oi_share": nearest_oi_share,
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

# 2026-06-10 (audit batch 2): staleness contract for the netflow feed. The
# BGeometrics /v1/exchange-netflow fetch died silently around 2026-04-10
# (exchange_netflow_history.jsonl froze at one entry) and nothing alerted for
# two months. When the newest history entry is older than this, we append a
# critical_errors.jsonl row (category data_feed_stale), at most once per the
# same window so the journal doesn't flood.
NETFLOW_STALE_AFTER_DAYS = 7
_NETFLOW_STALE_STATE_FILE = DATA_DIR / "netflow_staleness_state.json"


def _maybe_alert_netflow_stale(history):
    """Append a critical_errors row when netflow history is stale (>7d).

    Called from get_exchange_netflow_trend (1h cache upstream, so roughly
    hourly). Dedup: persists the last alert ts in a small state file and
    re-alerts at most once per NETFLOW_STALE_AFTER_DAYS. Never raises.
    """
    try:
        now = time.time()
        latest_ts = max((e.get("ts", 0) for e in history), default=0)
        age_days = (now - latest_ts) / 86400 if latest_ts else float("inf")
        if age_days < NETFLOW_STALE_AFTER_DAYS:
            return
        from portfolio.file_utils import atomic_write_json, load_json
        state = load_json(_NETFLOW_STALE_STATE_FILE, default={}) or {}
        last_alert = float(state.get("last_alert_ts", 0) or 0)
        if now - last_alert < NETFLOW_STALE_AFTER_DAYS * 86400:
            return
        from portfolio.claude_gate import record_critical_error
        if latest_ts:
            latest_str = (
                f"{time.strftime('%Y-%m-%d', time.gmtime(latest_ts))} "
                f"({age_days:.0f}d old, threshold {NETFLOW_STALE_AFTER_DAYS}d)"
            )
        else:
            latest_str = "none — history file empty/missing"
        record_critical_error(
            category="data_feed_stale",
            caller="crypto_macro_data.get_exchange_netflow_trend",
            message=(
                f"BGeometrics exchange-netflow feed stale: newest history "
                f"entry {latest_str}. crypto_macro exchange_netflow "
                f"sub-vote and onchain_btc netflow interp are silently HOLD."
            ),
            context={
                "history_file": str(NETFLOW_HISTORY_FILE),
                "latest_entry_ts": latest_ts,
                "age_days": round(age_days, 1) if latest_ts else None,
            },
        )
        state["last_alert_ts"] = now
        atomic_write_json(_NETFLOW_STALE_STATE_FILE, state)
    except Exception:
        logger.warning("Netflow staleness alert failed", exc_info=True)


def _daily_netflow_series(history):
    """Collapse 6h-cadence history samples to one value per UTC day.

    2026-06-10 (audit batch 2): BGeometrics netflow is a DAILY metric that
    _append_netflow_history samples up to 4x/day, so counting raw samples as
    "days" made the consecutive-negative BUY threshold fire ~4x too easily
    (5 "days" = ~30h) and inflated the sum_7d thresholds by the same factor.
    The last sample of each UTC day wins (latest revision of the daily value).

    Returns a list of (date, netflow) tuples in ascending date order.
    """
    by_day = {}
    for entry in history:
        ts = entry.get("ts", 0)
        if not ts:
            continue
        day = time.strftime("%Y-%m-%d", time.gmtime(ts))
        by_day[day] = entry.get("netflow", 0)
    return sorted(by_day.items())


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

        # Staleness contract — alert even on the insufficient-data path
        # (a frozen one-entry history is exactly the failure mode).
        _maybe_alert_netflow_stale(history)

        if not history or len(history) < 3:
            return {
                "current_netflow": netflow,
                "trend": "insufficient_data",
                "consecutive_negative": 0,
                "sum_7d": None,
            }

        # 2026-06-10 (audit batch 2): aggregate to one sample per UTC day so
        # "consecutive_negative" and the 7d/14d sums genuinely count DAYS —
        # consumers (crypto_macro._exchange_netflow_signal, _NETFLOW_ACCUM_DAYS)
        # always interpreted them as days.
        daily = _daily_netflow_series(history)
        cutoff_7d = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 7 * 86400))
        cutoff_14d = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 14 * 86400))
        recent_7d = [v for d, v in daily if d >= cutoff_7d]
        recent_14d = [v for d, v in daily if d >= cutoff_14d]

        # Count consecutive negative netflow DAYS (accumulation), walking back
        # from the most recent day. A calendar gap breaks the streak — missing
        # data is not evidence of accumulation.
        consecutive_neg = 0
        prev_day = None
        for day, value in reversed(daily):
            if prev_day is not None:
                gap = (
                    time.mktime(time.strptime(prev_day, "%Y-%m-%d"))
                    - time.mktime(time.strptime(day, "%Y-%m-%d"))
                ) / 86400
                if gap > 1.5:
                    break
            if value is not None and value < 0:
                consecutive_neg += 1
                prev_day = day
            else:
                break

        sum_7d = sum(v for v in recent_7d if v is not None) if recent_7d else None
        sum_14d = sum(v for v in recent_14d if v is not None) if recent_14d else None

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
