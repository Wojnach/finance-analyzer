"""Market health module — distribution days, FTD detection, breadth score.

Provides market-level context that the signal engine uses to penalize BUY
confidence in unhealthy markets.  All data comes from yfinance (SPY/QQQ),
cached hourly to avoid rate limits.

Key concepts:
- Distribution day (O'Neil): index closes down >=0.2% on higher volume
- Follow-Through Day: >=1.25% gain on day 4+ of rally on higher volume
- Breadth score: composite 0-100 from distribution days, FTD state, SMAs, trend
"""

import logging
import time
from datetime import UTC, datetime

from portfolio.file_utils import atomic_write_json, load_json
from portfolio.shared_state import _cached

logger = logging.getLogger("portfolio.market_health")

# Cache TTL: 1 hour — market health doesn't need minute-level freshness
MARKET_HEALTH_TTL = 3600

# Distribution day thresholds (O'Neil standard)
DIST_DAY_PRICE_DROP_PCT = -0.002  # >=0.2% decline
DIST_DAY_ROLLING_WINDOW = 25  # trading days
STALLING_UPPER_RANGE_PCT = 0.25  # top 25% of daily range

# FTD thresholds
FTD_CORRECTION_PCT = -0.05  # 5% drop from high = correction
FTD_MIN_RALLY_DAYS = 4  # FTD cannot occur before day 4
FTD_MIN_GAIN_PCT = 0.0125  # 1.25% gain minimum
FTD_FAILURE_WINDOW = 10  # days after FTD to watch for failure

# Breadth score component weights (sum = 100)
_WEIGHT_DIST_DAYS = 25
_WEIGHT_FTD_STATE = 25
_WEIGHT_SMA200 = 20
_WEIGHT_SMA50 = 15
_WEIGHT_TREND_10D = 15

# Zone thresholds
ZONE_DANGER = 30
ZONE_CAUTION = 50

# State file for FTD state machine persistence
import pathlib

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_STATE_FILE = _DATA_DIR / "market_health_state.json"


def _fetch_index_data(symbol: str, period: str = "60d") -> dict | None:
    """Fetch daily OHLCV for an index from yfinance.

    Returns dict with keys: closes, volumes, highs, lows, opens
    as lists of floats (oldest first).
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        if hist.empty or len(hist) < 10:
            return None

        return {
            "closes": hist["Close"].tolist(),
            "volumes": hist["Volume"].tolist(),
            "highs": hist["High"].tolist(),
            "lows": hist["Low"].tolist(),
            "opens": hist["Open"].tolist(),
        }
    except Exception:
        logger.warning("Failed to fetch %s data", symbol, exc_info=True)
        return None


def count_distribution_days(
    closes: list[float],
    volumes: list[float],
    highs: list[float],
    lows: list[float],
    window: int = DIST_DAY_ROLLING_WINDOW,
) -> dict:
    """Count distribution days in the trailing window.

    A distribution day: price closes down >=0.2% AND volume >= previous day.
    A stalling day: price closes in upper 25% of range on higher volume
    (institutions selling into strength — counts as distribution).

    Returns dict with distribution_days count, stalling_days, and detail list.
    """
    if len(closes) < 2:
        return {"distribution_days": 0, "stalling_days": 0, "details": []}

    n = len(closes)
    lookback_start = max(1, n - window)

    dist_days = 0
    stall_days = 0
    details = []

    for i in range(lookback_start, n):
        pct_change = (closes[i] / closes[i - 1]) - 1
        vol_higher = volumes[i] >= volumes[i - 1]
        day_range = highs[i] - lows[i]

        # Distribution day: down >=0.2% on higher volume
        if pct_change <= DIST_DAY_PRICE_DROP_PCT and vol_higher:
            dist_days += 1
            details.append({
                "day_offset": i - n + 1,  # negative offset from today
                "type": "distribution",
                "pct_change": round(pct_change * 100, 2),
            })
        # Stalling day: closes in upper 25% of range on higher volume
        # (but price didn't meaningfully advance — <0.2% gain)
        elif (
            vol_higher
            and day_range > 0
            and 0 <= pct_change < 0.002
            and (closes[i] - lows[i]) / day_range >= (1 - STALLING_UPPER_RANGE_PCT)
        ):
            stall_days += 1
            details.append({
                "day_offset": i - n + 1,
                "type": "stalling",
                "pct_change": round(pct_change * 100, 2),
            })

    return {
        "distribution_days": dist_days,
        "stalling_days": stall_days,
        "total_pressure": dist_days + stall_days,
        "details": details,
    }


# FTD state machine states
STATE_CORRECTING = "correcting"
STATE_RALLY_ATTEMPT = "rally_attempt"
STATE_FTD_CONFIRMED = "ftd_confirmed"
STATE_CONFIRMED_UPTREND = "confirmed_uptrend"


def detect_ftd_state(
    closes: list[float],
    volumes: list[float],
    prev_state: dict | None = None,
) -> dict:
    """Track Follow-Through Day state machine.

    States:
    - correcting: index is in a correction (down >=5% from recent high)
    - rally_attempt: first up day after correction, counting rally days
    - ftd_confirmed: FTD occurred (day 4+ of rally, >=1.25% gain, higher volume)
    - confirmed_uptrend: FTD has held (not undercut rally low within 10 days)

    Args:
        closes: daily close prices (oldest first)
        volumes: daily volumes (oldest first)
        prev_state: previous state dict for continuity (optional)

    Returns:
        dict with state, rally_day, rally_low, ftd_day_offset, etc.
    """
    if len(closes) < 20:
        return {
            "state": STATE_CORRECTING,
            "rally_day": 0,
            "rally_low": 0,
            "recent_high": 0,
            "ftd_day_offset": None,
        }

    # Use last 60 days for analysis
    n = len(closes)

    # Initialize state
    if prev_state:
        state = prev_state.get("state", STATE_CORRECTING)
        rally_day = prev_state.get("rally_day", 0)
        rally_low = prev_state.get("rally_low", 0)
        recent_high = prev_state.get("recent_high", 0)
        ftd_day_offset = prev_state.get("ftd_day_offset")
    else:
        state = STATE_CORRECTING
        rally_day = 0
        rally_low = min(closes[-20:])
        recent_high = max(closes[-60:]) if len(closes) >= 60 else max(closes)
        ftd_day_offset = None

    # Process the most recent day
    today_close = closes[-1]
    today_vol = volumes[-1]
    yesterday_close = closes[-2]
    yesterday_vol = volumes[-2]

    pct_change = (today_close / yesterday_close) - 1

    # Update recent high
    if today_close > recent_high:
        recent_high = today_close

    # Check correction from high
    drawdown = (today_close / recent_high) - 1 if recent_high > 0 else 0

    if state == STATE_CONFIRMED_UPTREND:
        # Check if uptrend is broken (new correction)
        if drawdown <= FTD_CORRECTION_PCT:
            state = STATE_CORRECTING
            rally_day = 0
            rally_low = today_close
            ftd_day_offset = None

    elif state == STATE_FTD_CONFIRMED:
        # Check if FTD holds or fails
        if today_close < rally_low:
            # FTD failed — undercut rally low
            state = STATE_CORRECTING
            rally_day = 0
            rally_low = today_close
            ftd_day_offset = None
        elif ftd_day_offset is not None and (n - 1 - ftd_day_offset) > FTD_FAILURE_WINDOW:
            # FTD has held past the failure window — confirmed uptrend
            state = STATE_CONFIRMED_UPTREND

    elif state == STATE_RALLY_ATTEMPT:
        if today_close < rally_low:
            # Rally failed — undercut rally low
            state = STATE_CORRECTING
            rally_day = 0
            rally_low = today_close
            ftd_day_offset = None
        elif pct_change > 0:
            rally_day += 1
            # Check for FTD
            if (
                rally_day >= FTD_MIN_RALLY_DAYS
                and pct_change >= FTD_MIN_GAIN_PCT
                and today_vol > yesterday_vol
            ):
                state = STATE_FTD_CONFIRMED
                ftd_day_offset = n - 1
        else:
            # Down day during rally — reset rally count but stay in rally_attempt
            # unless we undercut the low (handled above)
            pass

    elif state == STATE_CORRECTING:
        # Update rally low
        if today_close < rally_low or rally_low == 0:
            rally_low = today_close
        # Check for rally attempt start (first up day)
        if pct_change > 0 and drawdown <= FTD_CORRECTION_PCT:
            state = STATE_RALLY_ATTEMPT
            rally_day = 1
            rally_low = min(rally_low, yesterday_close)

    return {
        "state": state,
        "rally_day": rally_day,
        "rally_low": round(rally_low, 2),
        "recent_high": round(recent_high, 2),
        "ftd_day_offset": ftd_day_offset,
        "drawdown_pct": round(drawdown * 100, 2),
    }


def compute_breadth_score(
    dist_data: dict,
    ftd_state: dict,
    closes: list[float],
) -> dict:
    """Compute composite market breadth score (0-100).

    Components:
    - Distribution day severity (25 pts)
    - FTD state (25 pts)
    - SPY vs 200-SMA (20 pts)
    - SPY vs 50-SMA (15 pts)
    - 10-day return direction (15 pts)
    """
    components = {}

    # Component 1: Distribution day severity (fewer = better)
    total_pressure = dist_data.get("total_pressure", 0)
    if total_pressure <= 1:
        components["distribution"] = _WEIGHT_DIST_DAYS
    elif total_pressure <= 3:
        components["distribution"] = int(_WEIGHT_DIST_DAYS * 0.6)
    elif total_pressure <= 5:
        components["distribution"] = int(_WEIGHT_DIST_DAYS * 0.2)
    else:
        components["distribution"] = 0

    # Component 2: FTD state
    state = ftd_state.get("state", STATE_CORRECTING)
    ftd_scores = {
        STATE_CONFIRMED_UPTREND: _WEIGHT_FTD_STATE,
        STATE_FTD_CONFIRMED: int(_WEIGHT_FTD_STATE * 0.8),
        STATE_RALLY_ATTEMPT: int(_WEIGHT_FTD_STATE * 0.4),
        STATE_CORRECTING: 0,
    }
    components["ftd_state"] = ftd_scores.get(state, 0)

    # Component 3: Price vs 200-SMA
    if len(closes) >= 200:
        sma200 = sum(closes[-200:]) / 200
        components["sma200"] = _WEIGHT_SMA200 if closes[-1] > sma200 else 0
    elif len(closes) >= 50:
        # Fallback: use longest available SMA
        sma_n = len(closes)
        sma_val = sum(closes) / sma_n
        components["sma200"] = _WEIGHT_SMA200 if closes[-1] > sma_val else 0
    else:
        components["sma200"] = int(_WEIGHT_SMA200 * 0.5)  # neutral if insufficient data

    # Component 4: Price vs 50-SMA
    if len(closes) >= 50:
        sma50 = sum(closes[-50:]) / 50
        components["sma50"] = _WEIGHT_SMA50 if closes[-1] > sma50 else 0
    else:
        components["sma50"] = int(_WEIGHT_SMA50 * 0.5)

    # Component 5: 10-day return
    if len(closes) >= 11:
        ret_10d = (closes[-1] / closes[-11]) - 1
        if ret_10d > 0.02:
            components["trend_10d"] = _WEIGHT_TREND_10D
        elif ret_10d > 0:
            components["trend_10d"] = int(_WEIGHT_TREND_10D * 0.67)
        elif ret_10d > -0.02:
            components["trend_10d"] = int(_WEIGHT_TREND_10D * 0.33)
        else:
            components["trend_10d"] = 0
    else:
        components["trend_10d"] = int(_WEIGHT_TREND_10D * 0.5)

    score = sum(components.values())

    return {
        "score": score,
        "components": components,
    }


def _classify_zone(score: int) -> str:
    """Classify market health score into zones."""
    if score < ZONE_DANGER:
        return "danger"
    if score < ZONE_CAUTION:
        return "caution"
    return "healthy"


def _compute_market_health() -> dict | None:
    """Full market health computation from live data.

    Fetches SPY and QQQ, computes distribution days, FTD state,
    and breadth score.  Returns the complete health snapshot.
    """
    spy_data = _fetch_index_data("SPY", "90d")
    qqq_data = _fetch_index_data("QQQ", "90d")

    if not spy_data:
        logger.warning("SPY data unavailable — cannot compute market health")
        return None

    # Distribution days for both indices
    spy_dist = count_distribution_days(
        spy_data["closes"], spy_data["volumes"],
        spy_data["highs"], spy_data["lows"],
    )
    qqq_dist = (
        count_distribution_days(
            qqq_data["closes"], qqq_data["volumes"],
            qqq_data["highs"], qqq_data["lows"],
        )
        if qqq_data
        else {"distribution_days": 0, "stalling_days": 0, "total_pressure": 0}
    )

    # FTD state machine (use SPY as primary, persist state)
    prev_ftd = load_json(_STATE_FILE, default={}).get("ftd_state")
    ftd = detect_ftd_state(spy_data["closes"], spy_data["volumes"], prev_ftd)

    # Breadth score (based on SPY — the broad market)
    breadth = compute_breadth_score(spy_dist, ftd, spy_data["closes"])
    score = breadth["score"]
    zone = _classify_zone(score)

    # 10-day return for context
    closes = spy_data["closes"]
    ret_10d = ((closes[-1] / closes[-11]) - 1) * 100 if len(closes) >= 11 else 0

    # SMA status
    spy_above_200 = False
    spy_above_50 = False
    if len(closes) >= 200:
        spy_above_200 = closes[-1] > sum(closes[-200:]) / 200
    if len(closes) >= 50:
        spy_above_50 = closes[-1] > sum(closes[-50:]) / 50

    result = {
        "score": score,
        "zone": zone,
        "distribution_days_spy": spy_dist["distribution_days"],
        "distribution_days_qqq": qqq_dist["distribution_days"],
        "stalling_days_spy": spy_dist["stalling_days"],
        "total_pressure_spy": spy_dist["total_pressure"],
        "total_pressure_qqq": qqq_dist["total_pressure"],
        "ftd_state": ftd["state"],
        "ftd_rally_day": ftd["rally_day"],
        "ftd_drawdown_pct": ftd["drawdown_pct"],
        "spy_above_200sma": spy_above_200,
        "spy_above_50sma": spy_above_50,
        "spy_return_10d_pct": round(ret_10d, 2),
        "components": breadth["components"],
        "updated_at": datetime.now(UTC).isoformat(),
    }

    # Persist FTD state for continuity across restarts
    state_to_save = {
        "ftd_state": {
            "state": ftd["state"],
            "rally_day": ftd["rally_day"],
            "rally_low": ftd["rally_low"],
            "recent_high": ftd["recent_high"],
            "ftd_day_offset": ftd["ftd_day_offset"],
        },
        "last_updated": result["updated_at"],
    }
    atomic_write_json(_STATE_FILE, state_to_save)

    return result


def get_market_health(force: bool = False) -> dict | None:
    """Get cached market health snapshot.  Refreshes hourly.

    Args:
        force: bypass cache and recompute

    Returns:
        Market health dict or None on failure.
    """
    if force:
        return _compute_market_health()
    return _cached("market_health", MARKET_HEALTH_TTL, _compute_market_health)


def maybe_refresh_market_health() -> None:
    """Post-cycle hook: refresh market health if stale.

    Called from main.py's _run_post_cycle().  The _cached() call
    internally skips if data is fresh.
    """
    try:
        health = get_market_health()
        if health:
            logger.debug(
                "Market health: score=%d zone=%s dist_spy=%d ftd=%s",
                health["score"], health["zone"],
                health["distribution_days_spy"], health["ftd_state"],
            )
    except Exception:
        logger.warning("market health refresh failed", exc_info=True)


def get_confidence_penalty(action: str, health: dict | None = None) -> float:
    """Return confidence multiplier based on market health.

    Only penalizes BUY signals.  SELL and HOLD are unaffected.

    Returns:
        Multiplier in range [0.6, 1.1].  1.0 = no change.
    """
    if action != "BUY":
        return 1.0

    if health is None:
        return 1.0  # no data = no penalty

    score = health.get("score", 50)
    if score < ZONE_DANGER:
        return 0.6  # harsh penalty — danger zone
    if score < ZONE_CAUTION:
        return 0.8  # moderate penalty — caution zone
    if score >= 70:
        return 1.1  # slight boost — very healthy
    return 1.0  # healthy — no change
