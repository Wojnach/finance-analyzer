"""Futures flow signal — OI, long/short ratios, funding rate history.

Composite signal with 6 sub-indicators, majority vote, confidence capped at 0.7.
Only applicable to crypto tickers (BTC-USD, ETH-USD). Non-crypto → immediate HOLD.

Sub-indicators:
  1. oi_trend       — Rising OI + price direction → new longs/shorts
  2. oi_divergence  — OI/price divergence → de-risking or hidden buildup
  3. ls_extreme     — Contrarian: crowd overleveraged one side
  4. top_vs_crowd   — Top traders disagree with crowd → follow smart money
  5. funding_trend  — Funding rate extremes → contrarian
  6. oi_acceleration — Second derivative of OI → momentum confirmation

The ``context`` parameter is a dict with keys: ticker, config, macro.
"""

from __future__ import annotations

import logging

import pandas as pd

from portfolio.signal_utils import majority_vote
from portfolio.shared_state import _cached

logger = logging.getLogger("portfolio.signals.futures_flow")

# Confidence cap (same as news_event, econ_calendar)
_MAX_CONFIDENCE = 0.7

# Thresholds
_LS_EXTREME_HIGH = 2.0   # crowd overleveraged long
_LS_EXTREME_LOW = 0.7    # crowd overleveraged short
_FUNDING_HIGH = 0.0005   # 0.05% — contrarian SELL
_FUNDING_LOW = -0.0003   # -0.03% — contrarian BUY
_TOP_DIVERGE_THRESHOLD = 0.3  # top trader LS differs from crowd by this much

# Minimum data points needed for trend calculations
_MIN_HISTORY = 5


def _oi_trend(oi_history, df):
    """Sub-1: OI trend + price direction.

    Rising OI + price up = BUY (new longs entering)
    Rising OI + price down = SELL (new shorts entering)
    Falling OI = HOLD (deleveraging, no directional signal)
    """
    if not oi_history or len(oi_history) < _MIN_HISTORY:
        return "HOLD"

    recent_oi = [d["oi"] for d in oi_history[-_MIN_HISTORY:]]
    oi_change = (recent_oi[-1] - recent_oi[0]) / recent_oi[0] if recent_oi[0] else 0

    if oi_change <= 0.005:  # OI not meaningfully rising (< 0.5%)
        return "HOLD"

    # Price direction from OHLCV dataframe
    if df is not None and len(df) >= _MIN_HISTORY:
        price_start = df["close"].iloc[-_MIN_HISTORY]
        price_end = df["close"].iloc[-1]
        if price_start and price_end > price_start:
            return "BUY"   # rising OI + rising price = new longs
        elif price_start and price_end < price_start:
            return "SELL"  # rising OI + falling price = new shorts

    return "HOLD"


def _oi_divergence(oi_history, df):
    """Sub-2: OI/price divergence.

    Price up but OI falling = bearish divergence (SELL) — rally on thin leverage
    Price down but OI falling = de-risking/capitulation (BUY) — sellers exhausting
    """
    if not oi_history or len(oi_history) < _MIN_HISTORY:
        return "HOLD"

    recent_oi = [d["oi"] for d in oi_history[-_MIN_HISTORY:]]
    oi_change = (recent_oi[-1] - recent_oi[0]) / recent_oi[0] if recent_oi[0] else 0

    if abs(oi_change) < 0.005:  # OI flat — no divergence
        return "HOLD"

    if df is not None and len(df) >= _MIN_HISTORY:
        price_start = df["close"].iloc[-_MIN_HISTORY]
        price_end = df["close"].iloc[-1]
        price_change = (price_end - price_start) / price_start if price_start else 0

        if price_change > 0.005 and oi_change < -0.005:
            return "SELL"  # price up, OI down — bearish divergence
        if price_change < -0.005 and oi_change < -0.005:
            return "BUY"   # price down, OI down — de-risking, capitulation buy

    return "HOLD"


def _ls_extreme(ls_ratio):
    """Sub-3: Long/short ratio extreme — contrarian.

    LS > 2.0 = crowd overleveraged long → contrarian SELL
    LS < 0.7 = crowd overleveraged short → contrarian BUY
    """
    if not ls_ratio:
        return "HOLD"

    latest = ls_ratio[-1]["longShortRatio"]
    if latest > _LS_EXTREME_HIGH:
        return "SELL"
    elif latest < _LS_EXTREME_LOW:
        return "BUY"
    return "HOLD"


def _top_vs_crowd(top_position_ratio, ls_ratio):
    """Sub-4: Top trader vs crowd divergence — follow top traders.

    When top traders' positioning significantly differs from crowd,
    follow the top traders (they tend to be right).
    """
    if not top_position_ratio or not ls_ratio:
        return "HOLD"

    top_ls = top_position_ratio[-1]["longShortRatio"]
    crowd_ls = ls_ratio[-1]["longShortRatio"]

    diff = top_ls - crowd_ls
    if abs(diff) < _TOP_DIVERGE_THRESHOLD:
        return "HOLD"

    # Top traders more long than crowd → BUY
    if diff > _TOP_DIVERGE_THRESHOLD:
        return "BUY"
    # Top traders more short than crowd → SELL
    if diff < -_TOP_DIVERGE_THRESHOLD:
        return "SELL"

    return "HOLD"


def _funding_trend(funding_history):
    """Sub-5: Funding rate trend — contrarian.

    Rising funding > 0.05% = overleveraged longs → contrarian SELL
    Deeply negative < -0.03% = overleveraged shorts → contrarian BUY
    Also checks 3-period trend for strengthening signal.
    """
    if not funding_history or len(funding_history) < 3:
        return "HOLD"

    latest_rate = funding_history[-1]["fundingRate"]

    # 3-period average for trend
    recent_rates = [d["fundingRate"] for d in funding_history[-3:]]
    avg_rate = sum(recent_rates) / len(recent_rates)

    # Use average to smooth noise
    if avg_rate > _FUNDING_HIGH:
        return "SELL"
    elif avg_rate < _FUNDING_LOW:
        return "BUY"

    # Check if latest is extreme even if average isn't
    if latest_rate > _FUNDING_HIGH * 1.5:
        return "SELL"
    elif latest_rate < _FUNDING_LOW * 1.5:
        return "BUY"

    return "HOLD"


def _oi_acceleration(oi_history, df):
    """Sub-6: OI acceleration — second derivative of OI.

    Accelerating OI growth + aligned price = strong momentum confirmation.
    Decelerating OI growth = momentum fading.
    """
    if not oi_history or len(oi_history) < _MIN_HISTORY * 2:
        return "HOLD"

    oi_values = [d["oi"] for d in oi_history]

    # First derivative: rate of change
    mid = len(oi_values) // 2
    first_half_change = (oi_values[mid] - oi_values[0]) / oi_values[0] if oi_values[0] else 0
    second_half_change = (oi_values[-1] - oi_values[mid]) / oi_values[mid] if oi_values[mid] else 0

    # Second derivative: acceleration
    acceleration = second_half_change - first_half_change

    if abs(acceleration) < 0.005:  # not meaningful
        return "HOLD"

    # Check price alignment
    if df is not None and len(df) >= _MIN_HISTORY:
        price_start = df["close"].iloc[-_MIN_HISTORY]
        price_end = df["close"].iloc[-1]
        price_up = price_end > price_start

        if acceleration > 0.005:  # OI accelerating
            return "BUY" if price_up else "SELL"
        elif acceleration < -0.005:  # OI decelerating — momentum fading
            # Fading momentum → contrarian lean
            return "SELL" if price_up else "BUY"

    return "HOLD"


def compute_futures_flow_signal(df: pd.DataFrame, context: dict = None) -> dict:
    """Compute the futures flow composite signal.

    Args:
        df: OHLCV DataFrame for the ticker.
        context: dict with keys {ticker, config, macro}.

    Returns:
        dict with action, confidence, sub_signals, indicators.
    """
    ticker = context.get("ticker", "") if context else ""

    # Non-crypto → immediate HOLD
    from portfolio.tickers import CRYPTO_SYMBOLS
    if ticker not in CRYPTO_SYMBOLS:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {},
        }

    # Fetch all futures data
    from portfolio.futures_data import get_all_futures_data
    futures = _cached(
        f"futures_flow_data_{ticker}",
        300,
        get_all_futures_data,
        ticker,
    )

    if futures is None:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {"error": "fetch_failed"},
        }

    oi_history = futures.get("oi_history")
    ls_ratio = futures.get("ls_ratio")
    top_position = futures.get("top_position_ratio")
    funding_hist = futures.get("funding_history")
    current_oi = futures.get("open_interest")

    # Compute sub-signals
    sub = {
        "oi_trend": _oi_trend(oi_history, df),
        "oi_divergence": _oi_divergence(oi_history, df),
        "ls_extreme": _ls_extreme(ls_ratio),
        "top_vs_crowd": _top_vs_crowd(top_position, ls_ratio),
        "funding_trend": _funding_trend(funding_hist),
        "oi_acceleration": _oi_acceleration(oi_history, df),
    }

    # Majority vote — pass list of vote strings, not the dict
    action, confidence = majority_vote(list(sub.values()))
    confidence = min(confidence, _MAX_CONFIDENCE)

    # Build indicators dict for reporting
    indicators = {}
    if current_oi:
        indicators["open_interest"] = current_oi.get("oi")
    if oi_history and len(oi_history) >= 2:
        oi_vals = [d["oi"] for d in oi_history]
        indicators["oi_change_pct"] = round(
            (oi_vals[-1] - oi_vals[0]) / oi_vals[0] * 100 if oi_vals[0] else 0, 2
        )
    if ls_ratio:
        latest_ls = ls_ratio[-1]
        indicators["ls_ratio"] = round(latest_ls["longShortRatio"], 3)
        indicators["long_pct"] = round(latest_ls["longAccount"] * 100, 1)
        indicators["short_pct"] = round(latest_ls["shortAccount"] * 100, 1)
    if top_position:
        indicators["top_trader_ls"] = round(top_position[-1]["longShortRatio"], 3)
    if funding_hist and len(funding_hist) >= 1:
        indicators["funding_rate"] = funding_hist[-1]["fundingRate"]
        indicators["funding_rate_pct"] = round(funding_hist[-1]["fundingRate"] * 100, 4)
        if len(funding_hist) >= 3:
            recent = [d["fundingRate"] for d in funding_hist[-3:]]
            indicators["funding_3period_avg"] = round(sum(recent) / len(recent) * 100, 4)

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": sub,
        "indicators": indicators,
    }
