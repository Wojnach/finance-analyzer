"""Crypto derivatives composite — contrarian signal from z-score extremes.

Three sub-signals from Binance FAPI derivatives data:
  1. OI momentum    — OI rate-of-change z-score vs recent history
  2. Funding z-score — funding rate z-score identifies leverage extremes
  3. L/S contrarian  — long/short account ratio extremes

Crypto-only (BTC-USD, ETH-USD). Non-crypto -> immediate HOLD.
"""
from __future__ import annotations

import logging
import math

import pandas as pd

from portfolio.futures_data import (
    SYMBOL_MAP,
    get_funding_rate_history,
    get_long_short_ratio,
    get_open_interest_history,
)
from portfolio.signal_utils import majority_vote

logger = logging.getLogger("portfolio.signals.crypto_derivatives_composite")

_MAX_CONFIDENCE = 0.7
_MIN_HISTORY = 10

_OI_Z_THRESHOLD = 1.5
_FR_Z_THRESHOLD = 2.0
_LS_EXTREME_THRESHOLD = 0.62


def _zscore(values: list[float]) -> float:
    if len(values) < _MIN_HISTORY:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = var ** 0.5
    if std < 1e-10:
        return 0.0
    return (values[-1] - mean) / std


def _oi_momentum_vote(oi_history: list[dict] | None, df: pd.DataFrame) -> tuple[str, dict]:
    if not oi_history or len(oi_history) < _MIN_HISTORY:
        return "HOLD", {}

    oi_vals = [d.get("oi", 0) or 0 for d in oi_history]
    if not oi_vals or oi_vals[0] <= 0:
        return "HOLD", {}

    rocs = []
    for i in range(1, len(oi_vals)):
        if oi_vals[i - 1] > 0:
            rocs.append((oi_vals[i] - oi_vals[i - 1]) / oi_vals[i - 1])
    if len(rocs) < _MIN_HISTORY:
        return "HOLD", {}

    oi_z = _zscore(rocs)
    indicators = {"oi_roc_latest": rocs[-1] if rocs else 0.0, "oi_z": oi_z}

    price_falling = False
    price_rising = False
    if df is not None and len(df) >= 5:
        p0 = float(df["close"].iloc[-5])
        p1 = float(df["close"].iloc[-1])
        if not math.isnan(p0) and not math.isnan(p1) and p0 > 0:
            if p1 < p0:
                price_falling = True
            elif p1 > p0:
                price_rising = True

    if oi_z > _OI_Z_THRESHOLD and price_falling:
        return "SELL", indicators
    if oi_z < -_OI_Z_THRESHOLD and price_rising:
        return "BUY", indicators
    if oi_z > _OI_Z_THRESHOLD * 1.5:
        return "SELL", indicators
    if oi_z < -_OI_Z_THRESHOLD * 1.5:
        return "BUY", indicators

    return "HOLD", indicators


def _funding_zscore_vote(funding_history: list[dict] | None) -> tuple[str, dict]:
    if not funding_history or len(funding_history) < _MIN_HISTORY:
        return "HOLD", {}

    rates = [d.get("fundingRate", 0) or 0 for d in funding_history]
    fr_z = _zscore(rates)
    latest = rates[-1] if rates else 0.0
    apr = latest * 3 * 365 * 100

    indicators = {"funding_rate": latest, "funding_apr": apr, "funding_z": fr_z}

    if fr_z > _FR_Z_THRESHOLD:
        return "SELL", indicators
    if fr_z < -_FR_Z_THRESHOLD:
        return "BUY", indicators

    return "HOLD", indicators


def _ls_contrarian_vote(ls_history: list[dict] | None) -> tuple[str, dict]:
    if not ls_history or len(ls_history) < 3:
        return "HOLD", {}

    latest = ls_history[-1]
    long_pct = latest.get("longAccount", 0.5)
    short_pct = latest.get("shortAccount", 0.5)

    indicators = {"long_pct": long_pct, "short_pct": short_pct}

    if long_pct >= _LS_EXTREME_THRESHOLD:
        return "SELL", indicators
    if short_pct >= _LS_EXTREME_THRESHOLD:
        return "BUY", indicators

    return "HOLD", indicators


def compute_crypto_derivatives_composite_signal(
    df: pd.DataFrame, context: dict = None
) -> dict:
    empty = {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    if context is None:
        context = {}

    ticker = context.get("ticker", "")
    if ticker not in SYMBOL_MAP:
        return empty

    if df is None or len(df) < 5:
        return empty

    oi_hist = get_open_interest_history(ticker, period="5m", limit=30)
    funding_hist = get_funding_rate_history(ticker, limit=50)
    ls_hist = get_long_short_ratio(ticker, period="5m", limit=30)

    oi_vote, oi_ind = _oi_momentum_vote(oi_hist, df)
    fr_vote, fr_ind = _funding_zscore_vote(funding_hist)
    ls_vote, ls_ind = _ls_contrarian_vote(ls_hist)

    votes = [oi_vote, fr_vote, ls_vote]
    action, confidence = majority_vote(votes, count_hold=False)
    confidence = min(confidence, _MAX_CONFIDENCE)

    sub_signals = {
        "oi_momentum": oi_vote,
        "funding_zscore": fr_vote,
        "ls_contrarian": ls_vote,
    }
    indicators = {}
    indicators.update(oi_ind)
    indicators.update(fr_ind)
    indicators.update(ls_ind)

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": sub_signals,
        "indicators": indicators,
    }
