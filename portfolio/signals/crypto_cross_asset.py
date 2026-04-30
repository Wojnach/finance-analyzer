"""Cross-asset signal for BTC/ETH — correlated market indicators.

Mirrors `portfolio/signals/metals_cross_asset.py` for the crypto subsystem.
Combines 5 cross-asset sub-indicators via majority vote, capped at 0.7.

Sub-indicators:
  1. ETH/BTC ratio velocity — falling = BTC outperforming (alt season ending)
  2. Fear & Greed direction — contrarian (extreme fear = BUY, extreme greed = SELL)
  3. DXY momentum — DXY up = USD strong = crypto headwind
  4. SPY momentum — risk-on/off gauge (crypto correlates positively in risk-on)
  5. Gold/BTC ratio — capital rotation (gold up + BTC down = flight from risk)

Applicable to BTC-USD and ETH-USD only. Non-crypto -> immediate HOLD.

Designed as a complement to the existing `crypto_macro` signal which
focuses on options-derived metrics (max pain, P/C, exchange netflow).
This module covers cross-asset rotation signals without overlap.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import pandas as pd

from portfolio.signal_utils import majority_vote

logger = logging.getLogger("portfolio.signals.crypto_cross_asset")

_CRYPTO_TICKERS = {"BTC-USD", "ETH-USD"}
_MAX_CONFIDENCE = 0.7

# 1d-calibrated thresholds. Crypto's typical daily range is wider than metals,
# so absolute thresholds are 2-3x bigger.
_ETH_BTC_VEL_PCT = 1.0          # 1d ratio change — clear cross-asset rotation
_FEAR_GREED_LOW = 25            # extreme fear contrarian BUY
_FEAR_GREED_HIGH = 75           # extreme greed contrarian SELL
_DXY_MOVE_PCT = 0.5             # 1d DXY move — meaningful USD shift
_SPY_MOVE_PCT = 0.8             # 1d SPY move — risk-on/off threshold
_GOLD_BTC_RATIO_VEL_PCT = 1.5   # 1d gold-BTC ratio move


def _eth_btc_signal(eth_history: pd.DataFrame | None,
                    btc_history: pd.DataFrame | None,
                    ticker: str) -> tuple[str, dict]:
    """ETH/BTC ratio velocity. Rising ratio = ETH outperforming.

    For BTC: rising ETH/BTC = capital leaving BTC, slight bearish for BTC.
    For ETH: rising ETH/BTC = ETH outperforming, bullish for ETH.
    """
    if eth_history is None or btc_history is None:
        return "HOLD", {"reason": "no ratio data"}
    if eth_history.empty or btc_history.empty:
        return "HOLD", {"reason": "empty ratio data"}

    try:
        eth_close = eth_history["Close"].dropna()
        btc_close = btc_history["Close"].dropna()
        if len(eth_close) < 2 or len(btc_close) < 2:
            return "HOLD", {"reason": "not enough ratio history"}
        # Align on timestamp; fall back to last-N pairing if needed
        ratio = (eth_close / btc_close).dropna()
        if len(ratio) < 2:
            return "HOLD", {"reason": "ratio empty"}
        change_pct = float((ratio.iloc[-1] / ratio.iloc[-2] - 1.0) * 100.0)
    except (KeyError, ValueError, ZeroDivisionError):
        return "HOLD", {"reason": "ratio compute failed"}

    indicators = {"eth_btc_change_pct": round(change_pct, 3)}
    if abs(change_pct) < _ETH_BTC_VEL_PCT:
        return "HOLD", indicators

    rising = change_pct > 0
    if ticker == "ETH-USD":
        return ("BUY" if rising else "SELL", indicators)
    # BTC: rising ETH/BTC mildly bearish for BTC but contrarian — flat hold
    return "HOLD", indicators


def _fear_greed_signal(fg: dict | None) -> tuple[str, dict]:
    if not fg:
        return "HOLD", {"reason": "no fear/greed"}
    val = fg.get("value")
    if val is None:
        return "HOLD", {"reason": "no fear/greed value"}
    indicators = {"fear_greed": val,
                  "classification": fg.get("classification", "")}
    if val <= _FEAR_GREED_LOW:
        return "BUY", indicators
    if val >= _FEAR_GREED_HIGH:
        return "SELL", indicators
    return "HOLD", indicators


def _dxy_signal(dxy_history: pd.DataFrame | None) -> tuple[str, dict]:
    """Crypto inversely correlated with DXY (modest, regime-dependent)."""
    if dxy_history is None or dxy_history.empty:
        return "HOLD", {"reason": "no DXY"}
    try:
        closes = dxy_history["Close"].dropna()
        if len(closes) < 2:
            return "HOLD", {"reason": "DXY too short"}
        change_pct = float((closes.iloc[-1] / closes.iloc[-2] - 1.0) * 100.0)
    except (KeyError, ValueError, ZeroDivisionError):
        return "HOLD", {"reason": "DXY compute failed"}

    indicators = {"dxy_change_pct": round(change_pct, 3)}
    if abs(change_pct) < _DXY_MOVE_PCT:
        return "HOLD", indicators
    # DXY rising = USD strong = crypto headwind
    return ("SELL" if change_pct > 0 else "BUY", indicators)


def _spy_signal(spy_history: pd.DataFrame | None) -> tuple[str, dict]:
    """Risk-on/off via SPY. Crypto typically tracks risk-on regime."""
    if spy_history is None or spy_history.empty:
        return "HOLD", {"reason": "no SPY"}
    try:
        closes = spy_history["Close"].dropna()
        if len(closes) < 2:
            return "HOLD", {"reason": "SPY too short"}
        change_pct = float((closes.iloc[-1] / closes.iloc[-2] - 1.0) * 100.0)
    except (KeyError, ValueError, ZeroDivisionError):
        return "HOLD", {"reason": "SPY compute failed"}

    indicators = {"spy_change_pct": round(change_pct, 3)}
    if abs(change_pct) < _SPY_MOVE_PCT:
        return "HOLD", indicators
    return ("BUY" if change_pct > 0 else "SELL", indicators)


def _gold_btc_ratio_signal(gold_history: pd.DataFrame | None,
                           btc_history: pd.DataFrame | None,
                           ticker: str) -> tuple[str, dict]:
    """Gold/BTC ratio.

    Rising ratio = capital rotating from BTC to gold = flight to safety.
      - For BTC: bearish (capital leaving)
      - For ETH: bearish indirectly (whole crypto risk-off)
    Falling ratio = risk-on rotation INTO crypto = bullish.
    """
    if gold_history is None or btc_history is None:
        return "HOLD", {"reason": "no gold/BTC data"}
    if gold_history.empty or btc_history.empty:
        return "HOLD", {"reason": "empty gold/BTC data"}

    try:
        gold_close = gold_history["Close"].dropna()
        btc_close = btc_history["Close"].dropna()
        if len(gold_close) < 2 or len(btc_close) < 2:
            return "HOLD", {"reason": "not enough gold/BTC history"}
        ratio = (gold_close / btc_close).dropna()
        if len(ratio) < 2:
            return "HOLD", {"reason": "gold/BTC ratio empty"}
        change_pct = float((ratio.iloc[-1] / ratio.iloc[-2] - 1.0) * 100.0)
    except (KeyError, ValueError, ZeroDivisionError):
        return "HOLD", {"reason": "gold/BTC compute failed"}

    indicators = {"gold_btc_change_pct": round(change_pct, 3)}
    if abs(change_pct) < _GOLD_BTC_RATIO_VEL_PCT:
        return "HOLD", indicators
    # Rising = bearish for crypto, falling = bullish
    return ("SELL" if change_pct > 0 else "BUY", indicators)


def compute_crypto_cross_asset_signal(df: pd.DataFrame,
                                      context: dict | None = None) -> dict[str, Any]:
    """Aggregate crypto cross-asset sub-signals into a single voter.

    Args:
        df: OHLCV DataFrame for the crypto ticker (informational only here).
        context: dict with at least:
            - ticker (str)
            - cross_asset (dict, optional): {
                "eth_history": DataFrame,
                "btc_history": DataFrame,
                "dxy_history": DataFrame,
                "spy_history": DataFrame,
                "gold_history": DataFrame,
                "fear_greed": {"value": int, "classification": str},
              }

    Returns:
        dict with keys: signal ("BUY"|"SELL"|"HOLD"), confidence (float),
        sub_signals (dict), reason (str).

    The sub-data is fetched upstream (typically by `portfolio/crypto_precompute.py`
    and cached on `data/crypto_deep_context.json`). Non-crypto tickers
    immediately return HOLD with reason "non-crypto".
    """
    ticker = (context or {}).get("ticker", "")
    if ticker not in _CRYPTO_TICKERS:
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "reason": "non-crypto ticker",
        }

    cross = (context or {}).get("cross_asset") or {}
    eth_history = cross.get("eth_history")
    btc_history = cross.get("btc_history")
    dxy_history = cross.get("dxy_history")
    spy_history = cross.get("spy_history")
    gold_history = cross.get("gold_history")
    fear_greed = cross.get("fear_greed")

    # Check whether we got any cross-asset data at all
    has_any_data = any(
        x is not None and (not hasattr(x, "empty") or not x.empty)
        for x in (eth_history, btc_history, dxy_history, spy_history,
                  gold_history, fear_greed)
    )
    if not has_any_data:
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "reason": "no cross-asset data available",
        }

    sub_signals: dict[str, dict] = {}
    votes: list[str] = []

    for name, fn in (
        ("eth_btc_ratio", lambda: _eth_btc_signal(eth_history, btc_history, ticker)),
        ("fear_greed", lambda: _fear_greed_signal(fear_greed)),
        ("dxy", lambda: _dxy_signal(dxy_history)),
        ("spy", lambda: _spy_signal(spy_history)),
        ("gold_btc_ratio", lambda: _gold_btc_ratio_signal(gold_history, btc_history, ticker)),
    ):
        try:
            decision, ind = fn()
        except Exception as exc:  # noqa: BLE001
            logger.debug("sub %s failed: %s", name, exc)
            decision, ind = "HOLD", {"reason": f"error: {exc}"}
        sub_signals[name] = {"decision": decision, "indicators": ind}
        votes.append(decision)

    overall, _ = majority_vote(votes)
    vote_counts = Counter(votes)
    active = sum(1 for v in votes if v in ("BUY", "SELL"))
    if overall == "HOLD" or active == 0:
        confidence = 0.0
    else:
        # Confidence scales with majority margin and absolute participation.
        margin_winner = vote_counts.get(overall, 0)
        margin_loser = max(
            (vote_counts.get(v, 0) for v in ("BUY", "SELL") if v != overall),
            default=0,
        )
        margin = margin_winner - margin_loser
        confidence = min(_MAX_CONFIDENCE, 0.4 + 0.10 * margin + 0.05 * active)

    return {
        "signal": overall,
        "confidence": round(confidence, 3),
        "sub_signals": sub_signals,
        "reason": f"votes={dict(vote_counts)}, active={active}",
    }
