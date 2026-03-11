"""Optimal price targets with fill probabilities for buy and sell decisions.

Combines Monte Carlo running max/min simulation, first-passage-time analytics,
and structural price levels (BB bands) to produce ranked price targets with
fill probability and expected value.

Usage:
    from portfolio.price_targets import compute_targets
    result = compute_targets("XAG-USD", side="sell", price_usd=85.28,
                             atr_pct=0.59, p_up=0.45, hours_remaining=3.0)
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.stats import norm

from portfolio.monte_carlo import volatility_from_atr, drift_from_probability, MIN_VOLATILITY

logger = logging.getLogger("portfolio.price_targets")

# 24/7 asset suffixes (crypto, metals)
_24H_SUFFIXES = ("-USD",)


def _is_24h(ticker: str) -> bool:
    return any(ticker.upper().endswith(s) for s in _24H_SUFFIXES)


def _year_fraction(hours: float, is_24h: bool = True) -> float:
    """Convert hours to year fraction for GBM."""
    if is_24h:
        return hours / (252.0 * 24.0)
    return hours / (252.0 * 6.5)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def fill_probability(price: float, target: float, vol_annual: float,
                     drift_annual: float, hours_remaining: float,
                     is_24h: bool = True) -> float:
    """First-passage-time probability for GBM reaching *target* within window.

    For SELL (target >= price): probability running max hits target.
    For BUY  (target <= price): probability running min hits target.
    """
    if hours_remaining <= 0 or price <= 0 or vol_annual <= 0:
        return 1.0 if _on_easy_side(price, target, "sell") else 0.0

    # Target on the "easy" side (already filled)
    if target <= price:
        # For a sell order at or below current price -> instant fill
        return 1.0
    # target > price: compute first-passage for running max
    T = _year_fraction(hours_remaining, is_24h)
    sigma = max(vol_annual, MIN_VOLATILITY)
    mu_adj = drift_annual - 0.5 * sigma ** 2
    x = math.log(target / price)

    sqrt_T = math.sqrt(T)
    if sqrt_T * sigma < 1e-12:
        return 0.0

    d1 = (-x + mu_adj * T) / (sigma * sqrt_T)
    d2 = (-x - mu_adj * T) / (sigma * sqrt_T)

    exponent = 2.0 * mu_adj * x / (sigma ** 2)
    exponent = max(-500.0, min(500.0, exponent))  # clamp for numerical safety

    p = norm.cdf(d1) + math.exp(exponent) * norm.cdf(d2)
    return float(max(0.0, min(1.0, p)))


def _on_easy_side(price: float, target: float, side: str) -> bool:
    if side == "sell":
        return target <= price
    return target >= price


def fill_probability_buy(price: float, target: float, vol_annual: float,
                         drift_annual: float, hours_remaining: float,
                         is_24h: bool = True) -> float:
    """Fill probability for a BUY limit order (target <= price)."""
    if target >= price:
        return 1.0
    # Flip: P(min <= target) = P(max of -process >= -target)
    # Symmetry: negate drift, swap price/target relationship
    return fill_probability(price, price ** 2 / target if target > 0 else price,
                            vol_annual, -drift_annual, hours_remaining, is_24h)


def running_extremes(price: float, vol_annual: float, drift_annual: float,
                     hours_remaining: float, side: str = "sell",
                     n_paths: int = 10_000, n_steps: int = 50,
                     is_24h: bool = True) -> dict:
    """MC simulation of running max (sell) or running min (buy)."""
    if hours_remaining <= 0 or price <= 0:
        v = float(price) if price > 0 else 0.0
        return {k: v for k in ("p10", "p25", "p50", "p75", "p90")}

    T = _year_fraction(hours_remaining, is_24h)
    sigma = max(vol_annual, MIN_VOLATILITY)
    dt = T / n_steps
    drift_dt = (drift_annual - 0.5 * sigma ** 2) * dt
    vol_dt = sigma * math.sqrt(dt)

    rng = np.random.default_rng(42)
    n_half = n_paths // 2
    Z = rng.standard_normal((n_half, n_steps))
    Z_anti = -Z
    Z_all = np.concatenate([Z, Z_anti], axis=0)  # (n_paths, n_steps)

    log_increments = drift_dt + vol_dt * Z_all
    log_cum = np.cumsum(log_increments, axis=1)
    # Prepend zero column (start at spot)
    log_cum = np.concatenate([np.zeros((log_cum.shape[0], 1)), log_cum], axis=1)
    price_paths = price * np.exp(log_cum)

    if side == "sell":
        extremes = np.max(price_paths, axis=1)
    else:
        extremes = np.min(price_paths, axis=1)

    pcts = np.percentile(extremes, [10, 25, 50, 75, 90])
    return {
        "p10": round(float(pcts[0]), 4),
        "p25": round(float(pcts[1]), 4),
        "p50": round(float(pcts[2]), 4),
        "p75": round(float(pcts[3]), 4),
        "p90": round(float(pcts[4]), 4),
    }


def structural_levels(price: float, indicators: dict | None) -> dict:
    """Extract BB mid/upper/lower from indicator dict."""
    if not indicators:
        return {}
    levels = {}
    for key in ("bb_mid", "bb_upper", "bb_lower"):
        val = indicators.get(key)
        if val is not None and isinstance(val, (int, float)):
            levels[key] = float(val)
    return levels


def expected_value(fill_prob: float, gain_if_filled: float,
                   gain_at_fallback: float) -> float:
    """Probability-weighted expected value."""
    return fill_prob * gain_if_filled + (1.0 - fill_prob) * gain_at_fallback


def compute_targets(ticker: str, side: str, price_usd: float,
                    atr_pct: float, p_up: float, hours_remaining: float,
                    indicators: dict | None = None, warrant_leverage: float = 1.0,
                    position_units: int = 1, fx_rate: float = 1.0,
                    is_24h: bool = True, n_paths: int = 10_000) -> dict:
    """Main entry point: compute ranked price targets with fill probabilities."""
    result: dict = {
        "ticker": ticker,
        "side": side,
        "price_usd": price_usd,
        "hours_remaining": hours_remaining,
        "extremes": {},
        "targets": [],
        "recommended": None,
    }

    if hours_remaining <= 0 or price_usd <= 0 or atr_pct <= 0:
        return result

    vol = volatility_from_atr(atr_pct)
    if side == "buy":
        drift = drift_from_probability(1.0 - p_up, vol)
    else:
        drift = drift_from_probability(p_up, vol)

    # Structural levels
    levels = structural_levels(price_usd, indicators)

    # Running extremes
    extremes = running_extremes(price_usd, vol, drift, hours_remaining,
                                side=side, n_paths=n_paths, is_24h=is_24h)
    result["extremes"] = extremes

    # Build candidate targets
    candidates: list[tuple[float, str]] = []

    # MC quantiles
    for pkey in ("p25", "p50", "p75"):
        val = extremes.get(pkey)
        if val is not None:
            candidates.append((val, f"mc_{pkey}"))

    # Structural levels
    for label, val in levels.items():
        if side == "sell" and val > price_usd:
            candidates.append((val, label))
        elif side == "buy" and val < price_usd:
            candidates.append((val, label))

    # Fixed offsets
    offsets = [0.005, 0.01, 0.02]
    for off in offsets:
        pct_label = f"{off*100:.1f}%"
        if side == "sell":
            candidates.append((price_usd * (1 + off), f"+{pct_label}"))
        else:
            candidates.append((price_usd * (1 - off), f"-{pct_label}"))

    # Deduplicate (within 0.01% of each other)
    candidates.sort(key=lambda c: c[0])
    deduped: list[tuple[float, str]] = []
    for price_c, label_c in candidates:
        if deduped and abs(price_c - deduped[-1][0]) / max(price_usd, 1e-9) < 0.0001:
            continue
        deduped.append((price_c, label_c))

    # Compute fill probability and EV for each candidate
    min_fill = 0.05
    targets: list[dict] = []
    for target_price, label in deduped:
        if side == "sell":
            if target_price < price_usd:
                continue
            fp = fill_probability(price_usd, target_price, vol, drift,
                                  hours_remaining, is_24h)
            gain_if_filled = (target_price - price_usd) * position_units * warrant_leverage * fx_rate
        else:
            if target_price > price_usd:
                continue
            fp = fill_probability_buy(price_usd, target_price, vol, drift,
                                      hours_remaining, is_24h)
            gain_if_filled = (price_usd - target_price) * position_units * warrant_leverage * fx_rate

        if fp < min_fill:
            continue

        ev = expected_value(fp, gain_if_filled, 0.0)
        targets.append({
            "price": round(target_price, 4),
            "fill_prob": round(fp, 4),
            "ev_sek": round(ev, 2),
            "label": label,
        })

    # Sort by EV descending
    targets.sort(key=lambda t: t["ev_sek"], reverse=True)
    result["targets"] = targets
    result["recommended"] = targets[0] if targets else None
    return result


def compute_all_targets(agent_summary: dict, portfolio_states: dict,
                        config: dict) -> dict | None:
    """Batch wrapper for the reporting pipeline."""
    from portfolio.focus_analysis import hours_to_us_close

    default_hours = config.get("default_hours", 6)
    n_paths = config.get("n_paths", 10_000)
    signals = agent_summary.get("signals", {})
    focus_probs = agent_summary.get("focus_probabilities", {})

    # Collect tickers to process: held positions (SELL) + BUY consensus (BUY)
    tasks: list[tuple[str, str]] = []  # (ticker, side)

    held_tickers: set[str] = set()
    for _label, pf in portfolio_states.items():
        for tk, pos in pf.get("holdings", {}).items():
            if pos.get("shares", 0) > 0:
                held_tickers.add(tk)
                tasks.append((tk, "sell"))

    for tk, sig_data in signals.items():
        if sig_data.get("action") == "BUY" and tk not in held_tickers:
            tasks.append((tk, "buy"))

    if not tasks:
        return None

    results: dict = {}
    for ticker, side in tasks:
        if ticker in results:
            continue
        sig = signals.get(ticker, {})
        price = sig.get("price_usd", 0)
        if price <= 0:
            continue

        extra = sig.get("extra", {})
        atr_pct = extra.get("atr_pct") or sig.get("atr_pct", 2.0)
        is_24h_asset = _is_24h(ticker)

        # Directional probability
        tp = focus_probs.get(ticker, {}).get("1d", {})
        p_up = tp.get("probability", 0.5) if tp else 0.5

        # Hours remaining
        if is_24h_asset:
            hours = float(default_hours)
        else:
            hours = hours_to_us_close()
            if hours <= 0:
                hours = float(default_hours)

        # Indicators for structural levels
        indicators = {k: sig.get(k) for k in ("bb_mid", "bb_upper", "bb_lower")
                      if sig.get(k) is not None}
        if not indicators:
            indicators = {k: extra.get(k) for k in ("bb_mid", "bb_upper", "bb_lower")
                          if extra.get(k) is not None}

        try:
            res = compute_targets(
                ticker, side, price, atr_pct, p_up, hours,
                indicators=indicators or None, is_24h=is_24h_asset,
                n_paths=n_paths,
            )
            if res.get("targets"):
                results[ticker] = res
        except Exception:
            logger.warning("price_targets failed for %s", ticker, exc_info=True)

    return results if results else None
