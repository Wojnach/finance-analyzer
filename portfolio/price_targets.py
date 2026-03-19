"""Optimal price targets with fill probabilities for buy and sell decisions.

Combines Monte Carlo running max/min simulation, first-passage-time analytics,
and structural price levels (BB, Fibonacci, pivots, Keltner, Donchian, VWAP,
smart money swing levels) to produce ranked price targets with fill probability
and expected value.

Supports regime-aware confidence adjustment, BB squeeze warnings, and
Chronos forecast drift blending for improved accuracy.

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

from portfolio.monte_carlo import MIN_VOLATILITY, drift_from_probability, volatility_from_atr

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


def _is_valid_level(val: object) -> bool:
    """Return True if *val* is a finite positive number usable as a price level."""
    if not isinstance(val, (int, float)):
        return False
    if math.isnan(val) or math.isinf(val):
        return False
    return val > 0


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


def structural_levels(price: float, indicators: dict | None,
                      extra: dict | None = None) -> dict:
    """Extract all available price levels from indicators and enhanced signal data.

    Sources:
    - BB mid/upper/lower from main indicators
    - Fibonacci retracement levels, pivots, Camarilla, golden pocket, swings
    - Keltner channels, Donchian channels (volatility signal)
    - VWAP (volume flow signal)
    - Smart money swing highs/lows
    """
    if not indicators and not extra:
        return {}

    levels: dict[str, float] = {}

    # BB levels (from main indicators)
    if indicators:
        for key in ("bb_mid", "bb_upper", "bb_lower"):
            val = indicators.get(key)
            if _is_valid_level(val):
                levels[key] = float(val)

    if not extra:
        return levels

    # -- Fibonacci levels --------------------------------------------------
    fib_ind = extra.get("fibonacci_indicators", {})
    if fib_ind:
        fib_levels_dict = fib_ind.get("fib_levels", {})
        for ratio_key, val in fib_levels_dict.items():
            if _is_valid_level(val):
                # Convert "0.236" -> "fib_236", "0.5" -> "fib_5" etc.
                clean_key = str(ratio_key).replace("0.", "")
                levels[f"fib_{clean_key}"] = float(val)

        # Pivot levels
        _pivot_keys = [
            ("pivot", "pivot_pp"),
            ("r1", "pivot_r1"),
            ("r2", "pivot_r2"),
            ("s1", "pivot_s1"),
            ("s2", "pivot_s2"),
        ]
        for src_key, label in _pivot_keys:
            val = fib_ind.get(src_key)
            if _is_valid_level(val):
                levels[label] = float(val)

        # Camarilla pivots
        for key in ("cam_r3", "cam_s3", "cam_r4", "cam_s4"):
            val = fib_ind.get(key)
            if _is_valid_level(val):
                levels[key] = float(val)

        # Golden pocket
        for key in ("gp_upper", "gp_lower"):
            val = fib_ind.get(key)
            if _is_valid_level(val):
                levels[key] = float(val)

        # Fibonacci swing points
        for key in ("swing_high", "swing_low"):
            val = fib_ind.get(key)
            if _is_valid_level(val):
                levels[f"fib_{key}"] = float(val)

    # -- Volatility levels (Keltner, Donchian) -----------------------------
    vol_ind = extra.get("volatility_sig_indicators", {})
    if vol_ind:
        for key in ("keltner_upper", "keltner_lower",
                     "donchian_upper", "donchian_lower"):
            val = vol_ind.get(key)
            if _is_valid_level(val):
                levels[key] = float(val)

    # -- VWAP from volume flow ---------------------------------------------
    vf_ind = extra.get("volume_flow_indicators", {})
    if vf_ind:
        val = vf_ind.get("vwap")
        if _is_valid_level(val):
            levels["vwap"] = float(val)

    # -- Smart money swing levels ------------------------------------------
    sm_ind = extra.get("smart_money_indicators", {})
    if sm_ind:
        for key in ("last_swing_high", "last_swing_low"):
            val = sm_ind.get(key)
            if _is_valid_level(val):
                levels[f"smc_{key.replace('last_', '')}"] = float(val)

    return levels


def expected_value(fill_prob: float, gain_if_filled: float,
                   gain_at_fallback: float) -> float:
    """Probability-weighted expected value."""
    return fill_prob * gain_if_filled + (1.0 - fill_prob) * gain_at_fallback


def _apply_regime_adjustment(targets: list[dict], regime: str, side: str,
                             price_usd: float, bb_mid: float | None) -> None:
    """Mutate *targets* in-place with regime-aware confidence adjustments.

    - ranging/range-bound: penalize far targets, boost targets near bb_mid
    - trending-up + sell: boost fill_prob slightly for targets above price
    - trending-down + buy: boost fill_prob slightly for targets below price
    - high-vol: no fill_prob change (widening is handled by ATR already)
    """
    regime_lower = regime.lower().replace("-", "").replace("_", "")
    if not regime_lower:
        return

    for t in targets:
        tp = t["price"]
        fp = t["fill_prob"]

        if regime_lower in ("ranging", "rangebound"):
            # Penalize targets far from price (>1% away)
            pct_away = abs(tp - price_usd) / price_usd if price_usd > 0 else 0
            if pct_away > 0.01:
                t["fill_prob"] = round(fp * 0.85, 4)
            # Boost targets near bb_mid (mean-reversion)
            if bb_mid and bb_mid > 0:
                pct_from_mid = abs(tp - bb_mid) / bb_mid
                if pct_from_mid < 0.005:  # within 0.5% of bb_mid
                    t["fill_prob"] = round(min(fp * 1.15, 1.0), 4)

        elif regime_lower in ("trendingup",):
            if side == "sell" and tp > price_usd:
                t["fill_prob"] = round(min(fp * 1.10, 1.0), 4)
            elif side == "buy" and tp < price_usd:
                t["fill_prob"] = round(fp * 0.90, 4)

        elif regime_lower in ("trendingdown",):
            if side == "buy" and tp < price_usd:
                t["fill_prob"] = round(min(fp * 1.10, 1.0), 4)
            elif side == "sell" and tp > price_usd:
                t["fill_prob"] = round(fp * 0.90, 4)


def compute_targets(ticker: str, side: str, price_usd: float,
                    atr_pct: float, p_up: float, hours_remaining: float,
                    indicators: dict | None = None, extra: dict | None = None,
                    warrant_leverage: float = 1.0,
                    position_units: int = 1, fx_rate: float = 1.0,
                    is_24h: bool = True, n_paths: int = 10_000,
                    regime: str = "", bb_squeeze: bool = False,
                    chronos_drift: float | None = None) -> dict:
    """Main entry point: compute ranked price targets with fill probabilities.

    Parameters
    ----------
    extra : dict | None
        Enhanced signal indicator dicts (fibonacci_indicators, etc.)
        passed through to ``structural_levels``.
    regime : str
        Market regime string (e.g. "trending-up", "ranging").
        Used for regime-aware confidence adjustment.
    bb_squeeze : bool
        If True, Bollinger Band squeeze is active -- reduce confidence
        on all targets by 0.7x and flag ``squeeze_warning``.
    chronos_drift : float | None
        Annualised drift from Chronos 24h forecast.  When provided,
        blended 30/70 with the signal-based drift.
    """
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

    # Blend Chronos drift when available
    if chronos_drift is not None:
        drift = 0.7 * drift + 0.3 * chronos_drift

    # Structural levels (enriched with extra signal indicators)
    levels = structural_levels(price_usd, indicators, extra=extra)

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
        if side == "sell" and val > price_usd or side == "buy" and val < price_usd:
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

    # Regime-aware adjustments
    bb_mid_val = levels.get("bb_mid")
    if regime:
        _apply_regime_adjustment(targets, regime, side, price_usd, bb_mid_val)
        # Re-compute EV after fill_prob adjustment
        for t in targets:
            if side == "sell":
                gain = (t["price"] - price_usd) * position_units * warrant_leverage * fx_rate
            else:
                gain = (price_usd - t["price"]) * position_units * warrant_leverage * fx_rate
            t["ev_sek"] = round(expected_value(t["fill_prob"], gain, 0.0), 2)

    # BB squeeze warning: reduce confidence on all targets
    if bb_squeeze:
        result["squeeze_warning"] = True
        for t in targets:
            t["fill_prob"] = round(t["fill_prob"] * 0.7, 4)
            # Re-compute EV after squeeze adjustment
            if side == "sell":
                gain = (t["price"] - price_usd) * position_units * warrant_leverage * fx_rate
            else:
                gain = (price_usd - t["price"]) * position_units * warrant_leverage * fx_rate
            t["ev_sek"] = round(expected_value(t["fill_prob"], gain, 0.0), 2)

    # Filter out targets that dropped below min_fill after adjustments
    targets = [t for t in targets if t["fill_prob"] >= min_fill]

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

    # Forecast signals for Chronos drift
    forecast_signals = agent_summary.get("forecast_signals", {})

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

        # Indicators for structural levels (BB from main signal data)
        indicators = {k: sig.get(k) for k in ("bb_mid", "bb_upper", "bb_lower")
                      if sig.get(k) is not None}
        if not indicators:
            indicators = {k: extra.get(k) for k in ("bb_mid", "bb_upper", "bb_lower")
                          if extra.get(k) is not None}

        # Chronos drift from forecast signal
        chronos_drift_val = None
        fc_data = forecast_signals.get(ticker, {})
        chronos_pct = fc_data.get("chronos_24h_pct", 0)
        chronos_conf = fc_data.get("chronos_24h_conf", 0)
        if isinstance(chronos_conf, (int, float)) and chronos_conf > 0.3 \
                and isinstance(chronos_pct, (int, float)) and chronos_pct != 0:
            chronos_drift_val = (chronos_pct / 100.0) * math.sqrt(252)

        # BB squeeze detection
        vol_ind = extra.get("volatility_sig_indicators", {})
        squeeze = bool(vol_ind.get("bb_squeeze_on", False)) if vol_ind else False

        # Regime
        regime = sig.get("regime", "") or ""

        try:
            res = compute_targets(
                ticker, side, price, atr_pct, p_up, hours,
                indicators=indicators or None,
                extra=extra or None,
                is_24h=is_24h_asset,
                n_paths=n_paths,
                regime=regime,
                bb_squeeze=squeeze,
                chronos_drift=chronos_drift_val,
            )
            if res.get("targets"):
                results[ticker] = res
        except Exception:
            logger.warning("price_targets failed for %s", ticker, exc_info=True)

    return results if results else None
