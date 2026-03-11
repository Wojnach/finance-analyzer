"""Advisory entry/exit price recommendations for metals warrants.

This module bridges underlying-metal forecasts to actual Avanza instrument
prices. It is intentionally advisory-only: it scores candidate BUY and SELL
limit prices, but it does not place or queue orders.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any

from portfolio.monte_carlo import drift_from_probability, volatility_from_atr
from portfolio.price_targets import compute_targets, fill_probability, fill_probability_buy

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python 3.9+ in prod, keep fallback tiny
    ZoneInfo = None

try:
    from metals_swing_config import (
        MIN_BARRIER_DISTANCE_PCT,
        MIN_SPREAD_PCT,
        MIN_TRADE_SEK,
        POSITION_SIZE_PCT,
        TARGET_LEVERAGE,
    )
except ImportError:  # pragma: no cover - tests can run without swing config
    MIN_BARRIER_DISTANCE_PCT = 15.0
    MIN_SPREAD_PCT = 1.5
    MIN_TRADE_SEK = 500.0
    POSITION_SIZE_PCT = 30.0
    TARGET_LEVERAGE = 5.0


MARKET_CLOSE_CET = "21:55"
_BUY_SIGNAL_MIN = 0.53
_SELL_SIGNAL_MAX = 0.47
_MIN_FILL_PROB = 0.05
_DEFAULT_ATR = {"XAG-USD": 4.4, "XAU-USD": 1.9}


def hours_to_metals_close(now: _dt.datetime | None = None) -> float:
    """Hours until the Avanza metals trading close."""
    if ZoneInfo is not None:
        tz = ZoneInfo("Europe/Stockholm")
        now_cet = now.astimezone(tz) if now else _dt.datetime.now(tz)
    else:  # pragma: no cover
        utc_now = now or _dt.datetime.now(_dt.timezone.utc)
        now_cet = utc_now + _dt.timedelta(hours=1)
    close_cet = now_cet.replace(hour=21, minute=55, second=0, microsecond=0)
    return round(max(0.0, (close_cet - now_cet).total_seconds() / 3600.0), 4)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_or_none(value: Any, ndigits: int = 4) -> float | None:
    num = _safe_float(value, float("nan"))
    if math.isnan(num) or math.isinf(num):
        return None
    return round(num, ndigits)


def _position_direction(name: str | None, key: str | None, explicit: Any = None) -> int:
    if isinstance(explicit, str):
        txt = explicit.upper()
        if "SHORT" in txt or "BEAR" in txt:
            return -1
        if "LONG" in txt or "BULL" in txt:
            return 1
    joined = f"{name or ''} {key or ''}".upper()
    return -1 if ("SHORT" in joined or "BEAR" in joined) else 1


def _signal_probability(signal_data: dict | None, llm_signals: dict | None, ticker: str) -> float:
    """Blend local signal counts with LLM consensus into P(up)."""
    weights: list[tuple[float, float]] = []

    sig = (signal_data or {}).get(ticker) or {}
    buy_count = _safe_float(sig.get("buy_count"))
    sell_count = _safe_float(sig.get("sell_count"))
    total_votes = buy_count + sell_count
    if total_votes > 0:
        conf = max(0.2, min(1.0, _safe_float(sig.get("weighted_confidence"), 0.5)))
        weights.append((buy_count / total_votes, 1.0 + conf))
    else:
        action = str(sig.get("action", "")).upper()
        if action == "BUY":
            weights.append((0.58, 0.5))
        elif action == "SELL":
            weights.append((0.42, 0.5))

    llm = (llm_signals or {}).get(ticker) or {}
    consensus = llm.get("consensus") or {}
    if isinstance(consensus, dict):
        direction = str(consensus.get("direction", "")).lower()
        confidence = max(0.0, min(1.0, _safe_float(consensus.get("confidence"), 0.0)))
        if direction == "up":
            weights.append((0.5 + 0.4 * confidence, 1.0))
        elif direction == "down":
            weights.append((0.5 - 0.4 * confidence, 1.0))

    if not weights:
        return 0.5

    numer = sum(prob * weight for prob, weight in weights)
    denom = sum(weight for _, weight in weights) or 1.0
    return max(0.05, min(0.95, numer / denom))


def _signal_entry(signal_data: dict | None, ticker: str) -> dict:
    entry = (signal_data or {}).get(ticker) or {}
    return entry if isinstance(entry, dict) else {}


def _signal_extra(signal_data: dict | None, ticker: str) -> dict:
    extra = _signal_entry(signal_data, ticker).get("extra") or {}
    return extra if isinstance(extra, dict) else {}


def _chronos_drift(signal_data: dict | None, ticker: str) -> float | None:
    forecast_signals = (signal_data or {}).get("forecast_signals") or {}
    if not isinstance(forecast_signals, dict):
        return None
    forecast = forecast_signals.get(ticker) or {}
    if not isinstance(forecast, dict):
        return None
    chronos_pct = _safe_float(forecast.get("chronos_24h_pct"), 0.0)
    chronos_conf = _safe_float(forecast.get("chronos_24h_conf"), 0.0)
    if chronos_conf <= 0.3 or chronos_pct == 0:
        return None
    return (chronos_pct / 100.0) * math.sqrt(252.0)


def _expected_terminal_price(price: float, drift: float, hours_remaining: float, is_24h: bool = True) -> float:
    if price <= 0 or hours_remaining <= 0:
        return price
    trading_hours = 24.0 if is_24h else 6.5
    t_years = hours_remaining / (252.0 * trading_hours)
    return price * math.exp(drift * t_years)


def _instrument_price_from_underlying(
    current_price: float,
    current_underlying: float,
    target_underlying: float,
    leverage: float,
    direction_sign: int,
) -> float:
    if current_price <= 0 or current_underlying <= 0:
        return 0.0
    und_return = (target_underlying / current_underlying) - 1.0
    inst_return = direction_sign * leverage * und_return
    return max(0.01, current_price * (1.0 + inst_return))


def _warrant_vol_from_underlying(atr_pct: float, leverage: float) -> float:
    warrant_atr = max(0.1, atr_pct * max(1.0, abs(leverage)))
    return volatility_from_atr(warrant_atr)


def _build_candidate_rows(
    *,
    order_side: str,
    current_order_price: float,
    current_underlying: float,
    expected_terminal_order_price: float,
    underlying_candidates: list[dict],
    units: int,
    leverage: float,
    direction_sign: int,
    fill_vol: float,
    fill_drift: float,
    hours_remaining: float,
) -> list[dict]:
    rows: list[dict] = []
    current_value = current_order_price * max(units, 0)

    for candidate in underlying_candidates:
        target_underlying = _safe_float(candidate.get("price"))
        if target_underlying <= 0:
            continue

        target_order_price = _instrument_price_from_underlying(
            current_order_price,
            current_underlying,
            target_underlying,
            leverage,
            direction_sign,
        )
        if target_order_price <= 0:
            continue

        if order_side == "sell":
            if target_order_price < current_order_price:
                continue
            fill_prob = fill_probability(
                current_order_price, target_order_price, fill_vol, fill_drift, hours_remaining
            )
            filled_value = target_order_price * units
            fallback_value = expected_terminal_order_price * units
            ev_delta = fill_prob * (filled_value - current_value) + (1.0 - fill_prob) * (
                fallback_value - current_value
            )
        else:
            if target_order_price > current_order_price:
                continue
            fill_prob = fill_probability_buy(
                current_order_price, target_order_price, fill_vol, fill_drift, hours_remaining
            )
            filled_cost = target_order_price * units
            expected_terminal_value = expected_terminal_order_price * units
            ev_delta = fill_prob * (expected_terminal_value - filled_cost)
            filled_value = expected_terminal_value
            fallback_value = 0.0

        if fill_prob < _MIN_FILL_PROB:
            continue

        rows.append(
            {
                "label": candidate.get("label"),
                "underlying_target": round(target_underlying, 4),
                "target_price": round(target_order_price, 4),
                "fill_prob": round(fill_prob, 4),
                "ev_delta_sek": round(ev_delta, 2),
                "filled_value_sek": round(filled_value, 2),
                "fallback_value_sek": round(fallback_value, 2),
            }
        )

    rows.sort(key=lambda row: row["ev_delta_sek"], reverse=True)
    return rows


def _candidate_underlying_side(direction_sign: int, order_side: str) -> str:
    if order_side == "sell":
        return "sell" if direction_sign > 0 else "buy"
    return "buy" if direction_sign > 0 else "sell"


def _summary_filters(entry: dict) -> list[str]:
    reasons: list[str] = []
    spread_raw = entry.get("spread_pct")
    spread_pct = _safe_float(spread_raw, float("nan"))
    if not math.isnan(spread_pct) and spread_pct > MIN_SPREAD_PCT:
        reasons.append(f"spread {spread_pct:.2f}% > {MIN_SPREAD_PCT:.2f}%")
    barrier_raw = entry.get("barrier_distance_pct")
    barrier_distance_pct = _safe_float(barrier_raw, float("nan"))
    if not math.isnan(barrier_distance_pct) and barrier_distance_pct < MIN_BARRIER_DISTANCE_PCT:
        reasons.append(f"barrier {barrier_distance_pct:.1f}% < {MIN_BARRIER_DISTANCE_PCT:.1f}%")
    return reasons


def _planned_units(account: dict | None, ask: float) -> int:
    buying_power = _safe_float((account or {}).get("buying_power"))
    if buying_power <= 0 or ask <= 0:
        return 0
    allocation = buying_power * POSITION_SIZE_PCT / 100.0
    if allocation < MIN_TRADE_SEK:
        return 0
    return max(0, int(allocation // ask))


def _buy_direction_matches(signal_prob_up: float, instrument_direction: int) -> bool:
    if instrument_direction > 0:
        return signal_prob_up >= _BUY_SIGNAL_MIN
    return signal_prob_up <= _SELL_SIGNAL_MAX


def build_execution_recommendations(
    positions: dict,
    prices: dict,
    signal_data: dict | None = None,
    llm_signals: dict | None = None,
    warrant_catalog: dict | None = None,
    account: dict | None = None,
    hours_remaining: float | None = None,
) -> dict:
    """Build advisory BUY and SELL target recommendations for metals instruments."""
    if hours_remaining is None:
        hours_remaining = hours_to_metals_close()

    result = {
        "market_close_cet": MARKET_CLOSE_CET,
        "hours_remaining": round(max(0.0, hours_remaining), 2),
        "sell": {},
        "buy": {},
    }

    if hours_remaining <= 0:
        return result

    # Held positions: recommend sell targets for active positions.
    for key, pos in (positions or {}).items():
        if not isinstance(pos, dict) or not pos.get("active"):
            continue
        current = (prices or {}).get(key) or {}
        bid = _safe_float(current.get("bid"))
        underlying_price = _safe_float(current.get("underlying"))
        leverage = _safe_float(current.get("leverage") or pos.get("_leverage"), 1.0)
        if bid <= 0 or underlying_price <= 0 or leverage <= 0:
            continue

        ticker = "XAG-USD" if "silver" in key.lower() else "XAU-USD"
        direction_sign = _position_direction(pos.get("name"), key)
        prob_up = _signal_probability(signal_data, llm_signals, ticker)
        signal_entry = _signal_entry(signal_data, ticker)
        signal_extra = _signal_extra(signal_data, ticker)
        chronos_drift = _chronos_drift(signal_data, ticker)
        atr_pct = _safe_float(signal_entry.get("atr_pct"), _DEFAULT_ATR.get(ticker, 3.0))
        drift_under = drift_from_probability(prob_up, volatility_from_atr(atr_pct))
        if chronos_drift is not None:
            drift_under = 0.7 * drift_under + 0.3 * chronos_drift
        expected_underlying = _expected_terminal_price(underlying_price, drift_under, hours_remaining)
        expected_bid = _instrument_price_from_underlying(
            bid, underlying_price, expected_underlying, leverage, direction_sign
        )
        underlying_side = _candidate_underlying_side(direction_sign, "sell")
        underlying_plan = compute_targets(
            ticker=ticker,
            side=underlying_side,
            price_usd=underlying_price,
            atr_pct=atr_pct,
            p_up=prob_up,
            hours_remaining=hours_remaining,
            indicators=signal_entry,
            extra=signal_extra,
            is_24h=True,
            regime=str(signal_entry.get("regime", "") or ""),
            bb_squeeze=bool((signal_extra.get("volatility_sig_indicators") or {}).get("bb_squeeze_on", False)),
            chronos_drift=chronos_drift,
        )
        fill_vol = _warrant_vol_from_underlying(atr_pct, leverage)
        fill_drift = direction_sign * leverage * drift_under
        units = int(_safe_float(pos.get("units")))
        candidates = _build_candidate_rows(
            order_side="sell",
            current_order_price=bid,
            current_underlying=underlying_price,
            expected_terminal_order_price=expected_bid,
            underlying_candidates=underlying_plan.get("targets", []),
            units=units,
            leverage=leverage,
            direction_sign=direction_sign,
            fill_vol=fill_vol,
            fill_drift=fill_drift,
            hours_remaining=hours_remaining,
        )
        if not candidates:
            continue

        result["sell"][key] = {
            "instrument": pos.get("name", key),
            "underlying": ticker,
            "direction": "SHORT" if direction_sign < 0 else "LONG",
            "current_price": round(bid, 4),
            "current_underlying": round(underlying_price, 4),
            "expected_close_price": round(expected_bid, 4),
            "expected_close_underlying": round(expected_underlying, 4),
            "leverage": round(leverage, 3),
            "units": units,
            "prob_up": round(prob_up, 4),
            "plan_features": {
                "regime": signal_entry.get("regime"),
                "squeeze_warning": bool(underlying_plan.get("squeeze_warning", False)),
                "chronos_drift_annual": _round_or_none(chronos_drift, 4),
                "extra_level_count": len(signal_extra),
            },
            "spread_pct": _round_or_none(
                ((_safe_float(current.get("ask")) - bid) / bid * 100.0) if bid > 0 and _safe_float(current.get("ask")) > 0 else None,
                2,
            ),
            "recommended": candidates[0],
            "candidates": candidates[:5],
        }

    # Flat-state entries: recommend buy targets for warrant catalog.
    for catalog_key, info in (warrant_catalog or {}).items():
        if not isinstance(info, dict):
            continue
        ask = _safe_float(info.get("ask"))
        underlying_price = _safe_float(info.get("underlying_price"))
        leverage = _safe_float(info.get("current_leverage") or info.get("leverage"), 1.0)
        if ask <= 0 or underlying_price <= 0 or leverage <= 0:
            continue

        ticker = str(info.get("underlying") or "")
        if not ticker:
            continue
        direction_sign = _position_direction(info.get("name"), catalog_key, info.get("direction"))
        prob_up = _signal_probability(signal_data, llm_signals, ticker)
        signal_entry = _signal_entry(signal_data, ticker)
        signal_extra = _signal_extra(signal_data, ticker)
        chronos_drift = _chronos_drift(signal_data, ticker)
        if not _buy_direction_matches(prob_up, direction_sign):
            continue

        filter_reasons = _summary_filters(info)
        planned_units = _planned_units(account, ask)
        if planned_units <= 0:
            filter_reasons.append("insufficient buying power")
        if filter_reasons:
            result["buy"][catalog_key] = {
                "instrument": info.get("name", catalog_key),
                "underlying": ticker,
                "direction": "SHORT" if direction_sign < 0 else "LONG",
                "current_price": round(ask, 4),
                "current_underlying": round(underlying_price, 4),
                "leverage": round(leverage, 3),
                "prob_up": round(prob_up, 4),
                "plan_features": {
                    "regime": signal_entry.get("regime"),
                    "squeeze_warning": False,
                    "chronos_drift_annual": _round_or_none(chronos_drift, 4),
                    "extra_level_count": len(signal_extra),
                },
                "filtered_out": filter_reasons,
            }
            continue

        atr_pct = _safe_float(signal_entry.get("atr_pct"), _DEFAULT_ATR.get(ticker, 3.0))
        drift_under = drift_from_probability(prob_up, volatility_from_atr(atr_pct))
        if chronos_drift is not None:
            drift_under = 0.7 * drift_under + 0.3 * chronos_drift
        expected_underlying = _expected_terminal_price(underlying_price, drift_under, hours_remaining)
        expected_order_price = _instrument_price_from_underlying(
            ask, underlying_price, expected_underlying, leverage, direction_sign
        )
        underlying_side = _candidate_underlying_side(direction_sign, "buy")
        underlying_plan = compute_targets(
            ticker=ticker,
            side=underlying_side,
            price_usd=underlying_price,
            atr_pct=atr_pct,
            p_up=prob_up,
            hours_remaining=hours_remaining,
            indicators=signal_entry,
            extra=signal_extra,
            is_24h=True,
            regime=str(signal_entry.get("regime", "") or ""),
            bb_squeeze=bool((signal_extra.get("volatility_sig_indicators") or {}).get("bb_squeeze_on", False)),
            chronos_drift=chronos_drift,
        )
        fill_vol = _warrant_vol_from_underlying(atr_pct, leverage)
        fill_drift = direction_sign * leverage * drift_under
        candidates = _build_candidate_rows(
            order_side="buy",
            current_order_price=ask,
            current_underlying=underlying_price,
            expected_terminal_order_price=expected_order_price,
            underlying_candidates=underlying_plan.get("targets", []),
            units=planned_units,
            leverage=leverage,
            direction_sign=direction_sign,
            fill_vol=fill_vol,
            fill_drift=fill_drift,
            hours_remaining=hours_remaining,
        )
        if not candidates:
            result["buy"][catalog_key] = {
                "instrument": info.get("name", catalog_key),
                "underlying": ticker,
                "direction": "SHORT" if direction_sign < 0 else "LONG",
                "current_price": round(ask, 4),
                "current_underlying": round(underlying_price, 4),
                "leverage": round(leverage, 3),
                "prob_up": round(prob_up, 4),
                "plan_features": {
                    "regime": signal_entry.get("regime"),
                    "squeeze_warning": bool(underlying_plan.get("squeeze_warning", False)),
                    "chronos_drift_annual": _round_or_none(chronos_drift, 4),
                    "extra_level_count": len(signal_extra),
                },
                "filtered_out": ["no viable candidate targets"],
            }
            continue

        leverage_gap = abs(leverage - TARGET_LEVERAGE)
        recommendation = dict(candidates[0])
        recommendation["leverage_gap"] = round(leverage_gap, 3)

        result["buy"][catalog_key] = {
            "instrument": info.get("name", catalog_key),
            "underlying": ticker,
            "direction": "SHORT" if direction_sign < 0 else "LONG",
            "current_price": round(ask, 4),
            "current_underlying": round(underlying_price, 4),
            "expected_close_price": round(expected_order_price, 4),
            "expected_close_underlying": round(expected_underlying, 4),
            "leverage": round(leverage, 3),
            "prob_up": round(prob_up, 4),
            "planned_units": planned_units,
            "plan_features": {
                "regime": signal_entry.get("regime"),
                "squeeze_warning": bool(underlying_plan.get("squeeze_warning", False)),
                "chronos_drift_annual": _round_or_none(chronos_drift, 4),
                "extra_level_count": len(signal_extra),
            },
            "spread_pct": _round_or_none(info.get("spread_pct"), 2),
            "barrier_distance_pct": _round_or_none(info.get("barrier_distance_pct"), 2),
            "recommended": recommendation,
            "candidates": candidates[:5],
        }

    return result
