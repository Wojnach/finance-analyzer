"""Intraday ladder planning for metals instruments.

Pure planning utilities that bridge the underlying day-range model to actual
Avanza gold/silver certificates and warrants. The module is read-only and does
not place or cancel orders.
"""

from __future__ import annotations

from typing import Any

from portfolio.price_targets import compute_targets

SUPPORTED_UNDERLYINGS = {
    "gold": "XAU-USD",
    "silver": "XAG-USD",
}

US_OPEN_FLASH_PHASES = {"pre_open", "post_open_active"}
FLASH_RANGE_FRACTION = 0.35


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def map_underlying_name(name: str | None) -> str | None:
    """Map an Avanza underlying display name to the repo ticker."""
    text = (name or "").strip().lower()
    if text in SUPPORTED_UNDERLYINGS:
        return SUPPORTED_UNDERLYINGS[text]
    if "guld" in text or "gold" in text:
        return "XAU-USD"
    if "silver" in text:
        return "XAG-USD"
    return None


def translate_underlying_target(
    current_instrument_price: float,
    current_underlying_price: float,
    target_underlying_price: float,
    leverage: float,
    direction_sign: int = 1,
) -> float:
    """Approximate instrument price from an underlying target."""
    if current_instrument_price <= 0 or current_underlying_price <= 0:
        return 0.0
    underlying_return = (target_underlying_price / current_underlying_price) - 1.0
    instrument_return = direction_sign * leverage * underlying_return
    return round(max(0.01, current_instrument_price * (1.0 + instrument_return)), 4)


def flash_crash_drop_pct(analysis: dict | None) -> float:
    """Heuristic downside allowance for the US-open flash-crash window.

    The silver monitor records that the first US-open hour often has a much
    wider range than the rest of the day. For entry ladders, we only widen bids
    during the active/pre-open window; otherwise the reserve bid is disabled.
    """
    market_open = (analysis or {}).get("us_market_open") or {}
    phase = str(market_open.get("phase") or "").strip().lower()
    if phase not in US_OPEN_FLASH_PHASES:
        return 0.0

    stats = market_open.get("historical_stats") or {}
    mean_drop_pct = abs(_safe_float(stats.get("post_open_mean_pct")))
    avg_range_pct = abs(_safe_float(stats.get("post_open_avg_range_pct")))
    return max(mean_drop_pct, avg_range_pct * FLASH_RANGE_FRACTION)


def build_intraday_ladder(
    signal_entry: dict,
    focus_probabilities: dict | None,
    *,
    ticker: str,
    current_instrument_price: float,
    current_underlying_price: float,
    leverage: float,
    hours_remaining: float,
    analysis: dict | None = None,
    direction_sign: int = 1,
) -> dict:
    """Build a working bid / flash reserve / exit ladder for one instrument."""
    p_up = _safe_float((focus_probabilities or {}).get("3h", {}).get("probability"), 0.5)
    extra = signal_entry.get("extra") if isinstance(signal_entry, dict) else None
    squeeze_on = bool(((extra or {}).get("volatility_sig_indicators") or {}).get("bb_squeeze_on"))

    buy_targets = compute_targets(
        ticker,
        side="buy",
        price_usd=_safe_float(signal_entry.get("price_usd"), current_underlying_price),
        atr_pct=_safe_float(signal_entry.get("atr_pct"), 0.3),
        p_up=p_up,
        hours_remaining=hours_remaining,
        indicators=signal_entry,
        extra=extra,
        regime=str(signal_entry.get("regime") or ""),
        bb_squeeze=squeeze_on,
    )
    sell_targets = compute_targets(
        ticker,
        side="sell",
        price_usd=_safe_float(signal_entry.get("price_usd"), current_underlying_price),
        atr_pct=_safe_float(signal_entry.get("atr_pct"), 0.3),
        p_up=p_up,
        hours_remaining=hours_remaining,
        indicators=signal_entry,
        extra=extra,
        regime=str(signal_entry.get("regime") or ""),
        bb_squeeze=squeeze_on,
    )

    working_underlying = min(
        _safe_float((buy_targets.get("recommended") or {}).get("price"), current_underlying_price),
        _safe_float((buy_targets.get("extremes") or {}).get("p25"), current_underlying_price),
    )
    mean_underlying = _safe_float((buy_targets.get("recommended") or {}).get("price"), working_underlying)
    flash_drop_pct = flash_crash_drop_pct(analysis)
    flash_underlying = 0.0
    if flash_drop_pct > 0:
        flash_underlying = min(
            working_underlying,
            current_underlying_price * (1.0 - flash_drop_pct / 100.0),
        )

    exit_underlying = _safe_float((sell_targets.get("recommended") or {}).get("price"), current_underlying_price)
    stretch_exit_underlying = _safe_float((sell_targets.get("extremes") or {}).get("p75"), exit_underlying)

    result = {
        "ticker": ticker,
        "current_underlying": round(current_underlying_price, 4),
        "current_instrument": round(current_instrument_price, 4),
        "hours_remaining": round(hours_remaining, 2),
        "working_underlying": round(working_underlying, 4),
        "mean_underlying": round(mean_underlying, 4),
        "exit_underlying": round(exit_underlying, 4),
        "stretch_exit_underlying": round(stretch_exit_underlying, 4),
        "flash_crash_drop_pct": round(flash_drop_pct, 4),
        "working_price": translate_underlying_target(
            current_instrument_price,
            current_underlying_price,
            working_underlying,
            leverage,
            direction_sign=direction_sign,
        ),
        "mean_price": translate_underlying_target(
            current_instrument_price,
            current_underlying_price,
            mean_underlying,
            leverage,
            direction_sign=direction_sign,
        ),
        "exit_price": translate_underlying_target(
            current_instrument_price,
            current_underlying_price,
            exit_underlying,
            leverage,
            direction_sign=direction_sign,
        ),
        "stretch_exit_price": translate_underlying_target(
            current_instrument_price,
            current_underlying_price,
            stretch_exit_underlying,
            leverage,
            direction_sign=direction_sign,
        ),
        "buy_targets": buy_targets,
        "sell_targets": sell_targets,
    }
    result["flash_price"] = (
        translate_underlying_target(
            current_instrument_price,
            current_underlying_price,
            flash_underlying,
            leverage,
            direction_sign=direction_sign,
        )
        if flash_underlying > 0
        else 0.0
    )
    result["flash_underlying"] = round(flash_underlying, 4) if flash_underlying > 0 else 0.0
    return result
