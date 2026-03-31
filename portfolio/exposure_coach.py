"""Exposure coach — portfolio-level exposure recommendation.

Synthesizes market health score, regime detection, and portfolio state
into an exposure_ceiling multiplier (0.0-1.0) that scales maximum
allocation.  This is ADVISORY — it goes into agent_summary.json for
Layer 2 to consider, and optionally scales Kelly sizing.

Does NOT create new triggers or invoke Layer 2.
"""

import logging
from datetime import UTC, datetime

logger = logging.getLogger("portfolio.exposure_coach")

# Exposure ceiling floors (never go below this)
MIN_CEILING = 0.2

# Zone-based base ceilings
_ZONE_CEILINGS = {
    "danger": 0.3,
    "caution": 0.6,
    "healthy": 1.0,
}

# Regime adjustments (multiplicative)
_REGIME_MULTIPLIERS = {
    "trending-down": 0.7,
    "high-vol": 0.8,
    "range-bound": 0.9,
    "trending-up": 1.0,
}

# Bias classification
_BIAS_MAP = {
    "danger": "defensive",
    "caution": "defensive",
    "healthy": "neutral",
}


def compute_exposure_recommendation(
    market_health: dict | None = None,
    regime: str = "range-bound",
    portfolio_concentration: float | None = None,
) -> dict:
    """Compute portfolio-level exposure recommendation.

    Args:
        market_health: output of market_health.get_market_health()
        regime: current regime from indicators.detect_regime()
        portfolio_concentration: fraction of portfolio in single largest
            position (0.0-1.0), used to flag concentration risk

    Returns:
        dict with exposure_ceiling, rationale, bias, etc.
    """
    # Default: no data = neutral recommendation
    if market_health is None:
        return {
            "exposure_ceiling": 1.0,
            "rationale": "No market health data available — using default exposure",
            "market_health_zone": "unknown",
            "market_health_score": None,
            "regime": regime,
            "new_entries_allowed": True,
            "bias": "neutral",
            "updated_at": datetime.now(UTC).isoformat(),
        }

    zone = market_health.get("zone", "healthy")
    score = market_health.get("score", 50)

    # Base ceiling from zone
    ceiling = _ZONE_CEILINGS.get(zone, 1.0)

    # Regime adjustment
    regime_mult = _REGIME_MULTIPLIERS.get(regime, 1.0)
    ceiling *= regime_mult

    # Floor enforcement
    ceiling = max(ceiling, MIN_CEILING)

    # Round to 2 decimal places
    ceiling = round(ceiling, 2)

    # New entries allowed?
    # Block new entries only in danger zone with bearish regime
    new_entries = not (zone == "danger" and regime in ("trending-down", "high-vol"))

    # Bias
    if zone == "danger" or (zone == "caution" and regime == "trending-down"):
        bias = "defensive"
    elif zone == "healthy" and regime == "trending-up":
        bias = "growth"
    else:
        bias = "neutral"

    # Build rationale
    parts = []
    parts.append(f"Market {zone} (score {score})")
    if regime != "range-bound":
        parts.append(f"{regime} regime")
    if portfolio_concentration and portfolio_concentration > 0.3:
        parts.append(f"high concentration ({portfolio_concentration:.0%})")
    rationale = " + ".join(parts)

    return {
        "exposure_ceiling": ceiling,
        "rationale": rationale,
        "market_health_zone": zone,
        "market_health_score": score,
        "regime": regime,
        "new_entries_allowed": new_entries,
        "bias": bias,
        "updated_at": datetime.now(UTC).isoformat(),
    }
