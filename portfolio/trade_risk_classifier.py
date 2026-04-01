"""Trade Risk Classifier — scores proposed trades as LOW / MEDIUM / HIGH risk.

Scoring system (0-11 total points):
  Position size risk   (0-3): >20% = 3, >10% = 2, >5% = 1
  Regime risk          (0-3): trending-up=0, trending-down=1, ranging=2, high-vol=3, capitulation=3
  Counter-trend risk   (0-2): BUY in trending-down or SELL in trending-up = 2
  Weak consensus       (0-2): <60% = 2, <70% = 1
  Low confidence       (0-1): <50% = 1
  Concentration risk   (0-2): total exposure >40% = 2, >25% = 1

Thresholds: 0-3 = LOW, 4-6 = MEDIUM, 7+ = HIGH.
HOLD is always LOW (score 0, no factors).
"""

import logging

logger = logging.getLogger("portfolio.trade_risk_classifier")

# Regime risk mapping
_REGIME_SCORES = {
    "trending-up": 0,
    "trending-down": 1,
    "ranging": 2,
    "high-vol": 3,
    "capitulation": 3,
}


def classify_trade_risk(
    action: str,
    confidence: float,
    position_pct: float,
    regime: str,
    consensus_ratio: float,
    existing_exposure_pct: float = 0.0,
) -> dict:
    """Classify a proposed trade into LOW / MEDIUM / HIGH risk.

    Parameters
    ----------
    action : str
        Trade action: "BUY", "SELL", or "HOLD".
    confidence : float
        Signal confidence (0.0-1.0).
    position_pct : float
        Proposed position size as percentage of portfolio (0-100).
    regime : str
        Current market regime (trending-up, trending-down, ranging, high-vol, capitulation).
    consensus_ratio : float
        Signal consensus ratio (0.0-1.0). Fraction of voters that agree.
    existing_exposure_pct : float
        Current total portfolio exposure percentage (0-100).

    Returns
    -------
    dict
        {"level": "LOW"/"MEDIUM"/"HIGH", "score": int, "factors": list[str]}
    """
    action_upper = action.upper()

    # HOLD is always LOW risk
    if action_upper == "HOLD":
        return {"level": "LOW", "score": 0, "factors": []}

    score = 0
    factors = []

    # 1. Position size risk (0-3)
    if position_pct > 20:
        score += 3
        factors.append(f"large position ({position_pct:.1f}% > 20%)")
    elif position_pct > 10:
        score += 2
        factors.append(f"medium position ({position_pct:.1f}% > 10%)")
    elif position_pct > 5:
        score += 1
        factors.append(f"notable position ({position_pct:.1f}% > 5%)")

    # 2. Regime risk (0-3)
    regime_lower = regime.lower()
    regime_score = _REGIME_SCORES.get(regime_lower, 0)
    if regime_score > 0:
        score += regime_score
        factors.append(f"regime={regime_lower} (+{regime_score})")

    # 3. Counter-trend risk (0-2)
    if (action_upper == "BUY" and regime_lower == "trending-down") or \
       (action_upper == "SELL" and regime_lower == "trending-up"):
        score += 2
        factors.append(f"counter-trend {action_upper} in {regime_lower}")

    # 4. Weak consensus (0-2)
    if consensus_ratio < 0.60:
        score += 2
        factors.append(f"weak consensus ({consensus_ratio:.0%} < 60%)")
    elif consensus_ratio < 0.70:
        score += 1
        factors.append(f"moderate consensus ({consensus_ratio:.0%} < 70%)")

    # 5. Low confidence (0-1)
    if confidence < 0.50:
        score += 1
        factors.append(f"low confidence ({confidence:.0%} < 50%)")

    # 6. Concentration risk (0-2)
    if existing_exposure_pct > 40:
        score += 2
        factors.append(f"high concentration ({existing_exposure_pct:.1f}% > 40%)")
    elif existing_exposure_pct > 25:
        score += 1
        factors.append(f"moderate concentration ({existing_exposure_pct:.1f}% > 25%)")

    # Determine level
    if score <= 3:
        level = "LOW"
    elif score <= 6:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return {"level": level, "score": score, "factors": factors}
