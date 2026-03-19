"""Pre-trade validation -- sanity checks before order placement.

Validates trade parameters (price, volume, spread, cash, position size) before
any order is placed. Returns a ValidationResult with pass/fail, reason, and
optional warnings for near-limit conditions.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("portfolio.trade_validation")


@dataclass
class ValidationResult:
    """Result of pre-trade validation."""
    valid: bool
    reason: str = ""
    warnings: list = field(default_factory=list)


def validate_trade(
    action: str,
    price: float,
    volume: float,
    cash_available: float,
    bid: float | None = None,
    ask: float | None = None,
    last_known_price: float | None = None,
    max_spread_pct: float = 2.0,
    max_cash_pct: float = 50.0,
    min_order_sek: float = 500.0,
    max_price_deviation_pct: float = 5.0,
) -> ValidationResult:
    """Validate a trade before placing it.

    Checks (in order):
    1. Basic parameter validation (positive price, volume, valid action)
    2. Minimum order size
    3. Cash sufficiency (BUY only)
    4. Position size as % of cash (BUY only)
    5. Bid/ask spread width
    6. Price sanity vs last known price

    Returns ValidationResult with valid=True/False, reason, and warnings.
    """
    warnings: list[str] = []

    # --- 1. Basic parameter validation ---
    if action not in ("BUY", "SELL"):
        return ValidationResult(False, f"Invalid action: {action}")
    if price <= 0:
        return ValidationResult(False, f"Invalid price: {price}")
    if volume <= 0:
        return ValidationResult(False, f"Invalid volume: {volume}")

    order_value = price * volume

    # --- 2. Minimum order size ---
    if order_value < min_order_sek:
        return ValidationResult(
            False,
            f"Order value {order_value:.0f} SEK below minimum {min_order_sek:.0f} SEK",
        )

    # --- 3 & 4. BUY-specific checks ---
    if action == "BUY":
        # Cash sufficiency
        if order_value > cash_available:
            return ValidationResult(
                False,
                f"Insufficient cash: need {order_value:.0f} SEK, have {cash_available:.0f} SEK",
            )
        # Position size limit
        if cash_available > 0:
            cash_pct = (order_value / cash_available) * 100
            if cash_pct > max_cash_pct:
                return ValidationResult(
                    False,
                    f"Position too large: {cash_pct:.1f}% of cash (max {max_cash_pct:.1f}%)",
                )

    # --- 5. Bid/ask spread check ---
    if bid is not None and ask is not None and bid > 0:
        spread_pct = ((ask - bid) / bid) * 100
        if spread_pct > max_spread_pct:
            return ValidationResult(
                False,
                f"Spread too wide: {spread_pct:.2f}% (max {max_spread_pct:.1f}%)",
            )
        if spread_pct > max_spread_pct * 0.7:
            warnings.append(f"Spread warning: {spread_pct:.2f}% approaching limit")

    # --- 6. Price sanity vs last known ---
    if last_known_price is not None and last_known_price > 0:
        deviation_pct = abs(price - last_known_price) / last_known_price * 100
        if deviation_pct > max_price_deviation_pct:
            return ValidationResult(
                False,
                f"Price deviation {deviation_pct:.2f}% from last known "
                f"{last_known_price:.2f} (max {max_price_deviation_pct:.1f}%)",
            )
        if deviation_pct > max_price_deviation_pct * 0.7:
            warnings.append(f"Price moved {deviation_pct:.2f}% from last known")

    logger.debug(
        "Trade validated: %s %.4f @ %.2f SEK (value %.0f SEK)%s",
        action,
        volume,
        price,
        order_value,
        f" -- warnings: {warnings}" if warnings else "",
    )
    return ValidationResult(True, "All checks passed", warnings)
