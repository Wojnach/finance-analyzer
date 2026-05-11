"""Pure-function tier construction for the grid market-maker.

Given a live quote, a per-leg SEK budget, and optional barrier metadata,
``build_buy_ladder`` returns an ordered list of ``Tier`` records describing
each buy-limit level the grid fisher should place. The math is deliberately
isolated from order-placement code so it can be unit-tested without an
Avanza session.

Inputs are in warrant SEK price units. Tier spacing is applied to the bid
to find buy prices; knockout-proximity skip is done in underlying-price
space when ``barrier`` metadata is supplied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from portfolio.grid_fisher_config import (
    GRID_KNOCKOUT_SAFETY_PCT,
    GRID_LEG_SEK,
    GRID_TIER_SPACING_PCT,
    GRID_TIERS,
)


# Avanza minimum-courtage threshold. Orders below this incur a flat fee
# that destroys the per-leg edge. See memory ``feedback_min_order_size_1000_sek.md``.
MIN_LEG_SEK = 1000


@dataclass(frozen=True)
class Tier:
    """One rung of the buy ladder."""

    index: int
    price: float
    qty: int
    notional_sek: float
    spacing_pct: float
    skip_reason: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return self.skip_reason is None


def _round_price(price: float) -> float:
    """Round to two decimals — Avanza warrants quote to öre."""
    if price <= 0:
        return 0.0
    return round(price, 2)


def _underlying_move_for_warrant_drop(
    warrant_drop_pct: float, leverage: float
) -> float:
    """Approximate underlying-price move implied by a warrant-price drop.

    Constant-leverage approximation: dWarrant/Warrant ~ leverage *
    dUnderlying/Underlying. Real path-dependent compounding is ignored —
    we use this only for the barrier-proximity guard, which is conservative
    when the approximation is wrong (it overestimates the underlying move).
    """
    if leverage <= 0:
        return warrant_drop_pct
    return warrant_drop_pct / leverage


def _tier_skip_for_knockout(
    tier_price: float,
    underlying_price: Optional[float],
    barrier: Optional[float],
    leverage: Optional[float],
    direction: str,
    bid: float,
) -> Optional[str]:
    """Return a non-empty reason string if the tier sits too close to the barrier.

    Direction matters: LONG warrants get knocked out when underlying falls
    below barrier; SHORT warrants when underlying rises above. We compute the
    *implied* underlying level at the tier price and check distance.
    """
    if barrier is None or underlying_price is None or leverage is None:
        return None
    if bid <= 0 or underlying_price <= 0 or leverage <= 0:
        return None

    warrant_drop_pct = (bid - tier_price) / bid * 100.0
    underlying_move_pct = _underlying_move_for_warrant_drop(
        warrant_drop_pct, leverage
    )
    if direction == "LONG":
        implied_underlying = underlying_price * (1 - underlying_move_pct / 100.0)
        distance_pct = (implied_underlying - barrier) / barrier * 100.0
    else:
        implied_underlying = underlying_price * (1 + underlying_move_pct / 100.0)
        distance_pct = (barrier - implied_underlying) / barrier * 100.0

    if distance_pct < GRID_KNOCKOUT_SAFETY_PCT:
        return (
            f"knockout_proximity:{distance_pct:.1f}pct"
            f"<{GRID_KNOCKOUT_SAFETY_PCT}"
        )
    return None


def build_buy_ladder(
    *,
    bid: float,
    leg_sek: float = GRID_LEG_SEK,
    n_tiers: int = GRID_TIERS,
    spacing_pct: Sequence[float] = GRID_TIER_SPACING_PCT,
    direction: str = "LONG",
    underlying_price: Optional[float] = None,
    barrier: Optional[float] = None,
    leverage: Optional[float] = None,
) -> list[Tier]:
    """Construct the buy ladder for one instrument.

    Args:
        bid: current best bid in warrant SEK.
        leg_sek: target SEK notional per leg. Each leg's quantity rounds
            down so the actual notional never exceeds ``leg_sek``. Legs
            whose realised notional would be below ``MIN_LEG_SEK`` are
            marked with ``skip_reason="below_min_order"``.
        n_tiers: number of tiers to build.
        spacing_pct: per-tier pct below bid. Must have at least ``n_tiers``
            entries.
        direction: ``"LONG"`` or ``"SHORT"`` — controls barrier check direction.
        underlying_price: live spot of the underlying (e.g. XAG-USD price).
            If None, knockout safety is skipped (e.g. for certificates with
            no barrier).
        barrier: knockout barrier in underlying-price units. None => no
            knockout (e.g. constant-leverage certs).
        leverage: warrant leverage. None => no knockout check applied.

    Returns:
        A list of ``Tier`` records of length ``n_tiers``. Tiers that fail
        knockout or min-size checks are still returned with their
        ``skip_reason`` populated, so callers can log every rejection.
    """
    if bid <= 0 or not math.isfinite(bid):
        raise ValueError(f"bid must be a positive finite number, got {bid!r}")
    if n_tiers < 1:
        raise ValueError(f"n_tiers must be >= 1, got {n_tiers}")
    if len(spacing_pct) < n_tiers:
        raise ValueError(
            f"spacing_pct has {len(spacing_pct)} entries, need >= {n_tiers}"
        )
    if direction not in ("LONG", "SHORT"):
        raise ValueError(f"direction must be LONG or SHORT, got {direction!r}")

    tiers: list[Tier] = []
    for i in range(n_tiers):
        pct = float(spacing_pct[i])
        if pct <= 0:
            raise ValueError(
                f"spacing_pct[{i}] must be positive, got {pct} (deeper tiers below bid)"
            )
        raw_price = bid * (1 - pct / 100.0)
        price = _round_price(raw_price)
        if price <= 0:
            tiers.append(
                Tier(
                    index=i,
                    price=price,
                    qty=0,
                    notional_sek=0.0,
                    spacing_pct=pct,
                    skip_reason="non_positive_price",
                )
            )
            continue

        qty = int(leg_sek // price)
        notional = qty * price
        skip: Optional[str] = None
        if qty <= 0 or notional < MIN_LEG_SEK:
            skip = f"below_min_order:{notional:.0f}sek"

        if skip is None:
            knockout_skip = _tier_skip_for_knockout(
                tier_price=price,
                underlying_price=underlying_price,
                barrier=barrier,
                leverage=leverage,
                direction=direction,
                bid=bid,
            )
            if knockout_skip:
                skip = knockout_skip

        tiers.append(
            Tier(
                index=i,
                price=price,
                qty=qty,
                notional_sek=notional,
                spacing_pct=pct,
                skip_reason=skip,
            )
        )
    return tiers


def build_exit_levels(fill_price: float, target_pct: float, stop_pct: float) -> tuple[float, float]:
    """Return (sell_limit_price, stop_loss_price) for a filled buy at *fill_price*.

    Both prices are rounded to öre. ``target_pct`` and ``stop_pct`` are
    positive percentages — the function applies the sign.
    """
    if fill_price <= 0 or not math.isfinite(fill_price):
        raise ValueError(f"fill_price must be positive, got {fill_price!r}")
    if target_pct <= 0 or stop_pct <= 0:
        raise ValueError(
            f"target_pct and stop_pct must be positive, got "
            f"target={target_pct}, stop={stop_pct}"
        )
    sell_limit = _round_price(fill_price * (1 + target_pct / 100.0))
    stop_loss = _round_price(fill_price * (1 - stop_pct / 100.0))
    return sell_limit, stop_loss


def total_planned_notional(tiers: Sequence[Tier]) -> float:
    """Sum of notional across active (non-skipped) tiers."""
    return sum(t.notional_sek for t in tiers if t.is_active)


def active_tiers(tiers: Sequence[Tier]) -> list[Tier]:
    """Filter to tiers without a skip_reason."""
    return [t for t in tiers if t.is_active]
