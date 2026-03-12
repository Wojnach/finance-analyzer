"""Cost model for trade execution — fees, spread, and slippage.

Provides instrument-specific cost estimation for the exit optimizer.
Supports Avanza warrants, stocks, and crypto exchanges.

Usage:
    from portfolio.cost_model import get_cost_model
    costs = get_cost_model("warrant")
    exit_cost = costs.total_cost_sek(trade_value_sek=50000)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """All-in cost model for a single trade (one leg).

    Attributes:
        courtage_bps: Brokerage fee in basis points (e.g., 6.9 = 0.069%).
        min_fee_sek: Minimum fee in SEK (courtage floor).
        spread_bps: Half-spread cost in basis points. For a sell, you cross
            from mid to bid, losing half the spread.
        slippage_bps: Expected adverse price movement in basis points
            between decision and fill (market impact + latency).
        label: Human-readable name for this cost model.
    """
    courtage_bps: float = 0.0
    min_fee_sek: float = 0.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    label: str = "default"

    def total_cost_sek(self, trade_value_sek: float) -> float:
        """Compute total one-way cost for a trade of given value.

        Returns:
            Total cost in SEK (always non-negative).
        """
        if trade_value_sek <= 0:
            return 0.0
        courtage = max(trade_value_sek * self.courtage_bps / 10_000, self.min_fee_sek)
        spread = trade_value_sek * self.spread_bps / 10_000
        slippage = trade_value_sek * self.slippage_bps / 10_000
        return courtage + spread + slippage

    def total_cost_pct(self) -> float:
        """Total cost as a percentage of trade value (excluding min fee)."""
        return (self.courtage_bps + self.spread_bps + self.slippage_bps) / 100.0

    def round_trip_pct(self) -> float:
        """Round-trip cost (buy + sell) as a percentage."""
        return self.total_cost_pct() * 2


# ---------------------------------------------------------------------------
# Preset cost models for known instrument types
# ---------------------------------------------------------------------------

# Avanza warrants/certificates: 0 courtage on many, spread is the real cost.
# Typical MINI silver spread: 0.6-1.0% (30-50 bps half-spread).
WARRANT_COSTS = CostModel(
    courtage_bps=0.0,
    min_fee_sek=0.0,
    spread_bps=40.0,    # 0.40% half-spread (conservative)
    slippage_bps=10.0,   # 0.10% slippage on market orders
    label="avanza_warrant",
)

# Avanza stocks (Mini courtage class): 0.069% with 1 SEK minimum
STOCK_COSTS = CostModel(
    courtage_bps=6.9,
    min_fee_sek=1.0,
    spread_bps=5.0,      # 0.05% half-spread (liquid US stocks)
    slippage_bps=2.0,     # 0.02% slippage
    label="avanza_stock",
)

# Crypto (Binance-equivalent fees, used for simulated portfolio)
CRYPTO_COSTS = CostModel(
    courtage_bps=5.0,     # 0.05% taker fee
    min_fee_sek=0.0,
    spread_bps=5.0,       # 0.05% half-spread
    slippage_bps=5.0,     # 0.05% slippage
    label="crypto",
)

# Elongir silver bot (specific spread from config)
ELONGIR_COSTS = CostModel(
    courtage_bps=25.0,    # 0.25% commission
    min_fee_sek=0.0,
    spread_bps=40.0,      # 0.40% half-spread
    slippage_bps=10.0,    # 0.10% slippage
    label="elongir_silver",
)

_COST_MODELS = {
    "warrant": WARRANT_COSTS,
    "stock": STOCK_COSTS,
    "crypto": CRYPTO_COSTS,
    "elongir": ELONGIR_COSTS,
}


def get_cost_model(instrument_type: str) -> CostModel:
    """Look up cost model by instrument type.

    Args:
        instrument_type: One of "warrant", "stock", "crypto", "elongir".

    Returns:
        CostModel for the instrument type. Falls back to STOCK_COSTS if unknown.
    """
    return _COST_MODELS.get(instrument_type, STOCK_COSTS)
