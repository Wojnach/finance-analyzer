"""Single source of truth for asset correlation priors.

Used by:
- monte_carlo_risk.py: numeric correlation strengths for copula simulation
- risk_management.py: binary correlated-pair adjacency for position risk checks
"""

CORRELATION_PRIORS: dict[tuple[str, str], float] = {
    ("BTC-USD", "ETH-USD"): 0.75,
    ("XAG-USD", "XAU-USD"): 0.85,
}


def get_prior(ticker_a: str, ticker_b: str) -> float:
    """Look up prior correlation for a pair (order-independent). Returns 0.0 if unknown."""
    return CORRELATION_PRIORS.get(
        (ticker_a, ticker_b),
        CORRELATION_PRIORS.get((ticker_b, ticker_a), 0.0),
    )


def get_correlated_pairs() -> dict[str, list[str]]:
    """Derive binary adjacency list from numeric priors.

    Returns {ticker: [correlated_tickers]} for all pairs with prior > 0.
    """
    pairs: dict[str, list[str]] = {}
    for (a, b) in CORRELATION_PRIORS:
        pairs.setdefault(a, []).append(b)
        pairs.setdefault(b, []).append(a)
    return pairs
