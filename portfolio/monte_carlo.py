"""Monte Carlo price simulation engine.

Generates stochastic price paths using Geometric Brownian Motion (GBM)
with antithetic variates for variance reduction. Converts directional
signal probabilities into price quantile bands, stop-loss probabilities,
and expected return distributions.

Short-term focus: 3h, 1d, 3d horizons for active trading decisions.

Usage:
    from portfolio.monte_carlo import simulate_ticker
    result = simulate_ticker("BTC-USD", agent_summary)
    # result = {
    #   "price_bands_1d": {"p5": 82100, "p25": 84300, "p50": 86200, ...},
    #   "p_stop_hit_1d": 0.12,
    #   "expected_return_1d": {"mean_pct": 0.3, "std_pct": 2.1, "skew": 0.1},
    #   ...
    # }
"""

import logging
import math

import numpy as np
from scipy.stats import norm

logger = logging.getLogger("portfolio.monte_carlo")

# Default parameters
DEFAULT_N_PATHS = 10_000   # 5K pairs with antithetic variates
DEFAULT_HORIZONS = [1, 3]  # days
MIN_VOLATILITY = 0.05      # 5% annualized floor (prevents degenerate sims)


# ---------------------------------------------------------------------------
# Volatility & drift estimation from existing system data
# ---------------------------------------------------------------------------

def volatility_from_atr(atr_pct: float, period: int = 14) -> float:
    """Convert ATR% (14-period) to annualized volatility.

    ATR measures average true range over `period` candles. For hourly candles
    (our primary timeframe), we annualize: vol = atr_frac * sqrt(trading_periods/period).

    Uses 252 trading days (standard for stocks/crypto).

    Args:
        atr_pct: ATR as percentage of price (e.g., 3.5 means 3.5%).
        period: ATR lookback period (default 14).

    Returns:
        Annualized volatility as a decimal (e.g., 0.20 = 20%).
    """
    atr_frac = atr_pct / 100.0
    annual_factor = math.sqrt(252.0 / period)
    vol = atr_frac * annual_factor
    return max(vol, MIN_VOLATILITY)


def drift_from_probability(p_up: float, volatility: float) -> float:
    """Convert directional probability P(up) into annualized drift.

    Uses the inverse of the GBM CDF relationship:
        P(S_T > S_0) = N((mu - 0.5*sigma^2)*sqrt(T) / (sigma*sqrt(T)))

    For 1-day horizon (T = 1/252):
        mu = sigma * N_inv(p_up) * sqrt(252) + 0.5 * sigma^2

    This ensures the GBM simulation produces paths where the fraction
    ending above spot matches the input probability.

    Args:
        p_up: Probability of price being higher at horizon (0.0-1.0).
        volatility: Annualized volatility (decimal).

    Returns:
        Annualized drift (decimal). Positive = upward bias.
    """
    # Clamp p_up to avoid infinite drift at extremes
    p_up = max(0.01, min(0.99, p_up))

    # N_inv(p_up) gives the z-score for the desired probability
    z = norm.ppf(p_up)

    # mu = sigma * z * sqrt(252) + 0.5 * sigma^2
    # This is derived from P(S_T > S_0) = N((mu - 0.5*sigma^2)*sqrt(T) / sigma*sqrt(T))
    mu = volatility * z * math.sqrt(252.0) + 0.5 * volatility**2

    return mu


# ---------------------------------------------------------------------------
# Core simulation engine
# ---------------------------------------------------------------------------

class MonteCarloEngine:
    """Geometric Brownian Motion price path simulator with antithetic variates.

    Generates terminal price distributions for short-term horizons (hours to days).
    Uses antithetic variates for 50-75% variance reduction at zero extra cost.

    Args:
        price: Current asset price (USD).
        volatility: Annualized volatility (decimal, e.g., 0.20 = 20%).
        drift: Annualized drift (decimal). Use drift_from_probability() to derive.
        horizon_days: Simulation horizon in days (supports fractional, e.g., 0.125 = 3h).
        n_paths: Number of simulated terminal prices to generate.
        seed: Random seed for reproducibility.
    """

    def __init__(self, price: float, volatility: float, drift: float = 0.0,
                 horizon_days: float = 1.0, n_paths: int = DEFAULT_N_PATHS,
                 seed: int | None = None):
        self.price = price
        self.volatility = max(volatility, MIN_VOLATILITY)
        self.drift = drift
        self.horizon_days = horizon_days
        self.n_paths = n_paths
        self.seed = seed
        self._terminal_prices = None

    def simulate_paths(self) -> np.ndarray:
        """Generate terminal prices via GBM with antithetic variates.

        Formula: S_T = S0 * exp((mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)

        Antithetic variates: for each random draw Z, we also compute the
        path with -Z. This creates negative correlation between paired
        estimates, reducing variance of the mean estimator.

        Returns:
            1D array of terminal prices, shape (n_paths,).
        """
        rng = np.random.default_rng(self.seed)

        T = self.horizon_days / 252.0  # Convert to years
        sigma = self.volatility
        mu = self.drift

        # Draw half the paths, use antithetic for the other half
        n_half = self.n_paths // 2
        Z = rng.standard_normal(n_half)

        # Deterministic component
        drift_term = (mu - 0.5 * sigma**2) * T
        vol_term = sigma * math.sqrt(T)

        # Original paths
        log_returns_pos = drift_term + vol_term * Z
        # Antithetic paths (negate Z)
        log_returns_neg = drift_term + vol_term * (-Z)

        # Combine
        log_returns = np.concatenate([log_returns_pos, log_returns_neg])
        terminal_prices = self.price * np.exp(log_returns)

        # If n_paths is odd, add one more path
        if self.n_paths % 2 == 1:
            extra_Z = rng.standard_normal(1)
            extra_price = self.price * np.exp(drift_term + vol_term * extra_Z)
            terminal_prices = np.concatenate([terminal_prices, extra_price])

        self._terminal_prices = terminal_prices
        return terminal_prices

    def _ensure_simulated(self):
        """Run simulation if not already done."""
        if self._terminal_prices is None:
            self.simulate_paths()

    def price_quantiles(self, percentiles: list[int] | None = None) -> dict:
        """Extract price quantile bands from simulated distribution.

        Args:
            percentiles: List of percentiles to compute (default: [5, 25, 50, 75, 95]).

        Returns:
            Dict mapping percentile → price (e.g., {5: 87.2, 50: 90.1, 95: 93.5}).
        """
        self._ensure_simulated()
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]

        values = np.percentile(self._terminal_prices, percentiles)
        return {p: round(float(v), 2) for p, v in zip(percentiles, values)}

    def probability_below(self, threshold: float) -> float:
        """Compute P(terminal price < threshold).

        Useful for stop-loss probability: "what's the chance price hits my stop?"

        Args:
            threshold: Price level to test.

        Returns:
            Probability (0.0-1.0).
        """
        self._ensure_simulated()
        if threshold <= 0:
            return 0.0
        return float(np.mean(self._terminal_prices < threshold))

    def probability_above(self, threshold: float) -> float:
        """Compute P(terminal price > threshold).

        Useful for profit target probability.

        Args:
            threshold: Price level to test.

        Returns:
            Probability (0.0-1.0).
        """
        self._ensure_simulated()
        if threshold <= 0:
            return 1.0
        return float(np.mean(self._terminal_prices > threshold))

    def expected_return(self) -> dict:
        """Compute return distribution statistics.

        Returns:
            Dict with mean_pct, std_pct, skew of the return distribution.
        """
        self._ensure_simulated()
        returns = (self._terminal_prices - self.price) / self.price * 100.0

        mean_pct = float(np.mean(returns))
        std_pct = float(np.std(returns))

        # Skewness
        if std_pct > 0:
            skew = float(np.mean(((returns - mean_pct) / std_pct) ** 3))
        else:
            skew = 0.0

        return {
            "mean_pct": round(mean_pct, 2),
            "std_pct": round(std_pct, 2),
            "skew": round(skew, 2),
        }


# ---------------------------------------------------------------------------
# Convenience: simulate a single ticker from agent_summary data
# ---------------------------------------------------------------------------

def simulate_ticker(ticker: str, agent_summary: dict,
                    n_paths: int = DEFAULT_N_PATHS,
                    horizons: list[int] | None = None,
                    seed: int | None = None) -> dict | None:
    """Simulate price distribution for a ticker using agent_summary data.

    Extracts price, ATR volatility, and directional probability from the
    signal data, then runs GBM simulation at each horizon.

    Args:
        ticker: Instrument ticker (e.g., "BTC-USD").
        agent_summary: Full agent_summary dict (or compact version).
        n_paths: Number of MC paths per horizon.
        horizons: List of horizon days (default: [1, 3]).
        seed: Random seed.

    Returns:
        Dict with price bands, stop probability, expected return per horizon.
        None if ticker not found in summary.
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    signals = agent_summary.get("signals", {})
    ticker_data = signals.get(ticker)
    if not ticker_data:
        return None

    price = ticker_data.get("price_usd", 0)
    if price <= 0:
        return None

    extra = ticker_data.get("extra", {})
    atr_pct = extra.get("atr_pct") or ticker_data.get("atr_pct", 2.0)

    # Get volatility from ATR
    vol = volatility_from_atr(atr_pct)

    # Get directional probability for drift
    # Try to use existing probability computation if available
    p_up = _get_directional_probability(ticker, ticker_data, agent_summary)
    drift = drift_from_probability(p_up, vol)

    # Compute ATR-based stop level (2x ATR below entry/current)
    stop_price = price * (1 - 2 * atr_pct / 100)

    result = {}

    for h in horizons:
        h_key = f"{h}d" if h >= 1 else f"{int(h * 24)}h"

        mc = MonteCarloEngine(
            price=price,
            volatility=vol,
            drift=drift,
            horizon_days=h,
            n_paths=n_paths,
            seed=seed,
        )
        mc.simulate_paths()

        result[f"price_bands_{h_key}"] = mc.price_quantiles()
        result[f"p_stop_hit_{h_key}"] = round(mc.probability_below(stop_price), 3)
        result[f"expected_return_{h_key}"] = mc.expected_return()

    # Add metadata
    result["price_usd"] = price
    result["atr_pct"] = round(atr_pct, 2)
    result["volatility_annual"] = round(vol, 3)
    result["p_up"] = round(p_up, 3)
    result["drift_annual"] = round(drift, 4)
    result["stop_price"] = round(stop_price, 2)

    return result


def _get_directional_probability(ticker: str, ticker_data: dict,
                                  agent_summary: dict) -> float:
    """Extract directional probability from available data.

    Priority:
    1. Pre-computed focus_probabilities (from ticker_accuracy.py)
    2. Weighted confidence + action from signal engine
    3. Fallback: 0.5 (no edge)
    """
    # 1. Check focus_probabilities in summary
    focus_probs = agent_summary.get("focus_probabilities", {})
    ticker_probs = focus_probs.get(ticker, {})
    prob_1d = ticker_probs.get("1d", {})
    if prob_1d and "probability" in prob_1d:
        return prob_1d["probability"]

    # 2. Derive from weighted confidence + action
    extra = ticker_data.get("extra", {})
    action = extra.get("_weighted_action") or ticker_data.get("action", "HOLD")
    conf = extra.get("_weighted_confidence") or ticker_data.get("weighted_confidence", 0.5)

    if action == "BUY":
        return 0.5 + conf * 0.3  # Scale to 0.5-0.8 range
    elif action == "SELL":
        return 0.5 - conf * 0.3  # Scale to 0.2-0.5 range
    else:
        return 0.5  # HOLD = no directional edge


# ---------------------------------------------------------------------------
# Batch simulation for all interesting tickers
# ---------------------------------------------------------------------------

def simulate_all(agent_summary: dict, tickers: list[str] | None = None,
                 n_paths: int = DEFAULT_N_PATHS, seed: int = 42) -> dict:
    """Run MC simulation for multiple tickers.

    Args:
        agent_summary: Full agent summary dict.
        tickers: List of tickers to simulate. If None, uses held positions + focus tickers.
        n_paths: Paths per ticker per horizon.
        seed: Base seed (incremented per ticker for independence).

    Returns:
        Dict mapping ticker → simulation results.
    """
    if tickers is None:
        tickers = _interesting_tickers(agent_summary)

    results = {}
    for i, ticker in enumerate(tickers):
        try:
            result = simulate_ticker(ticker, agent_summary,
                                     n_paths=n_paths, seed=seed + i)
            if result:
                results[ticker] = result
        except Exception:
            logger.warning("MC simulation failed for %s", ticker, exc_info=True)

    return results


def _interesting_tickers(agent_summary: dict) -> list[str]:
    """Determine which tickers to simulate (held + focus + signaling)."""
    tickers = set()

    # Focus tickers (from config, surfaced in summary)
    for ft in agent_summary.get("focus_tickers", ["XAG-USD", "BTC-USD"]):
        tickers.add(ft)

    # Tickers with active consensus (BUY or SELL)
    for ticker, data in agent_summary.get("signals", {}).items():
        action = data.get("action", "HOLD")
        if action in ("BUY", "SELL"):
            tickers.add(ticker)

    return sorted(tickers)
