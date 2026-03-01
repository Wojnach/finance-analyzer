"""Portfolio VaR with t-copula correlated simulation.

Computes Value-at-Risk (VaR), Conditional VaR (Expected Shortfall), and
joint drawdown probabilities for multi-position portfolios using a Student-t
copula for tail dependence.

The t-copula captures the empirical fact that assets crash together more
often than a Gaussian copula predicts (tail dependence lambda ~0.18 at
df=4 vs. 0 for Gaussian).

Usage:
    from portfolio.monte_carlo_risk import compute_portfolio_var
    result = compute_portfolio_var(portfolio_state, agent_summary)
    # result = {
    #   "var_95_usd": -1234.56,
    #   "cvar_95_usd": -1567.89,
    #   "var_99_usd": -2345.67,
    #   "total_exposure_usd": 50000.0,
    #   ...
    # }
"""

import logging
import math

import numpy as np
from scipy.stats import t as t_dist

from portfolio.monte_carlo import (
    MIN_VOLATILITY,
    drift_from_probability,
    volatility_from_atr,
)

logger = logging.getLogger("portfolio.monte_carlo_risk")

DEFAULT_DF = 4        # Degrees of freedom for t-copula (4 = moderate fat tails)
DEFAULT_N_PATHS = 10_000


# ---------------------------------------------------------------------------
# Correlation matrix estimation
# ---------------------------------------------------------------------------

def estimate_correlation_matrix(
    returns: dict[str, np.ndarray],
    tickers: list[str],
) -> np.ndarray:
    """Estimate correlation matrix from historical return series.

    For tickers with insufficient data, defaults to zero correlation
    (independent assumption) which is conservative for VaR.

    Args:
        returns: Dict mapping ticker → array of log-returns.
        tickers: Ordered list of tickers (defines matrix row/column order).

    Returns:
        Correlation matrix, shape (n, n), guaranteed symmetric and PSD.
    """
    n = len(tickers)
    if n == 0:
        return np.array([]).reshape(0, 0)

    corr = np.eye(n)

    # Build return matrix for tickers with data
    for i in range(n):
        for j in range(i + 1, n):
            ri = returns.get(tickers[i])
            rj = returns.get(tickers[j])
            if ri is not None and rj is not None:
                # Align lengths (use shorter)
                min_len = min(len(ri), len(rj))
                if min_len >= 20:  # Need at least 20 observations
                    c = np.corrcoef(ri[:min_len], rj[:min_len])[0, 1]
                    if np.isfinite(c):
                        corr[i, j] = c
                        corr[j, i] = c

    # Ensure PSD via eigenvalue clipping (Higham nearest PSD)
    corr = _nearest_psd(corr)
    return corr


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Project matrix to nearest positive semi-definite correlation matrix.

    Uses eigenvalue clipping: set negative eigenvalues to a small positive
    value, then rescale diagonal to 1.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Clip negative eigenvalues
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    # Reconstruct
    result = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    # Rescale to correlation matrix (diagonal = 1)
    d = np.sqrt(np.diag(result))
    d[d == 0] = 1.0
    result = result / np.outer(d, d)
    # Enforce exact symmetry
    result = (result + result.T) / 2
    np.fill_diagonal(result, 1.0)
    return result


# ---------------------------------------------------------------------------
# Known correlation pairs (fallback when no historical data)
# ---------------------------------------------------------------------------

# From risk_management.py — approximate correlation strengths
CORRELATION_PRIORS = {
    ("BTC-USD", "ETH-USD"): 0.75,
    ("XAG-USD", "XAU-USD"): 0.85,
    ("NVDA", "AMD"): 0.70,
    ("NVDA", "AVGO"): 0.65,
    ("NVDA", "TSM"): 0.60,
    ("AMD", "AVGO"): 0.60,
    ("AMD", "TSM"): 0.55,
    ("AVGO", "TSM"): 0.55,
    ("GOOGL", "META"): 0.65,
    ("GOOGL", "AMZN"): 0.60,
    ("META", "AMZN"): 0.60,
    ("AAPL", "GOOGL"): 0.55,
    ("AAPL", "META"): 0.50,
    ("AAPL", "AMZN"): 0.55,
}


def _get_prior_correlation(ticker_a: str, ticker_b: str) -> float:
    """Look up prior correlation for a pair (order-independent)."""
    return CORRELATION_PRIORS.get(
        (ticker_a, ticker_b),
        CORRELATION_PRIORS.get((ticker_b, ticker_a), 0.0),
    )


def build_correlation_matrix(
    tickers: list[str],
    historical_returns: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Build correlation matrix using historical data with prior fallback.

    If historical returns are available and sufficient, uses empirical
    estimates. Otherwise falls back to hardcoded priors from CORRELATED_PAIRS.

    Args:
        tickers: Ordered list of tickers.
        historical_returns: Optional dict of ticker → log-return arrays.

    Returns:
        Correlation matrix (n x n).
    """
    n = len(tickers)
    if n == 0:
        return np.array([]).reshape(0, 0)

    if historical_returns:
        # Check if we have sufficient data (>= 30 observations per ticker)
        has_data = sum(
            1 for t in tickers
            if t in historical_returns and len(historical_returns[t]) >= 30
        )
        if has_data >= 2:
            return estimate_correlation_matrix(historical_returns, tickers)

    # Fallback: use priors
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = _get_prior_correlation(tickers[i], tickers[j])
            corr[i, j] = c
            corr[j, i] = c

    return _nearest_psd(corr)


# ---------------------------------------------------------------------------
# Portfolio risk simulator
# ---------------------------------------------------------------------------

class PortfolioRiskSimulator:
    """Multi-position portfolio risk simulator using Student-t copula.

    Generates correlated terminal returns for all positions jointly,
    then computes portfolio-level P&L, VaR, and CVaR.

    The t-copula with df=4 captures tail dependence (assets crash together)
    that a Gaussian copula misses entirely.

    Args:
        positions: Dict mapping ticker → {shares, price_usd, volatility, drift}.
        correlation_matrix: Correlation matrix for position tickers.
        horizon_days: Simulation horizon.
        n_paths: Number of simulation paths.
        df: Degrees of freedom for t-copula (lower = fatter tails).
        seed: Random seed.
    """

    def __init__(
        self,
        positions: dict,
        correlation_matrix: np.ndarray,
        horizon_days: float = 1.0,
        n_paths: int = DEFAULT_N_PATHS,
        df: int = DEFAULT_DF,
        seed: int | None = None,
    ):
        # Filter out zero-share positions
        self._tickers = [t for t, p in positions.items() if p.get("shares", 0) != 0]
        self._positions = {t: positions[t] for t in self._tickers}
        self._n_assets = len(self._tickers)

        # If we filtered tickers, extract the sub-matrix
        if self._n_assets > 0 and correlation_matrix.size > 0:
            all_tickers = list(positions.keys())
            indices = [all_tickers.index(t) for t in self._tickers]
            self._corr = correlation_matrix[np.ix_(indices, indices)]
        else:
            self._corr = np.array([]).reshape(0, 0)

        self.horizon_days = horizon_days
        self.n_paths = n_paths
        self.df = df
        self.seed = seed

        self._returns = None  # (n_paths, n_assets) log-returns
        self._pnl = None      # (n_paths,) portfolio P&L in USD

    def simulate_correlated_returns(self) -> np.ndarray:
        """Generate correlated returns using t-copula.

        Algorithm:
        1. Cholesky decompose correlation matrix: L = chol(R)
        2. Generate independent standard normals: Z ~ N(0, I)
        3. Correlate: W = Z @ L^T
        4. Generate chi-squared: S ~ chi2(df)
        5. Scale to t: T = W * sqrt(df / S) → multivariate t
        6. Transform marginals: apply per-asset drift and volatility

        Returns:
            Array of shape (n_paths, n_assets) with log-returns.
        """
        if self._n_assets == 0:
            self._returns = np.empty((self.n_paths, 0))
            return self._returns

        rng = np.random.default_rng(self.seed)
        T = self.horizon_days / 252.0

        # Step 1: Cholesky decomposition
        try:
            L = np.linalg.cholesky(self._corr)
        except np.linalg.LinAlgError:
            # Fall back to nearest PSD if not PD
            L = np.linalg.cholesky(_nearest_psd(self._corr))

        # Step 2: Independent standard normals
        Z = rng.standard_normal((self.n_paths, self._n_assets))

        # Step 3: Correlate
        W = Z @ L.T

        # Step 4: Chi-squared scaling for t-distribution
        S = rng.chisquare(self.df, size=self.n_paths)

        # Step 5: Scale to multivariate t
        # T_i = W_i * sqrt(df / S) for each path
        scale = np.sqrt(self.df / S)[:, np.newaxis]
        T_samples = W * scale

        # Step 6: Transform to uniform via t CDF, then to target marginals
        # U = F_t(T; df) → uniform on [0,1]
        U = t_dist.cdf(T_samples, df=self.df)

        # Transform each marginal to GBM log-return
        returns = np.empty_like(U)
        for i, ticker in enumerate(self._tickers):
            pos = self._positions[ticker]
            sigma = max(pos["volatility"], MIN_VOLATILITY)
            mu = pos.get("drift", 0.0)

            # Inverse normal CDF to get standard normal quantiles
            Z_marginal = t_dist.ppf(U[:, i], df=self.df)

            # GBM log-return: (mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z
            # But Z here is t-distributed, capturing fat tails
            drift_term = (mu - 0.5 * sigma**2) * T
            vol_term = sigma * math.sqrt(T)
            returns[:, i] = drift_term + vol_term * Z_marginal

        self._returns = returns
        return returns

    def _ensure_simulated(self):
        """Run simulation if not already done."""
        if self._returns is None:
            self.simulate_correlated_returns()

    def portfolio_pnl(self) -> np.ndarray:
        """Compute portfolio P&L distribution in USD.

        P&L = sum over assets of (shares * price * (exp(log_return) - 1))

        Returns:
            1D array of portfolio P&L values, shape (n_paths,).
        """
        self._ensure_simulated()

        if self._n_assets == 0:
            self._pnl = np.zeros(self.n_paths)
            return self._pnl

        pnl = np.zeros(self.n_paths)
        for i, ticker in enumerate(self._tickers):
            pos = self._positions[ticker]
            shares = pos["shares"]
            price = pos["price_usd"]
            # P&L per path = shares * price * (exp(return) - 1)
            pnl += shares * price * (np.exp(self._returns[:, i]) - 1)

        self._pnl = pnl
        return pnl

    def _ensure_pnl(self):
        """Compute P&L if not already done."""
        if self._pnl is None:
            self.portfolio_pnl()

    def var(self, confidence: float = 0.95) -> float:
        """Compute Value-at-Risk at given confidence level.

        VaR is the loss at the (1-confidence) percentile of the P&L distribution.
        Returns a negative number (loss).

        Args:
            confidence: Confidence level (0.95 or 0.99 typically).

        Returns:
            VaR in USD (negative = loss).
        """
        self._ensure_pnl()
        if self._n_assets == 0:
            return 0.0
        percentile = (1 - confidence) * 100
        return float(np.percentile(self._pnl, percentile))

    def cvar(self, confidence: float = 0.95) -> float:
        """Compute Conditional VaR (Expected Shortfall) at given confidence.

        CVaR is the average loss in the worst (1-confidence) scenarios.
        Always worse than VaR (further from zero).

        Args:
            confidence: Confidence level.

        Returns:
            CVaR in USD (negative = loss).
        """
        self._ensure_pnl()
        if self._n_assets == 0:
            return 0.0
        var_threshold = self.var(confidence)
        tail = self._pnl[self._pnl <= var_threshold]
        if len(tail) == 0:
            return var_threshold
        return float(np.mean(tail))

    def drawdown_probability(self, threshold_pct: float) -> float:
        """Probability of portfolio losing more than threshold_pct.

        Args:
            threshold_pct: Loss threshold as percentage (e.g., 5.0 = 5%).

        Returns:
            Probability (0.0-1.0) of exceeding the loss threshold.
        """
        self._ensure_pnl()
        if self._n_assets == 0:
            return 0.0

        total_value = sum(
            p["shares"] * p["price_usd"] for p in self._positions.values()
        )
        if total_value <= 0:
            return 0.0

        loss_threshold = -total_value * threshold_pct / 100.0
        return float(np.mean(self._pnl < loss_threshold))


# ---------------------------------------------------------------------------
# Convenience: compute portfolio VaR from system data structures
# ---------------------------------------------------------------------------

def compute_portfolio_var(
    portfolio_state: dict,
    agent_summary: dict,
    n_paths: int = DEFAULT_N_PATHS,
    historical_returns: dict[str, np.ndarray] | None = None,
    seed: int = 42,
) -> dict:
    """Compute portfolio VaR from portfolio state and agent summary.

    Extracts held positions, current prices, volatilities, and directional
    probabilities from the system data, then runs t-copula simulation.

    Args:
        portfolio_state: Portfolio state dict (holdings, cash_sek, etc.).
        agent_summary: Agent summary with signals, fx_rate.
        n_paths: Number of MC paths.
        historical_returns: Optional empirical return series for correlation.
        seed: Random seed.

    Returns:
        Dict with VaR metrics in both USD and SEK.
    """
    holdings = portfolio_state.get("holdings", {})
    signals = agent_summary.get("signals", {})
    fx_rate = agent_summary.get("fx_rate", 10.0)

    # Build positions dict
    positions = {}
    tickers = []

    for ticker, holding in holdings.items():
        shares = holding.get("shares", 0)
        if shares <= 0:
            continue

        ticker_data = signals.get(ticker, {})
        price = ticker_data.get("price_usd", 0)
        if price <= 0:
            continue

        extra = ticker_data.get("extra", {})
        atr_pct = extra.get("atr_pct") or ticker_data.get("atr_pct", 2.0)
        vol = volatility_from_atr(atr_pct)

        # Get directional probability for drift
        from portfolio.monte_carlo import _get_directional_probability
        p_up = _get_directional_probability(ticker, ticker_data, agent_summary)
        drift = drift_from_probability(p_up, vol)

        tickers.append(ticker)
        positions[ticker] = {
            "shares": shares,
            "price_usd": price,
            "volatility": vol,
            "drift": drift,
        }

    if not tickers:
        return {
            "var_95_usd": 0.0,
            "var_99_usd": 0.0,
            "cvar_95_usd": 0.0,
            "cvar_99_usd": 0.0,
            "var_95_sek": 0.0,
            "var_99_sek": 0.0,
            "cvar_95_sek": 0.0,
            "total_exposure_usd": 0.0,
            "total_exposure_sek": 0.0,
            "n_positions": 0,
            "drawdown_1pct_prob": 0.0,
            "drawdown_5pct_prob": 0.0,
        }

    # Build correlation matrix
    corr = build_correlation_matrix(tickers, historical_returns)

    # Run simulation
    sim = PortfolioRiskSimulator(
        positions=positions,
        correlation_matrix=corr,
        horizon_days=1,
        n_paths=n_paths,
        df=DEFAULT_DF,
        seed=seed,
    )
    sim.simulate_correlated_returns()

    total_exposure = sum(p["shares"] * p["price_usd"] for p in positions.values())

    var95 = sim.var(0.95)
    var99 = sim.var(0.99)
    cvar95 = sim.cvar(0.95)
    cvar99 = sim.cvar(0.99)

    return {
        "var_95_usd": round(var95, 2),
        "var_99_usd": round(var99, 2),
        "cvar_95_usd": round(cvar95, 2),
        "cvar_99_usd": round(cvar99, 2),
        "var_95_sek": round(var95 * fx_rate, 2),
        "var_99_sek": round(var99 * fx_rate, 2),
        "cvar_95_sek": round(cvar95 * fx_rate, 2),
        "total_exposure_usd": round(total_exposure, 2),
        "total_exposure_sek": round(total_exposure * fx_rate, 2),
        "n_positions": len(tickers),
        "drawdown_1pct_prob": round(sim.drawdown_probability(1.0), 3),
        "drawdown_5pct_prob": round(sim.drawdown_probability(5.0), 3),
    }
