"""Tests for Portfolio VaR with t-copula correlated simulation."""

import numpy as np
import pytest

from portfolio.monte_carlo_risk import (
    PortfolioRiskSimulator,
    compute_portfolio_var,
    estimate_correlation_matrix,
)


# ---------------------------------------------------------------------------
# Correlation matrix estimation
# ---------------------------------------------------------------------------

class TestCorrelationMatrix:
    """Test empirical correlation estimation."""

    def test_identity_for_independent_series(self):
        """Independent random series → near-identity correlation."""
        rng = np.random.default_rng(42)
        returns = {
            "A": rng.standard_normal(500),
            "B": rng.standard_normal(500),
            "C": rng.standard_normal(500),
        }
        tickers = ["A", "B", "C"]
        corr = estimate_correlation_matrix(returns, tickers)
        assert corr.shape == (3, 3)
        # Diagonal = 1
        np.testing.assert_array_almost_equal(np.diag(corr), [1, 1, 1])
        # Off-diagonal near zero (random, not exact)
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert abs(corr[i, j]) < 0.15

    def test_perfectly_correlated(self):
        """Identical series → correlation = 1."""
        series = np.random.default_rng(42).standard_normal(500)
        returns = {"A": series, "B": series.copy()}
        corr = estimate_correlation_matrix(returns, ["A", "B"])
        np.testing.assert_almost_equal(corr[0, 1], 1.0, decimal=5)

    def test_negatively_correlated(self):
        """Negated series → correlation = -1."""
        series = np.random.default_rng(42).standard_normal(500)
        returns = {"A": series, "B": -series}
        corr = estimate_correlation_matrix(returns, ["A", "B"])
        np.testing.assert_almost_equal(corr[0, 1], -1.0, decimal=5)

    def test_symmetric_and_positive_semidefinite(self):
        """Correlation matrix must be symmetric and PSD."""
        rng = np.random.default_rng(42)
        returns = {t: rng.standard_normal(200) for t in ["A", "B", "C", "D"]}
        corr = estimate_correlation_matrix(returns, ["A", "B", "C", "D"])
        # Symmetric
        np.testing.assert_array_almost_equal(corr, corr.T)
        # PSD: all eigenvalues >= 0
        eigenvalues = np.linalg.eigvalsh(corr)
        assert all(ev >= -1e-10 for ev in eigenvalues)

    def test_fallback_for_missing_ticker(self):
        """Ticker with no return history → row/col defaults to zero correlation."""
        returns = {"A": np.random.default_rng(42).standard_normal(100)}
        corr = estimate_correlation_matrix(returns, ["A", "B"])
        assert corr.shape == (2, 2)
        assert corr[0, 0] == 1.0
        assert corr[1, 1] == 1.0
        assert corr[0, 1] == 0.0  # No data for B → independent


# ---------------------------------------------------------------------------
# t-Copula correlated return generation
# ---------------------------------------------------------------------------

class TestTCopulaReturns:
    """Test that t-copula generates properly correlated fat-tailed returns."""

    def test_output_shape(self):
        """Correlated returns have correct shape."""
        positions = {
            "A": {"shares": 10, "price_usd": 100, "volatility": 0.20, "drift": 0.0},
            "B": {"shares": 5, "price_usd": 200, "volatility": 0.30, "drift": 0.0},
        }
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        sim = PortfolioRiskSimulator(
            positions=positions,
            correlation_matrix=corr,
            horizon_days=1,
            n_paths=1000,
            df=4,
            seed=42,
        )
        returns = sim.simulate_correlated_returns()
        assert returns.shape == (1000, 2)

    def test_correlation_preserved(self):
        """Simulated returns maintain approximate input correlation."""
        corr_input = np.array([[1.0, 0.8], [0.8, 1.0]])
        positions = {
            "A": {"shares": 10, "price_usd": 100, "volatility": 0.20, "drift": 0.0},
            "B": {"shares": 5, "price_usd": 200, "volatility": 0.20, "drift": 0.0},
        }
        sim = PortfolioRiskSimulator(
            positions=positions,
            correlation_matrix=corr_input,
            horizon_days=1,
            n_paths=50000,
            df=4,
            seed=42,
        )
        returns = sim.simulate_correlated_returns()
        empirical_corr = np.corrcoef(returns[:, 0], returns[:, 1])[0, 1]
        # t-copula correlation differs slightly from linear correlation
        # but should be close for these parameters
        assert abs(empirical_corr - 0.8) < 0.05

    def test_fat_tails_vs_gaussian(self):
        """t-copula (df=4) should produce more extreme events than Gaussian."""
        positions = {
            "A": {"shares": 1, "price_usd": 100, "volatility": 0.20, "drift": 0.0},
        }
        corr = np.array([[1.0]])

        # t-copula with df=4 (fat tails)
        sim_t = PortfolioRiskSimulator(
            positions=positions, correlation_matrix=corr,
            horizon_days=1, n_paths=50000, df=4, seed=42,
        )
        returns_t = sim_t.simulate_correlated_returns()

        # High df → approaches Gaussian
        sim_g = PortfolioRiskSimulator(
            positions=positions, correlation_matrix=corr,
            horizon_days=1, n_paths=50000, df=100, seed=42,
        )
        returns_g = sim_g.simulate_correlated_returns()

        # t(4) should have heavier tails → higher kurtosis
        kurtosis_t = float(np.mean(((returns_t - np.mean(returns_t)) / np.std(returns_t)) ** 4))
        kurtosis_g = float(np.mean(((returns_g - np.mean(returns_g)) / np.std(returns_g)) ** 4))
        assert kurtosis_t > kurtosis_g

    def test_reproducible_with_seed(self):
        """Same seed → identical results."""
        positions = {
            "A": {"shares": 10, "price_usd": 100, "volatility": 0.20, "drift": 0.0},
            "B": {"shares": 5, "price_usd": 200, "volatility": 0.30, "drift": 0.0},
        }
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        kwargs = dict(positions=positions, correlation_matrix=corr,
                      horizon_days=1, n_paths=100, df=4, seed=123)
        r1 = PortfolioRiskSimulator(**kwargs).simulate_correlated_returns()
        r2 = PortfolioRiskSimulator(**kwargs).simulate_correlated_returns()
        np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# Portfolio P&L distribution
# ---------------------------------------------------------------------------

class TestPortfolioPnL:
    """Test portfolio-level P&L aggregation."""

    def test_pnl_shape(self):
        """Portfolio P&L is a 1D array of n_paths."""
        positions = {
            "A": {"shares": 10, "price_usd": 100, "volatility": 0.20, "drift": 0.0},
        }
        corr = np.array([[1.0]])
        sim = PortfolioRiskSimulator(
            positions=positions, correlation_matrix=corr,
            horizon_days=1, n_paths=500, df=4, seed=42,
        )
        sim.simulate_correlated_returns()
        pnl = sim.portfolio_pnl()
        assert pnl.shape == (500,)

    def test_pnl_mean_near_zero_for_zero_drift(self):
        """Zero drift → E[P&L] near zero."""
        positions = {
            "A": {"shares": 100, "price_usd": 100, "volatility": 0.20, "drift": 0.0},
        }
        corr = np.array([[1.0]])
        sim = PortfolioRiskSimulator(
            positions=positions, correlation_matrix=corr,
            horizon_days=1, n_paths=50000, df=4, seed=42,
        )
        sim.simulate_correlated_returns()
        pnl = sim.portfolio_pnl()
        # Mean P&L should be near $0 for 1 day with zero drift
        assert abs(np.mean(pnl)) < 50  # $50 tolerance on $10K position

    def test_pnl_scales_with_position_size(self):
        """Larger position → wider P&L distribution."""
        corr = np.array([[1.0]])
        kwargs = dict(correlation_matrix=corr, horizon_days=1,
                      n_paths=10000, df=4, seed=42)

        small = PortfolioRiskSimulator(
            positions={"A": {"shares": 1, "price_usd": 100, "volatility": 0.20, "drift": 0.0}},
            **kwargs
        )
        large = PortfolioRiskSimulator(
            positions={"A": {"shares": 100, "price_usd": 100, "volatility": 0.20, "drift": 0.0}},
            **kwargs
        )
        small.simulate_correlated_returns()
        large.simulate_correlated_returns()
        assert np.std(large.portfolio_pnl()) > np.std(small.portfolio_pnl()) * 50

    def test_multi_position_pnl(self):
        """P&L aggregates across multiple positions."""
        positions = {
            "A": {"shares": 10, "price_usd": 100, "volatility": 0.20, "drift": 0.0},
            "B": {"shares": 5, "price_usd": 200, "volatility": 0.30, "drift": 0.0},
        }
        corr = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = PortfolioRiskSimulator(
            positions=positions, correlation_matrix=corr,
            horizon_days=1, n_paths=1000, df=4, seed=42,
        )
        sim.simulate_correlated_returns()
        pnl = sim.portfolio_pnl()
        assert pnl.shape == (1000,)
        # Total exposure is $1000 + $1000 = $2000
        # P&L should be in reasonable range
        assert np.max(np.abs(pnl)) < 2000  # Can't lose more than total in 1 day


# ---------------------------------------------------------------------------
# VaR and CVaR
# ---------------------------------------------------------------------------

class TestVaRCVaR:
    """Test Value-at-Risk and Conditional VaR computations."""

    @pytest.fixture()
    def sim(self):
        """Create a simulator with known properties."""
        positions = {
            "A": {"shares": 100, "price_usd": 100, "volatility": 0.20, "drift": 0.0},
        }
        corr = np.array([[1.0]])
        s = PortfolioRiskSimulator(
            positions=positions, correlation_matrix=corr,
            horizon_days=1, n_paths=50000, df=4, seed=42,
        )
        s.simulate_correlated_returns()
        return s

    def test_var_is_negative(self, sim):
        """VaR (loss) should be a negative number (loss)."""
        var95 = sim.var(0.95)
        assert var95 < 0

    def test_var_99_worse_than_95(self, sim):
        """99% VaR should be a larger loss than 95% VaR."""
        var95 = sim.var(0.95)
        var99 = sim.var(0.99)
        assert var99 < var95  # Both negative; 99% is more negative

    def test_cvar_worse_than_var(self, sim):
        """CVaR (expected shortfall) should be worse than VaR."""
        var95 = sim.var(0.95)
        cvar95 = sim.cvar(0.95)
        assert cvar95 < var95  # CVaR is the average of losses beyond VaR

    def test_cvar_is_negative(self, sim):
        """CVaR should be negative (a loss measure)."""
        assert sim.cvar(0.95) < 0
        assert sim.cvar(0.99) < 0

    def test_var_scales_with_volatility(self):
        """Higher volatility → larger VaR."""
        corr = np.array([[1.0]])
        kwargs = dict(correlation_matrix=corr, horizon_days=1,
                      n_paths=10000, df=4, seed=42)

        low_vol = PortfolioRiskSimulator(
            positions={"A": {"shares": 100, "price_usd": 100, "volatility": 0.10, "drift": 0.0}},
            **kwargs
        )
        high_vol = PortfolioRiskSimulator(
            positions={"A": {"shares": 100, "price_usd": 100, "volatility": 0.40, "drift": 0.0}},
            **kwargs
        )
        low_vol.simulate_correlated_returns()
        high_vol.simulate_correlated_returns()
        assert high_vol.var(0.95) < low_vol.var(0.95)  # More negative


# ---------------------------------------------------------------------------
# Correlated crash probability
# ---------------------------------------------------------------------------

class TestCorrelatedCrash:
    """Correlated positions should show higher joint drawdown risk."""

    def test_correlated_crash_worse_than_independent(self):
        """Highly correlated positions → worse portfolio VaR than independent."""
        positions = {
            "A": {"shares": 50, "price_usd": 100, "volatility": 0.25, "drift": 0.0},
            "B": {"shares": 50, "price_usd": 100, "volatility": 0.25, "drift": 0.0},
        }
        kwargs = dict(horizon_days=1, n_paths=20000, df=4, seed=42)

        # Independent positions
        corr_indep = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim_indep = PortfolioRiskSimulator(
            positions=positions, correlation_matrix=corr_indep, **kwargs
        )
        sim_indep.simulate_correlated_returns()

        # Highly correlated positions (like NVDA + AMD)
        corr_high = np.array([[1.0, 0.85], [0.85, 1.0]])
        sim_corr = PortfolioRiskSimulator(
            positions=positions, correlation_matrix=corr_high, **kwargs
        )
        sim_corr.simulate_correlated_returns()

        # Correlated portfolio has worse (more negative) VaR
        assert sim_corr.var(0.95) < sim_indep.var(0.95)

    def test_drawdown_probability(self):
        """Test drawdown probability computation."""
        positions = {
            "A": {"shares": 100, "price_usd": 100, "volatility": 0.20, "drift": 0.0},
        }
        corr = np.array([[1.0]])
        sim = PortfolioRiskSimulator(
            positions=positions, correlation_matrix=corr,
            horizon_days=1, n_paths=50000, df=4, seed=42,
        )
        sim.simulate_correlated_returns()

        # Probability of losing >1% of $10K position in 1 day
        p_1pct = sim.drawdown_probability(1.0)
        assert 0.0 < p_1pct < 1.0

        # Probability of losing >10% should be much smaller
        p_10pct = sim.drawdown_probability(10.0)
        assert p_10pct < p_1pct


# ---------------------------------------------------------------------------
# Single position (degenerate case)
# ---------------------------------------------------------------------------

class TestSinglePosition:
    """VaR for a single position should match individual MC."""

    def test_single_position_var(self):
        """Single-position portfolio VaR should be consistent."""
        from portfolio.monte_carlo import MonteCarloEngine

        positions = {
            "A": {"shares": 100, "price_usd": 100, "volatility": 0.20, "drift": 0.0},
        }
        corr = np.array([[1.0]])
        sim = PortfolioRiskSimulator(
            positions=positions, correlation_matrix=corr,
            horizon_days=1, n_paths=50000, df=100,  # High df → near-Gaussian
            seed=42,
        )
        sim.simulate_correlated_returns()
        var95 = sim.var(0.95)

        # Compare with MonteCarloEngine
        mc = MonteCarloEngine(price=100.0, volatility=0.20, drift=0.0,
                              horizon_days=1, n_paths=50000, seed=42)
        mc.simulate_paths()
        # VaR = 5th percentile of P&L for 100 shares
        mc_pnl = (mc._terminal_prices - 100.0) * 100
        mc_var = float(np.percentile(mc_pnl, 5))

        # Should be in the same ballpark (not exact due to different RNG paths)
        assert abs(var95 - mc_var) < abs(mc_var) * 0.3  # Within 30%


# ---------------------------------------------------------------------------
# Empty / no positions
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: no positions, single position, zero shares."""

    def test_no_positions_zero_var(self):
        """Empty portfolio → VaR = 0."""
        sim = PortfolioRiskSimulator(
            positions={},
            correlation_matrix=np.array([]).reshape(0, 0),
            horizon_days=1,
            n_paths=100,
            df=4,
            seed=42,
        )
        sim.simulate_correlated_returns()
        assert sim.var(0.95) == 0.0
        assert sim.cvar(0.95) == 0.0

    def test_zero_shares_ignored(self):
        """Position with zero shares should not affect VaR."""
        positions = {
            "A": {"shares": 0, "price_usd": 100, "volatility": 0.20, "drift": 0.0},
            "B": {"shares": 10, "price_usd": 100, "volatility": 0.20, "drift": 0.0},
        }
        corr = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = PortfolioRiskSimulator(
            positions=positions, correlation_matrix=corr,
            horizon_days=1, n_paths=1000, df=4, seed=42,
        )
        sim.simulate_correlated_returns()
        var95 = sim.var(0.95)
        assert var95 < 0  # Still has risk from B


# ---------------------------------------------------------------------------
# Convenience function: compute_portfolio_var
# ---------------------------------------------------------------------------

class TestComputePortfolioVar:
    """Test the high-level convenience function."""

    def test_returns_dict_with_expected_keys(self):
        portfolio_state = {
            "holdings": {
                "BTC-USD": {"shares": 0.5, "avg_cost_usd": 65000},
            },
            "cash_sek": 400000,
        }
        agent_summary = {
            "signals": {
                "BTC-USD": {
                    "price_usd": 67000,
                    "extra": {"atr_pct": 3.5, "_votes": {}},
                    "regime": "ranging",
                }
            },
            "fx_rate": 10.0,
        }
        result = compute_portfolio_var(portfolio_state, agent_summary, n_paths=1000)
        assert "var_95_usd" in result
        assert "var_99_usd" in result
        assert "cvar_95_usd" in result
        assert "var_95_sek" in result
        assert "total_exposure_usd" in result

    def test_empty_portfolio(self):
        portfolio_state = {"holdings": {}, "cash_sek": 500000}
        agent_summary = {"signals": {}, "fx_rate": 10.0}
        result = compute_portfolio_var(portfolio_state, agent_summary, n_paths=100)
        assert result["var_95_usd"] == 0.0
        assert result["total_exposure_usd"] == 0.0

    def test_var_in_sek(self):
        """VaR in SEK should be VaR in USD * fx_rate."""
        portfolio_state = {
            "holdings": {
                "XAG-USD": {"shares": 100, "avg_cost_usd": 30},
            },
            "cash_sek": 400000,
        }
        agent_summary = {
            "signals": {
                "XAG-USD": {
                    "price_usd": 32.5,
                    "extra": {"atr_pct": 2.0, "_votes": {}},
                    "regime": "trending-up",
                }
            },
            "fx_rate": 10.5,
        }
        result = compute_portfolio_var(portfolio_state, agent_summary, n_paths=5000)
        # SEK = USD * fx_rate
        np.testing.assert_almost_equal(
            result["var_95_sek"],
            result["var_95_usd"] * 10.5,
            decimal=0,
        )
