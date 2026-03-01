"""Tests for Monte Carlo price simulation engine."""

import numpy as np
import pytest

# We'll import after creating the module
MC_MODULE = "portfolio.monte_carlo"


class TestGBMPaths:
    """Test Geometric Brownian Motion path generation."""

    def test_path_shape(self):
        from portfolio.monte_carlo import MonteCarloEngine
        mc = MonteCarloEngine(price=100.0, volatility=0.20, drift=0.0,
                              horizon_days=1, n_paths=1000, seed=42)
        paths = mc.simulate_paths()
        # With antithetic: 1000 paths requested → 500 pairs → 1000 terminal prices
        assert paths.shape == (1000,)
        assert all(p > 0 for p in paths)  # Prices always positive (GBM property)

    def test_mean_near_spot_with_zero_drift(self):
        from portfolio.monte_carlo import MonteCarloEngine
        mc = MonteCarloEngine(price=100.0, volatility=0.20, drift=0.0,
                              horizon_days=1, n_paths=10000, seed=42)
        paths = mc.simulate_paths()
        # With zero drift, E[S_T] = S0 (risk-neutral), mean should be near 100
        assert abs(np.mean(paths) - 100.0) < 2.0  # within 2% of spot

    def test_positive_drift_raises_mean(self):
        from portfolio.monte_carlo import MonteCarloEngine
        mc = MonteCarloEngine(price=100.0, volatility=0.20, drift=0.10,
                              horizon_days=30, n_paths=10000, seed=42)
        paths = mc.simulate_paths()
        # Positive drift should push mean above spot
        assert np.mean(paths) > 100.0

    def test_negative_drift_lowers_mean(self):
        from portfolio.monte_carlo import MonteCarloEngine
        mc = MonteCarloEngine(price=100.0, volatility=0.20, drift=-0.10,
                              horizon_days=30, n_paths=10000, seed=42)
        paths = mc.simulate_paths()
        assert np.mean(paths) < 100.0

    def test_higher_vol_wider_distribution(self):
        from portfolio.monte_carlo import MonteCarloEngine
        mc_low = MonteCarloEngine(price=100.0, volatility=0.10, drift=0.0,
                                  horizon_days=30, n_paths=10000, seed=42)
        mc_high = MonteCarloEngine(price=100.0, volatility=0.50, drift=0.0,
                                   horizon_days=30, n_paths=10000, seed=42)
        std_low = np.std(mc_low.simulate_paths())
        std_high = np.std(mc_high.simulate_paths())
        assert std_high > std_low * 2  # 5x vol → much wider distribution

    def test_longer_horizon_wider_distribution(self):
        from portfolio.monte_carlo import MonteCarloEngine
        mc_1d = MonteCarloEngine(price=100.0, volatility=0.20, drift=0.0,
                                 horizon_days=1, n_paths=10000, seed=42)
        mc_30d = MonteCarloEngine(price=100.0, volatility=0.20, drift=0.0,
                                  horizon_days=30, n_paths=10000, seed=42)
        std_1d = np.std(mc_1d.simulate_paths())
        std_30d = np.std(mc_30d.simulate_paths())
        assert std_30d > std_1d

    def test_reproducible_with_seed(self):
        from portfolio.monte_carlo import MonteCarloEngine
        mc1 = MonteCarloEngine(price=100.0, volatility=0.20, drift=0.0,
                               horizon_days=1, n_paths=100, seed=123)
        mc2 = MonteCarloEngine(price=100.0, volatility=0.20, drift=0.0,
                               horizon_days=1, n_paths=100, seed=123)
        np.testing.assert_array_equal(mc1.simulate_paths(), mc2.simulate_paths())

    def test_different_seeds_different_results(self):
        from portfolio.monte_carlo import MonteCarloEngine
        mc1 = MonteCarloEngine(price=100.0, volatility=0.20, drift=0.0,
                               horizon_days=1, n_paths=100, seed=1)
        mc2 = MonteCarloEngine(price=100.0, volatility=0.20, drift=0.0,
                               horizon_days=1, n_paths=100, seed=2)
        assert not np.array_equal(mc1.simulate_paths(), mc2.simulate_paths())


class TestAntitheticVariates:
    """Test that antithetic variates reduce variance."""

    def test_antithetic_reduces_variance(self):
        """Compare variance of antithetic vs crude MC estimator."""
        from portfolio.monte_carlo import MonteCarloEngine

        # Run 50 batches and compare variance of the mean estimator
        means_crude = []
        means_antithetic = []

        for i in range(50):
            # Crude MC (no antithetic)
            rng = np.random.default_rng(seed=i)
            Z = rng.standard_normal(500)
            S0, sigma, T = 100.0, 0.20, 1/252
            crude_paths = S0 * np.exp(-0.5 * sigma**2 * T + sigma * np.sqrt(T) * Z)
            means_crude.append(np.mean(crude_paths))

            # Antithetic MC
            mc = MonteCarloEngine(price=100.0, volatility=0.20, drift=0.0,
                                  horizon_days=1, n_paths=500, seed=i)
            paths = mc.simulate_paths()
            means_antithetic.append(np.mean(paths))

        var_crude = np.var(means_crude)
        var_anti = np.var(means_antithetic)
        # Antithetic should have lower variance (typically 50-75% reduction)
        assert var_anti < var_crude


class TestQuantiles:
    """Test price quantile extraction."""

    def test_quantile_ordering(self):
        from portfolio.monte_carlo import MonteCarloEngine
        mc = MonteCarloEngine(price=100.0, volatility=0.20, drift=0.0,
                              horizon_days=1, n_paths=10000, seed=42)
        mc.simulate_paths()
        q = mc.price_quantiles([5, 25, 50, 75, 95])
        assert q[5] < q[25] < q[50] < q[75] < q[95]

    def test_median_near_spot_zero_drift(self):
        from portfolio.monte_carlo import MonteCarloEngine
        mc = MonteCarloEngine(price=100.0, volatility=0.20, drift=0.0,
                              horizon_days=1, n_paths=50000, seed=42)
        mc.simulate_paths()
        q = mc.price_quantiles([50])
        # Median of lognormal with zero drift is S0*exp(-0.5*sigma^2*T) ≈ 99.97
        assert abs(q[50] - 100.0) < 1.0

    def test_quantiles_all_positive(self):
        from portfolio.monte_carlo import MonteCarloEngine
        mc = MonteCarloEngine(price=50.0, volatility=0.40, drift=-0.05,
                              horizon_days=30, n_paths=10000, seed=42)
        mc.simulate_paths()
        q = mc.price_quantiles([1, 5, 50, 95, 99])
        for pct, price in q.items():
            assert price > 0, f"Quantile {pct}% should be positive, got {price}"


class TestProbabilities:
    """Test probability computations."""

    def test_probability_below_spot_near_50pct(self):
        from portfolio.monte_carlo import MonteCarloEngine
        mc = MonteCarloEngine(price=100.0, volatility=0.20, drift=0.0,
                              horizon_days=1, n_paths=50000, seed=42)
        mc.simulate_paths()
        p = mc.probability_below(100.0)
        # With zero drift, slightly above 50% end below (lognormal skew)
        assert 0.45 < p < 0.55

    def test_probability_above_very_high_near_zero(self):
        from portfolio.monte_carlo import MonteCarloEngine
        mc = MonteCarloEngine(price=100.0, volatility=0.20, drift=0.0,
                              horizon_days=1, n_paths=10000, seed=42)
        mc.simulate_paths()
        p = mc.probability_above(200.0)  # Double the price in 1 day
        assert p < 0.01

    def test_probability_below_zero_is_zero(self):
        from portfolio.monte_carlo import MonteCarloEngine
        mc = MonteCarloEngine(price=100.0, volatility=0.20, drift=0.0,
                              horizon_days=1, n_paths=10000, seed=42)
        mc.simulate_paths()
        assert mc.probability_below(0) == 0.0

    def test_probability_above_zero_is_one(self):
        from portfolio.monte_carlo import MonteCarloEngine
        mc = MonteCarloEngine(price=100.0, volatility=0.20, drift=0.0,
                              horizon_days=1, n_paths=10000, seed=42)
        mc.simulate_paths()
        assert mc.probability_above(0) == 1.0

    def test_stop_loss_probability_increases_with_vol(self):
        from portfolio.monte_carlo import MonteCarloEngine
        mc_low = MonteCarloEngine(price=100.0, volatility=0.10, drift=0.0,
                                  horizon_days=1, n_paths=10000, seed=42)
        mc_high = MonteCarloEngine(price=100.0, volatility=0.50, drift=0.0,
                                   horizon_days=1, n_paths=10000, seed=42)
        mc_low.simulate_paths()
        mc_high.simulate_paths()
        p_low = mc_low.probability_below(90.0)   # 10% stop
        p_high = mc_high.probability_below(90.0)
        assert p_high > p_low


class TestExpectedReturn:
    """Test expected return statistics."""

    def test_expected_return_dict_keys(self):
        from portfolio.monte_carlo import MonteCarloEngine
        mc = MonteCarloEngine(price=100.0, volatility=0.20, drift=0.0,
                              horizon_days=1, n_paths=1000, seed=42)
        mc.simulate_paths()
        ret = mc.expected_return()
        assert "mean_pct" in ret
        assert "std_pct" in ret
        assert "skew" in ret

    def test_expected_return_zero_drift_near_zero(self):
        from portfolio.monte_carlo import MonteCarloEngine
        mc = MonteCarloEngine(price=100.0, volatility=0.20, drift=0.0,
                              horizon_days=1, n_paths=50000, seed=42)
        mc.simulate_paths()
        ret = mc.expected_return()
        assert abs(ret["mean_pct"]) < 0.5  # Near zero for 1 day


class TestDriftFromProbability:
    """Test drift derivation from directional probability."""

    def test_drift_from_50pct_is_near_zero(self):
        from portfolio.monte_carlo import drift_from_probability
        d = drift_from_probability(0.5, 0.20)
        # With p=0.5, N_inv(0.5)=0, so mu = 0.5*sigma^2 = 0.02 (risk-neutral correction)
        assert abs(d) < 0.03

    def test_drift_from_70pct_is_positive(self):
        from portfolio.monte_carlo import drift_from_probability
        d = drift_from_probability(0.7, 0.20)
        assert d > 0

    def test_drift_from_30pct_is_negative(self):
        from portfolio.monte_carlo import drift_from_probability
        d = drift_from_probability(0.3, 0.20)
        assert d < 0


class TestVolatilityFromATR:
    """Test annualized volatility estimation from ATR%."""

    def test_atr_to_annual_vol(self):
        from portfolio.monte_carlo import volatility_from_atr
        # ATR% of 2% (14-period) → annualized
        vol = volatility_from_atr(2.0)
        # sqrt(252/14) ≈ 4.24, so 0.02 * 4.24 ≈ 0.085
        assert 0.05 < vol < 0.15

    def test_higher_atr_higher_vol(self):
        from portfolio.monte_carlo import volatility_from_atr
        vol_low = volatility_from_atr(1.0)
        vol_high = volatility_from_atr(5.0)
        assert vol_high > vol_low

    def test_zero_atr_gives_minimum_vol(self):
        from portfolio.monte_carlo import volatility_from_atr
        vol = volatility_from_atr(0.0)
        assert vol > 0  # Should have a floor, not zero


class TestSimulateTicker:
    """Test the convenience function for simulating a single ticker."""

    def test_simulate_ticker_returns_dict(self):
        from portfolio.monte_carlo import simulate_ticker
        # Mock agent_summary structure
        summary = {
            "signals": {
                "BTC-USD": {
                    "price_usd": 86000.0,
                    "extra": {
                        "atr_pct": 3.5,
                        "_votes": {"rsi": "BUY", "ema": "HOLD", "macd": "SELL"},
                    },
                    "regime": "ranging",
                }
            },
            "fx_rate": 10.5,
        }
        result = simulate_ticker("BTC-USD", summary)
        assert result is not None
        assert "price_bands_1d" in result
        assert "price_bands_3d" in result
        assert "p_stop_hit_1d" in result
        assert "expected_return_1d" in result

    def test_simulate_ticker_unknown_returns_none(self):
        from portfolio.monte_carlo import simulate_ticker
        result = simulate_ticker("FAKE-TICKER", {"signals": {}, "fx_rate": 10.5})
        assert result is None

    def test_bands_ordered(self):
        from portfolio.monte_carlo import simulate_ticker
        summary = {
            "signals": {
                "XAG-USD": {
                    "price_usd": 32.5,
                    "extra": {"atr_pct": 2.0, "_votes": {}},
                    "regime": "trending-up",
                }
            },
            "fx_rate": 10.5,
        }
        result = simulate_ticker("XAG-USD", summary)
        bands = result["price_bands_1d"]
        assert bands[5] < bands[25] < bands[50] < bands[75] < bands[95]
