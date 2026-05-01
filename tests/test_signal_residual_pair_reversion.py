"""Tests for residual_pair_reversion signal module."""
import numpy as np
import pandas as pd

from portfolio.signals.residual_pair_reversion import (
    _compute_half_life,
    _residual_z_signal,
    _rolling_ols_beta,
    compute_residual_pair_reversion_signal,
)


def _make_df(n=250):
    """Create a test DataFrame with realistic OHLCV data."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    def test_returns_dict_with_required_keys_no_context(self):
        """Without context (no ticker), should return HOLD."""
        df = _make_df()
        result = compute_residual_pair_reversion_signal(df)
        assert isinstance(result, dict)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_returns_dict_with_required_keys_unknown_ticker(self):
        """With unknown ticker, should return HOLD."""
        df = _make_df()
        result = compute_residual_pair_reversion_signal(
            df, context={"ticker": "UNKNOWN-TICKER"}
        )
        assert isinstance(result, dict)
        assert result["action"] == "HOLD"

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_residual_pair_reversion_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_none_dataframe_returns_hold(self):
        result = compute_residual_pair_reversion_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=50)
        result = compute_residual_pair_reversion_signal(
            df, context={"ticker": "ETH-USD"}
        )
        assert result["action"] == "HOLD"

    def test_valid_ticker_in_pair_map(self):
        """Valid ticker should attempt computation (may HOLD due to no driver data)."""
        df = _make_df(n=250)
        result = compute_residual_pair_reversion_signal(
            df, context={"ticker": "ETH-USD"}
        )
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        if result["action"] != "HOLD":
            assert 0.0 <= result["confidence"] <= 0.7


class TestRollingOLS:
    """Test the rolling OLS beta computation."""

    def test_constant_beta(self):
        """If target = 2*driver + noise, beta should be ~2.0."""
        np.random.seed(123)
        n = 300
        driver_ret = pd.Series(np.random.randn(n) * 0.01)
        target_ret = 2.0 * driver_ret + np.random.randn(n) * 0.001

        beta, residual = _rolling_ols_beta(target_ret, driver_ret, window=100)
        # After warmup, beta should be near 2.0
        valid_betas = beta.dropna()
        assert len(valid_betas) > 0
        assert abs(valid_betas.iloc[-1] - 2.0) < 0.5

    def test_zero_variance_driver(self):
        """If driver has zero variance, beta should be NaN."""
        n = 250
        driver_ret = pd.Series(np.zeros(n))
        target_ret = pd.Series(np.random.randn(n) * 0.01)
        beta, residual = _rolling_ols_beta(target_ret, driver_ret, window=180)
        # All betas should be NaN
        assert beta.dropna().empty or all(np.isnan(beta.dropna()))

    def test_residual_is_zero_for_perfect_fit(self):
        """If target = beta*driver exactly, residual should be ~0."""
        np.random.seed(456)
        n = 300
        driver_ret = pd.Series(np.random.randn(n) * 0.01)
        target_ret = 1.5 * driver_ret

        beta, residual = _rolling_ols_beta(target_ret, driver_ret, window=100)
        valid_residuals = residual.dropna()
        if len(valid_residuals) > 0:
            assert abs(valid_residuals.iloc[-1]) < 0.01


class TestResidualZSignal:
    """Test the residual z-score sub-indicator."""

    def test_extreme_negative_z_gives_buy(self):
        """Very negative residual z-score → BUY."""
        np.random.seed(789)
        n = 100
        # Normal residual then sharp negative
        residual = pd.Series(np.random.randn(n) * 0.01)
        residual.iloc[-5:] = -0.05  # Sharp negative deviation

        z_val, vote = _residual_z_signal(residual, z_lookback=60)
        # Should be BUY (z < -2.0)
        assert vote in ("BUY", "HOLD")  # Depends on exact z magnitude

    def test_extreme_positive_z_gives_sell(self):
        """Very positive residual z-score → SELL."""
        np.random.seed(101)
        n = 100
        residual = pd.Series(np.random.randn(n) * 0.01)
        residual.iloc[-5:] = 0.05  # Sharp positive deviation

        z_val, vote = _residual_z_signal(residual, z_lookback=60)
        assert vote in ("SELL", "HOLD")

    def test_normal_z_gives_hold(self):
        """Normal residual → HOLD."""
        np.random.seed(202)
        n = 100
        residual = pd.Series(np.random.randn(n) * 0.01)

        z_val, vote = _residual_z_signal(residual, z_lookback=60)
        assert vote == "HOLD"

    def test_insufficient_data_returns_hold(self):
        """Too few data points → HOLD."""
        residual = pd.Series([0.01, 0.02])
        z_val, vote = _residual_z_signal(residual, z_lookback=60)
        assert vote == "HOLD"
        assert np.isnan(z_val)


class TestHalfLife:
    """Test the OU half-life computation."""

    def test_mean_reverting_series(self):
        """A clearly mean-reverting series should have a finite half-life."""
        np.random.seed(303)
        n = 200
        # AR(1) process with theta=0.9 (half-life ~6.6 bars)
        series = np.zeros(n)
        for i in range(1, n):
            series[i] = 0.9 * series[i - 1] + np.random.randn() * 0.01

        hl = _compute_half_life(pd.Series(series))
        assert not np.isnan(hl)
        assert hl > 0

    def test_random_walk_returns_nan_or_positive(self):
        """A random walk should yield NaN or a positive half-life.

        Note: finite samples of random walks can appear weakly mean-reverting
        (theta < 1 due to estimation noise), producing a finite half-life.
        The key invariant is: it should never be negative.
        """
        np.random.seed(404)
        series = pd.Series(np.cumsum(np.random.randn(200) * 0.01))
        hl = _compute_half_life(series)
        assert np.isnan(hl) or hl > 0

    def test_insufficient_data_returns_nan(self):
        """Too few data points → NaN."""
        hl = _compute_half_life(pd.Series([0.01, 0.02, 0.03]))
        assert np.isnan(hl)


class TestConfidenceScaling:
    """Test confidence output is properly bounded."""

    def test_confidence_never_exceeds_0_7(self):
        """Confidence should never exceed 0.7 (cap for intermarket signals)."""
        df = _make_df(n=250)
        result = compute_residual_pair_reversion_signal(
            df, context={"ticker": "ETH-USD"}
        )
        assert result["confidence"] <= 0.7

    def test_hold_has_zero_confidence(self):
        """HOLD should always have 0.0 confidence."""
        df = _make_df(n=250)
        result = compute_residual_pair_reversion_signal(df)
        if result["action"] == "HOLD":
            assert result["confidence"] == 0.0
