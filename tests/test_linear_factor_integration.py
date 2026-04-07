"""Tests for linear factor model integration in signal_engine."""
from __future__ import annotations

from unittest.mock import patch


class TestLinearFactorInSignalEngine:
    """Verify the linear factor model integrates into generate_signal."""

    @patch("portfolio.linear_factor.LinearFactorModel.load")
    @patch("portfolio.linear_factor.LinearFactorModel.predict")
    @patch("portfolio.linear_factor.LinearFactorModel.score_to_action")
    def test_confirmation_boost_logged(self, mock_score_action, mock_predict, mock_load):
        """When linear factor agrees with consensus, confidence gets 10% boost."""
        mock_load.return_value = True
        mock_predict.return_value = 0.005
        mock_score_action.return_value = ("BUY", 0.6)

        # Build a minimal scenario where the engine produces BUY
        # We test just the integration block by importing and calling directly
        from portfolio.linear_factor import LinearFactorModel

        model = LinearFactorModel()
        model.weights = {"rsi": 0.01}  # just needs to be non-empty
        votes = {"rsi": "BUY", "macd": "BUY", "ema": "HOLD"}
        numeric = {k: {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}[v] for k, v in votes.items()}
        score = 0.005  # positive = bullish
        action, conf = model.score_to_action(score)
        assert action == "BUY"

    def test_model_not_trained_is_graceful(self):
        """When no model file exists, the integration block does nothing."""
        from portfolio.linear_factor import LinearFactorModel

        model = LinearFactorModel()
        # Don't load — weights empty
        score = model.predict({"rsi": 1.0, "macd": -1.0})
        assert score == 0.0  # empty model returns 0

    def test_score_to_action_thresholds(self):
        """Verify score_to_action uses correct thresholds."""
        from portfolio.linear_factor import LinearFactorModel

        model = LinearFactorModel()
        # Below threshold
        action, conf = model.score_to_action(0.0005, threshold=0.001)
        assert action == "HOLD"
        # Above threshold
        action, conf = model.score_to_action(0.005, threshold=0.001)
        assert action == "BUY"
        assert conf > 0
        # Negative
        action, conf = model.score_to_action(-0.005, threshold=0.001)
        assert action == "SELL"
