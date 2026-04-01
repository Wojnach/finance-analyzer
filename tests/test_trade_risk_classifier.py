"""Tests for the trade risk classifier."""

from portfolio.trade_risk_classifier import classify_trade_risk


class TestTradeRiskClassifier:

    def test_small_with_trend_is_low(self):
        """BUY, 0.7 conf, 5% pos, trending-up, 80% consensus -> LOW."""
        result = classify_trade_risk(
            action="BUY",
            confidence=0.7,
            position_pct=5.0,
            regime="trending-up",
            consensus_ratio=0.80,
        )
        assert result["level"] == "LOW"

    def test_large_counter_trend_is_high(self):
        """BUY, 0.5 conf, 25% pos, trending-down, 55% consensus -> HIGH."""
        result = classify_trade_risk(
            action="BUY",
            confidence=0.5,
            position_pct=25.0,
            regime="trending-down",
            consensus_ratio=0.55,
        )
        assert result["level"] == "HIGH"

    def test_medium_mixed_signals(self):
        """BUY, 0.6 conf, 12% pos, ranging, 65% consensus -> MEDIUM."""
        result = classify_trade_risk(
            action="BUY",
            confidence=0.6,
            position_pct=12.0,
            regime="ranging",
            consensus_ratio=0.65,
        )
        assert result["level"] == "MEDIUM"

    def test_hold_is_always_low(self):
        """HOLD action always returns LOW regardless of other params."""
        result = classify_trade_risk(
            action="HOLD",
            confidence=0.1,
            position_pct=50.0,
            regime="capitulation",
            consensus_ratio=0.10,
            existing_exposure_pct=80.0,
        )
        assert result["level"] == "LOW"
        assert result["score"] == 0
        assert result["factors"] == []

    def test_returns_risk_factors(self):
        """Result contains 'factors' list and 'score' int."""
        result = classify_trade_risk(
            action="BUY",
            confidence=0.4,
            position_pct=15.0,
            regime="high-vol",
            consensus_ratio=0.55,
            existing_exposure_pct=30.0,
        )
        assert isinstance(result["factors"], list)
        assert len(result["factors"]) > 0
        assert isinstance(result["score"], int)
        assert result["score"] > 0
        # Verify specific factors are present
        factor_text = " ".join(result["factors"])
        assert "position" in factor_text
        assert "regime" in factor_text
        assert "consensus" in factor_text
        assert "confidence" in factor_text
        assert "concentration" in factor_text
