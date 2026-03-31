"""Tests for portfolio.exposure_coach — exposure ceiling recommendations."""

import pytest

from portfolio.exposure_coach import MIN_CEILING, compute_exposure_recommendation


class TestExposureRecommendation:
    def test_healthy_trending_up(self):
        """Healthy market + trending up = full exposure."""
        health = {"zone": "healthy", "score": 80}
        result = compute_exposure_recommendation(health, regime="trending-up")

        assert result["exposure_ceiling"] == 1.0
        assert result["new_entries_allowed"] is True
        assert result["bias"] == "growth"

    def test_healthy_range_bound(self):
        """Healthy market + range bound = slight reduction."""
        health = {"zone": "healthy", "score": 60}
        result = compute_exposure_recommendation(health, regime="range-bound")

        assert result["exposure_ceiling"] == 0.9
        assert result["new_entries_allowed"] is True

    def test_caution_trending_up(self):
        """Caution market + trending up = moderate exposure."""
        health = {"zone": "caution", "score": 45}
        result = compute_exposure_recommendation(health, regime="trending-up")

        assert result["exposure_ceiling"] == 0.6
        assert result["new_entries_allowed"] is True

    def test_caution_trending_down(self):
        """Caution + trending down = defensive."""
        health = {"zone": "caution", "score": 40}
        result = compute_exposure_recommendation(health, regime="trending-down")

        assert result["exposure_ceiling"] == 0.42  # 0.6 * 0.7
        assert result["bias"] == "defensive"

    def test_danger_trending_down(self):
        """Danger + trending down = minimal exposure, no new entries."""
        health = {"zone": "danger", "score": 15}
        result = compute_exposure_recommendation(health, regime="trending-down")

        assert result["exposure_ceiling"] == 0.21  # 0.3 * 0.7
        assert result["new_entries_allowed"] is False
        assert result["bias"] == "defensive"

    def test_danger_high_vol(self):
        """Danger + high vol = no new entries."""
        health = {"zone": "danger", "score": 20}
        result = compute_exposure_recommendation(health, regime="high-vol")

        assert result["exposure_ceiling"] == 0.24  # 0.3 * 0.8
        assert result["new_entries_allowed"] is False

    def test_floor_enforcement(self):
        """Ceiling never goes below MIN_CEILING."""
        health = {"zone": "danger", "score": 5}
        result = compute_exposure_recommendation(health, regime="trending-down")

        assert result["exposure_ceiling"] >= MIN_CEILING

    def test_no_health_data(self):
        """No market health = default full exposure."""
        result = compute_exposure_recommendation(None, regime="trending-up")

        assert result["exposure_ceiling"] == 1.0
        assert result["market_health_zone"] == "unknown"
        assert result["new_entries_allowed"] is True

    def test_concentration_warning(self):
        """High concentration appears in rationale."""
        health = {"zone": "healthy", "score": 70}
        result = compute_exposure_recommendation(
            health, regime="trending-up", portfolio_concentration=0.45,
        )

        assert "high concentration" in result["rationale"]

    def test_result_structure(self):
        """Result has all expected keys."""
        health = {"zone": "healthy", "score": 75}
        result = compute_exposure_recommendation(health)

        expected_keys = {
            "exposure_ceiling", "rationale", "market_health_zone",
            "market_health_score", "regime", "new_entries_allowed",
            "bias", "updated_at",
        }
        assert expected_keys == set(result.keys())

    def test_healthy_neutral_bias(self):
        """Healthy + range-bound = neutral bias."""
        health = {"zone": "healthy", "score": 60}
        result = compute_exposure_recommendation(health, regime="range-bound")
        assert result["bias"] == "neutral"

    def test_danger_trending_up_still_allows_entries(self):
        """Danger + trending up = entries still allowed (mixed signal)."""
        health = {"zone": "danger", "score": 25}
        result = compute_exposure_recommendation(health, regime="trending-up")

        # Trending up but danger zone — ceiling is low but entries allowed
        assert result["exposure_ceiling"] == 0.3
        assert result["new_entries_allowed"] is True
