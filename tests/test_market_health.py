"""Tests for portfolio.market_health — distribution days, FTD, breadth score."""

import json
from unittest.mock import MagicMock, patch

import pytest

from portfolio.market_health import (
    STATE_CONFIRMED_UPTREND,
    STATE_CORRECTING,
    STATE_FTD_CONFIRMED,
    STATE_RALLY_ATTEMPT,
    ZONE_CAUTION,
    ZONE_DANGER,
    _classify_zone,
    compute_breadth_score,
    count_distribution_days,
    detect_ftd_state,
    get_confidence_penalty,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_series(n=30, start=100.0, daily_return=0.001, vol_base=1_000_000):
    """Generate synthetic OHLCV data."""
    closes = []
    opens = []
    highs = []
    lows = []
    volumes = []
    price = start
    for i in range(n):
        open_p = price
        close_p = price * (1 + daily_return)
        high_p = max(open_p, close_p) * 1.005
        low_p = min(open_p, close_p) * 0.995
        vol = vol_base + (i % 5) * 100_000  # slight volume variation
        opens.append(open_p)
        closes.append(close_p)
        highs.append(high_p)
        lows.append(low_p)
        volumes.append(vol)
        price = close_p
    return closes, volumes, highs, lows, opens


# ---------------------------------------------------------------------------
# Distribution day counting
# ---------------------------------------------------------------------------

class TestDistributionDays:
    def test_no_distribution_in_uptrend(self):
        """Steady uptrend with declining volume = no distribution."""
        closes = [100 + i * 0.5 for i in range(30)]
        volumes = [1_000_000 - i * 10_000 for i in range(30)]  # declining volume
        highs = [c + 0.3 for c in closes]
        lows = [c - 0.3 for c in closes]

        result = count_distribution_days(closes, volumes, highs, lows)
        assert result["distribution_days"] == 0

    def test_single_distribution_day(self):
        """One day with >=0.2% drop on higher volume."""
        closes = [100.0] * 10
        volumes = [1_000_000] * 10
        # Day 5: drop 0.3% on higher volume
        closes[5] = 99.7
        volumes[5] = 1_500_000

        highs = [c + 0.5 for c in closes]
        lows = [c - 0.5 for c in closes]

        result = count_distribution_days(closes, volumes, highs, lows)
        assert result["distribution_days"] == 1
        assert result["details"][0]["type"] == "distribution"

    def test_multiple_distribution_days(self):
        """Multiple distribution days accumulate."""
        closes = [100.0] * 30
        volumes = [1_000_000] * 30

        # Create 4 distribution days
        for i in [10, 15, 20, 25]:
            closes[i] = closes[i - 1] * 0.995  # -0.5%
            volumes[i] = volumes[i - 1] + 500_000  # higher volume

        highs = [c + 0.5 for c in closes]
        lows = [c - 0.5 for c in closes]

        result = count_distribution_days(closes, volumes, highs, lows)
        assert result["distribution_days"] == 4

    def test_stalling_day(self):
        """Close in upper 25% of range on higher volume with tiny gain."""
        closes = [100.0] * 10
        volumes = [1_000_000] * 10

        # Day 5: tiny gain, closes near high, higher volume
        closes[5] = 100.1  # +0.1% (< 0.2% threshold)
        volumes[5] = 1_500_000

        # Build highs/lows first, then override day 5
        highs = [c + 0.1 for c in closes]
        lows = [c - 0.1 for c in closes]
        highs[5] = 100.15
        lows[5] = 99.5  # range = 0.65, close at 100.1 is in upper 25%

        result = count_distribution_days(closes, volumes, highs, lows)
        assert result["stalling_days"] >= 1

    def test_not_distribution_if_volume_lower(self):
        """Price drop on LOWER volume is NOT a distribution day."""
        closes = [100.0] * 10
        volumes = [1_000_000] * 10
        closes[5] = 99.5  # -0.5%
        volumes[5] = 500_000  # LOWER volume

        highs = [c + 0.5 for c in closes]
        lows = [c - 0.5 for c in closes]

        result = count_distribution_days(closes, volumes, highs, lows)
        assert result["distribution_days"] == 0

    def test_window_limits(self):
        """Only count distribution days within the rolling window."""
        closes = [100.0] * 50
        volumes = [1_000_000] * 50

        # Distribution day at position 5 (outside 25-day window)
        closes[5] = 99.5
        volumes[5] = 1_500_000

        # Distribution day at position 40 (inside window)
        closes[40] = 99.5
        volumes[40] = 1_500_000

        highs = [c + 0.5 for c in closes]
        lows = [c - 0.5 for c in closes]

        result = count_distribution_days(closes, volumes, highs, lows, window=25)
        assert result["distribution_days"] == 1  # only the one inside window

    def test_empty_data(self):
        """Graceful handling of insufficient data."""
        result = count_distribution_days([], [], [], [])
        assert result["distribution_days"] == 0

    def test_single_bar(self):
        """Single bar = no distribution possible."""
        result = count_distribution_days([100], [1000], [101], [99])
        assert result["distribution_days"] == 0

    def test_total_pressure(self):
        """total_pressure = distribution_days + stalling_days."""
        closes = [100.0] * 10
        volumes = [1_000_000] * 10
        closes[5] = 99.5
        volumes[5] = 1_500_000

        highs = [c + 0.5 for c in closes]
        lows = [c - 0.5 for c in closes]

        result = count_distribution_days(closes, volumes, highs, lows)
        assert result["total_pressure"] == result["distribution_days"] + result["stalling_days"]


# ---------------------------------------------------------------------------
# FTD state machine
# ---------------------------------------------------------------------------

class TestFTDDetection:
    def test_correcting_state(self):
        """Index in correction = correcting state."""
        # Create data with a 7% drawdown from high
        closes = [100 + i * 0.5 for i in range(30)]  # uptrend
        peak = closes[-1]
        # Add 10 days of decline
        for i in range(10):
            closes.append(peak * (1 - 0.01 * (i + 1)))
        volumes = [1_000_000] * len(closes)

        result = detect_ftd_state(closes, volumes)
        assert result["state"] == STATE_CORRECTING

    def test_rally_attempt_starts(self):
        """First up day after correction = rally attempt."""
        # Create correction
        closes = [110 - i * 0.6 for i in range(30)]  # downtrend
        closes[-1] = closes[-2] * 1.005  # tiny up day
        volumes = [1_000_000] * len(closes)

        result = detect_ftd_state(closes, volumes)
        # Should be in rally_attempt or correcting depending on drawdown
        assert result["state"] in (STATE_RALLY_ATTEMPT, STATE_CORRECTING)

    def test_ftd_on_day_4(self):
        """FTD occurs on day 4+ with >=1.25% gain on higher volume."""
        # Build: peak → correction → rally attempt → FTD
        closes = [100 + i * 0.3 for i in range(20)]  # uptrend to ~106
        peak = closes[-1]

        # Correction: -7%
        for _ in range(5):
            closes.append(closes[-1] * 0.985)

        # Rally days 1-3 (small gains)
        for _ in range(3):
            closes.append(closes[-1] * 1.003)

        # Day 4: big gain (FTD candidate)
        closes.append(closes[-1] * 1.015)  # +1.5%

        volumes = [1_000_000] * len(closes)
        # FTD day needs higher volume than previous
        volumes[-1] = 1_500_000
        volumes[-2] = 1_000_000

        # Need to process state machine incrementally
        prev_state = {
            "state": STATE_RALLY_ATTEMPT,
            "rally_day": 3,
            "rally_low": min(closes[-8:-4]),
            "recent_high": peak,
            "ftd_day_offset": None,
        }

        result = detect_ftd_state(closes, volumes, prev_state)
        assert result["state"] == STATE_FTD_CONFIRMED

    def test_ftd_fails_on_undercut(self):
        """FTD fails if price undercuts rally low."""
        closes = [100.0] * 30
        volumes = [1_000_000] * 30

        prev_state = {
            "state": STATE_FTD_CONFIRMED,
            "rally_day": 5,
            "rally_low": 95.0,
            "recent_high": 105.0,
            "ftd_day_offset": 25,
        }

        # Today undercuts rally low
        closes[-1] = 94.0
        closes[-2] = 96.0

        result = detect_ftd_state(closes, volumes, prev_state)
        assert result["state"] == STATE_CORRECTING

    def test_confirmed_uptrend_after_window(self):
        """FTD holds past failure window = confirmed uptrend."""
        closes = [100.0 + i * 0.1 for i in range(40)]
        volumes = [1_000_000] * 40

        prev_state = {
            "state": STATE_FTD_CONFIRMED,
            "rally_day": 5,
            "rally_low": 95.0,
            "recent_high": 104.0,
            "ftd_day_offset": 20,  # 19 days ago (> FTD_FAILURE_WINDOW)
        }

        result = detect_ftd_state(closes, volumes, prev_state)
        assert result["state"] == STATE_CONFIRMED_UPTREND

    def test_confirmed_uptrend_breaks_on_correction(self):
        """Confirmed uptrend reverts to correcting on 5%+ drawdown."""
        closes = [100.0] * 30
        volumes = [1_000_000] * 30

        prev_state = {
            "state": STATE_CONFIRMED_UPTREND,
            "rally_day": 10,
            "rally_low": 95.0,
            "recent_high": 105.0,
            "ftd_day_offset": 15,
        }

        # Big drop below 5% correction threshold
        closes[-1] = 99.0  # 99/105 = -5.7%
        closes[-2] = 103.0

        result = detect_ftd_state(closes, volumes, prev_state)
        assert result["state"] == STATE_CORRECTING

    def test_insufficient_data(self):
        """With <20 bars, returns correcting as safe default."""
        result = detect_ftd_state([100.0] * 10, [1000] * 10)
        assert result["state"] == STATE_CORRECTING


# ---------------------------------------------------------------------------
# Breadth score
# ---------------------------------------------------------------------------

class TestBreadthScore:
    def test_perfect_score(self):
        """Healthy market = high score."""
        dist_data = {"total_pressure": 0}
        ftd_state = {"state": STATE_CONFIRMED_UPTREND}
        # 250 bars of uptrend
        closes = [100 + i * 0.2 for i in range(250)]

        result = compute_breadth_score(dist_data, ftd_state, closes)
        assert result["score"] >= 80

    def test_danger_score(self):
        """Unhealthy market = low score."""
        dist_data = {"total_pressure": 7}
        ftd_state = {"state": STATE_CORRECTING}
        # 250 bars of downtrend
        closes = [150 - i * 0.3 for i in range(250)]

        result = compute_breadth_score(dist_data, ftd_state, closes)
        assert result["score"] < ZONE_DANGER

    def test_caution_zone(self):
        """Mixed signals = moderate score (not extreme in either direction)."""
        dist_data = {"total_pressure": 3}
        ftd_state = {"state": STATE_RALLY_ATTEMPT}
        # Slight uptrend so SMA components contribute
        closes = [100 + i * 0.05 for i in range(250)]

        result = compute_breadth_score(dist_data, ftd_state, closes)
        # With 3 dist days, rally_attempt, above SMAs, slight uptrend
        # Not a perfect score, but not danger either
        assert 30 <= result["score"] <= 70

    def test_components_sum_to_score(self):
        """Score equals sum of components."""
        dist_data = {"total_pressure": 2}
        ftd_state = {"state": STATE_FTD_CONFIRMED}
        closes = [100 + i * 0.1 for i in range(250)]

        result = compute_breadth_score(dist_data, ftd_state, closes)
        assert result["score"] == sum(result["components"].values())

    def test_short_data_handled(self):
        """Short data (< 50 bars) uses neutral defaults."""
        dist_data = {"total_pressure": 0}
        ftd_state = {"state": STATE_CONFIRMED_UPTREND}
        closes = [100 + i * 0.1 for i in range(30)]

        result = compute_breadth_score(dist_data, ftd_state, closes)
        # Should not crash, should give partial score
        assert 0 <= result["score"] <= 100


# ---------------------------------------------------------------------------
# Zone classification
# ---------------------------------------------------------------------------

class TestZoneClassification:
    def test_danger(self):
        assert _classify_zone(20) == "danger"
        assert _classify_zone(0) == "danger"
        assert _classify_zone(29) == "danger"

    def test_caution(self):
        assert _classify_zone(30) == "caution"
        assert _classify_zone(45) == "caution"
        assert _classify_zone(49) == "caution"

    def test_healthy(self):
        assert _classify_zone(50) == "healthy"
        assert _classify_zone(75) == "healthy"
        assert _classify_zone(100) == "healthy"


# ---------------------------------------------------------------------------
# Confidence penalty
# ---------------------------------------------------------------------------

class TestConfidencePenalty:
    def test_buy_in_danger(self):
        health = {"score": 20}
        penalty = get_confidence_penalty("BUY", health)
        assert penalty == 0.6

    def test_buy_in_caution(self):
        health = {"score": 40}
        penalty = get_confidence_penalty("BUY", health)
        assert penalty == 0.8

    def test_buy_in_healthy(self):
        health = {"score": 55}
        penalty = get_confidence_penalty("BUY", health)
        assert penalty == 1.0

    def test_buy_in_very_healthy(self):
        health = {"score": 80}
        penalty = get_confidence_penalty("BUY", health)
        assert penalty == 1.1

    def test_sell_unaffected(self):
        """SELL signals are never penalized."""
        health = {"score": 10}  # danger zone
        penalty = get_confidence_penalty("SELL", health)
        assert penalty == 1.0

    def test_hold_unaffected(self):
        health = {"score": 10}
        penalty = get_confidence_penalty("HOLD", health)
        assert penalty == 1.0

    def test_no_health_data(self):
        """No data = no penalty."""
        assert get_confidence_penalty("BUY", None) == 1.0
        assert get_confidence_penalty("BUY") == 1.0


# ---------------------------------------------------------------------------
# Integration: get_market_health
# ---------------------------------------------------------------------------

class TestGetMarketHealth:
    @patch("portfolio.market_health._fetch_index_data")
    @patch("portfolio.market_health.load_json", return_value={})
    @patch("portfolio.market_health.atomic_write_json")
    def test_compute_returns_valid_structure(self, mock_write, mock_load, mock_fetch):
        """Full computation returns expected keys."""
        closes, volumes, highs, lows, opens = _make_series(90)
        mock_fetch.return_value = {
            "closes": closes,
            "volumes": volumes,
            "highs": highs,
            "lows": lows,
            "opens": opens,
        }

        from portfolio.market_health import _compute_market_health

        result = _compute_market_health()
        assert result is not None
        assert "score" in result
        assert "zone" in result
        assert result["zone"] in ("healthy", "caution", "danger")
        assert "distribution_days_spy" in result
        assert "ftd_state" in result
        assert "components" in result
        assert "updated_at" in result

    @patch("portfolio.market_health._fetch_index_data", return_value=None)
    def test_returns_none_on_data_failure(self, mock_fetch):
        from portfolio.market_health import _compute_market_health

        result = _compute_market_health()
        assert result is None
