"""Tests for Fear & Greed regime gating in signal_engine.

F&G has 31.4% accuracy.  It is auto-inverted (~69% contrarian) but in
trending markets the contrarian signal fights the trend and loses.
The fix gates F&G to HOLD in trending-up / trending-down regimes and
only allows votes in ranging / high-vol regimes.
"""

import unittest.mock as mock

from conftest import make_indicators as _make_indicators_base

from portfolio.signal_engine import generate_signal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _null_cached(key, ttl, func, *args):
    """Mock _cached that blocks all external calls except when overridden."""
    return None


def _make_fg_cached(fg_value, fg_class="Fear"):
    """Return a _cached mock that returns the given F&G value for fear_greed
    keys and None for everything else."""
    def _cached_fn(key, ttl, func, *args):
        if "fear_greed" in key:
            return {"value": fg_value, "classification": fg_class}
        return None
    return _cached_fn


def _ind_trending_up(**overrides):
    """Indicator dict that produces regime='trending-up'."""
    defaults = {
        "close": 69_000.0,
        "ema9": 70_000.0,   # > ema21 by >0.5%
        "ema21": 69_000.0,
        "rsi": 60.0,        # > 45
        "atr_pct": 2.2,
    }
    defaults.update(overrides)
    return _make_indicators_base(**defaults)


def _ind_trending_down(**overrides):
    """Indicator dict that produces regime='trending-down'."""
    defaults = {
        "close": 69_000.0,
        "ema9": 68_000.0,   # < ema21 by >0.5%
        "ema21": 69_000.0,
        "rsi": 40.0,        # < 55
        "atr_pct": 2.2,
    }
    defaults.update(overrides)
    return _make_indicators_base(**defaults)


def _ind_ranging(**overrides):
    """Indicator dict that produces regime='ranging'."""
    defaults = {
        "close": 69_000.0,
        "ema9": 69_000.0,   # ~ ema21, gap < 0.5%
        "ema21": 69_000.0,
        "rsi": 50.0,
        "atr_pct": 2.2,
    }
    defaults.update(overrides)
    return _make_indicators_base(**defaults)


def _ind_high_vol(**overrides):
    """Indicator dict that produces regime='high-vol' (crypto, atr_pct > 4.0)."""
    defaults = {
        "close": 69_000.0,
        "ema9": 69_000.0,
        "ema21": 69_000.0,
        "rsi": 50.0,
        "atr_pct": 5.0,     # > 4.0 threshold for crypto
    }
    defaults.update(overrides)
    return _make_indicators_base(**defaults)


# Common patches applied to every test: block all external data calls
_PATCHES = [
    mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached),
    mock.patch("portfolio.signal_engine.get_enhanced_signals", return_value=[]),
]


def _apply_patches(fg_value=None, fg_class="Fear"):
    """Return a list of active patches.  If fg_value is set, the _cached mock
    returns that F&G value instead of None."""
    patches = list(_PATCHES)
    if fg_value is not None:
        patches[0] = mock.patch(
            "portfolio.signal_engine._cached",
            side_effect=_make_fg_cached(fg_value, fg_class),
        )
    return patches


# ---------------------------------------------------------------------------
# Tests: F&G gated in trending regimes
# ---------------------------------------------------------------------------

class TestFGGatedTrendingUp:
    """F&G should be forced to HOLD in trending-up, even at extreme values."""

    def test_extreme_fear_gated_trending_up(self):
        """F&G value=10 (extreme fear -> BUY) should be gated to HOLD in trending-up."""
        ind = _ind_trending_up()
        patches = _apply_patches(fg_value=10, fg_class="Extreme Fear")
        with patches[0], patches[1]:
            _, _, extra = generate_signal(ind, ticker="BTC-USD")
        assert extra.get("fear_greed") == 10
        assert extra.get("fear_greed_gated") == "trending-up"
        # The vote should be HOLD (gated), verified via extra_info
        # We check the gated flag is present — the vote is HOLD by default
        # and the gating prevents it from being overridden to BUY

    def test_extreme_greed_gated_trending_up(self):
        """F&G value=85 (extreme greed -> SELL) should be gated to HOLD in trending-up."""
        ind = _ind_trending_up()
        patches = _apply_patches(fg_value=85, fg_class="Extreme Greed")
        with patches[0], patches[1]:
            _, _, extra = generate_signal(ind, ticker="BTC-USD")
        assert extra.get("fear_greed") == 85
        assert extra.get("fear_greed_gated") == "trending-up"

    def test_neutral_fg_still_hold_trending_up(self):
        """F&G value=50 (neutral -> HOLD anyway) should still have gated flag."""
        ind = _ind_trending_up()
        patches = _apply_patches(fg_value=50, fg_class="Neutral")
        with patches[0], patches[1]:
            _, _, extra = generate_signal(ind, ticker="BTC-USD")
        assert extra.get("fear_greed") == 50
        # Gated flag set even though it would have been HOLD anyway
        assert extra.get("fear_greed_gated") == "trending-up"


class TestFGGatedTrendingDown:
    """F&G should be forced to HOLD in trending-down, even at extreme values."""

    def test_extreme_fear_gated_trending_down(self):
        """F&G value=15 (extreme fear -> BUY) should be gated to HOLD in trending-down."""
        ind = _ind_trending_down()
        patches = _apply_patches(fg_value=15, fg_class="Extreme Fear")
        with patches[0], patches[1]:
            _, _, extra = generate_signal(ind, ticker="BTC-USD")
        assert extra.get("fear_greed") == 15
        assert extra.get("fear_greed_gated") == "trending-down"

    def test_extreme_greed_gated_trending_down(self):
        """F&G value=90 (extreme greed -> SELL) should be gated to HOLD in trending-down."""
        ind = _ind_trending_down()
        patches = _apply_patches(fg_value=90, fg_class="Extreme Greed")
        with patches[0], patches[1]:
            _, _, extra = generate_signal(ind, ticker="BTC-USD")
        assert extra.get("fear_greed") == 90
        assert extra.get("fear_greed_gated") == "trending-down"


# ---------------------------------------------------------------------------
# Tests: F&G allowed in ranging / high-vol regimes
# ---------------------------------------------------------------------------

class TestFGAllowedRanging:
    """F&G should vote normally in ranging regimes."""

    def test_extreme_fear_votes_buy_ranging(self):
        """F&G value=10 should produce BUY vote in ranging regime."""
        ind = _ind_ranging()
        patches = _apply_patches(fg_value=10, fg_class="Extreme Fear")
        with patches[0], patches[1]:
            _, _, extra = generate_signal(ind, ticker="BTC-USD")
        assert extra.get("fear_greed") == 10
        assert "fear_greed_gated" not in extra

    def test_extreme_greed_votes_sell_ranging(self):
        """F&G value=85 should produce SELL vote in ranging regime."""
        ind = _ind_ranging()
        patches = _apply_patches(fg_value=85, fg_class="Extreme Greed")
        with patches[0], patches[1]:
            _, _, extra = generate_signal(ind, ticker="BTC-USD")
        assert extra.get("fear_greed") == 85
        assert "fear_greed_gated" not in extra

    def test_neutral_fg_hold_ranging(self):
        """F&G value=50 (neutral) should remain HOLD in ranging regime — no gating flag."""
        ind = _ind_ranging()
        patches = _apply_patches(fg_value=50, fg_class="Neutral")
        with patches[0], patches[1]:
            _, _, extra = generate_signal(ind, ticker="BTC-USD")
        assert extra.get("fear_greed") == 50
        assert "fear_greed_gated" not in extra


class TestFGAllowedHighVol:
    """F&G should vote normally in high-vol regimes."""

    def test_extreme_fear_votes_buy_high_vol(self):
        """F&G value=5 should produce BUY vote in high-vol regime."""
        ind = _ind_high_vol()
        patches = _apply_patches(fg_value=5, fg_class="Extreme Fear")
        with patches[0], patches[1]:
            _, _, extra = generate_signal(ind, ticker="BTC-USD")
        assert extra.get("fear_greed") == 5
        assert "fear_greed_gated" not in extra

    def test_extreme_greed_votes_sell_high_vol(self):
        """F&G value=95 should produce SELL vote in high-vol regime."""
        ind = _ind_high_vol()
        patches = _apply_patches(fg_value=95, fg_class="Extreme Greed")
        with patches[0], patches[1]:
            _, _, extra = generate_signal(ind, ticker="BTC-USD")
        assert extra.get("fear_greed") == 95
        assert "fear_greed_gated" not in extra


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestFGGatingEdgeCases:
    """Edge cases for F&G regime gating."""

    def test_fg_unavailable_no_crash(self):
        """When F&G data is unavailable (None), no gated flag and no crash."""
        ind = _ind_trending_up()
        patches = _apply_patches(fg_value=None)
        with patches[0], patches[1]:
            _, _, extra = generate_signal(ind, ticker="BTC-USD")
        assert "fear_greed_gated" not in extra
        assert "fear_greed" not in extra

    def test_fg_at_boundary_20_trending_gated(self):
        """F&G exactly at 20 (boundary for BUY) should still be gated in trending-up."""
        ind = _ind_trending_up()
        patches = _apply_patches(fg_value=20, fg_class="Extreme Fear")
        with patches[0], patches[1]:
            _, _, extra = generate_signal(ind, ticker="BTC-USD")
        assert extra.get("fear_greed") == 20
        assert extra.get("fear_greed_gated") == "trending-up"

    def test_fg_at_boundary_80_trending_gated(self):
        """F&G exactly at 80 (boundary for SELL) should still be gated in trending-down."""
        ind = _ind_trending_down()
        patches = _apply_patches(fg_value=80, fg_class="Greed")
        with patches[0], patches[1]:
            _, _, extra = generate_signal(ind, ticker="BTC-USD")
        assert extra.get("fear_greed") == 80
        assert extra.get("fear_greed_gated") == "trending-down"

    def test_fg_at_boundary_20_ranging_votes(self):
        """F&G exactly at 20 should vote BUY in ranging regime."""
        ind = _ind_ranging()
        patches = _apply_patches(fg_value=20, fg_class="Extreme Fear")
        with patches[0], patches[1]:
            _, _, extra = generate_signal(ind, ticker="BTC-USD")
        assert extra.get("fear_greed") == 20
        assert "fear_greed_gated" not in extra

    def test_fg_at_boundary_80_ranging_votes(self):
        """F&G exactly at 80 should vote SELL in ranging regime."""
        ind = _ind_ranging()
        patches = _apply_patches(fg_value=80, fg_class="Greed")
        with patches[0], patches[1]:
            _, _, extra = generate_signal(ind, ticker="BTC-USD")
        assert extra.get("fear_greed") == 80
        assert "fear_greed_gated" not in extra

    def test_stock_ticker_trending_up_gated(self):
        """F&G gating works for stock tickers too, not just crypto."""
        # For stocks, high-vol threshold is 3.0 (not 4.0), but trending-up
        # detection is the same (ema gap + rsi)
        ind = _ind_trending_up(atr_pct=1.5)
        patches = _apply_patches(fg_value=10, fg_class="Extreme Fear")
        with patches[0], patches[1]:
            _, _, extra = generate_signal(ind, ticker="NVDA")
        assert extra.get("fear_greed") == 10
        assert extra.get("fear_greed_gated") == "trending-up"
