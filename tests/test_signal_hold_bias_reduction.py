"""Tests for Stage 2 Batch 1 — dead-zone soft directional votes.

2026-05-11: EMA / BB / MACD core voters abstained 88-94% of the time on
metals. The previously-HOLD branches now emit weak directional votes
based on the secondary derivative (EMA slope, BB band position, MACD
histogram slope). The strong-vote paths are unchanged. See the
module-level comment in portfolio/signal_engine.py for full rationale.

These tests pin the new behaviour at three layers:
1. The standalone helper functions return the right (vote, conf) pairs.
2. Genuine flat / mid-band inputs still return HOLD (we did NOT remove
   the abstention path — we just narrowed it).
3. Strong-vote conditions still hit the original logic with no change.
"""

import numpy as np
import pandas as pd
import pytest

from portfolio.signal_engine import (
    BB_INSIDE_SOFT_CONF,
    EMA_DEAD_ZONE_SOFT_CONF,
    MACD_DEAD_ZONE_SOFT_CONF,
    _bb_inside_band_vote,
    _ema_dead_zone_vote,
    _macd_dead_zone_vote,
)


def _make_df(closes):
    """Build a minimal OHLCV DataFrame whose `close` column drives the
    helper computations. high/low/open are synthesized around close
    because the helpers only read `close`."""
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.001,
            "low": closes * 0.999,
            "close": closes,
            "volume": np.ones_like(closes) * 100.0,
        }
    )


# ---------------------------------------------------------------------------
# EMA dead zone
# ---------------------------------------------------------------------------


class TestEmaDeadZoneVote:
    def test_rising_short_ema_emits_soft_buy(self):
        """Gap < 0.5% but EMA9 rising faster than EMA21 -> soft BUY at 0.20."""
        # Flat then sharp pickup at the tail. EMA9 reacts fast so its
        # slope over the last 3 bars is bigger than EMA21's slope, even
        # though the overall gap stays small.
        closes = [100.0] * 30 + [100.05, 100.12, 100.20, 100.30]
        df = _make_df(closes)
        ind = {"close": float(closes[-1])}
        vote, conf = _ema_dead_zone_vote(ind, df)
        assert vote == "BUY"
        assert conf == EMA_DEAD_ZONE_SOFT_CONF
        assert conf == 0.20

    def test_falling_short_ema_emits_soft_sell(self):
        """Gap < 0.5% but EMA9 falling faster than EMA21 -> soft SELL at 0.20."""
        closes = [100.0] * 30 + [99.95, 99.88, 99.80, 99.70]
        df = _make_df(closes)
        ind = {"close": float(closes[-1])}
        vote, conf = _ema_dead_zone_vote(ind, df)
        assert vote == "SELL"
        assert conf == EMA_DEAD_ZONE_SOFT_CONF

    def test_flat_slopes_stay_hold(self):
        """Truly flat close series -> EMA9 slope == EMA21 slope -> HOLD."""
        closes = [100.0] * 35
        df = _make_df(closes)
        ind = {"close": 100.0}
        vote, conf = _ema_dead_zone_vote(ind, df)
        assert vote == "HOLD"
        assert conf == 0.0

    def test_missing_df_returns_hold(self):
        """No df / too few rows -> HOLD, conf 0 (defensive fallback)."""
        ind = {"close": 100.0}
        assert _ema_dead_zone_vote(ind, None) == ("HOLD", 0.0)
        # Less than lookback + 21 bars -> not enough EMA21 history.
        short_df = _make_df([100.0] * 10)
        assert _ema_dead_zone_vote(ind, short_df) == ("HOLD", 0.0)


# ---------------------------------------------------------------------------
# Bollinger Bands inside-band
# ---------------------------------------------------------------------------


class TestBbInsideBandVote:
    def test_inside_but_near_upper_emits_soft_sell(self):
        """Price near the upper band (position > 0.6) -> soft SELL at 0.15."""
        ind = {
            "close": 100.7,
            "bb_mid": 100.0,
            "bb_upper": 101.0,
            "bb_lower": 99.0,
        }
        vote, conf = _bb_inside_band_vote(ind)
        assert vote == "SELL"
        assert conf == BB_INSIDE_SOFT_CONF
        assert conf == 0.15

    def test_inside_but_near_lower_emits_soft_buy(self):
        """Price near the lower band (position < -0.6) -> soft BUY at 0.15."""
        ind = {
            "close": 99.3,
            "bb_mid": 100.0,
            "bb_upper": 101.0,
            "bb_lower": 99.0,
        }
        vote, conf = _bb_inside_band_vote(ind)
        assert vote == "BUY"
        assert conf == BB_INSIDE_SOFT_CONF

    def test_mid_band_stays_hold(self):
        """Price right at mid-band (position ~ 0) -> HOLD, conf 0."""
        ind = {
            "close": 100.0,
            "bb_mid": 100.0,
            "bb_upper": 101.0,
            "bb_lower": 99.0,
        }
        vote, conf = _bb_inside_band_vote(ind)
        assert vote == "HOLD"
        assert conf == 0.0

    def test_just_inside_06_threshold_stays_hold(self):
        """Boundary check: position == 0.6 must NOT trigger a vote.

        (The branch is strict > 0.6 so we don't soft-vote on noise.)"""
        ind = {
            "close": 100.6,  # position = 0.6
            "bb_mid": 100.0,
            "bb_upper": 101.0,
            "bb_lower": 99.0,
        }
        assert _bb_inside_band_vote(ind) == ("HOLD", 0.0)

    def test_degenerate_band_returns_hold(self):
        """upper == mid (no band width) -> HOLD, conf 0 (no division)."""
        ind = {
            "close": 100.0,
            "bb_mid": 100.0,
            "bb_upper": 100.0,
            "bb_lower": 100.0,
        }
        assert _bb_inside_band_vote(ind) == ("HOLD", 0.0)


# ---------------------------------------------------------------------------
# MACD histogram dead zone
# ---------------------------------------------------------------------------


class TestMacdDeadZoneVote:
    def test_rising_histogram_emits_soft_buy(self):
        """|hist| small AND trending up -> soft BUY at 0.20."""
        # Long flat run, then accelerating uptick. The MACD histogram
        # is computed inside the helper from this close series.
        closes = [100.0] * 30 + [100.05, 100.10, 100.18, 100.30]
        df = _make_df(closes)
        ind = {"close": float(closes[-1])}
        vote, conf = _macd_dead_zone_vote(ind, df)
        assert vote == "BUY"
        assert conf == MACD_DEAD_ZONE_SOFT_CONF
        assert conf == 0.20

    def test_falling_histogram_emits_soft_sell(self):
        """|hist| small AND trending down -> soft SELL at 0.20."""
        closes = [100.0] * 30 + [99.95, 99.90, 99.82, 99.70]
        df = _make_df(closes)
        ind = {"close": float(closes[-1])}
        vote, conf = _macd_dead_zone_vote(ind, df)
        assert vote == "SELL"
        assert conf == MACD_DEAD_ZONE_SOFT_CONF

    def test_flat_histogram_stays_hold(self):
        """Truly flat close series -> histogram delta ~ 0 -> HOLD."""
        closes = [100.0] * 35
        df = _make_df(closes)
        ind = {"close": 100.0}
        vote, conf = _macd_dead_zone_vote(ind, df)
        assert vote == "HOLD"
        assert conf == 0.0

    def test_missing_df_returns_hold(self):
        """No df / too few rows -> HOLD (defensive)."""
        ind = {"close": 100.0}
        assert _macd_dead_zone_vote(ind, None) == ("HOLD", 0.0)
        short_df = _make_df([100.0] * 10)
        assert _macd_dead_zone_vote(ind, short_df) == ("HOLD", 0.0)


# ---------------------------------------------------------------------------
# Soft conf range guards
# ---------------------------------------------------------------------------


class TestSoftConfRanges:
    """Pin the soft conf constants in the 0.10-0.30 weak range so a
    future tuning PR can't silently push them into strong-vote territory
    without updating these tests."""

    def test_ema_soft_conf_is_weak(self):
        assert 0.10 <= EMA_DEAD_ZONE_SOFT_CONF <= 0.30

    def test_bb_soft_conf_is_weak(self):
        assert 0.10 <= BB_INSIDE_SOFT_CONF <= 0.30

    def test_macd_soft_conf_is_weak(self):
        assert 0.10 <= MACD_DEAD_ZONE_SOFT_CONF <= 0.30
