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
    MACD_DEAD_ZONE_MAGNITUDE_THRESHOLD,
    MACD_DEAD_ZONE_SOFT_CONF,
    _bb_inside_band_vote,
    _ema_dead_zone_vote,
    _macd_dead_zone_vote,
    _weighted_consensus,
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


# ---------------------------------------------------------------------------
# Codex Fix A: MACD magnitude gate
# ---------------------------------------------------------------------------


class TestMacdMagnitudeGate:
    """The MACD dead-zone helper must NOT fire when |current_hist| is
    well above the dead-zone threshold. The strong-vote path owns
    bars with meaningful histogram amplitude — emitting a soft vote
    there would overlap with (or contradict) the strong-vote logic.

    2026-05-11 Codex review Fix A. Prior to this fix the helper only
    inspected slope, so any non-crossover bar with a non-flat slope
    produced a soft vote regardless of histogram magnitude.
    """

    def test_macd_outside_dead_zone_returns_hold(self):
        """|hist| >> threshold + positive slope -> HOLD (strong path owns it)."""
        # Strong uptrend close series produces |hist| well above 0.05
        # under standard MACD(12, 26, 9). But we also pass macd_hist=0.5
        # in `ind` to make the magnitude gate explicit — the helper
        # should prefer the value carried in `ind` (Fix A behaviour).
        closes = [100.0 + i * 0.5 for i in range(60)]
        df = _make_df(closes)
        ind = {"close": float(closes[-1]), "macd_hist": 0.5}
        vote, conf = _macd_dead_zone_vote(ind, df)
        assert vote == "HOLD"
        assert conf == 0.0

    def test_macd_inside_dead_zone_emits_directional(self):
        """|hist| tiny (below threshold) + positive slope -> soft BUY."""
        # Same fixture as test_rising_histogram_emits_soft_buy: flat
        # then small uptick. The recomputed hist sits at ~0.028 which
        # is in the dead zone. With macd_hist=0.001 in ind we make the
        # gate explicitly pass without relying on the recomputed value.
        closes = [100.0] * 30 + [100.05, 100.10, 100.18, 100.30]
        df = _make_df(closes)
        ind = {"close": float(closes[-1]), "macd_hist": 0.001}
        vote, conf = _macd_dead_zone_vote(ind, df)
        assert vote == "BUY"
        assert conf == MACD_DEAD_ZONE_SOFT_CONF

    def test_macd_magnitude_threshold_constant_is_small(self):
        """Threshold sits comfortably below typical strong-vote amplitudes
        but above noise so the dead-zone branch can still find bars."""
        assert 0.0 < MACD_DEAD_ZONE_MAGNITUDE_THRESHOLD <= 0.10


# ---------------------------------------------------------------------------
# Codex Fix B: soft confs dampen weighted consensus
# ---------------------------------------------------------------------------


class TestSoftConsensusDampening:
    """An all-soft slate (ema/bb/macd dead-zone votes) must NOT produce
    full directional confidence — the soft_conf must dampen each vote's
    contribution. A single strong vote (e.g. RSI=25 oversold) at full
    weight should out-weight three soft votes combined.

    2026-05-11 Codex review Fix B. Prior to this fix _weighted_consensus
    treated soft votes as full-strength votes, so an all-soft slate
    produced full directional confidence — defeating the "weak weight"
    contract that the soft_conf values (0.15-0.20) were supposed to encode.
    """

    def test_all_soft_slate_has_lower_weighted_conf_than_one_strong(self):
        """3 soft BUYs (ema+bb+macd) should weigh less than 1 strong BUY."""
        # Accuracy data with consistent 60% for each signal so weight is
        # entirely driven by direction × soft_conf composition.
        accuracy_data = {
            "ema": {"accuracy": 0.60, "total": 1000,
                    "buy_accuracy": 0.60, "total_buy": 500,
                    "sell_accuracy": 0.60, "total_sell": 500},
            "bb": {"accuracy": 0.60, "total": 1000,
                   "buy_accuracy": 0.60, "total_buy": 500,
                   "sell_accuracy": 0.60, "total_sell": 500},
            "macd": {"accuracy": 0.60, "total": 1000,
                     "buy_accuracy": 0.60, "total_buy": 500,
                     "sell_accuracy": 0.60, "total_sell": 500},
            "rsi": {"accuracy": 0.60, "total": 1000,
                    "buy_accuracy": 0.60, "total_buy": 500,
                    "sell_accuracy": 0.60, "total_sell": 500},
        }
        # Scenario A: all-soft slate (ema/bb/macd BUY with soft_conf)
        soft_votes = {"ema": "BUY", "bb": "BUY", "macd": "BUY"}
        soft_confs = {
            "_soft_conf_ema": EMA_DEAD_ZONE_SOFT_CONF,
            "_soft_conf_bb": BB_INSIDE_SOFT_CONF,
            "_soft_conf_macd": MACD_DEAD_ZONE_SOFT_CONF,
        }
        action_soft, conf_soft = _weighted_consensus(
            soft_votes, accuracy_data, regime="ranging",
            soft_confidences=soft_confs,
        )
        # Scenario B: one strong BUY vote (RSI), no soft confs
        strong_votes = {"rsi": "BUY"}
        action_strong, conf_strong = _weighted_consensus(
            strong_votes, accuracy_data, regime="ranging",
            soft_confidences={},
        )
        # Both should pick BUY but the soft slate should not dominate.
        # The critical contract: 3 soft votes (each scaled by 0.15-0.20)
        # have combined weight ~3 * 0.18 * accuracy = ~0.54 * accuracy,
        # less than a single strong vote (~1.0 * accuracy). Since
        # _weighted_consensus normalises by total weight (buy/(buy+sell))
        # both unanimous-BUY scenarios may report similar normalised
        # confidence — so we assert on RAW buy_weight by directly
        # comparing aggregate weights. Use a SELL counterweight so the
        # normaliser sees both directions.
        mixed_soft = {**soft_votes, "rsi": "SELL"}
        action_mixed_soft, _ = _weighted_consensus(
            mixed_soft, accuracy_data, regime="ranging",
            soft_confidences=soft_confs,
        )
        # With 3 soft BUYs (~0.54x weight) vs 1 strong SELL (~1.0x),
        # SELL should win — confirming soft votes don't dominate.
        assert action_mixed_soft == "SELL", (
            f"All-soft slate of 3 BUYs incorrectly out-weighed 1 strong "
            f"SELL: got action={action_mixed_soft}. Soft votes must be "
            f"dampened by their soft_conf in _weighted_consensus."
        )

    def test_strong_vote_unaffected_when_no_soft_conf(self):
        """Strong votes (no _soft_conf_* key) keep full weight."""
        accuracy_data = {
            "rsi": {"accuracy": 0.60, "total": 1000,
                    "buy_accuracy": 0.60, "total_buy": 500,
                    "sell_accuracy": 0.60, "total_sell": 500},
        }
        # No soft_confidences -> full weight path
        action, conf = _weighted_consensus(
            {"rsi": "BUY"}, accuracy_data, regime="ranging",
            soft_confidences={},
        )
        assert action == "BUY"
        # Confidence should be 1.0 (only voter, BUY direction)
        assert conf >= 0.99

    def test_unknown_soft_conf_key_does_not_affect_strong_vote(self):
        """Vote whose _soft_conf_* key isn't present uses full weight."""
        accuracy_data = {
            "rsi": {"accuracy": 0.60, "total": 1000,
                    "buy_accuracy": 0.60, "total_buy": 500,
                    "sell_accuracy": 0.60, "total_sell": 500},
        }
        # soft_confidences carries entries for other signals — rsi
        # must not be affected.
        action, conf = _weighted_consensus(
            {"rsi": "BUY"}, accuracy_data, regime="ranging",
            soft_confidences={"_soft_conf_ema": 0.20},
        )
        assert action == "BUY"
        assert conf >= 0.99
