"""Tests for Stage 2 Batch 2 — candlestick + forecast dead-zone soft votes.

2026-05-11: extends the soft-directional pattern from Batch 1 (EMA/BB/MACD)
to two more high-HOLD enhanced voters. Candlestick abstained 87.6% on
metals (no recognised pattern); forecast abstained 87.0% (Chronos low
confidence / accuracy-gated). The new soft branches read a secondary
derivative (candle body direction / price+EMA21 slope alignment) and
emit a weak BUY/SELL when the secondary signal points cleanly one way.
Strong-vote paths and "no data available" paths are unchanged.

Soft confs are deliberately LOWER than Batch 1 (0.12-0.15 vs 0.15-0.20)
because the secondary derivative is weaker — it's a tiebreaker, not a
pattern claim.

These tests pin the new behaviour at three layers:
1. The standalone helper functions return the right (vote, conf) pairs.
2. Mixed / ambiguous secondary signals still return HOLD.
3. The forecast soft path is SKIPPED entirely when no Chronos data
   exists (models_disabled / error / chronos_1h_pct absent).
"""

import numpy as np
import pandas as pd

from portfolio.signal_engine import (
    CANDLESTICK_DEAD_ZONE_SOFT_CONF,
    FORECAST_DEAD_ZONE_SOFT_CONF,
    _candlestick_dead_zone_vote,
    _forecast_dead_zone_vote,
)


def _make_df(closes, opens=None):
    """Build a minimal OHLCV DataFrame.

    If `opens` is None, opens are set to closes shifted down by 1 so
    every bar is bullish (close>open). Pass opens explicitly to drive
    candle body direction.
    """
    closes = np.asarray(closes, dtype=float)
    if opens is None:
        opens = closes
    else:
        opens = np.asarray(opens, dtype=float)
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) * 1.001,
            "low": np.minimum(opens, closes) * 0.999,
            "close": closes,
            "volume": np.ones_like(closes) * 100.0,
        }
    )


# ---------------------------------------------------------------------------
# Candlestick dead zone
# ---------------------------------------------------------------------------


class TestCandlestickDeadZoneVote:
    def test_three_bullish_bodies_emit_soft_buy(self):
        """Last 3 bars all close>open -> soft BUY at 0.15."""
        closes = [100.0, 100.0, 100.0, 100.0, 100.0,
                  101.0, 102.0, 103.0]  # last 3 bullish
        opens = [100.0, 100.0, 100.0, 100.0, 100.0,
                 100.5, 101.5, 102.5]
        df = _make_df(closes, opens)
        vote, conf = _candlestick_dead_zone_vote(df)
        assert vote == "BUY"
        assert conf == CANDLESTICK_DEAD_ZONE_SOFT_CONF
        assert conf == 0.15

    def test_three_bearish_bodies_emit_soft_sell(self):
        """Last 3 bars all close<open -> soft SELL at 0.15."""
        closes = [100.0, 100.0, 100.0, 100.0, 100.0,
                  99.0, 98.0, 97.0]  # last 3 bearish
        opens = [100.0, 100.0, 100.0, 100.0, 100.0,
                 99.5, 98.5, 97.5]
        df = _make_df(closes, opens)
        vote, conf = _candlestick_dead_zone_vote(df)
        assert vote == "SELL"
        assert conf == CANDLESTICK_DEAD_ZONE_SOFT_CONF

    def test_mixed_two_bullish_one_bearish_stays_hold(self):
        """2/1 split -> HOLD (soft branch only fires on unanimity)."""
        # Last 3 bars: bullish, bearish, bullish
        closes = [100.0, 100.0, 101.0, 99.5, 101.0]
        opens = [100.0, 100.0, 100.5, 100.0, 100.0]
        df = _make_df(closes, opens)
        vote, conf = _candlestick_dead_zone_vote(df)
        assert vote == "HOLD"
        assert conf == 0.0

    def test_mixed_one_bullish_two_bearish_stays_hold(self):
        """1/2 split -> HOLD."""
        closes = [100.0, 100.0, 101.0, 99.5, 98.0]
        opens = [100.0, 100.0, 100.5, 100.5, 99.0]
        df = _make_df(closes, opens)
        vote, conf = _candlestick_dead_zone_vote(df)
        assert vote == "HOLD"
        assert conf == 0.0

    def test_doji_style_equal_open_close_stays_hold(self):
        """close == open on any of the 3 bars breaks unanimity -> HOLD."""
        closes = [100.0, 100.0, 101.0, 102.0, 102.0]
        opens = [100.0, 100.0, 100.5, 101.5, 102.0]  # last bar doji
        df = _make_df(closes, opens)
        vote, conf = _candlestick_dead_zone_vote(df)
        assert vote == "HOLD"
        assert conf == 0.0

    def test_missing_df_returns_hold(self):
        """No df / too few rows -> HOLD (defensive)."""
        assert _candlestick_dead_zone_vote(None) == ("HOLD", 0.0)
        short_df = _make_df([100.0, 100.0])  # < 3 bars
        assert _candlestick_dead_zone_vote(short_df) == ("HOLD", 0.0)


# ---------------------------------------------------------------------------
# Forecast dead zone
# ---------------------------------------------------------------------------


class TestForecastDeadZoneVote:
    def _rising_df(self):
        # Mildly accelerating uptrend so both close slope AND EMA21
        # slope are positive over the last 5 bars.
        closes = list(np.linspace(100.0, 100.4, 30)) + \
            [100.6, 100.8, 101.0, 101.3, 101.7]
        return _make_df(closes)

    def _falling_df(self):
        closes = list(np.linspace(100.0, 99.6, 30)) + \
            [99.4, 99.2, 99.0, 98.7, 98.3]
        return _make_df(closes)

    def _flat_df(self):
        # No drift -> ema21 slope ~0 and close slope ~0 -> HOLD.
        closes = [100.0] * 40
        return _make_df(closes)

    def _present_indicators(self):
        """Forecast pipeline ran and produced chronos output."""
        return {
            "chronos_1h_pct": 0.001,
            "chronos_24h_pct": 0.002,
            "chronos_ok": True,
            "kronos_ok": False,
            "forecast_gating": "raw",
        }

    def test_rising_price_and_ema_emits_soft_buy(self):
        """Both close and EMA21 rising -> soft BUY at 0.12."""
        df = self._rising_df()
        vote, conf = _forecast_dead_zone_vote(df, self._present_indicators())
        assert vote == "BUY"
        assert conf == FORECAST_DEAD_ZONE_SOFT_CONF
        assert conf == 0.12

    def test_falling_price_and_ema_emits_soft_sell(self):
        """Both close and EMA21 falling -> soft SELL at 0.12."""
        df = self._falling_df()
        vote, conf = _forecast_dead_zone_vote(df, self._present_indicators())
        assert vote == "SELL"
        assert conf == FORECAST_DEAD_ZONE_SOFT_CONF

    def test_mixed_slopes_stay_hold(self):
        """Price rising but EMA21 still falling (or vice versa) -> HOLD.

        Constructed by a brief rebound after a long decline so EMA21 is
        still falling while the last 5 close bars rise.
        """
        # Long downtrend, sharp upturn at the tail. EMA21 needs many
        # bars to react.
        closes = list(np.linspace(120.0, 100.0, 50)) + \
            [100.1, 100.3, 100.5, 100.8, 101.2]
        df = _make_df(closes)
        vote, conf = _forecast_dead_zone_vote(df, self._present_indicators())
        # Could be BUY only if EMA21 slope also already turned positive;
        # with 50 bars of decline + 5 bars of recovery, EMA21 should
        # still have a non-positive slope over the last 5 bars.
        assert vote == "HOLD"
        assert conf == 0.0

    def test_flat_slopes_stay_hold(self):
        """Truly flat close series -> HOLD."""
        df = self._flat_df()
        vote, conf = _forecast_dead_zone_vote(df, self._present_indicators())
        assert vote == "HOLD"
        assert conf == 0.0

    def test_missing_forecast_data_skips_soft_branch(self):
        """No Chronos output -> HOLD (don't substitute slope for forecast).

        Critical contract: when the forecast pipeline didn't run, the
        soft branch must NOT fire. Replacing an unrun forecast with a
        slope-following vote would silently expand the forecast voter
        into a generic momentum voter — exactly the kind of substitution
        the user said to guard against.
        """
        df = self._rising_df()
        # 1) Empty indicators -> chronos_1h_pct is None AND chronos_ok
        #    is falsy -> skip.
        assert _forecast_dead_zone_vote(df, {}) == ("HOLD", 0.0)
        # 2) models_disabled flag set -> skip even if other keys present.
        assert _forecast_dead_zone_vote(
            df, {"models_disabled": True, "chronos_1h_pct": 0.001},
        ) == ("HOLD", 0.0)
        # 3) error key set -> skip.
        assert _forecast_dead_zone_vote(
            df, {"error": "insufficient_candle_data"},
        ) == ("HOLD", 0.0)
        # 4) None indicators -> treat as empty -> skip.
        assert _forecast_dead_zone_vote(df, None) == ("HOLD", 0.0)

    def test_chronos_ok_without_pct_still_fires(self):
        """If chronos_ok is True the soft branch fires even without pct.

        Belt-and-suspenders: chronos_ok=True is the explicit "Chronos
        ran successfully" flag even if pct keys are missing (e.g., on
        a HOLD verdict where pct may have rounded to None).
        """
        df = self._rising_df()
        # chronos_1h_pct missing, but chronos_ok=True -> branch fires
        vote, conf = _forecast_dead_zone_vote(
            df, {"chronos_ok": True},
        )
        assert vote == "BUY"
        assert conf == FORECAST_DEAD_ZONE_SOFT_CONF

    def test_missing_df_returns_hold(self):
        """No df / too few rows -> HOLD."""
        assert _forecast_dead_zone_vote(None, self._present_indicators()) \
            == ("HOLD", 0.0)
        short_df = _make_df([100.0] * 10)  # < lookback + 21 = 26
        assert _forecast_dead_zone_vote(short_df, self._present_indicators()) \
            == ("HOLD", 0.0)


# ---------------------------------------------------------------------------
# Soft conf range guards
# ---------------------------------------------------------------------------


class TestBatch2SoftConfRanges:
    """Pin the Batch 2 soft conf constants in the 0.10-0.20 weak range.

    These must be LOWER than Batch 1 (0.15-0.20) because candlestick and
    forecast secondary derivatives are weaker than EMA/BB/MACD ones.
    """

    def test_candlestick_soft_conf_is_weak(self):
        assert 0.10 <= CANDLESTICK_DEAD_ZONE_SOFT_CONF <= 0.20

    def test_forecast_soft_conf_is_weak(self):
        assert 0.10 <= FORECAST_DEAD_ZONE_SOFT_CONF <= 0.20

    def test_forecast_soft_conf_is_lowest(self):
        """Forecast secondary derivative is weakest -> lowest soft conf.

        Pinning this ordering so a future tuning PR can't accidentally
        rank forecast above candlestick (the user's design says forecast
        soft votes are even weaker — they fall back to a plain momentum
        read, with no model output behind them).
        """
        assert FORECAST_DEAD_ZONE_SOFT_CONF <= CANDLESTICK_DEAD_ZONE_SOFT_CONF
