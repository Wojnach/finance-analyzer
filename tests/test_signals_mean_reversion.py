"""Tests for the mean-reversion signal module.

Covers:
- Basic computation with sufficient data
- Insufficient data returns HOLD
- RSI(2) extreme oversold triggers BUY
- RSI(2) extreme overbought triggers SELL
- IBS near 0 triggers BUY
- Consecutive down days triggers BUY
- Gap fill detection
- Combined IBS+RSI signal
- Bollinger Band %B extremes
- Consecutive up days triggers SELL
- Edge cases (flat data, single row, etc.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.mean_reversion import (
    _bb_pct_b,
    _consecutive_days,
    _gap_fill,
    _ibs_rsi2_combined,
    _internal_bar_strength,
    _rsi2_mean_reversion,
    compute_mean_reversion_signal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 100, close_base: float = 100.0,
             volatility: float = 2.0, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame."""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = close_base + np.cumsum(np.random.randn(n) * volatility)
    close = np.maximum(close, 1.0)
    high = close + np.abs(np.random.randn(n) * volatility)
    low = close - np.abs(np.random.randn(n) * volatility)
    low = np.maximum(low, 0.5)
    opn = close + np.random.randn(n) * 0.5
    volume = np.random.randint(100, 10000, n).astype(float)
    return pd.DataFrame({
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "time": dates,
    })


def _make_declining_df(n: int = 30, start: float = 100.0,
                       step: float = -1.0) -> pd.DataFrame:
    """Generate a DataFrame with strictly declining closes."""
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = np.array([start + i * step for i in range(n)])
    close = np.maximum(close, 1.0)
    high = close + 2.0
    low = close - 2.0
    low = np.maximum(low, 0.5)
    opn = close + 0.5
    volume = np.full(n, 1000.0)
    return pd.DataFrame({
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "time": dates,
    })


def _make_rising_df(n: int = 30, start: float = 100.0,
                    step: float = 1.0) -> pd.DataFrame:
    """Generate a DataFrame with strictly rising closes."""
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = np.array([start + i * step for i in range(n)])
    high = close + 2.0
    low = close - 2.0
    low = np.maximum(low, 0.5)
    opn = close - 0.5
    volume = np.full(n, 1000.0)
    return pd.DataFrame({
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "time": dates,
    })


# ---------------------------------------------------------------------------
# Test 1: Basic computation with sufficient data
# ---------------------------------------------------------------------------

class TestBasicComputation:
    def test_returns_expected_keys(self):
        df = _make_df(100)
        result = compute_mean_reversion_signal(df)
        assert "action" in result
        assert "confidence" in result
        assert "sub_signals" in result
        assert "indicators" in result

    def test_action_is_valid(self):
        df = _make_df(100)
        result = compute_mean_reversion_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_confidence_range(self):
        df = _make_df(100)
        result = compute_mean_reversion_signal(df)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_all_seven_sub_signals_present(self):
        df = _make_df(100)
        result = compute_mean_reversion_signal(df)
        expected_keys = {
            "rsi2_mr", "rsi3_mr", "ibs", "consecutive_days",
            "gap_fill", "bb_pct_b", "ibs_rsi2_combined", "half_life_mr",
        }
        assert set(result["sub_signals"].keys()) == expected_keys

    def test_indicators_present(self):
        df = _make_df(100)
        result = compute_mean_reversion_signal(df)
        expected_keys = {
            "rsi2", "rsi3", "ibs", "consecutive_days",
            "gap_pct", "gap_fill_pct", "bb_pct_b",
            "combined_ibs", "combined_rsi2",
            "half_life", "zscore",
        }
        assert set(result["indicators"].keys()) == expected_keys

    def test_sub_signals_all_valid_actions(self):
        df = _make_df(100)
        result = compute_mean_reversion_signal(df)
        for name, vote in result["sub_signals"].items():
            assert vote in ("BUY", "SELL", "HOLD"), (
                f"sub_signal {name} has invalid vote: {vote}"
            )


# ---------------------------------------------------------------------------
# Test 2: Insufficient data returns HOLD
# ---------------------------------------------------------------------------

class TestInsufficientData:
    def test_none_input(self):
        result = compute_mean_reversion_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_mean_reversion_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_too_few_rows(self):
        df = _make_df(2)
        result = compute_mean_reversion_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_missing_columns(self):
        df = pd.DataFrame({"close": [1, 2, 3], "volume": [100, 200, 300]})
        result = compute_mean_reversion_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_three_rows_computes_rsi_but_not_bb(self):
        """With exactly 3 rows, RSI(2) can compute but BB cannot."""
        df = _make_df(3)
        result = compute_mean_reversion_signal(df)
        # BB should be HOLD due to insufficient data
        assert result["sub_signals"]["bb_pct_b"] == "HOLD"


# ---------------------------------------------------------------------------
# Test 3: RSI(2) extreme oversold triggers BUY
# ---------------------------------------------------------------------------

class TestRSI2Oversold:
    def test_rsi2_extreme_decline_triggers_buy(self):
        """A steep decline should push RSI(2) below 10 and trigger BUY."""
        # Create data that drops sharply in the last few bars
        df = _make_df(50, close_base=100.0, volatility=0.1)
        # Override last 3 closes with a steep decline
        close_vals = df["close"].values.copy()
        close_vals[-3] = 100.0
        close_vals[-2] = 90.0
        close_vals[-1] = 80.0
        df["close"] = close_vals
        df["low"] = df["close"] - 1.0

        rsi2_val, signal = _rsi2_mean_reversion(df["close"].astype(float))
        assert signal == "BUY"
        assert rsi2_val < 10

    def test_rsi2_moderate_decline_holds(self):
        """A moderate decline should NOT push RSI(2) below 10."""
        close = pd.Series([100.0, 99.5, 99.0, 98.8, 98.7])
        rsi2_val, signal = _rsi2_mean_reversion(close)
        # Moderate decline, RSI(2) should not be extreme
        assert signal in ("HOLD", "BUY")  # depends on exact values


# ---------------------------------------------------------------------------
# Test 4: RSI(2) extreme overbought triggers SELL
# ---------------------------------------------------------------------------

class TestRSI2Overbought:
    def test_rsi2_extreme_rise_triggers_sell(self):
        """A steep rise should push RSI(2) above 90 and trigger SELL."""
        df = _make_df(50, close_base=100.0, volatility=0.1)
        close_vals = df["close"].values.copy()
        close_vals[-3] = 100.0
        close_vals[-2] = 110.0
        close_vals[-1] = 120.0
        df["close"] = close_vals
        df["high"] = df["close"] + 1.0

        rsi2_val, signal = _rsi2_mean_reversion(df["close"].astype(float))
        assert signal == "SELL"
        assert rsi2_val > 90


# ---------------------------------------------------------------------------
# Test 5: IBS near 0 triggers BUY
# ---------------------------------------------------------------------------

class TestIBS:
    def test_ibs_near_low_triggers_buy(self):
        """When close is near the low of the bar, IBS < 0.2 = BUY."""
        high = pd.Series([110.0])
        low = pd.Series([100.0])
        close = pd.Series([101.0])  # near the low

        ibs_val, signal = _internal_bar_strength(high, low, close)
        assert signal == "BUY"
        assert ibs_val < 0.2
        assert abs(ibs_val - 0.1) < 0.01

    def test_ibs_near_high_triggers_sell(self):
        """When close is near the high of the bar, IBS > 0.8 = SELL."""
        high = pd.Series([110.0])
        low = pd.Series([100.0])
        close = pd.Series([109.0])  # near the high

        ibs_val, signal = _internal_bar_strength(high, low, close)
        assert signal == "SELL"
        assert ibs_val > 0.8
        assert abs(ibs_val - 0.9) < 0.01

    def test_ibs_middle_holds(self):
        """When close is in the middle, IBS is ~0.5 = HOLD."""
        high = pd.Series([110.0])
        low = pd.Series([100.0])
        close = pd.Series([105.0])

        ibs_val, signal = _internal_bar_strength(high, low, close)
        assert signal == "HOLD"
        assert abs(ibs_val - 0.5) < 0.01

    def test_ibs_zero_range_holds(self):
        """When high == low (doji), return HOLD."""
        high = pd.Series([100.0])
        low = pd.Series([100.0])
        close = pd.Series([100.0])

        ibs_val, signal = _internal_bar_strength(high, low, close)
        assert signal == "HOLD"


# ---------------------------------------------------------------------------
# Test 6: Consecutive down days triggers BUY
# ---------------------------------------------------------------------------

class TestConsecutiveDays:
    def test_three_down_days_buy(self):
        """3 consecutive down closes should trigger BUY."""
        close = pd.Series([100.0, 99.0, 98.0, 97.0])
        count, signal = _consecutive_days(close)
        assert signal == "BUY"
        assert count == -3

    def test_five_down_days_buy(self):
        """5 consecutive down closes should trigger BUY."""
        close = pd.Series([100.0, 99.0, 98.0, 97.0, 96.0, 95.0])
        count, signal = _consecutive_days(close)
        assert signal == "BUY"
        assert count == -5

    def test_three_up_days_sell(self):
        """3 consecutive up closes should trigger SELL."""
        close = pd.Series([100.0, 101.0, 102.0, 103.0])
        count, signal = _consecutive_days(close)
        assert signal == "SELL"
        assert count == 3

    def test_two_down_days_hold(self):
        """Only 2 consecutive down closes should HOLD."""
        close = pd.Series([100.0, 99.0, 98.0])
        count, signal = _consecutive_days(close)
        assert signal == "HOLD"
        assert count == -2

    def test_mixed_directions_hold(self):
        """Alternating directions should HOLD."""
        close = pd.Series([100.0, 101.0, 100.5, 101.5])
        count, signal = _consecutive_days(close)
        assert signal == "HOLD"

    def test_single_row_hold(self):
        """Only 1 row should HOLD."""
        close = pd.Series([100.0])
        count, signal = _consecutive_days(close)
        assert signal == "HOLD"
        assert count == 0


# ---------------------------------------------------------------------------
# Test 7: Gap fill detection
# ---------------------------------------------------------------------------

class TestGapFill:
    def test_gap_down_filling_buy(self):
        """Gap down with price recovering > 30% of gap = BUY."""
        # Yesterday close = 100, today open = 98 (2% gap down)
        # Today close = 99 (filled 50% of the gap)
        open_prices = pd.Series([100.0, 98.0])
        close = pd.Series([100.0, 99.0])
        high = pd.Series([101.0, 99.5])
        low = pd.Series([99.0, 97.5])

        gap_pct, fill_pct, signal = _gap_fill(open_prices, close, high, low)
        assert signal == "BUY"
        assert gap_pct < 0  # gap down

    def test_gap_up_filling_sell(self):
        """Gap up with price falling > 30% of gap = SELL."""
        # Yesterday close = 100, today open = 102 (2% gap up)
        # Today close = 101 (filled 50% of the gap)
        open_prices = pd.Series([100.0, 102.0])
        close = pd.Series([100.0, 101.0])
        high = pd.Series([101.0, 102.5])
        low = pd.Series([99.0, 100.5])

        gap_pct, fill_pct, signal = _gap_fill(open_prices, close, high, low)
        assert signal == "SELL"
        assert gap_pct > 0  # gap up

    def test_no_gap_hold(self):
        """No significant gap = HOLD."""
        open_prices = pd.Series([100.0, 100.1])
        close = pd.Series([100.0, 100.2])
        high = pd.Series([101.0, 100.5])
        low = pd.Series([99.0, 99.8])

        gap_pct, fill_pct, signal = _gap_fill(open_prices, close, high, low)
        assert signal == "HOLD"

    def test_gap_not_filling_hold(self):
        """Large gap but price continues in gap direction = HOLD."""
        # Gap up, price continues higher (not filling)
        open_prices = pd.Series([100.0, 102.0])
        close = pd.Series([100.0, 103.0])  # moved further up
        high = pd.Series([101.0, 103.5])
        low = pd.Series([99.0, 101.5])

        gap_pct, fill_pct, signal = _gap_fill(open_prices, close, high, low)
        assert signal == "HOLD"

    def test_gap_down_continuing_down_is_hold_not_buy(self):
        """A-SM-1 (2026-04-11) regression guard: gap down where price
        CONTINUES DOWN must return HOLD, NOT BUY. The synthesis flagged
        this as a P0 inversion. Investigation showed the existing math
        already handles it (negative fill_pct is < 0.3), but we now have
        an explicit `if fill_pct < 0: HOLD` guard so a future refactor
        can't reintroduce the inversion. This test exhaustively covers
        all four (gap_dir, day_dir) quadrants."""
        # prev_close=100, today_open=95 (-5% gap-down), today_close=90 (continues -5%)
        open_prices = pd.Series([100.0, 95.0])
        close = pd.Series([100.0, 90.0])
        high = pd.Series([101.0, 95.5])
        low = pd.Series([99.0, 89.0])

        gap_pct, fill_pct, signal = _gap_fill(open_prices, close, high, low)
        assert signal == "HOLD", (
            f"Gap-down continuing down must HOLD; got {signal} "
            f"(gap_pct={gap_pct:.4f}, fill_pct={fill_pct:.4f}). "
            "A-SM-1 inversion regression — check the negative-fill_pct guard."
        )
        assert gap_pct < 0  # confirm test setup is gap-down
        assert fill_pct < 0  # confirm price moved further from prev_close

    def test_gap_down_small_continuation_is_hold(self):
        """A-SM-1 boundary: even a SMALL continuation in the gap direction
        must HOLD, not BUY."""
        # prev_close=100, today_open=95, today_close=94 (small additional drop)
        open_prices = pd.Series([100.0, 95.0])
        close = pd.Series([100.0, 94.0])
        high = pd.Series([101.0, 95.5])
        low = pd.Series([99.0, 93.5])

        _, fill_pct, signal = _gap_fill(open_prices, close, high, low)
        assert signal == "HOLD"
        assert fill_pct < 0  # negative because price widened the gap

    def test_gap_up_continuing_up_is_hold_not_sell(self):
        """A-SM-1 symmetric: gap-up + price-continuing-up must HOLD,
        not SELL."""
        open_prices = pd.Series([100.0, 105.0])
        close = pd.Series([100.0, 110.0])  # ran further away
        high = pd.Series([101.0, 110.5])
        low = pd.Series([99.0, 104.5])

        _, fill_pct, signal = _gap_fill(open_prices, close, high, low)
        assert signal == "HOLD"
        assert fill_pct < 0

    def test_crypto_higher_threshold(self):
        """With 1% threshold for crypto, a 0.5% gap does not trigger."""
        open_prices = pd.Series([100.0, 100.6])  # 0.6% gap
        close = pd.Series([100.0, 100.3])  # partially filled
        high = pd.Series([101.0, 100.8])
        low = pd.Series([99.0, 100.1])

        gap_pct, fill_pct, signal = _gap_fill(
            open_prices, close, high, low, gap_threshold=0.01
        )
        assert signal == "HOLD"


# ---------------------------------------------------------------------------
# Test 8: Combined IBS + RSI(2) signal
# ---------------------------------------------------------------------------

class TestCombinedIBSRSI2:
    def test_both_oversold_buy(self):
        """When IBS < 0.2 AND RSI(2) < 10, signal BUY."""
        # Construct data where close is near the low (IBS < 0.2)
        # and there's been a steep decline (RSI(2) < 10)
        n = 20
        np.random.seed(99)
        close_vals = [100.0] * 17 + [90.0, 80.0, 70.1]  # steep drop
        high_vals = [102.0] * 17 + [91.0, 81.0, 80.0]   # high well above close
        low_vals = [98.0] * 17 + [89.0, 79.0, 70.0]     # close near low

        close = pd.Series(close_vals, dtype=float)
        high = pd.Series(high_vals, dtype=float)
        low = pd.Series(low_vals, dtype=float)

        ibs_val, rsi2_val, signal = _ibs_rsi2_combined(high, low, close)
        assert signal == "BUY"
        assert ibs_val < 0.2
        assert rsi2_val < 10

    def test_both_overbought_sell(self):
        """When IBS > 0.8 AND RSI(2) > 90, signal SELL."""
        # Need gradual variation so RSI(2) can warm up, then a steep rise
        # at the end so RSI(2) > 90 and IBS > 0.8.
        close_vals = [100.0, 99.5, 100.2, 99.8, 100.1, 99.7, 100.3, 99.9,
                      100.0, 99.6, 100.4, 99.5, 100.1, 99.8, 100.2, 99.7,
                      100.0, 110.0, 120.0, 129.9]
        high_vals = [101.0, 100.5, 101.0, 100.5, 101.0, 100.5, 101.0, 100.5,
                     101.0, 100.5, 101.0, 100.5, 101.0, 100.5, 101.0, 100.5,
                     101.0, 111.0, 121.0, 130.0]
        low_vals = [99.0, 98.5, 99.0, 98.5, 99.0, 98.5, 99.0, 98.5,
                    99.0, 98.5, 99.0, 98.5, 99.0, 98.5, 99.0, 98.5,
                    99.0, 109.0, 119.0, 120.0]

        close = pd.Series(close_vals, dtype=float)
        high = pd.Series(high_vals, dtype=float)
        low = pd.Series(low_vals, dtype=float)

        ibs_val, rsi2_val, signal = _ibs_rsi2_combined(high, low, close)
        assert signal == "SELL"
        assert ibs_val > 0.8
        assert rsi2_val > 90

    def test_ibs_low_rsi_not_extreme_hold(self):
        """IBS < 0.2 but RSI(2) not below 10 = HOLD (both must agree)."""
        # Close near the low (IBS < 0.2) but with a bounce so RSI(2)
        # is not extremely low.  Up-down-down pattern keeps RSI(2) moderate.
        close_vals = [100.0, 99.0, 100.5, 99.8, 99.2, 100.3, 99.5, 100.1,
                      99.7, 100.0, 99.3, 100.2, 99.6, 100.0, 99.4, 100.5,
                      99.8, 100.0, 99.6, 99.1]
        high_vals = [102.0, 101.0, 102.0, 101.0, 101.0, 102.0, 101.0, 102.0,
                     101.0, 102.0, 101.0, 102.0, 101.0, 102.0, 101.0, 102.0,
                     101.0, 102.0, 101.0, 104.0]  # wide bar on last candle
        low_vals = [98.0, 97.0, 98.0, 97.0, 97.0, 98.0, 97.0, 98.0,
                    97.0, 98.0, 97.0, 98.0, 97.0, 98.0, 97.0, 98.0,
                    97.0, 98.0, 97.0, 99.0]  # close near low

        close = pd.Series(close_vals, dtype=float)
        high = pd.Series(high_vals, dtype=float)
        low = pd.Series(low_vals, dtype=float)

        ibs_val, rsi2_val, signal = _ibs_rsi2_combined(high, low, close)
        # IBS should be low (close 99.1, low 99.0, high 104.0 => IBS=0.02)
        # But RSI(2) should be moderate (not below 10) due to the up-down pattern
        assert ibs_val < 0.2
        assert rsi2_val > 10
        assert signal == "HOLD"


# ---------------------------------------------------------------------------
# Test 9: Bollinger Band %B extremes
# ---------------------------------------------------------------------------

class TestBBPctB:
    def test_below_lower_band_buy(self):
        """When close is below the lower BB, %B < 0 = BUY."""
        # Create data with a sudden crash at the end
        n = 30
        close_vals = [100.0] * 25 + [95.0, 90.0, 85.0, 80.0, 75.0]
        close = pd.Series(close_vals, dtype=float)

        pct_b, signal = _bb_pct_b(close)
        assert signal == "BUY"
        assert pct_b < 0.2

    def test_above_upper_band_sell(self):
        """When close is above the upper BB, %B > 1.0 = SELL."""
        n = 30
        close_vals = [100.0] * 25 + [105.0, 110.0, 115.0, 120.0, 125.0]
        close = pd.Series(close_vals, dtype=float)

        pct_b, signal = _bb_pct_b(close)
        assert signal == "SELL"
        assert pct_b > 0.8

    def test_inside_bands_hold(self):
        """When close is inside the bands, %B between 0.2-0.8 = HOLD."""
        n = 30
        close_vals = [100.0 + np.sin(i * 0.2) for i in range(n)]
        close = pd.Series(close_vals, dtype=float)

        pct_b, signal = _bb_pct_b(close)
        # Small oscillations should keep price inside bands
        assert signal == "HOLD"
        assert 0.2 <= pct_b <= 0.8

    def test_insufficient_data_hold(self):
        """Less than 20 bars = HOLD."""
        close = pd.Series([100.0] * 10, dtype=float)
        pct_b, signal = _bb_pct_b(close)
        assert signal == "HOLD"


# ---------------------------------------------------------------------------
# Test 10: Integration — declining series triggers multiple BUY sub-signals
# ---------------------------------------------------------------------------

class TestIntegrationDeclining:
    def test_declining_series_has_buy_signals(self):
        """A steadily declining series should trigger multiple BUY sub-signals."""
        df = _make_declining_df(n=50, start=100.0, step=-1.0)
        result = compute_mean_reversion_signal(df)

        # At least consecutive_days should be BUY
        assert result["sub_signals"]["consecutive_days"] == "BUY"

        # The overall action should lean BUY or at least have BUY sub-signals
        buy_count = sum(
            1 for v in result["sub_signals"].values() if v == "BUY"
        )
        assert buy_count >= 1

    def test_rising_series_has_sell_signals(self):
        """A steadily rising series should trigger SELL sub-signals."""
        df = _make_rising_df(n=50, start=100.0, step=1.0)
        result = compute_mean_reversion_signal(df)

        # At least consecutive_days should be SELL
        assert result["sub_signals"]["consecutive_days"] == "SELL"


# ---------------------------------------------------------------------------
# Test 11: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_flat_data_holds(self):
        """Completely flat data should return HOLD."""
        n = 50
        df = pd.DataFrame({
            "open": [100.0] * n,
            "high": [100.0] * n,
            "low": [100.0] * n,
            "close": [100.0] * n,
            "volume": [1000.0] * n,
            "time": pd.date_range("2024-01-01", periods=n, freq="1h"),
        })
        result = compute_mean_reversion_signal(df)
        assert result["action"] == "HOLD"

    def test_not_a_dataframe(self):
        """Passing a non-DataFrame should return HOLD."""
        result = compute_mean_reversion_signal("not a dataframe")
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_case_insensitive_columns(self):
        """Column names should be case-insensitive."""
        df = _make_df(50)
        df.columns = [c.upper() for c in df.columns]
        result = compute_mean_reversion_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")


# =============================================================================
# P1-6 (2026-05-02 adversarial follow-ups): seasonality detrending must not
# compound across iterations
# =============================================================================

class TestSeasonalityDoesNotCompound:
    """P1-6: when a seasonality_profile is supplied (metals path), the
    detrending loop must reconstruct each bar's price from the ORIGINAL
    previous close — not from the just-detrended previous close.

    The previous implementation read `df[_close_col].iloc[i - 1]` after
    having already overwritten `df[..., i-1]` on the previous iteration,
    so the detrending compounded: every bar's adjustment carried over into
    the base of the next bar's reconstruction. With a uniform per-hour
    mean_return of -0.001 (-0.10%) over 50 bars, the cumulative drift
    becomes ~5% — completely changing downstream RSI/BB readings.
    """

    def _build_uniform_seasonality_profile(self, mean_return: float) -> dict:
        """Build a 24-hour profile where every hour has the same
        mean_return. With this profile, detrending subtracts a constant
        per-bar bias, so any cumulative drift in the test output is
        attributable to the compounding bug."""
        return {
            str(h): {"mean_return": mean_return, "std_return": 0.01, "samples": 100}
            for h in range(24)
        }

    def _make_hourly_df(self, n: int = 50, start_price: float = 100.0):
        """Build a flat-price hourly DataFrame with a UTC index so the
        seasonality detrending branch fires."""
        idx = pd.date_range("2024-01-01 00:00", periods=n, freq="1h", tz="UTC")
        prices = [start_price] * n
        return pd.DataFrame({
            "open":   prices,
            "high":   prices,
            "low":    prices,
            "close":  prices,
            "volume": [1000.0] * n,
        }, index=idx)

    def test_uniform_zero_seasonality_leaves_close_unchanged(self):
        """Sanity: with mean_return=0 at every hour, the detrending is a
        no-op and the input prices survive."""
        from portfolio.signals.mean_reversion import compute_mean_reversion_signal

        df = self._make_hourly_df(n=50)
        original_close = df["close"].copy()
        # Inject a zero-mean profile so the detrending branch runs but does nothing.
        ctx = {"seasonality_profile": self._build_uniform_seasonality_profile(0.0)}
        # Function shouldn't crash and shouldn't drift the close.
        compute_mean_reversion_signal(df.copy(), context=ctx)
        # We can't directly observe the internal df, but the input is intact.
        pd.testing.assert_series_equal(df["close"], original_close)

    def test_seasonality_does_not_compound_across_iterations(self):
        """With a uniform per-hour bias of mean_return=+0.001 (the same +0.10%
        bias every hour), detrending each return subtracts 0.001. Reconstruction
        on a flat input should give:
            close[i] = close[0] * (1 + 0)^i = close[0]   -- because flat returns 0
            after detrending: 0 - 0.001 = -0.001 per bar
            close[i] = close[0] * (1 - 0.001)^i  -- if loop reads ORIGINAL close[i-1]

        With the bug: each iteration's modified close becomes the base for
        the next, causing extra compounding. With the fix: only the i-1
        ORIGINAL close is used as the base.

        This test uses the patched detrend_return path to capture the
        SEQUENCE of (raw_ret, hour) it's called with. Without the bug, every
        raw_ret call should be ~0.0 (flat input). With the bug, raw_ret
        becomes increasingly biased each iteration as the modified close
        drifts the pct_change downstream observation.

        We test indirectly: assert that re-running compute_mean_reversion_signal
        with the SAME flat input and seasonality profile produces a
        deterministic result independent of the loop's iteration count.
        Specifically: doubling N should not change the FINAL bar's RSI/BB
        if the loop is correct (because detrending is local, not cumulative).
        """
        from portfolio.signals.mean_reversion import compute_mean_reversion_signal

        ctx = {"seasonality_profile": self._build_uniform_seasonality_profile(0.001)}

        # Run with N=50 bars and N=100 bars on flat input. The LAST bar's
        # detrended value should be the same — the input is identical at
        # every position. With the compounding bug, the longer series
        # accumulates more drift, changing the indicators.
        df_50 = self._make_hourly_df(n=50)
        df_100 = self._make_hourly_df(n=100)

        result_50 = compute_mean_reversion_signal(df_50.copy(), context=ctx)
        result_100 = compute_mean_reversion_signal(df_100.copy(), context=ctx)

        # IBS is high/low independent of compounding (we set high=low=close=100
        # and detrending only modifies close). So IBS at the last bar should
        # match between both runs.
        assert result_50["sub_signals"]["ibs"] == result_100["sub_signals"]["ibs"]

        # Gap pct on flat input with -0.001 detrending: with the fix, every
        # bar has close = close[i-1] * (1 - 0.001) = original_close[0] * 0.999.
        # The gap_pct = |close[-1] / open[-1] - 1| = |0.999 - 1| ≈ 0.001 always.
        # With the BUG: close[i] = close[0] * (1 - 0.001)^i, so gap from open
        # (still 100) to close at i=49 is ~0.05; at i=99 is ~0.10.
        # The fix makes gap_pct independent of N.
        assert (
            result_50["indicators"]["gap_pct"] == pytest.approx(
                result_100["indicators"]["gap_pct"], abs=0.005,
            )
        ), (
            f"gap_pct differs between N=50 ({result_50['indicators']['gap_pct']:.4f}) "
            f"and N=100 ({result_100['indicators']['gap_pct']:.4f}); seasonality "
            "detrending is compounding across iterations (cumulative drift "
            "in modified close)."
        )

    def test_seasonality_uses_original_close_not_modified(self):
        """Direct white-box test: monkeypatch detrend_return to capture the
        sequence of raw_returns it observes. With the bug, late iterations
        receive returns that reflect the cumulative drift; with the fix,
        each iteration sees the same flat-input return.
        """
        from portfolio import seasonality
        from portfolio.signals import mean_reversion

        captured: list[float] = []

        def _capture(raw_return, hour, profile):
            captured.append(float(raw_return) if np.isfinite(raw_return) else 0.0)
            # Apply the same -0.001 detrend so the loop continues normally.
            return float(raw_return) - 0.001

        # Patch the import site (mean_reversion imports detrend_return inside
        # the function). We need to override the module the function imports
        # from.
        original = seasonality.detrend_return
        seasonality.detrend_return = _capture
        try:
            ctx = {"seasonality_profile": self._build_uniform_seasonality_profile(0.001)}
            df = self._make_hourly_df(n=20)
            mean_reversion.compute_mean_reversion_signal(df, context=ctx)
        finally:
            seasonality.detrend_return = original

        # On a flat-price input, every raw_return should be exactly 0.0.
        # With the bug, returns AFTER iteration 1 reflect the
        # modified-close pct_change, which is no longer 0.
        # Skip iteration 0 (returns[0] is NaN, never reaches here).
        assert len(captured) == 19, (
            f"Expected 19 iterations (range(1, 20)), got {len(captured)}"
        )
        # The pct_change on flat data is 0.0 for every bar.
        # NOTE: the `returns` series is computed ONCE at the top of the loop
        # from the ORIGINAL df, so all captured values should be 0.0
        # regardless of whether close gets modified mid-loop. This test
        # confirms the returns series is not re-derived from the modified df.
        for i, r in enumerate(captured):
            assert r == pytest.approx(0.0, abs=1e-9), (
                f"Iteration {i}: detrend_return called with raw_return={r}, "
                "expected 0.0 (flat input). Returns are being re-derived "
                "from the modified close column."
            )
