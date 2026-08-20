"""Monte Carlo volatility must reproduce realized volatility.

Regression tests for the 2026-08-20 defect: `volatility_from_atr` annualized a
one-bar ATR as if ATR(14) spanned 14 bars (`sqrt(trading_days / period)`),
understating volatility by sqrt(14) = 3.74x, and was additionally fed a
15-minute ATR where a daily one was assumed. Combined, both BTC and ETH reported
`volatility_annual = 0.05` — the MIN_VOLATILITY floor — against realized 33% and
55%. That made `p_stop_hit = 0.0` and `drawdown_1pct_prob = 0.0`.

The reference numbers below were measured on 2026-08-20 from Binance daily bars
(ATR(14) as a % of last close, and annualized stdev of 60d daily log returns).
They are fixtures, not live calls, so these tests stay deterministic.
"""

import math

import pytest

from portfolio.monte_carlo import (
    MIN_VOLATILITY,
    annualized_vol_from_atr,
    trading_days_for_ticker,
)

# (ticker, daily ATR(14) %, realized annualized vol as a decimal)
MEASURED = [
    ("BTC-USD", 2.147, 0.332),
    ("ETH-USD", 2.945, 0.549),
    ("XAU-USD", 1.690, 0.259),
    ("XAG-USD", 3.141, 0.452),
]


class TestAnnualizedVolFromAtr:
    @pytest.mark.parametrize("ticker,daily_atr_pct,realized", MEASURED)
    def test_reproduces_realized_vol_within_25pct(
        self, ticker, daily_atr_pct, realized
    ):
        """An ATR-derived vol is a proxy, but it must land in the right ballpark."""
        vol = annualized_vol_from_atr(
            daily_atr_pct, trading_days=trading_days_for_ticker(ticker)
        )
        ratio = vol / realized
        assert 0.75 <= ratio <= 1.25, (
            f"{ticker}: daily ATR {daily_atr_pct}% -> vol {vol:.3f} "
            f"vs realized {realized:.3f} (ratio {ratio:.2f})"
        )

    @pytest.mark.parametrize("ticker,daily_atr_pct,realized", MEASURED)
    def test_never_returns_the_floor_for_a_real_asset(
        self, ticker, daily_atr_pct, realized
    ):
        """The floor masked the bug. No real Tier-1 asset may land on it."""
        vol = annualized_vol_from_atr(
            daily_atr_pct, trading_days=trading_days_for_ticker(ticker)
        )
        assert (
            vol > MIN_VOLATILITY * 2
        ), f"{ticker} collapsed onto the MIN_VOLATILITY floor: {vol}"

    def test_does_not_divide_by_the_atr_period(self):
        """The sqrt(days/period) bug. A 2% daily ATR is ~30% vol, not ~8%."""
        vol = annualized_vol_from_atr(2.0, trading_days=365)
        assert 0.25 <= vol <= 0.36, vol

    def test_scales_as_sqrt_of_trading_days(self):
        """Doubling the periods per year scales vol by sqrt(2)."""
        a = annualized_vol_from_atr(2.0, trading_days=100)
        b = annualized_vol_from_atr(2.0, trading_days=200)
        assert b / a == pytest.approx(math.sqrt(2), rel=1e-6)

    def test_scales_linearly_in_atr(self):
        a = annualized_vol_from_atr(1.0, trading_days=365)
        b = annualized_vol_from_atr(3.0, trading_days=365)
        assert b / a == pytest.approx(3.0, rel=1e-6)

    def test_stocks_use_252_days(self):
        assert trading_days_for_ticker("MSTR") == 252
        stock = annualized_vol_from_atr(2.0, trading_days=252)
        crypto = annualized_vol_from_atr(2.0, trading_days=365)
        assert stock < crypto

    def test_floor_still_applies_to_a_degenerate_input(self):
        """A zero/absurd ATR must not produce a zero-vol degenerate simulation."""
        assert annualized_vol_from_atr(0.0, trading_days=365) == MIN_VOLATILITY

    def test_rejects_an_intraday_atr_masquerading_as_daily(self):
        """15m ATR is ~0.5% on BTC. Scaled as daily it yields nonsense-low vol.

        The function cannot detect the caller's bar interval, so the guard is
        that such an input lands on the floor rather than quietly returning a
        plausible-looking number. Callers must supply a DAILY ATR.
        """
        vol = annualized_vol_from_atr(0.54, trading_days=365)
        assert vol < 0.15, (
            "a 15m ATR passed as daily should look obviously wrong, "
            f"got {vol:.3f} which is plausible enough to hide the mistake"
        )
