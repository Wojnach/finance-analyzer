"""simulate_ticker must consume a DAILY ATR, never the 15-minute one.

Second half of the 2026-08-20 defect. `portfolio/main.py:546` builds the "Now"
horizon from `interval="15m"`, so `signals[tkr]["atr_pct"]` and
`signals[tkr]["extra"]["atr_pct"]` are 15-MINUTE ATR(14) values (BTC ~0.54%).
`simulate_ticker` consumed them as if they were daily, which both collapsed
volatility onto MIN_VOLATILITY and put the "2x ATR" stop ~1.1% below spot
instead of ~4.3%.

The daily ATR is already computed every cycle — the "7d" horizon fetches
`interval="1d"` (data_collector.TIMEFRAMES) and compute_indicators derives
atr_pct from it — reporting simply discarded it. These tests pin the contract
that simulate_ticker reads the daily value and ignores the intraday one.
"""

import pytest

from portfolio.monte_carlo import MIN_VOLATILITY, simulate_ticker

BTC_DAILY_ATR = 2.147  # measured 2026-08-20 on Binance daily bars
BTC_15M_ATR = 0.54  # what the "Now" horizon actually reports
BTC_PRICE = 71_450.0


def _summary(**signal_overrides):
    sig = {"price_usd": BTC_PRICE, "action": "HOLD", "confidence": 0.0}
    sig.update(signal_overrides)
    return {"signals": {"BTC-USD": sig}}


class TestDailyAtrIsUsed:
    def test_prefers_daily_atr_over_the_intraday_one(self):
        """Both present: the daily value must win."""
        res = simulate_ticker(
            "BTC-USD",
            _summary(daily_atr_pct=BTC_DAILY_ATR, atr_pct=BTC_15M_ATR),
            n_paths=2000,
            seed=1,
        )
        assert res["atr_pct"] == pytest.approx(BTC_DAILY_ATR, abs=0.01)
        # 2.147% daily ATR / 1.2 * sqrt(365) ~= 0.342
        assert 0.28 <= res["volatility_annual"] <= 0.40, res["volatility_annual"]

    def test_ignores_intraday_atr_in_extra(self):
        res = simulate_ticker(
            "BTC-USD",
            _summary(daily_atr_pct=BTC_DAILY_ATR, extra={"atr_pct": BTC_15M_ATR}),
            n_paths=2000,
            seed=1,
        )
        assert res["atr_pct"] == pytest.approx(BTC_DAILY_ATR, abs=0.01)

    def test_falls_back_to_daily_class_default_not_the_intraday_value(self):
        """No daily ATR available: use the daily-scale class default (3.5% crypto).

        Falling back to the 15m value would silently reintroduce the bug.
        """
        res = simulate_ticker(
            "BTC-USD",
            _summary(atr_pct=BTC_15M_ATR, extra={"atr_pct": BTC_15M_ATR}),
            n_paths=2000,
            seed=1,
        )
        assert res["atr_pct"] == pytest.approx(3.5, abs=0.01)
        assert res["volatility_annual"] > MIN_VOLATILITY * 3


class TestObservableSymptomsAreGone:
    @pytest.fixture
    def res(self):
        return simulate_ticker(
            "BTC-USD",
            _summary(daily_atr_pct=BTC_DAILY_ATR),
            n_paths=20000,
            seed=7,
        )

    def test_volatility_is_not_the_floor(self, res):
        assert res["volatility_annual"] > MIN_VOLATILITY * 2

    def test_stop_sits_a_realistic_distance_below_spot(self, res):
        """2x a 2.147% daily ATR is ~4.3%, not the ~1.1% the 15m value gave."""
        drop_pct = 100 * (1 - res["stop_price"] / BTC_PRICE)
        assert 3.5 <= drop_pct <= 5.0, drop_pct

    def test_stop_hit_probability_is_not_zero(self, res):
        """`p_stop_hit_1d = 0.0` was the headline symptom."""
        assert res["p_stop_hit_1d"] > 0.0

    def test_one_day_price_band_is_realistically_wide(self, res):
        """The p5-p95 band was +-0.4%; at 34% vol it should be several percent."""
        b = res["price_bands_1d"]
        spread_pct = 100 * (b[95] - b[5]) / BTC_PRICE
        assert spread_pct > 2.0, spread_pct

    def test_expected_return_std_reflects_daily_vol(self, res):
        """34% annualized is ~1.8% per day."""
        assert 1.0 <= res["expected_return_1d"]["std_pct"] <= 3.0
