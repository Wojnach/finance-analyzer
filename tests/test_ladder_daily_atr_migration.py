"""Ladder / fishing volatility must come from a DAILY ATR.

Second stage of the 2026-08-20 Monte Carlo volatility fix. `simulate_ticker`
and `compute_portfolio_var` were migrated first because they place no orders.
The callers migrated here DO size real Avanza limit ladders and stop levels:

    price_targets.compute_targets   <- metals_ladder, metals_execution_engine
    fin_fish._compute_vol_and_drift
    metals_execution_engine._warrant_vol_from_underlying

All of them were passing `signals[tkr]["atr_pct"]`, which is a 15-MINUTE ATR
(main.py:546 builds the "Now" horizon from interval="15m"), into
`volatility_from_atr`, whose sqrt(days/period) factor understates by a further
3.74x. Combined, XAG produced a 5.0% annualized vol (the MIN_VOLATILITY floor)
against ~58% realized, so every rung sat ~2x too close to spot for its own
stated fill probability.

Reference values measured live 2026-08-20:
    XAG-USD  15m ATR 0.32%   daily ATR 3.635%   spot ~68.20
    XAU-USD  15m ATR 0.16%   daily ATR 1.923%   spot ~4525.29
"""

import pytest

from portfolio.monte_carlo import MIN_VOLATILITY, annualized_vol_from_atr
from portfolio.price_targets import compute_targets

XAG_SPOT = 68.20
XAG_DAILY_ATR = 3.635
XAG_15M_ATR = 0.32


def _spread_pct(result, spot):
    """p10..p90 width of the running-extremes distribution, as % of spot."""
    e = result["extremes"]
    return 100.0 * (e["p90"] - e["p10"]) / spot


class TestComputeTargetsUsesDailyAtr:
    def test_extremes_widen_to_match_daily_vol(self):
        """At 3.6% daily ATR over 6h the p10-p90 band is ~1%+, not ~0.1%."""
        r = compute_targets(
            "XAG-USD",
            side="buy",
            price_usd=XAG_SPOT,
            atr_pct=XAG_DAILY_ATR,
            p_up=0.432,
            hours_remaining=6.0,
            n_paths=8000,
        )
        assert _spread_pct(r, XAG_SPOT) > 1.0, _spread_pct(r, XAG_SPOT)

    def test_daily_atr_no_longer_hits_the_floor(self):
        """The old formula floored XAG at MIN_VOLATILITY. Sanity-check the
        function the migration points at."""
        vol = annualized_vol_from_atr(XAG_DAILY_ATR, trading_days=365)
        assert vol > MIN_VOLATILITY * 5

    def test_buy_rung_sits_deeper_than_the_old_placement(self):
        """Same fill probability should now require a deeper limit.

        Measured: XAG buy recommendation moved -0.15% -> -0.31% from spot.
        """
        r = compute_targets(
            "XAG-USD",
            side="buy",
            price_usd=XAG_SPOT,
            atr_pct=XAG_DAILY_ATR,
            p_up=0.432,
            hours_remaining=6.0,
            n_paths=8000,
        )
        rec = r["recommended"]
        assert rec is not None
        depth_pct = 100.0 * (1 - rec["price"] / XAG_SPOT)
        assert depth_pct > 0.2, f"buy rung only {depth_pct:.3f}% below spot"

    def test_zero_atr_still_short_circuits(self):
        r = compute_targets(
            "XAG-USD",
            side="buy",
            price_usd=XAG_SPOT,
            atr_pct=0.0,
            p_up=0.5,
            hours_remaining=6.0,
        )
        assert r["targets"] == []
        assert r["recommended"] is None


class TestComputeAllTargetsReadsDailyField:
    def test_prefers_daily_atr_pct_over_intraday(self):
        """compute_all_targets must read signals[tkr].daily_atr_pct."""
        from portfolio.price_targets import compute_all_targets

        summary = {
            "signals": {
                "XAG-USD": {
                    "price_usd": XAG_SPOT,
                    "daily_atr_pct": XAG_DAILY_ATR,
                    "atr_pct": XAG_15M_ATR,
                    "action": "BUY",  # gets the ticker into the task list
                    "regime": "ranging",
                    "extra": {},
                }
            },
            "focus_probabilities": {"XAG-USD": {"1d": {"probability": 0.45}}},
        }
        out = compute_all_targets(summary, {}, {"default_hours": 6, "n_paths": 8000})
        res = out.get("XAG-USD")
        assert res, out
        assert _spread_pct(res, XAG_SPOT) > 1.0, _spread_pct(res, XAG_SPOT)


class TestFinFishExposesDailyAtr:
    def test_load_signal_data_surfaces_daily_atr_pct(self, tmp_path, monkeypatch):
        import json

        from portfolio import fin_fish

        summary = {
            "signals": {
                "XAG-USD": {
                    "price_usd": XAG_SPOT,
                    "rsi": 50.0,
                    "atr_pct": XAG_15M_ATR,
                    "daily_atr_pct": XAG_DAILY_ATR,
                    "regime": "ranging",
                    "action": "HOLD",
                    "extra": {},
                }
            }
        }
        p = tmp_path / "agent_summary.json"
        p.write_text(json.dumps(summary))
        monkeypatch.setattr(fin_fish, "SUMMARY_PATH", str(p))

        sig = fin_fish.load_signal_data("XAG-USD")
        assert sig["daily_atr_pct"] == XAG_DAILY_ATR

    def test_vol_falls_back_to_daily_atr_not_the_floor(self):
        """With no daily_ranges the ATR path is the only source of vol.

        It used to floor at 5% for metals; it must now reflect the daily ATR.
        """
        from portfolio.fin_fish import _compute_vol_and_drift

        signal = {
            "atr_pct": XAG_15M_ATR,
            "daily_atr_pct": XAG_DAILY_ATR,
            "focus": {"3h": {"probability": 0.45}},
            "entry": {},
        }
        vol, _drift = _compute_vol_and_drift(signal, [], "LONG")
        assert vol > 0.30, vol


class TestWarrantVolUsesDailyAtr:
    def test_leveraged_warrant_vol_scales_from_a_daily_atr(self):
        import importlib.util
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "_mee", root / "data" / "metals_execution_engine.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # 5x warrant on a 3.635% daily ATR underlying: vol must be large.
        vol = mod._warrant_vol_from_underlying(XAG_DAILY_ATR, 5.0)
        assert vol > 1.0, vol


class TestFlashReserveDoesNotDuplicateWorkingRung:
    """Correct (wider) vol can push the normal rung below the flash-crash level.

    `flash_underlying = min(working_underlying, spot*(1-flash_drop))`. Once the
    daily-ATR vol made `extremes.p25` deeper than the historical post-open
    drop, that min() returned `working_underlying` and the ladder emitted
    flash_price == working_price — i.e. two identical live orders.
    """

    def _ladder(self, daily_atr_pct):
        from portfolio.metals_ladder import build_intraday_ladder

        entry = {
            "price_usd": 86.9,
            "daily_atr_pct": daily_atr_pct,
            "atr_pct": 0.32,
            "regime": "ranging",
            "action": "BUY",
            "rsi": 50.0,
            "extra": {},
        }
        analysis = {
            "us_market_open": {
                "phase": "pre_open",
                "historical_stats": {
                    "post_open_mean_pct": -0.692,
                    "post_open_avg_range_pct": 3.537,
                },
            }
        }
        return build_intraday_ladder(
            entry,
            {"3h": {"probability": 0.55}},
            ticker="XAG-USD",
            current_instrument_price=12.7,
            current_underlying_price=86.9,
            leverage=6.3,
            hours_remaining=6.0,
            analysis=analysis,
        )

    @pytest.mark.parametrize("daily_atr_pct", [0.5, 2.0, 3.635, 8.0])
    def test_flash_rung_is_never_equal_to_the_working_rung(self, daily_atr_pct):
        lad = self._ladder(daily_atr_pct)
        if lad["flash_price"] > 0:
            assert lad["flash_price"] < lad["working_price"], (
                f"daily_atr={daily_atr_pct}: flash {lad['flash_price']} "
                f"== working {lad['working_price']} would place two identical orders"
            )

    def test_flash_reserve_disabled_when_working_rung_already_deeper(self):
        """A wide-vol ladder already fishing below the flash level needs no
        second rung — it must be switched off, not duplicated."""
        lad = self._ladder(8.0)
        assert lad["flash_price"] == 0.0
        assert lad["flash_underlying"] == 0.0
