"""Portfolio VaR must use the daily ATR too.

`compute_portfolio_var` shared the 2026-08-20 defect with simulate_ticker: it
read the 15-minute `atr_pct` and annualized it with the buggy
`sqrt(trading_days / period)` formula. The visible symptom in
`agent_summary.portfolio_var` was `drawdown_1pct_prob: 0.0` and
`drawdown_5pct_prob: 0.0` — the model asserting a 1% drawdown was impossible
on a book whose largest holding has ~34% annualized volatility.
"""

from portfolio.monte_carlo_risk import compute_portfolio_var

BTC_DAILY_ATR = 2.147
BTC_15M_ATR = 0.54
BTC_PRICE = 71_450.0


def _inputs(**sig_extra):
    sig = {"price_usd": BTC_PRICE, "action": "HOLD", "confidence": 0.0}
    sig.update(sig_extra)
    pf = {"holdings": {"BTC-USD": {"shares": 1.0}}, "cash_sek": 0}
    return pf, {"signals": {"BTC-USD": sig}}


class TestPortfolioVarUsesDailyAtr:
    def test_drawdown_probability_is_not_zero(self):
        """The headline symptom: a 1% drawdown is routine at 34% vol."""
        pf, summary = _inputs(daily_atr_pct=BTC_DAILY_ATR)
        res = compute_portfolio_var(pf, summary, n_paths=20000)
        assert res["n_positions"] == 1
        assert res["drawdown_1pct_prob"] > 0.05, res

    def test_var_is_a_material_fraction_of_exposure(self):
        """VaR95 of ~$5 on ~$1,800 of exposure was the broken output."""
        pf, summary = _inputs(daily_atr_pct=BTC_DAILY_ATR)
        res = compute_portfolio_var(pf, summary, n_paths=20000)
        var_frac = abs(res["var_95_usd"]) / res["total_exposure_usd"]
        assert var_frac > 0.005, f"VaR95 only {var_frac:.4%} of exposure"

    def test_ignores_the_intraday_atr(self):
        """A 15m ATR present alongside the daily one must not be preferred."""
        pf, summary = _inputs(
            daily_atr_pct=BTC_DAILY_ATR, extra={"atr_pct": BTC_15M_ATR}
        )
        res = compute_portfolio_var(pf, summary, n_paths=20000)
        assert res["drawdown_1pct_prob"] > 0.05

    def test_falls_back_to_daily_scale_default(self):
        """With only the 15m value present, use the daily class default."""
        pf, summary = _inputs(extra={"atr_pct": BTC_15M_ATR})
        res = compute_portfolio_var(pf, summary, n_paths=20000)
        assert res["drawdown_1pct_prob"] > 0.05
