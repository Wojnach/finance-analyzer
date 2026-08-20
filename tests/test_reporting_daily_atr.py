"""reporting must surface the DAILY ATR that the loop already computes.

The "7d" horizon fetches `interval="1d"` in both TIMEFRAMES and
STOCK_TIMEFRAMES (portfolio/data_collector.py), and compute_indicators derives
`atr_pct` from it every cycle. Before 2026-08-20 reporting discarded it, leaving
Monte Carlo with only the 15-minute "Now" value. These tests pin that the daily
value is extracted and published as `daily_atr_pct`.
"""

from portfolio.reporting import DAILY_ATR_HORIZON, daily_atr_pct_from_timeframes


def _entry(atr_pct):
    return {"indicators": {"atr_pct": atr_pct}, "action": "HOLD", "confidence": 0.0}


def _tf_entry(atr_pct):
    """A timeframe entry carrying every indicator the tf_list builder reads."""
    return {
        "action": "HOLD",
        "confidence": 0.0,
        "indicators": {
            "atr_pct": atr_pct,
            "rsi": 60.0,
            "macd_hist": 1.0,
            "ema9": 1.01,
            "ema21": 0.99,
            "price_vs_bb": "inside",
        },
    }


class TestDailyAtrPctFromTimeframes:
    def test_reads_the_daily_horizon(self):
        tf = [("Now", _entry(0.54)), ("7d", _entry(2.147))]
        assert daily_atr_pct_from_timeframes(tf) == 2.147

    def test_ignores_the_intraday_now_horizon(self):
        """0.54 is the 15m value — it must never be returned."""
        tf = [("Now", _entry(0.54)), ("12h", _entry(0.99))]
        assert daily_atr_pct_from_timeframes(tf) is None

    def test_returns_none_when_daily_horizon_absent(self):
        assert daily_atr_pct_from_timeframes([("Now", _entry(0.54))]) is None

    def test_returns_none_on_a_failed_daily_fetch(self):
        tf = [("Now", _entry(0.54)), ("7d", {"error": "timeout"})]
        assert daily_atr_pct_from_timeframes(tf) is None

    def test_returns_none_for_empty_input(self):
        assert daily_atr_pct_from_timeframes([]) is None
        assert daily_atr_pct_from_timeframes(None) is None

    def test_returns_none_for_a_zero_atr(self):
        """A zero ATR is a broken fetch, not a real reading."""
        tf = [("7d", _entry(0.0))]
        assert daily_atr_pct_from_timeframes(tf) is None

    def test_daily_horizon_constant_matches_data_collector(self):
        """If TIMEFRAMES is re-labelled, this test fails instead of silently
        reverting Monte Carlo to an intraday ATR."""
        from portfolio.data_collector import STOCK_TIMEFRAMES, TIMEFRAMES

        for spec in (TIMEFRAMES, STOCK_TIMEFRAMES):
            interval = next(iv for label, iv, *_ in spec if label == DAILY_ATR_HORIZON)
            assert (
                interval == "1d"
            ), f"{DAILY_ATR_HORIZON} is no longer a daily timeframe: {interval}"


class TestDailyAtrIsPublishedInSignals:
    """The helper is worthless unless write_agent_summary actually calls it."""

    def _run(self, tf_data):
        from unittest.mock import patch

        from portfolio.reporting import write_agent_summary

        signals = {
            "BTC-USD": {
                "action": "HOLD",
                "confidence": 0.5,
                "indicators": {
                    "close": 71450.0,
                    "rsi": 78.0,
                    "macd_hist": 0.5,
                    "ema9": 72000.0,
                    "ema21": 70000.0,
                    "price_vs_bb": "inside",
                    "atr": 385.0,
                    "atr_pct": 0.54,  # the 15-MINUTE value
                },
                "extra": {"_weighted_confidence": 0.5, "_confluence_score": 0.4},
            }
        }
        state = {"cash_sek": 500000, "initial_value_sek": 500000, "holdings": {}}
        with (
            patch("portfolio.reporting.atomic_write_json"),
            patch("portfolio.reporting._write_compact_summary"),
            patch("portfolio.reporting.detect_regime", return_value="range-bound"),
            patch("portfolio.reporting.portfolio_value", return_value=500000),
            patch("portfolio.reporting.get_enhanced_signals", return_value={}),
            patch("portfolio.reporting.load_json", return_value=None),
            patch("portfolio.reporting._cached", return_value=None),
            patch(
                "portfolio.api_utils.load_config",
                return_value={"notification": {}, "monte_carlo": {"enabled": False}},
            ),
        ):
            return write_agent_summary(
                signals, {"BTC-USD": 71450.0}, 9.5, state, tf_data
            )

    def test_daily_atr_pct_is_published(self):
        tf = {
            "BTC-USD": [
                ("Now", _tf_entry(0.54)),
                ("7d", _tf_entry(2.147)),
            ]
        }
        out = self._run(tf)
        assert out["signals"]["BTC-USD"]["daily_atr_pct"] == 2.147

    def test_daily_atr_pct_is_none_when_unavailable(self):
        """Absent, not silently set to the 15m value."""
        tf = {
            "BTC-USD": [
                ("Now", _tf_entry(0.54)),
            ]
        }
        out = self._run(tf)
        assert out["signals"]["BTC-USD"].get("daily_atr_pct") is None


def test_daily_atr_pct_is_rounded():
    """Keep 3dp — one more than the display-only `atr_pct` field, since this
    value feeds the volatility model.

    Observed live 2026-08-20: MSTR published 6.001724835787434.
    """
    assert daily_atr_pct_from_timeframes([("7d", _entry(6.001724835787434))]) == 6.002
