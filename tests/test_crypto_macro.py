"""Tests for crypto macro signal and data module."""

from unittest.mock import patch

import pandas as pd

# ---------------------------------------------------------------------------
# crypto_macro_data tests
# ---------------------------------------------------------------------------

class TestDeribitOptions:
    """Test Deribit options data parsing and max pain calculation."""

    def _make_book_summary(self, items):
        """Helper to create mock Deribit API response."""
        return {"result": items}

    def test_max_pain_simple(self):
        """Max pain should be the strike where option buyer losses are minimized."""
        from portfolio.crypto_macro_data import _fetch_deribit_options

        # Create simple options: calls at 70K and 75K, puts at 65K and 70K
        items = [
            {"instrument_name": "BTC-28MAR26-65000-P", "open_interest": 100},
            {"instrument_name": "BTC-28MAR26-70000-C", "open_interest": 200},
            {"instrument_name": "BTC-28MAR26-70000-P", "open_interest": 150},
            {"instrument_name": "BTC-28MAR26-75000-C", "open_interest": 100},
        ]

        with patch("portfolio.crypto_macro_data.fetch_json") as mock_fetch:
            mock_fetch.return_value = self._make_book_summary(items)
            result = _fetch_deribit_options("BTC")

        assert result is not None
        assert "max_pain" in result
        assert result["max_pain"] in (65000, 70000, 75000)
        assert result["nearest_pcr"] is not None
        assert result["total_pcr"] is not None

    def test_put_call_ratio_calculation(self):
        """Put/call ratio should be puts_oi / calls_oi."""
        from portfolio.crypto_macro_data import _fetch_deribit_options

        items = [
            {"instrument_name": "BTC-28MAR26-70000-C", "open_interest": 100},
            {"instrument_name": "BTC-28MAR26-70000-P", "open_interest": 120},
        ]

        with patch("portfolio.crypto_macro_data.fetch_json") as mock_fetch:
            mock_fetch.return_value = self._make_book_summary(items)
            result = _fetch_deribit_options("BTC")

        assert result is not None
        assert result["nearest_pcr"] == 1.2  # 120/100

    def test_empty_response(self):
        """Empty API response should return None."""
        from portfolio.crypto_macro_data import _fetch_deribit_options

        with patch("portfolio.crypto_macro_data.fetch_json") as mock_fetch:
            mock_fetch.return_value = None
            result = _fetch_deribit_options("BTC")

        assert result is None

    def test_no_open_interest(self):
        """Options with zero OI should be skipped."""
        from portfolio.crypto_macro_data import _fetch_deribit_options

        items = [
            {"instrument_name": "BTC-28MAR26-70000-C", "open_interest": 0},
            {"instrument_name": "BTC-28MAR26-70000-P", "open_interest": 0},
        ]

        with patch("portfolio.crypto_macro_data.fetch_json") as mock_fetch:
            mock_fetch.return_value = self._make_book_summary(items)
            result = _fetch_deribit_options("BTC")

        assert result is None


class TestGoldBtcRatio:
    """Test gold-BTC ratio computation."""

    def test_ratio_computed(self):
        """Ratio should be gold_price / btc_price."""
        from portfolio.crypto_macro_data import compute_gold_btc_ratio

        mock_summary = {
            "signals": {
                "BTC-USD": {"price_usd": 70000},
                "XAU-USD": {"price_usd": 4400},
            }
        }

        with patch("portfolio.file_utils.load_json", return_value=mock_summary), \
             patch("portfolio.crypto_macro_data._load_ratio_history", return_value=[]), \
             patch("portfolio.crypto_macro_data._append_ratio_history"):
            result = compute_gold_btc_ratio()

        assert result is not None
        assert abs(result["gold_btc_ratio"] - 4400 / 70000) < 0.0001
        assert result["trend"] == "flat"  # no history

    def test_missing_prices(self):
        """Should return None if prices are missing."""
        from portfolio.crypto_macro_data import compute_gold_btc_ratio

        mock_summary = {"signals": {"BTC-USD": {"price_usd": 0}}}

        with patch("portfolio.file_utils.load_json", return_value=mock_summary):
            result = compute_gold_btc_ratio()

        assert result is None


# ---------------------------------------------------------------------------
# crypto_macro signal tests
# ---------------------------------------------------------------------------

class TestCryptoMacroSignal:
    """Test the crypto macro composite signal."""

    def _make_df(self, close=70000):
        """Create minimal OHLCV DataFrame."""
        return pd.DataFrame({
            "open": [close] * 10,
            "high": [close * 1.01] * 10,
            "low": [close * 0.99] * 10,
            "close": [close] * 10,
            "volume": [100] * 10,
        })

    def test_non_crypto_returns_hold(self):
        """Non-crypto tickers should immediately return HOLD."""
        from portfolio.signals.crypto_macro import compute_crypto_macro_signal

        result = compute_crypto_macro_signal(
            self._make_df(135),
            context={"ticker": "MSTR"}
        )
        assert result["action"] == "HOLD"
        assert result["indicators"].get("skip_reason") == "non_crypto"

    def test_btc_returns_signal(self):
        """BTC should get a signal (may be HOLD if data unavailable)."""
        from portfolio.signals.crypto_macro import compute_crypto_macro_signal

        with patch("portfolio.signals.crypto_macro._cached", return_value=None):
            result = compute_crypto_macro_signal(
                self._make_df(70000),
                context={"ticker": "BTC-USD"}
            )

        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_options_gravity_below_max_pain(self):
        """Price well below max pain should signal BUY."""
        from portfolio.signals.crypto_macro import _options_gravity

        options = {"max_pain": 75000, "days_to_expiry": 3}
        action, ind = _options_gravity(options, 70000)
        assert action == "BUY"  # price 6.7% below max pain

    def test_options_gravity_above_max_pain(self):
        """Price well above max pain should signal SELL."""
        from portfolio.signals.crypto_macro import _options_gravity

        options = {"max_pain": 65000, "days_to_expiry": 3}
        action, ind = _options_gravity(options, 70000)
        assert action == "SELL"  # price 7.7% above max pain

    def test_options_gravity_near_max_pain(self):
        """Price near max pain should be HOLD."""
        from portfolio.signals.crypto_macro import _options_gravity

        options = {"max_pain": 70000, "days_to_expiry": 3}
        action, ind = _options_gravity(options, 70500)
        assert action == "HOLD"  # price 0.7% from max pain

    def test_options_gravity_far_expiry(self):
        """Far from expiry, gravity effect should be HOLD."""
        from portfolio.signals.crypto_macro import _options_gravity

        options = {"max_pain": 75000, "days_to_expiry": 14}
        action, ind = _options_gravity(options, 70000)
        assert action == "HOLD"  # too far from expiry

    def test_pcr_high_contrarian_buy(self):
        """High put/call ratio should be contrarian BUY."""
        from portfolio.signals.crypto_macro import _put_call_sentiment

        options = {"nearest_pcr": 1.5, "total_pcr": 1.3}
        action, ind = _put_call_sentiment(options)
        assert action == "BUY"

    def test_pcr_low_contrarian_sell(self):
        """Low put/call ratio should be contrarian SELL."""
        from portfolio.signals.crypto_macro import _put_call_sentiment

        options = {"nearest_pcr": 0.4, "total_pcr": 0.5}
        action, ind = _put_call_sentiment(options)
        assert action == "SELL"

    def test_gold_rotation_btc_outperforming(self):
        """BTC outperforming gold should signal BUY."""
        from portfolio.signals.crypto_macro import _gold_rotation

        data = {"trend": "btc_outperforming", "gold_btc_ratio": 0.06}
        action, ind = _gold_rotation(data)
        assert action == "BUY"

    def test_gold_rotation_gold_outperforming(self):
        """Gold outperforming BTC should signal SELL."""
        from portfolio.signals.crypto_macro import _gold_rotation

        data = {"trend": "gold_outperforming", "gold_btc_ratio": 0.07}
        action, ind = _gold_rotation(data)
        assert action == "SELL"

    def test_netflow_accumulation(self):
        """Strong accumulation should signal BUY."""
        from portfolio.signals.crypto_macro import _exchange_netflow_signal

        data = {
            "trend": "strong_accumulation",
            "consecutive_negative": 7,
            "sum_7d": -5000,
        }
        action, ind = _exchange_netflow_signal(data)
        assert action == "BUY"

    def test_netflow_distribution(self):
        """Strong distribution should signal SELL."""
        from portfolio.signals.crypto_macro import _exchange_netflow_signal

        data = {
            "trend": "strong_distribution",
            "consecutive_negative": 0,
            "sum_7d": 5000,
        }
        action, ind = _exchange_netflow_signal(data)
        assert action == "SELL"

    # 2026-06-10 (audit batch 2): expiry-proximity rewritten. The relief BUY
    # only fires for genuine quarterly expiries with meaningful OI share —
    # Deribit's daily expiries used to make this a permanent BUY vote.

    def test_quarterly_expiry_day_with_oi_buys(self):
        """Day of a genuine quarterly expiry with meaningful OI = relief BUY."""
        from portfolio.signals.crypto_macro import _expiry_proximity

        options = {
            "nearest_expiry_days": 0,
            "nearest_expiry": "27MAR26",
            "nearest_is_quarterly": True,
            "nearest_expiry_oi_share": 0.35,
        }
        action, ind = _expiry_proximity(options)
        assert action == "BUY"
        assert ind["is_quarterly"] is True

    def test_daily_expiry_holds(self):
        """A 0-1 DTE daily expiry must vote HOLD, not BUY (June collapse fix)."""
        from portfolio.signals.crypto_macro import _expiry_proximity

        options = {
            "nearest_expiry_days": 1,
            "nearest_expiry": "11JUN26",  # mid-month June daily — NOT quarterly
            "nearest_is_quarterly": False,
            "nearest_expiry_oi_share": 0.01,
        }
        action, ind = _expiry_proximity(options)
        assert action == "HOLD"
        assert ind["is_quarterly"] is False
        assert ind.get("expiry_risk_flag") is True

    def test_quarterly_without_oi_evidence_holds(self):
        """Quarterly expiry but thin/unknown OI share = HOLD (no BUY without evidence)."""
        from portfolio.signals.crypto_macro import _expiry_proximity

        # Thin OI share
        action, _ = _expiry_proximity({
            "nearest_expiry_days": 0,
            "nearest_expiry": "26JUN26",
            "nearest_is_quarterly": True,
            "nearest_expiry_oi_share": 0.02,
        })
        assert action == "HOLD"

        # Missing OI share (legacy cached dict)
        action, _ = _expiry_proximity({
            "days_to_expiry": 0,
            "nearest_expiry": "26JUN26",
        })
        assert action == "HOLD"

    def test_legacy_options_dict_quarterly_fallback(self):
        """Legacy dicts (pre-fix cache) classify quarterliness from the string."""
        from portfolio.signals.crypto_macro import _is_quarterly_expiry

        assert _is_quarterly_expiry("26JUN26") is True   # last week of June
        assert _is_quarterly_expiry("27MAR26") is True   # last week of March
        assert _is_quarterly_expiry("11JUN26") is False  # June daily, day < 22
        assert _is_quarterly_expiry("24APR26") is False  # non-quarter month
        assert _is_quarterly_expiry("") is False
        assert _is_quarterly_expiry("garbage") is False

    def test_quarterly_far_from_expiry_holds(self):
        """Quarterly expiry >1 day out should not BUY (warning window only)."""
        from portfolio.signals.crypto_macro import _expiry_proximity

        options = {
            "nearest_expiry_days": 3,
            "nearest_expiry": "26JUN26",
            "nearest_is_quarterly": True,
            "nearest_expiry_oi_share": 0.35,
        }
        action, ind = _expiry_proximity(options)
        assert action == "HOLD"
        assert ind.get("expiry_volatility_warning") is True


# ---------------------------------------------------------------------------
# 2026-06-10 (audit batch 2): options-chain expiry filter tests
# ---------------------------------------------------------------------------

class TestMetricsChainSelection:
    """Max pain / PCR / OI metrics must come from the monthly/quarterly chain,
    not the thin 0-1 DTE daily expiry Deribit also lists."""

    @staticmethod
    def _fmt(d):
        return d.strftime("%d%b%y").upper()

    def _dates(self):
        import datetime

        today = datetime.date.today()
        # Daily expiry: nearest future date with day-of-month < 22, so it can
        # never classify as monthly-like regardless of when the test runs.
        daily = today + datetime.timedelta(days=1)
        while daily.day >= 22:
            daily += datetime.timedelta(days=1)
        # Monthly-like expiry: first day-26 date strictly after the daily.
        monthly = daily + datetime.timedelta(days=1)
        while monthly.day != 26:
            monthly += datetime.timedelta(days=1)
        return today, daily, monthly

    def test_metrics_use_monthly_chain_not_daily(self):
        from portfolio.crypto_macro_data import _fetch_deribit_options

        today, daily, monthly = self._dates()
        d_str, m_str = self._fmt(daily), self._fmt(monthly)

        items = [
            # Thin daily chain (the raw nearest expiry)
            {"instrument_name": f"BTC-{d_str}-60000-C", "open_interest": 5},
            {"instrument_name": f"BTC-{d_str}-60000-P", "open_interest": 5},
            # Liquid monthly chain
            {"instrument_name": f"BTC-{m_str}-65000-C", "open_interest": 400},
            {"instrument_name": f"BTC-{m_str}-65000-P", "open_interest": 600},
        ]

        with patch("portfolio.crypto_macro_data.fetch_json") as mock_fetch:
            mock_fetch.return_value = {"result": items}
            result = _fetch_deribit_options("BTC")

        assert result is not None
        # Metrics chain = monthly
        assert result["metrics_expiry"] == m_str
        assert result["days_to_expiry"] == (monthly - today).days
        assert result["nearest_pcr"] == 1.5  # 600/400 from the monthly chain
        assert result["max_pain"] == 65000
        # Raw nearest expiry preserved for the expiry-proximity sub-signal
        assert result["nearest_expiry"] == d_str
        assert result["nearest_expiry_days"] == (daily - today).days
        assert result["nearest_is_quarterly"] is False  # day < 22 by construction
        assert result["nearest_expiry_oi_share"] is not None
        assert result["nearest_expiry_oi_share"] < 0.1  # 10/1010

    def test_no_monthly_listing_falls_back_to_most_liquid(self):
        from portfolio.crypto_macro_data import _fetch_deribit_options

        _, daily, _ = self._dates()
        d_str = self._fmt(daily)

        items = [
            {"instrument_name": f"BTC-{d_str}-60000-C", "open_interest": 50},
            {"instrument_name": f"BTC-{d_str}-60000-P", "open_interest": 60},
        ]

        with patch("portfolio.crypto_macro_data.fetch_json") as mock_fetch:
            mock_fetch.return_value = {"result": items}
            result = _fetch_deribit_options("BTC")

        assert result is not None
        assert result["metrics_expiry"] == d_str
        assert result["nearest_pcr"] == 1.2


# ---------------------------------------------------------------------------
# 2026-06-10 (audit batch 2): consecutive-negative netflow day-unit tests
# ---------------------------------------------------------------------------

class TestNetflowDayUnits:
    """consecutive_negative must count calendar DAYS, not 6h samples."""

    def _trend(self, history, monkeypatch, tmp_path):
        import portfolio.crypto_macro_data as cmd

        monkeypatch.setattr(cmd, "_NETFLOW_STALE_STATE_FILE",
                            tmp_path / "netflow_stale_state.json")
        with patch("portfolio.onchain_data.get_onchain_data", return_value=None), \
             patch("portfolio.crypto_macro_data._load_netflow_history",
                   return_value=history):
            return cmd.get_exchange_netflow_trend()

    def test_six_hour_samples_collapse_to_days(self, monkeypatch, tmp_path):
        import time as _time

        now = _time.time()
        # 8 negative samples at 6h cadence = 2 calendar days, not 8 "days".
        # Anchor to midday UTC so the 6h steps stay within their UTC day.
        day = 86400
        midday = (int(now) // day) * day - day + 43200  # yesterday 12:00 UTC
        history = []
        for d in range(2):
            for k in range(4):
                history.append({"ts": midday - d * day - k * 21600 + 32400,
                                "netflow": -50.0})
        history.sort(key=lambda e: e["ts"])

        result = self._trend(history, monkeypatch, tmp_path)
        assert result is not None
        # Old (buggy) behavior: 8. Fixed: number of distinct UTC days.
        assert result["consecutive_negative"] <= 3
        assert result["consecutive_negative"] >= 2

    def test_calendar_gap_breaks_streak(self, monkeypatch, tmp_path):
        import time as _time

        now = _time.time()
        day = 86400
        midday = (int(now) // day) * day - day + 43200
        history = [
            {"ts": midday - 5 * day, "netflow": -10.0},  # before the gap
            {"ts": midday - day, "netflow": -20.0},
            {"ts": midday, "netflow": -30.0},
        ]
        result = self._trend(history, monkeypatch, tmp_path)
        assert result is not None
        # The 4-day hole between samples breaks the streak at 2.
        assert result["consecutive_negative"] == 2

    def test_positive_day_breaks_streak(self, monkeypatch, tmp_path):
        import time as _time

        now = _time.time()
        day = 86400
        midday = (int(now) // day) * day - day + 43200
        history = [
            {"ts": midday - 2 * day, "netflow": -10.0},
            {"ts": midday - day, "netflow": 5.0},
            {"ts": midday, "netflow": -30.0},
        ]
        result = self._trend(history, monkeypatch, tmp_path)
        assert result is not None
        assert result["consecutive_negative"] == 1


# ---------------------------------------------------------------------------
# 2026-06-10 (audit batch 2): netflow staleness alert tests
# ---------------------------------------------------------------------------

class TestNetflowStalenessAlert:
    """exchange_netflow_history.jsonl older than 7d must raise a
    critical_errors row (category data_feed_stale), once per window."""

    def _run(self, history, monkeypatch, tmp_path):
        import portfolio.crypto_macro_data as cmd

        monkeypatch.setattr(cmd, "_NETFLOW_STALE_STATE_FILE",
                            tmp_path / "netflow_stale_state.json")
        with patch("portfolio.onchain_data.get_onchain_data", return_value=None), \
             patch("portfolio.crypto_macro_data._load_netflow_history",
                   return_value=history), \
             patch("portfolio.claude_gate.record_critical_error",
                   return_value=True) as mock_rce:
            result = cmd.get_exchange_netflow_trend()
        return result, mock_rce

    def test_stale_history_alerts(self, monkeypatch, tmp_path):
        import time as _time

        # One entry, 60 days old — the live Apr-10 failure mode.
        history = [{"ts": _time.time() - 60 * 86400, "netflow": -100.0}]
        result, mock_rce = self._run(history, monkeypatch, tmp_path)

        assert result is not None
        assert result["trend"] == "insufficient_data"
        assert mock_rce.call_count == 1
        kwargs = mock_rce.call_args.kwargs
        assert kwargs["category"] == "data_feed_stale"

    def test_alert_deduped_within_window(self, monkeypatch, tmp_path):
        import time as _time

        history = [{"ts": _time.time() - 60 * 86400, "netflow": -100.0}]
        _, first = self._run(history, monkeypatch, tmp_path)
        assert first.call_count == 1
        # Second run within the 7d window: state file dedups.
        _, second = self._run(history, monkeypatch, tmp_path)
        assert second.call_count == 0

    def test_fresh_history_no_alert(self, monkeypatch, tmp_path):
        import time as _time

        now = _time.time()
        history = [{"ts": now - k * 86400, "netflow": -10.0} for k in range(3)]
        _, mock_rce = self._run(history, monkeypatch, tmp_path)
        assert mock_rce.call_count == 0

    def test_empty_history_alerts(self, monkeypatch, tmp_path):
        """No history at all is the worst staleness — must alert too."""
        result, mock_rce = self._run([], monkeypatch, tmp_path)
        assert result is not None
        assert mock_rce.call_count == 1
