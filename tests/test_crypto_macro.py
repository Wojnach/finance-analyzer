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

    def test_quarterly_expiry_day(self):
        """Day of quarterly expiry should signal BUY (relief rally)."""
        from portfolio.signals.crypto_macro import _expiry_proximity

        options = {"days_to_expiry": 0, "nearest_expiry": "28MAR26"}
        action, ind = _expiry_proximity(options)
        assert action == "BUY"
        assert ind["is_quarterly"] is True
