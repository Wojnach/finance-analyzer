"""Tests for futures_data.py and signals/futures_flow.py."""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """OHLCV DataFrame with rising prices."""
    n = 30
    closes = list(np.linspace(100, 110, n))
    return pd.DataFrame({
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": [1000] * n,
    })


@pytest.fixture
def falling_df():
    """OHLCV DataFrame with falling prices."""
    n = 30
    closes = list(np.linspace(110, 100, n))
    return pd.DataFrame({
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": [1000] * n,
    })


@pytest.fixture
def oi_history_rising():
    """OI history with rising open interest."""
    return [
        {"oi": 1000 + i * 20, "oi_usdt": (1000 + i * 20) * 50000, "timestamp": 1000 + i}
        for i in range(30)
    ]


@pytest.fixture
def oi_history_falling():
    """OI history with falling open interest."""
    return [
        {"oi": 1600 - i * 20, "oi_usdt": (1600 - i * 20) * 50000, "timestamp": 1000 + i}
        for i in range(30)
    ]


@pytest.fixture
def ls_ratio_neutral():
    return [{"longShortRatio": 1.2, "longAccount": 0.545, "shortAccount": 0.455, "timestamp": 1000}]


@pytest.fixture
def ls_ratio_extreme_long():
    return [{"longShortRatio": 2.5, "longAccount": 0.71, "shortAccount": 0.29, "timestamp": 1000}]


@pytest.fixture
def ls_ratio_extreme_short():
    return [{"longShortRatio": 0.5, "longAccount": 0.33, "shortAccount": 0.67, "timestamp": 1000}]


@pytest.fixture
def top_position_bullish():
    """Top traders more long than crowd."""
    return [{"longShortRatio": 2.0, "longAccount": 0.67, "shortAccount": 0.33, "timestamp": 1000}]


@pytest.fixture
def top_position_bearish():
    """Top traders more short than crowd."""
    return [{"longShortRatio": 0.5, "longAccount": 0.33, "shortAccount": 0.67, "timestamp": 1000}]


@pytest.fixture
def funding_high():
    return [
        {"fundingRate": 0.0006, "fundingTime": 1000 + i, "symbol": "BTCUSDT"}
        for i in range(5)
    ]


@pytest.fixture
def funding_low():
    return [
        {"fundingRate": -0.0005, "fundingTime": 1000 + i, "symbol": "BTCUSDT"}
        for i in range(5)
    ]


@pytest.fixture
def funding_neutral():
    return [
        {"fundingRate": 0.0001, "fundingTime": 1000 + i, "symbol": "BTCUSDT"}
        for i in range(5)
    ]


# ---------------------------------------------------------------------------
# futures_data.py tests
# ---------------------------------------------------------------------------

class TestFuturesData:
    """Tests for portfolio.futures_data module."""

    def test_non_crypto_returns_none(self):
        from portfolio.futures_data import get_open_interest
        assert get_open_interest("NVDA") is None

    def test_non_crypto_all_returns_none(self):
        from portfolio.futures_data import get_all_futures_data
        assert get_all_futures_data("AAPL") is None

    def test_non_crypto_oi_history_returns_none(self):
        from portfolio.futures_data import get_open_interest_history
        assert get_open_interest_history("GOOGL") is None

    def test_non_crypto_ls_ratio_returns_none(self):
        from portfolio.futures_data import get_long_short_ratio
        assert get_long_short_ratio("MSTR") is None

    def test_non_crypto_top_position_returns_none(self):
        from portfolio.futures_data import get_top_trader_position_ratio
        assert get_top_trader_position_ratio("META") is None

    def test_non_crypto_top_account_returns_none(self):
        from portfolio.futures_data import get_top_trader_account_ratio
        assert get_top_trader_account_ratio("AMZN") is None

    def test_non_crypto_funding_returns_none(self):
        from portfolio.futures_data import get_funding_rate_history
        assert get_funding_rate_history("PLTR") is None

    @patch("portfolio.futures_data._fetch_json")
    def test_get_open_interest_success(self, mock_fetch):
        mock_fetch.return_value = {
            "openInterest": "12345.678",
            "symbol": "BTCUSDT",
            "time": 1700000000000,
        }
        from portfolio.futures_data import get_open_interest
        # Clear cache
        from portfolio.shared_state import _tool_cache
        _tool_cache.pop("futures_oi_BTC-USD", None)

        result = get_open_interest("BTC-USD")
        assert result is not None
        assert result["oi"] == 12345.678
        assert result["symbol"] == "BTCUSDT"

    @patch("portfolio.futures_data._fetch_json")
    def test_get_open_interest_failure(self, mock_fetch):
        mock_fetch.return_value = None
        from portfolio.futures_data import get_open_interest
        from portfolio.shared_state import _tool_cache
        _tool_cache.pop("futures_oi_ETH-USD", None)

        result = get_open_interest("ETH-USD")
        assert result is None

    @patch("portfolio.futures_data._fetch_json")
    def test_get_oi_history_success(self, mock_fetch):
        mock_fetch.return_value = [
            {"sumOpenInterest": "100", "sumOpenInterestValue": "5000000", "timestamp": 1000},
            {"sumOpenInterest": "110", "sumOpenInterestValue": "5500000", "timestamp": 2000},
        ]
        from portfolio.futures_data import get_open_interest_history
        from portfolio.shared_state import _tool_cache
        _tool_cache.pop("futures_oi_hist_BTC-USD_5m", None)

        result = get_open_interest_history("BTC-USD")
        assert result is not None
        assert len(result) == 2
        assert result[0]["oi"] == 100.0
        assert result[1]["oi_usdt"] == 5500000.0

    @patch("portfolio.futures_data._fetch_json")
    def test_get_ls_ratio_success(self, mock_fetch):
        mock_fetch.return_value = [
            {"longShortRatio": "1.5", "longAccount": "0.60", "shortAccount": "0.40", "timestamp": 1000},
        ]
        from portfolio.futures_data import get_long_short_ratio
        from portfolio.shared_state import _tool_cache
        _tool_cache.pop("futures_ls_BTC-USD_5m", None)

        result = get_long_short_ratio("BTC-USD")
        assert result is not None
        assert len(result) == 1
        assert result[0]["longShortRatio"] == 1.5

    @patch("portfolio.futures_data._fetch_json")
    def test_get_funding_history_success(self, mock_fetch):
        mock_fetch.return_value = [
            {"fundingRate": "0.0001", "fundingTime": 1000, "symbol": "BTCUSDT"},
            {"fundingRate": "0.0002", "fundingTime": 2000, "symbol": "BTCUSDT"},
        ]
        from portfolio.futures_data import get_funding_rate_history
        from portfolio.shared_state import _tool_cache
        _tool_cache.pop("futures_funding_hist_BTC-USD", None)

        result = get_funding_rate_history("BTC-USD")
        assert result is not None
        assert len(result) == 2
        assert result[0]["fundingRate"] == 0.0001

    @patch("portfolio.futures_data._fetch_json")
    def test_get_all_futures_data_partial_failure(self, mock_fetch):
        """Some endpoints fail but others succeed."""
        call_count = [0]
        def side_effect(url, **kwargs):
            call_count[0] += 1
            if "openInterest" in url and "Hist" not in url:
                return {"openInterest": "100", "symbol": "BTCUSDT", "time": 1000}
            return None  # all others fail

        mock_fetch.side_effect = side_effect
        from portfolio.futures_data import get_all_futures_data
        from portfolio.shared_state import _tool_cache
        # Clear all caches for BTC
        keys_to_clear = [k for k in list(_tool_cache.keys()) if "BTC" in k and "futures" in k]
        for k in keys_to_clear:
            _tool_cache.pop(k, None)

        result = get_all_futures_data("BTC-USD")
        assert result is not None
        assert result["open_interest"] is not None
        # Others should be None since fetch returned None
        assert result["oi_history"] is None

    def test_symbol_map_entries(self):
        from portfolio.futures_data import SYMBOL_MAP
        assert "BTC-USD" in SYMBOL_MAP
        assert "ETH-USD" in SYMBOL_MAP
        assert SYMBOL_MAP["BTC-USD"] == "BTCUSDT"
        assert SYMBOL_MAP["ETH-USD"] == "ETHUSDT"


# ---------------------------------------------------------------------------
# futures_flow.py sub-indicator tests
# ---------------------------------------------------------------------------

class TestOiTrend:
    def test_rising_oi_rising_price_buy(self, oi_history_rising, sample_df):
        from portfolio.signals.futures_flow import _oi_trend
        assert _oi_trend(oi_history_rising, sample_df) == "BUY"

    def test_rising_oi_falling_price_sell(self, oi_history_rising, falling_df):
        from portfolio.signals.futures_flow import _oi_trend
        assert _oi_trend(oi_history_rising, falling_df) == "SELL"

    def test_falling_oi_hold(self, oi_history_falling, sample_df):
        from portfolio.signals.futures_flow import _oi_trend
        assert _oi_trend(oi_history_falling, sample_df) == "HOLD"

    def test_no_data_hold(self, sample_df):
        from portfolio.signals.futures_flow import _oi_trend
        assert _oi_trend(None, sample_df) == "HOLD"
        assert _oi_trend([], sample_df) == "HOLD"


class TestOiDivergence:
    def test_price_up_oi_down_sell(self, oi_history_falling, sample_df):
        from portfolio.signals.futures_flow import _oi_divergence
        assert _oi_divergence(oi_history_falling, sample_df) == "SELL"

    def test_price_down_oi_down_buy(self, oi_history_falling, falling_df):
        from portfolio.signals.futures_flow import _oi_divergence
        assert _oi_divergence(oi_history_falling, falling_df) == "BUY"

    def test_no_data_hold(self, sample_df):
        from portfolio.signals.futures_flow import _oi_divergence
        assert _oi_divergence(None, sample_df) == "HOLD"


class TestLsExtreme:
    def test_extreme_long_sell(self, ls_ratio_extreme_long):
        from portfolio.signals.futures_flow import _ls_extreme
        assert _ls_extreme(ls_ratio_extreme_long) == "SELL"

    def test_extreme_short_buy(self, ls_ratio_extreme_short):
        from portfolio.signals.futures_flow import _ls_extreme
        assert _ls_extreme(ls_ratio_extreme_short) == "BUY"

    def test_neutral_hold(self, ls_ratio_neutral):
        from portfolio.signals.futures_flow import _ls_extreme
        assert _ls_extreme(ls_ratio_neutral) == "HOLD"

    def test_no_data_hold(self):
        from portfolio.signals.futures_flow import _ls_extreme
        assert _ls_extreme(None) == "HOLD"
        assert _ls_extreme([]) == "HOLD"


class TestTopVsCrowd:
    def test_top_more_long_buy(self, top_position_bullish, ls_ratio_neutral):
        from portfolio.signals.futures_flow import _top_vs_crowd
        assert _top_vs_crowd(top_position_bullish, ls_ratio_neutral) == "BUY"

    def test_top_more_short_sell(self, top_position_bearish, ls_ratio_neutral):
        from portfolio.signals.futures_flow import _top_vs_crowd
        assert _top_vs_crowd(top_position_bearish, ls_ratio_neutral) == "SELL"

    def test_no_data_hold(self, ls_ratio_neutral):
        from portfolio.signals.futures_flow import _top_vs_crowd
        assert _top_vs_crowd(None, ls_ratio_neutral) == "HOLD"
        assert _top_vs_crowd([], ls_ratio_neutral) == "HOLD"


class TestFundingTrend:
    def test_high_funding_sell(self, funding_high):
        from portfolio.signals.futures_flow import _funding_trend
        assert _funding_trend(funding_high) == "SELL"

    def test_low_funding_buy(self, funding_low):
        from portfolio.signals.futures_flow import _funding_trend
        assert _funding_trend(funding_low) == "BUY"

    def test_neutral_hold(self, funding_neutral):
        from portfolio.signals.futures_flow import _funding_trend
        assert _funding_trend(funding_neutral) == "HOLD"

    def test_no_data_hold(self):
        from portfolio.signals.futures_flow import _funding_trend
        assert _funding_trend(None) == "HOLD"
        assert _funding_trend([]) == "HOLD"
        # Less than 3 entries
        assert _funding_trend([{"fundingRate": 0.001, "fundingTime": 1}]) == "HOLD"


class TestOiAcceleration:
    def test_accelerating_oi_price_up_buy(self, sample_df):
        """OI accelerating + price up = BUY."""
        from portfolio.signals.futures_flow import _oi_acceleration
        # OI grows slowly first half, fast second half → positive acceleration
        oi = [{"oi": 1000 + i * 5, "oi_usdt": 0, "timestamp": i} for i in range(15)]
        oi += [{"oi": 1075 + i * 20, "oi_usdt": 0, "timestamp": 15 + i} for i in range(15)]
        assert _oi_acceleration(oi, sample_df) == "BUY"

    def test_no_data_hold(self, sample_df):
        from portfolio.signals.futures_flow import _oi_acceleration
        assert _oi_acceleration(None, sample_df) == "HOLD"
        assert _oi_acceleration([], sample_df) == "HOLD"


# ---------------------------------------------------------------------------
# Composite signal tests
# ---------------------------------------------------------------------------

class TestComputeFuturesFlowSignal:
    def test_non_crypto_hold(self, sample_df):
        from portfolio.signals.futures_flow import compute_futures_flow_signal
        result = compute_futures_flow_signal(sample_df, context={"ticker": "NVDA"})
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_no_context_hold(self, sample_df):
        from portfolio.signals.futures_flow import compute_futures_flow_signal
        result = compute_futures_flow_signal(sample_df, context=None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.signals.futures_flow._cached")
    def test_fetch_failure_hold(self, mock_cached, sample_df):
        mock_cached.return_value = None
        from portfolio.signals.futures_flow import compute_futures_flow_signal
        result = compute_futures_flow_signal(sample_df, context={"ticker": "BTC-USD"})
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0
        assert result["indicators"].get("error") == "fetch_failed"

    @patch("portfolio.signals.futures_flow._cached")
    def test_confidence_cap(self, mock_cached, sample_df):
        """Confidence should never exceed 0.7."""
        # Provide data that would give high confidence
        mock_cached.return_value = {
            "open_interest": {"oi": 50000, "symbol": "BTCUSDT", "time": 1000},
            "oi_history": [
                {"oi": 1000 + i * 50, "oi_usdt": (1000 + i * 50) * 50000, "timestamp": i}
                for i in range(30)
            ],
            "ls_ratio": [{"longShortRatio": 0.5, "longAccount": 0.33, "shortAccount": 0.67, "timestamp": 1}],
            "top_position_ratio": [{"longShortRatio": 0.5, "longAccount": 0.33, "shortAccount": 0.67, "timestamp": 1}],
            "top_account_ratio": None,
            "funding_history": [
                {"fundingRate": -0.0005, "fundingTime": i, "symbol": "BTCUSDT"}
                for i in range(5)
            ],
        }
        from portfolio.signals.futures_flow import compute_futures_flow_signal
        result = compute_futures_flow_signal(sample_df, context={"ticker": "BTC-USD"})
        assert result["confidence"] <= 0.7

    @patch("portfolio.signals.futures_flow._cached")
    def test_result_structure(self, mock_cached, sample_df):
        """Result should have all expected keys."""
        mock_cached.return_value = {
            "open_interest": {"oi": 50000, "symbol": "BTCUSDT", "time": 1000},
            "oi_history": [
                {"oi": 1000 + i * 10, "oi_usdt": (1000 + i * 10) * 50000, "timestamp": i}
                for i in range(30)
            ],
            "ls_ratio": [{"longShortRatio": 1.2, "longAccount": 0.545, "shortAccount": 0.455, "timestamp": 1}],
            "top_position_ratio": [{"longShortRatio": 1.3, "longAccount": 0.57, "shortAccount": 0.43, "timestamp": 1}],
            "top_account_ratio": None,
            "funding_history": [
                {"fundingRate": 0.0001, "fundingTime": i, "symbol": "BTCUSDT"}
                for i in range(5)
            ],
        }
        from portfolio.signals.futures_flow import compute_futures_flow_signal
        result = compute_futures_flow_signal(sample_df, context={"ticker": "BTC-USD"})

        assert "action" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 0.7
        assert "sub_signals" in result
        assert "indicators" in result
        # Check sub_signal keys
        expected_subs = {"oi_trend", "oi_divergence", "ls_extreme", "top_vs_crowd", "funding_trend", "oi_acceleration"}
        assert set(result["sub_signals"].keys()) == expected_subs
        # Check indicator keys
        assert "open_interest" in result["indicators"]
        assert "ls_ratio" in result["indicators"]
        assert "funding_rate" in result["indicators"]

    def test_metals_hold(self, sample_df):
        """Metals (XAU, XAG) should also return HOLD."""
        from portfolio.signals.futures_flow import compute_futures_flow_signal
        result = compute_futures_flow_signal(sample_df, context={"ticker": "XAU-USD"})
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Signal registration tests
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_registered_in_signal_names(self):
        from portfolio.tickers import SIGNAL_NAMES
        assert "futures_flow" in SIGNAL_NAMES

    def test_registered_in_registry(self):
        from portfolio.signal_registry import get_enhanced_signals
        enhanced = get_enhanced_signals()
        assert "futures_flow" in enhanced
        entry = enhanced["futures_flow"]
        assert entry["requires_context"] is True
        assert entry["module_path"] == "portfolio.signals.futures_flow"

    def test_load_signal_func(self):
        from portfolio.signal_registry import get_enhanced_signals, load_signal_func
        entry = get_enhanced_signals()["futures_flow"]
        func = load_signal_func(entry)
        assert func is not None
        assert callable(func)
