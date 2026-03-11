"""Tests for unified loop extensions in metals_loop.py, metals_llm.py, metals_risk.py.

Verifies crypto price fetching, signal reading, probability engine, Telegram
formatting, and trigger detection work for all 5 instruments (XAG, XAU, BTC, ETH, MSTR).
"""
import json
import os
import sys
import time
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))


# ---------------------------------------------------------------------------
# metals_llm.py — TRACKED_SYMBOLS + kline routing
# ---------------------------------------------------------------------------

class TestMetalsLLMExtensions:
    def test_tracked_symbols_includes_crypto(self):
        from metals_llm import TRACKED_SYMBOLS
        assert "BTC-USD" in TRACKED_SYMBOLS
        assert "ETH-USD" in TRACKED_SYMBOLS
        assert TRACKED_SYMBOLS["BTC-USD"] == "BTCUSDT"
        assert TRACKED_SYMBOLS["ETH-USD"] == "ETHUSDT"

    def test_metals_still_tracked(self):
        from metals_llm import TRACKED_SYMBOLS
        assert "XAG-USD" in TRACKED_SYMBOLS
        assert "XAU-USD" in TRACKED_SYMBOLS

    def test_backwards_compat_alias(self):
        from metals_llm import METALS_SYMBOLS, TRACKED_SYMBOLS
        assert METALS_SYMBOLS is TRACKED_SYMBOLS

    def test_crypto_tickers_set(self):
        from metals_llm import _CRYPTO_TICKERS
        assert "BTC-USD" in _CRYPTO_TICKERS
        assert "ETH-USD" in _CRYPTO_TICKERS
        assert "XAG-USD" not in _CRYPTO_TICKERS

    @patch("metals_llm.requests.get")
    def test_fetch_klines_crypto_uses_spot(self, mock_get):
        from metals_llm import _fetch_fapi_klines, SPOT_KLINES_BASE
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [[0, "100", "110", "90", "105", "1000", 0, "0", 0, "0", "0", "0"]],
        )
        mock_get.return_value.raise_for_status = lambda: None
        result = _fetch_fapi_klines("BTCUSDT", ticker="BTC-USD")
        assert result is not None
        assert len(result) == 1
        # Verify SPOT endpoint was called (not FAPI)
        call_url = mock_get.call_args[0][0]
        assert "api.binance.com" in call_url

    @patch("metals_llm.requests.get")
    def test_fetch_klines_metals_uses_fapi(self, mock_get):
        from metals_llm import _fetch_fapi_klines, FAPI_BASE
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [[0, "33", "34", "32", "33.5", "500", 0, "0", 0, "0", "0", "0"]],
        )
        mock_get.return_value.raise_for_status = lambda: None
        result = _fetch_fapi_klines("XAGUSDT", ticker="XAG-USD")
        assert result is not None
        call_url = mock_get.call_args[0][0]
        assert "fapi.binance.com" in call_url


# ---------------------------------------------------------------------------
# metals_risk.py — ATR defaults + position key mapping
# ---------------------------------------------------------------------------

class TestMetalsRiskExtensions:
    def test_atr_defaults_include_crypto(self):
        from metals_risk import ATR_DEFAULTS
        assert "BTC-USD" in ATR_DEFAULTS
        assert "ETH-USD" in ATR_DEFAULTS
        assert "MSTR" in ATR_DEFAULTS
        # BTC ATR should be ~3-4%, ETH ~4-5%
        assert 2.0 < ATR_DEFAULTS["BTC-USD"] < 6.0
        assert 2.0 < ATR_DEFAULTS["ETH-USD"] < 7.0

    def test_leverage_defaults_include_crypto(self):
        from metals_risk import _LEVERAGE_DEFAULTS
        assert "btc" in _LEVERAGE_DEFAULTS
        assert "eth" in _LEVERAGE_DEFAULTS
        assert _LEVERAGE_DEFAULTS["btc"] == 1.0  # spot, no leverage

    def test_position_key_to_ticker(self):
        from metals_risk import _position_key_to_ticker
        assert _position_key_to_ticker("silver301") == "XAG-USD"
        assert _position_key_to_ticker("gold") == "XAU-USD"
        assert _position_key_to_ticker("btc") == "BTC-USD"
        assert _position_key_to_ticker("xbt_tracker") == "BTC-USD"
        assert _position_key_to_ticker("eth") == "ETH-USD"
        assert _position_key_to_ticker("eth_tracker") == "ETH-USD"
        assert _position_key_to_ticker("mstr") == "MSTR"


# ---------------------------------------------------------------------------
# metals_loop.py — Constants + price fetching
# ---------------------------------------------------------------------------

class TestLoopConstants:
    def test_signal_tickers(self):
        from metals_loop import SIGNAL_TICKERS
        assert "XAG-USD" in SIGNAL_TICKERS
        assert "BTC-USD" in SIGNAL_TICKERS
        assert "ETH-USD" in SIGNAL_TICKERS
        # MSTR was removed from signals
        assert "MSTR" not in SIGNAL_TICKERS

    def test_all_tracked_tickers(self):
        from metals_loop import ALL_TRACKED_TICKERS
        assert "XAG-USD" in ALL_TRACKED_TICKERS
        assert "BTC-USD" in ALL_TRACKED_TICKERS
        assert "MSTR" in ALL_TRACKED_TICKERS
        assert len(ALL_TRACKED_TICKERS) == 5

    def test_crypto_symbols(self):
        from metals_loop import CRYPTO_SYMBOLS
        assert "BTC-USD" in CRYPTO_SYMBOLS
        assert CRYPTO_SYMBOLS["BTC-USD"] == "BTCUSDT"

    def test_check_interval_is_60(self):
        from metals_loop import CHECK_INTERVAL
        assert CHECK_INTERVAL == 60


# ---------------------------------------------------------------------------
# metals_loop.py — read_signal_data
# ---------------------------------------------------------------------------

class TestReadSignalData:
    @patch("metals_loop.os.path.exists", return_value=True)
    @patch("metals_loop.os.path.getmtime", return_value=time.time())
    @patch("builtins.open")
    def test_reads_crypto_tickers(self, mock_open, mock_mtime, mock_exists):
        from metals_loop import read_signal_data
        data = {
            "tickers": {
                "XAG-USD": {"action": "BUY", "confidence": 0.7, "rsi": 45,
                             "bb_position": "inside", "regime": "trending-up",
                             "atr_pct": 4.2, "extra": {"_buy_count": 5, "_sell_count": 1, "_voters": 6}},
                "BTC-USD": {"action": "SELL", "confidence": 0.6, "rsi": 68,
                             "bb_position": "inside", "regime": "range-bound",
                             "atr_pct": 3.1, "extra": {"_buy_count": 2, "_sell_count": 4, "_voters": 6}},
                "ETH-USD": {"action": "HOLD", "confidence": 0.0, "rsi": 50,
                             "bb_position": "inside", "regime": "range-bound",
                             "atr_pct": 4.5, "extra": {"_buy_count": 0, "_sell_count": 0, "_voters": 0}},
            },
            "timeframe_heatmap": {
                "BTC-USD": {"Now": "SELL", "12h": "HOLD"},
            },
        }
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_open.return_value.read = lambda: json.dumps(data)
        # Use json.load mock
        with patch("metals_loop.json.load", return_value=data):
            result = read_signal_data()
            assert "BTC-USD" in result
            assert result["BTC-USD"]["action"] == "SELL"
            assert result["BTC-USD"]["buy_count"] == 2
            assert "ETH-USD" in result

    @patch("metals_loop.os.path.exists", return_value=True)
    @patch("metals_loop.os.path.getmtime", return_value=time.time())
    @patch("builtins.open")
    def test_preserves_forecasts_and_extra_payload(self, mock_open, mock_mtime, mock_exists):
        from metals_loop import read_signal_data
        data = {
            "forecast_signals": {
                "XAG-USD": {"chronos_24h_pct": 6.5, "chronos_24h_conf": 0.81},
            },
            "cumulative_gains": {
                "XAG-USD": {"1d": 1.2},
            },
            "tickers": {
                "XAG-USD": {
                    "action": "BUY",
                    "confidence": 0.7,
                    "weighted_confidence": 0.72,
                    "rsi": 45,
                    "bb_position": "inside",
                    "regime": "trending-up",
                    "atr_pct": 4.2,
                    "price": 85.4,
                    "extra": {
                        "_buy_count": 5,
                        "_sell_count": 1,
                        "_voters": 6,
                        "fibonacci_indicators": {
                            "fib_levels": {"0.236": 85.6},
                        },
                    },
                },
            },
        }
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_open.return_value.read = lambda: json.dumps(data)
        with patch("metals_loop.json.load", return_value=data):
            result = read_signal_data()
            assert result["forecast_signals"]["XAG-USD"]["chronos_24h_pct"] == 6.5
            assert result["cumulative_gains"]["XAG-USD"]["1d"] == 1.2
            assert result["XAG-USD"]["extra"]["fibonacci_indicators"]["fib_levels"]["0.236"] == 85.6


# ---------------------------------------------------------------------------
# metals_loop.py — Telegram formatting
# ---------------------------------------------------------------------------

class TestBuildProbabilityTelegram:
    def _make_prob_report(self):
        """Create a mock probability report for all 5 instruments."""
        return {
            "XAG-USD": {
                "price": 33.45, "prob_up_pct": 72.0, "prob_down_pct": 28.0,
                "signal_action": "BUY", "signal_buy_count": 5, "signal_sell_count": 2,
                "signal_rsi": 55, "signal_regime": "trending-up",
                "momentum": {"velocity_pct": 0.01, "acceleration": 0, "trend": "flat"},
                "chronos_1h": {"direction": "up", "pct_move": 0.12},
            },
            "XAU-USD": {
                "price": 2890.0, "prob_up_pct": 61.0, "prob_down_pct": 39.0,
                "signal_action": "HOLD", "signal_buy_count": 1, "signal_sell_count": 1,
                "signal_rsi": 50, "signal_regime": "range-bound",
                "momentum": {"velocity_pct": 0, "acceleration": 0, "trend": "flat"},
            },
            "BTC-USD": {
                "price": 67200.0, "prob_up_pct": 42.0, "prob_down_pct": 58.0,
                "signal_action": "SELL", "signal_buy_count": 1, "signal_sell_count": 2,
                "signal_rsi": 48, "signal_regime": "range-bound",
                "momentum": {"velocity_pct": -0.01, "acceleration": 0, "trend": "flat"},
                "fear_greed": {"value": 7, "classification": "Extreme Fear"},
                "onchain": {"mvrv": 1.5, "zone": "accumulation"},
            },
            "ETH-USD": {
                "price": 1996.0, "prob_up_pct": 50.0, "prob_down_pct": 50.0,
                "signal_action": "HOLD", "signal_buy_count": 0, "signal_sell_count": 0,
                "signal_rsi": 50, "signal_regime": "range-bound",
                "momentum": {"velocity_pct": 0, "acceleration": 0, "trend": "flat"},
            },
            "MSTR": {
                "price": 287.0, "prob_up_pct": 45.0, "prob_down_pct": 55.0,
                "momentum": {"velocity_pct": 0, "acceleration": 0, "trend": "flat"},
                "btc_proxy": True,
            },
        }

    @patch("metals_loop._underlying_prices", {"BTC-USD": 67200, "ETH-USD": 1996})
    @patch("metals_loop.POSITIONS", {})
    @patch("metals_loop.price_history", [])
    @patch("metals_loop.check_count", 142)
    @patch("metals_loop.CRYPTO_DATA_AVAILABLE", False)
    def test_message_includes_all_tickers(self):
        from metals_loop import build_probability_telegram
        msg = build_probability_telegram(self._make_prob_report(), "14:32 CET")
        assert msg is not None
        assert "*PROB*" in msg
        # Check all tickers are mentioned
        assert "XAG" in msg
        assert "XAU" in msg
        assert "BTC" in msg
        assert "ETH" in msg
        assert "MSTR" in msg
        # Check footer
        assert "#142" in msg
        assert "14:32 CET" in msg

    @patch("metals_loop._underlying_prices", {"BTC-USD": 67200, "ETH-USD": 1996})
    @patch("metals_loop.POSITIONS", {})
    @patch("metals_loop.price_history", [])
    @patch("metals_loop.check_count", 100)
    @patch("metals_loop.CRYPTO_DATA_AVAILABLE", False)
    def test_watch_line_shows_top_movers(self):
        from metals_loop import build_probability_telegram
        msg = build_probability_telegram(self._make_prob_report(), "10:00 CET")
        first_line = msg.split("\n")[0]
        # XAG at 72% has highest deviation (22% from 50%)
        assert "XAG" in first_line

    @patch("metals_loop._underlying_prices", {"BTC-USD": 67200, "ETH-USD": 1996})
    @patch("metals_loop.POSITIONS", {})
    @patch("metals_loop.price_history", [])
    @patch("metals_loop.check_count", 100)
    @patch("metals_loop.CRYPTO_DATA_AVAILABLE", False)
    def test_crypto_context_shown(self):
        from metals_loop import build_probability_telegram
        msg = build_probability_telegram(self._make_prob_report(), "10:00 CET")
        # BTC should show F&G and MVRV
        assert "F&G: 7" in msg
        assert "MVRV: 1.50" in msg
        # ETH should show ETH/BTC ratio
        assert "ETH/BTC" in msg

    @patch("metals_loop._underlying_prices", {})
    @patch("metals_loop.POSITIONS", {})
    @patch("metals_loop.price_history", [])
    @patch("metals_loop.check_count", 1)
    @patch("metals_loop.CRYPTO_DATA_AVAILABLE", False)
    def test_under_4096_chars(self):
        from metals_loop import build_probability_telegram
        msg = build_probability_telegram(self._make_prob_report(), "10:00 CET")
        assert len(msg) < 4096  # Telegram limit


# ---------------------------------------------------------------------------
# metals_loop.py — _format_price
# ---------------------------------------------------------------------------

class TestFormatPrice:
    def test_btc_price(self):
        from metals_loop import _format_price
        assert _format_price(67200, "BTC-USD") == "$67.2K"

    def test_eth_price(self):
        from metals_loop import _format_price
        assert _format_price(1996, "ETH-USD") == "$1,996"

    def test_silver_price(self):
        from metals_loop import _format_price
        assert _format_price(33.45, "XAG-USD") == "$33.45"

    def test_gold_price(self):
        from metals_loop import _format_price
        assert _format_price(2890, "XAU-USD") == "$2,890"

    def test_mstr_price(self):
        from metals_loop import _format_price
        assert _format_price(287, "MSTR") == "$287.00"


# ---------------------------------------------------------------------------
# metals_signal_tracker.py — crypto ticker support
# ---------------------------------------------------------------------------

class TestSignalTrackerCrypto:
    @pytest.fixture(autouse=True)
    def _isolate_files(self, tmp_path, monkeypatch):
        import metals_signal_tracker as mod
        monkeypatch.setattr(mod, "SIGNAL_LOG", str(tmp_path / "signal_log.jsonl"))
        monkeypatch.setattr(mod, "OUTCOMES_LOG", str(tmp_path / "outcomes.jsonl"))
        monkeypatch.setattr(mod, "ACCURACY_CACHE_FILE", str(tmp_path / "accuracy.json"))
        yield

    def test_log_snapshot_records_crypto_signals(self):
        from metals_signal_tracker import log_snapshot, SIGNAL_LOG
        signal_data = {
            "BTC-USD": {
                "action": "SELL", "confidence": 0.6, "weighted_confidence": 0.5,
                "buy_count": 1, "sell_count": 3, "voters": 4, "rsi": 65,
                "regime": "range-bound", "vote_detail": "B:ema | S:rsi,macd,sentiment",
                "price": 67200,
            },
        }
        log_snapshot(1, {}, {}, signal_data, {}, False, [])

        with open(SIGNAL_LOG, "r") as f:
            entry = json.loads(f.readline())
        assert "BTC-USD" in entry["signals"]
        assert entry["signals"]["BTC-USD"]["action"] == "SELL"
        assert "BTC-USD" in entry["prices"]
        assert entry["prices"]["BTC-USD"] == 67200

    def test_resolve_outcome_handles_crypto(self):
        from metals_signal_tracker import _resolve_outcome
        import datetime
        now = time.time()
        entry = {
            "ts": datetime.datetime.fromtimestamp(now - 4000, tz=datetime.timezone.utc).isoformat(),
            "prices": {"BTC-USD": 67000, "XAG-USD": 33.0},
            "signals": {
                "BTC-USD": {"action": "BUY"},
            },
            "llm": {},
        }
        result = _resolve_outcome(entry, "1h", {"BTC-USD": 68000, "XAG-USD": 33.5}, now)
        assert result is not None
        assert "BTC-USD" in result
        assert result["BTC-USD"]["actual_dir"] == "up"
        assert result["BTC-USD"]["main_correct"] is True


def test_safe_print_fallback_on_unicode_encode_error(monkeypatch):
    import builtins
    import metals_loop as ml

    calls = {"n": 0, "msgs": []}

    def flaky_print(msg, flush=True):
        calls["n"] += 1
        calls["msgs"].append(msg)
        if calls["n"] == 1:
            raise UnicodeEncodeError("charmap", msg, 0, 1, "cannot encode")
        return None

    monkeypatch.setattr(builtins, "print", flaky_print)
    ml._safe_print("BTC ↑ 54%")

    # First call raises UnicodeEncodeError, second call uses sanitized fallback.
    assert calls["n"] >= 2
    assert "?" in calls["msgs"][-1]


def test_log_uses_safe_print(monkeypatch):
    import metals_loop as ml

    seen = []

    def capture(msg):
        seen.append(msg)

    monkeypatch.setattr(ml, "_safe_print", capture)
    ml.log("XAG ↑ 62%")

    assert seen
    assert "XAG ↑ 62%" in seen[0]


def test_singleton_lock_blocks_second_instance(tmp_path):
    import metals_loop as ml

    if ml.msvcrt is None:
        pytest.skip("Windows-only lock behavior")

    lock_path = tmp_path / "metals_loop.singleton.lock"

    holder = lock_path.open("a+", encoding="utf-8")
    ml.msvcrt.locking(holder.fileno(), ml.msvcrt.LK_NBLCK, 1)
    try:
        ml.release_singleton_lock()
        assert ml.acquire_singleton_lock(str(lock_path)) is False
    finally:
        ml.msvcrt.locking(holder.fileno(), ml.msvcrt.LK_UNLCK, 1)
        holder.close()
        ml.release_singleton_lock()


def test_singleton_lock_release_allows_reacquire(tmp_path):
    import metals_loop as ml

    if ml.msvcrt is None:
        pytest.skip("Windows-only lock behavior")

    lock_path = tmp_path / "metals_loop.singleton.lock"

    ml.release_singleton_lock()
    assert ml.acquire_singleton_lock(str(lock_path)) is True
    ml.release_singleton_lock()

    assert ml.acquire_singleton_lock(str(lock_path)) is True
    ml.release_singleton_lock()
