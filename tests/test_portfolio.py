"""Tests for the portfolio intelligence system.

Unit tests run locally without models or network.
Integration tests (marked @pytest.mark.integration) run locally with live models + GPU.
"""

import json
import pytest
import numpy as np
import pandas as pd

from portfolio.main import (
    compute_indicators,
    technical_signal,
    generate_signal,
    TIMEFRAMES,
    STOCK_TIMEFRAMES,
    ALPACA_INTERVAL_MAP,
    alpaca_klines,
    fetch_usd_sek,
    MIN_VOTERS_CRYPTO,
    MIN_VOTERS_STOCK,
)
from portfolio.sentiment import _aggregate_sentiments, _fetch_crypto_headlines

# --- Helpers ---


def make_candles(prices, volume=100.0):
    n = len(prices)
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [volume] * n,
            "time": pd.date_range("2026-01-01", periods=n, freq="15min"),
        }
    )


def make_indicators(**overrides):
    base = {
        "close": 69000.0,
        "rsi": 50.0,
        "macd_hist": 0.0,
        "macd_hist_prev": 0.0,
        "ema9": 69000.0,
        "ema21": 69000.0,
        "bb_upper": 70000.0,
        "bb_lower": 68000.0,
        "bb_mid": 69000.0,
        "price_vs_bb": "inside",
    }
    base.update(overrides)
    return base


# --- compute_indicators ---


class TestComputeIndicators:
    def test_returns_all_keys(self):
        df = make_candles([100 + i * 0.1 for i in range(100)])
        ind = compute_indicators(df)
        expected = {
            "close",
            "rsi",
            "macd_hist",
            "macd_hist_prev",
            "ema9",
            "ema21",
            "bb_upper",
            "bb_lower",
            "bb_mid",
            "price_vs_bb",
            "atr",
            "atr_pct",
            "rsi_p20",
            "rsi_p80",
        }
        assert expected == set(ind.keys())

    def test_rsi_range(self):
        df = make_candles([100 + i * 0.1 for i in range(100)])
        ind = compute_indicators(df)
        assert 0 <= ind["rsi"] <= 100

    def test_uptrend_rsi_above_50(self):
        prices = [100 + i * 2 for i in range(100)]
        ind = compute_indicators(make_candles(prices))
        assert ind["rsi"] > 50

    def test_downtrend_rsi_below_50(self):
        prices = [200 - i * 2 for i in range(100)]
        ind = compute_indicators(make_candles(prices))
        assert ind["rsi"] < 50

    def test_uptrend_ema9_above_ema21(self):
        prices = [100 + i * 2 for i in range(100)]
        ind = compute_indicators(make_candles(prices))
        assert ind["ema9"] > ind["ema21"]

    def test_uptrend_positive_macd(self):
        prices = [100 + i * 2 for i in range(100)]
        ind = compute_indicators(make_candles(prices))
        assert ind["macd_hist"] > 0

    def test_bb_bands_order(self):
        df = make_candles([100 + i * 0.1 for i in range(100)])
        ind = compute_indicators(df)
        assert ind["bb_lower"] < ind["bb_mid"] < ind["bb_upper"]

    def test_price_vs_bb_below(self):
        prices = [100.0] * 99 + [50.0]
        ind = compute_indicators(make_candles(prices))
        assert ind["price_vs_bb"] == "below_lower"

    def test_price_vs_bb_above(self):
        prices = [100.0] * 99 + [150.0]
        ind = compute_indicators(make_candles(prices))
        assert ind["price_vs_bb"] == "above_upper"


# --- technical_signal ---


class TestTechnicalSignal:
    def test_strong_buy(self):
        ind = make_indicators(
            rsi=30, macd_hist=5.0, ema9=70000, ema21=69000, close=70000, bb_mid=69000
        )
        action, conf = technical_signal(ind)
        assert action == "BUY"
        assert conf == 1.0

    def test_strong_sell(self):
        ind = make_indicators(
            rsi=70, macd_hist=-5.0, ema9=68000, ema21=69000, close=68000, bb_mid=69000
        )
        action, conf = technical_signal(ind)
        assert action == "SELL"
        assert conf == 1.0

    def test_mixed_hold(self):
        ind = make_indicators(
            rsi=30, macd_hist=5.0, ema9=68000, ema21=69000, close=68000, bb_mid=69000
        )
        action, conf = technical_signal(ind)
        assert action == "HOLD"
        assert conf == 0.5

    def test_three_buy_one_sell(self):
        ind = make_indicators(
            rsi=30, macd_hist=5.0, ema9=70000, ema21=69000, close=68000, bb_mid=69000
        )
        action, conf = technical_signal(ind)
        assert action == "BUY"
        assert conf == 0.75

    def test_confidence_is_ratio(self):
        ind = make_indicators(
            rsi=70, macd_hist=-5.0, ema9=68000, ema21=69000, close=70000, bb_mid=69000
        )
        action, conf = technical_signal(ind)
        assert action == "SELL"
        assert conf == 0.75


# --- generate_signal ---


class TestGenerateSignal:
    def test_hold_when_too_few_voters(self):
        ind = make_indicators(
            rsi=50, macd_hist=1.0, macd_hist_prev=2.0, price_vs_bb="inside"
        )
        action, conf, extra = generate_signal(ind)
        assert action == "HOLD"
        assert extra["_voters"] < MIN_VOTERS_CRYPTO

    def test_buy_on_rsi_oversold_ema_up(self):
        ind = make_indicators(
            rsi=25,
            macd_hist=1.0,
            macd_hist_prev=-1.0,
            ema9=70000,
            ema21=69000,
            price_vs_bb="below_lower",
        )
        action, conf, extra = generate_signal(ind)
        assert action == "BUY"
        assert conf > 0.5

    def test_sell_on_rsi_overbought_ema_down(self):
        ind = make_indicators(
            rsi=75,
            macd_hist=-1.0,
            macd_hist_prev=1.0,
            ema9=68000,
            ema21=69000,
            price_vs_bb="above_upper",
        )
        action, conf, extra = generate_signal(ind)
        assert action == "SELL"
        assert conf > 0.5

    def test_ema_always_votes(self):
        ind = make_indicators(
            rsi=50, macd_hist=0.5, macd_hist_prev=0.3, price_vs_bb="inside"
        )
        _, _, extra = generate_signal(ind)
        assert extra["_voters"] >= 1

    def test_confidence_range(self):
        ind = make_indicators(
            rsi=25,
            macd_hist=1.0,
            macd_hist_prev=-1.0,
            ema9=70000,
            ema21=69000,
            price_vs_bb="below_lower",
        )
        _, conf, _ = generate_signal(ind)
        assert 0.0 <= conf <= 1.0


# --- sentiment ---


class TestSentimentAggregation:
    def test_aggregate_positive(self):
        sentiments = [
            {"scores": {"positive": 0.8, "negative": 0.1, "neutral": 0.1}},
            {"scores": {"positive": 0.7, "negative": 0.2, "neutral": 0.1}},
        ]
        overall, avg = _aggregate_sentiments(sentiments)
        assert overall == "positive"
        assert avg["positive"] > avg["negative"]

    def test_aggregate_negative(self):
        sentiments = [
            {"scores": {"positive": 0.1, "negative": 0.8, "neutral": 0.1}},
            {"scores": {"positive": 0.1, "negative": 0.7, "neutral": 0.2}},
        ]
        overall, avg = _aggregate_sentiments(sentiments)
        assert overall == "negative"

    def test_aggregate_neutral(self):
        sentiments = [
            {"scores": {"positive": 0.2, "negative": 0.2, "neutral": 0.6}},
            {"scores": {"positive": 0.1, "negative": 0.1, "neutral": 0.8}},
        ]
        overall, avg = _aggregate_sentiments(sentiments)
        assert overall == "neutral"


# --- sentiment API edge case ---


class TestCryptoCompareAPI:
    def test_empty_dict_response_handled(self):
        """CryptoCompare sometimes returns Data:{} instead of Data:[]"""
        import portfolio.sentiment as sm
        import unittest.mock as mock

        fake_response = json.dumps({"Data": {}, "Type": 100}).encode()
        m = mock.MagicMock()
        m.read.return_value = fake_response
        m.__enter__ = lambda s: s
        m.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=m):
            result = _fetch_crypto_headlines("BTC")
        assert result == []

    def test_normal_list_response(self):
        import portfolio.sentiment as sm
        import unittest.mock as mock

        fake_articles = [
            {"title": "BTC pumps", "source": "Test", "published_on": 1700000000},
            {"title": "ETH dips", "source": "Test", "published_on": 1700001000},
        ]
        fake_response = json.dumps({"Data": fake_articles, "Type": 100}).encode()
        m = mock.MagicMock()
        m.read.return_value = fake_response
        m.__enter__ = lambda s: s
        m.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=m):
            result = _fetch_crypto_headlines("BTC", limit=5)
        assert len(result) == 2
        assert result[0]["title"] == "BTC pumps"


# --- ministral_trader ---


class TestMinistralTrader:
    def test_predict_output_format(self):
        from portfolio.ministral_trader import predict
        import unittest.mock as mock

        fake_response = {
            "choices": [{"text": "DECISION: BUY - Bullish signals dominate"}]
        }
        fake_model = mock.MagicMock(return_value=fake_response)

        with mock.patch(
            "portfolio.ministral_trader.load_model", return_value=fake_model
        ):
            ctx = {
                "ticker": "BTC",
                "price_usd": 69000.0,
                "rsi": 42.0,
                "macd_hist": 5.0,
                "ema_bullish": True,
                "bb_position": "inside",
                "fear_greed": 35,
                "fear_greed_class": "Fear",
                "news_sentiment": "neutral",
                "timeframe_summary": "",
                "headlines": "",
            }
            result = predict(ctx)

        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert "reasoning" in result
        assert result["model"] == "CryptoTrader-LM"
        assert result["action"] == "BUY"

    def test_predict_extracts_sell(self):
        from portfolio.ministral_trader import predict
        import unittest.mock as mock

        fake_response = {
            "choices": [{"text": "I recommend SELL due to bearish divergence"}]
        }
        fake_model = mock.MagicMock(return_value=fake_response)

        with mock.patch(
            "portfolio.ministral_trader.load_model", return_value=fake_model
        ):
            ctx = {"ticker": "ETH", "price_usd": 2000.0}
            result = predict(ctx)
        assert result["action"] == "SELL"

    def test_predict_defaults_hold(self):
        from portfolio.ministral_trader import predict
        import unittest.mock as mock

        fake_response = {
            "choices": [{"text": "Market conditions unclear, wait for confirmation"}]
        }
        fake_model = mock.MagicMock(return_value=fake_response)

        with mock.patch(
            "portfolio.ministral_trader.load_model", return_value=fake_model
        ):
            ctx = {"ticker": "BTC", "price_usd": 69000.0}
            result = predict(ctx)
        assert result["action"] == "HOLD"


# --- timeframes config ---


class TestTimeframesConfig:
    def test_seven_timeframes(self):
        assert len(TIMEFRAMES) == 7

    def test_now_is_first(self):
        assert TIMEFRAMES[0][0] == "Now"

    def test_now_has_no_cache(self):
        assert TIMEFRAMES[0][3] == 0

    def test_cache_ttl_increases(self):
        ttls = [tf[3] for tf in TIMEFRAMES]
        for i in range(1, len(ttls)):
            assert ttls[i] >= ttls[i - 1]

    def test_all_valid_intervals(self):
        valid = {
            "1m",
            "3m",
            "5m",
            "15m",
            "30m",
            "1h",
            "2h",
            "4h",
            "6h",
            "8h",
            "12h",
            "1d",
            "3d",
            "1w",
            "1M",
        }
        for _, interval, _, _ in TIMEFRAMES:
            assert interval in valid


# --- ministral_signal subprocess wrapper ---


class TestMinistralSignalWrapper:
    def test_parses_json_output(self):
        import unittest.mock as mock
        from portfolio.ministral_signal import get_ministral_signal

        fake_result = mock.MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = (
            '{"action": "BUY", "reasoning": "test", "model": "CryptoTrader-LM"}\n'
        )

        with mock.patch("subprocess.run", return_value=fake_result):
            result = get_ministral_signal({"ticker": "BTC", "price_usd": 69000})
        assert result["original"]["action"] == "BUY"
        assert result["original"]["model"] == "CryptoTrader-LM"

    def test_raises_on_failure(self):
        import unittest.mock as mock
        from portfolio.ministral_signal import get_ministral_signal

        fake_result = mock.MagicMock()
        fake_result.returncode = 1
        fake_result.stderr = "Error: model not found"

        with mock.patch("subprocess.run", return_value=fake_result):
            with pytest.raises(RuntimeError, match="Ministral failed"):
                get_ministral_signal({"ticker": "BTC", "price_usd": 69000})


# --- trigger system ---


class TestTriggerSystem:
    def setup_method(self):
        from portfolio.trigger import STATE_FILE

        self.state_file = STATE_FILE
        if self.state_file.exists():
            self._backup = self.state_file.read_text()
        else:
            self._backup = None
        if self.state_file.exists():
            self.state_file.unlink()

    def teardown_method(self):
        if self._backup:
            self.state_file.write_text(self._backup)
        elif self.state_file.exists():
            self.state_file.unlink()

    def _make_signals(self, btc_action="HOLD", eth_action="HOLD"):
        return {
            "BTC-USD": {"action": btc_action, "confidence": 0.5},
            "ETH-USD": {"action": eth_action, "confidence": 0.5},
        }

    def test_first_run_triggers_cooldown(self):
        from portfolio.trigger import check_triggers

        triggered, reasons = check_triggers(
            self._make_signals(), {"BTC-USD": 69000, "ETH-USD": 2000}, {}, {}
        )
        assert triggered
        assert any("cooldown" in r or "check-in" in r for r in reasons)

    def test_no_change_no_trigger(self):
        from portfolio.trigger import check_triggers

        sigs = self._make_signals()
        prices = {"BTC-USD": 69000, "ETH-USD": 2000}
        check_triggers(sigs, prices, {}, {})  # seed state
        triggered, reasons = check_triggers(sigs, prices, {}, {})
        assert not triggered

    def test_signal_flip_triggers(self):
        from portfolio.trigger import check_triggers

        prices = {"BTC-USD": 69000, "ETH-USD": 2000}
        # Seed with BUY → triggers on cooldown+consensus, saves BUY as triggered action
        check_triggers(self._make_signals("BUY", "HOLD"), prices, {}, {})
        # SUSTAINED_CHECKS=3: flip from BUY→HOLD must sustain 3 consecutive cycles
        hold_sigs = self._make_signals("HOLD", "HOLD")
        check_triggers(hold_sigs, prices, {}, {})
        check_triggers(hold_sigs, prices, {}, {})
        triggered, reasons = check_triggers(hold_sigs, prices, {}, {})
        assert triggered
        assert any("flipped" in r for r in reasons)

    def test_price_move_triggers(self):
        from portfolio.trigger import check_triggers

        sigs = self._make_signals()
        check_triggers(sigs, {"BTC-USD": 69000, "ETH-USD": 2000}, {}, {})
        triggered, reasons = check_triggers(
            sigs, {"BTC-USD": 71000, "ETH-USD": 2000}, {}, {}
        )
        assert triggered
        assert any("moved" in r and "BTC" in r for r in reasons)

    def test_small_price_move_no_trigger(self):
        from portfolio.trigger import check_triggers

        sigs = self._make_signals()
        check_triggers(sigs, {"BTC-USD": 69000, "ETH-USD": 2000}, {}, {})
        triggered, reasons = check_triggers(
            sigs, {"BTC-USD": 69500, "ETH-USD": 2000}, {}, {}
        )
        assert not triggered

    def test_fear_greed_threshold_triggers(self):
        from portfolio.trigger import check_triggers

        sigs = self._make_signals()
        prices = {"BTC-USD": 69000, "ETH-USD": 2000}
        fg = {"BTC-USD": {"value": 25}}
        check_triggers(sigs, prices, fg, {})
        fg2 = {"BTC-USD": {"value": 18}}
        triggered, reasons = check_triggers(sigs, prices, fg2, {})
        assert triggered
        assert any("F&G" in r for r in reasons)

    def test_sentiment_reversal_triggers(self):
        from portfolio.trigger import check_triggers

        sigs = self._make_signals()
        prices = {"BTC-USD": 69000, "ETH-USD": 2000}
        check_triggers(sigs, prices, {}, {"BTC-USD": "positive"})
        triggered, reasons = check_triggers(sigs, prices, {}, {"BTC-USD": "negative"})
        assert triggered
        assert any("sentiment" in r for r in reasons)

    def test_neutral_sentiment_no_trigger(self):
        from portfolio.trigger import check_triggers

        sigs = self._make_signals()
        prices = {"BTC-USD": 69000, "ETH-USD": 2000}
        check_triggers(sigs, prices, {}, {"BTC-USD": "positive"})
        triggered, reasons = check_triggers(sigs, prices, {}, {"BTC-USD": "neutral"})
        assert not triggered


# --- Alpaca klines ---


class TestAlpacaKlines:
    def test_parses_bars_response(self):
        import unittest.mock as mock

        fake_bars = {
            "bars": [
                {
                    "t": "2026-02-13T14:00:00Z",
                    "o": 130.0,
                    "h": 131.5,
                    "l": 129.5,
                    "c": 131.0,
                    "v": 50000,
                    "n": 200,
                    "vw": 130.8,
                },
                {
                    "t": "2026-02-13T14:15:00Z",
                    "o": 131.0,
                    "h": 132.0,
                    "l": 130.5,
                    "c": 131.5,
                    "v": 45000,
                    "n": 180,
                    "vw": 131.2,
                },
            ]
        }
        fake_resp = mock.MagicMock()
        fake_resp.json.return_value = fake_bars
        fake_resp.raise_for_status = mock.MagicMock()

        fake_config = json.dumps({"alpaca": {"key": "PK_TEST", "secret": "SK_TEST"}})
        with mock.patch("requests.get", return_value=fake_resp), mock.patch(
            "portfolio.main.CONFIG_FILE"
        ) as mock_cfg:
            mock_cfg.read_text.return_value = fake_config
            df = alpaca_klines("MSTR", interval="15m", limit=100)

        assert list(df.columns) >= ["open", "high", "low", "close", "volume", "time"]
        assert len(df) == 2
        assert df["close"].dtype == float
        assert df["volume"].dtype == float

    def test_empty_bars_raises(self):
        import unittest.mock as mock

        fake_resp = mock.MagicMock()
        fake_resp.json.return_value = {"bars": None}
        fake_resp.raise_for_status = mock.MagicMock()

        fake_config = json.dumps({"alpaca": {"key": "PK_TEST", "secret": "SK_TEST"}})
        with mock.patch("requests.get", return_value=fake_resp), mock.patch(
            "portfolio.main.CONFIG_FILE"
        ) as mock_cfg:
            mock_cfg.read_text.return_value = fake_config
            with pytest.raises(ValueError, match="No Alpaca data"):
                alpaca_klines("MSTR", interval="15m", limit=100)

    def test_unsupported_interval_raises(self):
        with pytest.raises(ValueError, match="Unsupported Alpaca interval"):
            alpaca_klines("MSTR", interval="3d", limit=100)


# --- USD/SEK via frankfurter.app ---


class TestFetchUsdSek:
    def test_parses_frankfurter_response(self):
        import unittest.mock as mock
        from portfolio.main import _fx_cache

        _fx_cache["rate"] = None
        _fx_cache["time"] = 0

        fake_resp = mock.MagicMock()
        fake_resp.json.return_value = {"rates": {"SEK": 10.85}}
        fake_resp.raise_for_status = mock.MagicMock()

        with mock.patch("portfolio.main.requests.get", return_value=fake_resp):
            rate = fetch_usd_sek()

        assert rate == 10.85

    def test_fallback_on_error(self):
        import unittest.mock as mock
        from portfolio.main import _fx_cache

        _fx_cache["rate"] = None
        _fx_cache["time"] = 0

        with mock.patch(
            "portfolio.main.requests.get", side_effect=Exception("timeout")
        ):
            rate = fetch_usd_sek()

        assert rate == 10.50


# --- Stock timeframes config ---


class TestStockTimeframes:
    def test_seven_horizons(self):
        assert len(STOCK_TIMEFRAMES) == 7

    def test_now_uses_15m(self):
        assert STOCK_TIMEFRAMES[0][0] == "Now"
        assert STOCK_TIMEFRAMES[0][1] == "15m"

    def test_has_12h_and_2d(self):
        labels = [tf[0] for tf in STOCK_TIMEFRAMES]
        assert "12h" in labels
        assert "2d" in labels

    def test_all_intervals_in_alpaca_map(self):
        for _, interval, _, _ in STOCK_TIMEFRAMES:
            assert (
                interval in ALPACA_INTERVAL_MAP
            ), f"{interval} not in ALPACA_INTERVAL_MAP"


# --- Integration tests (run locally with: pytest -m integration) ---


def _check_gpu():
    """Check if an NVIDIA GPU is available. Skip test if not."""
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            pytest.skip("No NVIDIA GPU detected (nvidia-smi failed)")
    except FileNotFoundError:
        pytest.skip("nvidia-smi not found — no GPU available")
    except subprocess.TimeoutExpired:
        pytest.skip("nvidia-smi timed out — GPU driver may be stuck")


@pytest.mark.integration
class TestIntegrationHerc2:
    def test_gpu_available(self):
        """Verify GPU is accessible and not stuck."""
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"nvidia-smi failed: {result.stderr}"
        info = result.stdout.strip()
        assert len(info) > 0, "nvidia-smi returned empty output"
        print(f"\nGPU: {info}")

    def test_ministral_gpu_inference(self):
        """Run actual CryptoTrader-LM inference on local GPU."""
        import subprocess
        import os

        _check_gpu()

        model_venv = r"Q:\models\.venv-llm\Scripts\python.exe"
        model_script = r"Q:\models\ministral_trader.py"

        if not os.path.exists(model_venv):
            pytest.skip(f"Model venv not found: {model_venv}")
        if not os.path.exists(model_script):
            pytest.skip(f"Model script not found: {model_script}")

        ctx = json.dumps(
            {
                "ticker": "BTC",
                "price_usd": 69000,
                "rsi": 45,
                "macd_hist": -2,
                "ema_bullish": False,
                "bb_position": "inside",
                "fear_greed": 30,
                "fear_greed_class": "Fear",
                "news_sentiment": "neutral",
                "timeframe_summary": "",
                "headlines": "",
            }
        )
        try:
            result = subprocess.run(
                [model_venv, model_script],
                input=ctx,
                capture_output=True,
                text=True,
                timeout=90,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(
                "GPU inference timed out after 90s — model may be stuck. "
                "Check: nvidia-smi for GPU memory, and ensure no other process holds the GPU."
            )
        assert result.returncode == 0, (
            f"Model exited with code {result.returncode}\n"
            f"stderr: {result.stderr[:500]}"
        )
        # Parse output — model may print loading messages before JSON
        stdout = result.stdout.strip()
        # Find the JSON object in output (last line or last {...})
        json_start = stdout.rfind("{")
        assert json_start >= 0, f"No JSON found in output:\n{stdout[:500]}"
        data = json.loads(stdout[json_start:])
        assert data["action"] in ("BUY", "SELL", "HOLD")
        assert data["model"] == "CryptoTrader-LM"
        assert len(data["reasoning"]) > 0

    def test_full_report(self):
        """Run --report end-to-end locally (no Telegram)."""
        import subprocess
        import os

        env = {**os.environ, "NO_TELEGRAM": "1"}
        result = subprocess.run(
            [
                r"Q:\finance-analyzer\.venv\Scripts\python.exe",
                r"Q:\finance-analyzer\portfolio\main.py",
                "--report",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            errors="replace",
            env=env,
        )
        assert result.returncode == 0, (
            f"Report exited with code {result.returncode}\n"
            f"stderr: {result.stderr[:500]}"
        )
        assert "Portfolio:" in result.stdout
        # Report should produce signal data for at least one ticker
        assert "BTC-USD" in result.stdout or "ETH-USD" in result.stdout
