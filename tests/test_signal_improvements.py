"""Tests for signal quality improvements (Phases 1-4).

Covers:
- ATR in compute_indicators
- Named votes in generate_signal
- log_signal_snapshot using passed votes
- ATR in agent_summary
- detect_regime classification
- Accuracy cache read/write with TTL
- Weighted consensus with accuracy data
- Regime-adaptive signal weights
- Confluence scoring
- Adaptive RSI thresholds
- Time-of-day confidence factor
- Cross-asset lead/lag detection
- Agent summary includes all new fields
"""

import json
import time
import unittest.mock as mock

import numpy as np
import pandas as pd
import pytest

from portfolio.main import (
    CRYPTO_SYMBOLS,
    STOCK_SYMBOLS,
    REGIME_WEIGHTS,
    compute_indicators,
    detect_regime,
    generate_signal,
    _weighted_consensus,
    _confluence_score,
    _time_of_day_factor,
    _cross_asset_signals,
    write_agent_summary,
)


def _null_cached(key, ttl, func, *args):
    return None


def make_df(n=100, close_base=100.0, volatility=2.0):
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = close_base + np.cumsum(np.random.randn(n) * volatility)
    close = np.maximum(close, 1.0)
    high = close + np.abs(np.random.randn(n) * volatility)
    low = close - np.abs(np.random.randn(n) * volatility)
    low = np.maximum(low, 0.5)
    opn = close + np.random.randn(n) * 0.5
    volume = np.random.randint(100, 10000, n).astype(float)
    return pd.DataFrame(
        {
            "open": opn,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "time": dates,
        }
    )


def make_indicators(**overrides):
    base = {
        "close": 130.0,
        "rsi": 50.0,
        "macd_hist": 0.0,
        "macd_hist_prev": 0.0,
        "ema9": 130.0,
        "ema21": 130.0,
        "bb_upper": 135.0,
        "bb_lower": 125.0,
        "bb_mid": 130.0,
        "price_vs_bb": "inside",
        "atr": 3.0,
        "atr_pct": 2.3,
        "rsi_p20": 30.0,
        "rsi_p80": 70.0,
    }
    base.update(overrides)
    return base


# --- Phase 1 Tests ---


class TestComputeIndicatorsATR:
    def test_atr_present_and_positive(self):
        df = make_df()
        ind = compute_indicators(df)
        assert ind is not None
        assert "atr" in ind
        assert "atr_pct" in ind
        assert ind["atr"] > 0
        assert ind["atr_pct"] > 0

    def test_atr_pct_is_percentage(self):
        df = make_df(close_base=100.0)
        ind = compute_indicators(df)
        assert ind["atr_pct"] < 100  # should be a small percentage

    def test_rsi_percentiles_present(self):
        df = make_df()
        ind = compute_indicators(df)
        assert "rsi_p20" in ind
        assert "rsi_p80" in ind


class TestNamedVotes:
    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_votes_dict_has_signal_names(self, _mock):
        ind = make_indicators()
        _, _, extra = generate_signal(ind, ticker="MSTR")
        votes = extra["_votes"]
        for name in [
            "rsi",
            "macd",
            "ema",
            "bb",
            "fear_greed",
            "sentiment",
            "ml",
            "funding",
            "volume",
        ]:
            assert name in votes
            assert votes[name] in ("BUY", "SELL", "HOLD")

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_votes_dict_crypto_has_all_25(self, _mock):
        ind = make_indicators(close=69000.0)
        _, _, extra = generate_signal(ind, ticker="BTC-USD")
        votes = extra["_votes"]
        assert "ministral" in votes
        # custom_lora fully disabled (20.9% accuracy, 97% SELL bias)
        assert "custom_lora" not in votes
        # 10 core (11 - custom_lora) + 19 enhanced composite signals (incl. forecast + claude_fundamental + futures_flow) = 29
        assert len(votes) == 29

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_buy_count_matches_votes(self, _mock):
        ind = make_indicators(rsi=25, macd_hist=1.0, macd_hist_prev=-1.0)
        _, _, extra = generate_signal(ind, ticker="MSTR")
        votes = extra["_votes"]
        buy_from_votes = sum(1 for v in votes.values() if v == "BUY")
        assert extra["_buy_count"] == buy_from_votes


class TestLogSignalSnapshotUsesPassedVotes:
    def test_uses_passed_votes_when_available(self):
        from portfolio.outcome_tracker import log_signal_snapshot, SIGNAL_NAMES

        passed_votes = {name: "HOLD" for name in SIGNAL_NAMES}
        passed_votes["rsi"] = "BUY"
        passed_votes["ema"] = "SELL"

        signals_dict = {
            "BTC-USD": {
                "action": "HOLD",
                "indicators": {
                    "close": 69000,
                    "rsi": 50,
                    "macd_hist": 0,
                    "macd_hist_prev": 0,
                    "ema9": 69000,
                    "ema21": 69000,
                    "price_vs_bb": "inside",
                },
                "extra": {"_votes": passed_votes},
            }
        }
        with mock.patch("portfolio.outcome_tracker.open", mock.mock_open()):
            entry = log_signal_snapshot(
                signals_dict, {"BTC-USD": 69000}, 10.5, ["test"]
            )

        logged = entry["tickers"]["BTC-USD"]["signals"]
        assert logged["rsi"] == "BUY"
        assert logged["ema"] == "SELL"
        assert logged["macd"] == "HOLD"


class TestAgentSummaryATR:
    @mock.patch("portfolio.reporting._atomic_write_json")
    @mock.patch("portfolio.reporting._cached", side_effect=_null_cached)
    def test_atr_in_agent_summary(self, _mock_cached, _mock_write):
        ind = make_indicators()
        signals = {
            "MSTR": {
                "action": "HOLD",
                "confidence": 0.0,
                "indicators": ind,
                "extra": {},
            }
        }
        state = {
            "cash_sek": 500000,
            "holdings": {},
            "transactions": [],
            "initial_value_sek": 500000,
            "start_date": "2024-01-01",
        }
        summary = write_agent_summary(signals, {"MSTR": 130.0}, 10.5, state, {})
        sig = summary["signals"]["MSTR"]
        assert "atr" in sig
        assert "atr_pct" in sig
        assert sig["atr"] > 0


# --- Phase 2 Tests ---


class TestDetectRegime:
    def test_trending_up(self):
        ind = make_indicators(ema9=135.0, ema21=130.0, rsi=55, atr_pct=2.0)
        assert detect_regime(ind, is_crypto=True) == "trending-up"

    def test_trending_down(self):
        ind = make_indicators(ema9=125.0, ema21=130.0, rsi=45, atr_pct=2.0)
        assert detect_regime(ind, is_crypto=True) == "trending-down"

    def test_ranging(self):
        ind = make_indicators(ema9=130.0, ema21=130.0, rsi=50, atr_pct=2.0)
        assert detect_regime(ind, is_crypto=True) == "ranging"

    def test_high_vol_crypto(self):
        ind = make_indicators(atr_pct=5.0)
        assert detect_regime(ind, is_crypto=True) == "high-vol"

    def test_high_vol_stock(self):
        ind = make_indicators(atr_pct=3.5)
        assert detect_regime(ind, is_crypto=False) == "high-vol"

    def test_stock_not_high_vol_at_3(self):
        ind = make_indicators(atr_pct=2.5)
        assert detect_regime(ind, is_crypto=False) != "high-vol"


class TestAccuracyCache:
    def test_cache_written_and_read(self, tmp_path):
        from portfolio.accuracy_stats import (
            load_cached_accuracy,
            write_accuracy_cache,
            ACCURACY_CACHE_FILE,
        )

        import portfolio.accuracy_stats as acc_mod

        orig = acc_mod.ACCURACY_CACHE_FILE
        acc_mod.ACCURACY_CACHE_FILE = tmp_path / "cache.json"
        try:
            data = {"rsi": {"accuracy": 0.65, "total": 30}}
            write_accuracy_cache("1d", data)
            result = load_cached_accuracy("1d")
            assert result == data
        finally:
            acc_mod.ACCURACY_CACHE_FILE = orig

    def test_cache_ttl_expired(self, tmp_path):
        from portfolio.accuracy_stats import load_cached_accuracy, ACCURACY_CACHE_FILE

        import portfolio.accuracy_stats as acc_mod

        orig = acc_mod.ACCURACY_CACHE_FILE
        acc_mod.ACCURACY_CACHE_FILE = tmp_path / "cache.json"
        try:
            expired = {
                "1d": {"rsi": {"accuracy": 0.5, "total": 10}},
                "time": time.time() - 7200,
            }
            (tmp_path / "cache.json").write_text(json.dumps(expired))
            result = load_cached_accuracy("1d")
            assert result is None
        finally:
            acc_mod.ACCURACY_CACHE_FILE = orig


# --- Phase 3 Tests ---


class TestWeightedConsensus:
    def test_high_accuracy_signal_dominates(self):
        votes = {"rsi": "BUY", "macd": "SELL", "ema": "HOLD"}
        acc = {
            "rsi": {"accuracy": 0.9, "total": 50},
            "macd": {"accuracy": 0.4, "total": 50},
        }
        action, conf = _weighted_consensus(votes, acc, "ranging")
        assert action == "BUY"

    def test_low_sample_uses_neutral_weight(self):
        votes = {"funding": "BUY", "volume": "SELL"}
        acc = {
            "funding": {"accuracy": 0.9, "total": 5},  # too few samples
            "volume": {"accuracy": 0.9, "total": 5},
        }
        action, conf = _weighted_consensus(votes, acc, "ranging")
        # Both get weight 0.5 (neutral), no regime mults → tie → HOLD
        assert action == "HOLD"

    def test_all_hold_returns_hold(self):
        votes = {"rsi": "HOLD", "macd": "HOLD"}
        action, conf = _weighted_consensus(votes, {}, "ranging")
        assert action == "HOLD"
        assert conf == 0.0


class TestRegimeWeights:
    def test_trend_signals_boosted_in_trending(self):
        votes = {"ema": "BUY", "rsi": "SELL"}
        acc = {
            "ema": {"accuracy": 0.6, "total": 50},
            "rsi": {"accuracy": 0.6, "total": 50},
        }
        action, _ = _weighted_consensus(votes, acc, "trending-up")
        # EMA gets 1.5x, RSI gets 0.7x → BUY should win
        assert action == "BUY"

    def test_reversion_signals_boosted_in_ranging(self):
        votes = {"rsi": "BUY", "ema": "SELL"}
        acc = {
            "rsi": {"accuracy": 0.6, "total": 50},
            "ema": {"accuracy": 0.6, "total": 50},
        }
        action, _ = _weighted_consensus(votes, acc, "ranging")
        # RSI gets 1.5x, EMA gets 0.5x → BUY should win
        assert action == "BUY"


class TestConfluenceScore:
    def test_unanimous_agreement(self):
        votes = {"rsi": "BUY", "macd": "BUY", "ema": "BUY"}
        score = _confluence_score(votes, {})
        assert score == 1.0

    def test_mixed_signals(self):
        votes = {"rsi": "BUY", "macd": "SELL", "ema": "BUY"}
        score = _confluence_score(votes, {})
        # 2/3 agree → 0.6667
        assert 0.6 < score < 0.7

    def test_all_hold_returns_zero(self):
        votes = {"rsi": "HOLD", "macd": "HOLD"}
        score = _confluence_score(votes, {})
        assert score == 0.0

    def test_volume_confirmation_bonus(self):
        votes = {"rsi": "BUY", "macd": "BUY", "ema": "SELL"}
        score_without = _confluence_score(votes, {})
        score_with = _confluence_score(votes, {"volume_action": "BUY"})
        assert score_with > score_without


class TestAdaptiveRSI:
    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_uses_percentiles(self, _mock):
        # RSI at 25, but adaptive lower bound is 20 → should NOT trigger BUY
        ind = make_indicators(rsi=25, rsi_p20=20.0, rsi_p80=75.0)
        _, _, extra = generate_signal(ind, ticker="MSTR")
        assert extra["_votes"]["rsi"] == "HOLD"

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_fallback_when_insufficient_data(self, _mock):
        # rsi_p20 defaults to 30 when not available
        ind = make_indicators(rsi=25)
        del ind["rsi_p20"]
        del ind["rsi_p80"]
        _, _, extra = generate_signal(ind, ticker="MSTR")
        # Falls back to 30/70, RSI 25 < 30 → BUY
        assert extra["_votes"]["rsi"] == "BUY"

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_floor_ceiling_enforced(self, _mock):
        # Even if percentile says lower=10, we clamp to 15
        ind = make_indicators(rsi=14, rsi_p20=10.0, rsi_p80=90.0)
        _, _, extra = generate_signal(ind, ticker="MSTR")
        # rsi_lower clamped to 15, RSI 14 < 15 → BUY
        assert extra["_votes"]["rsi"] == "BUY"


class TestTimeOfDayFactor:
    def test_quiet_hours(self):
        with mock.patch("portfolio.signal_engine.datetime") as mock_dt:
            mock_now = mock.MagicMock()
            mock_now.hour = 3
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            factor = _time_of_day_factor()
            assert factor == 0.8

    def test_active_hours(self):
        with mock.patch("portfolio.signal_engine.datetime") as mock_dt:
            mock_now = mock.MagicMock()
            mock_now.hour = 14
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            factor = _time_of_day_factor()
            assert factor == 1.0


# --- Phase 4 Tests ---


class TestCrossAssetSignals:
    def test_btc_leads_eth(self):
        signals = {
            "BTC-USD": {"action": "BUY"},
            "ETH-USD": {"action": "HOLD"},
            "MSTR": {"action": "HOLD"},
        }
        leads = _cross_asset_signals(signals)
        assert "ETH-USD" in leads
        assert leads["ETH-USD"]["leader"] == "BTC-USD"
        assert leads["ETH-USD"]["leader_action"] == "BUY"

    def test_btc_leads_eth(self):
        signals = {
            "BTC-USD": {"action": "SELL"},
            "ETH-USD": {"action": "HOLD"},
        }
        leads = _cross_asset_signals(signals)
        assert "ETH-USD" in leads

    def test_no_lead_when_btc_hold(self):
        signals = {
            "BTC-USD": {"action": "HOLD"},
            "ETH-USD": {"action": "BUY"},
        }
        leads = _cross_asset_signals(signals)
        assert leads == {}


class TestAgentSummaryAllFields:
    @mock.patch("portfolio.reporting._atomic_write_json")
    @mock.patch("portfolio.reporting._cached", side_effect=_null_cached)
    def test_includes_all_new_fields(self, _mock_cached, _mock_write):
        ind = make_indicators()
        signals = {
            "BTC-USD": {
                "action": "BUY",
                "confidence": 0.8,
                "indicators": ind,
                "extra": {
                    "_weighted_confidence": 0.75,
                    "_confluence_score": 0.9,
                    "_votes": {"rsi": "BUY"},
                },
            }
        }
        state = {
            "cash_sek": 500000,
            "holdings": {},
            "transactions": [],
            "initial_value_sek": 500000,
            "start_date": "2024-01-01",
        }
        summary = write_agent_summary(signals, {"BTC-USD": 69000}, 10.5, state, {})
        sig = summary["signals"]["BTC-USD"]
        assert "regime" in sig
        assert "atr" in sig
        assert "atr_pct" in sig
        assert "weighted_confidence" in sig
        assert "confluence_score" in sig
        assert sig["regime"] in ("trending-up", "trending-down", "ranging", "high-vol")
