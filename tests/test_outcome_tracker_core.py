"""Tests for _derive_signal_vote() and log_signal_snapshot() in outcome_tracker.

Covers all signal branches in _derive_signal_vote (RSI with adaptive thresholds,
MACD crossovers, EMA deadband, BB, Fear & Greed, sentiment, passthrough signals,
unknown signals) and log_signal_snapshot (vote passthrough, fallback derivation,
counting, entry structure, mock verification).
"""

from unittest.mock import MagicMock, patch

import pytest

from portfolio.outcome_tracker import _derive_signal_vote, log_signal_snapshot


# ---------------------------------------------------------------------------
# _derive_signal_vote — RSI (with adaptive thresholds, BUG-111 fix)
# ---------------------------------------------------------------------------


class TestDeriveSignalVoteRSI:
    """RSI signal derivation with adaptive p20/p80 thresholds."""

    def test_rsi_none_returns_hold(self):
        assert _derive_signal_vote("rsi", {}, {}) == "HOLD"

    def test_rsi_none_explicit(self):
        assert _derive_signal_vote("rsi", {"rsi": None}, {}) == "HOLD"

    def test_rsi_below_default_lower_returns_buy(self):
        """RSI=25, no adaptive thresholds -> default lower=30, so BUY."""
        assert _derive_signal_vote("rsi", {"rsi": 25}, {}) == "BUY"

    def test_rsi_above_default_upper_returns_sell(self):
        """RSI=75, no adaptive thresholds -> default upper=70, so SELL."""
        assert _derive_signal_vote("rsi", {"rsi": 75}, {}) == "SELL"

    def test_rsi_between_default_thresholds_returns_hold(self):
        """RSI=50, squarely between default 30 and 70."""
        assert _derive_signal_vote("rsi", {"rsi": 50}, {}) == "HOLD"

    def test_rsi_at_exactly_default_lower_returns_hold(self):
        """RSI=30 is NOT < 30, so HOLD."""
        assert _derive_signal_vote("rsi", {"rsi": 30}, {}) == "HOLD"

    def test_rsi_at_exactly_default_upper_returns_hold(self):
        """RSI=70 is NOT > 70, so HOLD."""
        assert _derive_signal_vote("rsi", {"rsi": 70}, {}) == "HOLD"

    def test_rsi_adaptive_lower_threshold_buy(self):
        """rsi_p20=25 sets lower threshold to 25; RSI=24 -> BUY."""
        indicators = {"rsi": 24, "rsi_p20": 25}
        assert _derive_signal_vote("rsi", indicators, {}) == "BUY"

    def test_rsi_adaptive_lower_threshold_hold(self):
        """rsi_p20=25 sets lower threshold to 25; RSI=28 -> HOLD (above threshold)."""
        indicators = {"rsi": 28, "rsi_p20": 25}
        assert _derive_signal_vote("rsi", indicators, {}) == "HOLD"

    def test_rsi_adaptive_upper_threshold_sell(self):
        """rsi_p80=65 sets upper threshold to 65; RSI=66 -> SELL."""
        indicators = {"rsi": 66, "rsi_p80": 65}
        assert _derive_signal_vote("rsi", indicators, {}) == "SELL"

    def test_rsi_adaptive_upper_threshold_hold(self):
        """rsi_p80=65 sets upper threshold to 65; RSI=64 -> HOLD."""
        indicators = {"rsi": 64, "rsi_p80": 65}
        assert _derive_signal_vote("rsi", indicators, {}) == "HOLD"

    def test_rsi_adaptive_lower_clamped_to_15(self):
        """rsi_p20=10 is clamped to max(10, 15) = 15. RSI=14 -> BUY."""
        indicators = {"rsi": 14, "rsi_p20": 10}
        assert _derive_signal_vote("rsi", indicators, {}) == "BUY"

    def test_rsi_adaptive_lower_clamped_hold_above_15(self):
        """rsi_p20=10, clamped to 15. RSI=16 -> HOLD (not below 15)."""
        indicators = {"rsi": 16, "rsi_p20": 10}
        assert _derive_signal_vote("rsi", indicators, {}) == "HOLD"

    def test_rsi_adaptive_upper_clamped_to_85(self):
        """rsi_p80=95 is clamped to min(95, 85) = 85. RSI=86 -> SELL."""
        indicators = {"rsi": 86, "rsi_p80": 95}
        assert _derive_signal_vote("rsi", indicators, {}) == "SELL"

    def test_rsi_adaptive_upper_clamped_hold_below_85(self):
        """rsi_p80=95, clamped to 85. RSI=84 -> HOLD (not above 85)."""
        indicators = {"rsi": 84, "rsi_p80": 95}
        assert _derive_signal_vote("rsi", indicators, {}) == "HOLD"

    def test_rsi_both_adaptive_thresholds(self):
        """Both rsi_p20=20 and rsi_p80=75 set. RSI=19 -> BUY."""
        indicators = {"rsi": 19, "rsi_p20": 20, "rsi_p80": 75}
        assert _derive_signal_vote("rsi", indicators, {}) == "BUY"

    def test_rsi_both_adaptive_thresholds_sell(self):
        """Both rsi_p20=20 and rsi_p80=75 set. RSI=76 -> SELL."""
        indicators = {"rsi": 76, "rsi_p20": 20, "rsi_p80": 75}
        assert _derive_signal_vote("rsi", indicators, {}) == "SELL"

    def test_rsi_both_adaptive_thresholds_hold(self):
        """Both rsi_p20=20 and rsi_p80=75 set. RSI=50 -> HOLD."""
        indicators = {"rsi": 50, "rsi_p20": 20, "rsi_p80": 75}
        assert _derive_signal_vote("rsi", indicators, {}) == "HOLD"


# ---------------------------------------------------------------------------
# _derive_signal_vote — MACD
# ---------------------------------------------------------------------------


class TestDeriveSignalVoteMACD:
    """MACD histogram crossover detection."""

    def test_macd_none_hist_returns_hold(self):
        assert _derive_signal_vote("macd", {"macd_hist": None}, {}) == "HOLD"

    def test_macd_none_hist_prev_returns_hold(self):
        assert _derive_signal_vote("macd", {"macd_hist": 1.0, "macd_hist_prev": None}, {}) == "HOLD"

    def test_macd_both_none_returns_hold(self):
        assert _derive_signal_vote("macd", {}, {}) == "HOLD"

    def test_macd_positive_crossover_returns_buy(self):
        """hist > 0 and hist_prev <= 0 -> BUY (negative-to-positive crossover)."""
        indicators = {"macd_hist": 0.5, "macd_hist_prev": -0.2}
        assert _derive_signal_vote("macd", indicators, {}) == "BUY"

    def test_macd_positive_crossover_from_zero(self):
        """hist > 0 and hist_prev == 0 -> BUY."""
        indicators = {"macd_hist": 0.1, "macd_hist_prev": 0.0}
        assert _derive_signal_vote("macd", indicators, {}) == "BUY"

    def test_macd_negative_crossover_returns_sell(self):
        """hist < 0 and hist_prev >= 0 -> SELL (positive-to-negative crossover)."""
        indicators = {"macd_hist": -0.3, "macd_hist_prev": 0.5}
        assert _derive_signal_vote("macd", indicators, {}) == "SELL"

    def test_macd_negative_crossover_from_zero(self):
        """hist < 0 and hist_prev == 0 -> SELL."""
        indicators = {"macd_hist": -0.1, "macd_hist_prev": 0.0}
        assert _derive_signal_vote("macd", indicators, {}) == "SELL"

    def test_macd_no_crossover_both_positive(self):
        """Both positive, no crossover -> HOLD."""
        indicators = {"macd_hist": 1.0, "macd_hist_prev": 0.5}
        assert _derive_signal_vote("macd", indicators, {}) == "HOLD"

    def test_macd_no_crossover_both_negative(self):
        """Both negative, no crossover -> HOLD."""
        indicators = {"macd_hist": -0.5, "macd_hist_prev": -1.0}
        assert _derive_signal_vote("macd", indicators, {}) == "HOLD"

    def test_macd_zero_hist_no_crossover(self):
        """hist == 0 is not > 0 and not < 0, so HOLD."""
        indicators = {"macd_hist": 0.0, "macd_hist_prev": -1.0}
        assert _derive_signal_vote("macd", indicators, {}) == "HOLD"


# ---------------------------------------------------------------------------
# _derive_signal_vote — EMA
# ---------------------------------------------------------------------------


class TestDeriveSignalVoteEMA:
    """EMA(9,21) crossover with deadband."""

    def test_ema_none_ema9_returns_hold(self):
        assert _derive_signal_vote("ema", {"ema21": 100}, {}) == "HOLD"

    def test_ema_none_ema21_returns_hold(self):
        assert _derive_signal_vote("ema", {"ema9": 100}, {}) == "HOLD"

    def test_ema_both_none_returns_hold(self):
        assert _derive_signal_vote("ema", {}, {}) == "HOLD"

    def test_ema_gap_above_half_pct_buy(self):
        """ema9 > ema21, gap > 0.5% -> BUY."""
        indicators = {"ema9": 101.0, "ema21": 100.0}  # 1% gap
        assert _derive_signal_vote("ema", indicators, {}) == "BUY"

    def test_ema_gap_above_half_pct_sell(self):
        """ema9 < ema21, gap > 0.5% -> SELL."""
        indicators = {"ema9": 99.0, "ema21": 100.0}  # 1% gap
        assert _derive_signal_vote("ema", indicators, {}) == "SELL"

    def test_ema_gap_below_half_pct_returns_hold(self):
        """ema9 near ema21, gap < 0.5% -> HOLD (deadband)."""
        indicators = {"ema9": 100.3, "ema21": 100.0}  # 0.3% gap
        assert _derive_signal_vote("ema", indicators, {}) == "HOLD"

    def test_ema_gap_exactly_half_pct_returns_buy(self):
        """Gap == 0.5% passes the < 0.5 check (strict less-than), so BUY."""
        indicators = {"ema9": 100.5, "ema21": 100.0}  # exactly 0.5%
        assert _derive_signal_vote("ema", indicators, {}) == "BUY"

    def test_ema_gap_just_above_half_pct_buy(self):
        """Gap just above 0.5% -> BUY."""
        indicators = {"ema9": 100.51, "ema21": 100.0}  # 0.51% gap
        assert _derive_signal_vote("ema", indicators, {}) == "BUY"

    def test_ema21_zero_returns_hold(self):
        """ema21 == 0 -> division by zero guard, gap = 0 -> HOLD."""
        indicators = {"ema9": 10.0, "ema21": 0.0}
        assert _derive_signal_vote("ema", indicators, {}) == "HOLD"


# ---------------------------------------------------------------------------
# _derive_signal_vote — Bollinger Bands
# ---------------------------------------------------------------------------


class TestDeriveSignalVoteBB:
    """Bollinger Bands position-based signal."""

    def test_bb_below_lower_returns_buy(self):
        assert _derive_signal_vote("bb", {"price_vs_bb": "below_lower"}, {}) == "BUY"

    def test_bb_above_upper_returns_sell(self):
        assert _derive_signal_vote("bb", {"price_vs_bb": "above_upper"}, {}) == "SELL"

    def test_bb_inside_returns_hold(self):
        assert _derive_signal_vote("bb", {"price_vs_bb": "inside"}, {}) == "HOLD"

    def test_bb_missing_returns_hold(self):
        assert _derive_signal_vote("bb", {}, {}) == "HOLD"

    def test_bb_unknown_value_returns_hold(self):
        assert _derive_signal_vote("bb", {"price_vs_bb": "on_lower"}, {}) == "HOLD"


# ---------------------------------------------------------------------------
# _derive_signal_vote — Fear & Greed
# ---------------------------------------------------------------------------


class TestDeriveSignalVoteFearGreed:
    """Fear & Greed index contrarian signal."""

    def test_fg_none_returns_hold(self):
        assert _derive_signal_vote("fear_greed", {}, {}) == "HOLD"

    def test_fg_explicit_none_returns_hold(self):
        assert _derive_signal_vote("fear_greed", {}, {"fear_greed": None}) == "HOLD"

    def test_fg_extreme_fear_returns_buy(self):
        """F&G <= 20 -> contrarian BUY."""
        assert _derive_signal_vote("fear_greed", {}, {"fear_greed": 5}) == "BUY"

    def test_fg_at_20_returns_buy(self):
        """F&G == 20, boundary inclusive -> BUY."""
        assert _derive_signal_vote("fear_greed", {}, {"fear_greed": 20}) == "BUY"

    def test_fg_extreme_greed_returns_sell(self):
        """F&G >= 80 -> contrarian SELL."""
        assert _derive_signal_vote("fear_greed", {}, {"fear_greed": 90}) == "SELL"

    def test_fg_at_80_returns_sell(self):
        """F&G == 80, boundary inclusive -> SELL."""
        assert _derive_signal_vote("fear_greed", {}, {"fear_greed": 80}) == "SELL"

    def test_fg_neutral_returns_hold(self):
        """F&G == 50 -> HOLD."""
        assert _derive_signal_vote("fear_greed", {}, {"fear_greed": 50}) == "HOLD"

    def test_fg_just_above_20_returns_hold(self):
        """F&G == 21 -> HOLD (not extreme fear)."""
        assert _derive_signal_vote("fear_greed", {}, {"fear_greed": 21}) == "HOLD"

    def test_fg_just_below_80_returns_hold(self):
        """F&G == 79 -> HOLD (not extreme greed)."""
        assert _derive_signal_vote("fear_greed", {}, {"fear_greed": 79}) == "HOLD"


# ---------------------------------------------------------------------------
# _derive_signal_vote — Sentiment
# ---------------------------------------------------------------------------


class TestDeriveSignalVoteSentiment:
    """CryptoBERT / TradingHero sentiment-based signal."""

    def test_sentiment_positive_high_conf_returns_buy(self):
        extra = {"sentiment": "positive", "sentiment_conf": 0.8}
        assert _derive_signal_vote("sentiment", {}, extra) == "BUY"

    def test_sentiment_negative_high_conf_returns_sell(self):
        extra = {"sentiment": "negative", "sentiment_conf": 0.6}
        assert _derive_signal_vote("sentiment", {}, extra) == "SELL"

    def test_sentiment_positive_low_conf_returns_hold(self):
        """Confidence <= 0.4 -> HOLD even with positive sentiment."""
        extra = {"sentiment": "positive", "sentiment_conf": 0.3}
        assert _derive_signal_vote("sentiment", {}, extra) == "HOLD"

    def test_sentiment_negative_low_conf_returns_hold(self):
        extra = {"sentiment": "negative", "sentiment_conf": 0.4}
        assert _derive_signal_vote("sentiment", {}, extra) == "HOLD"

    def test_sentiment_neutral_returns_hold(self):
        extra = {"sentiment": "neutral", "sentiment_conf": 0.9}
        assert _derive_signal_vote("sentiment", {}, extra) == "HOLD"

    def test_sentiment_missing_returns_hold(self):
        assert _derive_signal_vote("sentiment", {}, {}) == "HOLD"

    def test_sentiment_conf_at_boundary(self):
        """Confidence exactly 0.4 -> NOT > 0.4, so HOLD."""
        extra = {"sentiment": "positive", "sentiment_conf": 0.4}
        assert _derive_signal_vote("sentiment", {}, extra) == "HOLD"

    def test_sentiment_conf_just_above_boundary(self):
        """Confidence 0.41 -> > 0.4, so BUY with positive."""
        extra = {"sentiment": "positive", "sentiment_conf": 0.41}
        assert _derive_signal_vote("sentiment", {}, extra) == "BUY"


# ---------------------------------------------------------------------------
# _derive_signal_vote — Passthrough signals
# ---------------------------------------------------------------------------


class TestDeriveSignalVotePassthrough:
    """Signals that pass through from extra dict with a default of HOLD."""

    def test_ministral_passthrough_buy(self):
        assert _derive_signal_vote("ministral", {}, {"ministral_action": "BUY"}) == "BUY"

    def test_ministral_passthrough_sell(self):
        assert _derive_signal_vote("ministral", {}, {"ministral_action": "SELL"}) == "SELL"

    def test_ministral_default_hold(self):
        assert _derive_signal_vote("ministral", {}, {}) == "HOLD"

    def test_ml_passthrough_buy(self):
        assert _derive_signal_vote("ml", {}, {"ml_action": "BUY"}) == "BUY"

    def test_ml_default_hold(self):
        assert _derive_signal_vote("ml", {}, {}) == "HOLD"

    def test_funding_passthrough_sell(self):
        assert _derive_signal_vote("funding", {}, {"funding_action": "SELL"}) == "SELL"

    def test_funding_default_hold(self):
        assert _derive_signal_vote("funding", {}, {}) == "HOLD"

    def test_volume_passthrough_buy(self):
        assert _derive_signal_vote("volume", {}, {"volume_action": "BUY"}) == "BUY"

    def test_volume_default_hold(self):
        assert _derive_signal_vote("volume", {}, {}) == "HOLD"


# ---------------------------------------------------------------------------
# _derive_signal_vote — Unknown signal
# ---------------------------------------------------------------------------


class TestDeriveSignalVoteUnknown:
    """Unrecognized signal names should default to HOLD."""

    def test_unknown_signal_returns_hold(self):
        assert _derive_signal_vote("totally_made_up", {}, {}) == "HOLD"

    def test_enhanced_signal_not_derived_returns_hold(self):
        """Enhanced signals like 'trend' are not derived in _derive_signal_vote."""
        assert _derive_signal_vote("trend", {}, {}) == "HOLD"

    def test_empty_name_returns_hold(self):
        assert _derive_signal_vote("", {}, {}) == "HOLD"


# ---------------------------------------------------------------------------
# log_signal_snapshot tests
# ---------------------------------------------------------------------------


class TestLogSignalSnapshot:
    """Tests for log_signal_snapshot entry construction and writing."""

    @patch("portfolio.outcome_tracker.atomic_append_jsonl")
    @patch("portfolio.outcome_tracker.SignalDB", create=True)
    def test_uses_passed_votes_when_available(self, mock_db_cls, mock_append):
        """When _votes is present in extra, signals come from _votes, not derivation."""
        mock_db_cls.side_effect = ImportError("no db")

        signals_dict = {
            "BTC-USD": {
                "indicators": {"rsi": 25},  # would be BUY if derived
                "extra": {
                    "_votes": {"rsi": "SELL", "macd": "BUY"},  # override
                },
                "action": "HOLD",
            }
        }
        prices_usd = {"BTC-USD": 67000.0}
        entry = log_signal_snapshot(signals_dict, prices_usd, 10.5, ["test"])

        btc = entry["tickers"]["BTC-USD"]
        # RSI should be SELL (from _votes), not BUY (which derivation would give)
        assert btc["signals"]["rsi"] == "SELL"
        assert btc["signals"]["macd"] == "BUY"
        # Signals not in _votes default to HOLD
        assert btc["signals"]["ema"] == "HOLD"

    @patch("portfolio.outcome_tracker.atomic_append_jsonl")
    @patch("portfolio.outcome_tracker.SignalDB", create=True)
    def test_falls_back_to_derive_when_votes_missing(self, mock_db_cls, mock_append):
        """When _votes is not present, signals are derived from indicators/extra."""
        mock_db_cls.side_effect = ImportError("no db")

        signals_dict = {
            "ETH-USD": {
                "indicators": {"rsi": 25, "price_vs_bb": "below_lower"},
                "extra": {"volume_action": "BUY"},
                "action": "BUY",
            }
        }
        prices_usd = {"ETH-USD": 2000.0}
        entry = log_signal_snapshot(signals_dict, prices_usd, 10.5, ["consensus"])

        eth = entry["tickers"]["ETH-USD"]
        assert eth["signals"]["rsi"] == "BUY"  # RSI 25 < default 30
        assert eth["signals"]["bb"] == "BUY"  # below_lower
        assert eth["signals"]["volume"] == "BUY"  # from extra
        assert eth["signals"]["macd"] == "HOLD"  # no data -> HOLD

    @patch("portfolio.outcome_tracker.atomic_append_jsonl")
    @patch("portfolio.outcome_tracker.SignalDB", create=True)
    def test_correct_buy_sell_count(self, mock_db_cls, mock_append):
        """buy_count and sell_count match the actual signal values."""
        mock_db_cls.side_effect = ImportError("no db")

        # Use _votes to precisely control signal values
        votes = {name: "HOLD" for name in [
            "rsi", "macd", "ema", "bb", "fear_greed", "sentiment",
            "ministral", "ml", "funding", "volume", "qwen3",
            "trend", "momentum", "volume_flow", "volatility_sig",
            "candlestick", "structure", "fibonacci", "smart_money",
            "oscillators", "heikin_ashi", "mean_reversion", "calendar",
            "macro_regime", "momentum_factors", "news_event",
            "econ_calendar", "forecast", "claude_fundamental", "futures_flow",
        ]}
        votes["rsi"] = "BUY"
        votes["macd"] = "BUY"
        votes["ema"] = "SELL"

        signals_dict = {
            "XAG-USD": {
                "indicators": {},
                "extra": {"_votes": votes},
                "action": "BUY",
            }
        }
        entry = log_signal_snapshot(signals_dict, {"XAG-USD": 30.0}, 10.5, ["test"])
        xag = entry["tickers"]["XAG-USD"]
        assert xag["buy_count"] == 2
        assert xag["sell_count"] == 1
        assert xag["total_voters"] == 3

    @patch("portfolio.outcome_tracker.atomic_append_jsonl")
    @patch("portfolio.outcome_tracker.SignalDB", create=True)
    def test_price_from_prices_usd_preferred(self, mock_db_cls, mock_append):
        """Price in prices_usd dict is preferred over indicators.close."""
        mock_db_cls.side_effect = ImportError("no db")

        signals_dict = {
            "BTC-USD": {
                "indicators": {"close": 60000.0},
                "extra": {},
                "action": "HOLD",
            }
        }
        prices_usd = {"BTC-USD": 67500.0}
        entry = log_signal_snapshot(signals_dict, prices_usd, 10.5, ["test"])
        assert entry["tickers"]["BTC-USD"]["price_usd"] == 67500.0

    @patch("portfolio.outcome_tracker.atomic_append_jsonl")
    @patch("portfolio.outcome_tracker.SignalDB", create=True)
    def test_price_falls_back_to_indicator_close(self, mock_db_cls, mock_append):
        """When prices_usd has no entry, fall back to indicators['close']."""
        mock_db_cls.side_effect = ImportError("no db")

        signals_dict = {
            "NVDA": {
                "indicators": {"close": 185.0},
                "extra": {},
                "action": "HOLD",
            }
        }
        prices_usd = {}  # no NVDA price
        entry = log_signal_snapshot(signals_dict, prices_usd, 10.5, ["test"])
        assert entry["tickers"]["NVDA"]["price_usd"] == 185.0

    @patch("portfolio.outcome_tracker.atomic_append_jsonl")
    @patch("portfolio.outcome_tracker.SignalDB", create=True)
    def test_entry_structure_has_required_keys(self, mock_db_cls, mock_append):
        """Entry must contain ts, trigger_reasons, fx_rate, tickers, outcomes."""
        mock_db_cls.side_effect = ImportError("no db")

        signals_dict = {
            "BTC-USD": {
                "indicators": {},
                "extra": {},
                "action": "HOLD",
            }
        }
        entry = log_signal_snapshot(signals_dict, {"BTC-USD": 67000}, 10.5, ["price_move"])

        assert "ts" in entry
        assert entry["trigger_reasons"] == ["price_move"]
        assert entry["fx_rate"] == 10.5
        assert "tickers" in entry
        assert "BTC-USD" in entry["tickers"]
        assert entry["outcomes"] == {}

    @patch("portfolio.outcome_tracker.atomic_append_jsonl")
    @patch("portfolio.outcome_tracker.SignalDB", create=True)
    def test_consensus_from_sig_data_action(self, mock_db_cls, mock_append):
        """Consensus should come from sig_data['action'], not derived."""
        mock_db_cls.side_effect = ImportError("no db")

        signals_dict = {
            "ETH-USD": {
                "indicators": {},
                "extra": {},
                "action": "SELL",
            }
        }
        entry = log_signal_snapshot(signals_dict, {"ETH-USD": 2000}, 10.5, ["test"])
        assert entry["tickers"]["ETH-USD"]["consensus"] == "SELL"

    @patch("portfolio.outcome_tracker.atomic_append_jsonl")
    @patch("portfolio.outcome_tracker.SignalDB", create=True)
    def test_atomic_append_called(self, mock_db_cls, mock_append):
        """Verify atomic_append_jsonl is called with the SIGNAL_LOG path and entry."""
        mock_db_cls.side_effect = ImportError("no db")

        signals_dict = {
            "BTC-USD": {
                "indicators": {},
                "extra": {},
                "action": "HOLD",
            }
        }
        entry = log_signal_snapshot(signals_dict, {"BTC-USD": 67000}, 10.5, ["test"])

        mock_append.assert_called_once()
        call_args = mock_append.call_args
        # First positional arg is the log path, second is the entry dict
        assert call_args[0][1] is entry

    @patch("portfolio.outcome_tracker.atomic_append_jsonl")
    @patch("portfolio.outcome_tracker.SignalDB", create=True)
    def test_multiple_tickers(self, mock_db_cls, mock_append):
        """Snapshot with multiple tickers produces entries for each."""
        mock_db_cls.side_effect = ImportError("no db")

        signals_dict = {
            "BTC-USD": {
                "indicators": {"rsi": 25},
                "extra": {},
                "action": "BUY",
            },
            "XAG-USD": {
                "indicators": {"rsi": 75},
                "extra": {},
                "action": "SELL",
            },
        }
        prices_usd = {"BTC-USD": 67000, "XAG-USD": 30.0}
        entry = log_signal_snapshot(signals_dict, prices_usd, 10.5, ["consensus"])

        assert "BTC-USD" in entry["tickers"]
        assert "XAG-USD" in entry["tickers"]
        assert entry["tickers"]["BTC-USD"]["consensus"] == "BUY"
        assert entry["tickers"]["XAG-USD"]["consensus"] == "SELL"
        # BTC RSI 25 < 30 -> BUY derived
        assert entry["tickers"]["BTC-USD"]["signals"]["rsi"] == "BUY"
        # XAG RSI 75 > 70 -> SELL derived
        assert entry["tickers"]["XAG-USD"]["signals"]["rsi"] == "SELL"

    @patch("portfolio.outcome_tracker.atomic_append_jsonl")
    @patch("portfolio.outcome_tracker.SignalDB", create=True)
    def test_consensus_defaults_to_hold(self, mock_db_cls, mock_append):
        """If sig_data has no 'action' key, consensus defaults to HOLD."""
        mock_db_cls.side_effect = ImportError("no db")

        signals_dict = {
            "BTC-USD": {
                "indicators": {},
                "extra": {},
                # no "action" key
            }
        }
        entry = log_signal_snapshot(signals_dict, {"BTC-USD": 67000}, 10.5, ["test"])
        assert entry["tickers"]["BTC-USD"]["consensus"] == "HOLD"


# ---------------------------------------------------------------------------
# REF-20: Module-level logger (no more function-local import logging)
# ---------------------------------------------------------------------------


class TestModuleLogger:
    """Verify outcome_tracker uses a module-level logger."""

    def test_has_module_level_logger(self):
        import portfolio.outcome_tracker as ot
        import logging
        assert hasattr(ot, "logger")
        assert isinstance(ot.logger, logging.Logger)
        assert ot.logger.name == "portfolio.outcome_tracker"

    def test_no_function_local_logging_import(self):
        """Ensure no function bodies contain 'import logging as _logging'."""
        import inspect
        import portfolio.outcome_tracker as ot
        source = inspect.getsource(ot)
        assert "import logging as _logging" not in source
