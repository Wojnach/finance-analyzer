"""Tests for portfolio.analyze — deep instrument analysis module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from portfolio.analyze import (
    _build_analysis_prompt,
    _build_watch_prompt,
    _get_holdings,
    _load_journal_for_ticker,
    _parse_positions,
    _parse_watch_response,
    _should_call_claude,
    run_analysis,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SUMMARY = {
    "signals": {
        "ETH-USD": {
            "action": "SELL",
            "confidence": 0.625,
            "weighted_confidence": 0.551,
            "confluence_score": 0.725,
            "price_usd": 1921.46,
            "rsi": 23.1,
            "macd_hist": -4.13,
            "bb_position": "below_lower",
            "atr": 10.1,
            "atr_pct": 0.53,
            "regime": "ranging",
            "extra": {
                "fear_greed": 9,
                "fear_greed_class": "Extreme Fear",
                "sentiment": "neutral",
                "sentiment_conf": 0.61,
                "sentiment_model": "CryptoBERT",
                "ml_action": "SELL",
                "ml_confidence": 0.45,
                "funding_rate": -0.0019,
                "funding_action": "HOLD",
                "volume_ratio": 4.18,
                "volume_action": "SELL",
                "ministral_action": "SELL",
                "ministral_reasoning": "Bearish trend",
                "custom_lora_action": "SELL",
                "custom_lora_reasoning": "EMA crossover bearish",
                "_voters": 8,
                "_total_applicable": 11,
                "_buy_count": 3,
                "_sell_count": 5,
                "_votes": {
                    "rsi": "BUY", "macd": "HOLD", "ema": "SELL",
                    "bb": "BUY", "fear_greed": "BUY", "sentiment": "HOLD",
                    "ml": "SELL", "funding": "HOLD", "volume": "SELL",
                    "ministral": "SELL", "custom_lora": "SELL",
                },
                "_regime": "ranging",
                "_weighted_action": "SELL",
                "_weighted_confidence": 0.551,
                "_confluence_score": 0.725,
            },
        },
        "NVDA": {
            "action": "BUY",
            "confidence": 1.0,
            "weighted_confidence": 1.0,
            "confluence_score": 1.0,
            "price_usd": 187.81,
            "rsi": 49.8,
            "macd_hist": -0.27,
            "bb_position": "inside",
            "atr": 0.68,
            "atr_pct": 0.36,
            "regime": "ranging",
            "extra": {
                "fear_greed": 49,
                "fear_greed_class": "Neutral",
                "volume_ratio": 2.84,
                "volume_action": "BUY",
                "_voters": 3,
                "_total_applicable": 7,
                "_buy_count": 3,
                "_sell_count": 0,
                "_votes": {
                    "rsi": "BUY", "macd": "HOLD", "ema": "HOLD",
                    "bb": "HOLD", "fear_greed": "HOLD", "sentiment": "BUY",
                    "ml": "HOLD", "funding": "HOLD", "volume": "BUY",
                    "ministral": "HOLD", "custom_lora": "HOLD",
                },
                "_regime": "ranging",
                "_weighted_action": "BUY",
                "_weighted_confidence": 1.0,
                "_confluence_score": 1.0,
            },
        },
    },
    "timeframes": {
        "ETH-USD": [
            {"horizon": "Now", "action": "SELL", "confidence": 0.625, "rsi": 23.1, "macd_hist": -4.13, "ema_bullish": False, "bb_position": "below_lower"},
            {"horizon": "12h", "action": "SELL", "confidence": 0.75, "rsi": 29.4, "macd_hist": -3.05, "ema_bullish": False, "bb_position": "below_lower"},
            {"horizon": "2d", "action": "SELL", "confidence": 0.75, "rsi": 37.7, "macd_hist": -4.34, "ema_bullish": False, "bb_position": "below_lower"},
        ],
        "NVDA": [
            {"horizon": "Now", "action": "BUY", "confidence": 1.0, "rsi": 49.8, "macd_hist": -0.27, "ema_bullish": False, "bb_position": "inside"},
        ],
    },
    "fear_greed": {
        "ETH-USD": {"value": 9, "classification": "Extreme Fear"},
        "NVDA": {"value": 49, "classification": "Neutral"},
    },
    "macro": {
        "dxy": {"value": 97.92, "sma20": 97.2, "trend": "strong", "change_5d_pct": 1.02},
        "treasury": {"10y": {"yield_pct": 4.079, "change_5d": -2.23}, "spread_2s10s": 0.631, "curve": "normal"},
        "fed": {"next_fomc": "2026-03-17", "days_until": 26},
    },
    "signal_accuracy_1d": {
        "signals": {
            "rsi": {"accuracy": 0.406, "samples": 101},
            "macd": {"accuracy": 0.662, "samples": 154},
            "ema": {"accuracy": 0.4, "samples": 2030},
            "bb": {"accuracy": 0.432, "samples": 162},
            "fear_greed": {"accuracy": 0.561, "samples": 1078},
            "sentiment": {"accuracy": 0.344, "samples": 427},
            "ml": {"accuracy": 0.66, "samples": 612},
            "funding": {"accuracy": 0.88, "samples": 25},
            "volume": {"accuracy": 0.48, "samples": 893},
            "ministral": {"accuracy": 0.6, "samples": 936},
            "custom_lora": {"accuracy": 0.29, "samples": 582},
        },
    },
}


# ---------------------------------------------------------------------------
# _build_analysis_prompt tests
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    """Tests for _build_analysis_prompt."""

    def test_build_prompt_contains_ticker(self):
        """Prompt includes ticker name, price, signal vote breakdown, and OUTLOOK."""
        prompt = _build_analysis_prompt("ETH-USD", SAMPLE_SUMMARY)
        assert prompt is not None
        assert "ETH-USD" in prompt
        assert "1,921.46" in prompt
        assert "3B/5S/3H" in prompt
        assert "OUTLOOK:" in prompt

    def test_build_prompt_includes_macro(self):
        """Prompt includes DXY value, 10Y yield, and FOMC days."""
        prompt = _build_analysis_prompt("ETH-USD", SAMPLE_SUMMARY)
        assert "97.92" in prompt
        assert "4.079" in prompt
        assert "26" in prompt

    def test_build_prompt_includes_timeframes(self):
        """Prompt contains timeframe horizons and B/S/H tags."""
        prompt = _build_analysis_prompt("ETH-USD", SAMPLE_SUMMARY)
        assert "Now" in prompt
        assert "12h" in prompt
        assert "2d" in prompt
        # ETH-USD timeframes are all SELL, so tags should be S
        # Check the row contains S tags (the heatmap row)
        lines = prompt.splitlines()
        # Find the row that starts with the ticker abbreviation
        tf_row = [l for l in lines if l.strip().startswith("ETH-U") or l.strip().startswith("ETH")]
        assert len(tf_row) > 0
        row_text = tf_row[0]
        assert "S" in row_text  # All timeframes are SELL -> S

    def test_build_prompt_crypto_has_ministral(self):
        """Crypto ticker prompt includes Ministral line."""
        prompt = _build_analysis_prompt("ETH-USD", SAMPLE_SUMMARY)
        assert "Ministral:" in prompt

    def test_build_prompt_stock_no_ministral(self):
        """Stock ticker prompt does NOT include Ministral or LoRA lines."""
        prompt = _build_analysis_prompt("NVDA", SAMPLE_SUMMARY)
        assert prompt is not None
        assert "Ministral:" not in prompt
        assert "LoRA:" not in prompt

    def test_build_prompt_includes_accuracy(self):
        """Prompt contains accuracy percentages (RSI 0.406 -> 41%)."""
        prompt = _build_analysis_prompt("ETH-USD", SAMPLE_SUMMARY)
        # RSI accuracy 0.406 rounds to 41%
        assert "41%" in prompt
        # MACD accuracy 0.662 rounds to 66%
        assert "66%" in prompt

    def test_build_prompt_includes_confluence(self):
        """Prompt contains confluence percentage (ETH 0.725 -> 72%)."""
        prompt = _build_analysis_prompt("ETH-USD", SAMPLE_SUMMARY)
        assert "Confluence:" in prompt
        assert "72%" in prompt

    def test_build_prompt_missing_signal_returns_none(self):
        """Unknown ticker returns None from _build_analysis_prompt."""
        result = _build_analysis_prompt("FAKE-TICKER", SAMPLE_SUMMARY)
        assert result is None


# ---------------------------------------------------------------------------
# _load_journal_for_ticker tests
# ---------------------------------------------------------------------------

class TestLoadJournal:
    """Tests for _load_journal_for_ticker."""

    def test_load_journal_returns_matching_entries(self, tmp_path, monkeypatch):
        """Returns journal entries where the ticker has a non-neutral outlook."""
        journal = tmp_path / "layer2_journal.jsonl"
        entries = [
            {"ts": "2026-02-18T10:00:00Z", "tickers": {"ETH-USD": {"outlook": "bearish", "thesis": "Weak"}}, "prices": {"ETH-USD": 1900}},
            {"ts": "2026-02-18T12:00:00Z", "tickers": {"ETH-USD": {"outlook": "neutral", "thesis": ""}}, "prices": {"ETH-USD": 1910}},
            {"ts": "2026-02-18T14:00:00Z", "tickers": {"ETH-USD": {"outlook": "bullish", "thesis": "Recovery"}}, "prices": {"ETH-USD": 1930}},
        ]
        journal.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")
        monkeypatch.setattr("portfolio.analyze.JOURNAL_FILE", journal)

        result = _load_journal_for_ticker("ETH-USD", max_entries=5)
        # Only non-neutral entries returned
        assert len(result) == 2
        assert result[0]["tickers"]["ETH-USD"]["outlook"] == "bearish"
        assert result[1]["tickers"]["ETH-USD"]["outlook"] == "bullish"

    def test_load_journal_respects_max_entries(self, tmp_path, monkeypatch):
        """Returns only the last N entries."""
        journal = tmp_path / "layer2_journal.jsonl"
        entries = [
            {"ts": f"2026-02-18T{i:02d}:00:00Z", "tickers": {"BTC-USD": {"outlook": "bullish", "thesis": f"Entry {i}"}}, "prices": {"BTC-USD": 65000 + i * 100}}
            for i in range(10)
        ]
        journal.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")
        monkeypatch.setattr("portfolio.analyze.JOURNAL_FILE", journal)

        result = _load_journal_for_ticker("BTC-USD", max_entries=3)
        assert len(result) == 3
        # Should be the last 3
        assert result[0]["ts"] == "2026-02-18T07:00:00Z"

    def test_load_journal_missing_file(self, tmp_path, monkeypatch):
        """Returns empty list when journal file doesn't exist."""
        monkeypatch.setattr("portfolio.analyze.JOURNAL_FILE", tmp_path / "nonexistent.jsonl")
        result = _load_journal_for_ticker("ETH-USD")
        assert result == []


# ---------------------------------------------------------------------------
# _get_holdings tests
# ---------------------------------------------------------------------------

class TestGetHoldings:
    """Tests for _get_holdings."""

    def test_get_holdings_returns_both_strategies(self, tmp_path, monkeypatch):
        """Returns holdings from both patient and bold portfolios."""
        patient = tmp_path / "portfolio_state.json"
        bold = tmp_path / "portfolio_state_bold.json"
        patient.write_text(json.dumps({
            "holdings": {"ETH-USD": {"shares": 5.68, "avg_cost_usd": 1978.2}},
        }), encoding="utf-8")
        bold.write_text(json.dumps({
            "holdings": {"ETH-USD": {"shares": 2.5, "avg_cost_usd": 1950.0}},
        }), encoding="utf-8")
        monkeypatch.setattr("portfolio.analyze.PORTFOLIO_FILE", patient)
        monkeypatch.setattr("portfolio.analyze.BOLD_FILE", bold)

        result = _get_holdings("ETH-USD")
        assert "patient" in result
        assert "bold" in result
        assert result["patient"]["shares"] == 5.68

    def test_get_holdings_skips_zero_shares(self, tmp_path, monkeypatch):
        """Excludes holdings with zero shares."""
        patient = tmp_path / "portfolio_state.json"
        patient.write_text(json.dumps({
            "holdings": {"BTC-USD": {"shares": 0, "avg_cost_usd": 68000}},
        }), encoding="utf-8")
        bold = tmp_path / "portfolio_state_bold.json"
        bold.write_text(json.dumps({"holdings": {}}), encoding="utf-8")
        monkeypatch.setattr("portfolio.analyze.PORTFOLIO_FILE", patient)
        monkeypatch.setattr("portfolio.analyze.BOLD_FILE", bold)

        result = _get_holdings("BTC-USD")
        assert result == {}

    def test_get_holdings_missing_files(self, tmp_path, monkeypatch):
        """Returns empty dict when portfolio files don't exist."""
        monkeypatch.setattr("portfolio.analyze.PORTFOLIO_FILE", tmp_path / "nope.json")
        monkeypatch.setattr("portfolio.analyze.BOLD_FILE", tmp_path / "also_nope.json")
        result = _get_holdings("ETH-USD")
        assert result == {}


# ---------------------------------------------------------------------------
# run_analysis tests
# ---------------------------------------------------------------------------

class TestRunAnalysis:
    """Tests for run_analysis (integration-level with mocks)."""

    def test_unknown_ticker_prints_error(self, capsys):
        """run_analysis with an unknown ticker prints 'Unknown ticker'."""
        run_analysis("FAKE")
        captured = capsys.readouterr()
        assert "Unknown ticker" in captured.out

    def test_missing_summary_prints_error(self, monkeypatch, capsys, tmp_path):
        """run_analysis prints error when agent_summary.json doesn't exist."""
        monkeypatch.setattr(
            "portfolio.analyze.AGENT_SUMMARY_FILE",
            tmp_path / "nonexistent_summary.json",
        )
        run_analysis("ETH-USD")
        captured = capsys.readouterr()
        assert "No agent_summary.json" in captured.out


# ---------------------------------------------------------------------------
# Watch position tests
# ---------------------------------------------------------------------------

class TestParsePositions:
    """Tests for _parse_positions."""

    def test_parse_valid_positions(self):
        result = _parse_positions(["BTC:66500", "ETH:1920", "NVDA:125"])
        assert result == {"BTC-USD": 66500.0, "ETH-USD": 1920.0, "NVDA": 125.0}

    def test_parse_normalizes_short_names(self):
        result = _parse_positions(["btc:66500"])
        assert "BTC-USD" in result

    def test_parse_ignores_bad_format(self, capsys):
        result = _parse_positions(["BADFORMAT"])
        assert result == {}
        assert "Bad format" in capsys.readouterr().out

    def test_parse_ignores_unknown_ticker(self, capsys):
        result = _parse_positions(["FAKE:100"])
        assert result == {}
        assert "Unknown ticker" in capsys.readouterr().out

    def test_parse_ignores_bad_price(self, capsys):
        result = _parse_positions(["BTC:notanumber"])
        assert result == {}
        assert "Bad price" in capsys.readouterr().out


class TestBuildWatchPrompt:
    """Tests for _build_watch_prompt."""

    def test_watch_prompt_contains_positions(self):
        positions = {"ETH-USD": 1900.0}
        prompt = _build_watch_prompt(positions, SAMPLE_SUMMARY, elapsed_mins=30)
        assert "ETH-USD" in prompt
        assert "1,900.00" in prompt  # entry price
        assert "1,921.46" in prompt  # current price
        assert "30 minutes" in prompt or "30" in prompt

    def test_watch_prompt_shows_pnl(self):
        positions = {"ETH-USD": 1900.0}
        prompt = _build_watch_prompt(positions, SAMPLE_SUMMARY, elapsed_mins=10)
        # ETH is at 1921.46, entry 1900 = +1.13%
        assert "+1.1" in prompt or "+1.13" in prompt

    def test_watch_prompt_includes_signals(self):
        positions = {"ETH-USD": 1900.0}
        prompt = _build_watch_prompt(positions, SAMPLE_SUMMARY, elapsed_mins=10)
        assert "3B/5S/3H" in prompt
        assert "RSI:" in prompt
        assert "MACD:" in prompt

    def test_watch_prompt_includes_macro(self):
        positions = {"ETH-USD": 1900.0}
        prompt = _build_watch_prompt(positions, SAMPLE_SUMMARY, elapsed_mins=10)
        assert "DXY" in prompt
        assert "97.92" in prompt

    def test_watch_prompt_includes_hold_sell_instructions(self):
        positions = {"ETH-USD": 1900.0}
        prompt = _build_watch_prompt(positions, SAMPLE_SUMMARY, elapsed_mins=10)
        assert "HOLD or SELL" in prompt
        assert "REASON:" in prompt

    def test_watch_prompt_multiple_positions(self):
        positions = {"ETH-USD": 1900.0, "NVDA": 185.0}
        prompt = _build_watch_prompt(positions, SAMPLE_SUMMARY, elapsed_mins=10)
        assert "ETH-USD" in prompt
        assert "NVDA" in prompt

    def test_watch_prompt_crypto_has_ministral(self):
        positions = {"ETH-USD": 1900.0}
        prompt = _build_watch_prompt(positions, SAMPLE_SUMMARY, elapsed_mins=10)
        assert "Ministral" in prompt

    def test_watch_prompt_stock_no_ministral(self):
        positions = {"NVDA": 185.0}
        prompt = _build_watch_prompt(positions, SAMPLE_SUMMARY, elapsed_mins=10)
        assert "Ministral" not in prompt


class TestParseWatchResponse:
    """Tests for _parse_watch_response."""

    def test_parse_hold_response(self):
        output = (
            "ETH-USD: HOLD\n"
            "REASON: Signals improving, RSI recovering from oversold.\n"
            "\n"
            "OVERALL: Market in recovery phase."
        )
        decisions, overall = _parse_watch_response(output, ["ETH-USD"])
        assert decisions["ETH-USD"]["action"] == "HOLD"
        assert "RSI" in decisions["ETH-USD"]["reason"]
        assert "recovery" in overall.lower()

    def test_parse_sell_response(self):
        output = (
            "BTC-USD: SELL\n"
            "REASON: All timeframes bearish, MACD deteriorating.\n"
            "\n"
            "OVERALL: Bearish across the board."
        )
        decisions, overall = _parse_watch_response(output, ["BTC-USD"])
        assert decisions["BTC-USD"]["action"] == "SELL"

    def test_parse_multiple_positions(self):
        output = (
            "ETH-USD: HOLD\n"
            "REASON: Holding steady.\n"
            "\n"
            "BTC-USD: SELL\n"
            "REASON: Break down confirmed.\n"
            "\n"
            "OVERALL: Mixed signals."
        )
        decisions, overall = _parse_watch_response(output, ["ETH-USD", "BTC-USD"])
        assert decisions["ETH-USD"]["action"] == "HOLD"
        assert decisions["BTC-USD"]["action"] == "SELL"

    def test_parse_short_ticker_names(self):
        """Accepts 'BTC:' as well as 'BTC-USD:'."""
        output = "BTC: SELL\nREASON: Momentum fading."
        decisions, _ = _parse_watch_response(output, ["BTC-USD"])
        assert decisions["BTC-USD"]["action"] == "SELL"

    def test_parse_missing_ticker_returns_empty(self):
        output = "OVERALL: All quiet."
        decisions, _ = _parse_watch_response(output, ["ETH-USD"])
        assert "ETH-USD" not in decisions

    def test_parse_bold_markdown_response(self):
        """Handles Claude's **bold** formatting around tickers and OVERALL."""
        output = "**BTC-USD: SELL**\nREASON: Bearish.\n\n**OVERALL:** Bad market."
        decisions, overall = _parse_watch_response(output, ["BTC-USD"])
        assert decisions["BTC-USD"]["action"] == "SELL"
        assert "Bearish" in decisions["BTC-USD"]["reason"]
        assert "Bad market" in overall

    def test_parse_bold_markdown_hold(self):
        """Handles **ETH-USD: HOLD** with bold reason line."""
        output = "**ETH-USD: HOLD**\n**REASON:** Signals improving.\n\n**OVERALL:** Cautious."
        decisions, overall = _parse_watch_response(output, ["ETH-USD"])
        assert decisions["ETH-USD"]["action"] == "HOLD"
        assert "Cautious" in overall

    def test_parse_heading_markdown_response(self):
        """Handles ## heading formatting."""
        output = "## BTC-USD: SELL\nREASON: Momentum fading.\n\nOVERALL: Weak."
        decisions, overall = _parse_watch_response(output, ["BTC-USD"])
        assert decisions["BTC-USD"]["action"] == "SELL"
        assert "Weak" in overall


class TestShouldCallClaude:
    """Tests for _should_call_claude."""

    def test_first_call_always_triggers(self):
        should, reason = _should_call_claude(0, {}, {}, {}, {})
        assert should is True
        assert "initial" in reason

    def test_periodic_trigger_after_15_min(self):
        import time as t
        old_time = t.time() - 16 * 60  # 16 min ago
        should, reason = _should_call_claude(old_time, {}, {}, {}, {})
        assert should is True
        assert "periodic" in reason

    def test_no_trigger_within_interval(self):
        import time as t
        recent = t.time() - 60  # 1 min ago
        should, _ = _should_call_claude(
            recent,
            {"BTC-USD": 66500},
            {"BTC-USD": 66510},  # tiny move
            {"BTC-USD": "SELL"},
            {"BTC-USD": "SELL"},  # no flip
        )
        assert should is False

    def test_price_move_triggers(self):
        import time as t
        recent = t.time() - 60
        should, reason = _should_call_claude(
            recent,
            {"BTC-USD": 66500},
            {"BTC-USD": 67000},  # 0.75% move
            {"BTC-USD": "SELL"},
            {"BTC-USD": "SELL"},
        )
        assert should is True
        assert "moved" in reason

    def test_signal_flip_triggers(self):
        import time as t
        recent = t.time() - 60
        should, reason = _should_call_claude(
            recent,
            {"BTC-USD": 66500},
            {"BTC-USD": 66510},  # tiny move
            {"BTC-USD": "SELL"},
            {"BTC-USD": "BUY"},  # flipped!
        )
        assert should is True
        assert "flipped" in reason
