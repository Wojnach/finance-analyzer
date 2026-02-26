"""Tests for bull/bear adversarial debate schema in journal entries."""

import pytest
from portfolio.journal import _append_entry, _non_neutral_tickers
from portfolio.journal_index import _tokenize_entry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_entry(tickers=None, **kwargs):
    """Build a minimal journal entry dict."""
    base = {
        "ts": "2026-02-26T12:00:00+00:00",
        "trigger": "consensus",
        "regime": "trending-up",
        "decisions": {
            "patient": {"action": "HOLD", "reasoning": "No setup"},
            "bold": {"action": "HOLD", "reasoning": "No breakout"},
        },
        "tickers": tickers or {},
        "prices": {"BTC-USD": 67000},
        "watchlist": [],
    }
    base.update(kwargs)
    return base


def _ticker_with_debate(bull="Volume expansion", bear="RSI overbought", synthesis="Wait for pullback"):
    return {
        "outlook": "bullish",
        "thesis": "Breakout forming",
        "conviction": 0.7,
        "levels": [65000, 70000],
        "debate": {
            "bull": bull,
            "bear": bear,
            "synthesis": synthesis,
        },
    }


def _ticker_no_debate():
    return {
        "outlook": "bearish",
        "thesis": "Breakdown below support",
        "conviction": 0.5,
        "levels": [1800, 2000],
    }


# ---------------------------------------------------------------------------
# _append_entry tests — debate rendering
# ---------------------------------------------------------------------------

class TestAppendEntryDebate:
    def test_debate_rendered_when_present(self):
        entry = _make_entry(tickers={"BTC-USD": _ticker_with_debate()})
        lines = []
        _append_entry(lines, entry)
        text = "\n".join(lines)
        assert "Bull: Volume expansion" in text
        assert "Bear: RSI overbought" in text
        assert "Synthesis: Wait for pullback" in text

    def test_debate_not_rendered_when_absent(self):
        entry = _make_entry(tickers={"ETH-USD": _ticker_no_debate()})
        lines = []
        _append_entry(lines, entry)
        text = "\n".join(lines)
        assert "Bull:" not in text
        assert "Bear:" not in text
        assert "Synthesis:" not in text

    def test_partial_debate_only_renders_present_fields(self):
        ticker_info = _ticker_with_debate()
        del ticker_info["debate"]["bear"]
        entry = _make_entry(tickers={"BTC-USD": ticker_info})
        lines = []
        _append_entry(lines, entry)
        text = "\n".join(lines)
        assert "Bull: Volume expansion" in text
        assert "Bear:" not in text
        assert "Synthesis: Wait for pullback" in text

    def test_empty_debate_dict_no_crash(self):
        ticker_info = _ticker_with_debate()
        ticker_info["debate"] = {}
        entry = _make_entry(tickers={"BTC-USD": ticker_info})
        lines = []
        _append_entry(lines, entry)
        text = "\n".join(lines)
        assert "Bull:" not in text

    def test_debate_invalid_type_ignored(self):
        ticker_info = _ticker_with_debate()
        ticker_info["debate"] = "not a dict"
        entry = _make_entry(tickers={"BTC-USD": ticker_info})
        lines = []
        _append_entry(lines, entry)
        text = "\n".join(lines)
        assert "Bull:" not in text

    def test_debate_with_neutral_ticker_not_shown(self):
        """Neutral tickers are filtered by _non_neutral_tickers, so debate won't render."""
        entry = _make_entry(tickers={
            "BTC-USD": {
                "outlook": "neutral",
                "thesis": "",
                "conviction": 0.0,
                "debate": {"bull": "x", "bear": "y", "synthesis": "z"},
            }
        })
        lines = []
        _append_entry(lines, entry)
        text = "\n".join(lines)
        assert "Bull:" not in text

    def test_multiple_tickers_with_debate(self):
        entry = _make_entry(tickers={
            "BTC-USD": _ticker_with_debate(bull="BTC volume up"),
            "ETH-USD": {
                "outlook": "bearish",
                "thesis": "Weak",
                "conviction": 0.6,
                "debate": {
                    "bull": "ETH forming base",
                    "bear": "ETH below 200 SMA",
                    "synthesis": "Stay out",
                },
            },
        })
        lines = []
        _append_entry(lines, entry)
        text = "\n".join(lines)
        assert "Bull: BTC volume up" in text
        assert "Bull: ETH forming base" in text
        assert "Bear: ETH below 200 SMA" in text


# ---------------------------------------------------------------------------
# _tokenize_entry tests — debate tokenization
# ---------------------------------------------------------------------------

class TestTokenizeDebate:
    def test_debate_fields_tokenized(self):
        entry = _make_entry(tickers={"BTC-USD": _ticker_with_debate(
            bull="Volume expansion breakout",
            bear="RSI overbought resistance",
            synthesis="Wait for pullback confirmation",
        )})
        tokens = _tokenize_entry(entry)
        assert "volume" in tokens
        assert "expansion" in tokens
        assert "breakout" in tokens
        assert "overbought" in tokens
        assert "resistance" in tokens
        assert "pullback" in tokens
        assert "confirmation" in tokens

    def test_debate_missing_no_extra_tokens(self):
        entry = _make_entry(tickers={"ETH-USD": _ticker_no_debate()})
        tokens = _tokenize_entry(entry)
        # Should not have any debate-derived tokens beyond thesis
        assert "volume" not in tokens  # not in thesis "Breakdown below support"

    def test_debate_empty_dict_no_tokens(self):
        ticker_info = _ticker_with_debate()
        ticker_info["debate"] = {}
        entry = _make_entry(tickers={"BTC-USD": ticker_info})
        tokens = _tokenize_entry(entry)
        # Thesis tokens should still be there
        assert "breakout" in tokens
        assert "forming" in tokens

    def test_debate_invalid_type_no_crash(self):
        ticker_info = _ticker_with_debate()
        ticker_info["debate"] = 42
        entry = _make_entry(tickers={"BTC-USD": ticker_info})
        tokens = _tokenize_entry(entry)
        # Should not crash, just skip debate
        assert isinstance(tokens, list)

    def test_debate_stop_words_filtered(self):
        entry = _make_entry(tickers={"BTC-USD": _ticker_with_debate(
            bull="The volume is expanding",
            bear="A bearish divergence",
            synthesis="Not a good entry",
        )})
        tokens = _tokenize_entry(entry)
        assert "the" not in tokens
        assert "is" not in tokens
        assert "volume" in tokens
        assert "expanding" in tokens


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_old_entry_without_debate_renders_fine(self):
        """Entries from before the debate feature should render identically."""
        entry = _make_entry(tickers={
            "BTC-USD": {
                "outlook": "bullish",
                "thesis": "Breakout forming",
                "conviction": 0.7,
                "levels": [65000, 70000],
                # No "debate" key at all
            }
        })
        lines = []
        _append_entry(lines, entry)
        text = "\n".join(lines)
        assert "BTC-USD: bullish [70%]" in text
        assert "Breakout forming" in text
        assert "Bull:" not in text

    def test_old_entry_tokenizes_fine(self):
        entry = _make_entry(tickers={
            "BTC-USD": {
                "outlook": "bullish",
                "thesis": "Breakout forming",
                "conviction": 0.7,
            }
        })
        tokens = _tokenize_entry(entry)
        assert "breakout" in tokens
        assert "forming" in tokens
