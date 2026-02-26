"""Tests for ChromaDB vector memory module."""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from pathlib import Path

from portfolio.vector_memory import (
    entry_to_text,
    _entry_id,
    build_query_text,
    get_semantic_context,
    reset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(regime="trending-up", tickers=None, **kwargs):
    base = {
        "ts": "2026-02-26T12:00:00+00:00",
        "trigger": "consensus",
        "regime": regime,
        "reflection": "",
        "decisions": {
            "patient": {"action": "HOLD", "reasoning": "No setup"},
            "bold": {"action": "BUY BTC-USD", "reasoning": "Breakout confirmed"},
        },
        "tickers": tickers or {
            "BTC-USD": {
                "outlook": "bullish",
                "thesis": "Volume breakout above 68K",
                "conviction": 0.7,
            }
        },
        "prices": {"BTC-USD": 68000},
        "watchlist": ["BTC break above 70K"],
    }
    base.update(kwargs)
    return base


def _make_entry_with_debate():
    return _make_entry(tickers={
        "BTC-USD": {
            "outlook": "bullish",
            "thesis": "Breakout forming",
            "conviction": 0.8,
            "debate": {
                "bull": "12B consensus and volume expansion",
                "bear": "RSI 72 overbought",
                "synthesis": "Enter on pullback",
            },
        }
    })


# ---------------------------------------------------------------------------
# entry_to_text
# ---------------------------------------------------------------------------

class TestEntryToText:
    def test_basic_entry(self):
        text = entry_to_text(_make_entry())
        assert "regime: trending-up" in text
        assert "trigger: consensus" in text
        assert "BTC-USD: bullish" in text
        assert "Volume breakout above 68K" in text

    def test_decisions_included(self):
        text = entry_to_text(_make_entry())
        assert "patient: HOLD" in text
        assert "bold: BUY BTC-USD" in text

    def test_debate_fields_included(self):
        text = entry_to_text(_make_entry_with_debate())
        assert "bull: 12B consensus" in text
        assert "bear: RSI 72 overbought" in text
        assert "synthesis: Enter on pullback" in text

    def test_debate_missing_no_crash(self):
        text = entry_to_text(_make_entry())
        # No debate field in default entry — should not crash
        assert "bull:" not in text

    def test_watchlist_included(self):
        text = entry_to_text(_make_entry())
        assert "BTC break above 70K" in text

    def test_reflection_included(self):
        text = entry_to_text(_make_entry(reflection="Previous thesis was correct"))
        assert "Previous thesis was correct" in text

    def test_empty_entry(self):
        text = entry_to_text({})
        # Decisions always emit "patient: HOLD" / "bold: HOLD" even for empty entries
        assert "regime:" not in text
        assert "trigger:" not in text

    def test_debate_invalid_type_ignored(self):
        entry = _make_entry(tickers={
            "BTC-USD": {"outlook": "bullish", "thesis": "Test", "debate": "not a dict"}
        })
        text = entry_to_text(entry)
        assert "bull:" not in text


# ---------------------------------------------------------------------------
# _entry_id
# ---------------------------------------------------------------------------

class TestEntryId:
    def test_stable_id(self):
        entry = _make_entry()
        assert _entry_id(entry) == _entry_id(entry)

    def test_different_ts_different_id(self):
        e1 = _make_entry()
        e2 = _make_entry()
        e2["ts"] = "2026-02-26T13:00:00+00:00"
        assert _entry_id(e1) != _entry_id(e2)


# ---------------------------------------------------------------------------
# build_query_text
# ---------------------------------------------------------------------------

class TestBuildQueryText:
    def test_basic_query(self):
        state = {
            "regime": "trending-up",
            "held_tickers": ["BTC-USD"],
            "signals": {"ETH-USD": {"action": "BUY", "confidence": 0.7}},
        }
        text = build_query_text(state)
        assert "regime: trending-up" in text
        assert "BTC-USD" in text
        assert "ETH-USD: BUY" in text

    def test_hold_signals_excluded(self):
        state = {
            "regime": "range-bound",
            "held_tickers": [],
            "signals": {"BTC-USD": {"action": "HOLD", "confidence": 0.5}},
        }
        text = build_query_text(state)
        assert "BTC-USD" not in text

    def test_empty_state(self):
        text = build_query_text({})
        assert text == ""


# ---------------------------------------------------------------------------
# get_semantic_context — with chromadb mocked
# ---------------------------------------------------------------------------

class TestGetSemanticContext:
    def setup_method(self):
        reset()

    @patch("portfolio.vector_memory._load_journal_entries", return_value=[])
    @patch("portfolio.vector_memory._get_collection")
    def test_empty_collection(self, mock_coll, mock_entries):
        mock_coll.return_value.count.return_value = 0
        mock_coll.return_value.get.return_value = {"ids": []}
        state = {"regime": "trending-up", "signals": {"BTC-USD": {"action": "BUY", "confidence": 0.8}}}
        results = get_semantic_context(state)
        assert results == []

    def test_chromadb_not_installed(self):
        """When chromadb is not installed, should return empty list gracefully."""
        state = {"regime": "trending-up"}
        with patch("portfolio.vector_memory._get_collection",
                    side_effect=ImportError("No module named 'chromadb'")):
            results = get_semantic_context(state)
            assert results == []

    def test_config_disabled(self):
        """When config says disabled, _append_vector_memory_section skips."""
        from portfolio.journal import _append_vector_memory_section
        config = {"vector_memory": {"enabled": False}}
        md = "## Test"
        result = _append_vector_memory_section(md, config, None, [])
        assert result == md

    def test_deduplication_with_bm25(self):
        """Results with timestamps already in BM25 are filtered out."""
        mock_results = [
            {"text": "entry 1", "ts": "2026-02-26T12:00:00+00:00", "regime": "up", "distance": 0.1},
            {"text": "entry 2", "ts": "2026-02-26T13:00:00+00:00", "regime": "up", "distance": 0.2},
        ]
        with patch("portfolio.vector_memory._load_journal_entries", return_value=[]):
            with patch("portfolio.vector_memory.embed_entries", return_value=0):
                with patch("portfolio.vector_memory.query_similar", return_value=mock_results):
                    state = {"regime": "up", "signals": {"BTC-USD": {"action": "BUY", "confidence": 0.7}}}
                    # BM25 already has the first timestamp
                    results = get_semantic_context(
                        state,
                        bm25_timestamps={"2026-02-26T12:00:00+00:00"},
                    )
                    assert len(results) == 1
                    assert results[0]["ts"] == "2026-02-26T13:00:00+00:00"

    def test_error_returns_empty(self):
        """Any unexpected error returns empty list."""
        with patch("portfolio.vector_memory._load_journal_entries",
                    side_effect=RuntimeError("disk error")):
            results = get_semantic_context({"regime": "up"})
            assert results == []
