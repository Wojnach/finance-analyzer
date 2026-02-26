"""Tests for smart journal retrieval (BM25) in journal_index.py."""

import json
from datetime import datetime, timezone, timedelta

import pytest

from portfolio.journal_index import (
    BM25,
    JournalIndex,
    _tokenize_entry,
    _clean_words,
    _compute_importance,
    _price_bucket,
    _build_query_tokens,
    retrieve_relevant_entries,
)


# --- Helpers ---

def _make_entry(ts_offset_hours=0, regime="trending-up", tickers=None,
                decisions=None, watchlist=None, reflection="",
                prices=None, trigger="cooldown"):
    """Create a journal entry for testing."""
    base = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)
    ts = base + timedelta(hours=ts_offset_hours)
    return {
        "ts": ts.isoformat(),
        "trigger": trigger,
        "regime": regime,
        "reflection": reflection,
        "continues": None,
        "decisions": decisions or {
            "patient": {"action": "HOLD", "reasoning": "No setup"},
            "bold": {"action": "HOLD", "reasoning": "Waiting"},
        },
        "tickers": tickers or {},
        "watchlist": watchlist or [],
        "prices": prices or {},
    }


# --- BM25 ---

class TestBM25:
    def test_fit_and_score(self):
        bm25 = BM25()
        docs = [
            ["bitcoin", "buy", "breakout"],
            ["ethereum", "sell", "decline"],
            ["bitcoin", "hold", "ranging"],
        ]
        bm25.fit(docs)
        scores = bm25.score(["bitcoin"])
        assert len(scores) == 3
        # Doc 0 and 2 mention bitcoin, should score higher than doc 1
        assert scores[0] > scores[1]
        assert scores[2] > scores[1]

    def test_top_k(self):
        bm25 = BM25()
        docs = [
            ["btc", "buy", "breakout", "volume"],
            ["eth", "sell"],
            ["btc", "hold"],
            ["nvda", "buy", "breakout"],
        ]
        bm25.fit(docs)
        results = bm25.top_k(["btc", "buy", "breakout"], k=2)
        assert len(results) <= 2
        # First result should be doc 0 (most terms matched)
        assert results[0][0] == 0

    def test_empty_query(self):
        bm25 = BM25()
        bm25.fit([["a", "b"], ["c", "d"]])
        results = bm25.top_k([], k=5)
        assert results == []

    def test_empty_corpus(self):
        bm25 = BM25()
        bm25.fit([])
        scores = bm25.score(["test"])
        assert scores == []

    def test_no_match(self):
        bm25 = BM25()
        bm25.fit([["apple", "banana"], ["cherry", "date"]])
        results = bm25.top_k(["zebra"], k=5)
        assert results == []

    def test_single_doc(self):
        bm25 = BM25()
        bm25.fit([["bitcoin", "breakout", "volume"]])
        results = bm25.top_k(["bitcoin"], k=1)
        assert len(results) == 1
        assert results[0][0] == 0
        assert results[0][1] > 0

    def test_duplicate_terms_in_query(self):
        bm25 = BM25()
        bm25.fit([["btc", "buy"], ["eth", "sell"]])
        results = bm25.top_k(["btc", "btc", "btc"], k=2)
        # Should still work, btc matches doc 0
        assert results[0][0] == 0


# --- Tokenization ---

class TestTokenization:
    def test_tokenize_basic_entry(self):
        entry = _make_entry(regime="trending-up", tickers={
            "BTC-USD": {"outlook": "bullish", "thesis": "breakout forming", "conviction": 0.8},
        })
        tokens = _tokenize_entry(entry)
        assert "regime_trending-up" in tokens
        assert "btc-usd" in tokens
        assert "btc-usd_bullish" in tokens
        assert "breakout" in tokens
        assert "forming" in tokens
        assert "btc-usd_high_conviction" in tokens

    def test_tokenize_with_prices(self):
        entry = _make_entry(prices={"BTC-USD": 65000})
        tokens = _tokenize_entry(entry)
        # Should have a price bucket token
        bucket_tokens = [t for t in tokens if "btc-usd_" in t and ("above" in t or "below" in t)]
        assert len(bucket_tokens) > 0

    def test_tokenize_decisions(self):
        entry = _make_entry(decisions={
            "patient": {"action": "HOLD", "reasoning": "waiting for confirmation"},
            "bold": {"action": "BUY BTC-USD", "reasoning": "breakout confirmed"},
        })
        tokens = _tokenize_entry(entry)
        # Action "BUY BTC-USD" produces token "bold_buy btc-usd"
        assert any("bold_buy" in t for t in tokens)
        assert "breakout" in tokens
        assert "confirmed" in tokens

    def test_tokenize_empty_entry(self):
        entry = {"ts": "2026-01-01T00:00:00+00:00"}
        tokens = _tokenize_entry(entry)
        assert isinstance(tokens, list)

    def test_tokenize_watchlist(self):
        entry = _make_entry(watchlist=["BTC breakout above 67K", "ETH support at 1900"])
        tokens = _tokenize_entry(entry)
        assert "breakout" in tokens
        assert "67k" in tokens


class TestCleanWords:
    def test_basic(self):
        words = _clean_words("BTC breakout above 67K resistance")
        assert "btc" in words
        assert "breakout" in words
        assert "67k" in words

    def test_stop_words_removed(self):
        words = _clean_words("the price is above the resistance")
        assert "the" not in words
        assert "is" not in words

    def test_empty(self):
        assert _clean_words("") == []
        assert _clean_words(None) == []


# --- Price buckets ---

class TestPriceBuckets:
    def test_btc_below_bucket(self):
        assert _price_bucket("BTC-USD", 55000) == "BTC-USD_below_60000"

    def test_btc_above_bucket(self):
        assert _price_bucket("BTC-USD", 150000) == "BTC-USD_above_100000"

    def test_unknown_ticker(self):
        assert _price_bucket("NVDA", 180) is None

    def test_none_price(self):
        assert _price_bucket("BTC-USD", None) is None


# --- Importance scoring ---

class TestImportance:
    def test_recent_entry_higher(self):
        now = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)
        recent = _make_entry(ts_offset_hours=-1)
        old = _make_entry(ts_offset_hours=-24)
        assert _compute_importance(recent, now) > _compute_importance(old, now)

    def test_trade_action_boosts(self):
        now = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)
        hold = _make_entry(ts_offset_hours=-1)
        trade = _make_entry(ts_offset_hours=-1, decisions={
            "patient": {"action": "HOLD", "reasoning": ""},
            "bold": {"action": "BUY BTC-USD", "reasoning": "breakout"},
        })
        assert _compute_importance(trade, now) > _compute_importance(hold, now)

    def test_high_conviction_boosts(self):
        now = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)
        low_conv = _make_entry(ts_offset_hours=-1, tickers={
            "BTC-USD": {"outlook": "bullish", "conviction": 0.3},
        })
        high_conv = _make_entry(ts_offset_hours=-1, tickers={
            "BTC-USD": {"outlook": "bullish", "conviction": 0.9},
        })
        assert _compute_importance(high_conv, now) > _compute_importance(low_conv, now)

    def test_reflection_boosts(self):
        now = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)
        no_ref = _make_entry(ts_offset_hours=-1)
        with_ref = _make_entry(ts_offset_hours=-1, reflection="Previous thesis was correct")
        assert _compute_importance(with_ref, now) > _compute_importance(no_ref, now)

    def test_importance_capped_at_1(self):
        now = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)
        entry = _make_entry(
            ts_offset_hours=0,
            decisions={
                "patient": {"action": "BUY X", "reasoning": "strong"},
                "bold": {"action": "BUY Y", "reasoning": "very strong"},
            },
            tickers={"BTC-USD": {"outlook": "bullish", "conviction": 0.95}},
            reflection="Was right about everything",
        )
        assert _compute_importance(entry, now) <= 1.0


# --- JournalIndex ---

class TestJournalIndex:
    def test_build_and_query(self):
        entries = [
            _make_entry(ts_offset_hours=-10, regime="trending-up", tickers={
                "BTC-USD": {"outlook": "bullish", "thesis": "breakout"},
            }),
            _make_entry(ts_offset_hours=-8, regime="ranging", tickers={
                "ETH-USD": {"outlook": "bearish", "thesis": "breakdown"},
            }),
            _make_entry(ts_offset_hours=-6, regime="trending-up", tickers={
                "BTC-USD": {"outlook": "bullish", "thesis": "continuation"},
            }),
        ]
        idx = JournalIndex()
        idx.build(entries)

        # Query for BTC in trending-up regime
        results = idx.query({
            "held_tickers": ["BTC-USD"],
            "regime": "trending-up",
            "prices": {"BTC-USD": 65000},
            "signals": {},
        }, k=2)
        assert len(results) <= 2
        # BTC entries should rank higher
        tickers_in_results = set()
        for r in results:
            for t in r.get("tickers", {}):
                tickers_in_results.add(t)
        assert "BTC-USD" in tickers_in_results

    def test_empty_entries(self):
        idx = JournalIndex()
        idx.build([])
        results = idx.query({"held_tickers": [], "regime": "", "prices": {}, "signals": {}})
        assert results == []

    def test_query_with_signals(self):
        entries = [
            _make_entry(ts_offset_hours=-5, tickers={
                "NVDA": {"outlook": "bullish", "thesis": "earnings beat"},
            }),
            _make_entry(ts_offset_hours=-3, tickers={
                "BTC-USD": {"outlook": "bullish", "thesis": "volume expansion"},
            }),
        ]
        idx = JournalIndex()
        idx.build(entries)

        results = idx.query({
            "held_tickers": [],
            "regime": "trending-up",
            "prices": {},
            "signals": {"NVDA": {"action": "BUY"}},
        }, k=2)
        # NVDA entry should rank higher when NVDA has a BUY signal
        assert len(results) > 0


# --- Query token building ---

class TestBuildQueryTokens:
    def test_basic_tokens(self):
        tokens = _build_query_tokens({
            "held_tickers": ["BTC-USD"],
            "regime": "trending-up",
            "prices": {"BTC-USD": 65000},
            "signals": {},
        })
        assert "regime_trending-up" in tokens
        assert "btc-usd" in tokens

    def test_signal_action_tokens(self):
        tokens = _build_query_tokens({
            "held_tickers": [],
            "regime": "",
            "prices": {},
            "signals": {"NVDA": {"action": "BUY"}},
        })
        assert "nvda" in tokens
        assert "nvda_buy" in tokens

    def test_empty_state(self):
        tokens = _build_query_tokens({})
        assert tokens == []


# --- Integration: retrieve_relevant_entries ---

class TestRetrieveRelevantEntries:
    def test_returns_list(self, tmp_path):
        # Write a temp journal file
        journal = tmp_path / "layer2_journal.jsonl"
        entries = [
            _make_entry(ts_offset_hours=-5, tickers={
                "BTC-USD": {"outlook": "bullish", "thesis": "breakout"},
            }),
            _make_entry(ts_offset_hours=-3, tickers={
                "ETH-USD": {"outlook": "bearish", "thesis": "breakdown"},
            }),
        ]
        with open(journal, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        from unittest.mock import patch
        with patch("portfolio.journal_index.JOURNAL_FILE", journal):
            results = retrieve_relevant_entries(
                signals={"BTC-USD": {"action": "BUY"}},
                held_tickers=["BTC-USD"],
                regime="trending-up",
                prices={"BTC-USD": 65000},
                k=5,
            )
            assert isinstance(results, list)
            assert len(results) > 0

    def test_missing_file_returns_empty(self, tmp_path):
        from unittest.mock import patch
        with patch("portfolio.journal_index.JOURNAL_FILE", tmp_path / "nonexistent.jsonl"):
            results = retrieve_relevant_entries({}, [], "", {})
            assert results == []

    def test_empty_file_returns_empty(self, tmp_path):
        journal = tmp_path / "layer2_journal.jsonl"
        journal.write_text("")

        from unittest.mock import patch
        with patch("portfolio.journal_index.JOURNAL_FILE", journal):
            results = retrieve_relevant_entries({}, [], "", {})
            assert results == []

    def test_k_limits_results(self, tmp_path):
        journal = tmp_path / "layer2_journal.jsonl"
        entries = [_make_entry(ts_offset_hours=-i) for i in range(20)]
        with open(journal, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        from unittest.mock import patch
        with patch("portfolio.journal_index.JOURNAL_FILE", journal):
            results = retrieve_relevant_entries(
                signals={}, held_tickers=[], regime="trending-up",
                prices={}, k=3,
            )
            assert len(results) <= 3
