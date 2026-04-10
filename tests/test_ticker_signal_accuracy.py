"""Tests for per-ticker per-signal accuracy cross-tabulation."""

import pytest

# ---------------------------------------------------------------------------
# Helpers to build test data
# ---------------------------------------------------------------------------

def _make_entry(ts, tickers_data, outcomes_data=None):
    """Build a signal_log entry dict matching the JSONL/SQLite format."""
    return {
        "ts": ts,
        "trigger_reasons": ["test"],
        "fx_rate": 10.5,
        "tickers": tickers_data,
        "outcomes": outcomes_data or {},
    }


def _ticker_signals(consensus, signals_dict, price_usd=100.0):
    return {
        "price_usd": price_usd,
        "consensus": consensus,
        "buy_count": sum(1 for v in signals_dict.values() if v == "BUY"),
        "sell_count": sum(1 for v in signals_dict.values() if v == "SELL"),
        "total_voters": len(signals_dict),
        "signals": signals_dict,
    }


def _outcome(change_pct, price_usd=100.0):
    return {"1d": {"change_pct": change_pct, "price_usd": price_usd, "ts": "2026-01-02T00:00:00"}}


# ---------------------------------------------------------------------------
# Test accuracy_by_ticker_signal()
# ---------------------------------------------------------------------------

class TestAccuracyByTickerSignal:
    """Tests for accuracy_stats.accuracy_by_ticker_signal()."""

    def test_basic_cross_tabulation(self, monkeypatch):
        """RSI correct for BTC but wrong for ETH → different per-ticker accuracy."""
        entries = [
            _make_entry("2026-01-01T00:00:00", {
                "BTC-USD": _ticker_signals("BUY", {"rsi": "BUY", "macd": "HOLD"}),
                "ETH-USD": _ticker_signals("BUY", {"rsi": "BUY", "macd": "BUY"}),
            }, {
                "BTC-USD": _outcome(5.0),   # BUY correct (price went up)
                "ETH-USD": _outcome(-3.0),  # BUY wrong (price went down)
            }),
        ]
        from portfolio import accuracy_stats
        monkeypatch.setattr(accuracy_stats, "load_entries", lambda: entries)

        result = accuracy_stats.accuracy_by_ticker_signal("1d")

        assert "BTC-USD" in result
        assert "ETH-USD" in result
        assert result["BTC-USD"]["rsi"]["accuracy"] == 1.0
        assert result["BTC-USD"]["rsi"]["total"] == 1
        assert result["ETH-USD"]["rsi"]["accuracy"] == 0.0
        assert result["ETH-USD"]["macd"]["accuracy"] == 0.0

    def test_hold_votes_excluded(self, monkeypatch):
        """HOLD votes should not count toward accuracy."""
        entries = [
            _make_entry("2026-01-01T00:00:00", {
                "BTC-USD": _ticker_signals("HOLD", {"rsi": "HOLD", "macd": "BUY"}),
            }, {
                "BTC-USD": _outcome(5.0),
            }),
        ]
        from portfolio import accuracy_stats
        monkeypatch.setattr(accuracy_stats, "load_entries", lambda: entries)

        result = accuracy_stats.accuracy_by_ticker_signal("1d")
        # RSI voted HOLD → should not appear or have 0 total
        btc = result.get("BTC-USD", {})
        rsi = btc.get("rsi", {"total": 0})
        assert rsi["total"] == 0

    def test_multiple_entries_accumulate(self, monkeypatch):
        """Multiple entries should accumulate correct/total counts."""
        entries = [
            _make_entry("2026-01-01T00:00:00", {
                "BTC-USD": _ticker_signals("BUY", {"rsi": "BUY"}),
            }, {"BTC-USD": _outcome(5.0)}),
            _make_entry("2026-01-02T00:00:00", {
                "BTC-USD": _ticker_signals("BUY", {"rsi": "BUY"}),
            }, {"BTC-USD": _outcome(-2.0)}),
            _make_entry("2026-01-03T00:00:00", {
                "BTC-USD": _ticker_signals("SELL", {"rsi": "SELL"}),
            }, {"BTC-USD": _outcome(-3.0)}),
        ]
        from portfolio import accuracy_stats
        monkeypatch.setattr(accuracy_stats, "load_entries", lambda: entries)

        result = accuracy_stats.accuracy_by_ticker_signal("1d")
        btc_rsi = result["BTC-USD"]["rsi"]
        assert btc_rsi["total"] == 3
        assert btc_rsi["correct"] == 2  # first BUY correct, second BUY wrong, SELL correct
        assert abs(btc_rsi["accuracy"] - 2 / 3) < 0.001

    def test_directional_accuracy_fields(self, monkeypatch):
        """Per-ticker accuracy should include buy/sell directional breakdown."""
        entries = [
            _make_entry("2026-01-01T00:00:00", {
                "BTC-USD": _ticker_signals("BUY", {"rsi": "BUY"}),
            }, {"BTC-USD": _outcome(5.0)}),     # BUY correct
            _make_entry("2026-01-02T00:00:00", {
                "BTC-USD": _ticker_signals("BUY", {"rsi": "BUY"}),
            }, {"BTC-USD": _outcome(-2.0)}),    # BUY wrong
            _make_entry("2026-01-03T00:00:00", {
                "BTC-USD": _ticker_signals("SELL", {"rsi": "SELL"}),
            }, {"BTC-USD": _outcome(-3.0)}),    # SELL correct
            _make_entry("2026-01-04T00:00:00", {
                "BTC-USD": _ticker_signals("SELL", {"rsi": "SELL"}),
            }, {"BTC-USD": _outcome(1.0)}),     # SELL wrong
        ]
        from portfolio import accuracy_stats
        monkeypatch.setattr(accuracy_stats, "load_entries", lambda: entries)

        result = accuracy_stats.accuracy_by_ticker_signal("1d")
        btc_rsi = result["BTC-USD"]["rsi"]

        # Overall: 2 correct out of 4
        assert btc_rsi["total"] == 4
        assert btc_rsi["correct"] == 2

        # Directional: BUY 1/2 = 50%, SELL 1/2 = 50%
        assert btc_rsi["total_buy"] == 2
        assert btc_rsi["correct_buy"] == 1
        assert btc_rsi["buy_accuracy"] == 0.5
        assert btc_rsi["total_sell"] == 2
        assert btc_rsi["correct_sell"] == 1
        assert btc_rsi["sell_accuracy"] == 0.5

    def test_directional_accuracy_asymmetry(self, monkeypatch):
        """Verify asymmetric BUY/SELL accuracy is captured per ticker."""
        entries = [
            # 3 BUY votes: 1 correct, 2 wrong → buy_accuracy = 33.3%
            _make_entry("2026-01-01T00:00:00", {
                "XAG-USD": _ticker_signals("BUY", {"ministral": "BUY"}),
            }, {"XAG-USD": _outcome(1.0)}),     # correct
            _make_entry("2026-01-02T00:00:00", {
                "XAG-USD": _ticker_signals("BUY", {"ministral": "BUY"}),
            }, {"XAG-USD": _outcome(-2.0)}),    # wrong
            _make_entry("2026-01-03T00:00:00", {
                "XAG-USD": _ticker_signals("BUY", {"ministral": "BUY"}),
            }, {"XAG-USD": _outcome(-1.0)}),    # wrong
            # 2 SELL votes: 2 correct → sell_accuracy = 100%
            _make_entry("2026-01-04T00:00:00", {
                "XAG-USD": _ticker_signals("SELL", {"ministral": "SELL"}),
            }, {"XAG-USD": _outcome(-3.0)}),    # correct
            _make_entry("2026-01-05T00:00:00", {
                "XAG-USD": _ticker_signals("SELL", {"ministral": "SELL"}),
            }, {"XAG-USD": _outcome(-1.0)}),    # correct
        ]
        from portfolio import accuracy_stats
        monkeypatch.setattr(accuracy_stats, "load_entries", lambda: entries)

        result = accuracy_stats.accuracy_by_ticker_signal("1d")
        xag = result["XAG-USD"]["ministral"]

        # Overall: 3 correct out of 5 = 60%
        assert xag["total"] == 5
        assert xag["correct"] == 3

        # BUY: 1/3 = 33.3%, SELL: 2/2 = 100%
        assert xag["total_buy"] == 3
        assert xag["correct_buy"] == 1
        assert abs(xag["buy_accuracy"] - 1 / 3) < 0.01
        assert xag["total_sell"] == 2
        assert xag["correct_sell"] == 2
        assert xag["sell_accuracy"] == 1.0

    def test_sell_correct(self, monkeypatch):
        """SELL vote with negative change should be correct."""
        entries = [
            _make_entry("2026-01-01T00:00:00", {
                "XAG-USD": _ticker_signals("SELL", {"ema": "SELL"}),
            }, {"XAG-USD": _outcome(-2.5)}),
        ]
        from portfolio import accuracy_stats
        monkeypatch.setattr(accuracy_stats, "load_entries", lambda: entries)

        result = accuracy_stats.accuracy_by_ticker_signal("1d")
        assert result["XAG-USD"]["ema"]["accuracy"] == 1.0

    def test_min_samples_filter(self, monkeypatch):
        """Signals with fewer than min_samples should be excluded."""
        entries = [
            _make_entry("2026-01-01T00:00:00", {
                "BTC-USD": _ticker_signals("BUY", {"rsi": "BUY"}),
            }, {"BTC-USD": _outcome(5.0)}),
        ]
        from portfolio import accuracy_stats
        monkeypatch.setattr(accuracy_stats, "load_entries", lambda: entries)

        result = accuracy_stats.accuracy_by_ticker_signal("1d", min_samples=5)
        # Only 1 sample, min is 5 → should be filtered out
        btc = result.get("BTC-USD", {})
        assert "rsi" not in btc

    def test_empty_entries(self, monkeypatch):
        """Empty entries list should return empty dict."""
        from portfolio import accuracy_stats
        monkeypatch.setattr(accuracy_stats, "load_entries", lambda: [])

        result = accuracy_stats.accuracy_by_ticker_signal("1d")
        assert result == {}

    def test_no_outcomes(self, monkeypatch):
        """Entries without outcomes should be skipped."""
        entries = [
            _make_entry("2026-01-01T00:00:00", {
                "BTC-USD": _ticker_signals("BUY", {"rsi": "BUY"}),
            }),
        ]
        from portfolio import accuracy_stats
        monkeypatch.setattr(accuracy_stats, "load_entries", lambda: entries)

        result = accuracy_stats.accuracy_by_ticker_signal("1d")
        assert result == {}

    def test_pct_field_present(self, monkeypatch):
        """Each signal entry should have a pct field (accuracy * 100 rounded)."""
        entries = [
            _make_entry("2026-01-01T00:00:00", {
                "BTC-USD": _ticker_signals("BUY", {"rsi": "BUY"}),
            }, {"BTC-USD": _outcome(5.0)}),
        ]
        from portfolio import accuracy_stats
        monkeypatch.setattr(accuracy_stats, "load_entries", lambda: entries)

        result = accuracy_stats.accuracy_by_ticker_signal("1d")
        assert result["BTC-USD"]["rsi"]["pct"] == 100.0

    def test_multiple_signals_per_ticker(self, monkeypatch):
        """Each signal should track independently per ticker."""
        entries = [
            _make_entry("2026-01-01T00:00:00", {
                "BTC-USD": _ticker_signals("BUY", {"rsi": "BUY", "ema": "SELL", "bb": "BUY"}),
            }, {"BTC-USD": _outcome(2.0)}),
        ]
        from portfolio import accuracy_stats
        monkeypatch.setattr(accuracy_stats, "load_entries", lambda: entries)

        result = accuracy_stats.accuracy_by_ticker_signal("1d")
        btc = result["BTC-USD"]
        assert btc["rsi"]["accuracy"] == 1.0   # BUY + up = correct
        assert btc["ema"]["accuracy"] == 0.0   # SELL + up = wrong
        assert btc["bb"]["accuracy"] == 1.0    # BUY + up = correct


# ---------------------------------------------------------------------------
# Test SignalDB.ticker_signal_accuracy()
# ---------------------------------------------------------------------------

class TestSignalDBTickerSignalAccuracy:
    """Tests for the SQL-optimized version in SignalDB."""

    @pytest.fixture
    def db(self, tmp_path):
        from portfolio.signal_db import SignalDB
        db = SignalDB(tmp_path / "test.db")
        yield db
        db.close()

    def test_basic_sql_query(self, db):
        """SQL version should produce same results as Python version."""
        db.insert_snapshot(_make_entry("2026-01-01T00:00:00", {
            "BTC-USD": _ticker_signals("BUY", {"rsi": "BUY", "macd": "SELL"}),
        }, {"BTC-USD": _outcome(3.0)}))

        result = db.ticker_signal_accuracy("1d")
        assert result["BTC-USD"]["rsi"]["accuracy"] == 1.0
        assert result["BTC-USD"]["rsi"]["total"] == 1
        assert result["BTC-USD"]["macd"]["accuracy"] == 0.0

    def test_multiple_snapshots(self, db):
        """Should accumulate across multiple snapshots."""
        db.insert_snapshot(_make_entry("2026-01-01T00:00:00", {
            "BTC-USD": _ticker_signals("BUY", {"rsi": "BUY"}),
        }, {"BTC-USD": _outcome(5.0)}))
        db.insert_snapshot(_make_entry("2026-01-02T00:00:00", {
            "BTC-USD": _ticker_signals("SELL", {"rsi": "SELL"}),
        }, {"BTC-USD": _outcome(2.0)}))  # SELL but price went up → wrong

        result = db.ticker_signal_accuracy("1d")
        assert result["BTC-USD"]["rsi"]["total"] == 2
        assert result["BTC-USD"]["rsi"]["correct"] == 1

    def test_empty_db(self, db):
        """Empty database should return empty dict."""
        result = db.ticker_signal_accuracy("1d")
        assert result == {}

    def test_min_samples(self, db):
        """min_samples filter should exclude low-sample signals."""
        db.insert_snapshot(_make_entry("2026-01-01T00:00:00", {
            "BTC-USD": _ticker_signals("BUY", {"rsi": "BUY"}),
        }, {"BTC-USD": _outcome(5.0)}))

        result = db.ticker_signal_accuracy("1d", min_samples=3)
        assert result.get("BTC-USD", {}).get("rsi") is None

    def test_multiple_tickers(self, db):
        """Should track each ticker independently."""
        db.insert_snapshot(_make_entry("2026-01-01T00:00:00", {
            "BTC-USD": _ticker_signals("BUY", {"rsi": "BUY"}),
            "ETH-USD": _ticker_signals("SELL", {"rsi": "SELL"}),
        }, {
            "BTC-USD": _outcome(5.0),
            "ETH-USD": _outcome(-3.0),
        }))

        result = db.ticker_signal_accuracy("1d")
        assert result["BTC-USD"]["rsi"]["accuracy"] == 1.0
        assert result["ETH-USD"]["rsi"]["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Test top_signals_for_ticker() helper
# ---------------------------------------------------------------------------

class TestTopSignalsForTicker:
    """Tests for the convenience function that returns sorted signal rankings."""

    def test_ranking_order(self, monkeypatch):
        """Should return signals sorted by accuracy descending."""
        entries = []
        for i in range(10):
            entries.append(_make_entry(f"2026-01-{i+1:02d}T00:00:00", {
                "BTC-USD": _ticker_signals("BUY", {
                    "rsi": "BUY",
                    "ema": "SELL" if i < 7 else "BUY",  # ema wrong 7/10
                    "macd": "BUY" if i < 8 else "SELL",  # macd correct 8/10
                }),
            }, {"BTC-USD": _outcome(2.0)}))

        from portfolio import accuracy_stats
        monkeypatch.setattr(accuracy_stats, "load_entries", lambda: entries)

        top = accuracy_stats.top_signals_for_ticker("BTC-USD", "1d", min_samples=5)
        assert len(top) > 0
        # Should be sorted descending by accuracy
        accs = [s["accuracy"] for s in top]
        assert accs == sorted(accs, reverse=True)

    def test_unknown_ticker(self, monkeypatch):
        """Unknown ticker should return empty list."""
        from portfolio import accuracy_stats
        monkeypatch.setattr(accuracy_stats, "load_entries", lambda: [])

        top = accuracy_stats.top_signals_for_ticker("FAKE-USD", "1d")
        assert top == []
