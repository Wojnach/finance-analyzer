"""Tests for portfolio/kelly_metals.py.

Covers: recommended_metals_size, _get_ticker_accuracy, _get_outcome_stats,
_loss_reduction, format_kelly_line, leverage adjustment, edge cases.
"""

import json
import sqlite3

import pytest

from portfolio.kelly_metals import (
    _get_outcome_stats,
    _get_ticker_accuracy,
    _loss_reduction,
    format_kelly_line,
    recommended_metals_size,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _create_signal_db(db_path, ticker="XAG-USD", horizon="1d", rows=None):
    """Create a minimal signal_log.db with test data."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, trigger_reasons TEXT, fx_rate REAL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ticker_signals (
            snapshot_id INTEGER, ticker TEXT, price_usd REAL,
            consensus TEXT, buy_count INTEGER, sell_count INTEGER,
            total_voters INTEGER, signals TEXT, regime TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS outcomes (
            snapshot_id INTEGER, ticker TEXT, horizon TEXT,
            price_usd REAL, change_pct REAL, outcome_ts TEXT
        )
    """)

    if rows is None:
        # Default: 60 rows, ~55% win rate, avg_win ~3%, avg_loss ~2.5%
        rows = []
        for i in range(60):
            consensus = "BUY" if i % 3 != 2 else "SELL"
            # Wins: positive change for BUY, negative for SELL
            if i % 3 == 0:  # win
                change = 3.0 if consensus == "BUY" else -3.0
            elif i % 3 == 1:  # win
                change = 2.5 if consensus == "BUY" else -2.5
            else:  # loss (SELL that goes up)
                change = 2.0
            rows.append((consensus, change))

    for i, (consensus, change_pct) in enumerate(rows):
        sid = i + 1
        cur.execute("INSERT INTO snapshots VALUES (?, ?, ?, ?)",
                     (sid, f"2026-01-01T{i:02d}:00:00Z", "", 10.5))
        cur.execute("INSERT INTO ticker_signals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (sid, ticker, 30.0, consensus, 3, 1, 4, "{}", "neutral"))
        cur.execute("INSERT INTO outcomes VALUES (?, ?, ?, ?, ?, ?)",
                     (sid, ticker, horizon, 30.0 + change_pct, change_pct,
                      f"2026-01-02T{i:02d}:00:00Z"))

    conn.commit()
    conn.close()


# ===================================================================
# _loss_reduction
# ===================================================================

class TestLossReduction:
    def test_no_losses(self):
        assert _loss_reduction(0) == 1.0

    def test_one_loss(self):
        assert _loss_reduction(1) == 0.75

    def test_two_losses(self):
        assert _loss_reduction(2) == 0.50

    def test_three_losses(self):
        assert _loss_reduction(3) == 0.25

    def test_four_losses_sit_out(self):
        assert _loss_reduction(4) == 0.0

    def test_negative_losses(self):
        assert _loss_reduction(-1) == 1.0


# ===================================================================
# _get_ticker_accuracy
# ===================================================================

class TestGetTickerAccuracy:
    def test_reads_accuracy(self, tmp_path, monkeypatch):
        cache = {
            "per_ticker_consensus": {
                "XAG-USD": {"correct": 520, "total": 1000, "accuracy": 0.52, "pct": 52.0}
            }
        }
        cache_path = tmp_path / "accuracy_cache.json"
        _write_json(cache_path, cache)
        monkeypatch.setattr("portfolio.kelly_metals.ACCURACY_CACHE", cache_path)

        result = _get_ticker_accuracy("XAG-USD")
        assert result == pytest.approx(0.52)

    def test_returns_none_insufficient_samples(self, tmp_path, monkeypatch):
        cache = {
            "per_ticker_consensus": {
                "XAG-USD": {"correct": 10, "total": 20, "accuracy": 0.50}
            }
        }
        cache_path = tmp_path / "accuracy_cache.json"
        _write_json(cache_path, cache)
        monkeypatch.setattr("portfolio.kelly_metals.ACCURACY_CACHE", cache_path)

        assert _get_ticker_accuracy("XAG-USD") is None

    def test_returns_none_missing_ticker(self, tmp_path, monkeypatch):
        cache = {"per_ticker_consensus": {}}
        cache_path = tmp_path / "accuracy_cache.json"
        _write_json(cache_path, cache)
        monkeypatch.setattr("portfolio.kelly_metals.ACCURACY_CACHE", cache_path)

        assert _get_ticker_accuracy("XAU-USD") is None

    def test_returns_none_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("portfolio.kelly_metals.ACCURACY_CACHE",
                            tmp_path / "nonexistent.json")
        assert _get_ticker_accuracy("XAG-USD") is None


# ===================================================================
# _get_outcome_stats
# ===================================================================

class TestGetOutcomeStats:
    def test_computes_from_db(self, tmp_path, monkeypatch):
        db_path = tmp_path / "signal_log.db"
        _create_signal_db(db_path)
        monkeypatch.setattr("portfolio.kelly_metals.SIGNAL_DB", db_path)

        result = _get_outcome_stats("XAG-USD", "1d")
        assert result is not None
        assert result["n_trades"] == 60
        assert 0 < result["win_rate"] < 1
        assert result["avg_win_pct"] > 0
        assert result["avg_loss_pct"] > 0

    def test_returns_none_insufficient_data(self, tmp_path, monkeypatch):
        db_path = tmp_path / "signal_log.db"
        # Only 5 rows — below threshold of 30
        rows = [("BUY", 2.0)] * 5
        _create_signal_db(db_path, rows=rows)
        monkeypatch.setattr("portfolio.kelly_metals.SIGNAL_DB", db_path)

        assert _get_outcome_stats("XAG-USD", "1d") is None

    def test_returns_none_missing_db(self, tmp_path, monkeypatch):
        monkeypatch.setattr("portfolio.kelly_metals.SIGNAL_DB",
                            tmp_path / "nonexistent.db")
        assert _get_outcome_stats("XAG-USD") is None

    def test_different_horizon(self, tmp_path, monkeypatch):
        db_path = tmp_path / "signal_log.db"
        _create_signal_db(db_path, horizon="3h")
        monkeypatch.setattr("portfolio.kelly_metals.SIGNAL_DB", db_path)

        # 1d horizon should find nothing since DB has 3h
        assert _get_outcome_stats("XAG-USD", "1d") is None
        result = _get_outcome_stats("XAG-USD", "3h")
        assert result is not None


# ===================================================================
# recommended_metals_size
# ===================================================================

class TestRecommendedMetalsSize:
    @pytest.fixture(autouse=True)
    def _patch_data_sources(self, tmp_path, monkeypatch):
        """Patch file paths so tests don't hit real data."""
        monkeypatch.setattr("portfolio.kelly_metals.ACCURACY_CACHE",
                            tmp_path / "accuracy_cache.json")
        monkeypatch.setattr("portfolio.kelly_metals.SIGNAL_DB",
                            tmp_path / "signal_log.db")
        monkeypatch.setattr("portfolio.kelly_metals.AGENT_SUMMARY",
                            tmp_path / "agent_summary.json")

    def test_basic_sizing_with_defaults(self):
        """No data files — uses defaults, should still produce a result."""
        rec = recommended_metals_size(
            ticker="XAG-USD",
            leverage=5.0,
            buying_power_sek=5000,
            ask_price_sek=7.0,
        )
        assert "position_sek" in rec
        assert "units" in rec
        assert "kelly_pct" in rec
        assert "half_kelly_pct" in rec
        assert "win_rate" in rec
        assert "source" in rec
        assert rec["leverage"] == 5.0

    def test_positive_edge_produces_position(self, tmp_path):
        """With clear positive edge, should produce a nonzero position."""
        cache = {
            "per_ticker_consensus": {
                "XAG-USD": {"correct": 600, "total": 1000, "accuracy": 0.60}
            }
        }
        _write_json(tmp_path / "accuracy_cache.json", cache)

        db_path = tmp_path / "signal_log.db"
        # 60% win rate, avg_win=3%, avg_loss=2%
        rows = []
        for i in range(100):
            if i < 60:
                rows.append(("BUY", 3.0))   # win
            else:
                rows.append(("BUY", -2.0))  # loss
        _create_signal_db(db_path, rows=rows)

        rec = recommended_metals_size(
            ticker="XAG-USD",
            leverage=5.0,
            buying_power_sek=5000,
            ask_price_sek=7.0,
        )
        assert rec["position_sek"] > 0
        assert rec["units"] > 0
        assert rec["kelly_pct"] > 0
        assert rec["half_kelly_pct"] > 0
        assert rec["win_rate"] > 0.5

    def test_no_edge_produces_zero(self, tmp_path):
        """50/50 win rate with equal win/loss -> Kelly = 0 -> no position."""
        cache = {
            "per_ticker_consensus": {
                "XAG-USD": {"correct": 500, "total": 1000, "accuracy": 0.50}
            }
        }
        _write_json(tmp_path / "accuracy_cache.json", cache)

        db_path = tmp_path / "signal_log.db"
        rows = []
        for i in range(100):
            if i < 50:
                rows.append(("BUY", 2.0))
            else:
                rows.append(("BUY", -2.0))
        _create_signal_db(db_path, rows=rows)

        rec = recommended_metals_size(
            ticker="XAG-USD",
            leverage=5.0,
            buying_power_sek=5000,
            ask_price_sek=7.0,
        )
        assert rec["position_sek"] == 0
        assert rec["kelly_pct"] == 0.0

    def test_zero_buying_power(self):
        """Zero cash -> zero position."""
        rec = recommended_metals_size(
            ticker="XAG-USD",
            leverage=5.0,
            buying_power_sek=0,
            ask_price_sek=7.0,
        )
        assert rec["position_sek"] == 0
        assert rec["units"] == 0

    def test_consecutive_losses_reduce_size(self, tmp_path):
        """Consecutive losses should reduce position size."""
        cache = {
            "per_ticker_consensus": {
                "XAG-USD": {"correct": 650, "total": 1000, "accuracy": 0.65}
            }
        }
        _write_json(tmp_path / "accuracy_cache.json", cache)

        rec_0 = recommended_metals_size(
            ticker="XAG-USD", leverage=5.0,
            buying_power_sek=10000, ask_price_sek=7.0,
            consecutive_losses=0,
        )
        rec_2 = recommended_metals_size(
            ticker="XAG-USD", leverage=5.0,
            buying_power_sek=10000, ask_price_sek=7.0,
            consecutive_losses=2,
        )
        rec_4 = recommended_metals_size(
            ticker="XAG-USD", leverage=5.0,
            buying_power_sek=10000, ask_price_sek=7.0,
            consecutive_losses=4,
        )
        assert rec_0["position_sek"] > rec_2["position_sek"]
        assert rec_2["position_sek"] > 0
        assert rec_4["position_sek"] == 0  # 4 losses = sit out
        assert rec_2["loss_multiplier"] == 0.50
        assert rec_4["loss_multiplier"] == 0.0

    def test_higher_leverage_means_smaller_position(self, tmp_path):
        """Higher leverage should result in smaller position fraction."""
        cache = {
            "per_ticker_consensus": {
                "XAG-USD": {"correct": 600, "total": 1000, "accuracy": 0.60}
            }
        }
        _write_json(tmp_path / "accuracy_cache.json", cache)

        rec_5x = recommended_metals_size(
            ticker="XAG-USD", leverage=5.0,
            buying_power_sek=10000, ask_price_sek=7.0,
        )
        rec_20x = recommended_metals_size(
            ticker="XAG-USD", leverage=20.0,
            buying_power_sek=10000, ask_price_sek=7.0,
        )
        assert rec_5x["position_fraction"] > rec_20x["position_fraction"]

    def test_below_minimum_trade(self):
        """Position below 500 SEK -> zeroed out."""
        rec = recommended_metals_size(
            ticker="XAG-USD", leverage=5.0,
            buying_power_sek=200,
            ask_price_sek=7.0,
        )
        assert rec["position_sek"] == 0

    def test_agent_summary_fallback(self, tmp_path):
        """Uses agent_summary weighted_confidence when no other data."""
        summary = {
            "signals": {
                "XAG-USD": {"weighted_confidence": 0.62}
            }
        }
        rec = recommended_metals_size(
            ticker="XAG-USD", leverage=5.0,
            buying_power_sek=5000, ask_price_sek=7.0,
            agent_summary=summary,
        )
        assert "weighted_confidence" in rec["source"]
        assert rec["win_rate"] == pytest.approx(0.62)

    def test_growth_projections_included(self, tmp_path):
        """Result includes daily and monthly growth estimates."""
        cache = {
            "per_ticker_consensus": {
                "XAG-USD": {"correct": 600, "total": 1000, "accuracy": 0.60}
            }
        }
        _write_json(tmp_path / "accuracy_cache.json", cache)

        rec = recommended_metals_size(
            ticker="XAG-USD", leverage=5.0,
            buying_power_sek=5000, ask_price_sek=7.0,
        )
        assert "daily_log_growth" in rec
        assert "monthly_growth_pct" in rec
        if rec["position_sek"] > 0:
            assert rec["daily_log_growth"] > 0

    def test_gold_ticker(self, tmp_path):
        """Works with XAU-USD using its own defaults."""
        rec = recommended_metals_size(
            ticker="XAU-USD", leverage=3.0,
            buying_power_sek=5000, ask_price_sek=10.0,
        )
        assert rec["avg_win_pct"] == 2.10  # XAU default
        assert rec["avg_loss_pct"] == 1.80  # XAU default


# ===================================================================
# format_kelly_line
# ===================================================================

class TestFormatKellyLine:
    def test_basic_format(self):
        rec = {
            "half_kelly_pct": 0.07,
            "win_rate": 0.52,
            "position_sek": 1200,
            "loss_multiplier": 1.0,
        }
        line = format_kelly_line(rec)
        assert "Kelly:7.0%" in line
        assert "WR:52%" in line
        assert "Pos:1200kr" in line

    def test_includes_loss_adjustment(self):
        rec = {
            "half_kelly_pct": 0.07,
            "win_rate": 0.52,
            "position_sek": 600,
            "loss_multiplier": 0.50,
        }
        line = format_kelly_line(rec)
        assert "loss adj" in line
        assert "x0.50" in line
