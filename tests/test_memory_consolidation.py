"""Tests for portfolio.memory_consolidation — daily autoDream consolidation."""

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from portfolio.memory_consolidation import (
    MAX_OUTPUT_LINES,
    _compute_regime_stats,
    _compute_signal_accuracy,
    _compute_trade_stats,
    _load_recent_entries,
    consolidate_insights,
)

# ---------------------------------------------------------------------------
# Helpers — generate reproducible test data
# ---------------------------------------------------------------------------

SIGNAL_NAMES = ["rsi", "macd", "ema", "bb", "fear_greed", "sentiment", "ministral", "volume"]
TICKERS = ["BTC-USD", "ETH-USD", "NVDA"]
REGIMES = ["trending-up", "trending-down", "ranging", "unknown"]
ACTIONS = ["BUY", "SELL", "HOLD"]


def _make_signal_entries(rng, n=50, with_outcomes=True, base_time=None):
    """Generate n signal log entries with reproducible random data."""
    if base_time is None:
        base_time = datetime.now(UTC)
    entries = []
    for i in range(n):
        ts = (base_time - timedelta(hours=n - i)).isoformat()
        tickers = {}
        for ticker in TICKERS:
            signals = {}
            for sig in SIGNAL_NAMES:
                signals[sig] = rng.choice(ACTIONS)
            regime = rng.choice(REGIMES)
            tickers[ticker] = {
                "price_usd": round(float(rng.uniform(100, 70000)), 2),
                "consensus": rng.choice(ACTIONS),
                "signals": signals,
                "regime": regime,
            }

        entry = {"ts": ts, "tickers": tickers}

        if with_outcomes:
            outcomes = {}
            for ticker in TICKERS:
                outcomes[ticker] = {
                    "1d": {"change_pct": round(float(rng.uniform(-5, 5)), 2)},
                }
            entry["outcomes"] = outcomes

        entries.append(entry)
    return entries


def _make_journal_entries(rng, n=20, base_time=None):
    """Generate n journal entries with reproducible random data."""
    if base_time is None:
        base_time = datetime.now(UTC)
    entries = []
    for i in range(n):
        ts = (base_time - timedelta(hours=n - i)).isoformat()
        decisions = {}
        for strategy in ["patient", "bold"]:
            decisions[strategy] = {
                "action": rng.choice(ACTIONS),
                "reasoning": "Test reasoning.",
            }
        entry = {
            "ts": ts,
            "trigger": "test",
            "regime": rng.choice(REGIMES),
            "decisions": decisions,
        }
        entries.append(entry)
    return entries


def _write_jsonl(path, entries):
    """Write entries as JSONL to path."""
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# TestLoadRecentEntries
# ---------------------------------------------------------------------------


class TestLoadRecentEntries:
    """Tests for _load_recent_entries."""

    def test_loads_entries_within_window(self, tmp_path):
        """Only entries within the time window are returned."""
        now = datetime.now(UTC)
        entries = [
            {"ts": (now - timedelta(days=10)).isoformat(), "data": "old"},
            {"ts": (now - timedelta(days=5)).isoformat(), "data": "mid"},
            {"ts": (now - timedelta(days=2)).isoformat(), "data": "recent1"},
            {"ts": (now - timedelta(hours=3)).isoformat(), "data": "recent2"},
        ]
        path = tmp_path / "test.jsonl"
        _write_jsonl(path, entries)

        result = _load_recent_entries(path, days=7)

        assert len(result) == 3  # mid, recent1, recent2
        assert result[0]["data"] == "mid"
        assert result[-1]["data"] == "recent2"

    def test_handles_missing_file(self, tmp_path):
        """Missing file returns empty list."""
        path = tmp_path / "nonexistent.jsonl"
        result = _load_recent_entries(path, days=7)
        assert result == []

    def test_handles_empty_file(self, tmp_path):
        """Empty file returns empty list."""
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        result = _load_recent_entries(path, days=7)
        assert result == []

    def test_skips_entries_without_ts(self, tmp_path):
        """Entries missing 'ts' field are skipped."""
        now = datetime.now(UTC)
        entries = [
            {"data": "no_ts"},
            {"ts": (now - timedelta(hours=1)).isoformat(), "data": "has_ts"},
        ]
        path = tmp_path / "test.jsonl"
        _write_jsonl(path, entries)

        result = _load_recent_entries(path, days=7)
        assert len(result) == 1
        assert result[0]["data"] == "has_ts"


# ---------------------------------------------------------------------------
# TestConsolidateInsights
# ---------------------------------------------------------------------------


class TestConsolidateInsights:
    """Tests for the main consolidate_insights function."""

    def test_produces_markdown_output(self, tmp_path):
        """Output file is valid markdown with expected sections, <200 lines."""
        rng = np.random.default_rng(42)

        signal_path = tmp_path / "signal_log.jsonl"
        journal_path = tmp_path / "journal.jsonl"
        output_path = tmp_path / "trading_insights.md"

        _write_jsonl(signal_path, _make_signal_entries(rng, n=60))
        _write_jsonl(journal_path, _make_journal_entries(rng, n=25))

        result = consolidate_insights(
            signal_log_path=signal_path,
            journal_path=journal_path,
            output_path=output_path,
            days=7,
        )

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        # Must be under 200 lines
        assert len(lines) <= MAX_OUTPUT_LINES

        # Expected sections
        assert "# Trading Insights" in content
        assert "## Signal Performance" in content
        assert "## Regime Summary" in content
        assert "## Trade Decisions" in content
        assert "## Key Takeaways" in content

        # Return value sanity
        assert isinstance(result, dict)
        assert "best_signals" in result
        assert "worst_signals" in result
        assert "dominant_regime" in result
        assert "total_decisions" in result
        assert "entries_processed" in result
        assert result["entries_processed"] > 0

    def test_handles_empty_logs(self, tmp_path):
        """Gracefully handles empty log files."""
        signal_path = tmp_path / "signal_log.jsonl"
        journal_path = tmp_path / "journal.jsonl"
        output_path = tmp_path / "trading_insights.md"

        signal_path.write_text("")
        journal_path.write_text("")

        result = consolidate_insights(
            signal_log_path=signal_path,
            journal_path=journal_path,
            output_path=output_path,
            days=7,
        )

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "# Trading Insights" in content
        assert result["entries_processed"] == 0
        assert result["total_decisions"] == 0
        assert result["dominant_regime"] is None

    def test_handles_missing_logs(self, tmp_path):
        """Works when log files do not exist at all."""
        signal_path = tmp_path / "nonexistent_signal.jsonl"
        journal_path = tmp_path / "nonexistent_journal.jsonl"
        output_path = tmp_path / "trading_insights.md"

        result = consolidate_insights(
            signal_log_path=signal_path,
            journal_path=journal_path,
            output_path=output_path,
            days=7,
        )

        assert output_path.exists()
        assert result["entries_processed"] == 0

    def test_identifies_best_and_worst_signals(self, tmp_path):
        """Correctly identifies best (>55%) and worst (<45%) signals."""
        rng = np.random.default_rng(42)

        # Create entries where 'rsi' is always correct and 'ml' is always wrong
        now = datetime.now(UTC)
        entries = []
        for i in range(30):
            ts = (now - timedelta(hours=30 - i)).isoformat()
            entry = {
                "ts": ts,
                "tickers": {
                    "BTC-USD": {
                        "price_usd": 65000.0,
                        "consensus": "BUY",
                        "signals": {
                            "rsi": "BUY",     # Always BUY
                            "macd": "HOLD",
                            "ml": "SELL",      # Always SELL
                            "ema": "HOLD",
                        },
                    },
                },
                "outcomes": {
                    "BTC-USD": {
                        "1d": {"change_pct": 2.0},  # Price always goes UP
                    },
                },
            }
            entries.append(entry)

        signal_path = tmp_path / "signal_log.jsonl"
        journal_path = tmp_path / "journal.jsonl"
        output_path = tmp_path / "trading_insights.md"

        _write_jsonl(signal_path, entries)
        journal_path.write_text("")

        result = consolidate_insights(
            signal_log_path=signal_path,
            journal_path=journal_path,
            output_path=output_path,
            days=7,
        )

        # rsi (always BUY + price always up) = 100% accuracy -> best
        assert "rsi" in result["best_signals"]
        # ml (always SELL + price always up) = 0% accuracy -> worst
        assert "ml" in result["worst_signals"]

    def test_output_contains_regime_data(self, tmp_path):
        """Regime summary section is populated when regime data exists."""
        rng = np.random.default_rng(42)

        signal_path = tmp_path / "signal_log.jsonl"
        journal_path = tmp_path / "journal.jsonl"
        output_path = tmp_path / "trading_insights.md"

        _write_jsonl(signal_path, _make_signal_entries(rng, n=30))
        _write_jsonl(journal_path, _make_journal_entries(rng, n=10))

        consolidate_insights(
            signal_log_path=signal_path,
            journal_path=journal_path,
            output_path=output_path,
            days=7,
        )

        content = output_path.read_text(encoding="utf-8")
        # Should have regime table rows
        assert "trending" in content.lower() or "ranging" in content.lower() or "unknown" in content.lower()

    def test_trade_stats_counted(self, tmp_path):
        """Trade decisions are counted per strategy."""
        rng = np.random.default_rng(42)

        signal_path = tmp_path / "signal_log.jsonl"
        journal_path = tmp_path / "journal.jsonl"
        output_path = tmp_path / "trading_insights.md"

        signal_path.write_text("")
        _write_jsonl(journal_path, _make_journal_entries(rng, n=15))

        result = consolidate_insights(
            signal_log_path=signal_path,
            journal_path=journal_path,
            output_path=output_path,
            days=7,
        )

        # 15 entries x 2 strategies = 30 decisions
        assert result["total_decisions"] == 30

    def test_return_dict_structure(self, tmp_path):
        """Return dict has all required keys with correct types."""
        rng = np.random.default_rng(42)

        signal_path = tmp_path / "signal_log.jsonl"
        journal_path = tmp_path / "journal.jsonl"
        output_path = tmp_path / "trading_insights.md"

        _write_jsonl(signal_path, _make_signal_entries(rng, n=20))
        _write_jsonl(journal_path, _make_journal_entries(rng, n=10))

        result = consolidate_insights(
            signal_log_path=signal_path,
            journal_path=journal_path,
            output_path=output_path,
            days=7,
        )

        assert isinstance(result["best_signals"], list)
        assert isinstance(result["worst_signals"], list)
        assert isinstance(result["total_decisions"], int)
        assert isinstance(result["entries_processed"], int)
        # dominant_regime is str or None
        assert result["dominant_regime"] is None or isinstance(
            result["dominant_regime"], str
        )


# ---------------------------------------------------------------------------
# Unit tests for internal functions
# ---------------------------------------------------------------------------


class TestComputeSignalAccuracy:
    """Tests for _compute_signal_accuracy."""

    def test_empty_entries(self):
        assert _compute_signal_accuracy([]) == {}

    def test_no_outcomes(self):
        entries = [{"tickers": {"BTC": {"signals": {"rsi": "BUY"}}}}]
        assert _compute_signal_accuracy(entries) == {}

    def test_correct_buy(self):
        entries = [{
            "tickers": {"BTC": {"signals": {"rsi": "BUY"}}},
            "outcomes": {"BTC": {"1d": {"change_pct": 3.0}}},
        }]
        result = _compute_signal_accuracy(entries)
        assert result["rsi"]["correct"] == 1
        assert result["rsi"]["total"] == 1
        assert result["rsi"]["accuracy"] == 100.0

    def test_wrong_sell(self):
        entries = [{
            "tickers": {"BTC": {"signals": {"macd": "SELL"}}},
            "outcomes": {"BTC": {"1d": {"change_pct": 2.0}}},
        }]
        result = _compute_signal_accuracy(entries)
        assert result["macd"]["correct"] == 0
        assert result["macd"]["total"] == 1
        assert result["macd"]["accuracy"] == 0.0

    def test_hold_ignored(self):
        entries = [{
            "tickers": {"BTC": {"signals": {"rsi": "HOLD"}}},
            "outcomes": {"BTC": {"1d": {"change_pct": 3.0}}},
        }]
        result = _compute_signal_accuracy(entries)
        assert result == {}


class TestComputeRegimeStats:
    """Tests for _compute_regime_stats."""

    def test_empty(self):
        assert _compute_regime_stats([]) == {}

    def test_top_level_regime(self):
        entries = [
            {"regime": "trending-up"},
            {"regime": "trending-up"},
            {"regime": "ranging"},
        ]
        result = _compute_regime_stats(entries)
        assert result["trending-up"]["count"] == 2
        assert result["ranging"]["count"] == 1

    def test_per_ticker_regime(self):
        entries = [{
            "tickers": {
                "BTC": {"regime": "trending-up"},
                "ETH": {"regime": "ranging"},
            },
        }]
        result = _compute_regime_stats(entries)
        assert result["trending-up"]["count"] == 1
        assert result["ranging"]["count"] == 1


class TestComputeTradeStats:
    """Tests for _compute_trade_stats."""

    def test_empty(self):
        result = _compute_trade_stats([])
        assert result["total_decisions"] == 0

    def test_counts_actions(self):
        entries = [{
            "decisions": {
                "patient": {"action": "BUY"},
                "bold": {"action": "HOLD"},
            },
        }]
        result = _compute_trade_stats(entries)
        assert result["total_decisions"] == 2
        assert result["action_counts"]["BUY"] == 1
        assert result["action_counts"]["HOLD"] == 1
