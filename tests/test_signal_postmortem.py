"""Tests for portfolio.signal_postmortem — signal failure analysis."""


from portfolio.signal_postmortem import (
    compute_regime_insights,
    compute_signal_health_report,
    compute_vote_correlation,
)


class TestRegimeInsights:
    def test_regime_dependent_signal(self):
        """Signal with different accuracy across regimes is flagged."""
        regime_acc = {
            "trending-up": {
                "ema": {"accuracy": 0.75, "total": 50},
            },
            "ranging": {
                "ema": {"accuracy": 0.35, "total": 50},
            },
        }

        insights = compute_regime_insights(regime_acc)
        assert len(insights) == 1
        assert insights[0]["signal"] == "ema"
        assert insights[0]["best_regime"] == "trending-up"
        assert insights[0]["worst_regime"] == "ranging"
        assert insights[0]["spread_pp"] == 40.0

    def test_stable_signal_not_flagged(self):
        """Signal with similar accuracy across regimes is not flagged."""
        regime_acc = {
            "trending-up": {
                "rsi": {"accuracy": 0.55, "total": 50},
            },
            "ranging": {
                "rsi": {"accuracy": 0.52, "total": 50},
            },
        }

        insights = compute_regime_insights(regime_acc)
        assert len(insights) == 0

    def test_insufficient_samples_excluded(self):
        """Signals with too few samples are excluded."""
        regime_acc = {
            "trending-up": {
                "test_sig": {"accuracy": 0.90, "total": 5},
            },
            "ranging": {
                "test_sig": {"accuracy": 0.10, "total": 5},
            },
        }

        insights = compute_regime_insights(regime_acc)
        assert len(insights) == 0

    def test_single_regime_ignored(self):
        """Signal in only one regime cannot be compared."""
        regime_acc = {
            "trending-up": {
                "ema": {"accuracy": 0.75, "total": 50},
            },
        }

        insights = compute_regime_insights(regime_acc)
        assert len(insights) == 0

    def test_sorted_by_spread(self):
        """Results sorted by spread descending."""
        regime_acc = {
            "trending-up": {
                "ema": {"accuracy": 0.80, "total": 50},
                "rsi": {"accuracy": 0.70, "total": 50},
            },
            "ranging": {
                "ema": {"accuracy": 0.30, "total": 50},
                "rsi": {"accuracy": 0.40, "total": 50},
            },
        }

        insights = compute_regime_insights(regime_acc)
        assert len(insights) == 2
        assert insights[0]["signal"] == "ema"  # 50pp spread
        assert insights[1]["signal"] == "rsi"  # 30pp spread

    def test_empty_input(self):
        assert compute_regime_insights({}) == []
        assert compute_regime_insights(None) == []


class TestSignalHealthReport:
    def test_classification(self):
        """Signals classified as strong/marginal/weak correctly."""
        acc = {
            "rsi": {"accuracy": 0.65, "total": 100},
            "macd": {"accuracy": 0.50, "total": 100},
            "ml": {"accuracy": 0.28, "total": 100},
        }

        report = compute_signal_health_report(acc)
        cats = {r["signal"]: r["category"] for r in report}
        assert cats["rsi"] == "strong"
        assert cats["macd"] == "marginal"
        assert cats["ml"] == "weak"

    def test_sorted_by_accuracy(self):
        """Report sorted by accuracy descending."""
        acc = {
            "a": {"accuracy": 0.50, "total": 50},
            "b": {"accuracy": 0.70, "total": 50},
            "c": {"accuracy": 0.30, "total": 50},
        }

        report = compute_signal_health_report(acc)
        assert report[0]["signal"] == "b"
        assert report[-1]["signal"] == "c"

    def test_insufficient_samples_excluded(self):
        """Signals with few samples excluded from report."""
        acc = {
            "rsi": {"accuracy": 0.65, "total": 100},
            "new_sig": {"accuracy": 0.90, "total": 3},
        }

        report = compute_signal_health_report(acc)
        assert len(report) == 1
        assert report[0]["signal"] == "rsi"


class TestVoteCorrelation:
    def test_correlated_pair(self):
        """Two signals that always agree are detected."""
        entries = []
        for _ in range(50):
            entries.append({
                "tickers": {
                    "BTC-USD": {
                        "signals": {"rsi": "BUY", "stochrsi": "BUY", "macd": "SELL"},
                    }
                }
            })

        result = compute_vote_correlation(entries)
        # rsi and stochrsi always agree
        pairs = {(c["signal_a"], c["signal_b"]) for c in result}
        assert ("rsi", "stochrsi") in pairs

    def test_uncorrelated_ignored(self):
        """Pairs with low agreement not reported."""
        entries = []
        for i in range(50):
            entries.append({
                "tickers": {
                    "BTC-USD": {
                        "signals": {
                            "a": "BUY" if i % 2 == 0 else "SELL",
                            "b": "BUY" if i % 3 == 0 else "SELL",
                        },
                    }
                }
            })

        result = compute_vote_correlation(entries)
        # a and b have ~50% agreement — should not be reported
        assert len(result) == 0

    def test_hold_votes_excluded(self):
        """HOLD votes are not counted in correlation."""
        entries = []
        for _ in range(50):
            entries.append({
                "tickers": {
                    "BTC-USD": {
                        "signals": {"a": "HOLD", "b": "BUY"},
                    }
                }
            })

        result = compute_vote_correlation(entries)
        assert len(result) == 0  # no co-active pairs

    def test_empty_entries(self):
        assert compute_vote_correlation([]) == []
