"""Tests for probability_calibration in accuracy_stats."""
from portfolio.accuracy_stats import probability_calibration


def test_probability_calibration_returns_buckets(monkeypatch):
    """probability_calibration returns bucket dicts with correct schema."""
    import portfolio.accuracy_stats as mod

    fake_entries = [
        {
            "ts": "2026-03-01T00:00:00+00:00",
            "tickers": {
                "BTC-USD": {
                    "consensus": "BUY",
                    "buy_count": 7,
                    "sell_count": 3,
                    "total_voters": 10,
                    "signals": {},
                },
            },
            "outcomes": {
                "BTC-USD": {"1d": {"change_pct": 2.5}},
            },
        },
        {
            "ts": "2026-03-02T00:00:00+00:00",
            "tickers": {
                "BTC-USD": {
                    "consensus": "SELL",
                    "buy_count": 2,
                    "sell_count": 8,
                    "total_voters": 10,
                    "signals": {},
                },
            },
            "outcomes": {
                "BTC-USD": {"1d": {"change_pct": -1.0}},
            },
        },
    ]
    monkeypatch.setattr(mod, "load_entries", lambda: fake_entries)

    result = probability_calibration(horizon="1d")
    assert isinstance(result, list)
    # Both samples have confidence in 0.7-0.8 range, so one bucket should be populated
    filled = [b for b in result if b["sample_count"] > 0]
    assert len(filled) >= 1
    for b in filled:
        assert "predicted_confidence" in b
        assert "actual_accuracy" in b
        assert "sample_count" in b
        assert "correct_count" in b
        assert "bucket_low" in b
        assert "bucket_high" in b
        assert b["actual_accuracy"] == 1.0  # both predictions were correct


def test_probability_calibration_empty_data(monkeypatch):
    """Returns empty buckets when no data available."""
    import portfolio.accuracy_stats as mod

    monkeypatch.setattr(mod, "load_entries", lambda: [])
    result = probability_calibration()
    assert isinstance(result, list)
    assert len(result) > 0  # should still return bucket structure
    assert all(b["sample_count"] == 0 for b in result)
    assert all(b["actual_accuracy"] is None for b in result)


def test_probability_calibration_since_filter(monkeypatch):
    """Entries before 'since' cutoff are excluded."""
    import portfolio.accuracy_stats as mod

    fake_entries = [
        {
            "ts": "2026-02-01T00:00:00+00:00",
            "tickers": {
                "ETH-USD": {
                    "consensus": "BUY",
                    "buy_count": 6,
                    "sell_count": 4,
                    "total_voters": 10,
                    "signals": {},
                },
            },
            "outcomes": {
                "ETH-USD": {"1d": {"change_pct": 3.0}},
            },
        },
        {
            "ts": "2026-03-10T00:00:00+00:00",
            "tickers": {
                "ETH-USD": {
                    "consensus": "BUY",
                    "buy_count": 9,
                    "sell_count": 1,
                    "total_voters": 10,
                    "signals": {},
                },
            },
            "outcomes": {
                "ETH-USD": {"1d": {"change_pct": 1.5}},
            },
        },
    ]
    monkeypatch.setattr(mod, "load_entries", lambda: fake_entries)

    # With since=2026-03-01, only the second entry should be included
    result = probability_calibration(horizon="1d", since="2026-03-01T00:00:00+00:00")
    total_samples = sum(b["sample_count"] for b in result)
    assert total_samples == 1  # only the March entry


def test_probability_calibration_skips_neutral_outcomes(monkeypatch):
    """Outcomes with change_pct near zero are skipped."""
    import portfolio.accuracy_stats as mod

    fake_entries = [
        {
            "ts": "2026-03-01T00:00:00+00:00",
            "tickers": {
                "BTC-USD": {
                    "consensus": "BUY",
                    "buy_count": 8,
                    "sell_count": 2,
                    "total_voters": 10,
                    "signals": {},
                },
            },
            "outcomes": {
                "BTC-USD": {"1d": {"change_pct": 0.01}},  # below threshold
            },
        },
    ]
    monkeypatch.setattr(mod, "load_entries", lambda: fake_entries)

    result = probability_calibration(horizon="1d")
    total_samples = sum(b["sample_count"] for b in result)
    assert total_samples == 0  # skipped because change_pct < 0.05


def test_probability_calibration_custom_buckets(monkeypatch):
    """Custom bucket boundaries work correctly."""
    import portfolio.accuracy_stats as mod

    fake_entries = [
        {
            "ts": "2026-03-01T00:00:00+00:00",
            "tickers": {
                "XAG-USD": {
                    "consensus": "BUY",
                    "buy_count": 5,
                    "sell_count": 5,
                    "total_voters": 10,
                    "signals": {},
                },
            },
            "outcomes": {
                "XAG-USD": {"1d": {"change_pct": 2.0}},
            },
        },
    ]
    monkeypatch.setattr(mod, "load_entries", lambda: fake_entries)

    # confidence = 5/10 = 0.5, so it should land in first bucket [0.4, 0.55)
    result = probability_calibration(
        horizon="1d", buckets=[0.4, 0.55, 0.7, 1.01]
    )
    assert len(result) == 3  # 3 buckets
    total = sum(b["sample_count"] for b in result)
    assert total == 1
    assert result[0]["sample_count"] == 1  # 0.5 is in [0.4, 0.55)
    assert result[1]["sample_count"] == 0
    assert result[2]["sample_count"] == 0


def test_probability_calibration_incorrect_prediction(monkeypatch):
    """Incorrect predictions result in accuracy < 1.0."""
    import portfolio.accuracy_stats as mod

    fake_entries = [
        {
            "ts": "2026-03-01T00:00:00+00:00",
            "tickers": {
                "BTC-USD": {
                    "consensus": "BUY",
                    "buy_count": 7,
                    "sell_count": 3,
                    "total_voters": 10,
                    "signals": {},
                },
            },
            "outcomes": {
                "BTC-USD": {"1d": {"change_pct": -2.0}},  # wrong direction
            },
        },
        {
            "ts": "2026-03-02T00:00:00+00:00",
            "tickers": {
                "ETH-USD": {
                    "consensus": "BUY",
                    "buy_count": 7,
                    "sell_count": 3,
                    "total_voters": 10,
                    "signals": {},
                },
            },
            "outcomes": {
                "ETH-USD": {"1d": {"change_pct": 3.0}},  # correct direction
            },
        },
    ]
    monkeypatch.setattr(mod, "load_entries", lambda: fake_entries)

    result = probability_calibration(horizon="1d")
    filled = [b for b in result if b["sample_count"] > 0]
    assert len(filled) == 1
    assert filled[0]["sample_count"] == 2
    assert filled[0]["correct_count"] == 1
    assert filled[0]["actual_accuracy"] == 0.5


def test_probability_calibration_hold_consensus_skipped(monkeypatch):
    """Entries with HOLD consensus are skipped entirely."""
    import portfolio.accuracy_stats as mod

    fake_entries = [
        {
            "ts": "2026-03-01T00:00:00+00:00",
            "tickers": {
                "BTC-USD": {
                    "consensus": "HOLD",
                    "buy_count": 1,
                    "sell_count": 1,
                    "total_voters": 2,
                    "signals": {},
                },
            },
            "outcomes": {
                "BTC-USD": {"1d": {"change_pct": 5.0}},
            },
        },
    ]
    monkeypatch.setattr(mod, "load_entries", lambda: fake_entries)

    result = probability_calibration(horizon="1d")
    total_samples = sum(b["sample_count"] for b in result)
    assert total_samples == 0
