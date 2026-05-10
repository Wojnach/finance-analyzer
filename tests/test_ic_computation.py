"""Tests for IC computation module."""

from pathlib import Path

from portfolio.ic_computation import (
    _spearman_rank_correlation,
    compute_signal_ic,
    compute_signal_ic_per_ticker,
    get_signal_ic_ranking,
)


class TestSpearmanRankCorrelation:
    def test_perfect_positive(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
             11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        y = list(x)
        rho, n = _spearman_rank_correlation(x, y)
        assert n == 30
        assert abs(rho - 1.0) < 0.001

    def test_perfect_negative(self):
        x = list(range(1, 31))
        y = list(range(30, 0, -1))
        rho, n = _spearman_rank_correlation(x, y)
        assert n == 30
        assert abs(rho - (-1.0)) < 0.001

    def test_weak_correlation(self):
        x = list(range(1, 31))
        y = [15, 3, 28, 7, 22, 11, 19, 5, 26, 9,
             20, 1, 30, 8, 24, 13, 17, 4, 27, 10,
             21, 2, 29, 6, 23, 12, 18, 14, 25, 16]
        rho, n = _spearman_rank_correlation(x, y)
        assert n == 30
        assert abs(rho) < 0.4

    def test_too_few_samples(self):
        x = [1, 2, 3]
        y = [4, 5, 6]
        rho, n = _spearman_rank_correlation(x, y)
        assert n == 3
        assert rho == 0.0

    def test_ties_handled(self):
        x = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
             6, 6, 7, 7, 8, 8, 9, 9, 10, 10,
             11, 11, 12, 12, 13, 13, 14, 14, 15, 15]
        y = list(range(1, 31))
        rho, n = _spearman_rank_correlation(x, y)
        assert n == 30
        assert rho > 0.9

    def test_constant_values(self):
        x = [1] * 30
        y = list(range(30))
        rho, _ = _spearman_rank_correlation(x, y)
        assert rho == 0.0


class TestComputeSignalIC:
    def _make_entries(self, signal_name, votes_and_returns):
        entries = []
        for i, (vote, ret) in enumerate(votes_and_returns):
            entries.append({
                "ts": f"2026-04-01T{i:02d}:00:00+00:00",
                "tickers": {
                    "BTC-USD": {
                        "signals": {signal_name: vote},
                        "price_usd": 70000,
                    }
                },
                "outcomes": {
                    "BTC-USD": {
                        "1d": {"change_pct": ret, "price_after": 70000 + ret * 700}
                    }
                },
            })
        return entries

    def test_positive_ic_signal(self):
        data = [("BUY", 1.5)] * 20 + [("SELL", -1.2)] * 20
        entries = self._make_entries("rsi", data)
        result = compute_signal_ic(horizon="1d", entries=entries)
        assert result["rsi"]["ic"] > 0.5
        assert result["rsi"]["samples"] == 40

    def test_negative_ic_signal(self):
        data = [("BUY", -1.5)] * 20 + [("SELL", 1.2)] * 20
        entries = self._make_entries("macd", data)
        result = compute_signal_ic(horizon="1d", entries=entries)
        assert result["macd"]["ic"] < -0.5

    def test_hold_votes_excluded(self):
        data = [("HOLD", 1.5)] * 40
        entries = self._make_entries("ema", data)
        result = compute_signal_ic(horizon="1d", entries=entries)
        assert result["ema"]["samples"] == 0
        assert result["ema"]["ic"] == 0.0

    def test_insufficient_samples(self):
        data = [("BUY", 1.0)] * 5
        entries = self._make_entries("bb", data)
        result = compute_signal_ic(horizon="1d", entries=entries)
        assert result["bb"]["samples"] == 5
        assert result["bb"]["ic"] == 0.0


class TestComputeSignalICPerTicker:
    def test_per_ticker_separation(self):
        import random
        random.seed(42)
        entries = []
        for i in range(40):
            btc_vote = "BUY" if i % 2 == 0 else "SELL"
            btc_ret = (1.5 + random.random()) if btc_vote == "BUY" else (-1.2 - random.random())
            eth_vote = "SELL" if i % 2 == 0 else "BUY"
            eth_ret = (-1.0 - random.random()) if eth_vote == "SELL" else (0.8 + random.random())
            entries.append({
                "ts": f"2026-04-{(i % 28) + 1:02d}T{i % 24:02d}:00:00+00:00",
                "tickers": {
                    "BTC-USD": {"signals": {"rsi": btc_vote}, "price_usd": 70000},
                    "ETH-USD": {"signals": {"rsi": eth_vote}, "price_usd": 2000},
                },
                "outcomes": {
                    "BTC-USD": {"1d": {"change_pct": btc_ret}},
                    "ETH-USD": {"1d": {"change_pct": eth_ret}},
                },
            })
        result = compute_signal_ic_per_ticker(horizon="1d", entries=entries)
        assert "BTC-USD" in result
        assert "ETH-USD" in result
        assert result["BTC-USD"]["rsi"]["ic"] > 0.3
        assert result["ETH-USD"]["rsi"]["ic"] > 0.3


class TestGetSignalICRanking:
    def test_ranking_order(self, monkeypatch):
        # 2026-05-10: ranking now filters NEGATIVE IC signals — macd's
        # IC=-0.08 is no longer ranked at all. Returned list contains only
        # signals with positive IC, ordered by IC desc. Test was written
        # before the filter and asserted macd at position 2.
        mock_cache = {
            "time": 9999999999,
            "horizon": "1d",
            "global": {
                "rsi": {"ic": 0.15, "samples": 100},
                "macd": {"ic": -0.08, "samples": 50},  # negative — filtered
                "ema": {"ic": 0.02, "samples": 200},
            },
        }
        monkeypatch.setattr(
            "portfolio.ic_computation.load_cached_ic",
            lambda h: mock_cache,
        )
        ranked = get_signal_ic_ranking(horizon="1d", min_samples=30)
        names = [r[0] for r in ranked]
        assert names == ["rsi", "ema"]  # macd filtered out (negative IC)


class TestDataDirIsAbsolute:
    """Regression for adversarial review 05-01 P0-2: DATA_DIR was a relative
    `Path("data")` which silently broke when the scheduled task CWD differed
    from the repo root. Every other module uses
    `Path(__file__).resolve().parent.parent / "data"`.
    """

    def test_data_dir_is_absolute(self):
        """DATA_DIR must be an absolute path."""
        from portfolio.ic_computation import DATA_DIR
        assert DATA_DIR.is_absolute(), (
            f"DATA_DIR must be absolute, got {DATA_DIR!r}. If you reverted to "
            "Path('data'), please re-read adversarial review 05-01 P0-2."
        )

    def test_data_dir_matches_repo_data_dir(self):
        """DATA_DIR must point at the repo's `data` directory regardless of CWD."""
        from portfolio.ic_computation import DATA_DIR
        # Resolve repo root from this test file's location: tests/.. == repo root
        repo_root = Path(__file__).resolve().parent.parent
        expected = repo_root / "data"
        assert DATA_DIR == expected, (
            f"DATA_DIR={DATA_DIR!r} doesn't match expected repo data dir {expected!r}."
        )

    def test_ic_cache_file_under_data_dir(self):
        """IC_CACHE_FILE must live under DATA_DIR (absolute)."""
        from portfolio.ic_computation import DATA_DIR, IC_CACHE_FILE
        assert IC_CACHE_FILE.is_absolute()
        assert IC_CACHE_FILE.parent == DATA_DIR

    def test_signal_log_file_under_data_dir(self):
        """SIGNAL_LOG_FILE must live under DATA_DIR (absolute)."""
        from portfolio.ic_computation import DATA_DIR, SIGNAL_LOG_FILE
        assert SIGNAL_LOG_FILE.is_absolute()
        assert SIGNAL_LOG_FILE.parent == DATA_DIR
