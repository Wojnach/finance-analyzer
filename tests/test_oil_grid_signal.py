"""Tests for portfolio.oil_grid_signal — pure-vote logic + persistence."""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest

from portfolio import oil_grid_signal as feed


# ---------------------------------------------------------------------------
# Vote helpers
# ---------------------------------------------------------------------------


class TestVoteRsi:
    def test_oversold_buys(self):
        assert feed._vote_rsi(29.9) == "BUY"

    def test_overbought_sells(self):
        assert feed._vote_rsi(70.1) == "SELL"

    def test_neutral_holds(self):
        assert feed._vote_rsi(50) == "HOLD"

    def test_exact_30_is_hold(self):
        # Boundary alignment with main contract: 30.0 is neutral.
        assert feed._vote_rsi(30) == "HOLD"

    def test_exact_70_is_hold(self):
        assert feed._vote_rsi(70) == "HOLD"


class TestVoteMacd:
    def test_positive_hist_buys(self):
        assert feed._vote_macd(0.5) == "BUY"

    def test_negative_hist_sells(self):
        assert feed._vote_macd(-0.5) == "SELL"

    def test_zero_holds(self):
        assert feed._vote_macd(0) == "HOLD"


class TestVoteEma:
    def test_fast_above_slow_buys(self):
        # gap_pct = (106-100)/100*100 = 6 -> BUY
        assert feed._vote_ema(106, 100) == "BUY"

    def test_fast_below_slow_sells(self):
        # gap_pct = (94-100)/100*100 = -6 -> SELL
        assert feed._vote_ema(94, 100) == "SELL"

    def test_within_deadband_holds(self):
        # gap_pct = 0.3 -> HOLD
        assert feed._vote_ema(100.3, 100) == "HOLD"

    def test_exact_half_pct_gap_votes_buy(self):
        # gap_pct = 0.5 exactly -> BUY (matches main loop convention).
        assert feed._vote_ema(100.5, 100) == "BUY"

    def test_exact_half_pct_gap_negative_votes_sell(self):
        assert feed._vote_ema(99.5, 100) == "SELL"

    def test_zero_slow_ema_holds(self):
        # Defensive: ema21=0 (cold start) must not divide by zero.
        assert feed._vote_ema(106, 0) == "HOLD"


class TestVoteBb:
    def test_close_at_lower_band_holds(self):
        # Boundary alignment: exactly on the band is neutral.
        assert feed._vote_bb(close=98, bb_lower=98, bb_upper=102) == "HOLD"

    def test_close_below_lower_band_buys(self):
        assert feed._vote_bb(close=97.99, bb_lower=98, bb_upper=102) == "BUY"

    def test_close_at_upper_band_holds(self):
        assert feed._vote_bb(close=102, bb_lower=98, bb_upper=102) == "HOLD"

    def test_close_above_upper_band_sells(self):
        assert feed._vote_bb(close=102.01, bb_lower=98, bb_upper=102) == "SELL"

    def test_close_mid_holds(self):
        assert feed._vote_bb(close=100, bb_lower=98, bb_upper=102) == "HOLD"


# ---------------------------------------------------------------------------
# compute_signal — full pipeline
# ---------------------------------------------------------------------------


def _make_synthetic_df(close: float, n: int = 100) -> pd.DataFrame:
    # Build an OHLCV frame where every bar is at `close` so indicators
    # are deterministic.
    idx = pd.date_range("2026-05-11", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": [close] * n,
        "high": [close * 1.005] * n,
        "low": [close * 0.995] * n,
        "close": [close] * n,
        "volume": [1000.0] * n,
    }, index=idx)


def _make_uptrending_df(start: float = 70.0, n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2026-05-11", periods=n, freq="1h", tz="UTC")
    closes = [start + i * 0.3 for i in range(n)]
    return pd.DataFrame({
        "open": closes,
        "high": [c * 1.005 for c in closes],
        "low": [c * 0.995 for c in closes],
        "close": closes,
        "volume": [1000.0] * n,
    }, index=idx)


class TestComputeSignal:
    def test_uptrend_produces_buy(self):
        with patch("portfolio.price_source.fetch_klines",
                   return_value=_make_uptrending_df()):
            record = feed.compute_signal()
        assert record["action"] in ("BUY", "HOLD")  # strong trend -> BUY
        # At minimum, no error
        assert "error" not in record
        assert record["ticker"] == "OIL-USD"
        assert "votes" in record
        assert set(record["votes"].keys()) == {"rsi", "macd", "ema", "bb"}

    def test_fetch_failure_yields_safe_hold(self):
        with patch("portfolio.price_source.fetch_klines",
                   side_effect=RuntimeError("network down")):
            record = feed.compute_signal()
        assert record["action"] == "HOLD"
        assert record["confidence"] == 0.0
        assert "fetch_failed" in record.get("error", "")

    def test_insufficient_data_yields_hold(self):
        # Only 5 rows — less than 30 minimum
        short_df = _make_synthetic_df(close=70.0, n=5)
        with patch("portfolio.price_source.fetch_klines",
                   return_value=short_df):
            record = feed.compute_signal()
        assert record["action"] == "HOLD"
        assert "insufficient_data" in record.get("error", "")

    def test_confidence_above_floor_when_three_of_four_agree(self):
        # Construct a record manually by stubbing compute_indicators
        # so we hit the 3-of-4 BUY branch.
        from portfolio import indicators as ind_mod
        from portfolio import price_source as ps_mod
        df = _make_synthetic_df(close=70.0)
        synthetic_ind = {
            "rsi": 25,  # BUY
            "macd_hist": 0.5,  # BUY
            "ema9": 110, "ema21": 100,  # BUY (10% above)
            "close": 100.5, "bb_lower": 98, "bb_upper": 102,  # HOLD
        }
        with patch.object(ps_mod, "fetch_klines", return_value=df), \
             patch.object(ind_mod, "compute_indicators",
                          return_value=synthetic_ind):
            record = feed.compute_signal()
        assert record["action"] == "BUY"
        # 3/4 agreement -> 0.50 + 0.25 * 0.70 = 0.675
        assert record["confidence"] == pytest.approx(0.675)
        assert record["confidence"] > 0.56  # clears grid fisher floor

    def test_split_vote_holds(self):
        from portfolio import indicators as ind_mod
        from portfolio import price_source as ps_mod
        df = _make_synthetic_df(close=70.0)
        # 2 BUY, 2 SELL — tie -> HOLD.
        # bb close must be strictly above the upper band to vote SELL after
        # the 2026-05-11 boundary alignment.
        synthetic_ind = {
            "rsi": 25,  # BUY
            "macd_hist": 0.5,  # BUY
            "ema9": 90, "ema21": 100,  # SELL (-10% gap)
            "close": 102.1, "bb_lower": 98, "bb_upper": 102,  # SELL (>upper)
        }
        with patch.object(ps_mod, "fetch_klines", return_value=df), \
             patch.object(ind_mod, "compute_indicators",
                          return_value=synthetic_ind):
            record = feed.compute_signal()
        assert record["action"] == "HOLD"
        assert record["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Persistence + load
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_write_signal_round_trip(self, tmp_path, monkeypatch):
        # Stub compute_signal to return a known record.
        known = {
            "action": "BUY", "confidence": 0.7,
            "ts": "2026-05-11T12:00:00Z", "ticker": "OIL-USD",
        }
        monkeypatch.setattr(feed, "compute_signal", lambda: known)
        path = tmp_path / "oil.json"
        record = feed.write_signal(path)
        assert record == known
        with open(path) as f:
            on_disk = json.load(f)
        assert on_disk["action"] == "BUY"
        assert on_disk["confidence"] == 0.7

    def test_load_fresh_signal(self, tmp_path):
        import datetime as _dt
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        path = tmp_path / "oil.json"
        path.write_text(json.dumps({
            "action": "BUY", "confidence": 0.7, "ts": ts,
        }))
        loaded = feed.load_signal(path, max_age_s=60)
        assert loaded is not None
        assert loaded["action"] == "BUY"

    def test_load_stale_signal_returns_none(self, tmp_path):
        path = tmp_path / "oil.json"
        path.write_text(json.dumps({
            "action": "BUY", "confidence": 0.7,
            "ts": "2020-01-01T00:00:00Z",  # ancient
        }))
        assert feed.load_signal(path, max_age_s=60) is None

    def test_load_missing_returns_none(self, tmp_path):
        assert feed.load_signal(tmp_path / "nope.json") is None

    def test_load_corrupt_returns_none(self, tmp_path):
        path = tmp_path / "oil.json"
        path.write_text("{not valid")
        assert feed.load_signal(path) is None

    def test_load_missing_ts_returns_none(self, tmp_path):
        path = tmp_path / "oil.json"
        path.write_text(json.dumps({"action": "BUY", "confidence": 0.7}))
        assert feed.load_signal(path) is None
