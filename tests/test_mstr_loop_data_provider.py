"""Unit tests for portfolio.mstr_loop.data_provider."""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portfolio.mstr_loop import config
from portfolio.mstr_loop.data_provider import (
    _compute_weighted_scores,
    _parse_vote_detail,
    build_bundle,
)


# ---------------------------------------------------------------------------
# _parse_vote_detail
# ---------------------------------------------------------------------------


def test_parse_vote_detail_basic():
    votes = _parse_vote_detail(
        "B:sentiment,volume,calendar | S:smart_money,mean_reversion"
    )
    assert votes == {
        "sentiment": "BUY",
        "volume": "BUY",
        "calendar": "BUY",
        "smart_money": "SELL",
        "mean_reversion": "SELL",
    }


def test_parse_vote_detail_empty_string():
    assert _parse_vote_detail("") == {}


def test_parse_vote_detail_with_hold_segment():
    votes = _parse_vote_detail("B:a | H:b,c | S:d")
    assert votes == {"a": "BUY", "b": "HOLD", "c": "HOLD", "d": "SELL"}


def test_parse_vote_detail_whitespace_tolerant():
    votes = _parse_vote_detail(" B : sig1 , sig2 | S : sig3 ")
    assert votes == {"sig1": "BUY", "sig2": "BUY", "sig3": "SELL"}


def test_parse_vote_detail_drops_malformed_segments():
    # Missing colon in second segment → silently dropped.
    votes = _parse_vote_detail("B:a | garbage | S:b")
    assert votes == {"a": "BUY", "b": "SELL"}


# ---------------------------------------------------------------------------
# _compute_weighted_scores
# ---------------------------------------------------------------------------


def test_weighted_scores_top_signals_dominate(monkeypatch):
    """ministral (weight 3) + calendar (weight 2.5) BUY beats 3 default (1.0) SELLs."""
    monkeypatch.setattr(
        config, "MSTR_SIGNAL_WEIGHTS",
        {"ministral": 3.0, "calendar": 2.5},
    )
    monkeypatch.setattr(config, "DEFAULT_SIGNAL_WEIGHT", 1.0)
    votes = {
        "ministral": "BUY",
        "calendar": "BUY",
        "other1": "SELL",
        "other2": "SELL",
        "other3": "SELL",
    }
    long_s, short_s = _compute_weighted_scores(votes)
    # num: 3.0 + 2.5 = 5.5 LONG; 1.0 + 1.0 + 1.0 = 3.0 SHORT
    # denom: 8.5
    assert long_s == pytest.approx(5.5 / 8.5, abs=0.001)
    assert short_s == pytest.approx(3.0 / 8.5, abs=0.001)
    assert long_s > short_s


def test_weighted_scores_zero_weight_ignored(monkeypatch):
    """Signals with 0.0 weight don't contribute to numerator OR denominator."""
    monkeypatch.setattr(
        config, "MSTR_SIGNAL_WEIGHTS",
        {"trend": 0.0, "volatility_sig": 0.0, "ministral": 3.0},
    )
    monkeypatch.setattr(config, "DEFAULT_SIGNAL_WEIGHT", 1.0)
    votes = {
        "trend": "SELL",           # weight 0, ignored
        "volatility_sig": "SELL",  # weight 0, ignored
        "ministral": "BUY",        # weight 3
    }
    long_s, short_s = _compute_weighted_scores(votes)
    # Only ministral counts: 3/3 = 1.0 long
    assert long_s == pytest.approx(1.0)
    assert short_s == pytest.approx(0.0)


def test_weighted_scores_hold_dilutes():
    """HOLD votes count in denom but not numerator (dilute scores)."""
    votes = {"a": "BUY", "b": "BUY", "c": "HOLD", "d": "HOLD"}
    long_s, short_s = _compute_weighted_scores(votes)
    # 4 signals all weight 1.0 (default); 2 BUY + 2 HOLD
    # long = 2/4 = 0.5; short = 0/4 = 0
    assert long_s == pytest.approx(0.5)
    assert short_s == pytest.approx(0.0)


def test_weighted_scores_empty_votes():
    assert _compute_weighted_scores({}) == (0.0, 0.0)


def test_weighted_scores_all_zero_weights(monkeypatch):
    """If every voter has 0 weight, scores stay 0 — no div-by-zero."""
    monkeypatch.setattr(config, "MSTR_SIGNAL_WEIGHTS", {"x": 0.0, "y": 0.0})
    monkeypatch.setattr(config, "DEFAULT_SIGNAL_WEIGHT", 0.0)
    votes = {"x": "BUY", "y": "SELL"}
    long_s, short_s = _compute_weighted_scores(votes)
    assert long_s == 0.0
    assert short_s == 0.0


# ---------------------------------------------------------------------------
# build_bundle
# ---------------------------------------------------------------------------


def _write_agent_summary(tmp_path, sig_block=None, mc_block=None, timeframes=None):
    path = tmp_path / "agent_summary_compact.json"
    payload = {
        "signals": {"MSTR": sig_block or {
            "action": "BUY", "weighted_confidence": 0.6, "price_usd": 170.0,
            "rsi": 55, "macd_hist": 1.0, "bb_position": "inside",
            "atr_pct": 1.5, "regime": "trending-up", "stale": False,
            "extra": {
                "_voters": 8, "_buy_count": 5, "_sell_count": 3,
                "_vote_detail": "B:ministral,calendar,volume | S:rsi,bb,mean_reversion",
            },
        }},
        "monte_carlo": {"MSTR": mc_block or {
            "p_up": 0.65,
            "expected_return_1d": {"mean_pct": 0.3},
            "expected_return_3d": {"mean_pct": 0.8},
        }},
        "timeframes": {"MSTR": timeframes or [
            {"horizon": "Now", "action": "BUY"},
            {"horizon": "12h", "action": "BUY"},
        ]},
    }
    path.write_text(json.dumps(payload))
    return str(path)


def test_build_bundle_happy_path(tmp_path):
    path = _write_agent_summary(tmp_path)
    b = build_bundle(agent_summary_path=path, ticker="MSTR")
    assert b is not None
    assert b.price_usd == 170.0
    assert b.raw_action == "BUY"
    assert b.rsi == 55
    assert b.weighted_score_long > 0
    assert b.p_up_1d == 0.65
    assert b.exp_return_3d_pct == 0.8
    assert b.stale is False
    assert b.is_usable()


def test_build_bundle_missing_file_returns_none(tmp_path):
    b = build_bundle(agent_summary_path=str(tmp_path / "missing.json"))
    assert b is None


def test_build_bundle_missing_ticker_returns_none(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"signals": {}, "monte_carlo": {}, "timeframes": {}}))
    assert build_bundle(agent_summary_path=str(path), ticker="MSTR") is None


def test_build_bundle_is_usable_false_when_stale(tmp_path):
    path = _write_agent_summary(tmp_path, sig_block={
        "action": "BUY", "price_usd": 170.0, "rsi": 55, "regime": "trending-up",
        "stale": True,
        "extra": {"_vote_detail": "B:x | S:y"},
    })
    b = build_bundle(agent_summary_path=path)
    assert b is not None
    assert b.stale is True
    assert b.is_usable() is False


def test_build_bundle_is_usable_false_when_source_old(tmp_path, monkeypatch):
    path = _write_agent_summary(tmp_path)
    old_time = time.time() - 600  # 10 min old
    os.utime(path, (old_time, old_time))
    b = build_bundle(agent_summary_path=path)
    assert b is not None
    assert b.source_stale_seconds > 300
    assert b.is_usable() is False


def test_build_bundle_is_usable_false_when_price_zero(tmp_path):
    path = _write_agent_summary(tmp_path, sig_block={
        "action": "BUY", "price_usd": 0, "rsi": 55, "regime": "trending-up",
        "stale": False,
        "extra": {"_vote_detail": "B:x | S:y"},
    })
    b = build_bundle(agent_summary_path=path)
    assert b is not None
    assert b.is_usable() is False
