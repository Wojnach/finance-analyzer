"""Tests for the 2026-05-15 sentiment sub-voter fan-out.

`portfolio/sentiment.py._log_sub_vote()` emits one row per sentiment
sub-model (trading_hero / cryptobert / finbert / fingpt) into
`data/llm_probability_log.jsonl`. This breaks the 4-into-1 ensemble
masking that hid individual sub-model accuracy inside the aggregate
`sentiment` voter (46% on 40K samples).

These tests verify the contract without invoking real sentiment
inference (no model load, no network calls):

* The helper drops rows for non-LLM signals (defensive).
* It writes valid {BUY/HOLD/SELL} probs that sum to 1.
* It uses the canonical `BUY`/`HOLD`/`SELL` mapping for positive /
  neutral / negative labels.
* It tolerates missing or malformed avg_scores without raising.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import pytest

from portfolio import llm_probability_log as plog
from portfolio.sentiment import _log_sub_vote


@pytest.fixture
def isolated_log(monkeypatch):
    tmp = pathlib.Path(tempfile.mkdtemp()) / "log.jsonl"
    monkeypatch.setattr(plog, "_PROB_LOG", tmp)
    return tmp


def _rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_positive_sentiment_maps_to_buy(isolated_log):
    _log_sub_vote(
        "trading_hero", "BTC-USD", "positive",
        {"positive": 0.7, "negative": 0.2, "neutral": 0.1},
    )
    rows = _rows(isolated_log)
    assert len(rows) == 1
    assert rows[0]["signal"] == "trading_hero"
    assert rows[0]["ticker"] == "BTC-USD"
    assert rows[0]["chosen"] == "BUY"
    assert abs(sum(rows[0]["probs"].values()) - 1.0) < 1e-6


def test_negative_sentiment_maps_to_sell(isolated_log):
    _log_sub_vote(
        "cryptobert", "ETH-USD", "negative",
        {"positive": 0.1, "negative": 0.8, "neutral": 0.1},
    )
    rows = _rows(isolated_log)
    assert rows[0]["chosen"] == "SELL"
    assert rows[0]["probs"]["SELL"] > rows[0]["probs"]["BUY"]


def test_neutral_sentiment_maps_to_hold(isolated_log):
    _log_sub_vote(
        "finbert", "XAG-USD", "neutral",
        {"positive": 0.2, "negative": 0.3, "neutral": 0.5},
    )
    rows = _rows(isolated_log)
    assert rows[0]["chosen"] == "HOLD"


def test_fingpt_emits_via_helper(isolated_log):
    """fingpt rows arrive via flush_ab_log post-batch, but the same
    helper handles them — verify directly."""
    _log_sub_vote(
        "fingpt", "BTC-USD", "positive",
        {"positive": 0.6, "negative": 0.25, "neutral": 0.15},
    )
    rows = _rows(isolated_log)
    assert rows[0]["signal"] == "fingpt"


def test_non_llm_signal_silently_dropped(isolated_log):
    """Defense-in-depth: even though portfolio/sentiment.py only ever
    calls _log_sub_vote with sub-voter names, log_vote() inside it
    drops any row whose signal isn't in _LLM_SIGNALS so a typo could
    not pollute the log."""
    _log_sub_vote(
        "rsi", "BTC-USD", "positive",
        {"positive": 0.7, "negative": 0.2, "neutral": 0.1},
    )
    assert _rows(isolated_log) == []


def test_malformed_avg_scores_does_not_raise(isolated_log):
    """A buggy upstream that hands us malformed avg_scores must not
    blow up the sentiment-compute path. _log_sub_vote swallows
    exceptions by design."""
    # Missing the sentiment_label key entirely.
    _log_sub_vote("finbert", "MSTR", "positive", {"negative": 0.5})
    # Empty dict.
    _log_sub_vote("finbert", "MSTR", "positive", {})
    # None inside avg_scores.
    _log_sub_vote("finbert", "MSTR", "positive", {"positive": None})
    # All zeros — degenerate but well-formed.
    _log_sub_vote(
        "finbert", "MSTR", "positive",
        {"positive": 0.0, "negative": 0.0, "neutral": 0.0},
    )
    # No assertions: only that no exception escaped. log file may have
    # zero or more rows depending on how derive_probs_from_result
    # handles each malformed input — both behaviors are acceptable.
