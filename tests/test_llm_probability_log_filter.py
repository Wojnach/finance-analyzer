"""Tests for `portfolio.llm_probability_log.is_directional_prediction`.

This filter is the source of truth for which rows count toward accuracy
in BOTH `scripts/review_shadow_signals.py` and
`dashboard/app.py:_compute_llm_leaderboard`. Per the 2026-05-18 plan
(`.worktrees/shadow-gate-lora-20260518/docs/PLAN.md`) the two consumers
must agree or the promotion gate and the leaderboard will publish
different "accuracy" numbers for the same signal.
"""
from __future__ import annotations

from portfolio.llm_probability_log import is_directional_prediction


def test_real_buy_with_confidence_is_directional():
    row = {"signal": "ministral", "chosen": "BUY", "confidence": 0.65}
    assert is_directional_prediction(row) is True


def test_real_sell_with_confidence_is_directional():
    row = {"signal": "qwen3", "chosen": "SELL", "confidence": 0.55}
    assert is_directional_prediction(row) is True


def test_hold_vote_is_not_directional_even_with_high_confidence():
    # HOLD is non-information for direction; matches the accuracy_cache.json
    # methodology where total = total_buy + total_sell (HOLDs excluded).
    row = {"signal": "ministral", "chosen": "HOLD", "confidence": 0.8}
    assert is_directional_prediction(row) is False


def test_abstain_row_with_zero_confidence_is_not_directional():
    # Canonical _abstain() shape from portfolio/signals/finance_llama.py etc.
    row = {"signal": "cryptotrader_lm", "chosen": "HOLD", "confidence": 0.0}
    assert is_directional_prediction(row) is False


def test_buy_with_zero_confidence_is_not_directional():
    # Theoretical edge case: a model emits BUY with conf=0. We treat as
    # abstain because the abstain convention sets conf=0; allowing it
    # through would let a buggy scaffold game the gate by emitting BUY.
    row = {"signal": "meta_trader", "chosen": "BUY", "confidence": 0.0}
    assert is_directional_prediction(row) is False


def test_missing_confidence_is_not_directional():
    # Legacy rows pre-2026-04 lack the confidence field; treat them
    # conservatively as non-directional so only post-fix rows count.
    row = {"signal": "ministral", "chosen": "BUY"}
    assert is_directional_prediction(row) is False


def test_none_confidence_is_not_directional():
    row = {"signal": "ministral", "chosen": "BUY", "confidence": None}
    assert is_directional_prediction(row) is False


def test_non_numeric_confidence_is_not_directional():
    row = {"signal": "ministral", "chosen": "BUY", "confidence": "high"}
    assert is_directional_prediction(row) is False


def test_invalid_chosen_value_is_not_directional():
    row = {"signal": "ministral", "chosen": "MAYBE", "confidence": 0.7}
    assert is_directional_prediction(row) is False


def test_non_dict_row_is_not_directional():
    assert is_directional_prediction(None) is False
    assert is_directional_prediction("not-a-dict") is False
    assert is_directional_prediction([]) is False


def test_negative_confidence_is_not_directional():
    # Should never happen (log_vote rejects), but be defensive.
    row = {"signal": "ministral", "chosen": "SELL", "confidence": -0.1}
    assert is_directional_prediction(row) is False
