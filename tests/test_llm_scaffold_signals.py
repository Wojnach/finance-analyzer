"""Tests for the three LLM scaffold signal modules added 2026-05-15.

These modules deliberately return abstention (HOLD / conf=0 /
feature_unavailable=True) until real inference is wired. The contract
these tests pin down:

* They import without side effects (no model load at import time).
* They return the canonical enhanced-signal result dict shape.
* They never claim a non-HOLD action while feature_unavailable=True.
* cryptotrader_lm refuses to vote on non-crypto tickers.
* They are registered with signal_registry.

If these tests start failing because real inference shipped, the right
move is to fork them into per-model integration tests, not delete them.
"""

from __future__ import annotations

import pandas as pd
import pytest

from portfolio.signals.cryptotrader_lm import compute_cryptotrader_lm_signal
from portfolio.signals.finance_llama import compute_finance_llama_signal
from portfolio.signals.meta_trader import compute_meta_trader_signal


@pytest.fixture
def ohlcv():
    return pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2],
            "high": [1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1],
            "close": [1.1, 1.2, 1.3],
            "volume": [100, 110, 120],
        }
    )


def _assert_abstention(result, signal_name):
    assert result["action"] == "HOLD"
    assert result["confidence"] == 0.0
    assert result["sub_signals"] == {signal_name: "HOLD"}
    assert result["indicators"]["feature_unavailable"] is True


def test_finance_llama_returns_abstention(ohlcv):
    r = compute_finance_llama_signal(ohlcv, context={"ticker": "BTC-USD"})
    _assert_abstention(r, "finance_llama")
    assert r["indicators"]["reason"] == "scaffold"


def test_cryptotrader_lm_returns_abstention_on_crypto(ohlcv):
    r = compute_cryptotrader_lm_signal(ohlcv, context={"ticker": "BTC-USD"})
    _assert_abstention(r, "cryptotrader_lm")
    assert r["indicators"]["reason"] == "scaffold"


def test_cryptotrader_lm_refuses_non_crypto(ohlcv):
    """Even after real inference ships, this signal must refuse non-crypto
    tickers because the LoRA was trained only on BTC/ETH. The scaffold's
    refusal path must remain in place."""
    r = compute_cryptotrader_lm_signal(ohlcv, context={"ticker": "XAG-USD"})
    _assert_abstention(r, "cryptotrader_lm")
    assert r["indicators"]["reason"] == "non_crypto_ticker"
    assert r["indicators"]["ticker"] == "XAG-USD"


def test_meta_trader_returns_abstention(ohlcv):
    r = compute_meta_trader_signal(ohlcv, context={"ticker": "BTC-USD"})
    _assert_abstention(r, "meta_trader")
    assert r["indicators"]["reason"] == "scaffold"


def test_scaffolds_accept_no_context(ohlcv):
    """signal_engine may call enhanced compute fns without a context
    when requires_context=False (defensive — these all set
    requires_context=True so the dispatch always passes one, but the
    callable contract should still tolerate missing context)."""
    assert compute_finance_llama_signal(ohlcv)["action"] == "HOLD"
    assert compute_cryptotrader_lm_signal(ohlcv)["action"] == "HOLD"
    assert compute_meta_trader_signal(ohlcv)["action"] == "HOLD"


def test_scaffolds_registered_in_signal_registry():
    from portfolio.signal_registry import get_enhanced_signals

    sigs = get_enhanced_signals()
    for name in ("finance_llama", "cryptotrader_lm", "meta_trader"):
        assert name in sigs, f"{name} not registered"
        assert sigs[name]["requires_context"] is True
        assert sigs[name]["max_confidence"] == 0.7


def test_scaffolds_in_llm_probability_log_signal_set():
    """log_vote() will silently drop rows for any signal not in
    _LLM_SIGNALS. Verify the three scaffolds are members so abstention
    rows reach the log."""
    from portfolio.llm_probability_log import is_llm_signal

    for name in ("finance_llama", "cryptotrader_lm", "meta_trader"):
        assert is_llm_signal(name), f"{name} missing from _LLM_SIGNALS"


def test_scaffolds_results_validate_under_signal_engine():
    """The dispatch loop runs results through _validate_signal_result.
    Confirm our shape passes that gate so we don't silently get HOLD'd
    by the validator instead of by intent."""
    from portfolio.signal_engine import _validate_signal_result

    df = pd.DataFrame({"close": [1.0, 1.1]})
    for name, fn in (
        ("finance_llama", compute_finance_llama_signal),
        ("cryptotrader_lm", compute_cryptotrader_lm_signal),
        ("meta_trader", compute_meta_trader_signal),
    ):
        v = _validate_signal_result(
            fn(df, context={"ticker": "BTC-USD"}),
            sig_name=name,
            max_confidence=0.7,
        )
        assert v["action"] == "HOLD"
        assert v["confidence"] == 0.0
