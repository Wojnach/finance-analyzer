"""Integration tests for cryptotrader_lm real PEFT LoRA inference.

Mocks `portfolio.llama_server.query_llama_server` to keep tests offline.
Verifies the same contract as `finance_llama`:

* Non-crypto ticker still refused (regression guard for the LoRA
  out-of-distribution risk).
* Server-unavailable / Plex-VRAM / inference-error abstention paths.
* Well-formed JSON BUY/SELL/HOLD parsing via shared
  `ministral_trader._parse_response`.
* Confidence regex fallback.
* `_FEATURE_AVAILABLE=True` sentinel flipped.
* Result validates under `signal_engine._validate_signal_result`.
* Stop tokens forwarded.
* Crypto-only guard ordering: ticker check runs BEFORE
  `_FEATURE_AVAILABLE` so non-crypto requests never reach the
  inference path even if the feature is later disabled.
"""
from __future__ import annotations

import pandas as pd

import portfolio.llama_server as _llama_server
from portfolio.signals.cryptotrader_lm import compute_cryptotrader_lm_signal


def _ctx(ticker="BTC-USD", **overrides):
    base = {
        "ticker": ticker,
        "price_usd": 78_293.0,
        "rsi": 55.0,
        "macd_hist": 2.4,
        "ema_bullish": True,
        "ema_gap_pct": 1.0,
        "bb_position": "middle",
        "volume_ratio": 1.2,
        "fear_greed": 31,
        "fear_greed_class": "Fear",
        "news_sentiment": "neutral",
        "sentiment_confidence": 0.5,
        "timeframe_summary": "7d up",
        "headlines": "mixed",
        "change_24h": -0.3,
        "asset_type": "cryptocurrency",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Crypto-only refusal
# ---------------------------------------------------------------------------

def test_non_crypto_ticker_refused_unconditionally(monkeypatch):
    """LoRA was trained only on BTC/ETH. Metals and stocks must always
    receive the non_crypto_ticker abstention regardless of feature flag
    or server state."""
    # Spy on query_llama_server — must NOT be called for non-crypto.
    called = {"n": 0}
    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    monkeypatch.setattr(_llama_server, "query_llama_server",
                        lambda *a, **kw: (called.__setitem__("n", called["n"] + 1) or "BUY"))
    for non_crypto in ("XAG-USD", "XAU-USD", "MSTR", "SPY"):
        r = compute_cryptotrader_lm_signal(pd.DataFrame({"close": [1.0]}), context=_ctx(ticker=non_crypto))
        assert r["action"] == "HOLD"
        assert r["confidence"] == 0.0
        assert r["indicators"]["reason"] == "non_crypto_ticker"
        assert r["indicators"]["ticker"] == non_crypto
    assert called["n"] == 0, "non-crypto must short-circuit before query"


def test_btc_ticker_allowed(monkeypatch):
    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    monkeypatch.setattr(_llama_server, "query_llama_server",
                        lambda *a, **kw: '{"action": "BUY", "confidence": 70}')
    r = compute_cryptotrader_lm_signal(pd.DataFrame({"close": [1.0]}), context=_ctx(ticker="BTC-USD"))
    assert r["action"] == "BUY"
    assert r["confidence"] == 0.70


def test_eth_ticker_allowed(monkeypatch):
    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    monkeypatch.setattr(_llama_server, "query_llama_server",
                        lambda *a, **kw: '{"action": "SELL", "confidence": 65}')
    r = compute_cryptotrader_lm_signal(pd.DataFrame({"close": [1.0]}), context=_ctx(ticker="ETH-USD"))
    assert r["action"] == "SELL"
    assert r["confidence"] == 0.65


# ---------------------------------------------------------------------------
# Abstention paths
# ---------------------------------------------------------------------------

def test_abstains_when_llama_server_returns_none(monkeypatch):
    monkeypatch.setattr(_llama_server, "query_llama_server", lambda *a, **kw: None)
    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    r = compute_cryptotrader_lm_signal(pd.DataFrame({"close": [1.0]}), context=_ctx())
    assert r["action"] == "HOLD"
    assert r["confidence"] == 0.0
    assert r["indicators"]["reason"] == "server_unavailable"


def test_abstains_when_plex_vram_tight(monkeypatch):
    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: False)
    called = {"n": 0}

    def _fake_query(*a, **kw):
        called["n"] += 1
        return "should not happen"

    monkeypatch.setattr(_llama_server, "query_llama_server", _fake_query)
    r = compute_cryptotrader_lm_signal(pd.DataFrame({"close": [1.0]}), context=_ctx())
    assert r["action"] == "HOLD"
    assert r["indicators"]["reason"] == "plex_vram_tight"
    assert called["n"] == 0


def test_abstains_when_query_raises(monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("LoRA load failed")

    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    monkeypatch.setattr(_llama_server, "query_llama_server", _boom)
    r = compute_cryptotrader_lm_signal(pd.DataFrame({"close": [1.0]}), context=_ctx())
    assert r["action"] == "HOLD"
    assert r["indicators"]["reason"] == "inference_error"
    assert "LoRA load failed" in r["indicators"]["error"]


def test_missing_context_returns_abstention():
    r = compute_cryptotrader_lm_signal(pd.DataFrame({"close": [1.0]}), context=None)
    assert r["action"] == "HOLD"
    assert r["indicators"]["reason"] == "missing_context"


# ---------------------------------------------------------------------------
# Parser interop (uses shared ministral_trader._parse_response)
# ---------------------------------------------------------------------------

def test_recovers_confidence_via_regex_when_json_malformed(monkeypatch):
    """Same failure mode as ministral: ``` ```json codefences + literal
    newlines in reasoning break json.loads. Regex fallback recovers
    confidence from raw text."""
    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    bad_output = (
        '```json\n{"action": "BUY", "confidence": 72, '
        '"reasoning": "MVRV oversold\nfunding flipped"}\n```'
    )
    monkeypatch.setattr(_llama_server, "query_llama_server", lambda *a, **kw: bad_output)
    r = compute_cryptotrader_lm_signal(pd.DataFrame({"close": [1.0]}), context=_ctx())
    assert r["action"] == "BUY"
    assert 0.71 <= r["confidence"] <= 0.73


def test_confidence_none_defaults_to_50(monkeypatch):
    """Parser returns conf=None → default to 0.50 (NOT 0.0). Same
    rationale as finance_llama: preserves BUY/SELL argmax under
    derive_probs_from_result."""
    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    monkeypatch.setattr(
        _llama_server,
        "query_llama_server",
        lambda *a, **kw: "I think we should SELL based on the indicators",
    )
    r = compute_cryptotrader_lm_signal(pd.DataFrame({"close": [1.0]}), context=_ctx())
    assert r["action"] == "SELL"
    assert r["confidence"] == 0.50


# ---------------------------------------------------------------------------
# Wiring contracts
# ---------------------------------------------------------------------------

def test_feature_available_is_true():
    """Sentinel — drops to False when scaffold mode is re-enabled. If
    this test starts failing, real inference has been disabled and a
    regression note should explain why."""
    from portfolio.signals.cryptotrader_lm import _FEATURE_AVAILABLE
    assert _FEATURE_AVAILABLE is True


def test_uses_ministral8_lora_model_name(monkeypatch):
    """Verify the signal queries the right llama_server key — the one
    that has the LoRA adapter loaded via --lora extra-arg."""
    captured = {}

    def _spy(name, prompt, stop=None, **kw):
        captured["name"] = name
        captured["stop"] = stop
        captured["prompt_head"] = prompt[:50]
        return '{"action": "HOLD", "confidence": 60}'

    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    monkeypatch.setattr(_llama_server, "query_llama_server", _spy)
    compute_cryptotrader_lm_signal(pd.DataFrame({"close": [1.0]}), context=_ctx())
    assert captured["name"] == "ministral8_lora", (
        "must use the _MODEL_CONFIGS key that has the LoRA adapter"
    )
    assert "[INST]" in captured["prompt_head"], (
        "Ministral-instruct base requires [INST]...[/INST] markers"
    )
    assert captured["stop"] == ["[INST]"]


def test_result_validates_under_signal_engine(monkeypatch):
    from portfolio.signal_engine import _validate_signal_result

    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    monkeypatch.setattr(
        _llama_server,
        "query_llama_server",
        lambda *a, **kw: '{"action": "BUY", "confidence": 65}',
    )
    r = compute_cryptotrader_lm_signal(pd.DataFrame({"close": [1.0]}), context=_ctx())
    v = _validate_signal_result(r, sig_name="cryptotrader_lm", max_confidence=0.7)
    assert v["action"] == "BUY"
    assert v["confidence"] <= 0.7


def test_indicators_includes_model_lineage(monkeypatch):
    """Calibration dashboards need to distinguish LoRA-on-Ministral
    output from raw Ministral output. Verify model string captures
    both base + adapter."""
    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    monkeypatch.setattr(
        _llama_server,
        "query_llama_server",
        lambda *a, **kw: '{"action": "HOLD", "confidence": 50}',
    )
    r = compute_cryptotrader_lm_signal(pd.DataFrame({"close": [1.0]}), context=_ctx())
    assert "PEFT LoRA" in r["indicators"]["model"]
    assert "Ministral-8B" in r["indicators"]["model"]
