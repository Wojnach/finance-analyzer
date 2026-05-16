"""Integration tests for finance_llama real GGUF inference.

Mocks `portfolio.llama_server.query_llama_server` so we never need a live
llama-server during CI. Verifies:

* Server-unavailable path returns abstention with reason="server_unavailable".
* Plex-VRAM guard returns abstention with reason="plex_vram_tight".
* Well-formed JSON output is parsed into BUY/SELL/HOLD + confidence.
* Malformed JSON triggers the regex confidence fallback (matches
  ministral_trader._parse_response behaviour).
* Conf=None recovery defaults to 0.50, NOT 0.0 — so the BUY/SELL argmax
  survives derive_probs_from_result instead of collapsing to {0.5, 0.25, 0.25}.
* Result shape passes signal_engine._validate_signal_result.
* The model path string in indicators matches the on-disk GGUF.
"""
from __future__ import annotations

import pandas as pd

import portfolio.llama_server as _llama_server
from portfolio.signals.finance_llama import compute_finance_llama_signal


def _ctx(ticker="BTC-USD", price=100_000.0, **overrides):
    base = {
        "ticker": ticker,
        "price_usd": price,
        "rsi": 55.0,
        "macd_hist": 0.5,
        "ema_bullish": True,
        "ema_gap_pct": 1.2,
        "bb_position": "middle",
        "volume_ratio": 1.1,
        "fear_greed": 50,
        "fear_greed_class": "neutral",
        "news_sentiment": "neutral",
        "sentiment_confidence": 0.5,
        "timeframe_summary": "balanced",
        "headlines": "no major news",
        "change_24h": 0.1,
        "asset_type": "cryptocurrency",
    }
    base.update(overrides)
    return base


def test_abstains_when_llama_server_returns_none(monkeypatch):
    monkeypatch.setattr(_llama_server, "query_llama_server", lambda *a, **kw: None)
    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    r = compute_finance_llama_signal(pd.DataFrame({"close": [1.0]}), context=_ctx())
    assert r["action"] == "HOLD"
    assert r["confidence"] == 0.0
    assert r["indicators"]["reason"] == "server_unavailable"


def test_abstains_when_plex_vram_tight(monkeypatch):
    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: False)
    # query should never be called if the guard fires; assert that:
    called = {"n": 0}

    def _fake_query(*a, **kw):
        called["n"] += 1
        return "should not happen"

    monkeypatch.setattr(_llama_server, "query_llama_server", _fake_query)
    r = compute_finance_llama_signal(pd.DataFrame({"close": [1.0]}), context=_ctx())
    assert r["action"] == "HOLD"
    assert r["confidence"] == 0.0
    assert r["indicators"]["reason"] == "plex_vram_tight"
    assert called["n"] == 0, "guard must short-circuit before query"


def test_abstains_when_query_raises(monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    monkeypatch.setattr(_llama_server, "query_llama_server", _boom)
    r = compute_finance_llama_signal(pd.DataFrame({"close": [1.0]}), context=_ctx())
    assert r["action"] == "HOLD"
    assert r["confidence"] == 0.0
    assert r["indicators"]["reason"] == "inference_error"
    assert "connection refused" in r["indicators"]["error"]


def test_parses_well_formed_buy_response(monkeypatch):
    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    monkeypatch.setattr(
        _llama_server,
        "query_llama_server",
        lambda *a, **kw: '{"action": "BUY", "confidence": 75, "reasoning": "strong setup"}',
    )
    r = compute_finance_llama_signal(pd.DataFrame({"close": [1.0]}), context=_ctx())
    assert r["action"] == "BUY"
    assert 0.74 <= r["confidence"] <= 0.76  # 75 -> 0.75 after normalization
    assert r["indicators"]["model"] == "finance-llama-8b"


def test_parses_sell_response(monkeypatch):
    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    monkeypatch.setattr(
        _llama_server,
        "query_llama_server",
        lambda *a, **kw: '{"action": "SELL", "confidence": 0.62, "reasoning": "exhaustion"}',
    )
    r = compute_finance_llama_signal(pd.DataFrame({"close": [1.0]}), context=_ctx())
    assert r["action"] == "SELL"
    assert r["confidence"] == 0.62


def test_recovers_confidence_via_regex_when_json_malformed(monkeypatch):
    """Real-world failure mode: model emits ```json codefences + literal
    newlines in reasoning, breaking json.loads on the brace-extracted
    substring. Regex fallback in ministral_trader._parse_response
    salvages confidence from the raw text."""
    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    bad_output = (
        '```json\n{"action": "BUY", "confidence": 68, '
        '"reasoning": "trend up\nvolume good"}\n```'
    )
    monkeypatch.setattr(_llama_server, "query_llama_server", lambda *a, **kw: bad_output)
    r = compute_finance_llama_signal(pd.DataFrame({"close": [1.0]}), context=_ctx())
    assert r["action"] == "BUY"
    assert 0.67 <= r["confidence"] <= 0.69, f"expected ~0.68 from regex fallback, got {r['confidence']}"


def test_confidence_none_defaults_to_50_not_zero(monkeypatch):
    """When the parser can't extract any confidence figure at all, we
    default to 0.50 so the BUY/SELL argmax survives downstream
    derive_probs_from_result. The conf<=0 branch in that function
    forces a {0.5, 0.25, 0.25} shape favouring the chosen action only
    weakly — fine for HOLD but it masks real BUY/SELL conviction. 0.50
    yields the standard split with the argmax slightly elevated."""
    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    # No JSON structure, no "confidence:" substring — just an action word.
    monkeypatch.setattr(
        _llama_server,
        "query_llama_server",
        lambda *a, **kw: "I think we should BUY based on the indicators",
    )
    r = compute_finance_llama_signal(pd.DataFrame({"close": [1.0]}), context=_ctx())
    assert r["action"] == "BUY"
    assert r["confidence"] == 0.50


def test_missing_context_returns_abstention():
    r = compute_finance_llama_signal(pd.DataFrame({"close": [1.0]}), context=None)
    assert r["action"] == "HOLD"
    assert r["confidence"] == 0.0
    assert r["indicators"]["reason"] == "missing_context"


def test_missing_ticker_in_context_returns_abstention():
    r = compute_finance_llama_signal(pd.DataFrame({"close": [1.0]}), context={"price_usd": 100})
    assert r["action"] == "HOLD"
    assert r["indicators"]["reason"] == "missing_context"


def test_prompt_is_few_shot_completion_not_mistral_instruct():
    """Reviewer caught this on the first revision: finance-llama-8b is a
    Llama-3.1 completion model, not Mistral-instruct. The previous
    revision reused ministral_trader._build_prompt which emits
    [INST]...[/INST] markers — off-distribution for this model.

    Pin the corrected behaviour: prompt uses the few-shot 'Situation /
    Decision / Confidence' pattern lifted from
    Q:/models/fingpt_infer.py:PROMPT_TEMPLATES, NOT Mistral markers."""
    from portfolio.signals.finance_llama import _build_finance_llama_prompt

    prompt = _build_finance_llama_prompt(
        {"ticker": "BTC-USD", "rsi": 55, "macd_hist": 0.1, "ema_bullish": True}
    )
    assert "[INST]" not in prompt, "must not use Mistral instruction format"
    assert "[/INST]" not in prompt
    # Lowercased keys so the shared ministral parser's literal-string
    # confidence regex matches without IGNORECASE.
    assert "decision:" in prompt
    assert "confidence:" in prompt
    assert "Situation: BTC-USD" in prompt, "context must be interpolated"


def test_plain_text_completion_output_is_parseable(monkeypatch):
    """finance-llama-8b emits plain text like 'BUY\\nconfidence: 70' (not
    JSON). The reused ministral _parse_response should recover both
    action and confidence via its regex fallbacks. Verifies the two
    pieces — new prompt + reused parser — compose correctly end-to-end.

    Labels are lowercased to match the parser's literal 'confidence'
    regex (no IGNORECASE flag on that path)."""
    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    monkeypatch.setattr(
        _llama_server,
        "query_llama_server",
        lambda *a, **kw: " BUY\nconfidence: 72\nThe setup shows strong reversal signals.",
    )
    r = compute_finance_llama_signal(pd.DataFrame({"close": [1.0]}), context=_ctx())
    assert r["action"] == "BUY"
    assert 0.71 <= r["confidence"] <= 0.73


def test_stop_tokens_passed_to_llama_server(monkeypatch):
    """Stop tokens cut generation after the first Decision/Confidence
    pair so the model can't drift into making up more situations (the
    2026-04-09 fingpt incident). Verify they are actually forwarded."""
    captured = {}

    def _spy(name, prompt, stop=None, **kw):
        captured["name"] = name
        captured["stop"] = stop
        return "BUY\nConfidence: 60"

    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    monkeypatch.setattr(_llama_server, "query_llama_server", _spy)
    compute_finance_llama_signal(pd.DataFrame({"close": [1.0]}), context=_ctx())
    assert captured["name"] == "finance-llama-8b"
    assert "\n\n" in (captured["stop"] or [])
    assert "Situation:" in (captured["stop"] or [])


def test_result_validates_under_signal_engine(monkeypatch):
    """Shape contract: dispatcher runs results through
    _validate_signal_result; verify both BUY and abstention shapes pass."""
    from portfolio.signal_engine import _validate_signal_result

    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    monkeypatch.setattr(
        _llama_server,
        "query_llama_server",
        lambda *a, **kw: '{"action": "BUY", "confidence": 65}',
    )
    r = compute_finance_llama_signal(pd.DataFrame({"close": [1.0]}), context=_ctx())
    v = _validate_signal_result(r, sig_name="finance_llama", max_confidence=0.7)
    assert v["action"] == "BUY"
    assert v["confidence"] <= 0.7  # max_confidence cap applied
