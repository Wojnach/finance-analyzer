"""Tests for portfolio.signals.phi4_mini_reasoning.

Covers:
* Abstain shape when feature unavailable / server unreachable.
* Phi-4 <think> block stripping + decision/confidence extraction.
* Confidence fallback to 0.50 when no numeric confidence found.
* Prompt contains Phi-4 chat tokens (not Mistral [INST] or Qwen3 im_start).
* Result shape passes signal_engine._validate_signal_result.
* Stop tokens and slot name forwarded to query_llama_server.

xdist-safe: no module-level file-system paths; all file access is via
monkeypatched interfaces. Tests are pure unit — no live GPU, no llama-server.
"""
from __future__ import annotations

import pandas as pd
import pytest

import portfolio.llama_server as _llama_server
from portfolio.signals.phi4_mini_reasoning import (
    _build_phi4_prompt,
    _parse_phi4_response,
    _strip_think_block,
    compute_phi4_mini_signal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _df():
    return pd.DataFrame({"close": [1.0, 2.0, 3.0]})


# ---------------------------------------------------------------------------
# (a) Abstain paths
# ---------------------------------------------------------------------------

def test_abstains_when_server_returns_none(monkeypatch):
    monkeypatch.setattr(_llama_server, "query_llama_server", lambda *a, **kw: None)
    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    r = compute_phi4_mini_signal(_df(), context=_ctx())
    assert r["action"] == "HOLD"
    assert r["confidence"] == 0.0
    assert r["indicators"]["reason"] == "server_unavailable"


def test_abstains_when_plex_vram_tight(monkeypatch):
    """Plex-VRAM guard must short-circuit before any query is attempted."""
    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: False)
    called = {"n": 0}

    def _fake_query(*a, **kw):
        called["n"] += 1
        return "should not happen"

    monkeypatch.setattr(_llama_server, "query_llama_server", _fake_query)
    r = compute_phi4_mini_signal(_df(), context=_ctx())
    assert r["action"] == "HOLD"
    assert r["confidence"] == 0.0
    assert r["indicators"]["reason"] == "plex_vram_tight"
    assert called["n"] == 0, "guard must short-circuit before query"


def test_abstains_when_context_none(monkeypatch):
    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    monkeypatch.setattr(_llama_server, "query_llama_server", lambda *a, **kw: "BUY")
    r = compute_phi4_mini_signal(_df(), context=None)
    assert r["action"] == "HOLD"
    assert r["indicators"]["reason"] == "missing_context"


def test_abstains_when_ticker_missing(monkeypatch):
    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    monkeypatch.setattr(_llama_server, "query_llama_server", lambda *a, **kw: "BUY")
    r = compute_phi4_mini_signal(_df(), context={"price_usd": 100})
    assert r["action"] == "HOLD"
    assert r["indicators"]["reason"] == "missing_context"


def test_abstains_when_query_raises(monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    monkeypatch.setattr(_llama_server, "query_llama_server", _boom)
    r = compute_phi4_mini_signal(_df(), context=_ctx())
    assert r["action"] == "HOLD"
    assert r["confidence"] == 0.0
    assert r["indicators"]["reason"] == "inference_error"
    assert "connection refused" in r["indicators"]["error"]


# ---------------------------------------------------------------------------
# (b) Phi-4 <think> block + decision line parsing
# ---------------------------------------------------------------------------

def test_strip_think_block_removes_think_tags():
    raw = "<think>Step 1: RSI is 55, neutral.\nStep 2: EMA bullish.</think>\ndecision: BUY\nconfidence: 72"
    stripped = _strip_think_block(raw)
    assert "<think>" not in stripped
    assert "</think>" not in stripped
    assert "decision: BUY" in stripped


def test_strip_think_block_passthrough_when_no_tags():
    raw = "decision: SELL\nconfidence: 68"
    assert _strip_think_block(raw) == raw


def test_parse_phi4_response_with_think_block():
    """Real Phi-4 output format: <think>…</think> then structured decision lines."""
    raw = (
        "<think>\n"
        "RSI is 22 — deeply oversold. MACD histogram turning positive. "
        "Volume 1.8x confirms buyers stepping in.\n"
        "The setup looks like a capitulation bottom.\n"
        "</think>\n"
        "decision: BUY\n"
        "confidence: 74"
    )
    action, reasoning, confidence = _parse_phi4_response(raw)
    assert action == "BUY"
    assert confidence is not None
    assert 0.73 <= confidence <= 0.75, f"expected ~0.74, got {confidence}"


def test_parse_phi4_response_sell_with_think_block():
    raw = (
        "<think>RSI=82 overbought, momentum fading, volume drying up.</think>\n"
        "decision: SELL\nconfidence: 68"
    )
    action, _, confidence = _parse_phi4_response(raw)
    assert action == "SELL"
    assert 0.67 <= confidence <= 0.69


def test_parse_phi4_response_hold():
    raw = "<think>Mixed signals.</think>\ndecision: HOLD\nconfidence: 55"
    action, _, confidence = _parse_phi4_response(raw)
    assert action == "HOLD"
    assert 0.54 <= confidence <= 0.56


def test_parse_phi4_response_confidence_already_normalised():
    """Model may emit confidence as 0-1 float; must not double-divide."""
    raw = "decision: BUY\nconfidence: 0.72"
    action, _, confidence = _parse_phi4_response(raw)
    assert action == "BUY"
    assert 0.71 <= confidence <= 0.73


def test_parse_phi4_response_no_confidence_returns_none():
    """When no confidence line present, parser returns None — caller defaults to 0.50."""
    raw = "<think>Hard call.</think>\nI recommend BUY based on the signals."
    action, _, confidence = _parse_phi4_response(raw)
    assert action == "BUY"
    assert confidence is None


def test_compute_signal_confidence_none_defaults_to_50(monkeypatch):
    """conf=None from parser → signal returns 0.50, not 0.0 (preserves argmax)."""
    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    # No "confidence:" line — just a bare action word after the think block.
    monkeypatch.setattr(
        _llama_server,
        "query_llama_server",
        lambda *a, **kw: "<think>Step 1.</think>\nThe answer is SELL.",
    )
    r = compute_phi4_mini_signal(_df(), context=_ctx())
    assert r["action"] == "SELL"
    assert r["confidence"] == 0.50


# ---------------------------------------------------------------------------
# (c) Full compute path — well-formed response
# ---------------------------------------------------------------------------

def test_parses_buy_response_end_to_end(monkeypatch):
    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    monkeypatch.setattr(
        _llama_server,
        "query_llama_server",
        lambda *a, **kw: (
            "<think>RSI=22 oversold, volume spike, EMA bullish cross.</think>\n"
            "decision: BUY\nconfidence: 76"
        ),
    )
    r = compute_phi4_mini_signal(_df(), context=_ctx())
    assert r["action"] == "BUY"
    assert 0.75 <= r["confidence"] <= 0.77
    assert r["indicators"]["model"] == "phi4-mini-reasoning"


def test_result_shape_passes_validate_signal_result(monkeypatch):
    """signal_engine._validate_signal_result must not raise or strip the action."""
    from portfolio.signal_engine import _validate_signal_result

    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    monkeypatch.setattr(
        _llama_server,
        "query_llama_server",
        lambda *a, **kw: "decision: BUY\nconfidence: 65",
    )
    r = compute_phi4_mini_signal(_df(), context=_ctx())
    v = _validate_signal_result(r, sig_name="phi4_mini", max_confidence=0.7)
    assert v["action"] == "BUY"
    assert v["confidence"] <= 0.7  # max_confidence cap applied


# ---------------------------------------------------------------------------
# (d) Prompt format — Phi-4 tokens, not Mistral or Qwen3
# ---------------------------------------------------------------------------

def test_prompt_uses_phi4_chat_tokens_not_mistral():
    prompt = _build_phi4_prompt(_ctx())
    assert "[INST]" not in prompt, "must not use Mistral instruction format"
    assert "[/INST]" not in prompt
    assert "<|im_start|>" not in prompt, "must not use Qwen3/ChatML format"
    assert "<|system|>" in prompt
    assert "<|user|>" in prompt
    assert "<|assistant|>" in prompt
    assert "<|end|>" in prompt


def test_prompt_contains_ticker_and_instructions(monkeypatch):
    prompt = _build_phi4_prompt(_ctx(ticker="XAU-USD"))
    assert "XAU-USD" in prompt
    assert "decision: BUY|SELL|HOLD" in prompt
    assert "confidence:" in prompt


# ---------------------------------------------------------------------------
# (e) Slot name + stop tokens forwarded correctly
# ---------------------------------------------------------------------------

def test_slot_name_and_stop_tokens_forwarded(monkeypatch):
    captured = {}

    def _spy(name, prompt, stop=None, n_predict=None, **kw):
        captured["name"] = name
        captured["stop"] = stop
        captured["n_predict"] = n_predict
        return "decision: HOLD\nconfidence: 55"

    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    monkeypatch.setattr(_llama_server, "query_llama_server", _spy)
    compute_phi4_mini_signal(_df(), context=_ctx())

    assert captured["name"] == "phi4_mini", "must use phi4_mini slot, not another model"
    assert "<|end|>" in (captured["stop"] or []), "Phi-4 turn terminator must be a stop token"
    assert captured["n_predict"] is not None and captured["n_predict"] >= 256, (
        "reasoning models need token headroom for <think> block"
    )


# ---------------------------------------------------------------------------
# (c) Regression: truncated <think> must abstain, not fabricate a vote
#     Found by the 2026-06-01 live probe — n_predict=512 truncated mid-think
#     and the bare-word regex grabbed a spurious "BUY" from the reasoning prose.
# ---------------------------------------------------------------------------

def test_strip_think_block_unclosed_returns_empty():
    """Open <think> with no </think> (truncated generation) -> empty string."""
    raw = "<think>\nOkay, let's figure out whether to BUY, SELL, or HOLD. First,"
    assert _strip_think_block(raw) == ""


def test_parse_truncated_think_yields_no_action():
    """Truncated think must not regex-grab BUY/SELL from the reasoning text."""
    raw = "<think>\nShould I BUY here? RSI oversold suggests BUY but momentum"
    action, _reason, confidence = _parse_phi4_response(raw)
    assert action is None
    assert confidence is None


def test_compute_abstains_on_truncated_think(monkeypatch):
    """End-to-end: truncated-think response -> clean HOLD/conf=0 abstain."""
    truncated = "<think>\nLet me reason: BUY signals are RSI=28 oversold, but"
    monkeypatch.setattr(_llama_server, "model_load_safe", lambda *a, **kw: True)
    monkeypatch.setattr(_llama_server, "query_llama_server", lambda *a, **kw: truncated)
    r = compute_phi4_mini_signal(_df(), context=_ctx())
    assert r["action"] == "HOLD"
    assert r["confidence"] == 0.0
    assert r["indicators"]["reason"] == "think_truncated"


def test_parse_prose_confidence_from_conclusion():
    """LIVE-PROBE 2: model writes prose ('confidence score is **87**'), not
    the structured 'confidence: N'. Parser must still recover action+conf."""
    raw = ("<think>\nlong reasoning here about RSI and MACD...\n</think>\n"
           "**Conclusion:** strong oversold confluence points to a **BUY** "
           "signal. The confidence score is **87** due to alignment.")
    action, _reason, confidence = _parse_phi4_response(raw)
    assert action == "BUY"
    assert confidence == pytest.approx(0.87)


# ---------------------------------------------------------------------------
# (d) FGL-review regression: prose-confidence fallback must NOT scrape a
#     year / price / RSI value as a fabricated confidence on a real vote.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "<think>x</think>\ndecision: BUY\nGiven the 2026 outlook I have confidence, the year 2026 matters.",
    "<think>x</think>\ndecision: SELL\nI have confidence the price 98000 holds.",
    "<think>x</think>\ndecision: BUY\nconfidence in the RSI 72 reading is moderate.",
])
def test_prose_confidence_rejects_non_confidence_numbers(text):
    """Year/price/RSI near 'confidence' must NOT become the confidence."""
    action, _r, confidence = _parse_phi4_response(text)
    assert action in ("BUY", "SELL")
    # No structured/anchored confidence -> None (caller defaults 0.50), NOT a
    # fabricated 1.0 (2026/98000 clamped) or 0.72 (RSI).
    assert confidence is None


def test_prose_confidence_accepts_anchored_value():
    """Genuine anchored prose confidence is still recovered."""
    raw = "<think>x</think>\nThe **BUY** call. Confidence score is **87** here."
    action, _r, confidence = _parse_phi4_response(raw)
    assert action == "BUY"
    assert confidence == pytest.approx(0.87)
