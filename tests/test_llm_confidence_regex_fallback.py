"""Regression tests for the 2026-05-15 LLM confidence-recovery fix.

Both `ministral_trader._parse_response` and `qwen3_trader._parse_response`
now have a regex fallback for the `confidence` field that fires when
`_extract_json_payload` returns None (typically because the model
wrapped its JSON in a ```json codefence AND included raw newlines
inside the `reasoning` string value — both of which break json.loads).

Without the fallback, confidence stayed at None → signal_engine never
set `extra_info["{name}_confidence"]` → llm_probability_log treated
confidence as 0.0 and emitted the canonical conf-zero abstention shape
`{BUY: 0.25, HOLD: 0.5, SELL: 0.25}` for every vote. 9+ days of qwen3
+ ministral rows in `data/llm_probability_log.jsonl` are that exact
shape — Brier-score calibration was silently destroyed.

These tests pin the contract:

* Real production response shapes (codefence + raw newlines in reasoning)
  recover BOTH action and confidence.
* 0-100 confidence is renormalized to 0-1.
* 0-1 confidence is preserved.
* Pathological inputs (no JSON, no confidence field at all) return
  None confidence without raising.
* Whitespace-only confidence value is tolerated.
"""

from __future__ import annotations

import pytest

from portfolio.ministral_trader import _parse_response as parse_ministral
from portfolio.qwen3_trader import _parse_response as parse_qwen3


# Real production-shape responses captured 2026-05-15 from llama-server:
# codefence wrapper + raw newlines inside the reasoning string. This is
# the exact pathology that broke the system silently for 9+ days.
PROD_MINISTRAL = """```json
{
  "action": "BUY",
  "confidence": 75,
  "reasoning": "
1. **Trend**: EMA(9) > EMA(21) bullish crossover with bullish MACD.
2. **Volume**: 1.1x avg, supports momentum."
}
```"""

PROD_QWEN3 = """<think>
The market shows mixed signals but momentum favors buyers.
</think>
```json
{
  "action": "BUY",
  "confidence": 68,
  "reasoning": "
Multi-timeframe alignment with bullish technicals."
}
```"""


@pytest.mark.parametrize(
    "parser,text,expected_action,expected_conf",
    [
        (parse_ministral, PROD_MINISTRAL, "BUY", 0.75),
        (parse_qwen3, PROD_QWEN3, "BUY", 0.68),
    ],
)
def test_production_shape_recovers_confidence(parser, text, expected_action, expected_conf):
    """The exact production-shape response must recover BOTH action and
    confidence even though json.loads on the brace substring fails."""
    action, _reasoning, confidence = parser(text)
    assert action == expected_action
    assert confidence is not None, "confidence regex fallback did not fire"
    assert abs(confidence - expected_conf) < 1e-6


@pytest.mark.parametrize("parser", [parse_ministral, parse_qwen3])
def test_zero_to_one_confidence_preserved(parser):
    """Some prompt revisions ask for 0-1 fractions instead of 0-100.
    The regex fallback must NOT scale 0.7 to 0.007."""
    text = '{"action": "BUY", "confidence": 0.7}'
    _action, _reasoning, conf = parser(text)
    assert conf == 0.7


@pytest.mark.parametrize("parser", [parse_ministral, parse_qwen3])
def test_zero_to_100_confidence_renormalized(parser):
    """0-100 prompt convention is renormalized to 0-1."""
    text = '{"action": "SELL", "confidence": 82}'
    _action, _reasoning, conf = parser(text)
    assert abs(conf - 0.82) < 1e-6


@pytest.mark.parametrize("parser", [parse_ministral, parse_qwen3])
def test_no_confidence_field_returns_none(parser):
    """Pathological response with no confidence anywhere must return
    None — NOT 0.0, and NOT 1.0. Caller decides how to handle missing."""
    text = "I think we should BUY but I am not sure."
    _action, _reasoning, conf = parser(text)
    assert conf is None


@pytest.mark.parametrize("parser", [parse_ministral, parse_qwen3])
def test_empty_text_returns_none(parser):
    _action, _reasoning, conf = parser("")
    assert conf is None


@pytest.mark.parametrize("parser", [parse_ministral, parse_qwen3])
def test_bare_unquoted_confidence_key_recovered(parser):
    """Some Qwen3 thinking-mode outputs drop the outer quotes around
    field names. Defensive: still recover confidence."""
    text = "Result: action: BUY, confidence: 65, reasoning: bullish"
    _action, _reasoning, conf = parser(text)
    assert conf is not None
    assert abs(conf - 0.65) < 1e-6


@pytest.mark.parametrize("parser", [parse_ministral, parse_qwen3])
def test_clamps_to_zero_one_range(parser):
    """Model occasionally hallucinates out-of-range values. Clamp to
    [0, 1] so downstream log_vote() doesn't reject the row."""
    text = '{"action": "BUY", "confidence": 999}'
    _action, _reasoning, conf = parser(text)
    assert conf == 1.0
    text = '{"action": "BUY", "confidence": -5}'
    _action, _reasoning, conf = parser(text)
    assert conf == 0.0


@pytest.mark.parametrize("parser", [parse_ministral, parse_qwen3])
def test_action_still_recovered_when_confidence_recovers(parser):
    """Action regex was already recovered before this fix. Verify the
    new confidence fallback doesn't interfere with action recovery."""
    text = "Final answer: BUY. confidence: 72"
    action, _reasoning, conf = parser(text)
    assert action == "BUY"
    assert abs(conf - 0.72) < 1e-6
