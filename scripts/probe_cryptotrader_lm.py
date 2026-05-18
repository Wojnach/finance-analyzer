"""Validation probe for the cryptotrader_lm LoRA GGUF.

Why this exists
---------------
The v1 GGUF (`cryptotrader-lm-lora.gguf`, dated 2026-02-11) emitted
empty completions on every production prompt — it loaded cleanly and
served `completion_tokens=1, text=""` on the production
`portfolio.ministral_trader._build_prompt` template. The
auto-promotion gate (`scripts/review_shadow_signals.py`) was fed 992
HOLD/conf=0 abstain rows from that path and would have flipped
`cryptotrader_lm` into the production vote pool based on
HOLD-bias-against-outcome-backfill matching at ~64%.

We regenerated the GGUF on 2026-05-18 from
``Q:\\models\\cryptotrader-lm\\adapter_model.safetensors`` via the current
``Q:\\models\\llama.cpp\\convert_lora_to_gguf.py`` against
``Q:\\models\\ministral-8b-hf\\`` as ``--base``. v1 was archived as
``cryptotrader-lm-lora-v1-broken.gguf``.

This probe is the contract test that catches future regressions:
re-run after any swap of the GGUF or any change to the Ministral base
quantization / llama.cpp build. Premortem N3 (PLAN.md@e0f13449):
``completion_tokens > 5`` is not enough -- the GGUF must produce parseable
output on a real production prompt. The probe asserts:

* mean completion tokens >= 20 over N sampled prompts
* ``parse_rate >= 0.9`` -- output passes
  ``portfolio.ministral_trader._parse_response`` returning a non-None
  action
* at least one prompt produces a non-HOLD decision (model is not
  collapsing to safe default)

Exit codes:
    0  — probe PASS
    1  — probe FAIL with reasons printed

Usage
    .venv/Scripts/python.exe scripts/probe_cryptotrader_lm.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from portfolio.llama_server import query_llama_server  # noqa: E402
from portfolio.ministral_trader import _build_prompt, _parse_response  # noqa: E402

# Five diverse production-shaped contexts. The model was trained on
# crypto news + price data so all five tickers are BTC/ETH.
_PROBE_CONTEXTS = [
    {
        "ticker": "BTC-USD", "price_usd": 95000.0, "rsi": 55.0,
        "macd_hist": 10.0, "ema_bullish": True, "ema_gap_pct": 1.2,
        "bb_position": 0.7, "volume_ratio": 1.1, "fear_greed": 65,
        "fear_greed_class": "Greed", "news_sentiment": 0.2,
        "sentiment_confidence": 0.6,
        "timeframe_summary": "1h bullish, 4h bullish",
        "headlines": ["BTC ETF inflows continue"], "change_24h": 1.5,
    },
    {
        "ticker": "BTC-USD", "price_usd": 95000.0, "rsi": 78.0,
        "macd_hist": -5.0, "ema_bullish": False, "ema_gap_pct": -0.5,
        "bb_position": 0.95, "volume_ratio": 1.8, "fear_greed": 82,
        "fear_greed_class": "Extreme Greed", "news_sentiment": -0.3,
        "sentiment_confidence": 0.7,
        "timeframe_summary": "1h overbought, 4h ranging",
        "headlines": ["BTC profit-taking after ATH"], "change_24h": -2.1,
    },
    {
        "ticker": "ETH-USD", "price_usd": 3500.0, "rsi": 32.0,
        "macd_hist": -2.5, "ema_bullish": False, "ema_gap_pct": -1.8,
        "bb_position": 0.15, "volume_ratio": 0.9, "fear_greed": 25,
        "fear_greed_class": "Fear", "news_sentiment": -0.1,
        "sentiment_confidence": 0.5,
        "timeframe_summary": "1h oversold, 4h downtrend",
        "headlines": ["ETH staking yields dip"], "change_24h": -3.8,
    },
    {
        "ticker": "ETH-USD", "price_usd": 3500.0, "rsi": 50.0,
        "macd_hist": 0.5, "ema_bullish": True, "ema_gap_pct": 0.2,
        "bb_position": 0.5, "volume_ratio": 1.0, "fear_greed": 50,
        "fear_greed_class": "Neutral", "news_sentiment": 0.0,
        "sentiment_confidence": 0.4,
        "timeframe_summary": "1h ranging, 4h ranging",
        "headlines": [], "change_24h": 0.2,
    },
    {
        "ticker": "BTC-USD", "price_usd": 110000.0, "rsi": 62.0,
        "macd_hist": 25.0, "ema_bullish": True, "ema_gap_pct": 3.5,
        "bb_position": 0.85, "volume_ratio": 1.6, "fear_greed": 72,
        "fear_greed_class": "Greed", "news_sentiment": 0.5,
        "sentiment_confidence": 0.8,
        "timeframe_summary": "1h bullish, 4h bullish, 1d bullish",
        "headlines": ["BTC breaks 110K"], "change_24h": 4.2,
    },
]


def main() -> int:
    results = []
    print(f"Probing cryptotrader_lm LoRA with {len(_PROBE_CONTEXTS)} contexts...")
    for i, ctx in enumerate(_PROBE_CONTEXTS):
        prompt = _build_prompt(ctx)
        t0 = time.time()
        text = query_llama_server("ministral8_lora", prompt, stop=["[INST]"])
        wall = time.time() - t0
        decision, reasoning, confidence = _parse_response(text or "")
        n_tokens = len((text or "").split())
        parsed_ok = decision in ("BUY", "HOLD", "SELL")
        results.append({
            "i": i, "ticker": ctx["ticker"], "wall_s": wall,
            "n_tokens": n_tokens, "decision": decision,
            "confidence": confidence, "parsed_ok": parsed_ok,
        })
        print(
            f"  [{i}] {ctx['ticker']:8s} wall={wall:5.1f}s tokens={n_tokens:4d} "
            f"decision={decision!s:5s} conf={confidence!s:5s} ok={parsed_ok}"
        )

    n = len(results)
    parse_rate = sum(1 for r in results if r["parsed_ok"]) / n
    mean_tokens = sum(r["n_tokens"] for r in results) / n
    non_hold_count = sum(1 for r in results if r["decision"] in ("BUY", "SELL"))

    print()
    print(f"parse_rate     = {parse_rate:.3f} (>= 0.9 required)")
    print(f"mean_tokens    = {mean_tokens:.1f} (>= 20 required)")
    print(f"non_hold_count = {non_hold_count} (>= 1 required)")

    failures = []
    if parse_rate < 0.9:
        failures.append(f"parse_rate {parse_rate:.3f} < 0.9")
    if mean_tokens < 20:
        failures.append(f"mean_tokens {mean_tokens:.1f} < 20")
    if non_hold_count < 1:
        failures.append("zero non-HOLD decisions across all probes — model may be collapsing")

    if failures:
        print()
        print("PROBE FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print()
    print("PROBE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
