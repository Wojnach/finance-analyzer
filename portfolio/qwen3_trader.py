#!/usr/bin/env python3
"""Qwen3-8B trading signal via llama-cpp-python (GGUF).

Runs as a subprocess: reads JSON context from stdin, outputs JSON prediction.
Uses GPU lock to coordinate with Ministral (only one model on GPU at a time).
Supports ALL tickers (crypto, stocks, metals) — not crypto-only like Ministral.
"""

import argparse
import json
import os
import platform
import re
import sys

if platform.system() == "Windows":
    MODEL_PATH = r"Q:\models\qwen3-8b-gguf\Qwen3-8B-Q4_K_M.gguf"
else:
    MODEL_PATH = "/home/deck/models/qwen3-8b-gguf/Qwen3-8B-Q4_K_M.gguf"


def load_model():
    from llama_cpp import Llama

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Qwen3-8B model not found at {MODEL_PATH}")

    return Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
        chat_format="chatml",
    )


def _extract_json_payload(text):
    """Extract a JSON payload from model output when possible."""
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(stripped[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


def predict(context):
    model = load_model()

    # Build asset-type-aware prompt
    ticker = context.get("ticker", "UNKNOWN")
    asset_type = context.get("asset_type", "cryptocurrency")

    prompt = f"""<|im_start|>system
You are an expert financial analyst specializing in {asset_type} trading.
Analyze the market data and provide a single trading decision: BUY, SELL, or HOLD.
Respond with EXACTLY one JSON object: {{"action":"BUY|SELL|HOLD","reasoning":"one sentence"}}
Use HOLD when evidence is mixed or weak.<|im_end|>
<|im_start|>user
Asset: {ticker} ({asset_type})
Current Price: ${context.get('price_usd', 0):,.2f}

Technical Indicators:
- RSI(14): {context.get('rsi', 'N/A')}
- MACD Histogram: {context.get('macd_hist', 'N/A')}
- EMA(9) vs EMA(21): {'Bullish (9 > 21)' if context.get('ema_bullish') else 'Bearish (9 < 21)'} (gap: {context.get('ema_gap_pct', 'N/A')}%)
- Bollinger Bands: Price is {context.get('bb_position', 'N/A')}
- Volume Ratio: {context.get('volume_ratio', 'N/A')}x avg

Market Context:
- Fear & Greed: {context.get('fear_greed', 'N/A')}/100 ({context.get('fear_greed_class', '')})
- Sentiment: {context.get('news_sentiment', 'N/A')} (conf: {context.get('sentiment_confidence', 'N/A')})

Multi-timeframe: {context.get('timeframe_summary', 'N/A')}

Provide your trading decision as JSON. /no_think<|im_end|>
<|im_start|>assistant
"""

    response = model(
        prompt,
        max_tokens=200,
        temperature=0.1,
        top_p=0.3,
        stop=["<|im_end|>", "<|im_start|>"],
    )

    text = response["choices"][0]["text"].strip()

    # Strip thinking tags if present (Qwen3 thinking mode)
    if "<think>" in text:
        think_end = text.find("</think>")
        if think_end >= 0:
            text = text[think_end + 8:].strip()

    payload = _extract_json_payload(text)

    decision = None
    reasoning = text[:200]
    if isinstance(payload, dict):
        raw_action = str(payload.get("action", "")).upper()
        if raw_action in {"BUY", "SELL", "HOLD"}:
            decision = raw_action
        if payload.get("reasoning"):
            reasoning = str(payload["reasoning"])[:200]

    if decision is None:
        match = re.search(r"\b(BUY|SELL|HOLD)\b", text.upper())
        decision = match.group(1) if match else "HOLD"

    return {
        "action": decision,
        "reasoning": reasoning,
        "model": "Qwen3-8B",
    }


if __name__ == "__main__":
    context = json.loads(sys.stdin.read())
    result = predict(context)
    print(json.dumps(result))
