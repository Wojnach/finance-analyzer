#!/usr/bin/env python3
"""Ministral-8B trading signal via llama-cpp-python (GGUF).

Runs as a subprocess: reads JSON context from stdin, outputs JSON prediction.
Uses a separate venv (Q:/models/.venv-llm on Windows, ~/.venv-llm on Linux).
"""

import argparse
import json
import platform
import re
import sys

if platform.system() == "Windows":
    MODEL_PATH = r"Q:\models\ministral-8b-gguf\Ministral-8B-Instruct-2410-Q4_K_M.gguf"
    DEFAULT_LORA = r"Q:\models\cryptotrader-lm\cryptotrader-lm-lora.gguf"
else:
    MODEL_PATH = (
        "/home/deck/models/ministral-8b-gguf/Ministral-8B-Instruct-2410-Q4_K_M.gguf"
    )
    DEFAULT_LORA = "/home/deck/models/cryptotrader-lm/cryptotrader-lm-lora.gguf"


def load_model(lora_path=None):
    from llama_cpp import Llama

    return Llama(
        model_path=MODEL_PATH,
        lora_path=lora_path or DEFAULT_LORA,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
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


def predict(context, lora_path=None):
    model = load_model(lora_path)

    prompt = f"""[INST]You are an expert cryptocurrency trader. Based on the following market data, provide a single trading decision: BUY, SELL, or HOLD.

Market Data:
- Asset: {context['ticker']}
- Current Price: ${context['price_usd']:,.2f}
- 24h Change: {context.get('change_24h', 'N/A')}

Technical Indicators (1-hour candles):
- RSI(14): {context.get('rsi', 'N/A')}
- MACD Histogram: {context.get('macd_hist', 'N/A')}
- EMA(9) vs EMA(21): {'Bullish (9 > 21)' if context.get('ema_bullish') else 'Bearish (9 < 21)'} (gap: {context.get('ema_gap_pct', 'N/A')}%)
- Bollinger Bands: Price is {context.get('bb_position', 'N/A')}

Market Sentiment:
- Fear & Greed Index: {context.get('fear_greed', 'N/A')}/100 ({context.get('fear_greed_class', '')})
- News Sentiment: {context.get('news_sentiment', 'N/A')} (confidence: {context.get('sentiment_confidence', 'N/A')})

Multi-timeframe Analysis:
{context.get('timeframe_summary', 'N/A')}

Recent Headlines:
{context.get('headlines', 'N/A')}

Respond with EXACTLY one JSON object and no extra text.
Schema: {{"action":"BUY|SELL|HOLD","reasoning":"one sentence grounded in the data"}}
Use HOLD when the evidence is mixed or weak.[/INST]"""

    response = model(
        prompt,
        max_tokens=120,
        temperature=0.0,
        top_p=0.2,
        stop=["[INST]", "\n\n"],
    )

    text = response["choices"][0]["text"].strip()
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
        "model": "custom-lora" if lora_path else "CryptoTrader-LM",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lora", default=None, help="Path to LoRA GGUF (overrides default)"
    )
    args = parser.parse_args()

    context = json.loads(sys.stdin.read())
    result = predict(context, lora_path=args.lora)
    print(json.dumps(result))
