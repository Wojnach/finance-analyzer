#!/usr/bin/env python3
"""Ministral-8B trading signal via llama-cpp-python (GGUF).

Runs as a subprocess: reads JSON context from stdin, outputs JSON prediction.
Uses a separate venv (Q:/models/.venv-llm on Windows, ~/.venv-llm on Linux).
"""

import json
import platform
import sys

if platform.system() == "Windows":
    MODEL_PATH = r"Q:\models\ministral-8b-gguf\Ministral-8B-Instruct-2410-Q4_K_M.gguf"
else:
    MODEL_PATH = (
        "/home/deck/models/ministral-8b-gguf/Ministral-8B-Instruct-2410-Q4_K_M.gguf"
    )

_model = None


def load_model():
    global _model
    if _model is None:
        from llama_cpp import Llama

        _model = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False,
        )
    return _model


def predict(context):
    model = load_model()

    prompt = f"""[INST]You are an expert cryptocurrency trader. Based on the following market data, provide a single trading decision: BUY, SELL, or HOLD.

Market Data:
- Asset: {context['ticker']}
- Current Price: ${context['price_usd']:,.2f}
- 24h Change: {context.get('change_24h', 'N/A')}

Technical Indicators (15-minute candles):
- RSI(14): {context.get('rsi', 'N/A')}
- MACD Histogram: {context.get('macd_hist', 'N/A')}
- EMA(9) vs EMA(21): {'Bullish (9 > 21)' if context.get('ema_bullish') else 'Bearish (9 < 21)'}
- Bollinger Bands: Price is {context.get('bb_position', 'N/A')}

Market Sentiment:
- Fear & Greed Index: {context.get('fear_greed', 'N/A')}/100 ({context.get('fear_greed_class', '')})
- News Sentiment: {context.get('news_sentiment', 'N/A')}

Multi-timeframe Analysis:
{context.get('timeframe_summary', 'N/A')}

Recent Headlines:
{context.get('headlines', 'N/A')}

Respond with EXACTLY one of: BUY, SELL, or HOLD.
Then give a one-sentence reason.
Format: DECISION: [BUY/SELL/HOLD] - [reason][/INST]"""

    response = model(
        prompt,
        max_tokens=100,
        temperature=0.1,
        stop=["[INST]", "\n\n"],
    )

    text = response["choices"][0]["text"].strip()

    decision = "HOLD"
    for word in ["BUY", "SELL", "HOLD"]:
        if word in text.upper():
            decision = word
            break

    return {
        "action": decision,
        "reasoning": text[:200],
        "model": "Ministral-8B-Q4_K_M",
    }


if __name__ == "__main__":
    context = json.loads(sys.stdin.read())
    result = predict(context)
    print(json.dumps(result))
