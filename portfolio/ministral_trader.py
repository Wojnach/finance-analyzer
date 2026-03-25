#!/usr/bin/env python3
"""Ministral trading signal via llama.cpp (GGUF).

Runs as a subprocess: reads JSON context from stdin, outputs JSON prediction.

Two inference paths:
1. Ministral-3-8B via native llama-cli binary (bypasses Python bindings)
2. Legacy Ministral-8B via llama-cpp-python + CryptoTrader-LM LoRA (fallback)
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys

from portfolio.subprocess_utils import run_safe

if platform.system() == "Windows":
    MODEL_PATH = r"Q:\models\ministral-3-8b-gguf\Ministral-3-8B-Instruct-2512-Q5_K_M.gguf"
    LEGACY_MODEL_PATH = r"Q:\models\ministral-8b-gguf\Ministral-8B-Instruct-2410-Q4_K_M.gguf"
    LLAMA_CLI = r"Q:\models\llama-cpp-bin\cuda13\llama-completion.exe"
    DEFAULT_LORA = None
else:
    MODEL_PATH = "/home/deck/models/ministral-3-8b-gguf/Ministral-3-8B-Instruct-2512-Q5_K_M.gguf"
    LEGACY_MODEL_PATH = "/home/deck/models/ministral-8b-gguf/Ministral-8B-Instruct-2410-Q4_K_M.gguf"
    LLAMA_CLI = "/usr/local/bin/llama-cli"
    DEFAULT_LORA = None


def _can_use_native_cli():
    """Check if native llama-cli binary is available for Ministral-3."""
    return os.path.exists(MODEL_PATH) and os.path.exists(LLAMA_CLI)


def _predict_native(prompt):
    """Run Ministral-3 via native llama-cli binary (CUDA, supports mistral3 arch).

    Uses file-based prompt input and --no-cnv to prevent interactive chat mode.
    stdin is closed to prevent the process from hanging.
    """
    import tempfile
    prompt_file = os.path.join(tempfile.gettempdir(), "ministral3_prompt.txt")
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    proc = run_safe(
        [
            LLAMA_CLI,
            "-m", MODEL_PATH,
            "-ngl", "99",
            "-t", "4",  # cap CPU threads to prevent overheating
            "-c", "4096",
            "-n", "1024",
            "--temp", "0",
            "--no-display-prompt",
            "-f", prompt_file,
        ],
        capture_output=True,
        text=True,
        timeout=240,
        stdin=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"llama-cli failed: {proc.stderr[-300:]}")
    return proc.stdout.strip()


def load_model(lora_path=None):
    """Load legacy Ministral-8B via llama-cpp-python (fallback only)."""
    from llama_cpp import Llama

    if os.path.exists(LEGACY_MODEL_PATH):
        lora = lora_path or r"Q:\models\cryptotrader-lm\cryptotrader-lm-lora.gguf"
        kwargs = {
            "model_path": LEGACY_MODEL_PATH,
            "n_ctx": 4096,
            "n_gpu_layers": -1,
            "verbose": False,
        }
        if lora and os.path.exists(lora):
            kwargs["lora_path"] = lora
        return Llama(**kwargs)

    raise FileNotFoundError(f"No model found at {LEGACY_MODEL_PATH}")


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


def _build_prompt(context):
    return f"""[INST]You are an expert financial analyst. Analyze the following market data step by step before making a trading decision.

Think through these questions:
1. What is the overall trend? Are EMA and MACD aligned?
2. Is RSI signaling overbought/oversold, and does volume confirm?
3. Are multiple timeframes in agreement or conflicting?
4. Does sentiment support or contradict the technical picture?
5. What is your conviction level given the evidence?

Market Data:
- Asset: {context['ticker']}
- Current Price: ${context['price_usd']:,.2f}
- 24h Change: {context.get('change_24h', 'N/A')}

Technical Indicators:
- RSI(14): {context.get('rsi', 'N/A')}
- MACD Histogram: {context.get('macd_hist', 'N/A')}
- EMA(9) vs EMA(21): {'Bullish (9 > 21)' if context.get('ema_bullish') else 'Bearish (9 < 21)'} (gap: {context.get('ema_gap_pct', 'N/A')}%)
- Bollinger Bands: Price is {context.get('bb_position', 'N/A')}
- Volume Ratio: {context.get('volume_ratio', 'N/A')}x avg

Market Sentiment:
- Fear & Greed Index: {context.get('fear_greed', 'N/A')}/100 ({context.get('fear_greed_class', '')})
- News Sentiment: {context.get('news_sentiment', 'N/A')} (confidence: {context.get('sentiment_confidence', 'N/A')})

Multi-timeframe Analysis:
{context.get('timeframe_summary', 'N/A')}

Recent Headlines:
{context.get('headlines', 'N/A')}

First reason through the data step by step, then provide your final answer as a JSON object:
{{"action":"BUY|SELL|HOLD","confidence":0-100,"reasoning":"2-3 sentences explaining your logic"}}

Confidence guide: 80+ = strong conviction, 60-79 = moderate, 40-59 = weak/mixed, <40 = default to HOLD.[/INST]"""


def _parse_response(text):
    """Parse model output into action + reasoning + confidence."""
    payload = _extract_json_payload(text)
    decision = None
    reasoning = text[:200]
    confidence = None
    if isinstance(payload, dict):
        raw_action = str(payload.get("action", "")).upper()
        if raw_action in {"BUY", "SELL", "HOLD"}:
            decision = raw_action
        if payload.get("reasoning"):
            reasoning = str(payload["reasoning"])[:200]
        if payload.get("confidence") is not None:
            try:
                confidence = int(float(payload["confidence"]))
                confidence = max(0, min(100, confidence))
            except (ValueError, TypeError):
                pass
    if decision is None:
        match = re.search(r"\b(BUY|SELL|HOLD)\b", text.upper())
        decision = match.group(1) if match else "HOLD"
    return decision, reasoning, confidence


def predict(context, lora_path=None):
    prompt = _build_prompt(context)

    # Path 1: Native llama-cli binary (Ministral-3, CUDA, supports mistral3 arch)
    if _can_use_native_cli():
        try:
            text = _predict_native(prompt)
            decision, reasoning, confidence = _parse_response(text)
            result = {"action": decision, "reasoning": reasoning, "model": "Ministral-3-8B"}
            if confidence is not None:
                result["confidence"] = confidence
            return result
        except Exception as e:
            print(f"Native CLI failed ({e}), falling back to legacy", file=sys.stderr)

    # Path 2: Legacy llama-cpp-python (Ministral-8B + LoRA)
    model = load_model(lora_path)
    response = model(
        prompt,
        max_tokens=1024,
        temperature=0.0,
        top_p=0.2,
        stop=["[INST]"],
    )
    text = response["choices"][0]["text"].strip()
    decision, reasoning, confidence = _parse_response(text)
    result = {"action": decision, "reasoning": reasoning, "model": "CryptoTrader-LM"}
    if confidence is not None:
        result["confidence"] = confidence
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lora", default=None, help="Path to LoRA GGUF (overrides default)"
    )
    args = parser.parse_args()

    context = json.loads(sys.stdin.read())
    result = predict(context, lora_path=args.lora)
    print(json.dumps(result))
