#!/usr/bin/env python3
"""Qwen3-8B trading signal via native llama-completion binary (GGUF).

Runs as a subprocess: reads JSON context from stdin, outputs JSON prediction.
Uses the native llama.cpp binary (b8391+) for inference — no llama-cpp-python needed.
Supports single and batch mode (load model once for multiple tickers).
"""

import json
import os
import platform
import re
import subprocess
import sys
import tempfile

from portfolio.subprocess_utils import run_safe

if platform.system() == "Windows":
    MODEL_PATH = r"Q:\models\qwen3-8b-gguf\Qwen3-8B-Q4_K_M.gguf"
    LLAMA_CLI = r"Q:\models\llama-cpp-bin\cuda13\llama-completion.exe"
else:
    MODEL_PATH = "/home/deck/models/qwen3-8b-gguf/Qwen3-8B-Q4_K_M.gguf"
    LLAMA_CLI = "/usr/local/bin/llama-completion"


def _can_use_native():
    return os.path.exists(MODEL_PATH) and os.path.exists(LLAMA_CLI)


def load_model():
    """Validate native Qwen3 assets and return the resolved runtime paths."""
    missing = []
    if not os.path.exists(MODEL_PATH):
        missing.append(MODEL_PATH)
    if not os.path.exists(LLAMA_CLI):
        missing.append(LLAMA_CLI)
    if missing:
        raise FileNotFoundError(
            f"Qwen3 native assets missing: {', '.join(missing)}"
        )
    return {"model_path": MODEL_PATH, "llama_cli": LLAMA_CLI}


def _run_native(prompt, max_tokens=2048):
    """Run inference via native llama-completion binary."""
    assets = load_model()
    prompt_file = os.path.join(tempfile.gettempdir(), "qwen3_prompt.txt")
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    proc = run_safe(
        [
            assets["llama_cli"],
            "-m", assets["model_path"],
            "-ngl", "99",
            "-t", "4",  # cap CPU threads to prevent overheating
            "-c", "4096",
            "-n", str(max_tokens),
            "--temp", "0.7",
            "--top-p", "0.8",
            "--no-display-prompt",
            "-f", prompt_file,
        ],
        capture_output=True,
        text=True,
        timeout=240,
        stdin=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"llama-completion failed: {proc.stderr[-300:]}")
    return proc.stdout.strip()


def _extract_json_payload(text):
    """Extract a JSON payload from model output."""
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
    ticker = context.get("ticker", "UNKNOWN")
    asset_type = context.get("asset_type", "cryptocurrency")

    return f"""<|im_start|>system
You are an expert financial analyst specializing in {asset_type} trading.

Your job is to deeply analyze market data and produce a high-quality trading decision.
Think carefully before answering. Consider:
1. Are the technical indicators confirming each other or diverging?
2. Is the trend strengthening or weakening across timeframes?
3. Does volume support the move? Low volume signals are unreliable.
4. Is sentiment aligned with technicals, or is it contrarian?
5. What is the risk/reward setup? Is there a clear edge or is it ambiguous?

After your analysis, respond with a JSON object:
{{"action":"BUY|SELL|HOLD","confidence":0-100,"reasoning":"2-3 sentences explaining your logic"}}

Use HOLD when evidence is mixed, weak, or conflicting. A confident HOLD is better than a low-confidence BUY/SELL.
Confidence guide: 80+ = strong conviction, 60-79 = moderate, 40-59 = weak/mixed, <40 = default to HOLD.<|im_end|>
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

Analyze the data thoroughly, then provide your trading decision as JSON.<|im_end|>
<|im_start|>assistant
"""


def _parse_response(text):
    """Parse model output into action + reasoning + confidence."""
    # Strip thinking tags if present (Qwen3 native thinking mode)
    if "<think>" in text:
        think_end = text.find("</think>")
        if think_end >= 0:
            text = text[think_end + 8:].strip()

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


def predict(context):
    """Single-ticker prediction."""
    prompt = _build_prompt(context)
    text = _run_native(prompt)
    decision, reasoning, confidence = _parse_response(text)
    result = {"action": decision, "reasoning": reasoning, "model": "Qwen3-8B"}
    if confidence is not None:
        result["confidence"] = confidence
    return result


def predict_batch(contexts):
    """Process multiple tickers — one model load, sequential inference.

    Note: The native binary loads/unloads per call, so batch mode here
    just runs them sequentially. True batching requires llama-server.
    """
    results = []
    for ctx in contexts:
        try:
            results.append(predict(ctx))
        except Exception as e:
            results.append({"action": "HOLD", "reasoning": f"error: {e}", "model": "Qwen3-8B"})
    return results


if __name__ == "__main__":
    data = json.loads(sys.stdin.read())
    if isinstance(data, list):
        results = predict_batch(data)
        print(json.dumps(results))
    else:
        result = predict(data)
        print(json.dumps(result))
