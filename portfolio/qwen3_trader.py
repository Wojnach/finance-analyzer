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


def _run_native(prompt, max_tokens=256):
    """Run inference via native llama-completion binary."""
    assets = load_model()
    prompt_file = os.path.join(tempfile.gettempdir(), "qwen3_prompt.txt")
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    proc = subprocess.run(
        [
            assets["llama_cli"],
            "-m", assets["model_path"],
            "-ngl", "99",
            "-c", "2048",
            "-n", str(max_tokens),
            "--temp", "0.7",
            "--top-p", "0.8",
            "--no-display-prompt",
            "-f", prompt_file,
        ],
        capture_output=True,
        text=True,
        timeout=120,
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

Provide your trading decision as JSON.<|im_end|>
<|im_start|>assistant
"""


def _parse_response(text):
    """Parse model output into action + reasoning."""
    # Strip thinking tags if present
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
    return decision, reasoning


def predict(context):
    """Single-ticker prediction."""
    prompt = _build_prompt(context)
    text = _run_native(prompt)
    decision, reasoning = _parse_response(text)
    return {"action": decision, "reasoning": reasoning, "model": "Qwen3-8B"}


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
