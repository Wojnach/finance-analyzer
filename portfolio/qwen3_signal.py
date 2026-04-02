"""Wrapper to call Qwen3-8B trading model.

Prefers persistent llama-server (HTTP), swapping models as needed.
Falls back to subprocess if server unavailable.
Uses GPU lock to coordinate with Ministral.

Supports batch mode: multiple tickers processed in one model-load cycle.
"""

import json
import logging
import platform
import subprocess
import time
from pathlib import Path

from portfolio.gpu_gate import gpu_gate
from portfolio.llama_server import query_llama_server
from portfolio.subprocess_utils import kill_orphaned_llama, run_safe

logger = logging.getLogger("portfolio.qwen3_signal")

# Batch queue — accumulates contexts, flushed when get_qwen3_batch() is called
_batch_queue: list[dict] = []
_batch_results: dict[str, dict] = {}  # ticker -> result, populated by flush


def _extract_json_from_stdout(stdout):
    """Extract JSON (object or array) from subprocess stdout."""
    if not stdout:
        return None
    text = stdout.strip()
    if not text:
        return None
    # Try parsing as-is (could be array for batch mode)
    if text.startswith("[") or text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # Find first [ or { and parse from there
    for start_char in ("[", "{"):
        idx = text.find(start_char)
        if idx >= 0:
            try:
                return json.loads(text[idx:])
            except json.JSONDecodeError:
                pass
    # Last resort: scan lines in reverse
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith(("{", "[")):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _call_qwen3(context):
    """Call Qwen3-8B, preferring persistent llama-server, with subprocess fallback."""
    from portfolio.qwen3_trader import _build_prompt, _parse_response
    prompt = _build_prompt(context)

    text = query_llama_server("qwen3", prompt, n_predict=1024, temperature=0.6,
                              top_p=0.95, stop=["<|endoftext|>", "<|im_end|>"])
    if text is not None:
        decision, reasoning, confidence = _parse_response(text)
        result = {"action": decision, "reasoning": reasoning, "model": "Qwen3-8B"}
        if confidence is not None:
            result["confidence"] = confidence
        return result

    # Fallback: subprocess (cold start)
    logger.info("llama-server unavailable for qwen3, falling back to subprocess")
    repo_root = Path(__file__).resolve().parent.parent
    if platform.system() == "Windows":
        python = str(repo_root / ".venv" / "Scripts" / "python.exe")
    else:
        python = str(repo_root / ".venv" / "bin" / "python")

    script = repo_root / "portfolio" / "qwen3_trader.py"
    cmd = [python, str(script)]

    try:
        result = run_safe(
            cmd,
            input=json.dumps(context),
            capture_output=True,
            text=True,
            timeout=240,
        )
    except subprocess.TimeoutExpired as e:
        stderr_text = e.stderr[-500:] if e.stderr else "(no stderr)"
        logger.error("Qwen3 subprocess timed out after 240s — stderr: %s", stderr_text)
        raise
    if result.returncode != 0:
        raise RuntimeError(f"Qwen3 failed: {result.stderr[-500:]}")
    payload = _extract_json_from_stdout(result.stdout)
    if payload is None:
        raise RuntimeError(f"Qwen3 returned invalid JSON: {result.stdout[-500:]}")
    return payload


def _call_qwen3_batch(contexts):
    """Call Qwen3-8B inference subprocess in batch mode.

    Loads model once, processes all tickers, returns list of results.
    Saves ~5s model load per additional ticker vs single-ticker mode.
    """
    if not contexts:
        return []

    repo_root = Path(__file__).resolve().parent.parent
    if platform.system() == "Windows":
        python = str(repo_root / ".venv" / "Scripts" / "python.exe")
    else:
        python = str(repo_root / ".venv" / "bin" / "python")

    script = repo_root / "portfolio" / "qwen3_trader.py"
    cmd = [python, str(script)]

    t0 = time.time()
    # Send as JSON array to trigger batch mode in qwen3_trader.py
    result = run_safe(
        cmd,
        input=json.dumps(contexts),
        capture_output=True,
        text=True,
        timeout=60 + 30 * len(contexts),  # 60s base + 30s per ticker (extended for deeper reasoning)
    )
    elapsed = time.time() - t0
    logger.info("Qwen3 batch: %d tickers in %.1fs (%.1fs/ticker)",
                len(contexts), elapsed, elapsed / len(contexts) if contexts else 0)

    if result.returncode != 0:
        raise RuntimeError(f"Qwen3 batch failed: {result.stderr[-500:]}")
    payload = _extract_json_from_stdout(result.stdout)
    if not isinstance(payload, list):
        raise RuntimeError(f"Qwen3 batch returned non-list: {type(payload)}")
    return payload


def get_qwen3_signal(context):
    """Get trading signal from Qwen3-8B with GPU gating.

    Returns dict with 'action', 'reasoning', 'model' keys.
    """
    try:
        killed = kill_orphaned_llama()
        if killed:
            logger.warning("Reaped %d orphaned llama process(es)", killed)
    except Exception:
        pass
    with gpu_gate("qwen3", timeout=300) as acquired:
        if not acquired:
            logger.warning("GPU gate timeout — returning HOLD")
            return {"action": "HOLD", "reasoning": "GPU busy", "model": "Qwen3-8B"}
        # Check VRAM before loading model — Qwen3 Q4_K_M needs ~5GB
        from portfolio.gpu_gate import get_vram_usage
        vram = get_vram_usage()
        if vram and vram["free_mb"] < 5500:
            logger.warning("Insufficient VRAM for Qwen3: %dMB free (need 5500MB) — returning HOLD", vram["free_mb"])
            return {"action": "HOLD", "reasoning": f"low VRAM ({vram['free_mb']}MB free)", "model": "Qwen3-8B"}
        return _call_qwen3(context)


def get_qwen3_signal_batch(contexts):
    """Get trading signals for multiple tickers in one model-load cycle.

    Args:
        contexts: list of context dicts, each with 'ticker' key.

    Returns:
        dict mapping ticker -> result dict.
    """
    if not contexts:
        return {}

    try:
        results = _call_qwen3_batch(contexts)
        # Map results back to tickers
        mapped = {}
        for ctx, res in zip(contexts, results):
            ticker = ctx.get("ticker", "UNKNOWN")
            mapped[ticker] = res
        return mapped
    except Exception as e:
        logger.warning("Qwen3 batch failed (%s), returning HOLD for all", e)
        return {
            ctx.get("ticker", "UNKNOWN"): {
                "action": "HOLD",
                "reasoning": f"batch error: {e}",
                "model": "Qwen3-8B",
            }
            for ctx in contexts
        }
