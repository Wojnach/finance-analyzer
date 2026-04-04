"""Batch queue for LLM signals — eliminates model swap overhead.

During parallel ticker processing, expired LLM cache entries are enqueued
instead of triggering immediate model loads. After all tickers finish,
flush_llm_batch() processes them grouped by model: load Ministral once →
query all tickers → swap to Qwen3 once → query all tickers.

Result: max 1 model swap per cycle instead of N swaps.

Uses query_llama_server_batch() which holds the file lock for the entire
model phase, preventing the metals loop from swapping mid-batch (Codex #4).
"""

import logging
import threading
import time

logger = logging.getLogger("portfolio.llm_batch")

_lock = threading.Lock()
_ministral_queue: list[tuple[str, dict]] = []   # (cache_key, context)
_qwen3_queue: list[tuple[str, dict]] = []       # (cache_key, context)


def enqueue_ministral(cache_key, context):
    """Add a Ministral cache miss to the batch queue."""
    with _lock:
        if not any(k == cache_key for k, _ in _ministral_queue):
            _ministral_queue.append((cache_key, context))


def enqueue_qwen3(cache_key, context):
    """Add a Qwen3 cache miss to the batch queue."""
    with _lock:
        if not any(k == cache_key for k, _ in _qwen3_queue):
            _qwen3_queue.append((cache_key, context))


def _flush_via_server(model_name, batch, build_prompt_fn, parse_response_fn, stop_tokens):
    """Flush a batch using query_llama_server_batch (atomic, lock held for entire phase)."""
    try:
        from portfolio.llama_server import query_llama_server_batch
    except ImportError:
        return {}

    prompts_and_params = []
    for _cache_key, ctx in batch:
        prompt = build_prompt_fn(ctx)
        prompts_and_params.append({
            "prompt": prompt,
            "stop": stop_tokens,
        })

    texts = query_llama_server_batch(model_name, prompts_and_params)

    results = {}
    for (cache_key, _ctx), text in zip(batch, texts):
        if text is not None:
            parsed = parse_response_fn(text)
            if parsed:
                results[cache_key] = parsed
    return results


def flush_llm_batch():
    """Process all queued LLM requests, batched by model.

    Called once after ThreadPoolExecutor completes in main.py.
    Returns dict of {cache_key: result} for cache updates.
    """
    with _lock:
        m_batch = list(_ministral_queue)
        q_batch = list(_qwen3_queue)
        _ministral_queue.clear()
        _qwen3_queue.clear()

    if not m_batch and not q_batch:
        return {}

    results = {}
    t0 = time.monotonic()

    # Phase 1: All Ministral queries (lock held for entire phase)
    if m_batch:
        logger.info("LLM batch: %d Ministral queries", len(m_batch))
        try:
            from portfolio.ministral_trader import _build_prompt, _parse_response

            def _parse_ministral(text):
                decision, reasoning, confidence = _parse_response(text)
                result = {
                    "original": {"action": decision, "reasoning": reasoning, "model": "Ministral-3-8B"},
                    "custom": None,
                }
                if confidence is not None:
                    result["original"]["confidence"] = confidence
                return result

            phase = _flush_via_server("ministral3", m_batch, _build_prompt, _parse_ministral, ["[INST]"])
            results.update(phase)
        except Exception as e:
            logger.warning("LLM batch Ministral failed: %s", e)

    # Phase 2: All Qwen3 queries (lock held for entire phase)
    if q_batch:
        logger.info("LLM batch: %d Qwen3 queries", len(q_batch))
        try:
            from portfolio.qwen3_trader import _build_prompt as _qwen_build
            from portfolio.qwen3_trader import _parse_response as _qwen_parse_raw

            def _parse_qwen3(text):
                decision, reasoning, confidence = _qwen_parse_raw(text)
                result = {"action": decision, "reasoning": reasoning, "model": "Qwen3-8B"}
                if confidence is not None:
                    result["confidence"] = confidence
                return result

            phase = _flush_via_server(
                "qwen3", q_batch, _qwen_build, _parse_qwen3,
                ["<|endoftext|>", "<|im_end|>"],
            )
            results.update(phase)
        except Exception as e:
            logger.warning("LLM batch Qwen3 failed: %s", e)

    elapsed = time.monotonic() - t0
    logger.info("LLM batch: %d results in %.1fs (M:%d Q:%d)",
                len(results), elapsed, len(m_batch), len(q_batch))
    return results
