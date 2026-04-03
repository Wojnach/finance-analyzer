"""Batch queue for LLM signals — eliminates model swap overhead.

During parallel ticker processing, expired LLM cache entries are enqueued
instead of triggering immediate model loads. After all tickers finish,
flush_llm_batch() processes them grouped by model: load Ministral once →
query all tickers → swap to Qwen3 once → query all tickers.

Result: max 1 model swap per cycle instead of N swaps.
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
        # Deduplicate by cache_key
        if not any(k == cache_key for k, _ in _ministral_queue):
            _ministral_queue.append((cache_key, context))


def enqueue_qwen3(cache_key, context):
    """Add a Qwen3 cache miss to the batch queue."""
    with _lock:
        if not any(k == cache_key for k, _ in _qwen3_queue):
            _qwen3_queue.append((cache_key, context))


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

    # Phase 1: All Ministral queries (server loads ministral3 once)
    if m_batch:
        logger.info("LLM batch: %d Ministral queries", len(m_batch))
        try:
            from portfolio.ministral_signal import get_ministral_signal
            for cache_key, ctx in m_batch:
                try:
                    result = get_ministral_signal(ctx)
                    if result:
                        results[cache_key] = result
                except Exception as e:
                    logger.warning("LLM batch Ministral %s failed: %s", cache_key, e)
        except ImportError:
            logger.debug("ministral_signal not available")

    # Phase 2: All Qwen3 queries (server swaps to qwen3 once)
    if q_batch:
        logger.info("LLM batch: %d Qwen3 queries", len(q_batch))
        try:
            from portfolio.qwen3_signal import get_qwen3_signal
            for cache_key, ctx in q_batch:
                try:
                    result = get_qwen3_signal(ctx)
                    if result:
                        results[cache_key] = result
                except Exception as e:
                    logger.warning("LLM batch Qwen3 %s failed: %s", cache_key, e)
        except ImportError:
            logger.debug("qwen3_signal not available")

    elapsed = time.monotonic() - t0
    logger.info("LLM batch: %d results in %.1fs (M:%d Q:%d)",
                len(results), elapsed, len(m_batch), len(q_batch))
    return results
