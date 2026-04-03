# Plan: LLM Signal Batching — Eliminate Model Swap Overhead

**Date**: 2026-04-02
**Status**: Ready to implement

## Problem

The shared `llama-server.exe` (port 8787) serves Ministral-3, Qwen3, and
Ministral-8B+LoRA, but can only hold one model in VRAM at a time. The signal
engine runs 8 parallel threads, each processing one ticker. Within each ticker,
signals are computed sequentially: first Ministral, then Qwen3. This causes
model swapping mid-cycle:

```
Thread 1: PLTR → get_ministral_signal → loads ministral3 (~10s)
Thread 1: PLTR → get_qwen3_signal    → swaps to qwen3 (~10s)
Thread 2: NVDA → get_ministral_signal → swaps to ministral3 (~10s)
Thread 2: NVDA → get_qwen3_signal    → swaps to qwen3 (~10s)
...
```

The 15-minute cache (MINISTRAL_TTL=900) means only 2-4 tickers expire per
120s cycle. But each expiry triggers 2 model swaps (ministral3 → qwen3),
and concurrent threads compound the swapping. The dogpile prevention in
`_cached()` (BUG-166) mitigates some of this by making threads wait for
stale data when another thread is loading, but the fundamental problem
remains: alternating models = constant swapping.

**Impact**: ~10s per swap × 2-8 swaps per cycle = 20-80s of server
restart overhead where the GPU is loading models from disk instead of
doing inference.

## Solution: Post-Cycle Batch Flush

### Architecture

Split LLM signal computation into two phases:

**Phase 1 (during parallel ticker processing)**: When a Ministral/Qwen3
cache entry expires, don't call the model immediately. Instead:
- Return the stale cached value (the dogpile `_MAX_STALE_FACTOR=3` allows
  data up to 45 minutes old)
- Add the ticker+context to a thread-safe batch queue

**Phase 2 (after ThreadPoolExecutor completes)**: Flush the batch queue:
1. Load ministral3 on llama-server
2. Query ALL queued Ministral tickers sequentially (~1s each)
3. Swap to qwen3 (one swap, ~10s)
4. Query ALL queued Qwen3 tickers sequentially (~1s each)
5. Update the signal cache with fresh values

**Result**: Maximum 1 model swap per cycle (ministral3 → qwen3), regardless
of how many tickers expired. With N expired tickers:
- Old: N × 2 swaps × ~10s = ~20N seconds
- New: 1 swap × ~10s + 2N queries × ~1s = ~10 + 2N seconds
- For N=4: old ~80s, new ~18s

### Trade-off

LLM signals are delayed by one cycle (~120s) on first cache miss after TTL
expiry. This is acceptable:
- Signals have 15-min TTL — 120s delay is <2% of the cache lifetime
- LLM signals are advisory, not latency-critical
- The stale value is still directionally useful

### Detailed Design

#### New: `portfolio/llm_batch.py` (new file, ~80 lines)

Thread-safe batch queue + flush function:

```python
"""Batch queue for LLM signals — collects expired cache entries during
parallel ticker processing, flushes after cycle completes."""

import threading
from portfolio.llama_server import query_llama_server

_lock = threading.Lock()
_ministral_queue = []   # list of (cache_key, context)
_qwen3_queue = []       # list of (cache_key, context)

def enqueue_ministral(cache_key, context):
    """Add a Ministral cache miss to the batch queue."""
    with _lock:
        _ministral_queue.append((cache_key, context))

def enqueue_qwen3(cache_key, context):
    """Add a Qwen3 cache miss to the batch queue."""
    with _lock:
        _qwen3_queue.append((cache_key, context))

def flush_llm_batch():
    """Process all queued LLM requests, batched by model.
    
    Called once after ThreadPoolExecutor completes.
    Returns dict of {cache_key: result} for cache update.
    """
    with _lock:
        m_batch = list(_ministral_queue)
        q_batch = list(_qwen3_queue)
        _ministral_queue.clear()
        _qwen3_queue.clear()
    
    results = {}
    
    # Phase 1: All Ministral queries (server loads ministral3 once)
    for cache_key, ctx in m_batch:
        result = _query_ministral(ctx)
        if result:
            results[cache_key] = result
    
    # Phase 2: All Qwen3 queries (server swaps to qwen3 once)
    for cache_key, ctx in q_batch:
        result = _query_qwen3(ctx)
        if result:
            results[cache_key] = result
    
    return results
```

#### Modified: `portfolio/signal_engine.py`

Change the Ministral/Qwen3 blocks (~lines 1126-1171) to enqueue instead
of calling directly when cache is expired:

```python
# Current:
ms = _cached("ministral_BTC", MINISTRAL_TTL, get_ministral_signal, ctx)

# New:
ms = _cached_or_enqueue("ministral_BTC", MINISTRAL_TTL,
                         get_ministral_signal, ctx,
                         enqueue_fn=enqueue_ministral)
```

The `_cached_or_enqueue` helper checks the cache — if fresh, returns it.
If expired, enqueues the request and returns stale data. The actual model
call happens in `flush_llm_batch()` after the cycle.

#### Modified: `portfolio/main.py`

After the `ThreadPoolExecutor` block (~line 406), add the batch flush:

```python
# After all tickers processed:
from portfolio.llm_batch import flush_llm_batch
batch_results = flush_llm_batch()
# Update cache with fresh LLM results
for cache_key, result in batch_results.items():
    _update_cache(cache_key, result)
```

#### Modified: `portfolio/shared_state.py`

Add `_cached_or_enqueue()` variant and `_update_cache()` function:

```python
def _cached_or_enqueue(key, ttl, func, *args, enqueue_fn=None):
    """Like _cached, but on miss: enqueue for batch instead of calling func.
    
    Returns stale data if available, None otherwise.
    """
    now = time.time()
    with _cache_lock:
        if key in _tool_cache and now - _tool_cache[key]["time"] < ttl:
            return _tool_cache[key]["data"]
        # Cache miss — enqueue for batch processing
        if enqueue_fn:
            enqueue_fn(key, args[0] if args else None)  # args[0] = context
        # Return stale if available
        if key in _tool_cache:
            age = now - _tool_cache[key]["time"]
            if age <= ttl * _MAX_STALE_FACTOR:
                return _tool_cache[key]["data"]
    return None

def _update_cache(key, data, ttl=None):
    """Update a cache entry directly (for batch flush results)."""
    with _cache_lock:
        _tool_cache[key] = {
            "data": data,
            "time": time.time(),
            "ttl": ttl or MINISTRAL_TTL,
        }
```

### Files Changed

| File | Change | Lines |
|------|--------|-------|
| `portfolio/llm_batch.py` | **NEW** — batch queue + flush | ~80 |
| `portfolio/shared_state.py` | Add `_cached_or_enqueue`, `_update_cache` | ~25 |
| `portfolio/signal_engine.py` | Ministral/Qwen3 use enqueue instead of direct call | ~10 |
| `portfolio/main.py` | Call `flush_llm_batch()` after ticker processing | ~5 |

### What Stays the Same

- `portfolio/llama_server.py` — no changes, server manages model swapping as before
- `portfolio/ministral_signal.py` — still called from batch flush, same interface
- `portfolio/qwen3_signal.py` — still called from batch flush, same interface
- `data/metals_llm.py` — metals loop unaffected (has own timing, low frequency)
- Cache TTLs, dogpile prevention, accuracy tracking — all unchanged

### Execution Order

1. Create `portfolio/llm_batch.py`
2. Add `_cached_or_enqueue` + `_update_cache` to `shared_state.py`
3. Modify `signal_engine.py` Ministral/Qwen3 blocks
4. Modify `main.py` to call `flush_llm_batch()` post-cycle
5. Run tests
6. Verify in logs that model swaps are reduced

### Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| One-cycle delay on cache miss | Acceptable: 120s << 900s TTL |
| Stale data returned on very first run (no cache) | `_cached_or_enqueue` returns None → HOLD default |
| Queue grows unbounded if flush fails | Queue is cleared at start of flush, max size = N tickers × 2 models |
| Metals loop triggers swap mid-batch | File lock in `llama_server.py` handles this — metals loop waits |

### Revert Plan

`git revert` the commit. The `_cached()` function still works — it's the
fallback path. Ministral/Qwen3 subprocess fallbacks remain in both signal
modules.

### Alternative Considered: Ministral-Only Server

Simpler option: only Ministral uses llama-server, Qwen3 stays on subprocess.
Rejected because:
- Qwen3 cold start (~10s) is also expensive
- We already have the server infrastructure
- Batching is cleaner and benefits both models

---
*Updated: 2026-04-02*
