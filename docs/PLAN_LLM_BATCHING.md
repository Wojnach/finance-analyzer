# Plan: LLM Signal Batching — Eliminate Model Swap Overhead

**Status**: Planned for 2026-04-02  
**Priority**: High — affects market-hours signal latency  

## Problem

The shared `llama-server.exe` serves both Ministral-3 and Qwen3, but can only
load one model at a time (10GB VRAM). The signal engine processes tickers in
parallel (8 threads), and each ticker calls Ministral then Qwen3 sequentially.
This causes constant model swapping:

```
Thread 1: ticker PLTR → Ministral (load ministral3) → Qwen3 (swap to qwen3)
Thread 2: ticker NVDA → Ministral (swap to ministral3) → Qwen3 (swap to qwen3)
...ping-pong swaps, ~10s each
```

Each swap = stop server + start with new model + wait for health = ~10s.
With 15-min TTL cache, ~2-4 tickers refresh per 120s cycle. Worst case:
4 tickers × 2 models × 10s swap = 80s of swapping per cycle.

**Before** (subprocess per call): ~5-10s cold start per call, but Ministral
and Qwen3 could run in parallel on separate threads (each loaded their own
model independently). Total: ~10-20s.

**Current** (shared server with swapping): similar or worse due to swap overhead.

## Solution: Batch-by-Model

Change the signal engine to batch all Ministral calls together, then all Qwen3
calls together. One model swap per cycle instead of per-ticker:

```
Phase 1: Load ministral3 → query PLTR, NVDA, BTC, ETH (all cached-expired tickers)
Phase 2: Swap to qwen3 → query PLTR, NVDA, BTC, ETH
Total: 1 swap (~10s) + N queries (~1s each) = ~10 + 4s = 14s
vs current: N swaps × 10s = ~40-80s
```

### Implementation

**File: `portfolio/signal_engine.py`**

The current flow (per ticker, parallel):
```python
# Inside generate_signals() per ticker:
votes["ministral"] = _cached("ministral_PLTR", 900, get_ministral_signal, ctx)
votes["qwen3"] = _cached("qwen3_PLTR", 900, get_qwen3_signal, ctx)
```

Proposed flow:
1. **Collect phase**: During parallel ticker processing, if Ministral/Qwen3 cache
   is expired, DON'T call the model. Instead, add the ticker+context to a batch queue.
   Return the stale cached value (or HOLD).
2. **Batch phase**: After all tickers are processed, run the batch:
   - Load ministral3 on llama-server
   - Query all queued Ministral tickers (fast, ~1s each)
   - Swap to qwen3
   - Query all queued Qwen3 tickers
3. **Update cache**: Store results back in the signal cache.
4. **Next cycle**: The freshly cached values are used.

**Trade-off**: LLM signals are delayed by one cycle (~120s) on first cache miss.
This is acceptable — the signals have 15-min TTL and are advisory, not latency-critical.

### Files to change

1. `portfolio/signal_engine.py` — add batch queue + post-cycle batch flush
2. `portfolio/ministral_signal.py` — add batch query function
3. `portfolio/qwen3_signal.py` — already has `get_qwen3_signal_batch()`
4. `portfolio/llama_server.py` — no changes needed (server stays as-is)

### Alternative: Simpler approach

If batching is too complex, a simpler option:
- Only Ministral uses llama-server (stays loaded, no swaps)
- Qwen3 continues using subprocess (cold start, but cached 15min)
- No model swapping at all

This loses the persistent-server benefit for Qwen3 but avoids all swap overhead.

### Risks

- Batch approach delays LLM signals by one cycle — monitor accuracy impact
- Queue management across parallel threads needs thread-safe data structure
- Qwen3 `get_qwen3_signal_batch()` already exists but uses subprocess internally

### Revert

If batching causes issues: `git revert` the batch commit. The fallback subprocess
paths in both signal modules ensure nothing breaks.

---
*Written: 2026-04-02 01:35 CET*
