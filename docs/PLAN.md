# Plan: Unified Persistent LLM Server

## Problem
Three separate GGUF models run via cold-start subprocesses, each loading the model
from disk every call (~5-10s CPU spike each time):
1. **Ministral-3-8B** (main loop, `portfolio/ministral_trader.py` → `llama-completion.exe`)
2. **Qwen3-8B** (main loop, `portfolio/qwen3_trader.py` → `llama-completion.exe`)
3. **Ministral-8B + LoRA** (metals loop, `data/metals_llm.py` → `Q:\models\ministral_trader.py`)

Plus Chronos (PyTorch, separate — stays as its own persistent process).

## Solution: Unified llama-server.exe

### Architecture
- **One `llama-server.exe` process** on port 8787, shared by both loops
- **Model swapping**: only one model fits in 10GB VRAM at a time. Server stops/starts
  with different model when a different model is requested.
- **Cross-process coordination**: file-based lock at `data/llama_server.lock` prevents
  both loops from swapping simultaneously.
- **Chronos unchanged**: PyTorch model, stays as its own persistent stdin/stdout server.

### Model configs
| Name | Model Path | VRAM | Used by |
|------|-----------|------|---------|
| ministral3 | ministral-3-8b-gguf/Q5_K_M.gguf | ~5.7GB | main loop |
| qwen3 | qwen3-8b-gguf/Q4_K_M.gguf | ~4.7GB | main loop |
| ministral8_lora | ministral-8b-gguf/Q4_K_M.gguf + LoRA | ~5GB | metals loop |

### Flow
1. Main loop signal cycle: load ministral3 → query N tickers → swap to qwen3 → query N tickers
2. Metals loop (every 5-30min): acquire lock → swap to ministral8_lora → query 4 tickers → release
3. Main loop waits if metals loop is mid-query (file lock)

### Files changed
- `portfolio/llama_server.py` — add ministral8_lora config, file-based lock
- `data/metals_llm.py` — use portfolio.llama_server instead of stdin/stdout Ministral server
- `portfolio/ministral_signal.py` — already wired (done)
- `portfolio/qwen3_signal.py` — already wired (done)

### Revert plan
If this fails:
1. `git revert HEAD` on main
2. Both loops fall back to subprocess.run (the fallback paths are preserved in all callers)
3. Metals loop falls back to its own stdin/stdout server (code still exists)

### What stays the same
- Chronos persistent server (metals_llm.py, stdin/stdout) — PyTorch, not GGUF
- GPU gate (`portfolio/gpu_gate.py`) — still used for non-server callers
- Prediction accuracy tracking — unaffected
- Signal formats — unchanged

---
*Written: 2026-04-02*
