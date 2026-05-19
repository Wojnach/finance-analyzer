# Unsloth Notes vs March 17 Runtime Work

Updated: 2026-03-18

## Purpose

This note separates two related but different threads:

1. the older **Unsloth / custom LoRA training** idea
2. the **runtime and inference changes actually implemented on 2026-03-17**

The distinction matters because no new Unsloth-based training pipeline was landed on
March 17. What landed was model/runtime hardening around local inference.

## The Unsloth Part

The Unsloth guidance lives in
`docs/plans/lora-custom-training-plan.md`.

Main takeaway from that plan:

- Unsloth is still the preferred future path for custom LoRA training on the RTX 3080
  10GB setup because it should allow faster iteration and lower VRAM pressure than a
  plain Transformers + PEFT flow.
- That applies to **training** a custom adapter, not to the production **inference**
  path currently used by the live trading system.

In other words: Unsloth is still relevant for future fine-tuning work, but it was not
the thing that was implemented in the March 17 runtime session.

## What Was Actually Implemented on 2026-03-17

The March 17 work upgraded and stabilized the local model runtime:

- `2346e8a`: upgraded Chronos to v2, upgraded Ministral to 3-8B, and added
  Qwen3-8B as signal `#31`
- `9f69d26` -> `56b6c57`: Qwen3 thinking mode was first disabled, then re-enabled
  with a tighter `256` token budget
- `57f365a`: added Qwen3 batch mode so one model load can serve multiple tickers
- `4d262ef`: moved Qwen3 onto native CUDA `llama-completion`
- `47783e0`: moved Ministral-3 onto native CUDA `llama-completion`
- `a9f9e3b` and `53ad9af`: added a GPU gate and expanded it across all four GPU
  model paths: Ministral, Qwen3, Chronos, and Kronos
- `0eab17b` and `b917217`: tuned model and GPU-gate timeouts after measuring real
  runtime behavior
- `1b63191`: fixed the Windows loop launcher by setting `PYTHONPATH` to stop
  detached crash loops

## Practical Learnings

### 1. Native llama.cpp was the stable inference path

The big operational win was moving Qwen3 and Ministral onto the native CUDA
`llama-completion` path. That became the reliable path for local inference.

### 2. Qwen3 needed tighter control to be usable

Qwen3 was not just a drop-in model swap. It needed:

- explicit thinking-mode tuning
- a small thinking budget
- batch mode to avoid repeated model-load overhead

Without that, the latency and runtime profile were not good enough.

### 3. GPU contention was a real system problem

Once Qwen3, Ministral, Chronos, and Kronos all shared the same GPU, serialized access
became necessary. The GPU gate was not an optimization detail; it was required for
runtime stability.

### 4. Operational plumbing mattered as much as model quality

Even after the model upgrades, the Windows loop environment could still fail for boring
reasons like missing `PYTHONPATH`. The session reinforced that runtime reliability is
not just about model choice.

## Current Conclusion

- **Unsloth** remains the recommended next step for future custom LoRA training and
  rapid fine-tuning iteration.
- **March 17** was primarily an inference/runtime stabilization session, not a new
  training pipeline session.
- The concrete wins from that session were native inference, batching, GPU
  coordination, and loop hardening.

## Verification Captured in Session Notes

From `memory/2026-03-17.md`:

```powershell
.venv\Scripts\python.exe -m pytest -q tests\test_message_store.py tests\test_model_upgrades.py tests\test_digest.py
```

Result:

```text
40 passed
```
