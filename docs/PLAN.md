# Plan — Plex-aware Model Swap Coordination

**Date:** 2026-05-11
**Branch:** `feat/plex-vram-coord`
**Worktree:** `/mnt/q/finance-analyzer/.worktrees/plex-vram-coord`

## Why

Plex Media Server hard-crashes / HTTP-hangs when finance loop swaps LLM models while Plex is hardware-transcoding to the LG OLED TV. Two confirmed incidents on 2026-05-10:

| Time | Plex symptom | Finance event 0-60s prior |
|------|--------------|----------------------------|
| 17:21:33 | Hard crash (Plex Crash Uploader fired) | `llama-server` PID 21096 (Qwen3-8B Q4) spawned 17:20:42, loaded 5+ GB into VRAM |
| 18:50:17 | HTTP hang (process alive, unresponsive) | `llama-server` PID 27048 (Ministral-8B Q4 + LoRA) spawned 18:50:17 |

Hercules2 hardware: RTX 3080 10 GB VRAM. Intel iGPU is BIOS-disabled (firmware setting, no Windows path to fix today). All Plex transcoding currently goes through NVENC, sharing VRAM with finance loop's LLM models.

## Root cause (verified, not guessed)

The existing locking is **correct in design but insufficient in tolerance**.

**Architecture:**
- `portfolio/llama_server.py` is the single source of truth for llama-server lifecycle. Both `portfolio/main.py --loop` and `data/metals_loop.py` import `query_llama_server` / `query_llama_server_batch` from it.
- Cross-process serialization is via a file lock at `data/llama_server.lock` (acquired in `_acquire_file_lock`, held around the entire swap+query).
- Within `_start_server` (line 313), the sequence is: `_stop_server()` (kills old llama-server) → `_wait_for_vram_reclaim(min_free_mb=5632, max_wait=4.0)` (waits up to 4 s for ≥5.5 GB free) → `subprocess.Popen(...)` spawns new llama-server with `-ngl 99` (full GPU offload).

**What goes wrong:**
1. Plex `Plex Transcoder.exe` holds a CUDA NVENC encoder context (~300–500 MB VRAM) while transcoding.
2. Finance loop wants to swap from model A to model B. Kills A, waits 4 s, spawns B.
3. Windows VRAM release is async — A's ~6 GB takes several seconds to actually clear. Within the 4 s window, free VRAM may not yet reach 5.5 GB.
4. The wait returns at the 4 s ceiling regardless of whether the threshold was met (it proceeds anyway). New B starts loading.
5. B's `-ngl 99` allocates 5–6 GB. Combined with not-yet-released A (still 2–3 GB) and Plex NVENC (~500 MB), total briefly exceeds 10 GB.
6. CUDA driver evicts Plex's encoder context to satisfy B's allocation. Plex's NVENC session fails. Plex crashes (in-process, not BSOD) or hangs.

**Why it doesn't always crash:**
- Plex idle / direct-playing → no NVENC context → nothing to evict → safe
- Plex transcoding AND swap in progress → race

**Why existing `gpu_gate` doesn't catch it:**
- `gpu_gate` is used by some callers (`ministral_signal.py`, `qwen3_signal.py`, `signals/forecast.py`) but `query_llama_server` itself does not call `gpu_gate`. Instead, `llama_server.py` has its own file lock. Both locks correctly serialize finance-loop callers against each other, but neither knows anything about Plex.

## Fix — single-file change to `portfolio/llama_server.py`

Add Plex-transcode awareness inside `_wait_for_vram_reclaim` and the call site in `_start_server`. Detection is via `nvidia-smi --query-compute-apps` — direct evidence that Plex is currently holding a CUDA context. No Plex HTTP / token plumbing needed.

### Helper: `_plex_transcode_active() -> bool`

```python
def _plex_transcode_active() -> bool:
    """True iff Plex Transcoder.exe currently holds a CUDA compute/encode context.

    Cached for 5 s to avoid hammering nvidia-smi during VRAM polling loops.
    Falls back to False on any error — never blocks the finance loop.
    """
```

Implementation:
- `subprocess.run(["nvidia-smi", "--query-compute-apps=process_name", "--format=csv,noheader"], timeout=2)`
- Match `"Plex Transcoder"` (case-insensitive) in any line of stdout
- Module-level cache: `(timestamp, result)` with 5 s TTL

### Modify `_wait_for_vram_reclaim` (line 271)

Add an optional `plex_safe` mode that uses raised thresholds:
```python
def _wait_for_vram_reclaim(min_free_mb: int = 5632, max_wait: float = 4.0,
                           plex_safe: bool = False) -> float:
    if plex_safe:
        min_free_mb = max(min_free_mb, 7168)  # ≥7 GB free
        max_wait = max(max_wait, 30.0)        # up to 30 s
    ...
```

Reasoning for 7168 MB (7 GB) target:
- 8B Q4 model weights: ~4.5 GB
- KV cache @ c=4096: ~0.8 GB
- Working set during load (transient peaks): ~1 GB
- Plex NVENC reserve: ~0.5 GB headroom
- Total: ~6.8 GB → 7 GB threshold

### Modify `_start_server` (line 313)

Detect Plex once at the start of each swap, pass through to the wait, and abort the swap if VRAM headroom can't be reached:

```python
def _start_server(name):
    ...
    plex_active = _plex_transcode_active()
    if plex_active:
        logger.info("llama-server: Plex transcoding active, using safe VRAM reclaim (>=7 GB / 30 s)")
    _stop_server()
    waited = _wait_for_vram_reclaim(plex_safe=plex_active)
    if plex_active and (_query_free_vram_mb() or 0) < 7168:
        # VRAM did not clear in time AND Plex is still transcoding. Abandoning
        # the swap is safer than crashing Plex - the caller falls back to
        # subprocess inference (slower but never racing).
        logger.warning("llama-server: aborting %s swap, Plex transcoding and VRAM insufficient", name)
        return False
    ... # spawn as before
```

Returning `False` propagates through `_ensure_model` → `query_llama_server` → caller, which then returns `None` and the caller falls back (existing behaviour for "server unavailable").

## Files to modify (one)

| File | Change |
|------|--------|
| `portfolio/llama_server.py` | Add `_plex_transcode_active`; add `plex_safe` param to `_wait_for_vram_reclaim`; raise threshold + abort-on-timeout in `_start_server` |

## Files to NOT modify

- `data/metals_loop.py`, `data/metals_llm.py`, `portfolio/main.py` — they're callers, no change needed.
- `portfolio/gpu_gate.py`, `Q:/models/gpu_lock.py` — orthogonal concern.

## Tests

New file: `tests/test_llama_server_plex_aware.py`.

Cases (all use monkeypatched `subprocess.run` for `nvidia-smi`):

1. **`test_plex_active_detection_positive`** — `nvidia-smi` stdout contains `"Plex Transcoder.exe"` → returns True.
2. **`test_plex_active_detection_negative`** — stdout has only other processes → returns False.
3. **`test_plex_active_detection_cache_hits`** — call twice within 5 s, second call returns cached, `subprocess.run` called once total.
4. **`test_plex_active_detection_nvidia_smi_failure`** — `subprocess.run` raises → returns False (never True on error).
5. **`test_wait_reclaim_plex_safe_raises_threshold`** — passing `plex_safe=True` with `min_free_mb=5632`, polled VRAM = 6500 MB → still waits (6500 < 7168).
6. **`test_wait_reclaim_plex_safe_extends_timeout`** — `max_wait=4.0` overridden to ≥30.0.
7. **`test_start_server_aborts_when_plex_busy_and_low_vram`** — mock `_plex_transcode_active=True` + `_query_free_vram_mb=4000` → `_start_server` returns False without calling `subprocess.Popen`.
8. **`test_start_server_proceeds_when_plex_idle`** — `_plex_transcode_active=False` + low VRAM → proceeds (existing behaviour, regression guard).

## Verification (manual, after merge)

1. Start a 4K HEVC transcode to LG TV (force by setting client quality to 1080p / 8 Mbps).
2. Confirm `nvidia-smi` shows `Plex Transcoder.exe` in compute apps.
3. Trigger a finance-loop model swap (or wait for natural cycle ~5 min).
4. Watch loop logs for either `Plex transcoding active, using safe VRAM reclaim` or `aborting <model> swap`.
5. Confirm Plex playback continues without buffering / interruption.
6. Stop the stream → `nvidia-smi` no longer shows Plex Transcoder → next swap proceeds normally with old thresholds.

## Rollback

Single-file change. `git revert <SHA>` is clean. No data migration, no schema change. Safe to deploy.

## Execution batches (per /fgl)

| Batch | Files | What |
|-------|-------|------|
| 1 | `portfolio/llama_server.py` + `tests/test_llama_server_plex_aware.py` | Implement helper + raised thresholds + tests |
| 2 | (optional follow-up) | If telemetry shows abort firing too often, expose `plex_safe` thresholds via env vars |

## Out of scope

Per the parent plan at `/root/.claude/plans/i-want-us-to-agile-liskov.md`:
- Plex Direct Play optimisation (subtitle conversion etc.) is a manual workstream
- BIOS iGPU enable + Plex QuickSync switch is a separate manual workstream
- Watchdog (`C:\NSSM\plex-watchdog.ps1`) and Windows LocalDumps are already in place from earlier in the session
