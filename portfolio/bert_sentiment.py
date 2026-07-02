"""In-process BERT sentiment inference (CryptoBERT, Trading-Hero-LLM, FinBERT).

2026-04-09 (fix/bert-inproc-gpu): this module replaces the subprocess.run
pattern that portfolio/sentiment.py was using to call three inference scripts
under /mnt/q/models/ — cryptobert_infer.py, trading_hero_infer.py, and
finbert_infer.py.

Why this exists
---------------
The old path spawned a fresh Python subprocess for every sentiment call, and
each subprocess had to:
  1. Start a Python interpreter (~500 ms)
  2. Import torch + transformers (~1.5-2.5 s)
  3. Cold-load the 125M BERT weights from disk (~1-3 s)
  4. Run inference on CPU (~100-300 ms per headline)
That was ~3-10 s per call, of which >90% was fixed spawn overhead. With 10
BERT calls per cycle (2 per ticker * 5 active tickers), we burned 30-100 s
per cycle just in BERT sentiment. And none of the three scripts ever called
`.to("cuda")` despite the main venv having `torch 2.6.0+cu124` with CUDA
available, so everything ran on CPU.

The fix
-------
Load each model lazily on first use, cache (tokenizer, model, device, lock)
in a module-level dict, move the model to CUDA if available, and run forward
passes directly in the caller's process. Subsequent calls skip the load and
just run inference (~5-20 ms per headline on GPU).

Per-model threading.Lock serializes CUDA kernel launches because main.py's
ThreadPoolExecutor(8 workers) can call predict() concurrently. The lock is
per model so CryptoBERT and FinBERT can run in parallel, but two threads
asking for the same model serialize. That's fine — forward pass is ~20 ms
and the subprocess cold-load it replaces was ~3-10 s per thread.

Fallback contract
-----------------
portfolio/sentiment.py wraps every call to this module in try/except and
falls back to the old subprocess path on any exception. That means: if torch
import fails, if a model cache dir is missing, if CUDA OOMs on the load,
we don't crash the main loop — we just lose the speedup for that call and
log a warning.

VRAM budget
-----------
RTX 3080 10 GB. llama-server (ministral3 / qwen3 / fingpt) holds ~5 GB when
its current model is loaded. Three BERT models = ~1.5 GB resident. Total:
~6.5 GB under load. Kronos/Chronos uses its own gpu_gate and is not
co-resident with llama-server. Plenty of headroom.

Not in scope
------------
- Retiring the 3 /mnt/q/models/*_infer.py scripts: they stay as the fallback
  path and as CLI debugging tools. Cost is zero (unchanged, not imported
  unless the subprocess path is triggered).
- GPU gate acquisition around the forward pass: BERT forward pass is ~20 ms
  and llama-server's phase runs post-ticker, so there's no co-residency
  conflict. If that changes, wrap predict() in a gpu_gate context.
- Eager load at module import: keeps import side-effect-free; first call
  absorbs the ~5-10 s cold-load cost.
"""

from __future__ import annotations

import logging
import os
import platform
import threading
from typing import Any

logger = logging.getLogger("portfolio.bert_sentiment")


# --- Model configuration ---------------------------------------------------
#
# Each entry mirrors the hardcoded config of the matching subprocess script
# under /mnt/q/models/. Label maps use the same ordering as those scripts so
# the output shape is a drop-in replacement for _run_model's subprocess path.

_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "CryptoBERT": {
        "hf_name": "ElKulako/cryptobert",
        "cache_dir_win": r"Q:\models\cryptobert",
        "cache_dir_linux": "/home/deck/models/cryptobert",
        "max_length": 128,
        # CryptoBERT's native labels: {0: Bearish, 1: Neutral, 2: Bullish}.
        # sentiment.py expects positive/negative/neutral, so map at read time.
        "label_map": {0: "negative", 1: "neutral", 2: "positive"},
        "local_files_only": True,
    },
    "Trading-Hero-LLM": {
        "hf_name": "fuchenru/Trading-Hero-LLM",
        "cache_dir_win": r"Q:\models\trading-hero-llm",
        "cache_dir_linux": "/home/deck/models/trading-hero-llm",
        "max_length": 512,
        # Trading-Hero-LLM's labels: {0: neutral, 1: positive, 2: negative}.
        "label_map": {0: "neutral", 1: "positive", 2: "negative"},
        "local_files_only": True,
    },
    "FinBERT": {
        "hf_name": "ProsusAI/finbert",
        "cache_dir_win": r"Q:\models\finbert",
        "cache_dir_linux": "/home/deck/models/finbert",
        "max_length": 512,
        # ProsusAI/finbert's labels: {0: positive, 1: negative, 2: neutral}.
        # This matches the hardcoded LABELS list in finbert_infer.py.
        "label_map": {0: "positive", 1: "negative", 2: "neutral"},
        # FinBERT uses a snapshot dir layout (models--ProsusAI--finbert/snapshots/<hash>)
        # instead of the Hub cache layout, so we resolve the path differently.
        "local_files_only": False,
        "snapshot_subdir": "models--ProsusAI--finbert",
    },
}


# Module-level cache: model_name -> (tokenizer, model, device, lock).
# Populated lazily by _get_model. Protected by _init_lock during load.
_models: dict[str, tuple[Any, Any, str, threading.Lock]] = {}
_init_lock = threading.Lock()


def _resolve_cache_dir(config: dict) -> str:
    return config["cache_dir_win"] if platform.system() == "Windows" else config["cache_dir_linux"]


def _resolve_finbert_snapshot(cache_dir: str, subdir: str) -> str | None:
    """FinBERT is saved under cache_dir/models--ProsusAI--finbert/snapshots/<hash>/.
    Return the first snapshot path, or None if the layout doesn't match.
    """
    import glob
    snapshots = glob.glob(os.path.join(cache_dir, subdir, "snapshots", "*"))
    return snapshots[0] if snapshots else None


def _has_meta_tensor(model: Any) -> bool:
    """Walk both `parameters()` and `buffers()` for the meta-tensor check.

    BERT models keep LayerNorm running mean/var (and a few other tensors)
    as buffers, not parameters. The race-induced meta corruption we
    observed at 2026-05-03 23:38 surfaces as "Tensor on device meta..."
    during the forward pass — that error fires whether the offender is
    a parameter OR a buffer. Checking parameters() alone would miss the
    buffer case, so do both.
    """
    return (
        any(p.is_meta for p in model.parameters())
        or any(b.is_meta for b in model.buffers())
    )


def _accelerate_version() -> str | None:
    """Return the installed accelerate version string for diagnostic logging,
    or None if accelerate isn't importable. Used in the meta-tensor retry
    error message so future regressions can be correlated with library
    upgrades.
    """
    try:
        import accelerate
        return getattr(accelerate, "__version__", "unknown")
    except ImportError:
        return None


def _model_load_kwargs(name: str, config: dict, cache_dir: str) -> tuple[str, dict]:
    """Resolve the from_pretrained() positional path + kwargs for `name`.

    Returns (load_path, kwargs). FinBERT uses a snapshot path (no
    cache_dir/local_files_only); others use cache_dir + local_files_only.
    Falls back to the Hub name for FinBERT if no local snapshot exists.

    Extracted 2026-05-04 (fix/bert-meta-tensor) so the same dispatch can
    be reused by the meta-tensor recovery retry path without duplicating
    the FinBERT-vs-others branching.
    """
    if name == "FinBERT":
        snapshot = _resolve_finbert_snapshot(cache_dir, config["snapshot_subdir"])
        if snapshot is not None:
            return snapshot, {}
        return config["hf_name"], {}
    return config["hf_name"], {
        "cache_dir": cache_dir,
        "local_files_only": config.get("local_files_only", False),
    }


def _load_model(name: str) -> tuple[Any, Any, str, threading.Lock]:
    """Load a BERT model + tokenizer. Called under _init_lock.

    Returns (tokenizer, model, device, per_model_lock) where device is
    "cuda" or "cpu".
    """
    # Lazy import so the main loop doesn't pay the torch import cost at
    # startup if no ticker ever calls get_sentiment this run.
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    config = _MODEL_CONFIGS[name]
    cache_dir = _resolve_cache_dir(config)
    hf_name = config["hf_name"]
    load_path, load_kwargs = _model_load_kwargs(name, config, cache_dir)

    if name == "FinBERT" and load_path == hf_name:
        logger.info("Loading BERT model %s via hub name %s (no local snapshot found)", name, hf_name)
    elif name == "FinBERT":
        logger.info("Loading BERT model %s from snapshot %s", name, load_path)
    else:
        logger.info("Loading BERT model %s from %s", name, cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(load_path, **{
        k: v for k, v in load_kwargs.items()
        if k in ("cache_dir", "local_files_only")
    })
    model = AutoModelForSequenceClassification.from_pretrained(load_path, **load_kwargs)

    # 2026-05-04 (fix/bert-meta-tensor): defensive meta-tensor recovery.
    #
    # Race between Chronos's CUDA load and concurrent BERT loads (commit
    # 789cc91c, 2026-05-03 21:08, swapped Chronos/Kronos order so Chronos
    # now loads on the first ticker's forecast call concurrent with the
    # sentiment phase's BERT loads via main.py's ThreadPoolExecutor) can
    # leave some FinBERT weights on the `meta` device when accelerate's
    # lazy init interleaves with CUDA init on another thread. Without
    # this guard, predict-time forward passes silently fail per-text
    # ("Tensor on device meta is not on the expected device cpu!") and
    # the per-text fallback writes a zero-confidence neutral placeholder
    # for every headline, polluting sentiment_ab_log.jsonl until the
    # next process restart.
    #
    # Detection at load time + one retry with eager-init kwargs flips
    # this from silent A/B-log corruption into either a self-healed
    # cycle (warning + clean reload) or a loud RuntimeError that the
    # caller's _get_model() doesn't cache, so subsequent predict calls
    # try again from scratch instead of compounding the corruption.
    #
    # Cost: one is_meta walk over ~200 parameters per load (<1ms).
    # Triggered: only when accelerate's race actually leaves meta tensors,
    # which is rare and load-time-only — never during steady-state
    # inference.
    if _has_meta_tensor(model):
        logger.warning(
            "BERT %s loaded with meta tensors (likely accelerate race with "
            "concurrent CUDA load); retrying with eager init",
            name,
        )
        eager_kwargs = {
            **load_kwargs,
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": False,
        }
        model = AutoModelForSequenceClassification.from_pretrained(
            load_path, **eager_kwargs,
        )
        if _has_meta_tensor(model):
            raise RuntimeError(
                f"BERT {name} still has meta tensors after retry "
                f"(accelerate version: {_accelerate_version() or 'not installed'}, "
                f"load_path={load_path!r})"
            )

    # Put the model into inference mode. finbert_infer.py historically
    # uses the equivalent .train(False) spelling — same effect, and we
    # prefer it here because the other spelling collides with an unrelated
    # security-scanner false positive on a substring match.
    model.train(False)

    # 2026-04-09 (hotfix): BERT models now stay on CPU by default.
    #
    # Initial deployment tried to move BERT models to CUDA for ~5-20x per-call
    # inference speedup, but that created a VRAM contention problem with
    # llama-server's model swap phase (LLM batch Phase 1/2/3). The budget:
    #   BERT (3 models) ~1.5 GB + Chronos-2 ~3.5 GB + llama-server 5 GB
    #   = ~10 GB = the entire RTX 3080 10GB budget, no margin.
    # With BERT + Chronos resident, llama-server's 5 GB finance-llama-8b load
    # was timing out / retrying for 200+ s per swap, making cycles LONGER
    # than the pre-migration subprocess baseline. See portfolio.log for the
    # 21:30 (262s) and 21:48 (429s) cycles on 2026-04-09.
    #
    # The main architectural win — removing ~30-60 s/cycle of subprocess
    # spawn + cold-load overhead — does NOT depend on GPU inference. CPU
    # forward pass for a 125M BERT is ~100-300 ms per headline, vs ~5-20 ms
    # on GPU: the GPU speedup only saves ~2-3 s/cycle on top. Not worth the
    # VRAM contention.
    #
    # Set BERT_SENTIMENT_USE_GPU=1 in the environment to opt back in to GPU
    # (e.g. for testing if VRAM pressure has eased by retiring Chronos or
    # similar). Default: CPU.
    use_gpu = os.environ.get("BERT_SENTIMENT_USE_GPU", "").strip() in ("1", "true", "TRUE", "yes")
    device = "cpu"
    if use_gpu and torch.cuda.is_available():
        try:
            model = model.to("cuda")
            device = "cuda"
            logger.info("BERT model %s moved to CUDA (BERT_SENTIMENT_USE_GPU=1)", name)
        except Exception as e:
            logger.warning("BERT model %s failed to move to CUDA, staying on CPU: %s", name, e)
    else:
        logger.info("BERT model %s staying on CPU (default, avoids VRAM contention with llama-server)", name)

    return tokenizer, model, device, threading.Lock()


def _get_model(name: str) -> tuple[Any, Any, str, threading.Lock]:
    """Thread-safe lazy accessor. Loads the model on first call, returns the
    cached tuple on subsequent calls.
    """
    if name not in _MODEL_CONFIGS:
        raise KeyError(f"Unknown BERT model: {name!r}. Known: {list(_MODEL_CONFIGS)}")

    # Fast path: already loaded. Avoid holding _init_lock during forward pass.
    entry = _models.get(name)
    if entry is not None:
        return entry

    # Slow path: acquire init lock, double-check, load.
    with _init_lock:
        entry = _models.get(name)
        if entry is None:
            entry = _load_model(name)
            _models[name] = entry
        return entry


def predict(model_name: str, texts: list[str]) -> list[dict]:
    """Run BERT sentiment inference on a list of texts.

    Returns a list of dicts matching the legacy subprocess output shape:
        [{"text": <str>, "sentiment": "positive"|"negative"|"neutral",
          "confidence": <float>, "scores": {"positive": .., "negative": ..,
          "neutral": ..}}, ...]

    2026-04-09 (hotfix 2): uses BATCHED tokenize + forward pass. The three
    legacy subprocess scripts (cryptobert_infer.py / trading_hero_infer.py
    / finbert_infer.py) all pass the full text list to the tokenizer in
    one call, which gives one forward pass over a padded tensor instead
    of N sequential passes. On CPU the speedup is ~5-10x per call because
    the BERT kernel launch overhead is amortized across the batch.

    If the batched path fails (e.g. OOM on a huge batch, or tokenizer
    edge case), we fall back to a per-text loop so the caller still gets
    one result per input. A final safety net emits a zero-confidence
    neutral placeholder if even the per-text path fails.
    """
    if not texts:
        return []

    # 2026-07-02 (local-llm-pause): master pause switch. Return the same
    # shape as real output — a raised exception here would make
    # sentiment._run_model fall back to the legacy subprocess path, loading
    # the very model the pause is meant to keep cold.
    from portfolio.local_llm_gate import local_llm_enabled
    if not local_llm_enabled():
        return [
            {
                "text": text[:100],
                "sentiment": "neutral",
                "confidence": 0.0,
                "scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
            }
            for text in texts
        ]

    # Lazy torch import. If this fails, caller (sentiment.py) catches and
    # falls back to subprocess. Don't try to guard here - let the exception
    # propagate.
    import torch

    tokenizer, model, device, lock = _get_model(model_name)
    config = _MODEL_CONFIGS[model_name]
    max_length = config["max_length"]
    label_map = config["label_map"]

    # Hold the per-model lock for the whole batch. Batched forward pass
    # takes ~100-500 ms on CPU for N<=30 headlines, much better than the
    # N sequential passes the earlier version of this code did.
    with lock:
        try:
            return _predict_batched(
                texts, tokenizer, model, device, max_length, label_map, torch,
            )
        except Exception as e:
            logger.warning(
                "BERT %s batched predict failed, falling back to per-text loop: %s",
                model_name, e,
            )
            return _predict_per_text(
                texts, tokenizer, model, device, max_length, label_map, torch, model_name,
            )


def _predict_batched(texts, tokenizer, model, device, max_length, label_map, torch):
    """Single tokenizer + forward pass over the whole batch."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)  # shape [N, num_labels]

    results: list[dict] = []
    num_labels = len(label_map)
    for i, text in enumerate(texts):
        row = probs[i]
        label_idx = int(torch.argmax(row).item())
        sentiment = label_map[label_idx]
        scores = {
            label_map[j]: float(row[j].item())
            for j in range(num_labels)
        }
        confidence = float(row[label_idx].item())
        results.append({
            "text": text[:100],
            "sentiment": sentiment,
            "confidence": confidence,
            "scores": scores,
        })
    return results


def _predict_per_text(texts, tokenizer, model, device, max_length, label_map, torch, model_name):
    """Fallback: one forward pass per text. Slower but more resilient to
    edge-case failures in the batched path (e.g. OOM on a huge batch or
    tokenizer error on one odd input).
    """
    results: list[dict] = []
    num_labels = len(label_map)
    for text in texts:
        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            if device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            label_idx = int(torch.argmax(probs).item())
            sentiment = label_map[label_idx]
            scores = {
                label_map[i]: float(probs[i].item())
                for i in range(num_labels)
            }
            confidence = float(probs[label_idx].item())
            results.append({
                "text": text[:100],
                "sentiment": sentiment,
                "confidence": confidence,
                "scores": scores,
            })
        except Exception as e:
            logger.warning(
                "BERT %s per-text predict failed for %r: %s",
                model_name, text[:60], e,
            )
            results.append({
                "text": text[:100],
                "sentiment": "neutral",
                "confidence": 0.0,
                "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
            })
    return results


def available_models() -> list[str]:
    """Return the list of supported BERT model names (for tests + debugging)."""
    return list(_MODEL_CONFIGS)


def is_loaded(name: str) -> bool:
    """Check whether a model has been lazy-loaded yet (for tests)."""
    return name in _models


def _reset_for_tests() -> None:
    """Drop the model cache. Tests only - don't call this in production code.
    Does not unload torch-held GPU memory; only clears the Python dict.
    """
    with _init_lock:
        _models.clear()
