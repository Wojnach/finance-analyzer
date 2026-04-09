#!/usr/bin/env python3
"""Long-lived FinGPT inference daemon.

Loads a GGUF sentiment model once at startup and services per-request NDJSON
inference over stdin/stdout, so the main loop doesn't pay the 40-60s cold-load
cost on every subprocess invocation. Called from portfolio/sentiment.py.

Protocol
--------
Startup:
  → reads sys.path injection, imports fingpt_infer from /mnt/q/models
  → loads model under gpu_lock("fingpt", timeout=120)  (single warm load)
  → emits {"ready": true, "model": "<name>"} on stdout, flushes

Per request (one NDJSON line per request, one NDJSON line per response):
  in:  {"mode": "headlines"|"cumulative", "texts": [...], "ticker": "BTC", "request_id": 42}
  out: {"request_id": 42, "result": [...]}      (headlines)
       {"request_id": 42, "result": {...}}      (cumulative)
       {"request_id": 42, "error": "message"}   (on failure)

Shutdown:
  stdin EOF or {"quit": true} line → clean exit

Runs in the .venv-llm python environment (llama-cpp-python + CUDA). Does NOT
import anything from the finance-analyzer portfolio package — standalone.
"""

import json
import logging
import os
import sys
import traceback
from pathlib import Path

# ── Path injection for fingpt_infer import ─────────────────────────────
# fingpt_infer.py lives at /mnt/q/models/fingpt_infer.py (Windows: Q:\models).
# The .venv-llm python doesn't know about the finance-analyzer package tree,
# so we only need the Q:\models path and the standard library.
_MODELS_DIR = Path(r"Q:\models") if sys.platform == "win32" else Path.home() / "models"
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

# Configure minimal logging to stderr (stdout is reserved for the NDJSON protocol)
logging.basicConfig(
    level=logging.INFO,
    format="[fingpt_daemon] %(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("fingpt_daemon")


def _emit(obj: dict) -> None:
    """Write a single NDJSON line to stdout and flush."""
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _warm_load():
    """Load the fingpt model on CPU (no GPU contention). Returns (name, path, model).

    History of this function:
    - v1 (2026-04-09 merge 0e1697a): loaded via fingpt_infer._load_model which
      uses ``n_gpu_layers=-1`` (all layers on GPU). Paired with llama-server
      (~5 GB VRAM) this exhausted the 10 GB RTX 3080 budget and inference
      timed out after 60s. Measured in live logs immediately after merge.
    - v2 (2026-04-09 merge 5ab383f): preferred the 1.2B LFM2 sentiment model
      which would have fit in 1.5 GB, but the installed llama-cpp-python can
      NOT load the LFM2 architecture — it raises
      ``ValueError: Failed to load model from file``. Verified manually.
    - v3 (this commit): load finance-llama-8b on CPU via a direct Llama()
      call with ``n_gpu_layers=0``. No VRAM contention at all. Inference is
      slower (10-30s per batch instead of 1-3s on GPU) but fits comfortably
      inside the daemon's 60s request timeout and the new 600s loop cadence.
      GPU stays free for Kronos, Chronos, and llama-server.
    """
    import fingpt_infer  # imported from /mnt/q/models via sys.path injection above
    from llama_cpp import Llama

    name, path = fingpt_infer._find_model()
    if not path or not Path(path).exists():
        raise RuntimeError(
            f"No sentiment GGUF model found. Checked: {list(fingpt_infer.MODEL_PATHS.values())}"
        )

    # Load on CPU — no gpu_lock needed, no VRAM consumed.
    logger.info("Loading model %s on CPU (n_gpu_layers=0)", name)
    model = Llama(
        model_path=path,
        n_ctx=2048,
        n_gpu_layers=0,
        n_threads=4,   # same cap as fingpt_infer._load_model
        verbose=False,
    )

    logger.info("Loaded model %s from %s (pid %d)", name, path, os.getpid())
    return name, path, model


def _predict_headlines_warm(fingpt_infer, model, name: str, headlines: list) -> list:
    """Inference loop for per-headline mode, using a pre-loaded CPU model.

    Mirrors fingpt_infer.predict_headlines lines 151-195 but skips the model
    load (already warm) and skips the GPU lock entirely — v3 runs on CPU
    so there is no VRAM or CUDA context to guard against.
    """
    template = fingpt_infer.PROMPT_TEMPLATES.get(
        name, fingpt_infer.PROMPT_TEMPLATES["finance-llama-8b"]
    )

    results = []
    if True:  # keep indentation level for minimal diff vs prior version
        for headline in headlines:
            try:
                prompt = template.format(headline=headline)
                response = model(
                    prompt,
                    max_tokens=20,
                    temperature=0.1,
                    stop=["\n", "<|eot_id|>", "[INST]"],
                )
                text = response["choices"][0]["text"].strip()
                sentiment = fingpt_infer._parse_sentiment(text)
                confidence = fingpt_infer._estimate_confidence(text, sentiment)

                scores = {"positive": 0.1, "negative": 0.1, "neutral": 0.1}
                scores[sentiment] = confidence
                remaining = 1.0 - confidence
                other_labels = [lb for lb in fingpt_infer.SENTIMENT_LABELS if lb != sentiment]
                for ol in other_labels:
                    scores[ol] = remaining / len(other_labels)

                results.append({
                    "text": headline[:100],
                    "sentiment": sentiment,
                    "confidence": round(confidence, 4),
                    "scores": {k: round(v, 4) for k, v in scores.items()},
                    "model": f"fingpt:{name}",
                })
            except Exception as e:
                logger.warning("Inference failed for headline: %s", e)
                results.append({
                    "text": headline[:100],
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                    "model": f"fingpt:{name}:error",
                })
    return results


def _predict_cumulative_warm(fingpt_infer, model, name: str, headlines: list) -> dict:
    """Inference for cumulative/clustered mode, using a pre-loaded CPU model.

    Mirrors fingpt_infer.predict_cumulative lines 233-265 but skips the load
    and the GPU lock (v3 runs on CPU).
    """
    if not headlines:
        return {"sentiment": "neutral", "confidence": 0.0, "scores": {}, "model": "none"}

    headlines_block = "\n".join(f"- {h}" for h in headlines[:20])  # cap at 20
    prompt = fingpt_infer.CUMULATIVE_PROMPT.format(
        count=len(headlines),
        headlines_block=headlines_block,
    )

    if True:  # keep indentation level for minimal diff vs prior version
        try:
            response = model(
                prompt,
                max_tokens=30,
                temperature=0.1,
                stop=["\n", "<|eot_id|>"],
            )
            text = response["choices"][0]["text"].strip()
            sentiment = fingpt_infer._parse_sentiment(text)
            confidence = fingpt_infer._estimate_confidence(text, sentiment)

            # Boost confidence for cumulative (more data = more certainty)
            if len(headlines) >= 5:
                confidence = min(confidence + 0.1, 0.95)

            scores = {"positive": 0.1, "negative": 0.1, "neutral": 0.1}
            scores[sentiment] = confidence
            remaining = 1.0 - confidence
            for ol in [lb for lb in fingpt_infer.SENTIMENT_LABELS if lb != sentiment]:
                scores[ol] = remaining / 2

            return {
                "sentiment": sentiment,
                "confidence": round(confidence, 4),
                "scores": {k: round(v, 4) for k, v in scores.items()},
                "model": "fingpt:cumulative",
                "headline_count": len(headlines),
            }
        except Exception as e:
            logger.warning("Cumulative inference failed: %s", e)
            return {"sentiment": "neutral", "confidence": 0.0, "scores": {}, "model": "error"}


def main() -> int:
    # Warm load the model once
    try:
        import fingpt_infer
        name, path, model = _warm_load()
    except Exception as exc:
        logger.error("Warm load failed: %s\n%s", exc, traceback.format_exc())
        _emit({"ready": False, "error": str(exc)})
        return 1

    _emit({"ready": True, "model": name})

    # Request loop — one NDJSON line in, one NDJSON line out
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except Exception as exc:
            _emit({"error": f"invalid JSON: {exc}"})
            continue

        if req.get("quit"):
            logger.info("Quit requested, exiting cleanly")
            return 0

        req_id = req.get("request_id")
        mode = req.get("mode", "headlines")
        texts = req.get("texts", []) or []

        try:
            if mode == "cumulative":
                result = _predict_cumulative_warm(fingpt_infer, model, name, texts)
            else:
                result = _predict_headlines_warm(fingpt_infer, model, name, texts)
            _emit({"request_id": req_id, "result": result})
        except Exception as exc:
            logger.error("Request %s failed: %s\n%s", req_id, exc, traceback.format_exc())
            _emit({"request_id": req_id, "error": str(exc)})

    logger.info("stdin closed, exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
