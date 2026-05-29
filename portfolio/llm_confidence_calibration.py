"""Fitted confidence calibration map for LLM-family signals.

Why this exists (2026-05-29)
----------------------------
For ministral & qwen3 the per-class probabilities logged to
`data/llm_probability_log.jsonl` are SYNTHETIC: the model emits only an
argmax action plus a self-reported scalar `confidence`, and
`llm_probability_log.derive_probs_from_result()` confidence-splits that
scalar (chosen action gets `confidence`, the other two split the rest).

Measured against realized outcomes this is WORSE than a uniform guess —
production Brier was ministral 0.785, qwen3 0.793 vs. uniform 0.667. The
root cause is that these models are *anti-calibrated*: on the directional
(BUY/SELL) rows that joined to an outcome, ministral self-reported a mean
confidence of 0.686 while only being right 23% of the time; qwen3 reported
0.716 mean confidence while being right 38% of the time. High confidence on
wrong calls is the textbook signature of overconfidence, and feeding that
raw confidence straight into the probability assigned to the chosen class
inflates Brier far past uniform.

What we do instead
------------------
We DO NOT try to read real token logprobs out of llama.cpp. The
ministral/qwen3 hot path is batched (signal_engine -> llm_batch
`_flush_via_server` -> `query_llama_server_batch` -> `_query_http`) and the
parsers only ever see `text: str`; the verdict token is emitted after
freeform reasoning as multi-token JSON (and qwen3 runs at temp 0.6 with a
`<think>` block). Locating the action-token position in that stream and
marginalizing BPE subwords is fragile and would put garbage probabilities
on a LIVE trading loop. The forced-single-token + `/completion` `n_probs`
path is documented as an opt-in follow-up (see FOLLOW-UP note at the bottom
of this module) but is intentionally NOT wired up here.

Instead this module loads an offline-fitted **empirical calibration map**:
for each signal, raw self-reported confidence is binned and replaced by the
empirically-measured P(chosen action == realized outcome) within that bin.
Anti-calibrated confidence collapses toward the true (low) hit rate, which
provably lowers Brier (offline, on the directional joined set: ministral
0.959 -> 0.551, qwen3 0.809 -> 0.591, both below uniform). The map is
produced by `scripts/fit_llm_confidence_calibration.py` from the same
log join outcomes the dashboard / shadow gate already use.

Safety contract (this is on a live loop — degrade gracefully)
-------------------------------------------------------------
* `calibrate()` NEVER raises. Any failure (file missing, malformed JSON,
  unknown signal, bin not found, out-of-range value) returns the input
  confidence UNCHANGED. Identity == today's behavior, so a broken or
  absent map can never make Brier worse than the synthetic baseline.
* The map is loaded lazily and cached with a TTL (default 300 s) so the
  hot loop does not re-read the JSON every call. Thread-safe.
* This module is only consumed by
  `llm_probability_log.derive_probs_from_result()` in its confidence-split
  fallback branch (NEVER the `avg_scores` rich-shape branch — sentiment's
  per-class scores are already real and must not be touched).

Map schema (`data/llm_confidence_calibration.json`)
---------------------------------------------------
```json
{
  "fitted_at": "2026-05-29T...",
  "method": "equal-width-bins",
  "signals": {
    "ministral": {
      "n": 5222,
      "bins": [[0.0, 0.2, 0.31, 812], [0.2, 0.4, 0.28, 640], ...]
    },
    "qwen3": { "n": 6055, "bins": [...] }
  }
}
```
Each bin is `[lo, hi, p_correct, count]`. `lo <= confidence < hi`
(the top bin is inclusive of 1.0). `p_correct` is the empirical hit rate
of the chosen action vs. realized outcome in that bin, or null when the bin
is too thin to trust (the consumer treats null as identity).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from portfolio.file_utils import load_json

logger = logging.getLogger("portfolio.llm_confidence_calibration")

_BASE_DIR = Path(__file__).resolve().parent.parent
_MAP_PATH = _BASE_DIR / "data" / "llm_confidence_calibration.json"

# TTL cache. The fitting job runs at most daily, so a 300 s TTL is plenty and
# keeps the hot signal loop from re-reading the JSON on every vote. Mirrors the
# is_promoted() cache cadence in shadow_registry.py.
_TTL_S = 300.0
_lock = threading.Lock()
_cache: dict = {"data": None, "loaded_at": 0.0, "path": None}


def _load_map(path: Path | str | None = None) -> dict:
    """Return the parsed calibration map's signals dict, or {} on any
    problem. TTL-cached + thread-safe. Never raises.

    A test or caller can pass an explicit `path` to bypass the production
    file; the cache keys on the path so test isolation holds.
    """
    p = Path(path) if path else _MAP_PATH
    now = time.time()
    with _lock:
        fresh = (
            _cache["data"] is not None
            and _cache["path"] == str(p)
            and (now - _cache["loaded_at"]) < _TTL_S
        )
        if fresh:
            return _cache["data"]
        data = load_json(str(p), default=None)
        signals: dict = {}
        if isinstance(data, dict):
            raw = data.get("signals")
            if isinstance(raw, dict):
                signals = raw
        _cache["data"] = signals
        _cache["loaded_at"] = now
        _cache["path"] = str(p)
        return signals


def _invalidate_cache() -> None:
    """Test hook — drop the TTL cache so a freshly written map is observed."""
    with _lock:
        _cache["data"] = None
        _cache["loaded_at"] = 0.0
        _cache["path"] = None


def calibrate(
    signal: str,
    action: str,
    confidence: float,
    *,
    path: Path | str | None = None,
) -> float:
    """Map a model's raw self-reported `confidence` to the empirically
    calibrated probability that its chosen `action` is correct.

    Returns the calibrated confidence in [0, 1] when a fitted bin is found
    for `signal`, otherwise returns the input `confidence` UNCHANGED. Never
    raises — the hot loop calls this and must never break.

    `action` is accepted for forward compatibility (a future map could be
    per-action) but is not used by the current equal-width-bin map. It is
    intentionally part of the signature so callers don't have to change when
    the map gains per-action resolution.
    """
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        return confidence
    # Out-of-range or non-finite confidence: pass through unchanged. The
    # caller (derive_probs_from_result) already clamps/handles these.
    if not (0.0 <= conf <= 1.0):
        return confidence

    try:
        signals = _load_map(path=path)
        entry = signals.get(signal)
        if not isinstance(entry, dict):
            return conf  # no map for this signal -> identity
        bins = entry.get("bins")
        if not isinstance(bins, list) or not bins:
            return conf
        for b in bins:
            # Each bin is [lo, hi, p_correct, count]. Be defensive about shape.
            if not isinstance(b, (list, tuple)) or len(b) < 3:
                continue
            lo, hi, p_correct = b[0], b[1], b[2]
            try:
                lo = float(lo)
                hi = float(hi)
            except (TypeError, ValueError):
                continue
            # Top bin is inclusive of the upper edge so conf==1.0 matches.
            in_bin = (lo <= conf < hi) or (conf >= hi and hi >= 1.0)
            if in_bin:
                if p_correct is None:
                    return conf  # under-sampled bin -> identity
                try:
                    p = float(p_correct)
                except (TypeError, ValueError):
                    return conf
                if 0.0 <= p <= 1.0:
                    return p
                return conf
        # No bin matched (shouldn't happen with full coverage) -> identity.
        return conf
    except Exception as e:  # pragma: no cover - defensive; never break the loop
        logger.debug("calibrate failed for %s/%s: %s", signal, action, e)
        return confidence


# ---------------------------------------------------------------------------
# FOLLOW-UP (opt-in, NOT implemented 2026-05-29): real token logprobs.
#
# llama.cpp's native /completion endpoint supports `n_probs: N`, returning a
# `completion_probabilities` array of per-generated-token top-N logprobs.
# A genuinely model-derived per-class distribution would:
#   1. Constrain the model to emit a final single verdict token after its
#      reasoning (e.g. a "VERDICT: <BUY|HOLD|SELL>" tail with a grammar or a
#      forced continuation), so the class token position is deterministic.
#   2. Read top_logprobs at that position and softmax the three class logits.
#   3. Thread the real probs from the trader through llm_batch into
#      derive_probs_from_result (replacing the confidence-split for these
#      signals only when real probs are present).
# This touches the BATCHED hot path (_query_http return contract, the two
# parse layers, the prewarmer) and is fragile on a live loop, so it stays
# behind a future config flag. The calibration map above is the low-risk
# fix that measurably lowers Brier today without touching inference.
# ---------------------------------------------------------------------------
