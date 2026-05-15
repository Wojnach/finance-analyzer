"""Finance-Llama-8B signal — scaffold (2026-05-15).

Status: SCAFFOLD. Returns HOLD with feature_unavailable=True until the
llama-cpp-python loader for `Q:/models/finance-llama-8b-gguf/
wiroai-finance-llama-8b-q4_k_m.gguf` is implemented and verified.

Once inference is wired, the contract is:

    1. Build a prompt from recent OHLCV summary + macro context.
    2. Run the model via portfolio.llm_batch (llama-server queue) so it
       shares the GPU multiplexer with ministral/qwen3.
    3. Extract probabilities from token logprobs at the answer position
       (the {BUY, HOLD, SELL} token IDs), normalize to sum=1.
    4. Return {"action": argmax, "confidence": max_prob,
               "sub_signals": {"finance_llama": action},
               "indicators": {"probs": probs, ...}}.

Until then this scaffold exists so:

  * The signal is registered (signal_registry / signal_engine dispatch).
  * shadow_registry.json has an entry tracking days-in-shadow.
  * llm_probability_log accepts "finance_llama" rows (already added to
    _LLM_SIGNALS in batch 1).

The scaffold returns HOLD with confidence=0.0, which is filtered out of
consensus computation at signal_engine._weighted_consensus (HOLD votes
are dropped before BUY/SELL weighting). It cannot influence trade
decisions.

Removing this scaffold:
  Once real inference ships, replace `compute_finance_llama_signal()`
  body with the real implementation. The registry registration and
  shadow entry can stay as-is.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("portfolio.signals.finance_llama")

# Module-level constants intentionally kept simple — no model load happens
# at import time. Loading the GGUF (~5GB) on every signal_engine reload
# would tank cycle startup and stress GPU memory unnecessarily.
_MODEL_PATH = r"Q:\models\finance-llama-8b-gguf\wiroai-finance-llama-8b-q4_k_m.gguf"
_FEATURE_AVAILABLE = False  # flip to True once inference is wired


def compute_finance_llama_signal(df, context=None):
    """Scaffold: always returns HOLD with feature_unavailable=True.

    Conforms to the enhanced-signal compute interface (df + optional
    context kwarg, returning the standard result dict shape consumed by
    signal_engine._validate_signal_result).

    df: pandas.DataFrame with OHLCV (unused while scaffold).
    context: dict passed by signal_engine; expected keys once inference
             is live: ticker, horizon, recent_news, macro snapshot.

    Returns the canonical abstention shape. confidence=0.0 means HOLD
    is filtered out of active_votes in _weighted_consensus, so this
    cannot drive a consensus shift.
    """
    if _FEATURE_AVAILABLE:
        # Sentinel branch — never executed in the scaffold. Once inference
        # is added the body goes here.
        raise NotImplementedError(
            "finance_llama inference not yet implemented — scaffold only. "
            "See module docstring for the wiring contract."
        )

    return {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {"finance_llama": "HOLD"},
        "indicators": {
            "feature_unavailable": True,
            "reason": "scaffold",
            "model_path": _MODEL_PATH,
        },
    }
