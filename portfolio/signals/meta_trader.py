"""Custom Meta-Trader (Qwen2-36L unsloth) signal — scaffold (2026-05-15).

Status: SCAFFOLD. Returns HOLD with feature_unavailable=True until the
unsloth/transformers loader for `Q:/models/custom-meta-trader/` is
implemented and verified.

Role
----
Meta-model. Unlike the per-asset LLM voters (ministral, qwen3,
finance_llama, cryptotrader_lm), meta-trader is designed to consume
the OUTPUTS of other voters from the same cycle and produce a
synthesized verdict. This means it must be dispatched AFTER the other
LLMs in the signal_engine ordering, with their predictions available
in the context dict.

Once inference is wired, the contract is:

    1. Read context["upstream_llm_votes"]: a dict
       {sig_name: {"action": ..., "confidence": ...}} aggregated by the
       dispatch loop for the current ticker.
    2. Build a structured prompt around the per-model verdicts plus the
       OHLCV summary.
    3. Run Qwen2-36L (32K context, ~6GB VRAM) and parse the verdict.
    4. Return the canonical result dict.

cycle_modulo=5 in shadow_registry — the heaviest LLM in our stack, so
it runs at most every 5 minutes during the trail period.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("portfolio.signals.meta_trader")

_MODEL_DIR = r"Q:\models\custom-meta-trader"
_FEATURE_AVAILABLE = False  # flip to True once inference is wired


def compute_meta_trader_signal(df, context=None):
    """Scaffold: HOLD on every call, with feature_unavailable=True."""
    if _FEATURE_AVAILABLE:
        raise NotImplementedError(
            "meta_trader inference not yet implemented — scaffold only. "
            "See module docstring for the wiring contract."
        )

    return {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {"meta_trader": "HOLD"},
        "indicators": {
            "feature_unavailable": True,
            "reason": "scaffold",
            "model_dir": _MODEL_DIR,
        },
    }
