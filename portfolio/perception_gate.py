"""Perception gate — filters low-value Layer 2 invocations.

A rule-based pre-invocation filter (NOT an LLM call). Checks signal
consensus strength and trigger importance. If the gate decides to skip,
the agent is not invoked, saving tokens and latency.

Config:
    "perception_gate": {
        "enabled": true,
        "min_signal_strength": 0.3,
        "skip_tiers": [1]
    }
"""

import json
import logging
from pathlib import Path

from portfolio.api_utils import load_config

logger = logging.getLogger("portfolio.perception_gate")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Trigger keywords that always bypass the gate
_BYPASS_KEYWORDS = ("consensus", "F&G crossed", "post-trade")


def should_invoke(reasons, tier, config=None):
    """Decide whether to invoke Layer 2.

    Args:
        reasons: list[str] of trigger reasons.
        tier: int (1, 2, or 3).
        config: optional config dict. Loaded from disk if None.

    Returns:
        (should_invoke: bool, reason: str) explaining the decision.
    """
    if config is None:
        config = load_config()

    gate_cfg = config.get("perception_gate", {})
    if not gate_cfg.get("enabled", False):
        return True, "gate disabled"

    skip_tiers = gate_cfg.get("skip_tiers", [1])
    if tier not in skip_tiers:
        return True, f"T{tier} not in skip_tiers"

    # Force-bypass for important triggers
    for reason in reasons:
        for keyword in _BYPASS_KEYWORDS:
            if keyword in reason:
                return True, f"bypass: {keyword!r} in trigger"

    # Check signals from compact summary
    min_strength = gate_cfg.get("min_signal_strength", 0.3)
    summary = _load_compact_summary()
    if summary is None:
        return True, "no summary available, pass through"

    signals = summary.get("signals", {})
    if not signals:
        return False, "no signals in summary"

    max_confidence = 0.0
    non_hold_count = 0
    for ticker, sig in signals.items():
        if not isinstance(sig, dict):
            continue
        action = sig.get("action", "HOLD")
        conf = sig.get("confidence", 0.0)
        if action != "HOLD":
            non_hold_count += 1
            if conf > max_confidence:
                max_confidence = conf

    if non_hold_count == 0:
        return False, "no non-HOLD signals"

    if max_confidence < min_strength:
        return False, f"max confidence {max_confidence:.2f} < {min_strength}"

    return True, f"{non_hold_count} active signals, max conf {max_confidence:.2f}"


def _load_compact_summary():
    """Load the compact summary JSON."""
    path = DATA_DIR / "agent_summary_compact.json"
    if not path.exists():
        path = DATA_DIR / "agent_summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
