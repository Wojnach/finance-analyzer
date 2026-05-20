"""Online CUSUM change detection for signal accuracy.

Complements the batch accuracy_degradation.py (daily snapshot, 7d window)
with online detection that can catch accuracy shifts within 3-7 observations
instead of waiting for a full batch window.

Uses a two-sided CUSUM (Page's test): tracks cumulative positive and negative
deviations from a reference accuracy. When either side exceeds the control
limit h, a shift is detected.

State persisted to data/cusum_accuracy_state.json. Called from outcome_tracker
after each outcome backfill updates accuracy.

Public API:
    update_cusum(signal_name, was_correct, reference_accuracy) -> dict | None
        Returns alert dict if shift detected, else None.
    get_cusum_state() -> dict
        Returns current CUSUM state for all signals.
    reset_signal(signal_name)
        Resets CUSUM counters for a signal (e.g. after acknowledged degradation).
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from portfolio.file_utils import atomic_write_json, load_json

logger = logging.getLogger("portfolio.cusum_accuracy_monitor")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATE_FILE = DATA_DIR / "cusum_accuracy_state.json"

_lock = threading.Lock()

ALLOWABLE_SLACK = 0.03
CONTROL_LIMIT_H = 3.0
MIN_OBSERVATIONS = 20


def _load_state() -> dict:
    state = load_json(STATE_FILE)
    if not state or not isinstance(state, dict):
        return {"signals": {}, "alerts": []}
    return state


def _save_state(state: dict) -> None:
    atomic_write_json(STATE_FILE, state)


def _get_signal_state(state: dict, signal_name: str) -> dict:
    signals = state.setdefault("signals", {})
    if signal_name not in signals:
        signals[signal_name] = {
            "s_pos": 0.0,
            "s_neg": 0.0,
            "n": 0,
            "last_alert_n": 0,
            "reference_accuracy": None,
        }
    return signals[signal_name]


def update_cusum(
    signal_name: str,
    was_correct: bool,
    reference_accuracy: float | None = None,
) -> dict[str, Any] | None:
    """Update CUSUM for a signal after observing an outcome.

    Args:
        signal_name: Signal identifier.
        was_correct: Whether the signal's prediction was correct.
        reference_accuracy: Expected accuracy (0-1). If None, uses stored reference.

    Returns:
        Alert dict if CUSUM detects a shift, else None.
    """
    with _lock:
        state = _load_state()
        sig = _get_signal_state(state, signal_name)

        if reference_accuracy is not None:
            sig["reference_accuracy"] = reference_accuracy

        ref = sig.get("reference_accuracy")
        if ref is None or ref <= 0 or ref >= 1:
            _save_state(state)
            return None

        x = 1.0 if was_correct else 0.0
        deviation = x - ref

        sig["s_pos"] = max(0.0, sig["s_pos"] + deviation - ALLOWABLE_SLACK)
        sig["s_neg"] = max(0.0, sig["s_neg"] - deviation - ALLOWABLE_SLACK)
        sig["n"] += 1

        alert = None
        if sig["n"] >= MIN_OBSERVATIONS and sig["n"] > sig.get("last_alert_n", 0) + 10:
            if sig["s_neg"] > CONTROL_LIMIT_H:
                alert = {
                    "signal": signal_name,
                    "type": "accuracy_degradation",
                    "cusum_neg": round(sig["s_neg"], 3),
                    "observations": sig["n"],
                    "reference_accuracy": ref,
                    "detected_at": datetime.now(UTC).isoformat(),
                    "message": (
                        f"CUSUM detected accuracy degradation for {signal_name}: "
                        f"S-={sig['s_neg']:.2f} > h={CONTROL_LIMIT_H} "
                        f"after {sig['n']} observations (ref={ref:.1%})"
                    ),
                }
                sig["last_alert_n"] = sig["n"]
                alerts_list = state.setdefault("alerts", [])
                alerts_list.append(alert)
                state["alerts"] = alerts_list[-100:]
                logger.warning("CUSUM alert: %s", alert["message"])

            elif sig["s_pos"] > CONTROL_LIMIT_H:
                alert = {
                    "signal": signal_name,
                    "type": "accuracy_improvement",
                    "cusum_pos": round(sig["s_pos"], 3),
                    "observations": sig["n"],
                    "reference_accuracy": ref,
                    "detected_at": datetime.now(UTC).isoformat(),
                    "message": (
                        f"CUSUM detected accuracy improvement for {signal_name}: "
                        f"S+={sig['s_pos']:.2f} > h={CONTROL_LIMIT_H} "
                        f"after {sig['n']} observations (ref={ref:.1%})"
                    ),
                }
                sig["last_alert_n"] = sig["n"]
                alerts_list = state.setdefault("alerts", [])
                alerts_list.append(alert)
                state["alerts"] = alerts_list[-100:]
                logger.info("CUSUM improvement: %s", alert["message"])

        _save_state(state)
        return alert


def get_cusum_state() -> dict:
    """Return current CUSUM state for all signals."""
    with _lock:
        return _load_state()


def reset_signal(signal_name: str) -> None:
    """Reset CUSUM counters for a signal."""
    with _lock:
        state = _load_state()
        signals = state.get("signals", {})
        if signal_name in signals:
            ref = signals[signal_name].get("reference_accuracy")
            signals[signal_name] = {
                "s_pos": 0.0,
                "s_neg": 0.0,
                "n": 0,
                "last_alert_n": 0,
                "reference_accuracy": ref,
            }
            _save_state(state)
            logger.info("CUSUM reset for %s", signal_name)
