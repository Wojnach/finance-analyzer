"""
Signal generation and evaluation for the Finance Analyzer.
Trigger + guards entry model: any trigger fires a candidate, all guards must pass.
"""

from typing import Dict, Any
from collections import defaultdict


def evaluate_buy_conditions(
    triggers: Dict[str, bool],
    guards: Dict[str, bool],
) -> Dict[str, Any]:
    """
    Evaluate conditions for a buy signal using trigger + guards model.

    Args:
        triggers: Dict of trigger names to booleans (OR'd — any one fires)
        guards: Dict of guard names to booleans (AND'd — all must be true)

    Returns:
        Dict with should_enter, active_triggers, and failed_guards
    """
    active = [k for k, v in triggers.items() if v]
    failed = [k for k, v in guards.items() if not v]

    return {
        "should_enter": len(active) > 0 and len(failed) == 0,
        "active_triggers": active,
        "failed_guards": failed,
    }


def evaluate_sell_conditions(
    conditions: Dict[str, Any],
    rsi_overbought: float = 70,
    base_confidence: float = 0.25,
) -> Dict[str, Any]:
    """
    Evaluate conditions for a sell signal.

    Args:
        conditions: Dict containing indicator values:
            - rsi: Current RSI value
            - macd_bearish_cross: Boolean for MACD bearish crossover
            - ema_bearish_cross: Boolean for EMA bearish crossover
            - volume_spike: Boolean for volume spike
        rsi_overbought: RSI threshold for overbought (default 70)
        base_confidence: Base confidence per signal (default 0.25)

    Returns:
        Dict with signal flags and confidence score
    """
    result = {
        "rsi_signal": False,
        "macd_signal": False,
        "ema_signal": False,
        "volume_signal": False,
        "confidence": 0.0,
    }

    signal_count = 0

    if conditions.get("rsi", 50) > rsi_overbought:
        result["rsi_signal"] = True
        signal_count += 1

    if conditions.get("macd_bearish_cross", False):
        result["macd_signal"] = True
        signal_count += 1

    if conditions.get("ema_bearish_cross", False):
        result["ema_signal"] = True
        signal_count += 1

    if conditions.get("volume_spike", False):
        result["volume_signal"] = True
        signal_count += 0.5

    result["confidence"] = min(signal_count * base_confidence, 1.0)

    return result


def should_execute_trade(signal: Dict[str, Any], min_confidence: float = 0.5) -> bool:
    """
    Determine if a trade should be executed.

    For trigger+guard signals: checks should_enter flag.
    For confidence-based signals (sell): checks confidence threshold.
    """
    if "should_enter" in signal:
        return signal["should_enter"]
    return signal.get("confidence", 0) >= min_confidence


class SignalTracker:
    """
    Tracks signals to prevent duplicate alerts and manage cooldowns.
    """

    def __init__(self, cooldown_periods: int = 3):
        self.cooldown_periods = cooldown_periods
        self.current_period = 0
        self._signals: Dict[str, Dict[str, int]] = defaultdict(dict)

    def can_signal(self, pair: str, signal_type: str) -> bool:
        if pair not in self._signals:
            return True
        if signal_type not in self._signals[pair]:
            return True
        last_signal = self._signals[pair][signal_type]
        periods_since = self.current_period - last_signal
        return periods_since >= self.cooldown_periods

    def record_signal(self, pair: str, signal_type: str) -> None:
        self._signals[pair][signal_type] = self.current_period

    def advance_period(self) -> None:
        self.current_period += 1

    def reset(self) -> None:
        self._signals.clear()
        self.current_period = 0
