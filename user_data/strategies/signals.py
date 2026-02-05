"""
Signal generation and evaluation for the Finance Analyzer.
Combines multiple indicators into buy/sell signals with confidence scores.
"""

from typing import Dict, Any
from collections import defaultdict


def evaluate_buy_conditions(
    conditions: Dict[str, Any], rsi_oversold: float = 30, base_confidence: float = 0.25
) -> Dict[str, Any]:
    """
    Evaluate conditions for a buy signal.

    Args:
        conditions: Dict containing indicator values:
            - rsi: Current RSI value
            - macd_bullish_cross: Boolean for MACD bullish crossover
            - ema_bullish_cross: Boolean for EMA bullish crossover
            - volume_spike: Boolean for volume spike
        rsi_oversold: RSI threshold for oversold (default 30)
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

    # RSI oversold check
    if conditions.get("rsi", 50) < rsi_oversold:
        result["rsi_signal"] = True
        signal_count += 1

    # MACD bullish crossover
    if conditions.get("macd_bullish_cross", False):
        result["macd_signal"] = True
        signal_count += 1

    # EMA bullish crossover
    if conditions.get("ema_bullish_cross", False):
        result["ema_signal"] = True
        signal_count += 1

    # Volume spike (confirmation, adds extra confidence)
    if conditions.get("volume_spike", False):
        result["volume_signal"] = True
        signal_count += 0.5  # Volume is confirmation, half weight

    # Calculate confidence
    result["confidence"] = min(signal_count * base_confidence, 1.0)

    return result


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

    # RSI overbought check
    if conditions.get("rsi", 50) > rsi_overbought:
        result["rsi_signal"] = True
        signal_count += 1

    # MACD bearish crossover
    if conditions.get("macd_bearish_cross", False):
        result["macd_signal"] = True
        signal_count += 1

    # EMA bearish crossover
    if conditions.get("ema_bearish_cross", False):
        result["ema_signal"] = True
        signal_count += 1

    # Volume spike (confirmation)
    if conditions.get("volume_spike", False):
        result["volume_signal"] = True
        signal_count += 0.5

    result["confidence"] = min(signal_count * base_confidence, 1.0)

    return result


def should_execute_trade(signal: Dict[str, Any], min_confidence: float = 0.5) -> bool:
    """
    Determine if a trade should be executed based on signal confidence.

    Args:
        signal: Signal dict from evaluate_buy/sell_conditions
        min_confidence: Minimum confidence threshold (default 0.5)

    Returns:
        Boolean indicating whether to execute the trade
    """
    return signal.get("confidence", 0) >= min_confidence


class SignalTracker:
    """
    Tracks signals to prevent duplicate alerts and manage cooldowns.
    """

    def __init__(self, cooldown_periods: int = 3):
        """
        Initialize signal tracker.

        Args:
            cooldown_periods: Number of periods to wait before allowing same signal
        """
        self.cooldown_periods = cooldown_periods
        self.current_period = 0
        # Structure: {pair: {signal_type: last_signal_period}}
        self._signals: Dict[str, Dict[str, int]] = defaultdict(dict)

    def can_signal(self, pair: str, signal_type: str) -> bool:
        """
        Check if a signal can be generated (not in cooldown).

        Args:
            pair: Trading pair (e.g., 'BTC/USDT')
            signal_type: Type of signal ('buy' or 'sell')

        Returns:
            Boolean indicating if signal is allowed
        """
        if pair not in self._signals:
            return True

        if signal_type not in self._signals[pair]:
            return True

        last_signal = self._signals[pair][signal_type]
        periods_since = self.current_period - last_signal

        return periods_since >= self.cooldown_periods

    def record_signal(self, pair: str, signal_type: str) -> None:
        """
        Record that a signal was generated.

        Args:
            pair: Trading pair
            signal_type: Type of signal
        """
        self._signals[pair][signal_type] = self.current_period

    def advance_period(self) -> None:
        """Advance the internal period counter (called each candle)."""
        self.current_period += 1

    def reset(self) -> None:
        """Reset all tracking state."""
        self._signals.clear()
        self.current_period = 0
