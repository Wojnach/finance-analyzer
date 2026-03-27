"""Configuration for 3-hour prediction horizon.

Research-backed constants for optimizing signal accuracy at 3h:
- Slow signals (trend, fibonacci, macro_regime) are noise at 3h (38-45% accuracy)
- Peak trading hours (10-17 UTC) have 43-45% accuracy; quiet hours (20-01 UTC) have 52-55%
- Confidence above 75% at 3h is anti-correlated with accuracy (90%+ bucket = 28% actual)
"""

# Signals that need daily+ data and degrade at 3h.
# trend: 38.8% at 3h (needs SMA200 = 33+ days context)
# fibonacci: 43.3% at 3h (swing detection needs broad history)
# macro_regime: 45.1% at 3h (200-SMA regime is daily-scale)
SLOW_SIGNALS_3H = frozenset({"trend", "fibonacci", "macro_regime"})

# 3h confidence cap. The 90%+ confidence bucket has 28.1% actual accuracy.
# The 70-80% band is the best-performing at 58.9%. Cap at 75%.
CONFIDENCE_CAP_3H = 0.75

# Time-of-day scaling for 3h predictions (UTC hours).
# Based on measured consensus accuracy by hour.
_TOD_SCALE = {
    # Quiet hours (20:00-01:00 UTC) — 52-55% accuracy
    20: 1.10, 21: 1.10, 22: 1.08, 23: 1.08, 0: 1.10,
    # Asian session (1-6 UTC) — 48-50%, slightly below baseline
    1: 0.97, 2: 0.95, 3: 0.95, 4: 0.95, 5: 0.95, 6: 0.97,
    # Pre-EU (7-9 UTC) — neutral
    7: 1.0, 8: 1.0, 9: 1.0,
    # Peak noise (10-17 UTC) — 43-45% accuracy
    10: 0.88, 11: 0.88, 12: 0.85, 13: 0.88, 14: 0.88,
    15: 0.88, 16: 0.85, 17: 0.85,
    # Transition (18-19 UTC) — near neutral
    18: 1.0, 19: 1.0,
}


def is_slow_signal_3h(signal_name: str) -> bool:
    """Check if a signal should be disabled for 3h horizon predictions."""
    return signal_name in SLOW_SIGNALS_3H


def time_of_day_scale_3h(hour: int) -> float:
    """Return confidence scaling factor for 3h predictions at given UTC hour."""
    return float(_TOD_SCALE.get(hour % 24, 1.0))
