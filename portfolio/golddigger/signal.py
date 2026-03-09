"""Composite signal engine for GoldDigger.

Computes 30-second log-returns, rolling z-scores (60-min window), and a
weighted composite score S_t that drives entry/exit decisions.

S_t = w_gold * z_gold - w_fx * z_fx - w_yield * z_yield

Positive S = gold up + USD weak + yields down = bullish for BULL GULD certificate.
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from portfolio.golddigger.data_provider import MarketSnapshot

logger = logging.getLogger("portfolio.golddigger.signal")

EPSILON = 1e-9


@dataclass
class SignalState:
    """Current state of the composite signal engine."""
    composite_s: float = 0.0
    z_gold: float = 0.0
    z_fx: float = 0.0
    z_yield: float = 0.0
    r_gold: float = 0.0
    r_fx: float = 0.0
    delta_y_bps: float = 0.0
    confirm_count: int = 0     # consecutive polls above theta_in
    window_size: int = 0       # current number of samples in rolling window
    valid: bool = False        # True when enough data for z-scores


def _log_return(current: float, previous: float) -> float:
    """Compute log return ln(current / previous). Returns 0 if invalid."""
    if current <= 0 or previous <= 0:
        return 0.0
    return math.log(current / previous)


def _zscore(value: float, window: deque, min_samples: int = 10) -> float:
    """Compute z-score of value relative to rolling window.

    Returns 0.0 if fewer than min_samples available.
    """
    if len(window) < min_samples:
        return 0.0
    mean = sum(window) / len(window)
    variance = sum((x - mean) ** 2 for x in window) / len(window)
    std = math.sqrt(variance) + EPSILON
    return (value - mean) / std


class CompositeSignal:
    """Rolling z-score composite signal for gold intraday trading.

    Maintains deques of 30s returns for gold, FX, and yield. Computes
    z-scores over a configurable rolling window (default 120 = 60 min).
    """

    def __init__(
        self,
        window_n: int = 120,
        min_window: int = 10,
        w_gold: float = 0.50,
        w_fx: float = 0.30,
        w_yield: float = 0.20,
        theta_in: float = 1.0,
        theta_out: float = 0.2,
        confirm_polls: int = 2,
    ):
        self.window_n = window_n
        self.min_window = min_window
        self.w_gold = w_gold
        self.w_fx = w_fx
        self.w_yield = w_yield
        self.theta_in = theta_in
        self.theta_out = theta_out
        self.confirm_polls = confirm_polls

        # Rolling return windows
        self._gold_returns: deque = deque(maxlen=window_n)
        self._fx_returns: deque = deque(maxlen=window_n)
        self._yield_changes: deque = deque(maxlen=window_n)

        # Previous snapshot values for computing returns
        self._prev_gold: Optional[float] = None
        self._prev_fx: Optional[float] = None
        self._prev_yield: Optional[float] = None

        # Confirmation counter
        self._confirm_count: int = 0

    def reset(self):
        """Reset all state (call on new trading day)."""
        self._gold_returns.clear()
        self._fx_returns.clear()
        self._yield_changes.clear()
        self._prev_gold = None
        self._prev_fx = None
        self._prev_yield = None
        self._confirm_count = 0

    def update(self, snap: MarketSnapshot) -> SignalState:
        """Process a new market snapshot and return the current signal state.

        Call this every poll_seconds (30s). The first call per day establishes
        the baseline; returns become valid after min_window samples.
        """
        state = SignalState()

        # Need at least a previous value to compute returns
        if self._prev_gold is None:
            self._prev_gold = snap.gold
            self._prev_fx = snap.usdsek
            self._prev_yield = snap.us10y
            state.window_size = 0
            return state

        # Compute returns
        r_gold = _log_return(snap.gold, self._prev_gold)
        r_fx = _log_return(snap.usdsek, self._prev_fx)
        delta_y_bps = 10000.0 * (snap.us10y - self._prev_yield) if self._prev_yield else 0.0

        # Append to rolling windows
        self._gold_returns.append(r_gold)
        self._fx_returns.append(r_fx)
        self._yield_changes.append(delta_y_bps)

        # Update previous values
        self._prev_gold = snap.gold
        self._prev_fx = snap.usdsek
        self._prev_yield = snap.us10y

        # Compute z-scores
        z_gold = _zscore(r_gold, self._gold_returns, self.min_window)
        z_fx = _zscore(r_fx, self._fx_returns, self.min_window)
        z_yield = _zscore(delta_y_bps, self._yield_changes, self.min_window)

        # Composite score: gold UP, USD DOWN, yields DOWN = bullish
        s = self.w_gold * z_gold - self.w_fx * z_fx - self.w_yield * z_yield

        # Track confirmation count
        valid = len(self._gold_returns) >= self.min_window
        if valid and s >= self.theta_in and z_gold > 0:
            self._confirm_count += 1
        else:
            self._confirm_count = 0

        state.composite_s = s
        state.z_gold = z_gold
        state.z_fx = z_fx
        state.z_yield = z_yield
        state.r_gold = r_gold
        state.r_fx = r_fx
        state.delta_y_bps = delta_y_bps
        state.confirm_count = self._confirm_count
        state.window_size = len(self._gold_returns)
        state.valid = valid

        return state

    def should_enter(self, state: SignalState, spread_pct: Optional[float] = None, spread_max: float = 0.02) -> bool:
        """Check if all entry conditions are met.

        Requires:
        1. Signal valid (enough data in window)
        2. S_t >= theta_in for confirm_polls consecutive polls
        3. z_gold > 0 (gold momentum confirmation)
        4. Spread within limits (if certificate quote available)
        """
        if not state.valid:
            return False
        if state.confirm_count < self.confirm_polls:
            return False
        if state.z_gold <= 0:
            return False
        if spread_pct is not None and spread_pct > spread_max:
            logger.info("Entry blocked: spread %.2f%% > max %.2f%%", spread_pct * 100, spread_max * 100)
            return False
        return True

    def should_exit(self, state: SignalState) -> bool:
        """Check if composite signal has decayed below exit threshold."""
        if not state.valid:
            return False
        return state.composite_s <= self.theta_out
