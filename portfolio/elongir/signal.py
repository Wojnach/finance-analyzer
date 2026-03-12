"""Dip/reversal detection state machine for the Elongir bot.

States: SCANNING -> DIP_DETECTED -> CONFIRMING_BUY -> (BUY triggered)
        IN_POSITION -> TRAILING -> (SELL triggered)

The DipDetector looks for oversold silver dips and waits for confirmation
before triggering a buy. The ReversalDetector monitors open positions for
exit conditions.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

from portfolio.elongir.indicators import IndicatorSet
from portfolio.elongir.config import ElongirConfig

logger = logging.getLogger("portfolio.elongir.signal")


# ---------------------------------------------------------------------------
# DipDetector -- scans for dip entries
# ---------------------------------------------------------------------------

class DipDetector:
    """State machine for detecting silver dips and confirming buy entries.

    Flow:
        SCANNING -- no dip detected
        DIP_DETECTED -- dip criteria met (RSI oversold + BB lower + drop from high)
        CONFIRMING_BUY -- MACD improving for N consecutive polls + RSI turning up
        -> returns "BUY" when RSI crosses above recovery threshold
    """

    def __init__(self, config: ElongirConfig):
        self.cfg = config
        self._state = "SCANNING"
        self._macd_improving_count = 0
        self._prev_macd_hist: Optional[float] = None
        self._prev_rsi: Optional[float] = None

    @property
    def state(self) -> str:
        return self._state

    def reset(self) -> None:
        """Reset to SCANNING state."""
        self._state = "SCANNING"
        self._macd_improving_count = 0
        self._prev_macd_hist = None
        self._prev_rsi = None

    def update(
        self,
        indicators: IndicatorSet,
        silver_price: float,
    ) -> Optional[str]:
        """Process one poll cycle. Returns "BUY" if entry triggered, else None.

        Dip criteria (all must be true):
            1. 5m RSI < oversold threshold (30)
            2. Price below 5m BB lower OR 15m BB lower
            3. Drop >= min_dip_pct from 1h high

        Confirmation criteria (after dip detected):
            1. MACD histogram improving for N consecutive polls
            2. RSI starting to turn up (current > previous)

        Buy trigger:
            RSI crosses back above recovery threshold (35)
        """
        tf5 = indicators.tf_5m
        tf15 = indicators.tf_15m

        rsi_5m = tf5.rsi
        macd_hist_5m = tf5.macd_histogram
        bb_lower_5m = tf5.bb_lower
        bb_lower_15m = tf15.bb_lower
        high_1h = tf5.high_1h or tf15.high_1h

        # --- SCANNING: look for dip ---
        if self._state == "SCANNING":
            if self._check_dip_criteria(
                silver_price, rsi_5m, bb_lower_5m, bb_lower_15m, high_1h
            ):
                self._state = "DIP_DETECTED"
                self._macd_improving_count = 0
                logger.info(
                    "DIP_DETECTED: RSI=%.1f, price=%.2f, bb_lower_5m=%.2f",
                    rsi_5m or 0, silver_price, bb_lower_5m or 0,
                )
            self._update_prev(rsi_5m, macd_hist_5m)
            return None

        # --- DIP_DETECTED: wait for MACD to start improving ---
        if self._state == "DIP_DETECTED":
            # Check if dip conditions still hold (or we're recovering)
            improving = self._check_macd_improving(macd_hist_5m)
            rsi_turning = self._check_rsi_turning_up(rsi_5m)

            if improving:
                self._macd_improving_count += 1
            else:
                self._macd_improving_count = 0

            if (self._macd_improving_count >= self.cfg.macd_improving_checks
                    and rsi_turning):
                self._state = "CONFIRMING_BUY"
                logger.info(
                    "CONFIRMING_BUY: MACD improving %dx, RSI turning up",
                    self._macd_improving_count,
                )

            # Reset if price recovered without MACD confirmation
            if rsi_5m is not None and rsi_5m > self.cfg.rsi_overbought:
                logger.info("DIP fizzled: RSI recovered to %.1f without MACD confirmation", rsi_5m)
                self._state = "SCANNING"

            self._update_prev(rsi_5m, macd_hist_5m)
            return None

        # --- CONFIRMING_BUY: wait for RSI recovery above threshold ---
        if self._state == "CONFIRMING_BUY":
            if rsi_5m is not None and rsi_5m > self.cfg.rsi_recovery:
                logger.info("BUY TRIGGER: RSI %.1f crossed above %.1f", rsi_5m, self.cfg.rsi_recovery)
                self._state = "SCANNING"  # Reset after triggering
                self._update_prev(rsi_5m, macd_hist_5m)
                return "BUY"

            # Timeout: if RSI keeps dropping instead of recovering, go back
            if rsi_5m is not None and rsi_5m < self.cfg.rsi_oversold - 10:
                logger.info("CONFIRMING_BUY failed: RSI dropped further to %.1f", rsi_5m)
                self._state = "DIP_DETECTED"  # back to waiting

            self._update_prev(rsi_5m, macd_hist_5m)
            return None

        self._update_prev(rsi_5m, macd_hist_5m)
        return None

    def _check_dip_criteria(
        self,
        silver_price: float,
        rsi_5m: Optional[float],
        bb_lower_5m: Optional[float],
        bb_lower_15m: Optional[float],
        high_1h: Optional[float],
    ) -> bool:
        """Check all dip detection criteria."""
        # 1. RSI oversold
        if rsi_5m is None or rsi_5m >= self.cfg.rsi_oversold:
            return False

        # 2. Price below BB lower (5m or 15m)
        below_bb = False
        if bb_lower_5m is not None and silver_price < bb_lower_5m:
            below_bb = True
        if bb_lower_15m is not None and silver_price < bb_lower_15m:
            below_bb = True
        if not below_bb:
            return False

        # 3. Drop from 1h high >= min_dip_pct
        if high_1h is not None and high_1h > 0:
            drop_pct = (high_1h - silver_price) / high_1h * 100.0
            if drop_pct < self.cfg.min_dip_pct:
                return False
        else:
            # No high data -- cannot confirm dip magnitude
            return False

        return True

    def _check_macd_improving(self, macd_hist: Optional[float]) -> bool:
        """Check if MACD histogram is improving (less negative or more positive)."""
        if macd_hist is None or self._prev_macd_hist is None:
            return False
        return macd_hist > self._prev_macd_hist

    def _check_rsi_turning_up(self, rsi: Optional[float]) -> bool:
        """Check if RSI is turning up (current > previous)."""
        if rsi is None or self._prev_rsi is None:
            return False
        return rsi > self._prev_rsi

    def _update_prev(self, rsi: Optional[float], macd_hist: Optional[float]) -> None:
        """Update previous values for next comparison."""
        if rsi is not None:
            self._prev_rsi = rsi
        if macd_hist is not None:
            self._prev_macd_hist = macd_hist


# ---------------------------------------------------------------------------
# ReversalDetector -- monitors open positions for exit conditions
# ---------------------------------------------------------------------------

class ReversalDetector:
    """Monitors an open position for exit conditions.

    Exit reasons:
        SELL_SIGNAL -- 5m RSI overbought + MACD histogram declining
        STOP -- hard stop: silver dropped 2% below entry
        TRAILING_STOP -- trailing: price dropped 0.7% from peak (after 1.5% gain)
        TAKE_PROFIT -- price reached 2% above entry
        TIME_STOP -- held longer than max_hold_hours
    """

    def __init__(self, config: ElongirConfig):
        self.cfg = config
        self._prev_macd_hist: Optional[float] = None

    def update(
        self,
        indicators: IndicatorSet,
        silver_price: float,
        entry_silver_usd: float,
        entry_time_iso: str,
        trailing_peak_usd: float,
        trailing_active: bool,
    ) -> Optional[str]:
        """Check exit conditions. Returns exit reason string or None.

        Returns one of: "SELL_SIGNAL", "STOP", "TRAILING_STOP",
                        "TAKE_PROFIT", "TIME_STOP", or None.
        """
        gain_pct = (silver_price - entry_silver_usd) / entry_silver_usd * 100.0

        # --- Hard stop ---
        if gain_pct <= -self.cfg.stop_loss_pct:
            return "STOP"

        # --- Take profit ---
        if gain_pct >= self.cfg.take_profit_pct:
            return "TAKE_PROFIT"

        # --- Time stop ---
        try:
            entry_dt = datetime.fromisoformat(entry_time_iso)
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            elapsed_hours = (
                datetime.now(timezone.utc) - entry_dt
            ).total_seconds() / 3600.0
            if elapsed_hours >= self.cfg.max_hold_hours:
                return "TIME_STOP"
        except (ValueError, TypeError):
            pass

        # --- Trailing stop ---
        if trailing_active:
            drop_from_peak = (trailing_peak_usd - silver_price) / trailing_peak_usd * 100.0
            if drop_from_peak >= self.cfg.trailing_distance_pct:
                return "TRAILING_STOP"

        # --- Signal-based sell ---
        tf5 = indicators.tf_5m
        rsi_5m = tf5.rsi
        macd_hist = tf5.macd_histogram

        macd_declining = False
        if macd_hist is not None and self._prev_macd_hist is not None:
            macd_declining = macd_hist < self._prev_macd_hist

        if macd_hist is not None:
            self._prev_macd_hist = macd_hist

        if (rsi_5m is not None
                and rsi_5m > self.cfg.rsi_overbought
                and macd_declining):
            return "SELL_SIGNAL"

        return None

    def should_activate_trailing(
        self,
        silver_price: float,
        entry_silver_usd: float,
    ) -> bool:
        """Check if trailing stop should be activated."""
        gain_pct = (silver_price - entry_silver_usd) / entry_silver_usd * 100.0
        return gain_pct >= self.cfg.trailing_start_pct
