"""Intraday fishing decision engine -- designed to run inside metals loop.

Pure decision logic. No data fetching, no Avanza execution.
The caller (metals loop) provides state and executes decisions.

Usage::

    engine = FishEngine()
    decision = engine.tick(state)
    if decision['action'] == 'BUY':
        # caller executes via Avanza
        engine.confirm_entry(direction, entry_price_cert, volume, underlying_price)
    elif decision['action'] == 'SELL':
        # caller executes via Avanza
        engine.confirm_exit(pnl)

State dict the engine expects each tick::

    state = {
        'silver_price': float,
        'gold_price': float,
        'gold_5min_change': float,
        'signal_action': str,           # BUY/SELL/HOLD
        'signal_buy_count': int,
        'signal_sell_count': int,
        'rsi': float,
        'mc_p_up': float,
        'metals_action': str,           # same as signal_action (unified)
        'regime': str,
        'news_action': str,
        'econ_action': str,
        'focus_1d_dir': str,
        'focus_1d_prob': float,
        'orb_range': dict or None,      # {high, low, formed}
        'vol_scalar': float,
        'hour_cet': int,
        'minute_cet': int,
        'day_of_week': int,             # 0=Mon
        'velocity': float or None,
        'trade_guard_ok': bool,
        'spread_pct': float,            # current bid/ask spread %
        'news_spike': bool,
        'headline_sentiment': str,      # 'positive'/'negative'/''
        'event_hours': float,           # hours until next econ event
        'high_impact_near': bool,
        # Layer 2 journal context
        'layer2_outlook': str,          # 'bullish'/'bearish'/''
        'layer2_conviction': float,     # 0-1
        'layer2_levels': list,          # [support, resistance]
        'layer2_action': str,           # 'BUY'/'SELL'/'HOLD'
        'layer2_ts': str,              # ISO timestamp
        # Monte Carlo bands
        'mc_bands_1d': dict,            # {'5': float, '25': float, '75': float, '95': float}
        # Chronos forecast
        'chronos_1h_pct': float,
        'chronos_24h_pct': float,
        # Prophecy belief
        'prophecy_target': float,
        'prophecy_conviction': float,
    }
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Instrument IDs (Rule 2+5: only AVA products, prefer higher-priced certs)
# ---------------------------------------------------------------------------
LONG_OB = "1650161"     # BULL SILVER X5 AVA 4 (~7 SEK, 0.14% spread)
SHORT_OB = "2286417"    # BEAR SILVER X5 AVA 12 (~2.5 SEK, 0.39% spread)

# ---------------------------------------------------------------------------
# Trading rules constants
# ---------------------------------------------------------------------------
MIN_ORDER_SEK = 1000        # Rule 1: courtage-free threshold
MIN_EXPECTED_GAIN = 50      # Rule 6: min expected gain
MAX_SPREAD_PCT = 1.0        # Rule 9: skip instruments with >1% spread
COOLDOWN_NORMAL = 300       # Rule 4: 5 min between trades
COOLDOWN_HIGH_CONV = 120    # Rule 4: 2 min for high-conviction exits
MIN_VOTES = 2               # Rule 3: 2+ tactics must agree
LEVERAGE = 5                # Silver X5 certs

# Exit thresholds (backtested)
EXIT_LONG_RSI = 62
EXIT_LONG_MC = 0.35
EXIT_SHORT_RSI = 30
EXIT_SIGNAL_FLIP_MARGIN = 4
EXIT_METALS_DISAGREE_COUNT = 15  # fishing is contrarian — expect disagreement, don't panic
EXIT_TP_PCT = 2.0           # +2% underlying
EXIT_SL_PCT = -3.0          # -3% underlying
MAX_HOLD_NORMAL = 120       # minutes
MAX_HOLD_EVENT = 60         # minutes when event within 24h

# Straddle mode
STRADDLE_CANCEL_HOUR = 18
STRADDLE_CANCEL_MIN = 55
STRADDLE_FLOOR_PCT = 3.0
STRADDLE_CEIL_PCT = 2.0

# Time gating
US_SESSION_START = 14       # 14:00 CET
US_SESSION_END = 17         # 17:00 CET
DEAD_ZONE_START = 10        # 10:00 CET
DEAD_ZONE_END = 14          # 14:00 CET

# Gold-leads-silver
GOLD_LEAD_THRESHOLD = 0.5   # gold must move >0.5% in 5 min
SILVER_LAG_THRESHOLD = 0.2  # silver hasn't moved >0.2% yet

# Temporal patterns
TEMPORAL_MIN_PROBABILITY = 68

# Trade log
TRADE_LOG = "data/fish_trades.jsonl"


class FishEngine:
    """Intraday fishing decision engine -- designed to run inside metals loop.

    Pure decision logic. No data fetching, no Avanza execution.
    The caller (metals loop) provides state and executes decisions.
    """

    def __init__(
        self,
        *,
        temporal_patterns: list[dict] | None = None,
        long_ob: str = LONG_OB,
        short_ob: str = SHORT_OB,
        trade_log_path: str = TRADE_LOG,
        time_func=None,
    ):
        # Position tracking
        self.position: dict | None = None  # {direction, entry_underlying, entry_cert, volume, ob_id, entry_ts}
        self.metals_disagree_count = 0

        # Session P&L
        self.session_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0

        # Mode: 'momentum' or 'straddle'
        self.mode = "momentum"
        self.straddle_floor = 0.0
        self.straddle_ceil = 0.0
        self.straddle_bull_filled = False
        self.straddle_bear_filled = False
        self.consecutive_losses = 0

        # MC history for momentum tactic (need 2 consecutive checks)
        self._mc_history: list[float] = []

        # Cooldown tracking
        self._last_trade_ts = 0.0
        self._cooldown_seconds = 0.0

        # Instrument IDs
        self.long_ob = long_ob
        self.short_ob = short_ob

        # Temporal patterns
        self._temporal_patterns: dict[tuple[int, int], dict] = {}
        if temporal_patterns:
            for p in temporal_patterns:
                key = (p["day"], p["hour_cet"])
                self._temporal_patterns[key] = p

        # ORB state (can be set externally or from state dict)
        self.orb_range_high = 0.0
        self.orb_range_low = 0.0
        self.orb_range_formed = False

        # Trade log path
        self._trade_log_path = trade_log_path

        # Allow injecting time function for testing
        self._time = time_func or time.time

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process one cycle of market state and return a decision.

        Args:
            state: Market state dict (see module docstring for schema).

        Returns:
            Decision dict with keys: action, direction, reason,
            tactics_agreed, size_scalar, exit_reason, confidence,
            instrument_ob, hold_minutes.
        """
        now = self._time()

        # Auto-disable after 21:55 CET
        hour = state.get("hour_cet", 0)
        minute = state.get("minute_cet", 0)
        if hour > 21 or (hour == 21 and minute >= 55):
            if self.position is not None:
                pos = self.position
                hold_minutes = (now - pos["entry_ts"]) / 60
                return {
                    "action": "SELL",
                    "direction": pos["direction"],
                    "reason": "session end 21:55 CET",
                    "tactics_agreed": [],
                    "size_scalar": 0,
                    "exit_reason": "SESSION_END",
                    "confidence": 0,
                    "instrument_ob": pos["ob_id"],
                    "hold_minutes": round(hold_minutes, 1),
                    "volume": pos["volume"],
                }
            return self._hold("market closed")

        # Update ORB from state if provided
        orb = state.get("orb_range")
        if orb and orb.get("formed") and not self.orb_range_formed:
            self.orb_range_high = orb["high"]
            self.orb_range_low = orb["low"]
            self.orb_range_formed = True

        # Update vol_scalar from state
        vol_scalar = state.get("vol_scalar", 1.0)

        # Update MC history
        mc = state.get("mc_p_up", 0.5)
        self._mc_history.append(mc)
        if len(self._mc_history) > 3:
            self._mc_history.pop(0)

        silver_price = state.get("silver_price", 0)
        if silver_price <= 0:
            return self._hold("no silver price")

        # --- If we have a position, evaluate exit ---
        if self.position is not None:
            return self._evaluate_exit(state, now)

        # --- No position: evaluate entry via voting ---
        return self._evaluate_entry(state, now, vol_scalar)

    def confirm_entry(
        self,
        direction: str,
        entry_price_cert: float,
        volume: int,
        underlying_price: float,
    ) -> None:
        """Caller confirms a BUY was executed. Updates internal position."""
        now = self._time()
        ob_id = self.long_ob if direction == "LONG" else self.short_ob
        self.position = {
            "direction": direction,
            "entry_underlying": underlying_price,
            "entry_cert": entry_price_cert,
            "volume": volume,
            "ob_id": ob_id,
            "entry_ts": now,
        }
        self.metals_disagree_count = 0
        self._last_trade_ts = now

        # Log the entry
        self._log_trade(
            "BUY",
            "BULL X5" if direction == "LONG" else "BEAR X5",
            volume,
            entry_price_cert,
            -(entry_price_cert * volume),
        )

        if self.mode == "straddle":
            if direction == "LONG":
                self.straddle_bull_filled = True
            else:
                self.straddle_bear_filled = True

    def confirm_exit(self, pnl: float, exit_cert_price: float = 0) -> None:
        """Caller confirms a SELL was executed. Updates session P&L."""
        if self.position is None:
            return

        self.session_pnl += pnl
        self.trade_count += 1

        if pnl > 0:
            self.win_count += 1
            self.consecutive_losses = 0
        else:
            self.loss_count += 1
            self.consecutive_losses += 1

        # Rule 8: Auto-switch to straddle after 3 consecutive losses
        if self.mode == "momentum" and self.consecutive_losses >= 3:
            self._switch_to_straddle()

        # Log the exit
        direction = self.position["direction"]
        volume = self.position["volume"]
        self._log_trade(
            "SELL",
            "BULL X5" if direction == "LONG" else "BEAR X5",
            volume,
            exit_cert_price,
            exit_cert_price * volume if exit_cert_price else 0,
        )

        self.position = None

    def set_mode(self, mode: str, floor: float = 0, ceil: float = 0) -> None:
        """Explicitly set the trading mode.

        Args:
            mode: 'momentum' or 'straddle'
            floor: straddle floor price (only if mode='straddle')
            ceil: straddle ceiling price (only if mode='straddle')
        """
        self.mode = mode
        if mode == "straddle":
            self.straddle_floor = floor
            self.straddle_ceil = ceil
            self.straddle_bull_filled = False
            self.straddle_bear_filled = False

    def set_orb_range(self, high: float, low: float) -> None:
        """Set the Opening Range Breakout levels."""
        if high > low > 0:
            self.orb_range_high = high
            self.orb_range_low = low
            self.orb_range_formed = True

    def load_temporal_patterns(self, patterns: list[dict]) -> None:
        """Load temporal patterns from data."""
        self._temporal_patterns = {}
        for p in patterns:
            key = (p["day"], p["hour_cet"])
            self._temporal_patterns[key] = p

    def get_session_stats(self) -> dict:
        """Return session statistics."""
        return {
            "session_pnl": round(self.session_pnl, 2),
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": round(self.win_count / self.trade_count, 3)
            if self.trade_count > 0
            else 0,
            "mode": self.mode,
            "consecutive_losses": self.consecutive_losses,
            "has_position": self.position is not None,
        }

    @property
    def has_position(self) -> bool:
        return self.position is not None

    # ------------------------------------------------------------------
    # Exit evaluation
    # ------------------------------------------------------------------

    def _evaluate_exit(self, state: dict, now: float) -> dict[str, Any]:
        """Evaluate whether to exit the current position."""
        pos = self.position
        assert pos is not None

        d = pos["direction"]
        silver_price = state["silver_price"]
        ep = pos["entry_underlying"]
        rsi = state.get("rsi", 50)
        mc = state.get("mc_p_up", 0.5)
        buy_count = state.get("signal_buy_count", 0)
        sell_count = state.get("signal_sell_count", 0)
        metals_action = state.get("metals_action", "HOLD")
        event_hours = state.get("event_hours", 999)
        high_impact = state.get("high_impact_near", False)

        # Underlying move %
        if d == "LONG":
            mv_pct = (silver_price - ep) / ep * 100 if ep > 0 else 0
        else:
            mv_pct = (ep - silver_price) / ep * 100 if ep > 0 else 0

        cert_pct = mv_pct * LEVERAGE
        hold_minutes = (now - pos["entry_ts"]) / 60

        # Max hold: shorten if event proximity
        max_hold = (
            MAX_HOLD_EVENT
            if (event_hours < 24 or high_impact)
            else MAX_HOLD_NORMAL
        )

        # Track metals disagree
        if d == "LONG" and metals_action == "SELL":
            self.metals_disagree_count += 1
        elif d == "SHORT" and metals_action == "BUY":
            self.metals_disagree_count += 1
        else:
            self.metals_disagree_count = 0

        # --- Exit rules (priority order) ---

        # Dynamic TP/SL from Monte Carlo (if available)
        try:
            mc_bands = state.get('mc_bands_1d', {})
            if d == 'LONG' and mc_bands:
                tp_level = float(mc_bands.get('75', 0))
                sl_level = float(mc_bands.get('5', 0))
                if tp_level > 0 and ep > 0:
                    tp_pct = (tp_level - ep) / ep * 100
                    sl_pct = (sl_level - ep) / ep * 100
                else:
                    tp_pct, sl_pct = EXIT_TP_PCT, EXIT_SL_PCT
            elif d == 'SHORT' and mc_bands:
                tp_level = float(mc_bands.get('25', 0))
                sl_level = float(mc_bands.get('95', 0))
                if tp_level > 0 and ep > 0:
                    tp_pct = (ep - tp_level) / ep * 100
                    sl_pct = (ep - sl_level) / ep * 100
                else:
                    tp_pct, sl_pct = EXIT_TP_PCT, EXIT_SL_PCT
            else:
                tp_pct, sl_pct = EXIT_TP_PCT, EXIT_SL_PCT
        except Exception:
            tp_pct, sl_pct = EXIT_TP_PCT, EXIT_SL_PCT

        # Enforce minimum thresholds to cover friction
        tp_pct = max(tp_pct, 1.0)
        sl_pct = min(sl_pct, -1.0)

        # TP: dynamic or +2% underlying
        if mv_pct >= tp_pct:
            mc_tag = '(MC)' if state.get('mc_bands_1d') else ''
            return self._sell(
                f"TP{mc_tag}",
                f"take profit at {mv_pct:+.1f}% ({cert_pct:+.0f}% cert) [tp={tp_pct:.1f}%{mc_tag}]",
                high_conviction=False,
            )

        # SL: dynamic or -3% underlying
        if mv_pct <= sl_pct:
            mc_tag = '(MC)' if state.get('mc_bands_1d') else ''
            return self._sell(
                f"SL{mc_tag}",
                f"stop loss at {mv_pct:+.1f}% ({cert_pct:+.0f}% cert) [sl={sl_pct:.1f}%{mc_tag}]",
                high_conviction=False,
            )

        # LONG exits
        if d == "LONG":
            # Combined RSI>62 + MC<35% (66.7% backtested win rate)
            if rsi > EXIT_LONG_RSI and mc < EXIT_LONG_MC:
                return self._sell(
                    "COMB",
                    f"RSI={rsi:.0f}>62 + MC={mc:.0%}<35% combo exit",
                    high_conviction=True,
                )
            # RSI extreme
            if rsi > 70:
                return self._sell(
                    "RSI",
                    f"RSI={rsi:.0f} overbought",
                    high_conviction=True,
                )
            # Signal flip: 4+ vote margin
            if sell_count > buy_count + EXIT_SIGNAL_FLIP_MARGIN:
                return self._sell(
                    "SELL flip",
                    f"signal flipped to SELL ({sell_count}S vs {buy_count}B)",
                    high_conviction=True,
                )

        # SHORT exits
        else:
            # RSI<30 solo
            if rsi < EXIT_SHORT_RSI:
                return self._sell(
                    "RSI",
                    f"RSI={rsi:.0f} oversold",
                    high_conviction=True,
                )
            # Signal flip: 4+ vote margin
            if buy_count > sell_count + EXIT_SIGNAL_FLIP_MARGIN:
                return self._sell(
                    "BUY flip",
                    f"signal flipped to BUY ({buy_count}B vs {sell_count}S)",
                    high_conviction=True,
                )

        # Metals disagree: N consecutive (but NOT in extreme RSI zones)
        # Fishing is contrarian — when RSI is extreme, disagreement is expected.
        # Selling an oversold LONG or overbought SHORT on metals disagree
        # is exactly the wrong move (learned from 2026-04-07 live test: -590 SEK).
        # Note: the per-direction RSI exits above (RSI>70 for LONG, RSI<30 for SHORT)
        # handle same-direction extremes. This guard covers the contrarian case:
        # LONG held in oversold (RSI<30) or SHORT held in overbought (RSI>70).
        rsi_in_extreme = rsi < 30 or rsi > 70
        if self.metals_disagree_count >= EXIT_METALS_DISAGREE_COUNT and not rsi_in_extreme:
            return self._sell(
                f"MD{self.metals_disagree_count}",
                f"metals disagree {self.metals_disagree_count} consecutive checks (RSI={rsi:.0f})",
                high_conviction=False,
            )

        # Time: max hold
        if hold_minutes >= max_hold:
            return self._sell(
                f"{max_hold}m hold",
                f"max hold time {max_hold}m reached ({hold_minutes:.0f}m held)",
                high_conviction=False,
            )

        # --- Still holding ---
        return {
            "action": "HOLD",
            "direction": d,
            "reason": f"holding {d} {mv_pct:+.1f}% ({cert_pct:+.0f}% cert) {hold_minutes:.0f}m",
            "tactics_agreed": [],
            "size_scalar": 0,
            "exit_reason": None,
            "confidence": 0,
            "instrument_ob": pos["ob_id"],
            "hold_minutes": round(hold_minutes, 1),
            "underlying_move_pct": round(mv_pct, 2),
            "cert_move_pct": round(cert_pct, 1),
        }

    # ------------------------------------------------------------------
    # Entry evaluation (voting system)
    # ------------------------------------------------------------------

    def _evaluate_entry(
        self, state: dict, now: float, vol_scalar: float
    ) -> dict[str, Any]:
        """Evaluate whether to enter a new position via tactic voting."""

        # Check cooldown (Rule 4)
        elapsed_since_trade = now - self._last_trade_ts
        if elapsed_since_trade < self._cooldown_seconds:
            remaining = self._cooldown_seconds - elapsed_since_trade
            return self._hold(f"cooldown {remaining:.0f}s remaining")

        # Check trade guard
        if not state.get("trade_guard_ok", True):
            return self._hold("trade guard blocked")

        # Check spread (Rule 9)
        spread_pct = state.get("spread_pct", 0)
        if spread_pct > MAX_SPREAD_PCT:
            return self._hold(f"spread {spread_pct:.1f}% > {MAX_SPREAD_PCT}% limit")

        silver_price = state["silver_price"]
        hour = state.get("hour_cet", 12)
        minute = state.get("minute_cet", 0)
        day_of_week = state.get("day_of_week", 0)

        # Time gating (Tactic 8)
        is_us_session = US_SESSION_START <= hour < US_SESSION_END
        is_dead_zone = DEAD_ZONE_START <= hour < DEAD_ZONE_END

        # Collect tactic votes
        votes: dict[str, str] = {}  # tactic_name -> 'LONG'/'SHORT'

        # --- Tactic 1: Momentum ---
        if self.mode == "momentum":
            self._vote_momentum(state, votes)

        # --- Tactic 2: Straddle ---
        elif self.mode == "straddle":
            self._vote_straddle(state, votes, hour, minute, silver_price)

        # --- Tactic 3: Gold-leads-silver ---
        self._vote_gold_lead(state, votes, is_dead_zone)

        # --- Tactic 4: ORB breakout ---
        self._vote_orb(silver_price, votes)

        # --- Tactic 5: Temporal pattern ---
        self._vote_temporal(hour, day_of_week, votes)

        # --- Tactic 6: Sentiment velocity ---
        self._vote_sentiment(state, votes)

        # --- Tactic 9: Layer 2 journal vote (weight 2) ---
        try:
            l2_vote = self._vote_layer2(state)
            if l2_vote:
                votes['layer2'] = l2_vote
                # Layer 2 counts as 2 votes -- add a second entry
                votes['layer2_w'] = l2_vote  # duplicate to get weight 2
        except Exception:
            pass

        # --- Count votes (Rule 3: 2+ tactics must agree) ---
        longs = [k for k, v in votes.items() if v == "LONG"]
        shorts = [k for k, v in votes.items() if v == "SHORT"]

        # --- Tactic 7: Vol-targeting (scales size, doesn't vote) ---
        size_scalar = vol_scalar

        # --- Tactic 8: Time gating (boost/penalty on size) ---
        if is_us_session:
            size_scalar *= 1.2  # US session boost
        elif is_dead_zone:
            size_scalar *= 0.7  # dead zone penalty

        size_scalar = max(0.25, min(2.0, size_scalar))

        if len(longs) >= MIN_VOTES and len(longs) > len(shorts):
            confidence = min(1.0, len(longs) / 4.0)
            # Chronos confidence modifier
            try:
                chronos_pct = state.get('chronos_24h_pct', 0)
                if chronos_pct < -0.3:
                    confidence *= 0.7  # Chronos disagrees with LONG
            except Exception:
                pass
            return {
                "action": "BUY",
                "direction": "LONG",
                "reason": f"{len(longs)} tactics agree LONG: {', '.join(longs)}",
                "tactics_agreed": longs,
                "size_scalar": round(size_scalar, 2),
                "exit_reason": None,
                "confidence": round(confidence, 2),
                "instrument_ob": self.long_ob,
                "hold_minutes": 0,
            }

        if len(shorts) >= MIN_VOTES and len(shorts) > len(longs):
            confidence = min(1.0, len(shorts) / 4.0)
            # Chronos confidence modifier
            try:
                chronos_pct = state.get('chronos_24h_pct', 0)
                if chronos_pct > 0.3:
                    confidence *= 0.7  # Chronos disagrees with SHORT
            except Exception:
                pass
            return {
                "action": "BUY",
                "direction": "SHORT",
                "reason": f"{len(shorts)} tactics agree SHORT: {', '.join(shorts)}",
                "tactics_agreed": shorts,
                "size_scalar": round(size_scalar, 2),
                "exit_reason": None,
                "confidence": round(confidence, 2),
                "instrument_ob": self.short_ob,
                "hold_minutes": 0,
            }

        # Conflict or insufficient votes
        if longs and shorts:
            return self._hold(
                f"vote conflict: {', '.join(longs)} LONG vs {', '.join(shorts)} SHORT"
            )

        if longs or shorts:
            singles = longs or shorts
            direction = "LONG" if longs else "SHORT"
            return self._hold(
                f"only 1 tactic ({singles[0]}) votes {direction}, need {MIN_VOTES}+"
            )

        return self._hold("no tactic votes")

    # ------------------------------------------------------------------
    # Tactic implementations
    # ------------------------------------------------------------------

    def _vote_momentum(self, state: dict, votes: dict) -> None:
        """Tactic 1: Both loops agree + MC stable 2 consecutive checks."""
        signal_action = state.get("signal_action", "HOLD")
        metals_action = state.get("metals_action", "HOLD")

        # MC stability: 2 consecutive checks above/below threshold
        mc_stable_bull = (
            len(self._mc_history) >= 2
            and all(x > 0.70 for x in self._mc_history[-2:])
        )
        mc_stable_bear = (
            len(self._mc_history) >= 2
            and all(x < 0.30 for x in self._mc_history[-2:])
        )

        if signal_action == "BUY" and metals_action == "BUY" and mc_stable_bull:
            votes["momentum"] = "LONG"
        elif signal_action == "SELL" and metals_action == "SELL" and mc_stable_bear:
            votes["momentum"] = "SHORT"

    def _vote_straddle(
        self,
        state: dict,
        votes: dict,
        hour: int,
        minute: int,
        silver_price: float,
    ) -> None:
        """Tactic 2: Floor/ceiling hit (straddle mode)."""
        # Don't place new straddle entries after cancel time
        past_cancel = hour > STRADDLE_CANCEL_HOUR or (
            hour == STRADDLE_CANCEL_HOUR and minute >= STRADDLE_CANCEL_MIN
        )
        if past_cancel:
            return

        if self.straddle_floor <= 0 or self.straddle_ceil <= 0:
            return

        if silver_price <= self.straddle_floor and not self.straddle_bull_filled:
            votes["straddle"] = "LONG"
        elif silver_price >= self.straddle_ceil and not self.straddle_bear_filled:
            votes["straddle"] = "SHORT"

    def _vote_gold_lead(
        self, state: dict, votes: dict, is_dead_zone: bool
    ) -> None:
        """Tactic 3: Gold moved >0.5% in 5 min, silver hasn't followed."""
        gold_5min_change = state.get("gold_5min_change", 0)
        # Determine if silver has already followed (approximate from velocity or
        # by comparing gold change to expected lag).
        # Simplified: if gold moved big and we have no velocity showing silver
        # following, treat it as a lead signal.
        if gold_5min_change > GOLD_LEAD_THRESHOLD:
            confidence = min(1.0, gold_5min_change / 1.0)
            if not is_dead_zone or confidence >= 0.7:
                votes["gold_lead"] = "LONG"
        elif gold_5min_change < -GOLD_LEAD_THRESHOLD:
            confidence = min(1.0, abs(gold_5min_change) / 1.0)
            if not is_dead_zone or confidence >= 0.7:
                votes["gold_lead"] = "SHORT"

    def _vote_orb(self, silver_price: float, votes: dict) -> None:
        """Tactic 4: Price breaks morning range."""
        if not self.orb_range_formed:
            return
        if self.orb_range_high <= self.orb_range_low:
            return
        if silver_price > self.orb_range_high:
            votes["orb"] = "LONG"
        elif silver_price < self.orb_range_low:
            votes["orb"] = "SHORT"

    def _vote_temporal(self, hour: int, day_of_week: int, votes: dict) -> None:
        """Tactic 5: Recurring time-of-day patterns."""
        key = (day_of_week, hour)
        p = self._temporal_patterns.get(key)
        if p and p.get("probability", 0) >= TEMPORAL_MIN_PROBABILITY:
            direction = p["direction"]
            d_map = {"BULL": "LONG", "BEAR": "SHORT"}
            mapped = d_map.get(direction, direction)
            if mapped in ("LONG", "SHORT"):
                votes["temporal"] = mapped

    def _vote_sentiment(self, state: dict, votes: dict) -> None:
        """Tactic 6: News spike with directional sentiment."""
        if not state.get("news_spike"):
            return
        sentiment = state.get("headline_sentiment", "")
        if sentiment == "negative":
            votes["sentiment"] = "SHORT"
        elif sentiment == "positive":
            votes["sentiment"] = "LONG"

    def _vote_layer2(self, state: dict) -> str | None:
        """Tactic 9: Layer 2 journal reasoning -- highest trust vote (weight 2)."""
        outlook = state.get('layer2_outlook', '')
        conviction = state.get('layer2_conviction', 0)
        ts = state.get('layer2_ts', '')

        if not outlook or conviction < 0.4:
            return None

        # Check staleness -- ignore if >4h old
        if ts:
            try:
                from datetime import datetime as _dt, timezone, timedelta
                entry_time = _dt.fromisoformat(ts.replace('Z', '+00:00'))
                age_hours = (_dt.now(timezone.utc) - entry_time).total_seconds() / 3600
                if age_hours > 4:
                    return None
            except Exception:
                pass

        if outlook == 'bullish':
            return 'LONG'
        elif outlook == 'bearish':
            return 'SHORT'
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sell(
        self, exit_code: str, reason: str, *, high_conviction: bool
    ) -> dict[str, Any]:
        """Build a SELL decision and set appropriate cooldown."""
        pos = self.position
        assert pos is not None

        now = self._time()
        if high_conviction:
            self._cooldown_seconds = COOLDOWN_HIGH_CONV
        else:
            self._cooldown_seconds = COOLDOWN_NORMAL
        self._last_trade_ts = now

        hold_minutes = (now - pos["entry_ts"]) / 60

        return {
            "action": "SELL",
            "direction": pos["direction"],
            "reason": reason,
            "tactics_agreed": [],
            "size_scalar": 0,
            "exit_reason": exit_code,
            "confidence": 0,
            "instrument_ob": pos["ob_id"],
            "hold_minutes": round(hold_minutes, 1),
            "volume": pos["volume"],
        }

    def _hold(self, reason: str) -> dict[str, Any]:
        """Build a HOLD decision."""
        return {
            "action": "HOLD",
            "direction": self.position["direction"]
            if self.position
            else "",
            "reason": reason,
            "tactics_agreed": [],
            "size_scalar": 0,
            "exit_reason": None,
            "confidence": 0,
            "instrument_ob": "",
            "hold_minutes": 0,
        }

    def _switch_to_straddle(self) -> None:
        """Auto-switch from momentum to straddle after 3 consecutive losses."""
        logger.info("3 consecutive losses -- switching to STRADDLE mode")
        self.mode = "straddle"
        self.straddle_bull_filled = False
        self.straddle_bear_filled = False
        # Note: caller must set floor/ceil via set_mode() or provide in state

    def _log_trade(
        self,
        action: str,
        instrument: str,
        units: int,
        price: float,
        amount: float,
        tactic: str = "",
    ) -> None:
        """Rule 7: Log every trade with exact amounts."""
        try:
            from portfolio.file_utils import atomic_append_jsonl

            entry = {
                "ts": datetime.datetime.now(datetime.UTC).isoformat(),
                "action": action,
                "instrument": instrument,
                "units": units,
                "price_sek": round(price, 2),
                "amount_sek": round(amount, 2),
                "tactic": tactic,
                "session_pnl": round(self.session_pnl, 2),
                "mode": self.mode,
            }
            atomic_append_jsonl(self._trade_log_path, entry)
        except Exception:
            logger.debug("Failed to log trade to %s", self._trade_log_path, exc_info=True)

    # ------------------------------------------------------------------
    # Serialization (for persisting engine state across restarts)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize engine state to a dict."""
        return {
            "position": self.position,
            "metals_disagree_count": self.metals_disagree_count,
            "session_pnl": self.session_pnl,
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "mode": self.mode,
            "straddle_floor": self.straddle_floor,
            "straddle_ceil": self.straddle_ceil,
            "straddle_bull_filled": self.straddle_bull_filled,
            "straddle_bear_filled": self.straddle_bear_filled,
            "consecutive_losses": self.consecutive_losses,
            "mc_history": self._mc_history,
            "last_trade_ts": self._last_trade_ts,
            "cooldown_seconds": self._cooldown_seconds,
            "orb_range_high": self.orb_range_high,
            "orb_range_low": self.orb_range_low,
            "orb_range_formed": self.orb_range_formed,
        }

    def from_dict(self, d: dict) -> None:
        """Restore engine state from a dict."""
        self.position = d.get("position")
        self.metals_disagree_count = d.get("metals_disagree_count", 0)
        self.session_pnl = d.get("session_pnl", 0)
        self.trade_count = d.get("trade_count", 0)
        self.win_count = d.get("win_count", 0)
        self.loss_count = d.get("loss_count", 0)
        self.mode = d.get("mode", "momentum")
        self.straddle_floor = d.get("straddle_floor", 0)
        self.straddle_ceil = d.get("straddle_ceil", 0)
        self.straddle_bull_filled = d.get("straddle_bull_filled", False)
        self.straddle_bear_filled = d.get("straddle_bear_filled", False)
        self.consecutive_losses = d.get("consecutive_losses", 0)
        self._mc_history = d.get("mc_history", [])
        self._last_trade_ts = d.get("last_trade_ts", 0)
        self._cooldown_seconds = d.get("cooldown_seconds", 0)
        self.orb_range_high = d.get("orb_range_high", 0)
        self.orb_range_low = d.get("orb_range_low", 0)
        self.orb_range_formed = d.get("orb_range_formed", False)
