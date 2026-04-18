"""momentum_rider — ride MSTR trends LONG using MSTR-weighted signal score.

v1 LONG-only. Entry when the MSTR-weighted BUY score crosses the configured
threshold with a sane RSI and cooldown cleared. Exit on signal flip,
trailing stop, hard stop, or EOD flatten.

The weighted score uses `config.MSTR_SIGNAL_WEIGHTS` — the signals that
have historical >60% accuracy on MSTR (ministral, econ_calendar, calendar,
qwen3, volume_flow) dominate the decision; low-accuracy signals (trend,
volatility_sig, macro_regime) contribute zero.

This is the v1 reference strategy. Future strategies (mean_reversion,
earnings_play, etc.) drop alongside as separate files without touching
the loop orchestrator.
"""

from __future__ import annotations

import datetime
import logging

from portfolio.mstr_loop import config
from portfolio.mstr_loop.data_provider import MstrBundle
from portfolio.mstr_loop.state import BotState
from portfolio.mstr_loop.strategies.base import Decision

logger = logging.getLogger(__name__)

STRATEGY_KEY = "momentum_rider"


class MomentumRider:
    """Trend-riding LONG strategy on MSTR via BULL MSTR X5 SG4."""

    key: str = STRATEGY_KEY
    enabled: bool = True

    def step(self, bundle: MstrBundle, state: BotState) -> Decision | None:
        """Main decision logic. None = no action this cycle."""
        if not bundle.is_usable():
            return None

        existing = state.get_position(self.key)

        if existing is not None:
            return self._evaluate_exit(bundle, existing)
        return self._evaluate_entry(bundle, state)

    # -----------------------------------------------------------------
    # Entry
    # -----------------------------------------------------------------
    def _evaluate_entry(self, bundle: MstrBundle, state: BotState) -> Decision | None:
        # Cross-cutting risk gates first — drawdown halts, earnings blackout,
        # BTC regime. Centralised in portfolio.mstr_loop.risk so every
        # strategy gets the same macro/portfolio-level refusals for free.
        from portfolio.mstr_loop import risk as risk_mod
        halted, reason = risk_mod.any_entry_halt_active(state, bundle, direction="LONG")
        if halted:
            logger.debug("momentum_rider: entry halted — %s", reason)
            return None

        # Cooldown after last exit
        if not state.cooldown_elapsed(self.key, config.MOMENTUM_RIDER_COOLDOWN_MINUTES):
            return None

        # RSI zone — fade blow-offs, skip already-weak entries.
        # We DO want to enter during a confirmed trend (RSI 50-78 is the
        # sweet spot) but we DO NOT want to chase into RSI 85+ tops.
        if bundle.rsi < config.MOMENTUM_RIDER_RSI_MIN:
            return None
        if bundle.rsi > config.MOMENTUM_RIDER_RSI_MAX:
            return None

        # Weighted-score threshold using MSTR-specific reliability weights.
        # This is the core of the "different strategies weight accurate
        # signals" architecture: momentum_rider fires when the signals
        # that actually work on MSTR agree.
        if bundle.weighted_score_long < config.MOMENTUM_RIDER_BUY_THRESHOLD:
            return None

        # Don't enter on a stale signal block (MSTR closed hours).
        if bundle.stale:
            return None

        rationale = (
            f"weighted_long={bundle.weighted_score_long:.2f} ≥ "
            f"{config.MOMENTUM_RIDER_BUY_THRESHOLD}, "
            f"rsi={bundle.rsi:.1f}, regime={bundle.regime}, "
            f"p_up_1d={bundle.p_up_1d:.2f}"
        )
        return Decision(
            strategy_key=self.key,
            action="BUY",
            direction="LONG",
            cert_ob_id=config.BULL_MSTR_OB_ID,
            rationale=rationale,
            stop_pct_underlying=config.MOMENTUM_RIDER_HARD_STOP_PCT,
            tp_pct_underlying=None,  # no fixed TP — trailing stop handles it
            confidence=min(bundle.weighted_score_long,
                           config.MOMENTUM_RIDER_MIN_CONFIDENCE),
        )

    # -----------------------------------------------------------------
    # Exit
    # -----------------------------------------------------------------
    def _evaluate_exit(self, bundle: MstrBundle, pos) -> Decision | None:  # noqa: ANN001
        pnl_pct = pos.unrealized_underlying_pct(bundle.price_usd)

        # Hard stop (absolute worst-case defense)
        if pnl_pct <= -config.MOMENTUM_RIDER_HARD_STOP_PCT:
            return self._exit_decision(pos, "stop",
                                       f"pnl={pnl_pct:.2f}% ≤ hard stop")

        # Signal flip — MSTR-weighted SHORT score exceeds its threshold.
        # Uses the same weighted score as entry (mirrored) so the exit
        # decision uses the same MSTR-accurate signals.
        if bundle.weighted_score_short >= config.MOMENTUM_RIDER_SELL_THRESHOLD:
            return self._exit_decision(
                pos, "signal_flip",
                f"weighted_short={bundle.weighted_score_short:.2f}"
            )

        # Trailing stop: once execution.update_trail_state has flipped
        # trail_active on (after pnl hit TRAIL_ACTIVATION_PCT in some prior
        # cycle), fire on pullbacks >= TRAIL_DISTANCE_PCT from the peak.
        # Distance is ATR-adaptive when enabled so quiet sessions use a
        # tighter trail and wild sessions a wider one.
        if pos.trail_active and pos.peak_underlying_price > 0:
            from portfolio.mstr_loop import risk as risk_mod
            trail_dist = risk_mod.effective_trail_distance_pct(
                bundle, config.MOMENTUM_RIDER_TRAIL_DISTANCE_PCT,
            )
            pullback_pct = (
                (pos.peak_underlying_price - bundle.price_usd) /
                pos.peak_underlying_price * 100
            )
            if pullback_pct >= trail_dist:
                return self._exit_decision(
                    pos, "trail",
                    f"pullback={pullback_pct:.2f}% ≥ {trail_dist:.2f}% from peak "
                    f"${pos.peak_underlying_price:.2f}"
                )

        # EOD flatten — checked here so the strategy itself decides to exit;
        # the loop also enforces it as a backstop.
        from portfolio.mstr_loop import session
        if session.in_eod_flatten_window():
            return self._exit_decision(pos, "eod", "21:45 CET EOD flatten")

        return None

    def _exit_decision(self, pos, reason: str, rationale: str) -> Decision:  # noqa: ANN001
        return Decision(
            strategy_key=self.key,
            action="SELL",
            direction=pos.direction,  # same direction, it's a close
            cert_ob_id=pos.cert_ob_id,
            rationale=rationale,
            exit_reason=reason,
        )


STRATEGY = MomentumRider()
