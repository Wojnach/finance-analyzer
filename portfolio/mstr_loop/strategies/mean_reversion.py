"""mean_reversion — SHORT MSTR when weighted signals are overbought.

Mirror of momentum_rider in the opposite direction. Fires SHORT on
BEAR MSTR cert when weighted_score_short crosses threshold AND RSI > 75.
Other gates (drawdown, earnings, BTC regime) shared via portfolio.mstr_loop.risk.

v2 Tier 2 — adds the second execution direction. Enabled only when
BEAR_MSTR_OB_ID is populated in config (so a misconfigured bot refuses
to fire SHORT blindly).
"""

from __future__ import annotations

import logging

from portfolio.mstr_loop import config
from portfolio.mstr_loop.data_provider import MstrBundle
from portfolio.mstr_loop.state import BotState
from portfolio.mstr_loop.strategies.base import Decision

logger = logging.getLogger(__name__)

STRATEGY_KEY = "mean_reversion"


class MeanReversion:
    """SHORT-side trend-fade strategy on BEAR MSTR cert."""

    key: str = STRATEGY_KEY
    enabled: bool = True

    def step(self, bundle: MstrBundle, state: BotState) -> Decision | None:
        if not bundle.is_usable():
            return None
        if config.BEAR_MSTR_OB_ID is None:
            # Fail-safe: bot cannot SHORT without a resolved BEAR cert ob_id.
            # This also means strategy is effectively disabled until an
            # operator resolves the cert and sets the env/config.
            logger.debug("mean_reversion: BEAR_MSTR_OB_ID not configured — SHORT entries refused")
            return None
        existing = state.get_position(self.key)
        if existing is not None:
            return self._evaluate_exit(bundle, existing)
        return self._evaluate_entry(bundle, state)

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------
    def _evaluate_entry(self, bundle: MstrBundle, state: BotState) -> Decision | None:
        # Shared risk gates — same set as momentum_rider, direction-aware.
        from portfolio.mstr_loop import risk as risk_mod
        halted, reason = risk_mod.any_entry_halt_active(state, bundle, direction="SHORT")
        if halted:
            logger.debug("mean_reversion: entry halted — %s", reason)
            return None

        if not state.cooldown_elapsed(self.key, config.MEAN_REVERSION_COOLDOWN_MINUTES):
            return None
        # SHORT is specifically a fade-the-blow-off play. RSI must be
        # overbought OR the weighted short score must be extreme.
        if bundle.rsi < config.MEAN_REVERSION_RSI_MIN:
            return None
        if bundle.weighted_score_short < config.MEAN_REVERSION_SELL_THRESHOLD:
            return None
        if bundle.stale:
            return None

        rationale = (
            f"weighted_short={bundle.weighted_score_short:.2f} ≥ "
            f"{config.MEAN_REVERSION_SELL_THRESHOLD}, "
            f"rsi={bundle.rsi:.1f} ≥ {config.MEAN_REVERSION_RSI_MIN}, "
            f"regime={bundle.regime}, p_up_1d={bundle.p_up_1d:.2f}"
        )
        return Decision(
            strategy_key=self.key,
            action="BUY",  # from execution's perspective, opening a SHORT = buying BEAR cert
            direction="SHORT",
            cert_ob_id=config.BEAR_MSTR_OB_ID,
            rationale=rationale,
            stop_pct_underlying=config.MEAN_REVERSION_HARD_STOP_PCT,
            tp_pct_underlying=None,
            confidence=min(
                bundle.weighted_score_short,
                config.MEAN_REVERSION_MIN_CONFIDENCE,
            ),
        )

    # ------------------------------------------------------------------
    # Exit (direction-aware: SHORT wins when price falls)
    # ------------------------------------------------------------------
    def _evaluate_exit(self, bundle: MstrBundle, pos) -> Decision | None:  # noqa: ANN001
        pnl_pct = pos.unrealized_underlying_pct(bundle.price_usd)

        if pnl_pct <= -config.MEAN_REVERSION_HARD_STOP_PCT:
            return self._exit_decision(pos, "stop",
                                       f"pnl={pnl_pct:.2f}% ≤ hard stop")

        # Signal flip — weighted LONG score exceeds threshold → close SHORT.
        if bundle.weighted_score_long >= config.MEAN_REVERSION_SELL_THRESHOLD:
            return self._exit_decision(
                pos, "signal_flip",
                f"weighted_long={bundle.weighted_score_long:.2f}"
            )

        # Trailing stop — peak for SHORT is the LOWEST price seen since entry.
        # execution.update_trail_state tracks this direction-aware peak.
        if pos.trail_active and pos.peak_underlying_price > 0:
            # For SHORT, pullback = current rising above the peak
            pullback_pct = (
                (bundle.price_usd - pos.peak_underlying_price) /
                pos.peak_underlying_price * 100
            )
            if pullback_pct >= config.MEAN_REVERSION_TRAIL_DISTANCE_PCT:
                return self._exit_decision(
                    pos, "trail",
                    f"bounce={pullback_pct:.2f}% from low "
                    f"${pos.peak_underlying_price:.2f}"
                )

        # EOD flatten backstop
        from portfolio.mstr_loop import session
        if session.in_eod_flatten_window():
            return self._exit_decision(pos, "eod", "21:45 CET EOD flatten")

        return None

    def _exit_decision(self, pos, reason: str, rationale: str) -> Decision:  # noqa: ANN001
        return Decision(
            strategy_key=self.key,
            action="SELL",
            direction=pos.direction,
            cert_ob_id=pos.cert_ob_id,
            rationale=rationale,
            exit_reason=reason,
        )


STRATEGY = MeanReversion()
