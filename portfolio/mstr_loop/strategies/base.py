"""Strategy base protocol + Decision dataclass for MSTR Loop.

Each strategy implements `step(bundle, state) -> Decision | None`. The loop
iterates enabled strategies per cycle, executes any returned Decision via
`execution.py`. Strategies do NOT touch Avanza directly — execution is
centralized so DRY_RUN/paper/live gating is in one place.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

from portfolio.mstr_loop.data_provider import MstrBundle
from portfolio.mstr_loop.state import BotState


@dataclasses.dataclass
class Decision:
    """What a strategy wants to do this cycle."""
    strategy_key: str
    action: str                  # "BUY" or "SELL"
    direction: str               # "LONG" or "SHORT"
    cert_ob_id: str              # which Avanza cert to trade
    rationale: str               # human-readable why (goes in journal)
    # Size hints for execution layer
    notional_sek_hint: float | None = None  # override Kelly if set
    # Stop/TP expressed in underlying % (execution layer converts to cert price)
    stop_pct_underlying: float | None = None
    tp_pct_underlying: float | None = None
    # Metadata
    confidence: float = 0.0      # 0-1, strategy's own confidence in the call
    exit_reason: str | None = None  # for SELL: "signal_flip" | "trail" | "stop" | "eod"


class Strategy(Protocol):
    """Interface every strategy must implement.

    Strategies are stateless where possible — state lives in BotState, not
    in the strategy instance. The only exception is strategy-internal
    heuristic tracking (e.g. in-memory recent-trade counters) that doesn't
    need persistence across restarts.
    """
    key: str
    enabled: bool

    def step(self, bundle: MstrBundle, state: BotState) -> Decision | None:
        """Decide what to do this cycle. None = no action."""
        ...
