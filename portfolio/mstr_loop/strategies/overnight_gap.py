"""overnight_gap — Monday-open gap-fill play on MSTR (STUB).

When BTC moves ≥3% over the weekend (Sat 00:00 → Sun 23:59 UTC) but
MSTR closed flat on Friday, MSTR often gaps ≥5% at Monday US open and
either:
  (a) continues in the direction of the BTC move (follow-through), or
  (b) partially fills the gap within the first 30 min.

This strategy would enter at Monday 15:30 CET with direction matching
the BTC move and exit within 30 min of open, using a tight 2% stop.

STATUS: scaffold only. Requires:
  1. Weekend BTC move detection (Binance 24h ticker + historical
     archive).
  2. Friday-close MSTR snapshot from the signal log.
  3. Backtest against the last 6 months of Monday opens.

To activate:
  1. Add is_monday_open() session helper (first 30min of Monday session).
  2. Implement step() with weekend-BTC-move gate.
  3. Add ≥6 tests.
  4. Flip STRATEGY_TOGGLES["overnight_gap"] = True.
"""

from __future__ import annotations

import logging

from portfolio.mstr_loop.data_provider import MstrBundle
from portfolio.mstr_loop.state import BotState
from portfolio.mstr_loop.strategies.base import Decision

logger = logging.getLogger(__name__)

STRATEGY_KEY = "overnight_gap"


class OvernightGap:
    key: str = STRATEGY_KEY
    enabled: bool = False  # stub

    def step(self, bundle: MstrBundle, state: BotState) -> Decision | None:
        return None


STRATEGY = OvernightGap()
