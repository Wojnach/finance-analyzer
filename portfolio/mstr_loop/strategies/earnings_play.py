"""earnings_play — straddle-style MSTR earnings strategy (STUB).

MSTR has historically produced ±15-25% moves on earnings prints. This
strategy would open a small position (LONG or SHORT depending on pre-
print technicals) 2 trading days before earnings, flat it at open
post-print.

STATUS: scaffold only. DO NOT enable. Requires backtesting against
historical MSTR earnings dates before activation.

To activate:
  1. Fetch MSTR earnings calendar via portfolio.earnings_calendar.
  2. Implement entry rule (LONG if pre-print BTC regime up + MSTR
     technicals trending, SHORT if overbought + post-print expected).
  3. Add test_mstr_loop_earnings_play.py with ≥8 tests.
  4. Flip STRATEGY_TOGGLES["earnings_play"] = True in config.
"""

from __future__ import annotations

import logging

from portfolio.mstr_loop.data_provider import MstrBundle
from portfolio.mstr_loop.state import BotState
from portfolio.mstr_loop.strategies.base import Decision

logger = logging.getLogger(__name__)

STRATEGY_KEY = "earnings_play"


class EarningsPlay:
    key: str = STRATEGY_KEY
    enabled: bool = False  # stub

    def step(self, bundle: MstrBundle, state: BotState) -> Decision | None:
        # Intentional no-op. See module docstring for activation checklist.
        return None


STRATEGY = EarningsPlay()
