"""premium_arb — MSTR NAV premium mean-reversion (STUB).

MSTR's market cap typically trades at 10-40% premium to the USD value
of its BTC holdings. When the premium expands beyond ~30% it tends to
mean-revert within days. This strategy would:
  - SHORT MSTR (via BEAR cert) when premium > 30%
  - LONG MSTR when premium < 10%
  - Flat otherwise

STATUS: scaffold only. Requires:
  1. Live MSTR NAV calculation: need shares outstanding + BTC held.
     Could use portfolio.alpha_vantage + published MicroStrategy BTC
     counts, or a dedicated scraper.
  2. Historical premium series for threshold calibration.
  3. Integration test against Mar 2021 premium blowoff (when MSTR
     NAV premium exceeded 100% and reverted catastrophically).

To activate:
  1. Build portfolio/mstr_loop/mstr_nav.py: compute premium from live data.
  2. Wire into data_provider.MstrBundle as mstr_nav_premium_pct.
  3. Implement this step() using premium thresholds.
  4. Add ≥10 tests incl. NAV computation + threshold gating.
  5. Flip STRATEGY_TOGGLES["premium_arb"] = True.
"""

from __future__ import annotations

import logging

from portfolio.mstr_loop.data_provider import MstrBundle
from portfolio.mstr_loop.state import BotState
from portfolio.mstr_loop.strategies.base import Decision

logger = logging.getLogger(__name__)

STRATEGY_KEY = "premium_arb"


class PremiumArb:
    key: str = STRATEGY_KEY
    enabled: bool = False  # stub

    def step(self, bundle: MstrBundle, state: BotState) -> Decision | None:
        return None


STRATEGY = PremiumArb()
