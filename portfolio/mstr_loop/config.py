"""MSTR Loop configuration — all tunable knobs.

Three-phase rollout:
- PHASE = "shadow" — no execution, log decisions only
- PHASE = "paper"  — paper portfolio with 100K SEK starting cash
- PHASE = "live"   — real orders on Avanza via BankID session
"""

from __future__ import annotations

import os
from typing import Literal

Phase = Literal["shadow", "paper", "live"]

# ---------------------------------------------------------------------------
# Phase switch (override with MSTR_LOOP_PHASE env var)
# ---------------------------------------------------------------------------
PHASE: Phase = os.environ.get("MSTR_LOOP_PHASE", "shadow")  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Instruments (live-verified 2026-04-17)
# ---------------------------------------------------------------------------
MSTR_UNDERLYING = "MSTR"  # ticker for signal lookup in agent_summary
BULL_MSTR_OB_ID = "2257847"  # BULL MSTR X5 SG4, Nordic MTF cert, SEK
BULL_MSTR_LEVERAGE = 5.0  # directional P&L sensitivity vs underlying
# No BEAR cert in v1 (LONG-only). Post-v1 mean_reversion strategy resolves
# BEAR via live `api_post("/_api/search/filtered-search")` at that time.

# ---------------------------------------------------------------------------
# MSTR-specific signal reliability weights (from system_lessons.json audit)
# Higher weight = signal is historically more accurate on MSTR.
# Signals with 0.0 weight are forced neutral (below-coin-flip on MSTR).
# ---------------------------------------------------------------------------
MSTR_SIGNAL_WEIGHTS: dict[str, float] = {
    # Top-reliability signals on MSTR
    "ministral": 3.0,           # 70.9% acc, n=347
    "econ_calendar": 3.0,       # 83.6% acc, n=440
    "calendar": 2.5,            # 68.0% acc, n=322
    "qwen3": 2.5,               # 63.8% acc, n=389
    "volume_flow": 2.0,         # 59.6% acc, n=3122 (high sample count)
    # Default weight (everything not listed) is 1.0. The get() fallback
    # in the scoring helper handles new signals gracefully.
    # Force-ignored: below-coin-flip reliability on MSTR
    "trend": 0.0,               # 41.1% acc, n=433
    "volatility_sig": 0.0,      # 39.9% acc, n=348
    "macro_regime": 0.0,        # 34.7% acc, n=969
}
DEFAULT_SIGNAL_WEIGHT: float = 1.0

# ---------------------------------------------------------------------------
# Strategy toggles — edit these to enable/disable strategies without code
# ---------------------------------------------------------------------------
STRATEGY_TOGGLES: dict[str, bool] = {
    "momentum_rider": True,
    # Future slots, wired in when implemented:
    # "mean_reversion": False,
    # "earnings_play": False,
    # "premium_arb": False,
    # "overnight_gap": False,
}

# ---------------------------------------------------------------------------
# Session window (US market hours in CET)
# ---------------------------------------------------------------------------
SESSION_OPEN_CET_HOUR = 15
SESSION_OPEN_CET_MINUTE = 30
SESSION_CLOSE_CET_HOUR = 22
SESSION_CLOSE_CET_MINUTE = 0
EOD_FLATTEN_CET_HOUR = 21
EOD_FLATTEN_CET_MINUTE = 45  # hard deadline — market-close all positions
# NOTE: NASDAQ hours shift with DST (21:30-22:00 depending on week).
# Hardcoded 22:00 close is correct ~51 weeks/year; DST-gap weeks
# (Mar 8-29 and Oct 25-Nov 1) shift real close 1h earlier. EOD at 21:45
# still works then (20:45 buffer). Revisit if DST gap coincides with a
# live trading day and we see issues.

# ---------------------------------------------------------------------------
# Cycle cadence
# ---------------------------------------------------------------------------
CYCLE_INTERVAL_SEC = 60

# ---------------------------------------------------------------------------
# Sizing — matches metals_swing_config pattern for Kelly-with-floor
# ---------------------------------------------------------------------------
ACCOUNT_ID = "1625505"           # Avanza ISK (shared with metals swing)
INITIAL_PAPER_CASH_SEK = 100_000  # Phase C starting capital
SHADOW_NOTIONAL_SEK = 30_000      # Phase B hypothetical position size
MIN_TRADE_SEK = 1000              # Avanza min courtage floor
POSITION_SIZE_PCT = 30            # fallback when Kelly unavailable
MAX_CONCURRENT_POSITIONS = 1      # one per strategy in v1

# ---------------------------------------------------------------------------
# momentum_rider strategy knobs
# ---------------------------------------------------------------------------
MOMENTUM_RIDER_BUY_THRESHOLD = 0.55      # weighted score required for LONG entry
MOMENTUM_RIDER_SELL_THRESHOLD = 0.55     # weighted inverse score required for exit
MOMENTUM_RIDER_MIN_CONFIDENCE = 0.50     # confidence cap during Phase B
MOMENTUM_RIDER_RSI_MAX = 78              # above this, skip entry (avoid blow-offs)
MOMENTUM_RIDER_RSI_MIN = 40              # below this, skip entry (already weak)
MOMENTUM_RIDER_COOLDOWN_MINUTES = 30     # between entries on this strategy
MOMENTUM_RIDER_TRAIL_ACTIVATION_PCT = 1.5  # % underlying profit to start trailing
MOMENTUM_RIDER_TRAIL_DISTANCE_PCT = 2.0    # % underlying pullback triggers exit
MOMENTUM_RIDER_HARD_STOP_PCT = 2.0         # % underlying loss triggers exit

# ---------------------------------------------------------------------------
# File paths (all atomic-write via portfolio.file_utils)
# ---------------------------------------------------------------------------
STATE_FILE = "data/mstr_loop_state.json"
TRADES_LOG = "data/mstr_loop_trades.jsonl"
SHADOW_LOG = "data/mstr_loop_shadow.jsonl"
POLL_LOG = "data/mstr_loop_poll.jsonl"
KILL_SWITCH_FILE = "data/mstr_loop.disabled"
SINGLETON_LOCK_FILE = "data/mstr_loop.singleton.lock"
