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

# BEAR MSTR cert (v2 Tier 2 — SHORT support via mean_reversion strategy).
# ob_id is None until resolved live; mean_reversion strategy refuses entries
# when None (fail-safe). To resolve live, run:
#     scripts/resolve_mstr_bear_cert.py
# or set MSTR_LOOP_BEAR_OB_ID env var to override.
BEAR_MSTR_OB_ID: str | None = os.environ.get("MSTR_LOOP_BEAR_OB_ID") or None
BEAR_MSTR_LEVERAGE = 3.0

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
    "mean_reversion": False,   # v2: SHORT strategy — enable when BEAR_MSTR_OB_ID is set
    "earnings_play": False,    # v2 stub
    "premium_arb": False,      # v2 stub
    "overnight_gap": False,    # v2 stub
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
# mean_reversion (SHORT) strategy knobs — v2 Tier 2
# ---------------------------------------------------------------------------
MEAN_REVERSION_SELL_THRESHOLD = 0.55
MEAN_REVERSION_MIN_CONFIDENCE = 0.50
MEAN_REVERSION_RSI_MIN = 75                # SHORT only makes sense when overbought
MEAN_REVERSION_COOLDOWN_MINUTES = 30
MEAN_REVERSION_TRAIL_ACTIVATION_PCT = 1.5
MEAN_REVERSION_TRAIL_DISTANCE_PCT = 2.0
MEAN_REVERSION_HARD_STOP_PCT = 2.0

# ---------------------------------------------------------------------------
# Drawdown circuit breaker (v2 Tier 1)
# ---------------------------------------------------------------------------
DRAWDOWN_DAILY_HALT_PCT = -3.0             # halt entries if today P&L <= -3% of peak
DRAWDOWN_WEEKLY_HALT_PCT = -8.0            # halt all strategies for 7d if week <= -8%
DRAWDOWN_CHECK_ENABLED = True

# ---------------------------------------------------------------------------
# BTC-regime gate (v2 Tier 2)
# ---------------------------------------------------------------------------
BTC_REGIME_GATE_ENABLED = True
BTC_REGIME_DOWN_TAGS = frozenset({"trending-down"})
BTC_REGIME_UP_TAGS = frozenset({"trending-up"})

# ---------------------------------------------------------------------------
# Earnings blackout (v2 Tier 2)
# ---------------------------------------------------------------------------
EARNINGS_BLACKOUT_ENABLED = True
EARNINGS_BLACKOUT_DAYS_BEFORE = 2
EARNINGS_BLACKOUT_DAYS_AFTER = 1

# ---------------------------------------------------------------------------
# Partial-exit ladder (v2 Tier 2)
# ---------------------------------------------------------------------------
PARTIAL_EXIT_LADDER_ENABLED = True
# Sequence of (profit_pct, fraction_of_original_units) tranches.
# Final third rides the trail stop (no explicit tranche).
PARTIAL_EXIT_TRANCHES: list[tuple[float, float]] = [
    (2.0, 1 / 3),
    (4.0, 1 / 3),
]

# ---------------------------------------------------------------------------
# ATR-adaptive trail (v2 Tier 3)
# ---------------------------------------------------------------------------
ATR_ADAPTIVE_TRAIL_ENABLED = True
ATR_ADAPTIVE_MULT = 1.5
ATR_ADAPTIVE_TRAIL_MIN_PCT = 1.5
ATR_ADAPTIVE_TRAIL_MAX_PCT = 4.0

# ---------------------------------------------------------------------------
# Dynamic signal re-weighting (v2 Tier 3 — disabled by default)
# ---------------------------------------------------------------------------
DYNAMIC_WEIGHTS_ENABLED = False
DYNAMIC_WEIGHTS_REFRESH_EVERY_CYCLES = 60
DYNAMIC_WEIGHTS_MIN_SAMPLES = 30
DYNAMIC_WEIGHTS_SYSTEM_LESSONS_FILE = "data/system_lessons.json"

# ---------------------------------------------------------------------------
# Telegram (v2 Tier 1)
# ---------------------------------------------------------------------------
TELEGRAM_ENABLED = True
TELEGRAM_HOURLY_REPORT_MINUTES = 60        # cadence for status reports
TELEGRAM_PER_TRADE_ALERTS = True
TELEGRAM_STATE_FILE = "data/mstr_loop_telegram_state.json"

# ---------------------------------------------------------------------------
# Auto-scorecard (v2 Tier 1) — writes after every closed trade
# ---------------------------------------------------------------------------
SCORECARD_FILE = "data/mstr_loop_scorecard.json"
SCORECARD_UPDATE_ENABLED = True

# ---------------------------------------------------------------------------
# File paths (all atomic-write via portfolio.file_utils)
# ---------------------------------------------------------------------------
STATE_FILE = "data/mstr_loop_state.json"
TRADES_LOG = "data/mstr_loop_trades.jsonl"
SHADOW_LOG = "data/mstr_loop_shadow.jsonl"
POLL_LOG = "data/mstr_loop_poll.jsonl"
KILL_SWITCH_FILE = "data/mstr_loop.disabled"
SINGLETON_LOCK_FILE = "data/mstr_loop.singleton.lock"
