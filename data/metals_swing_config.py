"""Metals swing trader configuration — warrant catalog and trading parameters.

All tunable constants for the autonomous swing trader. Edit here, not in the trader.
"""

# ---------------------------------------------------------------------------
# Warrant catalog — all tradeable instruments with metadata
# ---------------------------------------------------------------------------
WARRANT_CATALOG = {
    "MINI_L_SILVER_SG": {
        "ob_id": "2043157",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "leverage": 1.56,
        "barrier": 32.45,
        "parity": 10,
        "name": "MINI L SILVER SG",
    },
    "MINI_L_SILVER_AVA_301": {
        "ob_id": "2334960",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "leverage": 6.3,
        "barrier": 75.03,
        "parity": 10,
        "name": "MINI L SILVER AVA 301",
    },
    "BEAR_SILVER_X5_AVA_12": {
        "ob_id": "2286417",
        "api_type": "certificate",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 5.0,
        "barrier": None,
        "parity": 1,
        "name": "BEAR SILVER X5 AVA 12",
    },
}

# ---------------------------------------------------------------------------
# Execution mode
# ---------------------------------------------------------------------------
DRY_RUN = False               # True = log decisions + Telegram without placing orders
BUY_MODE = "AUTO"             # "AUTO" or "CONFIRM" (Telegram confirmation for BUYs)
SELL_MODE = "AUTO"            # SELLs always autonomous

# ---------------------------------------------------------------------------
# Sizing & limits
# ---------------------------------------------------------------------------
ACCOUNT_ID = "1625505"
INITIAL_BUDGET_SEK = 10000    # fallback budget when Avanza API fails to return cash
POSITION_SIZE_PCT = 30        # % of available cash per trade
MAX_CONCURRENT = 2            # max simultaneous positions
TARGET_LEVERAGE = 5.0         # preferred leverage (user prefers 5x)
MIN_ACCEPTABLE_LEVERAGE = 3.0 # SKIP_BUY if best candidate is below this (fail-closed)
MIN_BARRIER_DISTANCE_PCT = 10 # minimum distance to barrier (was 15, but that excluded
                              # all high-lev AVA MINIs and forced fallbacks to trackers)
MIN_SPREAD_PCT = 1.5          # max acceptable bid-ask spread %
MIN_TRADE_SEK = 1000          # minimum trade size (Avanza min courtage threshold)

# ---------------------------------------------------------------------------
# Entry rules
# ---------------------------------------------------------------------------
MIN_BUY_VOTERS = 3            # minimum agreeing BUY signals
MIN_BUY_CONFIDENCE = 0.60     # minimum calibrated signal confidence (user rule: no sub-60% trades)
MIN_BUY_TF_RATIO = 0.43       # 3/7 timeframes must agree
RSI_ENTRY_LOW = 35            # RSI buy zone lower bound
RSI_ENTRY_HIGH = 68           # RSI buy zone upper bound (avoid overbought)
MACD_IMPROVING_CHECKS = 2     # MACD must be improving for N consecutive checks
REGIME_CONFIRM_CHECKS = 2     # require N consecutive BUY checks in same regime
                              # (rejects single-check flips from trending-down → ranging BUY)

# ---------------------------------------------------------------------------
# Momentum-entry override (2026-04-17)
# ---------------------------------------------------------------------------
# When metals_loop's entry-side fast-tick detects an upside breakout it writes
# a momentum candidate to MOMENTUM_STATE_FILE. If the swing trader sees a
# fresh candidate for the ticker it is evaluating, the snapshot-conviction
# gates relax: momentum has already price-confirmed what the voter count
# would eventually confirm on its own 2-3 cycles later (by which time the
# move is often half over).
#
# The relaxed override only touches MIN_BUY_CONFIDENCE and MIN_BUY_VOTERS.
# RSI zone, MACD-improving, regime-confirm, and TF-alignment stay at their
# regular thresholds — those reject *false* breakouts and the user still
# benefits from them during momentum bursts.
#
# SHORT-side momentum is not yet supported. The fast-tick only writes LONG
# candidates; SHORT_ENABLED=False so there is no production path.
MOMENTUM_ENTRY_ENABLED = True
MOMENTUM_MIN_BUY_CONFIDENCE = 0.50   # relaxed from MIN_BUY_CONFIDENCE=0.60
MOMENTUM_MIN_BUY_VOTERS = 2          # relaxed from MIN_BUY_VOTERS=3
MOMENTUM_CANDIDATE_TTL_SEC = 300     # candidates older than 5 min are ignored
MOMENTUM_STATE_FILE = "data/metals_momentum_state.json"

# ---------------------------------------------------------------------------
# Entry-gate hardening (2026-04-18 — post-mortem for the
# MINI L SILVER AVA 336 loss of -5.07% / -50.56 SEK on 2026-04-17).
#
# The standard entry gates passed at the edge (conf 0.66 just above 0.60;
# RSI 68 exactly at RSI_ENTRY_HIGH; MACD +0.015 barely positive after
# decaying from +0.22). Three new gates address specific equation bugs
# exposed by that trade. Full debate including steelman counter-arguments
# in docs/plans/2026-04-18-entry-gates.md.
#
# NB: these gates apply to the standard-gate path only. The momentum-
# override path (fresh fast-tick breakout candidate) stays on the relaxed
# MOMENTUM_MIN_BUY_CONFIDENCE / MOMENTUM_MIN_BUY_VOTERS thresholds and
# skips these three — momentum trades explicitly need to catch
# fast-moving breakouts without 2-cycle persistence lag.

# Gate A — signal persistence. Require confidence >= MIN_BUY_CONFIDENCE
# for N consecutive cycles. The 2026-04-17 trade had conf
# 0.40 -> 0.80 -> 0.66 -> 0.00 across 70 minutes; a 2-cycle requirement
# blocks any single-cycle phantom spike while costing at most ~2 min of
# entry lag.
SIGNAL_PERSISTENCE_CHECKS = 2

# Gate B — MACD decay. Reject entries where current MACD is less than
# MACD_DECAY_MIN_RATIO of the max |MACD| over the last
# MACD_DECAY_PEAK_LOOKBACK cycles. Catches "fading momentum after peak":
# on 2026-04-17 MACD decayed from +0.22 to +0.001 over an hour, then
# ticked up +0.001 -> +0.015 (+14x in % terms, noise in absolute
# terms). Ratio to recent peak would have been 7% -> rejected. Asset-
# agnostic (no fixed threshold that would need per-symbol tuning).
MACD_DECAY_PEAK_LOOKBACK = 20        # capped at current macd_history length
MACD_DECAY_MIN_RATIO = 0.30          # current |MACD| must be >= 30% of recent peak |MACD|

# Gate C — RSI slope / recent-dip. RSI alone at 68 says nothing about
# momentum direction; combined with slope it does. Accept if EITHER:
#   - RSI is not falling over RSI_SLOPE_LOOKBACK_CHECKS (current >=
#     value N cycles ago), OR
#   - RSI dipped below RSI_DIP_BELOW_LEVEL in the last
#     RSI_DIP_LOOKBACK_CHECKS (proves we're buying a pullback, not the
#     tail end of a rally).
# Yesterday RSI went 79 -> 77 -> 75 -> 72 -> 68 -> 66 -> 68 (declining),
# never reaching 55 -> would have been rejected.
RSI_SLOPE_LOOKBACK_CHECKS = 5        # ≈5 min
RSI_DIP_LOOKBACK_CHECKS = 20         # ≈20 min
RSI_DIP_BELOW_LEVEL = 55             # must have touched <= 55 recently OR be rising

# History buffers for the new gates. Mirrored on MACD history (length 20).
CONFIDENCE_HISTORY_MAX = 20
RSI_HISTORY_MAX = 30                 # covers both lookbacks comfortably

# ---------------------------------------------------------------------------
# Exit rules
# ---------------------------------------------------------------------------
TAKE_PROFIT_UNDERLYING_PCT = 3.0   # 2026-04-14 raised from 2.0 — was exiting too early on
                                   # intraday noise during strong trends. 3.0% underlying =
                                   # ~+15% warrant on 5x / ~+14% on 4.75x. If user wants
                                   # even more room, raise further; user's note: "we can
                                   # always buy in again" — rebuy on pullback after exit.
TRAILING_START_PCT = 1.5           # start trailing after 1.5% underlying gain
TRAILING_DISTANCE_PCT = 1.0        # trail 1% behind underlying peak
HARD_STOP_UNDERLYING_PCT = 2.0     # -2% underlying = hard exit
SIGNAL_REVERSAL_EXIT = True        # exit on SELL consensus with >= MIN_BUY_VOTERS

# 2026-04-20: warrant-side exit rules. Added after MINI L SILVER AVA 331
# peaked at 15.48 (+5.9% from 14.62 entry) around 16:00 CET while silver
# underlying only moved +1.26% — below the 3% underlying TP threshold, so
# the existing rules never fired. Market makers can mark the warrant up
# independently of the underlying (spread widening, momentum premium),
# which the old underlying-only logic ignored.
#
# These rules fire IN PARALLEL with the underlying-side rules; whichever
# trips first wins via the `if not exit_reason` chain. Track
# ``peak_warrant_bid`` per position on every eval.
WARRANT_TAKE_PROFIT_PCT = 5.0        # exit when warrant bid >= entry * 1.05
WARRANT_TRAILING_START_PCT = 3.0     # activate warrant trailing at +3% from entry
WARRANT_TRAILING_DISTANCE_PCT = 1.5  # exit on 1.5% retrace from warrant peak

# 2026-04-17 momentum-exit tuning pass. Background: MINI L SILVER AVA 331
# bought 13:33 CET (6B/1S, conf 0.603) and sold 55 seconds later as
# "MOMENTUM_EXIT: 3 declining checks (-0.64%)". Silver rallied +5.4% off
# the sell price. Three compounding bugs:
#
#   1. Pre-entry history contaminates exit check — ``_und_history`` isn't
#      reset on entry, so ticks from before the buy triggered the exit
#      on the same cycle that verified the fill.
#   2. No minimum hold — rule evaluates from t=0 of the position.
#   3. -0.3% over 3×60s ticks is below the XAG/XAU noise floor (typical
#      60s tick 0.03-0.13%; three can sum to >0.3% on pure noise).
#
# Fix: hard stop + trailing + signal-reversal still fire from t=0. Only
# the 3-tick counter-trend heuristic waits for a minimum hold AND
# requires a larger cumulative move. ``_und_history`` is cleared on
# entry at ``metals_swing_trader._execute_buy``.
MOMENTUM_EXIT_MIN_HOLD_SECONDS = 300   # 5 min hold before 3-tick counter-trend
                                       # rule can fire. Hard stop + trailing +
                                       # signal reversal still active from t=0,
                                       # so real catastrophic moves still exit
                                       # fast via HARD_STOP (-2% underlying).
                                       # 5m ≈ 5 post-entry 60s ticks — leaves
                                       # 2-3 cycles past the minimum 3-tick
                                       # window while still blocking the
                                       # fill-verification-same-cycle exit.
MOMENTUM_EXIT_THRESHOLD_PCT = 0.8      # Min cumulative counter-trend move over
                                       # the 3-tick window. Raised from 0.3.
# 2026-04-10: user removed the 5h time limit in favor of an EOD-only forced
# sell just before US market close. Set to 24h so the safety net still
# exists for catastrophic edge cases (position orphaned by a crash) but
# intraday exits are rule-driven (TAKE_PROFIT, TRAILING, HARD_STOP,
# SIGNAL_REVERSAL, MOMENTUM, EXIT_OPTIMIZER).
#
# *** EOD_EXIT_MINUTES_BEFORE and the DST gap trap ***
#
# The check in metals_swing_trader._check_exits compares minutes_to_close
# against this value, where close_cet is HARDCODED to 21:55 CET at
# metals_swing_trader.py:1156. That hardcode is correct for ~51 weeks/year
# (standard DST-aligned US close = 22:00 CET → 21:55 "practical close").
# During the DST gap weeks (roughly Mar 8-29 and Oct 25-Nov 1) US is on DST
# while EU is not, shifting the real close down to 21:00 CET. With a 10-min
# buffer, EOD would fire at 21:45 — 45 MINUTES AFTER real close, so orders
# would not execute and positions would bleed overnight.
#
# To be safe during DST gaps AND still close reasonably near the real close
# in normal weeks, use a 25-min buffer:
#   - Normal weeks: real close 22:00, EOD fires at 21:30 (30 min early).
#   - DST gap: real close 21:00, EOD fires at 20:30 (30 min early — still in
#     the trading window).
# This is a compromise until dynamic todayClosingTime lookup is added (see
# .claude/rules/metals-avanza.md). TODO: ship get_session_close_cet() in
# portfolio/avanza_session.py and set this back to 10.
MAX_HOLD_HOURS = 24                # 24h safety net only — real time-based exit is EOD
EOD_EXIT_MINUTES_BEFORE = 0        # 2026-04-13 user override: disabled EOD blind exit.
                                   # Statistical basis: XAG 1d UP base rate 55.7%, current
                                   # SELL consensus only 32.5% accurate at 1d horizon (n=813).
                                   # Holding overnight is probabilistically favorable. The
                                   # smart breakeven evaluator (rule 1b in _check_exits) +
                                   # HARD_STOP + TRAILING still manage risk.
                                   # REVERT to 25 after current position closes.

# ---------------------------------------------------------------------------
# Cooldowns
# ---------------------------------------------------------------------------
BUY_COOLDOWN_MINUTES = 30     # min time between BUYs
LOSS_ESCALATION = {0: 1, 1: 2, 2: 4, 3: 8}  # consecutive losses -> cooldown multiplier

# ---------------------------------------------------------------------------
# Stop-loss (hardware, on Avanza)
# ---------------------------------------------------------------------------
STOP_LOSS_UNDERLYING_PCT = 2.5     # stop at -2.5% underlying from entry
STOP_LOSS_VALID_DAYS = 8           # stop order validity (calendar days)

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
TELEGRAM_SUMMARY_INTERVAL = 20     # send summary every N checks (~30 min at 90s)

# ---------------------------------------------------------------------------
# State files
# ---------------------------------------------------------------------------
STATE_FILE = "data/metals_swing_state.json"
DECISIONS_LOG = "data/metals_swing_decisions.jsonl"
TRADES_LOG = "data/metals_swing_trades.jsonl"
