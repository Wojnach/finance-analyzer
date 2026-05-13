"""Grid market-maker configuration — tier math, sizing, EOD timing.

The grid fisher places a multi-tier ladder of buy limits on a single
warrant per ticker+direction, rotates each fill into an opposite-side
sell limit plus stop, and reconciles continuously. One direction is
active per ticker at a time; flips are gated by a cooldown.

User-visible knobs live here. Code in ``portfolio/grid_fisher.py`` and
``portfolio/grid_tiers.py`` reads from this module only — no magic
constants inline.

Hold profile: minutes to hours (5x leverage + barrier risk make overnight
asymmetric; see memory ``feedback_fishing_hold_time.md``).
"""

# ---------------------------------------------------------------------------
# Master switch
# ---------------------------------------------------------------------------
# 2026-05-11 PM: User confirmed account 1625505
# (INVESTERINGSSPARKONTO/ISK) is the intended warrant trading account.
# Swedish ISK accounts legally trade leveraged certs. The
# avanza_account_check verifier no longer disallows ISK by default
# (DISALLOWED_CATEGORY_FRAGMENTS is empty). Flipping PROBE_ONLY back
# to False so real placements route to the configured account.
GRID_FISHER_ENABLED = True

# Probe mode: when True, the tick computes intended placements and writes
# them to the decision log without calling Avanza.
GRID_FISHER_PROBE_ONLY = False

# ---------------------------------------------------------------------------
# Tier construction
# ---------------------------------------------------------------------------
GRID_TIERS = 2
# Each tier sits this far below the prevailing bid. Index 0 is closest to
# market (highest fill probability); higher indices are deeper dips.
# Two tiers (vs the original three) keep total deployed notional inside
# the user's 7 000 SEK budget: 2 tiers × 1 200 SEK × 3 instruments
# worst-case = 7 200 SEK. The global cap below enforces 6 500 SEK so we
# always have headroom for one rotation cycle without breaching budget.
GRID_TIER_SPACING_PCT = (0.4, 1.2)
# When a buy fills at price P:
#   sell limit = P * (1 + GRID_TARGET_PCT/100)
#   stop loss  = P * (1 - GRID_STOP_PCT/100)
# Target must clear courtage (~2 SEK round-trip on 1200 SEK = 0.17%) plus
# warrant spread (~0.5%) with margin. 1.2% gives ~0.5% net edge per cycle.
GRID_TARGET_PCT = 1.2
# Stop sits outside the typical 15-minute volatility band so noise does
# not fire it (memory ``feedback_stops_outside_volatility.md``).
GRID_STOP_PCT = 3.5

# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------
# Per-leg order size in SEK. 1200 stays just above Avanza's 1000 SEK
# minimum-courtage threshold. User's stated cap.
GRID_LEG_SEK = 1200
# Hard cap on resting + filled inventory per instrument. With 2 tiers,
# placement totals 2 400 SEK; cap leaves headroom for one rotation cycle
# (one tier still resting plus newly-filled inventory waiting on exit)
# without breaching the global budget.
GRID_PER_INSTRUMENT_MAX_SEK = 3000
# Global cap across all instruments. Enforced inside tick() before any
# new placement — if planned-notional summed across all active
# instruments would exceed this, the cycle skips with a logged
# `skip_global_cap` decision. Sized to fit inside a 7 000 SEK trading
# budget with headroom for rotation legs and Avanza margin reserve.
GRID_GLOBAL_MAX_SEK = 6500
# Session loss budget per instrument. Breaching this freezes new
# placements until the next session.
GRID_PER_SESSION_LOSS_LIMIT_SEK = 500
# Reserve held back from the live Avanza buying power before the cap is
# computed. Covers courtage, FX spread, and rotation legs that haven't
# rebooked yet. The effective global cap each tick =
#   min(GRID_GLOBAL_MAX_SEK, max(0, live_buying_power - this buffer)).
# 2026-05-13: added after grid_fisher attempted OLJAB placements with
# only ~3 097 SEK available (memory project_avanza_account_mismatch_20260511);
# previously the cap was hardcoded against a 6 500 SEK budget and never
# consulted the live account, so insufficient-cash rejections came from
# Avanza rather than being filtered locally.
GRID_CASH_SAFETY_BUFFER_SEK = 500
# Buying-power readings are cached this many seconds so a multi-instrument
# tick doesn't hit /_api/account-overview six times. Refreshed lazily on
# the next tick after expiry.
GRID_BUYING_POWER_CACHE_SECS = 60
# When a buying-power fetch fails (None) but a recent good reading exists
# within this many seconds, the cached value is reused. Past that horizon
# we fail-closed (effective cap = 0) so a stale session can't keep
# placing orders against unknown cash. Better to skip a cycle than to
# over-order against a stale balance.
GRID_BUYING_POWER_STALE_GRACE_SECS = 300

# ---------------------------------------------------------------------------
# Signal gating
# ---------------------------------------------------------------------------
# Minimum aggregated confidence to arm a direction. Matches the metals
# swing trader floor (calibration-aware reanchor, 2026-05-04).
GRID_MIN_SIGNAL_CONFIDENCE = 0.56
# Cooldown after a BULL/BEAR flip on the same ticker. Prevents whipsaw
# placement when signal oscillates near consensus boundary.
GRID_DIRECTION_FLIP_COOLDOWN_MIN = 30
# When ADX(14) exceeds this, the counter-trend ladder is skipped — only
# the with-trend direction is allowed. Protects against MM-in-trend
# death spirals (memory ``project_fish_engine_live_test.md``).
GRID_ADX_TREND_FILTER = 35

# ---------------------------------------------------------------------------
# Knockout / barrier safety
# ---------------------------------------------------------------------------
# Skip tiers whose implied underlying move would land within this fraction
# of the warrant's knockout barrier. Reused threshold from exit_optimizer.
GRID_KNOCKOUT_SAFETY_PCT = 8.0

# ---------------------------------------------------------------------------
# EOD behaviour
# ---------------------------------------------------------------------------
# Minutes before todayClosingTime to cancel unfilled buy tiers and
# tighten sell limits. After cancel, no new placements for the day.
GRID_EOD_SWEEP_MINUTES_BEFORE = 10
# Minutes before close to market-sell any remaining inventory. Trumps
# the per-instrument exit ladder.
GRID_EOD_MARKET_SELL_MINUTES_BEFORE = 5

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
# Maximum orders placed per minute across all instruments. Spreads
# placement of 9 startup orders (3 instruments x 3 tiers) over time
# instead of hammering Avanza at once.
GRID_MAX_ORDERS_PER_MIN = 10
# Seconds between successive order placements within a single cycle.
GRID_ORDER_PLACE_DELAY_S = 0.6

# ---------------------------------------------------------------------------
# Active instruments map — orderbook IDs per (ticker, direction)
# ---------------------------------------------------------------------------
# Maintained in lockstep with data/fin_fish_config.py PREFERRED_INSTRUMENTS
# (which carries name + spread metadata). When swapping a preferred cert
# here, update fin_fish_config too.
GRID_ACTIVE_INSTRUMENTS = {
    "XAG-USD": {"LONG": "1650161", "SHORT": "2286417"},
    "XAU-USD": {"LONG": "738811", "SHORT": "1047859"},
    "OIL-USD": {"LONG": "2367797", "SHORT": "2367803"},
}

# ---------------------------------------------------------------------------
# State files
# ---------------------------------------------------------------------------
GRID_STATE_FILE = "data/grid_fisher_state.json"
GRID_DECISIONS_LOG = "data/grid_fisher_decisions.jsonl"

# ---------------------------------------------------------------------------
# State-schema version. Bump when state-file shape changes; older files
# are reset (and logged) rather than crashing.
# ---------------------------------------------------------------------------
GRID_STATE_SCHEMA_VERSION = 1
