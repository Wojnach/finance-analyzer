"""Single source of truth for all ticker lists, source mappings, and symbol constants.

Every module that needs ticker definitions should import from here instead
of maintaining its own copy.
"""

import re
from functools import lru_cache
from pathlib import Path

# ── Tier 1: Full signals (30 signals, 7 timeframes) ──────────────────────

SYMBOLS = {
    # Crypto (Binance spot)
    "BTC-USD": {"binance": "BTCUSDT"},
    "ETH-USD": {"binance": "ETHUSDT"},
    # Metals (Binance futures)
    "XAU-USD": {"binance_fapi": "XAUUSDT"},
    "XAG-USD": {"binance_fapi": "XAGUSDT"},
    # US Equities (Alpaca IEX) — MSTR kept as BTC NAV-premium reference for metals_loop
    "MSTR": {"alpaca": "MSTR"},
    # Removed Mar 15: AMD, GOOGL, AMZN, AAPL, AVGO, META, SOUN, LMT
    # Removed Apr 09: PLTR, NVDA, MU, SMCI, TSM, TTWO, VRT
    #   Reduces main loop load to stay under 60s cadence. Cycle p50 was 143s with
    #   12 tickers — dropping to 5 is expected to bring p50 under target. MSTR retained
    #   because data/metals_loop.py uses it for BTC NAV-premium tracking.
}

# ── Asset-class subsets ───────────────────────────────────────────────────

CRYPTO_SYMBOLS = {"BTC-USD", "ETH-USD"}
METALS_SYMBOLS = {"XAU-USD", "XAG-USD"}
STOCK_SYMBOLS = {"MSTR"}

# All known tickers (union of all subsets)
ALL_TICKERS = CRYPTO_SYMBOLS | METALS_SYMBOLS | STOCK_SYMBOLS

# ── Derived mappings (all from SYMBOLS — single source of truth) ─────────

BINANCE_SPOT_MAP = {
    t: sym for t, src in SYMBOLS.items()
    for k, sym in src.items() if k == "binance"
}
BINANCE_FAPI_MAP = {
    t: sym for t, src in SYMBOLS.items()
    for k, sym in src.items() if k == "binance_fapi"
}
BINANCE_MAP = {**BINANCE_SPOT_MAP, **BINANCE_FAPI_MAP}

# Ticker -> (source_type, symbol) mapping (used by macro_context)
TICKER_SOURCE_MAP = {
    t: next(iter(src.items())) for t, src in SYMBOLS.items()
}

# Yahoo Finance symbol mapping — stock tickers map to themselves
YF_MAP = {t: t for t in STOCK_SYMBOLS}

# ── Signal names (used by outcome_tracker, accuracy_stats) ───────────────
# Canonical source is portfolio.signal_registry.get_signal_names().
# This static list is kept for backward compatibility with modules that
# import SIGNAL_NAMES directly (outcome_tracker, accuracy_stats).

# Signals that are force-HOLD (disabled due to poor accuracy).
# Kept in SIGNAL_NAMES for historical tracking but excluded from active reports.
DISABLED_SIGNALS = {
    "ml",               # 41.7% accuracy (1714 sam) — worse than coin flip
    "fibonacci",        # 2026-04-29: 43.6% at 1d (17024 sam), 43.3% at 3h (8811 sam).
                        # Consistently below coin flip across ALL horizons and tickers
                        # with massive sample size. Was accuracy-gated but still computed
                        # every cycle (~50ms wasted). Formal disable saves CPU.
    # "cot_positioning" re-enabled 2026-04-13 for shadow validation (was
    # force-HOLD pending live validation, 0 samples). COT is a weekly signal
    # (CFTC Friday release) — expected to contribute mostly at 3d/5d horizons
    # where the system already has edge (XAG 5d consensus 61.2%). The
    # existing accuracy gate in signal_engine.py auto-disables any signal
    # below 45% accuracy once 30+ samples accumulate, so re-enabling is
    # self-correcting.
    "futures_basis",    # 0 accuracy samples — pending live validation
    "hurst_regime",     # pending live validation (added 2026-04-11)
    "shannon_entropy",  # pending live validation (added 2026-04-12)
    "vix_term_structure",  # pending live validation (added 2026-04-13)
    "gold_real_yield_paradox",  # pending live validation (added 2026-04-14)
    "cross_asset_tsmom",  # pending live validation (added 2026-04-15)
    "copper_gold_ratio",  # pending live validation (added 2026-04-17)
    # "statistical_jump_regime" RE-ENABLED 2026-04-29: 52.7% accuracy (110 sam)
    # at 1d — above 47% gate, marginal but worth live validation. Shadow-safe
    # since 2026-04-18. If it degrades below 47% the accuracy gate auto-disables.
    "network_momentum",  # pending live validation (added 2026-04-19)
    "ovx_metals_spillover",  # pending live validation (added 2026-04-20)
    "xtrend_equity_spillover",  # pending live validation (added 2026-04-21)
    "complexity_gap_regime",  # pending live validation (added 2026-04-22)
    "realized_skewness",  # KILLED 2026-04-29: 33.3% at 1d (90 sam). Below coin flip.
    "mahalanobis_turbulence",  # pending live validation (added 2026-04-24)
    # "crypto_evrp" RE-ENABLED 2026-05-18: 80.5% 1d_recent (77 sam), 92.4% 3d
    # (105 sam), 66.7% 1d all-time. Anti-correlated (0% agreement) with the
    # degraded macro_external cluster (crypto_macro 32.8%, credit_spread 19.3%).
    # Strongest shadow-to-live promotion candidate. Accuracy gate auto-disables
    # if performance degrades below 47%.
    "hash_ribbons",  # pending live validation (added 2026-04-26)
    "drift_regime_gate",  # pending live validation (added 2026-04-28)
    "vol_ratio_regime",  # pending live validation (added 2026-04-29)
    "residual_pair_reversion",  # pending live validation (added 2026-04-30)
    "williams_vix_fix",  # pending live validation (added 2026-05-01)
    "treasury_risk_rotation",  # pending live validation (added 2026-05-07)
    "intraday_seasonality",  # pending live validation (added 2026-05-08)
    "cubic_trend_persistence",  # pending live validation (added 2026-05-09)
    "vwap_zscore_mr",  # pending live validation (added 2026-05-10)
    "gold_overnight_bias",  # pending live validation (added 2026-05-11)
    "metals_vrp",  # pending live validation (added 2026-05-12)
    "breakeven_inflation_momentum",  # pending live validation (added 2026-05-13)
    "trend_slope_momentum",  # pending live validation (added 2026-05-14)
    "ttm_squeeze",  # pending live validation (added 2026-05-16)
    "tsi_chop_mr",  # pending live validation (added 2026-05-17)
    "amihud_illiquidity_regime",  # pending live validation (added 2026-05-18)
    "absorption_ratio_regime",  # pending live validation (added 2026-05-19)
    "funding",          # 2026-05-13: 30.8% at 1d (743 sam). BUY-only signal, all wrong.
                        # 0% activation on all tickers in last 50 entries — phantom voter
                        # that escapes accuracy gate because it rarely votes. When it does
                        # vote BUY, it's wrong 69.2% of the time. Horizon-gated to 3h/4h
                        # per CLAUDE.md but still enabled at other horizons. Full disable.
    "macro_regime",     # 2026-05-13: 47.0% at 1d (29,626 sam). BUY accuracy 41.3% —
                        # actively harmful on BUY side. 88% activation on XAG where it
                        # consistently votes BUY in ranging market. 29K samples confirm
                        # this is noise, not bad luck. Should be caught by high-sample gate
                        # (50% for 7K+ sam) but at 47% it sits between the standard 47%
                        # gate and the 50% high-sample gate. Formal disable to eliminate
                        # the single biggest noise source in XAG consensus.
    "calendar",         # 2026-05-09: 29.3% recent accuracy — structural BUY bias
                        # (6/8 sub-signals BUY-only). Actively harmful. Was per-horizon
                        # blacklisted at 1d but still voted at other horizons.
    "futures_flow",     # 2026-05-07: 38.3% at 1d (2168 sam). Actively harmful —
                        # 12pp worse than coin flip. In cross_asset_flow cluster
                        # but still wastes compute. Was accuracy-gated at runtime
                        # but formal disable saves ~50ms/cycle.
    "trend",            # 2026-05-07: 46.1% at 1d (17880 sam), 40.3% at 3h.
                        # Massive sample, consistently below threshold across ALL
                        # horizons. 92-100% correlated with ema/macro_regime in
                        # pure_trend cluster. In ranging regime (current) this is
                        # pure noise. ema (50.0%) is the cluster leader.
    "macd",             # 2026-05-07: 44.2% at 1d (6136 sam), 43.7% at 3h.
                        # Below threshold across all horizons. Only 5.3% activation
                        # on XAG. In oscillator_trend cluster where momentum_factors
                        # (53.2%) is the better signal.
    # "econ_calendar" RE-ENABLED 2026-04-23. BUG-218 fixed: added post_event_relief
    # sub-signal that emits BUY after high-impact events pass (4-24h relief window)
    # and during event-free calm windows (>72h to next event). The composite is now
    # 5 sub-signals (3 SELL + 1 BUY + 1 neutral) instead of 4 SELL-only.
    # 62.6% accuracy before disabling. Accuracy gate will auto-gate if BUY
    # signals degrade the composite.
    "orderbook_flow",   # 2026-04-11: 51.1% accuracy (360 sam), 93.3% activation rate,
                        # no recent data. Pure noise in every consensus decision.
                        # Re-evaluate after 2 weeks of accuracy data collection.
    # "forecast" RE-ENABLED 2026-04-21. The 36-39% accuracy measured on 2026-04-12
    # was polluted by Kronos voting 100% HOLD in shadow mode — Kronos occupied 3 of 6
    # slots in _health_weighted_vote whenever its subprocess succeeded, dragging every
    # composite vote toward HOLD regardless of Chronos's verdict. With Kronos retired
    # in portfolio/signals/forecast.py (same PR), the composite is now Chronos-only.
    # Chronos effective accuracy: 1h=45.4%, 24h=52.4% (4d ago). The 47% tiered
    # accuracy gate will force-HOLD 1h while letting 24h contribute. Forecast stayed
    # in this set for 10 days, which ALSO silenced forecast_predictions.jsonl and
    # forecast_health.jsonl because signal_engine.py skips disabled signals before
    # invocation — so we lost all shadow/health visibility while the signal was off.
    # Re-enabling restores both the signal and the logging. If accuracy degrades
    # again post-Kronos-retire, move into REGIME_GATED_SIGNALS (24h-only) rather
    # than re-disabling blindly.
    "oscillators",      # 2026-04-14: below 45% on ALL tickers at 1d (BTC 35.8%, ETH 36.3%,
                        # XAG 34.9%, XAU 40.2%, MSTR 42.6%; 5065 total sam). Also weak at
                        # 3h (34-45% per ticker). Regime-gated in ranging but noise everywhere.
    "smart_money",      # 2026-04-24: below 40% on ALL Tier 1 tickers at 1d — BTC 39.8% (123),
                        # ETH 34.9% (146), MSTR 33.3% (264), XAU N/A. Not salvageable.
                        # Cross-ticker consistent failure. 51.6% aggregate masks per-ticker disaster.
    "claude_fundamental",  # 2026-05-03: CRASHED to 19.8% recent 1d (222 sam) from 57.9%
                        # all-time. Root cause: Opus tier has 95% BUY bias (76/80 votes BUY),
                        # Sonnet 73% BUY bias. Haiku 83% abstention (useless). In ranging
                        # market these BUY calls are mostly wrong. Bias detectors (added
                        # 2026-04-25) couldn't prevent structural LLM bullish lean.
                        # Re-enable after fixing bias detector thresholds.
    "sentiment",        # 2026-05-03: 33.8% at 3h recent (3629 sam), 45.9% all-time (39579 sam).
                        # CryptoBERT predictions are noise. High-volume signal actively hurting
                        # consensus. Always in macro_external cluster but dragging down peers.
    "volatility_sig",   # 2026-05-12: 46.1% at 1d (7933 sam), 43.9% recent (628 sam).
                        # Below 47% gate with massive sample — statistically certain.
                        # 47.2% at 3h (marginal). No horizon provides reliable edge.
                        # Was regime-gated in ranging but fails across all regimes.
    "dxy_cross_asset",  # 2026-05-12: 43.3% at 1d (30 sam). SELL accuracy catastrophic
                        # at 14.3%. Metals-only signal. Not salvageable at current sample.
    "forecast",         # 2026-05-12: 25.6% recent 1d (223 sam), crashed from 47.0%
                        # all-time (6966 sam). Chronos-2 time-series model failing in
                        # ranging regime — oscillating predictions flip rapidly.
                        # Previous comment said "move to REGIME_GATED_SIGNALS (24h-only)"
                        # but 25.6% is catastrophic even at 24h. Full disable until
                        # regime transitions to trending and accuracy recovers.
                        # Forecast logging (forecast_predictions.jsonl) will be silenced
                        # again — acceptable tradeoff vs consensus pollution.
    "heikin_ashi",      # 2026-05-15: 43.5% blended 1d (27562 sam), 49.1% all-time.
                        # 52% activation rate — loud + below coin flip. 53% at 3h is
                        # marginal. Trend-following signal failing in persistent ranging
                        # regime. High-sample gate (50% for 7K+) already blocks at 1d.
    "candlestick",      # 2026-05-15: 44.1% blended 1d (12966 sam). 28% activation.
                        # Below gate at all horizons except 3h (51.6%). Pattern
                        # recognition signals degrade when volatility compresses.
    "volume",           # 2026-05-15: 45.4% blended 1d (24762 sam). 19% activation.
                        # Low activation limits damage but signal is structurally
                        # below coin flip. Volume confirmation adds noise not signal.
    "volume_flow",      # 2026-05-15: 47.2% blended 1d (65983 sam). 66% activation —
                        # highest-activation signal below gate. 66K samples at 47% is
                        # statistically conclusive. OBV/VWAP/A-D/CMF/MFI composite
                        # is noise at 1d. Marginal at 3h (47.6%).
    "ema",              # 2026-05-15: 46.2% blended 1d (17917 sam). 62% activation.
                        # High activation + below gate = biggest noise contributor.
                        # 59.7% at 3h is the only horizon with edge. In ranging regime
                        # EMA crossovers whipsaw constantly. Trend signal with no trend.
    "structure",        # 2026-05-15: 45.7% blended 1d (16652 sam). 42% activation.
                        # BUY-biased (25.5% vs 16.6%). High/Low breakout and Donchian
                        # signals fail in ranging markets. 49.9% at 3h (marginal).
    "connors_rsi2",     # 2026-05-19: NEW. RSI(2) ultra-short MR for BTC/ETH only.
                        # Shadow mode — accumulate accuracy data before enabling.
    "adx_regime_switch",  # 2026-05-19: NEW. ADX dual regime meta-signal. Detects
                        # trend/range transitions via ADX + DI spread. All assets.
                        # Shadow mode — accumulate accuracy data before enabling.
}
# 2026-04-11 research session changes:
# - orderbook_flow DISABLED: 93.3% active, 51.1% accuracy, 0 recent data. Noise.
# - credit_spread_risk ENABLED: 66.9% accuracy (257 sam), BUY 80.3%. Directional
#   gate at 40% will auto-gate SELL (49.1%) while allowing strong BUY votes.
# - crypto_macro ENABLED: 56.5% accuracy (1273 sam). BUY-biased (93%) so bias
#   penalty (0.5x) applies. Provides crypto-specific on-chain edge.
# funding: removed from DISABLED — 74.2% at 3h (535 samples) but 29.9% at 1d.
# Horizon-gated via REGIME_GATED_SIGNALS to only vote at 3h/4h.

# 2026-05-05: Surface the disable reason to the dashboard tooltip by parsing the
# inline comments next to each DISABLED_SIGNALS entry. Done via source-file
# parsing (rather than a parallel dict) so the comments stay the single source
# of truth. Falls back to None if the file shape changes.
_DISABLED_REASON_ENTRY_RE = re.compile(
    r'^(\s*)"([a-z_][a-z0-9_]*)"\s*,\s*(?:#\s*(.*))?$'
)
_DISABLED_REASON_CONT_RE = re.compile(r'^(\s+)#\s*(.*)$')


def _clean_disabled_reason(lines: list[str]) -> str:
    """Join continuation comments and trim to a single short summary."""
    if not lines:
        return ""
    text = " ".join(lines).strip()
    for sep in (". ", " — "):
        if sep in text:
            text = text.split(sep, 1)[0].rstrip(".")
            break
    return text[:160].rstrip()


@lru_cache(maxsize=1)
def _parse_disabled_reasons() -> dict[str, str]:
    """Parse the DISABLED_SIGNALS literal in this file into {name: reason}.

    A continuation comment is recognised when its `#` is indented strictly
    further than the entry name's column, which excludes flush-left
    separator comments (e.g. the commented-out re-enable notes) from
    bleeding into the previous entry's reason.
    """
    try:
        src = Path(__file__).resolve().read_text(encoding="utf-8")
    except OSError:
        return {}
    block_match = re.search(
        r'^DISABLED_SIGNALS\s*=\s*\{(.*?)^\}',
        src, re.MULTILINE | re.DOTALL,
    )
    if not block_match:
        return {}
    out: dict[str, str] = {}
    current: str | None = None
    current_lines: list[str] = []
    entry_indent = 0
    for raw in block_match.group(1).splitlines():
        m_entry = _DISABLED_REASON_ENTRY_RE.match(raw)
        if m_entry:
            if current is not None:
                out[current] = _clean_disabled_reason(current_lines)
            current = m_entry.group(2)
            entry_indent = len(m_entry.group(1))
            first = (m_entry.group(3) or "").strip()
            current_lines = [first] if first else []
            continue
        m_cont = _DISABLED_REASON_CONT_RE.match(raw)
        if m_cont and current is not None:
            indent = len(m_cont.group(1))
            if indent > entry_indent:
                txt = m_cont.group(2).strip()
                if txt:
                    current_lines.append(txt)
    if current is not None:
        out[current] = _clean_disabled_reason(current_lines)
    return out


def get_disabled_reason(signal_name: str) -> str | None:
    """Return a short reason for why `signal_name` is disabled, or None.

    Returns None for signals not in DISABLED_SIGNALS, and for disabled
    signals whose comment was empty or unparseable.
    """
    if signal_name not in DISABLED_SIGNALS:
        return None
    reasons = _parse_disabled_reasons()
    reason = reasons.get(signal_name)
    return reason if reason else None


# Signals that require local GPU inference.
# Skipped for US stocks outside market hours to save GPU resources.
# claude_fundamental excluded — uses remote API, has its own market-hours gate.
GPU_SIGNALS = frozenset({"ministral", "qwen3", "forecast"})

SIGNAL_NAMES = [
    "rsi",
    "macd",
    "ema",
    "bb",
    "fear_greed",
    "sentiment",
    "ministral",
    "ml",
    "funding",
    "volume",
    "qwen3",
    # custom_lora removed — disabled signal, was polluting accuracy stats
    # Enhanced composite signals
    "trend",
    "momentum",
    "volume_flow",
    "volatility_sig",
    "candlestick",
    "structure",
    "fibonacci",
    "smart_money",
    "oscillators",
    "heikin_ashi",
    "mean_reversion",
    "calendar",
    "macro_regime",
    "momentum_factors",
    "news_event",
    "econ_calendar",
    "forecast",
    "claude_fundamental",
    "futures_flow",
    "crypto_macro",
    "orderbook_flow",
    "metals_cross_asset",
    "dxy_cross_asset",
    "cot_positioning",
    "credit_spread_risk",
    "onchain",
    "futures_basis",
    "hurst_regime",
    "shannon_entropy",
    "vix_term_structure",
    "gold_real_yield_paradox",
    "cross_asset_tsmom",
    "copper_gold_ratio",
    "statistical_jump_regime",
    "network_momentum",
    "ovx_metals_spillover",
    "xtrend_equity_spillover",
    "complexity_gap_regime",
    "realized_skewness",
    "mahalanobis_turbulence",
    "crypto_evrp",
    "hash_ribbons",
    "drift_regime_gate",
    "vol_ratio_regime",
    "residual_pair_reversion",
    "williams_vix_fix",
    "treasury_risk_rotation",
    "intraday_seasonality",
    "cubic_trend_persistence",
    "vwap_zscore_mr",
    "btc_proxy",
    "metals_vrp",
    "breakeven_inflation_momentum",
    "trend_slope_momentum",
    "ttm_squeeze",
    "tsi_chop_mr",
    "amihud_illiquidity_regime",
    "absorption_ratio_regime",
]
