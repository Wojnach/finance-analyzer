"""Instrument profiles for smart fishing — per-metal signal trust and behavior.

Each metal gets a "personality" defining:
- Which signals to trust (>70% historical accuracy on this ticker)
- Which signals to ignore (<45% accuracy, noise)
- Cross-asset drivers with correlation and lead time
- Regime-specific behavior (direction bias, TP multiplier)
- Precomputed context file path
- Typical volatility characteristics

Usage:
    from portfolio.instrument_profile import get_profile
    profile = get_profile("XAG-USD")
    trusted = profile["trusted_signals"]
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Signal trust tiers — derived from signal_reliability in agent_summary
# Updated periodically based on accuracy_cache.json
# ---------------------------------------------------------------------------

_SILVER_TRUSTED = [
    "econ_calendar",       # 94.7% (396 samples)
    "fear_greed",          # 91.5% (212)
    "claude_fundamental",  # 85.4% (247)
    "momentum_factors",    # 76.7% (630)
    "structure",           # 76.4% (467)
    "mean_reversion",      # 72.9% (3d horizon)
    "fibonacci",           # 68.2% (decent, context-dependent)
]

_SILVER_IGNORED = [
    "sentiment",           # 4.5% (89) — noise
    "ministral",           # 18.9% (95) — LLM fails on silver
    "oscillators",         # 40.6% (143) — below gate
    "custom_lora",         # disabled globally
    "ml",                  # disabled globally
]

_GOLD_TRUSTED = [
    "econ_calendar",       # high accuracy on metals broadly
    "macro_regime",        # gold responds to macro shifts
    "claude_fundamental",  # LLM analysis works for gold
    "fibonacci",           # structural levels reliable
    "structure",           # breakout detection
    "smart_money",         # institutional flow detection
]

_GOLD_IGNORED = [
    "sentiment",           # noise for metals
    "custom_lora",         # disabled
    "ml",                  # disabled
]

# ---------------------------------------------------------------------------
# Cross-asset drivers — what moves this instrument and how fast
# ---------------------------------------------------------------------------

_SILVER_CROSS_ASSETS = {
    "DXY": {
        "ticker": "DX-Y.NYB",
        "correlation": -0.65,
        "lead_minutes": 15,
        "description": "Strong USD = weak silver",
        "threshold_pct": 0.3,  # alert if DXY moves >0.3%
    },
    "copper": {
        "ticker": "HG=F",
        "correlation": 0.55,
        "lead_minutes": 30,
        "description": "Industrial demand proxy — copper leads silver",
        "threshold_pct": 1.5,
    },
    "gold": {
        "ticker": "GC=F",
        "correlation": 0.85,
        "lead_minutes": 5,
        "description": "Primary correlation — gold leads silver in most moves",
        "threshold_pct": 0.5,
    },
    "gold_silver_ratio": {
        "ticker": None,  # computed from GC=F / SI=F
        "mean": 62,
        "revert_threshold": 5,  # alert if ratio deviates >5 from mean
        "description": "High ratio = silver undervalued relative to gold",
    },
    "oil": {
        "ticker": "CL=F",
        "correlation": 0.30,
        "lead_minutes": 60,
        "description": "Inflation proxy — oil up = metals up",
        "threshold_pct": 2.0,
    },
}

_GOLD_CROSS_ASSETS = {
    "DXY": {
        "ticker": "DX-Y.NYB",
        "correlation": -0.70,
        "lead_minutes": 10,
        "description": "Strong USD = weak gold (strongest inverse correlation)",
        "threshold_pct": 0.3,
    },
    "real_yields": {
        "ticker": "^TNX",
        "correlation": -0.60,
        "lead_minutes": 60,
        "description": "Rising real yields = gold opportunity cost rises",
        "threshold_pct": 0.05,  # yield moves in absolute terms
    },
    "VIX": {
        "ticker": "^VIX",
        "correlation": 0.40,
        "lead_minutes": 15,
        "description": "Fear gauge — VIX spike = gold safe haven bid",
        "threshold_pct": 5.0,
    },
    "SPY": {
        "ticker": "SPY",
        "correlation": -0.25,
        "lead_minutes": 30,
        "description": "Risk-off = gold up, but weak correlation",
        "threshold_pct": 0.8,
    },
}

# ---------------------------------------------------------------------------
# Full instrument profiles
# ---------------------------------------------------------------------------

PROFILES: dict[str, dict[str, Any]] = {
    "XAG-USD": {
        "name": "Silver",
        "binance_symbol": "XAGUSDT",
        "trusted_signals": _SILVER_TRUSTED,
        "ignored_signals": _SILVER_IGNORED,
        "cross_asset_drivers": _SILVER_CROSS_ASSETS,
        "regime_behaviors": {
            "trending-up": {
                "preferred_direction": "LONG",
                "tp_multiplier": 1.5,
                "conviction_boost": 10,
            },
            "trending-down": {
                "preferred_direction": "SHORT",
                "tp_multiplier": 1.5,
                "conviction_boost": 10,
            },
            "ranging": {
                "preferred_direction": "BOTH",
                "tp_multiplier": 1.0,
                "conviction_boost": 0,
            },
            "high-vol": {
                "preferred_direction": "SHORT",  # vol expansion favors shorts
                "tp_multiplier": 2.0,
                "conviction_boost": -5,  # lower conviction in chaos
            },
        },
        "precompute_file": "data/silver_deep_context.json",
        "prophecy_key": "silver_bull_2026",
        "typical_daily_range_pct": 5.0,
        "typical_hourly_vol_pct": 0.4,
        "margin_hike_risk_levels": [100, 120],  # CME hikes near these
        "overnight_correlation_with_asia": 0.3,
    },
    "XAU-USD": {
        "name": "Gold",
        "binance_symbol": "XAUUSDT",
        "trusted_signals": _GOLD_TRUSTED,
        "ignored_signals": _GOLD_IGNORED,
        "cross_asset_drivers": _GOLD_CROSS_ASSETS,
        "regime_behaviors": {
            "trending-up": {
                "preferred_direction": "LONG",
                "tp_multiplier": 1.3,
                "conviction_boost": 10,
            },
            "trending-down": {
                "preferred_direction": "SHORT",
                "tp_multiplier": 1.3,
                "conviction_boost": 10,
            },
            "ranging": {
                "preferred_direction": "BOTH",
                "tp_multiplier": 1.0,
                "conviction_boost": 0,
            },
            "high-vol": {
                "preferred_direction": "LONG",  # gold is safe haven
                "tp_multiplier": 1.5,
                "conviction_boost": 5,
            },
        },
        "precompute_file": "data/gold_deep_context.json",
        "prophecy_key": None,
        "typical_daily_range_pct": 2.9,
        "typical_hourly_vol_pct": 0.25,
        "margin_hike_risk_levels": [],
        "overnight_correlation_with_asia": 0.5,  # gold more Asia-sensitive
    },
}


def get_profile(ticker: str) -> dict[str, Any] | None:
    """Get instrument profile for a ticker, or None if not profiled."""
    return PROFILES.get(ticker)


def get_trusted_signals(ticker: str) -> list[str]:
    """Get list of trusted signal names for this ticker."""
    profile = PROFILES.get(ticker)
    return profile["trusted_signals"] if profile else []


def get_ignored_signals(ticker: str) -> list[str]:
    """Get list of signals to ignore for this ticker."""
    profile = PROFILES.get(ticker)
    return profile["ignored_signals"] if profile else []


def get_cross_asset_drivers(ticker: str) -> dict[str, dict]:
    """Get cross-asset driver configuration for this ticker."""
    profile = PROFILES.get(ticker)
    return profile["cross_asset_drivers"] if profile else {}


def get_regime_behavior(ticker: str, regime: str) -> dict[str, Any]:
    """Get regime-specific behavior for this ticker."""
    profile = PROFILES.get(ticker)
    if not profile:
        return {"preferred_direction": "BOTH", "tp_multiplier": 1.0, "conviction_boost": 0}
    return profile["regime_behaviors"].get(
        regime,
        {"preferred_direction": "BOTH", "tp_multiplier": 1.0, "conviction_boost": 0},
    )


def format_profile_briefing(ticker: str, signal_data: dict | None = None) -> str:
    """Format a human-readable instrument briefing.

    Parameters
    ----------
    ticker : str
        Instrument ticker (e.g., "XAG-USD")
    signal_data : dict, optional
        Signal reliability data from agent_summary_compact.json

    Returns
    -------
    str
        Formatted briefing text
    """
    profile = PROFILES.get(ticker)
    if not profile:
        return f"No profile available for {ticker}"

    lines = [
        f"  Instrument: {profile['name']} ({ticker})",
        f"  Typical daily range: {profile['typical_daily_range_pct']:.1f}%",
        f"  Hourly vol: {profile['typical_hourly_vol_pct']:.2f}%",
    ]

    # Signal trust tiers
    if signal_data:
        reliability = signal_data.get("signal_reliability", {}).get(ticker, {})
        if reliability:
            trusted_with_acc = []
            for sig in profile["trusted_signals"]:
                acc_data = reliability.get(sig, {})
                if acc_data:
                    acc = acc_data.get("accuracy", 0)
                    n = acc_data.get("total", 0)
                    trusted_with_acc.append(f"{sig} ({acc:.0%}/{n})")
                else:
                    trusted_with_acc.append(sig)
            lines.append(f"  Trusted signals: {', '.join(trusted_with_acc)}")
        else:
            lines.append(f"  Trusted signals: {', '.join(profile['trusted_signals'])}")
    else:
        lines.append(f"  Trusted signals: {', '.join(profile['trusted_signals'])}")

    lines.append(f"  Ignored signals: {', '.join(profile['ignored_signals'])}")

    # Cross-asset drivers
    lines.append("  Cross-asset drivers:")
    for name, driver in profile["cross_asset_drivers"].items():
        corr = driver.get("correlation", 0)
        lead = driver.get("lead_minutes", 0)
        desc = driver.get("description", "")
        if lead:
            lines.append(f"    {name}: corr {corr:+.2f}, leads {lead}min -- {desc}")
        else:
            lines.append(f"    {name}: {desc}")

    # Prophecy
    if profile.get("prophecy_key"):
        lines.append(f"  Active prophecy: {profile['prophecy_key']}")

    return "\n".join(lines)
