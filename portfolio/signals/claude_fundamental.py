"""Claude Fundamental Signal (#28) — three-tier LLM cascade for fundamental analysis.

Tier 1 (Haiku):  Quick directional pulse, every ~1 min
Tier 2 (Sonnet): Full 5-dimension fundamental analysis, every ~10 min
Tier 3 (Opus):   Deep conviction with cross-asset reasoning, every ~30 min

Highest-tier fresh analysis wins (Opus > Sonnet > Haiku).
Sub-signals: fundamental_quality, sector_positioning, valuation,
             catalyst_assessment, macro_sensitivity.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import pandas as pd

from portfolio.signal_utils import majority_vote

logger = logging.getLogger("portfolio.signals.claude_fundamental")
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

_MAX_CONFIDENCE = 0.7

SUB_SIGNAL_NAMES = [
    "fundamental_quality",
    "sector_positioning",
    "valuation",
    "catalyst_assessment",
    "macro_sensitivity",
]

# --- Three-tier cache ---
_cache = {
    "haiku":  {"results": {}, "ts": 0},
    "sonnet": {"results": {}, "ts": 0},
    "opus":   {"results": {}, "ts": 0},
}
_lock = threading.Lock()

_DEFAULT_HOLD = {
    "action": "HOLD",
    "confidence": 0.0,
    "sub_signals": {},
    "indicators": {},
}


def _get_cooldowns(config):
    """Get per-tier cooldowns from config with defaults."""
    cf = config.get("claude_fundamental", {})
    return {
        "haiku":  cf.get("haiku_cooldown", 60),
        "sonnet": cf.get("sonnet_cooldown", 600),
        "opus":   cf.get("opus_cooldown", 1800),
    }


def _get_models(config):
    """Get per-tier model aliases from config with defaults."""
    cf = config.get("claude_fundamental", {})
    return {
        "haiku":  cf.get("haiku_model", "haiku"),
        "sonnet": cf.get("sonnet_model", "sonnet"),
        "opus":   cf.get("opus_model", "opus"),
    }


def _get_timeouts(config):
    """Get per-tier CLI timeouts from config with defaults."""
    cf = config.get("claude_fundamental", {})
    return {
        "haiku":  cf.get("haiku_timeout", 30),
        "sonnet": cf.get("sonnet_timeout", 60),
        "opus":   cf.get("opus_timeout", 120),
    }


def _needs_refresh(tier, cooldowns):
    """Check if a tier's cache has expired."""
    return time.time() - _cache[tier]["ts"] > cooldowns[tier]


def _build_ticker_grid(summary):
    """Build compact ticker grid from agent_summary_compact data."""
    lines = []
    tickers = summary.get("tickers", {})
    for ticker, data in tickers.items():
        price = data.get("price_usd", data.get("price", "?"))
        rsi_val = data.get("rsi", "?")
        macd_val = data.get("macd_hist", "?")
        regime = data.get("regime", "?")
        consensus = data.get("consensus", "HOLD")
        buy_count = data.get("buy_count", 0)
        sell_count = data.get("sell_count", 0)
        hold_count = data.get("hold_count", data.get("abstain_count", 0))
        vol_ratio = data.get("volume_ratio", "?")
        lines.append(
            f"{ticker}: ${price} RSI={rsi_val} MACD={macd_val} "
            f"regime={regime} vol={vol_ratio} "
            f"consensus={consensus} ({buy_count}B/{sell_count}S/{hold_count}H)"
        )
    return "\n".join(lines)


def _build_macro_block(macro):
    """Build macro context string from macro data."""
    if not macro:
        return "Macro data unavailable."
    parts = []
    dxy = macro.get("dxy", {})
    if dxy:
        parts.append(f"DXY: {dxy.get('value', '?')} ({dxy.get('change_5d', '?')} 5d)")
    treasury = macro.get("treasury", {})
    if treasury:
        y10 = treasury.get("10y", "?")
        y2 = treasury.get("2y", "?")
        spread = treasury.get("2s10s", "?")
        parts.append(f"10Y: {y10}% | 2Y: {y2}% | 2s10s: {spread}")
    fed = macro.get("fed", {})
    if fed:
        parts.append(f"FOMC: {fed.get('next_date', '?')} ({fed.get('days_until', '?')}d)")
    fg = macro.get("fear_greed", {})
    if fg:
        parts.append(f"F&G: {fg.get('crypto', '?')}/{fg.get('stock', '?')}")
    return " | ".join(parts) if parts else "Macro data unavailable."


def _build_haiku_prompt(summary, macro):
    """Build Haiku prompt for quick directional pulse."""
    from datetime import datetime, timezone

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    macro_line = _build_macro_block(macro)
    ticker_grid = _build_ticker_grid(summary)

    return f"""You are a fast fundamental screener for a trading system.
Date: {date_str}. {macro_line}

For each ticker below, give a quick fundamental directional read.
Focus ONLY on: business quality, sector momentum, and obvious valuation.
Do NOT duplicate technical analysis — other signals handle RSI, MACD, etc.

Use your knowledge of each company's fundamentals, earnings trajectory,
competitive position, and sector dynamics. This is what makes you valuable —
technical signals cannot see business quality or upcoming catalysts.

Tickers:
{ticker_grid}

Respond with ONLY a JSON object (no markdown, no explanation):
{{"TICKER": {{"action": "BUY|SELL|HOLD", "confidence": 0.0-0.7}}, ...}}

Rules:
- confidence max 0.7
- HOLD with 0.0 confidence if no strong fundamental view
- BUY only if fundamentals are clearly positive (strong earnings, moat, tailwinds)
- SELL only if fundamentals are clearly negative (deteriorating margins, headwinds)
"""


def _build_sonnet_prompt(summary, macro):
    """Build Sonnet prompt for full 5-dimension fundamental analysis."""
    from datetime import datetime, timezone

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    macro_line = _build_macro_block(macro)
    ticker_grid = _build_ticker_grid(summary)

    return f"""You are a fundamental analyst signal for a trading system.
Date: {date_str}. {macro_line}

For each ticker, evaluate 5 fundamental dimensions:
1. fundamental_quality — earnings trajectory, margins, competitive moat, revenue growth
2. sector_positioning — sector tailwinds/headwinds, AI cycle position, rotation dynamics
3. valuation — price relative to growth, analyst consensus targets, historical P/E norms
4. catalyst_assessment — upcoming earnings, product launches, regulatory events, M&A
5. macro_sensitivity — how DXY/yields/FOMC specifically affect THIS name

Do NOT duplicate technical analysis. Your value is FUNDAMENTAL knowledge that
candlestick patterns cannot capture — business quality, competitive dynamics,
earnings trajectory, sector rotation, and macro sensitivity per name.

Tickers:
{ticker_grid}

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "TICKER": {{
    "sub_signals": {{
      "fundamental_quality": "BUY|SELL|HOLD",
      "sector_positioning": "BUY|SELL|HOLD",
      "valuation": "BUY|SELL|HOLD",
      "catalyst_assessment": "BUY|SELL|HOLD",
      "macro_sensitivity": "BUY|SELL|HOLD"
    }},
    "reasoning": "1 sentence"
  }},
  ...
}}

Rules:
- Each sub_signal votes independently: BUY, SELL, or HOLD
- HOLD means no strong fundamental view on that dimension
- Be specific in reasoning — name the catalyst, the moat, the headwind
"""


def _build_opus_prompt(summary, macro, portfolios):
    """Build Opus prompt for deep conviction with cross-asset reasoning."""
    from datetime import datetime, timezone

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    macro_line = _build_macro_block(macro)
    ticker_grid = _build_ticker_grid(summary)

    # Build portfolio context
    portfolio_lines = []
    for pf_name, pf_data in (portfolios or {}).items():
        label = "Patient" if "bold" not in pf_name else "Bold"
        cash = pf_data.get("cash_sek", "?")
        holdings = pf_data.get("holdings", {})
        held = ", ".join(
            f"{t} {h.get('shares', 0):.2f}sh@${h.get('avg_cost_usd', 0):.2f}"
            for t, h in holdings.items()
            if h.get("shares", 0) > 0
        ) or "none"
        portfolio_lines.append(f"{label}: {cash:.0f} SEK cash | Holdings: {held}")
    portfolio_ctx = "\n".join(portfolio_lines) if portfolio_lines else "No portfolio data."

    return f"""You are a deep fundamental conviction analyst for a trading system.
Date: {date_str}. {macro_line}

Portfolio context:
{portfolio_ctx}

For each ticker, evaluate 5 fundamental dimensions:
1. fundamental_quality — earnings trajectory, margins, competitive moat, revenue growth
2. sector_positioning — sector tailwinds/headwinds, AI cycle position, rotation dynamics
3. valuation — price relative to growth, analyst consensus targets, historical P/E norms
4. catalyst_assessment — upcoming earnings, product launches, regulatory events, M&A
5. macro_sensitivity — how DXY/yields/FOMC specifically affect THIS name

ADDITIONALLY, provide:
- Cross-asset reasoning: if BTC breaks down, how does MSTR follow? If semis rally, which names lead?
- Contrarian flags: where do fundamentals STRONGLY disagree with the technical consensus?
- Portfolio-aware sizing: are we too concentrated in one sector?

Do NOT duplicate technical analysis. Focus on what you know that charts cannot show.

Tickers:
{ticker_grid}

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "TICKER": {{
    "sub_signals": {{
      "fundamental_quality": "BUY|SELL|HOLD",
      "sector_positioning": "BUY|SELL|HOLD",
      "valuation": "BUY|SELL|HOLD",
      "catalyst_assessment": "BUY|SELL|HOLD",
      "macro_sensitivity": "BUY|SELL|HOLD"
    }},
    "conviction": 0.0-1.0,
    "reasoning": "2-3 sentences with specific fundamental justification",
    "contrarian_flag": true|false
  }},
  ...
}}

Rules:
- conviction 0.0-1.0 reflects how strongly fundamentals support a directional view
- contrarian_flag=true when fundamentals STRONGLY disagree with technical consensus
- Be specific: name the earnings beat, the AI contract, the tariff exposure, the yield sensitivity
"""


def _call_claude_cli(model, prompt, timeout=60):
    """Call claude CLI and return text response.

    Uses Claude Code Max subscription via ``claude -p``.
    """
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)  # prevent nested session error

    cmd = [
        "claude", "-p",
        "--model", model,
        "--output-format", "text",
        "--no-session-persistence",
    ]

    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (rc={result.returncode}): {result.stderr[:500]}"
        )

    return result.stdout


def _extract_json(text):
    """Robustly extract JSON from LLM response (handles markdown blocks, etc)."""
    # Try direct parse first
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding the outermost { ... }
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    logger.warning("Could not extract JSON from response: %.200s", text)
    return {}


def _parse_haiku_response(text):
    """Parse Haiku JSON → {ticker: {"action": ..., "confidence": ...}}."""
    data = _extract_json(text)
    results = {}
    for ticker, val in data.items():
        if not isinstance(val, dict):
            continue
        action = val.get("action", "HOLD").upper()
        if action not in ("BUY", "SELL", "HOLD"):
            action = "HOLD"
        conf = min(float(val.get("confidence", 0.0)), _MAX_CONFIDENCE)
        results[ticker] = {
            "action": action,
            "confidence": round(conf, 4),
            "sub_signals": {},
            "indicators": {"_tier": "haiku"},
        }
    return results


def _parse_sonnet_response(text):
    """Parse Sonnet JSON → {ticker: {full signal result with sub_signals}}."""
    data = _extract_json(text)
    results = {}
    for ticker, val in data.items():
        if not isinstance(val, dict):
            continue
        sub_signals = val.get("sub_signals", {})
        # Validate sub_signal values
        clean_subs = {}
        for name in SUB_SIGNAL_NAMES:
            vote = sub_signals.get(name, "HOLD").upper()
            if vote not in ("BUY", "SELL", "HOLD"):
                vote = "HOLD"
            clean_subs[name] = vote

        # Majority vote across sub-signals
        votes = list(clean_subs.values())
        action, conf = majority_vote(votes)
        conf = min(conf, _MAX_CONFIDENCE)

        results[ticker] = {
            "action": action,
            "confidence": round(conf, 4),
            "sub_signals": clean_subs,
            "indicators": {
                "_tier": "sonnet",
                "reasoning": val.get("reasoning", ""),
            },
        }
    return results


def _parse_opus_response(text):
    """Parse Opus JSON → {ticker: {full signal result + contrarian_flag}}."""
    data = _extract_json(text)
    results = {}
    for ticker, val in data.items():
        if not isinstance(val, dict):
            continue
        sub_signals = val.get("sub_signals", {})
        clean_subs = {}
        for name in SUB_SIGNAL_NAMES:
            vote = sub_signals.get(name, "HOLD").upper()
            if vote not in ("BUY", "SELL", "HOLD"):
                vote = "HOLD"
            clean_subs[name] = vote

        votes = list(clean_subs.values())
        action, conf = majority_vote(votes)
        conf = min(conf, _MAX_CONFIDENCE)

        # Opus provides its own conviction — use it to scale confidence
        opus_conviction = min(float(val.get("conviction", 0.5)), 1.0)
        # Blend: majority_vote confidence weighted by Opus conviction
        conf = min(conf * opus_conviction, _MAX_CONFIDENCE)

        results[ticker] = {
            "action": action,
            "confidence": round(conf, 4),
            "sub_signals": clean_subs,
            "indicators": {
                "_tier": "opus",
                "reasoning": val.get("reasoning", ""),
                "conviction": opus_conviction,
                "contrarian_flag": bool(val.get("contrarian_flag", False)),
            },
        }
    return results


def _refresh_tier(tier, context):
    """Refresh one tier's cache by calling the claude CLI."""
    config = context.get("config", {})
    models = _get_models(config)
    timeouts = _get_timeouts(config)

    # Read the compact summary
    summary_path = DATA_DIR / "agent_summary_compact.json"
    if not summary_path.exists():
        logger.warning("agent_summary_compact.json not found, skipping %s refresh", tier)
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    macro = context.get("macro")
    # Also try to get macro from summary if not in context
    if not macro:
        macro = summary.get("macro", {})

    if tier == "haiku":
        prompt = _build_haiku_prompt(summary, macro)
        raw = _call_claude_cli(models["haiku"], prompt, timeout=timeouts["haiku"])
        results = _parse_haiku_response(raw)
    elif tier == "sonnet":
        prompt = _build_sonnet_prompt(summary, macro)
        raw = _call_claude_cli(models["sonnet"], prompt, timeout=timeouts["sonnet"])
        results = _parse_sonnet_response(raw)
    else:  # opus
        portfolios = {}
        for pf in ("portfolio_state.json", "portfolio_state_bold.json"):
            pf_path = DATA_DIR / pf
            if pf_path.exists():
                portfolios[pf] = json.loads(pf_path.read_text(encoding="utf-8"))
        prompt = _build_opus_prompt(summary, macro, portfolios)
        raw = _call_claude_cli(models["opus"], prompt, timeout=timeouts["opus"])
        results = _parse_opus_response(raw)

    with _lock:
        _cache[tier]["results"] = results
        _cache[tier]["ts"] = time.time()

    logger.info("Claude fundamental %s refreshed: %d tickers", tier, len(results))


def _get_best_result(ticker):
    """Cascade: Opus > Sonnet > Haiku. Return best available result for ticker."""
    # Prefer higher tier with a non-HOLD vote
    for tier in ("opus", "sonnet", "haiku"):
        result = _cache[tier]["results"].get(ticker)
        if result and result.get("action") != "HOLD":
            return result
    # All tiers say HOLD — return highest-tier available
    for tier in ("opus", "sonnet", "haiku"):
        result = _cache[tier]["results"].get(ticker)
        if result:
            return result
    return None


def compute_claude_fundamental_signal(df: pd.DataFrame, context: dict = None) -> dict:
    """Signal entry point — called once per ticker by signal_engine.

    Args:
        df: OHLCV DataFrame (not used directly — fundamentals come from LLM knowledge)
        context: dict with keys: ticker, config, macro

    Returns:
        dict with action, confidence, sub_signals, indicators
    """
    if context is None:
        return dict(_DEFAULT_HOLD)

    config = context.get("config", {})
    cf_config = config.get("claude_fundamental", {})
    if not cf_config.get("enabled", True):
        return dict(_DEFAULT_HOLD)

    cooldowns = _get_cooldowns(config)

    # Refresh any tier whose cooldown has expired (double-check locking)
    for tier in ("haiku", "sonnet", "opus"):
        if _needs_refresh(tier, cooldowns):
            with _lock:
                if _needs_refresh(tier, cooldowns):
                    try:
                        _refresh_tier(tier, context)
                    except Exception as e:
                        logger.warning("Claude fundamental %s failed: %s", tier, e)

    # Cascade lookup for this ticker
    ticker = context.get("ticker", "")
    result = _get_best_result(ticker)
    if result:
        return dict(result)  # copy to avoid mutation
    return dict(_DEFAULT_HOLD)
