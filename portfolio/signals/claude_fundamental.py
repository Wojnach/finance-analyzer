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
import re
import threading
import time
from datetime import UTC
from pathlib import Path

import pandas as pd

from portfolio.file_utils import load_json
from portfolio.signal_utils import majority_vote

logger = logging.getLogger("portfolio.signals.claude_fundamental")
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

_MAX_CONFIDENCE = 0.7
_CF_LOG = DATA_DIR / "claude_fundamental_log.jsonl"

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
    """Get per-tier cooldowns from config with defaults.

    Defaults are tuned for accuracy tracking — need enough samples to
    measure hit rate. Haiku is cheap/fast so runs often. Opus is expensive
    so runs sparingly.
    """
    cf = config.get("claude_fundamental", {})
    return {
        "haiku":  cf.get("haiku_cooldown", 300),    # 5 min default
        "sonnet": cf.get("sonnet_cooldown", 1800),   # 30 min default
        "opus":   cf.get("opus_cooldown", 7200),      # 2h default
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
    tickers = summary.get("signals", summary.get("tickers", {}))
    if not tickers:
        logger.warning("claude_fundamental: empty ticker grid — summary has keys %s "
                        "but no 'signals' or 'tickers' data. Signal will return HOLD.",
                        list(summary.keys())[:5])
        return ""
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


# Earnings calendar cache (refreshed every 12h)
_earnings_cache = {"data": {}, "ts": 0}
_EARNINGS_CACHE_TTL = 43200  # 12 hours


def _get_earnings_calendar():
    """Fetch next earnings dates from yfinance for all stock tickers.

    Cached for 12h to avoid excessive Yahoo Finance requests.
    Returns dict of {ticker: {date, eps_estimate, days_until}}.
    """
    now = time.time()
    if now - _earnings_cache["ts"] < _EARNINGS_CACHE_TTL and _earnings_cache["data"]:
        return _earnings_cache["data"]

    from datetime import datetime

    try:
        from portfolio.tickers import STOCK_SYMBOLS
    except ImportError:
        return {}

    result = {}
    for ticker in STOCK_SYMBOLS:
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            cal = t.calendar
            if cal is None or (hasattr(cal, "empty") and cal.empty):
                continue
            if isinstance(cal, dict):
                dates = cal.get("Earnings Date", [])
                eps_avg = cal.get("Earnings Average")
            elif hasattr(cal, "loc"):
                dates = cal.loc["Earnings Date"] if "Earnings Date" in cal.index else []
                eps_avg = cal.loc["Earnings Average"] if "Earnings Average" in cal.index else None
            else:
                continue

            if dates:
                next_date = dates[0] if isinstance(dates, list) else dates
                if hasattr(next_date, "date"):
                    next_date = next_date.date()
                today = datetime.now(UTC).date()
                days_until = (next_date - today).days
                result[ticker] = {
                    "date": str(next_date),
                    "eps_estimate": float(eps_avg) if eps_avg is not None else None,
                    "days_until": days_until,
                }
        except Exception:
            continue

    _earnings_cache["data"] = result
    _earnings_cache["ts"] = now
    return result


def _build_fundamentals_block(ticker, fundamentals, tier="haiku", earnings=None):
    """Build a fundamentals data string for a ticker based on tier detail level.

    Args:
        ticker: Stock ticker symbol
        fundamentals: Dict of all fundamentals from alpha_vantage cache
        tier: "haiku" (one-liner), "sonnet"/"opus" (detailed block)

    Returns:
        Formatted string with fundamentals data, or empty string if none available.
    """
    fund = fundamentals.get(ticker) if fundamentals else None
    if not fund:
        return ""

    earn = (earnings or {}).get(ticker, {})

    if tier == "haiku":
        # One-liner: key metrics only
        parts = [ticker + ":"]
        pe = fund.get("pe_ratio")
        if pe is not None:
            parts.append(f"PE={pe:.1f}")
        rev_growth = fund.get("revenue_growth_yoy")
        if rev_growth is not None:
            parts.append(f"RevGrowth={rev_growth:+.0%}")
        target = fund.get("analyst_target")
        if target is not None:
            parts.append(f"Target=${target:.0f}")
        margin = fund.get("profit_margin")
        if margin is not None:
            parts.append(f"Margin={margin:.0%}")
        if earn.get("days_until") is not None:
            parts.append(f"Earnings={earn['days_until']}d")
        return " ".join(parts) if len(parts) > 1 else ""

    # Sonnet/Opus: detailed block
    lines = [f"  {ticker} Fundamentals:"]
    pe = fund.get("pe_ratio")
    fpe = fund.get("forward_pe")
    peg = fund.get("peg_ratio")
    if pe is not None or fpe is not None:
        pe_str = f"PE={pe:.1f}" if pe else "PE=N/A"
        fpe_str = f"FwdPE={fpe:.1f}" if fpe else "FwdPE=N/A"
        peg_str = f"PEG={peg:.2f}" if peg else ""
        lines.append(f"    Valuation: {pe_str} {fpe_str} {peg_str}".rstrip())
    eps = fund.get("eps")
    margin = fund.get("profit_margin")
    rev_growth = fund.get("revenue_growth_yoy")
    earn_growth = fund.get("earnings_growth_yoy")
    if eps is not None or margin is not None:
        eps_str = f"EPS=${eps:.2f}" if eps else ""
        margin_str = f"Margin={margin:.1%}" if margin else ""
        rg_str = f"RevGrowth={rev_growth:+.1%}" if rev_growth is not None else ""
        eg_str = f"EarnGrowth={earn_growth:+.1%}" if earn_growth is not None else ""
        lines.append(f"    Earnings: {eps_str} {margin_str} {rg_str} {eg_str}".rstrip())
    target = fund.get("analyst_target")
    ratings = fund.get("analyst_ratings", {})
    if target is not None or ratings:
        t_str = f"Target=${target:.2f}" if target else ""
        sb = ratings.get("strong_buy", 0)
        b = ratings.get("buy", 0)
        h = ratings.get("hold", 0)
        s = ratings.get("sell", 0)
        ss = ratings.get("strong_sell", 0)
        r_str = f"Analysts={sb}SB/{b}B/{h}H/{s}S/{ss}SS" if any([sb, b, h, s, ss]) else ""
        lines.append(f"    Consensus: {t_str} {r_str}".rstrip())
    w52h = fund.get("w52_high")
    w52l = fund.get("w52_low")
    beta = fund.get("beta")
    if w52h is not None or w52l is not None:
        range_str = f"52W=${w52l:.2f}-${w52h:.2f}" if w52l and w52h else ""
        beta_str = f"Beta={beta:.2f}" if beta else ""
        lines.append(f"    Range: {range_str} {beta_str}".rstrip())
    sector = fund.get("sector")
    industry = fund.get("industry")
    if sector:
        ind_str = f"/{industry}" if industry else ""
        lines.append(f"    Sector: {sector}{ind_str}")
    if earn.get("date"):
        earn_parts = [f"NextEarnings={earn['date']} ({earn.get('days_until', '?')}d)"]
        if earn.get("eps_estimate") is not None:
            earn_parts.append(f"EPS_Est=${earn['eps_estimate']:.2f}")
        lines.append(f"    Catalyst: {' '.join(earn_parts)}")

    return "\n".join(lines) if len(lines) > 1 else ""


def _get_fundamentals_data():
    """Load fundamentals from alpha_vantage cache (returns empty dict on failure)."""
    try:
        from portfolio.alpha_vantage import get_all_fundamentals
        return get_all_fundamentals()
    except Exception as e:
        logger.warning("Fundamentals fetch failed: %s", e, exc_info=True)
        return {}


def _build_haiku_prompt(summary, macro):
    """Build Haiku prompt for quick directional pulse."""
    from datetime import datetime

    date_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    macro_line = _build_macro_block(macro)
    ticker_grid = _build_ticker_grid(summary)

    # Add one-liner fundamentals for each ticker
    fundamentals = _get_fundamentals_data()
    earnings = _get_earnings_calendar()
    fund_lines = []
    tickers = summary.get("signals", summary.get("tickers", {}))
    for ticker in tickers:
        line = _build_fundamentals_block(ticker, fundamentals, tier="haiku", earnings=earnings)
        if line:
            fund_lines.append(line)
    fund_section = "\n".join(fund_lines) if fund_lines else ""
    fund_prompt = f"\n\nReal-time fundamentals (Alpha Vantage):\n{fund_section}" if fund_section else ""

    return f"""You are a fast fundamental screener for a trading system.
Date: {date_str}. {macro_line}

For each ticker below, give a quick fundamental directional read.
Focus ONLY on: business quality, sector momentum, and obvious valuation.
Do NOT duplicate technical analysis — other signals handle RSI, MACD, etc.

Use the real fundamental data provided below (P/E, revenue growth, analyst targets)
combined with your knowledge of each company's competitive position and sector dynamics.
{fund_prompt}

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
    from datetime import datetime

    date_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    macro_line = _build_macro_block(macro)
    ticker_grid = _build_ticker_grid(summary)

    # Add detailed fundamentals per ticker
    fundamentals = _get_fundamentals_data()
    earnings = _get_earnings_calendar()
    fund_blocks = []
    tickers = summary.get("signals", summary.get("tickers", {}))
    for ticker in tickers:
        block = _build_fundamentals_block(ticker, fundamentals, tier="sonnet", earnings=earnings)
        if block:
            fund_blocks.append(block)
    fund_section = "\n".join(fund_blocks) if fund_blocks else ""
    fund_prompt = f"\n\nReal fundamental data (Alpha Vantage — use these numbers, not estimates):\n{fund_section}" if fund_section else ""

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
{fund_prompt}

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
    from datetime import datetime

    date_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
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

    # Add detailed fundamentals + cross-sector comparison for Opus
    fundamentals = _get_fundamentals_data()
    earnings = _get_earnings_calendar()
    fund_blocks = []
    tickers = summary.get("signals", summary.get("tickers", {}))
    for ticker in tickers:
        block = _build_fundamentals_block(ticker, fundamentals, tier="opus", earnings=earnings)
        if block:
            fund_blocks.append(block)
    fund_section = "\n".join(fund_blocks) if fund_blocks else ""
    fund_prompt = f"\n\nReal fundamental data (Alpha Vantage — use these numbers, not estimates):\n{fund_section}" if fund_section else ""

    return f"""You are a deep fundamental conviction analyst for a trading system.
Date: {date_str}. {macro_line}

Portfolio context:
{portfolio_ctx}
{fund_prompt}

For each ticker, evaluate 5 fundamental dimensions:
1. fundamental_quality — earnings trajectory, margins, competitive moat, revenue growth
2. sector_positioning — sector tailwinds/headwinds, AI cycle position, rotation dynamics
3. valuation — price relative to growth, analyst consensus targets, historical P/E norms
4. catalyst_assessment — upcoming earnings, product launches, regulatory events, M&A
5. macro_sensitivity — how DXY/yields/FOMC specifically affect THIS name

ADDITIONALLY, provide:
- Cross-asset reasoning: if BTC breaks down, how does ETH follow? If semis rally, which names lead?
- Contrarian flags: where do fundamentals STRONGLY disagree with the technical consensus?
- Portfolio-aware sizing: are we too concentrated in one sector?
- Cross-sector valuation comparison using the real P/E, PEG, and margin data above

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

    Routes through ``claude_gate.invoke_claude_text()`` for kill switch,
    rate limiting, and invocation logging.  Falls back to raw subprocess
    only if the gate module cannot be imported (should not happen in
    normal operation).
    """
    from portfolio.claude_gate import invoke_claude_text

    text, success, exit_code = invoke_claude_text(
        prompt=prompt,
        caller=f"claude_fundamental_{model}",
        model=model,
        timeout=timeout,
    )
    if not success:
        raise RuntimeError(f"claude_gate returned exit_code={exit_code}")
    return text


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


def _journal_refresh(tier: str, results: dict) -> None:
    """Persist tier refresh results for accuracy tracking and debugging.

    2026-04-21 v2: LOG EVERYTHING. An earlier v1 of this function silently
    skipped rows that looked like abstentions (Haiku HOLD-0.0-empty-reason).
    That was the wrong call — those rows are load-bearing evidence of how
    often each tier declines to make a call, and the abstention rate itself
    is a health signal. The revised approach is to ALWAYS write the row but
    tag intentional abstentions with `is_abstention: true` so downstream
    analysis can filter at read-time without losing the data. See the
    `feedback_log_everything.md` memory for the full rationale.
    """
    import datetime as _dt

    from portfolio.file_utils import atomic_append_jsonl

    ts = _dt.datetime.now(_dt.UTC).isoformat()
    for ticker, result in results.items():
        action = result.get("action", "HOLD")
        confidence = result.get("confidence", 0.0)
        reasoning = result.get("indicators", {}).get("reasoning", "")
        is_abstention = (
            action == "HOLD" and confidence == 0.0 and not reasoning
        )
        entry = {
            "ts": ts,
            "tier": tier,
            "ticker": ticker,
            "action": action,
            "confidence": confidence,
            "sub_signals": result.get("sub_signals", {}),
            "reasoning": reasoning,
            "contrarian_flag": result.get("indicators", {}).get("contrarian_flag", False),
            "is_abstention": is_abstention,
        }
        try:
            atomic_append_jsonl(_CF_LOG, entry)
        except Exception as e:
            logger.warning("Failed to journal cf result: %s", e)


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
    summary = load_json(summary_path, default={})
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
                portfolios[pf] = load_json(pf_path, default={})
        prompt = _build_opus_prompt(summary, macro, portfolios)
        raw = _call_claude_cli(models["opus"], prompt, timeout=timeouts["opus"])
        results = _parse_opus_response(raw)

    with _lock:
        _cache[tier]["results"] = results
        _cache[tier]["ts"] = time.time()

    logger.info("Claude fundamental %s refreshed: %d tickers", tier, len(results))

    # Journal the results for accuracy tracking and debugging
    _journal_refresh(tier, results)


def _is_tier_biased(tier: str) -> bool:
    """Detect BUY or SELL bias from recent journal entries.

    If a tier has >75% of its recent non-abstention votes in one direction,
    it's structurally biased and should be downweighted in the cascade.
    Uses the last 30 journal entries for the tier.

    2026-04-25: Added to fix Sonnet (83% BUY) and Opus (78% BUY) bias that
    collapsed claude_fundamental from 59.4% to 37.9% recent accuracy.
    """
    _BIAS_THRESHOLD = 0.75
    _BIAS_MIN_SAMPLES = 10

    try:
        from portfolio.file_utils import load_jsonl_tail
        entries = load_jsonl_tail(_CF_LOG, max_entries=200)
    except Exception:
        return False

    tier_votes = [
        e.get("action", "HOLD")
        for e in entries
        if e.get("tier") == tier and not e.get("is_abstention", False)
    ]
    # Only check most recent entries
    tier_votes = tier_votes[-30:] if len(tier_votes) > 30 else tier_votes

    non_hold = [v for v in tier_votes if v != "HOLD"]
    if len(non_hold) < _BIAS_MIN_SAMPLES:
        return False

    from collections import Counter
    counts = Counter(non_hold)
    most_common_count = counts.most_common(1)[0][1]
    bias_rate = most_common_count / len(non_hold)

    if bias_rate > _BIAS_THRESHOLD:
        logger.info(
            "Claude fundamental %s tier biased: %.0f%% %s (%d non-HOLD votes)",
            tier, bias_rate * 100, counts.most_common(1)[0][0], len(non_hold),
        )
        return True
    return False


def _get_best_result(ticker):
    """Cascade: Opus > Sonnet > Haiku. Return best available result for ticker.

    2026-04-25: Added bias detection. When a higher tier has >75% BUY (or SELL)
    bias, its non-HOLD vote is treated as HOLD for cascade purposes. This lets
    Haiku's prudent HOLD win over Sonnet/Opus's structural optimism in ranging
    markets. The bias detector uses a rolling 30-entry window from the journal.
    """
    # Prefer higher tier with a non-HOLD vote, unless that tier is biased
    for tier in ("opus", "sonnet", "haiku"):
        result = _cache[tier]["results"].get(ticker)
        if result and result.get("action") != "HOLD":
            if not _is_tier_biased(tier):
                return result
            # Tier is biased — skip its non-HOLD vote, treat as HOLD
            logger.debug(
                "Skipping biased %s tier vote (%s) for %s",
                tier, result.get("action"), ticker,
            )
    # All tiers say HOLD (or biased tiers skipped) — return highest-tier available
    for tier in ("opus", "sonnet", "haiku"):
        result = _cache[tier]["results"].get(ticker)
        if result:
            return result
    return None


def _bg_refresh(tier, context):
    """Background refresh — runs in a daemon thread, never blocks the loop."""
    try:
        _refresh_tier(tier, context)
    except Exception as e:
        logger.warning("Claude fundamental %s bg-refresh failed: %s", tier, e)


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

    # Market hours gate — only refresh during EU+US hours (07:00-21:00 UTC weekdays).
    # Fundamentals don't change overnight and we don't want to waste claude calls.
    # Cached results from the last open-hours refresh are still served during off-hours.
    from portfolio.market_timing import get_market_state
    market_state, _, _ = get_market_state()
    skip_refresh = market_state in ("closed", "weekend", "holiday")

    cooldowns = _get_cooldowns(config)

    # Refresh in background thread — never block the signal loop.
    # Fundamentals change on a hours/days timescale, not minutes.
    if not skip_refresh:
        for tier in ("haiku", "sonnet", "opus"):
            if _needs_refresh(tier, cooldowns):
                with _lock:
                    if _needs_refresh(tier, cooldowns):
                        # Mark as refreshing to prevent duplicate spawns
                        _cache[tier]["ts"] = time.time()
                        t = threading.Thread(
                            target=_bg_refresh, args=(tier, context),
                            daemon=True, name=f"cf-{tier}",
                        )
                        t.start()

    # Cascade lookup for this ticker
    ticker = context.get("ticker", "")
    result = _get_best_result(ticker)
    if result:
        return dict(result)  # copy to avoid mutation
    return dict(_DEFAULT_HOLD)
