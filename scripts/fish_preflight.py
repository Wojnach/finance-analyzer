"""Fish Preflight — GO/NO-GO gate for intraday metals fishing.

Aggregates overnight move, Monte Carlo P(up), focus probabilities, RSI,
signal consensus, Chronos drift, and regime to produce a directional
conviction score (0-100) for both BULL and BEAR fishing.

Score interpretation:
    80-100  Strong GO — high-conviction direction
    60-79   GO — decent setup, proceed with smaller size
    40-59   MARGINAL — only if other factors confirm
    0-39    NO-GO — don't fish this direction

Usage:
    .venv/Scripts/python.exe scripts/fish_preflight.py [--ticker XAG-USD]
    .venv/Scripts/python.exe scripts/fish_preflight.py --all

Designed to run BEFORE every /fin-fish session as a gate.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portfolio.file_utils import load_json

logger = logging.getLogger("fish_preflight")

SUMMARY_PATH = ROOT / "data" / "agent_summary_compact.json"
FORECAST_PATH = ROOT / "data" / "forecast_predictions.jsonl"
METALS_SIGNAL_PATH = ROOT / "data" / "metals_signal_log.jsonl"

# ---------------------------------------------------------------------------
# Component weights (sum = 100)
# ---------------------------------------------------------------------------
W_OVERNIGHT = 25      # Overnight move direction — strongest single factor
W_MC = 15             # Monte Carlo P(up) 1d
W_FOCUS_1D = 15       # Focus probability 1d direction
W_FOCUS_3H = 10       # Focus probability 3h (intraday-relevant)
W_RSI = 10            # RSI position
W_CONSENSUS = 10      # Signal consensus (buy/sell count)
W_CHRONOS = 10        # Chronos 24h drift
W_REGIME = 5          # Regime type penalty/bonus


def _safe_float(val, default=0.0):
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_overnight_move(ticker: str) -> float | None:
    """Get overnight move % from Binance FAPI (last close vs current)."""
    import requests
    symbol_map = {"XAG-USD": "XAGUSDT", "XAU-USD": "XAUUSDT"}
    symbol = symbol_map.get(ticker)
    if not symbol:
        return None
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/ticker/24hr",
            params={"symbol": symbol},
            timeout=5,
        )
        d = r.json()
        return float(d["priceChangePercent"])
    except Exception:
        return None


def _load_summary_data(ticker: str) -> dict:
    """Load signal, MC, and focus data from compact summary."""
    summary = load_json(SUMMARY_PATH) or {}

    signals = (summary.get("signals") or {}).get(ticker, {})
    mc = (summary.get("monte_carlo") or {}).get(ticker, {})
    focus = (summary.get("focus_probabilities") or {}).get(ticker, {})

    return {
        "rsi": _safe_float(signals.get("rsi")),
        "action": signals.get("action", "HOLD"),
        "confidence": _safe_float(signals.get("confidence")),
        "regime": signals.get("regime", ""),
        "buy_count": (signals.get("extra") or {}).get("_buy_count", 0),
        "sell_count": (signals.get("extra") or {}).get("_sell_count", 0),
        "voters": (signals.get("extra") or {}).get("_voters", 0),
        "mc_return_1d": _safe_float((mc.get("expected_return_1d") or {}).get("mean_pct")),
        "focus_1d": focus.get("1d", {}),
        "focus_3h": focus.get("3h", {}),
    }


def _load_chronos_drift(ticker: str) -> float | None:
    """Get latest Chronos 24h prediction from forecast log."""
    try:
        lines = FORECAST_PATH.read_text().strip().split("\n")
        for line in reversed(lines[-50:]):
            d = json.loads(line)
            if d.get("ticker") == ticker:
                raw = d.get("raw_sub_signals", {})
                # Try to get numeric drift
                indicators = d.get("indicators", {})
                chronos_pct = indicators.get("chronos_24h_pct")
                if chronos_pct is not None:
                    return float(chronos_pct)
                # Fallback: map action to estimate
                action = raw.get("chronos_24h", "HOLD")
                if action == "BUY":
                    return 0.5
                elif action == "SELL":
                    return -0.5
                return 0.0
    except Exception:
        pass
    return None


def _load_metals_loop_data(ticker: str) -> dict | None:
    """Get latest metals loop signal for the ticker."""
    try:
        lines = METALS_SIGNAL_PATH.read_text().strip().split("\n")
        for line in reversed(lines[-20:]):
            d = json.loads(line)
            signals = d.get("signals", {})
            if ticker in signals:
                return signals[ticker]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_preflight(ticker: str) -> dict:
    """Compute BULL and BEAR conviction scores (0-100).

    Each component contributes points toward the direction it favors.
    """
    bull_points = 0.0
    bear_points = 0.0
    components = {}

    # --- 1. Overnight move (25 pts) ---
    overnight = _load_overnight_move(ticker)
    if overnight is not None:
        if overnight > 1.0:
            # Big up overnight → DON'T go bear (yesterday's lesson)
            bull_points += W_OVERNIGHT * min(overnight / 3.0, 1.0)
            components["overnight"] = f"+{overnight:.1f}% -> BULL {bull_points:.0f}pts"
        elif overnight < -1.0:
            bear_points += W_OVERNIGHT * min(abs(overnight) / 3.0, 1.0)
            components["overnight"] = f"{overnight:.1f}% -> BEAR {bear_points:.0f}pts"
        elif overnight > 0.3:
            bull_points += W_OVERNIGHT * 0.3
            components["overnight"] = f"+{overnight:.1f}% -> slight BULL"
        elif overnight < -0.3:
            bear_points += W_OVERNIGHT * 0.3
            components["overnight"] = f"{overnight:.1f}% -> slight BEAR"
        else:
            components["overnight"] = f"{overnight:+.1f}% -> NEUTRAL"
    else:
        components["overnight"] = "NO DATA"

    # --- 2. Load summary data ---
    data = _load_summary_data(ticker)

    # --- 3. Monte Carlo 1d return (15 pts) ---
    mc_ret = data["mc_return_1d"]
    if mc_ret > 0.1:
        bull_points += W_MC * min(mc_ret / 0.5, 1.0)
        components["mc_1d"] = f"+{mc_ret:.2f}% -> BULL"
    elif mc_ret < -0.1:
        bear_points += W_MC * min(abs(mc_ret) / 0.5, 1.0)
        components["mc_1d"] = f"{mc_ret:.2f}% -> BEAR"
    else:
        components["mc_1d"] = f"{mc_ret:+.2f}% -> NEUTRAL"

    # --- 4. Focus probability 1d (15 pts) ---
    focus_1d = data["focus_1d"]
    if focus_1d:
        direction_1d = focus_1d.get("direction", "")
        prob_1d = _safe_float(focus_1d.get("probability"), 0.5)
        if direction_1d == "up" and prob_1d > 0.55:
            bull_points += W_FOCUS_1D * min((prob_1d - 0.5) * 4, 1.0)
            components["focus_1d"] = f"UP {prob_1d:.0%} -> BULL"
        elif direction_1d == "down" and prob_1d > 0.55:
            bear_points += W_FOCUS_1D * min((prob_1d - 0.5) * 4, 1.0)
            components["focus_1d"] = f"DOWN {prob_1d:.0%} -> BEAR"
        else:
            components["focus_1d"] = f"{direction_1d} {prob_1d:.0%} -> NEUTRAL"
    else:
        components["focus_1d"] = "NO DATA"

    # --- 5. Focus probability 3h (10 pts) ---
    focus_3h = data["focus_3h"]
    if focus_3h:
        direction_3h = focus_3h.get("direction", "")
        prob_3h = _safe_float(focus_3h.get("probability"), 0.5)
        if direction_3h == "up" and prob_3h > 0.55:
            bull_points += W_FOCUS_3H * min((prob_3h - 0.5) * 4, 1.0)
            components["focus_3h"] = f"UP {prob_3h:.0%} -> BULL"
        elif direction_3h == "down" and prob_3h > 0.55:
            bear_points += W_FOCUS_3H * min((prob_3h - 0.5) * 4, 1.0)
            components["focus_3h"] = f"DOWN {prob_3h:.0%} -> BEAR"
        else:
            components["focus_3h"] = f"{direction_3h} {prob_3h:.0%} -> NEUTRAL"
    else:
        components["focus_3h"] = "NO DATA"

    # --- 6. RSI (10 pts) ---
    rsi = data["rsi"]
    if rsi > 0:
        if rsi > 70:
            bear_points += W_RSI
            components["rsi"] = f"{rsi:.1f} (overbought) -> BEAR"
        elif rsi > 65:
            bear_points += W_RSI * 0.6
            components["rsi"] = f"{rsi:.1f} (high) -> lean BEAR"
        elif rsi < 30:
            bull_points += W_RSI
            components["rsi"] = f"{rsi:.1f} (oversold) -> BULL"
        elif rsi < 45:
            bull_points += W_RSI * 0.6
            components["rsi"] = f"{rsi:.1f} (low) -> lean BULL"
        else:
            components["rsi"] = f"{rsi:.1f} -> NEUTRAL"
    else:
        components["rsi"] = "NO DATA"

    # --- 7. Signal consensus (10 pts) ---
    buy_c = data["buy_count"]
    sell_c = data["sell_count"]
    total = buy_c + sell_c
    if total > 0:
        buy_ratio = buy_c / total
        if buy_ratio > 0.65:
            bull_points += W_CONSENSUS * min((buy_ratio - 0.5) * 4, 1.0)
            components["consensus"] = f"{buy_c}B/{sell_c}S -> BULL"
        elif buy_ratio < 0.35:
            bear_points += W_CONSENSUS * min((0.5 - buy_ratio) * 4, 1.0)
            components["consensus"] = f"{buy_c}B/{sell_c}S -> BEAR"
        else:
            components["consensus"] = f"{buy_c}B/{sell_c}S -> NEUTRAL"
    else:
        components["consensus"] = f"action={data['action']} -> NEUTRAL"

    # --- 8. Chronos 24h (10 pts) ---
    chronos = _load_chronos_drift(ticker)
    if chronos is not None:
        if chronos > 0.2:
            bull_points += W_CHRONOS * min(chronos / 1.0, 1.0)
            components["chronos"] = f"+{chronos:.2f}% -> BULL"
        elif chronos < -0.2:
            bear_points += W_CHRONOS * min(abs(chronos) / 1.0, 1.0)
            components["chronos"] = f"{chronos:.2f}% -> BEAR"
        else:
            components["chronos"] = f"{chronos:+.2f}% -> NEUTRAL"
    else:
        components["chronos"] = "NO DATA"

    # --- 9. Regime (5 pts bonus/penalty) ---
    regime = data["regime"]
    if regime == "trending-up":
        bull_points += W_REGIME * 0.6
        bear_points -= W_REGIME * 0.3  # Penalty for fighting trend
        components["regime"] = f"{regime} -> lean BULL"
    elif regime == "trending-down":
        bear_points += W_REGIME * 0.6
        bull_points -= W_REGIME * 0.3
        components["regime"] = f"{regime} -> lean BEAR"
    elif regime == "ranging":
        # Ranging is neutral — both directions viable for mean reversion
        components["regime"] = f"{regime} -> NEUTRAL"
    else:
        components["regime"] = f"{regime or 'unknown'}"

    # Clamp to 0-100
    bull_score = max(0, min(100, round(bull_points)))
    bear_score = max(0, min(100, round(bear_points)))

    # Determine verdict
    if bull_score >= 60 and bull_score > bear_score + 15:
        verdict = "GO BULL"
    elif bear_score >= 60 and bear_score > bull_score + 15:
        verdict = "GO BEAR"
    elif bull_score >= 40 or bear_score >= 40:
        best = "BULL" if bull_score >= bear_score else "BEAR"
        verdict = f"MARGINAL {best}"
    else:
        verdict = "NO-GO"

    return {
        "ticker": ticker,
        "bull_score": bull_score,
        "bear_score": bear_score,
        "verdict": verdict,
        "components": components,
        "overnight_move": overnight,
        "rsi": data["rsi"],
        "regime": regime,
    }


# ---------------------------------------------------------------------------
# CLI display
# ---------------------------------------------------------------------------

def _score_bar(score: int, width: int = 20) -> str:
    filled = round(score / 100 * width)
    bar = "#" * filled + "-" * (width - filled)
    if score >= 80:
        label = "STRONG GO"
    elif score >= 60:
        label = "GO"
    elif score >= 40:
        label = "MARGINAL"
    else:
        label = "NO-GO"
    return f"[{bar}] {score}/100 {label}"


def print_preflight(result: dict) -> None:
    ticker = result["ticker"]
    print(f"\n{'='*60}")
    print(f"  FISH PREFLIGHT: {ticker}")
    print(f"{'='*60}")
    print()
    print(f"  BULL score: {_score_bar(result['bull_score'])}")
    print(f"  BEAR score: {_score_bar(result['bear_score'])}")
    print(f"  Verdict:    {result['verdict']}")
    print()
    print("  Components:")
    for name, detail in result["components"].items():
        print(f"    {name:12s}  {detail}")
    print()

    # Yesterday's lesson reminder
    if result["overnight_move"] is not None:
        om = result["overnight_move"]
        if abs(om) > 3.0:
            print(f"  !! OVERNIGHT MOVE {om:+.1f}% -- regime may have shifted !!")
            print(f"  !! Don't apply yesterday's pattern to today !!")
            print()


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Fish preflight GO/NO-GO check")
    parser.add_argument("--ticker", default="XAG-USD", help="Ticker to check")
    parser.add_argument("--all", action="store_true", help="Check XAG-USD and XAU-USD")
    args = parser.parse_args()

    tickers = ["XAG-USD", "XAU-USD"] if args.all else [args.ticker]

    for ticker in tickers:
        result = compute_preflight(ticker)
        print_preflight(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
