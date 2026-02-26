"""Periodic trade reflection — computes structured performance metrics.

Pure Python (no LLM call). After every N trades, generates a reflection
summary stored in data/reflections.jsonl. Layer 2 reads the latest
reflection as context for self-assessment.

Config:
    "reflection": {
        "enabled": true,
        "trade_interval": 10,
        "max_age_days": 7
    }
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

from portfolio.file_utils import atomic_append_jsonl, load_json, load_jsonl

logger = logging.getLogger("portfolio.reflection")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REFLECTIONS_FILE = DATA_DIR / "reflections.jsonl"
PORTFOLIO_FILE = DATA_DIR / "portfolio_state.json"
BOLD_FILE = DATA_DIR / "portfolio_state_bold.json"
JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"


def _count_trades(portfolio):
    """Count total trades in a portfolio state."""
    return len(portfolio.get("transactions", []))


def _compute_strategy_metrics(portfolio):
    """Compute win rate, avg PnL, and total PnL for a strategy.

    Returns dict with trades, win_rate, avg_pnl_pct, total_pnl_pct, holdings.
    """
    txns = portfolio.get("transactions", [])
    initial = portfolio.get("initial_value_sek", 500000)
    cash = portfolio.get("cash_sek", initial)
    holdings = portfolio.get("holdings", {})
    holding_tickers = [t for t, h in holdings.items() if h.get("shares", 0) > 0]

    if not txns:
        return {
            "trades": 0,
            "win_rate": None,
            "avg_pnl_pct": None,
            "total_pnl_pct": round((cash - initial) / initial * 100, 2),
            "holdings": holding_tickers,
        }

    # Match BUY/SELL pairs per ticker for PnL
    buys = {}  # ticker -> list of (price_sek, shares)
    sells = []  # list of (ticker, pnl_pct)

    for tx in txns:
        ticker = tx.get("ticker", "")
        action = tx.get("action", "")
        shares = tx.get("shares", 0)
        price = tx.get("price_sek", 0)

        if action == "BUY" and price > 0:
            buys.setdefault(ticker, []).append((price, shares))
        elif action == "SELL" and price > 0:
            # Compute PnL against avg cost of prior buys
            buy_list = buys.get(ticker, [])
            if buy_list:
                total_cost = sum(p * s for p, s in buy_list)
                total_shares = sum(s for _, s in buy_list)
                avg_cost = total_cost / total_shares if total_shares > 0 else price
                pnl_pct = (price - avg_cost) / avg_cost * 100
                sells.append((ticker, pnl_pct))

    wins = sum(1 for _, pnl in sells if pnl > 0)
    win_rate = round(wins / len(sells), 2) if sells else None
    avg_pnl = round(sum(pnl for _, pnl in sells) / len(sells), 2) if sells else None
    total_pnl_pct = round((cash - initial) / initial * 100, 2)

    return {
        "trades": len(txns),
        "win_rate": win_rate,
        "avg_pnl_pct": avg_pnl,
        "total_pnl_pct": total_pnl_pct,
        "holdings": holding_tickers,
    }


def _regime_distribution():
    """Count regime occurrences in recent journal entries."""
    entries = load_jsonl(JOURNAL_FILE, limit=100)
    dist = {}
    for e in entries:
        regime = e.get("regime", "unknown")
        dist[regime] = dist.get(regime, 0) + 1
    return dist


def _generate_insights(patient_metrics, bold_metrics):
    """Generate human-readable insights from metrics."""
    insights = []

    for label, m in [("Patient", patient_metrics), ("Bold", bold_metrics)]:
        trades = m.get("trades", 0)
        if trades == 0:
            insights.append(f"{label}: no trades yet")
            continue

        win_rate = m.get("win_rate")
        if win_rate is not None:
            if win_rate == 0:
                insights.append(f"{label}: all {trades} closed trades were losses")
            elif win_rate >= 0.7:
                insights.append(f"{label}: strong {win_rate:.0%} win rate over {trades} trades")
            elif win_rate < 0.4:
                insights.append(f"{label}: weak {win_rate:.0%} win rate — review entry criteria")

        total_pnl = m.get("total_pnl_pct", 0)
        if total_pnl < -5:
            insights.append(f"{label}: down {abs(total_pnl):.1f}% — consider reducing size")
        elif total_pnl > 5:
            insights.append(f"{label}: up {total_pnl:.1f}% — strategy working")

    return insights


def should_reflect(config=None):
    """Check whether a new reflection is due.

    Returns True if:
    - Feature is enabled
    - Total trade count crossed the interval threshold since last reflection
    - OR last reflection is older than max_age_days
    """
    if config is None:
        from portfolio.api_utils import load_config
        config = load_config()

    ref_cfg = config.get("reflection", {})
    if not ref_cfg.get("enabled", False):
        return False

    interval = ref_cfg.get("trade_interval", 10)
    max_age_days = ref_cfg.get("max_age_days", 7)

    # Count total trades across both portfolios
    patient = load_json(PORTFOLIO_FILE, {})
    bold = load_json(BOLD_FILE, {})
    total_trades = _count_trades(patient) + _count_trades(bold)

    # Load last reflection
    reflections = load_jsonl(REFLECTIONS_FILE)
    if not reflections:
        return total_trades >= interval

    last = reflections[-1]
    last_trade_count = last.get("trade_count_total", 0)

    # Check trade interval
    if total_trades - last_trade_count >= interval:
        return True

    # Check age
    try:
        last_ts = datetime.fromisoformat(last["ts"])
        age = datetime.now(timezone.utc) - last_ts
        if age > timedelta(days=max_age_days):
            return True
    except (KeyError, ValueError):
        return True

    return False


def compute_reflection():
    """Compute a reflection entry from current portfolio states.

    Returns a reflection dict ready to be saved.
    """
    patient = load_json(PORTFOLIO_FILE, {})
    bold = load_json(BOLD_FILE, {})

    patient_metrics = _compute_strategy_metrics(patient)
    bold_metrics = _compute_strategy_metrics(bold)
    regime_dist = _regime_distribution()
    insights = _generate_insights(patient_metrics, bold_metrics)

    total_trades = _count_trades(patient) + _count_trades(bold)

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "patient": patient_metrics,
        "bold": bold_metrics,
        "regime_distribution": regime_dist,
        "trade_count_total": total_trades,
        "insights": insights,
    }


def save_reflection(reflection):
    """Save a reflection entry to the JSONL file."""
    atomic_append_jsonl(REFLECTIONS_FILE, reflection)
    logger.info("Reflection saved: %d total trades", reflection.get("trade_count_total", 0))


def maybe_reflect(config=None):
    """Check if a reflection is due and compute/save one if so.

    Called from main loop. Non-blocking, non-critical.
    """
    try:
        if should_reflect(config):
            reflection = compute_reflection()
            save_reflection(reflection)
            return True
    except Exception as e:
        logger.warning("reflection failed: %s", e)
    return False


def load_latest_reflection(max_age_days=7):
    """Load the most recent reflection if it exists and isn't too old.

    Returns the reflection dict or None.
    """
    reflections = load_jsonl(REFLECTIONS_FILE)
    if not reflections:
        return None

    last = reflections[-1]
    try:
        ts = datetime.fromisoformat(last["ts"])
        age = datetime.now(timezone.utc) - ts
        if age > timedelta(days=max_age_days):
            return None
    except (KeyError, ValueError):
        pass

    return last
