"""Weekly performance digest sent via Telegram.

Summarizes: trades taken, P&L change, best/worst signals, regime distribution,
portfolio comparison (Patient vs Bold).

Usage:
    python -m portfolio.weekly_digest           # send digest now
    python -m portfolio.weekly_digest --dry-run # print without sending
"""

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIG_FILE = BASE_DIR / "config.json"
PATIENT_FILE = DATA_DIR / "portfolio_state.json"
BOLD_FILE = DATA_DIR / "portfolio_state_bold.json"
SIGNAL_LOG = DATA_DIR / "signal_log.jsonl"
JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"


def _load_json(path):
    """Load a JSON file, returning empty dict on failure."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return {}


def _load_jsonl(path, since=None):
    """Load JSONL file, optionally filtering entries since a datetime."""
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if since is not None:
            ts_str = entry.get("ts", "")
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts < since:
                    continue
            except (ValueError, TypeError):
                continue
        entries.append(entry)
    return entries


def _portfolio_summary(state, label):
    """Build summary dict for a portfolio state."""
    cash = state.get("cash_sek", 0)
    initial = state.get("initial_value_sek", 500000)
    fees = state.get("total_fees_sek", 0) or 0
    transactions = state.get("transactions", [])
    holdings = state.get("holdings", {})
    active_holdings = [t for t, h in holdings.items() if h.get("shares", 0) > 0]
    pnl_sek = cash - initial  # simplified: ignores unrealized P&L from holdings
    pnl_pct = (pnl_sek / initial * 100) if initial else 0.0

    return {
        "label": label,
        "cash_sek": cash,
        "initial_sek": initial,
        "pnl_sek": pnl_sek,
        "pnl_pct": pnl_pct,
        "total_fees_sek": fees,
        "total_trades": len(transactions),
        "active_holdings": active_holdings,
    }


def _trades_this_week(transactions, since):
    """Count trades within the time window."""
    week_trades = []
    for t in transactions:
        ts_str = t.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts >= since:
                week_trades.append(t)
        except (ValueError, TypeError):
            continue
    return week_trades


def _signal_accuracy_this_week(signal_entries):
    """Compute per-signal accuracy from signal log entries with 1d outcomes."""
    from portfolio.accuracy_stats import SIGNAL_NAMES

    stats = {s: {"correct": 0, "total": 0} for s in SIGNAL_NAMES}

    for entry in signal_entries:
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            outcome = outcomes.get(ticker, {}).get("1d")
            if not outcome:
                continue
            change_pct = outcome.get("change_pct", 0)
            signals = tdata.get("signals", {})

            for sig_name in SIGNAL_NAMES:
                vote = signals.get(sig_name, "HOLD")
                if vote == "HOLD":
                    continue
                stats[sig_name]["total"] += 1
                if (vote == "BUY" and change_pct > 0) or (vote == "SELL" and change_pct < 0):
                    stats[sig_name]["correct"] += 1

    result = {}
    for sig_name in SIGNAL_NAMES:
        s = stats[sig_name]
        if s["total"] > 0:
            result[sig_name] = {
                "accuracy": s["correct"] / s["total"],
                "pct": round(s["correct"] / s["total"] * 100, 1),
                "samples": s["total"],
            }
    return result


def _regime_distribution(journal_entries):
    """Compute regime distribution from journal entries."""
    regimes = [e.get("regime", "unknown") for e in journal_entries if e.get("regime")]
    if not regimes:
        return {}
    counter = Counter(regimes)
    total = len(regimes)
    return {regime: round(count / total * 100, 1) for regime, count in counter.most_common()}


def generate_weekly_digest():
    """Generate the weekly performance digest message string.

    Returns:
        str: Formatted Telegram message.
    """
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)

    # Load portfolio states
    patient_state = _load_json(PATIENT_FILE)
    bold_state = _load_json(BOLD_FILE)
    patient = _portfolio_summary(patient_state, "Patient")
    bold = _portfolio_summary(bold_state, "Bold")

    # Count trades this week
    patient_week_trades = _trades_this_week(patient_state.get("transactions", []), week_ago)
    bold_week_trades = _trades_this_week(bold_state.get("transactions", []), week_ago)

    # Categorize trades
    patient_buys = [t for t in patient_week_trades if t.get("action") == "BUY"]
    patient_sells = [t for t in patient_week_trades if t.get("action") == "SELL"]
    bold_buys = [t for t in bold_week_trades if t.get("action") == "BUY"]
    bold_sells = [t for t in bold_week_trades if t.get("action") == "SELL"]

    # Load signal log for accuracy this week
    signal_entries = _load_jsonl(SIGNAL_LOG, since=week_ago)
    weekly_accuracy = _signal_accuracy_this_week(signal_entries)

    # Best/worst signals this week (min 5 samples)
    qualified = {k: v for k, v in weekly_accuracy.items() if v["samples"] >= 5}
    best_signal = None
    worst_signal = None
    if qualified:
        best_name = max(qualified, key=lambda k: qualified[k]["accuracy"])
        worst_name = min(qualified, key=lambda k: qualified[k]["accuracy"])
        best_signal = (best_name, qualified[best_name]["pct"], qualified[best_name]["samples"])
        worst_signal = (worst_name, qualified[worst_name]["pct"], qualified[worst_name]["samples"])

    # Load journal for regime distribution and invocation count
    journal_entries = _load_jsonl(JOURNAL_FILE, since=week_ago)
    regimes = _regime_distribution(journal_entries)
    invocation_count = len(journal_entries)

    # Build message
    lines = []
    lines.append("*WEEKLY DIGEST*")
    lines.append(f"_{now.strftime('%b %d')} — 7 day summary_")
    lines.append("")

    # Portfolio comparison
    lines.append("*Portfolio Performance*")
    for pf in (patient, bold):
        sign = "+" if pf["pnl_pct"] >= 0 else ""
        holdings_str = ", ".join(pf["active_holdings"]) if pf["active_holdings"] else "none"
        lines.append(
            f"`{pf['label']:<8} {pf['cash_sek']:>10,.0f} SEK  "
            f"{sign}{pf['pnl_pct']:.2f}%`"
        )
        lines.append(f"  Holdings: {holdings_str}")
        lines.append(f"  Fees: {pf['total_fees_sek']:,.0f} SEK total")
    lines.append("")

    # Trades this week
    lines.append("*Trades This Week*")
    lines.append(
        f"Patient: {len(patient_buys)} buy, {len(patient_sells)} sell "
        f"({len(patient_week_trades)} total)"
    )
    lines.append(
        f"Bold: {len(bold_buys)} buy, {len(bold_sells)} sell "
        f"({len(bold_week_trades)} total)"
    )

    # Detail recent trades
    all_week_trades = []
    for t in patient_week_trades:
        all_week_trades.append(("Patient", t))
    for t in bold_week_trades:
        all_week_trades.append(("Bold", t))

    if all_week_trades:
        lines.append("")
        for strat, t in sorted(
            all_week_trades,
            key=lambda x: x[1].get("timestamp", ""),
        ):
            action = t.get("action", "?")
            ticker = t.get("ticker", "?")
            total_sek = t.get("total_sek", 0)
            price_usd = t.get("price_usd", 0)
            lines.append(
                f"  {strat} {action} {ticker} @ ${price_usd:,.2f} "
                f"({total_sek:,.0f} SEK)"
            )
    else:
        lines.append("  No trades this week.")
    lines.append("")

    # Signal accuracy this week
    lines.append("*Signal Accuracy (7d)*")
    if best_signal:
        lines.append(f"  Best: {best_signal[0]} {best_signal[1]}% ({best_signal[2]} samples)")
    if worst_signal:
        lines.append(f"  Worst: {worst_signal[0]} {worst_signal[1]}% ({worst_signal[2]} samples)")
    if not best_signal and not worst_signal:
        lines.append("  Insufficient data (need 5+ samples per signal)")

    # Top 5 signals by accuracy
    if qualified:
        sorted_sigs = sorted(qualified.items(), key=lambda x: x[1]["accuracy"], reverse=True)
        lines.append("")
        for name, data in sorted_sigs[:5]:
            lines.append(f"  `{name:<18} {data['pct']:>5.1f}%  ({data['samples']} samples)`")
    lines.append("")

    # Regime distribution
    lines.append("*Regime Distribution (7d)*")
    if regimes:
        for regime, pct in regimes.items():
            bar_len = int(pct / 5)  # rough visual bar
            bar = "=" * bar_len
            lines.append(f"  `{regime:<16} {pct:>5.1f}%  {bar}`")
    else:
        lines.append("  No journal data this week.")
    lines.append("")

    # Invocations
    lines.append(f"_Layer 2 invocations: {invocation_count}_")

    msg = "\n".join(lines)
    return msg


def send_digest(msg):
    """Send digest message via Telegram.

    Args:
        msg: The message string to send.

    Returns:
        requests.Response or None on error.
    """
    config = _load_json(CONFIG_FILE)
    token = config.get("telegram", {}).get("token")
    chat_id = config.get("telegram", {}).get("chat_id")

    if not token or not chat_id:
        print("ERROR: Telegram config missing token or chat_id")
        return None

    # Save locally first
    log_file = DATA_DIR / "telegram_messages.jsonl"
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "text": msg,
        "type": "weekly_digest",
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Send via Telegram
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
            timeout=30,
        )
        print(f"Telegram response: {resp.status_code}")
        return resp
    except requests.RequestException as e:
        print(f"ERROR sending Telegram: {e}")
        return None


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    msg = generate_weekly_digest()

    if dry_run:
        print(msg)
        print("\n--- (dry run, not sent) ---")
    else:
        print(msg)
        print()
        send_digest(msg)
