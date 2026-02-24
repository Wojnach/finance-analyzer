"""4-hour digest message builder and sender.

Sends via message_store with category "digest" (always delivered to Telegram).
Enhanced with: invocation count, success/failure, consensus breakdown, L1→L2 triggers.
"""

import json
import logging
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from portfolio.portfolio_mgr import _atomic_write_json, portfolio_value, load_state
from portfolio.message_store import send_or_store
from portfolio.telegram_notifications import escape_markdown_v1

logger = logging.getLogger("portfolio.digest")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INVOCATIONS_FILE = DATA_DIR / "invocations.jsonl"
AGENT_SUMMARY_FILE = DATA_DIR / "agent_summary.json"
BOLD_STATE_FILE = DATA_DIR / "portfolio_state_bold.json"
SIGNAL_LOG_FILE = DATA_DIR / "signal_log.jsonl"

DIGEST_INTERVAL = 14400  # 4 hours


def _get_last_digest_time():
    try:
        state = json.loads(
            (DATA_DIR / "trigger_state.json").read_text(encoding="utf-8")
        )
        return state.get("last_digest_time", 0)
    except Exception:
        return 0


def _set_last_digest_time(t):
    path = DATA_DIR / "trigger_state.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    state["last_digest_time"] = t
    _atomic_write_json(path, state)


JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"


def _build_digest_message():
    from portfolio.stats import load_jsonl

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=DIGEST_INTERVAL)

    # --- Invocations (Layer 1 trigger → Layer 2) ---
    entries = load_jsonl(INVOCATIONS_FILE)
    recent = [e for e in entries if datetime.fromisoformat(e["ts"]) >= cutoff]
    reason_counts = Counter()
    status_counts = Counter()
    for e in recent:
        status_counts[e.get("status", "invoked")] += 1
        for r in e.get("reasons", []):
            if "flipped" in r:
                reason_counts["signal_flip"] += 1
            elif "moved" in r:
                reason_counts["price_move"] += 1
            elif "F&G" in r:
                reason_counts["fear_greed"] += 1
            elif "sentiment" in r:
                reason_counts["sentiment"] += 1
            elif "cooldown" in r or "check-in" in r:
                reason_counts["check_in"] += 1
            elif "consensus" in r:
                reason_counts["consensus"] += 1
            elif "post-trade" in r:
                reason_counts["post_trade"] += 1
            else:
                reason_counts["other"] += 1

    # --- Layer 2 decisions from journal ---
    journal = load_jsonl(JOURNAL_FILE)
    recent_journal = [e for e in journal if datetime.fromisoformat(e["ts"]) >= cutoff]
    l2_decisions = {"patient": Counter(), "bold": Counter()}
    for e in recent_journal:
        decisions = e.get("decisions", {})
        for strat in ("patient", "bold"):
            action = decisions.get(strat, {}).get("action", "HOLD")
            # Normalize: "BUY SMCI" → "BUY", "SELL BTC-USD" → "SELL"
            action_key = action.split()[0] if action else "HOLD"
            l2_decisions[strat][action_key] += 1

    # --- Signal consensus breakdown from signal_log ---
    signal_entries = load_jsonl(SIGNAL_LOG_FILE)
    recent_signals = [e for e in signal_entries if datetime.fromisoformat(e.get("ts", "2000-01-01")) >= cutoff]
    consensus_counts = Counter()
    for e in recent_signals:
        for ticker_data in e.get("signals", {}).values():
            action = ticker_data.get("action", "HOLD")
            consensus_counts[action] += 1

    # --- Portfolio values ---
    summary = json.loads(AGENT_SUMMARY_FILE.read_text(encoding="utf-8"))
    fx_rate = summary.get("fx_rate", 10.5)
    prices_usd = {t: s["price_usd"] for t, s in summary.get("signals", {}).items()}

    state = load_state()
    p_total = portfolio_value(state, prices_usd, fx_rate)
    p_pnl = ((p_total - state["initial_value_sek"]) / state["initial_value_sek"]) * 100
    p_holdings = [
        t for t, h in state.get("holdings", {}).items() if h.get("shares", 0) > 0
    ]

    # --- Build message ---
    lines = ["*4H DIGEST*", ""]
    lines.append(
        f"_{cutoff.strftime('%H:%M')} - {now.strftime('%H:%M UTC')} ({now.strftime('%b %d')})_"
    )

    # Layer 2 (Claude Code) activity — invocations, success, failures
    invoked = status_counts.get("invoked", 0)
    skipped_busy = status_counts.get("skipped_busy", 0)
    skipped_offhours = status_counts.get("skipped_offhours", 0)
    l2_analyses = len(recent_journal)
    l2_failures = max(0, invoked - l2_analyses)  # invoked but no journal = failure

    lines.append("")
    lines.append("*Claude Code Activity*")
    lines.append(f"`Invoked:    {invoked}`")
    lines.append(f"`Succeeded:  {l2_analyses}`")
    if l2_failures > 0:
        lines.append(f"`Failed:     {l2_failures}`")
    if skipped_busy > 0:
        lines.append(f"`Skipped:    {skipped_busy} (busy)`")
    if skipped_offhours > 0:
        lines.append(f"`Off-hours:  {skipped_offhours}`")

    # Layer 2 decisions per strategy
    for strat in ("patient", "bold"):
        counts = l2_decisions[strat]
        if counts:
            parts = []
            for action in ("HOLD", "BUY", "SELL"):
                if counts[action]:
                    parts.append(f"{counts[action]}{action[0]}")
            lines.append(f"`  {strat:<8} {'/'.join(parts)}`")

    # Layer 1 triggers → Layer 2 count
    l1_to_l2 = len(recent)  # total trigger events that reached Layer 2 decision
    lines.append("")
    lines.append(f"*L1 Triggers: {l1_to_l2}*")
    for reason, count in reason_counts.most_common(6):
        lines.append(f"`{reason:<14} {count}`")

    # Signal consensus breakdown (across all tickers in period)
    if consensus_counts:
        total_signals = sum(consensus_counts.values())
        buy_pct = consensus_counts.get("BUY", 0) / total_signals * 100 if total_signals else 0
        sell_pct = consensus_counts.get("SELL", 0) / total_signals * 100 if total_signals else 0
        hold_pct = consensus_counts.get("HOLD", 0) / total_signals * 100 if total_signals else 0
        lines.append("")
        lines.append(
            f"_Consensus: {consensus_counts.get('BUY', 0)}B/{consensus_counts.get('SELL', 0)}S/{consensus_counts.get('HOLD', 0)}H "
            f"({buy_pct:.0f}/{sell_pct:.0f}/{hold_pct:.0f}%)_"
        )

    # Portfolio summaries
    lines.append("")
    p_holdings_str = escape_markdown_v1(', '.join(p_holdings)) if p_holdings else 'cash'
    lines.append(
        f"_Patient: {p_total:,.0f} SEK ({p_pnl:+.1f}%) · {p_holdings_str}_"
    )

    if BOLD_STATE_FILE.exists():
        bold = json.loads(BOLD_STATE_FILE.read_text(encoding="utf-8"))
        b_total = portfolio_value(bold, prices_usd, fx_rate)
        b_pnl = (
            (b_total - bold["initial_value_sek"]) / bold["initial_value_sek"]
        ) * 100
        b_holdings = [
            t for t, h in bold.get("holdings", {}).items() if h.get("shares", 0) > 0
        ]
        b_holdings_str = escape_markdown_v1(', '.join(b_holdings)) if b_holdings else 'cash'
        lines.append(
            f"_Bold: {b_total:,.0f} SEK ({b_pnl:+.1f}%) · {b_holdings_str}_"
        )

    return "\n".join(lines)


def _maybe_send_digest(config):
    last = _get_last_digest_time()
    if last and (time.time() - last) < DIGEST_INTERVAL:
        return
    try:
        msg = _build_digest_message()
        send_or_store(msg, config, category="digest")
        _set_last_digest_time(time.time())
        logger.info("4h digest sent")
    except Exception as e:
        logger.warning("digest failed: %s", e)
