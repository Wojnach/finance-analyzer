import json
import re
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
CONTEXT_FILE = DATA_DIR / "layer2_context.md"
PORTFOLIO_FILE = DATA_DIR / "portfolio_state.json"
BOLD_FILE = DATA_DIR / "portfolio_state_bold.json"

TICKERS = [
    "BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD",
    "MSTR", "PLTR", "NVDA", "AMD", "BABA", "GOOGL", "AMZN", "AAPL",
    "AVGO", "AI", "GRRR", "IONQ", "MRVL", "META", "MU", "PONY",
    "RXRX", "SOUN", "SMCI", "TSM", "TTWO", "TEM", "UPST", "VERI",
    "VRT", "QQQ", "LMT",
]

TIER_FULL = 2
TIER_COMPACT = 4


def load_recent(max_entries=10, max_age_hours=8):
    if not JOURNAL_FILE.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    entries = []
    for line in JOURNAL_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            ts = datetime.fromisoformat(entry["ts"])
            if ts >= cutoff:
                entries.append(entry)
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return entries[-max_entries:]


def _is_all_hold(entry):
    decisions = entry.get("decisions", {})
    for strat in ("patient", "bold"):
        d = decisions.get(strat, {})
        if d.get("action", "HOLD") != "HOLD":
            return False
    return True


def _non_neutral_tickers(entry):
    tickers = entry.get("tickers", {})
    return {
        k: v for k, v in tickers.items() if v.get("outlook", "neutral") != "neutral"
    }


def _fmt_time(ts_str):
    ts = datetime.fromisoformat(ts_str)
    return ts.strftime("%H:%M UTC")


def _fmt_time_range(ts_start, ts_end):
    t0 = datetime.fromisoformat(ts_start).strftime("%H:%M")
    t1 = datetime.fromisoformat(ts_end).strftime("%H:%M UTC")
    return f"{t0}–{t1}"


def _entry_age_hours(entry, now=None):
    if now is None:
        now = datetime.now(timezone.utc)
    ts = datetime.fromisoformat(entry["ts"])
    return (now - ts).total_seconds() / 3600


def _append_entry(lines, entry):
    ts = _fmt_time(entry["ts"])
    trigger = entry.get("trigger", "unknown")
    regime = entry.get("regime", "unknown")

    lines.append(f"**{ts}** | trigger: {trigger}")

    reflection = entry.get("reflection")
    if reflection:
        lines.append(f"_Reflection: {reflection}_")

    lines.append(f"regime: {regime}")

    decisions = entry.get("decisions", {})
    for strat in ("patient", "bold"):
        d = decisions.get(strat, {})
        action = d.get("action", "HOLD")
        reasoning = d.get("reasoning", "")
        lines.append(f"{strat}: {action} — {reasoning}")

    for ticker, info in _non_neutral_tickers(entry).items():
        outlook = info.get("outlook", "neutral")
        thesis = info.get("thesis", "")
        levels = info.get("levels", [])
        level_str = f" (S:{levels[0]} R:{levels[1]})" if len(levels) == 2 else ""
        conviction = info.get("conviction")
        conv_str = f" [{int(conviction * 100)}%]" if conviction else ""
        lines.append(f"{ticker}: {outlook}{conv_str} — {thesis}{level_str}")

    lines.append("")


def _append_entry_compact(lines, entry):
    ts = _fmt_time(entry["ts"])
    decisions = entry.get("decisions", {})
    p_action = decisions.get("patient", {}).get("action", "HOLD")
    b_action = decisions.get("bold", {}).get("action", "HOLD")

    ticker_parts = []
    for ticker, info in _non_neutral_tickers(entry).items():
        outlook = info.get("outlook", "neutral")
        conviction = info.get("conviction")
        conv_str = f"({int(conviction * 100)}%)" if conviction else ""
        ticker_parts.append(f"{ticker}={outlook}{conv_str}")

    ticker_str = " | " + ", ".join(ticker_parts) if ticker_parts else ""
    lines.append(f"**{ts}** | patient: {p_action} / bold: {b_action}{ticker_str}")
    lines.append("")


def _append_entry_oneline(lines, entry):
    ts = _fmt_time(entry["ts"])
    regime = entry.get("regime", "unknown")
    decisions = entry.get("decisions", {})
    p_action = decisions.get("patient", {}).get("action", "HOLD")
    b_action = decisions.get("bold", {}).get("action", "HOLD")
    lines.append(f"{ts} {regime} P:{p_action}/B:{b_action}")


def _build_continuation_chains(entries):
    ts_map = {}
    for e in entries:
        ts_map[e["ts"]] = e

    children = defaultdict(list)
    for e in entries:
        parent_ts = e.get("continues")
        if parent_ts and parent_ts in ts_map:
            children[parent_ts].append(e["ts"])

    roots = set()
    for e in entries:
        parent_ts = e.get("continues")
        if parent_ts and parent_ts in ts_map:
            continue
        if e["ts"] in children:
            roots.add(e["ts"])

    chains = []
    for root_ts in sorted(roots):
        chain = [root_ts]
        current = root_ts
        while current in children:
            next_ts = children[current][0]
            chain.append(next_ts)
            current = next_ts
        if len(chain) >= 2:
            chains.append(chain)

    return chains, ts_map


def _load_portfolio_pnl():
    data = {}
    for label, filepath in [("patient", PORTFOLIO_FILE), ("bold", BOLD_FILE)]:
        if not filepath.exists():
            continue
        try:
            pf = json.loads(filepath.read_text(encoding="utf-8"))
            holdings = pf.get("holdings", {})
            holding_tickers = [t for t, h in holdings.items() if h.get("shares", 0) > 0]
            data[label] = {
                "cash_sek": pf.get("cash_sek", 0),
                "initial_value_sek": pf.get("initial_value_sek", 500000),
                "total_fees_sek": pf.get("total_fees_sek", 0),
                "trades": len(pf.get("transactions", [])),
                "holdings": holding_tickers,
            }
        except (json.JSONDecodeError, ValueError, AttributeError):
            continue
    return data


def _detect_warnings(entries):
    if not entries:
        return []
    warnings = []

    ticker_runs = defaultdict(list)
    for e in entries:
        tickers = e.get("tickers", {})
        prices = e.get("prices", {})
        for ticker, info in tickers.items():
            outlook = info.get("outlook", "neutral")
            if outlook != "neutral":
                price = prices.get(ticker)
                ticker_runs[ticker].append((outlook, price))

    for ticker, runs in ticker_runs.items():
        if len(runs) >= 3:
            outlooks = [r[0] for r in runs]
            prices_list = [r[1] for r in runs if r[1] is not None]
            if len(set(outlooks)) == 1 and len(prices_list) >= 2:
                outlook = outlooks[0]
                first_price = prices_list[0]
                last_price = prices_list[-1]
                pct_change = (last_price - first_price) / first_price
                if outlook == "bullish" and pct_change < -0.005:
                    warnings.append(
                        f"{ticker}: thesis (bullish) contradicted — price dropped {abs(pct_change):.1%}"
                    )
                elif outlook == "bearish" and pct_change > 0.005:
                    warnings.append(
                        f"{ticker}: thesis (bearish) contradicted — price rose {pct_change:.1%}"
                    )

    for strat in ("patient", "bold"):
        actions = []
        for e in entries:
            d = e.get("decisions", {}).get(strat, {})
            action_str = d.get("action", "HOLD")
            match = re.match(r"(BUY|SELL)\s+(\S+)", action_str)
            if match:
                actions.append((match.group(1), match.group(2)))
            else:
                actions.append((action_str, None))

        for i in range(len(actions) - 2):
            a1, t1 = actions[i]
            a3, t3 = actions[i + 2]
            if t1 and t3 and t1 == t3:
                if (a1 == "BUY" and a3 == "SELL") or (a1 == "SELL" and a3 == "BUY"):
                    warnings.append(
                        f"{strat}: whipsaw on {t1} ({a1}→{a3} within 3 entries)"
                    )

        ticker_trade_count = defaultdict(int)
        for action, ticker in actions:
            if ticker and action in ("BUY", "SELL"):
                ticker_trade_count[ticker] += 1
        for ticker, count in ticker_trade_count.items():
            if count >= 3:
                warnings.append(
                    f"{strat}: churning {ticker} ({count} trades in window)"
                )

    if len(entries) >= 2:
        regimes = [e.get("regime", "unknown") for e in entries]
        if len(set(regimes)) == 1:
            t0 = datetime.fromisoformat(entries[0]["ts"])
            t1 = datetime.fromisoformat(entries[-1]["ts"])
            span_hours = (t1 - t0).total_seconds() / 3600
            if span_hours >= 8:
                warnings.append(
                    f"Regime stuck: {regimes[0]} for {span_hours:.0f}h — reassess"
                )

    return warnings


def build_context(entries, portfolio_data=None, now=None):
    if not entries:
        return "## Your Memory\n\nNo previous invocations. Fresh start.\n"

    if now is None:
        now = datetime.now(timezone.utc)

    lines = []

    regimes = [e.get("regime", "unknown") for e in entries]
    last_regime = regimes[-1]
    streak = 0
    for r in reversed(regimes):
        if r == last_regime:
            streak += 1
        else:
            break
    hours_span = 0
    if len(entries) >= 2:
        t0 = datetime.fromisoformat(entries[0]["ts"])
        t1 = datetime.fromisoformat(entries[-1]["ts"])
        hours_span = (t1 - t0).total_seconds() / 3600

    lines.append(f"## Your Memory (last {hours_span:.0f}h, {len(entries)} invocations)")
    lines.append("")
    lines.append(
        f"**Regime:** {last_regime} ({streak} invocation{'s' if streak != 1 else ''})"
    )
    lines.append("")
    lines.append("### Recent Decisions")
    lines.append("")

    i = 0
    while i < len(entries):
        entry = entries[i]
        age = _entry_age_hours(entry, now)

        if _is_all_hold(entry):
            hold_start = i
            while i < len(entries) and _is_all_hold(entries[i]):
                i += 1
            hold_count = i - hold_start

            if hold_count == 1 and age < TIER_FULL:
                _append_entry(lines, entry)
            elif hold_count == 1:
                _append_entry_oneline(lines, entry)
            else:
                ts_range = _fmt_time_range(
                    entries[hold_start]["ts"], entries[i - 1]["ts"]
                )
                lines.append(f"**{ts_range}** | {hold_count}x HOLD (no setups)")
                lines.append("")
        else:
            if age < TIER_FULL:
                _append_entry(lines, entry)
            elif age < TIER_COMPACT:
                _append_entry_compact(lines, entry)
            else:
                _append_entry_oneline(lines, entry)
            i += 1

    watchlist = []
    for e in reversed(entries):
        wl = e.get("watchlist", [])
        if wl:
            watchlist = wl
            break
    if watchlist:
        lines.append("### Watchlist")
        lines.append("")
        for item in watchlist:
            lines.append(f"- {item}")
        lines.append("")

    chains, ts_map = _build_continuation_chains(entries)
    if chains:
        lines.append("### Thesis Chains")
        lines.append("")
        for chain in chains:
            time_parts = [_fmt_time(ts).replace(" UTC", "") for ts in chain]
            tickers_in_chain = set()
            for ts in chain:
                e = ts_map[ts]
                for t in _non_neutral_tickers(e):
                    tickers_in_chain.add(t)
            ticker_str = (
                ", ".join(sorted(tickers_in_chain)) if tickers_in_chain else "general"
            )
            lines.append(f"{'  →  '.join(time_parts)} UTC: {ticker_str}")
        lines.append("")

    last = entries[-1]
    prices = last.get("prices", {})
    if prices:
        lines.append("### Prices at Last Entry")
        lines.append("")
        parts = []
        for t in TICKERS:
            p = prices.get(t)
            if p is not None:
                parts.append(f"{t}: ${p:,.2f}" if p >= 100 else f"{t}: ${p:,.4f}")
        lines.append(" | ".join(parts))
        lines.append("")

    if portfolio_data:
        lines.append("### Portfolio Snapshot")
        lines.append("")
        for label in ("patient", "bold"):
            d = portfolio_data.get(label)
            if not d:
                continue
            cash = d.get("cash_sek", 0)
            fees = d.get("total_fees_sek", 0) or 0
            trades = d.get("trades", 0)
            holdings = d.get("holdings", [])
            holding_str = ", ".join(holdings) if holdings else "none"
            lines.append(
                f"**{label.title()}:** {cash:,.0f} SEK cash | "
                f"{trades} trades | {fees:,.0f} fees | holding: {holding_str}"
            )
        lines.append("")

    warns = _detect_warnings(entries)
    if warns:
        lines.append("### Warnings")
        lines.append("")
        for w in warns:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)


def write_context():
    entries = load_recent()
    portfolio_data = _load_portfolio_pnl()
    md = build_context(entries, portfolio_data=portfolio_data)
    CONTEXT_FILE.write_text(md, encoding="utf-8")
    return len(entries)
