import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
CONTEXT_FILE = DATA_DIR / "layer2_context.md"

TICKERS = ["BTC-USD", "ETH-USD", "MSTR", "PLTR", "NVDA"]


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


def build_context(entries):
    if not entries:
        return "## Your Memory\n\nNo previous invocations. Fresh start.\n"

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
        if _is_all_hold(entry):
            hold_start = i
            while i < len(entries) and _is_all_hold(entries[i]):
                i += 1
            hold_count = i - hold_start
            if hold_count == 1:
                _append_entry(lines, entry)
            else:
                ts_range = _fmt_time_range(
                    entries[hold_start]["ts"], entries[i - 1]["ts"]
                )
                lines.append(f"**{ts_range}** | {hold_count}x HOLD (no setups)")
                lines.append("")
        else:
            _append_entry(lines, entry)
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

    return "\n".join(lines)


def _append_entry(lines, entry):
    ts = _fmt_time(entry["ts"])
    trigger = entry.get("trigger", "unknown")
    regime = entry.get("regime", "unknown")

    lines.append(f"**{ts}** | trigger: {trigger}")
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
        lines.append(f"{ticker}: {outlook} — {thesis}{level_str}")

    lines.append("")


def write_context():
    entries = load_recent()
    md = build_context(entries)
    CONTEXT_FILE.write_text(md, encoding="utf-8")
    return len(entries)
