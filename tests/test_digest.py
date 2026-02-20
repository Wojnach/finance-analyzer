import json
from datetime import datetime, timedelta, timezone
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data")
INVOCATIONS_FILE = DATA_DIR / "invocations.jsonl"
AGENT_SUMMARY_FILE = DATA_DIR / "agent_summary.json"

now = datetime.now(timezone.utc)
cutoff = now - timedelta(hours=12)

entries = []
for line in INVOCATIONS_FILE.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line:
        entries.append(json.loads(line))

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
        else:
            reason_counts["other"] += 1

summary = json.loads(AGENT_SUMMARY_FILE.read_text(encoding="utf-8"))
fx_rate = summary.get("fx_rate", 10.5)
prices_usd = {t: s["price_usd"] for t, s in summary.get("signals", {}).items()}

state = json.loads((DATA_DIR / "portfolio_state.json").read_text(encoding="utf-8"))
cash = state["cash_sek"]
for t, h in state.get("holdings", {}).items():
    if h.get("shares", 0) > 0 and t in prices_usd:
        cash += h["shares"] * prices_usd[t] * fx_rate
p_total = cash
p_pnl = ((p_total - state["initial_value_sek"]) / state["initial_value_sek"]) * 100
p_holdings = [t for t, h in state.get("holdings", {}).items() if h.get("shares", 0) > 0]

lines = ["*12H DIGEST*", ""]
lines.append(
    f'_{cutoff.strftime("%H:%M")} - {now.strftime("%H:%M UTC")} ({now.strftime("%b %d")})_'
)
lines.append(f"_Triggers: {len(recent)}_")
for reason, count in reason_counts.most_common():
    lines.append(f"`{reason:<14} {count}`")
lines.append("")
lines.append(f"_Patient: {p_total:,.0f} SEK ({p_pnl:+.1f}%)_")

bold_path = DATA_DIR / "portfolio_state_bold.json"
if bold_path.exists():
    bold = json.loads(bold_path.read_text(encoding="utf-8"))
    bc = bold["cash_sek"]
    for t, h in bold.get("holdings", {}).items():
        if h.get("shares", 0) > 0 and t in prices_usd:
            bc += h["shares"] * prices_usd[t] * fx_rate
    b_pnl = ((bc - bold["initial_value_sek"]) / bold["initial_value_sek"]) * 100
    lines.append(f"_Bold: {bc:,.0f} SEK ({b_pnl:+.1f}%)_")

invoked = status_counts.get("invoked", 0)
skipped = status_counts.get("skipped_busy", 0)
lines.append(f"_Agent: {invoked} runs, {skipped} skipped_")

msg = "\n".join(lines)
print("=== MESSAGE ===")
print(msg)
print("=== SENDING ===")

import requests

c = json.load(open("config.json"))
tok = c["telegram"]["token"]
cid = c["telegram"]["chat_id"]
r = requests.post(
    f"https://api.telegram.org/bot{tok}/sendMessage",
    json={"chat_id": cid, "text": msg, "parse_mode": "Markdown"},
    timeout=10,
)
print(
    f"Status: {r.status_code}, ok: {r.json().get('ok')}, desc: {r.json().get('description', '')}"
)
