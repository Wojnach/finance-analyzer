"""System Health Contract checker with auto-remediation.

Runs invariant checks defined in docs/SYSTEM_HEALTH_CONTRACT.md.
Automatically fixes what it can (stale locks, stopped tasks).
Reports unfixable issues via Telegram.

Three tiers control scope:
  --tier full      (11:00 CET)  All 10 sections
  --tier pre-us    (15:25 CET)  Sections 1-6
  --tier post-us   (22:05 CET)  Sections 1, 5

Usage:
  .venv/Scripts/python.exe scripts/health_check.py --tier full
  .venv/Scripts/python.exe scripts/health_check.py --tier full --dry-run
"""

import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time

import requests

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

sys.path.insert(0, str(BASE_DIR))
from portfolio.api_utils import load_config
from portfolio.file_utils import load_json, load_jsonl_tail
from portfolio.message_store import send_or_store

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


def _age_minutes(iso_ts):
    """Return age in minutes of an ISO timestamp string."""
    if not iso_ts:
        return float("inf")
    try:
        dt = datetime.datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return (_utcnow() - dt).total_seconds() / 60
    except Exception:
        return float("inf")


def _age_minutes_epoch(epoch):
    """Return age in minutes of a Unix epoch timestamp."""
    if not epoch:
        return float("inf")
    try:
        return (time.time() - float(epoch)) / 60
    except Exception:
        return float("inf")


def _file_age_hours(path):
    """Return age in hours of a file's mtime."""
    try:
        return (time.time() - os.path.getmtime(path)) / 3600
    except Exception:
        return float("inf")


def _ps(cmd):
    """Run a PowerShell command and return stdout."""
    try:
        r = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", cmd],
            capture_output=True, text=True, timeout=15,
        )
        return r.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"


def _tail_file(path, lines=50):
    """Read last N lines of a file."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            chunk = min(size, lines * 200)
            f.seek(max(0, size - chunk))
            data = f.read().decode("utf-8", errors="replace")
        return data.split("\n")[-lines:]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Result type: (status, detail, [remediation_actions])
#   status: "ok", "fixed", "fail"
#   detail: human-readable summary
# ---------------------------------------------------------------------------

def check_1_process_liveness(dry_run=False):
    """Section 1: Process Liveness."""
    fails = []
    fixes = []

    for task in ("PF-DataLoop", "PF-MetalsLoop"):
        out = _ps(f'(Get-ScheduledTask -TaskName "{task}").State')
        if "Running" not in out:
            if dry_run:
                fails.append(f"{task} stopped (would restart)")
            else:
                _ps(f'Start-ScheduledTask -TaskName "{task}"')
                time.sleep(3)
                verify = _ps(f'(Get-ScheduledTask -TaskName "{task}").State')
                if "Running" in verify:
                    fixes.append(f"Restarted {task}")
                else:
                    fails.append(f"{task} restart failed (state={verify})")

    # Check main loop PID
    trigger = load_json(DATA_DIR / "trigger_state.json", {})
    loop_pid = trigger.get("last_loop_pid")
    if loop_pid:
        out = _ps(f"(Get-Process -Id {loop_pid} -ErrorAction SilentlyContinue).Id")
        if str(loop_pid) not in out:
            fails.append(f"Main loop PID {loop_pid} not found")

    if fails:
        return "fail", "; ".join(fails + fixes)
    if fixes:
        return "fixed", "; ".join(fixes)
    return "ok", "PF-DataLoop + PF-MetalsLoop running"


def check_2_heartbeat(dry_run=False):
    """Section 2: Heartbeat & Cycle Health."""
    fails = []
    hs = load_json(DATA_DIR / "health_state.json", {})

    hb_age = _age_minutes(hs.get("last_heartbeat"))
    if hb_age > 5:
        fails.append(f"Heartbeat {hb_age:.0f}m old (>5m)")

    errs = hs.get("error_count", 0)
    if errs > 0:
        fails.append(f"error_count={errs}")

    sh = hs.get("signal_health", {})
    for mod, data in sh.items():
        recent = data.get("recent_results", [])
        false_count = sum(1 for r in recent[-10:] if not r)
        if false_count > 0:
            fails.append(f"signal '{mod}' {false_count} recent failures")

    acceptable = {"monte_carlo", "price_targets", "equity_curve"}
    mf = hs.get("last_module_failures", {}).get("modules", [])
    unexpected = [m for m in mf if m not in acceptable]
    if unexpected:
        fails.append(f"module failures: {unexpected}")

    cycle = hs.get("cycle_count", "?")
    if fails:
        return "fail", "; ".join(fails)
    return "ok", f"Heartbeat {hb_age:.0f}m, cycle #{cycle}"


def check_3_signal_coverage(dry_run=False):
    """Section 3: Signal Coverage."""
    fails = []

    # Active tickers (8 stocks removed Mar 15 2026, commit 40dec15)
    expected = {
        "BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD",
        "PLTR", "NVDA", "MU", "SMCI", "TSM", "TTWO", "VRT", "MSTR",
    }

    # US stocks are only in the summary during market hours
    now_cet = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=2))
    )
    us_open = 15 <= now_cet.hour < 22 and now_cet.weekday() < 5
    if not us_open:
        expected_now = {"BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD"}
    else:
        expected_now = expected

    summary = load_json(DATA_DIR / "agent_summary.json", {})
    tickers_found = set()
    if isinstance(summary, dict):
        for key in summary:
            if key in expected:
                tickers_found.add(key)
        for section in ("tickers", "signals", "data"):
            sub = summary.get(section, {})
            if isinstance(sub, dict):
                for key in sub:
                    if key in expected:
                        tickers_found.add(key)

    missing = expected_now - tickers_found
    if missing:
        fails.append(f"Missing tickers: {sorted(missing)}")

    acc_age = _file_age_hours(DATA_DIR / "accuracy_cache.json")
    if acc_age > 48:
        fails.append(f"accuracy_cache.json {acc_age:.0f}h old")

    if fails:
        return "fail", "; ".join(fails)
    return "ok", f"{len(tickers_found)}/{len(expected)} tickers"


def check_4_llm_inference(dry_run=False):
    """Section 4: LLM Inference (Metals Loop)."""
    fails = []
    fixes = []
    expected = {"XAG-USD", "XAU-USD", "BTC-USD", "ETH-USD"}
    found_chronos = set()
    found_ministral = set()

    lines = _tail_file(DATA_DIR / "metals_loop_out.txt", 200)

    for line in lines:
        if "[LLM] Chronos" in line:
            for t in expected:
                if t in line:
                    found_chronos.add(t)
        if "[LLM] Ministral" in line and "error" not in line.lower():
            for t in expected:
                if t in line:
                    found_ministral.add(t)

    missing_chronos = expected - found_chronos
    if missing_chronos:
        # Try restarting metals loop
        if not dry_run:
            _ps('Stop-ScheduledTask -TaskName "PF-MetalsLoop"')
            time.sleep(3)
            _ps('Start-ScheduledTask -TaskName "PF-MetalsLoop"')
            fixes.append(f"Restarted PF-MetalsLoop (missing Chronos for {sorted(missing_chronos)})")
        else:
            fails.append(f"No Chronos for: {sorted(missing_chronos)} (would restart)")

    missing_ministral = expected - found_ministral
    # Ministral runs every 5 min, so missing is less urgent
    if missing_ministral and not missing_chronos:
        fails.append(f"No recent Ministral for: {sorted(missing_ministral)}")

    if fails:
        return "fail", "; ".join(fails + fixes)
    if fixes:
        return "fixed", "; ".join(fixes)
    return "ok", f"Chronos {len(found_chronos)}/4, Ministral {len(found_ministral)}/4"


def check_5_layer2_agent(dry_run=False):
    """Section 5: Layer 2 Agent."""
    agent_log = DATA_DIR / "agent.log"
    if not agent_log.exists():
        return "ok", "agent.log absent (Layer 2 disabled)"

    lines = _tail_file(agent_log, 15)
    text = "\n".join(lines)

    if "Not logged in" in text:
        login_count = sum(1 for l in lines if "Not logged in" in l)
        if login_count >= 5:
            return "fail", f"Claude CLI auth expired ({login_count}x). Run: claude /login"

    if "nested session" in text.lower():
        return "fail", "Nested session errors"

    hs = load_json(DATA_DIR / "health_state.json", {})
    inv_age = _age_minutes(hs.get("last_invocation_ts"))
    return "ok", f"Last invocation {inv_age:.0f}m ago"


def check_6_telegram(dry_run=False):
    """Section 6: Telegram Delivery."""
    fails = []

    entries = load_jsonl_tail(DATA_DIR / "telegram_messages.jsonl", max_entries=20)
    unsent_old = 0
    for entry in entries:
        if not entry.get("sent", True):
            age = _age_minutes(entry.get("ts"))
            if age > 30:
                unsent_old += 1

    if unsent_old > 2:
        fails.append(f"{unsent_old} unsent msgs >30m old")

    # Digest uses digest_state.json (migrated from trigger_state.json)
    digest_state = load_json(DATA_DIR / "digest_state.json", {})
    digest_age = _age_minutes_epoch(digest_state.get("last_digest_time"))
    if digest_age > 300:
        fails.append(f"Digest {digest_age / 60:.0f}h old")

    daily_state = load_json(DATA_DIR / "daily_digest_state.json", {})
    daily_ts = daily_state.get("last_daily_digest_time") or \
        load_json(DATA_DIR / "trigger_state.json", {}).get("last_daily_digest_time")
    daily_age = _age_minutes_epoch(daily_ts)
    if daily_age > 1500:
        fails.append(f"Daily digest {daily_age / 60:.0f}h old")

    if fails:
        return "fail", "; ".join(fails)
    return "ok", "Delivery OK"


def check_7_data_freshness(dry_run=False):
    """Section 7: Data Freshness."""
    fails = []

    if load_json(DATA_DIR / "portfolio_state.json") is None:
        fails.append("portfolio_state.json missing/corrupt")

    fund_age = _file_age_hours(DATA_DIR / "fundamentals_cache.json")
    if fund_age > 48:
        fails.append(f"fundamentals_cache {fund_age:.0f}h old")

    if load_json(DATA_DIR / "prophecy.json") is None:
        fails.append("prophecy.json missing")

    fh_age = _file_age_hours(DATA_DIR / "forecast_health.jsonl")
    if fh_age > 24:
        fails.append(f"forecast_health {fh_age:.0f}h old")

    if fails:
        return "fail", "; ".join(fails)
    return "ok", "Data files fresh"


def check_8_singleton_locks(dry_run=False):
    """Section 8: Singleton Locks."""
    fails = []
    fixes = []

    for lock_file in DATA_DIR.glob("*.singleton.lock"):
        size = lock_file.stat().st_size if lock_file.exists() else 0

        if size == 0:
            if dry_run:
                fails.append(f"{lock_file.name} 0 bytes (would delete)")
            else:
                lock_file.unlink(missing_ok=True)
                fixes.append(f"Deleted stale {lock_file.name}")
            continue

        # Check if PID is alive
        try:
            content = lock_file.read_text().strip()
            pid_data = json.loads(content) if content.startswith("{") else {"pid": content}
            pid = str(pid_data.get("pid", content))
            out = _ps(f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue).Id")
            if pid not in out:
                if dry_run:
                    fails.append(f"{lock_file.name} PID {pid} dead (would delete)")
                else:
                    lock_file.unlink(missing_ok=True)
                    fixes.append(f"Deleted stale {lock_file.name} (PID {pid} dead)")
        except Exception:
            age_h = _file_age_hours(lock_file)
            if age_h > 24:
                if dry_run:
                    fails.append(f"{lock_file.name} unparseable, {age_h:.0f}h old (would delete)")
                else:
                    lock_file.unlink(missing_ok=True)
                    fixes.append(f"Deleted stale {lock_file.name}")

    if fails:
        return "fail", "; ".join(fails + fixes)
    if fixes:
        return "fixed", "; ".join(fixes)
    return "ok", "No stale locks"


def check_9_api_connectivity(dry_run=False):
    """Section 9: External API Connectivity."""
    fails = []

    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
            timeout=5,
        )
        if r.status_code != 200:
            fails.append(f"Binance spot HTTP {r.status_code}")
    except Exception as e:
        fails.append(f"Binance spot: {e}")

    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/ticker/price?symbol=XAGUSDT",
            timeout=5,
        )
        if r.status_code != 200:
            fails.append(f"Binance FAPI HTTP {r.status_code}")
    except Exception as e:
        fails.append(f"Binance FAPI: {e}")

    if fails:
        return "fail", "; ".join(fails)
    return "ok", "Binance spot + FAPI OK"


def check_10_dashboard(dry_run=False):
    """Section 10: Dashboard."""
    out = _ps('(Get-ScheduledTask -TaskName "PF-Dashboard" -ErrorAction SilentlyContinue).State')
    if "Running" in out:
        try:
            r = requests.get("http://localhost:5055/api/health", timeout=3)
            if r.status_code == 200:
                return "ok", "Dashboard responsive"
            return "fail", f"Dashboard HTTP {r.status_code}"
        except Exception:
            return "fail", "Dashboard task running but port 5055 dead"
    return "ok", f"Dashboard {out or 'Disabled'} (optional)"


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

ALL_CHECKS = {
    1: ("Process Liveness", check_1_process_liveness),
    2: ("Heartbeat & Cycles", check_2_heartbeat),
    3: ("Signal Coverage", check_3_signal_coverage),
    4: ("LLM Inference", check_4_llm_inference),
    5: ("Layer 2 Agent", check_5_layer2_agent),
    6: ("Telegram Delivery", check_6_telegram),
    7: ("Data Freshness", check_7_data_freshness),
    8: ("Singleton Locks", check_8_singleton_locks),
    9: ("API Connectivity", check_9_api_connectivity),
    10: ("Dashboard", check_10_dashboard),
}

TIERS = {
    "full": list(range(1, 11)),
    "pre-us": list(range(1, 7)),
    "post-us": [1, 5],
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_checks(tier="full", dry_run=False):
    sections = TIERS[tier]
    results = []
    for num in sections:
        name, fn = ALL_CHECKS[num]
        try:
            status, detail = fn(dry_run=dry_run)
        except Exception as e:
            status, detail = "fail", f"Check crashed: {e}"
        results.append((num, name, status, detail))
    return results


def format_message(results, tier):
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=2)))
    ts = now.strftime("%Y-%m-%d %H:%M CET")
    tier_label = {"full": "Full", "pre-us": "Pre-US", "post-us": "Post-US"}[tier]

    ok_count = sum(1 for _, _, s, _ in results if s == "ok")
    fixed_count = sum(1 for _, _, s, _ in results if s == "fixed")
    fail_count = sum(1 for _, _, s, _ in results if s == "fail")
    total = len(results)
    all_ok = fail_count == 0

    lines = []
    if all_ok and fixed_count == 0:
        hs = load_json(DATA_DIR / "health_state.json", {})
        cycle = hs.get("cycle_count", "?")
        uptime_d = hs.get("uptime_seconds", 0) / 86400
        lines.append(f"*Health Check ({tier_label}) — {ts}*")
        lines.append(f"All {total} checks passed. Cycle #{cycle}, {uptime_d:.1f}d uptime.")
    elif all_ok and fixed_count > 0:
        lines.append(f"*Health Check ({tier_label}) — {ts}*")
        lines.append(f"{ok_count} OK, {fixed_count} auto-fixed.")
        lines.append("")
        lines.append("*AUTO-FIXED:*")
        for num, name, status, detail in results:
            if status == "fixed":
                lines.append(f"  [{num}] {name}: {detail}")
    else:
        lines.append(f"*Health Check ({tier_label}) — {ts}*")
        if fixed_count > 0:
            lines.append("")
            lines.append("*AUTO-FIXED:*")
            for num, name, status, detail in results:
                if status == "fixed":
                    lines.append(f"  [{num}] {name}: {detail}")
        lines.append("")
        lines.append("*NEEDS ATTENTION:*")
        for num, name, status, detail in results:
            if status == "fail":
                lines.append(f"  [{num}] {name}: {detail}")
        lines.append("")
        lines.append(f"{ok_count} OK / {fixed_count} fixed / {fail_count} failed")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="System Health Contract checker")
    parser.add_argument("--tier", choices=["full", "pre-us", "post-us"], default="full")
    parser.add_argument("--dry-run", action="store_true", help="Print but don't fix or send")
    args = parser.parse_args()

    print(f"Health check: tier={args.tier} dry_run={args.dry_run}")
    results = run_checks(args.tier, dry_run=args.dry_run)

    for num, name, status, detail in results:
        tag = {"ok": "OK  ", "fixed": "FIX ", "fail": "FAIL"}[status]
        print(f"  [{num}] {tag} {name}: {detail}")

    msg = format_message(results, args.tier)

    fail_count = sum(1 for _, _, s, _ in results if s == "fail")
    fixed_count = sum(1 for _, _, s, _ in results if s == "fixed")

    if args.dry_run:
        print(f"\n--- Telegram message (dry run) ---\n{msg}")
    elif fail_count > 0 or fixed_count > 0:
        config = load_config()
        ok = send_or_store(msg, config, category="health")
        print(f"\nTelegram: {'sent' if ok else 'FAILED'}")
    else:
        print("\nAll OK — no Telegram needed.")


if __name__ == "__main__":
    main()
