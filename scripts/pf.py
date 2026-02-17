#!/usr/bin/env python3
"""pf — Portfolio CLI for mobile SSH."""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATE_FILE = DATA_DIR / "portfolio_state.json"
SUMMARY_FILE = DATA_DIR / "agent_summary.json"
TRIGGER_FILE = DATA_DIR / "trigger_state.json"
CONFIG_FILE = BASE_DIR / "config.json"

TICKER_ALIASES = {
    "btc": "BTC-USD",
    "btcusd": "BTC-USD",
    "btc-usd": "BTC-USD",
    "eth": "ETH-USD",
    "ethusd": "ETH-USD",
    "eth-usd": "ETH-USD",
    "mstr": "MSTR",
    "pltr": "PLTR",
    "nvda": "NVDA",
}

CRYPTO_SYMBOLS = {"BTC-USD", "ETH-USD"}
FEE_CRYPTO = 0.0005
FEE_STOCK = 0.001
MIN_TRADE_SEK = 500
W = 80


def load_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading {path.name}: {e}")
        sys.exit(1)


def _atomic_write_json(path, data):
    import os, tempfile

    path.parent.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def normalize_ticker(raw):
    return TICKER_ALIASES.get(raw.lower().replace(" ", ""), raw.upper())


def fmt_sek(v):
    if abs(v) >= 1_000_000:
        return f"{v:,.0f}"
    return f"{v:,.0f}"


def fmt_usd(v):
    if v >= 10000:
        return f"${v:,.0f}"
    if v >= 100:
        return f"${v:,.1f}"
    return f"${v:,.2f}"


def fmt_shares(v):
    if v >= 100:
        return f"{v:.1f}"
    if v >= 1:
        return f"{v:.4f}"
    if v >= 0.001:
        return f"{v:.6f}"
    return f"{v:.8f}"


def time_ago(iso_str):
    try:
        dt = datetime.fromisoformat(iso_str)
        delta = datetime.now(timezone.utc) - dt
        mins = int(delta.total_seconds() / 60)
        if mins < 1:
            return "just now"
        if mins < 60:
            return f"{mins}m ago"
        hours = mins // 60
        if hours < 24:
            return f"{hours}h{mins % 60}m ago"
        return f"{delta.days}d ago"
    except (ValueError, TypeError):
        return "unknown"


def file_age_minutes(path):
    try:
        return (time.time() - path.stat().st_mtime) / 60
    except OSError:
        return 999


def send_telegram(msg):
    try:
        config = json.loads(CONFIG_FILE.read_text())
        token = config["telegram"]["token"]
        chat_id = config["telegram"]["chat_id"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        print("Warning: Telegram not configured")
        return False
    try:
        import requests

        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
            timeout=30,
        )
        return r.ok
    except Exception as e:
        print(f"Telegram error: {e}")
        return False


# --- Commands ---


def cmd_status(_args):
    state = load_json(STATE_FILE)
    summary = load_json(SUMMARY_FILE)

    cash = state["cash_sek"]
    fx = summary.get("fx_rate", 0)
    initial = state.get("initial_value_sek", 500000)
    signals = summary.get("signals", {})

    prices = {t: s["price_usd"] for t, s in signals.items()}
    total = cash
    for ticker, h in state.get("holdings", {}).items():
        if h["shares"] > 0 and ticker in prices:
            total += h["shares"] * prices[ticker] * fx

    pnl = ((total - initial) / initial) * 100

    print(f"Portfolio  {fmt_sek(total)} SEK  ({pnl:+.2f}%)")
    print(f"Cash       {fmt_sek(cash)} SEK")
    print(f"USD/SEK    {fx:.2f}")
    print()

    holdings = state.get("holdings", {})
    has_holdings = any(h["shares"] > 0 for h in holdings.values())
    if has_holdings:
        print("HOLDINGS")
        for ticker, h in sorted(holdings.items()):
            if h["shares"] <= 0:
                continue
            price = prices.get(ticker, 0)
            val_sek = h["shares"] * price * fx
            cost_sek = h["shares"] * h["avg_cost_usd"] * fx
            pnl_h = ((val_sek - cost_sek) / cost_sek * 100) if cost_sek > 0 else 0
            print(
                f"  {ticker:<10} {fmt_shares(h['shares']):>12}"
                f"  {fmt_usd(price):>8}"
                f"  {fmt_sek(val_sek):>8} SEK"
                f"  {pnl_h:+.1f}%"
            )
    else:
        print("No holdings")

    age = file_age_minutes(SUMMARY_FILE)
    print(f"\nUpdated: {age:.0f}m ago")


def cmd_signals(_args):
    summary = load_json(SUMMARY_FILE)
    signals = summary.get("signals", {})
    timeframes = summary.get("timeframes", {})
    fg = summary.get("fear_greed", {})

    for ticker in sorted(signals.keys()):
        s = signals[ticker]
        action = s["action"]
        conf = s["confidence"]
        price = s["price_usd"]
        rsi = s.get("rsi", 0)
        macd = "+" if s.get("macd_hist", 0) >= 0 else "-"
        bb = s.get("bb_position", "?")[:2]
        extra = s.get("extra", {})

        conf_pct = int(conf * 100)
        print(
            f"{ticker:<10} {fmt_usd(price):>8}  "
            f"{action} {conf_pct}%  "
            f"RSI:{rsi:.0f} MACD:{macd} BB:{bb}"
        )

        parts = []
        fgi = fg.get(ticker, {})
        if fgi:
            val = fgi.get("value", "?")
            cls = fgi.get("classification", "")
            short_cls = cls.replace("Extreme ", "Ext")
            parts.append(f"F&G:{val}({short_cls})")

        sent = extra.get("sentiment")
        if sent and sent != "neutral":
            parts.append(f"News:{sent}")

        m_action = extra.get("ministral_action")
        if m_action:
            parts.append(f"8B:{m_action}")

        if parts:
            print(f"  {' '.join(parts)}")

        tfs = timeframes.get(ticker, [])
        tf_parts = []
        for tf in tfs:
            if tf["horizon"] == "Now":
                continue
            tf_action = tf["action"]
            tf_conf = int(tf["confidence"] * 100)
            if tf_action != "HOLD":
                tf_parts.append(f"{tf['horizon']} {tf_action} {tf_conf}%")
        if tf_parts:
            print(f"  {' '.join(tf_parts)}")
        print()


def cmd_trades(args):
    state = load_json(STATE_FILE)
    txns = state.get("transactions", [])
    n = args.n or 10
    recent = txns[-n:]

    if not recent:
        print("No trades yet")
        return

    for t in reversed(recent):
        ts = t.get("timestamp") or t.get("time", "")
        try:
            dt = datetime.fromisoformat(ts)
            date_str = dt.strftime("%m-%d %H:%M")
        except (ValueError, TypeError):
            date_str = ts[:16]
        action = t["action"]
        ticker = t["ticker"]
        shares = t["shares"]
        price = t["price_usd"]
        val_sek = t.get("price_sek", 0) * shares if "price_sek" in t else 0
        conf = int(t.get("confidence", 0) * 100)
        source = t.get("source", "")
        src_tag = " [M]" if source == "manual" else ""

        print(
            f"{date_str}  {action:<4} {ticker:<10} "
            f"{fmt_shares(shares):>10} @ {fmt_usd(price):<8} "
            f"{fmt_sek(val_sek):>8} SEK  {conf}%{src_tag}"
        )


def cmd_log(_args):
    age_summary = file_age_minutes(SUMMARY_FILE)
    age_trigger = file_age_minutes(TRIGGER_FILE)
    age_state = file_age_minutes(STATE_FILE)

    print(f"Data ages:")
    print(f"  agent_summary.json   {age_summary:5.1f}m")
    print(f"  trigger_state.json   {age_trigger:5.1f}m")
    print(f"  portfolio_state.json {age_state:5.1f}m")
    print()

    if TRIGGER_FILE.exists():
        trigger = load_json(TRIGGER_FILE)
        last_t = trigger.get("last_trigger_time", 0)
        if last_t:
            dt = datetime.fromtimestamp(last_t, tz=timezone.utc)
            print(
                f"Last trigger: {dt.strftime('%m-%d %H:%M')} UTC"
                f" ({time_ago(dt.isoformat())})"
            )
        last = trigger.get("last", {})
        sigs = last.get("signals", {})
        if sigs:
            print("Last signals:")
            for t, s in sorted(sigs.items()):
                print(f"  {t:<10} {s['action']} {int(s['confidence']*100)}%")
    print()

    state = load_json(STATE_FILE)
    txns = state.get("transactions", [])
    if txns:
        print(f"Total trades: {len(txns)}")
        last = txns[-1]
        ts = last.get("time", "")
        print(f"Last trade:   {last['action']} {last['ticker']} " f"({time_ago(ts)})")


def cmd_report(_args):
    print("Triggering Telegram report...")
    main_py = BASE_DIR / "portfolio" / "main.py"
    try:
        r = subprocess.run(
            [sys.executable, str(main_py), "--report"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r.returncode == 0:
            print("Report sent.")
        else:
            print(f"Error (exit {r.returncode}):")
            if r.stderr:
                print(r.stderr[-500:])
    except subprocess.TimeoutExpired:
        print("Timed out after 120s")


def cmd_buy(args):
    ticker = normalize_ticker(args.ticker)
    pct = (args.pct or 20) / 100.0

    summary = load_json(SUMMARY_FILE)
    state = load_json(STATE_FILE)
    signals = summary.get("signals", {})
    fx = summary.get("fx_rate", 0)

    if ticker not in signals:
        print(f"Unknown ticker: {ticker}")
        print(f"Available: {', '.join(sorted(signals.keys()))}")
        return

    age = file_age_minutes(SUMMARY_FILE)
    if age > 10:
        print(f"WARNING: Data is {age:.0f}m old!")

    price_usd = signals[ticker]["price_usd"]
    alloc_sek = state["cash_sek"] * pct
    if alloc_sek < MIN_TRADE_SEK:
        print(
            f"Allocation too small: {fmt_sek(alloc_sek)} SEK " f"(min {MIN_TRADE_SEK})"
        )
        return

    price_sek = price_usd * fx
    fee_rate = FEE_CRYPTO if ticker in CRYPTO_SYMBOLS else FEE_STOCK
    fee = alloc_sek * fee_rate
    net_alloc = alloc_sek - fee
    shares = net_alloc / price_sek

    cur = state.setdefault("holdings", {}).get(ticker, {"shares": 0, "avg_cost_usd": 0})
    total_shares = cur["shares"] + shares
    avg_cost = (
        (cur["shares"] * cur["avg_cost_usd"] + shares * price_usd) / total_shares
        if total_shares > 0
        else price_usd
    )

    state["holdings"][ticker] = {
        "shares": total_shares,
        "avg_cost_usd": avg_cost,
    }
    state["cash_sek"] -= alloc_sek
    state["total_fees_sek"] = round(state.get("total_fees_sek", 0) + fee, 2)

    trade = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "action": "BUY",
        "shares": shares,
        "price_usd": price_usd,
        "price_sek": price_sek,
        "total_sek": alloc_sek,
        "fee_sek": round(fee, 2),
        "confidence": 0,
        "fx_rate": fx,
        "source": "manual",
    }
    state.setdefault("transactions", []).append(trade)
    _atomic_write_json(STATE_FILE, state)

    print(f"BUY  {ticker}  {fmt_shares(shares)} @ {fmt_usd(price_usd)}")
    print(
        f"     {fmt_sek(alloc_sek)} SEK  ({pct*100:.0f}% of cash, fee {fmt_sek(fee)})"
    )
    print(f"     Cash remaining: {fmt_sek(state['cash_sek'])} SEK")

    msg = (
        f"*MANUAL BUY {ticker}*\n"
        f"{fmt_shares(shares)} @ {fmt_usd(price_usd)}\n"
        f"{fmt_sek(alloc_sek)} SEK ({pct*100:.0f}% of cash)"
    )
    if send_telegram(msg):
        print("     Telegram sent")


def cmd_sell(args):
    ticker = normalize_ticker(args.ticker)
    pct = (args.pct or 50) / 100.0

    summary = load_json(SUMMARY_FILE)
    state = load_json(STATE_FILE)
    signals = summary.get("signals", {})
    fx = summary.get("fx_rate", 0)

    holdings = state.get("holdings", {})
    if ticker not in holdings or holdings[ticker]["shares"] <= 0:
        print(f"No position in {ticker}")
        return

    if ticker not in signals:
        print(f"No price data for {ticker}")
        return

    age = file_age_minutes(SUMMARY_FILE)
    if age > 10:
        print(f"WARNING: Data is {age:.0f}m old!")

    price_usd = signals[ticker]["price_usd"]
    price_sek = price_usd * fx
    fee_rate = FEE_CRYPTO if ticker in CRYPTO_SYMBOLS else FEE_STOCK
    cur = holdings[ticker]
    sell_shares = cur["shares"] * pct
    proceeds = sell_shares * price_sek
    fee = proceeds * fee_rate
    net_proceeds = proceeds - fee

    cur["shares"] -= sell_shares
    state["holdings"][ticker] = cur
    state["cash_sek"] += net_proceeds
    state["total_fees_sek"] = round(state.get("total_fees_sek", 0) + fee, 2)

    trade = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "action": "SELL",
        "shares": sell_shares,
        "price_usd": price_usd,
        "price_sek": price_sek,
        "total_sek": net_proceeds,
        "fee_sek": round(fee, 2),
        "confidence": 0,
        "fx_rate": fx,
        "source": "manual",
    }
    state.setdefault("transactions", []).append(trade)
    _atomic_write_json(STATE_FILE, state)

    print(f"SELL {ticker}  {fmt_shares(sell_shares)} @ {fmt_usd(price_usd)}")
    print(
        f"     {fmt_sek(net_proceeds)} SEK  ({pct*100:.0f}% of position, fee {fmt_sek(fee)})"
    )
    print(f"     Cash now: {fmt_sek(state['cash_sek'])} SEK")

    msg = (
        f"*MANUAL SELL {ticker}*\n"
        f"{fmt_shares(sell_shares)} @ {fmt_usd(price_usd)}\n"
        f"{fmt_sek(proceeds)} SEK ({pct*100:.0f}% of position)"
    )
    if send_telegram(msg):
        print("     Telegram sent")


def cmd_pause(_args):
    r = subprocess.run(
        ["schtasks", "/Change", "/TN", "PF-DataLoop", "/DISABLE"],
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        print("PF-DataLoop DISABLED")
    else:
        print(f"Error: {r.stderr.strip()}")


def cmd_resume(_args):
    r = subprocess.run(
        ["schtasks", "/Change", "/TN", "PF-DataLoop", "/ENABLE"],
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        print("PF-DataLoop ENABLED")
    else:
        print(f"Error: {r.stderr.strip()}")


def main():
    p = argparse.ArgumentParser(
        prog="pf",
        description="Portfolio CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("status", help="Portfolio overview")
    sub.add_parser("signals", help="Latest signals")
    sub.add_parser("log", help="System status + trigger info")
    sub.add_parser("report", help="Force Telegram report")
    sub.add_parser("pause", help="Disable PF-DataLoop")
    sub.add_parser("resume", help="Enable PF-DataLoop")

    sp_trades = sub.add_parser("trades", help="Recent trades")
    sp_trades.add_argument("n", nargs="?", type=int, help="Number of trades")

    sp_buy = sub.add_parser("buy", help="Manual buy")
    sp_buy.add_argument("ticker", help="Ticker (btc, eth, mstr, pltr)")
    sp_buy.add_argument(
        "pct", nargs="?", type=float, help="Percent of cash (default 20)"
    )

    sp_sell = sub.add_parser("sell", help="Manual sell")
    sp_sell.add_argument("ticker", help="Ticker")
    sp_sell.add_argument(
        "pct", nargs="?", type=float, help="Percent of position (default 50)"
    )

    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        return

    cmds = {
        "status": cmd_status,
        "signals": cmd_signals,
        "trades": cmd_trades,
        "log": cmd_log,
        "report": cmd_report,
        "buy": cmd_buy,
        "sell": cmd_sell,
        "pause": cmd_pause,
        "resume": cmd_resume,
    }
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()
