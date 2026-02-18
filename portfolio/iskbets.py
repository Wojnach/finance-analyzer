"""ISKBETS — Intraday quick-gamble mode.

Scans for entry conditions every 60s cycle, sends Telegram alerts, monitors
exit conditions using an ATR-based ladder. User trades manually on Avanza
and confirms via Telegram replies.
"""

import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CONFIG_FILE = DATA_DIR / "iskbets_config.json"
STATE_FILE = DATA_DIR / "iskbets_state.json"

BINANCE_BASE = "https://api.binance.com/api/v3"
ALPACA_BASE = "https://data.alpaca.markets/v2"

# Ticker → source mapping (mirrors main.py SYMBOLS)
TICKER_SOURCES = {
    "BTC-USD": {"binance": "BTCUSDT"},
    "ETH-USD": {"binance": "ETHUSDT"},
    "MSTR": {"alpaca": "MSTR"},
    "PLTR": {"alpaca": "PLTR"},
    "NVDA": {"alpaca": "NVDA"},
}
CRYPTO_TICKERS = {"BTC-USD", "ETH-USD"}


# ── State I/O ────────────────────────────────────────────────────────────


def _load_config():
    """Load per-session ISKBETS config. Returns dict or None if disabled/expired."""
    if not CONFIG_FILE.exists():
        return None
    try:
        cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not cfg.get("enabled", False):
        return None
    # Check expiry
    expiry = cfg.get("expiry")
    if expiry:
        try:
            exp_dt = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) > exp_dt:
                # Auto-disable
                cfg["enabled"] = False
                _save_config(cfg)
                print("  ISKBETS: Session expired, auto-disabled")
                return None
        except (ValueError, TypeError):
            pass
    return cfg


def _save_config(cfg):
    """Atomic write of iskbets config."""
    DATA_DIR.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=DATA_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, default=str)
        os.replace(tmp, str(CONFIG_FILE))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_state():
    """Load ISKBETS runtime state."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"active_position": None, "trade_history": []}


def _save_state(state):
    """Atomic write of ISKBETS state."""
    DATA_DIR.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=DATA_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, str(STATE_FILE))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── Telegram ─────────────────────────────────────────────────────────────


def _send_telegram(msg, config):
    """Send a Telegram message."""
    token = config["telegram"]["token"]
    chat_id = config["telegram"]["chat_id"]
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
        timeout=30,
    )
    if not r.ok:
        print(f"  ISKBETS Telegram error: {r.status_code} {r.text[:200]}")


def _log_telegram(msg):
    """Append message to telegram log."""
    log = DATA_DIR / "telegram_messages.jsonl"
    try:
        with open(log, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"ts": datetime.now(timezone.utc).isoformat(), "text": msg},
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:
        pass


# ── ATR Computation ──────────────────────────────────────────────────────


def _get_alpaca_headers(config):
    acfg = config.get("alpaca", {})
    return {
        "APCA-API-KEY-ID": acfg.get("key", ""),
        "APCA-API-SECRET-KEY": acfg.get("secret", ""),
    }


def compute_atr_15m(ticker, config):
    """Fetch 15-min candles and compute ATR(14). Returns ATR value in USD."""
    source = TICKER_SOURCES.get(ticker)
    if not source:
        raise ValueError(f"Unknown ticker: {ticker}")

    if "binance" in source:
        r = requests.get(
            f"{BINANCE_BASE}/klines",
            params={
                "symbol": source["binance"],
                "interval": "15m",
                "limit": 20,
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(
            data,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_vol", "trades", "taker_buy_vol",
                "taker_buy_quote_vol", "ignore",
            ],
        )
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
    else:
        # Alpaca stocks
        end = datetime.now(timezone.utc)
        start = end - pd.Timedelta(days=2)
        r = requests.get(
            f"{ALPACA_BASE}/stocks/{source['alpaca']}/bars",
            headers=_get_alpaca_headers(config),
            params={
                "timeframe": "15Min",
                "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "feed": "iex",
                "adjustment": "split",
            },
            timeout=10,
        )
        r.raise_for_status()
        bars = r.json().get("bars") or []
        if not bars:
            raise ValueError(f"No Alpaca 15m data for {ticker}")
        df = pd.DataFrame(bars)
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df = df.tail(20)

    close = df["close"]
    high = df["high"]
    low = df["low"]

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(span=14, adjust=False).mean().iloc[-1]
    return float(atr)


# ── Entry Evaluation ─────────────────────────────────────────────────────


def _evaluate_entry(ticker, signals, prices_usd, tf_data, iskbets_cfg, app_config):
    """Check if entry conditions are met for a ticker.

    Returns (should_enter, conditions_list).
    Gate 1: ≥min_bigbet_conditions of 6 bigbet conditions.
    Gate 2: ≥min_buy_votes from main signal grid.
    Time gate: No entry after entry_cutoff_et.
    """
    # Time gate
    cutoff_str = iskbets_cfg.get("entry_cutoff_et", "14:30")
    if not _before_cutoff(cutoff_str):
        return False, ["Past entry cutoff"]

    min_bigbet = iskbets_cfg.get("min_bigbet_conditions", 2)
    min_votes = iskbets_cfg.get("min_buy_votes", 3)

    # Gate 1: Reuse bigbet condition evaluation
    from portfolio.bigbet import _evaluate_conditions

    bull_conds, _bear_conds, _extra = _evaluate_conditions(
        ticker, signals, prices_usd, tf_data
    )

    if len(bull_conds) < min_bigbet:
        return False, []

    # Gate 2: Buy vote count from signals
    sig = signals.get(ticker)
    if not sig:
        return False, []
    extra = sig.get("extra", {})
    buy_count = extra.get("_buy_count", 0)

    if buy_count < min_votes:
        return False, []

    conditions = list(bull_conds)
    conditions.append(f"Signal grid: {buy_count} BUY votes")
    return True, conditions


def _before_cutoff(cutoff_str):
    """Check if current time is before the ET cutoff. cutoff_str like '14:30'."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo

    now_et = datetime.now(ZoneInfo("America/New_York"))
    parts = cutoff_str.split(":")
    cutoff_hour = int(parts[0])
    cutoff_min = int(parts[1]) if len(parts) > 1 else 0
    return (now_et.hour < cutoff_hour) or (
        now_et.hour == cutoff_hour and now_et.minute < cutoff_min
    )


def _past_time_exit(time_exit_str):
    """Check if current time is past the ET time exit."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo

    now_et = datetime.now(ZoneInfo("America/New_York"))
    parts = time_exit_str.split(":")
    exit_hour = int(parts[0])
    exit_min = int(parts[1]) if len(parts) > 1 else 0
    return (now_et.hour > exit_hour) or (
        now_et.hour == exit_hour and now_et.minute >= exit_min
    )


# ── Layer 2 Gate ─────────────────────────────────────────────────────────


GATE_LOG_FILE = DATA_DIR / "iskbets_gate_log.jsonl"


def _build_gate_prompt(ticker, price, conditions, signals, tf_data, atr, config):
    """Build a minimal prompt for the Layer 2 APPROVE/SKIP gate."""
    sig = signals.get(ticker, {})
    extra = sig.get("extra", {})
    ind = sig.get("indicators", {})

    buy_c = extra.get("_buy_count", 0)
    sell_c = extra.get("_sell_count", 0)
    hold_c = extra.get("_total_applicable", 11) - buy_c - sell_c

    rsi = ind.get("rsi", "N/A")
    macd = ind.get("macd_hist", "N/A")
    bb = ind.get("price_vs_bb", "N/A")
    atr_pct = (atr / price * 100) if price > 0 else 0

    fg = extra.get("fear_greed", "N/A")

    # Build TF heatmap row for this ticker
    tf_row = ""
    tf_list = tf_data.get(ticker, [])
    if tf_list:
        labels = []
        actions = []
        for label, td in tf_list:
            labels.append(label)
            a = td.get("action")
            if a == "BUY":
                actions.append("B")
            elif a == "SELL":
                actions.append("S")
            else:
                actions.append("H")
        tf_row = "/".join(labels) + ": " + " ".join(actions)

    # FOMC proximity from agent_summary if available
    fomc_days = "N/A"
    try:
        summary_file = DATA_DIR / "agent_summary.json"
        if summary_file.exists():
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            macro = summary.get("macro", {})
            fomc_days = macro.get("fomc_days_until", "N/A")
    except Exception:
        pass

    cond_str = "\n".join(f"- {c}" for c in conditions)

    return (
        f"You are a fast entry gate for ISKBETS (intraday trading).\n\n"
        f"Layer 1 conditions PASSED for {ticker} at ${price:,.2f}:\n"
        f"{cond_str}\n\n"
        f"Signals: {buy_c}B/{sell_c}S/{hold_c}H\n"
        f"RSI {rsi} | MACD {macd} | BB {bb} | ATR ${atr:,.2f} ({atr_pct:.1f}%)\n\n"
        f"Timeframes ({tf_row})\n\n"
        f"F&G: {fg} | FOMC: {fomc_days}d away\n\n"
        f"Respond EXACTLY:\n"
        f"DECISION: APPROVE or SKIP\n"
        f"REASONING: One sentence.\n\n"
        f"APPROVE if setup is clean. SKIP only for clear red flags (all long TFs opposing, "
        f"chasing end of move, imminent FOMC). Default to APPROVE when uncertain."
    )


def _parse_gate_response(output):
    """Parse DECISION: APPROVE|SKIP and REASONING: ... from gate output.

    Returns (approved: bool, reasoning: str). Defaults to (True, "") on parse failure.
    """
    approved = True
    reasoning = ""

    for line in output.strip().splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("DECISION:"):
            val = line.split(":", 1)[1].strip().upper()
            if "SKIP" in val:
                approved = False
            # APPROVE is the default, so only SKIP changes it
        elif upper.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()

    return approved, reasoning


def invoke_layer2_gate(ticker, price, conditions, signals, tf_data, atr, iskbets_cfg, config):
    """Invoke Claude CLI for a fast APPROVE/SKIP decision on an ISKBETS entry.

    Returns (approved: bool, reasoning: str).
    Defaults to (True, "") on any failure — Layer 2 is additive, never blocking.
    """
    if not iskbets_cfg.get("layer2_gate", False):
        return True, ""

    prompt = _build_gate_prompt(ticker, price, conditions, signals, tf_data, atr, config)

    t0 = time.time()
    approved = True
    reasoning = ""

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--max-turns", "1"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        elapsed = time.time() - t0
        output = result.stdout.strip()

        if result.returncode == 0 and output:
            approved, reasoning = _parse_gate_response(output)
        else:
            print(f"  ISKBETS L2 GATE: claude returned code {result.returncode}")
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  ISKBETS L2 GATE: timeout after {elapsed:.1f}s")
    except FileNotFoundError:
        elapsed = time.time() - t0
        print("  ISKBETS L2 GATE: claude not found in PATH")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ISKBETS L2 GATE: error — {e}")

    # Log decision
    try:
        GATE_LOG_FILE.parent.mkdir(exist_ok=True)
        with open(GATE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "ticker": ticker,
                        "price": price,
                        "approved": approved,
                        "reasoning": reasoning,
                        "elapsed_s": round(time.time() - t0, 2),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:
        pass

    return approved, reasoning


# ── Exit Checks ──────────────────────────────────────────────────────────


def check_exits(state, prices_usd, signals, tf_data, iskbets_cfg):
    """Check exit conditions for an active position.

    Returns (exit_type, detail_msg) or None.
    Priority: hard_stop > time_exit > stage1 > trailing > signal_reversal.
    """
    pos = state.get("active_position")
    if not pos:
        return None

    ticker = pos["ticker"]
    price = prices_usd.get(ticker, 0)
    if price <= 0:
        return None

    entry_price = pos["entry_price_usd"]
    atr = pos["atr_15m"]
    stop_at_breakeven = pos.get("stop_at_breakeven", False)

    hard_stop_mult = iskbets_cfg.get("hard_stop_atr_mult", 2.0)
    stage1_mult = iskbets_cfg.get("stage1_atr_mult", 1.5)
    trailing_mult = iskbets_cfg.get("trailing_atr_mult", 1.0)
    time_exit_str = iskbets_cfg.get("time_exit_et", "15:50")

    # Update highest price
    highest = max(pos.get("highest_price", entry_price), price)
    pos["highest_price"] = highest

    # Priority 1: Hard stop
    hard_stop = entry_price - (hard_stop_mult * atr)
    if price <= hard_stop:
        return "hard_stop", f"Hard stop hit at ${price:,.2f} (stop ${hard_stop:,.2f})"

    # Priority 2: Time exit
    if _past_time_exit(time_exit_str):
        return "time_exit", f"Time exit at ${price:,.2f} (past {time_exit_str} ET)"

    # Priority 3: Stage 1 target → move stop to breakeven
    stage1_target = entry_price + (stage1_mult * atr)
    if not stop_at_breakeven and price >= stage1_target:
        pos["stop_at_breakeven"] = True
        pos["stop_loss"] = entry_price  # Move stop to breakeven
        return "stage1_hit", f"Stage 1 hit at ${price:,.2f} (target ${stage1_target:,.2f}). Stop moved to breakeven ${entry_price:,.2f}"

    # Priority 4: Trailing stop
    if stop_at_breakeven:
        trailing_stop = max(entry_price, highest - (trailing_mult * atr))
    else:
        trailing_stop = entry_price - (hard_stop_mult * atr)

    if stop_at_breakeven and price <= trailing_stop:
        return "trailing_stop", f"Trailing stop hit at ${price:,.2f} (stop ${trailing_stop:,.2f})"

    # Priority 5: Signal reversal (≥3 sell votes for 2 consecutive cycles)
    sig = signals.get(ticker, {})
    extra = sig.get("extra", {})
    sell_count = extra.get("_sell_count", 0)

    if sell_count >= 3:
        streak = pos.get("sell_signal_streak", 0) + 1
        pos["sell_signal_streak"] = streak
        if streak >= 2:
            return "signal_reversal", f"Signal reversal: {sell_count} SELL votes for {streak} consecutive cycles"
    else:
        pos["sell_signal_streak"] = 0

    # Update stop_loss in state for display
    pos["stop_loss"] = trailing_stop if stop_at_breakeven else hard_stop

    return None


# ── Alert Formatting ─────────────────────────────────────────────────────


def format_entry_alert(ticker, price, conditions, atr, iskbets_cfg, signals=None, l2_reasoning=""):
    """Format Telegram entry alert message."""
    hard_stop_mult = iskbets_cfg.get("hard_stop_atr_mult", 2.0)
    stage1_mult = iskbets_cfg.get("stage1_atr_mult", 1.5)

    stop = price - (hard_stop_mult * atr)
    stage1 = price + (stage1_mult * atr)
    stop_pct = ((stop - price) / price) * 100
    stage1_pct = ((stage1 - price) / price) * 100

    short_ticker = ticker.replace("-USD", "")
    lines = [
        f"\U0001f7e1 *ISKBETS BUY {short_ticker}* @ ${price:,.2f}",
        "",
    ]

    # Signal votes grid
    if signals and ticker in signals:
        sig = signals[ticker]
        extra = sig.get("extra", {})
        votes = extra.get("_votes", {})
        buy_c = extra.get("_buy_count", 0)
        sell_c = extra.get("_sell_count", 0)
        hold_c = extra.get("_total_applicable", 11) - buy_c - sell_c

        buy_names = [k.upper() for k, v in votes.items() if v == "BUY"]
        sell_names = [k.upper() for k, v in votes.items() if v == "SELL"]

        lines.append(f"`Signals: {buy_c}B/{sell_c}S/{hold_c}H`")
        if buy_names:
            lines.append(f"`BUY:  {', '.join(buy_names)}`")
        if sell_names:
            lines.append(f"`SELL: {', '.join(sell_names)}`")

        # Key indicator values
        ind = sig.get("indicators", {})
        rsi = ind.get("rsi")
        macd = ind.get("macd_hist")
        bb = ind.get("price_vs_bb", "")
        if rsi is not None:
            lines.append(f"`RSI {rsi:.0f} | MACD {macd:+.0f} | BB {bb}`")
        lines.append("")

    # Why this entry triggered
    if conditions:
        lines.append("_Why:_")
        for c in conditions:
            lines.append(f"  \u2022 {c}")
        lines.append("")

    # Layer 2 reasoning (if gate was invoked)
    if l2_reasoning:
        lines.append(f"_Claude: {l2_reasoning}_")
        lines.append("")

    # Price levels — the key info
    lines.append(f"*If you buy:*")
    lines.append(f"`Stop loss:  ${stop:,.0f} ({stop_pct:+.1f}%)`")
    lines.append(f"`Target #1:  ${stage1:,.0f} ({stage1_pct:+.1f}%)`")
    lines.append(f"_After target #1, stop moves to breakeven._")
    lines.append(f"_Then trailing stop locks in profit._")
    lines.append("")
    lines.append(f"_Bought? Reply:_ `bought {short_ticker} PRICE AMOUNT`")
    lines.append(f"_Example:_ `bought {short_ticker} {price:.0f} 100000`")

    return "\n".join(lines)


def format_exit_alert(ticker, price, exit_type, entry_price, amount_sek, entry_time, fx_rate, exit_time=None):
    """Format Telegram exit alert message."""
    pnl_pct = ((price - entry_price) / entry_price) * 100
    shares = amount_sek / (entry_price * fx_rate)
    pnl_sek = shares * (price - entry_price) * fx_rate

    # Duration
    try:
        entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
        now = exit_time if exit_time else datetime.now(timezone.utc)
        duration = now - entry_dt
        hours = int(duration.total_seconds() // 3600)
        mins = int((duration.total_seconds() % 3600) // 60)
        dur_str = f"{hours}h {mins}min"
    except Exception:
        dur_str = "unknown"

    emoji_map = {
        "hard_stop": "\U0001f534",
        "trailing_stop": "\U0001f534",
        "time_exit": "\u23f0",
        "signal_reversal": "\U0001f7e0",
        "stage1_hit": "\U0001f7e2",
    }
    emoji = emoji_map.get(exit_type, "\U0001f534")

    type_labels = {
        "hard_stop": "Hard stop triggered",
        "trailing_stop": "Trailing stop triggered",
        "time_exit": "Time exit (market close)",
        "signal_reversal": "Signal reversal",
        "stage1_hit": "Stage 1 target hit",
    }
    label = type_labels.get(exit_type, exit_type)

    short_ticker = ticker.replace("-USD", "")

    if exit_type == "stage1_hit":
        lines = [
            f"{emoji} *{short_ticker} hit target #1* @ ${price:,.2f}",
            "",
            f"`Entry:  ${entry_price:,.2f}`",
            f"`Now:    ${price:,.2f} ({pnl_pct:+.1f}%)`",
            "",
            f"Your stop just moved to *breakeven* (${entry_price:,.2f}).",
            f"You can't lose money on this trade anymore.",
            f"Trailing stop will now lock in profit as price rises.",
            "",
            f"_No action needed — hold and let it run._",
        ]
    else:
        lines = [
            f"{emoji} *SELL {short_ticker}* @ ${price:,.2f}",
            "",
            f"`Entry:  ${entry_price:,.2f}`",
            f"`Exit:   ${price:,.2f}`",
            f"`P&L:    {pnl_sek:+,.0f} SEK ({pnl_pct:+.1f}%)`",
            f"`Held:   {dur_str}`",
            "",
            f"_{label}. Sell now on Avanza._",
            "",
            f"_Sold? Reply:_ `sold`",
        ]

    return "\n".join(lines)


def format_position_status(pos, price, fx_rate):
    """Format a status reply for the 'status' command."""
    ticker = pos["ticker"]
    entry_price = pos["entry_price_usd"]
    amount_sek = pos["amount_sek"]
    pnl_pct = ((price - entry_price) / entry_price) * 100
    shares = amount_sek / (entry_price * fx_rate)
    pnl_sek = shares * (price - entry_price) * fx_rate
    stop = pos.get("stop_loss", 0)
    stage1 = entry_price + (pos.get("atr_15m", 0) * 1.5)
    be = pos.get("stop_at_breakeven", False)

    try:
        entry_dt = datetime.fromisoformat(pos["entry_time"].replace("Z", "+00:00"))
        duration = datetime.now(timezone.utc) - entry_dt
        hours = int(duration.total_seconds() // 3600)
        mins = int((duration.total_seconds() % 3600) // 60)
        dur_str = f"{hours}h {mins}min"
    except Exception:
        dur_str = "unknown"

    lines = [
        f"\U0001f4ca *ISKBETS Status — {ticker}*",
        "",
        f"Entry: ${entry_price:,.2f} | Now: ${price:,.2f}",
        f"P&L: {pnl_sek:+,.0f} SEK ({pnl_pct:+.1f}%)",
        f"Stop: ${stop:,.2f} {'(breakeven)' if be else '(hard)'}",
        f"Stage 1: ${stage1:,.2f} {'(HIT)' if be else ''}",
        f"Highest: ${pos.get('highest_price', entry_price):,.2f}",
        f"Duration: {dur_str}",
    ]
    return "\n".join(lines)


# ── Main Entry Point ─────────────────────────────────────────────────────


def check_iskbets(signals, prices_usd, fx_rate, tf_data, config):
    """Main ISKBETS check — called every loop cycle.

    Scans for entries or monitors active position exits.
    """
    iskbets_cfg = config.get("iskbets", {})

    session_cfg = _load_config()
    if not session_cfg:
        return

    state = _load_state()
    changed = False

    if state.get("active_position"):
        # Monitor active position
        result = check_exits(state, prices_usd, signals, tf_data, iskbets_cfg)
        if result:
            exit_type, detail = result
            pos = state["active_position"]
            ticker = pos["ticker"]
            price = prices_usd.get(ticker, 0)

            msg = format_exit_alert(
                ticker, price, exit_type,
                pos["entry_price_usd"], pos["amount_sek"],
                pos["entry_time"], fx_rate,
            )
            _log_telegram(msg)
            try:
                _send_telegram(msg, config)
            except Exception as e:
                print(f"  ISKBETS: Telegram send failed: {e}")

            if exit_type != "stage1_hit":
                print(f"  ISKBETS EXIT: {exit_type} — {ticker} ${price:,.2f}")
            else:
                print(f"  ISKBETS: Stage 1 hit — {ticker}, stop moved to breakeven")

            changed = True
        else:
            # Just update state with highest_price etc.
            changed = True
    else:
        # Scan configured tickers for entry
        tickers = session_cfg.get("tickers", [])
        for ticker in tickers:
            if ticker not in signals or ticker not in prices_usd:
                continue
            price = prices_usd.get(ticker, 0)
            if price <= 0:
                continue

            should_enter, conditions = _evaluate_entry(
                ticker, signals, prices_usd, tf_data, iskbets_cfg, config
            )
            if should_enter:
                try:
                    atr = compute_atr_15m(ticker, config)
                except Exception as e:
                    print(f"  ISKBETS: ATR computation failed for {ticker}: {e}")
                    continue

                # Layer 2 gate — APPROVE/SKIP decision
                approved, l2_reasoning = invoke_layer2_gate(
                    ticker, price, conditions, signals, tf_data, atr, iskbets_cfg, config
                )
                if not approved:
                    print(f"  ISKBETS L2 GATE: SKIP {ticker} — {l2_reasoning}")
                    continue  # keep scanning other tickers

                msg = format_entry_alert(ticker, price, conditions, atr, iskbets_cfg, signals=signals, l2_reasoning=l2_reasoning)
                _log_telegram(msg)
                try:
                    _send_telegram(msg, config)
                except Exception as e:
                    print(f"  ISKBETS: Telegram send failed: {e}")

                print(f"  ISKBETS ENTRY ALERT: {ticker} ${price:,.2f} ({len(conditions)} conditions)")
                # Don't auto-set position — user must confirm via "bought" command
                break  # One alert at a time

    if changed:
        _save_state(state)


# ── Command Handler (called by TelegramPoller) ───────────────────────────


def handle_command(cmd, args, config):
    """Handle a Telegram command. Returns response text."""
    cmd = cmd.lower().strip()

    if cmd == "bought":
        return _handle_bought(args, config)
    elif cmd == "sold":
        return _handle_sold(args, config)
    elif cmd == "cancel":
        return _handle_cancel()
    elif cmd == "status":
        return _handle_status(config)
    else:
        return None  # Unknown command, ignore


def _handle_bought(args, config):
    """Handle 'bought TICKER PRICE AMOUNT' command."""
    parts = args.strip().split()
    if len(parts) < 3:
        return "Usage: `bought TICKER PRICE AMOUNT`"

    ticker = parts[0].upper()
    try:
        price_usd = float(parts[1])
        amount_sek = float(parts[2])
    except ValueError:
        return "Invalid price or amount. Usage: `bought TICKER PRICE AMOUNT`"

    if ticker not in TICKER_SOURCES:
        return f"Unknown ticker: {ticker}"

    # Load FX rate
    try:
        from portfolio.main import fetch_usd_sek
        fx_rate = fetch_usd_sek()
    except Exception:
        fx_rate = 10.5  # Fallback

    iskbets_cfg = config.get("iskbets", {})

    # Compute ATR
    try:
        atr = compute_atr_15m(ticker, config)
    except Exception as e:
        atr = price_usd * 0.02  # Fallback: 2% of price
        print(f"  ISKBETS: ATR fallback used: {e}")

    hard_stop_mult = iskbets_cfg.get("hard_stop_atr_mult", 2.0)
    stage1_mult = iskbets_cfg.get("stage1_atr_mult", 1.5)

    shares = amount_sek / (price_usd * fx_rate)
    stop = price_usd - (hard_stop_mult * atr)
    stage1 = price_usd + (stage1_mult * atr)

    state = _load_state()
    state["active_position"] = {
        "ticker": ticker,
        "entry_price_usd": price_usd,
        "amount_sek": amount_sek,
        "shares": round(shares, 6),
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "atr_15m": round(atr, 4),
        "stop_loss": round(stop, 2),
        "stage1_target": round(stage1, 2),
        "stop_at_breakeven": False,
        "highest_price": price_usd,
        "sell_signal_streak": 0,
        "fx_rate": fx_rate,
    }
    _save_state(state)

    stop_pct = ((stop - price_usd) / price_usd) * 100
    stage1_pct = ((stage1 - price_usd) / price_usd) * 100

    return (
        f"\u2705 Position tracked.\n"
        f"Stop: ${stop:,.2f} ({stop_pct:+.1f}%) | Stage 1: ${stage1:,.2f} ({stage1_pct:+.1f}%)\n"
        f"Trailing thereafter | Hard close: {iskbets_cfg.get('time_exit_et', '15:50')} ET"
    )


def _handle_sold(args, config):
    """Handle 'sold' command — close position and log P&L."""
    state = _load_state()
    pos = state.get("active_position")
    if not pos:
        return "No active position to close."

    ticker = pos["ticker"]
    entry_price = pos["entry_price_usd"]
    amount_sek = pos["amount_sek"]
    fx_rate = pos.get("fx_rate", 10.5)

    # Try to get current price
    current_price = None
    parts = args.strip().split() if args else []
    if parts:
        try:
            current_price = float(parts[0])
        except ValueError:
            pass

    if current_price is None:
        current_price = pos.get("highest_price", entry_price)

    shares = amount_sek / (entry_price * fx_rate)
    pnl_sek = shares * (current_price - entry_price) * fx_rate
    pnl_pct = ((current_price - entry_price) / entry_price) * 100

    try:
        entry_dt = datetime.fromisoformat(pos["entry_time"].replace("Z", "+00:00"))
        duration = datetime.now(timezone.utc) - entry_dt
        hours = int(duration.total_seconds() // 3600)
        mins = int((duration.total_seconds() % 3600) // 60)
        dur_str = f"{hours}h {mins}min"
    except Exception:
        dur_str = "unknown"

    # Log to trade history
    trade = {
        "ticker": ticker,
        "entry_price_usd": entry_price,
        "exit_price_usd": current_price,
        "amount_sek": amount_sek,
        "pnl_sek": round(pnl_sek, 2),
        "pnl_pct": round(pnl_pct, 2),
        "entry_time": pos["entry_time"],
        "exit_time": datetime.now(timezone.utc).isoformat(),
        "duration": dur_str,
    }
    state["trade_history"].append(trade)
    state["active_position"] = None
    _save_state(state)

    return f"\U0001f4ca ISKBETS closed. {ticker} {pnl_sek:+,.0f} SEK ({pnl_pct:+.1f}%) in {dur_str}"


def _handle_cancel():
    """Handle 'cancel' command — disable ISKBETS mode."""
    cfg = _load_config()
    if cfg:
        cfg["enabled"] = False
        _save_config(cfg)

    state = _load_state()
    state["active_position"] = None
    _save_state(state)

    return "\u274c ISKBETS mode disabled."


def _handle_status(config):
    """Handle 'status' command — send position summary."""
    state = _load_state()
    pos = state.get("active_position")
    if not pos:
        # Check if session is active
        session_cfg = _load_config()
        if session_cfg:
            tickers = session_cfg.get("tickers", [])
            return f"\U0001f50d ISKBETS scanning: {', '.join(tickers)}. No active position."
        return "ISKBETS is not active."

    ticker = pos["ticker"]

    # Try to get current price
    try:
        from portfolio.main import fetch_usd_sek
        fx_rate = fetch_usd_sek()
    except Exception:
        fx_rate = pos.get("fx_rate", 10.5)

    # Get live price
    price = _get_current_price(ticker, config)
    if price is None:
        price = pos.get("highest_price", pos["entry_price_usd"])

    return format_position_status(pos, price, fx_rate)


def _get_current_price(ticker, config):
    """Fetch current price for a ticker."""
    source = TICKER_SOURCES.get(ticker)
    if not source:
        return None
    try:
        if "binance" in source:
            r = requests.get(
                f"{BINANCE_BASE}/ticker/price",
                params={"symbol": source["binance"]},
                timeout=5,
            )
            r.raise_for_status()
            return float(r.json()["price"])
        else:
            # Alpaca snapshot
            acfg = config.get("alpaca", {})
            headers = {
                "APCA-API-KEY-ID": acfg.get("key", ""),
                "APCA-API-SECRET-KEY": acfg.get("secret", ""),
            }
            r = requests.get(
                f"{ALPACA_BASE}/stocks/{source['alpaca']}/snapshot",
                headers=headers,
                params={"feed": "iex"},
                timeout=5,
            )
            r.raise_for_status()
            return float(r.json()["latestTrade"]["p"])
    except Exception:
        return None
