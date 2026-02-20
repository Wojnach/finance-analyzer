"""Manual instrument analysis and position watchdog.

Usage:
  python main.py --analyze ETH-USD
  python main.py --watch BTC-USD:66500 ETH-USD:1920 MSTR:125
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent


def _clean_env():
    """Return env dict without CLAUDECODE to avoid nested-session errors."""
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    return env
DATA_DIR = BASE_DIR / "data"
AGENT_SUMMARY_FILE = DATA_DIR / "agent_summary.json"
JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
PORTFOLIO_FILE = DATA_DIR / "portfolio_state.json"
BOLD_FILE = DATA_DIR / "portfolio_state_bold.json"
CONFIG_FILE = BASE_DIR / "config.json"
ANALYSIS_LOG_FILE = DATA_DIR / "analysis_log.jsonl"
WATCH_LOG_FILE = DATA_DIR / "watch_log.jsonl"

from portfolio.tickers import (
    ALL_TICKERS as KNOWN_TICKERS,
    CRYPTO_SYMBOLS as CRYPTO_TICKERS,
    METALS_SYMBOLS as METALS_TICKERS,
)

# Position watch exit thresholds
STOP_PCT = -2.0       # hard stop
TARGET_PCT = 2.0      # take profit
WARN_PCT = -1.5       # early warning
TIME_WARN_MINS = 180  # 3h — consider exit
TIME_MAX_MINS = 300   # 5h — hard time exit


def _load_journal_for_ticker(ticker, max_entries=5):
    """Load last N journal entries that mention this ticker with a non-neutral outlook."""
    if not JOURNAL_FILE.exists():
        return []
    entries = []
    for line in JOURNAL_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            tickers = entry.get("tickers", {})
            info = tickers.get(ticker, {})
            if info.get("outlook", "neutral") != "neutral":
                entries.append(entry)
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return entries[-max_entries:]


def _get_holdings(ticker):
    """Get holdings for this ticker from both portfolio strategies."""
    holdings = {}
    for label, filepath in [("patient", PORTFOLIO_FILE), ("bold", BOLD_FILE)]:
        if not filepath.exists():
            continue
        try:
            pf = json.loads(filepath.read_text(encoding="utf-8"))
            h = pf.get("holdings", {}).get(ticker, {})
            if h.get("shares", 0) > 0:
                holdings[label] = h
        except Exception:
            continue
    return holdings


def _build_analysis_prompt(ticker, summary):
    """Build the rich analysis prompt for Claude."""
    sig = summary.get("signals", {}).get(ticker)
    if not sig:
        return None

    extra = sig.get("extra", {})
    votes = extra.get("_votes", {})
    price = sig["price_usd"]
    regime = sig.get("regime", "unknown")
    rsi = sig["rsi"]
    macd = sig["macd_hist"]
    bb = sig["bb_position"]
    atr_pct = sig.get("atr_pct", 0)
    vol_raw = extra.get("volume_ratio")
    vol = f"{vol_raw:.2f}" if isinstance(vol_raw, (int, float)) else "N/A"
    buy_c = extra.get("_buy_count", 0)
    sell_c = extra.get("_sell_count", 0)
    total = extra.get("_total_applicable", 21)
    hold_c = total - buy_c - sell_c
    w_conf = sig.get("weighted_confidence", 0)
    confluence = sig.get("confluence_score", 0)

    # Signal accuracy
    acc_data = summary.get("signal_accuracy_1d", {}).get("signals", {})

    def _vote_str(name):
        vote = votes.get(name, "HOLD")
        acc = acc_data.get(name, {})
        acc_pct = f"{acc['accuracy']*100:.0f}%" if acc else "N/A"
        return f"{vote} ({acc_pct})"

    lines = [
        f"You are analyzing {ticker} for a trader who wants your honest assessment.",
        "",
        "CURRENT DATA:",
        f"Price: ${price:,.2f} | Regime: {regime}",
        f"Signals: {buy_c}B/{sell_c}S/{hold_c}H (weighted conf: {w_conf:.0%})",
        f"RSI {rsi:.1f} | MACD {macd:+.2f} | BB {bb} | ATR {atr_pct:.2f}% | Vol {vol}x",
        f"Confluence: {confluence:.0%}",
        "",
        "Per-signal votes (accuracy in parens):",
        f"  RSI: {_vote_str('rsi')} | MACD: {_vote_str('macd')} | EMA: {_vote_str('ema')}",
        f"  BB: {_vote_str('bb')} | F&G: {_vote_str('fear_greed')} | Sentiment: {_vote_str('sentiment')}",
        f"  Volume: {_vote_str('volume')}",
    ]

    if ticker in CRYPTO_TICKERS:
        lines.append(f"  ML: {_vote_str('ml')} | Funding: {_vote_str('funding')}")
        m_reason = extra.get("ministral_reasoning", "")
        l_reason = extra.get("custom_lora_reasoning", "")
        lines.append(f"  Ministral: {votes.get('ministral', 'HOLD')} — \"{m_reason}\"")
        lines.append(f"  LoRA: {votes.get('custom_lora', 'HOLD')} — \"{l_reason}\"")

    # Timeframes
    tf_list = summary.get("timeframes", {}).get(ticker, [])
    if tf_list:
        lines.append("")
        lines.append("Timeframes:")
        header = "       "
        row = f"  {ticker[:5]:<5} "
        detail_lines = []
        for tf in tf_list:
            h = tf["horizon"]
            a = tf.get("action", "HOLD")
            header += f"{h:>4} "
            tag = "B" if a == "BUY" else "S" if a == "SELL" else "H"
            row += f"  {tag}  "
            detail_lines.append(
                f"  {h}: {a} | RSI {tf.get('rsi', 'N/A')} | MACD {tf.get('macd_hist', 'N/A')}"
            )
        lines.append(header)
        lines.append(row)
        lines.extend(detail_lines)

    # Macro
    macro = summary.get("macro", {})
    dxy = macro.get("dxy", {})
    treasury = macro.get("treasury", {})
    fed = macro.get("fed", {})
    fg_data = summary.get("fear_greed", {}).get(ticker, {})

    lines.append("")
    dxy_val = dxy.get("value", "N/A")
    dxy_trend = dxy.get("trend", "")
    yield_10y = treasury.get("10y", {}).get("yield_pct", "N/A")
    spread = treasury.get("spread_2s10s", "N/A")
    fomc_days = fed.get("days_until", "N/A")
    lines.append(f"Macro: DXY {dxy_val} ({dxy_trend}) | 10Y {yield_10y}% | 2s10s {spread} | FOMC {fomc_days}d")

    fg_val = fg_data.get("value", "N/A")
    fg_class = fg_data.get("classification", "")
    lines.append(f"F&G: {fg_val} ({fg_class})")

    # Journal history
    journal_entries = _load_journal_for_ticker(ticker)
    if journal_entries:
        lines.append("")
        lines.append("Recent history:")
        for entry in journal_entries:
            ts = entry["ts"]
            info = entry.get("tickers", {}).get(ticker, {})
            outlook = info.get("outlook", "neutral")
            thesis = info.get("thesis", "")
            old_price = entry.get("prices", {}).get(ticker)
            if old_price and old_price > 0:
                pct = ((price - old_price) / old_price) * 100
                lines.append(
                    f"- {ts}: outlook={outlook}, thesis=\"{thesis}\", "
                    f"price was ${old_price:,.2f} (now ${price:,.2f}, {pct:+.1f}%)"
                )
            else:
                lines.append(f"- {ts}: outlook={outlook}, thesis=\"{thesis}\"")

    # Holdings
    holdings = _get_holdings(ticker)
    if holdings:
        lines.append("")
        parts = []
        for strat, h in holdings.items():
            shares = h.get("shares", 0)
            avg_cost = h.get("avg_cost_usd", 0)
            parts.append(f"{strat.title()} holds {shares:.4f} @ ${avg_cost:,.2f}")
        lines.append(f"Portfolio exposure: {' | '.join(parts)}")

    lines.extend([
        "",
        "Respond with:",
        "OUTLOOK: BULLISH / BEARISH / NEUTRAL",
        "CONVICTION: X/10",
        "SUMMARY: 3-5 sentences. Cover signal alignment, TF structure, macro context, regime.",
        "KEY LEVELS: Support $X, Resistance $Y",
        "RISK FACTORS: 1-2 key risks",
        "TRADE IDEA: If actionable, what setup? If not, what would change your mind?",
        "SHORT-TERM SCALP: Rate a 3-5h leveraged trade: SKIP / MARGINAL / GOOD / EXCELLENT.",
        "  If not SKIP, specify direction (LONG/SHORT), entry timing, and what invalidates it.",
    ])

    return "\n".join(lines)


def _log_analysis(ticker, output, elapsed):
    """Append to analysis_log.jsonl."""
    try:
        ANALYSIS_LOG_FILE.parent.mkdir(exist_ok=True)
        with open(ANALYSIS_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "ticker": ticker,
                        "elapsed_s": round(elapsed, 2),
                        "output": output[:2000],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:
        pass


def _send_telegram(msg, config):
    if os.environ.get("NO_TELEGRAM"):
        print("[NO_TELEGRAM] Skipping send")
        return
    token = config["telegram"]["token"]
    chat_id = config["telegram"]["chat_id"]
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
        timeout=30,
    )
    if not r.ok:
        print(f"Telegram error: {r.status_code} {r.text[:200]}")


def run_analysis(ticker):
    """Run a deep analysis for a single instrument."""
    ticker = ticker.upper()

    # Normalize short names
    short_map = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
    ticker = short_map.get(ticker, ticker)

    if ticker not in KNOWN_TICKERS:
        print(f"Unknown ticker: {ticker}")
        print(f"Valid tickers: {', '.join(sorted(KNOWN_TICKERS))}")
        return

    if not AGENT_SUMMARY_FILE.exists():
        print("No agent_summary.json found. Run --report first to generate signal data.")
        return

    summary = json.loads(AGENT_SUMMARY_FILE.read_text(encoding="utf-8"))

    if ticker not in summary.get("signals", {}):
        print(f"No signal data for {ticker} in agent_summary.json. Run --report first.")
        return

    prompt = _build_analysis_prompt(ticker, summary)
    if not prompt:
        print(f"Failed to build analysis prompt for {ticker}.")
        return

    print(f"Analyzing {ticker}...\n")

    t0 = time.time()
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--max-turns", "1"],
            capture_output=True,
            text=True,
            timeout=120,
            env=_clean_env(),
            stdin=subprocess.DEVNULL,
        )
        elapsed = time.time() - t0
        output = result.stdout.strip()

        if result.returncode != 0 or not output:
            print(f"Claude returned code {result.returncode}")
            if result.stderr:
                print(f"stderr: {result.stderr[:500]}")
            return

        print(output)
        print(f"\n({elapsed:.1f}s)")

        _log_analysis(ticker, output, elapsed)

        # Send to Telegram
        try:
            config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            price = summary["signals"][ticker]["price_usd"]
            tg_msg = f"*ANALYSIS: {ticker}* (${price:,.2f})\n\n{output}"
            _send_telegram(tg_msg, config)
            print("Sent to Telegram.")
        except Exception as e:
            print(f"Telegram failed: {e}")

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"Claude timed out after {elapsed:.0f}s")
    except FileNotFoundError:
        print("claude not found in PATH")
    except Exception as e:
        print(f"Error: {e}")


# ---------------------------------------------------------------------------
# Position watchdog (--watch) — Claude-powered scalp exit monitor
# ---------------------------------------------------------------------------

SHORT_MAP = {"BTC": "BTC-USD", "ETH": "ETH-USD", "XAU": "XAU-USD", "XAG": "XAG-USD"}

# Claude analysis triggers
CLAUDE_INTERVAL_MINS = 15       # max time between Claude calls
CLAUDE_PRICE_THRESHOLD = 0.5    # % price move triggers re-analysis


def _normalize_ticker(t):
    t = t.upper()
    return SHORT_MAP.get(t, t)


def _parse_positions(args):
    """Parse 'TICKER:ENTRY_PRICE' pairs from CLI args."""
    positions = {}
    for arg in args:
        if ":" not in arg:
            print(f"Bad format: {arg} — expected TICKER:ENTRY_PRICE")
            continue
        ticker_raw, price_raw = arg.split(":", 1)
        ticker = _normalize_ticker(ticker_raw)
        if ticker not in KNOWN_TICKERS:
            print(f"Unknown ticker: {ticker}")
            continue
        try:
            entry_price = float(price_raw)
        except ValueError:
            print(f"Bad price: {price_raw}")
            continue
        positions[ticker] = entry_price
    return positions


def _load_summary():
    """Load the full agent_summary.json."""
    if not AGENT_SUMMARY_FILE.exists():
        return None
    try:
        return json.loads(AGENT_SUMMARY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def _get_signal_state(ticker, summary=None):
    """Get current signal state from agent_summary.json."""
    if summary is None:
        summary = _load_summary()
    if not summary:
        return None
    sig = summary.get("signals", {}).get(ticker)
    if not sig:
        return None
    extra = sig.get("extra", {})
    tfs = summary.get("timeframes", {}).get(ticker, [])
    sell_tfs = sum(1 for tf in tfs if tf.get("action") == "SELL")
    buy_tfs = sum(1 for tf in tfs if tf.get("action") == "BUY")
    return {
        "action": sig["action"],
        "price": sig["price_usd"],
        "buy_count": extra.get("_buy_count", 0),
        "sell_count": extra.get("_sell_count", 0),
        "sell_tfs": sell_tfs,
        "buy_tfs": buy_tfs,
        "rsi": sig.get("rsi", 50),
        "macd": sig.get("macd_hist", 0),
    }


def _log_watch(event):
    """Append watch event to log."""
    try:
        WATCH_LOG_FILE.parent.mkdir(exist_ok=True)
        with open(WATCH_LOG_FILE, "a", encoding="utf-8") as f:
            event["ts"] = datetime.now(timezone.utc).isoformat()
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _build_watch_prompt(positions, summary, elapsed_mins):
    """Build a Claude prompt focused on scalp exit timing.

    Args:
        positions: dict of {ticker: entry_price}
        summary: full agent_summary.json dict
        elapsed_mins: minutes since positions were opened
    """
    lines = [
        "You are monitoring open leveraged positions for a short-term scalp trader.",
        "The trader uses 5x leveraged instruments. A 2% move in the underlying = 10% on the position.",
        f"Positions have been open for {elapsed_mins:.0f} minutes. Max hold time is 5 hours.",
        "Your job: for each position, decide HOLD or SELL based on all available signals.",
        "",
        "OPEN POSITIONS:",
    ]

    acc_data = summary.get("signal_accuracy_1d", {}).get("signals", {})

    for ticker, entry_price in positions.items():
        sig = summary.get("signals", {}).get(ticker)
        if not sig:
            lines.append(f"\n{ticker}: NO SIGNAL DATA")
            continue

        extra = sig.get("extra", {})
        votes = extra.get("_votes", {})
        price = sig["price_usd"]
        pct = ((price - entry_price) / entry_price) * 100
        regime = sig.get("regime", "unknown")
        rsi = sig["rsi"]
        macd = sig["macd_hist"]
        bb = sig["bb_position"]
        atr_pct = sig.get("atr_pct", 0)
        vol_raw = extra.get("volume_ratio")
        vol = f"{vol_raw:.2f}" if isinstance(vol_raw, (int, float)) else "N/A"
        buy_c = extra.get("_buy_count", 0)
        sell_c = extra.get("_sell_count", 0)
        total = extra.get("_total_applicable", 21)
        hold_c = total - buy_c - sell_c
        w_conf = sig.get("weighted_confidence", 0)

        def _vote_str(name):
            vote = votes.get(name, "HOLD")
            acc = acc_data.get(name, {})
            acc_pct = f"{acc['accuracy']*100:.0f}%" if acc else "N/A"
            return f"{vote}({acc_pct})"

        lines.extend([
            f"\n--- {ticker} ---",
            f"Entry: ${entry_price:,.2f} | Now: ${price:,.2f} | P&L: {pct:+.2f}% (={pct*5:+.1f}% on 5x)",
            f"Regime: {regime} | RSI {rsi:.1f} | MACD {macd:+.2f} | BB {bb} | ATR {atr_pct:.2f}% | Vol {vol}x",
            f"Signals: {buy_c}B/{sell_c}S/{hold_c}H (wconf {w_conf:.0%})",
            f"  RSI:{_vote_str('rsi')} MACD:{_vote_str('macd')} EMA:{_vote_str('ema')} "
            f"BB:{_vote_str('bb')} F&G:{_vote_str('fear_greed')} Sent:{_vote_str('sentiment')} Vol:{_vote_str('volume')}",
        ])

        if ticker in CRYPTO_TICKERS:
            m_reason = extra.get("ministral_reasoning", "")[:80]
            l_reason = extra.get("custom_lora_reasoning", "")[:80]
            lines.append(
                f"  ML:{_vote_str('ml')} Fund:{_vote_str('funding')} "
                f"Ministral:{votes.get('ministral','HOLD')} LoRA:{votes.get('custom_lora','HOLD')}"
            )
            if m_reason:
                lines.append(f"  Ministral says: \"{m_reason}\"")
            if l_reason:
                lines.append(f"  LoRA says: \"{l_reason}\"")

        # Timeframes
        tfs = summary.get("timeframes", {}).get(ticker, [])
        if tfs:
            tf_tags = " ".join(
                f"{tf['horizon']}={'B' if tf.get('action')=='BUY' else 'S' if tf.get('action')=='SELL' else 'H'}"
                for tf in tfs
            )
            sell_tfs = sum(1 for tf in tfs if tf.get("action") == "SELL")
            buy_tfs = sum(1 for tf in tfs if tf.get("action") == "BUY")
            lines.append(f"  TFs: {tf_tags}  ({sell_tfs}S/{buy_tfs}B)")

    # Macro context
    macro = summary.get("macro", {})
    dxy = macro.get("dxy", {})
    treasury = macro.get("treasury", {})
    fed = macro.get("fed", {})
    lines.extend([
        "",
        f"Macro: DXY {dxy.get('value','N/A')} ({dxy.get('trend','')}) | "
        f"10Y {treasury.get('10y',{}).get('yield_pct','N/A')}% | "
        f"2s10s {treasury.get('spread_2s10s','N/A')} | "
        f"FOMC {fed.get('days_until','N/A')}d",
    ])

    # F&G for all watched tickers
    fg = summary.get("fear_greed", {})
    fg_parts = []
    for ticker in positions:
        val = fg.get(ticker, {}).get("value", "?")
        fg_parts.append(f"{ticker}: {val}")
    lines.append(f"F&G: {' | '.join(fg_parts)}")

    # Instructions
    lines.extend([
        "",
        "RULES:",
        "- This is a short-term scalp. 3-5 hour max hold, never overnight.",
        "- Take profit target: +2% underlying (+10% on 5x leverage).",
        "- Hard stop: -2% underlying (-10% on 5x leverage).",
        "- If signals are deteriorating and P&L is negative, recommend exit early.",
        "- If signals are improving, it may be worth holding even if slightly negative.",
        "- Weigh higher-accuracy signals more heavily (accuracy % shown in parens).",
        "- Consider regime + timeframe alignment. ALL TFs SELL = get out.",
        "",
        "For EACH position, respond with exactly this format:",
        "TICKER: HOLD or SELL",
        "REASON: 1-2 sentence explanation",
        "",
        "Then add:",
        "OVERALL: Brief 1-sentence market read",
    ])

    return "\n".join(lines)


def _parse_watch_response(output, tickers):
    """Parse Claude's watch response into per-ticker decisions.

    Returns dict of {ticker: {"action": "HOLD"|"SELL", "reason": str}}
    """
    decisions = {}
    lines = output.strip().splitlines()

    for ticker in tickers:
        # Look for "TICKER: HOLD" or "TICKER: SELL" pattern
        # Strip markdown bold/heading markers (Claude often wraps in **bold**)
        short = ticker.replace("-USD", "")
        for i, line in enumerate(lines):
            stripped = line.strip().lstrip("*#- ").rstrip("*")
            stripped_upper = stripped.upper()
            if (stripped_upper.startswith(f"{ticker}:") or
                    stripped_upper.startswith(f"{short}:")):
                action = "SELL" if "SELL" in stripped_upper else "HOLD"
                reason = ""
                # Look for REASON line after
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip().lstrip("*#- ").rstrip("*")
                    if next_line.upper().startswith("REASON:"):
                        reason = next_line[7:].strip()
                decisions[ticker] = {"action": action, "reason": reason}
                break

    # Extract overall market read
    overall = ""
    for line in lines:
        stripped = line.strip().lstrip("*#- ").rstrip("*")
        if stripped.upper().startswith("OVERALL:"):
            overall = stripped[8:].strip()
            break

    return decisions, overall


def _should_call_claude(last_claude_time, last_claude_prices, current_prices,
                        last_claude_actions, current_actions):
    """Determine if we should invoke Claude for analysis.

    Returns (should_call, reason) tuple.
    """
    now = time.time()

    # First call
    if last_claude_time == 0:
        return True, "initial"

    # Time-based: every 15 minutes
    mins_since = (now - last_claude_time) / 60
    if mins_since >= CLAUDE_INTERVAL_MINS:
        return True, f"periodic ({mins_since:.0f}min since last)"

    # Price-based: >0.5% move since last Claude call
    for ticker, price in current_prices.items():
        old = last_claude_prices.get(ticker)
        if old and old > 0:
            pct_move = abs((price - old) / old) * 100
            if pct_move >= CLAUDE_PRICE_THRESHOLD:
                return True, f"{ticker} moved {pct_move:.1f}% since last analysis"

    # Signal flip: consensus action changed
    for ticker, action in current_actions.items():
        old_action = last_claude_actions.get(ticker)
        if old_action and action != old_action:
            return True, f"{ticker} signal flipped {old_action} -> {action}"

    return False, ""


def watch_positions(position_args, interval=60):
    """Claude-powered position watchdog for short-term scalp trades.

    Monitors positions with mechanical guardrails (hard stop, target, time limit)
    and periodically invokes Claude to interpret the full signal picture.

    Args:
        position_args: list of "TICKER:ENTRY_PRICE" strings
        interval: seconds between checks (default 60)
    """
    positions = _parse_positions(position_args)
    if not positions:
        print("No valid positions to watch.")
        return

    try:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        config = None
        print("WARNING: config.json not found — Telegram alerts disabled")

    start_time = time.time()
    resolved = set()                   # tickers that hit hard exits
    last_claude_time = 0               # epoch of last Claude call
    last_claude_prices = {}            # prices at last Claude call
    last_claude_actions = {}           # signal actions at last Claude call
    claude_call_count = 0

    print(f"Watching {len(positions)} position(s) with Claude analysis. Ctrl+C to stop.")
    for ticker, entry in positions.items():
        print(f"  {ticker}: entry ${entry:,.2f}")
    print()

    # Send initial Telegram notification
    if config:
        pos_lines = "\n".join(
            f"  {t}: entry ${p:,.2f}" for t, p in positions.items()
        )
        _send_telegram(
            f"*WATCH STARTED*\n\nMonitoring {len(positions)} position(s):\n{pos_lines}\n\n"
            f"_Claude-powered analysis every {CLAUDE_INTERVAL_MINS}min or on signal changes_",
            config,
        )

    try:
        while True:
            now_dt = datetime.now(timezone.utc)
            elapsed_mins = (time.time() - start_time) / 60
            ts_str = now_dt.strftime("%H:%M:%S UTC")

            summary = _load_summary()
            if not summary:
                print(f"  [{ts_str}] No agent_summary.json — waiting...")
                time.sleep(interval)
                continue

            # --- Phase 1: Mechanical guardrails (fire immediately) ---
            current_prices = {}
            current_actions = {}

            for ticker, entry_price in positions.items():
                if ticker in resolved:
                    continue

                state = _get_signal_state(ticker, summary)
                if not state:
                    print(f"  [{ts_str}] {ticker}: no signal data")
                    continue

                price = state["price"]
                current_prices[ticker] = price
                current_actions[ticker] = state["action"]
                pct = ((price - entry_price) / entry_price) * 100
                lev_pct = pct * 5

                # Hard stop — immediate exit
                if pct <= STOP_PCT:
                    msg = (
                        f"*STOP HIT: {ticker}*\n\n"
                        f"P&L: {pct:+.2f}% ({lev_pct:+.1f}% on 5x)\n"
                        f"Price: ${price:,.2f} (entry ${entry_price:,.2f})\n"
                        f"_SELL NOW — hard stop breached_"
                    )
                    print(f"  ** {ticker} STOP HIT {pct:+.2f}% **")
                    if config:
                        _send_telegram(msg, config)
                    _log_watch({"ticker": ticker, "type": "stop", "price": price,
                                "entry": entry_price, "pct": round(pct, 2)})
                    resolved.add(ticker)
                    continue

                # Take profit — immediate exit
                if pct >= TARGET_PCT:
                    msg = (
                        f"*TARGET HIT: {ticker}*\n\n"
                        f"P&L: {pct:+.2f}% ({lev_pct:+.1f}% on 5x)\n"
                        f"Price: ${price:,.2f} (entry ${entry_price:,.2f})\n"
                        f"_Take profit — target reached_"
                    )
                    print(f"  ** {ticker} TARGET HIT {pct:+.2f}% **")
                    if config:
                        _send_telegram(msg, config)
                    _log_watch({"ticker": ticker, "type": "target", "price": price,
                                "entry": entry_price, "pct": round(pct, 2)})
                    resolved.add(ticker)
                    continue

                # Time limit — immediate exit
                if elapsed_mins >= TIME_MAX_MINS:
                    msg = (
                        f"*TIME LIMIT: {ticker}*\n\n"
                        f"P&L: {pct:+.2f}% ({lev_pct:+.1f}% on 5x)\n"
                        f"Price: ${price:,.2f} (entry ${entry_price:,.2f})\n"
                        f"_5h max hold time exceeded — close position_"
                    )
                    print(f"  ** {ticker} TIME LIMIT — close position **")
                    if config:
                        _send_telegram(msg, config)
                    _log_watch({"ticker": ticker, "type": "time_max", "price": price,
                                "entry": entry_price, "pct": round(pct, 2)})
                    resolved.add(ticker)
                    continue

            # Check if all done
            active = [t for t in positions if t not in resolved]
            if not active:
                print("\nAll positions resolved. Stopping watch.")
                break

            # --- Phase 2: Claude-powered analysis ---
            should_call, reason = _should_call_claude(
                last_claude_time, last_claude_prices, current_prices,
                last_claude_actions, current_actions,
            )

            if should_call:
                # Only analyze active (non-resolved) positions
                active_positions = {t: positions[t] for t in active}
                prompt = _build_watch_prompt(active_positions, summary, elapsed_mins)
                claude_call_count += 1

                print(f"\n  [{ts_str}] Calling Claude (#{claude_call_count}, reason: {reason})...")
                t0 = time.time()

                try:
                    result = subprocess.run(
                        ["claude", "-p", prompt, "--max-turns", "1"],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        env=_clean_env(),
                        stdin=subprocess.DEVNULL,
                    )
                    c_elapsed = time.time() - t0
                    output = result.stdout.strip()

                    if result.returncode == 0 and output:
                        decisions, overall = _parse_watch_response(output, active)
                        print(f"  Claude ({c_elapsed:.1f}s): {overall}")

                        for ticker in active:
                            dec = decisions.get(ticker, {})
                            dec_action = dec.get("action", "HOLD")
                            dec_reason = dec.get("reason", "")
                            entry_price = positions[ticker]
                            price = current_prices.get(ticker, 0)
                            pct = ((price - entry_price) / entry_price) * 100 if price else 0
                            lev_pct = pct * 5

                            state = _get_signal_state(ticker, summary)
                            sell_tfs = state["sell_tfs"] if state else "?"
                            buy_tfs = state["buy_tfs"] if state else "?"
                            action = state["action"] if state else "?"

                            if dec_action == "SELL":
                                print(f"  ** Claude: SELL {ticker} — {dec_reason} **")
                                if config:
                                    tg_msg = (
                                        f"*WATCH SELL: {ticker}*\n\n"
                                        f"Claude recommends EXIT\n"
                                        f"P&L: {pct:+.2f}% ({lev_pct:+.1f}% on 5x)\n"
                                        f"Price: ${price:,.2f} (entry ${entry_price:,.2f})\n"
                                        f"Signals: {action} | TFs: {sell_tfs}S/{buy_tfs}B\n"
                                        f"_Reason: {dec_reason}_\n"
                                        f"_Hold time: {elapsed_mins:.0f}min_"
                                    )
                                    _send_telegram(tg_msg, config)
                                _log_watch({
                                    "ticker": ticker, "type": "claude_sell",
                                    "price": price, "entry": entry_price,
                                    "pct": round(pct, 2), "reason": dec_reason,
                                })
                            else:
                                print(f"  Claude: HOLD {ticker} — {dec_reason}")

                        _log_watch({
                            "type": "claude_analysis", "call_num": claude_call_count,
                            "reason": reason, "elapsed_s": round(c_elapsed, 2),
                            "decisions": {t: decisions.get(t, {}) for t in active},
                            "overall": overall,
                        })
                    else:
                        print(f"  Claude returned code {result.returncode} ({c_elapsed:.1f}s)")
                        if result.stderr:
                            print(f"  stderr: {result.stderr[:200]}")

                except subprocess.TimeoutExpired:
                    print(f"  Claude timed out after 60s")
                except FileNotFoundError:
                    print("  claude not found in PATH — falling back to mechanical only")
                except Exception as e:
                    print(f"  Claude error: {e}")

                last_claude_time = time.time()
                last_claude_prices = dict(current_prices)
                last_claude_actions = dict(current_actions)

            else:
                # Status line (no Claude call this tick)
                for ticker in active:
                    price = current_prices.get(ticker, 0)
                    entry_price = positions[ticker]
                    pct = ((price - entry_price) / entry_price) * 100 if price else 0
                    state = _get_signal_state(ticker, summary)
                    action = state["action"] if state else "?"
                    sell_tfs = state["sell_tfs"] if state else "?"
                    buy_tfs = state["buy_tfs"] if state else "?"
                    print(f"  [{ts_str}] {ticker} ${price:,.2f} ({pct:+.2f}%) "
                          f"| {action} | {sell_tfs}S/{buy_tfs}B TFs")

            time.sleep(interval)

    except KeyboardInterrupt:
        elapsed_mins = (time.time() - start_time) / 60
        print(f"\nWatch stopped after {elapsed_mins:.0f}min, {claude_call_count} Claude calls.")
        if config:
            # Final status
            status_parts = []
            for ticker, entry_price in positions.items():
                price = current_prices.get(ticker, 0)
                if price:
                    pct = ((price - entry_price) / entry_price) * 100
                    status_parts.append(f"{ticker}: ${price:,.2f} ({pct:+.2f}%)")
            _send_telegram(
                f"*WATCH STOPPED*\n\n"
                f"Duration: {elapsed_mins:.0f}min | Claude calls: {claude_call_count}\n"
                + "\n".join(status_parts),
                config,
            )
