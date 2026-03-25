"""Scheduled crypto analysis — periodic BTC/ETH/MSTR reports to Telegram.

Runs at configured hours (default 08:00, 13:00, 18:00 CET) as part of the
main loop post-cycle. Generates a structured report from signal data and
crypto_macro data (Deribit options, gold-BTC ratio, exchange netflow),
then sends to Telegram.

No Claude usage — pure Python analysis. Use /fin-crypto manually for
deep adversarial analysis when needed.

Pattern: follows digest.py (state file + sentinel + time check).

Usage (in main.py _run_post_cycle):
    from portfolio.crypto_scheduler import maybe_run_crypto_report
    maybe_run_crypto_report(config)
"""

import logging
import time
import zoneinfo
from datetime import datetime
from pathlib import Path

from portfolio.file_utils import atomic_write_json, load_json, atomic_append_jsonl
from portfolio.message_store import send_or_store

logger = logging.getLogger("portfolio.crypto_scheduler")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

STATE_FILE = DATA_DIR / "crypto_scheduler_state.json"
LOG_FILE = DATA_DIR / "fin_command_log.jsonl"

# Default schedule: 08:00, 13:00, 18:00 CET (Europe/Stockholm)
DEFAULT_HOURS = [8, 13, 18]
DEFAULT_TZ = "Europe/Stockholm"

# Minimum gap between reports (50 min — prevents double-fires within same hour)
MIN_GAP_SECONDS = 3000


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _get_state():
    """Load scheduler state from file."""
    state = load_json(STATE_FILE, default={})
    return state if isinstance(state, dict) else {}


def _save_state(state):
    """Save scheduler state to file."""
    atomic_write_json(STATE_FILE, state)


def _should_run(config):
    """Check if it's time for a crypto report.

    Returns (should_run: bool, hour: int).
    """
    notification = config.get("notification", {})
    crypto_cfg = notification.get("crypto_scheduler", {})

    if not crypto_cfg.get("enabled", True):
        return False, 0

    hours = crypto_cfg.get("hours_local", DEFAULT_HOURS)
    tz_name = crypto_cfg.get("timezone", DEFAULT_TZ)

    try:
        tz = zoneinfo.ZoneInfo(tz_name)
    except Exception:
        tz = zoneinfo.ZoneInfo(DEFAULT_TZ)

    now_local = datetime.now(tz)
    current_hour = now_local.hour

    if current_hour not in hours:
        return False, 0

    # Check if we already ran this hour
    state = _get_state()
    last_ts = state.get("last_report_time", 0)
    if last_ts and (time.time() - last_ts) < MIN_GAP_SECONDS:
        return False, 0

    # Check if we ran in this specific hour today
    last_hour = state.get("last_report_hour", -1)
    last_date = state.get("last_report_date", "")
    today_str = now_local.strftime("%Y-%m-%d")

    if last_date == today_str and last_hour == current_hour:
        return False, 0

    return True, current_hour


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _build_crypto_report(config):
    """Build a structured crypto report from current signal data.

    Returns (message_text, log_entry) tuple.
    """
    # Load signal data
    summary = load_json(DATA_DIR / "agent_summary_compact.json")
    if not summary:
        return None, None

    signals = summary.get("signals", {})
    macro = summary.get("macro", {})
    futures = summary.get("futures_data", {})
    prophecy = summary.get("prophecy", {})
    mc = summary.get("monte_carlo", {})
    forecast = summary.get("forecast_signals", {})
    price_levels = summary.get("price_levels", {})

    # Load crypto macro data (options, gold-BTC ratio)
    try:
        from portfolio.crypto_macro_data import get_crypto_macro_data
        btc_macro = get_crypto_macro_data("BTC-USD")
        eth_macro = get_crypto_macro_data("ETH-USD")
    except Exception:
        btc_macro = None
        eth_macro = None

    # Extract ticker data
    btc = signals.get("BTC-USD", {})
    eth = signals.get("ETH-USD", {})
    mstr = signals.get("MSTR", {})

    btc_price = btc.get("price_usd", 0)
    eth_price = eth.get("price_usd", 0)
    mstr_price = mstr.get("price_usd", 0)

    # Fear & Greed
    fg = summary.get("fear_greed", {})
    crypto_fg = fg.get("crypto", {}).get("value") if isinstance(fg, dict) else None

    # DXY
    dxy = macro.get("dxy", {})
    dxy_val = dxy.get("value", "?")
    dxy_chg = dxy.get("change_5d_pct", "?")

    # FOMC
    fed = macro.get("fed", {})
    fomc_days = fed.get("days_until", "?")

    # VIX
    vix = macro.get("vix", {})
    vix_val = vix.get("value", "?")

    # Futures
    btc_futures = futures.get("BTC-USD", {})
    eth_futures = futures.get("ETH-USD", {})

    # Prophecy
    beliefs = prophecy.get("beliefs", []) if isinstance(prophecy, dict) else []
    btc_prophecy = next((b for b in beliefs if b.get("ticker") == "BTC-USD"), {})
    eth_prophecy = next((b for b in beliefs if b.get("ticker") == "ETH-USD"), {})

    # Monte Carlo
    btc_mc = mc.get("BTC-USD", {})

    # Options data
    btc_options = btc_macro.get("options", {}) if btc_macro else {}
    eth_options = eth_macro.get("options", {}) if eth_macro else {}
    gold_btc = btc_macro.get("gold_btc_ratio", {}) if btc_macro else {}

    # Compute ETH/BTC ratio
    eth_btc_ratio = round(eth_price / btc_price, 4) if btc_price > 0 else 0

    # Get current local time
    try:
        tz = zoneinfo.ZoneInfo(DEFAULT_TZ)
        now_local = datetime.now(tz)
        time_str = now_local.strftime("%H:%M CET")
        date_str = now_local.strftime("%Y-%m-%d")
    except Exception:
        time_str = "?"
        date_str = "?"

    # Helper for vote breakdown
    def _votes(sig):
        extra = sig.get("extra", {})
        b = extra.get("_buy_count", 0)
        s = extra.get("_sell_count", 0)
        total = extra.get("_total_applicable", 0)
        h = total - b - s
        action = extra.get("_weighted_action", sig.get("action", "HOLD"))
        conf = extra.get("_weighted_confidence", 0)
        return f"{action} ({b}B/{s}S/{h}H) wConf {conf:.0%}"

    # Build message
    lines = []
    lines.append(f"*CRYPTO REPORT — {date_str} {time_str}*")
    lines.append("")

    # Macro header
    fg_str = f"F&G {crypto_fg}" if crypto_fg else "F&G ?"
    if crypto_fg and crypto_fg <= 10:
        fg_str += " ⚠️ EXTREME FEAR"
    elif crypto_fg and crypto_fg >= 80:
        fg_str += " ⚠️ EXTREME GREED"

    lines.append(f"📊 {fg_str} | DXY {dxy_val} ({dxy_chg}%) | VIX {vix_val} | FOMC {fomc_days}d")
    lines.append("")

    # BTC section
    lines.append(f"*BTC — ${btc_price:,.0f}*")
    lines.append(f"Signal: {_votes(btc)} | RSI {btc.get('rsi', '?')}")

    if btc_options:
        mp = btc_options.get("max_pain")
        pcr = btc_options.get("nearest_pcr")
        days_exp = btc_options.get("days_to_expiry")
        exp_date = btc_options.get("nearest_expiry", "?")
        if mp:
            dist = ((btc_price - mp) / mp * 100) if mp > 0 else 0
            pcr_signal = ""
            if pcr and pcr > 1.2:
                pcr_signal = " (contrarian BUY)"
            elif pcr and pcr < 0.6:
                pcr_signal = " (contrarian SELL)"
            lines.append(f"Options: MaxPain ${mp:,.0f} ({dist:+.1f}%) | PCR {pcr}{pcr_signal}")
            if days_exp is not None and days_exp <= 3:
                lines.append(f"⚠️ Expiry {exp_date} in {days_exp}d!")

    btc_funding = btc_futures.get("funding_rate_pct", "?")
    btc_ls = btc_futures.get("ls_ratio", "?")
    lines.append(f"Futures: Funding {btc_funding}% | L/S {btc_ls}")

    if btc_prophecy:
        prog = btc_prophecy.get("progress_pct", "?")
        target = btc_prophecy.get("target_price", 100000)
        lines.append(f"Prophecy: ${target:,.0f} target, {prog}% progress")

    if btc_mc:
        p_up = btc_mc.get("p_up")
        bands_1d = btc_mc.get("price_bands_1d", {})
        if p_up is not None:
            lines.append(f"MC: P(up)={p_up:.0%} | 1d: ${bands_1d.get('5', 0):,.0f}-${bands_1d.get('95', 0):,.0f}")

    lines.append("")

    # ETH section
    lines.append(f"*ETH — ${eth_price:,.0f}*")
    lines.append(f"Signal: {_votes(eth)} | RSI {eth.get('rsi', '?')}")

    if eth_options:
        mp = eth_options.get("max_pain")
        pcr = eth_options.get("nearest_pcr")
        if mp:
            dist = ((eth_price - mp) / mp * 100) if mp > 0 else 0
            lines.append(f"Options: MaxPain ${mp:,.0f} ({dist:+.1f}%) | PCR {pcr}")

    eth_funding = eth_futures.get("funding_rate_pct", "?")
    eth_ls = eth_futures.get("ls_ratio", "?")
    lines.append(f"Futures: Funding {eth_funding}% | L/S {eth_ls}")
    lines.append(f"ETH/BTC: {eth_btc_ratio}")

    if eth_prophecy:
        prog = eth_prophecy.get("progress_pct", "?")
        lines.append(f"Prophecy: $4K target, {prog}% progress")

    lines.append("")

    # MSTR section
    lines.append(f"*MSTR — ${mstr_price:,.1f}*")
    lines.append(f"Signal: {_votes(mstr)} | RSI {mstr.get('rsi', '?')}")

    # Fundamentals from cache
    try:
        fund_cache = load_json(DATA_DIR / "fundamentals_cache.json", default={})
        mstr_fund = fund_cache.get("MSTR", {})
        if mstr_fund:
            target = mstr_fund.get("analyst_target")
            beta = mstr_fund.get("beta")
            if target:
                lines.append(f"Analyst target: ${target:.0f} | Beta: {beta:.1f}x")
    except Exception:
        logger.debug("Failed to load MSTR fundamentals from cache", exc_info=True)

    lines.append("")

    # Gold-BTC rotation
    if gold_btc:
        ratio = gold_btc.get("gold_btc_ratio")
        trend = gold_btc.get("trend", "flat")
        if ratio:
            trend_emoji = "↗️" if trend == "btc_outperforming" else "↘️" if trend == "gold_outperforming" else "➡️"
            lines.append(f"Gold/BTC: {ratio:.4f} {trend_emoji} ({trend.replace('_', ' ')})")

    # Key levels
    btc_levels = price_levels.get("BTC-USD", {})
    if btc_levels:
        vwap = btc_levels.get("vwap")
        gp = btc_levels.get("gp_upper")
        swing_hi = btc_levels.get("smc_swing_high")
        swing_lo = btc_levels.get("smc_swing_low")
        if vwap and swing_hi:
            lines.append(f"BTC levels: S ${swing_lo:,.0f} / VWAP ${vwap:,.0f} | R ${swing_hi:,.0f}")

    msg = "\n".join(lines)

    # Build log entry
    log_entry = {
        "ts": datetime.now().astimezone().isoformat(),
        "command": "crypto-scheduler",
        "tickers": ["BTC-USD", "ETH-USD", "MSTR"],
        "crypto_fear_greed": crypto_fg,
        "dxy": dxy_val if isinstance(dxy_val, (int, float)) else None,
        "fomc_days": fomc_days if isinstance(fomc_days, int) else None,
        "eth_btc_ratio": eth_btc_ratio,
        "btc": {
            "price_usd": btc_price,
            "signal_consensus": btc.get("extra", {}).get("_weighted_action", "HOLD"),
            "rsi": btc.get("rsi"),
            "max_pain": btc_options.get("max_pain"),
            "pcr": btc_options.get("nearest_pcr"),
            "funding_rate": btc_futures.get("funding_rate_pct"),
        },
        "eth": {
            "price_usd": eth_price,
            "signal_consensus": eth.get("extra", {}).get("_weighted_action", "HOLD"),
            "rsi": eth.get("rsi"),
            "max_pain": eth_options.get("max_pain"),
            "pcr": eth_options.get("nearest_pcr"),
            "funding_rate": eth_futures.get("funding_rate_pct"),
        },
        "mstr": {
            "price_usd": mstr_price,
            "signal_consensus": mstr.get("extra", {}).get("_weighted_action", "HOLD"),
            "rsi": mstr.get("rsi"),
        },
        "gold_btc_ratio": gold_btc.get("gold_btc_ratio") if gold_btc else None,
        "gold_btc_trend": gold_btc.get("trend") if gold_btc else None,
    }

    return msg, log_entry


# ---------------------------------------------------------------------------
# Sentinel function (called from main loop)
# ---------------------------------------------------------------------------

def maybe_run_crypto_report(config):
    """Check if it's time and send a crypto report.

    Called from _run_post_cycle() in main.py every loop cycle.
    """
    should, hour = _should_run(config)
    if not should:
        return

    logger.info("Crypto scheduler: running %02d:00 report", hour)

    try:
        msg, log_entry = _build_crypto_report(config)
        if not msg:
            logger.warning("Crypto scheduler: failed to build report (no data)")
            return

        # Send to Telegram
        send_or_store(msg, config, category="crypto_report")

        # Log to fin_command_log.jsonl
        if log_entry:
            atomic_append_jsonl(LOG_FILE, log_entry)

        # Update state
        try:
            tz = zoneinfo.ZoneInfo(DEFAULT_TZ)
            now_local = datetime.now(tz)
            today_str = now_local.strftime("%Y-%m-%d")
        except Exception:
            today_str = ""

        state = _get_state()
        state["last_report_time"] = time.time()
        state["last_report_hour"] = hour
        state["last_report_date"] = today_str
        _save_state(state)

        logger.info("Crypto scheduler: report sent for %02d:00", hour)

    except Exception:
        logger.warning("Crypto scheduler failed", exc_info=True)
