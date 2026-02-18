#!/usr/bin/env python3
"""iskbet_replay — Replay historical candles through the ISKBETS pipeline.

Fetches real Binance 15m candles, recomputes indicators at each step,
and runs the ISKBETS entry/exit logic with real Telegram alerts.

Usage:
    python scripts/iskbet_replay.py                         # BTC, Feb 13 12:00 UTC, 4h
    python scripts/iskbet_replay.py --start 2026-02-13T14:00
    python scripts/iskbet_replay.py --hours 6 --speed 20
    python scripts/iskbet_replay.py --ticker ETH-USD
"""

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from portfolio.main import compute_indicators, technical_signal
from portfolio.iskbets import (
    _evaluate_entry,
    check_exits,
    format_entry_alert,
    format_exit_alert,
    invoke_layer2_gate,
    _send_telegram,
    _log_telegram,
    _load_state,
    _save_state,
    DATA_DIR,
    CONFIG_FILE as ISKBETS_CONFIG_FILE,
    STATE_FILE as ISKBETS_STATE_FILE,
    compute_atr_15m,
)

APP_CONFIG_FILE = BASE_DIR / "config.json"
REPLAY_STATE_FILE = DATA_DIR / "iskbets_state_replay.json"

BINANCE_BASE = "https://api.binance.com/api/v3"

# Timeframe resample specs: (label, target_interval, min_candles_needed)
# We resample 15m candles to build multi-TF data
RESAMPLE_MAP = {
    "Now":  ("15min", 100),
    "12h":  ("1h",    100),
    "2d":   ("4h",    100),
    "7d":   ("1D",    100),
    "1mo":  ("3D",    100),
    "3mo":  ("1W",    100),
    "6mo":  ("1ME",   48),
}


# ── Data download ─────────────────────────────────────────────────────


def download_15m_candles(symbol, start_dt, end_dt, lookback_days=30):
    """Download 15m candles from Binance spot API.

    Fetches from (start - lookback) to end to have enough data for
    indicator warmup and multi-TF resampling.
    """
    fetch_start = start_dt - timedelta(days=lookback_days)
    start_ms = int(fetch_start.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    all_rows = []
    cursor = start_ms

    print(f"  Downloading {symbol} 15m candles...")
    print(f"  Range: {fetch_start.strftime('%Y-%m-%d %H:%M')} -> {end_dt.strftime('%Y-%m-%d %H:%M')} UTC")

    while cursor < end_ms:
        r = requests.get(
            f"{BINANCE_BASE}/klines",
            params={
                "symbol": symbol,
                "interval": "15m",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for k in batch:
            all_rows.append({
                "open_time": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        cursor = int(batch[-1][0]) + 1
        time.sleep(0.1)

    df = (
        pd.DataFrame(all_rows)
        .drop_duplicates("open_time")
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    print(f"  Downloaded {len(df)} candles")
    return df


# ── Multi-timeframe construction ──────────────────────────────────────


def resample_candles(df_15m, target_freq):
    """Resample 15m OHLCV candles to a larger timeframe."""
    df = df_15m.set_index("timestamp")
    resampled = df.resample(target_freq).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return resampled.reset_index()


def build_tf_data(df_15m, candle_idx):
    """Build the tf_data dict for a single ticker at a given candle index.

    Uses candles up to candle_idx (inclusive) to compute indicators
    at each timeframe resolution.
    """
    available = df_15m.iloc[:candle_idx + 1].copy()
    tf_data = []

    for label, (freq, min_candles) in RESAMPLE_MAP.items():
        if label == "Now":
            # Use raw 15m candles, last 100
            window = available.tail(100).reset_index(drop=True)
        else:
            resampled = resample_candles(available, freq)
            window = resampled.tail(min_candles).reset_index(drop=True)

        if len(window) < 26:
            tf_data.append((label, {"indicators": {}, "action": None, "confidence": None}))
            continue

        ind = compute_indicators(window)
        if ind is None:
            tf_data.append((label, {"indicators": {}, "action": None, "confidence": None}))
            continue

        if label == "Now":
            action, conf = None, None
        else:
            action, conf = technical_signal(ind)

        tf_data.append((label, {"indicators": ind, "action": action, "confidence": conf}))

    return tf_data


# ── Signal construction ───────────────────────────────────────────────


def build_signals(indicators, tf_data_list, ticker, fg_value=8):
    """Build a signals dict matching what check_iskbets/_evaluate_entry expects.

    Computes rule-based votes from indicators. Stubs external signals.
    """
    ind = indicators
    votes = {}
    extra = {}

    # RSI — adaptive thresholds
    rsi_lower = ind.get("rsi_p20", 30)
    rsi_upper = ind.get("rsi_p80", 70)
    rsi_lower = max(rsi_lower, 15)
    rsi_upper = min(rsi_upper, 85)
    if ind["rsi"] < rsi_lower:
        votes["rsi"] = "BUY"
    elif ind["rsi"] > rsi_upper:
        votes["rsi"] = "SELL"
    else:
        votes["rsi"] = "HOLD"

    # MACD — crossover only
    if ind["macd_hist"] > 0 and ind["macd_hist_prev"] <= 0:
        votes["macd"] = "BUY"
    elif ind["macd_hist"] < 0 and ind["macd_hist_prev"] >= 0:
        votes["macd"] = "SELL"
    else:
        votes["macd"] = "HOLD"

    # EMA — gap > 0.5%
    ema_gap = abs(ind["ema9"] - ind["ema21"]) / ind["ema21"] * 100 if ind["ema21"] != 0 else 0
    if ema_gap >= 0.5:
        votes["ema"] = "BUY" if ind["ema9"] > ind["ema21"] else "SELL"
    else:
        votes["ema"] = "HOLD"

    # BB — extremes only
    if ind["price_vs_bb"] == "below_lower":
        votes["bb"] = "BUY"
    elif ind["price_vs_bb"] == "above_upper":
        votes["bb"] = "SELL"
    else:
        votes["bb"] = "HOLD"

    # F&G — stubbed to historical value
    extra["fear_greed"] = fg_value
    extra["fear_greed_class"] = "Extreme Fear" if fg_value <= 20 else "Neutral"
    if fg_value <= 20:
        votes["fear_greed"] = "BUY"
    elif fg_value >= 80:
        votes["fear_greed"] = "SELL"
    else:
        votes["fear_greed"] = "HOLD"

    # Stubbed signals
    votes["sentiment"] = "HOLD"
    votes["ml"] = "HOLD"
    votes["funding"] = "HOLD"
    votes["ministral"] = "HOLD"
    votes["custom_lora"] = "HOLD"

    # Volume — compute from candles (look at Now TF indicators)
    # We compute volume ratio from the 15m candles directly
    votes["volume"] = "HOLD"
    # Volume info will be injected by the caller from candle data

    # Count votes
    buy = sum(1 for v in votes.values() if v == "BUY")
    sell = sum(1 for v in votes.values() if v == "SELL")
    total_applicable = 11  # crypto

    extra["_buy_count"] = buy
    extra["_sell_count"] = sell
    extra["_voters"] = buy + sell
    extra["_total_applicable"] = total_applicable
    extra["_votes"] = votes
    extra["volume_ratio"] = 1.0  # default, updated by caller
    extra["volume_action"] = "HOLD"

    # Build the signal dict
    active = buy + sell
    if active < 3:
        action = "HOLD"
        conf = 0.0
    else:
        buy_conf = buy / active
        sell_conf = sell / active
        if buy_conf > sell_conf and buy_conf >= 0.5:
            action = "BUY"
            conf = buy_conf
        elif sell_conf > buy_conf and sell_conf >= 0.5:
            action = "SELL"
            conf = sell_conf
        else:
            action = "HOLD"
            conf = max(buy_conf, sell_conf)

    return {
        "action": action,
        "confidence": conf,
        "price_usd": ind["close"],
        "indicators": ind,
        "extra": extra,
    }


def compute_volume_signal(df_15m, candle_idx):
    """Compute volume ratio and direction from 15m candles."""
    available = df_15m.iloc[:candle_idx + 1]
    if len(available) < 22:
        return 1.0, "HOLD"

    vol = available["volume"].astype(float)
    close = available["close"].astype(float)

    last_vol = float(vol.iloc[-2])  # last completed candle
    avg20 = float(vol.iloc[:-1].rolling(20).mean().iloc[-1])
    ratio = last_vol / avg20 if avg20 > 0 else 1.0

    # Price direction over last 3 completed candles
    if len(close) >= 5:
        price_change = float(close.iloc[-2] / close.iloc[-5] - 1)
    else:
        price_change = 0.0

    if ratio > 1.5:
        if price_change > 0:
            return ratio, "BUY"
        elif price_change < 0:
            return ratio, "SELL"
    return ratio, "HOLD"


# ── ATR from candles (no live fetch) ──────────────────────────────────


def compute_atr_from_candles(df_15m, candle_idx):
    """Compute ATR(14) from 15m candles up to candle_idx."""
    window = df_15m.iloc[max(0, candle_idx - 19):candle_idx + 1].copy()
    if len(window) < 15:
        return None

    close = window["close"].astype(float)
    high = window["high"].astype(float)
    low = window["low"].astype(float)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=14, adjust=False).mean().iloc[-1]
    return float(atr)


# ── Replay state (isolated from production) ───────────────────────────


def load_replay_state():
    if REPLAY_STATE_FILE.exists():
        try:
            return json.loads(REPLAY_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"active_position": None, "trade_history": []}


def save_replay_state(state):
    DATA_DIR.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=DATA_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, str(REPLAY_STATE_FILE))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── Main replay loop ──────────────────────────────────────────────────


def run_replay(ticker, symbol, start_dt, hours, speed, fg_value, config,
               auto_entry=False, amount_sek=100000, layer2_gate=False):
    end_dt = start_dt + timedelta(hours=hours)
    iskbets_cfg = config.get("iskbets", {})
    fx_rate = 9.0  # approximate USD/SEK for Feb 13

    # Download candles
    df = download_15m_candles(symbol, start_dt, end_dt, lookback_days=30)
    if df.empty:
        print("ERROR: No candles downloaded")
        return

    # Find the candle index where replay starts
    start_ts = pd.Timestamp(start_dt)
    replay_mask = df["timestamp"] >= start_ts
    if not replay_mask.any():
        print(f"ERROR: No candles after {start_dt}")
        return

    first_replay_idx = replay_mask.idxmax()
    end_ts = pd.Timestamp(end_dt)
    end_mask = df["timestamp"] <= end_ts
    last_replay_idx = end_mask[::-1].idxmax() if end_mask.any() else len(df) - 1

    replay_candles = range(first_replay_idx, last_replay_idx + 1)
    total = len(replay_candles)

    print(f"\n{'='*65}")
    print(f"  ISKBETS REPLAY — {ticker}")
    print(f"  {start_dt.strftime('%Y-%m-%d %H:%M')} ->{end_dt.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  {total} candles × {speed}s = ~{total * speed // 60} min real time")
    print(f"  F&G stubbed to {fg_value} (Extreme Fear)")
    min_bb = iskbets_cfg.get("min_bigbet_conditions", 2)
    min_bv = iskbets_cfg.get("min_buy_votes", 3)
    print(f"  Gates: bigbet >= {min_bb}, buy_votes >= {min_bv}")
    print(f"  Ctrl+C to stop")
    print(f"{'='*65}\n")

    # Clear replay state
    state = {"active_position": None, "trade_history": []}
    save_replay_state(state)

    # Also write to the production state file so check_exits/handle_command work
    # We'll restore it after replay
    orig_state_backup = None
    if ISKBETS_STATE_FILE.exists():
        orig_state_backup = ISKBETS_STATE_FILE.read_text(encoding="utf-8")

    entry_alert_sent = False
    cooldown_until = 0  # candle index — no re-entry until past this
    COOLDOWN_CANDLES = 8  # 8 × 15min = 2h cooldown after exit

    try:
        for step, idx in enumerate(replay_candles):
            row = df.iloc[idx]
            sim_time = row["timestamp"]
            price = row["close"]

            # Build indicators from candles up to this point
            ind = compute_indicators(df.iloc[:idx + 1].tail(100).reset_index(drop=True))
            if ind is None:
                print(f"  {sim_time.strftime('%H:%M')} — insufficient data for indicators")
                time.sleep(1)
                continue

            # Build multi-TF data
            tf_data_list = build_tf_data(df, idx)
            tf_data = {ticker: tf_data_list}

            # Compute volume signal
            vol_ratio, vol_action = compute_volume_signal(df, idx)

            # Build signals dict
            sig = build_signals(ind, tf_data_list, ticker, fg_value)
            sig["extra"]["volume_ratio"] = vol_ratio
            sig["extra"]["volume_action"] = vol_action

            # Update volume vote
            if vol_action != "HOLD":
                sig["extra"]["_votes"]["volume"] = vol_action
                # Recount
                buy = sum(1 for v in sig["extra"]["_votes"].values() if v == "BUY")
                sell = sum(1 for v in sig["extra"]["_votes"].values() if v == "SELL")
                sig["extra"]["_buy_count"] = buy
                sig["extra"]["_sell_count"] = sell
                sig["extra"]["_voters"] = buy + sell

            signals = {ticker: sig}
            prices_usd = {ticker: price}

            # Load current state
            state = load_replay_state()

            # Status line
            votes = sig["extra"]["_votes"]
            buy_names = [k for k, v in votes.items() if v == "BUY"]
            sell_names = [k for k, v in votes.items() if v == "SELL"]
            buy_c = sig["extra"]["_buy_count"]
            sell_c = sig["extra"]["_sell_count"]
            rsi_val = ind["rsi"]
            macd_val = ind["macd_hist"]

            if state.get("active_position"):
                # Monitor exits
                pos = state["active_position"]
                entry_price = pos["entry_price_usd"]
                pnl_pct = ((price - entry_price) / entry_price) * 100
                stop = pos.get("stop_loss", 0)
                be = pos.get("stop_at_breakeven", False)

                # Run exit check
                result = check_exits(state, prices_usd, signals, tf_data, iskbets_cfg)

                if result:
                    exit_type, detail = result

                    if exit_type == "stage1_hit":
                        # Stage 1 — notify but don't close
                        msg = format_exit_alert(
                            ticker, price, exit_type,
                            entry_price, pos["amount_sek"],
                            pos["entry_time"], fx_rate,
                            exit_time=sim_time,
                        )
                        _log_telegram(msg)
                        try:
                            _send_telegram(msg, config)
                        except Exception as e:
                            print(f"  Telegram error: {e}")
                        print(f"\n  {'='*60}")
                        print(f"  STAGE 1 HIT — {ticker} ${price:,.2f} ({pnl_pct:+.1f}%)")
                        print(f"  Stop moved to breakeven ${entry_price:,.2f}")
                        print(f"  {'='*60}\n")
                        save_replay_state(state)
                    else:
                        # Real exit
                        msg = format_exit_alert(
                            ticker, price, exit_type,
                            entry_price, pos["amount_sek"],
                            pos["entry_time"], fx_rate,
                            exit_time=sim_time,
                        )
                        _log_telegram(msg)
                        try:
                            _send_telegram(msg, config)
                        except Exception as e:
                            print(f"  Telegram error: {e}")

                        # Log trade
                        shares = pos["amount_sek"] / (entry_price * fx_rate)
                        pnl_sek = shares * (price - entry_price) * fx_rate
                        state["trade_history"].append({
                            "ticker": ticker,
                            "entry_price_usd": entry_price,
                            "exit_price_usd": price,
                            "amount_sek": pos["amount_sek"],
                            "pnl_sek": round(pnl_sek, 2),
                            "pnl_pct": round(pnl_pct, 2),
                            "exit_type": exit_type,
                            "entry_time": pos["entry_time"],
                            "exit_time": sim_time.isoformat(),
                        })
                        state["active_position"] = None
                        save_replay_state(state)

                        print(f"\n  {'='*60}")
                        print(f"  EXIT ({exit_type}) — {ticker} ${price:,.2f} ({pnl_pct:+.1f}%)")
                        print(f"  P&L: {pnl_sek:+,.0f} SEK")
                        print(f"  {'='*60}\n")
                        entry_alert_sent = False
                        cooldown_until = idx + COOLDOWN_CANDLES
                else:
                    save_replay_state(state)

                # Terminal status (no Telegram for hold)
                stop_str = f"stop ${stop:,.0f}" + (" BE" if be else "")
                print(
                    f"  {sim_time.strftime('%H:%M')} ${price:,.0f} "
                    f"| {pnl_pct:+.1f}% | {stop_str} "
                    f"| {buy_c}B/{sell_c}S"
                    f"  [{step+1}/{total}]"
                )
            else:
                # Scan for entry
                # Monkey-patch _before_cutoff to use simulated time
                sim_et_hour = (sim_time - timedelta(hours=5)).hour  # rough UTC→ET
                sim_et_min = (sim_time - timedelta(hours=5)).minute
                cutoff_str = iskbets_cfg.get("entry_cutoff_et", "14:30")
                cutoff_parts = cutoff_str.split(":")
                cutoff_h = int(cutoff_parts[0])
                cutoff_m = int(cutoff_parts[1]) if len(cutoff_parts) > 1 else 0
                before_cutoff = (sim_et_hour < cutoff_h) or (sim_et_hour == cutoff_h and sim_et_min < cutoff_m)

                should_enter = False
                conditions = []

                if before_cutoff:
                    with patch("portfolio.iskbets._before_cutoff", return_value=True):
                        should_enter, conditions = _evaluate_entry(
                            ticker, signals, prices_usd, tf_data, iskbets_cfg, config
                        )

                if should_enter and not entry_alert_sent and idx >= cooldown_until:
                    # Compute ATR from candles instead of live fetch
                    atr = compute_atr_from_candles(df, idx)
                    if atr is None:
                        atr = price * 0.02

                    # Layer 2 gate (optional)
                    l2_reasoning = ""
                    if layer2_gate:
                        gate_cfg = dict(iskbets_cfg)
                        gate_cfg["layer2_gate"] = True
                        approved, l2_reasoning = invoke_layer2_gate(
                            ticker, price, conditions, signals, tf_data, atr, gate_cfg, config
                        )
                        if not approved:
                            print(f"  L2 GATE: SKIP — {l2_reasoning}")
                            print(
                                f"  {sim_time.strftime('%H:%M')} ${price:,.0f} "
                                f"| RSI {rsi_val:.0f} MACD {macd_val:+.0f} "
                                f"| {buy_c}B/{sell_c}S | L2 SKIP"
                                f"  [{step+1}/{total}]"
                            )
                            time.sleep(speed)
                            continue

                    msg = format_entry_alert(ticker, price, conditions, atr, iskbets_cfg, signals=signals, l2_reasoning=l2_reasoning)
                    # Tag as replay
                    msg = msg.replace("ISKBETS BUY", "ISKBETS REPLAY")
                    _log_telegram(msg)
                    try:
                        _send_telegram(msg, config)
                    except Exception as e:
                        print(f"  Telegram error: {e}")

                    entry_alert_sent = True

                    hard_stop_mult = iskbets_cfg.get("hard_stop_atr_mult", 2.0)
                    stage1_mult = iskbets_cfg.get("stage1_atr_mult", 1.5)
                    stop = price - (hard_stop_mult * atr)
                    stage1 = price + (stage1_mult * atr)

                    print(f"\n  {'='*60}")
                    print(f"  ENTRY ALERT — {ticker} ${price:,.2f}")
                    for c in conditions:
                        print(f"    * {c}")
                    print(f"  ATR(15m): ${atr:,.2f}")
                    print(f"  Stop: ${stop:,.2f} | Stage 1: ${stage1:,.2f}")
                    print(f"  {'='*60}")

                    if auto_entry:
                        # Auto-confirm the entry
                        _create_position(state, ticker, price, atr, amount_sek, fx_rate, iskbets_cfg, sim_time)
                        save_replay_state(state)
                        print(f"  AUTO-ENTRY: {amount_sek:,.0f} SEK @ ${price:,.2f}")
                        print(f"  Monitoring exits...\n")
                    else:
                        print(f"  Type amount (SEK) to confirm, or Enter to skip:")
                        try:
                            import select
                            # Non-blocking input with timeout
                            user_input = input(f"  > ").strip()
                        except EOFError:
                            user_input = ""

                        if user_input:
                            try:
                                amt = float(user_input)
                            except ValueError:
                                amt = amount_sek
                            _create_position(state, ticker, price, atr, amt, fx_rate, iskbets_cfg, sim_time)
                            save_replay_state(state)
                            print(f"  BOUGHT: {amt:,.0f} SEK @ ${price:,.2f}")
                            print(f"  Monitoring exits...\n")
                        else:
                            print(f"  Skipped. Continuing scan...\n")

                # Terminal status
                bb_pos = ind.get("price_vs_bb", "in")[:2]
                if state.get("active_position"):
                    status_str = ""
                elif idx < cooldown_until:
                    status_str = "cooldown"
                elif should_enter:
                    status_str = "ENTRY!"
                else:
                    status_str = "no entry"
                print(
                    f"  {sim_time.strftime('%H:%M')} ${price:,.0f} "
                    f"| RSI {rsi_val:.0f} MACD {macd_val:+.0f} BB:{bb_pos} "
                    f"| {buy_c}B/{sell_c}S ({','.join(buy_names) or '-'}) "
                    f"| {status_str}"
                    f"  [{step+1}/{total}]"
                )

            # Check if user wants to stop (non-blocking)
            time.sleep(speed)

        # End of replay
        state = load_replay_state()
        print(f"\n{'='*65}")
        print(f"  REPLAY COMPLETE — {ticker}")
        print(f"  {start_dt.strftime('%Y-%m-%d %H:%M')} ->{end_dt.strftime('%Y-%m-%d %H:%M')} UTC")

        history = state.get("trade_history", [])
        if history:
            total_pnl = sum(t["pnl_sek"] for t in history)
            print(f"  Trades: {len(history)}")
            for t in history:
                print(f"    {t['ticker']} {t['pnl_pct']:+.1f}% ({t['pnl_sek']:+,.0f} SEK) — {t['exit_type']}")
            print(f"  Total P&L: {total_pnl:+,.0f} SEK")
        else:
            print(f"  No trades completed")

        pos = state.get("active_position")
        if pos:
            final_price = df.iloc[last_replay_idx]["close"]
            pnl = ((final_price - pos["entry_price_usd"]) / pos["entry_price_usd"]) * 100
            print(f"  Open position: {pos['ticker']} @ ${pos['entry_price_usd']:,.2f} ->${final_price:,.2f} ({pnl:+.1f}%)")

        print(f"{'='*65}")

    except KeyboardInterrupt:
        print(f"\n\n  Replay stopped by user.")
        state = load_replay_state()
        if state.get("active_position"):
            pos = state["active_position"]
            print(f"  Open position: {pos['ticker']} @ ${pos['entry_price_usd']:,.2f}")

    finally:
        # Restore original state if we backed it up
        if orig_state_backup is not None:
            ISKBETS_STATE_FILE.write_text(orig_state_backup, encoding="utf-8")
            print("  Restored original iskbets_state.json")
        elif ISKBETS_STATE_FILE.exists():
            # We wrote replay state to production file — clean it up
            pass


def _create_position(state, ticker, price, atr, amount_sek, fx_rate, iskbets_cfg, sim_time=None):
    """Create an active position in state (simulating 'bought' confirmation)."""
    hard_stop_mult = iskbets_cfg.get("hard_stop_atr_mult", 2.0)
    stage1_mult = iskbets_cfg.get("stage1_atr_mult", 1.5)

    shares = amount_sek / (price * fx_rate)
    stop = price - (hard_stop_mult * atr)
    stage1 = price + (stage1_mult * atr)

    entry_time = sim_time.isoformat() if sim_time else datetime.now(timezone.utc).isoformat()

    state["active_position"] = {
        "ticker": ticker,
        "entry_price_usd": price,
        "amount_sek": amount_sek,
        "shares": round(shares, 6),
        "entry_time": entry_time,
        "atr_15m": round(atr, 4),
        "stop_loss": round(stop, 2),
        "stage1_target": round(stage1, 2),
        "stop_at_breakeven": False,
        "highest_price": price,
        "sell_signal_streak": 0,
        "fx_rate": fx_rate,
    }


TICKER_MAP = {
    "BTC-USD":  "BTCUSDT",
    "ETH-USD":  "ETHUSDT",
}


def main():
    parser = argparse.ArgumentParser(
        description="Replay historical candles through ISKBETS pipeline"
    )
    parser.add_argument(
        "--ticker", default="BTC-USD",
        help="Ticker to replay (default: BTC-USD)"
    )
    parser.add_argument(
        "--start", default="2026-02-13T12:00",
        help="Replay start time in UTC ISO format (default: 2026-02-13T12:00)"
    )
    parser.add_argument(
        "--hours", type=float, default=4,
        help="Hours of data to replay (default: 4)"
    )
    parser.add_argument(
        "--speed", type=float, default=15,
        help="Seconds per 15m candle (default: 15)"
    )
    parser.add_argument(
        "--fg", type=int, default=8,
        help="Fear & Greed value to stub (default: 8, was 5-8 on Feb 13)"
    )
    parser.add_argument(
        "--min-bigbet", type=int, default=None,
        help="Override min_bigbet_conditions (default: from config, usually 2)"
    )
    parser.add_argument(
        "--min-votes", type=int, default=None,
        help="Override min_buy_votes (default: from config, usually 3)"
    )
    parser.add_argument(
        "--auto-entry", action="store_true",
        help="Auto-confirm entries (no waiting, default 100K SEK)"
    )
    parser.add_argument(
        "--amount", type=float, default=100000,
        help="SEK amount for auto-entry or default prompt (default: 100000)"
    )
    parser.add_argument(
        "--layer2-gate", action="store_true",
        help="Enable Layer 2 (Claude) APPROVE/SKIP gate on entries"
    )

    args = parser.parse_args()

    ticker = args.ticker.upper()
    if ticker in ("BTC", "BTCUSD"):
        ticker = "BTC-USD"
    elif ticker in ("ETH", "ETHUSD"):
        ticker = "ETH-USD"

    symbol = TICKER_MAP.get(ticker)
    if not symbol:
        print(f"ERROR: Replay only supports crypto tickers (BTC-USD, ETH-USD)")
        print(f"  Stocks need Alpaca historical data which has different access patterns.")
        sys.exit(1)

    # Parse start time
    try:
        start_dt = datetime.fromisoformat(args.start)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
    except ValueError:
        print(f"ERROR: Invalid start time: {args.start}")
        print(f"  Use ISO format like: 2026-02-13T12:00")
        sys.exit(1)

    # Load config
    try:
        config = json.loads(APP_CONFIG_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR: Cannot read config.json: {e}")
        sys.exit(1)

    # Warn if production loop may be running
    summary_file = DATA_DIR / "agent_summary.json"
    if summary_file.exists():
        age_min = (time.time() - summary_file.stat().st_mtime) / 60
        if age_min < 5:
            print(f"\n  WARNING: agent_summary.json is {age_min:.0f}m old — production loop may be running!")
            print(f"  The TelegramPoller may conflict. Consider pausing the loop first.")
            print(f"  Press Ctrl+C to abort or wait 5s to continue...")
            try:
                time.sleep(5)
            except KeyboardInterrupt:
                sys.exit(0)

    # Apply threshold overrides
    if args.min_bigbet is not None:
        config.setdefault("iskbets", {})["min_bigbet_conditions"] = args.min_bigbet
    if args.min_votes is not None:
        config.setdefault("iskbets", {})["min_buy_votes"] = args.min_votes

    run_replay(
        ticker, symbol, start_dt, args.hours, args.speed, args.fg, config,
        auto_entry=args.auto_entry, amount_sek=args.amount,
        layer2_gate=args.layer2_gate,
    )


if __name__ == "__main__":
    main()
