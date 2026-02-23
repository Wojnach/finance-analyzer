"""Agent summary reporting — builds JSON summaries for Layer 2 consumption."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from portfolio.shared_state import _cached
from portfolio.indicators import detect_regime
from portfolio.portfolio_mgr import _atomic_write_json, portfolio_value
from portfolio.tickers import CRYPTO_SYMBOLS, STOCK_SYMBOLS
from portfolio.signal_registry import get_enhanced_signals

import portfolio.shared_state as _ss

logger = logging.getLogger("portfolio.reporting")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
AGENT_SUMMARY_FILE = DATA_DIR / "agent_summary.json"
COMPACT_SUMMARY_FILE = DATA_DIR / "agent_summary_compact.json"
TIER1_FILE = DATA_DIR / "agent_context_t1.json"
TIER2_FILE = DATA_DIR / "agent_context_t2.json"


def _cross_asset_signals(all_signals):
    btc = all_signals.get("BTC-USD", {})
    btc_action = btc.get("action", "HOLD")
    if btc_action == "HOLD":
        return {}

    followers = {"ETH-USD": "BTC-USD", "MSTR": "BTC-USD"}
    leads = {}
    for follower, leader in followers.items():
        f_data = all_signals.get(follower, {})
        f_action = f_data.get("action", "HOLD")
        if f_action == "HOLD" and btc_action != "HOLD":
            leads[follower] = {
                "leader": leader,
                "leader_action": btc_action,
                "note": f"{leader} is {btc_action} but {follower} hasn't moved yet",
            }
    return leads


def write_agent_summary(
    signals, prices_usd, fx_rate, state, tf_data, trigger_reasons=None
):
    total = portfolio_value(state, prices_usd, fx_rate)
    pnl_pct = ((total - state["initial_value_sek"]) / state["initial_value_sek"]) * 100

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trigger_reasons": trigger_reasons or [],
        "fx_rate": round(fx_rate, 2),
        "portfolio": {
            "total_sek": round(total),
            "pnl_pct": round(pnl_pct, 2),
            "cash_sek": round(state["cash_sek"]),
            "holdings": state.get("holdings", {}),
            "num_transactions": len(state.get("transactions", [])),
        },
        "signals": {},
        "timeframes": {},
        "fear_greed": {},
    }

    for name, sig in signals.items():
        extra = sig.get("extra", {})
        ind = sig["indicators"]
        # Collect enhanced signal summaries (compact, no sub_signals detail in top-level)
        enhanced = {}
        for esig_name in get_enhanced_signals():
            eaction = extra.get(f"{esig_name}_action", "HOLD")
            econf = extra.get(f"{esig_name}_confidence", 0.0)
            enhanced[esig_name] = {"action": eaction, "confidence": econf}

        # Strip bulky sub_signals from extra to keep agent_summary.json lean
        extra_clean = {k: v for k, v in extra.items() if not k.endswith("_sub_signals")}

        sig_entry = {
            "action": sig["action"],
            "confidence": sig["confidence"],
            "weighted_confidence": extra.get("_weighted_confidence", 0.0),
            "confluence_score": extra.get("_confluence_score", 0.0),
            "price_usd": ind["close"],
            "rsi": round(ind["rsi"], 1),
            "macd_hist": round(ind["macd_hist"], 2),
            "bb_position": ind["price_vs_bb"],
            "atr": round(ind.get("atr", 0), 4),
            "atr_pct": round(ind.get("atr_pct", 0), 2),
            "regime": detect_regime(ind, is_crypto=name in CRYPTO_SYMBOLS),
            "enhanced_signals": enhanced,
            "extra": extra_clean,
        }
        # Mark extended-hours data for stocks (yfinance prepost during off-hours)
        if name in STOCK_SYMBOLS and _ss._current_market_state in ("closed", "weekend"):
            sig_entry["extended_hours"] = True
        summary["signals"][name] = sig_entry
        if "fear_greed" in extra:
            summary["fear_greed"][name] = {
                "value": extra["fear_greed"],
                "classification": extra.get("fear_greed_class", ""),
            }

        tf_list = []
        for label, entry in tf_data.get(name, []):
            if "error" in entry:
                tf_list.append({"horizon": label, "error": entry["error"]})
            else:
                ei = entry["indicators"]
                tf_list.append(
                    {
                        "horizon": label,
                        "action": entry["action"] if label != "Now" else sig["action"],
                        "confidence": (
                            entry["confidence"] if label != "Now" else sig["confidence"]
                        ),
                        "rsi": round(ei["rsi"], 1),
                        "macd_hist": round(ei["macd_hist"], 2),
                        "ema_bullish": ei["ema9"] > ei["ema21"],
                        "bb_position": ei["price_vs_bb"],
                    }
                )
        summary["timeframes"][name] = tf_list

    # Macro context (non-voting, for Claude Code reasoning)
    try:
        from portfolio.macro_context import get_dxy, get_fed_calendar, get_treasury

        macro = {}
        dxy = _cached("dxy", 3600, get_dxy)
        if dxy:
            macro["dxy"] = dxy
        treasury = _cached("treasury", 3600, get_treasury)
        if treasury:
            macro["treasury"] = treasury
        fed = get_fed_calendar()
        if fed:
            macro["fed"] = fed
        if macro:
            summary["macro"] = macro
    except (ImportError, Exception):
        pass

    try:
        from portfolio.accuracy_stats import (
            signal_accuracy,
            consensus_accuracy,
            best_worst_signals,
            load_cached_accuracy,
            write_accuracy_cache,
        )

        # Use cached accuracy to avoid redundant full-log scans
        sig_acc = load_cached_accuracy("1d")
        if not sig_acc:
            sig_acc = signal_accuracy("1d")
            if sig_acc:
                write_accuracy_cache("1d", sig_acc)
        cons_acc = consensus_accuracy("1d")
        bw = best_worst_signals("1d", acc=sig_acc)
        qualified = {k: v for k, v in sig_acc.items() if v["total"] >= 5}
        if qualified:
            summary["signal_accuracy_1d"] = {
                "signals": {
                    k: {"accuracy": round(v["accuracy"], 3), "samples": v["total"]}
                    for k, v in qualified.items()
                },
                "consensus": {
                    "accuracy": round(cons_acc["accuracy"], 3),
                    "samples": cons_acc["total"],
                },
                "best": bw.get("best"),
                "worst": bw.get("worst"),
            }
    except Exception:
        pass

    # Signal activation rates (normalized weights for Layer 2 reference)
    try:
        from portfolio.accuracy_stats import load_cached_activation_rates
        act_rates = load_cached_activation_rates()
        if act_rates:
            summary["signal_weights"] = {
                name: {
                    "activation_rate": d["activation_rate"],
                    "normalized_weight": d["normalized_weight"],
                    "bias": d["bias"],
                }
                for name, d in act_rates.items()
                if d.get("samples", 0) > 0
            }
    except Exception:
        pass

    cross_leads = _cross_asset_signals(signals)
    if cross_leads:
        summary["cross_asset_leads"] = cross_leads

    # Avanza-tracked instruments (Tier 2: Nordic equities, Tier 3: warrants)
    try:
        from portfolio.avanza_tracker import fetch_avanza_prices
        avanza_prices = fetch_avanza_prices()
        if avanza_prices:
            summary["avanza_instruments"] = avanza_prices
    except Exception:
        pass

    # Preserve stale data for instruments not in current cycle (e.g. stocks off-hours)
    # so Layer 2 always sees all instruments
    if AGENT_SUMMARY_FILE.exists():
        try:
            prev = json.loads(AGENT_SUMMARY_FILE.read_text(encoding="utf-8"))
            for section in ("signals", "timeframes", "fear_greed"):
                prev_section = prev.get(section, {})
                for ticker, data in prev_section.items():
                    if ticker not in summary[section]:
                        if section == "signals" and isinstance(data, dict):
                            data["stale"] = True
                        summary[section][ticker] = data
        except Exception:
            pass

    _atomic_write_json(AGENT_SUMMARY_FILE, summary)
    _write_compact_summary(summary)
    return summary


def _get_held_tickers():
    """Return set of tickers held in either Patient or Bold portfolio."""
    held = set()
    for fname in ("portfolio_state.json", "portfolio_state_bold.json"):
        try:
            state = json.loads((DATA_DIR / fname).read_text(encoding="utf-8"))
            for ticker, pos in state.get("holdings", {}).items():
                if pos.get("shares", 0) > 0:
                    held.add(ticker)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    return held


def _write_compact_summary(summary):
    """Write a stripped version of agent_summary for Layer 2 (<15K tokens).

    Full detail (including per-signal _votes) only for "interesting" tickers:
      - Non-HOLD consensus (action != "HOLD")
      - Active position in either portfolio
    HOLD tickers with no position get a minimal entry (no _votes dict,
    no per-signal extras) to save ~25 lines each.
    """
    KEEP_EXTRA_FULL = {
        "fear_greed", "fear_greed_class",
        "sentiment", "sentiment_conf",
        "ml_action", "ml_confidence",
        "funding_rate", "funding_action",
        "volume_ratio", "volume_action",
        "ministral_action",
        "_voters", "_total_applicable", "_buy_count", "_sell_count",
        "_votes", "_weighted_action", "_weighted_confidence",
        "_confluence_score",
    }
    # Minimal extra for HOLD-no-position tickers (just counts, no per-signal votes)
    KEEP_EXTRA_MINIMAL = {
        "_voters", "_total_applicable", "_buy_count", "_sell_count",
        "_weighted_action", "_weighted_confidence",
    }
    # Minimal top-level fields for HOLD-no-position tickers
    KEEP_TICKER_MINIMAL = {
        "action", "confidence", "price_usd", "rsi", "regime", "extra",
    }

    held_tickers = _get_held_tickers()

    # Build compact version without full deep copy — only copy sub-dicts we modify
    compact = {k: v for k, v in summary.items()
               if k not in ("signals", "timeframes", "fear_greed", "signal_weights")}

    compact["signals"] = {}
    for ticker, ticker_data in summary.get("signals", {}).items():
        action = ticker_data.get("action", "HOLD")
        is_held = ticker in held_tickers
        is_interesting = action != "HOLD" or is_held

        if is_interesting:
            # Keep all fields except enhanced_signals
            td = {k: v for k, v in ticker_data.items() if k != "enhanced_signals"}
            if "extra" in td:
                extra = {k: v for k, v in td["extra"].items()
                         if k in KEEP_EXTRA_FULL}
                # For non-held tickers, collapse _votes dict into a compact string
                # to save ~23 lines per ticker while preserving the info
                if not is_held and "_votes" in extra:
                    votes = extra["_votes"]
                    buys = [s for s, v in votes.items() if v == "BUY"]
                    sells = [s for s, v in votes.items() if v == "SELL"]
                    parts = []
                    if buys:
                        parts.append("B:" + ",".join(buys))
                    if sells:
                        parts.append("S:" + ",".join(sells))
                    extra["_vote_detail"] = " | ".join(parts) if parts else "none"
                    del extra["_votes"]
                td["extra"] = extra
        else:
            # Minimal — just enough to know price and that it's HOLD
            td = {k: v for k, v in ticker_data.items()
                  if k in KEEP_TICKER_MINIMAL and k != "extra"}
            if "extra" in ticker_data:
                td["extra"] = {k: v for k, v in ticker_data["extra"].items()
                               if k in KEEP_EXTRA_MINIMAL}

        compact["signals"][ticker] = td

    # Only include timeframes for interesting tickers
    compact["timeframes"] = {}
    for ticker, tf_list in summary.get("timeframes", {}).items():
        action = summary.get("signals", {}).get(ticker, {}).get("action", "HOLD")
        if action != "HOLD" or ticker in held_tickers:
            compact["timeframes"][ticker] = [
                {"horizon": tf["horizon"], "action": tf.get("action", "HOLD")}
                if "error" not in tf else {"horizon": tf["horizon"], "error": tf["error"]}
                for tf in tf_list
            ]

    _atomic_write_json(COMPACT_SUMMARY_FILE, compact)


# ---------------------------------------------------------------------------
# Tiered summary generators (T1 quick-check, T2 signal-analysis)
# T3 uses the existing compact summary unchanged.
# ---------------------------------------------------------------------------

def write_tiered_summary(summary, tier, triggered_tickers=None):
    """Write a tier-specific context file for Layer 2 invocation.

    Args:
        summary: Full agent_summary dict (from write_agent_summary).
        tier: 1, 2, or 3.
        triggered_tickers: Set of tickers that caused the trigger (for T2 filtering).
    """
    if tier == 1:
        _write_tier1_summary(summary)
    elif tier == 2:
        _write_tier2_summary(summary, triggered_tickers)
    # Tier 3 uses existing agent_summary_compact.json — no extra file needed


def _portfolio_snapshot(state_file):
    """Load a portfolio state file and return a compact snapshot dict."""
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
        cash = state.get("cash_sek", 0)
        initial = state.get("initial_value_sek", 500000)
        pnl_pct = round(((cash - initial) / initial) * 100, 2) if initial else 0
        return {
            "cash_sek": round(cash),
            "total_sek": round(cash),  # approximate — no live prices in snapshot
            "pnl_pct": pnl_pct,
        }
    except (FileNotFoundError, json.JSONDecodeError):
        return {"cash_sek": 0, "total_sek": 0, "pnl_pct": 0}


def _macro_headline(summary):
    """Build a one-line macro headline from summary data."""
    parts = []
    macro = summary.get("macro", {})
    dxy = macro.get("dxy", {})
    if dxy:
        val = dxy.get("value", dxy.get("close"))
        change_5d = dxy.get("change_5d_pct", 0)
        arrow = "↑" if change_5d > 0 else "↓" if change_5d < 0 else ""
        if val:
            parts.append(f"DXY {val:.0f}{arrow}" if val > 10 else f"DXY {val}{arrow}")

    treasury = macro.get("treasury", {})
    y10 = treasury.get("10y")
    if y10:
        change_5d = treasury.get("10y_change_5d_pct", 0)
        arrow = "↑" if change_5d > 0 else "↓" if change_5d < 0 else ""
        parts.append(f"10Y {y10}{arrow}")

    # F&G: try to get crypto and stock values
    fg = summary.get("fear_greed", {})
    fg_vals = []
    for ticker in ("BTC-USD", "ETH-USD"):
        v = fg.get(ticker, {}).get("value")
        if v is not None:
            fg_vals.append(str(v))
            break
    # Use first stock F&G if available
    for ticker, fgd in fg.items():
        if ticker not in ("BTC-USD", "ETH-USD") and isinstance(fgd, dict):
            v = fgd.get("value")
            if v is not None:
                fg_vals.append(str(v))
                break
    if fg_vals:
        parts.append(f"F&G {'/'.join(fg_vals)}")

    fed = macro.get("fed", {})
    days = fed.get("days_until")
    if days is not None:
        parts.append(f"FOMC {days}d")

    return " · ".join(parts) if parts else ""


def _write_tier1_summary(summary):
    """Tier 1: Quick check — held positions only + macro headline + all prices."""
    held_tickers = _get_held_tickers()
    signals = summary.get("signals", {})
    timeframes = summary.get("timeframes", {})

    t1 = {
        "tier": 1,
        "timestamp": summary.get("timestamp", ""),
        "trigger_reasons": summary.get("trigger_reasons", []),
        "fx_rate": summary.get("fx_rate", 0),
        "held_positions": {},
        "portfolio_patient": _portfolio_snapshot(DATA_DIR / "portfolio_state.json"),
        "portfolio_bold": _portfolio_snapshot(DATA_DIR / "portfolio_state_bold.json"),
        "macro_headline": _macro_headline(summary),
        "all_prices": {},
    }

    # Held positions with actionable detail
    for ticker in held_tickers:
        sig = signals.get(ticker, {})
        extra = sig.get("extra", {})
        entry = {
            "price_usd": sig.get("price_usd", 0),
            "action": sig.get("action", "HOLD"),
            "confidence": sig.get("confidence", 0),
            "rsi": sig.get("rsi", 0),
            "regime": sig.get("regime", "unknown"),
            "atr_pct": sig.get("atr_pct", 0),
            "votes": f"{extra.get('_buy_count', 0)}B/{extra.get('_sell_count', 0)}S/{extra.get('_voters', 0) - extra.get('_buy_count', 0) - extra.get('_sell_count', 0)}H",
        }
        # Compact timeframe heatmap
        tf_list = timeframes.get(ticker, [])
        if tf_list:
            heatmap = ""
            for tf in tf_list:
                a = tf.get("action", "HOLD")
                heatmap += "B" if a == "BUY" else "S" if a == "SELL" else "·"
            entry["timeframes"] = heatmap
        t1["held_positions"][ticker] = entry

    # All prices for context comparison
    for ticker, sig in signals.items():
        t1["all_prices"][ticker] = round(sig.get("price_usd", 0), 2)

    _atomic_write_json(TIER1_FILE, t1)


def _write_tier2_summary(summary, triggered_tickers=None):
    """Tier 2: Signal analysis — triggered + held + top 5 interesting tickers."""
    held_tickers = _get_held_tickers()
    triggered_tickers = triggered_tickers or set()
    signals = summary.get("signals", {})
    timeframes = summary.get("timeframes", {})

    KEEP_EXTRA_FULL = {
        "fear_greed", "fear_greed_class",
        "sentiment", "sentiment_conf",
        "ml_action", "ml_confidence",
        "funding_rate", "funding_action",
        "volume_ratio", "volume_action",
        "ministral_action",
        "_voters", "_total_applicable", "_buy_count", "_sell_count",
        "_votes", "_weighted_action", "_weighted_confidence",
        "_confluence_score",
    }

    # Categorize tickers
    full_detail_tickers = held_tickers | triggered_tickers
    remaining = []
    for ticker, sig in signals.items():
        if ticker not in full_detail_tickers:
            buy_count = sig.get("extra", {}).get("_buy_count", 0) or 0
            sell_count = sig.get("extra", {}).get("_sell_count", 0) or 0
            active = buy_count + sell_count
            remaining.append((ticker, active, sig))

    # Top 5 non-triggered non-held by active voter count
    remaining.sort(key=lambda x: x[1], reverse=True)
    medium_tickers = {t for t, _, _ in remaining[:5]}

    t2 = {
        "tier": 2,
        "timestamp": summary.get("timestamp", ""),
        "trigger_reasons": summary.get("trigger_reasons", []),
        "fx_rate": summary.get("fx_rate", 0),
        "signals": {},
        "timeframes": {},
    }

    for ticker, sig in signals.items():
        if ticker in full_detail_tickers:
            # Full detail — same as compact summary for held/interesting tickers
            td = {k: v for k, v in sig.items() if k != "enhanced_signals"}
            if "extra" in td:
                td["extra"] = {k: v for k, v in td["extra"].items()
                               if k in KEEP_EXTRA_FULL}
            t2["signals"][ticker] = td
            # Include timeframes
            tf_list = timeframes.get(ticker, [])
            if tf_list:
                t2["timeframes"][ticker] = [
                    {"horizon": tf["horizon"], "action": tf.get("action", "HOLD")}
                    if "error" not in tf else {"horizon": tf["horizon"], "error": tf["error"]}
                    for tf in tf_list
                ]
        elif ticker in medium_tickers:
            # Medium detail — vote detail string + timeframes, no full _votes
            td = {k: v for k, v in sig.items() if k != "enhanced_signals"}
            if "extra" in td:
                extra = {k: v for k, v in td["extra"].items() if k in KEEP_EXTRA_FULL}
                if "_votes" in extra:
                    votes = extra["_votes"]
                    buys = [s for s, v in votes.items() if v == "BUY"]
                    sells = [s for s, v in votes.items() if v == "SELL"]
                    parts = []
                    if buys:
                        parts.append("B:" + ",".join(buys))
                    if sells:
                        parts.append("S:" + ",".join(sells))
                    extra["_vote_detail"] = " | ".join(parts) if parts else "none"
                    del extra["_votes"]
                td["extra"] = extra
            t2["signals"][ticker] = td
            tf_list = timeframes.get(ticker, [])
            if tf_list:
                t2["timeframes"][ticker] = [
                    {"horizon": tf["horizon"], "action": tf.get("action", "HOLD")}
                    if "error" not in tf else {"horizon": tf["horizon"], "error": tf["error"]}
                    for tf in tf_list
                ]
        else:
            # Price-only one-liner
            t2["signals"][ticker] = {
                "action": sig.get("action", "HOLD"),
                "price_usd": sig.get("price_usd", 0),
            }

    # Include macro, accuracy, portfolio sections from full summary
    for key in ("macro", "signal_accuracy_1d", "cross_asset_leads",
                "avanza_instruments", "portfolio"):
        if key in summary:
            t2[key] = summary[key]

    _atomic_write_json(TIER2_FILE, t2)
