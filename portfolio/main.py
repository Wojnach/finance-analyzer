#!/usr/bin/env python3
"""Portfolio Intelligence System — Simulated Trading on Binance Real-Time Data

This is the orchestrator module. All logic has been extracted to:
- shared_state.py — mutable globals, caching, rate limiters
- market_timing.py — DST-aware market hours, agent window
- fx_rates.py — USD/SEK exchange rate fetching
- indicators.py — compute_indicators, detect_regime, technical_signal
- data_collector.py — Binance/Alpaca/yfinance kline fetchers
- signal_engine.py — 29-signal voting system, generate_signal
- portfolio_mgr.py — portfolio state load/save/value
- reporting.py — agent_summary.json builder
- telegram_notifications.py — Telegram send/escape/alert
- digest.py — 4-hour digest builder
- agent_invocation.py — Layer 2 Claude Code subprocess
- logging_config.py — structured logging setup
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
DATA_DIR = BASE_DIR / "data"

logger = logging.getLogger("portfolio.loop")

# --- Re-exports for backwards compatibility ---
# External code (tests, trigger.py, etc.) that does `from portfolio.main import X`
# continues to work via these re-exports.

from portfolio.tickers import SYMBOLS, CRYPTO_SYMBOLS, STOCK_SYMBOLS, METALS_SYMBOLS  # noqa: E402, F401
from portfolio.api_utils import load_config as _load_config  # noqa: E402, F401
from portfolio.http_retry import fetch_with_retry  # noqa: E402, F401

# Shared state re-exports
from portfolio.shared_state import (  # noqa: E402, F401
    _tool_cache, _cached, _run_cycle_id, _current_market_state,
    _regime_cache, _regime_cache_cycle,
    _RateLimiter, _alpaca_limiter, _binance_limiter, _yfinance_limiter,
    _alpha_vantage_limiter,
    FUNDAMENTALS_TTL, FEAR_GREED_TTL, SENTIMENT_TTL, MINISTRAL_TTL,
    ML_SIGNAL_TTL, FUNDING_RATE_TTL, VOLUME_TTL, _RETRY_COOLDOWN,
)
import portfolio.shared_state as _ss

# Market timing re-exports
from portfolio.market_timing import (  # noqa: E402, F401
    _is_us_dst, _market_close_hour_utc, _is_agent_window,
    get_market_state, MARKET_OPEN_HOUR,
    INTERVAL_MARKET_OPEN, INTERVAL_MARKET_CLOSED, INTERVAL_WEEKEND,
)

# FX rates re-exports
from portfolio.fx_rates import fetch_usd_sek, _fx_cache  # noqa: E402, F401

# Indicators re-exports
from portfolio.indicators import (  # noqa: E402, F401
    compute_indicators, detect_regime, technical_signal,
)

# Data collector re-exports
from portfolio.data_collector import (  # noqa: E402, F401
    binance_klines, binance_fapi_klines, alpaca_klines, yfinance_klines,
    _fetch_klines, collect_timeframes,
    TIMEFRAMES, STOCK_TIMEFRAMES, ALPACA_INTERVAL_MAP,
    BINANCE_BASE, BINANCE_FAPI_BASE, ALPACA_BASE,
)

# Signal engine re-exports
from portfolio.signal_engine import (  # noqa: E402, F401
    generate_signal,
    MIN_VOTERS_CRYPTO, MIN_VOTERS_STOCK,
    _prev_sentiment, _prev_sentiment_loaded,
    _load_prev_sentiments, _get_prev_sentiment, _set_prev_sentiment,
    REGIME_WEIGHTS, _weighted_consensus, _confluence_score, _time_of_day_factor,
)

# Portfolio manager re-exports
from portfolio.portfolio_mgr import (  # noqa: E402, F401
    load_state, save_state, _atomic_write_json, portfolio_value,
    STATE_FILE, INITIAL_CASH_SEK,
)

# Reporting re-exports
from portfolio.reporting import (  # noqa: E402, F401
    write_agent_summary, _write_compact_summary, _cross_asset_signals,
    AGENT_SUMMARY_FILE, COMPACT_SUMMARY_FILE,
)

# Telegram re-exports
from portfolio.telegram_notifications import (  # noqa: E402, F401
    send_telegram, escape_markdown_v1, _maybe_send_alert,
    BOLD_STATE_FILE,
)

# Agent invocation re-exports
from portfolio.agent_invocation import (  # noqa: E402, F401
    invoke_agent, _log_trigger,
    INVOCATIONS_FILE, TIER_CONFIG,
)

# Digest re-exports
from portfolio.digest import _maybe_send_digest  # noqa: E402, F401

CONFIG_FILE = BASE_DIR / "config.json"


# --- Helpers ---

import re as _re

def _extract_triggered_tickers(reasons):
    """Parse ticker names from trigger reason strings.

    Examples:
        "MU consensus BUY (79%)" -> "MU"
        "BTC-USD moved 3.1% up" -> "BTC-USD"
        "ETH-USD flipped SELL->BUY (sustained)" -> "ETH-USD"
    """
    tickers = set()
    # Known ticker patterns: uppercase letters/digits optionally followed by -USD
    ticker_pat = _re.compile(r'^([A-Z][A-Z0-9]*(?:-[A-Z]+)?)\s+(?:consensus|moved|flipped)')
    for reason in reasons:
        m = ticker_pat.match(reason)
        if m:
            tickers.add(m.group(1))
    return tickers


# --- Main orchestrator ---


def run(force_report=False, active_symbols=None):
    _ss._run_cycle_id += 1

    config = _load_config()
    state = load_state()
    fx_rate = fetch_usd_sek()

    market_state, default_symbols, _ = get_market_state()
    _ss._current_market_state = market_state
    active = active_symbols or default_symbols

    skipped = set(SYMBOLS.keys()) - active
    skip_note = f" (skipped: {', '.join(sorted(skipped))})" if skipped else ""
    logger.info("USD/SEK: %.2f | market: %s%s", fx_rate, market_state, skip_note)

    signals_ok = 0
    signals_failed = 0
    signals = {}
    prices_usd = {}
    tf_data = {}

    for name, source in SYMBOLS.items():
        if name not in active:
            continue
        try:
            tfs = collect_timeframes(source)
            tf_data[name] = tfs

            now_entry = tfs[0][1] if tfs else None
            now_df = None
            if now_entry and "indicators" in now_entry:
                ind = now_entry["indicators"]
                now_df = now_entry.get("_df")
            else:
                now_df = _fetch_klines(source, interval="15m", limit=100)
                ind = compute_indicators(now_df)

            if ind is None:
                logger.info("%s: insufficient data, skipping", name)
                signals_failed += 1
                continue
            price = ind["close"]
            prices_usd[name] = price

            action, conf, extra = generate_signal(
                ind, ticker=name, config=config, timeframes=tfs, df=now_df
            )
            signals[name] = {
                "action": action,
                "confidence": conf,
                "indicators": ind,
                "extra": extra,
            }

            extra_str = ""
            if extra:
                parts = []
                if "fear_greed" in extra:
                    parts.append(f"F&G:{extra['fear_greed']}")
                if "sentiment" in extra:
                    parts.append(f"News:{extra['sentiment']}")
                if "ministral_action" in extra:
                    parts.append(f"8B:{extra['ministral_action']}")
                if "ml_action" in extra:
                    parts.append(f"ML:{extra['ml_action']}")
                if "funding_action" in extra:
                    parts.append(f"FR:{extra['funding_rate']}%")
                if "volume_action" in extra and extra["volume_action"] != "HOLD":
                    parts.append(f"Vol:{extra['volume_ratio']}x")
                if parts:
                    extra_str = f" | {' '.join(parts)}"
            enh_parts = []
            for esig in ("trend", "momentum", "volume_flow", "volatility_sig",
                         "candlestick", "structure", "fibonacci", "smart_money",
                         "oscillators", "heikin_ashi", "mean_reversion", "calendar",
                         "macro_regime", "momentum_factors"):
                ea = extra.get(f"{esig}_action", "HOLD")
                if ea != "HOLD":
                    enh_parts.append(f"{esig[:4].title()}:{ea[0]}")
            enh_str = f" | Enh: {' '.join(enh_parts)}" if enh_parts else ""

            logger.info(
                "%s: $%s | RSI %.0f | MACD %+.1f%s%s | %s (%.0f%%)",
                name, "{:,.2f}".format(price), ind['rsi'], ind['macd_hist'], extra_str, enh_str, action, conf * 100
            )
            signals_ok += 1

            for label, entry in tfs[1:]:
                if "error" in entry:
                    logger.warning("%s: %s", label, entry['error'])
                else:
                    ei = entry["indicators"]
                    logger.info(
                        "%s: %s %.0f%% | RSI %.0f | MACD %+.1f",
                        label, entry['action'], entry['confidence'] * 100, ei['rsi'], ei['macd_hist']
                    )

        except Exception as e:
            signals_failed += 1
            logger.error("%s: %s", name, e)

    total = portfolio_value(state, prices_usd, fx_rate)
    pnl_pct = ((total - state["initial_value_sek"]) / state["initial_value_sek"]) * 100
    logger.info("Portfolio: %s SEK (%+.2f%%) | Cash: %s SEK", "{:,.0f}".format(total), pnl_pct, "{:,.0f}".format(state['cash_sek']))

    if not STATE_FILE.exists():
        save_state(state)

    # Smart trigger
    from portfolio.trigger import check_triggers

    fear_greeds = {}
    sentiments = {}
    for name, sig in signals.items():
        extra = sig.get("extra", {})
        if "fear_greed" in extra:
            fear_greeds[name] = {
                "value": extra["fear_greed"],
                "classification": extra.get("fear_greed_class", ""),
            }
        if "sentiment" in extra:
            sentiments[name] = extra["sentiment"]

    triggered, reasons = check_triggers(signals, prices_usd, fear_greeds, sentiments)

    if triggered or force_report:
        reasons_list = reasons if reasons else ["startup"]
        summary = write_agent_summary(signals, prices_usd, fx_rate, state, tf_data, reasons_list)
        logger.info("Trigger: %s", ', '.join(reasons_list))

        # Classify tier and write tier-specific context
        from portfolio.trigger import classify_tier, update_tier_state
        from portfolio.reporting import write_tiered_summary
        tier = classify_tier(reasons_list)
        triggered_tickers = _extract_triggered_tickers(reasons_list)
        write_tiered_summary(summary, tier, triggered_tickers)
        update_tier_state(tier)
        logger.info("Tier: T%d (%s)", tier, TIER_CONFIG.get(tier, {}).get('label', 'UNKNOWN'))

        try:
            from portfolio.outcome_tracker import log_signal_snapshot
            log_signal_snapshot(signals, prices_usd, fx_rate, reasons_list)
        except Exception as e:
            logger.warning("signal logging failed: %s", e)

        layer2_cfg = config.get("layer2", {})
        if os.environ.get("NO_TELEGRAM"):
            logger.info("[NO_TELEGRAM] Skipping agent invocation")
            _log_trigger(reasons_list, "skipped_test", tier=tier)
        elif layer2_cfg.get("enabled", True):
            if _is_agent_window():
                result = invoke_agent(reasons_list, tier=tier)
                _log_trigger(reasons_list, "invoked" if result else "skipped_busy", tier=tier)
            else:
                logger.info("Layer 2: outside market window, skipping")
                _log_trigger(reasons_list, "skipped_offhours", tier=tier)
        else:
            logger.info("Layer 2 disabled — skipping agent invocation")
            _maybe_send_alert(
                config, signals, prices_usd, fx_rate, state, reasons_list, tf_data
            )
            _log_trigger(reasons_list, "alert_only", tier=tier)
    else:
        write_agent_summary(signals, prices_usd, fx_rate, state, tf_data)
        logger.info("No trigger.")

    # Big Bet detection
    bigbet_cfg = config.get("bigbet", {})
    if bigbet_cfg.get("enabled", False):
        try:
            from portfolio.bigbet import check_bigbet
            check_bigbet(signals, prices_usd, fx_rate, tf_data, config)
        except Exception as e:
            logger.warning("Big Bet check failed: %s", e)

    # ISKBETS monitoring
    try:
        from portfolio.iskbets import check_iskbets
        check_iskbets(signals, prices_usd, fx_rate, tf_data, config)
    except Exception as e:
        logger.warning("ISKBETS check failed: %s", e)

    # Avanza pending order confirmations
    try:
        from portfolio.avanza_orders import check_pending_orders
        check_pending_orders(config)
    except Exception as e:
        logger.warning("Avanza order check failed: %s", e)

    # Periodic trade reflection
    try:
        from portfolio.reflection import maybe_reflect
        maybe_reflect(config)
    except Exception as e:
        logger.warning("reflection check failed: %s", e)

    # Health update
    try:
        from portfolio.health import update_health
        trigger_reason = reasons[0] if (triggered or force_report) and reasons else None
        update_health(
            cycle_count=_ss._run_cycle_id,
            signals_ok=signals_ok,
            signals_failed=signals_failed,
            last_trigger_reason=trigger_reason,
        )
    except Exception as e:
        logger.warning("health update failed: %s", e)


def _crash_alert(error_msg):
    """Save crash alert to message log (not sent to Telegram)."""
    try:
        config_path = Path(__file__).resolve().parent.parent / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        text = f"LOOP CRASH\n\n{error_msg[:3000]}"
        from portfolio.message_store import send_or_store
        send_or_store(text, config, category="error")
    except Exception:
        pass


def loop(interval=None):
    from portfolio.logging_config import setup_logging
    setup_logging()

    # Validate config on startup (fail fast if misconfigured)
    from portfolio.config_validator import validate_config_file
    validate_config_file()

    # Check if previous loop crashed (stale heartbeat)
    heartbeat_file = DATA_DIR / "heartbeat.txt"
    if heartbeat_file.exists():
        try:
            last_beat = datetime.fromisoformat(heartbeat_file.read_text().strip())
            age_seconds = (datetime.now(timezone.utc) - last_beat).total_seconds()
            if age_seconds > 300:  # 5 minutes — previous loop likely crashed
                age_min = int(age_seconds // 60)
                msg = f"_LOOP RESTARTED_ — previous heartbeat was {age_min}m ago. Possible crash."
                logger.warning(msg)
                try:
                    config = _load_config()
                    from portfolio.message_store import send_or_store
                    send_or_store(msg, config, category="error")
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Failed to check heartbeat staleness: %s", e)

    logger.info("Loop started")

    # Load Alpha Vantage fundamentals cache from disk
    try:
        from portfolio.alpha_vantage import load_persistent_cache
        load_persistent_cache()
    except Exception as e:
        logger.warning("Failed to load fundamentals cache: %s", e)

    config = _load_config()
    logger.info("Starting loop with market-aware scheduling. Ctrl+C to stop.")

    try:
        from portfolio.telegram_poller import TelegramPoller
        from portfolio.iskbets import handle_command
        poller = TelegramPoller(config, on_command=handle_command)
        poller.start()
        logger.info("ISKBETS Telegram poller started")
    except Exception as e:
        logger.warning("ISKBETS poller failed to start: %s", e)

    try:
        run(force_report=True)
        _maybe_send_digest(config)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        import traceback
        _crash_alert(traceback.format_exc())
        logger.error("in initial run: %s", e)
        time.sleep(10)

    last_state = None
    while True:
        market_state, active_symbols, sleep_interval = get_market_state()
        if interval:
            sleep_interval = interval
        if market_state != last_state:
            logger.info(
                "Schedule: %s — %d instruments, %ds interval",
                market_state, len(active_symbols), sleep_interval
            )
            last_state = market_state
        time.sleep(sleep_interval)
        try:
            run(force_report=False, active_symbols=active_symbols)
            _maybe_send_digest(config)
            # Alpha Vantage fundamentals batch refresh (off-hours only)
            try:
                from portfolio.alpha_vantage import should_batch_refresh, refresh_fundamentals_batch
                if should_batch_refresh(config):
                    refresh_fundamentals_batch(config)
            except Exception as e_av:
                logger.warning("Alpha Vantage refresh failed: %s", e_av)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            import traceback
            _crash_alert(traceback.format_exc())
            logger.error("in run: %s", e)
            try:
                from portfolio.health import update_health
                update_health(cycle_count=_ss._run_cycle_id, signals_ok=0, signals_failed=0,
                              error=str(e))
            except Exception:
                pass
            time.sleep(10)
        try:
            (DATA_DIR / "heartbeat.txt").write_text(datetime.now(timezone.utc).isoformat())
        except Exception:
            pass


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--check-outcomes" in args:
        print("=== Outcome Backfill ===")
        from portfolio.outcome_tracker import backfill_outcomes
        updated = backfill_outcomes()
        print(f"Updated {updated} entries")
    elif "--accuracy" in args:
        from portfolio.accuracy_stats import print_accuracy_report
        print_accuracy_report()
    elif "--retrain" in args:
        print("=== ML Retraining ===")
        print("Refreshing data from Binance API...")
        from portfolio.data_refresh import refresh_all
        refresh_all(days=365)
        print("\nTraining model...")
        from portfolio.ml_trainer import load_data, train_final
        data = load_data()
        feature_cols = [c for c in data.columns if c not in ("target", "month")]
        print(f"Dataset: {len(data):,} rows, {len(feature_cols)} features")
        train_final(data, feature_cols)
        print("Done.")
    elif "--analyze" in args:
        idx = args.index("--analyze")
        if idx + 1 >= len(args):
            print("Usage: --analyze TICKER (e.g. --analyze ETH-USD)")
            sys.exit(1)
        ticker = args[idx + 1].upper()
        from portfolio.analyze import run_analysis
        run_analysis(ticker)
    elif "--watch" in args:
        idx = args.index("--watch")
        pos_args = args[idx + 1:]
        if not pos_args:
            print("Usage: --watch TICKER:ENTRY [TICKER:ENTRY ...]")
            print("Example: --watch BTC:66500 ETH:1920 MSTR:125")
            sys.exit(1)
        from portfolio.analyze import watch_positions
        watch_positions(pos_args)
    elif "--avanza-status" in args:
        from portfolio.avanza_client import get_positions, get_portfolio_value
        positions = get_positions()
        value = get_portfolio_value()
        print(f"Portfolio value: {value:,.0f} SEK")
        if positions:
            for p in positions:
                print(f"  {p}")
    elif "--loop" in args:
        idx = args.index("--loop")
        override = int(args[idx + 1]) if idx + 1 < len(args) else None
        loop(interval=override)
    else:
        run(force_report="--report" in args)
