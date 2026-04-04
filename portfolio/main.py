#!/usr/bin/env python3
"""Portfolio Intelligence System — Simulated Trading on Binance Real-Time Data

This is the orchestrator module. All logic has been extracted to:
- shared_state.py — mutable globals, caching, rate limiters
- market_timing.py — DST-aware market hours, agent window
- fx_rates.py — USD/SEK exchange rate fetching
- indicators.py — compute_indicators, detect_regime, technical_signal
- data_collector.py — Binance/Alpaca/yfinance kline fetchers
- signal_engine.py — 30-signal voting system, generate_signal
- portfolio_mgr.py — portfolio state load/save/value
- reporting.py — agent_summary.json builder
- telegram_notifications.py — Telegram send/escape/alert
- digest.py — 4-hour digest builder
- daily_digest.py — morning daily digest (focus instruments + movers)
- message_throttle.py — analysis message rate limiting
- agent_invocation.py — Layer 2 Claude Code subprocess
- logging_config.py — structured logging setup
"""

import atexit
import logging
import os
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import load_json

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
DATA_DIR = BASE_DIR / "data"

logger = logging.getLogger("portfolio.loop")

# --- Singleton guard (same pattern as metals_loop.py) ---
try:
    import msvcrt
except ImportError:
    msvcrt = None

_SINGLETON_LOCK_FILE = str(DATA_DIR / "main_loop.singleton.lock")
_DUPLICATE_EXIT_CODE = 11
_singleton_lock_fh = None


def _acquire_singleton_lock():
    """Acquire single-instance lock for main loop (non-blocking)."""
    global _singleton_lock_fh
    if _singleton_lock_fh is not None:
        return True
    if msvcrt is None:
        return True

    os.makedirs(os.path.dirname(_SINGLETON_LOCK_FILE), exist_ok=True)
    fh = open(_SINGLETON_LOCK_FILE, "a+", encoding="utf-8")
    try:
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        fh.close()
        return False

    fh.seek(0)
    fh.truncate()
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    _singleton_lock_fh = fh
    return True


def _release_singleton_lock():
    """Release single-instance lock if held."""
    global _singleton_lock_fh
    if _singleton_lock_fh is None:
        return
    try:
        if msvcrt is not None:
            _singleton_lock_fh.seek(0)
            msvcrt.locking(_singleton_lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
    finally:
        with suppress(Exception):
            _singleton_lock_fh.close()
        _singleton_lock_fh = None


# --- Re-exports for backwards compatibility ---
# External code (tests, trigger.py, etc.) that does `from portfolio.main import X`
# continues to work via these re-exports.

import portfolio.shared_state as _ss

# Agent invocation re-exports
from portfolio.agent_invocation import (  # noqa: E402, F401
    INVOCATIONS_FILE,
    TIER_CONFIG,
    _log_trigger,
    check_agent_completion,
    invoke_agent,
)
from portfolio.api_utils import load_config as _load_config  # noqa: E402, F401

# Data collector re-exports
from portfolio.data_collector import (  # noqa: E402, F401
    ALPACA_BASE,
    ALPACA_INTERVAL_MAP,
    BINANCE_BASE,
    BINANCE_FAPI_BASE,
    STOCK_TIMEFRAMES,
    TIMEFRAMES,
    _fetch_klines,
    alpaca_klines,
    binance_fapi_klines,
    binance_klines,
    collect_timeframes,
    yfinance_klines,
)

# Digest re-exports
from portfolio.digest import _maybe_send_digest  # noqa: E402, F401

# FX rates re-exports
from portfolio.fx_rates import _fx_cache, fetch_usd_sek  # noqa: E402, F401
from portfolio.http_retry import fetch_with_retry  # noqa: E402, F401

# Indicators re-exports
from portfolio.indicators import (  # noqa: E402, F401
    compute_indicators,
    detect_regime,
    technical_signal,
)

# Market timing re-exports
from portfolio.market_timing import (  # noqa: E402, F401
    INTERVAL_MARKET_CLOSED,
    INTERVAL_MARKET_OPEN,
    INTERVAL_WEEKEND,
    MARKET_OPEN_HOUR,
    _is_agent_window,
    _is_us_dst,
    _market_close_hour_utc,
    get_market_state,
)

# Portfolio manager re-exports
from portfolio.portfolio_mgr import (  # noqa: E402, F401
    INITIAL_CASH_SEK,
    STATE_FILE,
    _atomic_write_json,
    load_state,
    portfolio_value,
    save_state,
)

# Reporting re-exports
from portfolio.reporting import (  # noqa: E402, F401
    AGENT_SUMMARY_FILE,
    COMPACT_SUMMARY_FILE,
    _cross_asset_signals,
    _write_compact_summary,
    write_agent_summary,
)

# Shared state re-exports
from portfolio.shared_state import (  # noqa: E402, F401
    _RETRY_COOLDOWN,
    FEAR_GREED_TTL,
    FUNDAMENTALS_TTL,
    FUNDING_RATE_TTL,
    MINISTRAL_TTL,
    ML_SIGNAL_TTL,
    SENTIMENT_TTL,
    VOLUME_TTL,
    _alpaca_limiter,
    _alpha_vantage_limiter,
    _binance_limiter,
    _cached,
    _current_market_state,
    _RateLimiter,
    _regime_cache,
    _regime_cache_cycle,
    _run_cycle_id,
    _tool_cache,
    _yfinance_limiter,
)

# Signal engine re-exports
from portfolio.signal_engine import (  # noqa: E402, F401
    MIN_VOTERS_CRYPTO,
    MIN_VOTERS_STOCK,
    REGIME_WEIGHTS,
    _confluence_score,
    _get_prev_sentiment,
    _load_prev_sentiments,
    _prev_sentiment,
    _prev_sentiment_loaded,
    _set_prev_sentiment,
    _time_of_day_factor,
    _weighted_consensus,
    generate_signal,
)

# Telegram re-exports
from portfolio.telegram_notifications import (  # noqa: E402, F401
    BOLD_STATE_FILE,
    _maybe_send_alert,
    escape_markdown_v1,
    send_telegram,
)
from portfolio.tickers import CRYPTO_SYMBOLS, METALS_SYMBOLS, STOCK_SYMBOLS, SYMBOLS  # noqa: E402, F401

CONFIG_FILE = BASE_DIR / "config.json"


# --- Helpers ---

import re as _re

_TICKER_PAT = _re.compile(r'^([A-Z][A-Z0-9]*(?:-[A-Z]+)?)\s+(?:consensus|moved|flipped)')


def _extract_triggered_tickers(reasons):
    """Parse ticker names from trigger reason strings.

    Examples:
        "MU consensus BUY (79%)" -> "MU"
        "BTC-USD moved 3.1% up" -> "BTC-USD"
        "ETH-USD flipped SELL->BUY (sustained)" -> "ETH-USD"
    """
    tickers = set()
    for reason in reasons:
        m = _TICKER_PAT.match(reason)
        if m:
            tickers.add(m.group(1))
    return tickers


def _run_post_cycle(config, report=None):
    """Post-cycle housekeeping: digest, daily digest, message throttle flush, AV refresh.

    Args:
        config: Application config dict.
        report: Optional CycleReport to track task success/failure.
    """
    def _track(name, func, *args):
        """Run a post-cycle task, tracking success on report if provided."""
        try:
            func(*args)
            if report is not None:
                report.post_cycle_results[name] = True
        except Exception as e:
            logger.warning("%s failed: %s", name, e)
            if report is not None:
                report.post_cycle_results[name] = False
                report.errors.append((name, str(e)))

    _maybe_send_digest(config)
    # Market health refresh (hourly via internal cache, self-checking)
    try:
        from portfolio.market_health import maybe_refresh_market_health
        _track("market_health", maybe_refresh_market_health)
    except Exception as e_mh:
        logger.warning("market health import failed: %s", e_mh)
    try:
        from portfolio.daily_digest import maybe_send_daily_digest
        _track("daily_digest", maybe_send_daily_digest, config)
    except Exception as e_dd:
        logger.warning("daily digest import failed: %s", e_dd)
    try:
        from portfolio.message_throttle import flush_and_send
        _track("message_throttle", flush_and_send, config)
    except Exception as e_mt:
        logger.warning("message throttle import failed: %s", e_mt)
    try:
        from portfolio.alpha_vantage import refresh_fundamentals_batch, should_batch_refresh
        if should_batch_refresh(config):
            _track("alpha_vantage", refresh_fundamentals_batch, config)
    except Exception as e_av:
        logger.warning("Alpha Vantage import failed: %s", e_av)
    try:
        from portfolio.local_llm_report import maybe_export_local_llm_report
        export = maybe_export_local_llm_report(config=config)
        if export:
            logger.info(
                "local LLM report exported: %s (%sd window)",
                export["date"],
                export["days"],
            )
        if report is not None:
            report.post_cycle_results["local_llm_report"] = True
    except Exception as e_report:
        logger.warning("local LLM report export failed: %s", e_report)
        if report is not None:
            report.post_cycle_results["local_llm_report"] = False
    # Metals deep context precompute (every 4h, self-checking)
    try:
        from portfolio.metals_precompute import maybe_precompute_metals
        _track("metals_precompute", maybe_precompute_metals, config)
    except Exception as e_metals:
        logger.warning("Metals precompute import failed: %s", e_metals)
    # Oil deep context precompute (every 2h, self-checking)
    try:
        from portfolio.oil_precompute import maybe_precompute_oil
        _track("oil_precompute", maybe_precompute_oil, config)
    except Exception as e_oil:
        logger.warning("Oil precompute import failed: %s", e_oil)
    # Prune unbounded JSONL files to prevent disk exhaustion (BUG-59)
    try:
        from portfolio.file_utils import prune_jsonl
        for name in ("invocations.jsonl", "layer2_journal.jsonl", "telegram_messages.jsonl"):
            prune_jsonl(DATA_DIR / name, max_entries=5000)
        if report is not None:
            report.post_cycle_results["jsonl_prune"] = True
    except Exception as e_prune:
        logger.warning("JSONL prune failed: %s", e_prune)
        if report is not None:
            report.post_cycle_results["jsonl_prune"] = False
    # Fin command self-improvement: backfill outcomes + evolve lessons (daily)
    try:
        from portfolio.fin_evolve import maybe_evolve
        _track("fin_evolve", maybe_evolve, config)
    except Exception as e_evolve:
        logger.warning("Fin evolve import failed: %s", e_evolve)
    # Scheduled crypto analysis report (08:00, 13:00, 18:00 CET)
    try:
        from portfolio.crypto_scheduler import maybe_run_crypto_report
        _track("crypto_scheduler", maybe_run_crypto_report, config)
    except Exception as e_crypto:
        logger.warning("Crypto scheduler import failed: %s", e_crypto)
    # Signal postmortem (daily — uses accuracy cache, generates once per day)
    try:
        from portfolio.file_utils import load_json as _lj
        from portfolio.signal_postmortem import POSTMORTEM_FILE, generate_postmortem
        pm = _lj(POSTMORTEM_FILE)
        # Regenerate if missing or stale (>20 hours old)
        if not pm or (time.time() - pm.get("_epoch", 0)) > 72000:
            result = generate_postmortem()
            if result:
                result["_epoch"] = time.time()
                from portfolio.file_utils import atomic_write_json as _awj
                _awj(POSTMORTEM_FILE, result)
        if report is not None:
            report.post_cycle_results["signal_postmortem"] = True
    except Exception as e_pm:
        logger.warning("Signal postmortem failed: %s", e_pm)
        if report is not None:
            report.post_cycle_results["signal_postmortem"] = False


# --- Main orchestrator ---


def run(force_report=False, active_symbols=None):
    _ss._run_cycle_id += 1

    # Check if a previously-spawned agent has completed (BUG-39)
    try:
        completion = check_agent_completion()
        if completion:
            logger.info(
                "Agent completed: status=%s tier=%s duration=%.1fs",
                completion.get("status"), completion.get("tier"),
                completion.get("duration_s", 0),
            )
    except Exception as e:
        logger.warning("check_agent_completion failed: %s", e)

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

    _run_start = time.monotonic()

    # Loop contract: track what actually happens this cycle
    from portfolio.loop_contract import CycleReport
    report = CycleReport(cycle_id=_ss._run_cycle_id, active_tickers=set(active))
    report.cycle_start = _run_start

    # --- Fully parallel: data collection + signal generation per ticker ---
    # Each ticker: fetch timeframes, compute indicators, generate signals — all threaded.
    # Rate limiters, cache locks, and GPU gate are already thread-safe.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    active_items = [(name, source) for name, source in SYMBOLS.items() if name in active]

    def _process_ticker(name, source):
        """Fetch data + generate signals for one ticker. Fully thread-safe."""
        try:
            t0 = time.monotonic()
            tfs = collect_timeframes(source)
            tf_elapsed = time.monotonic() - t0

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
                return name, None

            price = ind["close"]

            sig_start = time.monotonic()
            action, conf, extra = generate_signal(
                ind, ticker=name, config=config, timeframes=tfs, df=now_df
            )
            sig_elapsed = time.monotonic() - sig_start
            total_elapsed = time.monotonic() - t0
            logger.info(
                "%s: timing: tf=%.1fs sig=%.1fs total=%.1fs",
                name, tf_elapsed, sig_elapsed, total_elapsed,
            )

            extra_str = ""
            if extra:
                parts = []
                if extra.get("_gpu_signals_skipped"):
                    parts.append("GPU:skip")
                if "fear_greed" in extra:
                    parts.append(f"F&G:{extra['fear_greed']}")
                if "sentiment" in extra:
                    parts.append(f"News:{extra['sentiment']}")
                if "ministral_action" in extra:
                    parts.append(f"8B:{extra['ministral_action']}")
                if "ml_action" in extra:
                    parts.append(f"ML:{extra['ml_action']}")
                if "volume_action" in extra and extra["volume_action"] != "HOLD":
                    parts.append(f"Vol:{extra['volume_ratio']}x")
                if parts:
                    extra_str = f" | {' '.join(parts)}"
            enh_parts = []
            for esig in ("trend", "momentum", "volume_flow", "volatility_sig",
                         "candlestick", "structure", "fibonacci", "smart_money",
                         "oscillators", "heikin_ashi", "mean_reversion", "calendar",
                         "macro_regime", "momentum_factors", "futures_flow"):
                ea = extra.get(f"{esig}_action", "HOLD")
                if ea != "HOLD":
                    enh_parts.append(f"{esig[:4].title()}:{ea[0]}")
            enh_str = f" | Enh: {' '.join(enh_parts)}" if enh_parts else ""

            logger.info(
                "%s: $%s | RSI %.0f | MACD %+.1f%s%s | %s (%.0f%%)",
                name, f"{price:,.2f}", ind['rsi'], ind['macd_hist'], extra_str, enh_str, action, conf * 100
            )

            for label, entry in tfs[1:]:
                if "error" in entry:
                    logger.warning("%s: %s", label, entry['error'])
                else:
                    ei = entry["indicators"]
                    logger.info(
                        "%s: %s %.0f%% | RSI %.0f | MACD %+.1f",
                        label, entry['action'], entry['confidence'] * 100, ei['rsi'], ei['macd_hist']
                    )

            return name, {
                "tfs": tfs, "ind": ind, "now_df": now_df, "price": price,
                "action": action, "confidence": conf, "extra": extra,
            }
        except Exception as e:
            logger.error("%s: %s", name, e, exc_info=True)
            return name, None

    max_workers = min(len(active_items), 8)

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ticker") as pool:
        futures = {
            pool.submit(_process_ticker, name, source): name
            for name, source in active_items
        }
        for future in as_completed(futures):
            name, result = future.result()
            if result is not None:
                tf_data[name] = result["tfs"]
                prices_usd[name] = result["price"]
                signals[name] = {
                    "action": result["action"],
                    "confidence": result["confidence"],
                    "indicators": result["ind"],
                    "extra": result["extra"],
                }
                signals_ok += 1
            else:
                signals_failed += 1

    # --- Post-cycle LLM batch flush ---
    # Ministral/Qwen3 cache misses were enqueued during parallel ticker processing.
    # Now flush them sequentially, grouped by model (one swap max).
    try:
        from portfolio.llm_batch import flush_llm_batch
        from portfolio.shared_state import MINISTRAL_TTL, _update_cache
        batch_results = flush_llm_batch()
        for cache_key, result in batch_results.items():
            _update_cache(cache_key, result, ttl=MINISTRAL_TTL)
        report.llm_batch_flushed = True
    except Exception as e_batch:
        logger.warning("LLM batch flush failed: %s", e_batch)
        report.errors.append(("llm_batch_flush", str(e_batch)))

    _run_elapsed = time.monotonic() - _run_start
    logger.info(
        "Signal loop done: %d OK, %d failed in %.1fs (%.1fs/ticker avg)",
        signals_ok, signals_failed, _run_elapsed,
        _run_elapsed / max(signals_ok + signals_failed, 1),
    )

    # BUG-85: Flush batched sentiment state to disk once per cycle (not per-ticker)
    try:
        from portfolio.signal_engine import flush_sentiment_state
        flush_sentiment_state()
    except Exception:
        logger.warning("Failed to flush sentiment state", exc_info=True)

    # --- Cycle failure alert via Telegram ---
    # Collect per-ticker signal failures from this cycle
    _cycle_signal_failures = {}
    for _tk, _sig in signals.items():
        _sf = _sig.get("extra", {}).get("_signal_failures", [])
        if _sf:
            _cycle_signal_failures[_tk] = _sf

    if signals_failed > 0 or _cycle_signal_failures:
        _parts = []
        if signals_failed > 0:
            _parts.append(f"{signals_failed} ticker(s) failed entirely")
        if _cycle_signal_failures:
            _sf_total = sum(len(v) for v in _cycle_signal_failures.values())
            _sf_tickers = ", ".join(
                f"{tk}({len(sigs)})" for tk, sigs in _cycle_signal_failures.items()
            )
            _parts.append(f"{_sf_total} signal failures: {_sf_tickers}")
        _fail_msg = f"*LOOP ERRORS* ({int(_run_elapsed)}s cycle)\n" + "\n".join(_parts)
        try:
            from portfolio.message_store import send_or_store
            send_or_store(_fail_msg, config, category="error")
        except Exception as _e:
            logger.warning("Failed to send cycle error alert: %s", _e)

    total = portfolio_value(state, prices_usd, fx_rate)
    # BUG-103: Guard against zero/missing initial_value_sek to prevent ZeroDivisionError
    initial_val = state.get("initial_value_sek") or INITIAL_CASH_SEK
    pnl_pct = ((total - initial_val) / initial_val) * 100
    logger.info("Portfolio: %s SEK (%+.2f%%) | Cash: %s SEK", f"{total:,.0f}", pnl_pct, "{:,.0f}".format(state['cash_sek']))

    if not STATE_FILE.exists():
        save_state(state)

    # Log hourly price snapshot for cumulative tracking
    try:
        from portfolio.cumulative_tracker import maybe_log_hourly_snapshot
        maybe_log_hourly_snapshot(prices_usd)
    except Exception as e:
        logger.warning("hourly snapshot failed: %s", e)

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
        report.summary_written = True
        logger.info("Trigger: %s", ', '.join(reasons_list))

        # Classify tier and write tier-specific context
        from portfolio.reporting import write_tiered_summary
        from portfolio.trigger import classify_tier, update_tier_state
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
            logger.info("Layer 2 disabled — autonomous mode")
            from portfolio.autonomous import autonomous_decision
            autonomous_decision(
                config, signals, prices_usd, fx_rate, state,
                reasons_list, tf_data, tier, triggered_tickers,
            )
            _log_trigger(reasons_list, "autonomous", tier=tier)
    else:
        write_agent_summary(signals, prices_usd, fx_rate, state, tf_data)
        report.summary_written = True
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
        report.health_updated = True
    except Exception as e:
        logger.warning("health update failed: %s", e)
        report.errors.append(("health_update", str(e)))

    # Periodic safeguard checks (every 100 cycles ≈ 100 min)
    if _ss._run_cycle_id % 100 == 0 and _ss._run_cycle_id > 0:
        try:
            from portfolio.health import check_dead_signals, check_outcome_staleness
            outcome_status = check_outcome_staleness()
            if outcome_status["stale"]:
                age = outcome_status["newest_outcome_age_hours"]
                msg = (f"⚠️ SAFEGUARD: Outcome backfill stale! "
                       f"Newest outcome: {age:.0f}h ago. "
                       f"Entries missing outcomes: {outcome_status['entries_without_outcomes']}/50. "
                       f"Accuracy data is degrading.")
                logger.warning(msg)
                try:
                    from portfolio.telegram_notifications import send_telegram
                    send_telegram(msg)
                except Exception:
                    logger.debug("Failed to send outcome staleness alert", exc_info=True)

            dead_signals = check_dead_signals()
            if dead_signals:
                msg = (f"⚠️ SAFEGUARD: Dead signals (100% HOLD in last 20 entries): "
                       f"{', '.join(dead_signals)}. "
                       f"These signals contribute nothing to consensus.")
                logger.warning(msg)
                try:
                    from portfolio.telegram_notifications import send_telegram
                    send_telegram(msg)
                except Exception:
                    logger.debug("Failed to send dead signals alert", exc_info=True)
        except Exception as e:
            logger.debug("safeguard checks failed: %s", e)

    # Log portfolio equity snapshot for dashboard chart
    try:
        from portfolio.risk_management import log_portfolio_value
        log_portfolio_value()
    except Exception as e:
        logger.warning("equity snapshot failed: %s", e)

    # Finalize cycle report
    report.signals_ok = signals_ok
    report.signals_failed = signals_failed
    report.signals = signals
    report.cycle_end = time.monotonic()
    return report


_consecutive_crashes = 0
_MAX_CRASH_ALERTS = 5  # stop sending alerts after this many consecutive crashes
_MAX_CRASH_BACKOFF = 300  # max sleep between crashes (5 min)


def _crash_alert(error_msg):
    """Save crash alert to message log with crash-loop protection.

    After _MAX_CRASH_ALERTS consecutive crashes, stops sending alerts
    to prevent Telegram spam.  Sleep backoff is handled by the caller.
    """
    global _consecutive_crashes
    _consecutive_crashes += 1

    if _consecutive_crashes > _MAX_CRASH_ALERTS:
        # Silently log — don't spam Telegram
        logger.error(
            "Crash #%d (alerts suppressed after %d): %s",
            _consecutive_crashes, _MAX_CRASH_ALERTS, error_msg[:200],
        )
        return

    try:
        config_path = Path(__file__).resolve().parent.parent / "config.json"
        config = load_json(config_path, default={})
        text = f"LOOP CRASH #{_consecutive_crashes}\n\n{error_msg[:3000]}"
        if _consecutive_crashes == _MAX_CRASH_ALERTS:
            text += "\n\n_Further crash alerts suppressed until recovery._"
        from portfolio.message_store import send_or_store
        send_or_store(text, config, category="error")
    except Exception as e:
        logger.debug("Crash alert send failed: %s", e)


def _crash_sleep():
    """Exponential backoff sleep for consecutive crashes."""
    delay = min(10 * (2 ** (_consecutive_crashes - 1)), _MAX_CRASH_BACKOFF)
    logger.info("Crash backoff: sleeping %ds (crash #%d)", delay, _consecutive_crashes)
    time.sleep(delay)


def _reset_crash_counter():
    """Reset crash counter after a successful run cycle."""
    global _consecutive_crashes
    if _consecutive_crashes > 0:
        logger.info("Recovered after %d consecutive crashes", _consecutive_crashes)
        _consecutive_crashes = 0


def _sleep_for_next_cycle(previous_cycle_started, interval_s):
    """Sleep until the next scheduled cycle start.

    Anchors cadence to cycle start time so the loop period remains close to the
    configured interval instead of drifting by the work duration each cycle.
    """
    elapsed = time.monotonic() - previous_cycle_started
    remaining = interval_s - elapsed
    if remaining > 0:
        time.sleep(remaining)
        return
    logger.warning("Loop overran target cadence by %.1fs", abs(remaining))


def loop(interval=None):
    from portfolio.logging_config import setup_logging
    setup_logging()

    # Prevent duplicate loop instances
    if not _acquire_singleton_lock():
        logger.warning("Duplicate main loop instance detected; exiting.")
        sys.exit(_DUPLICATE_EXIT_CODE)
    atexit.register(_release_singleton_lock)

    # Validate config on startup (fail fast if misconfigured)
    from portfolio.config_validator import validate_config_file
    validate_config_file()

    # Check if previous loop crashed (stale heartbeat)
    heartbeat_file = DATA_DIR / "heartbeat.txt"
    if heartbeat_file.exists():
        try:
            last_beat = datetime.fromisoformat(heartbeat_file.read_text().strip())
            age_seconds = (datetime.now(UTC) - last_beat).total_seconds()
            if age_seconds > 300:  # 5 minutes — previous loop likely crashed
                age_min = int(age_seconds // 60)
                msg = f"_LOOP RESTARTED_ — previous heartbeat was {age_min}m ago. Possible crash."
                logger.warning(msg)
                try:
                    config = _load_config()
                    from portfolio.message_store import send_or_store
                    send_or_store(msg, config, category="error")
                except Exception as e2:
                    logger.debug("Restart notification failed: %s", e2)
        except Exception as e:
            logger.warning("Failed to check heartbeat staleness: %s", e)

    # Reset session start_time so uptime_seconds is accurate for this session
    from portfolio.health import reset_session_start
    reset_session_start()

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
        from portfolio.iskbets import handle_command
        from portfolio.telegram_poller import TelegramPoller
        poller = TelegramPoller(config, on_command=handle_command)
        poller.start()
        logger.info("ISKBETS Telegram poller started")
    except Exception as e:
        logger.warning("ISKBETS poller failed to start: %s", e)

    try:
        initial_report = run(force_report=True)
        _run_post_cycle(config, report=initial_report)
        _reset_crash_counter()
        try:
            (DATA_DIR / "heartbeat.txt").write_text(datetime.now(UTC).isoformat())
            if initial_report is not None:
                initial_report.heartbeat_updated = True
        except Exception as e:
            logger.debug("Heartbeat write after initial run failed: %s", e)
    except KeyboardInterrupt:
        logger.info("Loop interrupted during initial run, shutting down cleanly")
        return
    except Exception as e:
        import traceback
        _crash_alert(traceback.format_exc())
        logger.error("in initial run: %s", e)
        _crash_sleep()

    last_state = None
    last_cycle_started = time.monotonic()
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
        _sleep_for_next_cycle(last_cycle_started, sleep_interval)
        cycle_started = time.monotonic()
        try:
            report = run(force_report=False, active_symbols=active_symbols)
            _run_post_cycle(config, report=report)
            _reset_crash_counter()
        except KeyboardInterrupt:
            logger.info("Loop interrupted, shutting down cleanly")
            break
        except Exception as e:
            import traceback
            _crash_alert(traceback.format_exc())
            logger.error("in run: %s", e)
            try:
                from portfolio.health import update_health
                update_health(cycle_count=_ss._run_cycle_id, signals_ok=0, signals_failed=0,
                              error=str(e))
            except Exception as e2:
                logger.debug("Health update after crash failed: %s", e2)
            _crash_sleep()
            report = None
        last_cycle_started = cycle_started
        try:
            (DATA_DIR / "heartbeat.txt").write_text(datetime.now(UTC).isoformat())
            if report is not None:
                report.heartbeat_updated = True
        except Exception as e:
            logger.debug("Heartbeat write failed: %s", e)
        # Loop contract verification
        if report is not None:
            try:
                from portfolio.loop_contract import verify_and_act
                verify_and_act(report, config)
            except Exception as e_contract:
                logger.warning("Loop contract check failed: %s", e_contract)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--check-outcomes" in args:
        print("=== Outcome Backfill ===")
        from portfolio.outcome_tracker import backfill_outcomes
        updated = backfill_outcomes()
        print(f"Updated {updated} entries")
        # Also backfill forecast outcomes
        print("\n=== Forecast Outcome Backfill ===")
        try:
            from portfolio.forecast_accuracy import backfill_forecast_outcomes
            fc_updated = backfill_forecast_outcomes()
            print(f"Updated {fc_updated} forecast entries")
        except Exception as e:
            print(f"Forecast backfill failed: {e}")
        # Backfill Layer 2 decision outcomes
        print("\n=== Decision Outcome Backfill ===")
        try:
            from portfolio.decision_outcome_tracker import backfill_decision_outcomes
            dec_updated = backfill_decision_outcomes()
            print(f"Updated {dec_updated} decision outcome entries")
        except Exception as e:
            print(f"Decision outcome backfill failed: {e}")
    elif "--accuracy" in args:
        from portfolio.accuracy_stats import print_accuracy_report
        print_accuracy_report()
    elif "--forecast-accuracy" in args:
        from portfolio.forecast_accuracy import print_forecast_accuracy_report
        print_forecast_accuracy_report()
    elif "--local-llm-report" in args:
        from portfolio.local_llm_report import print_local_llm_report
        idx = args.index("--local-llm-report")
        days = int(args[idx + 1]) if idx + 1 < len(args) and not args[idx + 1].startswith("--") else 30
        print_local_llm_report(days=days)
    elif "--export-local-llm-report" in args:
        from portfolio.local_llm_report import HISTORY_FILE, LATEST_REPORT_FILE, export_local_llm_report
        idx = args.index("--export-local-llm-report")
        days = int(args[idx + 1]) if idx + 1 < len(args) and not args[idx + 1].startswith("--") else 30
        export = export_local_llm_report(days=days)
        print(f"Exported local LLM report for {export['date']} ({days}d window)")
        print(f"Latest: {LATEST_REPORT_FILE}")
        print(f"History: {HISTORY_FILE}")
    elif "--prophecy-review" in args:
        from portfolio.prophecy import print_prophecy_review
        print_prophecy_review()
    elif "--forecast-outcomes" in args:
        print("=== Forecast Outcome Backfill ===")
        from portfolio.forecast_accuracy import backfill_forecast_outcomes
        updated = backfill_forecast_outcomes()
        print(f"Updated {updated} forecast entries with actual outcomes")
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
    elif "--analyze-focus" in args:
        idx = args.index("--analyze-focus")
        raw = args[idx + 1] if idx + 1 < len(args) and not args[idx + 1].startswith("--") else ""
        tickers = [t.strip() for t in raw.split(",") if t.strip()] if raw else None
        from portfolio.focus_analysis import run_focus_analysis
        msg = run_focus_analysis(tickers=tickers)
        print(msg)
    elif "--watch" in args:
        idx = args.index("--watch")
        pos_args = args[idx + 1:]
        if not pos_args:
            print("Usage: --watch TICKER:ENTRY [TICKER:ENTRY ...]")
            print("Example: --watch BTC:66500 ETH:1920 AMD:150")
            sys.exit(1)
        from portfolio.analyze import watch_positions
        watch_positions(pos_args)
    elif "--price-targets" in args:
        from pathlib import Path as _Path
        _data = _Path("data")
        _summary = load_json(_data / "agent_summary.json", default={})
        _patient = load_json(_data / "portfolio_state.json", default={})
        _bold = load_json(_data / "portfolio_state_bold.json", default={})
        from portfolio.api_utils import load_config as _pt_load
        from portfolio.price_targets import compute_all_targets
        _pt_cfg = _pt_load().get("price_targets", {})
        results = compute_all_targets(_summary, {"patient": _patient, "bold": _bold}, _pt_cfg)
        for ticker, data in (results or {}).items():
            side = data["side"].upper()
            print(f"\n=== {ticker} {side} @ ${data['price_usd']:.2f} ({data['hours_remaining']:.1f}h left) ===")
            ext = data.get("extremes", {})
            label = "Running max" if data["side"] == "sell" else "Running min"
            print(f"{label}: p25=${ext.get('p25',0):.2f}  p50=${ext.get('p50',0):.2f}  p75=${ext.get('p75',0):.2f}  p90=${ext.get('p90',0):.2f}")
            for t in data.get("targets", [])[:5]:
                print(f"  {t['label']:<14} ${t['price']:.2f}  fill={t['fill_prob']:.0%}  EV={t['ev_sek']:+,.0f} SEK")
            rec = data.get("recommended")
            if rec:
                print(f"  >>> RECOMMENDED: ${rec['price']:.2f}  fill={rec['fill_prob']:.0%}  EV={rec['ev_sek']:+,.0f} SEK")
    elif "--avanza-status" in args:
        from portfolio.avanza_client import get_portfolio_value, get_positions
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
