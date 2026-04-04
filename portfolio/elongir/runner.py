"""Elongir runner -- main loop and Telegram notifications.

Orchestrates the poll cycle, crash recovery, and hourly Telegram reports.

Usage:
    python -m portfolio.elongir              # run main loop
    python -m portfolio.elongir --once       # single poll cycle
"""

import logging
import os
import time
import traceback
from pathlib import Path

from portfolio.elongir.bot import ElongirBot
from portfolio.elongir.config import ElongirConfig
from portfolio.elongir.state import (
    effective_leverage,
    sell_price,
    warrant_price_sek,
)
from portfolio.file_utils import load_json
from portfolio.loop_contract import BotCycleReport, ViolationTracker, verify_and_act, verify_bot_contract
from portfolio.process_lock import acquire_lock_file, release_lock_file

logger = logging.getLogger("portfolio.elongir.runner")

# Crash protection
MAX_CONSECUTIVE_ERRORS = 5
BACKOFF_BASE = 10  # seconds
BACKOFF_MAX = 300  # 5 minutes

_singleton_lock_fh = None


def _load_config() -> dict:
    """Load config.json with worktree-aware fallback order."""
    worktree_root = Path(__file__).resolve().parent.parent.parent
    candidate_paths = []

    env_path = os.environ.get("PORTFOLIO_CONFIG_PATH") or os.environ.get("ELONGIR_CONFIG_PATH")
    if env_path:
        candidate_paths.append(Path(env_path))

    candidate_paths.append(worktree_root / "config.json")

    # Clean worktrees usually don't carry the untracked live config
    parents = Path(__file__).resolve().parents
    if len(parents) >= 5 and parents[3].name == ".worktrees":
        candidate_paths.append(parents[4] / "config.json")

    for config_path in candidate_paths:
        if config_path.exists():
            if config_path.parent != worktree_root:
                logger.info("Using shared config from %s", config_path)
            data = load_json(config_path, default=None)
            if data is None:
                raise ValueError(f"Config corrupt or unreadable: {config_path}")
            return data

    searched = ", ".join(str(p) for p in candidate_paths)
    raise FileNotFoundError(f"No config.json found. Checked: {searched}")


def _send_telegram(msg: str, config: dict) -> bool:
    """Send a Telegram notification via send_or_store."""
    try:
        from portfolio.message_store import send_or_store
        return send_or_store(msg, config, category="elongir")
    except Exception as e:
        logger.warning("Telegram send failed: %s", e)
        return False


def _acquire_lock(lock_path: str) -> bool:
    """Acquire the Elongir singleton lock."""
    global _singleton_lock_fh
    if _singleton_lock_fh is not None:
        return True

    fh = acquire_lock_file(
        lock_path,
        owner="elongir",
        metadata={"script": "portfolio.elongir.runner"},
    )
    if fh is None:
        return False

    _singleton_lock_fh = fh
    return True


def _release_lock() -> None:
    """Release the singleton lock if held."""
    global _singleton_lock_fh
    release_lock_file(_singleton_lock_fh)
    _singleton_lock_fh = None


def _format_hourly_report(bot: ElongirBot, silver_usd: float, fx_rate: float) -> str:
    """Build an hourly Telegram report."""
    state = bot.state
    w_mid = warrant_price_sek(silver_usd, fx_rate, bot.cfg.financing_level)
    lev = effective_leverage(silver_usd, bot.cfg.financing_level)
    equity = state.equity(silver_usd, fx_rate)
    pnl_pct = (equity - bot.cfg.equity_sek) / bot.cfg.equity_sek * 100.0

    lines = [
        f"*ELONGIR 1H* Silver: ${silver_usd:.2f}",
        f"`Equity: {equity / 1000:.1f}K SEK ({pnl_pct:+.1f}%)`",
        f"`Warrant: {w_mid:.2f} SEK | Lev: {lev:.1f}x`",
        f"`Daily P&L: {state.daily_pnl:+,.0f} SEK | Trades: {state.daily_trades}`",
        f"`Total P&L: {state.total_pnl:+,.0f} SEK | W/L: {state.wins}/{state.losses}`",
        f"`Max DD: {state.max_drawdown_pct:.1f}% | Fees: {state.total_fees:,.0f} SEK`",
    ]
    if state.has_position():
        pos = state.position
        gain_pct = (silver_usd - pos.entry_silver_usd) / pos.entry_silver_usd * 100.0
        w_bid = sell_price(w_mid, bot.cfg.spread_pct)
        mtm_pnl = pos.quantity * w_bid - pos.cost_sek
        lines.append(
            f"`Open: {pos.quantity}x @ {pos.entry_warrant_sek:.2f} | "
            f"Silver {gain_pct:+.1f}% | MTM {mtm_pnl:+,.0f} SEK`"
        )
        if pos.trailing_active:
            lines.append(f"`Trailing: peak=${pos.trailing_peak_usd:.2f}`")
    else:
        lines.append(f"`State: {state.signal_state}`")

    return "\n".join(lines)


def _format_trade_message(action: dict, bot: ElongirBot) -> str:
    """Build a Telegram message for a trade execution."""
    state = bot.state
    equity = state.equity(action.get("silver_usd", 0), 10.5)  # approximate

    if action["type"] == "BUY":
        return (
            f"*ELONGIR BUY* ${action['silver_usd']:.2f}\n"
            f"`{action['quantity']}x @ {action['warrant_ask_sek']:.2f} SEK`\n"
            f"`Cost: {action['cost_sek']:,.0f} SEK | Fee: {action['fee_sek']:.0f}`\n"
            f"`Stop: ${action['stop_usd']:.2f} | TP: ${action['tp_usd']:.2f}`\n"
            f"`Lev: {action['leverage']:.1f}x | RSI: {action.get('rsi_5m', '?')}`\n"
            f"_{action['reason']}_"
        )
    else:  # SELL
        return (
            f"*ELONGIR SELL* ${action['silver_usd']:.2f}\n"
            f"`{action['quantity']}x @ {action['warrant_bid_sek']:.2f} SEK`\n"
            f"`P&L: {action['pnl_sek']:+,.0f} SEK ({action['silver_gain_pct']:+.1f}%)`\n"
            f"`Held: {action['hold_minutes']:.0f} min | Fee: {action['fee_sek']:.0f}`\n"
            f"`Equity: {equity / 1000:.1f}K | W/L: {state.wins}/{state.losses}`\n"
            f"_{action['reason']}_"
        )


def run(once: bool = False) -> None:
    """Main loop -- runs until killed or session close."""
    config = None
    cfg = None
    bot = None

    try:
        config = _load_config()
        cfg = ElongirConfig.from_config(config)

        if not _acquire_lock(cfg.lock_file):
            logger.warning("Duplicate Elongir instance detected; exiting.")
            return

        bot = ElongirBot(cfg)
        logger.info(
            "Elongir starting (poll: %ds, equity: %,.0f SEK)",
            cfg.poll_seconds, cfg.equity_sek,
        )

        # Startup notification
        _send_telegram(
            f"*ELONGIR STARTED*\n"
            f"`Equity: {cfg.equity_sek:,.0f} SEK`\n"
            f"`Poll: {cfg.poll_seconds}s | Stop: {cfg.stop_loss_pct}%`\n"
            f"`TP: {cfg.take_profit_pct}% | Trail: {cfg.trailing_start_pct}%/{cfg.trailing_distance_pct}%`\n"
            f"`Max hold: {cfg.max_hold_hours}h | Max trades/day: {cfg.max_daily_trades}`",
            config,
        )

        consecutive_errors = 0
        last_report_time = time.time()
        _contract_tracker = ViolationTracker(Path("data/elongir_contract_state.json"))
        _cycle_count = 0

        while True:
            try:
                _cycle_count += 1
                _report = BotCycleReport(cycle_id=_cycle_count, bot_name="elongir")
                _report.cycle_start = time.monotonic()

                # Collect data and run one step
                from portfolio.elongir.data_provider import collect_snapshot
                snapshot = collect_snapshot()
                _report.snapshot_collected = True
                action = bot.step(snapshot)
                _report.bot_step_completed = True

                # Send trade notification
                if action is not None:
                    msg = _format_trade_message(action, bot)
                    _send_telegram(msg, config)

                # Hourly report
                if time.time() - last_report_time >= cfg.telegram_report_interval:
                    if snapshot.is_complete():
                        report = _format_hourly_report(bot, snapshot.silver_usd, snapshot.fx_rate)
                        _send_telegram(report, config)
                    last_report_time = time.time()

                consecutive_errors = 0
                _report.consecutive_errors = consecutive_errors
                _report.max_consecutive_errors = MAX_CONSECUTIVE_ERRORS

                if once:
                    logger.info(
                        "Single-cycle complete (%s)",
                        "action" if action else "hold",
                    )
                    break

                _report.cycle_end = time.monotonic()
                try:
                    verify_and_act(_report, config or {}, tracker=_contract_tracker,
                                   verify_fn=verify_bot_contract, loop_name="elongir")
                except Exception:
                    pass
                time.sleep(cfg.poll_seconds)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                consecutive_errors += 1
                _report.consecutive_errors = consecutive_errors
                _report.max_consecutive_errors = MAX_CONSECUTIVE_ERRORS
                logger.error(
                    "Poll error (%d/%d): %s",
                    consecutive_errors, MAX_CONSECUTIVE_ERRORS, e,
                )
                traceback.print_exc()

                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    msg = f"_ELONGIR HALTED: {consecutive_errors} consecutive errors_"
                    logger.error(msg)
                    if config:
                        _send_telegram(msg, config)
                    break

                backoff = min(BACKOFF_BASE * (2 ** (consecutive_errors - 1)), BACKOFF_MAX)
                logger.info("Backing off %ds before retry", backoff)
                time.sleep(backoff)

    except KeyboardInterrupt:
        logger.info("Elongir stopped by user")
    except Exception as e:
        logger.error("Fatal error: %s", e)
        traceback.print_exc()
    finally:
        # Save state and send shutdown message
        if bot is not None and cfg is not None:
            bot.state.save(cfg.state_file)
            if config is not None:
                _send_telegram(
                    f"*ELONGIR STOPPED*\n"
                    f"`Cash: {bot.state.cash_sek:,.0f} SEK`\n"
                    f"`Total P&L: {bot.state.total_pnl:+,.0f} SEK`\n"
                    f"`Trades: {bot.state.total_trades} | W/L: {bot.state.wins}/{bot.state.losses}`",
                    config,
                )

        _release_lock()
        logger.info("Elongir shutdown complete")


def main() -> None:
    """Entry point called from __main__.py."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Elongir -- Simulated silver dip-trading bot"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single poll cycle and exit",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run(once=args.once)
