"""GoldDigger runner — main loop and Avanza execution layer.

Orchestrates the 30-second poll cycle, Playwright session management,
Avanza order execution, and Telegram notifications.

Usage:
    python -m portfolio.golddigger              # live mode (dry-run by default)
    python -m portfolio.golddigger --live       # live execution via Avanza
    python -m portfolio.golddigger --dry-run    # paper trade (default)
"""

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from portfolio.golddigger.config import GolddiggerConfig, DATA_DIR
from portfolio.golddigger.bot import GolddiggerBot

logger = logging.getLogger("portfolio.golddigger.runner")

# Crash protection
MAX_CONSECUTIVE_ERRORS = 5
BACKOFF_BASE = 10  # seconds
BACKOFF_MAX = 300  # 5 minutes


def _load_config() -> dict:
    """Load main config.json."""
    config_path = Path(__file__).resolve().parent.parent.parent / "config.json"
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def _send_telegram(msg: str, config: dict):
    """Send a Telegram notification."""
    try:
        import requests
        token = config["telegram"]["token"]
        chat_id = config["telegram"]["chat_id"]
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
            timeout=15,
        )
    except Exception as e:
        logger.warning("Telegram send failed: %s", e)


def _setup_playwright():
    """Initialize Playwright and load Avanza session.

    Returns (playwright_instance, browser, page) or (None, None, None) on failure.
    """
    try:
        from playwright.sync_api import sync_playwright
        storage_path = DATA_DIR / "avanza_storage_state.json"
        if not storage_path.exists():
            logger.error("No Avanza storage state at %s", storage_path)
            return None, None, None

        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(storage_state=str(storage_path))
        page = context.new_page()
        # Navigate to Avanza to establish session
        page.goto("https://www.avanza.se/min-ekonomi/oversikt.html", wait_until="domcontentloaded")
        logger.info("Playwright session loaded")
        return pw, browser, page
    except Exception as e:
        logger.error("Playwright setup failed: %s", e)
        return None, None, None


def _execute_order(page, action: dict, config: dict, account_id: str) -> bool:
    """Execute an order on Avanza via Playwright.

    Uses the same order placement as metals_avanza_helpers.
    Returns True if order was placed successfully.
    """
    try:
        sys.path.insert(0, str(DATA_DIR))
        from metals_avanza_helpers import place_order

        ob_id = action["orderbook_id"]
        side = action["action"]  # BUY or SELL
        price = action["price"]
        qty = action["quantity"]

        success, result = place_order(page, account_id, ob_id, side, price, qty)
        if success:
            logger.info("Order placed: %s %d @ %.2f (order_id: %s)",
                        side, qty, price, result.get("order_id", "?"))
            _send_telegram(
                f"*GOLDDIGGER {side}* {qty}x @ {price:.2f} SEK\n"
                f"Gold: ${action.get('gold_price', 0):.0f}\n"
                f"S={action.get('composite_s', 0):.2f}\n"
                f"_{action.get('reason', '')}_",
                config,
            )
            return True
        else:
            logger.error("Order FAILED: %s", result)
            _send_telegram(
                f"_GOLDDIGGER ORDER FAILED: {side} {qty}x @ {price:.2f}_\n"
                f"Error: {result.get('error', 'unknown')}",
                config,
            )
            return False
    except Exception as e:
        logger.error("Order execution error: %s", e)
        return False


def run(live: bool = False):
    """Main loop — runs until killed or market close."""
    config = _load_config()
    cfg = GolddiggerConfig.from_config(config)
    dry_run = not live

    bot = GolddiggerBot(cfg, dry_run=dry_run)

    mode = "LIVE" if live else "DRY-RUN"
    logger.info("GoldDigger starting in %s mode (poll: %ds)", mode, cfg.poll_seconds)

    # Setup Playwright for live mode
    pw, browser, page = None, None, None
    if live and cfg.bull_orderbook_id:
        pw, browser, page = _setup_playwright()
        if page:
            bot.set_page(page)
        else:
            logger.warning("No Playwright session — running without certificate prices")

    if cfg.telegram_alerts:
        _send_telegram(
            f"*GOLDDIGGER STARTED* ({mode})\n"
            f"Equity: {cfg.equity_sek:,.0f} SEK\n"
            f"Poll: {cfg.poll_seconds}s | Window: {cfg.window_n}\n"
            f"Entry: S >= {cfg.theta_in} | Exit: S <= {cfg.theta_out}\n"
            f"Stop: {cfg.stop_loss_pct*100:.0f}% | TP: {cfg.take_profit_pct*100:.0f}%",
            config,
        )

    consecutive_errors = 0

    try:
        while True:
            try:
                action = bot.step()

                if action:
                    if live and page:
                        _execute_order(page, action, config, cfg.avanza_account_id)
                    elif not live:
                        logger.info("[DRY-RUN] Would execute: %s %d @ %.2f — %s",
                                    action["action"], action["quantity"],
                                    action["price"], action.get("reason", ""))
                        if cfg.telegram_alerts:
                            _send_telegram(
                                f"*GOLDDIGGER {action['action']}* (dry-run)\n"
                                f"{action['quantity']}x @ {action['price']:.2f} SEK\n"
                                f"Gold: ${action.get('gold_price', 0):.0f} | "
                                f"S={action.get('composite_s', 0):.2f}\n"
                                f"_{action.get('reason', '')}_",
                                config,
                            )

                consecutive_errors = 0
                time.sleep(cfg.poll_seconds)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                consecutive_errors += 1
                logger.error("Poll error (%d/%d): %s",
                             consecutive_errors, MAX_CONSECUTIVE_ERRORS, e)
                traceback.print_exc()

                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    msg = f"_GOLDDIGGER HALTED: {consecutive_errors} consecutive errors_"
                    logger.error(msg)
                    if cfg.telegram_alerts:
                        _send_telegram(msg, config)
                    break

                backoff = min(BACKOFF_BASE * (2 ** (consecutive_errors - 1)), BACKOFF_MAX)
                logger.info("Backing off %ds before retry", backoff)
                time.sleep(backoff)

    except KeyboardInterrupt:
        logger.info("GoldDigger stopped by user")
    finally:
        # Save state
        bot.state.save(cfg.state_file)

        # Cleanup Playwright
        if browser:
            try:
                browser.close()
            except Exception:
                pass
        if pw:
            try:
                pw.stop()
            except Exception:
                pass

        if cfg.telegram_alerts:
            _send_telegram(
                f"*GOLDDIGGER STOPPED*\n"
                f"Equity: {bot.state.cash_sek:,.0f} SEK\n"
                f"Daily P&L: {bot.state.daily_pnl:,.0f} SEK\n"
                f"Total P&L: {bot.state.total_pnl:,.0f} SEK\n"
                f"Total trades: {bot.state.total_trades}",
                config,
            )

        logger.info("GoldDigger shutdown complete")
