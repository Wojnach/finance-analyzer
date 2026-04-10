"""Autonomous metals swing trader — rule-based warrant BUY/SELL on Avanza.

Integrates into metals_loop.py. Called every price cycle. Shares Playwright session.
No LLM dependency — decisions in <1s based on signal consensus + price rules.

Usage from metals_loop.py:
    from metals_swing_trader import SwingTrader
    trader = SwingTrader(page)
    trader.evaluate_and_execute(prices, signal_data)
"""

import datetime
import json
import logging
import time

import requests

# 2026-04-09 Stage 1 log migration (docs/LOG_MIGRATION_AUDIT_20260409.md):
# `_log()` is a thin shim that delegates to `logger.info()`. This module is
# ALWAYS imported from `metals_loop.py` in production, never run directly.
#
# Logger namespace (codex adversarial review finding HIGH round 3,
# 2026-04-09): the logger is named `metals_loop.swing_trader` so it is a
# CHILD of metals_loop's logger in Python's dotted hierarchy. Child loggers
# inherit the parent's effective level and propagate records up to the
# parent's handlers, which means metals_loop.py's `_install_stage1_logging()`
# (installed only under `if __name__ == "__main__":`) automatically
# configures level AND handler for this module too. Without the dotted
# parent name, `metals_swing_trader` would be a sibling of `metals_loop`
# and would NOT inherit level — its INFO records would be filtered by the
# default WARNING level, dropping `[SWING] Cash synced`, `[SWING] BUY`,
# etc. from metals_loop_out.txt.
#
# Library discipline: this module still does NOT install any handlers
# itself. Under pytest, `caplog.at_level(logging.INFO, logger="metals_loop")`
# on the parent captures records from this child via propagation. Under a
# custom embedding, the caller is free to attach their own handlers to
# either `metals_loop` or `metals_loop.swing_trader` directly.
logger = logging.getLogger("metals_loop.swing_trader")


def _has_ancestor_emitter(lg: logging.Logger, target_level: int) -> bool:
    """Walk the logger hierarchy for a handler that would emit target_level.

    Codex review v7 finding HIGH (2026-04-09): this is a local duplicate
    of metals_loop._has_ancestor_emitter. The previous version lazy-
    imported that helper inside `_log()` at call time, which in turn
    imported metals_loop — dragging in Playwright as a hard dependency
    and changing cwd, breaking every standalone/unit-test use of this
    module. Duplicating the 15-line helper here keeps _log() side-effect
    free and allows metals_swing_trader to be imported in isolation
    without pulling in the full metals_loop stack.

    See data/metals_loop.py:_has_ancestor_emitter for the upstream
    docstring — they stay in sync by code review, not by runtime link.
    """
    current = lg
    while current is not None:
        for h in current.handlers:
            if isinstance(h, logging.NullHandler):
                continue
            if h.level == logging.NOTSET or h.level <= target_level:
                return True
        if not current.propagate:
            break
        current = current.parent
    return False
from metals_swing_config import (
    ACCOUNT_ID,
    BUY_COOLDOWN_MINUTES,
    DECISIONS_LOG,
    DRY_RUN,
    EOD_EXIT_MINUTES_BEFORE,
    HARD_STOP_UNDERLYING_PCT,
    INITIAL_BUDGET_SEK,
    LOSS_ESCALATION,
    MACD_IMPROVING_CHECKS,
    MAX_CONCURRENT,
    MAX_HOLD_HOURS,
    MIN_ACCEPTABLE_LEVERAGE,
    MIN_BARRIER_DISTANCE_PCT,
    MIN_BUY_CONFIDENCE,
    MIN_BUY_TF_RATIO,
    MIN_BUY_VOTERS,
    MIN_SPREAD_PCT,
    MIN_TRADE_SEK,
    POSITION_SIZE_PCT,
    REGIME_CONFIRM_CHECKS,
    RSI_ENTRY_HIGH,
    RSI_ENTRY_LOW,
    SIGNAL_REVERSAL_EXIT,
    STATE_FILE,
    STOP_LOSS_UNDERLYING_PCT,
    STOP_LOSS_VALID_DAYS,
    TAKE_PROFIT_UNDERLYING_PCT,
    TARGET_LEVERAGE,
    TELEGRAM_SUMMARY_INTERVAL,
    TRADES_LOG,
    TRAILING_DISTANCE_PCT,
    TRAILING_START_PCT,
)
from metals_swing_config import (
    WARRANT_CATALOG as STATIC_WARRANT_CATALOG,
)

from portfolio.file_utils import atomic_append_jsonl, atomic_write_json, load_json

# Dynamic warrant catalog refresher — replaces the stale hardcoded list.
try:
    from metals_warrant_refresh import load_catalog_or_fetch
    _REFRESHER_AVAILABLE = True
except ImportError:
    _REFRESHER_AVAILABLE = False

from portfolio.avanza_control import (
    delete_order_live,
    delete_stop_loss,
    fetch_account_cash,
    fetch_page_positions,
    fetch_price,
    place_order,
    place_stop_loss,
)

SEND_PERIODIC_SUMMARY = False

# --- SHORT support (Fix 8, 2026-04-09) ---
# XAU (and occasionally XAG) has been printing SELL signals all day with no
# way to act on them because _check_exits was LONG-only. Fix 8 adds direction-
# aware exit math. Ships disabled by default — the user flips SHORT_ENABLED
# to True and populates SHORT_CANARY_WARRANTS with a single low-leverage
# warrant key to activate SHORT entries on a canary instrument only.
#
# When ready to open up the whole catalog, either drop the allowlist (keep
# SHORT_ENABLED=True) or move the list to config.json.
SHORT_ENABLED: bool = False
SHORT_CANARY_WARRANTS: frozenset[str] = frozenset()

# --- Reliability thresholds (added 2026-04-09) ---
# Fill verification — how long to wait before checking Avanza for a recent buy,
# and how long before rolling back an unfilled order.
FILL_VERIFY_MIN_AGE_S = 15   # first verification check after this many seconds
FILL_VERIFY_MAX_AGE_S = 90   # rollback if still unfilled after this many seconds

# Sell-failed cooldown — if _execute_sell() fails, suppress exit re-evaluation
# on the same position for this many seconds to avoid tight cascade loops.
SELL_FAILED_COOLDOWN_SECONDS = 300  # 5 minutes

# Reconciliation — how often to cross-check swing state positions against
# Avanza holdings, and how many consecutive failures before alerting.
RECON_THROTTLE_CYCLES = 3    # run every N cycles (after the first unconditional one)
RECON_FAILURE_STREAK_ALERT = 10  # alert on Telegram after N consecutive failures


def _log(msg):
    # 2026-04-09 Stage 1 shim: delegate to logger.info. The [SWING] tag is
    # preserved in the message body so existing grep-based operator workflows
    # (e.g. `grep SWING metals_loop_out.txt`) still work. The timestamp +
    # level prefix comes from metals_loop.py's Stage 1 handler (we're a
    # child logger under `metals_loop.swing_trader`, so we inherit level
    # and propagate to the parent's _LazyStdoutHandler).
    #
    # Codex review v6/v7 (2026-04-09): the visibility check walks
    # ancestor handlers via our LOCAL _has_ancestor_emitter (above) —
    # NOT a lazy import from metals_loop, because that would drag in
    # Playwright and mutate cwd as a side effect of a simple log call.
    # Under pytest caplog.at_level on "metals_loop", caplog adds a
    # capture handler to the parent; the walk finds it via propagation
    # and takes the logger path. Without setup, the fallback path
    # writes directly to stdout.
    if logger.isEnabledFor(logging.INFO) and _has_ancestor_emitter(logger, logging.INFO):
        logger.info(f"[SWING] {msg}")
    else:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [INFO] [SWING] {msg}", flush=True)


def _now_utc():
    return datetime.datetime.now(datetime.UTC)


def _cet_hour():
    """Get current CET hour as float. Uses zoneinfo (DST-safe)."""
    try:
        from zoneinfo import ZoneInfo
        now = datetime.datetime.now(ZoneInfo("Europe/Stockholm"))
        return now.hour + now.minute / 60
    except ImportError:
        now = datetime.datetime.now(datetime.UTC)
        return ((now.hour + 1) % 24) + now.minute / 60


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _load_state():
    """Load swing trader state from disk."""
    result = load_json(STATE_FILE)
    if result is not None:
        return result
    return _default_state()


def _default_state():
    return {
        "cash_sek": 0,
        "positions": {},
        "consecutive_losses": 0,
        "last_buy_ts": None,
        "total_trades": 0,
        "total_pnl_sek": 0,
        "session_trades": 0,
        "macd_history": {},
    }


def _save_state(state):
    try:
        atomic_write_json(STATE_FILE, state, indent=2, ensure_ascii=False)
    except Exception:
        # 2026-04-09 Stage 2: use logger.exception for free stack trace.
        # State save failure risks data loss (positions won't persist across
        # restarts), so this is ERROR-level, not WARNING.
        logger.exception("[SWING] _save_state: atomic_write_json failed — position state may drift")


def _delete_stop_loss(page, stop_id):
    """Delete a stop-loss order by ID."""
    success, _ = delete_stop_loss(page, ACCOUNT_ID, stop_id)
    return success


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

_tg_config = None

def _send_telegram(msg):
    global _tg_config
    if _tg_config is None:
        try:
            with open("config.json", encoding="utf-8") as f:
                cfg = json.load(f)
            _tg_config = {
                "token": cfg["telegram"]["token"],
                "chat_id": cfg["telegram"]["chat_id"],
            }
        except Exception:
            logger.warning("[SWING] _send_telegram: config.json telegram block read failed, telegram disabled for this process", exc_info=True)
            _tg_config = {}

    if not _tg_config.get("token"):
        _log("Telegram not configured")
        return

    # Check mute_all from config
    try:
        with open("config.json", encoding="utf-8") as f:
            _mute = json.load(f).get("telegram", {}).get("mute_all", False)
        if _mute:
            _log(f"[TG muted] {msg[:80]}")
            return
    except Exception:
        logger.debug("_send_telegram: mute_all check failed, proceeding with send", exc_info=True)

    try:
        requests.post(
            f"https://api.telegram.org/bot{_tg_config['token']}/sendMessage",
            json={"chat_id": _tg_config["chat_id"], "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception:
        # 2026-04-09 Stage 2: WARNING (not ERROR) because Telegram is a
        # notification channel — losing one message is a degradation, not
        # data loss. exc_info=True preserves the stack for debugging network
        # issues, auth drift, rate-limit responses, etc.
        logger.warning("[SWING] _send_telegram: requests.post to Telegram API failed", exc_info=True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_decision(decision):
    try:
        atomic_append_jsonl(DECISIONS_LOG, decision)
    except Exception:
        # 2026-04-09 Stage 2: ERROR-level — audit trail loss on write failure.
        # decisions_log.jsonl is how we reconstruct swing trader behavior
        # post-hoc; silent drops mean we lose visibility into trade reasoning.
        logger.exception("[SWING] _log_decision: atomic_append_jsonl failed — decision audit lost")


def _log_trade(trade):
    try:
        atomic_append_jsonl(TRADES_LOG, trade)
    except Exception:
        # 2026-04-09 Stage 2: ERROR-level — audit trail loss. Same rationale
        # as _log_decision but for executed trades (metals_trades.jsonl is the
        # source of truth for P&L attribution and accuracy backfill).
        logger.exception("[SWING] _log_trade: atomic_append_jsonl failed — trade audit lost")


# ---------------------------------------------------------------------------
# SwingTrader class
# ---------------------------------------------------------------------------

class SwingTrader:
    """Autonomous rule-based swing trader for metals warrants on Avanza."""

    # Class-level defaults for reliability tracking (Fix 1/2/4, 2026-04-09).
    # Declared at class level so tests that bypass __init__ (e.g. via a
    # factory helper) inherit safe defaults instead of crashing on
    # AttributeError. Instance __init__ overrides these.
    cash_sync_ok: bool = False
    cash_sync_was_ok: bool = False
    recon_failure_streak: int = 0
    reconciled_once: bool = False

    def __init__(self, page):
        self.page = page
        self.state = _load_state()
        self.check_count = 0

        # Track consecutive signal history per ticker (action, regime) — used
        # to reject single-check regime flips from trending-down → ranging BUY.
        # Keyed by ticker; value is a list of (action, regime) tuples.
        self.regime_history: dict[str, list[tuple[str, str]]] = {}

        # Reliability tracking (added 2026-04-09):
        #   cash_sync_ok      — False until _sync_cash() confirms a live fetch.
        #                       _check_entries refuses to place orders while False.
        #   cash_sync_was_ok  — previous state, for transition detection (Telegram).
        #   recon_failure_streak — consecutive failed position reconciliations.
        #   reconciled_once   — ensures reconciliation runs unconditionally on
        #                       the first evaluate_and_execute call (catches
        #                       startup phantoms).
        self.cash_sync_ok: bool = False
        self.cash_sync_was_ok: bool = False
        self.recon_failure_streak: int = 0
        self.reconciled_once: bool = False

        # Load dynamic warrant catalog (live refresh from Avanza). Falls back
        # to the static hardcoded catalog if the refresh fails entirely.
        self.warrant_catalog = self._load_warrant_catalog()

        # Sync cash from Avanza on init
        self._sync_cash()
        _log(f"SwingTrader init: cash={self.state['cash_sek']:.0f} SEK, "
             f"cash_sync_ok={self.cash_sync_ok}, "
             f"positions={len(self.state['positions'])}, "
             f"catalog={len(self.warrant_catalog)} warrants, "
             f"DRY_RUN={DRY_RUN}")

    def _load_warrant_catalog(self) -> dict:
        """Load dynamic warrant catalog, falling back to static config.

        The refresher fetches live data from Avanza and filters dead/knocked
        out warrants. If it returns an empty dict (total network failure),
        we fall back to the static config catalog so the trader can still
        operate (with whatever stale entries it contains).
        """
        if not _REFRESHER_AVAILABLE:
            _log("Refresher unavailable — using static catalog")
            return dict(STATIC_WARRANT_CATALOG)
        try:
            # Thread the metals_loop page object through so the refresher can
            # reuse the existing sync_playwright context (page.context.request),
            # instead of importing portfolio.avanza_session.api_post which tries
            # to open a second sync_playwright and crashes with "Sync API inside
            # the asyncio loop". See 2026-04-10 fix notes in metals_warrant_refresh.py.
            catalog = load_catalog_or_fetch(self.page)
        except Exception as e:  # noqa: BLE001
            _log(f"Catalog refresh raised {type(e).__name__}: {e} — using static")
            return dict(STATIC_WARRANT_CATALOG)
        if not catalog:
            _log("Refresher returned empty — using static catalog")
            return dict(STATIC_WARRANT_CATALOG)
        return catalog

    def _sync_cash(self):
        """Fetch real ISK buying power from Avanza and update state.

        Sets self.cash_sync_ok to True on success, False on failure. While
        False, _check_entries refuses to place new orders (Fix 1, 2026-04-09).

        Falls back to INITIAL_BUDGET_SEK when the API fails and no saved
        balance exists (e.g. first startup with empty state file).
        """
        acc = fetch_account_cash(self.page, ACCOUNT_ID)
        if acc and acc.get("buying_power") is not None:
            self.state["cash_sek"] = float(acc["buying_power"])
            _save_state(self.state)
            _log(f"Cash synced: {self.state['cash_sek']:.0f} SEK")
            self.cash_sync_ok = True
            if not self.cash_sync_was_ok:
                # Transition False → True: announce recovery (skip on first sync).
                if self.check_count > 0:
                    _send_telegram(
                        f"_SWING: cash sync recovered — {self.state['cash_sek']:.0f} SEK_"
                    )
                self.cash_sync_was_ok = True
            return

        # Failure path — degrade safely.
        self.cash_sync_ok = False
        if self.state["cash_sek"] == 0:
            self.state["cash_sek"] = float(INITIAL_BUDGET_SEK)
            _save_state(self.state)
            _log(f"Cash sync failed, using configured budget: {self.state['cash_sek']:.0f} SEK")
        else:
            _log(f"Cash sync failed, using saved: {self.state['cash_sek']:.0f} SEK")
        if self.cash_sync_was_ok:
            # Transition True → False: alert once per failure streak.
            _send_telegram(
                "⚠ _SWING: cash sync failed — entries paused until recovery_"
            )
            self.cash_sync_was_ok = False

    def evaluate_and_execute(self, prices, signal_data):
        """Main entry point — called every loop cycle during market hours.

        Args:
            prices: dict from metals_loop price fetch (keyed by position name)
            signal_data: dict from read_signal_data() with XAG-USD/XAU-USD signals
        """
        self.check_count += 1

        # Re-sync cash from Avanza every 30 checks (~30 min) to catch manual deposits
        # and recover from initial API failures.
        if self.check_count % 30 == 0:
            self._sync_cash()

        # Periodic catalog refresh (every 360 checks ≈ 6h with 60s loop)
        if self.check_count % 360 == 0:
            try:
                fresh = self._load_warrant_catalog()
                if fresh:
                    self.warrant_catalog = fresh
                    _log(f"Catalog auto-refreshed: {len(self.warrant_catalog)} warrants")
            except Exception:  # noqa: BLE001
                # 2026-04-09 Stage 2: WARNING — auto-refresh failure falls
                # back to cached catalog, so this is a degradation not a
                # blocker. Stack trace via exc_info helps diagnose Avanza
                # API shape drift (the most common cause of refresh break).
                logger.warning(
                    "[SWING] _evaluate: periodic catalog auto-refresh failed, continuing with cached catalog (%d warrants)",
                    len(self.warrant_catalog),
                    exc_info=True,
                )

        # Position reconciliation against Avanza holdings (Fix 2, 2026-04-09).
        # Runs UNCONDITIONALLY on the first call after init to catch startup
        # phantoms, then throttled to every RECON_THROTTLE_CYCLES cycles.
        # Must run BEFORE _check_exits so phantoms are pruned before the exit
        # loop has a chance to fire on them and trigger a cascade.
        if (not self.reconciled_once) or (self.check_count % RECON_THROTTLE_CYCLES == 0):
            self._reconcile_swing_positions()
            self.reconciled_once = True

        # Fill verification for recent BUYs (Fix 4, 2026-04-09). Runs before
        # _check_exits so unfilled positions don't enter the exit path.
        self._verify_recent_fills()

        # Update regime history BEFORE entry checks so the gate sees the latest state.
        self._update_regime_history(signal_data)

        # Check exits first (protect capital)
        self._check_exits(prices, signal_data)

        # Then check entries
        self._check_entries(prices, signal_data)

        # Track MACD history for improving-checks requirement
        self._update_macd_history(signal_data)

        # Periodic Telegram summary (when SEND_PERIODIC_SUMMARY enabled).
        # Restored after f6b491c accidentally dropped this call site.
        if SEND_PERIODIC_SUMMARY and self.check_count % TELEGRAM_SUMMARY_INTERVAL == 0:
            self._send_summary(signal_data)

    # -------------------------------------------------------------------
    # Reliability: reconciliation & fill verification (2026-04-09)
    # -------------------------------------------------------------------

    def _reconcile_swing_positions(self):
        """Cross-check swing state positions against Avanza holdings; prune phantoms.

        Fix 2 (2026-04-09). Before this method existed, swing state positions
        had no reconciliation path — a phantom position from a prior run could
        live forever in self.state["positions"] and cause a SELL cascade when
        _check_exits tried to close it (see 08:25 UTC incident).

        Failure semantics:
          - fetch_page_positions returns None     → transient; increment streak
          - fetch_page_positions returns {}       → valid "flat account"; prune all
          - fetch_page_positions returns {ob:...} → prune anything not in the set
        """
        held = fetch_page_positions(self.page, ACCOUNT_ID)
        if held is None:
            self.recon_failure_streak += 1
            if self.recon_failure_streak == RECON_FAILURE_STREAK_ALERT:
                _log(f"Reconciliation failed {self.recon_failure_streak} cycles in a row")
                _send_telegram(
                    f"⚠ _SWING: position reconciliation failing for "
                    f"{self.recon_failure_streak} cycles — session may be dead_"
                )
            return
        # Success — reset streak
        if self.recon_failure_streak >= RECON_FAILURE_STREAK_ALERT:
            _send_telegram("_SWING: position reconciliation recovered_")
        self.recon_failure_streak = 0

        if not self.state.get("positions"):
            return  # nothing to reconcile

        held_ob_ids = set(held.keys())
        phantoms = []
        for pos_id, pos in list(self.state["positions"].items()):
            if str(pos.get("ob_id", "")) not in held_ob_ids:
                phantoms.append((pos_id, pos))

        for pos_id, pos in phantoms:
            name = pos.get("warrant_name", pos_id)
            units = pos.get("units", 0)
            entry = pos.get("entry_price", 0)
            _log(f"PHANTOM CLEARED: {name} (was {units}u @ {entry}) — not on Avanza")
            _send_telegram(
                f"⚠ *SWING PHANTOM CLEARED* {name}\n"
                f"`{units}u @ {entry} — not held on Avanza, removing from state`"
            )
            # Cancel any orphan stop-loss attached to the phantom
            stop_id = pos.get("stop_order_id")
            if stop_id and stop_id != "DRY_RUN":
                try:
                    _delete_stop_loss(self.page, stop_id)
                except Exception:  # noqa: BLE001
                    # 2026-04-09 Stage 2: WARNING — best-effort cleanup of
                    # an orphan stop-loss. If this fails the next reconcile
                    # cycle will retry. Stack trace helps diagnose persistent
                    # failures (e.g. stop-loss API auth drift).
                    logger.warning(
                        "[SWING] _reconcile_swing_positions: phantom stop-loss cancel failed stop_id=%s",
                        stop_id,
                        exc_info=True,
                    )
            del self.state["positions"][pos_id]
        if phantoms:
            _save_state(self.state)

    def _verify_recent_fills(self):
        """Verify recently-placed BUY orders actually resulted in held positions.

        Fix 4 (2026-04-09). Before this method existed, a BUY order that
        returned `success=True` from place_order was trusted as filled even
        if Avanza never held the position (e.g. limit order expired
        unfilled). This caused the 06:17 UTC ghost trade.

        For each position with fill_verified == False:
          - age < FILL_VERIFY_MIN_AGE_S: skip, let Avanza settle
          - FILL_VERIFY_MIN_AGE_S <= age < FILL_VERIFY_MAX_AGE_S: check holdings.
            If found, mark fill_verified. If not found, keep waiting.
          - age >= FILL_VERIFY_MAX_AGE_S and still not found: roll back.
            Cancel the buy order, cancel the stop-loss, restore cash, delete
            the position from state, send Telegram alert.
        """
        unverified_ids = [
            pid for pid, p in self.state.get("positions", {}).items()
            if not p.get("fill_verified", False)
        ]
        if not unverified_ids:
            return

        held = None  # fetched lazily, only if at least one position is old enough
        for pos_id in unverified_ids:
            pos = self.state["positions"].get(pos_id)
            if not pos:
                continue
            try:
                entry_ts = datetime.datetime.fromisoformat(pos["entry_ts"])
            except (KeyError, ValueError, TypeError):
                # Corrupt entry_ts — let Fix 3 entry_ts hardening handle it.
                continue
            age_s = (_now_utc() - entry_ts).total_seconds()
            if age_s < FILL_VERIFY_MIN_AGE_S:
                continue

            if held is None:
                held = fetch_page_positions(self.page, ACCOUNT_ID)
                if held is None:
                    return  # transient, retry next cycle

            if pos.get("ob_id") in held:
                pos["fill_verified"] = True
                _save_state(self.state)
                _log(f"FILL VERIFIED: {pos.get('warrant_name', pos_id)} after {age_s:.0f}s")
                continue

            if age_s < FILL_VERIFY_MAX_AGE_S:
                continue  # not yet timed out, keep waiting

            # Rollback — order never filled within the max window.
            name = pos.get("warrant_name", pos_id)
            units = pos.get("units", 0)
            entry_price = pos.get("entry_price", 0)
            cost = units * entry_price
            _log(f"UNFILLED after {age_s:.0f}s: {name} — rolling back")

            # Cancel the buy order if we have its id (best-effort)
            buy_order_id = pos.get("buy_order_id")
            if buy_order_id:
                try:
                    delete_order_live(self.page, ACCOUNT_ID, buy_order_id)
                except Exception:  # noqa: BLE001
                    # 2026-04-09 Stage 2: WARNING — best-effort rollback
                    # cancel. Failure here means the order may still exist
                    # on Avanza after we've already cleaned up our state;
                    # reconciliation next cycle will catch the divergence.
                    logger.warning(
                        "[SWING] _verify_recent_fills: rollback buy-order cancel failed buy_order_id=%s",
                        buy_order_id,
                        exc_info=True,
                    )

            # Cancel the stop-loss if one was placed
            stop_id = pos.get("stop_order_id")
            if stop_id and stop_id != "DRY_RUN":
                try:
                    _delete_stop_loss(self.page, stop_id)
                except Exception:  # noqa: BLE001
                    # 2026-04-09 Stage 2: WARNING — same pattern as the
                    # buy-order cancel above. Best-effort cleanup in the
                    # rollback path; reconcile next cycle is the safety net.
                    logger.warning(
                        "[SWING] _verify_recent_fills: rollback stop-loss cancel failed stop_id=%s",
                        stop_id,
                        exc_info=True,
                    )

            # Restore cash and remove position
            self.state["cash_sek"] += cost
            del self.state["positions"][pos_id]
            _save_state(self.state)

            _send_telegram(
                f"⚠ *SWING UNFILLED ROLLBACK* {name}\n"
                f"`{units}u @ {entry_price} = {cost:.0f} SEK restored`\n"
                f"`Order didn't fill within {FILL_VERIFY_MAX_AGE_S}s`"
            )

    def _update_regime_history(self, signal_data):
        """Append (action, regime) snapshot per ticker, capped at last 10 entries."""
        if not signal_data:
            return
        for ticker in ("XAG-USD", "XAU-USD"):
            sig = signal_data.get(ticker)
            if not sig:
                continue
            entry = (sig.get("action", "HOLD"), sig.get("regime", "unknown"))
            hist = self.regime_history.setdefault(ticker, [])
            hist.append(entry)
            # Keep last 10 to bound memory
            if len(hist) > 10:
                self.regime_history[ticker] = hist[-10:]

    def _regime_confirmed(self, ticker: str, action: str, regime: str) -> bool:
        """Return True if the (action, regime) pair held for REGIME_CONFIRM_CHECKS in a row.

        Rejects single-check flips like trending-down → ranging BUY in one tick.
        """
        hist = self.regime_history.get(ticker, [])
        if len(hist) < REGIME_CONFIRM_CHECKS:
            return False
        recent = hist[-REGIME_CONFIRM_CHECKS:]
        return all(a == action and r == regime for a, r in recent)

    # -------------------------------------------------------------------
    # Entry logic
    # -------------------------------------------------------------------

    def _check_entries(self, prices, signal_data):
        """Scan for BUY opportunities on XAG-USD and XAU-USD."""
        if not signal_data:
            return

        # Fix 1 (2026-04-09): refuse entries while cash sync is broken.
        # Exit path continues unaffected — protection > perfect accounting.
        if not self.cash_sync_ok:
            if self.check_count % 30 == 0:  # log reminder every ~30 min
                _log("Entries paused: cash sync not ok (awaiting recovery)")
            return

        # Fix 2 (2026-04-09): pause entries while reconciliation is failing
        # — we can't trust our view of Avanza holdings.
        if self.recon_failure_streak >= RECON_FAILURE_STREAK_ALERT:
            if self.check_count % 30 == 0:
                _log(f"Entries paused: reconciliation failing ({self.recon_failure_streak} cycles)")
            return

        active_count = len(self.state["positions"])
        if active_count >= MAX_CONCURRENT:
            return

        # Check cooldown
        if not self._cooldown_cleared():
            return

        for underlying_ticker in ["XAG-USD", "XAU-USD"]:
            sig = signal_data.get(underlying_ticker)
            if not sig:
                continue

            # Already have a position in this underlying?
            if self._has_position(underlying_ticker):
                continue

            # Evaluate entry criteria
            entry_ok, reason = self._evaluate_entry(sig, underlying_ticker)
            if not entry_ok:
                decision = {
                    "ts": _now_utc().isoformat(),
                    "check": self.check_count,
                    "underlying": underlying_ticker,
                    "action": "SKIP_BUY",
                    "reason": reason,
                    "signal": _compact_signal(sig),
                }
                _log_decision(decision)
                continue

            # Select best warrant
            direction = "SHORT" if sig.get("action") == "SELL" else "LONG"
            warrant = self._select_warrant(underlying_ticker, direction)
            if not warrant:
                _log(f"No valid warrant for {underlying_ticker} {direction}")
                continue

            # Fix 8 canary gate (2026-04-09): for SHORT entries, require the
            # warrant key to be in the canary allowlist. This exercises the
            # full SHORT code path in prod on a single low-risk instrument
            # before opening the whole catalog. Leave SHORT_CANARY_WARRANTS
            # empty to keep SHORT entries fully disabled even when
            # SHORT_ENABLED=True — canary opt-in is explicit.
            if direction == "SHORT":
                if not SHORT_CANARY_WARRANTS:
                    decision = {
                        "ts": _now_utc().isoformat(),
                        "check": self.check_count,
                        "underlying": underlying_ticker,
                        "action": "SKIP_BUY",
                        "reason": "SHORT canary allowlist empty — no warrant whitelisted",
                        "signal": _compact_signal(sig),
                    }
                    _log_decision(decision)
                    continue
                if warrant["key"] not in SHORT_CANARY_WARRANTS:
                    decision = {
                        "ts": _now_utc().isoformat(),
                        "check": self.check_count,
                        "underlying": underlying_ticker,
                        "action": "SKIP_BUY",
                        "reason": f"SHORT canary-gated: {warrant['key']} not in allowlist",
                        "signal": _compact_signal(sig),
                    }
                    _log_decision(decision)
                    continue

            # Calculate position size
            cash = self.state["cash_sek"]
            alloc = cash * POSITION_SIZE_PCT / 100
            if alloc < MIN_TRADE_SEK:
                _log(f"Insufficient cash: {cash:.0f} SEK (need {MIN_TRADE_SEK})")
                continue

            ask_price = warrant["live_ask"]
            if ask_price <= 0:
                continue

            units = int(alloc / ask_price)
            if units < 1:
                continue

            total_cost = units * ask_price

            # Execute BUY (direction passed through for state recording)
            self._execute_buy(warrant, units, ask_price, underlying_ticker, sig, total_cost, direction)

    def _evaluate_entry(self, sig, ticker):
        """Check if signal data meets entry criteria. Returns (ok, reason).

        Gates today (post-SG-incident hardening + 2026-04-09 SHORT unlock):
        - action == BUY (LONG) or action == SELL (SHORT, if SHORT_ENABLED)
        - confidence >= MIN_BUY_CONFIDENCE (user rule: no sub-60% trades)
        - majority_count > minority_count (not just majority_count >= MIN)
        - RSI in entry zone (LONG) or mirrored zone (SHORT)
        - MACD improving (LONG) or declining (SHORT) for N checks
        - regime stable for N consecutive checks (no single-check flips)

        SHORT path (Fix 8, 2026-04-09): direction-aware exits are now wired
        in _check_exits so SHORT entries can be safely opened. Gated to
        SHORT_CANARY_WARRANTS for canary observation before full rollout.
        The warrant-level canary check happens in _check_entries after
        warrant selection, not here (this method doesn't know the warrant).
        """
        buy_count = sig.get("buy_count", 0)
        sell_count = sig.get("sell_count", 0)
        rsi = sig.get("rsi", 50)
        action = sig.get("action", "HOLD")
        confidence = float(sig.get("confidence", 0) or 0)

        # Direction check — LONG needs BUY, SHORT needs SELL + SHORT_ENABLED.
        if action == "BUY":
            majority = buy_count
            minority = sell_count
            direction = "LONG"
        elif action == "SELL":
            if not SHORT_ENABLED:
                return False, "action=SELL — SHORT disabled (SHORT_ENABLED=False)"
            majority = sell_count
            minority = buy_count
            direction = "SHORT"
        else:
            return False, f"action={action} (need BUY or SELL consensus)"

        # User rule: no signal trades below 60% calibrated confidence
        if confidence < MIN_BUY_CONFIDENCE:
            return False, f"confidence {confidence:.2f} < {MIN_BUY_CONFIDENCE}"

        # Require strict majority, not just minimum voters. The 2026-04-09
        # incident fired BUY with buy=3 sell=4 because only MIN_BUY_VOTERS
        # was checked. Now we also require majority > minority.
        if majority <= minority:
            return False, f"no strict majority: {direction}={majority} vs other={minority}"

        # Minimum voter threshold (belt-and-suspenders alongside majority check)
        if majority < MIN_BUY_VOTERS:
            return False, f"{direction}_count={majority} < {MIN_BUY_VOTERS}"

        # Timeframe alignment — count timeframes voting in our direction.
        tf = sig.get("timeframes", {})
        if tf:
            target_label = "BUY" if direction == "LONG" else "SELL"
            aligned_tfs = sum(1 for v in tf.values() if v == target_label)
            total_tfs = len(tf)
            if total_tfs > 0 and aligned_tfs / total_tfs < MIN_BUY_TF_RATIO:
                return False, f"TF alignment {aligned_tfs}/{total_tfs} {target_label} < {MIN_BUY_TF_RATIO:.0%}"

        # RSI zone (LONG: not oversold, not overbought; SHORT: mirrored)
        if direction == "LONG":
            if rsi < RSI_ENTRY_LOW:
                return False, f"RSI {rsi:.1f} < {RSI_ENTRY_LOW} (oversold, wait for bounce)"
            if rsi > RSI_ENTRY_HIGH:
                return False, f"RSI {rsi:.1f} > {RSI_ENTRY_HIGH} (overbought)"
        else:  # SHORT — mirror the zone (high RSI = overbought = good short entry)
            if rsi > 100 - RSI_ENTRY_LOW:
                return False, f"RSI {rsi:.1f} > {100 - RSI_ENTRY_LOW} (overbought reversal, wait)"
            if rsi < 100 - RSI_ENTRY_HIGH:
                return False, f"RSI {rsi:.1f} < {100 - RSI_ENTRY_HIGH} (already oversold)"

        # MACD trend — LONG needs MACD rising, SHORT needs MACD falling.
        macd_hist = self.state.get("macd_history", {}).get(ticker, [])
        if len(macd_hist) >= MACD_IMPROVING_CHECKS:
            recent = macd_hist[-MACD_IMPROVING_CHECKS:]
            if direction == "LONG":
                trending = all(recent[i] > recent[i - 1] for i in range(1, len(recent)))
                if not trending:
                    return False, f"MACD not improving for {MACD_IMPROVING_CHECKS} checks"
            else:  # SHORT
                trending = all(recent[i] < recent[i - 1] for i in range(1, len(recent)))
                if not trending:
                    return False, f"MACD not declining for {MACD_IMPROVING_CHECKS} checks"
        # If not enough MACD history yet, skip this check (allow entry)

        # Regime confirmation — require N consecutive (action, regime) checks.
        # The 2026-04-09 incident fired BUY in one cycle after 20+ trending-down
        # checks; the regime flipped to "ranging" + action flipped to "BUY" in
        # a single tick. Reject these single-check flips.
        regime = sig.get("regime", "unknown")
        if not self._regime_confirmed(ticker, action, regime):
            return False, f"regime not confirmed: need {REGIME_CONFIRM_CHECKS}x ({action},{regime})"

        # EOD check — don't buy near close
        h = _cet_hour()
        close_cet = 21.0 + 55 / 60  # 21:55
        minutes_to_close = (close_cet - h) * 60
        if minutes_to_close < EOD_EXIT_MINUTES_BEFORE + 60:
            return False, f"Too close to EOD ({minutes_to_close:.0f}min left)"

        return True, "entry criteria met"

    def _select_warrant(self, underlying, direction):
        """Pick best warrant by leverage/barrier/spread/issuer.

        Uses the dynamic catalog (refreshed from Avanza at startup). Candidates
        are scored on four factors:
        - leverage proximity to TARGET_LEVERAGE (weight 0.35)
        - barrier distance (weight 0.30)
        - bid-ask spread (weight 0.15)
        - issuer fee preference: AVA products get 1.0, others 0.5 (weight 0.20)

        AVA-issued warrants have 0 SEK courtage on Avanza while SG/VT/BNP
        charge regular courtage — the fee_score penalizes non-AVA products.

        Fails closed: returns None if the best candidate's leverage is below
        MIN_ACCEPTABLE_LEVERAGE, so the trader SKIPs instead of falling back
        to a 1.5x tracker (which was the 2026-04-09 SG incident).
        """
        candidates = []
        for key, w in self.warrant_catalog.items():
            if w.get("underlying") != underlying or w.get("direction") != direction:
                continue

            data = fetch_price(self.page, w["ob_id"], w["api_type"])
            if not data or not data.get("bid") or not data.get("ask"):
                continue

            bid = data["bid"]
            ask = data["ask"]
            if bid <= 0 or ask <= 0:
                continue

            spread_pct = (ask - bid) / bid * 100
            live_leverage = data.get("leverage") or w.get("leverage")
            live_barrier = data.get("barrier") or w.get("barrier")

            # Barrier distance — LONG: price above barrier, SHORT: price below barrier.
            barrier_dist = 999
            if live_barrier and live_barrier > 0:
                underlying_price = data.get("underlying", 0)
                if underlying_price > 0:
                    if direction == "LONG":
                        barrier_dist = (underlying_price - live_barrier) / underlying_price * 100
                    else:  # SHORT
                        barrier_dist = (live_barrier - underlying_price) / underlying_price * 100

            # Filter: minimum thresholds
            if barrier_dist < MIN_BARRIER_DISTANCE_PCT:
                _log(f"  {w['name']}: barrier too close ({barrier_dist:.1f}% < {MIN_BARRIER_DISTANCE_PCT}%)")
                continue
            if spread_pct > MIN_SPREAD_PCT:
                _log(f"  {w['name']}: spread too wide ({spread_pct:.1f}% > {MIN_SPREAD_PCT}%)")
                continue

            # Score: prefer leverage near target, far from barrier, tight spread, AVA issuer
            leverage_score = 1.0 / (1 + abs(live_leverage - TARGET_LEVERAGE))
            barrier_score = min(barrier_dist / 30, 1.0)
            spread_score = 1.0 / (1 + spread_pct)
            # isAza=True → 0 courtage on Avanza; other issuers pay regular fees
            is_aza = bool(w.get("isAza", False))
            fee_score = 1.0 if is_aza else 0.5
            score = (
                leverage_score * 0.35
                + barrier_score * 0.30
                + spread_score * 0.15
                + fee_score * 0.20
            )

            candidates.append({
                **w,
                "key": key,
                "live_bid": bid,
                "live_ask": ask,
                "live_leverage": live_leverage,
                "live_barrier": live_barrier,
                "barrier_dist": barrier_dist,
                "spread_pct": spread_pct,
                "underlying_price": data.get("underlying", 0),
                "is_aza": is_aza,
                "score": score,
            })

        if not candidates:
            _log(f"  No valid {direction} candidates for {underlying} in catalog "
                 f"(catalog size: {len(self.warrant_catalog)})")
            return None

        best = max(candidates, key=lambda w: w["score"])

        # Fail-closed: if the best candidate is under-leveraged, SKIP the trade
        # entirely rather than silently buy a low-leverage tracker. The 2026-04-09
        # SG incident bought 1.75x because it was the only candidate that passed
        # the barrier gate — now we explicitly reject that degenerate case.
        if best["live_leverage"] < MIN_ACCEPTABLE_LEVERAGE:
            _log(f"  SKIP_BUY: best candidate {best['name']} has lev "
                 f"{best['live_leverage']:.2f}x < MIN_ACCEPTABLE_LEVERAGE "
                 f"({MIN_ACCEPTABLE_LEVERAGE}). Not falling back to a tracker.")
            return None

        _log(f"  Selected: {best['name']} lev={best['live_leverage']:.2f}x "
             f"barrier={best['barrier_dist']:.1f}% spread={best['spread_pct']:.2f}% "
             f"AVA={best['is_aza']} score={best['score']:.3f}")
        return best

    def _execute_buy(self, warrant, units, ask_price, underlying_ticker, sig, total_cost, direction="LONG"):
        """Execute a BUY order and set stop-loss.

        `direction` is "LONG" or "SHORT" — determines whether _check_exits
        interprets profit as underlying rising (LONG) or falling (SHORT).
        For SHORT, the user still BUYs an inverse warrant (BEAR/MINI S) on
        Avanza — the warrant itself is a long position, just inversely
        correlated with the underlying.
        """
        pos_id = f"pos_{int(time.time())}"

        _log(f"BUY {warrant['name']}: {units}u @ {ask_price} = {total_cost:.0f} SEK "
             f"(underlying: {underlying_ticker}, dir: {direction}, lev: {warrant['live_leverage']:.1f}x)")

        trade_record = {
            "ts": _now_utc().isoformat(),
            "action": "BUY",
            "pos_id": pos_id,
            "warrant_key": warrant["key"],
            "warrant_name": warrant["name"],
            "underlying": underlying_ticker,
            "direction": direction,
            "units": units,
            "price": ask_price,
            "total_sek": round(total_cost, 2),
            "underlying_price": warrant.get("underlying_price", 0),
            "leverage": warrant["live_leverage"],
            "signal": _compact_signal(sig),
            "dry_run": DRY_RUN,
        }

        result = None
        if DRY_RUN:
            _log(f"  [DRY RUN] Would place BUY order: {units}u @ {ask_price}")
            trade_record["result"] = "DRY_RUN"
        else:
            success, result = place_order(self.page, ACCOUNT_ID, warrant["ob_id"], "BUY", ask_price, units)
            trade_record["result"] = result
            if not success:
                _log(f"  BUY FAILED: {result}")
                _log_trade(trade_record)
                _send_telegram(f"*SWING BUY FAILED* {warrant['name']}\n{result.get('parsed', {}).get('message', str(result)[:100])}")
                return

        _log_trade(trade_record)

        # Update state. `fill_verified` defaults False — _verify_recent_fills
        # will flip it True once Avanza confirms the position, or roll back
        # the order after FILL_VERIFY_MAX_AGE_S if the order never filled
        # (Fix 4, 2026-04-09). `buy_order_id` is captured from the order
        # result so rollback can cancel the order. `direction` is recorded
        # for direction-aware exit logic (Fix 8).
        underlying_price = warrant.get("underlying_price", 0)
        buy_order_id = None
        if isinstance(result, dict):
            buy_order_id = result.get("order_id") or result.get("parsed", {}).get("orderId")
        self.state["positions"][pos_id] = {
            "warrant_key": warrant["key"],
            "warrant_name": warrant["name"],
            "ob_id": warrant["ob_id"],
            "api_type": warrant["api_type"],
            "underlying": underlying_ticker,
            "direction": direction,  # "LONG" or "SHORT"
            "units": units,
            "entry_price": ask_price,
            "entry_underlying": underlying_price,
            "entry_ts": _now_utc().isoformat(),
            "peak_underlying": underlying_price,      # LONG: tracks max
            "trough_underlying": underlying_price,    # SHORT: tracks min
            "trailing_active": False,
            "stop_order_id": None,
            "leverage": abs(warrant["live_leverage"]),  # always magnitude
            "fill_verified": DRY_RUN,  # DRY_RUN positions count as verified
            "buy_order_id": buy_order_id,
        }
        self.state["cash_sek"] -= total_cost
        self.state["last_buy_ts"] = _now_utc().isoformat()
        self.state["total_trades"] += 1
        self.state["session_trades"] += 1
        _save_state(self.state)

        # Place hardware stop-loss
        self._set_stop_loss(pos_id)

        # Telegram
        msg = (f"{'*[DRY] ' if DRY_RUN else '*'}SWING BUY* {warrant['name']}\n"
               f"`{units}u @ {ask_price} = {total_cost:.0f} SEK`\n"
               f"`Lev: {warrant['live_leverage']:.1f}x | Underlying: {underlying_price:.2f}`\n"
               f"`Signals: {sig.get('buy_count', 0)}B/{sig.get('sell_count', 0)}S | RSI {sig.get('rsi', 0):.0f}`\n"
               f"_TP: +{TAKE_PROFIT_UNDERLYING_PCT}% und | Stop: -{HARD_STOP_UNDERLYING_PCT}% und_")
        _send_telegram(msg)

        decision = {
            "ts": _now_utc().isoformat(),
            "check": self.check_count,
            "underlying": underlying_ticker,
            "action": "BUY",
            "warrant": warrant["name"],
            "units": units,
            "price": ask_price,
            "total_sek": round(total_cost, 2),
            "signal": _compact_signal(sig),
            "dry_run": DRY_RUN,
        }
        _log_decision(decision)

    def _set_stop_loss(self, pos_id):
        """Place hardware stop-loss for a position."""
        pos = self.state["positions"].get(pos_id)
        if not pos:
            return

        entry_und = pos.get("entry_underlying", 0)
        leverage = pos.get("leverage", 5.0)
        entry_price = pos["entry_price"]

        if entry_und <= 0 or entry_price <= 0:
            _log("  Cannot set stop: no entry underlying price")
            return

        # Stop at -STOP_LOSS_UNDERLYING_PCT% on underlying, translated to warrant price
        und_drop_pct = STOP_LOSS_UNDERLYING_PCT / 100
        warrant_drop_pct = und_drop_pct * leverage
        trigger_price = round(entry_price * (1 - warrant_drop_pct), 2)
        sell_price = round(trigger_price * 0.99, 2)  # sell 1% below trigger for fill

        if trigger_price <= 0:
            _log("  Stop price would be <=0, skipping")
            return

        _log(f"  Setting stop-loss: trigger={trigger_price} sell={sell_price} "
             f"(und -{STOP_LOSS_UNDERLYING_PCT}% * {leverage:.1f}x lev)")

        if DRY_RUN:
            _log(f"  [DRY RUN] Would place stop-loss @ {trigger_price}")
            pos["stop_order_id"] = "DRY_RUN"
            _save_state(self.state)
            return

        success, stop_id = place_stop_loss(
            self.page, ACCOUNT_ID, pos["ob_id"], trigger_price, sell_price,
            pos["units"], valid_days=STOP_LOSS_VALID_DAYS,
        )
        if success:
            pos["stop_order_id"] = stop_id
            _save_state(self.state)
            _log(f"  Stop-loss placed: {stop_id}")
        else:
            _log("  Stop-loss FAILED")
            _send_telegram(f"_SWING: stop-loss failed for {pos['warrant_name']}_")

    # -------------------------------------------------------------------
    # Exit logic
    # -------------------------------------------------------------------

    def _check_exits(self, prices, signal_data):
        """Check exit conditions on all open positions."""
        now = _now_utc()
        h = _cet_hour()
        close_cet = 21.0 + 55 / 60  # 21:55 CET
        minutes_to_close = (close_cet - h) * 60

        to_remove = []
        corrupt_ids = []  # positions with unusable entry_ts — drop without selling

        for pos_id, pos in list(self.state["positions"].items()):
            # Fix 3b (2026-04-09): sell-failed cooldown. If a previous SELL
            # attempt on this position failed, skip exit re-evaluation for a
            # few minutes. Prevents tight cascade loops where the same
            # position keeps hitting the same failing SELL path every cycle.
            sell_failed_at = pos.get("sell_failed_at")
            if sell_failed_at:
                try:
                    age = (now - datetime.datetime.fromisoformat(sell_failed_at)).total_seconds()
                    if age < SELL_FAILED_COOLDOWN_SECONDS:
                        continue
                    # Cooldown expired — clear flags and retry
                    pos.pop("sell_failed_at", None)
                    pos.pop("sell_failed_reason", None)
                except (ValueError, TypeError):
                    # Corrupt timestamp — just clear the flag and retry
                    pos.pop("sell_failed_at", None)
                    pos.pop("sell_failed_reason", None)

            # Fix 3 (2026-04-09): entry_ts hardening. If entry_ts is missing or
            # unparseable, the position is phantom/corrupt — drop it without
            # attempting a sell. Don't default to "now" or a synthetic value.
            try:
                entry_ts = datetime.datetime.fromisoformat(pos["entry_ts"])
            except (KeyError, ValueError, TypeError) as e:
                _log(f"CORRUPT POS DROPPED: {pos_id} invalid entry_ts: {e}")
                _send_telegram(
                    f"⚠ _SWING: dropped corrupt position {pos.get('warrant_name', pos_id)} "
                    f"— invalid entry_ts ({e})_"
                )
                corrupt_ids.append(pos_id)
                continue

            # Get current underlying price
            underlying_price = self._get_underlying_price(pos, prices)
            if underlying_price <= 0:
                continue

            entry_und = pos.get("entry_underlying", 0)
            if entry_und <= 0:
                continue

            # Direction-aware extreme tracking + profit/loss computation
            # (Fix 8, 2026-04-09). For LONG, "extreme" = max underlying seen
            # since entry; profit = (current - entry)/entry. For SHORT,
            # "extreme" = min underlying seen; profit = (entry - current)/entry.
            # `und_change_pct` below is direction-aware "profit percentage":
            # positive = in profit, negative = in drawdown, regardless of
            # direction. This keeps the downstream TAKE_PROFIT / HARD_STOP /
            # TRAILING_STOP conditions unchanged — they still check "profit
            # vs threshold" but the profit definition flips for SHORT.
            direction = pos.get("direction", "LONG")  # default for legacy positions
            if direction == "LONG":
                if underlying_price > pos.get("peak_underlying", 0):
                    pos["peak_underlying"] = underlying_price
                und_change_pct = (underlying_price - entry_und) / entry_und * 100
                extreme_und = pos.get("peak_underlying", entry_und)
                # from_peak: how far underlying has retraced DOWN from its max
                from_peak_pct = (underlying_price - extreme_und) / extreme_und * 100 if extreme_und > 0 else 0
            else:  # SHORT
                # Initialize trough if missing (legacy positions that somehow
                # got marked SHORT without a trough field)
                if "trough_underlying" not in pos or pos.get("trough_underlying", 0) <= 0:
                    pos["trough_underlying"] = entry_und
                if underlying_price < pos["trough_underlying"]:
                    pos["trough_underlying"] = underlying_price
                und_change_pct = (entry_und - underlying_price) / entry_und * 100
                extreme_und = pos.get("trough_underlying", entry_und)
                # from_peak: how far underlying has retraced UP from its min
                # (this is a LOSS for a SHORT, so sign convention matches LONG
                # where retracement from peak is a LOSS — both are negative
                # when we're giving back profit).
                from_peak_pct = (extreme_und - underlying_price) / extreme_und * 100 if extreme_und > 0 else 0

            # Get current warrant price for P&L
            warrant_data = fetch_price(self.page, pos["ob_id"], pos["api_type"])
            current_bid = warrant_data.get("bid", 0) if warrant_data else 0

            exit_reason = None

            # --- Exit optimizer: probabilistic exit assessment ---
            # SHORT positions SKIP the optimizer entirely (Fix 8, 2026-04-09 —
            # Option A): Position dataclass is LONG-only and would compute
            # inverted EV, potentially steering toward bad exits. SHORTs fall
            # through to the simple exit rules below.
            try:
                from portfolio.cost_model import get_cost_model
                from portfolio.exit_optimizer import MarketSnapshot, Position, compute_exit_plan
                from portfolio.session_calendar import get_session_info
                sess = get_session_info("warrant", underlying=pos.get("underlying"))
                if direction == "LONG" and sess.is_open and sess.remaining_minutes >= 2:
                    opt_pos = Position(
                        symbol=pos.get("underlying", ""),
                        qty=pos.get("units", 0),
                        entry_price_sek=pos.get("entry_price", 0),
                        entry_underlying_usd=entry_und,
                        entry_ts=datetime.datetime.fromisoformat(pos["entry_ts"]) if pos.get("entry_ts") else _now_utc(),
                        instrument_type="warrant",
                        leverage=pos.get("leverage", 5.0),
                        financing_level=pos.get("financing_level"),
                    )
                    # NOTE: MarketSnapshot.bid is documented as the underlying's
                    # USD bid (not the warrant SEK bid). Passing the warrant bid
                    # here was the source of the 2026-04-09 -2430 SEK fake-loss
                    # bug — _compute_pnl_sek treated warrant_bid_sek as if it
                    # were a silver USD price and computed a -46% "move".
                    # Pass None so the optimizer uses market.price (the correct
                    # underlying USD reference) for the market-exit estimate.
                    opt_market = MarketSnapshot(
                        asof_ts=_now_utc(),
                        price=underlying_price,
                        bid=None,
                        usdsek=10.85,
                    )
                    exit_plan = compute_exit_plan(
                        opt_pos, opt_market, sess.session_end,
                        costs=get_cost_model("warrant"), n_paths=2000,
                    )
                    pos["_exit_plan"] = {
                        "recommended": exit_plan.recommended.action,
                        "rec_price": exit_plan.recommended.price_usd,
                        "rec_ev": exit_plan.recommended.ev_sek,
                        "rec_fill_prob": exit_plan.recommended.fill_prob,
                        "stop_hit_prob": exit_plan.stop_hit_prob,
                        "risk_flags": list(exit_plan.recommended.risk_flags),
                        "market_exit_pnl": exit_plan.market_exit.pnl_sek,
                    }
                    # Override: if optimizer says market exit due to risk override
                    if (exit_plan.recommended.action == "market"
                            and any(f in exit_plan.recommended.risk_flags
                                    for f in ("KNOCKOUT_DANGER", "SESSION_END_IMMINENT"))):
                        exit_reason = f"EXIT_OPTIMIZER: {', '.join(exit_plan.recommended.risk_flags)} (EV {exit_plan.recommended.ev_sek:+,.0f} SEK)"
                    # Override: if stop hit probability is very high
                    elif exit_plan.stop_hit_prob > 0.30:
                        exit_reason = f"EXIT_OPTIMIZER: stop hit prob {exit_plan.stop_hit_prob:.0%} > 30% (EV {exit_plan.recommended.ev_sek:+,.0f} SEK)"
            except Exception:
                # 2026-04-09 Stage 2: WARNING — exit optimizer is a bonus
                # layer; failure falls through to rule-based exit logic
                # below (take profit / trailing / stop / barrier). exc_info
                # surfaces whatever broke inside the probabilistic model.
                logger.warning(
                    "[SWING] _check_exits: exit_optimizer raised — falling back to rule-based exit logic",
                    exc_info=True,
                )

            # 1. Take profit
            if not exit_reason and und_change_pct >= TAKE_PROFIT_UNDERLYING_PCT:
                exit_reason = f"TAKE_PROFIT: underlying +{und_change_pct:.2f}% >= +{TAKE_PROFIT_UNDERLYING_PCT}%"

            # 2. Trailing stop
            if not exit_reason and und_change_pct >= TRAILING_START_PCT:
                pos["trailing_active"] = True
                if from_peak_pct <= -TRAILING_DISTANCE_PCT:
                    exit_reason = f"TRAILING_STOP: {from_peak_pct:.2f}% from peak (trail={TRAILING_DISTANCE_PCT}%)"

            # 3. Hard stop
            if not exit_reason and und_change_pct <= -HARD_STOP_UNDERLYING_PCT:
                exit_reason = f"HARD_STOP: underlying {und_change_pct:.2f}% <= -{HARD_STOP_UNDERLYING_PCT}%"

            # 4. Signal reversal (direction-aware, Fix 8 2026-04-09).
            # LONG exits on SELL consensus; SHORT exits on BUY consensus.
            if not exit_reason and SIGNAL_REVERSAL_EXIT and signal_data:
                sig = signal_data.get(pos["underlying"], {})
                if direction == "LONG":
                    rev_count = sig.get("sell_count", 0)
                    rev_label = "SELL"
                else:
                    rev_count = sig.get("buy_count", 0)
                    rev_label = "BUY"
                tf = sig.get("timeframes", {})
                rev_tfs = sum(1 for v in tf.values() if v == rev_label) if tf else 0
                total_tfs = len(tf) if tf else 7
                if rev_count >= MIN_BUY_VOTERS and total_tfs > 0 and rev_tfs / total_tfs >= MIN_BUY_TF_RATIO:
                    exit_reason = f"SIGNAL_REVERSAL: {rev_count}{rev_label[0]}, {rev_tfs}/{total_tfs} TFs {rev_label}"

            # 5. Time limit (entry_ts already parsed at top of loop)
            if not exit_reason:
                held_hours = (now - entry_ts).total_seconds() / 3600
                if held_hours >= MAX_HOLD_HOURS:
                    exit_reason = f"TIME_LIMIT: held {held_hours:.1f}h >= {MAX_HOLD_HOURS}h"

            # 6. EOD exit
            if not exit_reason and minutes_to_close <= EOD_EXIT_MINUTES_BEFORE:
                exit_reason = f"EOD_EXIT: {minutes_to_close:.0f}min to close"

            # 7. Momentum exit (direction-aware, Fix 8 2026-04-09).
            # LONG exits on 3 consecutive declines (underlying going against
            # us); SHORT exits on 3 consecutive rises.
            if not exit_reason and len(self.state.get("_und_history", {}).get(pos["underlying"], [])) >= 3:
                hist = self.state["_und_history"][pos["underlying"]][-3:]
                if direction == "LONG":
                    monotonic = all(hist[i] < hist[i - 1] for i in range(1, len(hist)))
                    move_rate = (hist[-1] - hist[0]) / hist[0] * 100
                    if monotonic and move_rate < -0.3:
                        exit_reason = f"MOMENTUM_EXIT: 3 declining checks ({move_rate:.2f}%)"
                else:  # SHORT
                    monotonic = all(hist[i] > hist[i - 1] for i in range(1, len(hist)))
                    move_rate = (hist[-1] - hist[0]) / hist[0] * 100
                    if monotonic and move_rate > 0.3:
                        exit_reason = f"MOMENTUM_EXIT: 3 rising checks ({move_rate:+.2f}%)"

            if exit_reason:
                self._execute_sell(pos_id, pos, current_bid, underlying_price, exit_reason)
                to_remove.append(pos_id)

        # Update underlying price history for momentum tracking
        if signal_data:
            if "_und_history" not in self.state:
                self.state["_und_history"] = {}
            for ticker in ["XAG-USD", "XAU-USD"]:
                und_price = self._get_ticker_underlying_price(ticker, prices)
                if und_price > 0:
                    hist = self.state["_und_history"].setdefault(ticker, [])
                    hist.append(und_price)
                    if len(hist) > 10:
                        hist.pop(0)

        # Fix 3b (2026-04-09): actually delete positions that were sold OR
        # flagged corrupt. Prior to this fix, `to_remove.append(pos_id)` was
        # populated but never iterated — `_execute_sell` handled deletion on
        # success but returned early on failure, leaving phantom positions
        # in state forever and causing the 08:25 UTC cascade.
        dirty = False
        for pos_id in to_remove:
            # Only remove positions where _execute_sell succeeded.
            # Failed sells leave sell_failed_at set — respected via the
            # cooldown at the top of the loop next cycle. See Fix 3b.
            if pos_id in self.state["positions"] and not self.state["positions"][pos_id].get("sell_failed_at"):
                del self.state["positions"][pos_id]
                dirty = True
        for pos_id in corrupt_ids:
            if pos_id in self.state["positions"]:
                del self.state["positions"][pos_id]
                dirty = True
        if dirty:
            _save_state(self.state)

    def _execute_sell(self, pos_id, pos, current_bid, underlying_price, reason):
        """Execute a SELL order for a position."""
        units = pos["units"]
        entry_price = pos["entry_price"]
        warrant_pnl_pct = ((current_bid / entry_price) - 1) * 100 if entry_price > 0 else 0
        proceeds = units * current_bid

        _log(f"SELL {pos['warrant_name']}: {units}u @ {current_bid} = {proceeds:.0f} SEK "
             f"(PnL: {warrant_pnl_pct:+.1f}%) — {reason}")

        # Safe held_hours calculation — _check_exits has already validated
        # entry_ts (Fix 3), but this method can be called from other paths
        # so we guard here too.
        try:
            held_hours = round(
                (_now_utc() - datetime.datetime.fromisoformat(pos["entry_ts"])).total_seconds() / 3600,
                2,
            )
        except (KeyError, ValueError, TypeError):
            held_hours = 0.0

        trade_record = {
            "ts": _now_utc().isoformat(),
            "action": "SELL",
            "pos_id": pos_id,
            "warrant_key": pos["warrant_key"],
            "warrant_name": pos["warrant_name"],
            "underlying": pos["underlying"],
            "units": units,
            "price": current_bid,
            "total_sek": round(proceeds, 2),
            "entry_price": entry_price,
            "pnl_pct": round(warrant_pnl_pct, 2),
            "pnl_sek": round(proceeds - (units * entry_price), 2),
            "underlying_price": underlying_price,
            "entry_underlying": pos.get("entry_underlying", 0),
            "reason": reason,
            "held_hours": held_hours,
            "dry_run": DRY_RUN,
        }

        if DRY_RUN:
            _log(f"  [DRY RUN] Would place SELL order: {units}u @ {current_bid}")
            trade_record["result"] = "DRY_RUN"
        else:
            success, result = place_order(self.page, ACCOUNT_ID, pos["ob_id"], "SELL", current_bid, units)
            trade_record["result"] = result
            if not success:
                _log(f"  SELL FAILED: {result}")
                _log_trade(trade_record)
                _send_telegram(f"*SWING SELL FAILED* {pos['warrant_name']}\n{result.get('parsed', {}).get('message', str(result)[:100])}")
                # Fix 3b (2026-04-09): mark the failure time so _check_exits
                # will skip re-evaluating this position for
                # SELL_FAILED_COOLDOWN_SECONDS. Prevents the cascade bug
                # where a failing SELL re-fired every cycle.
                pos["sell_failed_at"] = _now_utc().isoformat()
                parsed_msg = ""
                try:
                    parsed_msg = str(result.get("parsed", {}).get("message", ""))[:200]
                except (AttributeError, TypeError):
                    parsed_msg = str(result)[:200]
                pos["sell_failed_reason"] = parsed_msg
                _save_state(self.state)
                return

        _log_trade(trade_record)

        # Cancel hardware stop-loss
        if pos.get("stop_order_id") and pos["stop_order_id"] != "DRY_RUN" and not DRY_RUN:
            ok = _delete_stop_loss(self.page, pos["stop_order_id"])
            _log(f"  Stop-loss cancelled: {ok}")

        # Update state
        pnl_sek = proceeds - (units * entry_price)
        self.state["cash_sek"] += proceeds
        self.state["total_pnl_sek"] += pnl_sek
        self.state["total_trades"] += 1
        self.state["session_trades"] += 1

        if pnl_sek < 0:
            self.state["consecutive_losses"] = self.state.get("consecutive_losses", 0) + 1
        else:
            self.state["consecutive_losses"] = 0

        # Remove position
        del self.state["positions"][pos_id]
        _save_state(self.state)

        # Telegram
        pnl_emoji = "+" if pnl_sek >= 0 else ""
        msg = (f"{'*[DRY] ' if DRY_RUN else '*'}SWING SELL* {pos['warrant_name']}\n"
               f"`{units}u @ {current_bid} = {proceeds:.0f} SEK`\n"
               f"`PnL: {pnl_emoji}{warrant_pnl_pct:.1f}% ({pnl_emoji}{pnl_sek:.0f} SEK)`\n"
               f"`Reason: {reason}`\n"
               f"_Cash: {self.state['cash_sek']:.0f} SEK | Total PnL: {self.state['total_pnl_sek']:+.0f} SEK_")
        _send_telegram(msg)

        decision = {
            "ts": _now_utc().isoformat(),
            "check": self.check_count,
            "underlying": pos["underlying"],
            "action": "SELL",
            "warrant": pos["warrant_name"],
            "units": units,
            "price": current_bid,
            "pnl_pct": round(warrant_pnl_pct, 2),
            "pnl_sek": round(pnl_sek, 2),
            "reason": reason,
            "dry_run": DRY_RUN,
        }
        _log_decision(decision)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _has_position(self, underlying_ticker):
        """Check if we already have a position in this underlying."""
        return any(pos["underlying"] == underlying_ticker for pos in self.state["positions"].values())

    def _cooldown_cleared(self):
        """Check if BUY cooldown has elapsed (with loss escalation)."""
        last_ts = self.state.get("last_buy_ts")
        if not last_ts:
            return True

        try:
            last_dt = datetime.datetime.fromisoformat(last_ts)
            elapsed_min = (_now_utc() - last_dt).total_seconds() / 60

            losses = self.state.get("consecutive_losses", 0)
            multiplier = LOSS_ESCALATION.get(min(losses, max(LOSS_ESCALATION.keys())), 1)
            required_min = BUY_COOLDOWN_MINUTES * multiplier

            if elapsed_min < required_min:
                return False
        except Exception:
            logger.warning("[SWING] SwingTrader._cooldown_cleared: corrupt last_buy_ts=%r — clearing cooldown", last_ts, exc_info=True)

        return True

    def _get_underlying_price(self, pos, prices):
        """Get current underlying price from loop's price data or fetch directly."""
        # Try from loop's price data (keyed by position name, not ticker)
        for key, p in prices.items():
            if isinstance(p, dict) and p.get("underlying"):
                ticker = "XAG-USD" if "silver" in key.lower() else "XAU-USD"
                if ticker == pos.get("underlying"):
                    return p["underlying"]

        # Fallback: fetch warrant price and extract underlying
        data = fetch_price(self.page, pos["ob_id"], pos["api_type"])
        if data and data.get("underlying"):
            return data["underlying"]

        return 0

    def _get_ticker_underlying_price(self, ticker, prices):
        """Get underlying price for a ticker from loop price data."""
        for key, p in prices.items():
            if isinstance(p, dict) and p.get("underlying"):
                mapped = "XAG-USD" if "silver" in key.lower() else "XAU-USD"
                if mapped == ticker:
                    return p["underlying"]
        return 0

    def _update_macd_history(self, signal_data):
        """Track MACD histogram values across checks."""
        if not signal_data:
            return

        if "macd_history" not in self.state:
            self.state["macd_history"] = {}

        for ticker in ["XAG-USD", "XAU-USD"]:
            sig = signal_data.get(ticker)
            if not sig:
                continue

            macd_hist = sig.get("macd_hist")
            if macd_hist is None:
                continue

            history = self.state["macd_history"].setdefault(ticker, [])
            history.append(macd_hist)
            if len(history) > 20:
                history.pop(0)

        _save_state(self.state)

    def _send_summary(self, signal_data):
        """Send periodic Telegram summary."""
        positions = self.state["positions"]
        cash = self.state["cash_sek"]

        if not positions:
            return

        lines = [f"*SWING #{self.check_count}* {len(positions)} position(s)"]
        for _pid, pos in positions.items():
            data = fetch_price(self.page, pos["ob_id"], pos["api_type"])
            bid = data.get("bid", 0) if data else 0
            pnl = ((bid / pos["entry_price"]) - 1) * 100 if pos["entry_price"] > 0 else 0
            held = (_now_utc() - datetime.datetime.fromisoformat(pos["entry_ts"])).total_seconds() / 3600
            trail = " TRAIL" if pos.get("trailing_active") else ""
            lines.append(f"`{pos['warrant_name']}: {bid} ({pnl:+.1f}%) {held:.1f}h{trail}`")
        lines.append(f"`Cash: {cash:.0f} | Trades: {self.state['total_trades']} | PnL: {self.state['total_pnl_sek']:+.0f}`")
        if DRY_RUN:
            lines.append("_DRY RUN mode_")
        _send_telegram("\n".join(lines))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _compact_signal(sig):
    """Create compact signal summary for logging."""
    if not sig:
        return {}
    return {
        "action": sig.get("action"),
        "buy": sig.get("buy_count", 0),
        "sell": sig.get("sell_count", 0),
        "rsi": round(sig.get("rsi", 0), 1),
        "macd": sig.get("macd_hist"),
        "regime": sig.get("regime"),
        "confidence": sig.get("confidence", 0),
    }
