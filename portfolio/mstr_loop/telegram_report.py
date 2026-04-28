"""MSTR Loop Telegram reporter — hourly status + per-trade alerts.

Thin wrapper around portfolio.message_store.send_or_store. Throttling via
portfolio.message_throttle pattern + local state file that remembers when
we last sent each category of message.

Falls back to a no-op if config.TELEGRAM_ENABLED is False or the message
infrastructure is unavailable (no config.json, no bot token). Non-fatal
in all failure modes — telemetry must not block trading.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any

from portfolio.mstr_loop import config
from portfolio.mstr_loop.state import BotState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local "last sent" state (one file, tracks cadence per message category)
# ---------------------------------------------------------------------------

def _load_ts_state(path: str = config.TELEGRAM_STATE_FILE) -> dict[str, str]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _save_ts_state(state: dict[str, str], path: str = config.TELEGRAM_STATE_FILE) -> None:
    try:
        from portfolio.file_utils import atomic_write_json
        atomic_write_json(path, state, ensure_ascii=False)
    except Exception:
        logger.debug("telegram_report: state save failed", exc_info=True)


def _can_send(category: str, min_interval_minutes: float, ts_state: dict[str, str]) -> bool:
    """True if more than min_interval minutes have elapsed since last category send."""
    last = ts_state.get(category)
    if not last:
        return True
    try:
        last_dt = datetime.datetime.fromisoformat(last)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=datetime.UTC)
    except (TypeError, ValueError):
        return True
    elapsed_min = (datetime.datetime.now(datetime.UTC) - last_dt).total_seconds() / 60
    return elapsed_min >= min_interval_minutes


def _mark_sent(category: str, ts_state: dict[str, str]) -> None:
    ts_state[category] = datetime.datetime.now(datetime.UTC).isoformat()
    _save_ts_state(ts_state)


# ---------------------------------------------------------------------------
# Message renderers
# ---------------------------------------------------------------------------

def _render_hourly_report(state: BotState) -> str:
    """Compact status line: cash, open position, day P&L, win rate."""
    win_rate = (state.wins / state.total_trades * 100) if state.total_trades else 0.0
    lines = [
        "*MSTR Loop — hourly status*",
        f"`phase: {config.PHASE}  cash: {state.cash_sek:,.0f} SEK`",
        f"`trades: {state.total_trades} ({state.wins}W/{state.losses}L, {win_rate:.0f}%)`",
        f"`total_pnl: {state.total_pnl_sek:+,.0f} SEK`",
    ]
    if state.positions:
        for pos in state.positions.values():
            lines.append(
                f"`OPEN {pos.strategy_key} {pos.direction} "
                f"{pos.units} units @ ${pos.entry_underlying_price:.2f}`"
            )
    else:
        lines.append("`open: (none)`")
    return "\n".join(lines)


def _render_trade_alert(action: str, pos_or_decision: Any, context: dict[str, Any]) -> str:
    """Per-trade alert for BUY/SELL events."""
    if action == "BUY":
        d = pos_or_decision
        return (
            f"*MSTR Loop — {config.PHASE.upper()} {action}*\n"
            f"`{d.strategy_key} {d.direction} @ ${context.get('underlying_price', 0):.2f}`\n"
            f"`units={context.get('units', 0)} cert=${context.get('cert_price', 0):.2f}`\n"
            f"_reason: {d.rationale}_"
        )
    # SELL
    pos = pos_or_decision
    pnl_sek = context.get("pnl_sek", 0)
    exit_reason = context.get("exit_reason", "?")
    return (
        f"*MSTR Loop — {config.PHASE.upper()} {action}*\n"
        f"`{getattr(pos, 'strategy_key', '?')} {getattr(pos, 'direction', '?')} "
        f"exit @ ${context.get('underlying_price', 0):.2f}`\n"
        f"`pnl={pnl_sek:+,.0f} SEK  reason={exit_reason}`"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def maybe_send_hourly(state: BotState) -> None:
    """Send hourly status report if cadence elapsed. No-op otherwise."""
    if not config.TELEGRAM_ENABLED:
        return
    ts_state = _load_ts_state()
    if not _can_send("hourly", config.TELEGRAM_HOURLY_REPORT_MINUTES, ts_state):
        return
    _send(_render_hourly_report(state), category="analysis")
    _mark_sent("hourly", ts_state)


def send_trade_alert(action: str, pos_or_decision: Any, context: dict[str, Any]) -> None:
    """Fire-and-forget per-trade alert. Called from execution on fill."""
    if not config.TELEGRAM_ENABLED or not config.TELEGRAM_PER_TRADE_ALERTS:
        return
    msg = _render_trade_alert(action, pos_or_decision, context)
    _send(msg, category="analysis")


def _send(text: str, category: str = "analysis") -> None:
    """Backend send — routes through portfolio.message_store.send_or_store."""
    try:
        from portfolio.config import load_config
        from portfolio.message_store import send_or_store
        cfg = load_config()
        send_or_store(text, cfg, category=category)
    except Exception:
        # Try fallback: raw send_telegram. Final fallback: log only.
        try:
            from portfolio.config import load_config
            from portfolio.telegram_notifications import send_telegram
            cfg = load_config()
            send_telegram(text, cfg)
        except Exception:
            logger.info("telegram_report: delivery unavailable — logging instead: %s",
                        text.replace("\n", " | "))
