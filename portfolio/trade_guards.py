"""Trade guards — overtrading prevention for the trading agent.

Three guards:
1. Per-ticker cooldown: No re-trade on same ticker within N minutes.
2. Consecutive-loss escalation: After losses, increase cooldown multiplier.
3. Position rate limit: Max N new positions per time window.

State is persisted to data/trade_guard_state.json.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

from portfolio.file_utils import atomic_write_json, load_json

logger = logging.getLogger("portfolio.trade_guards")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATE_FILE = DATA_DIR / "trade_guard_state.json"

# Defaults
DEFAULT_TICKER_COOLDOWN_MINUTES = 30
DEFAULT_BOLD_POSITION_LIMIT = 1       # max new positions per window
DEFAULT_BOLD_POSITION_WINDOW_H = 4    # hours
DEFAULT_PATIENT_POSITION_LIMIT = 1
DEFAULT_PATIENT_POSITION_WINDOW_H = 8
LOSS_ESCALATION = {0: 1, 1: 1, 2: 2, 3: 4, 4: 8}  # consecutive_losses -> cooldown multiplier


def _load_state():
    """Load trade guard state from disk."""
    return load_json(str(STATE_FILE), default={
        "ticker_trades": {},
        "consecutive_losses": {"patient": 0, "bold": 0},
        "new_position_timestamps": {"patient": [], "bold": []},
    })


def _save_state(state):
    """Persist trade guard state to disk."""
    atomic_write_json(STATE_FILE, state)


def _get_cooldown_multiplier(consecutive_losses):
    """Get cooldown multiplier based on consecutive loss count."""
    if consecutive_losses >= 4:
        return LOSS_ESCALATION[4]
    return LOSS_ESCALATION.get(consecutive_losses, 1)


def check_overtrading_guards(ticker, action, strategy, portfolio, config=None):
    """Check all trade guards for a proposed trade.

    Args:
        ticker: Instrument ticker (e.g., "BTC-USD").
        action: "BUY" or "SELL".
        strategy: "patient" or "bold".
        portfolio: Portfolio state dict.
        config: Optional config dict with trade_guards settings.

    Returns:
        list of warning dicts, each with:
            - guard: str (guard name)
            - severity: "warning" or "block"
            - message: str
            - details: dict (guard-specific data)
        Empty list means all guards pass.
    """
    cfg = (config or {}).get("trade_guards", {})
    if cfg.get("enabled") is False:
        return []

    warnings = []
    state = _load_state()
    now = datetime.now(timezone.utc)

    # --- Guard 1: Per-ticker cooldown ---
    base_cooldown = cfg.get("ticker_cooldown_minutes", DEFAULT_TICKER_COOLDOWN_MINUTES)
    consecutive = state.get("consecutive_losses", {}).get(strategy, 0)
    multiplier = _get_cooldown_multiplier(consecutive)
    effective_cooldown = base_cooldown * multiplier

    key = f"{strategy}:{ticker}"
    ticker_trades = state.get("ticker_trades", {})
    last_trade_str = ticker_trades.get(key)
    if last_trade_str:
        try:
            last_trade = datetime.fromisoformat(last_trade_str)
            elapsed = (now - last_trade).total_seconds() / 60
            if elapsed < effective_cooldown:
                remaining = effective_cooldown - elapsed
                warnings.append({
                    "guard": "ticker_cooldown",
                    "severity": "warning",
                    "message": (
                        f"{ticker} traded {elapsed:.0f}m ago by {strategy}. "
                        f"Cooldown: {effective_cooldown:.0f}m (base {base_cooldown}m × {multiplier}x). "
                        f"{remaining:.0f}m remaining."
                    ),
                    "details": {
                        "ticker": ticker,
                        "strategy": strategy,
                        "elapsed_min": round(elapsed, 1),
                        "cooldown_min": effective_cooldown,
                        "multiplier": multiplier,
                        "remaining_min": round(remaining, 1),
                    },
                })
        except (ValueError, TypeError):
            pass

    # --- Guard 2: Consecutive-loss escalation (informational) ---
    if consecutive >= 2:
        warnings.append({
            "guard": "consecutive_losses",
            "severity": "warning",
            "message": (
                f"{strategy}: {consecutive} consecutive losses. "
                f"Cooldown multiplier: {multiplier}x."
            ),
            "details": {
                "strategy": strategy,
                "consecutive_losses": consecutive,
                "multiplier": multiplier,
            },
        })

    # --- Guard 3: Position rate limit (BUY only) ---
    if action == "BUY":
        is_bold = strategy == "bold"
        limit = cfg.get(
            f"{'bold' if is_bold else 'patient'}_position_limit",
            DEFAULT_BOLD_POSITION_LIMIT if is_bold else DEFAULT_PATIENT_POSITION_LIMIT,
        )
        window_h = cfg.get(
            f"{'bold' if is_bold else 'patient'}_position_window_h",
            DEFAULT_BOLD_POSITION_WINDOW_H if is_bold else DEFAULT_PATIENT_POSITION_WINDOW_H,
        )
        cutoff = now - timedelta(hours=window_h)

        timestamps = state.get("new_position_timestamps", {}).get(strategy, [])
        recent = []
        for ts_str in timestamps:
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts >= cutoff:
                    recent.append(ts)
            except (ValueError, TypeError):
                continue

        if len(recent) >= limit:
            warnings.append({
                "guard": "position_rate_limit",
                "severity": "warning",
                "message": (
                    f"{strategy}: {len(recent)} new position(s) in last {window_h}h "
                    f"(limit: {limit})."
                ),
                "details": {
                    "strategy": strategy,
                    "recent_count": len(recent),
                    "limit": limit,
                    "window_hours": window_h,
                },
            })

    return warnings


def record_trade(ticker, direction, strategy, pnl_pct=None, config=None):
    """Record a completed trade for guard tracking.

    Call this after executing a trade to update cooldowns and loss streaks.

    Args:
        ticker: Instrument ticker.
        direction: "BUY" or "SELL".
        strategy: "patient" or "bold".
        pnl_pct: Realized P&L percentage (for SELL trades). None for BUY.
        config: Optional config dict.
    """
    state = _load_state()
    now = datetime.now(timezone.utc)
    now_str = now.isoformat()

    # Update ticker trade timestamp
    key = f"{strategy}:{ticker}"
    if "ticker_trades" not in state:
        state["ticker_trades"] = {}
    state["ticker_trades"][key] = now_str

    # Update consecutive losses on SELL
    if direction == "SELL" and pnl_pct is not None:
        if "consecutive_losses" not in state:
            state["consecutive_losses"] = {"patient": 0, "bold": 0}
        if pnl_pct < 0:
            state["consecutive_losses"][strategy] = (
                state["consecutive_losses"].get(strategy, 0) + 1
            )
        else:
            state["consecutive_losses"][strategy] = 0

    # Track new position timestamps (BUY only)
    if direction == "BUY":
        if "new_position_timestamps" not in state:
            state["new_position_timestamps"] = {"patient": [], "bold": []}
        if strategy not in state["new_position_timestamps"]:
            state["new_position_timestamps"][strategy] = []
        state["new_position_timestamps"][strategy].append(now_str)

        # Prune old timestamps (keep last 24h)
        cutoff = now - timedelta(hours=24)
        state["new_position_timestamps"][strategy] = [
            ts for ts in state["new_position_timestamps"][strategy]
            if datetime.fromisoformat(ts) >= cutoff
        ]

    _save_state(state)


def get_all_guard_warnings(signals, patient_pf, bold_pf, config=None):
    """Get trade guard warnings for all tickers with BUY/SELL signals.

    Args:
        signals: Dict of ticker -> signal data (from agent_summary).
        patient_pf: Patient portfolio state dict.
        bold_pf: Bold portfolio state dict.
        config: Optional config dict.

    Returns:
        dict with:
            - warnings: list of warning dicts
            - summary: str (human-readable summary)
    """
    cfg = (config or {}).get("trade_guards", {})
    if cfg.get("enabled") is False:
        return {"warnings": [], "summary": "Trade guards disabled"}

    all_warnings = []

    for ticker, sig in signals.items():
        action = sig.get("action", "HOLD")
        if action == "HOLD":
            continue

        for strategy, portfolio in [("patient", patient_pf), ("bold", bold_pf)]:
            warns = check_overtrading_guards(
                ticker, action, strategy, portfolio, config
            )
            all_warnings.extend(warns)

    summary_parts = []
    if all_warnings:
        by_guard = {}
        for w in all_warnings:
            guard = w["guard"]
            by_guard.setdefault(guard, []).append(w)
        for guard, warns in by_guard.items():
            summary_parts.append(f"{guard}: {len(warns)} warning(s)")

    return {
        "warnings": all_warnings,
        "summary": "; ".join(summary_parts) if summary_parts else "All clear",
    }
