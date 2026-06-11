"""Trade guards — overtrading prevention for the trading agent.

Three guards:
1. Per-ticker cooldown: No re-trade on same ticker within N minutes.
2. Consecutive-loss escalation: After losses, increase cooldown multiplier.
3. Position rate limit: Max N new positions per time window.

State is persisted to data/trade_guard_state.json.
"""

import logging
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

from portfolio.file_utils import atomic_write_json, load_json

logger = logging.getLogger("portfolio.trade_guards")


def _default_state():
    """Fresh guard state. Factory (not a constant) so each load gets its own
    mutable containers — never aliases a module global."""
    return {
        "ticker_trades": {},
        "consecutive_losses": {"patient": 0, "bold": 0},
        "last_loss_ts": {"patient": None, "bold": None},
        "new_position_timestamps": {"patient": [], "bold": []},
    }

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATE_FILE = DATA_DIR / "trade_guard_state.json"

# Defaults
DEFAULT_TICKER_COOLDOWN_MINUTES = 30
DEFAULT_BOLD_POSITION_LIMIT = 1       # max new positions per window
DEFAULT_BOLD_POSITION_WINDOW_H = 4    # hours
DEFAULT_PATIENT_POSITION_LIMIT = 1
DEFAULT_PATIENT_POSITION_WINDOW_H = 8
LOSS_ESCALATION = {0: 1, 1: 1, 2: 2, 3: 4, 4: 8}  # consecutive_losses -> cooldown multiplier
LOSS_DECAY_HOURS = 24  # halve escalation multiplier every N hours without a trade

_state_lock = threading.Lock()


def _load_state():
    """Load trade guard state from disk.

    2026-06-11 (audit batch 7): a corrupt trade_guard_state.json used to fall
    through to a fresh default SILENTLY (only file_utils' generic "corrupt JSON"
    WARNING). This guard exists specifically to throttle a strategy that is ON A
    LOSING STREAK; a silent reset wipes consecutive_losses + the up-to-8× cooldown
    escalation memory, and the very next record_trade() _save_state overwrites the
    corrupt file with zeros — the escalation history is gone with no trace. Mirror
    the portfolio_mgr 2026-06-01 hardening: when the file EXISTS but parses to None
    (corruption, not just missing), quarantine the bytes to a .corrupt sidecar and
    append a critical_errors row BEFORE starting fresh.
    """
    loaded = load_json(str(STATE_FILE), default=None)
    if loaded is not None:
        return loaded

    # default=None came back: either the file is missing (normal first run) or
    # it exists but is unparseable (corruption — surface + preserve loudly).
    if STATE_FILE.exists():
        try:
            corrupt_bytes = STATE_FILE.read_bytes()
        except OSError:
            corrupt_bytes = b""
        if corrupt_bytes.strip():
            _quarantine_corrupt_state(corrupt_bytes)

    return _default_state()


def _quarantine_corrupt_state(corrupt_bytes: bytes) -> None:
    """Preserve a corrupt guard-state file + surface it before fresh defaults
    overwrite it. Never raises — evidence preservation must not crash the read
    path (mirrors portfolio_mgr._quarantine_corrupt_state, 2026-06-01)."""
    try:
        qpath = STATE_FILE.with_name(f"{STATE_FILE.name}.corrupt")
        try:
            qpath.write_bytes(corrupt_bytes)
        except OSError as e:
            logger.warning("trade_guards: could not write quarantine %s: %s", qpath.name, e)
        logger.warning(
            "trade_guards: %s is corrupt — quarantined to %s and reset to fresh "
            "state. Loss-escalation / cooldown memory was lost.",
            STATE_FILE.name, qpath.name,
        )
        try:
            from portfolio.claude_gate import record_critical_error
            record_critical_error(
                category="trade_guard_state_corrupt",
                caller="portfolio.trade_guards",
                message=(
                    f"{STATE_FILE.name} unparseable — quarantined to {qpath.name} and "
                    f"reset to fresh state. Consecutive-loss counts and the up-to-8× "
                    f"cooldown escalation were wiped; overtrading guards restart from "
                    f"zero until trades re-accumulate."
                ),
                context={"path": str(STATE_FILE), "quarantine": str(qpath), "bytes": len(corrupt_bytes)},
            )
        except Exception:  # noqa: BLE001 — surfacing is best-effort
            logger.exception("trade_guards: failed to record critical_errors entry")
    except Exception:  # noqa: BLE001
        logger.exception("trade_guards: quarantine of corrupt state failed")


def _save_state(state):
    """Persist trade guard state to disk."""
    atomic_write_json(STATE_FILE, state)


def _portfolios_have_transactions():
    """Return True if any portfolio file has at least one recorded transaction.

    Used by the C4 sanity check to distinguish "no trades happened yet"
    (quiet startup state) from "trades happened but weren't recorded"
    (broken wiring — real bug).

    2026-04-22 follow-up: include warrants portfolio — CLAUDE.md lists it as
    an independent strategy state file, and warrants-only activity would
    have left C4 silent forever.
    """
    for pf_name in (
        "portfolio_state.json",
        "portfolio_state_bold.json",
        "portfolio_state_warrants.json",
    ):
        pf = load_json(str(DATA_DIR / pf_name), default={})
        if pf and pf.get("transactions"):
            return True
    return False


def _get_cooldown_multiplier(consecutive_losses, last_loss_ts_str=None):
    """Get cooldown multiplier based on consecutive loss count with time decay.

    After LOSS_DECAY_HOURS without a new trade, the multiplier halves
    repeatedly (geometric decay). E.g. 8x → 4x after 24h → 2x after 48h → 1x.
    """
    if consecutive_losses >= 4:
        base = LOSS_ESCALATION[4]
    else:
        base = LOSS_ESCALATION.get(consecutive_losses, 1)

    if base <= 1 or not last_loss_ts_str:
        return base

    # Apply time-based decay
    try:
        last_loss = datetime.fromisoformat(
            last_loss_ts_str.replace("Z", "+00:00")
        )
        if last_loss.tzinfo is None:
            last_loss = last_loss.replace(tzinfo=UTC)
        elapsed_hours = (datetime.now(UTC) - last_loss).total_seconds() / 3600
        if elapsed_hours > LOSS_DECAY_HOURS:
            halvings = int(elapsed_hours // LOSS_DECAY_HOURS)
            base = max(1, base >> halvings)  # bit-shift right = halve
    except (ValueError, TypeError, OverflowError):
        pass

    return base


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

    # 2026-06-11 (audit batch 7): does the strategy currently hold this ticker?
    # Used to (a) let SELLs that reduce/close an existing position bypass the
    # cooldown + loss-escalation blocks, and (b) treat a BUY into an already-held
    # ticker as a scale-in, not a "new position", for the rate limit. The
    # `portfolio` arg was previously threaded through get_all_guard_warnings and
    # never used — this wires it.
    holdings = (portfolio or {}).get("holdings", {}) if isinstance(portfolio, dict) else {}
    held_shares = 0
    _h = holdings.get(ticker) if isinstance(holdings, dict) else None
    if isinstance(_h, dict):
        try:
            held_shares = float(_h.get("shares", 0) or 0)
        except (TypeError, ValueError):
            held_shares = 0
    currently_holds = held_shares > 0
    # A SELL while holding is a position-reducing EXIT. Cooldown + loss-escalation
    # exist to PREVENT OVERTRADING (re-entering), not to lock a strategy into a
    # losing position: blocking exits hardest right after losses (the 8× / 240min
    # escalation fires exactly when the position is underwater) inverts risk
    # management. Exits therefore skip Guard 1 + Guard 2 entirely; entries (BUY,
    # and any SELL with no position to reduce, e.g. a short — none today) keep the
    # original block semantics unchanged.
    is_exit = action == "SELL" and currently_holds

    warnings = []
    with _state_lock:
        state = _load_state()
    now = datetime.now(UTC)

    # --- Guard 1: Per-ticker cooldown (ENTRIES only) ---
    base_cooldown = cfg.get("ticker_cooldown_minutes", DEFAULT_TICKER_COOLDOWN_MINUTES)
    consecutive = state.get("consecutive_losses", {}).get(strategy, 0)
    last_loss_ts = state.get("last_loss_ts", {}).get(strategy)
    multiplier = _get_cooldown_multiplier(consecutive, last_loss_ts)
    effective_cooldown = base_cooldown * multiplier

    key = f"{strategy}:{ticker}"
    ticker_trades = state.get("ticker_trades", {})
    last_trade_str = ticker_trades.get(key)
    if last_trade_str and not is_exit:
        try:
            last_trade = datetime.fromisoformat(last_trade_str)
            # M8: ensure aware datetime before comparison with aware now
            if last_trade.tzinfo is None:
                last_trade = last_trade.replace(tzinfo=UTC)
            elapsed = (now - last_trade).total_seconds() / 60
            if elapsed < effective_cooldown:
                remaining = effective_cooldown - elapsed
                warnings.append({
                    "guard": "ticker_cooldown",
                    "severity": "block",
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
        except (ValueError, TypeError) as e:
            logger.warning("trade_guards: corrupt timestamp in cooldown check for %s: %s", ticker, e)

    # --- Guard 2: Consecutive-loss escalation (informational, ENTRIES only) ---
    if consecutive >= 2 and not is_exit:
        base_mult = _get_cooldown_multiplier(consecutive, None)
        warnings.append({
            "guard": "consecutive_losses",
            "severity": "warning",
            "message": (
                f"{strategy}: {consecutive} consecutive losses. "
                f"Cooldown multiplier: {multiplier}x"
                f"{f' (decayed from {base_mult}x)' if multiplier < base_mult else ''}."
            ),
            "details": {
                "strategy": strategy,
                "consecutive_losses": consecutive,
                "multiplier": multiplier,
                "base_multiplier": base_mult,
                "decayed": multiplier < base_mult,
            },
        })

    # --- Guard 3: New-position rate limit (BUY opening a NEW position only) ---
    # 2026-06-11 (audit batch 7): a BUY into an already-held ticker is a scale-in,
    # not a new position, so it must not consume the new-position budget (with
    # patient_position_limit=1/8h a single add-on used to exhaust the window).
    # `currently_holds` comes from the portfolio arg; record_trade is told the
    # same so it only stamps new_position_timestamps for genuine opens.
    if action == "BUY" and not currently_holds:
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
                # M8: ensure aware datetime before comparison with aware cutoff
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                if ts >= cutoff:
                    recent.append(ts)
            except (ValueError, TypeError):
                continue

        if len(recent) >= limit:
            warnings.append({
                "guard": "position_rate_limit",
                "severity": "block",
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


_wiring_confirmed = False  # process-scoped flag — positive proof for C4


def record_trade(ticker, direction, strategy, pnl_pct=None, config=None, is_new_position=True):
    """Record a completed trade for guard tracking.

    Call this after executing a trade to update cooldowns and loss streaks.

    Args:
        ticker: Instrument ticker.
        direction: "BUY" or "SELL".
        strategy: "patient" or "bold".
        pnl_pct: Realized P&L percentage (for SELL trades). None for BUY.
        config: Optional config dict.
        is_new_position: 2026-06-11 (audit batch 7) — only stamp the
            new-position rate-limit clock when this BUY genuinely OPENED a
            position (ticker not already held before the buy). Scale-ins pass
            False so they don't consume the new-position budget. Defaults True
            for back-compat with callers that can't tell (treated as an open).
    """
    # 2026-04-22 follow-up: positive-proof wiring check. The previous C4
    # warning was *reactive* — it could only tell you after a trade had
    # already slipped through unguarded. Log INFO once per process the first
    # time this function fires, so operators get explicit confirmation the
    # BUG-219/PR-R4-4 wiring is alive rather than having to infer it from
    # absence-of-warnings.
    global _wiring_confirmed
    if not _wiring_confirmed:
        logger.info(
            "C4: record_trade() wiring confirmed — first call this process "
            "(ticker=%s direction=%s strategy=%s)",
            ticker, direction, strategy,
        )
        _wiring_confirmed = True

    with _state_lock:
        state = _load_state()
        now = datetime.now(UTC)
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
            if "last_loss_ts" not in state:
                state["last_loss_ts"] = {"patient": None, "bold": None}
            if pnl_pct < 0:
                state["consecutive_losses"][strategy] = (
                    state["consecutive_losses"].get(strategy, 0) + 1
                )
                state["last_loss_ts"][strategy] = now_str
            else:
                state["consecutive_losses"][strategy] = 0
                state["last_loss_ts"][strategy] = None

        # Prune stale ticker_trades entries (>90 days old) to prevent
        # unbounded growth. new_position_timestamps has its own 24h pruning
        # below; ticker_trades had none.
        _prune_cutoff = now - timedelta(days=90)
        _ticker_trades = state.get("ticker_trades", {})
        _stale_keys = []
        for _k, _v in _ticker_trades.items():
            try:
                _dt = datetime.fromisoformat(_v)
                if _dt.tzinfo is None:
                    _dt = _dt.replace(tzinfo=UTC)
                if _dt < _prune_cutoff:
                    _stale_keys.append(_k)
            except (ValueError, TypeError):
                _stale_keys.append(_k)
        for _k in _stale_keys:
            del _ticker_trades[_k]

        # Track new position timestamps (BUY that OPENS a new position only —
        # scale-ins into an already-held ticker pass is_new_position=False and
        # must not consume the new-position rate-limit budget). 2026-06-11.
        if direction == "BUY" and is_new_position:
            if "new_position_timestamps" not in state:
                state["new_position_timestamps"] = {"patient": [], "bold": []}
            if strategy not in state["new_position_timestamps"]:
                state["new_position_timestamps"][strategy] = []
            state["new_position_timestamps"][strategy].append(now_str)

            # Prune old timestamps (keep last 24h).
            cutoff = now - timedelta(hours=24)
            pruned = []
            for ts in state["new_position_timestamps"][strategy]:
                try:
                    dt = datetime.fromisoformat(ts)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=UTC)
                    if dt >= cutoff:
                        pruned.append(ts)
                except (ValueError, TypeError):
                    continue
            state["new_position_timestamps"][strategy] = pruned

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

    # C4: Detect broken record_trade() wiring.
    # 2026-04-22: original check fired every cycle whenever state was empty,
    # even when no trades had happened yet (portfolios untouched) — noisy and
    # misleading post-BUG-219/PR-R4-4 which wired _record_new_trades().
    # Now only warn when portfolios DO have transactions but guard state is
    # still empty — that's the real signal the wiring is broken.
    with _state_lock:
        state = _load_state()
    if not _wiring_confirmed and not state.get("ticker_trades") and all_warnings == [] and _portfolios_have_transactions():
        logger.warning(
            "C4: portfolios have transactions but trade_guard_state.json "
            "has no recorded trades — record_trade() wiring appears broken. "
            "Overtrading guards are NON-FUNCTIONAL."
        )

    return {
        "warnings": all_warnings,
        "summary": "; ".join(summary_parts) if summary_parts else "All clear",
    }


def should_block_trade(guard_result):
    """Check if any guard warning has 'block' severity.

    ARCH-29: Convenience function for Layer 2 go/no-go decisions.

    Args:
        guard_result: Return value from get_all_guard_warnings().

    Returns:
        True if any warning has severity="block", False otherwise.
    """
    warnings = guard_result.get("warnings", [])
    return any(w.get("severity") == "block" for w in warnings)
