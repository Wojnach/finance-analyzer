"""Kelly-optimal position sizing for metals warrant trading.

Computes leverage-adjusted Kelly fraction using real signal accuracy data
and historical outcome statistics from signal_log.db. Designed to replace
fixed position sizing (30% of buying power) with edge-aware sizing.

Usage::

    from portfolio.kelly_metals import recommended_metals_size

    rec = recommended_metals_size(
        ticker="XAG-USD",
        leverage=5.0,
        buying_power_sek=5000,
        consecutive_losses=0,
    )
    # rec["position_sek"]   -> how much to allocate (SEK)
    # rec["units"]          -> how many cert units at current ask
    # rec["kelly_pct"]      -> full Kelly fraction (0-1)
    # rec["half_kelly_pct"] -> half Kelly (recommended)
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

from portfolio.file_utils import load_json
from portfolio.kelly_sizing import kelly_fraction

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ACCURACY_CACHE = DATA_DIR / "accuracy_cache.json"
SIGNAL_DB = DATA_DIR / "signal_log.db"
AGENT_SUMMARY = DATA_DIR / "agent_summary.json"

# Defaults when no historical data is available
_DEFAULT_AVG_WIN = {"XAG-USD": 3.09, "XAU-USD": 2.10}
_DEFAULT_AVG_LOSS = {"XAG-USD": 2.43, "XAU-USD": 1.80}
_DEFAULT_WIN_RATE = 0.52

# Conservative sizing limits
MIN_TRADE_SEK = 500.0
MAX_POSITION_FRACTION = 0.95  # never go above 95% of buying power
LOSS_REDUCTION_STEP = 0.25    # reduce Kelly by 25% per consecutive loss


def _get_ticker_accuracy(ticker: str) -> float | None:
    """Read per-ticker consensus accuracy from accuracy_cache.json."""
    cache = load_json(str(ACCURACY_CACHE), default={})
    per_ticker = cache.get("per_ticker_consensus", {})
    entry = per_ticker.get(ticker, {})
    if not isinstance(entry, dict):
        return None
    total = entry.get("total", 0)
    if total < 30:
        return None
    acc = entry.get("accuracy")
    if acc is not None and 0 < acc < 1:
        return acc
    return None


def _get_outcome_stats(ticker: str, horizon: str = "1d") -> dict | None:
    """Compute win rate and avg win/loss from signal_log.db outcomes.

    Joins ticker_signals (consensus BUY/SELL) with outcomes to measure
    how well the consensus predicted direction over the given horizon.

    Returns:
        dict with win_rate, avg_win_pct, avg_loss_pct, n_trades
        or None if insufficient data.
    """
    if not SIGNAL_DB.exists():
        return None

    try:
        conn = sqlite3.connect(str(SIGNAL_DB))
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ts.consensus, o.change_pct
            FROM ticker_signals ts
            JOIN outcomes o ON o.snapshot_id = ts.snapshot_id
                           AND o.ticker = ts.ticker
            WHERE ts.ticker = ? AND o.horizon = ?
              AND ts.consensus IN ('BUY', 'SELL')
            """,
            (ticker, horizon),
        )
        rows = cur.fetchall()
        conn.close()
    except Exception:
        return None

    if len(rows) < 30:
        return None

    wins: list[float] = []
    losses: list[float] = []
    for consensus, change_pct in rows:
        correct = (consensus == "BUY" and change_pct > 0) or (
            consensus == "SELL" and change_pct < 0
        )
        if correct:
            wins.append(abs(change_pct))
        else:
            losses.append(abs(change_pct))

    total = len(wins) + len(losses)
    if total == 0:
        return None

    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    return {
        "win_rate": len(wins) / total,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "n_trades": total,
    }


def _loss_reduction(consecutive_losses: int) -> float:
    """Reduce position size after consecutive losses.

    Returns a multiplier in (0, 1]. After 4+ losses, position is 0 (sit out).
    """
    if consecutive_losses <= 0:
        return 1.0
    reduction = 1.0 - consecutive_losses * LOSS_REDUCTION_STEP
    return max(0.0, reduction)


def recommended_metals_size(
    ticker: str = "XAG-USD",
    leverage: float = 5.0,
    buying_power_sek: float = 0.0,
    ask_price_sek: float = 0.0,
    consecutive_losses: int = 0,
    agent_summary: dict | None = None,
    horizon: str = "1d",
) -> dict:
    """Compute Kelly-optimal position size for a metals warrant trade.

    Args:
        ticker: Underlying ticker (e.g. "XAG-USD").
        leverage: Certificate leverage (e.g. 5.0 for BULL SILVER X5).
        buying_power_sek: Available cash on Avanza.
        ask_price_sek: Current ask price of the certificate (SEK).
        consecutive_losses: Number of consecutive losses (reduces sizing).
        agent_summary: Optional agent_summary dict for signal-based win prob.
        horizon: Outcome horizon for accuracy stats ("1d", "3h").

    Returns:
        dict with sizing recommendation and all computation details.
    """
    # --- Step 1: Estimate win probability ---
    source_parts: list[str] = []
    win_rate = None

    # Try per-ticker accuracy from cache
    ticker_acc = _get_ticker_accuracy(ticker)
    if ticker_acc is not None:
        win_rate = ticker_acc
        source_parts.append(f"per_ticker_consensus ({win_rate:.1%})")

    # Try outcome stats from SQLite
    outcome_stats = _get_outcome_stats(ticker, horizon)
    if outcome_stats is not None:
        db_win_rate = outcome_stats["win_rate"]
        if win_rate is not None:
            # Blend: 60% DB (more data), 40% cache (includes recent)
            win_rate = 0.6 * db_win_rate + 0.4 * win_rate
            source_parts.append(f"blended with DB ({db_win_rate:.1%}, n={outcome_stats['n_trades']})")
        else:
            win_rate = db_win_rate
            source_parts.append(f"signal_log.db ({win_rate:.1%}, n={outcome_stats['n_trades']})")

    # Try agent_summary weighted confidence
    if win_rate is None and agent_summary:
        signals = agent_summary.get("signals", {})
        ticker_data = signals.get(ticker, {})
        if isinstance(ticker_data, dict):
            wc = ticker_data.get("weighted_confidence")
            if wc is not None and 0 < wc < 1:
                win_rate = wc
                source_parts.append(f"weighted_confidence ({win_rate:.1%})")

    # Fallback
    if win_rate is None:
        win_rate = _DEFAULT_WIN_RATE
        source_parts.append(f"default ({win_rate:.1%})")

    # --- Step 2: Estimate avg win/loss (underlying %) ---
    if outcome_stats and outcome_stats["avg_win_pct"] > 0 and outcome_stats["avg_loss_pct"] > 0:
        avg_win = outcome_stats["avg_win_pct"]
        avg_loss = outcome_stats["avg_loss_pct"]
        source_parts.append(f"outcome W={avg_win:.2f}% L={avg_loss:.2f}%")
    else:
        avg_win = _DEFAULT_AVG_WIN.get(ticker, 3.0)
        avg_loss = _DEFAULT_AVG_LOSS.get(ticker, 2.5)
        source_parts.append(f"default W={avg_win:.2f}% L={avg_loss:.2f}%")

    # --- Step 3: Compute Kelly fraction (on underlying) ---
    full_kelly = kelly_fraction(win_rate, avg_win, avg_loss)
    half_kelly = full_kelly / 2.0

    # --- Step 4: Convert to leveraged position fraction ---
    # Kelly says risk X% of capital. With leverage L, a loss of avg_loss%
    # on underlying = avg_loss * L % on the certificate.
    # Position fraction = half_kelly / (avg_loss * leverage / 100)
    cert_loss_frac = avg_loss * leverage / 100.0
    if cert_loss_frac > 0:
        position_fraction = half_kelly / cert_loss_frac
    else:
        position_fraction = 0.0

    position_fraction = min(position_fraction, MAX_POSITION_FRACTION)

    # --- Step 5: Apply consecutive loss reduction ---
    loss_mult = _loss_reduction(consecutive_losses)
    adjusted_fraction = position_fraction * loss_mult

    # --- Step 6: Compute final SEK allocation ---
    position_sek = buying_power_sek * adjusted_fraction
    if position_sek < MIN_TRADE_SEK:
        position_sek = 0.0

    # --- Step 7: Compute units ---
    units = 0
    if ask_price_sek > 0 and position_sek > 0:
        units = int(position_sek / ask_price_sek)
        if units <= 0:
            position_sek = 0.0

    # --- Step 8: Expected growth rate (daily log-growth at half-Kelly) ---
    cert_win_frac = avg_win * leverage / 100.0
    f = adjusted_fraction
    if f > 0 and cert_loss_frac > 0:
        daily_log_growth = (
            win_rate * math.log(1 + f * cert_win_frac)
            + (1 - win_rate) * math.log(max(1e-10, 1 - f * cert_loss_frac))
        )
    else:
        daily_log_growth = 0.0

    monthly_growth = math.exp(daily_log_growth * 22) - 1 if daily_log_growth > 0 else 0.0

    return {
        # Sizing
        "position_sek": round(position_sek, 0),
        "position_fraction": round(adjusted_fraction, 4),
        "units": units,
        # Kelly components
        "kelly_pct": round(full_kelly, 4),
        "half_kelly_pct": round(half_kelly, 4),
        "position_fraction_raw": round(position_fraction, 4),
        # Inputs
        "win_rate": round(win_rate, 4),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "leverage": leverage,
        "consecutive_losses": consecutive_losses,
        "loss_multiplier": round(loss_mult, 2),
        # Growth projections
        "daily_log_growth": round(daily_log_growth, 6),
        "monthly_growth_pct": round(monthly_growth * 100, 1),
        # Provenance
        "source": " | ".join(source_parts),
    }


def format_kelly_line(rec: dict) -> str:
    """One-line summary for Telegram notifications."""
    k = rec["half_kelly_pct"] * 100
    wr = rec["win_rate"] * 100
    pos = rec["position_sek"]
    lm = rec["loss_multiplier"]
    parts = [f"Kelly:{k:.1f}% WR:{wr:.0f}% Pos:{pos:.0f}kr"]
    if lm < 1.0:
        parts.append(f"(x{lm:.2f} loss adj)")
    return " ".join(parts)
