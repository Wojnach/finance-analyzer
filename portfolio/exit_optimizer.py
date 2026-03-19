"""Quant exit engine — probabilistic exit optimization for intraday positions.

Three-layer architecture:
1. **Opportunity layer**: Monte Carlo path simulation for remaining-session
   price distribution (max/min/terminal).
2. **Execution layer**: Fill probability and time-to-hit estimation from
   simulated paths.
3. **Decision layer**: EV ranking of candidate exits, net of costs, with
   risk overrides (knock-out proximity, session end, volatility shock).

Designed for Avanza MINI futures (gold/silver warrants) but works for any
instrument with price, volatility, and session data.

Usage:
    from portfolio.exit_optimizer import compute_exit_plan, Position, MarketSnapshot
    plan = compute_exit_plan(position, market, session_end, cost_model)
    print(plan.recommended)  # Best exit by EV

Reference: docs/deep research/deep-research-report.md
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from portfolio.cost_model import CostModel, get_cost_model

logger = logging.getLogger("portfolio.exit_optimizer")

# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketSnapshot:
    """Current market state for the position's instrument.

    Prices are in the underlying's native currency (USD for metals/crypto).
    """
    asof_ts: datetime
    price: float              # Current underlying price (USD)
    bid: float | None = None
    ask: float | None = None
    volatility_annual: float | None = None   # Annualized vol (decimal)
    atr_pct: float | None = None             # ATR% for vol estimation
    usdsek: float = 10.85                       # FX rate
    drift: float = 0.0                          # Annualized drift (0 = neutral)


@dataclass(frozen=True)
class Position:
    """A held position to evaluate for exit.

    For warrants: prices in SEK, with underlying in USD.
    For stocks/crypto: prices in USD.
    """
    symbol: str                          # Underlying ticker (e.g., "XAG-USD")
    qty: float                           # Units held
    entry_price_sek: float               # What we paid per unit (SEK)
    entry_underlying_usd: float          # Underlying price at entry (USD)
    entry_ts: datetime
    instrument_type: str = "warrant"     # "warrant", "stock", "crypto"
    leverage: float = 1.0                # Effective leverage at entry
    financing_level: float | None = None  # MINI future financing level (USD)
    trailing_peak_usd: float | None = None  # Highest underlying since entry


@dataclass(frozen=True)
class CandidateExit:
    """A ranked exit candidate with probabilistic assessment.

    Attributes:
        price_usd: Target exit price in underlying USD.
        action: Exit method — "limit", "market", "hold_to_close".
        fill_prob: P(price reaches target before session end), 0.0-1.0.
        expected_fill_time_min: E[time to hit target | hit], in minutes.
        pnl_sek: Net P&L if filled at target price (after costs).
        ev_sek: Expected value = fill_prob × pnl + (1-fill_prob) × fallback.
        pnl_pct: P&L as percentage of position value.
        risk_flags: List of active risk warnings.
        quantile: Which quantile of session-max this candidate represents.
    """
    price_usd: float
    action: str
    fill_prob: float
    expected_fill_time_min: float
    pnl_sek: float
    ev_sek: float
    pnl_pct: float
    risk_flags: tuple[str, ...] = ()
    quantile: float | None = None


@dataclass
class ExitPlan:
    """Complete exit plan with ranked candidates.

    Attributes:
        symbol: Underlying ticker.
        asof_ts: When this plan was computed.
        remaining_minutes: Minutes until session close.
        candidates: All evaluated exit candidates, sorted by EV descending.
        recommended: The top candidate (highest EV, respecting risk overrides).
        market_exit: Immediate market exit candidate (always available).
        session_max_distribution: Quantiles of the remaining-session max price.
        session_min_distribution: Quantiles of the remaining-session min price.
        stop_hit_prob: P(price drops to stop level before session end).
        provenance: Audit trail (model version, parameters, data sources).
    """
    symbol: str
    asof_ts: datetime
    remaining_minutes: float
    candidates: list[CandidateExit]
    recommended: CandidateExit
    market_exit: CandidateExit
    session_max_distribution: dict[str, float] = field(default_factory=dict)
    session_min_distribution: dict[str, float] = field(default_factory=dict)
    stop_hit_prob: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line summary for logging/Telegram."""
        rec = self.recommended
        mkt = self.market_exit
        return (
            f"Exit plan: recommended {rec.action} @ ${rec.price_usd:.2f} "
            f"(EV {rec.ev_sek:+,.0f} SEK, fill {rec.fill_prob:.0%}, "
            f"{rec.expected_fill_time_min:.0f}min) | "
            f"market exit {mkt.pnl_sek:+,.0f} SEK | "
            f"{self.remaining_minutes:.0f}min left"
        )

    def to_dict(self) -> dict:
        """Serialize for JSON (agent_summary integration)."""
        return {
            "symbol": self.symbol,
            "remaining_min": round(self.remaining_minutes),
            "recommended": {
                "price": round(self.recommended.price_usd, 2),
                "action": self.recommended.action,
                "fill_prob": round(self.recommended.fill_prob, 3),
                "ev_sek": round(self.recommended.ev_sek),
                "pnl_pct": round(self.recommended.pnl_pct, 2),
                "time_min": round(self.recommended.expected_fill_time_min),
                "risk_flags": list(self.recommended.risk_flags),
            },
            "market_exit_sek": round(self.market_exit.pnl_sek),
            "stop_hit_prob": round(self.stop_hit_prob, 3),
            "session_max": self.session_max_distribution,
            "session_min": self.session_min_distribution,
            "n_candidates": len(self.candidates),
        }


# ---------------------------------------------------------------------------
# Intraday Monte Carlo path engine
# ---------------------------------------------------------------------------

# Trading minutes per day by instrument type (for annualization)
_TRADING_MINUTES = {
    "warrant": 820,    # 08:15-21:55 CET = ~13.67h
    "stock": 390,      # 6.5h
    "crypto": 1440,    # 24h
}
_TRADING_DAYS_PER_YEAR = 252
_MIN_VOLATILITY = 0.05  # 5% annualized floor


def _estimate_volatility(market: MarketSnapshot) -> float:
    """Get annualized volatility from market snapshot."""
    if market.volatility_annual and market.volatility_annual > _MIN_VOLATILITY:
        return market.volatility_annual
    if market.atr_pct and market.atr_pct > 0:
        # Convert ATR% (14-period) to annualized vol
        atr_frac = market.atr_pct / 100.0
        return max(atr_frac * math.sqrt(252.0 / 14), _MIN_VOLATILITY)
    return 0.20  # Default 20% annual vol


def simulate_intraday_paths(
    price: float,
    volatility: float,
    drift: float,
    remaining_minutes: int,
    instrument_type: str = "warrant",
    n_paths: int = 5000,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate GBM price paths at 1-minute resolution.

    Uses antithetic variates for variance reduction (~50% lower variance).

    Args:
        price: Current underlying price (USD).
        volatility: Annualized volatility (decimal, e.g., 0.25 = 25%).
        drift: Annualized drift (decimal). 0 = neutral.
        remaining_minutes: Minutes until session close.
        instrument_type: For annualization ("warrant", "stock", "crypto").
        n_paths: Number of paths to simulate. Even number recommended.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_paths, remaining_minutes + 1) where column 0 is
        the current price and each subsequent column is 1 minute later.
    """
    rng = np.random.default_rng(seed)

    n_steps = max(1, int(remaining_minutes))
    min_per_year = _TRADING_MINUTES.get(instrument_type, 390) * _TRADING_DAYS_PER_YEAR
    dt = 1.0 / min_per_year  # 1 minute in annualized trading time

    vol = max(volatility, _MIN_VOLATILITY)
    drift_per_step = (drift - 0.5 * vol ** 2) * dt
    vol_per_step = vol * math.sqrt(dt)

    # Antithetic variates: generate half, mirror the other half
    n_half = n_paths // 2
    Z = rng.standard_normal((n_half, n_steps))
    Z_all = np.vstack([Z, -Z])

    # If odd n_paths, add one extra
    if n_paths % 2 == 1:
        extra = rng.standard_normal((1, n_steps))
        Z_all = np.vstack([Z_all, extra])

    # Log-return increments → cumulative → price paths
    log_inc = drift_per_step + vol_per_step * Z_all  # (n_paths, n_steps)
    log_cum = np.cumsum(log_inc, axis=1)

    # Prepend zero column (current price)
    zeros = np.zeros((Z_all.shape[0], 1))
    log_paths = np.hstack([zeros, log_cum])

    paths = price * np.exp(log_paths)
    return paths


def _path_statistics(paths: np.ndarray) -> dict:
    """Extract key statistics from simulated paths.

    Returns:
        Dict with session_max, session_min, terminal arrays and quantile dicts.
    """
    session_max = np.max(paths[:, 1:], axis=1)  # Exclude t=0
    session_min = np.min(paths[:, 1:], axis=1)
    terminal = paths[:, -1]

    quantiles = [5, 10, 20, 35, 50, 65, 80, 90, 95]
    max_q = {f"p{q}": round(float(v), 4)
             for q, v in zip(quantiles, np.percentile(session_max, quantiles))}
    min_q = {f"p{q}": round(float(v), 4)
             for q, v in zip(quantiles, np.percentile(session_min, quantiles))}

    return {
        "session_max": session_max,
        "session_min": session_min,
        "terminal": terminal,
        "max_quantiles": max_q,
        "min_quantiles": min_q,
    }


def _first_hit_times(paths: np.ndarray, target: float, direction: str = "above") -> np.ndarray:
    """Compute first passage time for each path to reach target.

    Args:
        paths: Price paths, shape (n_paths, n_steps+1).
        target: Price level to hit.
        direction: "above" (sell target) or "below" (stop level).

    Returns:
        Array of shape (n_paths,). Values are minute indices (1-based).
        -1 means the path never hit the target.
    """
    if direction == "above":
        hits = paths[:, 1:] >= target
    else:
        hits = paths[:, 1:] <= target

    # argmax on axis=1 returns first True index (0-based in the sliced array)
    first_idx = np.argmax(hits, axis=1)

    # Distinguish never-hit: if first_idx=0 but that cell isn't True → never hit
    never_hit = ~np.any(hits, axis=1)
    result = first_idx + 1  # Convert to 1-based minute index
    result[never_hit] = -1

    return result


# ---------------------------------------------------------------------------
# P&L computation
# ---------------------------------------------------------------------------

def _compute_pnl_sek(
    position: Position,
    exit_price_usd: float,
    market: MarketSnapshot,
    costs: CostModel,
) -> float:
    """Compute net P&L in SEK for exiting at given underlying price.

    For warrants (MINI futures):
        warrant_value = (underlying - financing_level) × usdsek
        pnl = (exit_value - entry_value) × qty - costs

    For stocks/crypto:
        pnl = (exit_price - entry_price) × qty × usdsek - costs
    """
    fx = market.usdsek

    if position.instrument_type == "warrant" and position.financing_level is not None:
        # MINI future: warrant price = (underlying - financing_level) × fx
        exit_warrant_sek = (exit_price_usd - position.financing_level) * fx
        exit_warrant_sek = max(exit_warrant_sek, 0)  # Can't go below 0 (knock-out)
        exit_value = exit_warrant_sek * position.qty
        entry_value = position.entry_price_sek * position.qty
    elif position.instrument_type == "warrant":
        # Leveraged product without explicit financing level
        pct_move = (exit_price_usd - position.entry_underlying_usd) / position.entry_underlying_usd
        warrant_move = pct_move * position.leverage
        exit_warrant_sek = position.entry_price_sek * (1 + warrant_move)
        exit_warrant_sek = max(exit_warrant_sek, 0)
        exit_value = exit_warrant_sek * position.qty
        entry_value = position.entry_price_sek * position.qty
    else:
        # Direct position (stock/crypto)
        exit_value = position.qty * exit_price_usd * fx
        entry_value = position.qty * position.entry_underlying_usd * fx

    cost = costs.total_cost_sek(exit_value)
    return exit_value - entry_value - cost


def _pnl_pct(pnl_sek: float, position: Position) -> float:
    """P&L as percentage of initial investment."""
    entry_value = position.entry_price_sek * position.qty
    if entry_value <= 0:
        return 0.0
    return pnl_sek / entry_value * 100.0


# ---------------------------------------------------------------------------
# Risk flags
# ---------------------------------------------------------------------------

def _compute_risk_flags(
    target_price: float | None,
    position: Position,
    market: MarketSnapshot,
    remaining_minutes: float,
    session_max: np.ndarray | None = None,
    session_min: np.ndarray | None = None,
) -> list[str]:
    """Generate risk warnings for a candidate exit."""
    flags = []

    # 1. Session end proximity
    if remaining_minutes < 30:
        flags.append("SESSION_END_IMMINENT")
    elif remaining_minutes < 60:
        flags.append("SESSION_END_NEAR")

    # 2. Knock-out proximity (MINI futures)
    if position.financing_level and position.financing_level > 0:
        distance_pct = (market.price - position.financing_level) / market.price * 100
        if distance_pct < 3:
            flags.append("KNOCKOUT_DANGER")
        elif distance_pct < 8:
            flags.append("KNOCKOUT_WARNING")

    # 3. Target far from current price (low fill probability expected)
    if target_price and market.price > 0:
        target_distance_pct = abs(target_price - market.price) / market.price * 100
        if target_distance_pct > 5:
            flags.append("TARGET_DISTANT")

    # 4. Underlying session mismatch (warrant still trading but underlying closed)
    # This would be detected by session_calendar, passed as a flag

    # 5. Position aging
    if position.entry_ts:
        hold_hours = (market.asof_ts - position.entry_ts).total_seconds() / 3600
        if hold_hours > 5:
            flags.append("HOLD_TIME_EXTENDED")

    # 6. Stop-loss proximity from MC paths
    if session_min is not None and position.financing_level:
        stop_buffer = position.financing_level * 1.03  # 3% above financing
        p_knockout = float(np.mean(session_min <= stop_buffer))
        if p_knockout > 0.10:
            flags.append(f"KNOCKOUT_PROB_{p_knockout:.0%}")

    return flags


# ---------------------------------------------------------------------------
# Risk overrides
# ---------------------------------------------------------------------------

def _apply_risk_overrides(
    candidates: list[CandidateExit],
    position: Position,
    market: MarketSnapshot,
    remaining_minutes: float,
    session_min: np.ndarray | None = None,
) -> CandidateExit:
    """Apply hard risk overrides and select recommended exit.

    Risk overrides can force a market exit even if EV says hold:
    - Knock-out danger (< 3% from financing level)
    - Session end imminent (< 5 min remaining)
    - Stop probability too high (> 25% chance of knock-out)
    """
    if not candidates:
        raise ValueError("No candidates to evaluate")

    # Find the market exit candidate
    market_exits = [c for c in candidates if c.action == "market"]
    market_exit = market_exits[0] if market_exits else candidates[-1]

    # Override 1: Knock-out danger → force market exit
    if position.financing_level and position.financing_level > 0:
        distance_pct = (market.price - position.financing_level) / market.price * 100
        if distance_pct < 3:
            logger.warning("RISK OVERRIDE: Knock-out danger (%.1f%% from barrier), "
                           "forcing market exit", distance_pct)
            return market_exit

    # Override 2: Session about to end → force market exit
    if remaining_minutes < 5:
        logger.info("RISK OVERRIDE: Session ending in %.0f min, forcing market exit",
                     remaining_minutes)
        return market_exit

    # Override 3: High knock-out probability → prefer market exit
    if session_min is not None and position.financing_level:
        stop_buffer = position.financing_level * 1.03
        p_knockout = float(np.mean(session_min <= stop_buffer))
        if p_knockout > 0.25:
            logger.warning("RISK OVERRIDE: %.0f%% knock-out probability, "
                           "forcing market exit", p_knockout * 100)
            return market_exit

    # No override triggered — return highest-EV candidate
    return candidates[0]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

DEFAULT_N_PATHS = 5000
DEFAULT_QUANTILES = [0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95]


def compute_exit_plan(
    position: Position,
    market: MarketSnapshot,
    session_end: datetime,
    costs: CostModel | None = None,
    *,
    n_paths: int = DEFAULT_N_PATHS,
    quantiles: list[float] | None = None,
    stop_price_usd: float | None = None,
    seed: int | None = None,
) -> ExitPlan:
    """Compute a full exit plan for a held position.

    This is the main function. It:
    1. Simulates remaining-session price paths (Monte Carlo GBM)
    2. Extracts session-max/min distributions
    3. Generates candidate exits at quantile levels of session max
    4. Computes fill probability, time-to-hit, and EV for each
    5. Adds market exit and hold-to-close baselines
    6. Ranks by EV and applies risk overrides

    Args:
        position: The held position to evaluate.
        market: Current market snapshot.
        session_end: UTC datetime of session close.
        costs: Cost model. If None, auto-selects by instrument type.
        n_paths: Number of Monte Carlo paths.
        quantiles: Quantile levels for candidate generation.
        stop_price_usd: Explicit stop level (for stop-hit probability).
        seed: Random seed for reproducibility.

    Returns:
        ExitPlan with ranked candidates and recommendation.
    """
    if costs is None:
        costs = get_cost_model(position.instrument_type)

    if quantiles is None:
        quantiles = DEFAULT_QUANTILES

    now = market.asof_ts
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    if session_end.tzinfo is None:
        session_end = session_end.replace(tzinfo=UTC)

    remaining_min = max(0, (session_end - now).total_seconds() / 60)

    # ---- Edge case: session over or almost over ----
    if remaining_min < 1:
        mkt_pnl = _compute_pnl_sek(position, market.bid or market.price, market, costs)
        mkt_candidate = CandidateExit(
            price_usd=market.bid or market.price,
            action="market",
            fill_prob=1.0,
            expected_fill_time_min=0,
            pnl_sek=mkt_pnl,
            ev_sek=mkt_pnl,
            pnl_pct=_pnl_pct(mkt_pnl, position),
            risk_flags=("SESSION_ENDED",),
        )
        return ExitPlan(
            symbol=position.symbol,
            asof_ts=now,
            remaining_minutes=0,
            candidates=[mkt_candidate],
            recommended=mkt_candidate,
            market_exit=mkt_candidate,
            provenance={"reason": "session_ended"},
        )

    # ---- 1. Estimate volatility and simulate paths ----
    vol = _estimate_volatility(market)
    drift = market.drift

    paths = simulate_intraday_paths(
        price=market.price,
        volatility=vol,
        drift=drift,
        remaining_minutes=int(remaining_min),
        instrument_type=position.instrument_type,
        n_paths=n_paths,
        seed=seed,
    )

    # ---- 2. Extract path statistics ----
    stats = _path_statistics(paths)
    session_max = stats["session_max"]
    session_min = stats["session_min"]
    terminal = stats["terminal"]

    # ---- 3. Generate candidate exits at session-max quantiles ----
    target_prices = np.quantile(session_max, quantiles)
    candidates: list[CandidateExit] = []

    # Fallback P&L: what we get if we hold to close (median terminal price)
    median_terminal = float(np.median(terminal))
    fallback_pnl = _compute_pnl_sek(position, median_terminal, market, costs)

    for q, target in zip(quantiles, target_prices):
        target = float(target)

        # Skip if target is below current price (can't profit from selling below)
        if target <= market.price * 0.999:
            continue

        # Fill probability: fraction of paths where max >= target
        fill_prob = float(np.mean(session_max >= target))

        # Time to hit
        hit_times = _first_hit_times(paths, target, direction="above")
        hitting_times = hit_times[hit_times > 0]
        expected_time = float(np.mean(hitting_times)) if len(hitting_times) > 0 else remaining_min

        # P&L if filled
        pnl = _compute_pnl_sek(position, target, market, costs)

        # Expected value: fill_prob × conditional_pnl + (1-fill_prob) × fallback
        ev = fill_prob * pnl + (1 - fill_prob) * fallback_pnl

        flags = _compute_risk_flags(target, position, market, remaining_min,
                                     session_max, session_min)

        candidates.append(CandidateExit(
            price_usd=round(target, 4),
            action="limit",
            fill_prob=round(fill_prob, 4),
            expected_fill_time_min=round(expected_time, 1),
            pnl_sek=round(pnl, 2),
            ev_sek=round(ev, 2),
            pnl_pct=round(_pnl_pct(pnl, position), 2),
            risk_flags=tuple(flags),
            quantile=q,
        ))

    # ---- 4. Market exit candidate (immediate fill, certain) ----
    bid = market.bid or market.price
    mkt_pnl = _compute_pnl_sek(position, bid, market, costs)
    market_candidate = CandidateExit(
        price_usd=round(bid, 4),
        action="market",
        fill_prob=1.0,
        expected_fill_time_min=0,
        pnl_sek=round(mkt_pnl, 2),
        ev_sek=round(mkt_pnl, 2),
        pnl_pct=round(_pnl_pct(mkt_pnl, position), 2),
        risk_flags=tuple(_compute_risk_flags(None, position, market, remaining_min)),
    )
    candidates.append(market_candidate)

    # ---- 5. Hold-to-close candidate ----
    # EV of holding = mean terminal P&L (expected value across all paths)
    terminal_pnls = np.array([
        _compute_pnl_sek(position, float(p), market, costs)
        for p in np.percentile(terminal, [10, 25, 50, 75, 90])
    ])
    hold_ev = float(np.mean(terminal_pnls))

    hold_candidate = CandidateExit(
        price_usd=round(median_terminal, 4),
        action="hold_to_close",
        fill_prob=1.0,
        expected_fill_time_min=round(remaining_min, 1),
        pnl_sek=round(fallback_pnl, 2),
        ev_sek=round(hold_ev, 2),
        pnl_pct=round(_pnl_pct(fallback_pnl, position), 2),
        risk_flags=tuple(_compute_risk_flags(None, position, market, remaining_min,
                                              session_max, session_min)),
    )
    candidates.append(hold_candidate)

    # ---- 6. Sort by EV descending ----
    candidates.sort(key=lambda c: c.ev_sek, reverse=True)

    # ---- 7. Stop-loss hit probability ----
    stop_prob = 0.0
    if stop_price_usd and stop_price_usd > 0:
        stop_prob = float(np.mean(session_min <= stop_price_usd))
    elif position.financing_level:
        # Use knock-out level + 3% buffer as effective stop
        stop_buffer = position.financing_level * 1.03
        stop_prob = float(np.mean(session_min <= stop_buffer))

    # ---- 8. Apply risk overrides to select recommendation ----
    recommended = _apply_risk_overrides(
        candidates, position, market, remaining_min, session_min
    )

    return ExitPlan(
        symbol=position.symbol,
        asof_ts=now,
        remaining_minutes=round(remaining_min, 1),
        candidates=candidates,
        recommended=recommended,
        market_exit=market_candidate,
        session_max_distribution=stats["max_quantiles"],
        session_min_distribution=stats["min_quantiles"],
        stop_hit_prob=round(stop_prob, 4),
        provenance={
            "model": "GBM_antithetic",
            "n_paths": n_paths,
            "volatility": round(vol, 4),
            "drift": round(drift, 4),
            "remaining_min": round(remaining_min),
            "instrument_type": position.instrument_type,
            "cost_model": costs.label,
        },
    )


# ---------------------------------------------------------------------------
# Convenience: compute exit plan from existing system data
# ---------------------------------------------------------------------------

def compute_exit_plan_from_summary(
    ticker: str,
    agent_summary: dict,
    position_state: dict,
    session_end: datetime,
    *,
    instrument_type: str = "warrant",
    financing_level: float | None = None,
    leverage: float = 1.0,
    n_paths: int = DEFAULT_N_PATHS,
) -> ExitPlan | None:
    """Build exit plan from agent_summary and portfolio state data.

    Convenience wrapper that extracts price, volatility, and position data
    from the standard system data structures.

    Args:
        ticker: Underlying ticker (e.g., "XAG-USD").
        agent_summary: Agent summary dict with signals and prices.
        position_state: Position dict with shares, avg_cost, entry info.
        session_end: Session close time (UTC).
        instrument_type: "warrant", "stock", "crypto".
        financing_level: For MINI futures, the knock-out level.
        leverage: Effective leverage.
        n_paths: MC paths.

    Returns:
        ExitPlan or None if insufficient data.
    """
    signals = agent_summary.get("signals", {})
    ticker_data = signals.get(ticker, {})
    if not ticker_data:
        return None

    price = ticker_data.get("price_usd", 0)
    if price <= 0:
        return None

    extra = ticker_data.get("extra", {})
    atr_pct = extra.get("atr_pct") or ticker_data.get("atr_pct")
    fx_rate = agent_summary.get("fx_rate", 10.85)

    # Build MarketSnapshot
    market = MarketSnapshot(
        asof_ts=datetime.now(UTC),
        price=price,
        atr_pct=atr_pct,
        usdsek=fx_rate,
    )

    # Build Position
    shares = position_state.get("shares", position_state.get("qty", 0))
    entry_price = position_state.get("entry_price_sek",
                                      position_state.get("entry_price", 0))
    entry_underlying = position_state.get("entry_underlying_usd",
                                           position_state.get("entry_underlying", price))
    entry_ts_str = position_state.get("entry_ts")
    entry_ts = datetime.now(UTC)
    if entry_ts_str:
        try:
            entry_ts = datetime.fromisoformat(entry_ts_str)
        except (ValueError, TypeError):
            pass

    position = Position(
        symbol=ticker,
        qty=shares,
        entry_price_sek=entry_price,
        entry_underlying_usd=entry_underlying,
        entry_ts=entry_ts,
        instrument_type=instrument_type,
        leverage=leverage,
        financing_level=financing_level,
    )

    return compute_exit_plan(position, market, session_end, n_paths=n_paths)
