"""Risk management utilities for portfolio intelligence system.

Provides:
- Maximum drawdown circuit breaker
- ATR-based trailing stop-loss tracking
- Position age tracking
- Portfolio value history logging
- Transaction cost analysis
"""

import datetime
import json
import pathlib

from portfolio.file_utils import load_json

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

INITIAL_VALUE_DEFAULT = 500_000  # SEK


def _compute_portfolio_value(portfolio: dict, agent_summary: dict) -> float:
    """Compute current total portfolio value in SEK.

    Value = cash_sek + sum(shares * current_price_sek) for each holding.
    current_price_sek is derived from agent_summary prices * fx_rate.
    """
    cash = portfolio.get("cash_sek", 0)
    holdings = portfolio.get("holdings", {})
    fx_rate = agent_summary.get("fx_rate", 1.0)
    signals = agent_summary.get("signals", {})

    holdings_value = 0.0
    for ticker, pos in holdings.items():
        shares = pos.get("shares", 0)
        if shares <= 0:
            continue
        # Try to get current price from agent_summary signals
        if ticker in signals:
            price_usd = signals[ticker].get("price_usd", 0)
            holdings_value += shares * price_usd * fx_rate
        else:
            # Fallback: use avg_cost_usd from holdings if no live price
            avg_cost = pos.get("avg_cost_usd", 0)
            holdings_value += shares * avg_cost * fx_rate

    return cash + holdings_value


def check_drawdown(portfolio_path: str, max_drawdown_pct: float = 20.0,
                   agent_summary_path: str | None = None) -> dict:
    """Check if portfolio has exceeded maximum drawdown threshold.

    Computes current portfolio value against the initial value and the peak
    value recorded in portfolio_value_history.jsonl (if available).

    Args:
        portfolio_path: Path to portfolio_state JSON file.
        max_drawdown_pct: Maximum allowed drawdown percentage (default 20%).
        agent_summary_path: Path to agent_summary.json for live prices.
            If None, uses DATA_DIR / "agent_summary.json".

    Returns:
        dict with:
            - breached: bool -- True if drawdown exceeds threshold
            - current_drawdown_pct: float -- current drawdown from peak (positive number)
            - peak_value: float -- highest portfolio value seen
            - current_value: float -- current portfolio value in SEK
            - initial_value: float -- starting portfolio value
    """
    portfolio = load_json(portfolio_path, default={})
    initial_value = portfolio.get("initial_value_sek", INITIAL_VALUE_DEFAULT)

    if agent_summary_path is None:
        agent_summary_path = str(DATA_DIR / "agent_summary.json")

    # If portfolio has no holdings, value is just cash
    if not portfolio.get("holdings"):
        current_value = portfolio.get("cash_sek", initial_value)
    else:
        summary = load_json(agent_summary_path, default={})
        if summary:
            current_value = _compute_portfolio_value(portfolio, summary)
        else:
            # Fallback: cash only (conservative estimate)
            current_value = portfolio.get("cash_sek", initial_value)

    # Determine peak value from history file or initial value
    peak_value = initial_value
    history_path = DATA_DIR / "portfolio_value_history.jsonl"
    pf_name = pathlib.Path(portfolio_path).stem  # e.g. "portfolio_state" or "portfolio_state_bold"
    is_bold = "bold" in pf_name

    if history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    value_key = "bold_value_sek" if is_bold else "patient_value_sek"
                    val = entry.get(value_key, 0)
                    if val > peak_value:
                        peak_value = val
        except (json.JSONDecodeError, IOError):
            pass

    # Also compare against current value in case it's a new peak
    if current_value > peak_value:
        peak_value = current_value

    # Calculate drawdown
    if peak_value > 0:
        current_drawdown_pct = ((peak_value - current_value) / peak_value) * 100
    else:
        current_drawdown_pct = 0.0

    return {
        "breached": current_drawdown_pct > max_drawdown_pct,
        "current_drawdown_pct": round(current_drawdown_pct, 4),
        "peak_value": round(peak_value, 2),
        "current_value": round(current_value, 2),
        "initial_value": initial_value,
    }


def compute_stop_levels(holdings: dict, agent_summary: dict) -> dict:
    """Compute ATR-based stop-loss levels for all positions.

    For each holding with shares > 0, calculates stop-loss levels based on
    2x ATR (Average True Range) from the entry price.

    Args:
        holdings: The "holdings" dict from portfolio state.
            Each entry: {ticker: {"shares": N, "avg_cost_usd": X, ...}}
        agent_summary: Parsed agent_summary.json dict.

    Returns:
        dict keyed by ticker, each with:
            - entry_price_usd: float (avg_cost_usd)
            - current_price_usd: float (from agent_summary)
            - atr_pct: float (from agent_summary)
            - stop_price_usd: float (entry_price * (1 - 2 * atr_pct/100))
            - triggered: bool (current_price < stop_price)
            - distance_to_stop_pct: float (positive = above stop, negative = below)
            - pnl_pct: float (current vs entry)
    """
    signals = agent_summary.get("signals", {})
    result = {}

    for ticker, pos in holdings.items():
        shares = pos.get("shares", 0)
        if shares <= 0:
            continue

        entry_price = pos.get("avg_cost_usd", 0)
        if entry_price <= 0:
            continue

        # Get current price and ATR from agent_summary
        if ticker not in signals:
            # Ticker not in current summary (e.g., stock after hours)
            result[ticker] = {
                "entry_price_usd": entry_price,
                "current_price_usd": None,
                "atr_pct": None,
                "stop_price_usd": None,
                "triggered": False,
                "distance_to_stop_pct": None,
                "pnl_pct": None,
                "note": "No live data available (market closed or ticker not in summary)",
            }
            continue

        sig = signals[ticker]
        current_price = sig.get("price_usd", 0)
        atr_pct = sig.get("atr_pct", 0)

        # 2x ATR stop-loss
        stop_price = entry_price * (1 - 2 * atr_pct / 100)

        # Distance from current price to stop
        if stop_price > 0:
            distance_to_stop_pct = ((current_price - stop_price) / stop_price) * 100
        else:
            distance_to_stop_pct = float("inf")

        triggered = current_price < stop_price if current_price > 0 else False
        pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0

        result[ticker] = {
            "entry_price_usd": round(entry_price, 6),
            "current_price_usd": round(current_price, 6),
            "atr_pct": round(atr_pct, 4),
            "stop_price_usd": round(stop_price, 6),
            "triggered": triggered,
            "distance_to_stop_pct": round(distance_to_stop_pct, 4),
            "pnl_pct": round(pnl_pct, 4),
        }

    return result


def get_position_ages(portfolio: dict) -> dict:
    """Calculate age of each position from first BUY transaction.

    Args:
        portfolio: Full portfolio state dict (with "holdings" and "transactions").

    Returns:
        dict keyed by ticker (only tickers currently held with shares > 0):
            - age_hours: float
            - age_days: float
            - first_buy: str (ISO-8601 timestamp of first BUY)
            - num_buys: int (total BUY transactions for this ticker)
            - num_sells: int (total SELL transactions for this ticker)
    """
    holdings = portfolio.get("holdings", {})
    transactions = portfolio.get("transactions", [])
    now = datetime.datetime.now(datetime.timezone.utc)
    result = {}

    for ticker, pos in holdings.items():
        shares = pos.get("shares", 0)
        if shares <= 0:
            continue

        # Find all BUY and SELL transactions for this ticker
        first_buy_ts = None
        num_buys = 0
        num_sells = 0

        for tx in transactions:
            if tx.get("ticker") != ticker:
                continue
            action = tx.get("action", "")
            ts_str = tx.get("timestamp", "")

            if action == "BUY":
                num_buys += 1
                try:
                    ts = datetime.datetime.fromisoformat(ts_str)
                    if first_buy_ts is None or ts < first_buy_ts:
                        first_buy_ts = ts
                except (ValueError, TypeError):
                    pass
            elif action == "SELL":
                num_sells += 1

        if first_buy_ts is not None:
            # Ensure timezone-aware comparison
            if first_buy_ts.tzinfo is None:
                first_buy_ts = first_buy_ts.replace(tzinfo=datetime.timezone.utc)
            age_delta = now - first_buy_ts
            age_hours = age_delta.total_seconds() / 3600
            age_days = age_hours / 24

            result[ticker] = {
                "age_hours": round(age_hours, 2),
                "age_days": round(age_days, 2),
                "first_buy": first_buy_ts.isoformat(),
                "num_buys": num_buys,
                "num_sells": num_sells,
            }

    return result


def log_portfolio_value(patient_path: str | None = None,
                        bold_path: str | None = None,
                        agent_summary_path: str | None = None):
    """Append current portfolio values to data/portfolio_value_history.jsonl.

    Each entry contains:
        - ts: ISO-8601 UTC timestamp
        - patient_value_sek: total patient portfolio value
        - bold_value_sek: total bold portfolio value
        - patient_pnl_pct: patient P&L percentage
        - bold_pnl_pct: bold P&L percentage
        - prices: dict of current USD prices from agent_summary

    Args:
        patient_path: Path to patient portfolio state JSON.
        bold_path: Path to bold portfolio state JSON.
        agent_summary_path: Path to agent_summary.json.
    """
    if patient_path is None:
        patient_path = str(DATA_DIR / "portfolio_state.json")
    if bold_path is None:
        bold_path = str(DATA_DIR / "portfolio_state_bold.json")
    if agent_summary_path is None:
        agent_summary_path = str(DATA_DIR / "agent_summary.json")

    patient = load_json(patient_path, default={})
    bold = load_json(bold_path, default={})
    summary = load_json(agent_summary_path, default={"signals": {}, "fx_rate": 1.0})

    patient_value = _compute_portfolio_value(patient, summary)
    bold_value = _compute_portfolio_value(bold, summary)

    patient_initial = patient.get("initial_value_sek", INITIAL_VALUE_DEFAULT)
    bold_initial = bold.get("initial_value_sek", INITIAL_VALUE_DEFAULT)

    patient_pnl_pct = ((patient_value - patient_initial) / patient_initial) * 100 if patient_initial > 0 else 0
    bold_pnl_pct = ((bold_value - bold_initial) / bold_initial) * 100 if bold_initial > 0 else 0

    # Collect current prices
    prices = {}
    for ticker, sig in summary.get("signals", {}).items():
        price = sig.get("price_usd")
        if price is not None:
            prices[ticker] = price

    entry = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "patient_value_sek": round(patient_value, 2),
        "bold_value_sek": round(bold_value, 2),
        "patient_pnl_pct": round(patient_pnl_pct, 4),
        "bold_pnl_pct": round(bold_pnl_pct, 4),
        "fx_rate": summary.get("fx_rate", 1.0),
        "prices": prices,
    }

    history_path = DATA_DIR / "portfolio_value_history.jsonl"
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def transaction_cost_analysis(portfolio: dict) -> dict:
    """Analyze transaction costs for a portfolio.

    Examines all transactions and accumulated fees to produce a cost report.

    Args:
        portfolio: Full portfolio state dict.

    Returns:
        dict with:
            - total_fees_sek: float -- accumulated fees
            - fees_as_pct_of_initial: float -- total_fees / initial_value * 100
            - avg_fee_per_trade: float -- average fee per transaction
            - total_trades: int -- number of transactions
            - fees_as_pct_of_pnl: float | None -- total_fees / abs(pnl) * 100
              (None if no P&L to compare against)
            - total_buy_volume_sek: float -- sum of BUY allocs
            - total_sell_volume_sek: float -- sum of SELL proceeds
            - buy_count: int
            - sell_count: int
    """
    transactions = portfolio.get("transactions", [])
    initial_value = portfolio.get("initial_value_sek", INITIAL_VALUE_DEFAULT)
    cash = portfolio.get("cash_sek", initial_value)

    # total_fees_sek from portfolio state
    total_fees_from_state = portfolio.get("total_fees_sek", 0) or 0

    # Also compute fees from transaction records (fee_sek field)
    computed_fees = 0.0
    total_buy_volume = 0.0
    total_sell_volume = 0.0
    buy_count = 0
    sell_count = 0

    for tx in transactions:
        fee = tx.get("fee_sek", 0) or 0
        computed_fees += fee
        action = tx.get("action", "")
        total_sek = tx.get("total_sek", 0) or 0

        if action == "BUY":
            buy_count += 1
            total_buy_volume += total_sek
        elif action == "SELL":
            sell_count += 1
            total_sell_volume += total_sek

    # Use the larger of state fees vs computed fees (handles missing fee_sek fields)
    total_fees = max(total_fees_from_state, computed_fees)

    total_trades = len(transactions)
    avg_fee = total_fees / total_trades if total_trades > 0 else 0

    fees_as_pct_initial = (total_fees / initial_value) * 100 if initial_value > 0 else 0

    # PnL: cash + holdings_value - initial_value
    # For simplicity here, we approximate with cash - initial (since holdings
    # value requires live prices). A full PnL needs _compute_portfolio_value.
    # However, if portfolio is all-cash (no holdings), this is exact.
    holdings = portfolio.get("holdings", {})
    has_open_positions = any(
        pos.get("shares", 0) > 0 for pos in holdings.values()
    )

    if has_open_positions:
        # PnL is approximate (doesn't include unrealized gains)
        pnl_note = "approximate (excludes unrealized gains/losses)"
        pnl = cash - initial_value  # unrealized not included
    else:
        pnl_note = "exact (all positions closed)"
        pnl = cash - initial_value

    if abs(pnl) > 0.01:
        fees_as_pct_pnl = (total_fees / abs(pnl)) * 100
    else:
        fees_as_pct_pnl = None

    return {
        "total_fees_sek": round(total_fees, 2),
        "fees_as_pct_of_initial": round(fees_as_pct_initial, 4),
        "avg_fee_per_trade": round(avg_fee, 2),
        "total_trades": total_trades,
        "fees_as_pct_of_pnl": round(fees_as_pct_pnl, 4) if fees_as_pct_pnl is not None else None,
        "total_buy_volume_sek": round(total_buy_volume, 2),
        "total_sell_volume_sek": round(total_sell_volume, 2),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "pnl_sek": round(pnl, 2),
        "pnl_note": pnl_note,
    }
