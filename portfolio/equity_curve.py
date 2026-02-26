"""Equity curve analysis and portfolio metrics.

Loads portfolio value history from the JSONL log and computes performance
metrics useful for charting and strategy comparison.
"""

import json
import math
import pathlib
import datetime
from collections import defaultdict

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

DEFAULT_HISTORY_PATH = DATA_DIR / "portfolio_value_history.jsonl"
INITIAL_VALUE = 500_000  # SEK
RISK_FREE_RATE_ANNUAL = 0.035  # 3.5% Swedish risk-free rate (approximate)


def load_equity_curve(path: str | None = None) -> list[dict]:
    """Load portfolio value history for charting.

    Reads the JSONL file and returns a list of dicts sorted by timestamp.
    Each dict contains:
        - ts: ISO-8601 timestamp
        - patient_value_sek: float
        - bold_value_sek: float
        - patient_pnl_pct: float
        - bold_pnl_pct: float
        - fx_rate: float
        - prices: dict of ticker -> USD price

    Args:
        path: Path to the portfolio_value_history.jsonl file.
            Defaults to data/portfolio_value_history.jsonl.

    Returns:
        list of dicts sorted by timestamp (oldest first).
        Empty list if file doesn't exist or is empty.
    """
    if path is None:
        path = str(DEFAULT_HISTORY_PATH)

    result = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    result.append(entry)
                except json.JSONDecodeError:
                    continue  # Skip malformed lines
    except FileNotFoundError:
        return []

    # Sort by timestamp
    result.sort(key=lambda x: x.get("ts", ""))
    return result


def _parse_ts(ts_str: str) -> datetime.datetime:
    """Parse an ISO-8601 timestamp string to a timezone-aware datetime."""
    dt = datetime.datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def _daily_returns(curve: list[dict], value_key: str) -> list[float]:
    """Compute daily returns from the equity curve.

    Groups entries by date, takes the last entry per day, and computes
    day-over-day percentage returns.

    Args:
        curve: Sorted equity curve list.
        value_key: Either "patient_value_sek" or "bold_value_sek".

    Returns:
        list of daily return percentages.
    """
    if not curve:
        return []

    # Group by date, take last value per day
    daily_values = {}
    for entry in curve:
        ts_str = entry.get("ts", "")
        if not ts_str:
            continue
        try:
            dt = _parse_ts(ts_str)
            date_key = dt.date()
            value = entry.get(value_key, 0)
            if value > 0:
                daily_values[date_key] = value
        except (ValueError, TypeError):
            continue

    if len(daily_values) < 2:
        return []

    sorted_dates = sorted(daily_values.keys())
    returns = []
    for i in range(1, len(sorted_dates)):
        prev_val = daily_values[sorted_dates[i - 1]]
        curr_val = daily_values[sorted_dates[i]]
        if prev_val > 0:
            daily_ret = ((curr_val - prev_val) / prev_val) * 100
            returns.append(daily_ret)

    return returns


def compute_metrics(curve: list[dict], strategy: str) -> dict:
    """Compute portfolio metrics from equity curve.

    Args:
        curve: List of equity curve entries (from load_equity_curve).
        strategy: "patient" or "bold".

    Returns:
        dict with:
            - max_drawdown_pct: float -- maximum peak-to-trough drawdown
            - sharpe_ratio: float -- annualized Sharpe ratio (or None if insufficient data)
            - sortino_ratio: float -- annualized Sortino ratio (or None)
            - win_rate: float -- percentage of positive-return days
            - avg_daily_return_pct: float -- mean daily return
            - best_day_pct: float -- best single-day return
            - worst_day_pct: float -- worst single-day return
            - days_in_drawdown: int -- number of days below previous peak
            - total_return_pct: float -- total return from start to end
            - annualized_return_pct: float -- annualized return (or None if < 1 day)
            - volatility_annual_pct: float -- annualized daily volatility
            - num_data_points: int -- number of entries in curve
            - date_range: tuple of (first_ts, last_ts)
    """
    value_key = f"{strategy}_value_sek"

    result = {
        "max_drawdown_pct": 0.0,
        "sharpe_ratio": None,
        "sortino_ratio": None,
        "win_rate": 0.0,
        "avg_daily_return_pct": 0.0,
        "best_day_pct": 0.0,
        "worst_day_pct": 0.0,
        "days_in_drawdown": 0,
        "total_return_pct": 0.0,
        "annualized_return_pct": None,
        "volatility_annual_pct": 0.0,
        "num_data_points": len(curve),
        "date_range": None,
    }

    if not curve:
        return result

    # Extract values
    values = []
    timestamps = []
    for entry in curve:
        val = entry.get(value_key)
        ts = entry.get("ts", "")
        if val is not None and val > 0 and ts:
            values.append(val)
            timestamps.append(ts)

    if not values:
        return result

    result["num_data_points"] = len(values)
    result["date_range"] = (timestamps[0], timestamps[-1])

    # --- Total return ---
    first_val = values[0]
    last_val = values[-1]
    total_return_pct = ((last_val - first_val) / first_val) * 100
    result["total_return_pct"] = round(total_return_pct, 4)

    # --- Annualized return ---
    try:
        first_dt = _parse_ts(timestamps[0])
        last_dt = _parse_ts(timestamps[-1])
        days_elapsed = (last_dt - first_dt).total_seconds() / 86400
        if days_elapsed >= 1:
            years = days_elapsed / 365.25
            # Annualized return = (final/initial)^(1/years) - 1
            if first_val > 0 and last_val > 0:
                annualized = (pow(last_val / first_val, 1 / years) - 1) * 100
                result["annualized_return_pct"] = round(annualized, 4)
    except (ValueError, TypeError, ZeroDivisionError):
        pass

    # --- Maximum drawdown ---
    peak = values[0]
    max_dd = 0.0
    days_below_peak = 0
    # Group by date for drawdown day counting
    date_was_below = set()

    for i, val in enumerate(values):
        if val > peak:
            peak = val
        dd = ((peak - val) / peak) * 100
        if dd > max_dd:
            max_dd = dd
        if dd > 0.01:  # Meaningfully below peak
            try:
                dt = _parse_ts(timestamps[i])
                date_was_below.add(dt.date())
            except (ValueError, TypeError):
                pass

    result["max_drawdown_pct"] = round(max_dd, 4)
    result["days_in_drawdown"] = len(date_was_below)

    # --- Daily returns ---
    daily_rets = _daily_returns(curve, value_key)

    if daily_rets:
        result["avg_daily_return_pct"] = round(sum(daily_rets) / len(daily_rets), 6)
        result["best_day_pct"] = round(max(daily_rets), 4)
        result["worst_day_pct"] = round(min(daily_rets), 4)

        # Win rate
        positive_days = sum(1 for r in daily_rets if r > 0)
        result["win_rate"] = round((positive_days / len(daily_rets)) * 100, 2)

        # Volatility (annualized)
        if len(daily_rets) >= 2:
            mean_ret = sum(daily_rets) / len(daily_rets)
            variance = sum((r - mean_ret) ** 2 for r in daily_rets) / (len(daily_rets) - 1)
            daily_vol = math.sqrt(variance)
            annual_vol = daily_vol * math.sqrt(252)  # Trading days
            result["volatility_annual_pct"] = round(annual_vol, 4)

            # Sharpe ratio (annualized)
            daily_rf = RISK_FREE_RATE_ANNUAL / 252  # Daily risk-free rate as decimal
            # Convert daily returns to decimal for Sharpe
            daily_rets_dec = [r / 100 for r in daily_rets]
            mean_excess = sum(r - daily_rf for r in daily_rets_dec) / len(daily_rets_dec)
            if daily_vol > 0:
                # Annualize Sharpe: mean_excess / daily_std * sqrt(252)
                daily_std_dec = math.sqrt(
                    sum((r - sum(daily_rets_dec) / len(daily_rets_dec)) ** 2
                        for r in daily_rets_dec) / (len(daily_rets_dec) - 1)
                )
                if daily_std_dec > 0:
                    sharpe = (mean_excess / daily_std_dec) * math.sqrt(252)
                    result["sharpe_ratio"] = round(sharpe, 4)

            # Sortino ratio (using downside deviation)
            downside_returns = [r / 100 - daily_rf for r in daily_rets if r / 100 < daily_rf]
            if downside_returns:
                downside_var = sum(r ** 2 for r in downside_returns) / len(downside_returns)
                downside_dev = math.sqrt(downside_var)
                if downside_dev > 0:
                    sortino = (mean_excess / downside_dev) * math.sqrt(252)
                    result["sortino_ratio"] = round(sortino, 4)

    return result


def compare_strategies(curve: list[dict]) -> dict:
    """Compare patient vs bold strategy performance.

    Args:
        curve: Equity curve from load_equity_curve().

    Returns:
        dict with:
            - patient: metrics dict
            - bold: metrics dict
            - comparison: dict with relative performance
    """
    patient = compute_metrics(curve, "patient")
    bold = compute_metrics(curve, "bold")

    comparison = {
        "return_diff_pct": round(patient["total_return_pct"] - bold["total_return_pct"], 4),
        "leader": "patient" if patient["total_return_pct"] > bold["total_return_pct"] else "bold",
        "drawdown_diff_pct": round(patient["max_drawdown_pct"] - bold["max_drawdown_pct"], 4),
        "lower_drawdown": "patient" if patient["max_drawdown_pct"] < bold["max_drawdown_pct"] else "bold",
    }

    # Risk-adjusted comparison
    if patient.get("sharpe_ratio") is not None and bold.get("sharpe_ratio") is not None:
        comparison["sharpe_leader"] = "patient" if patient["sharpe_ratio"] > bold["sharpe_ratio"] else "bold"
        comparison["sharpe_diff"] = round(patient["sharpe_ratio"] - bold["sharpe_ratio"], 4)

    return {
        "patient": patient,
        "bold": bold,
        "comparison": comparison,
    }


def get_latest_values(curve: list[dict]) -> dict | None:
    """Get the most recent portfolio values from the curve.

    Returns:
        dict with patient_value_sek, bold_value_sek, ts, or None if curve is empty.
    """
    if not curve:
        return None
    latest = curve[-1]
    return {
        "ts": latest.get("ts"),
        "patient_value_sek": latest.get("patient_value_sek"),
        "bold_value_sek": latest.get("bold_value_sek"),
        "patient_pnl_pct": latest.get("patient_pnl_pct"),
        "bold_pnl_pct": latest.get("bold_pnl_pct"),
    }


def _pair_round_trips(transactions):
    """Match BUY and SELL transactions into round-trip pairs.

    Uses FIFO matching: each SELL is paired with the earliest available BUY
    shares for that ticker.

    Args:
        transactions: List of transaction dicts from portfolio state.

    Returns:
        list of round-trip dicts, each with:
            - ticker: str
            - buy_ts: str (ISO-8601)
            - sell_ts: str (ISO-8601)
            - buy_price_sek: float (per share)
            - sell_price_sek: float (per share)
            - shares: float
            - pnl_pct: float
            - pnl_sek: float
            - hold_hours: float
            - fee_sek: float (total fees for this round trip)
    """
    from collections import defaultdict

    # Group BUYs by ticker — maintain FIFO order
    buy_queues = defaultdict(list)
    for tx in transactions:
        if tx.get("action") == "BUY":
            ticker = tx.get("ticker", "")
            shares = tx.get("shares", 0)
            total_sek = tx.get("total_sek", 0)
            fee = tx.get("fee_sek", 0) or 0
            if shares > 0:
                price_per_share = total_sek / shares
                buy_queues[ticker].append({
                    "ts": tx.get("timestamp", ""),
                    "remaining_shares": shares,
                    "price_per_share": price_per_share,
                    "fee_sek": fee,
                })

    round_trips = []

    for tx in transactions:
        if tx.get("action") != "SELL":
            continue
        ticker = tx.get("ticker", "")
        sell_shares = tx.get("shares", 0)
        sell_total = tx.get("total_sek", 0)
        sell_fee = tx.get("fee_sek", 0) or 0
        sell_ts = tx.get("timestamp", "")

        if sell_shares <= 0 or ticker not in buy_queues:
            continue

        sell_price_per_share = sell_total / sell_shares if sell_shares > 0 else 0
        shares_to_match = sell_shares

        while shares_to_match > 0 and buy_queues[ticker]:
            buy = buy_queues[ticker][0]
            matched = min(shares_to_match, buy["remaining_shares"])

            # Compute hold time
            hold_hours = 0
            try:
                buy_dt = datetime.datetime.fromisoformat(buy["ts"])
                sell_dt = datetime.datetime.fromisoformat(sell_ts)
                if buy_dt.tzinfo is None:
                    buy_dt = buy_dt.replace(tzinfo=datetime.timezone.utc)
                if sell_dt.tzinfo is None:
                    sell_dt = sell_dt.replace(tzinfo=datetime.timezone.utc)
                hold_hours = (sell_dt - buy_dt).total_seconds() / 3600
            except (ValueError, TypeError):
                pass

            buy_price = buy["price_per_share"]
            pnl_pct = ((sell_price_per_share - buy_price) / buy_price * 100) if buy_price > 0 else 0
            pnl_sek = (sell_price_per_share - buy_price) * matched

            # Proportional fees
            buy_fee_share = (buy["fee_sek"] * matched / buy["remaining_shares"]) if buy["remaining_shares"] > 0 else 0
            sell_fee_share = (sell_fee * matched / sell_shares) if sell_shares > 0 else 0

            round_trips.append({
                "ticker": ticker,
                "buy_ts": buy["ts"],
                "sell_ts": sell_ts,
                "buy_price_sek": round(buy_price, 4),
                "sell_price_sek": round(sell_price_per_share, 4),
                "shares": round(matched, 8),
                "pnl_pct": round(pnl_pct, 4),
                "pnl_sek": round(pnl_sek, 2),
                "hold_hours": round(hold_hours, 2),
                "fee_sek": round(buy_fee_share + sell_fee_share, 2),
            })

            buy["remaining_shares"] -= matched
            shares_to_match -= matched

            if buy["remaining_shares"] <= 1e-10:
                buy_queues[ticker].pop(0)

    return round_trips


def compute_trade_metrics(transactions, initial_value=INITIAL_VALUE):
    """Compute per-trade performance metrics from transaction history.

    Args:
        transactions: List of transaction dicts from portfolio state.
        initial_value: Starting portfolio value in SEK.

    Returns:
        dict with:
            - profit_factor: gross_profit / gross_loss (None if no losses)
            - avg_hold_hours: average hold time per round trip
            - trade_frequency_per_week: trades per 7 calendar days
            - win_loss_ratio: avg_win_pct / avg_loss_pct (None if no losses)
            - max_consecutive_wins: longest win streak
            - max_consecutive_losses: longest loss streak
            - expectancy_pct: (win_rate * avg_win) - (loss_rate * avg_loss)
            - calmar_ratio: annualized_return / max_drawdown (None if insufficient data)
            - round_trips: int (number of paired trades)
            - total_pnl_sek: float (sum of round-trip P&L)
    """
    trips = _pair_round_trips(transactions)

    result = {
        "profit_factor": None,
        "avg_hold_hours": 0,
        "trade_frequency_per_week": 0,
        "win_loss_ratio": None,
        "max_consecutive_wins": 0,
        "max_consecutive_losses": 0,
        "expectancy_pct": 0,
        "calmar_ratio": None,
        "round_trips": len(trips),
        "total_pnl_sek": 0,
    }

    if not trips:
        return result

    # Gross profit/loss
    gross_profit = sum(t["pnl_sek"] for t in trips if t["pnl_sek"] > 0)
    gross_loss = abs(sum(t["pnl_sek"] for t in trips if t["pnl_sek"] < 0))

    if gross_loss > 0:
        result["profit_factor"] = round(gross_profit / gross_loss, 4)

    # Average hold time
    hold_hours = [t["hold_hours"] for t in trips if t["hold_hours"] > 0]
    if hold_hours:
        result["avg_hold_hours"] = round(sum(hold_hours) / len(hold_hours), 2)

    # Trade frequency
    try:
        timestamps = []
        for t in trips:
            ts_str = t.get("sell_ts") or t.get("buy_ts")
            if ts_str:
                timestamps.append(datetime.datetime.fromisoformat(ts_str))
        if len(timestamps) >= 2:
            span_days = (max(timestamps) - min(timestamps)).total_seconds() / 86400
            if span_days > 0:
                result["trade_frequency_per_week"] = round(len(trips) / span_days * 7, 2)
    except (ValueError, TypeError):
        pass

    # Win/loss stats
    wins = [t for t in trips if t["pnl_pct"] > 0]
    losses = [t for t in trips if t["pnl_pct"] <= 0]
    win_count = len(wins)
    loss_count = len(losses)
    total_count = len(trips)

    avg_win_pct = sum(t["pnl_pct"] for t in wins) / win_count if wins else 0
    avg_loss_pct = abs(sum(t["pnl_pct"] for t in losses) / loss_count) if losses else 0

    if avg_loss_pct > 0:
        result["win_loss_ratio"] = round(avg_win_pct / avg_loss_pct, 4)

    # Streaks
    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0
    for t in trips:
        if t["pnl_pct"] > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)

    result["max_consecutive_wins"] = max_wins
    result["max_consecutive_losses"] = max_losses

    # Expectancy
    win_rate = win_count / total_count if total_count > 0 else 0
    loss_rate = loss_count / total_count if total_count > 0 else 0
    result["expectancy_pct"] = round(
        (win_rate * avg_win_pct) - (loss_rate * avg_loss_pct), 4
    )

    # Total P&L
    result["total_pnl_sek"] = round(sum(t["pnl_sek"] for t in trips), 2)

    # Calmar ratio (annualized return / max drawdown)
    # Compute a mini equity curve from round-trip PnLs
    if initial_value > 0 and len(trips) >= 2:
        equity = [initial_value]
        for t in trips:
            equity.append(equity[-1] + t["pnl_sek"])

        peak = equity[0]
        max_dd = 0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        total_return = (equity[-1] - equity[0]) / equity[0] if equity[0] > 0 else 0
        try:
            first_ts = datetime.datetime.fromisoformat(trips[0]["buy_ts"])
            last_ts = datetime.datetime.fromisoformat(trips[-1]["sell_ts"])
            days = (last_ts - first_ts).total_seconds() / 86400
            if days >= 1 and max_dd > 0:
                years = days / 365.25
                annualized = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
                result["calmar_ratio"] = round(annualized / max_dd, 4)
        except (ValueError, TypeError):
            pass

    return result


if __name__ == "__main__":
    curve = load_equity_curve()
    if not curve:
        print("No equity curve data found in", DEFAULT_HISTORY_PATH)
        print("Run risk_management.log_portfolio_value() to start logging.")
    else:
        print(f"Loaded {len(curve)} data points")
        results = compare_strategies(curve)

        for strategy in ("patient", "bold"):
            m = results[strategy]
            print(f"\n{'='*50}")
            print(f"  {strategy.upper()} STRATEGY METRICS")
            print(f"{'='*50}")
            print(f"  Total return:     {m['total_return_pct']:+.2f}%")
            print(f"  Max drawdown:     {m['max_drawdown_pct']:.2f}%")
            if m["sharpe_ratio"] is not None:
                print(f"  Sharpe ratio:     {m['sharpe_ratio']:.2f}")
            if m["sortino_ratio"] is not None:
                print(f"  Sortino ratio:    {m['sortino_ratio']:.2f}")
            print(f"  Win rate:         {m['win_rate']:.1f}%")
            print(f"  Best day:         {m['best_day_pct']:+.2f}%")
            print(f"  Worst day:        {m['worst_day_pct']:+.2f}%")
            print(f"  Days in drawdown: {m['days_in_drawdown']}")
            print(f"  Volatility (ann): {m['volatility_annual_pct']:.2f}%")
            if m["date_range"]:
                print(f"  Date range:       {m['date_range'][0][:10]} to {m['date_range'][1][:10]}")

        c = results["comparison"]
        print(f"\n{'='*50}")
        print(f"  COMPARISON")
        print(f"{'='*50}")
        print(f"  Return leader:    {c['leader']} (by {abs(c['return_diff_pct']):.2f}%)")
        print(f"  Lower drawdown:   {c['lower_drawdown']} (by {abs(c['drawdown_diff_pct']):.2f}%)")
        if "sharpe_leader" in c:
            print(f"  Better Sharpe:    {c['sharpe_leader']} (by {abs(c['sharpe_diff']):.2f})")
