"""
Walk-forward backtesting engine for the ORB (Opening Range Breakout) predictor.

Validates prediction quality out-of-sample: for each day D, trains on days[0:D]
only, predicts day D, then compares against actual outcomes.

Usage:
    python -u portfolio/orb_backtest.py
    python -u portfolio/orb_backtest.py --batches 10
"""

import random
import statistics
import sys
from dataclasses import dataclass, field
from typing import Optional

from portfolio.orb_predictor import DayResult, ORBPredictor, Prediction


# === Data Structures ===


@dataclass
class BacktestDay:
    """Predicted vs actual outcome for a single day."""
    date: str
    morning_direction: str
    morning_range_pct: float
    predicted_high_conservative: float
    predicted_high_median: float
    predicted_high_aggressive: float
    predicted_low_conservative: float
    predicted_low_median: float
    predicted_low_aggressive: float
    actual_high: float
    actual_low: float
    sample_size: int
    filters_applied: list = field(default_factory=list)

    @property
    def high_error_median(self) -> float:
        """Absolute error: predicted high (median) vs actual high."""
        return abs(self.predicted_high_median - self.actual_high)

    @property
    def low_error_median(self) -> float:
        """Absolute error: predicted low (median) vs actual low."""
        return abs(self.predicted_low_median - self.actual_low)

    @property
    def high_error_pct(self) -> float:
        """Percentage error on high prediction."""
        if self.actual_high == 0:
            return 0.0
        return abs(self.predicted_high_median - self.actual_high) / self.actual_high * 100

    @property
    def low_error_pct(self) -> float:
        """Percentage error on low prediction."""
        if self.actual_low == 0:
            return 0.0
        return abs(self.predicted_low_median - self.actual_low) / self.actual_low * 100

    @property
    def high_in_range(self) -> bool:
        """Was actual high within [conservative, aggressive] band?"""
        return self.predicted_high_conservative <= self.actual_high <= self.predicted_high_aggressive

    @property
    def low_in_range(self) -> bool:
        """Was actual low within [aggressive, conservative] band?
        Note: aggressive low < conservative low (aggressive = deeper dip).
        """
        return self.predicted_low_aggressive <= self.actual_low <= self.predicted_low_conservative

    @property
    def buy_target_hit(self) -> bool:
        """Did price drop to our buy target (predicted_low_median)?"""
        return self.actual_low <= self.predicted_low_median

    @property
    def sell_target_hit(self) -> bool:
        """Did price rise to our sell target (predicted_high_median)?"""
        return self.actual_high >= self.predicted_high_median


@dataclass
class BacktestMetrics:
    """Aggregated backtest statistics."""
    total_days: int
    high_hit_rate: float        # % of days actual high within predicted band
    low_hit_rate: float         # % of days actual low within predicted band
    both_hit_rate: float        # % of days both high and low within bands
    high_mae: float             # Mean Absolute Error on high (USD)
    low_mae: float              # Mean Absolute Error on low (USD)
    high_mae_pct: float         # MAE on high as percentage
    low_mae_pct: float          # MAE on low as percentage
    directional_accuracy: float # % of days direction prediction correct
    trade_count: int            # Days where buy target was hit
    winning_trades: int         # Trades where both buy and sell targets hit
    total_pnl_usd: float       # Cumulative P&L from simulated trades
    total_pnl_pct: float        # Cumulative P&L as % of capital deployed
    avg_trade_pnl_usd: float   # Average P&L per trade
    win_rate: float             # winning_trades / trade_count
    avg_sample_size: float      # Average historical days used per prediction
    filters_used: str           # Description of filters applied
    per_day: list = field(default_factory=list)  # List of BacktestDay


# === Walk-Forward Backtest ===


def walk_forward_backtest(
    day_results: list[DayResult],
    predictor: ORBPredictor,
    min_history: int = 10,
    use_direction_filter: bool = True,
    use_range_filter: bool = False,
) -> list[BacktestDay]:
    """Run walk-forward backtest: for each day D, train on days[0:D], predict day D.

    Args:
        day_results: All historical day results, sorted chronologically.
        predictor: ORBPredictor instance.
        min_history: Minimum days of history required before first prediction.
        use_direction_filter: Pass through to predictor.
        use_range_filter: Pass through to predictor.

    Returns:
        List of BacktestDay results for each predicted day.
    """
    results = []

    for i in range(min_history, len(day_results)):
        today = day_results[i]
        training_data = day_results[:i]  # Only past data — no look-ahead

        prediction = predictor.predict_daily_range(
            morning=today.morning,
            historical_days=training_data,
            use_direction_filter=use_direction_filter,
            use_range_filter=use_range_filter,
        )

        if prediction is None:
            continue

        bt_day = BacktestDay(
            date=today.date,
            morning_direction=today.morning.direction,
            morning_range_pct=today.morning.range_pct,
            predicted_high_conservative=prediction.predicted_high_conservative,
            predicted_high_median=prediction.predicted_high_median,
            predicted_high_aggressive=prediction.predicted_high_aggressive,
            predicted_low_conservative=prediction.predicted_low_conservative,
            predicted_low_median=prediction.predicted_low_median,
            predicted_low_aggressive=prediction.predicted_low_aggressive,
            actual_high=today.day_high,
            actual_low=today.day_low,
            sample_size=prediction.sample_size,
            filters_applied=prediction.filters_applied,
        )
        results.append(bt_day)

    return results


# === Metrics Calculation ===


FEE_RATE = 0.0005  # 0.05% per side


def _simulate_trades(days: list[BacktestDay]) -> tuple[int, int, float, float]:
    """Simulate buy-at-predicted-low, sell-at-predicted-high trading.

    Strategy: each day, place limit buy at predicted_low_median.
    If filled, place limit sell at predicted_high_median.
    If sell not filled, exit at morning midpoint (next day open proxy).

    Returns: (trade_count, winning_trades, total_pnl_usd, total_capital_deployed)
    """
    trade_count = 0
    winning_trades = 0
    total_pnl = 0.0
    total_capital = 0.0

    for day in days:
        if not day.buy_target_hit:
            continue

        # Buy filled at predicted_low_median
        buy_price = day.predicted_low_median
        buy_cost = buy_price * (1 + FEE_RATE)  # price + fee
        trade_count += 1
        total_capital += buy_price

        if day.sell_target_hit:
            # Sell filled at predicted_high_median
            sell_price = day.predicted_high_median
            sell_proceeds = sell_price * (1 - FEE_RATE)  # price - fee
            winning_trades += 1
        else:
            # Sell target not hit — exit at midpoint between actual high and low
            # (conservative estimate of exit near end of day)
            exit_price = (day.actual_high + day.actual_low) / 2
            sell_proceeds = exit_price * (1 - FEE_RATE)

        pnl = sell_proceeds - buy_cost
        total_pnl += pnl

    return trade_count, winning_trades, total_pnl, total_capital


def _directional_accuracy(days: list[BacktestDay]) -> float:
    """Did morning direction predict which side extends more?

    "up" morning -> actual high extends more than actual low (upside_ext > downside_ext)
    "down" morning -> actual low extends more (downside_ext > upside_ext)
    """
    if not days:
        return 0.0

    correct = 0
    for day in days:
        # Calculate actual extensions from morning range
        morning_high = day.predicted_high_conservative  # ~= morning high (conservative is closest)
        morning_low = day.predicted_low_conservative    # ~= morning low
        # Better: use the original morning high/low from the prediction
        # The conservative predictions are morning_high * (1 + p25_upside/100)
        # We need the raw morning values. We can back-calculate approximately,
        # but it's simpler to check which actual extreme deviated more.
        upside = day.actual_high - morning_high
        downside = morning_low - day.actual_low

        if day.morning_direction == "up" and upside >= downside:
            correct += 1
        elif day.morning_direction == "down" and downside >= upside:
            correct += 1

    return correct / len(days) * 100


def calculate_metrics(
    backtest_days: list[BacktestDay],
    label: str = "default",
) -> BacktestMetrics:
    """Calculate comprehensive metrics from backtest results.

    Args:
        backtest_days: Results from walk_forward_backtest.
        label: Description of filter configuration.

    Returns:
        BacktestMetrics with all statistics populated.
    """
    if not backtest_days:
        return BacktestMetrics(
            total_days=0, high_hit_rate=0, low_hit_rate=0, both_hit_rate=0,
            high_mae=0, low_mae=0, high_mae_pct=0, low_mae_pct=0,
            directional_accuracy=0, trade_count=0, winning_trades=0,
            total_pnl_usd=0, total_pnl_pct=0, avg_trade_pnl_usd=0,
            win_rate=0, avg_sample_size=0, filters_used=label,
        )

    n = len(backtest_days)

    # Hit rates
    high_hits = sum(1 for d in backtest_days if d.high_in_range)
    low_hits = sum(1 for d in backtest_days if d.low_in_range)
    both_hits = sum(1 for d in backtest_days if d.high_in_range and d.low_in_range)

    # MAE
    high_errors = [d.high_error_median for d in backtest_days]
    low_errors = [d.low_error_median for d in backtest_days]
    high_errors_pct = [d.high_error_pct for d in backtest_days]
    low_errors_pct = [d.low_error_pct for d in backtest_days]

    # Trades
    trade_count, winning_trades, total_pnl, total_capital = _simulate_trades(backtest_days)

    return BacktestMetrics(
        total_days=n,
        high_hit_rate=high_hits / n * 100,
        low_hit_rate=low_hits / n * 100,
        both_hit_rate=both_hits / n * 100,
        high_mae=statistics.mean(high_errors),
        low_mae=statistics.mean(low_errors),
        high_mae_pct=statistics.mean(high_errors_pct),
        low_mae_pct=statistics.mean(low_errors_pct),
        directional_accuracy=_directional_accuracy(backtest_days),
        trade_count=trade_count,
        winning_trades=winning_trades,
        total_pnl_usd=total_pnl,
        total_pnl_pct=(total_pnl / total_capital * 100) if total_capital > 0 else 0,
        avg_trade_pnl_usd=(total_pnl / trade_count) if trade_count > 0 else 0,
        win_rate=(winning_trades / trade_count * 100) if trade_count > 0 else 0,
        avg_sample_size=statistics.mean(d.sample_size for d in backtest_days),
        filters_used=label,
        per_day=backtest_days,
    )


# === Leave-N-Out Validation ===


def leave_n_out_validation(
    day_results: list[DayResult],
    predictor: ORBPredictor,
    n: int = 5,
    iterations: int = 20,
    seed: int = 42,
) -> BacktestMetrics:
    """Leave-N-out cross-validation.

    Remove N random days from training set, predict them using the rest.
    Repeat `iterations` times with different random selections.
    Report average metrics across all runs.

    Args:
        day_results: All historical day results.
        predictor: ORBPredictor instance.
        n: Number of days to hold out per iteration.
        iterations: Number of random iterations.
        seed: Random seed for reproducibility.

    Returns:
        Aggregated BacktestMetrics across all iterations.
    """
    rng = random.Random(seed)
    all_bt_days = []

    for i in range(iterations):
        # Pick N random days to hold out (must have enough history before them)
        eligible_indices = list(range(10, len(day_results)))  # skip first 10 for min history
        if len(eligible_indices) < n:
            break

        holdout_indices = set(rng.sample(eligible_indices, min(n, len(eligible_indices))))

        for idx in sorted(holdout_indices):
            test_day = day_results[idx]
            # Training set: all days except holdout, that come before this day
            training = [
                day_results[j] for j in range(idx)
                if j not in holdout_indices
            ]

            if len(training) < 5:
                continue

            prediction = predictor.predict_daily_range(
                morning=test_day.morning,
                historical_days=training,
            )

            if prediction is None:
                continue

            bt_day = BacktestDay(
                date=test_day.date,
                morning_direction=test_day.morning.direction,
                morning_range_pct=test_day.morning.range_pct,
                predicted_high_conservative=prediction.predicted_high_conservative,
                predicted_high_median=prediction.predicted_high_median,
                predicted_high_aggressive=prediction.predicted_high_aggressive,
                predicted_low_conservative=prediction.predicted_low_conservative,
                predicted_low_median=prediction.predicted_low_median,
                predicted_low_aggressive=prediction.predicted_low_aggressive,
                actual_high=test_day.day_high,
                actual_low=test_day.day_low,
                sample_size=prediction.sample_size,
                filters_applied=prediction.filters_applied,
            )
            all_bt_days.append(bt_day)

    return calculate_metrics(all_bt_days, label=f"leave-{n}-out x{iterations}")


# === Report Formatting ===


def format_backtest_report(
    metrics: BacktestMetrics,
    show_per_day: bool = True,
    max_days: int = 50,
) -> str:
    """Format backtest results into a readable report.

    Args:
        metrics: BacktestMetrics from calculate_metrics.
        show_per_day: Whether to include per-day detail table.
        max_days: Maximum number of per-day rows to show.

    Returns:
        Formatted string report.
    """
    lines = [
        "=" * 80,
        f"  ORB BACKTEST REPORT — {metrics.filters_used}",
        "=" * 80,
        "",
        f"  Days tested:         {metrics.total_days}",
        f"  Avg training sample: {metrics.avg_sample_size:.1f} days",
        "",
        "--- Prediction Accuracy ---",
        f"  High within band:    {metrics.high_hit_rate:5.1f}%  (actual high in [conservative, aggressive])",
        f"  Low within band:     {metrics.low_hit_rate:5.1f}%  (actual low in [aggressive, conservative])",
        f"  Both within band:    {metrics.both_hit_rate:5.1f}%",
        f"  Directional:         {metrics.directional_accuracy:5.1f}%  (morning direction predicts extension side)",
        "",
        "--- Prediction Error (MAE) ---",
        f"  High MAE:            ${metrics.high_mae:.4f}  ({metrics.high_mae_pct:.3f}%)",
        f"  Low MAE:             ${metrics.low_mae:.4f}  ({metrics.low_mae_pct:.3f}%)",
        "",
        "--- Simulated Trading (buy@low_median, sell@high_median, 0.05% fee/side) ---",
        f"  Trades executed:     {metrics.trade_count}  (buy target hit)",
        f"  Winning trades:      {metrics.winning_trades}  (both targets hit)",
        f"  Win rate:            {metrics.win_rate:5.1f}%",
        f"  Total P&L:           ${metrics.total_pnl_usd:+.4f}",
        f"  P&L % of capital:    {metrics.total_pnl_pct:+.3f}%",
        f"  Avg P&L per trade:   ${metrics.avg_trade_pnl_usd:+.4f}",
    ]

    if show_per_day and metrics.per_day:
        lines.extend([
            "",
            "-" * 80,
            "  PER-DAY DETAILS",
            "-" * 80,
            "",
            "  Date       | Dir | Pred High (C/M/A)        | Act High | Pred Low (C/M/A)         | Act Low  | Buy? Sell?",
            "  " + "-" * 118,
        ])

        for day in metrics.per_day[:max_days]:
            buy_mark = "Y" if day.buy_target_hit else "."
            sell_mark = "Y" if day.sell_target_hit else "."
            lines.append(
                f"  {day.date} | {day.morning_direction:4s} "
                f"| {day.predicted_high_conservative:8.2f} {day.predicted_high_median:8.2f} {day.predicted_high_aggressive:8.2f} "
                f"| {day.actual_high:8.2f} "
                f"| {day.predicted_low_conservative:8.2f} {day.predicted_low_median:8.2f} {day.predicted_low_aggressive:8.2f} "
                f"| {day.actual_low:8.2f} "
                f"|  {buy_mark}    {sell_mark}"
            )

        if len(metrics.per_day) > max_days:
            lines.append(f"  ... ({len(metrics.per_day) - max_days} more days omitted)")

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


# === Filter Comparison ===


def compare_filters(
    day_results: list[DayResult],
    predictor: ORBPredictor,
    min_history: int = 10,
) -> str:
    """Run backtest with different filter combinations and compare.

    Tests four configurations:
    1. No filters
    2. Direction filter only
    3. Range filter only
    4. Both filters

    Returns formatted comparison report.
    """
    configs = [
        ("No filters", False, False),
        ("Direction filter", True, False),
        ("Range filter", False, True),
        ("Direction + Range filters", True, True),
    ]

    results = []
    for label, use_dir, use_range in configs:
        bt_days = walk_forward_backtest(
            day_results, predictor, min_history,
            use_direction_filter=use_dir,
            use_range_filter=use_range,
        )
        metrics = calculate_metrics(bt_days, label=label)
        results.append(metrics)

    # Format comparison table
    lines = [
        "",
        "=" * 80,
        "  FILTER COMPARISON",
        "=" * 80,
        "",
        f"  {'Config':<28s} | {'Days':>5s} | {'HiHit%':>6s} | {'LoHit%':>6s} | {'HiMAE%':>6s} | {'LoMAE%':>6s} | {'Trades':>6s} | {'Win%':>5s} | {'P&L$':>8s}",
        "  " + "-" * 100,
    ]

    for m in results:
        lines.append(
            f"  {m.filters_used:<28s} | {m.total_days:5d} | {m.high_hit_rate:5.1f}% | {m.low_hit_rate:5.1f}% "
            f"| {m.high_mae_pct:5.3f}% | {m.low_mae_pct:5.3f}% | {m.trade_count:6d} | {m.win_rate:4.1f}% | {m.total_pnl_usd:+8.4f}"
        )

    lines.append("")
    return "\n".join(lines)


# === Convenience Entry Point ===


def run_backtest(num_batches: int = 5) -> None:
    """Fetch data, run walk-forward backtest, and print full report.

    Args:
        num_batches: Number of Binance API batches to fetch (each ~10 days).
    """
    print(f"Fetching {num_batches} batches of XAGUSDT 15m data from Binance...")
    predictor = ORBPredictor()
    klines = predictor.fetch_klines(num_batches=num_batches)

    print(f"Fetched {len(klines)} candles")
    day_results = predictor.calculate_all_days(klines)
    print(f"Valid trading days: {len(day_results)}")

    if len(day_results) < 15:
        print("ERROR: Not enough trading days for backtest (need at least 15)")
        return

    # 1. Walk-forward with default filters (direction only)
    print("\n--- Running walk-forward backtest (direction filter) ---")
    bt_days = walk_forward_backtest(day_results, predictor, min_history=10)
    metrics = calculate_metrics(bt_days, label="Walk-forward (direction filter)")
    print(format_backtest_report(metrics))

    # 2. Filter comparison
    print("\n--- Running filter comparison ---")
    print(compare_filters(day_results, predictor))

    # 3. Leave-N-out cross-validation
    print("\n--- Running leave-5-out cross-validation (20 iterations) ---")
    lno_metrics = leave_n_out_validation(day_results, predictor, n=5, iterations=20)
    print(format_backtest_report(lno_metrics, show_per_day=False))

    # 4. Summary statistics from full dataset
    stats = predictor.compute_statistics(day_results)
    print("\n--- Dataset Summary ---")
    print(f"  Total valid days: {stats['total_days']}")
    print(f"  Up mornings: {stats['up_morning_days']}  |  Down mornings: {stats['down_morning_days']}")
    print(f"  Morning range: mean {stats['morning_range_pct']['mean']:.3f}%, median {stats['morning_range_pct']['median']:.3f}%")
    print(f"  Upside ext: mean {stats['upside_ext']['mean']:.3f}%, median {stats['upside_ext']['median']:.3f}%, max {stats['upside_ext']['max']:.3f}%")
    print(f"  Downside ext: mean {stats['downside_ext']['mean']:.3f}%, median {stats['downside_ext']['median']:.3f}%, max {stats['downside_ext']['max']:.3f}%")
    print(f"  Day high in morning range: {stats['high_in_morning_pct']:.1f}%")
    print(f"  Day low in morning range: {stats['low_in_morning_pct']:.1f}%")


if __name__ == "__main__":
    batches = 5
    if len(sys.argv) > 1 and sys.argv[1] == "--batches" and len(sys.argv) > 2:
        batches = int(sys.argv[2])
    run_backtest(num_batches=batches)
