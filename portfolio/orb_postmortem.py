"""
ORB Postmortem — End-of-day analysis comparing ORB predictions to actual results.

Tracks prediction accuracy over time, identifies which filters work best,
and generates actionable recommendations.

Usage:
    python -u portfolio/orb_postmortem.py
"""

import json
import statistics
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from portfolio.orb_predictor import ORBPredictor, Prediction

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
POSTMORTEM_PATH = DATA_DIR / "orb_postmortem.jsonl"
PREDICTIONS_TODAY_PATH = DATA_DIR / "orb_predictions_today.json"


@dataclass
class PostmortemResult:
    """Result of comparing one day's ORB prediction to actual outcome."""
    date: str
    # Predicted values
    predicted_high_conservative: float
    predicted_high_median: float
    predicted_high_aggressive: float
    predicted_low_conservative: float
    predicted_low_median: float
    predicted_low_aggressive: float
    morning_direction: str
    morning_range_pct: float
    sample_size: int
    filters_applied: list
    # Actual values
    actual_high: float
    actual_low: float
    # Errors
    high_error_abs: float       # actual_high - predicted_high_median
    high_error_pct: float       # error as % of predicted
    low_error_abs: float        # actual_low - predicted_low_median
    low_error_pct: float        # error as % of predicted
    # Target hit analysis
    high_within_conservative: bool  # actual_high <= predicted_high_aggressive
    high_within_aggressive: bool    # actual_high >= predicted_high_conservative
    low_within_conservative: bool   # actual_low >= predicted_low_aggressive
    low_within_aggressive: bool     # actual_low <= predicted_low_conservative
    # P&L simulation (if traded median targets)
    buy_target_hit: bool        # actual_low <= predicted_low_median
    sell_target_hit: bool       # actual_high >= predicted_high_median
    simulated_pnl_pct: float    # % P&L if bought at pred_low_med and sold at pred_high_med


def run_postmortem(prediction: Prediction, actual_high: float, actual_low: float) -> PostmortemResult:
    """Compare predicted vs actual highs/lows for a single day.

    Args:
        prediction: The day's ORB prediction
        actual_high: Actual day high from market data
        actual_low: Actual day low from market data

    Returns:
        PostmortemResult with errors, hit/miss analysis, and simulated P&L
    """
    # Errors (positive = actual exceeded prediction, negative = actual fell short)
    high_error_abs = actual_high - prediction.predicted_high_median
    high_error_pct = high_error_abs / prediction.predicted_high_median * 100

    low_error_abs = actual_low - prediction.predicted_low_median
    low_error_pct = low_error_abs / prediction.predicted_low_median * 100

    # Target hit analysis
    # "Within conservative" means actual stayed within the tighter bounds
    high_within_conservative = actual_high <= prediction.predicted_high_aggressive
    high_within_aggressive = actual_high >= prediction.predicted_high_conservative
    low_within_conservative = actual_low >= prediction.predicted_low_aggressive
    low_within_aggressive = actual_low <= prediction.predicted_low_conservative

    # Buy/sell target hit
    buy_target_hit = actual_low <= prediction.predicted_low_median
    sell_target_hit = actual_high >= prediction.predicted_high_median

    # Simulated P&L: if we bought at predicted low median and sold at predicted high median
    if buy_target_hit and sell_target_hit:
        # Both targets hit -- full predicted spread captured
        simulated_pnl_pct = (prediction.predicted_high_median - prediction.predicted_low_median) / prediction.predicted_low_median * 100
    elif buy_target_hit:
        # Bought at low target, but high target never reached -- use actual high as exit
        simulated_pnl_pct = (actual_high - prediction.predicted_low_median) / prediction.predicted_low_median * 100
    elif sell_target_hit:
        # Never got buy fill -- no trade
        simulated_pnl_pct = 0.0
    else:
        # Neither target hit -- no trade
        simulated_pnl_pct = 0.0

    return PostmortemResult(
        date=prediction.date,
        predicted_high_conservative=prediction.predicted_high_conservative,
        predicted_high_median=prediction.predicted_high_median,
        predicted_high_aggressive=prediction.predicted_high_aggressive,
        predicted_low_conservative=prediction.predicted_low_conservative,
        predicted_low_median=prediction.predicted_low_median,
        predicted_low_aggressive=prediction.predicted_low_aggressive,
        morning_direction=prediction.morning_direction,
        morning_range_pct=prediction.morning_range_pct,
        sample_size=prediction.sample_size,
        filters_applied=prediction.filters_applied,
        actual_high=actual_high,
        actual_low=actual_low,
        high_error_abs=round(high_error_abs, 4),
        high_error_pct=round(high_error_pct, 3),
        low_error_abs=round(low_error_abs, 4),
        low_error_pct=round(low_error_pct, 3),
        high_within_conservative=high_within_conservative,
        high_within_aggressive=high_within_aggressive,
        low_within_conservative=low_within_conservative,
        low_within_aggressive=low_within_aggressive,
        buy_target_hit=buy_target_hit,
        sell_target_hit=sell_target_hit,
        simulated_pnl_pct=round(simulated_pnl_pct, 4),
    )


def log_postmortem(result: PostmortemResult, filepath: str = str(POSTMORTEM_PATH)) -> None:
    """Append one JSON line per day to the postmortem log."""
    entry = asdict(result)
    entry["logged_at"] = datetime.now(timezone.utc).isoformat()
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_postmortem_history(filepath: str = str(POSTMORTEM_PATH)) -> list[PostmortemResult]:
    """Read all past postmortems and return as list of PostmortemResult."""
    path = Path(filepath)
    if not path.exists():
        return []

    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Remove non-dataclass fields
                data.pop("logged_at", None)
                results.append(PostmortemResult(**data))
            except (json.JSONDecodeError, TypeError):
                continue
    return results


def format_lessons_learned(history: list[PostmortemResult]) -> str:
    """Analyze postmortem history for patterns and output recommendations."""
    if not history:
        return "No postmortem history available yet."

    lines = [f"=== ORB Lessons Learned ({len(history)} days) ===\n"]

    # Overall accuracy
    buy_hits = sum(1 for r in history if r.buy_target_hit)
    sell_hits = sum(1 for r in history if r.sell_target_hit)
    both_hits = sum(1 for r in history if r.buy_target_hit and r.sell_target_hit)
    lines.append(f"Buy target hit rate:  {buy_hits}/{len(history)} ({buy_hits/len(history)*100:.0f}%)")
    lines.append(f"Sell target hit rate: {sell_hits}/{len(history)} ({sell_hits/len(history)*100:.0f}%)")
    lines.append(f"Both targets hit:     {both_hits}/{len(history)} ({both_hits/len(history)*100:.0f}%)")

    # High prediction accuracy
    high_errors = [abs(r.high_error_pct) for r in history]
    low_errors = [abs(r.low_error_pct) for r in history]
    lines.append(f"\nHigh prediction error: median {statistics.median(high_errors):.2f}%, mean {statistics.mean(high_errors):.2f}%")
    lines.append(f"Low prediction error:  median {statistics.median(low_errors):.2f}%, mean {statistics.mean(low_errors):.2f}%")

    # Simulated P&L
    pnls = [r.simulated_pnl_pct for r in history]
    traded_pnls = [p for p in pnls if p != 0.0]
    lines.append(f"\nSimulated P&L (all days): total {sum(pnls):.3f}%, mean {statistics.mean(pnls):.3f}%")
    if traded_pnls:
        lines.append(f"Simulated P&L (traded days only): total {sum(traded_pnls):.3f}%, mean {statistics.mean(traded_pnls):.3f}%")
        lines.append(f"Trade days: {len(traded_pnls)}/{len(history)}")

    # Direction analysis
    up_days = [r for r in history if r.morning_direction == "up"]
    down_days = [r for r in history if r.morning_direction == "down"]

    if up_days:
        up_both = sum(1 for r in up_days if r.buy_target_hit and r.sell_target_hit)
        up_pnl = sum(r.simulated_pnl_pct for r in up_days)
        lines.append(f"\nUp mornings ({len(up_days)} days): both-hit {up_both/len(up_days)*100:.0f}%, total P&L {up_pnl:.3f}%")

    if down_days:
        down_both = sum(1 for r in down_days if r.buy_target_hit and r.sell_target_hit)
        down_pnl = sum(r.simulated_pnl_pct for r in down_days)
        lines.append(f"Down mornings ({len(down_days)} days): both-hit {down_both/len(down_days)*100:.0f}%, total P&L {down_pnl:.3f}%")

    # Range size analysis
    if len(history) >= 6:
        sorted_by_range = sorted(history, key=lambda r: r.morning_range_pct)
        half = len(sorted_by_range) // 2
        small_range = sorted_by_range[:half]
        large_range = sorted_by_range[half:]

        small_both = sum(1 for r in small_range if r.buy_target_hit and r.sell_target_hit)
        large_both = sum(1 for r in large_range if r.buy_target_hit and r.sell_target_hit)
        lines.append(f"\nSmall morning range ({len(small_range)} days): both-hit {small_both/len(small_range)*100:.0f}%")
        lines.append(f"Large morning range ({len(large_range)} days): both-hit {large_both/len(large_range)*100:.0f}%")

    # Recommendations
    lines.append("\n--- Recommendations ---")
    if len(history) < 10:
        lines.append("- Need more data (< 10 days). Keep tracking.")
    else:
        if buy_hits / len(history) < 0.5:
            lines.append("- Low buy target hit rate. Consider using conservative (25th pctl) instead of median for buy targets.")
        if sell_hits / len(history) < 0.5:
            lines.append("- Low sell target hit rate. Consider using conservative (25th pctl) instead of median for sell targets.")
        if both_hits / len(history) > 0.6:
            lines.append("- Good both-target hit rate. Median targets are reliable for this market.")
        if up_days and down_days:
            up_rate = sum(1 for r in up_days if r.buy_target_hit and r.sell_target_hit) / len(up_days)
            down_rate = sum(1 for r in down_days if r.buy_target_hit and r.sell_target_hit) / len(down_days)
            if up_rate > down_rate + 0.15:
                lines.append("- Up mornings significantly more predictable. Consider direction filter.")
            elif down_rate > up_rate + 0.15:
                lines.append("- Down mornings significantly more predictable. Consider direction filter.")

    return "\n".join(lines)


def generate_daily_report() -> Optional[PostmortemResult]:
    """Run end-of-day postmortem for today.

    Reads today's prediction from orb_predictions_today.json,
    fetches actual day data from Binance, runs postmortem, logs it.

    Returns:
        PostmortemResult if successful, None otherwise
    """
    # Load today's prediction
    if not PREDICTIONS_TODAY_PATH.exists():
        print("No prediction found for today (data/orb_predictions_today.json missing)")
        return None

    with open(PREDICTIONS_TODAY_PATH, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    # Remove non-Prediction fields
    pred_data.pop("generated_at", None)
    prediction = Prediction(**pred_data)

    print(f"Loaded prediction for {prediction.date}")
    print(f"  Morning: HIGH ${prediction.morning_high:.2f} LOW ${prediction.morning_low:.2f} DIR={prediction.morning_direction}")
    print(f"  Predicted HIGH (med): ${prediction.predicted_high_median:.2f}")
    print(f"  Predicted LOW  (med): ${prediction.predicted_low_median:.2f}")

    # Fetch actual day data from Binance
    predictor = ORBPredictor()
    print("\nFetching today's actual data from Binance...")
    try:
        klines = predictor.fetch_klines(num_batches=1, limit=200)
    except Exception as e:
        print(f"Error fetching klines: {e}")
        return None

    days = predictor.group_by_day(klines, weekdays_only=False)
    today_candles = days.get(prediction.date, [])

    if not today_candles:
        print(f"No candles found for {prediction.date}")
        return None

    # Filter to trading hours (08:00-22:00 UTC)
    day_candles = [
        c for c in today_candles
        if predictor.day_start_utc <= c["hour"] <= predictor.day_end_utc
    ]

    if len(day_candles) < 10:
        print(f"Insufficient day candles ({len(day_candles)}), market may still be open")
        return None

    actual_high = max(c["high"] for c in day_candles)
    actual_low = min(c["low"] for c in day_candles)

    print(f"  Actual: HIGH ${actual_high:.2f} LOW ${actual_low:.2f}")

    # Run postmortem
    result = run_postmortem(prediction, actual_high, actual_low)

    # Log it
    log_postmortem(result)
    print(f"\nPostmortem logged to {POSTMORTEM_PATH}")

    # Print result summary
    print(f"\n=== Postmortem for {result.date} ===")
    print(f"High: predicted ${result.predicted_high_median:.2f} vs actual ${result.actual_high:.2f} (error: {result.high_error_pct:+.2f}%)")
    print(f"Low:  predicted ${result.predicted_low_median:.2f} vs actual ${result.actual_low:.2f} (error: {result.low_error_pct:+.2f}%)")
    print(f"Buy target hit:  {'YES' if result.buy_target_hit else 'NO'}")
    print(f"Sell target hit: {'YES' if result.sell_target_hit else 'NO'}")
    print(f"Simulated P&L:   {result.simulated_pnl_pct:+.3f}%")

    # Load history and print lessons
    history = load_postmortem_history()
    if history:
        print(f"\n{format_lessons_learned(history)}")

    return result


if __name__ == "__main__":
    generate_daily_report()
