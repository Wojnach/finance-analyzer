# Backtest Task

Backtest the trading system against historical data from February 27th, 2025.

## What to do

1. **Read the existing trading system** to understand how it makes decisions (signals, indicators, news interpretation, entry/exit logic).

2. **Fetch or load historical price data** for all instruments the system trades, covering Feb 27th in the same intervals the system normally uses.

3. **Run the system against this data chronologically** — simulate it as if the system were live that day. For each interval:
   - Record the system's prediction: price direction (up/down) and confidence/probability.
   - Record the actual price movement that followed.
   - Log every signal, indicator value, and decision made.

4. **Calculate accuracy:** % of predictions where direction was correct.

5. **If accuracy is below 70%**, analyze which signals or conditions led to wrong predictions and propose specific improvements.

## Output

- Write full results to `docs/BACKTEST_FEB27.md`: every prediction vs actual, running accuracy, breakdown by instrument, and analysis of failures.
- Send a Telegram summary with: overall accuracy %, per-instrument accuracy, best/worst calls, and whether the 70% threshold was met.

## Important

- **Do not modify the trading system during the backtest.** Run it as-is first.
- If the system needs adapters to run against historical data instead of live data, build those as a separate backtesting module — keep it reusable for future dates.
- After the initial run, if accuracy is below 70%, propose changes in `docs/PLAN.md` but do not implement without a second backtest proving improvement.
