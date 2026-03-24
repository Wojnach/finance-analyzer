Show trade history, performance stats, and fee analysis for both strategies.

## Steps

1. **Read portfolio states** (parallel):
   - `data/portfolio_state.json` — Patient transactions
   - `data/portfolio_state_bold.json` — Bold transactions

2. **Read journal** — Read last 20 entries from `data/layer2_journal.jsonl` for trade reasoning and context.

3. **Read signal accuracy** — Run `.venv/Scripts/python.exe -u portfolio/main.py --accuracy` for per-signal hit rates (to correlate: did we trade on accurate signals?).

4. **For each strategy, compute:**

   **Transaction log** (all trades, chronological):
   - Date, ticker, action, shares, price (USD + SEK), fee, reason
   - For sells: calculate realized P&L vs avg_cost

   **Performance metrics:**
   - Total realized P&L (sum of all closed trades)
   - Total unrealized P&L (current holdings vs avg_cost)
   - Win rate (% of sells that were profitable)
   - Average win size vs average loss size
   - Largest win and largest loss
   - Total fees paid + fee as % of total traded volume
   - Average hold time (entry to exit, for completed round-trips)

   **Signal correlation:**
   - For each trade: what was the consensus at entry? What was the confidence?
   - Did high-confidence trades outperform low-confidence ones?
   - Which signals were most often right when we traded?

5. **Fee drag analysis:**
   - Total fees per strategy
   - Fees as % of starting capital
   - Average fee per trade
   - Projected annual fee drag at current trading frequency

## Output format

```
## Trade History — {date}

### Patient Strategy
Trades: {count} ({buys} buys, {sells} sells)
| Date       | Action | Ticker | Shares | Price    | Fee   | P&L     | Reason          |
|------------|--------|--------|--------|----------|-------|---------|-----------------|
| ...        |        |        |        |          |       |         |                 |

Realized P&L: {total} SEK
Unrealized P&L: {total} SEK
Win Rate: {pct}% ({wins}/{total_sells})
Avg Win: {sek} SEK | Avg Loss: {sek} SEK
Total Fees: {total} SEK ({pct}% of volume)

### Bold Strategy
{same format}

### Comparison
| Metric          | Patient | Bold    |
|-----------------|---------|---------|
| Total Return    |         |         |
| Win Rate        |         |         |
| Avg Hold Time   |         |         |
| Fee Drag        |         |         |
| Sharpe (est.)   |         |         |

### Lessons
{2-3 observations from the trade data: e.g., "Bold's losses came from averaging down against trend", "Patient's patience cost missed entries on 3 breakouts", etc.}
```

Use exact numbers. Cross-reference with journal reasoning to understand WHY trades were made and whether the reasoning held up.
