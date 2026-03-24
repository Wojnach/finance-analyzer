Run a fresh signal report and display current consensus for all instruments. Read-only — do NOT trade.

## Steps

1. **Check the time** — Run `powershell.exe -NoProfile -Command "Get-Date -Format 'dddd yyyy-MM-dd HH:mm:ss'"` to know which markets are active.

2. **Run signal report** — Execute: `.venv/Scripts/python.exe -u portfolio/main.py --report`
   This refreshes all 30 signals across all 19 instruments. Wait for completion (may take 1-2 min).

3. **Read compact summary** — Read `data/agent_summary_compact.json` for the full picture:
   - Per-ticker: consensus, confidence, weighted_confidence, vote breakdown, regime, RSI, MACD, volume ratio
   - Timeframe heatmaps (Now through 6mo)
   - Macro context: DXY, treasury yields, F&G, FOMC proximity
   - Focus probabilities (Mode B): directional probabilities at 3h/1d/3d for XAG-USD and BTC-USD
   - Signal accuracy stats

4. **Read prophecy** — Read `data/prophecy.json` for active beliefs and checkpoint status.

5. **Read notification config** — Read `config.json` → `notification.mode` and `focus_tickers` to know which format to use.

## Output format

Use the notification mode from config:

### Mode B (probability) — for focus tickers (XAG-USD, BTC-USD by default):
```
{TICKER}  ${price}  ↑/↓{pct}% 3h  ↑/↓{pct}% 1d  ↑/↓{pct}% 3d
  accuracy: {pct}% 1d ({samples} sam) | 7d: {cumulative_gain}%
  regime: {regime} | RSI: {rsi} | MACD: {macd}
  signals: {vote_breakdown} — {top BUY signals} vs {top SELL signals}
  heatmap: {7-char BHS heatmap Now→6mo}
```

### All tickers (Mode A grid):
```
{TICKER}  ${price}  {consensus}  {XB/YS/ZH}  {7-char heatmap}
```

### Summary sections:
```
## Signal Report — {date} {time} CET

### Focus Instruments (Mode B)
{rich format for focus tickers}

### All Instruments
{grid for remaining tickers, sorted: BUY first, then SELL, then HOLD}

### Macro Context
DXY: {value} ({5d_change}) | 10Y: {yield} | 2s10s: {spread} | F&G: {crypto}/{stock} | FOMC: {days}d

### Prophecy Check
{active beliefs with progress toward targets}

### Top Signals (by accuracy)
{top 5 most accurate signals with sample count}

### Actionable
{any tickers with BUY/SELL consensus — highlight these}
{if none: "No actionable signals. All HOLD."}
```

Be concise. Prices rounded aggressively ($68K, $426, $32, $1,949). Use the monospace ticker grid format from CLAUDE.md.
