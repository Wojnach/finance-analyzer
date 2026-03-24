Perform a full Layer 2 analysis cycle — the complete analysis described in CLAUDE.md. This is the big one.

You are the decision-making layer. Read everything, analyze, decide for both strategies, journal, and notify.

## Steps (follow CLAUDE.md sections 1-6 exactly)

### 1. Read the data
Read these files (parallel where possible):
- `data/layer2_context.md` — your memory from previous invocations
- `data/agent_summary_compact.json` — all 30 signals, timeframes, indicators, macro (read this, NOT agent_summary.json)
- `data/portfolio_state.json` — Patient strategy
- `data/portfolio_state_bold.json` — Bold strategy
- `data/portfolio_state_warrants.json` — Warrant positions
- `data/prophecy.json` — Active beliefs and checkpoints
- `config.json` — Notification mode, focus tickers
- Last 5 entries from `data/layer2_journal.jsonl` — your previous theses and prices

### 2. Analyze
Follow CLAUDE.md Section 2 exactly:
- Compare previous thesis prices with current — were you right? Write reflection.
- Review all 30 signals across all timeframes for each instrument
- Check macro context: DXY, yields, curve, FOMC proximity
- Assess portfolio risk: concentration, drawdown, cash reserves
- Check recent transactions: avoid whipsaw
- Consider market regime per instrument
- Check ATR-based exits for held positions (2x ATR stop-loss guide)
- Check cross-asset leads
- Compare weighted_confidence with raw confidence

### 3. Decide (for EACH strategy independently)
Follow CLAUDE.md Section 3:
- **For any BUY or SELL consideration:** Run the structured adversarial debate (bull/bear/synthesis)
- **Patient:** Apply "The Regime Reader" personality. Bias toward patience. 15% cash BUY, 50% SELL.
- **Bold:** Apply "The Breakout Trend Rider" personality. Bias toward action on confirmed setups. 30% cash BUY, 100% SELL.
- Check pre-trade guards: position limits, averaging-down rules, cooldowns

### 4. Execute (if trading)
Follow CLAUDE.md Section 4 math EXACTLY:
- BUY: alloc, fee, net_alloc, shares_bought, new avg_cost, deduct from cash
- SELL: sell_shares, proceeds, fee, net_proceeds, add to cash
- Post-trade validation: fee total, holdings integrity, cash check
- Append transaction record

### 5. Write journal
Append one JSON line to `data/layer2_journal.jsonl` per CLAUDE.md Section 5 schema.
Include: ts, trigger, regime, reflection, decisions, tickers (with debate for any trade), watchlist, prices.

### 6. Notify via Telegram
Follow CLAUDE.md Section 6:
- Save message to `data/telegram_messages.jsonl`
- Send via Telegram API using config.json credentials
- Use Mode A or Mode B format based on `config.notification.mode`
- First line MUST fit Apple Watch (~60 chars)
- Include ticker grid, context line, reasoning

**IMPORTANT:** This is a manual invocation by the user. The trigger reason is "manual_invoke". Always send the Telegram message — the user wants to see your analysis.
