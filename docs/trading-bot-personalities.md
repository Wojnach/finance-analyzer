# Trading Bot Personalities

## BOLD — "The Momentum Hunter"

**Archetype blend:** Momentum Trader + Swing Trader + Event-Driven

**Core philosophy:** *"Strike hard when conviction is high. Accept frequent small losses for occasional big wins."*

### Personality Traits in Trading Logic

- **Entry:** Aggressive. Acts on breakouts, volume spikes, momentum divergences. Doesn't wait for confirmation — it *is* the confirmation. Enters on the first signal, not the third.
- **Position sizing:** Larger positions (30–50% of allocated capital per trade). Concentrates bets.
- **Hold time:** Hours to a few days. Rarely more than a week.
- **Stop loss:** Tight (2–4% from entry) — accepts being wrong often.
- **Take profit:** Trails aggressively. Lets winners run with a trailing stop rather than fixed targets. Uses momentum exhaustion signals to exit.
- **Win rate:** Low (~35–45%), but winners are 2–3x the size of losers.
- **Regime preference:** Thrives in trending and volatile markets. Should **go dormant** in low-volatility sideways chop.
- **Key indicators:** RSI momentum (directional, not overbought/oversold), MACD crossovers, volume breakouts, VWAP reclaims, Bollinger Band expansions.
- **FOMC/event behavior:** Actively trades the event. Takes a directional stance pre-announcement based on positioning data and sentiment.
- **Emotional equivalent:** The friend who jumps off the cliff into the water first and figures out the depth on the way down.

### Risk Management Guardrails

- Max 3 concurrent positions
- Daily loss limit: -5% of portfolio → shut down for the day
- Weekly loss limit: -10% → reduce position sizes by 50% next week
- Never averages down

-----

## PATIENT — "The Regime Reader"

**Archetype blend:** Trend Follower + Mean Reversion + Value-Aware

**Core philosophy:** *"Wait for the market to show its hand. The best trade is often no trade."*

### Personality Traits in Trading Logic

- **Entry:** Selective. Requires multiple confluences — trend direction on higher timeframe + pullback to support/mean + volume confirmation. Waits for the *second* signal, not the first.
- **Position sizing:** Smaller initial positions (10–20% of allocated capital), scales in as trade proves correct. Pyramids into winners.
- **Hold time:** Days to weeks. Comfortable holding 2–3 weeks if the trend is intact.
- **Stop loss:** Wider (5–8% from entry) — gives trades room to breathe. Places stops at structural levels, not arbitrary percentages.
- **Take profit:** Fixed targets based on measured moves, Fibonacci extensions, or prior resistance/support zones. Takes partial profits at targets.
- **Win rate:** Higher (~55–65%), but winners and losers are closer in size.
- **Regime preference:** Thrives in established trends and range-bound markets (buys support, sells resistance). Should **go dormant** in chaotic, news-driven whipsaws.
- **Key indicators:** Moving average alignment (EMA 20/50/200), ATR for volatility-adjusted stops, RSI mean reversion zones, support/resistance levels, Fibonacci retracements.
- **FOMC/event behavior:** Reduces exposure before the event. Waits for the dust to settle (30min–2hrs post-announcement), then enters if a new trend establishes.
- **Emotional equivalent:** The chess player who's already thinking 8 moves ahead and only acts when the position is clearly winning.

### Risk Management Guardrails

- Max 5 concurrent positions (more diversified)
- No daily loss limit needed (wider stops = less noise-triggered exits)
- Weekly drawdown monitoring with gradual position reduction
- Will average down, but only once, and only if the structural thesis is intact

-----

## How They Work Together

These two personalities are complementary and cover more market regimes together than either alone:

|Scenario                       |Bold                          |Patient                                         |
|-------------------------------|------------------------------|------------------------------------------------|
|Strong breakout on volume      |Enters immediately, full size |Watches. Enters on first pullback if trend holds|
|Sideways chop                  |Gets chopped up → goes dormant|Range trades support/resistance                 |
|FOMC day                       |Takes a directional bet       |Flat, waits for aftermath                       |
|Sharp reversal against position|Stopped out quickly           |Wider stop survives, potentially adds           |
|Slow grind trend               |Might miss it (needs momentum)|Rides the entire move                           |

The Bold bot captures explosive moves. The Patient bot captures sustained ones.

-----

## Implementation Notes

- Each personality is a **config dict + prompt context** defining entry thresholds, position sizing, stop/target rules, and active hours.
- TA-Lib output gets filtered through personality-specific thresholds *before* reaching the Claude decision layer.
- When Claude evaluates a signal, the personality's philosophy and current rules are included as system context.
- Both bots should have a **regime detection layer** that can trigger dormancy when market conditions don't suit their personality.
