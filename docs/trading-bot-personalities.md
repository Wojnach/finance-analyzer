# Trading Bot Personalities

## BOLD — "The Breakout Trend Rider"

**Archetype blend:** Aggressive Trend Follower + Breakout Trader

**Core philosophy:** *"Catch the trend early, size up with conviction, and ride it until the structure breaks."*

This personality is built on what consistently ranks #1 across academic research and real-world performance: systematic trend following. The "bold" aspect isn't recklessness — it's willingness to commit larger size when a trend confirms, and to act on breakouts before the crowd piles in.

### Personality Traits in Trading Logic

- **Entry:** Acts on confirmed breakouts from consolidation zones. Requires a structural break (higher high after a base, or breakdown below support) backed by expanding volume. Not chasing momentum — entering when a new trend *begins*.
- **Position sizing:** Aggressive once confirmed. 30–50% of allocated capital on high-conviction setups. Uses ATR-based sizing so position size adapts to current volatility — bigger positions in calm markets, smaller in volatile ones.
- **Hold time:** Days to weeks. Holds as long as the trend structure is intact. Not married to the trade — exits when structure breaks, not on arbitrary time limits.
- **Stop loss:** Structural, not percentage-based. Places stops below the breakout level or the last higher low. Typically lands around 2–5% depending on ATR. If the breakout level fails, the thesis is dead.
- **Take profit:** No fixed targets. Trails using a chandelier stop (ATR-based trailing stop) or exits when price closes below a short-term moving average (e.g., EMA 10 or 20). Lets the trend decide when it's over.
- **Win rate:** Moderate (~40–50%), but the average winner is 2–4x the average loser. Expectancy is driven by size of wins, not frequency.
- **Regime preference:** Thrives when markets transition from consolidation to trend. Should **go dormant** when ATR is compressing and no breakout setups are forming. Has a built-in "no trade" mode.
- **Key indicators:** ATR (volatility and position sizing), volume expansion on breakout, price structure (higher highs/lows or lower highs/lows), EMA 20 as trend health check, Donchian channels or Bollinger Band expansion for breakout detection.
- **FOMC/event behavior:** Does not trade the event itself. Watches for breakouts that form *after* the event settles (1–4 hours post). Events create the volatility that forms new trends — this bot catches the trend, not the noise.
- **What makes it "bold":** The sizing. When a breakout confirms, it commits meaningfully. Most retail traders under-size their best setups. This bot doesn't have that problem.

### Risk Management Guardrails

- Max 3 concurrent positions
- ATR-based position sizing caps risk per trade at 2% of total portfolio
- Daily loss limit: -5% of portfolio → no new entries for 24 hours
- Weekly loss limit: -10% → reduce position sizes by 50% next week
- Never averages down — if the breakout fails, the trade is wrong
- Dormancy trigger: If 3 consecutive trades are stopped out, pause for 48 hours and reassess regime

### Why This Ranks High

Trend following is the only active trading strategy with 100+ years of out-of-sample evidence. Adding aggressive breakout entry timing and conviction sizing to a trend following core creates the highest expected value profile for an active retail trader. The key insight is that "bold" doesn't mean "reckless" — it means sizing up when probabilities are genuinely in your favor.

-----

## PATIENT — "The Regime Reader"

**Archetype blend:** Systematic Trend Follower + Mean Reversion

**Core philosophy:** *"Wait for the market to show its hand. The best trade is often no trade."*

This personality captures the other side of what works: patience, confluence, and position management. Where Bold commits early and big, Patient builds positions gradually and extracts value from established moves.

### Personality Traits in Trading Logic

- **Entry:** Selective. Requires multiple confluences — trend direction confirmed on a higher timeframe + pullback to a mean or support zone + volume confirmation on the bounce/continuation. Enters on the *second* confirmation, not the first.
- **Position sizing:** Smaller initial positions (10–20% of allocated capital). Scales in as the trade proves correct. Pyramids into winners — adds to the position at the first and second pullback within a confirmed trend.
- **Hold time:** Days to weeks. Comfortable holding 2–3 weeks if the trend is intact. Treats time in the market as an asset, not a risk.
- **Stop loss:** Wider (5–8% from entry) — gives trades room to breathe. Places stops at structural levels (below swing lows in an uptrend, above swing highs in a downtrend), not arbitrary percentages. Uses ATR to validate that stops are outside normal noise.
- **Take profit:** Uses measured moves, Fibonacci extensions, or prior resistance/support zones as targets. Takes partial profits (e.g., 50% at first target, trail the rest). This locks in gains while keeping upside exposure.
- **Win rate:** Higher (~55–65%), but winners and losers are closer in size (1.2–1.8x reward/risk). Expectancy is driven by consistency and low drawdowns.
- **Regime preference:** Thrives in established trends and range-bound markets. In trends, rides pullbacks. In ranges, buys support and sells resistance. Should **go dormant** in chaotic, news-driven whipsaw environments where no structure exists.
- **Key indicators:** Moving average alignment (EMA 20/50/200) for trend health, ATR for volatility-adjusted stops and position sizing, RSI for mean reversion entry zones (40–50 in uptrends, 50–60 in downtrends), Fibonacci retracements for pullback entry levels, support/resistance from prior price structure.
- **FOMC/event behavior:** Reduces exposure before the event. Does not enter new positions within 4 hours of a major announcement. Waits for the dust to settle, then enters if a new trend establishes or the prior trend resumes with confirmation.
- **What makes it "patient":** It will sit in cash for days if no setup meets its criteria. It treats missed trades as zero-cost events and bad trades as the only real loss.

### Risk Management Guardrails

- Max 5 concurrent positions (more diversified across setups)
- ATR-based position sizing caps initial risk per trade at 1% of total portfolio (scales to 2% if pyramided)
- No daily loss limit needed — wider stops and smaller sizing mean individual trades rarely cause sharp drawdowns
- Weekly drawdown monitoring: if portfolio drops -7% in a week, reduce all position sizes by 50%
- Will scale into a losing position once (not averaging down — adding at a pre-planned pullback level within the thesis), but only if the structural thesis is intact and the higher timeframe trend hasn't changed
- Dormancy trigger: If win rate drops below 40% over the last 10 trades, pause and reassess whether the market regime has shifted

### Why This Ranks High

The combination of trend following (proven edge) and mean reversion entry timing (better entries within the trend) creates a smoother equity curve than either approach alone. Pyramiding into winners is one of the few position management techniques with genuine academic support. The patience to wait and the discipline to scale creates a compounding machine.

-----

## How They Work Together

These two personalities are complementary and cover more market regimes together than either alone:

|Scenario                                  |Bold                                                        |Patient                                                     |
|------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|
|Breakout from consolidation               |Enters immediately with conviction size                     |Watches. Enters on first pullback if trend holds            |
|Established trending market               |Already in from breakout, trailing stop                     |Scaling in at pullbacks, building full position             |
|Sideways range / chop                     |Dormant — no breakouts forming                              |Range trades support/resistance with mean reversion         |
|FOMC / major event day                    |No new entries during event. Watches for post-event breakout|Flat, waits for aftermath trend confirmation                |
|Sharp reversal against position           |Stopped out at breakout level — clean exit                  |Wider stop survives initial move, evaluates if thesis intact|
|Slow grinding trend with no clear breakout|May miss it entirely — needs a structural trigger           |Catches it via pullback entries and MA alignment            |
|Volatility spike / crash                  |Dormant if no clean breakout forms                          |Dormant — no structure to trade                             |

**Bold captures the beginning of moves. Patient captures the middle and duration. Together they cover trend initiation and trend continuation, which are the two highest-value phases of any market move.**

### Portfolio Allocation Between Bots

A suggested split based on the research:

- **60% Patient / 40% Bold** in normal conditions — Patient has higher expected Sharpe ratio
- **50/50** during high-breakout environments (post-consolidation, post-earnings season)
- **80% Patient / 20% Bold** during uncertain or transitional regimes

-----

## Implementation Notes

- Each personality is a **config dict + prompt context** defining entry thresholds, position sizing, stop/target rules, and active conditions.
- TA-Lib output gets filtered through personality-specific thresholds *before* reaching the Claude decision layer.
- When Claude evaluates a signal, the personality's philosophy and current rules are included as system context.
- Both bots should have a **regime detection layer** that can trigger dormancy when market conditions don't suit their personality. This can be as simple as ATR percentile ranking (high ATR = trending, low ATR = consolidation) combined with ADX for trend strength.
- **Key difference from v1:** Bold is no longer a momentum chaser. It's a trend follower that enters aggressively at the point of highest conviction (the breakout). This aligns with what actually produces the best long-term returns for active retail traders.
- Both bots use ATR-based position sizing, which means they automatically adapt to changing volatility without manual intervention.
