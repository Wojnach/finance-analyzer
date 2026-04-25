# After-Hours Research Plan — 2026-04-25

## Bugs & Problems Found

1. **claude_fundamental BUY bias escapes accuracy gate at 3h**
   - ROOT CAUSE: Sonnet has 83% BUY rate, Opus has 78% BUY rate (last 500 entries)
   - Cascade picks highest-tier non-HOLD vote → always BUY for crypto/metals
   - At 1d horizon: blended accuracy ~40.1% (fast blend), correctly gated by 50% high-sample gate
   - At 3h horizon: 0 samples, accuracy defaults to 0.5, PASSES the 47% gate
   - BUG: claude_fundamental is a fundamentals signal (hours/days timescale) but votes at 3h
   - FIX: Add claude_fundamental to 3h regime gating, OR use 1d accuracy as cross-horizon fallback
   - File: `portfolio/signal_engine.py` lines ~660-690 (REGIME_GATED_SIGNALS)

2. **Signal correlation clusters waste voter diversity**
   - BB, EMA, MACD vote the same way ~85% of the time across all tickers
   - Sentiment and candlestick cluster with them on metals (85-89% agreement)
   - 3 correlated votes carry ~1 signal's worth of information
   - File: `portfolio/signal_engine.py` (correlation group gating exists but may not cover these)

3. **HOLD dilution from disabled signals**
   - 73-78% of all signal votes are HOLD across tickers
   - 18 disabled signals + accuracy-gated signals inflate voter denominator
   - Consensus percentages are artificially suppressed

4. **ETH-USD consensus below coin flip (47.7% 3h, 48.5% 1d)**
   - Missing ETH-specific signals: ETH/BTC ratio momentum, ETF flows, staking dynamics
   - System treats ETH like BTC but ETH has unique L2 value leakage and structural drivers

5. **MSTR consensus below coin flip (45.9% 3h, 49.4% 1d)**
   - Generic stock signals (RSI/MACD on MSTR price) inappropriate for BTC-leveraged treasury
   - Missing: mNAV premium/discount signal, BTC signal inheritance

## Improvements Prioritized (impact x ease)

### Tier 1: Implement NOW (high impact, easy/medium)
1. **Fix claude_fundamental 3h gate escape** — Add to 3h/4h regime gating in ranging
   - Impact: HIGH (removes ~33% phantom BUY votes from 3h consensus)
   - Effort: 1 line change in REGIME_GATED_SIGNALS
   - Risk: LOW (only gates in ranging regime at 3h/4h)
   
2. **Add claude_fundamental bias detector** — When Sonnet/Opus BUY rate >75% over last 30 samples, prefer Haiku's HOLD
   - Impact: HIGH (fixes structural BUY bias at source)
   - Effort: ~20 lines in claude_fundamental.py
   - Risk: LOW (only triggers when bias is extreme)

3. **Verify structure and fibonacci gating** — Both have terrible recent accuracy
   - structure: 32.6% recent 1d → blended ~37% → should be gated
   - fibonacci: 43.2% all-time → below 47% gate
   - Verify these are actually being gated at all horizons

### Tier 2: Implement NOW (medium impact, easy)
4. **Add BB+EMA+MACD to correlation group** — They vote identically 85% of the time
   - Only let the highest-accuracy of the three vote, gate the other two
   - File: `portfolio/signal_engine.py` correlation groups section

### Tier 3: Defer (high impact, hard)
5. **IC-weighted signal voting** — Replace equal-weight with rolling IC per signal per ticker
   - Already in memory/quant_research_priorities.md as #1 priority
   - Defer: requires accuracy_stats changes + extensive testing
   
6. **ETH/BTC ratio momentum signal** — New signal for ETH-specific driver
   - High impact for ETH accuracy
   - Defer: new signal module requires full validation pipeline

7. **mNAV premium signal for MSTR** — Track MSTR market cap / BTC holdings value
   - High impact for MSTR accuracy
   - Defer: requires new data source integration

## Execution Plan

### Batch 1: Claude Fundamental Fix + Signal Gating (3 files)
1. `portfolio/signal_engine.py` — Add claude_fundamental to 3h/4h ranging regime gate
2. `portfolio/signals/claude_fundamental.py` — Add tier BUY-bias detector to cascade
3. Tests: Run test_claude_fundamental.py + test_signal_engine.py

### Batch 2: Correlation Group Update (1 file)
1. `portfolio/signal_engine.py` — Add BB+EMA+MACD correlation group
2. Tests: Run test_signal_engine.py

### Batch 3: Commit plan + deliverables
1. Commit plan, all research deliverables, and code changes
2. Merge to main, push

## What to Defer (→ docs/IMPROVEMENT_BACKLOG.md)
- IC-weighted signal voting (needs architectural change)
- ETH/BTC ratio momentum signal (needs new signal module)
- mNAV premium signal for MSTR (needs data source)
- Bull/bear debate architecture for claude_fundamental (TradingAgents pattern)
- Walk-forward parameter optimization
- Per-regime accuracy tracking split
