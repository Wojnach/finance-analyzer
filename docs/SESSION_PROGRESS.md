# Session Progress — Auto-Improvement Session (2026-04-12)

## Status: IN PROGRESS

Deep codebase exploration and improvement session. 6 parallel agents explored the codebase,
many agent findings were false positives (verified by reading code). Implemented only
confirmed bugs and improvements.

### What shipped (4 commits, pushed to main)
- `c299acc` docs: adversarial review round 5 plan
- `c467f13` docs: independent adversarial review round 5
- `383c718` docs: adversarial review round 5 — synthesis + independent review
- `a948733` docs: add portfolio-risk agent review + update synthesis

### Key findings (19 total: 3 P0, 10 P1, 7 P2, 1 P3)

**Independent review (12 findings)**:
- SO-1 [P0]: check_drawdown() STILL dead code — 3 rounds open, #1 priority
- IR-1 [P1]: fx_rate=1.0 fallback would cause 10x portfolio miscalculation
- IR-2 [P1]: record_trade() never called — both risk gates disconnected
- IR-8 [P1]: /mode command breaks config.json symlink on Windows
- SO-2 [P1]: POSITIONS dict thread-safety in metals_loop
- SO-3 [P1]: Naked position on stop-loss failure
- IR-7 [P1]: _handle_buy_fill POSITIONS race window
- + 5 P2/P3 findings

**Portfolio-risk agent (10 findings, 7 NEW)**:
- PR-R5-3 [P1]: warrant_portfolio averaging doesn't update underlying entry price
- PR-R5-4 [P1]: trade_validation min order 500 vs Avanza 1000 SEK
- PR-R5-5 [P1]: atr_stop_proximity "CHECK" sentinel
- PR-R5-6 [P1]: Sharpe ratio dead guard
- PR-R5-7 [P1]: Kelly fee asymmetry
- PR-R5-8 [P2]: Monte Carlo negative shares allowed
- PR-R5-9 [P2]: Kelly metals near-zero loss → 95% position

### What's next
- **#1 PRIORITY**: Wire check_drawdown() into main.py (fix IR-1 first to prevent phantom drawdown)
- **#2**: Wire record_trade() into Layer 2 journal path
- **#3**: Add POSITIONS lock in metals_loop.py
- **#4**: Fix warrant_portfolio averaging-in (PR-R5-3)
- **#5**: Fix trade_validation min order floor (PR-R5-4)

### Previous session notes (2026-04-10)

### What shipped (2 commits, pushed)
- `6ec4be9` feat(signals): per-ticker directional accuracy + raise directional gate to 40%
  - `accuracy_stats.py`: `accuracy_by_ticker_signal()` now returns `buy_accuracy`/`sell_accuracy` per ticker×signal
  - `signal_engine.py`: BUG-158 override propagates directional fields; `_DIRECTIONAL_GATE_THRESHOLD` raised 0.35 → 0.40
  - New tests: directional accuracy fields, asymmetry test, macro_regime BUY gating at 40%
- `0d81282` docs(research): after-hours session 2026-04-10 — signal audit + research plan

### Signal audit key findings
- 12/32 active signals below 50% accuracy at 1d horizon
- Extreme per-ticker variance: ministral 71.7% on MSTR vs 20.4% on XAG (51.3pp)
- 6 directional asymmetries >15pp (biggest: qwen3 BUY 30.4% vs SELL 74.3%)
- Volatility cluster (volatility_sig, oscillators, volume, structure) has 94.9% agreement rate — all underperforming
- Momentum cluster (rsi, bb, mean_reversion) has 100% agreement rate — best-performing
- 14 signals gated in ranging regime, only 18 active

### Implementation impact
- macro_regime BUY (38.9%) now gated at 0.40 threshold — previously passed both gates
- fibonacci SELL (35.9%) now caught by raised threshold
- Per-ticker directional data flows through BUG-158 override → directional gate in `_weighted_consensus()`
- 68 tests passed (50 signal_engine + 18 ticker_signal_accuracy)

### Research deliverables written
- `data/daily_research_signal_audit.json` — full signal audit with recommendations
- `data/daily_research_ticker_deep_dive.json` — XAG-USD, MSTR, BTC-USD deep dives
- `data/daily_research_macro.json` — macro research (Hormuz, CPI, sentiment, Fed)
- `data/morning_briefing.json` — Apr 11 morning briefing
- `docs/RESEARCH_PLAN.md` — updated implementation plan

### Quant research findings (from background agent)
- **P0 next session**: Direction-specific weight scaling — use buy_accuracy/sell_accuracy as weights in _weighted_consensus (~5-line change). Data foundation shipped tonight.
- **Critical**: Fear & Greed blended accuracy = 0.357, below 0.45 gate. Should already be force-HOLD'd — VERIFY.
- **P1**: MSTR-BTC proxy signal inheritance (beta 1.31-1.41, correlation >0.80). +5-8pp on MSTR.
- **P1**: XAG cross-asset enrichment (DXY, copper lead, real yields). Paper: 85-90% directional accuracy at 20d.
- **P2**: IC-based weighting (Spearman correlation, not hit rate). Paper: ~4x returns vs single-model.
- **P2**: HMM regime detection (probabilistic, per-instrument). Paper: +1-4% annualized.

### Signal audit deep findings (from background agent)
- 10 signal pairs have 100% agreement rate — completely redundant as independent voters
- Trending-up: oscillators (24.4%), momentum_factors (23.7%) vote SELL into uptrends — catastrophic
- Trending-down: news_event (0%), econ_calendar (2.7%) near-zero
- Recommendation: raise accuracy gate to 48% (gates 4 additional coin-flip signals)

### Next priorities
1. **P0**: Direction-specific weight scaling (use directional accuracy as weight, ~5 lines)
2. **P0**: Verify fear_greed is actually gated (blended 0.357 < 0.45 gate)
3. Per-ticker signal blacklisting (ministral on XAG 20.4%)
4. Raise accuracy gate to 47-48%
5. MSTR-BTC proxy signal module
6. Monitor directional gating impact over 48h
7. Bank earnings week: GS Mon, JPM/WFC/C Tue, BAC/MS Wed
