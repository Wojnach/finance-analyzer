# After-Hours Research Plan — 2026-05-23

## Findings Summary

### Phase 0: Daily Review
- System healthy: 0 errors, 87 cycles, 14.5h uptime, all 48 signal modules operational
- Only 1 Layer 2 journal entry today (BTC-USD BUY consensus → HOLD/DORMANT)
- 27 unresolved critical errors (13 contract violations, 12 accuracy degradation, 2 Avanza session expired)
- No metals trades since March — warrant trading inactive
- XAG last trigger: BUY→SELL flip at 19:40 UTC

### Phase 1: Market Research
- **Weekend binary event**: Iran ceasefire — deal = oil drops $8-12, risk-on, gold dips; no deal = Brent $114+, haven bid
- **Memorial Day**: US closed Mon May 25 — extended weekend exposure, thin liquidity
- **Heavy data Thursday May 28**: GDP 2nd estimate + PCE deflator + durable goods + claims
- **No FOMC/CPI/NFP this week**. Next: FOMC Jun 16-17, NFP Jun 5, CPI Jun 10
- BTC at $74.3-75.6K near critical $74K support, F&G 29, ETF outflows $1.15B but whale accumulation massive (270K BTC)

### Phase 2: Quant Research
- **Direction-specific gating is the #1 finding** — validates existing system architecture
- Walk-forward parameterized windows (arxiv 2602.10785) — window length is a hyperparameter
- Regime-adaptive ensemble learning outperforms flat penalty multipliers
- LLM confidence gate — suppress when self-reported confidence < 60%
- Bull/Bear structured prompt for Layer 2 (cheapest multi-agent approximation)

### Phase 3: Signal Audit — Key Findings
1. **Qwen3**: Best signal overall (60.1%), SELL accuracy 73.5% but BUY only 33.1% — directional gate already handling this
2. **btc_proxy**: 44.6% overall (gated), but SELL accuracy 69.4% on 49 samples — directional rescue should save SELL votes. Need to verify rescue is working.
3. **claude_fundamental**: Disabled correctly — crashed to 19.8% recent (95% BUY bias from Opus tier)
4. **calendar**: Disabled correctly — 29.3% recent with structural BUY bias
5. **crypto_evrp**: 54.5% 1d but 92.6% 5d on 229 samples — exceptional medium-term signal
6. **Fear & Greed**: 58.6% on 10K+ samples but BUY-only (0 SELL signals ever) — is this intentional?
7. **cot_positioning**: 80% but only 10 samples — statistically meaningless
8. **williams_vix_fix**: 85.7% SELL accuracy but only 35 samples — monitor
9. **Directional asymmetry > 25pp in 12 signals** — system already has directional gate at 40%, working correctly

## Bugs & Problems Found

### B1: Verify btc_proxy directional rescue is working
- btc_proxy overall 44.6% → accuracy gate forces HOLD
- But SELL accuracy 69.4% on 49 samples (≥30 threshold, ≥55% rescue threshold)
- Directional rescue SHOULD rescue SELL votes at 0.70x weight
- **Action**: Write a test that verifies directional rescue for btc_proxy SELL
- **Files**: `tests/test_signal_engine.py`

### B2: Layer 2 timeout rate too high
- Only 1/N triggers resulted in journal entries today
- Contract violations show triggers firing but agents timing out
- **Action**: Investigate timeout settings, possibly increase T1 timeout from 180s
- **Files**: `portfolio/agent_invocation.py`
- **Priority**: DEFER — needs careful investigation, not a research-session fix

### B3: Avanza session expired since May 20
- 2 critical errors for session expiry
- **Action**: Note for user — needs manual BankID re-authentication
- **Priority**: DEFER — requires human interaction

## Improvements Prioritized (Impact × Ease)

### Tier 1: Implement NOW (this session)

#### I1: Raise directional gate threshold from 40% to 43%
- **Impact**: Medium — catches more poor-direction votes
- **Effort**: Trivial — single constant change
- **Risk**: Low — existing tests should catch regressions
- **Files**: `portfolio/signal_engine.py` (line 457)
- **Rationale**: 40% is generous. Signals at 41-42% directional accuracy are still noise.
  The assertion at line 1051 requires relaxed accuracy gate (47%-2%=45%) > directional gate.
  So max safe value is 44%. Setting to 43% gives 2pp buffer.

#### I2: Add LLM confidence gate for Qwen3 and Ministral
- **Impact**: Medium — reduces noise from low-confidence LLM calls
- **Effort**: Easy — parse existing confidence output, gate below 60%
- **Files**: `portfolio/signals/ministral_signal.py`, `portfolio/signals/qwen3_signal.py`
- **Rationale**: Research shows LLM signals should be suppressed when self-reported confidence is low.
  Both models already output confidence scores. Just need to add a threshold check.

#### I3: Add directional rescue verification test
- **Impact**: Low (test, not feature) — ensures btc_proxy SELL rescue works
- **Effort**: Easy
- **Files**: `tests/test_signal_engine.py`

#### I4: Log directional gate/rescue decisions for observability
- **Impact**: Medium — helps diagnose which signals are being gated/rescued
- **Effort**: Easy — add structured logging to existing gate logic
- **Files**: `portfolio/signal_engine.py`

### Tier 2: Defer to future sessions

#### D1: Regime-specific weight vectors (replace flat penalty multipliers)
- **Impact**: High — per-regime signal weights dramatically outperform flat penalties
- **Effort**: 5 days — needs regime classifier, weight optimization, validation
- **Files**: `portfolio/signal_engine.py`, `portfolio/signals/regime_classifier.py`

#### D2: Walk-forward validation framework
- **Impact**: High — prevents small-sample illusions (Chronos crash from 76%→54%)
- **Effort**: 4 days — parameterized windows, per-signal optimization
- **Files**: `scripts/walk_forward_validation.py`, `portfolio/accuracy_stats.py`

#### D3: IC-based signal weighting
- **Impact**: High — replace equal weights with rolling IC weights
- **Effort**: 3 days — extend existing `ic_computation.py`
- **Files**: `portfolio/signal_engine.py`, `portfolio/ic_computation.py`

#### D4: Structured Bull/Bear Layer 2 prompt
- **Impact**: Medium — cheapest multi-agent approximation
- **Effort**: 1 day — prompt restructuring
- **Files**: `portfolio/agent_invocation.py`, `docs/TRADING_PLAYBOOK.md`

#### D5: BTC ETF flow signal
- **Impact**: High for BTC — quant research identifies this as biggest signal gap
- **Effort**: 3 days — needs data source, signal module, testing
- **Files**: new `portfolio/signals/btc_etf_flow.py`

## Execution Order

### Batch 1: Signal engine improvements (I1, I4)
- Raise directional gate threshold to 43%
- Add structured logging for gate/rescue decisions
- Run tests, commit

### Batch 2: LLM confidence gate (I2)
- Add confidence threshold to Qwen3 and Ministral signals
- Run tests, commit

### Batch 3: Test coverage (I3)
- Add directional rescue verification test
- Run tests, commit

### Batch 4: Merge, verify, ship
- Full test suite
- Merge to main, push
- Update SESSION_PROGRESS.md

## Deferred Items → IMPROVEMENT_BACKLOG.md
- D1-D5 above
- BTC ETF flow signal (new module needed)
- Regime-adaptive ensemble (per-regime weight vectors)
- Walk-forward validation with parameterized windows
