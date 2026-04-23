# Cross-Critique — 2026-04-23 Dual Adversarial Review

**Date:** 2026-04-23
**Method:** Stream A (8 parallel code-reviewer agents) vs Stream B (independent Opus review).
Each stream's findings are critiqued by the other for false positives, missed severity,
and blind spots.

---

## A → B: Agent findings the independent review missed

### Confirmed important findings agents caught that I missed:

1. **S-CORE-009 (signal_db):** SQL-based accuracy functions don't apply the `_MIN_CHANGE_PCT`
   neutral-outcome filter. Python path filters ±0.05% changes as neutral; SQL path counts
   them as wrong. This systematically depresses accuracy in the active code path.
   *I missed this because I focused on signal_engine.py and didn't read signal_db.py's SQL.*

2. **S-CORE-004 (ic_computation):** Uses relative `Path("data")` instead of `__file__`-based
   path. Breaks when CWD differs from repo root (scheduled tasks, tests). IC data silently
   unavailable. *I missed this — a subtle but impactful portability bug.*

3. **ORCH-005 (multi_agent_layer2):** `proc.kill()` in specialist timeout leaves Node.js
   child trees alive — fixed in `claude_gate.py` with `_kill_process_tree()` but not
   propagated to `multi_agent_layer2.py`. *I didn't read multi_agent_layer2.py.*

4. **ORCH-007:** Trades by timed-out agents skip `record_trade()` — overtrading guards
   never activate for those trades. *Subtle sequencing issue I couldn't catch from
   reading main.py alone.*

5. **METAL-001 (microstructure_state):** `persist_state()` double-records OFI values by
   re-calling `get_microstructure_state()`, compressing z-score variance and biasing the
   orderbook_flow signal toward neutral. *I didn't read microstructure_state.py.*

6. **METAL-006 (microstructure):** OFI ask-side sign convention is inverted relative to
   Cont et al. (2014). Rising ask produces positive OFI contribution when it should
   produce negative. Structural BUY bias. *Mathematical — needs domain expertise.*

7. **AVZ-001 (avanza_orders):** Pending orders can be lost due to unprotected read-modify-write
   between `check_pending_orders` and `request_order`. *I noted the order lock bypass but
   missed the file-level race in the pending orders flow.*

8. **AVZ-008 (warrant_portfolio):** Average-down on BUY doesn't update
   `underlying_entry_price_usd`, corrupting warrant P&L calculations.
   *I didn't read warrant_portfolio.py.*

9. **SIG-006 (news_event):** Bare "cut" keyword defaults to positive sentiment. Headlines
   like "Tesla cuts workforce" generate false BUY signals. *Simple but impactful.*

10. **DATA-002 (sentiment):** MSTR routed through stock-model sentiment but its price is
    BTC-driven. Structurally wrong sentiment for a Tier-1 instrument. *Architecture issue
    I couldn't see from signal_engine.py alone.*

### Agent findings I consider false positives or overstated:

1. **S-CORE-001:** Agent says the high-sample gate relaxation formula is "wrong" because
   `max(gate_val, high_gate_val)` might invert under hypothetical parameter changes. In
   practice, the current values (0.50 > 0.47) make this correct. The deeper concern about
   coin-flip signals clearing the relaxed gate IS valid but is a design choice, not a bug.
   *Downgrade from P1 to P2.*

2. **RISK-005 (kelly_metals):** Agent claims leverage conversion "inflates allocations by
   10x+". The formula `position_fraction = half_kelly / cert_loss_frac` where
   `cert_loss_frac = avg_loss * leverage / 100` is a standard risk-of-ruin sizing formula,
   not Kelly per se. The result (0.66 for a 5x cert) is aggressive but is capped by
   `MAX_POSITION_FRACTION`. Need to verify the actual MAX_POSITION_FRACTION value before
   calling this a bug. *Possibly valid but needs deeper analysis.*

3. **ORCH-003 (loop_contract):** Agent says the `"invoked"` status check is "permanently dead."
   But tracing the code, `_log_trigger()` IS called with status `"invoked"` from main.py:822.
   The real issue is timing — the status persists for only one cycle. The agent correctly
   identified the race but overstated it as "permanently dead." *Downgrade from P1 to P2.*

---

## B → A: Independent findings the agents missed or under-rated

### Findings only the independent review caught:

1. **MY-SC-001 (_voters count disconnect):** No agent identified this specific disconnect
   where Stage 4's min_voters check uses pre-accuracy-gating counts. The signals-core agent
   flagged the high-sample gate relaxation and the utility boost but missed this fundamental
   voter-count mismatch. *This is arguably the highest-impact consensus logic bug because
   it allows thin-slate consensus to pass the quorum check.*

2. **MY-ORCH-001 (post-crash zero-sleep):** The orchestration agent flagged this as ORCH-001
   independently — CONFIRMED CONVERGENCE. Both streams identified the same bug. This
   increases confidence to 95%+.

3. **MY-RISK-001 (stale-price drawdown):** The portfolio-risk agent flagged this as RISK-003
   independently — CONFIRMED CONVERGENCE. The agent's description is more detailed
   (specifically noting the silent path vs the WARNING-only path).

### Findings where agents and I disagree on severity:

1. **RISK-001 (Kelly bankroll base):** Agent rates P0. I didn't flag this independently but
   reviewing the claim: Kelly criterion using uninvested cash instead of total portfolio
   value systematically underbets. For a portfolio 80% invested, the bankroll should be
   500K but the code uses 100K. However, this makes the system MORE conservative, not less.
   Underbetting is not a financial risk — it's a missed opportunity. *Downgrade from P0 to P2.*

2. **RISK-002 (round-trip P&L excludes fees):** Agent rates P1 (all metrics overstated).
   Valid finding but the fees are tracked separately and are visible. The overstated metrics
   affect Kelly inputs, which compounds with RISK-001. Together they make sizing inaccurate.
   *Keep at P1.*

---

## Convergence Analysis

Findings independently identified by both streams (highest confidence):

| Finding | Agent ID | Independent ID | Description |
|---------|----------|----------------|-------------|
| Post-crash zero-sleep | ORCH-001 | MY-ORCH-001 | Crashed cycles fire immediately after backoff |
| Stale-price drawdown | RISK-003 | MY-RISK-001 | Circuit breaker uses stale prices silently |
| No stop-proximity guard | AVZ-006 | MY-AVZ-001 | 3% stop-loss rule has zero enforcement |
| trade_guards no lock | RISK-006 | MY-RISK-002 | Race condition on guard state file |
| Confirmed orders TOTP path | AVZ-002/007 | MY-AVZ-002 | Orders bypass order lock |
| Utility boost bypasses gate | S-CORE-007 | MY-SC-002 | Accuracy can be artificially inflated |
| Persistence cold-start | S-CORE-010 | MY-SC-003 | Filter bypassed on first cycle |
| Thread leak on timeout | ORCH-006 | MY-ORCH-002 | Stuck threads persist forever |
| Raw file I/O violations | INFRA-001/002 | MY-INFRA-001 | Multiple atomic I/O rule breaches |
| Log rotation no fsync | INFRA-003 | MY-INFRA-002 | Data loss on crash during rotation |

**10 out of 12 independent findings were also found by agents.** This indicates high
review coverage and strong signal. The 2 unique independent findings (MY-SC-001 voter
count disconnect, MY-INFRA-003 weekly_digest memory) suggest agents sometimes miss
cross-function data flow issues.

---

## Blind Spots

### Neither stream caught:
1. Thread safety of `flush_sentiment_state()` — called from main.py post-cycle, but
   `_set_prev_sentiment()` is called from within `_process_ticker` threads. The lock
   protects individual operations but the batched disk write may race with a concurrent
   `_set_prev_sentiment()` that sets `_sentiment_dirty = True` after the snapshot is
   taken but before `_sentiment_dirty = False` is set. The fix is already in place
   (dirty flag cleared only after successful write — BUG-101), so this is mitigated.

2. The `_phase_log_per_ticker` eviction at 50% on overflow means old phase data is lost
   mid-session. Not a functional bug but a diagnostics gap.
