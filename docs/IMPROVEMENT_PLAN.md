# Improvement Plan — 2026-05-30 Auto Session

**Branch:** `improve/auto-session-2026-05-30`
**Sources:** FGL adversarial review (2026-05-29, 126 findings) cross-referenced
with direct code verification in this session.

---

## 1. Verified Bugs

| ID | File | Line | Bug | Severity |
|----|------|------|-----|----------|
| T0-1 | autonomous.py | 80-90 | Exception before journal write → no stub → false CRITICAL | P0 |
| T0-2 | loop_contract.py | 365-371 | `autonomous_*` status not in skip list | P0 |
| P0-1 | warrant_portfolio.py | 257 | SELL of non-existent position silently ignored | P0 |
| P0-2 | avanza_orders.py | 379 | orderId="?" placeholder saved as real ID | P0 |
| P0-3 | avanza_session.py | 610 | orderbook_id not validated before POST | P0 |
| P0-4 | multi_agent_layer2.py | 181 | Raw Popen bypasses claude_gate | P0 |
| TB-1 | amihud_illiquidity_regime.py | 115 | Regime gate emits BUY/SELL | P1 |
| TB-2 | choppiness_regime_gate.py | 122-128 | Regime gate emits BUY/SELL | P1 |
| TB-3 | signal_engine.py | ~2427 | No enforcement that regime gates return HOLD | P1 |
| TF-1 | loop_health.py | 214 | `status` hardcoded "ok" | P1 |
| TA-1 | http_retry.py | 76 | Fatal-vs-transient conflation | P1 |

## 2. Refuted (verified correct)

| Claim | Why refuted |
|-------|-------------|
| Theme D: grid_fisher cancel-before-replace | Sequence is correct |
| Theme E: health.py RMW unprotected | `_health_lock` wraps all RMW |
| Theme E: shared_state dogpile no bound | `_LOADING_TIMEOUT=120s` evicts |
| EOD_EXIT_MINUTES_BEFORE=0 | Explicit user override 2026-04-13 |

## 3. Batch Execution Order

### Batch 1 — Tier 0: Restore the detector (2 files)
1. `portfolio/autonomous.py`: Add failure journal stub
2. `portfolio/loop_contract.py`: Add `autonomous`/`autonomous_failed` to status handling

### Batch 2 — P0: Avanza + Warrant validation (3 files)
1. `portfolio/warrant_portfolio.py`: Refuse SELL of non-existent position
2. `portfolio/avanza_orders.py`: Reject missing orderId on SUCCESS
3. `portfolio/avanza_session.py`: Validate orderbook_id before POST

### Batch 3 — P0: Multi-agent gate bypass (1 file)
1. `portfolio/multi_agent_layer2.py`: Add claude_gate protections

### Batch 4 — Theme B: Regime gate enforcement (3 files)
1. `portfolio/signals/amihud_illiquidity_regime.py`: Return multiplier only
2. `portfolio/signals/choppiness_regime_gate.py`: Same fix
3. `portfolio/signal_engine.py`: Engine-level guard

### Batch 5 — Health + HTTP reliability (2 files)
1. `portfolio/loop_health.py`: Use `ok` param in status
2. `portfolio/http_retry.py`: Fatal-vs-transient typing

## 4. Not In Scope

- EOD_EXIT_MINUTES_BEFORE: explicit user override
- Leverage-aware ATR stops: needs verification of which paths affect warrants
- Cross-process JSON lock: architectural, dedicated session
- metals_loop monolith split: XL effort, no functional bug
