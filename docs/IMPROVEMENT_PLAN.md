# Improvement Plan — 2026-05-14

**Status:** IN PROGRESS

## Session: 2026-05-14 — Deep Validation + Safety Fixes

### Methodology

Deployed 4 parallel exploration agents (signals, infra, dashboard, trading).
Raw reports contained ~80 findings. Manual validation against actual source code
**rejected ~60% as false positives** (agents hallucinated line numbers, misread
control flow, or flagged intentional design choices as bugs).

### Validated Bugs

#### B1: risk_management.py:208 — price_usd=0 falsely zeros position value [CRITICAL]
- `signals[ticker].get("price_usd", 0)` — if entry exists but price is 0,
  position valued at $0 instead of falling through to avg_cost
- Impact: false drawdown circuit breaker trip during data collection failures

#### B2: avanza_orders.py:199-210 — Expired orders can still be confirmed [MEDIUM]
- Confirmation check runs before expiry check (`elif` means expiry skipped on confirm)
- A CONFIRM arriving after 5-min window still executes

#### B3: dashboard/app.py:1274 — metals-accuracy returns HTTP 200 on no data [LOW]
- Missing `, 404` status code; only endpoint with this inconsistency

#### B4: signal_engine.py:613 — Persistence filter condition inverted [LOW]
- `if 1 >= min_cycles:` works by accident; should be `if min_cycles <= 1:`

#### B5: signal_engine.py:2799 — ADX cache eviction comment wrong [LOW]
- Says "LRU" but does insertion-order FIFO eviction

#### B6: risk_management.py:762 — concentration check silently disabled on zero prices [MEDIUM]
- All prices 0 → total_value=0 → returns None (no risk flag)

### Test Coverage Gaps
- `_compute_portfolio_value` with price_usd=0 — NOT TESTED
- `check_concentration_risk` — ZERO tests
- `compute_all_risk_flags` — ZERO tests
- avanza_orders expired-but-confirmed — NOT TESTED

### Implementation Batches

**Batch 1: Safety fixes + tests** (5 files)
1. Fix B1: price_usd guard in _compute_portfolio_value
2. Fix B2: expiry check before confirmation
3. Fix B3: 404 on metals-accuracy no-data
4. Test: price_usd=0, concentration_risk, expired-confirmed order

**Batch 2: Code clarity** (1 file)
1. Fix B4: persistence filter condition
2. Fix B5: ADX cache comment

**Batch 3: Risk test coverage** (1 file)
1. Test: compute_all_risk_flags end-to-end

### Rejected Findings (Agent False Positives)
- portfolio_mgr._get_lock race — return IS inside `with` block
- Easter day+1 off-by-one — standard Gregorian algorithm
- US DST calculation errors — verified 2024-2030
- Singleton lock not released — registered via atexit
- file_utils sidecar lock race — benign
- Accuracy stats null inflation — correctly excluded
- Grid fisher cooldown not enforced — cooldown_until works
- ThreadPoolExecutor shutdown — intentional (OR-I-001)

---

# Improvement Plan — 2026-05-13

**Status:** COMPLETE — 18 fixes shipped across 4 batches. 7 items deferred.

## Scope

Focused on **highest-impact fixes** from adversarial reviews (2026-05-11, 2026-05-12)
and deep exploration findings. Prioritized: security > correctness > reliability > quality.

---

## Batch 1: Security & Safety Critical (7 files)

### 1a. PowerShell command injection (`portfolio/subprocess_utils.py:214-218`)
- **Bug:** `$pattern` f-spliced into PowerShell `-like` mask. No sanitization.
- **Fix:** Escape PowerShell wildcards/special chars, or use `-eq` with exact match.
- **Impact:** Low blast radius (only used for process name matching), but P0 security.

### 1b. Avanza 1000 SEK floor (3 files)
- `portfolio/trade_validation.py:32` — min order 500 → 1000 SEK
- `portfolio/kelly_sizing.py:326` — min order 500 → 1000 SEK
- `portfolio/kelly_metals.py:44` — min order 500 → 1000 SEK
- **Fix:** Change literal 500 → 1000 in all three. Grep for other 500 SEK floors.

### 1c. CORS headers leak (`dashboard/app.py:52-61`)
- **Bug:** CORS method/header headers sent even when origin not whitelisted.
- **Fix:** Move all CORS headers inside the origin-check `if` block.

### 1d. Confirm token logging (`portfolio/avanza_orders.py:139-142`)
- **Bug:** Full confirmation token logged at INFO level.
- **Fix:** Log only first 4 chars + masked remainder.

### 1e. NODE_OPTIONS overwrite (2 files)
- `portfolio/agent_invocation.py:847` — overwrites NODE_OPTIONS
- `portfolio/multi_agent_layer2.py:145` — overwrites NODE_OPTIONS
- **Fix:** Append to existing NODE_OPTIONS instead of overwriting.

---

## Batch 2: Signal System Correctness (2 files, focused edits)

### 2a. Gate relaxation violates 47% rule (`portfolio/signal_engine.py`)
- **Bug:** `_GATE_RELAXATION_MAX = 0.06` allows 47% gate to drop to 41%.
- **Fix:** Reduce to 0.0 (strict 47%). Docs say force-HOLD below 47%.

### 2b. MIN_VOTERS_METALS = 2 instead of 3 (`portfolio/signal_engine.py`)
- **Bug:** Metals use 2 minimum voters, violating MIN_VOTERS = 3 rule.
- **Fix:** Change MIN_VOTERS_METALS to 3.

### 2c. Ticker accuracy SELL inversion (`portfolio/signals/ticker_accuracy.py`)
- **Bug:** Maps SELL to `1 - accuracy`. 40% SELL → 60% P(up). Forbidden.
- **Fix:** Remove inversion; use raw accuracy for both BUY and SELL.

### 2d. Remove unused _cross_ticker_consensus (`portfolio/signal_engine.py`)
- Dead dict + lock. Remove.

---

## Batch 3: Infrastructure Reliability (5 files)

### 3a. Silent lock creation failure (`portfolio/file_utils.py:237-238`)
- **Bug:** `except OSError: pass` on sidecar lock creation.
- **Fix:** Log warning on failure.

### 3b. Cache timestamp corruption (`portfolio/shared_state.py:124`)
- **Bug:** Error recovery backdates timestamp, breaking fallback path.
- **Fix:** Set `time = now` with a flag to retry after cooldown.

### 3c. trade_guards.py save outside lock (`portfolio/trade_guards.py:312`)
- **Bug:** `_save_state()` called after releasing `_state_lock`.
- **Fix:** Move inside lock block.

### 3d. Process lock silent flush (`portfolio/process_lock.py:102-104`)
- **Bug:** `except Exception: pass` swallows write+flush.
- **Fix:** Log warning on failure.

### 3e. GPU gate fd leak on write failure (`portfolio/gpu_gate.py:217-219`)
- **Bug:** Corrupt lock file traps subsequent callers.
- **Fix:** Delete lock file on write failure before re-raising.

---

## Batch 4: Risk & Data Quality (5 files)

### 4a. Risk mgmt silent avg_cost fallback (`portfolio/risk_management.py`)
- **Fix:** Log WARNING when falling back to avg_cost.

### 4b. Grid fisher duplicate EOD close (`portfolio/grid_fisher.py`)
- **Fix:** Store order_id, decrement inventory after sell.

### 4c. Warrant hours wrong (`portfolio/golddigger.py`, `portfolio/elongir.py`)
- **Fix:** 08:30-21:30 → 08:15-21:55

### 4d. Signal decay alert raw file access (`portfolio/signal_decay_alert.py`)
- **Fix:** Replace raw open()/json.load() with file_utils.load_json().

---

## Deferred (TODO: MANUAL REVIEW)

- Dashboard CF-Access JWT bypass — needs Cloudflare integration knowledge
- Avanza account whitelist — needs live session testing
- Warrant state non-atomic mutations — architectural redesign
- Layer 2 child not Job-bound — Windows Job Object integration
- Forecast horizon contamination — signal already disabled
- IC computation sort order — needs accuracy data testing
- LLM prewarmer blocking — risky for loop stability

---

## Execution Order

Batch 1 → 2 → 3 → 4. Test after each batch. Commit per batch.
