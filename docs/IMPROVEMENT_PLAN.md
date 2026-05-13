# Improvement Plan — 2026-05-13

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
