# Improvement Plan — Auto Session 2026-04-26

Based on deep exploration by 5 parallel agents (signal engine, portfolio/risk,
infrastructure, metals/trading, test coverage) plus manual verification of all
findings against actual code.

Previous sessions fixed: BUG-219 through BUG-225, IC weighting, dynamic
correlation (agreement rate), per-ticker gating, regime accuracy overrides.

## Exploration Summary

### Agent Finding Triage (5 agents, ~25 raw findings)

**False Positives Rejected (12):**
- CancelledError in data_collector.py — `as_completed` only yields completed futures before
  timeout; `_fetch_one_timeframe` catches all exceptions (line 312). No code path to hit.
- BUG-226 exit optimizer cost model — hold-to-close path already calls `_compute_pnl_sek()`
  with costs consistently (line 618). Original report was wrong.
- Loss escalation BUY reset — correct by design: only SELLs update streak counter (line 240).
  BUYs don't represent completed trades.
- Dynamic correlation dead code — already replaced with agreement rate (line 1012-1029).
  Memory file was stale.
- IC weighting missing — already implemented at signal_engine.py:1821-1836.
- Per-ticker gating missing — implemented at signal_engine.py:3056-3078.
- HOLD dilution of voter denominator — disabled signals vote HOLD, `active_voters = buy + sell`
  correctly excludes them. No dilution.
- TOCTOU race in file_utils.py lock file init — `open("ab")` is atomic on NTFS; the exists()
  pre-check is redundant but harmless.
- Subprocess child cleanup — mitigated by orphan reaper in subprocess_utils.py.
- Monte Carlo path count off-by-one — intentional (documented).
- Transaction count baseline stale — baseline captured after spawn (correct).
- Hold-to-close double-cost-counting — both paths use `_compute_pnl_sek` consistently.

### Confirmed Bugs

- **BUG-227: Post-persistence voter gates use pre-filter count**
  `signal_engine.py:2957, 3173, 2155` — The `min_voters` gate and dynamic
  min_voters penalty both use `active_voters` computed BEFORE the persistence
  filter (line 2948). After `_apply_persistence_filter()` at line 3143 reduces
  voters (flipping unstable BUY/SELL back to HOLD), the gates should use
  `post_persistence_voters` (line 3155). Currently, weak consensus can pass
  the gate when actual post-filter voters are below the threshold.
  - **Severity**: P1 — consensus credibility inflated, affects trade decisions
  - **Impact**: When persistence filter reduces e.g. 6 voters to 3, the gate
    still sees 6, letting a weak 3-voter consensus through that should be HOLD.

- **BUG-228: fin_snipe_manager entry underlying recovery not retriable**
  `fin_snipe_manager.py:340-357` — When `position_average_price` is 0 or missing
  from API, the formula returns `current_underlying` as fallback (line 350).
  This gets saved permanently as `entry_underlying`, meaning the position's
  entry is set to the current price (0% P&L). Subsequent cycles use the saved
  bad value (line 341-343) and never retry.
  - **Severity**: P2 — exit optimizer gets wrong entry_underlying_usd
  - **Fix**: Return 0.0 instead of current_underlying on failure, so the caller
    knows the estimate is unknown and can retry next cycle.

### Test Coverage Gaps

- **portfolio_mgr.py** (180 LOC) — zero tests. Handles safety-critical financial
  state: atomic load/save, backup rotation, corruption recovery, concurrent writes.
- **trigger.py** (475 LOC) — zero tests. Decision gate for Layer 2 invocation:
  sustained-check debounce, price threshold, ranging dampening, startup grace.

### Documentation

- SYSTEM_OVERVIEW.md signal counts and module counts need updating.
- Several memory files are stale (dynamic_corr_bug, quant_research_priorities).

---

## Implementation Batches

### Batch 1: BUG-227 Post-Persistence Voter Gate (2 files)

**Files**: `portfolio/signal_engine.py`, `tests/test_signal_engine_circuit_breaker.py`

1. In `generate_signal()`, replace the second min_voters gate (line 3173) to use
   `post_persistence_voters` instead of `active_voters`:
   ```python
   # Apply core gate AND MIN_VOTERS gate to weighted consensus too
   if core_active == 0 or post_persistence_voters < min_voters:
       weighted_action = "HOLD"
       weighted_conf = 0.0
   ```

2. The first min_voters gate (line 2957) stays as-is — it runs BEFORE the
   persistence filter and correctly uses the pre-filter count for the initial
   consensus. Only the post-filter gates need the post-filter count.

3. In `apply_confidence_penalties()` (line 2155), read `_voters_post_filter`
   instead of `_voters` for the dynamic min_voters check:
   ```python
   active_voters = extra_info.get("_voters_post_filter",
                                   extra_info.get("_voters", 0))
   ```
   The fallback to `_voters` maintains backward compatibility with cached
   extra_info from before BUG-224 was fixed.

4. Write tests:
   - `test_weighted_consensus_uses_post_filter_voters`: mock a scenario where
     pre-filter voters = 5 but post-filter = 2. Verify weighted consensus is
     forced to HOLD.
   - `test_confidence_penalty_uses_post_filter_voters`: verify dynamic_min_voters
     penalty reads post-filter count.

### Batch 2: BUG-228 Entry Underlying Recovery (2 files)

**Files**: `portfolio/fin_snipe_manager.py`, `tests/test_fin_snipe_manager.py`

1. Change `_estimate_entry_underlying()` fallback from `current_underlying` to
   `0.0` when formula inputs are invalid (line 350).
2. Add a warning log when falling back to 0.0.
3. In the caller that saves the result, skip saving if `entry_underlying == 0.0`.
4. Write tests:
   - `test_entry_underlying_returns_zero_on_bad_inputs`: verify 0.0 returned
     when entry_price=0 or leverage=0.
   - `test_entry_underlying_saved_value_used`: verify saved value takes priority.

### Batch 3: Test Coverage — portfolio_mgr.py (1 new file)

**Files**: `tests/test_portfolio_mgr.py` (new)

Tests for the most critical paths:
1. `test_load_state_returns_defaults_for_missing_file`
2. `test_save_load_roundtrip`
3. `test_backup_rotation_creates_backups`
4. `test_corruption_recovery_from_backup`
5. `test_update_state_atomic_read_modify_write`
6. `test_validated_state_fills_missing_keys`

### Batch 4: Test Coverage — trigger.py (1 new file)

**Files**: `tests/test_trigger.py` (new)

Tests for sustained-check debounce and gate logic:
1. `test_update_sustained_increments_on_same_value`
2. `test_update_sustained_resets_on_changed_value`
3. `test_update_sustained_count_gate_fires_at_threshold`
4. `test_update_sustained_duration_gate_fires_at_threshold`
5. `test_ranging_dampening_blocks_low_confidence`

### Batch 5: Documentation

**Files**: `docs/SYSTEM_OVERVIEW.md`, `CLAUDE.md`

1. Update signal counts to reflect current state (51 modules, 33 active,
   18 disabled — verify against tickers.py before writing).
2. Update module line counts where they've drifted.

---

## Backlog (deferred — not this session)

- **Regime-conditioned per-signal weights**: Replace static REGIME_WEIGHTS dict
  with data-driven multipliers from regime_accuracy. Requires validation study.
- **Session expiry pre-check**: Add session time check before order POST.
  Requires Avanza session architecture change.
- **Buying power TOCTOU**: Buying power read unguarded by order lock. Requires
  caller restructuring.
- **Hold-to-close reporting**: pnl_sek uses median, ev_sek uses quintile mean.
  Semantically defensible but confusing. Low priority.
- **outcome_tracker JSONL→SQLite**: Performance optimization.

## Dependency Ordering

Batch 1 → Batch 2 → Batch 3 → Batch 4 → Batch 5

No cross-dependencies. Sequential ordering keeps commits clean.
