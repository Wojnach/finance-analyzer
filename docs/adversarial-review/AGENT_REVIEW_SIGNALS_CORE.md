# Agent Adversarial Review: signals-core

**Agent**: feature-dev:code-reviewer
**Subsystem**: signals-core (5,640 lines, 11 files)
**Duration**: ~240 seconds
**Findings**: 10 (4 P1, 4 P2, 2 P3)

---

## P1 Findings

### A-SC-1: Per-Ticker Accuracy Override Strips Directional Fields — Directional Gate Silently Disabled
- **File**: `portfolio/signal_engine.py:1840-1849`
- **Description**: When per-ticker accuracy data is used to override global accuracy, the constructed dict includes only `accuracy`, `total`, `correct`, `pct` — it does NOT include `buy_accuracy`, `sell_accuracy`, `total_buy`, `total_sell`. The directional accuracy gate in `_weighted_consensus()` (lines 829-837) uses `stats.get("buy_accuracy", acc)` which falls back to overall accuracy when the key is missing.
- **Impact**: Signals with extreme directional bias (e.g., qwen3 BUY=30%, SELL=74%) are NOT gated when per-ticker accuracy data exists. The directional gate's `dir_acc < 0.35` condition never fires because `dir_acc` falls back to the overall accuracy (~50%).
- **Fix**: Extend `accuracy_by_ticker_signal` to compute directional accuracy per-ticker, or copy directional fields from global accuracy before overriding.

### A-SC-2: Regime Accuracy Cache Uses Single Shared Timestamp
- **File**: `portfolio/accuracy_stats.py:878`
- **Description**: `write_regime_accuracy_cache()` writes a single `"time"` key for all horizons. The main accuracy cache (BUG-133 fix) correctly uses per-horizon timestamps (`time_{horizon}`), but the regime cache was never updated.
- **Impact**: Writing "3h" regime accuracy marks the "1d" regime data as fresh. Regime accuracy overlay in `generate_signal()` (lines 1829-1832) uses potentially stale cross-horizon data.
- **Fix**: Mirror the BUG-133 fix: write `cache[f"time_{horizon}"]` in `write_regime_accuracy_cache`.

### A-SC-3: Ministral Applicable-Count vs Actual Vote Inconsistency
- **File**: `portfolio/signal_engine.py:510-511` vs `1459-1465`
- **Description**: `_compute_applicable_count()` excludes ministral for non-crypto tickers, but `generate_signal()` sets `votes["ministral"] = "HOLD"` and runs the ministral block for ALL tickers. Comment says "all tickers" but count function says crypto-only.
- **Impact**: `total_applicable` is understated by 1 for metals/stocks. If a stale cache key produces a non-HOLD vote for a non-crypto ticker, it would count in `active_voters` for a ticker where ministral is supposedly inapplicable.
- **Fix**: Align the comment, applicable count, and vote-setting logic.

### A-SC-4: Regime Accuracy Overlay Ordering vs Per-Ticker Override
- **File**: `portfolio/signal_engine.py:1829-1832` and `1841-1849`
- **Description**: Regime accuracy overlay runs BEFORE per-ticker override. Per-ticker override completely replaces whatever regime overlay set. This means regime overlay is wasted computation for the primary instruments (BTC, ETH, XAU, XAG, MSTR) that have per-ticker data.
- **Impact**: Not a bug, but wasted CPU and a design confusion that could trip future contributors.

---

## P2 Findings

### A-SC-5: blend_accuracy_data Uses max() for Total — Overstates Sample Size
- **File**: `portfolio/accuracy_stats.py:648-649`
- **Description**: `total = max(at_samples, rc_samples)` means a signal with 35 all-time and 5 recent samples reports `total=35`, making the accuracy gate treat the 70%-recent blend as if backed by 35 samples.
- **Impact**: Signals with few recent samples can be force-gated based on inflated sample counts.

### A-SC-6: signal_history.py Read-Modify-Write Race Condition
- **File**: `portfolio/signal_history.py:53-82`
- **Description**: `update_history()` does read → append → trim → write without locking. Under 8-worker ThreadPoolExecutor, concurrent writes from two tickers silently clobber each other.
- **Impact**: Lost signal history entries. Affects streak tracking and noisy-signal detection.

### A-SC-7: _load_accuracy_snapshots Reads Entire File Without Size Guard
- **File**: `portfolio/accuracy_stats.py:907-920`
- **Description**: Uses `read_text().splitlines()` instead of `load_jsonl_tail()`. File grows unboundedly.
- **Impact**: OOM risk on long-running system. Should use `load_jsonl_tail(path, max_entries=1000)`.

### A-SC-8: SQLite SignalDB Missing check_same_thread=False
- **File**: `portfolio/signal_db.py:33`
- **Description**: SQLite connection created without `check_same_thread=False`. Safe under current usage patterns (fresh connection per call) but fragile for future refactors.
- **Fix**: Add `check_same_thread=False` — zero-cost defensive hardening.

---

## P3 Findings

### A-SC-9: signal_weight_optimizer.py Uses Relative Path
- **File**: `portfolio/signal_weight_optimizer.py:27`
- **Description**: `_RESULTS_FILE = Path("data/models/walkforward_results.json")` is relative, not `Path(__file__).resolve().parent.parent / "data"`. Breaks under pytest xdist.

### A-SC-10: signal_accuracy_cost_adjusted Documentation Inaccuracy
- **File**: `portfolio/accuracy_stats.py:257`
- **Description**: Minor docstring confusion about bps thresholds. No functional impact.
