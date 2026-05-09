# Improvement Plan — auto-session-2026-05-09

## Exploration Summary

Deep exploration via 4 parallel agents (signal pipeline, orchestration, portfolio/risk,
metals/dashboard/infra) + manual reading. Prior session (2026-05-08) already fixed B2-B8
and batches 2-4 from the old plan. This session found new issues.

---

## 1. Bugs & Problems Found

### P0 — Critical

**B1: Calendar signal at 29.3% — actively harmful, not disabled**
- File: `portfolio/tickers.py` (DISABLED_SIGNALS)
- Calendar signal dropped from 48.8% baseline to 29.3% recent. Critical error has
  escalated 26x consecutive. Has structural BUY bias (6 of 8 sub-signals are BUY-only).
  Already per-horizon blacklisted at 1d for every ticker but still runs and votes at
  other horizons. accuracy_degradation monitor spamming critical_errors.jsonl.
- Fix: Add `"calendar"` to DISABLED_SIGNALS. Add to `_SHADOW_SAFE_SIGNALS` in
  signal_engine.py. Resolve critical error entries.

### P1 — High

**B2: fx_rate hardcoded fallback 10.0 in monte_carlo_risk.py**
- File: `portfolio/monte_carlo_risk.py:430`
- `compute_portfolio_var()` uses `agent_summary.get("fx_rate", 10.0)` — bypasses the
  3-tier cache chain in risk_management.py. On feed outage, SEK VaR is wrong by ~5%.
- Fix: Extract FX constants to `fx_rates.py` and import shared fallback function.

**B3: Division by zero in journal.py `_detect_warnings`**
- File: `portfolio/journal.py:225`
- `(last_price - first_price) / first_price` with no guard for `first_price=0`.
  Propagates to `write_context()`, aborting Layer 2 context generation.
- Fix: Add `if first_price > 0` guard.

**B4: `signal_accuracy_cost_adjusted()` crashes on None change_pct**
- File: `portfolio/accuracy_stats.py:432`
- `abs(change_pct)` raises TypeError when change_pct is None (missing backfill).
  Base `signal_accuracy()` has a None guard but this function does not.
- Fix: Add `if change_pct is None: continue`.

**B5: `update_module_failures()` / `update_health()` clobber race**
- File: `portfolio/health.py`
- Both do independent read-modify-write of health_state.json. `update_module_failures()`
  is called from `reporting.py` before `update_health()` in main.py. The second write
  clobbers `last_module_failures` from the first.
- Fix: Merge the two into a single update call, or make `update_health()` preserve
  existing fields it doesn't own.

**B6: `load_jsonl` missing OSError guard**
- File: `portfolio/file_utils.py:117`
- `load_jsonl` opens file with plain `open()`, no OSError guard. On Windows,
  PermissionError (antivirus lock) propagates uncaught. `load_jsonl_tail` catches this.
- Fix: Add `except OSError` guard matching `load_jsonl_tail` pattern.

### P2 — Medium

**B7: btc_proxy vote never tracked for accuracy**
- File: `portfolio/signal_engine.py:3162-3172`
- Synthetic btc_proxy vote for MSTR injected but not in SIGNAL_NAMES. Never logged
  by outcome_tracker. Accumulates zero accuracy data. Bypasses accuracy gate trivially.
- Fix: Add "btc_proxy" to SIGNAL_NAMES in tickers.py.

**B8: Double Telegram alert race in fx_rates.py**
- File: `portfolio/fx_rates.py:87`
- `_last_fx_alert` updated after network call outside lock. Two threads can both pass
  cooldown check and both send alerts.
- Fix: Set `_last_fx_alert` under lock before network call (optimistic lock).

**B9: outcome_tracker timestamp parsing without guard**
- File: `portfolio/outcome_tracker.py:421`
- `datetime.fromisoformat(entry["ts"])` without try/except. Corrupt JSONL entry
  aborts entire backfill batch.
- Fix: Wrap in try/except like other timestamp parsing in the file.

**B10: Leaked module-level loop variables**
- File: `portfolio/signal_engine.py:541`
- After `_TICKER_DISABLED_BY_HORIZON` validation, `_tk` and `_sigs` remain in namespace.
  `_k` and `_inner` deleted but not these.
- Fix: Add `del _tk, _sigs`.

### P3 — Low

**B11: Dead tod_factor application to discarded conf**
- File: `portfolio/signal_engine.py:3593`
- `conf *= tod_factor` on raw unweighted conf that is immediately overwritten. Dead code.
- Fix: Remove.

**B12: Dead correlation pairs for removed tickers**
- File: `portfolio/risk_management.py:731-738`
- CORRELATED_PAIRS lists AMD, AVGO, TSM, GOOGL, META, AMZN, AAPL — all removed.
- Fix: Remove dead pairs.

**B13: `import math` inside hot-path function**
- File: `portfolio/risk_management.py:290`
- `import math` inside `check_drawdown()` called every 60s. Python caches but bad style.
- Fix: Move to top-level.

**B14: FX constants duplicated without shared constant**
- Files: `portfolio/fx_rates.py:66`, `portfolio/risk_management.py:121`
- Both use 10.50 and band 7.0-15.0 independently.
- Fix: Extract constants to fx_rates.py.

**B15: mojibake dict duplicate keys in message_store**
- File: `portfolio/message_store.py:37-48`
- Multiple `'â'` dict keys — Python keeps last only. Some mojibake goes unrepaired.
- Fix: Deduplicate or restructure.

---

## 2. Execution Batches

### Batch 1: Critical signal fix + crash guards (4 files, low risk)
1. B1: Disable calendar signal in tickers.py + shadow-track in signal_engine.py
2. B3: Division-by-zero guard in journal.py
3. B4: None guard in accuracy_stats.py
4. B10: Clean leaked module variables in signal_engine.py

### Batch 2: Financial correctness (3 files, medium risk)
1. B2: fx_rate fallback chain — shared constants in fx_rates.py, import in
   monte_carlo_risk.py and risk_management.py
2. B12: Remove dead correlation pairs in risk_management.py
3. B14: FX constants consolidation (same files as B2)

### Batch 3: State integrity + thread safety (4 files, medium risk)
1. B5: Health state clobber fix in health.py
2. B6: OSError guard in file_utils.py load_jsonl
3. B8: FX double-alert race in fx_rates.py
4. B9: Timestamp guard in outcome_tracker.py

### Batch 4: Cleanup (4 files, low risk)
1. B7: btc_proxy in SIGNAL_NAMES (tickers.py)
2. B11: Dead tod_factor line (signal_engine.py)
3. B13: import math to top level (risk_management.py)
4. B15: mojibake dict fix (message_store.py)

### Batch 5: Verify + document + ship
1. Full test suite
2. Update docs/SYSTEM_OVERVIEW.md
3. Merge, push, restart loops

---

## 3. Risk Assessment

- **Batch 1**: Low risk. Calendar disable is additive (already accuracy-gated everywhere).
  Guards are defensive additions.
- **Batch 2**: Low risk. Constant consolidation, dead code removal.
- **Batch 3**: Medium risk. health.py change alters write pattern. file_utils and
  fx_rates changes add guards. Need careful testing.
- **Batch 4**: Low risk. All additive or cleanup changes.

---

## 4. Deferred (too risky or too large for autonomous session)

- `generate_signal()` is 1100+ lines. Decomposition would be high-value but high-risk.
- signal_accuracy_ewma() dead code removal — may be needed for research.
- Metals loop crash recovery + Telegram unification — large scope, touches production.
- Dashboard auth hardening (CF header validation) — policy decision needed.
