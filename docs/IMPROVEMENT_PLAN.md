# Improvement Plan — Auto-Session 2026-04-10

Updated: 2026-04-10
Branch: improve/auto-session-2026-04-10
Status: **COMPLETE**

## Context

The codebase is extremely mature — 182+ bugs fixed across dozens of sessions,
242 test files, ~5,994 tests, solid architecture with atomic I/O, thread-safe
caching, and comprehensive signal gating. This session focuses on:

1. Fixing stale documentation (signal counts, ticker counts, test counts)
2. Cleaning remaining ruff violations (90 total, 3 auto-fixable)
3. Fixing reimplemented builtins and collapsible patterns in metals modules
4. Updating SYSTEM_OVERVIEW.md to reflect current reality

---

## 1. Documentation Drift (CRITICAL)

### DOC-1: CLAUDE.md signal/ticker counts are stale
- **File**: `CLAUDE.md`
- **Issue**: Says "30 active signals" and "12 Tier-1 instruments" but:
  - Actual: 36 total signals, 32 active, 4 disabled
  - Actual: 5 Tier-1 instruments (BTC-USD, ETH-USD, XAU-USD, XAG-USD, MSTR)
  - Removed tickers (PLTR, NVDA, MU, SMCI, TSM, TTWO, VRT) on Apr 9 to stay
    under 60s cycle cadence, but CLAUDE.md still lists them
  - Signal list is also stale: onchain (#35) and credit_spread_risk (#36) added
    but not reflected
- **Fix**: Update all counts, ticker table, signal inventory, and applicable
  signal counts (crypto/stocks/metals)
- **Impact**: Layer 2 Claude reads CLAUDE.md for context — stale counts cause
  confusion and suboptimal decisions

### DOC-2: SYSTEM_OVERVIEW.md is stale
- **File**: `docs/SYSTEM_OVERVIEW.md`
- **Issue**: Says "34 total, 30 active" signals, "20 instruments", "159 test files".
  Currently: 36 total, 32 active, 5 instruments, 242 test files.
  Also missing: funding rate re-enable (3h-only), onchain signal, G/S ratio
  velocity, and the ticker reduction.
- **Fix**: Full update of section headers and counts
- **Impact**: Onboarding/reference accuracy

---

## 2. Lint Cleanup

### LINT-1: 3 unsorted import blocks (auto-fixable)
- `portfolio/bert_sentiment.py:142`
- `portfolio/llm_batch.py:199`
- `portfolio/signal_engine.py:3`
- **Fix**: `ruff check --fix --select I001`
- **Impact**: None (cosmetic)

### LINT-2: 4 suppressible exceptions in metals_loop.py
- Lines 821, 3020, 6049, 6773
- `try/except/pass` → `contextlib.suppress()`
- **Fix**: Manual — these are in production-critical code
- **Impact**: Code clarity, ~4 lines saved per site

### LINT-3: 2 reimplemented builtins (SIM110)
- `data/metals_loop.py:938` — for loop → `return any(...)`
- `data/metals_swing_trader.py:1519` — for loop → `return any(...)`
- **Fix**: Replace with `any()` builtin
- **Impact**: Readability

### LINT-4: 5 collapsible if statements (SIM102)
- `data/metals_loop.py:2949, 5511, 6569`
- `data/metals_swing_trader.py:1386, 1468`
- **Fix**: Merge nested `if` into single `if x and y:`
- **Impact**: Readability

---

## 3. Test Count Verification

### TEST-1: Verify test suite baseline
- Run full test suite in parallel
- Document current pass/fail counts
- Confirm pre-existing failures list in TESTING.md is still accurate

---

## 4. Ordering & Dependencies

### Batch 1: Documentation fixes (no code changes, no tests needed)
1. `CLAUDE.md` — Update signal counts, ticker table, applicable counts
2. `docs/SYSTEM_OVERVIEW.md` — Full refresh of all count-dependent sections

### Batch 2: Ruff auto-fix (safe, broad)
1. Run `ruff check --fix --select I001` on portfolio/
2. Run tests to verify

### Batch 3: Manual lint fixes — metals modules
1. `data/metals_loop.py` — SIM105 (4 sites), SIM110 (1 site), SIM102 (3 sites)
2. `data/metals_swing_trader.py` — SIM110 (1 site), SIM102 (2 sites)
3. Run tests to verify

### Batch 4: Documentation finalization
1. Update `docs/TESTING.md` test counts if changed
2. Update SYSTEM_OVERVIEW known issues section
3. Final accuracy of all documentation

---

## 5. Risk Assessment

- **Very low risk**: Documentation updates and lint fixes only
- **No behavioral changes** to production trading logic
- **No new features** — pure accuracy and quality improvement
- **Full test suite** after each code batch verifies no regressions
- metals_loop.py changes are cosmetic (contextlib.suppress, any(), if-merge)
  and cannot change behavior
