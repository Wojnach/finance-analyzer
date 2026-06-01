# Improvement Plan — 2026-06-01 Auto Session

**Branch:** `improve/auto-session-2026-06-01`
**Sources:** 5 parallel exploration agents + P0 verification agent + cross-reference
with May 21 (419 findings) and May 31 (81 findings) adversarial review syntheses.
All P0 findings independently verified against current code at `f10c63ce`.

---

## 1. Bugs & Problems Found

### P0 — Fix immediately

| # | File | Bug | Impact | Fix |
|---|------|-----|--------|-----|
| B1 | `portfolio/outcome_tracker.py:218-252` | `_fetch_historical_price` uses `interval="1h"` and returns close (`data[0][4]`). Target price can be up to 59 min late. | All short-horizon accuracy stats biased — 33% error at 3h, up to 100% at 1h. Every accuracy-gated decision, IC weight, and Mode-B probability affected. **#1 correctness bug.** | Change to `interval="1m"`, return `data[0][1]` (open price). |
| B2 | `portfolio/agent_invocation.py:~1628-1680` | `auth_error` status writes no journal stub — `failed` and `incomplete` do but `auth_error` falls through. | Auth outages leave zero journal record. Contract violations fire. Outage invisible to journal consumers. | Add `elif status == "auth_error"` branch mirroring `failed`. |
| B3 | `portfolio/signal_engine.py:650,3908,4582` | `_cross_ticker_consensus` keyed by `ticker` only — all 7 horizons overwrite the same entry. | MSTR btc_proxy reads whichever BTC horizon finished last. Can flip vote direction. | Key by `(ticker, horizon)`, lookup by `(ticker, horizon)`. |
| B4 | `portfolio/fx_rates.py:47-53` | Sanity-check failure falls through to stale-cache block silently — serves old rate when live API returns bad data. | FX rate could be days stale with no distinct warning. Portfolio valuations drift. | Explicit early return after sanity-check failure with logged distinction. |
| B5 | `data/metals_loop.py:~6986` | `for t, _ in SILVER_ALERT_LEVELS` then `t[0]` — `t` is a float, `TypeError`. | Startup crash when silver positions active. | Fix unpack to use float directly: `f'{t}%'`. |

### P1 — Fix this session

| # | File | Bug | Impact | Fix |
|---|------|-----|--------|-----|
| B6 | `portfolio/risk_management.py:728-739` | `_CORRELATED_PAIRS = []` on transient import failure permanently disables correlation risk. | Correlation risk gate dead for entire process lifetime. | Use `_NOT_LOADED` sentinel; retry on next call. |
| B7 | `dashboard/app.py:1115-1126` | Signal heatmap hardcodes 30 signals from outdated layout. Missing active signals, includes removed. | Heatmap shows phantom HOLDs, missing real signals. | Generate dynamically from `tickers.SIGNAL_NAMES` + signal registry. |
| B8 | `scripts/check_critical_errors.py` | No auto-resolve: stale `contract_violation` entries (fixed May 30) and mislabeled `avanza_account_mismatch` stay unresolved forever. | Fix-agent budget burn. 31 phantom unresolved entries. Operator can't triage. | Add auto-resolve: resolve when category hasn't fired in 3+ days and a post-dated resolution exists. |

### P2 — Document and defer

| # | File | Bug | Note |
|---|------|-----|------|
| B9 | `signal_engine.py:4175` | 3d/5d/10d horizons collapse to 1d accuracy | Acknowledged TODO. Complex fix, needs data migration. |
| B10 | `claude_gate.py:343` | `_count_today_invocations()` full JSONL scan on every call | Perf degradation. Switch to `load_jsonl_tail`. |
| B11 | `metals_loop.py:1692` | `_underlying_prices` dict race (fast-tick vs LLM read) | Structurally racy but GIL-protected for single-key ops. |

---

## 2. Documentation Fixes

| # | What | Where | Fix |
|---|------|-------|-----|
| D1 | Route count stale | `CLAUDE.md` | Update "33 endpoints" → actual count (~55 including house blueprint). |
| D2 | Signal counts | `docs/SYSTEM_OVERVIEW.md` | Sync module/signal counts with current state. |
| D3 | Known issues | `docs/SYSTEM_OVERVIEW.md` | Update "Known Issues" section with what's fixed vs open. |

---

## 3. Batch Plan

### Batch 1: Critical correctness (5 files, 5 P0 fixes)

**Files:** `portfolio/outcome_tracker.py`, `portfolio/agent_invocation.py`,
`portfolio/signal_engine.py`, `portfolio/fx_rates.py`, `data/metals_loop.py`

| Change | Risk | Test needed |
|--------|------|-------------|
| B1: 1h→1m + open price | LOW | Update existing `test_outcome_tracker.py` |
| B2: auth_error journal stub | LOW | Add test for auth_error stub write |
| B3: cache key (ticker,horizon) | MED | Update `test_cross_ticker_consensus` |
| B4: fx_rates explicit return | LOW | Add test for sanity-check failure path |
| B5: SILVER_ALERT_LEVELS unpack | LOW | Add test for alert level formatting |

### Batch 2: P1 reliability (3 files)

**Files:** `portfolio/risk_management.py`, `dashboard/app.py`,
`scripts/check_critical_errors.py`

| Change | Risk | Test needed |
|--------|------|-------------|
| B6: sentinel + retry | LOW | Test import failure → retry succeeds |
| B7: dynamic signal heatmap | LOW | Test signal list matches registry |
| B8: auto-resolve stale criticals | MED | Test auto-resolve logic with fixtures |

### Batch 3: Documentation (2 files)

**Files:** `CLAUDE.md`, `docs/SYSTEM_OVERVIEW.md`

No code risk. Update counts and known issues.

---

## 4. Dependencies

All Batch 1 fixes are independent. Batch 2 has no Batch 1 deps.
Batch 3 depends on Batch 1+2 (docs reflect final state).

## 5. Impact Assessment

- B1 changes outcome backfill precision but does NOT retroactively fix existing entries.
  A re-backfill of short-horizon outcomes would be needed to fully correct accuracy stats.
- B3 changes cache key format — no persistence, pure in-memory. No migration needed.
- B7 changes the heatmap API response shape (signal list). Dashboard frontend
  should handle this gracefully since it iterates the returned list.
- B8 adds new logic to a startup script — if buggy, worst case is stale errors
  persist (same as today). Fail-safe.
