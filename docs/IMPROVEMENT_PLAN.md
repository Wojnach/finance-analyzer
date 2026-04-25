# Improvement Plan — Auto Session 2026-04-25

Based on deep exploration by 4 parallel agents (signal engine, portfolio/risk,
infrastructure, metals/trading) plus manual verification of all P0/P1 findings.

Previous sessions fixed: BUG-219 pnl_pct (P0), BUG-220 outcome_tracker (P2),
BUG-221 daily_digest tz (P2), rate limiter (P1), drawdown NaN (P1), cache None (P2),
smart_money disable + per-ticker blacklist (2026-04-24).

## Exploration Summary

### Agent Finding Triage (4 agents, ~50 raw findings)

**False Positives Rejected (32):**
- `file_utils.py:117` file handle leak — FileNotFoundError prevents `open()`, no handle created
- `shared_state.py:89` _loading_keys race — `add()` is INSIDE the `with _cache_lock:` block
- `health.py:23` concurrent read — `atomic_write_json` uses `os.replace` (atomic on NTFS)
- `market_timing.py:42` DST formula — verified correct for 2026 (March 29 = Sunday)
- `equity_curve.py:245` Sortino wrong — H19 comment is correct per Morningstar standard
- `risk_management.py:640` concentration logic — multiple tickers ≤40% each is by design
- `agent_invocation.py:200` monotonic poisoning — `_agent_start` always from `time.monotonic()`
- `golddigger/risk.py:98` divide-by-zero — already guarded with `if entry_ask <= 0: return`
- `monte_carlo_risk.py:278` C9 t-copula — ALREADY FIXED in current code
- `signal_engine.py` P2-A core_active gate — intentional design: enhanced signals can't create consensus alone
- `signal_engine.py:1613-1614` double regime gating — idempotent (HOLD→HOLD = no-op)
- `signal_engine.py:1884` confluence > 1.0 — already capped with `min(..., 1.0)`
- `signal_engine.py:34` ADX cache — bounded by `_ADX_CACHE_MAX = 200` + GC
- `metals_loop.py:1593-1608` signal age — already handles internal timestamp vs mtime correctly
- `avanza_session.py:1059` cancel_all_stop_losses race — fail-closed semantics correctly implemented
- Plus 17 other findings that were either already fixed, by-design, or code-quality-only

### Confirmed Bugs

- **BUG-223: place_stop_loss missing sell_price validation for non-trailing stops**
  `avanza_session.py:762-780` — For MONETARY (non-trailing) stop-loss orders, `sell_price`
  is passed directly to the API payload without validating `sell_price > 0`. A sell_price
  of 0 for a LESS_OR_EQUAL trigger would execute as a market sell at whatever price exists.
  The trailing stop (FOLLOW_DOWNWARDS with sell_price=0) is intentional per the docstring,
  but the code doesn't distinguish.
  - **Severity**: P1 — safety-critical, could cause market sell at worst price
  - **Fix**: Guard `sell_price > 0` for MONETARY+non-trailing stop types. Allow 0 only for
    trailing stops (trigger_type in {"FOLLOW_DOWNWARDS", "FOLLOW_UPWARDS"}).

- **BUG-224: extra_info["_voters"] records pre-persistence-filter count**
  `signal_engine.py:3167` — After `_apply_persistence_filter()` at line 3129 reduces
  active voters, the `active_voters` variable (computed at line 2934) is still recorded
  as the voter count. Accuracy tracking and Layer 2 decisions see inflated voter counts.
  - **Severity**: P2 — metadata inaccuracy, affects accuracy tracking downstream
  - **Fix**: Compute post-persistence voter count and record both pre/post in extra_info.

- **BUG-225: Sharpe inner mean recomputed N times**
  `equity_curve.py:236` — `sum(daily_rets_dec) / len(daily_rets_dec)` is evaluated
  inside the generator expression, making it O(n^2) instead of O(n). Not a correctness
  bug — the result is identical — but wastes CPU on portfolios with years of history.
  - **Severity**: P3 — performance only
  - **Fix**: Extract `mean_dec` before the generator.

- **BUG-226: Exit optimizer hold-to-close EV omits cost model**
  `exit_optimizer.py:~623-634` — Market exit candidate uses `_compute_pnl_sek()` which
  includes costs. Hold-to-close candidate computes `fallback_pnl` without explicitly
  applying cost model. Biases EV toward holding.
  - **Severity**: P2 — subtly inflates hold recommendation probability
  - **Fix**: Apply cost model consistently in hold-to-close path.

### Documentation Updates Needed

- **Signal counts**: CLAUDE.md says "50 Modules . 34 Active" but actual is
  51 modules / 33 active / 18 disabled (mahalanobis_turbulence added, smart_money disabled)

---

## Implementation Batches

### Batch 1: BUG-223 stop-loss sell_price validation (2 files, ~15 lines)

**Files**: `portfolio/avanza_session.py`, `tests/test_avanza_session.py`

1. In `place_stop_loss()` after line 748, add validation:
   ```python
   # Trailing stops (FOLLOW_DOWNWARDS/UPWARDS) legitimately use sell_price=0
   # (market order on trigger). Non-trailing MONETARY stops must have sell_price > 0.
   _TRAILING_TYPES = {"FOLLOW_DOWNWARDS", "FOLLOW_UPWARDS"}
   if trigger_type not in _TRAILING_TYPES and value_type == "MONETARY":
       if sell_price <= 0:
           raise ValueError(
               f"Non-trailing stop-loss requires sell_price > 0, got {sell_price}"
           )
   ```

2. Write tests:
   - `test_stop_loss_rejects_zero_sell_price_monetary` — verify ValueError
   - `test_stop_loss_allows_zero_sell_price_trailing` — verify trailing is allowed

### Batch 2: BUG-224 voters count + BUG-225 Sharpe fix (3 files, ~15 lines)

**Files**: `portfolio/signal_engine.py`, `portfolio/equity_curve.py`,
`tests/test_signal_engine_circuit_breaker.py`

1. In `generate_signal()` after line 3136 (persistence filter), compute post-filter count:
   ```python
   post_persistence_voters = sum(
       1 for v in consensus_votes.values() if v in ("BUY", "SELL")
   )
   ```
   At line 3167, record both:
   ```python
   extra_info["_voters"] = active_voters  # pre-filter (for compatibility)
   extra_info["_voters_post_filter"] = post_persistence_voters
   ```

2. In `equity_curve.py:236`, extract mean before generator:
   ```python
   mean_dec = sum(daily_rets_dec) / len(daily_rets_dec)
   daily_std_dec = math.sqrt(
       sum((r - mean_dec) ** 2 for r in daily_rets_dec)
       / (len(daily_rets_dec) - 1)
   )
   ```

### Batch 3: Documentation updates (2 files, ~20 lines)

**Files**: `CLAUDE.md`, `docs/SYSTEM_OVERVIEW.md`

1. Update signal counts: "50 Modules . 34 Active" -> "51 Modules . 33 Active . 18 Disabled"
2. Update disabled signals list (add smart_money, mahalanobis_turbulence)
3. This plan document gets committed.

---

## Backlog (deferred -- not this session)

- **BUG-226**: Exit optimizer cost model — needs deeper investigation of `_compute_pnl_sek`
  call chain; deferred to avoid regression risk in live trading
- **ARCH-1**: outcome_tracker JSONL->SQLite migration (performance)
- **Signal engine**: linear factor model using post-gated votes (P2, needs validation study)
- **GPU gate**: psutil absence fallback for stale lock recovery (P2 edge case)

## Dependency Ordering

Batch 1 -> Batch 2 -> Batch 3

No cross-dependencies between batches. Sequential ordering keeps commits clean.
