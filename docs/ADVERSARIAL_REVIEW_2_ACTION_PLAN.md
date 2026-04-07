# Adversarial Review #2 — Prioritized Action Plan
**Date:** 2026-04-07
**Source:** Independent review + Agent reviews
**Total findings:** 33+ (N1-N33, plus prior C3/C6 still open)

---

## IMMEDIATE (Day 1) — Dead Code Risk Protection

These are the highest-impact, lowest-effort fixes. The system has defense-in-depth
architecture on paper but **none of the defensive layers are wired into production**.

### 1. Wire `record_trade()` into all execution paths (N1/C4)
**Effort:** 2 hours | **Impact:** Enables all overtrading guards
**Files:** Create a post-trade hook in Layer 2 execution path.
The challenge is Layer 2 runs as a Claude CLI subprocess — need to add
`record_trade()` call wherever transactions are appended to portfolio state.

### 2. Wire `check_drawdown()` into the main loop (N22)
**Effort:** 1 hour | **Impact:** Enables drawdown circuit breaker
**Where:** `portfolio/main.py` — call after computing portfolio value (line ~582).
If breached, set a flag that Layer 2 reads to suppress BUY signals.

### 3. Add account ID validation (N4)
**Effort:** 15 minutes | **Impact:** Prevents pension account orders
**Where:** `portfolio/avanza_session.py:_place_order()` — add:
```python
ALLOWED_ACCOUNTS = {"1625505"}  # ISK only
if str(account_id or DEFAULT_ACCOUNT_ID) not in ALLOWED_ACCOUNTS:
    raise ValueError(f"Account {account_id} not in allowlist")
```

### 4. Fix `_loading_keys` leak in `_cached_or_enqueue` (N21)
**Effort:** 30 minutes | **Impact:** Prevents permanent LLM signal loss
**Where:** `portfolio/shared_state.py` — add timeout check:
```python
# In _cached_or_enqueue, before checking _loading_keys:
_LOADING_EXPIRE = 300  # 5 minutes
# Clean stale loading keys
stale = [k for k in _loading_keys if ...]  # need timestamp tracking
```
Or simpler: clear ALL `_loading_keys` at start of each cycle in `main.py`.

### 5. Add locking to `health.py` read-modify-write (N10)
**Effort:** 30 minutes | **Impact:** Prevents health data loss
**Where:** `portfolio/health.py` — add a module-level `threading.Lock()`.

### 6. Fix `/mode` symlink destruction — TICKING TIME BOMB (N31)
**Effort:** 5 minutes | **Impact:** Prevents permanent config breakage
**Where:** `portfolio/telegram_poller.py:150-160` — resolve symlink before write:
```python
resolved_path = Path(CONFIG_FILE).resolve()
atomic_write_json(resolved_path, config)
```

### 7. Redact Telegram bot token from retry logs (N32)
**Effort:** 15 minutes | **Impact:** Prevents credential leakage
**Where:** `portfolio/http_retry.py` — mask URLs containing `api.telegram.org/bot`

### 8. Fix Sortino ratio denominator (N23)
**Effort:** 5 minutes | **Impact:** Correct risk-adjusted metrics
**Where:** `portfolio/equity_curve.py:246` — change:
```python
# BEFORE:
downside_var = sum(r ** 2 for r in downside_returns) / len(downside_returns)
# AFTER:
downside_var = sum(r ** 2 for r in downside_returns) / len(daily_rets_dec)
```

---

## SHORT-TERM (This Week)

### 7. Add file locking to `atomic_append_jsonl` (N2)
### 8. Fix circuit breaker HALF_OPEN stuck state (N24)
### 9. Cache peak_value instead of scanning history (N5)
### 10. Delete or wire `SignalWeightManager` (C6)
### 11. Fix Avanza order confirmation to use order ID (N11)
### 12. Add pre-trade volume invariant to order flow (C3)

---

## MEDIUM-TERM (This Month)

### 13. Refactor metals_loop to use signal_engine.py (N3/A1)
### 14. Add HALF_OPEN timeout to circuit breaker (N24)
### 15. Fix Monte Carlo ATR-to-vol candle frequency assumption
### 16. Upgrade FOMC dates to dynamic fetching (N13)
### 17. Convert trade guards to `severity: "block"` (N14)
### 18. Run post-cycle tasks asynchronously (N12)

---

## Meta-Observation

The most alarming finding is not any single bug — it's the **pattern** of defense
layers that exist as code but are never activated:

| Layer | Function | Status |
|-------|----------|--------|
| Overtrading guards | `record_trade()` + `check_overtrading_guards()` | DEAD CODE |
| Drawdown breaker | `check_drawdown()` | DEAD CODE |
| Signal weight adaptation | `SignalWeightManager.batch_update()` | DEAD CODE |
| Trade guard blocking | `severity: "block"` | NEVER USED |

The system **appears** well-protected but has **zero automated risk management** in production.
All four safety nets exist in code, pass their tests, but are never wired into the runtime.

This is a classic "defense theater" anti-pattern in trading systems. The fix priority should be:
1. Wire the defenses (days 1-2)
2. Add integration tests that verify the defenses FIRE in realistic scenarios
3. Add a "defense status" dashboard panel showing which guards are active vs dormant
