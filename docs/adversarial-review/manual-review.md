# Independent Manual Adversarial Review -- 2026-04-26

Reviewer: Claude Opus 4.6 (manual deep-read, independent of agent reviews)

## Methodology
Read key files in each subsystem. Focused on: consensus logic, order execution,
portfolio state management, data integrity, thread safety, and financial math.

---

## 1. signals-core

### P1-SC1: signal_engine.py:1505 -- _weighted_consensus is a 380-line god function
The consensus function applies 10+ sequential filter/gate/penalty steps: accuracy
gate, directional gate, correlation dedup, regime gating, horizon gating, crisis
mode, IC weighting, bias penalty, activation cap, persistence filter, Top-N gate,
circuit breaker relaxation. Each step mutates votes or weights in place. A single
bug in any step silently corrupts all downstream steps.

**Impact**: Any regression in one of the ~10 filter layers is invisible until it
causes a bad trade. The function complexity is itself a risk factor.

**Fix**: Extract each gate/filter into a named function. Test each independently.

### P1-SC2: signal_engine.py:1563 -- import math inside hot-path function
_weighted_consensus imports math inside the function body on every call.
CPython caches this, but it is inconsistent with the codebase and adds overhead
to a function called 5 tickers x 7 horizons = 35 times per cycle.

### P2-SC3: signal_engine.py:1 -- Stale docstring "32-signal" vs actual 33/52
Module docstring says "32-signal voting system" but CLAUDE.md documents 33 active
from 52 registered modules.

### P2-SC4: Signal persistence filter cold-start
_apply_persistence_filter (line 244) allows all signals through unfiltered on
the first cycle after process restart. Every loop restart bypasses the
documented "single-check MACD/RSI improvements are noise" protection for one cycle.

### P2-SC5: _adx_cache keyed by id(df) -- fragile
Line 34: _adx_cache is keyed by Python object ID. After GC, a new DataFrame could
get the same ID and return stale ADX. The _ADX_CACHE_MAX=200 bound limits this,
but the correctness guarantee is weak.

---

## 2. orchestration

### P1-OR1: main.py:646 -- pool.shutdown(wait=False, cancel_futures=True) does not stop running threads
cancel_futures=True only prevents queued futures from starting. Threads already
running inside _process_ticker continue to completion, potentially writing to
shared state (signals, caches) after the main loop has moved past the pool timeout.

**Impact**: After a BUG-178 timeout, orphaned threads may update _tool_cache,
_last_signal_per_ticker, or write signal log entries for a completed cycle.

### P1-OR2: main.py:1-235 -- ~235 lines of re-exports
The first 235 lines are backward-compat re-exports. This creates a false
dependency graph and means main.py cannot be imported without side effects.

### P2-OR3: trigger.py:155 -- set stored in JSON-destined dict
state["_current_tickers"] = set(signals.keys()). Python sets are not JSON-serializable.
Currently popped before save, but fragile.

### P2-OR4: Agent invocation lacks prompt sanitization
agent_invocation.py builds prompts with trigger reasons via string formatting.
No sanitization is applied to the reason strings.

---

## 3. portfolio-risk

### P0-PR1: risk_management.py:66 -- fx_rate defaults to 1.0 instead of ~10.5
When agent_summary is stale and fx_rate is missing, portfolio value is
computed with fx_rate=1.0. A $50K BTC position would be valued at 50K SEK
instead of 525K SEK. The drawdown circuit breaker triggers falsely.

**Impact**: False circuit breaker trigger blocks all Layer 2 invocations whenever
agent_summary is stale. Single most dangerous finding across subsystems.

### P1-PR2: trade_guards.py -- No lock on read-modify-write
record_trade and check_overtrading_guards both do load-mutate-save without
any lock. Multi-agent invocations can cause lost updates.

### P1-PR3: kelly_sizing.py:290 -- 500 SEK min vs documented 1000 SEK
The minimum trade threshold is 500 SEK but project rules require 1000 SEK.

### P2-PR4: equity_curve.py -- Round-trip PnL excludes fees
pnl_sek is gross PnL. Kelly criterion and all downstream metrics (Sharpe,
Calmar, expectancy) are computed on gross. For warrants with 0.8-1.0%
round-trip costs, this materially overestimates edge.

---

## 4. metals-core

### P0-MC1: fin_snipe_manager.py:64 -- MIN_STOP_DISTANCE_PCT = 1.0 violates 3% rule
The documented rule (metals-avanza.md, MEMORY.md) says "NEVER place stop-loss
within 3% of current bid." The code enforces only 1%. A stop 1.5% below bid on
a volatile 5x warrant will fill from normal spread movement.

### P1-MC2: orb_predictor.py:32-35 -- DST-blind morning window
ORB morning constants are hardcoded for UTC+1 (winter). From late March through
late October, the ORB window is off by 1 hour.

### P1-MC3: metals_loop.py:542 -- Raw json.load() instead of file_utils
Violates CLAUDE.md rule 4. Under concurrent write, json.load() can see a
truncated file and lose position state.

### P1-MC4: fin_snipe_manager.py:1590 -- Budget assigned per-instrument, not split
--budget 50000 --orderbook A --orderbook B gives 50K to each, deploying 100K total.

---

## 5. avanza-api

### P1-AV1: avanza_session.py:49 -- RLock is correct but indicates design smell
The need for reentrant locking indicates callers hold the lock while calling
methods that also acquire it. Should be restructured.

### P2-AV2: avanza_orders.py:131 -- CONFIRM matches most recent, not specific order
When multiple orders are pending, user cannot confirm a specific one.

### P2-AV3: avanza_session.py ALLOWED_ACCOUNT_IDS -- Good
Account whitelist {"1625505"} correctly prevents wrong-account trading. Positive.

---

## 6. signals-modules

### P2-SM1: Error handling pattern -- modules return HOLD on exception
Nearly all 40 modules use catch-all except Exception. Programming errors
(TypeError, AttributeError) are silently converted to HOLD. Should log at WARNING.

### P2-SM2: signals/forecast.py -- Chronos accuracy 54.5% barely passes 47% gate
The signal contributes near-zero edge while consuming GPU resources.

---

## 7. data-external

### P2-DE1: No circuit breaker for yfinance
data_collector.py has circuit breakers for Binance and Alpaca but NOT for
yfinance. A yfinance outage causes repeated slow timeouts with no backoff.

### P2-DE2: Binance 10m interval correctly avoided
The known invalid "10m" interval is not used anywhere. Positive observation.

---

## 8. infrastructure

### P1-IN1: dashboard/app.py:46 -- CORS Access-Control-Allow-Origin: *
Combined with optional token auth, any website on the local network can read
portfolio data. On a shared network this is exploitable.

### P2-IN2: file_utils.py:128 -- load_jsonl logs malformed lines at DEBUG
Corrupt JSONL lines are silently skipped with only DEBUG log. In production,
DEBUG is not enabled, so JSONL corruption is invisible.

---

## Cross-Cutting Findings

### P1-XC1: Inconsistent fx_rate defaults across modules
- risk_management.py: 1.0 (DANGEROUS)
- monte_carlo_risk.py: 10.0
- daily_digest.py: 10.5
- exit_optimizer.py: 10.85
There should be ONE default in a single location.

### P2-XC2: Thread safety model is ad-hoc
Each module implements its own locking strategy. Some use threading.Lock
(portfolio_mgr, health), some use threading.RLock (avanza_session), some
have no locks (trade_guards). No documented concurrency model.

### P2-XC3: Timestamp convention inconsistency
Most code uses datetime.now(UTC).isoformat(). crypto_scheduler.py uses
datetime.now().astimezone().isoformat() (local time). This causes 1-2h offsets
in cross-log analysis.
