# Dual Adversarial Review — 2026-04-12

**Date**: 2026-04-12
**Scope**: Full codebase, 8 subsystems, 142 modules
**Method**: Dual adversarial review — independent human review + 8 parallel code-reviewer
agents, followed by cross-critique in both directions.
**HEAD**: `56d7180` (main, "fix(signals): reduce consensus noise")

---

## Executive Summary

The finance-analyzer codebase is a mature, feature-rich quantitative trading system with
extensive safety mechanisms (circuit breakers, trade guards, account whitelists, atomic I/O).
The codebase shows evidence of iterative hardening through numerous "BUG-NNN" fixes (180+
tracked bugs).

**This review found 3 P1 bugs and 10+ P2 bugs** that affect signal accuracy, loop
reliability, and risk management:

| Finding | Severity | Subsystem | One-liner |
|---------|----------|-----------|-----------|
| SC-I-001 | **P1** | signals-core | Double regime gating nullifies BUG-158 per-ticker exemptions |
| CROSS-001 | **P1** | cross-subsystem | C10 dead-signal-trap fix incomplete — `_raw_votes` never consumed |
| OR-I-001 | **P1** | orchestration | ThreadPoolExecutor `with` deadlocks on stuck threads |
| SC-I-002 | P2 | signals-core | Dynamic horizon weights cache stale static fallback 1h |
| OR-I-003 | P2 | orchestration | No timeout on pre-loop FX rate fetch |
| PR-I-001 | P2 | portfolio-risk | Drawdown uses avg_cost when live prices missing |
| PR-I-003 | P2 | portfolio-risk | No negative-cash validation |
| MC-I-002 | P2 | metals-core | CET/CEST timezone fallback hardcodes UTC+1 |
| MC-I-003 | P2 | metals-core | Safety param fallback values may diverge |
| AV-I-003 | P2 | avanza-api | CORS wildcard on dashboard |
| SM-I-001 | P2 | signals-modules | social_sentiment uses raw json.loads |
| DE-I-001 | P2 | data-external | Rate limit enforced only by caller discipline |
| DE-I-002 | P2 | data-external | FX rate fallback cascades wrong valuations |
| IN-I-001 | P2 | infrastructure | atomic_append_jsonl not truly atomic |

---

## Subsystem Health Matrix

| Subsystem | P1 | P2 | P3 | Health |
|-----------|----|----|----| -------|
| signals-core | 1 | 1 | 2 | ⚠️ WARN |
| orchestration | 1 | 1 | 1 | ⚠️ WARN |
| portfolio-risk | 0 | 2 | 1 | ✅ OK |
| metals-core | 0 | 2 | 0 | ✅ OK |
| avanza-api | 0 | 1 | 1 | ✅ GOOD |
| signals-modules | 0 | 1 | 1 | ✅ GOOD |
| data-external | 0 | 2 | 0 | ✅ OK |
| infrastructure | 0 | 1 | 2 | ✅ GOOD |
| cross-subsystem | 1 | 0 | 0 | ⚠️ WARN |

---

## P1 Findings — Must Fix

### 1. SC-I-001: Double regime gating nullifies BUG-158 per-ticker exemptions

**Location**: `portfolio/signal_engine.py:749` vs `signal_engine.py:1754-1759`

**What happens**: `generate_signal()` at line 1754-1759 exempts signals with ≥60%
per-ticker accuracy from regime gating (BUG-158 fix). The exempted signal keeps its
original vote (e.g., `fear_greed = "BUY"` on XAG-USD with 93.8% accuracy). Then at
line 1971, `_weighted_consensus(votes, ...)` is called. Inside that function at line
749, a NEW local `votes` dict is created with the FULL regime gate applied:

```python
regime_gated = _get_regime_gated(regime, horizon)
votes = {k: ("HOLD" if k in regime_gated else v) for k, v in votes.items()}
```

This `_get_regime_gated()` call returns the full gate set WITHOUT per-ticker exemptions.
So `fear_greed` gets re-gated to HOLD inside the weighted consensus, which is the
PRIMARY action used (line 2011: `action = weighted_action`).

**Impact**: Every per-ticker exemption granted by BUG-158 is silently nullified.
Proven alpha is being thrown away on every cycle. The BUG-158 exemption is dead code.

**Fix**: Add a `regime_gated_override` parameter to `_weighted_consensus()`:
```python
def _weighted_consensus(votes, ..., regime_gated_override=None):
    regime_gated = regime_gated_override or _get_regime_gated(regime, horizon)
```
Then pass `regime_gated_effective` from `generate_signal()`.

---

### 2. CROSS-001: C10 dead-signal-trap fix incomplete — `_raw_votes` never consumed

**Location**: `portfolio/signal_engine.py:2002` (writer), `portfolio/outcome_tracker.py:122` (reader)

**What happens**: The C10 fix at line 1726 captures `raw_votes = dict(votes)` before any
gating rewrites votes to HOLD. These are stored at line 2002:
```python
extra_info["_raw_votes"] = raw_votes
```

But `outcome_tracker.py:log_signal_snapshot()` at line 122-124 reads:
```python
passed_votes = extra.get("_votes")  # POST-gating votes!
```

Since `_votes` contains post-gating data where regime-gated signals are HOLD, and
accuracy_stats.py skips HOLD votes, gated signals never accumulate accuracy data.
Searching for `_raw_votes` across the entire codebase (`grep -r "_raw_votes"`) confirms
NO downstream module reads this field.

**Impact**: 13+ signals gated in ranging regime, 7+ in trending-up, 8+ in trending-down
are permanently trapped. They can never demonstrate improved accuracy to earn un-gating.
This is a systemic issue that prevents the signal system from self-healing.

**Fix**: In `outcome_tracker.py:log_signal_snapshot()`:
```python
passed_votes = extra.get("_raw_votes") or extra.get("_votes")
```

---

### 3. OR-I-001: ThreadPoolExecutor `with` block deadlocks on stuck threads

**Location**: `portfolio/main.py:555`

**What happens**: The main loop uses:
```python
with ThreadPoolExecutor(max_workers=max_workers) as pool:
    futures = {pool.submit(_process_ticker, name, source): name ...}
    try:
        for future in as_completed(futures, timeout=180):
            ...
    except TimeoutError:
        for f in futures:
            f.cancel()  # Only cancels not-yet-started futures!
```

When the `with` block exits (line 555), Python calls `pool.__exit__()` which calls
`pool.shutdown(wait=True)`. This BLOCKS until ALL submitted futures complete — including
stuck threads that `cancel()` couldn't stop.

If a ticker thread is hung on blocking network I/O (e.g., Binance API with no timeout),
the main loop blocks indefinitely. The system's crash recovery (exponential backoff,
Telegram alerts) only handles exceptions, not indefinite blocking.

**Impact**: A single network hang can freeze the entire 24/7 trading system. The BUG-178
timeout handler fires correctly but the `with` block prevents the loop from continuing.

**Fix**: Replace the `with` block with explicit pool management:
```python
pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ticker")
try:
    # ... same future logic ...
except TimeoutError:
    for f in futures:
        f.cancel()
finally:
    pool.shutdown(wait=False, cancel_futures=True)  # Python 3.9+
```

---

## P2 Findings — Should Fix

### 4. SC-I-002: Dynamic horizon weights cache stale static fallback for 1 hour

When `accuracy_cache.json` is temporarily unavailable (I/O glitch, file locked),
`_compute_dynamic_horizon_weights()` returns the static `HORIZON_SIGNAL_WEIGHTS`
fallback. This result gets cached by `_cached()` for `_DYNAMIC_HORIZON_WEIGHT_TTL =
3600` (1 hour). When the accuracy cache becomes available again (~60s later), stale
static weights persist for up to 59 more minutes.

### 5. OR-I-003: No timeout on pre-loop FX rate fetch

`fetch_usd_sek()` at `main.py:417` runs before the thread pool, outside any timeout.
If the FX API hangs, the entire cycle stalls before signal processing begins.

### 6. PR-I-001: Drawdown uses avg_cost_usd when live prices missing

`risk_management.py:79-82` falls back to entry price when `agent_summary` lacks live
data. If prices have dropped 50% since entry, the drawdown calculation sees the
original price, potentially missing a real drawdown that should trigger the circuit
breaker.

### 7. PR-I-003: No negative-cash validation in portfolio_mgr

`portfolio_mgr.py:_validated_state()` doesn't check `cash_sek >= 0`. A double-deduction
from concurrent trades could create negative cash.

### 8. MC-I-002: CET/CEST timezone fallback hardcodes UTC+1

`metals_execution_engine.py:51-52` uses `timedelta(hours=1)` when `zoneinfo` is
unavailable. During CEST (late March to late October), this is off by 1 hour, causing
`hours_to_metals_close()` to report 1 extra hour of trading time.

### 9. MC-I-003: Safety parameter fallback values may diverge

When `metals_swing_config` import fails silently, fallback values for
`MIN_BARRIER_DISTANCE_PCT`, `MIN_SPREAD_PCT` etc. may not match production.

### 10. AV-I-003: CORS wildcard on dashboard

`dashboard/app.py:42` sets `Access-Control-Allow-Origin: *`, allowing any website to
read portfolio data via the dashboard API.

### 11. SM-I-001: social_sentiment uses raw json.loads

`social_sentiment.py:34,67` uses `json.loads(resp.read())` instead of requests.json(),
violating project I/O conventions.

### 12. DE-I-001: Rate limit enforced only by caller discipline

Alpha Vantage 25/day limit is enforced by callers using `_alpha_vantage_limiter`, not
by the API module. Direct calls bypass protection.

### 13. DE-I-002: FX rate fallback cascades wrong valuations

If all FX sources fail, stale/wrong rate cascades to portfolio value, drawdown, and
position sizing calculations.

### 14. IN-I-001: atomic_append_jsonl is not truly atomic

A crash between `f.write(line)` and `os.fsync()` can leave a partial JSON line.
`load_jsonl` handles this by skipping malformed lines, but the data is lost.

---

## P3 Findings — Nice to Fix

| ID | Subsystem | Description |
|----|-----------|-------------|
| SC-I-003 | signals-core | Unanimity penalty uses raw vote counts, not weighted |
| SC-I-004 | signals-core | Correlation clustering is order-dependent (greedy) |
| OR-I-002 | orchestration | main.py re-exports 50+ internal symbols |
| AV-I-002 | avanza-api | Telegram CONFIRM matches most recent order only |
| SM-I-002 | signals-modules | Signal module exceptions produce silent HOLD |
| IN-I-002 | infrastructure | Dashboard cache has no max-size bound |
| IN-I-003 | infrastructure | Journal entries not validated before append |

---

## Validated Non-Issues

| Claim | Verdict | Evidence |
|-------|---------|----------|
| AV-I-001: Account whitelist not enforced in avanza_session.py | **FALSE POSITIVE** | Both `avanza_client.py:31` and `avanza_session.py:35,522,665` enforce `ALLOWED_ACCOUNT_IDS = {"1625505"}` with ValueError raises |
| Config validator might wipe config | **FALSE POSITIVE** | `config_validator.py` is read-only validation, never writes |
| Dashboard endpoints unauthenticated | **FALSE POSITIVE** | All `/api/*` routes have `@require_auth` decorator; auth degrades to open only when no token configured |

---

## Codex Agent Review Findings

*(Sections below populated as each background agent completes its review.)*

### signals-core agent
*(pending)*

### orchestration agent
*(pending)*

### portfolio-risk agent
*(pending)*

### metals-core agent
*(pending)*

### avanza-api agent
*(pending)*

### signals-modules agent
*(pending)*

### data-external agent
*(pending)*

### infrastructure agent
*(pending)*

---

## Cross-Critique

### Independent → Codex (Findings agents may have missed)

1. **CROSS-001 (dead signal trap)**: This requires tracing data flow across two subsystems
   (signal_engine.py → outcome_tracker.py). Single-subsystem reviewers would miss it
   because each file looks correct in isolation.

2. **SC-I-001 (double gating)**: The bug spans two functions in the same file. A reviewer
   focused on `_weighted_consensus` alone would see correct regime gating logic. A reviewer
   focused on `generate_signal` alone would see correct per-ticker exemptions. Only by
   tracing the data flow between them does the double-gating become visible.

### Codex → Independent (Findings agents found that I missed)

*(To be completed after agent results arrive.)*

---

## Recommendations

### Immediate (before next trading week)

1. **Fix SC-I-001**: Pass `regime_gated_effective` to `_weighted_consensus()` as override.
   Estimated effort: 30 minutes. Risk: low (additive change, no behavior change when
   no exemptions are active).

2. **Fix CROSS-001**: Use `_raw_votes` in `outcome_tracker.py:log_signal_snapshot()`.
   Estimated effort: 15 minutes. Risk: low (affects accuracy logging only, not live
   trading decisions). But impact is high — unlocks self-healing for ~28 gated signals.

3. **Fix OR-I-001**: Replace `with ThreadPoolExecutor(...)` with explicit shutdown.
   Estimated effort: 20 minutes. Risk: medium (need to verify thread cleanup doesn't
   leave resources dangling).

### Near-term (this week)

4. Fix PR-I-001: Flag stale drawdown valuations.
5. Fix MC-I-002: Require `zoneinfo`, remove UTC+1 fallback.
6. Restrict CORS to localhost/LAN.

### Backlog

7. Cache drawdown peak value (PR-I-002).
8. Add negative-cash guard.
9. Move rate limiting into API modules.
10. Bound dashboard cache size.

---

## Comparison with Previous Review (2026-04-05)

The April 5 review covered the same 8 subsystems. Key differences:

- **SC-I-001 is NEW**: The double-gating bug was introduced or obscured by the BUG-158
  fix itself (the exemption code was added but the inner gating wasn't updated).
- **CROSS-001 is NEW**: The C10 fix was added between reviews but the downstream consumer
  was never updated.
- **OR-I-001 persists**: The ThreadPoolExecutor issue was flagged in the April 5 review
  infrastructure section (config wipe hazard) but the pool deadlock aspect was not
  identified. The BUG-178 timeout handler (added April 10) partially addresses this
  but doesn't solve the `shutdown(wait=True)` blocking.
- **AV-I-001 resolved**: The account whitelist is now confirmed enforced in both paths
  (was a concern in the April 5 review).

---

## Methodology

This review used a "dual adversarial" methodology:

1. **Independent review**: Primary reviewer reads source code directly, traces data flows
   across module boundaries, searches for anti-patterns, and documents findings with
   file paths, line numbers, and code evidence.

2. **Codex agents**: 8 specialized `feature-dev:code-reviewer` agents launched in parallel,
   one per subsystem. Each agent receives the file list for its subsystem and instructions
   to look for bugs, race conditions, data corruption, and security vulnerabilities.

3. **Cross-critique**: Each side reviews the other's findings for:
   - **False positives**: Non-issues flagged incorrectly (downgrade or remove)
   - **False negatives**: Real issues missed (upgrade and add)
   - **Severity disagreements**: Different severity assessment of the same issue

The dual approach catches bugs that a single reviewer would miss due to mental model
bias and attention fatigue. Cross-critique forces evidence-based justification.
