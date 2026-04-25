# Dual Adversarial Review #8 — Synthesis Document
**Date:** 2026-04-25
**Methodology:** 8 parallel agent reviews + 1 independent review, cross-critiqued

## Executive Summary

Two independent review passes across 8 subsystems found **~52 confirmed P1/P2 findings** and
**~25 P3 quality issues**. The system has improved significantly since review #7 (Codex round
sanitization, per-horizon gating, IC-based weighting, loop contract hardening), but several
classes of bug persist:

1. **P1: Hardware trailing stop STILL broken** — Review #7 P1-12 (tuple/dict mismatch) is
   confirmed unfixed by metals-core agent. Every trade-queue buy fill has zero broker-level
   stop protection since the feature was enabled.
2. **P1: New signal `mahalanobis_turbulence` has wrong `_cached()` call** — Will crash with
   TypeError when enabled. Latent but blocks signal rollout.
3. **P1: Loss escalation permanently dead** — `pnl_pct` is never written to transactions,
   so the consecutive-loss cooldown multiplier (2x/4x/8x) never activates.
4. **P1: Avanza account whitelist gaps** — `metals_avanza_helpers` and `avanza_client` paths
   bypass the whitelist enforcement that `avanza_session` implements.
5. **P1: Digest crash kills all post-cycle tasks** — `_maybe_send_digest` is the only
   post-cycle task not wrapped in `_track()`, so a crash cascades to ~15 tasks.
6. **Systemic: `_cached()` dogpile returns None** — Multiple callers don't handle None,
   causing cold-start crashes or degraded signal quality.

---

## Cross-Critique: Agent Findings vs. Independent Review

### Agent found, Independent missed:
| Finding | Subsystem | Why missed |
|---------|-----------|------------|
| `pnl_pct` never written → loss escalation dead | portfolio-risk | Didn't trace write path from Layer 2 to transaction schema |
| `_maybe_send_digest` not in `_track()` → cascade failure | orchestration | Didn't read `_run_post_cycle` wrapper pattern carefully |
| Double-logging triggers on skip paths | orchestration | Didn't trace `_log_trigger` calls from both caller and callee |
| `change_pct=None` crashes `signal_accuracy_cost_adjusted` | signals-core | Focused on `signal_accuracy()` fix, missed parallel functions |
| Regime/ticker accuracy cache shared timestamp across horizons | signals-core | Didn't read regime cache write path (focused on read path) |
| Directional accuracy uses winner-take-all, not blended | signals-core | Didn't read `blend_accuracy_data` deeply enough |
| Drawdown blocker blocks both strategies on one breach | orchestration | Saw the loop but didn't trace the early-return semantics |
| Log rotation fsync gap + race with `atomic_append_jsonl` | infrastructure | Didn't read rotation write path closely |
| `_cached` stores pre-call timestamp (stale `now`) | infrastructure | Missed this subtle timing issue |
| `metals_avanza_helpers` no account whitelist | avanza-api | Didn't read `data/metals_avanza_helpers.py` |
| `avanza_client._place_order` no max order size cap | avanza-api | Didn't read TOTP client path |
| ORB predictor hardcodes UTC offsets for winter only | metals-core | Didn't read `orb_predictor.py` |
| fin_snipe_manager 5% cert stop too tight for 5x leverage | metals-core | Didn't read `_compute_stop_plan` |
| Fear & Greed unguarded `body["data"][0]` | data-external | Didn't read `fear_greed.py` deeply |
| `onchain_data._load_onchain_cache` missing `_coerce_epoch` | data-external | Didn't read onchain cache path |

### Independent found, Agent missed:
| Finding | Subsystem | Why missed |
|---------|-----------|------------|
| `_get_horizon_weights()` None during dogpile → TypeError | signals-core | Agents focused on accuracy logic, not cache-miss paths |
| `smart_money` globally disabled but still computed every cycle | signals-modules | Agents reviewed module internals, not dispatch efficiency |
| Persistence filter seeds `cycles=_PERSISTENCE_MIN_CYCLES` | signals-core | Agent found same issue but framed differently |
| `_extract_ticker` defaults to XAG-USD | orchestration | **Agent confirmed** (Finding 5 in orchestration) |
| CONFIRM flow matches most recent order (no order-ID) | avanza-api | **Agent confirmed** (P2-6 in avanza-api) |

### Both found independently (high confidence):
| Finding | Subsystem |
|---------|-----------|
| Persistence filter cold-start double-permissive | signals-core |
| Multi-agent specialist timeout blocks main loop | orchestration |
| `fx_rate=1.0` silent fallback | data-external + portfolio-risk |
| CONFIRM flow no order-ID verification | avanza-api |
| `_streaming_max` reads entire JSONL every call | portfolio-risk |
| Hardware trailing stop broken (review #7 unfixed) | metals-core |
| Playwright browser context no atexit cleanup | avanza-api |
| Lockfile accumulation in data/ | infrastructure |

---

## Consolidated P1 Findings (Highest Priority)

### P1-01: Hardware trailing stop NEVER placed — wrong function imported + wrong return type
**Source:** Agent (metals-core), also review #7 P1-12
**File:** `data/metals_loop.py:4773-4802`
**Impact:** Every trade-queue buy fill runs with ZERO broker-level stop protection since
`HARDWARE_TRAILING_ENABLED` was turned on. `place_stop_loss` from `avanza_control` doesn't
accept `trigger_type`/`value_type` kwargs → TypeError caught by bare `except Exception`.
Additionally, `result.get("status")` on a tuple return → AttributeError also caught silently.
**Status:** NOT FIXED since review #7.

### P1-02: `_cached()` arg order swapped in TWO disabled signals — will crash on enable
**Source:** Independent review + Agent (signals-modules)
**Files:** `portfolio/signals/mahalanobis_turbulence.py:99`, `portfolio/signals/complexity_gap_regime.py:92`
**Impact:** Both signals pass `_do_fetch` as the `ttl` parameter and `ttl=_CACHE_TTL` as keyword.
Python raises `TypeError: got multiple values for argument 'ttl'`. Both are disabled pending
validation. The instant either is enabled, every cycle crashes silently.
**Fix:** `return _cached("key", _CACHE_TTL, _do_fetch)` — swap positions 2 and 3.

### P1-03: `pnl_pct` never written to transactions → consecutive-loss escalation dead
**Source:** Agent (portfolio-risk)
**File:** `portfolio/trade_guards.py:240`, `portfolio/agent_invocation.py:720`
**Impact:** The overtrading prevention system's loss-streak escalation (2x/4x/8x cooldown)
has never activated. `pnl_pct` doesn't exist in the transaction schema.

### P1-04: `_maybe_send_digest` crash kills all ~15 post-cycle tasks
**Source:** Agent (orchestration)
**File:** `portfolio/main.py:282`
**Impact:** Unlike every other post-cycle task, digest is NOT wrapped in `_track()`. A crash
cascades through JSONL pruning, log rotation, accuracy snapshots, etc.
**Fix:** Wrap in `_track("digest", _maybe_send_digest, config)`

### P1-05: Avanza account whitelist bypassed in `metals_avanza_helpers` + `avanza_client`
**Source:** Agent (avanza-api)
**Files:** `data/metals_avanza_helpers.py:253,330,457`, `portfolio/avanza_client.py:326`
**Impact:** Page-based and TOTP order paths accept arbitrary account IDs without whitelist
check. Config corruption or future dynamic account_id could send orders to wrong account.

### P1-06: `fear_greed.py` unguarded `body["data"][0]` on empty API response
**Source:** Agent (data-external)
**File:** `portfolio/fear_greed.py:100`
**Impact:** Empty API response during maintenance → IndexError → entire F&G voter dead for cycle.

### P1-07: Log rotation no fsync before os.replace + race with atomic_append_jsonl
**Source:** Agent (infrastructure)
**File:** `portfolio/log_rotation.py:242-249`
**Impact:** Power failure during rotation can lose entire signal_log.jsonl kept portion.
Concurrent appends during rotation window are silently dropped.

### P1-08: Emergency mode sell+stop both placed for full position volume (overfill)
**Source:** Agent (metals-core)
**File:** `portfolio/fin_snipe_manager.py:948-984`
**Impact:** When `sell_naked=True` and `stop_naked=True`, both orders encumber full position.
Total = 2x position → Avanza rejection `short.sell.not.allowed`.

### P1-09: Double-logging of triggers on skip paths corrupts invocation stats
**Source:** Agent (orchestration)
**File:** `portfolio/main.py:822` + `portfolio/agent_invocation.py`
**Impact:** Skip paths log once from agent_invocation, then again from main.py with "skipped_busy",
inflating invocation counts and masking real skip reasons in loop contract.

---

## Consolidated P2 Findings (Important)

| # | Subsystem | File | Finding |
|---|-----------|------|---------|
| P2-01 | signals-core | signal_engine.py:901-909 | `_get_horizon_weights()` returns None during dogpile → TypeError |
| P2-02 | signals-core | signal_engine.py:258-268 | Persistence filter cold-start double-permissive (unfixed #7) |
| P2-03 | signals-core | accuracy_stats.py:359,610 | `change_pct=None` crashes `signal_accuracy_cost_adjusted` |
| P2-04 | signals-core | accuracy_stats.py:1117,1535 | Regime/ticker accuracy cache shared timestamp across horizons |
| P2-05 | signals-core | accuracy_stats.py:840-845 | Directional accuracy blend uses winner-take-all, not blended |
| P2-06 | signals-core | signal_engine.py:3087-3100 | Utility boost asymmetric — no penalty for negative-return signals |
| P2-07 | orchestration | agent_invocation.py:428 | Multi-agent specialists block main loop 30s synchronously |
| P2-08 | orchestration | agent_invocation.py:151 | `_extract_ticker` hardcoded XAG-USD fallback |
| P2-09 | orchestration | agent_invocation.py:382-392 | Drawdown blocker blocks both strategies when one breaches |
| P2-10 | orchestration | main.py:1081 | Stale config for all post-cycle tasks |
| P2-11 | orchestration | agent_invocation.py:754-776 | Agent timeout enforced only at cycle boundary (+600s overrun) |
| P2-12 | portfolio-risk | trade_guards.py | No file-level locking on trade_guard_state read-modify-write |
| P2-13 | portfolio-risk | risk_management.py:79-81 | Fallback to avg_cost_usd when live price missing |
| P2-14 | portfolio-risk | risk_management.py:21-51 | `_streaming_max` unbounded JSONL scan |
| P2-15 | metals-core | metals_swing_trader.py:2406 | Hardcoded 21:55 close time ignoring DST |
| P2-16 | metals-core | metals_swing_trader.py:2826 | Hardcoded usdsek=10.85 |
| P2-17 | metals-core | orb_predictor.py:32-35 | ORB predictor UTC offsets hardcoded for winter |
| P2-18 | metals-core | fin_snipe_manager.py:514 | 5% cert stop too tight for 5x leverage |
| P2-19 | avanza-api | avanza_orders.py:130-135 | CONFIRM matches most recent order, no order-ID |
| P2-20 | avanza-api | avanza_session.py:628 | `cancel_order` no account whitelist |
| P2-21 | avanza-api | avanza_orders.py:209 | CONFIRM flow uses stale price (up to 5min) |
| P2-22 | data-external | fx_rates.py | FX rate defaults to 1.0 on API failure |
| P2-23 | data-external | onchain_data.py:102 | `_load_onchain_cache` missing `_coerce_epoch` |
| P2-24 | infrastructure | shared_state.py:100 | `_cached` stores pre-call timestamp → early expiry |
| P2-25 | infrastructure | health.py | Health state r/m/w only thread-safe, not process-safe |
| P2-26 | infrastructure | file_utils.py:233 | `msvcrt.locking(LK_LOCK)` retries for only 1 second |

---

## Recurring Bug Classes

### 1. Silent failures hidden by bare `except Exception`
Still the most pervasive issue. The hardware trailing stop (P1-01), mahalanobis signal
(P1-02), fear_greed (P1-06), and signal utility (P2-03) all have bugs caught by broad
exception handlers that silently degrade to HOLD/None/fallback.

### 2. Account/order safety gaps in parallel code paths
The Avanza subsystem has THREE order placement paths (session, client/TOTP, metals_helpers/page).
Safety guards (whitelist, size cap, order lock) are applied inconsistently. The session path
is the most hardened; the other two are missing critical guards.

### 3. `_cached()` dogpile None returns
At least 3 callers don't handle None: `_get_horizon_weights`, `_get_correlation_groups` (handled
via `or`), and potentially IC data. The `_cached` API should either never return None or clearly
document the contract.

### 4. Timestamp/clock inconsistencies
Multiple subsystems hardcode UTC offsets for CET winter time (ORB predictor, swing trader
close time, `_cet_hour` fallback) without DST adjustment. This is a recurring issue across
metals-core.

### 5. Read-modify-write without cross-process locking
trade_guards, health_state, contract_state, and portfolio_state_* files are all shared across
processes (main loop, metals loop, golddigger, fix agents) but only have threading locks,
not file-level locks. Concurrent processes can produce lost updates.

---

## Signal Module Status (from signals-modules agent)

| Previously Reported Bug | Status |
|------------------------|--------|
| Fibonacci extensions 10x | **FIXED** |
| VWAP cumulative (no session reset) | **NOT FIXED** — still uses full-series cumsum |
| Annualizer sqrt(365) vs sqrt(252) | **NOT FIXED** — inconsistent within volatility.py |
| News "cut" keyword positive mapping | **PARTIALLY FIXED** — rate/guidance cut handled, bare "cut" still matches negatives |
| COT relative file paths | **NOT FIXED** — still uses `"data/..."` relative paths |
| `_cached()` arg swap (mahalanobis + complexity_gap) | **NEW BUG** — never worked, will crash on enable |

---

## Verified Fixes from Review #7

| #7 Finding | Status | Notes |
|------------|--------|-------|
| P1-01 Fibonacci extensions 10x | Not re-verified | Agent should check |
| P1-02 record_trade never receives pnl_pct | CONFIRMED STILL BROKEN (P1-03 above) |
| P1-05 Fear & Greed sustained-fear gate | Not re-verified |
| P1-06 fear_greed unguarded data[0] | CONFIRMED STILL BROKEN (P1-06 above) |
| P1-10 health.py conflates trigger/invocation time | Partially addressed by loop_contract per-tier grace |
| P1-11 log_rotation no fsync | CONFIRMED STILL BROKEN (P1-07 above) |
| P1-12 Hardware trailing stop tuple/dict | CONFIRMED STILL BROKEN (P1-01 above) |
| P1-13 Emergency mode sell+stop overfill | CONFIRMED STILL BROKEN (P1-08 above) |

**4 of 8 sampled P1s from review #7 are confirmed unfixed.** This is a concerning pattern.

---

## Recommendations

### Immediate (before next trading session):
1. **Fix P1-01** (hardware trailing stop) — import correct function or fix kwargs
2. **Fix P1-02** (mahalanobis _cached call) — swap arguments before enabling signal
3. **Fix P1-04** (digest in _track) — one-line fix, prevents cascade failure
4. **Fix P1-06** (fear_greed guard) — add `if not body.get("data"): return None`
5. **Fix P1-09** (double-logging) — remove _log_trigger from skip paths in invoke_agent

### Short-term (this week):
6. Add account whitelist to `metals_avanza_helpers` and `avanza_client` (P1-05)
7. Fix log rotation fsync + append race (P1-07)
8. Wire `pnl_pct` into transaction schema (P1-03)
9. Fix `_get_horizon_weights` None guard (P2-01)
10. Fix persistence filter cold-start seeding (P2-02)

### Medium-term (next sprint):
11. Consolidate Avanza order paths to single hardened implementation
12. Add file-level locking to shared state files (trade_guards, health, contract)
13. DST-aware close times throughout metals subsystem
14. Make `_cached()` never return None (return empty sentinel instead)
15. Decompose metals_loop.py monolith

---

## Methodology Notes

- **Independent review:** Manual code reading of ~15 key files across all 8 subsystems.
  Found 17 issues (1 P1, 14 P2, 2 P3). Strongest on signals-core and infrastructure;
  weakest on metals-core internals (5000-line monolith limits depth).
- **Agent reviews:** 8 parallel agents, each assigned one subsystem with specific focus areas.
  Found ~80 raw findings, consolidated to ~60 confirmed after dedup and false-positive removal.
- **Cross-critique:** The comparison table above shows where each approach added unique value.
  Agents were stronger on deep call-chain tracing (pnl_pct wiring, log_rotation fsync).
  Independent review was stronger on cross-cutting patterns (_cached dogpile, dispatch efficiency).
