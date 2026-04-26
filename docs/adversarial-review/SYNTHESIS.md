# Adversarial Review #8 -- SYNTHESIS (2026-04-26)

**Date**: 2026-04-26 (supersedes 2026-04-19 review)
**Methodology**: Dual review -- 8 parallel code-reviewer agents + independent manual deep-read
**Scope**: Full codebase (149+ portfolio files, 40 signal modules, 20+ support files)
**Commit baseline**: b7610fb3 (main)

---

## Executive Summary

Third full adversarial review. Eight parallel code-reviewer agents examined the codebase
partitioned into 8 subsystems, while an independent manual review read critical files
directly. Cross-critique identified 11 findings confirmed by both reviewers, 37 unique
to agents, and 12 unique to the manual review.

**Key theme**: The two highest-risk subsystems (metals-core, avanza-api) both handle real
money and have the most critical findings. Financial math bugs (fx_rate defaults, stop-loss
distances, PnL calculations) are the primary risk vector.

---

## TOP 10 CRITICAL FINDINGS

### 1. [P0] risk_management.py:66 -- fx_rate defaults to 1.0 (CONFIRMED BOTH)
When agent_summary is stale, portfolio value computed with fx_rate=1.0 instead of ~10.5.
USD positions undervalued by 10.5x. Drawdown circuit breaker false-triggers, blocking ALL
Layer 2 decisions. **Single most dangerous bug -- affects both strategies simultaneously.**

### 2. [P0] fin_snipe_manager.py:64 -- MIN_STOP_DISTANCE_PCT=1.0 violates 3% rule (CONFIRMED BOTH)
Documented rule says "NEVER place stop-loss within 3% of bid." Code enforces 1%. A stop
1.5% below bid on a 5x warrant fills from normal spread movement. Replicates Mar 3 incident.

### 3. [P0] avanza_orders.py:17 -- CONFIRM orders route through unguarded TOTP path (AGENT)
place_buy/sell_order imported from avanza_control resolves to avanza_client (TOTP path)
with NO 50K cap, NO order lock, NO account whitelist. Any Telegram CONFIRM bypasses safety.

### 4. [P0] ic_computation.py:19 -- Relative DATA_DIR path (AGENT)
DATA_DIR = Path("data") is relative. In any non-CWD context (subprocess, scheduled task),
IC cache silently misses. IC-based weight multipliers become 1.0x for everything.

### 5. [P0] volatility.py:311 -- No null guard on active signal (AGENT)
df.columns accessed without df is None check. Only active signal module missing this guard.
AttributeError crashes the signal for all 5 tickers when df=None.

### 6. [P0] fin_snipe_manager.py:1590 -- Budget per-instrument, not split (AGENT)
--budget 50000 --orderbook A --orderbook B gives 50K each, deploying 100K total.
Double-spend of stated budget cap.

### 7. [P0] onchain_data.py:101 -- _load_onchain_cache skips _coerce_epoch (AGENT)
Fallback path when no API token crashes on ISO timestamp. Half-applied fix silently
disables on-chain voter.

### 8. [P1] avanza_session.py -- POST retry on browser-dead = double order (AGENT)
_with_browser_recovery retries POST after TargetClosedError. If first POST succeeded
but response read failed, retry places duplicate order. Real-money double-order risk.

### 9. [P1] signal_engine.py:2864 -- Regime gating uses global not per-ticker accuracy (AGENT)
Regime gating exemption uses global recent accuracy. A signal with 55% global but 25%
on XAG-USD bypasses regime gating for XAG, allowing harmful signals to vote.

### 10. [P1] log_rotation.py:242 -- Missing fsync before os.replace (AGENT)
rotate_jsonl writes temp file without flush+fsync before rename. Power loss can produce
zero-length replacement file. Irrecoverable data loss for signal_log/journal.

---

## CONFIRMED FINDINGS (Both reviewers independently identified)

| # | Subsystem | Finding | Impact |
|---|-----------|---------|--------|
| 1 | portfolio-risk | fx_rate default 1.0 | False circuit breaker |
| 2 | metals-core | Stop distance 1% vs 3% rule | Stop fills from spread |
| 3 | portfolio-risk | trade_guards no lock | Lost updates |
| 4 | portfolio-risk | kelly_sizing 500 vs 1000 SEK | Below Avanza minimum |
| 5 | metals-core | ORB DST-blind morning window | Wrong ORB range in summer |
| 6 | metals-core | metals_loop raw json.load | Position state corruption |
| 7 | metals-core | Budget per-instrument not split | Double-spend |
| 8 | orchestration | crypto_scheduler local timestamp | 1-2h offset in logs |
| 9 | infrastructure | Dashboard CORS wildcard | Data exfiltration |
| 10 | portfolio-risk | Equity curve fees excluded | Kelly overestimates edge |
| 11 | orchestration | set in JSON-destined dict | Latent serialization crash |

---

## SUBSYSTEM HEALTH SCORES

| Subsystem | P0 | P1 | P2 | Health |
|-----------|----|----|----|----|
| signals-core | 1 | 6 | 6 | 40/100 Poor |
| orchestration | 2 | 4 | 3 | 45/100 Poor |
| portfolio-risk | 2 | 4 | 4 | 45/100 Poor |
| metals-core | 3 | 5 | 6 | 30/100 CRITICAL |
| avanza-api | 2 | 4 | 4 | 35/100 CRITICAL |
| signals-modules | 2 | 3 | 3 | 50/100 Fair |
| data-external | 2 | 5 | 6 | 40/100 Poor |
| infrastructure | 1 | 6 | 6 | 40/100 Poor |

---

## PRIORITY FIX LIST

### Immediate (fix today -- real money at stake)
1. risk_management.py:66 -- fx_rate default 1.0 -> 10.5
2. fin_snipe_manager.py:64 -- MIN_STOP_DISTANCE_PCT 1.0 -> 3.0
3. avanza_orders.py:17 -- Import from avanza_session, not avanza_control
4. ic_computation.py:19 -- DATA_DIR to Path(__file__).resolve().parent.parent / "data"
5. volatility.py:311 -- Add df is None guard

### This week
6. avanza_session.py -- No POST retry on browser-dead (double-order risk)
7. trade_guards.py -- Add threading.Lock for read-modify-write
8. kelly_sizing.py:290 -- Change 500 to 1000 SEK minimum
9. orb_predictor.py:32-35 -- DST-aware morning window
10. onchain_data.py:101 -- Apply _coerce_epoch
11. fin_snipe_manager.py:1590 -- Divide budget by len(orderbook_filter)
12. log_rotation.py:242 -- Add fsync before os.replace
13. funding_rate.py -- Add _binance_limiter.wait()
14. social_sentiment.py -- Replace print() with logger, use http_retry

### Next sprint
15. signal_engine.py -- Refactor _weighted_consensus into sub-functions
16. Centralize fx_rate default across all modules
17. metals_loop.py -- Replace raw json.load with file_utils
18. Dashboard CORS -- Restrict from * to localhost/LAN
19. credit_spread.py -- Add threading.Lock on _oas_cache
20. complexity_gap_regime.py + mahalanobis_turbulence.py -- Fix _cached arg order

---

## METHODOLOGY NOTES

- **Agent reviews**: 8 feature-dev:code-reviewer agents, each given complete file list
  with specific review criteria (7 categories: bugs, security, reliability, data integrity,
  performance, architecture, correctness). Total agent runtime: ~3-6 min each.
- **Manual review**: Direct reading of 15+ critical files (~5K lines in detail).
- **Cross-referencing**: Independent reviews conducted without knowledge of each other.
  11 findings confirmed by both (highest confidence). 37 agent-only + 12 manual-only.
- Prior reviews: Apr 12 (#1), Apr 17 (#6), Apr 19 (#7), Apr 24 (#7.5). This is #8.

---

## POSITIVE PATTERNS

1. **Fail-closed accuracy gate**: When stats loading fails, ALL signals gated at 0%.
2. **Account whitelist**: ALLOWED_ACCOUNT_IDS in avanza_session.py prevents pension trades.
3. **Atomic I/O layer**: file_utils.py with fsync+replace is used in most critical paths.
4. **Dogpile prevention**: shared_state._cached prevents thundering herd on cache misses.
5. **Circuit breakers**: Per-API circuit breakers (Binance, Alpaca) with CLOSED/OPEN/HALF_OPEN.
6. **BUG tracking**: 180+ named bugs tracked inline, creating strong audit trail.

---

Generated: 2026-04-26 by Claude Opus 4.6 (adversarial review #8)
Prior reviews: ADVERSARIAL_REVIEW_2026_04_12.md, 2026-04-17.md, 2026-04-24.md
