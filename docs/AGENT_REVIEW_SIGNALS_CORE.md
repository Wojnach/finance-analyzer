# Adversarial Review: signals-core (Agent Findings)

Reviewer: Code-reviewer subagent (feature-dev:code-reviewer)
Date: 2026-04-08

---

## CRITICAL

### CS1. Per-ticker consensus cache is horizon-blind — 3h data read by 1d callers [92% confidence]
**File**: `portfolio/signal_engine.py:1609-1613`

Cache written as `write_accuracy_cache("per_ticker_consensus", _ptc)` where `_ptc` uses
`acc_horizon` (could be "3h", "4h", or "1d"). Key is always `"per_ticker_consensus"`.
First caller (3h horizon) populates with 3h accuracy. Second caller (1d horizon) reads
same key and gets 3h data. This feeds the `_PER_TICKER_CONSENSUS_GATE = 0.38` check that
suppresses trades on AMD (24.8%), GOOGL (31.3%), META (34.2%). Wrong horizon accuracy
can flip the gate either direction on boundary tickers.

**Fix**: Use `f"per_ticker_consensus_{acc_horizon}"` as cache key.

### CS2. Ministral applicable count is crypto-only but runtime runs for ALL tickers [95% confidence]
**File**: `portfolio/signal_engine.py:452`

`_compute_applicable_count` skips ministral for non-crypto (`if sig == "ministral" and not
is_crypto: continue`). But at runtime (line 1321), ministral runs for ALL tickers with no
crypto filter. `total_applicable` is under-counted for stocks/metals. Doesn't directly
suppress trades (MIN_VOTERS uses raw counts) but inflates abstention rate in reports,
misleading Layer 2 decisions.

### CS3. SignalDB SQLite not thread-safe for concurrent writes [88% confidence]
**File**: `portfolio/signal_db.py:25-37`

Single `sqlite3.Connection` per instance. `backfill_outcomes` holds one connection for
minutes while `log_signal_snapshot` from the 60s loop creates new instances writing
concurrently. WAL mode helps but doesn't eliminate concurrent write contention. Failed
writes logged at DEBUG level → snapshots silently dropped.

### CS4. write_accuracy_cache read-then-write race — concurrent threads clobber each other [85% confidence]
**File**: `portfolio/accuracy_stats.py:699-707`

`write_accuracy_cache()`: load file → merge → write, no lock. 20 tickers processed in
parallel threads each trigger writes on cache miss. Thread A's write is clobbered by
Thread B's stale-base write. Per-ticker accuracy cache (BUG-158 regime gating) is
especially vulnerable — partial cross-tab fed to regime gating exemption check.

---

## HIGH

### HCS1. Regime accuracy overlay discards the 70/30 recency blend [85% confidence]
**File**: `portfolio/signal_engine.py:1643-1646`

After careful recency-blend (`blend_accuracy_data` with 70/30 weights), the regime
overlay replaces blended accuracy with raw all-time regime accuracy for any signal with
30+ samples. No recency blend applied to regime data. Recent regime performance changes
(last 7 days) are invisible. Per-ticker override (line 1655) has final say but only
for tickers WITH per-ticker data.

### HCS2. Fear & Greed sustained fear logic: allows BUY AFTER 30 days of extreme fear [92% confidence]
**File**: `portfolio/signal_engine.py:1206-1213`

Code allows BUY when `fear_days > 30`, suppresses in first 30 days. Comment warns that
during sustained fear (2022) "prices dropped another -40% after signal." With current
46+ day fear streak, this is actively generating BUY votes for fear_greed RIGHT NOW.
The intent appears reversed — allowing contrarian BUY during the exact prolonged-fear
scenario the historical warning describes.

### HCS3. outcome_tracker fear_greed derivation ignores sustained fear gate — inflates accuracy [88% confidence]
**File**: `portfolio/outcome_tracker.py:74-91`

`_derive_signal_vote` for fear_greed uses hardcoded `fg <= 20 → BUY` with no sustained
fear gate or regime gate. Historical entries (before `passed_votes` was added) have
inflated fear_greed BUY counts. These entries permanently skew accuracy data.

### HCS4. `_weighted_consensus` doesn't know about per-ticker consensus gate — inflated confidence logged [90% confidence]
**File**: `portfolio/signal_engine.py:636-817`

Weighted consensus generates a result for gated tickers (AMD 24.8%). The per-ticker gate
fires AFTER consensus (line 1844), setting action=HOLD. But `extra_info["_weighted_confidence"]`
contains the pre-gate inflated value. Layer 2 sees this misleading confidence.

### HCS5. meta_learner: no train/test gap rejection — overfitting models deployed silently [90% confidence]
**File**: `portfolio/meta_learner.py:156-162`

10-day holdout with LightGBM, 30+ features, ~600 test rows after dedup. A model with
75% train / 52% test accuracy gets saved and used without any rejection threshold.
No gap check between train and test accuracy means overfitting models go to production.

### HCS6. SQLite insert_snapshot partial write — no rollback on mid-insert failure [82% confidence]
**File**: `portfolio/signal_db.py:93-153`

Inserts snapshot row, then ticker_signals, then outcomes, commits once at end. Exception
mid-way propagates without rollback. Partial data (snapshot without ticker data) can be
committed by the next unrelated operation on the same connection.

---

## MEDIUM

### MCS1. ADX cache id(df) reuse within a cycle [82% confidence]
**File**: `portfolio/signal_engine.py:947-992`

Agent confirms Claude's C1 finding. Within a single cycle, 8 concurrent threads can
trigger GC → address reuse → stale ADX from different ticker.

### MCS2. Dynamic correlation groups only see post-gated votes [80% confidence]
**File**: `portfolio/signal_engine.py:508-595`

`log_signal_snapshot` stores post-gated `_votes`. Regime-gated signals are always HOLD
→ zero variance → excluded from correlation clustering. Correlation between gated signals
goes unmeasured, degrading deduplication quality over time.

### MCS3. outcome_tracker fromisoformat fails on Python 3.10 for timezone-aware strings [80% confidence]
**File**: `portfolio/outcome_tracker.py:329`

`fromisoformat("...+00:00")` fails on Python 3.10. Would silently prevent ALL outcome
backfilling → accuracy stats always zero → all signals get default 0.5 weight.

### MCS4. blend_accuracy_data: no-recent-data signals get all-time accuracy (correct but fragile) [80% confidence]
**File**: `portfolio/accuracy_stats.py:612-658`

Signals that stopped firing in the last 7 days get all-time accuracy as blend. May not
reflect current regime. Corner case for newly gated signals.

---

## LOW

### LCS1. signal_weights.py: MWU weights computed but never read by signal_engine [100% confidence]
Dead code. `SignalWeightManager` saves to `signal_weights.json` but `generate_signal()`
never calls `get_weight()`. Zero effect on trading.

### LCS2. ticker_accuracy uses change_pct > 0 without min threshold — inflates accuracy [90% confidence]
**File**: `portfolio/ticker_accuracy.py:61`

`accuracy_stats.py` uses `_MIN_CHANGE_PCT = 0.05` threshold. `ticker_accuracy.py` uses
bare `change_pct > 0`. A 0.001% move counts as "correct." Per-ticker accuracy is
systematically higher → P(up) probability overstated in Mode B notifications.

### LCS3. RECENCY_MIN_SAMPLES=30 overrides function default of 50 — undocumented [90% confidence]
**File**: `portfolio/signal_engine.py:40-45`

Blend activates with only 30 recent samples (vs documented 50). Noisier blending.

---

## Cross-Critique: Claude Direct vs Signals-Core Agent

### Agent found that Claude missed:
1. **CS1**: Per-ticker consensus cache horizon-blind — **CRITICAL** — total miss
2. **CS2**: Ministral applicable count crypto-only mismatch — missed
3. **CS3**: SignalDB thread safety — missed (I noted signal_db broadly)
4. **CS4**: write_accuracy_cache race — missed
5. **HCS1**: Regime accuracy discards recency blend — missed
6. **HCS2**: Fear & Greed allows BUY after 30 days of fear — missed (dangerous in current market)
7. **HCS3**: outcome_tracker inflates fear_greed accuracy — missed
8. **HCS4**: Weighted consensus doesn't know per-ticker gate — missed
9. **HCS5**: meta_learner no overfit rejection — missed
10. **HCS6**: SQLite partial write no rollback — missed
11. **LCS2**: ticker_accuracy threshold mismatch — missed

### Claude found that agent confirmed:
1. **C1/MCS1**: ADX cache id(df) — both found (agent confirmed at MEDIUM)
2. **H1**: Sentiment flush TOCTOU — agent didn't re-raise
3. **H2**: Dynamic correlation mega-groups — agent found related MCS2 (post-gate votes)
4. **H3**: Utility boost cap at 0.95 — not re-raised by agent

### Net assessment:
The signals-core agent found **4 CRITICAL + 6 HIGH + 4 MEDIUM + 3 LOW = 17 net-new issues**.
The horizon-blind per-ticker consensus cache (CS1) is the most impactful — it can flip
the gate on boundary tickers between "allow trading" and "suppress trading" depending
on which horizon computed first in the cycle. The fear & greed logic reversal (HCS2) is
actively generating wrong signals in the current 46-day fear streak.

Agent was **dramatically stronger** — the voting engine is complex enough that only
line-by-line tracing reveals the interaction bugs between caches, gates, and overlays.
