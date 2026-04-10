# Cross-Critique — Dual Adversarial Review

**Date**: 2026-04-10
**Format**: Independent review (I) vs Agent reviews (A) — each side critiques the other
**Status**: signals-core + avanza-api agents complete; 6 agents still running

---

## Part 1: Agent Review → Independent Review Critique

*Findings from agent reviews that the independent review MISSED or UNDERRATED.*

### 1.1 Independent MISSED: Per-Ticker Accuracy Strips Directional Fields [A-SC-1, P1]
The independent review examined `_weighted_consensus` and noted the directional gate logic, but did NOT check whether the per-ticker accuracy override at line 1844 preserves the `buy_accuracy`/`sell_accuracy` fields. The agent correctly identified that these fields are dropped during the override, silently disabling the directional gate for tickers with sufficient per-ticker data. This is the highest-impact finding across both reviews.

### 1.2 Independent MISSED: Regime Accuracy Cache Shared Timestamp [A-SC-2, P1]
The independent review noted the accuracy system's blending but did not audit the regime accuracy cache's timestamp management. The agent found that `write_regime_accuracy_cache` uses a single `"time"` key (unlike the main accuracy cache which was fixed in BUG-133). Cross-horizon contamination is a real risk.

### 1.3 Independent MISSED: signal_history.py Race Condition [A-SC-6, P2]
The independent review focused on `file_utils.py` atomicity and `shared_state.py` thread safety but missed `signal_history.py`'s unprotected read-modify-write. This is a genuine race under ThreadPoolExecutor.

### 1.4 Independent MISSED: blend_accuracy_data Sample Count Inflation [A-SC-5, P2]
The independent review noted the accuracy gate but didn't trace how `total` is computed in `blend_accuracy_data`. Using `max(at_samples, rc_samples)` overstates sample count when the blend is dominated by a small recent window.

### 1.5 Independent MISSED: Ministral Applicable Count Mismatch [A-SC-3, P1]
The independent review noted the `_compute_applicable_count` function but didn't compare it against the actual vote-setting code. The comment "all tickers" vs the count function's "crypto-only" exclusion is a real inconsistency.

### 1.6 Independent UNDERRATED: Playwright Context Thread Safety [A-AV-1, P0 vs I-AV-3 P2]
The independent review flagged "Playwright context is module-level singleton without health check" as P2. The avanza-api agent identified the MUCH more severe issue: `_pw_lock` is only held during context *creation*, not during API calls. This means concurrent requests from multiple threads can corrupt Playwright's internal HTTP state. The independent review missed that the lock scope is too narrow — it focused on the health check (what happens after the context dies) rather than the fundamental thread-safety issue (what happens while the context is alive).

### 1.7 Independent UNDERRATED: Account Whitelist Gap [A-AV-2, P0 vs I-AV-2 P2]
The independent review flagged the account whitelist gap as P2 ("relies on trust chain"). The avanza-api agent correctly identified this as P0 — the TOTP path (`avanza_client._place_order`) has NO account check at all, and if Avanza re-orders accounts in the API response, trades execute on the pension account. The independent review's P2 rating was too mild for a finding that could cause real-money trades on the wrong account.

### 1.8 Independent MISSED: Pending Orders TOCTOU Race [A-AV-9, P1]
The independent review examined `avanza_orders.py` but focused on the confirmation flow, not the concurrent-access race condition on the pending orders file. The agent correctly identified that concurrent calls to `check_pending_orders()` could double-execute an order.

### 1.9 Independent MISSED: get_positions/get_portfolio_value Include Pension Account [A-AV-3, P1]
The independent review didn't check whether `avanza_client` functions filter by account ID. The agent found they iterate ALL accounts, inflating portfolio value and position data.

---

## Part 2: Independent Review → Agent Review Critique

*Findings from the independent review that the signals-core agent MISSED.*

### 2.1 Agent MISSED: Subprocess governance bypass [IN-7, P1]
The agent only reviewed signals-core files. The independent review's cross-cutting scan found that `bigbet.py` and `iskbets.py` bypass `claude_gate.py` entirely — this affects the broader system, not just signals-core.

### 2.2 Agent MISSED: JSONL append atomicity on Windows [IN-1, P1]
The agent noted `signal_history.py` race conditions but didn't audit `file_utils.atomic_append_jsonl` for NTFS non-atomicity. The independent review caught this broader infrastructure issue.

### 2.3 Agent MISSED: Portfolio drawdown fallback [PR-1, P1]
Outside the agent's subsystem scope — the independent review's cross-subsystem view caught this financial safety issue in risk_management.py.

### 2.4 Agent MISSED: Hardware trailing stop failure [MC-4, P1]
Outside scope — metals-core subsystem finding from independent review.

### 2.5 Assessment of Agent Findings
All 10 agent findings for signals-core are **valid and well-evidenced**. Specific notes:
- A-SC-1 (directional gate) is the most impactful — CONFIRMED, should be Priority 1
- A-SC-2 (regime cache) is CONFIRMED — reproducible by inspection
- A-SC-4 (regime overlay ordering) is correctly assessed as "not a bug" but a design clarity issue — AGREE with P1 downgrade to P2
- A-SC-8 (SQLite check_same_thread) is CONFIRMED — defensive hardening recommended
- A-SC-10 (docstring) is correctly rated P3

---

## Part 3: Agreement Matrix

*Findings that both sides independently identified — these have highest confidence.*

| Finding | Independent ID | Agent ID | Subsystem | Severity |
|---------|---------------|----------|-----------|----------|
| ADX cache key collision potential | SC-1 | (not directly flagged) | signals-core | P2 |
| Consensus 50% tie returns BUY | SC-2 | A-SC-4 (related: confidence semantics) | signals-core | P2 |
| Sentiment hysteresis dirty flag | SC-5 | (not flagged — assessed differently) | signals-core | P2 |

---

## Part 4: Disputed Findings

*Findings where one side disagrees with the other's assessment.*

### 4.1 Regime Weight Double-Counting [Agent A-SC-4 vs Independent]
- **Agent**: Flagged as P2 — regime weights applied in `_weighted_consensus` AND `apply_confidence_penalties`.
- **Independent**: Did not flag this. After analysis, I AGREE with the agent's assessment that this is confusing but not a bug: `_weighted_consensus` regime weights adjust relative signal importance, while `apply_confidence_penalties` applies an absolute confidence discount. The two are not double-counting.
- **Resolution**: AGREE — P2 for clarity, not a functional bug.

### 4.2 ADX Cache Key [Independent SC-1 vs Agent]
- **Independent**: Flagged `(id(df), len(df), close[-1])` as a potential collision risk.
- **Agent**: Did not flag this specifically, but noted the content-based key approach (C1 comment).
- **Resolution**: AGREE with independent — the risk is real but low-probability. The ticker-based cache key extension would be a small fix.

---

## Part 5: False Alarms

*Findings that were investigated and determined to be non-issues.*

| Finding | Source | Reason for dismissal |
|---------|--------|---------------------|
| PR-3: Kelly division by zero | Independent | Code properly guards against all zero-denominator cases (lines 38-41) |
| A-SC-10: Cost-adjusted accuracy double-filter | Agent | The `continue` correctly skips BOTH `total` and `correct` increments |
| MC-3: Stop-loss 3% check uses bid vs ask | Independent | Using bid is correct — stops trigger on sell-side price |
| SC-7: Fail-closed accuracy gate | Independent | POSITIVE finding — excellent defensive design, not a bug |

---

## Summary

The dual review found **60+ unique findings** across both reviewers (2 agents complete, 6 pending):
- **2 P0 findings** from the avanza-api agent that the independent review UNDERRATED (Playwright thread safety, account whitelist)
- **5 new P1 findings** from the signals-core agent that the independent review missed
- **4 new P1 findings** from the avanza-api agent that the independent review missed entirely
- **4 P1 findings** from the independent review that agents couldn't find (different subsystem scope)

**The most impactful finding** across both reviews is **A-AV-1: Playwright context used outside lock** (P0), which can cause corrupt trade responses when concurrent API calls are made. The **second most impactful** is **A-AV-2: TOTP path has no account whitelist** (P0), which can route orders to the pension account.

The independent review's overall health assessment was too optimistic for the avanza-api subsystem. The agent review correctly identified it as the weakest link in the system's financial safety chain.
